//! Resident ownership and semantic lowering for the reviewed Gemma 4 MoE.
//!
//! Expert source planes are never individually resident.  Each layer is
//! packed once into the version-2 provider blob and that blob is the only
//! expert allocation visible to execution.  Attention is deliberately kept
//! behind an explicit hook: the current stateless causal-attention semantic
//! accepts BF16 K/V tensors and cannot truthfully represent this model's
//! opaque, unit-scale E4M3 state (including its sliding-window retention
//! contract).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::gemma4_moe::{
    GEMMA4_MOE_LAYER_BLOB_BYTES, GEMMA4_MOE_LAYER_BLOB_PREFIX, GEMMA4_MOE_MODEL_FINGERPRINT,
    GEMMA4_MOE_PER_EXPERT_SCALES_OFFSET, GEMMA4_MOE_REPOSITORY, GEMMA4_MOE_REVISION,
    GEMMA4_MOE_SEMANTIC_REPOSITORY, Gemma4MoeConfig, Gemma4MoeExpertPlanes,
    Gemma4MoeExpertProjection, Gemma4MoeModelError, Gemma4MoeTensorPlane, VerifiedGemma4Moe,
    VerifiedGgufGemma4Moe, gemma4_moe_layer_blob_name, gemma4_moe_layer_blob_pack_inputs,
    gemma4_moe_per_expert_scale_destination,
};
use crate::gemma4_moe_graph::{
    Gemma4MoeExecutionBoundary, Gemma4MoeGraph, Gemma4MoeGraphNodeKind, Gemma4MoeGraphTensorDtype,
    Gemma4MoeGraphTensorEncoding, Gemma4MoeGraphTensorSpec, Gemma4MoeKvDescriptor,
    Gemma4MoeLinearRole, Gemma4MoeNormRole, Gemma4MoeRmsScaleMode, Gemma4MoeValueShape,
    build_gemma4_moe_graph_from_config_with_identity, expected_gemma4_moe_text_tensor_catalog,
};
use crate::op::{RmsNormScaleMode, SemanticOpDescriptor, SemanticOpKind, SplitHalfRotaryContract};
use crate::prepared_execution::{
    ExecutionAuditAccumulator, ExecutionBoundaryKind, PreparedCachePolicy, PreparedDynamicIdentity,
    PreparedSemanticCache,
};
use crate::weights::{
    VerifiedWeightPlanMetadata, WEIGHT_LOAD_CHUNK_BYTES, WeightClassification, WeightConsumer,
    WeightConsumerKey, WeightLoadEntry, WeightLoadPlan,
};
use crate::{
    AccessMode, AllocationCategory, CausalAttentionDescriptor, DType, ExecutionBuffer,
    ExecutionQueue, ExecutionSession, ExecutionSessionId, ExecutionState, ExecutionStateImageV1,
    KvCacheEncoding, KvPhysicalMemorySnapshot, KvState, KvStateDescriptor, OwnedTensorBinding,
    PreparedExecutionAudit, PreparedOperation, StateForkAuditV1, StateForkModeV1, TensorDType,
    TensorView,
};
use crate::{
    CheckpointIdentity, CheckpointPayload, SessionCheckpoint, StateOwnerKindV1, StatePlaneKindV1,
};

const ROUTE_EXPERT_COUNT: u64 = 128;
const ROUTE_TOP_K: u64 = 8;

/// Container-neutral access required by resident provisioning.
///
/// Implementations are only supplied for fully verified source types.  The
/// exact container identity is part of the load plan and request graph.
pub trait Gemma4MoeWeightSource: Send + Sync {
    fn config(&self) -> &Gemma4MoeConfig;
    fn repository(&self) -> &str;
    fn resolved_revision(&self) -> &str;
    fn source_container_identity(&self) -> &str;
    fn direct_tensors(&self) -> &[Gemma4MoeTensorPlane];
    fn read_direct_tensor(&self, logical_name: &str) -> Result<Vec<u8>, Gemma4MoeModelError>;
    fn read_expert_planes(
        &self,
        layer: u16,
        expert: u16,
        projection: Gemma4MoeExpertProjection,
    ) -> Result<Gemma4MoeExpertPlanes, Gemma4MoeModelError>;
}

impl Gemma4MoeWeightSource for VerifiedGemma4Moe {
    fn config(&self) -> &Gemma4MoeConfig {
        self.config()
    }

    fn repository(&self) -> &str {
        GEMMA4_MOE_REPOSITORY
    }

    fn resolved_revision(&self) -> &str {
        GEMMA4_MOE_REVISION
    }

    fn source_container_identity(&self) -> &str {
        GEMMA4_MOE_MODEL_FINGERPRINT
    }

    fn direct_tensors(&self) -> &[Gemma4MoeTensorPlane] {
        self.text_planes()
    }

    fn read_direct_tensor(&self, logical_name: &str) -> Result<Vec<u8>, Gemma4MoeModelError> {
        self.read_tensor(logical_name)
    }

    fn read_expert_planes(
        &self,
        layer: u16,
        expert: u16,
        projection: Gemma4MoeExpertProjection,
    ) -> Result<Gemma4MoeExpertPlanes, Gemma4MoeModelError> {
        self.read_expert_planes(layer, expert, projection)
    }
}

impl Gemma4MoeWeightSource for VerifiedGgufGemma4Moe {
    fn config(&self) -> &Gemma4MoeConfig {
        self.config()
    }

    fn repository(&self) -> &str {
        GEMMA4_MOE_SEMANTIC_REPOSITORY
    }

    fn resolved_revision(&self) -> &str {
        self.file_sha256()
    }

    fn source_container_identity(&self) -> &str {
        self.file_sha256()
    }

    fn direct_tensors(&self) -> &[Gemma4MoeTensorPlane] {
        self.direct_planes()
    }

    fn read_direct_tensor(&self, logical_name: &str) -> Result<Vec<u8>, Gemma4MoeModelError> {
        self.read_tensor(logical_name)
    }

    fn read_expert_planes(
        &self,
        layer: u16,
        expert: u16,
        projection: Gemma4MoeExpertProjection,
    ) -> Result<Gemma4MoeExpertPlanes, Gemma4MoeModelError> {
        self.read_expert_planes(layer, expert, projection)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MoeExecutionError(String);

impl Gemma4MoeExecutionError {
    fn invalid(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Gemma4MoeExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid Gemma 4 MoE execution: {}", self.0)
    }
}

impl std::error::Error for Gemma4MoeExecutionError {}

impl From<Gemma4MoeModelError> for Gemma4MoeExecutionError {
    fn from(error: Gemma4MoeModelError) -> Self {
        Self::invalid(error.to_string())
    }
}

fn is_expert_source_tensor(name: &str) -> bool {
    name.contains(".experts.")
}

fn is_embedded_per_expert_scale(name: &str) -> bool {
    name.ends_with(".router.per_expert_scale")
}

fn bytes_for_spec(spec: &Gemma4MoeGraphTensorSpec) -> Result<u64, Gemma4MoeExecutionError> {
    let element_bytes: u64 = match spec.dtype {
        Gemma4MoeGraphTensorDtype::Bf16 => 2,
        Gemma4MoeGraphTensorDtype::F32 => 4,
        Gemma4MoeGraphTensorDtype::Fp8E4M3 | Gemma4MoeGraphTensorDtype::U8 => 1,
    };
    spec.stored_shape
        .iter()
        .try_fold(element_bytes, |bytes, dimension| {
            bytes.checked_mul(*dimension)
        })
        .ok_or_else(|| Gemma4MoeExecutionError::invalid("resident tensor byte count overflowed"))
}

fn graph_dtype_to_tensor(dtype: Gemma4MoeGraphTensorDtype) -> TensorDType {
    match dtype {
        Gemma4MoeGraphTensorDtype::Bf16 => TensorDType::Bf16,
        Gemma4MoeGraphTensorDtype::F32 => TensorDType::F32,
        Gemma4MoeGraphTensorDtype::Fp8E4M3 | Gemma4MoeGraphTensorDtype::U8 => TensorDType::U8,
    }
}

fn direct_consumer(name: &str) -> Option<WeightConsumerKey> {
    let layer = name
        .strip_prefix("model.language_model.layers.")
        .and_then(|tail| tail.split('.').next())
        .and_then(|layer| layer.parse::<u64>().ok());
    let role = if name.ends_with("embed_tokens.weight") {
        WeightConsumer::EmbeddingAndTiedOutput
    } else if name.ends_with("model.language_model.norm.weight") {
        WeightConsumer::FinalNorm
    } else if name.ends_with("input_layernorm.weight") {
        WeightConsumer::InputNorm
    } else if name.ends_with("post_attention_layernorm.weight") {
        WeightConsumer::PostAttentionNorm
    } else if name.ends_with("pre_feedforward_layernorm.weight") {
        WeightConsumer::PreFeedforwardNorm
    } else if name.ends_with("pre_feedforward_layernorm_2.weight") {
        WeightConsumer::Gemma4MoePreFeedforwardNorm2
    } else if name.ends_with("post_feedforward_layernorm_1.weight") {
        WeightConsumer::Gemma4MoePostFeedforwardNorm1
    } else if name.ends_with("post_feedforward_layernorm_2.weight") {
        WeightConsumer::Gemma4MoePostFeedforwardNorm2
    } else if name.ends_with("post_feedforward_layernorm.weight") {
        WeightConsumer::PostFeedforwardNorm
    } else if name.ends_with("self_attn.q_proj.weight") {
        WeightConsumer::AttentionQ
    } else if name.ends_with("self_attn.k_proj.weight") {
        WeightConsumer::AttentionK
    } else if name.ends_with("self_attn.v_proj.weight") {
        WeightConsumer::AttentionV
    } else if name.ends_with("self_attn.o_proj.weight") {
        WeightConsumer::AttentionO
    } else if name.ends_with("self_attn.q_norm.weight") {
        WeightConsumer::AttentionQNorm
    } else if name.ends_with("self_attn.k_norm.weight") {
        WeightConsumer::AttentionKNorm
    } else if name.ends_with("mlp.gate_proj.weight") {
        WeightConsumer::MlpGate
    } else if name.ends_with("mlp.up_proj.weight") {
        WeightConsumer::MlpUp
    } else if name.ends_with("mlp.down_proj.weight") {
        WeightConsumer::MlpDown
    } else if name.ends_with("router.proj.weight") {
        WeightConsumer::MoeRouter
    } else if name.ends_with("router.scale") {
        WeightConsumer::Gemma4MoeRouterScale
    } else if name.ends_with("layer_scalar") {
        WeightConsumer::LayerScalar
    } else {
        return None;
    };
    Some(WeightConsumerKey { layer, role })
}

/// Constructs the one-to-one resident plan shared by safetensors and GGUF.
/// Source expert planes and learned per-expert scales are represented only by
/// their synthetic layer blob; they never receive destination offsets.
pub fn build_gemma4_moe_resident_weight_load_plan(
    source: &dyn Gemma4MoeWeightSource,
) -> Result<WeightLoadPlan, Gemma4MoeExecutionError> {
    let catalog = expected_gemma4_moe_text_tensor_catalog(source.config())
        .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
    let expected_direct = catalog
        .iter()
        .filter(|spec| !is_expert_source_tensor(&spec.name))
        .map(|spec| (spec.name.as_str(), spec))
        .collect::<BTreeMap<_, _>>();
    let mut actual_direct = BTreeMap::new();
    for plane in source.direct_tensors().iter().filter(|plane| {
        plane.source_name.starts_with("model.language_model.")
            && !is_expert_source_tensor(&plane.source_name)
    }) {
        if actual_direct
            .insert(plane.source_name.as_str(), plane)
            .is_some()
        {
            return Err(Gemma4MoeExecutionError::invalid(format!(
                "duplicate direct tensor: {}",
                plane.source_name
            )));
        }
    }
    if actual_direct.len() != expected_direct.len() {
        return Err(Gemma4MoeExecutionError::invalid(format!(
            "direct tensor count differs: expected {}, got {}",
            expected_direct.len(),
            actual_direct.len()
        )));
    }
    for (name, spec) in &expected_direct {
        let plane = actual_direct.get(name).ok_or_else(|| {
            Gemma4MoeExecutionError::invalid(format!("direct tensor is absent: {name}"))
        })?;
        if plane.shape != spec.stored_shape
            || (spec.dtype == Gemma4MoeGraphTensorDtype::Bf16 && plane.dtype != "BF16")
        {
            return Err(Gemma4MoeExecutionError::invalid(format!(
                "direct tensor metadata differs: {name}"
            )));
        }
    }

    let mut entries = Vec::with_capacity(expected_direct.len() + 30);
    let mut destination = 0_u64;
    for (name, spec) in expected_direct {
        if is_embedded_per_expert_scale(name) {
            continue;
        }
        if spec.encoding != Gemma4MoeGraphTensorEncoding::Plain {
            return Err(Gemma4MoeExecutionError::invalid(
                "non-expert direct tensor unexpectedly has an encoded layout",
            ));
        }
        let plane = actual_direct[name];
        let bytes = bytes_for_spec(spec)?;
        let start = destination;
        destination = destination
            .checked_add(bytes)
            .ok_or_else(|| Gemma4MoeExecutionError::invalid("resident bytes overflowed"))?;
        entries.push(WeightLoadEntry {
            tensor_name: name.to_owned(),
            classification: WeightClassification::Required,
            consumer: direct_consumer(name),
            dtype: graph_dtype_to_tensor(spec.dtype),
            shape: spec.stored_shape.clone(),
            source_file: plane.source_file.clone(),
            locked_file_size: plane.absolute_byte_range[1],
            locked_file_sha256: source.source_container_identity().to_owned(),
            source_range: plane.absolute_byte_range,
            destination_start: Some(start),
            chunks: Vec::new(),
        });
    }
    for layer in 0..source.config().layer_count {
        let start = destination;
        destination = destination
            .checked_add(GEMMA4_MOE_LAYER_BLOB_BYTES)
            .ok_or_else(|| Gemma4MoeExecutionError::invalid("resident bytes overflowed"))?;
        entries.push(WeightLoadEntry {
            tensor_name: gemma4_moe_layer_blob_name(layer),
            classification: WeightClassification::Required,
            consumer: Some(WeightConsumerKey {
                layer: Some(u64::from(layer)),
                role: WeightConsumer::Gemma4MoeLayerBlob,
            }),
            dtype: TensorDType::U8,
            shape: vec![GEMMA4_MOE_LAYER_BLOB_BYTES],
            source_file: "<synthetic-gemma4-moe-layer-blob>".to_owned(),
            locked_file_size: 0,
            locked_file_sha256: source.source_container_identity().to_owned(),
            source_range: [0, 0],
            destination_start: Some(start),
            chunks: Vec::new(),
        });
    }
    WeightLoadPlan::from_verified_entries(
        VerifiedWeightPlanMetadata {
            schema_version: "gemma4-moe-resident-plan-v1".to_owned(),
            repo_id: source.repository().to_owned(),
            resolved_revision: source.resolved_revision().to_owned(),
            lock_fingerprint: GEMMA4_MOE_MODEL_FINGERPRINT.to_owned(),
            tied_embeddings: true,
            chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
            total_destination_bytes: destination,
        },
        entries,
    )
    .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Gemma4MoeTensorBacking {
    ResidentWeight(String),
    ResidentExpertBlob { layer: u32 },
    TokenIds,
    Positions,
    ConstantBf16 { bits: u16, width: usize },
    Workspace,
    Alias { tensor: usize },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MoeExecutionTensor {
    id: usize,
    name: String,
    view: TensorView,
    backing: Gemma4MoeTensorBacking,
}

impl Gemma4MoeExecutionTensor {
    pub const fn id(&self) -> usize {
        self.id
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub const fn view(&self) -> &TensorView {
        &self.view
    }
    pub const fn backing(&self) -> &Gemma4MoeTensorBacking {
        &self.backing
    }
}

/// Exact opaque-state attention contract attached to a lowered graph node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gemma4MoeAttentionHook {
    layer: u32,
    kv: Gemma4MoeKvDescriptor,
    score_scale_bits: u32,
    sliding_api_required: bool,
}

impl Gemma4MoeAttentionHook {
    pub const fn layer(self) -> u32 {
        self.layer
    }
    pub const fn kv(self) -> Gemma4MoeKvDescriptor {
        self.kv
    }
    pub const fn score_scale_bits(self) -> u32 {
        self.score_scale_bits
    }
    pub const fn sliding_api_required(self) -> bool {
        self.sliding_api_required
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Gemma4MoeLowering {
    Semantic(Box<SemanticOpDescriptor>),
    StaticFp8Attention(Gemma4MoeAttentionHook),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MoeExecutionNode {
    graph_node_id: usize,
    stage: u8,
    label: String,
    lowering: Gemma4MoeLowering,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    boundary_after: Option<Gemma4MoeExecutionBoundary>,
}

impl Gemma4MoeExecutionNode {
    pub const fn graph_node_id(&self) -> usize {
        self.graph_node_id
    }
    pub const fn stage(&self) -> u8 {
        self.stage
    }
    pub fn label(&self) -> &str {
        &self.label
    }
    pub const fn lowering(&self) -> &Gemma4MoeLowering {
        &self.lowering
    }
    pub fn inputs(&self) -> &[usize] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[usize] {
        &self.outputs
    }
    pub const fn boundary_after(&self) -> Option<Gemma4MoeExecutionBoundary> {
        self.boundary_after
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MoeExecutionSegment {
    first_node: usize,
    end_node: usize,
    boundary: Gemma4MoeExecutionBoundary,
}

/// One request transition submitted to every sliding-attention layer.
/// Before the ring saturates a transition may contain a chunk. At and after
/// saturation it is exactly one token so no row can overwrite a K/V slot
/// still needed by another query in the same transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gemma4MoeTransitionSegment {
    start_position: u64,
    token_count: u64,
    expected_length: u64,
    saturated_sliding_ring: bool,
}

impl Gemma4MoeTransitionSegment {
    pub const fn start_position(self) -> u64 {
        self.start_position
    }
    pub const fn token_count(self) -> u64 {
        self.token_count
    }
    pub const fn expected_length(self) -> u64 {
        self.expected_length
    }
    pub const fn saturated_sliding_ring(self) -> bool {
        self.saturated_sliding_ring
    }
}

pub fn plan_gemma4_moe_transitions(
    start_position: u64,
    token_count: u64,
) -> Result<Vec<Gemma4MoeTransitionSegment>, Gemma4MoeExecutionError> {
    if token_count == 0 {
        return Err(Gemma4MoeExecutionError::invalid(
            "transition token count must be non-zero",
        ));
    }
    let final_length = start_position
        .checked_add(token_count)
        .ok_or_else(|| Gemma4MoeExecutionError::invalid("transition length overflowed"))?;
    let mut segments = Vec::new();
    let mut cursor = start_position;
    if cursor < 1_024 {
        let chunk = token_count.min(1_024 - cursor);
        let expected_length = cursor + chunk;
        segments.push(Gemma4MoeTransitionSegment {
            start_position: cursor,
            token_count: chunk,
            expected_length,
            saturated_sliding_ring: expected_length == 1_024,
        });
        cursor = expected_length;
    }
    while cursor < final_length {
        segments.push(Gemma4MoeTransitionSegment {
            start_position: cursor,
            token_count: 1,
            expected_length: cursor + 1,
            saturated_sliding_ring: true,
        });
        cursor += 1;
    }
    Ok(segments)
}

impl Gemma4MoeExecutionSegment {
    pub const fn first_node(&self) -> usize {
        self.first_node
    }
    pub const fn end_node(&self) -> usize {
        self.end_node
    }
    pub const fn boundary(&self) -> Gemma4MoeExecutionBoundary {
        self.boundary
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MoeExecutionLayout {
    model_fingerprint: String,
    source_container_identity: String,
    plan_digest: [u8; 32],
    token_count: u64,
    tensors: Vec<Gemma4MoeExecutionTensor>,
    nodes: Vec<Gemma4MoeExecutionNode>,
    segments: Vec<Gemma4MoeExecutionSegment>,
    transitions: Vec<Gemma4MoeTransitionSegment>,
    terminal_readback_tensor: usize,
    resident_weight_bytes: u64,
    workspace_bytes: u64,
}

impl Gemma4MoeExecutionLayout {
    pub fn model_fingerprint(&self) -> &str {
        &self.model_fingerprint
    }
    pub fn source_container_identity(&self) -> &str {
        &self.source_container_identity
    }
    pub const fn plan_digest(&self) -> &[u8; 32] {
        &self.plan_digest
    }
    pub const fn token_count(&self) -> u64 {
        self.token_count
    }
    pub fn tensors(&self) -> &[Gemma4MoeExecutionTensor] {
        &self.tensors
    }
    pub fn nodes(&self) -> &[Gemma4MoeExecutionNode] {
        &self.nodes
    }
    pub fn segments(&self) -> &[Gemma4MoeExecutionSegment] {
        &self.segments
    }
    pub fn transitions(&self) -> &[Gemma4MoeTransitionSegment] {
        &self.transitions
    }
    pub const fn terminal_readback_tensor(&self) -> usize {
        self.terminal_readback_tensor
    }
    pub const fn resident_weight_bytes(&self) -> u64 {
        self.resident_weight_bytes
    }
    pub const fn workspace_bytes(&self) -> u64 {
        self.workspace_bytes
    }
    pub fn attention_hooks(&self) -> impl Iterator<Item = Gemma4MoeAttentionHook> + '_ {
        self.nodes.iter().filter_map(|node| match node.lowering {
            Gemma4MoeLowering::StaticFp8Attention(hook) => Some(hook),
            Gemma4MoeLowering::Semantic(_) => None,
        })
    }
}

fn contiguous(dtype: DType, shape: &[u64]) -> Result<TensorView, Gemma4MoeExecutionError> {
    let shape = shape
        .iter()
        .map(|dimension| usize::try_from(*dimension))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| Gemma4MoeExecutionError::invalid("tensor extent exceeds usize"))?;
    TensorView::contiguous(dtype, &shape)
        .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))
}

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let upper = bits >> 16;
    let lower = bits & 0xffff;
    (upper + u32::from(lower > 0x8000 || (lower == 0x8000 && upper & 1 != 0))) as u16
}

struct LayoutBuilder<'a> {
    graph: &'a Gemma4MoeGraph,
    plan: &'a WeightLoadPlan,
    tensors: Vec<Gemma4MoeExecutionTensor>,
    nodes: Vec<Gemma4MoeExecutionNode>,
    outputs: Vec<Vec<usize>>,
    weights: BTreeMap<String, usize>,
    token_ids: usize,
    positions: usize,
    workspace_bytes: u64,
}

impl<'a> LayoutBuilder<'a> {
    fn new(
        graph: &'a Gemma4MoeGraph,
        plan: &'a WeightLoadPlan,
    ) -> Result<Self, Gemma4MoeExecutionError> {
        let expected_revision = if graph.source_container_identity() == GEMMA4_MOE_MODEL_FINGERPRINT
        {
            GEMMA4_MOE_REVISION
        } else {
            graph.source_container_identity()
        };
        if graph.model_fingerprint() != GEMMA4_MOE_MODEL_FINGERPRINT
            || graph.model_fingerprint() != plan.lock_fingerprint
            || plan.resolved_revision != expected_revision
            || plan
                .entries
                .iter()
                .any(|entry| entry.locked_file_sha256 != graph.source_container_identity())
        {
            return Err(Gemma4MoeExecutionError::invalid(
                "graph and weight-plan identity differ",
            ));
        }
        if !plan
            .has_valid_digest()
            .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?
        {
            return Err(Gemma4MoeExecutionError::invalid(
                "weight-plan digest is invalid",
            ));
        }
        let catalog = expected_gemma4_moe_text_tensor_catalog(graph.config())
            .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?
            .into_iter()
            .map(|spec| (spec.name.clone(), spec))
            .collect::<BTreeMap<_, _>>();
        let mut builder = Self {
            graph,
            plan,
            tensors: Vec::new(),
            nodes: Vec::new(),
            outputs: vec![Vec::new(); graph.nodes().len()],
            weights: BTreeMap::new(),
            token_ids: usize::MAX,
            positions: usize::MAX,
            workspace_bytes: 0,
        };
        let mut names = BTreeSet::new();
        for entry in &plan.entries {
            if !names.insert(entry.tensor_name.as_str()) {
                return Err(Gemma4MoeExecutionError::invalid(
                    "resident plan contains a duplicate entry",
                ));
            }
            let (view, backing) = if let Some(layer) = entry
                .tensor_name
                .strip_prefix(GEMMA4_MOE_LAYER_BLOB_PREFIX)
                .and_then(|layer| layer.parse::<u32>().ok())
            {
                (
                    contiguous(DType::U8, &[GEMMA4_MOE_LAYER_BLOB_BYTES])?,
                    Gemma4MoeTensorBacking::ResidentExpertBlob { layer },
                )
            } else {
                let spec = catalog.get(&entry.tensor_name).ok_or_else(|| {
                    Gemma4MoeExecutionError::invalid(format!(
                        "resident plan tensor is outside the graph catalog: {}",
                        entry.tensor_name
                    ))
                })?;
                if is_expert_source_tensor(&spec.name) || is_embedded_per_expert_scale(&spec.name) {
                    return Err(Gemma4MoeExecutionError::invalid(
                        "individual expert storage is forbidden",
                    ));
                }
                let dtype = match spec.dtype {
                    Gemma4MoeGraphTensorDtype::Bf16 => DType::Bf16,
                    Gemma4MoeGraphTensorDtype::F32 => DType::F32,
                    Gemma4MoeGraphTensorDtype::Fp8E4M3 => DType::F8E4M3Fn,
                    Gemma4MoeGraphTensorDtype::U8 => DType::U8,
                };
                (
                    contiguous(dtype, &spec.stored_shape)?,
                    Gemma4MoeTensorBacking::ResidentWeight(entry.tensor_name.clone()),
                )
            };
            let id = builder.push_tensor(entry.tensor_name.clone(), view, backing);
            builder.weights.insert(entry.tensor_name.clone(), id);
        }
        builder.token_ids = builder.push_tensor(
            "request.token_ids",
            contiguous(DType::I32, &[graph.token_count()])?,
            Gemma4MoeTensorBacking::TokenIds,
        );
        builder.positions = builder.push_tensor(
            "request.positions",
            contiguous(DType::I32, &[graph.token_count()])?,
            Gemma4MoeTensorBacking::Positions,
        );
        Ok(builder)
    }

    fn push_tensor(
        &mut self,
        name: impl Into<String>,
        view: TensorView,
        backing: Gemma4MoeTensorBacking,
    ) -> usize {
        let id = self.tensors.len();
        self.tensors.push(Gemma4MoeExecutionTensor {
            id,
            name: name.into(),
            view,
            backing,
        });
        id
    }

    fn workspace(
        &mut self,
        name: impl Into<String>,
        view: TensorView,
    ) -> Result<usize, Gemma4MoeExecutionError> {
        self.workspace_bytes = self
            .workspace_bytes
            .checked_add(view.payload_bytes())
            .ok_or_else(|| Gemma4MoeExecutionError::invalid("workspace bytes overflowed"))?;
        Ok(self.push_tensor(name, view, Gemma4MoeTensorBacking::Workspace))
    }

    fn alias(
        &mut self,
        name: impl Into<String>,
        tensor: usize,
        view: TensorView,
    ) -> Result<usize, Gemma4MoeExecutionError> {
        if view.payload_bytes() != self.tensors[tensor].view.payload_bytes() {
            return Err(Gemma4MoeExecutionError::invalid(
                "execution alias changes payload bytes",
            ));
        }
        Ok(self.push_tensor(name, view, Gemma4MoeTensorBacking::Alias { tensor }))
    }

    fn constant(
        &mut self,
        label: &str,
        bits: u16,
        width: usize,
    ) -> Result<usize, Gemma4MoeExecutionError> {
        Ok(self.push_tensor(
            format!("{label}.constant"),
            TensorView::contiguous(DType::Bf16, &[width])
                .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?,
            Gemma4MoeTensorBacking::ConstantBf16 { bits, width },
        ))
    }

    fn weight(&self, name: &str) -> Result<usize, Gemma4MoeExecutionError> {
        self.weights.get(name).copied().ok_or_else(|| {
            Gemma4MoeExecutionError::invalid(format!("resident weight is absent: {name}"))
        })
    }

    fn predecessor(
        &self,
        node: usize,
        predecessor: usize,
        output: usize,
    ) -> Result<usize, Gemma4MoeExecutionError> {
        let predecessor = *self.graph.nodes()[node]
            .predecessors()
            .get(predecessor)
            .ok_or_else(|| Gemma4MoeExecutionError::invalid("graph predecessor is absent"))?;
        self.outputs[predecessor]
            .get(output)
            .copied()
            .ok_or_else(|| Gemma4MoeExecutionError::invalid("graph predecessor output is absent"))
    }

    fn views(&self, tensors: &[usize]) -> Vec<TensorView> {
        tensors
            .iter()
            .map(|tensor| self.tensors[*tensor].view.clone())
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn semantic(
        &mut self,
        graph_node_id: usize,
        stage: u8,
        label: impl Into<String>,
        descriptor: SemanticOpDescriptor,
        inputs: Vec<usize>,
        outputs: Vec<usize>,
        boundary_after: Option<Gemma4MoeExecutionBoundary>,
    ) {
        self.nodes.push(Gemma4MoeExecutionNode {
            graph_node_id,
            stage,
            label: label.into(),
            lowering: Gemma4MoeLowering::Semantic(Box::new(descriptor)),
            inputs,
            outputs,
            boundary_after,
        });
    }

    fn generic_semantic(
        &mut self,
        graph_node_id: usize,
        kind: SemanticOpKind,
        inputs: Vec<usize>,
        outputs: Vec<usize>,
    ) -> Result<(), Gemma4MoeExecutionError> {
        let descriptor = SemanticOpDescriptor::new(kind, self.views(&inputs), self.views(&outputs))
            .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
        let node = &self.graph.nodes()[graph_node_id];
        self.semantic(
            graph_node_id,
            0,
            node.label(),
            descriptor,
            inputs,
            outputs,
            node.boundary_after(),
        );
        Ok(())
    }

    fn output_shape(&self, node: usize) -> Result<TensorView, Gemma4MoeExecutionError> {
        match self.graph.nodes()[node].output_shape() {
            Gemma4MoeValueShape::Dense { rows, width } => contiguous(DType::Bf16, &[*rows, *width]),
            Gemma4MoeValueShape::TokenIndices { rows } => contiguous(DType::I32, &[*rows]),
            Gemma4MoeValueShape::Routes { rows, .. } => {
                let bytes = rows
                    .checked_mul(ROUTE_TOP_K)
                    .and_then(|pairs| pairs.checked_mul(16))
                    .and_then(|bytes| bytes.checked_add(ROUTE_EXPERT_COUNT * 4))
                    .and_then(|bytes| bytes.checked_add((ROUTE_EXPERT_COUNT + 1) * 4))
                    .and_then(|bytes| bytes.checked_add(4))
                    .ok_or_else(|| Gemma4MoeExecutionError::invalid("route metadata overflows"))?;
                contiguous(DType::U8, &[bytes])
            }
            Gemma4MoeValueShape::QueryAndKey { .. } => Err(Gemma4MoeExecutionError::invalid(
                "query-and-key shape requires rotary-specific lowering",
            )),
        }
    }

    fn lower(&mut self) -> Result<(), Gemma4MoeExecutionError> {
        for graph_node_id in 0..self.graph.nodes().len() {
            let node = &self.graph.nodes()[graph_node_id];
            let output = match node.kind() {
                Gemma4MoeGraphNodeKind::Embedding { weight } => {
                    let output = self.workspace(
                        format!("{}.output", node.label()),
                        self.output_shape(graph_node_id)?,
                    )?;
                    self.generic_semantic(
                        graph_node_id,
                        SemanticOpKind::Embedding,
                        vec![self.weight(weight)?, self.token_ids],
                        vec![output],
                    )?;
                    vec![output]
                }
                Gemma4MoeGraphNodeKind::ScaleConstant { value_bits } => {
                    let input = self.predecessor(graph_node_id, 0, 0)?;
                    let scalar = self.constant(
                        node.label(),
                        f32_to_bf16_rne(f32::from_bits(*value_bits)),
                        1,
                    )?;
                    let output = self.workspace(
                        format!("{}.output", node.label()),
                        self.output_shape(graph_node_id)?,
                    )?;
                    self.generic_semantic(
                        graph_node_id,
                        SemanticOpKind::ScalarMul,
                        vec![input, scalar],
                        vec![output],
                    )?;
                    vec![output]
                }
                Gemma4MoeGraphNodeKind::RmsNorm {
                    role,
                    epsilon_bits,
                    scale_mode,
                    weight,
                } => {
                    let source = self.predecessor(graph_node_id, 0, 0)?;
                    let (input, output_view, scale_width) = match role {
                        Gemma4MoeNormRole::Query => {
                            let layer = node.layer().ok_or_else(|| {
                                Gemma4MoeExecutionError::invalid("query norm has no layer")
                            })?;
                            let head_dim =
                                u64::from(self.graph.layers()[layer as usize].attention.head_dim);
                            let heads =
                                u64::from(self.graph.layers()[layer as usize].attention.q_heads);
                            let view = contiguous(
                                DType::Bf16,
                                &[self.graph.token_count() * heads, head_dim],
                            )?;
                            (
                                self.alias(
                                    format!("{}.input", node.label()),
                                    source,
                                    view.clone(),
                                )?,
                                view,
                                head_dim as usize,
                            )
                        }
                        Gemma4MoeNormRole::Key | Gemma4MoeNormRole::ValueUnitScale => {
                            let layer = node.layer().ok_or_else(|| {
                                Gemma4MoeExecutionError::invalid("KV norm has no layer")
                            })?;
                            let attention = self.graph.layers()[layer as usize].attention;
                            let head_dim = u64::from(attention.head_dim);
                            let heads = u64::from(attention.kv_heads);
                            let view = contiguous(
                                DType::Bf16,
                                &[self.graph.token_count() * heads, head_dim],
                            )?;
                            (
                                self.alias(
                                    format!("{}.input", node.label()),
                                    source,
                                    view.clone(),
                                )?,
                                view,
                                head_dim as usize,
                            )
                        }
                        _ => (
                            source,
                            self.output_shape(graph_node_id)?,
                            self.graph.config().hidden_size as usize,
                        ),
                    };
                    let scale = if *scale_mode == Gemma4MoeRmsScaleMode::NoAffineScale {
                        self.constant(node.label(), f32_to_bf16_rne(1.0), scale_width)?
                    } else {
                        self.weight(weight.as_deref().ok_or_else(|| {
                            Gemma4MoeExecutionError::invalid("affine RMSNorm weight is absent")
                        })?)?
                    };
                    let output = self.workspace(format!("{}.output", node.label()), output_view)?;
                    let descriptor = SemanticOpDescriptor::new_rms_norm(
                        self.views(&[input, scale]),
                        self.views(&[output]),
                        f32::from_bits(*epsilon_bits),
                        RmsNormScaleMode::Direct,
                    )
                    .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
                    self.semantic(
                        graph_node_id,
                        0,
                        node.label(),
                        descriptor,
                        vec![input, scale],
                        vec![output],
                        node.boundary_after(),
                    );
                    vec![output]
                }
                Gemma4MoeGraphNodeKind::Linear {
                    weight,
                    input_features,
                    output_features,
                    role,
                } => {
                    let source = self.predecessor(graph_node_id, 0, 0)?;
                    let input = self.alias(
                        format!("{}.input", node.label()),
                        source,
                        contiguous(DType::Bf16, &[self.graph.token_count(), *input_features])?,
                    )?;
                    let output = self.workspace(
                        format!("{}.output", node.label()),
                        contiguous(DType::Bf16, &[self.graph.token_count(), *output_features])?,
                    )?;
                    if *role == Gemma4MoeLinearRole::RouterProjection
                        && *output_features != ROUTE_EXPERT_COUNT
                    {
                        return Err(Gemma4MoeExecutionError::invalid(
                            "router projection width differs",
                        ));
                    }
                    self.generic_semantic(
                        graph_node_id,
                        SemanticOpKind::Matmul,
                        vec![input, self.weight(weight)?],
                        vec![output],
                    )?;
                    vec![output]
                }
                Gemma4MoeGraphNodeKind::Rotary(rope) => {
                    let q_source = self.predecessor(graph_node_id, 0, 0)?;
                    let k_source = self.predecessor(graph_node_id, 1, 0)?;
                    let q_view = contiguous(
                        DType::Bf16,
                        &[
                            self.graph.token_count(),
                            u64::from(rope.q_heads),
                            u64::from(rope.head_dim),
                        ],
                    )?;
                    let k_view = contiguous(
                        DType::Bf16,
                        &[
                            self.graph.token_count(),
                            u64::from(rope.kv_heads),
                            u64::from(rope.head_dim),
                        ],
                    )?;
                    let q = self.alias(format!("{}.q", node.label()), q_source, q_view.clone())?;
                    let k = self.alias(format!("{}.k", node.label()), k_source, k_view.clone())?;
                    let q_out = self.workspace(format!("{}.q_out", node.label()), q_view)?;
                    let k_out = self.workspace(format!("{}.k_out", node.label()), k_view)?;
                    let descriptor = SemanticOpDescriptor::new_rotary(
                        self.views(&[q, k, self.positions]),
                        self.views(&[q_out, k_out]),
                        SplitHalfRotaryContract::new(
                            rope.q_heads,
                            rope.kv_heads,
                            rope.head_dim,
                            rope.rotary_dim,
                            rope.theta as f32,
                            self.graph.start_position(),
                            self.graph.token_count(),
                            self.graph.config().max_position_embeddings,
                        )
                        .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?,
                    )
                    .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
                    self.semantic(
                        graph_node_id,
                        0,
                        node.label(),
                        descriptor,
                        vec![q, k, self.positions],
                        vec![q_out, k_out],
                        node.boundary_after(),
                    );
                    vec![q_out, k_out]
                }
                Gemma4MoeGraphNodeKind::CausalAttention(attention) => {
                    if attention.scaling_bits != 1.0_f32.to_bits() {
                        return Err(Gemma4MoeExecutionError::invalid(
                            "Gemma 4 MoE attention score scale must be exactly one",
                        ));
                    }
                    let q = self.predecessor(graph_node_id, 0, 0)?;
                    let k = self.predecessor(graph_node_id, 0, 1)?;
                    let v_source = self.predecessor(graph_node_id, 1, 0)?;
                    let v = self.alias(
                        format!("{}.v", node.label()),
                        v_source,
                        contiguous(
                            DType::Bf16,
                            &[
                                self.graph.token_count(),
                                u64::from(attention.kv_heads),
                                u64::from(attention.head_dim),
                            ],
                        )?,
                    )?;
                    let output = self.workspace(
                        format!("{}.output", node.label()),
                        contiguous(
                            DType::Bf16,
                            &[
                                self.graph.token_count(),
                                u64::from(attention.q_heads),
                                u64::from(attention.head_dim),
                            ],
                        )?,
                    )?;
                    let layer = node.layer().ok_or_else(|| {
                        Gemma4MoeExecutionError::invalid("attention node has no layer")
                    })?;
                    self.nodes.push(Gemma4MoeExecutionNode {
                        graph_node_id,
                        stage: 0,
                        label: node.label().to_owned(),
                        lowering: Gemma4MoeLowering::StaticFp8Attention(Gemma4MoeAttentionHook {
                            layer,
                            kv: self.graph.layers()[layer as usize].kv,
                            score_scale_bits: attention.scaling_bits,
                            sliding_api_required: attention.sliding_window.is_some(),
                        }),
                        inputs: vec![q, k, v],
                        outputs: vec![output],
                        boundary_after: node.boundary_after(),
                    });
                    vec![output]
                }
                Gemma4MoeGraphNodeKind::GeluTanhMul => {
                    let inputs = vec![
                        self.predecessor(graph_node_id, 0, 0)?,
                        self.predecessor(graph_node_id, 1, 0)?,
                    ];
                    let output = self.workspace(
                        format!("{}.output", node.label()),
                        self.output_shape(graph_node_id)?,
                    )?;
                    self.generic_semantic(
                        graph_node_id,
                        SemanticOpKind::GeluTanhMul,
                        inputs,
                        vec![output],
                    )?;
                    vec![output]
                }
                Gemma4MoeGraphNodeKind::RouterRootScale {
                    scale_weight,
                    hidden_root_reciprocal_bits,
                } => {
                    let input = self.predecessor(graph_node_id, 0, 0)?;
                    let multiplied = self.workspace(
                        format!("{}.broadcast", node.label()),
                        self.output_shape(graph_node_id)?,
                    )?;
                    let broadcast = SemanticOpDescriptor::new(
                        SemanticOpKind::BroadcastMul,
                        self.views(&[input, self.weight(scale_weight)?]),
                        self.views(&[multiplied]),
                    )
                    .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
                    self.semantic(
                        graph_node_id,
                        0,
                        format!("{}.broadcast", node.label()),
                        broadcast,
                        vec![input, self.weight(scale_weight)?],
                        vec![multiplied],
                        None,
                    );
                    let scalar = self.constant(
                        node.label(),
                        f32_to_bf16_rne(f32::from_bits(*hidden_root_reciprocal_bits)),
                        1,
                    )?;
                    let output = self.workspace(
                        format!("{}.output", node.label()),
                        self.output_shape(graph_node_id)?,
                    )?;
                    let descriptor = SemanticOpDescriptor::new(
                        SemanticOpKind::ScalarMul,
                        self.views(&[multiplied, scalar]),
                        self.views(&[output]),
                    )
                    .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
                    self.semantic(
                        graph_node_id,
                        1,
                        format!("{}.root_scalar", node.label()),
                        descriptor,
                        vec![multiplied, scalar],
                        vec![output],
                        node.boundary_after(),
                    );
                    vec![output]
                }
                Gemma4MoeGraphNodeKind::StableTopKRouter {
                    expert_count,
                    top_k,
                    full_softmax,
                    renormalize_selected_weights,
                    stable_tie_break_by_lower_expert,
                    ..
                } => {
                    if (
                        *expert_count,
                        *top_k,
                        *full_softmax,
                        *renormalize_selected_weights,
                        *stable_tie_break_by_lower_expert,
                    ) != (128, 8, true, true, true)
                    {
                        return Err(Gemma4MoeExecutionError::invalid(
                            "router semantic contract differs",
                        ));
                    }
                    let input = self.predecessor(graph_node_id, 0, 0)?;
                    let output = self.workspace(
                        format!("{}.metadata", node.label()),
                        self.output_shape(graph_node_id)?,
                    )?;
                    self.generic_semantic(
                        graph_node_id,
                        SemanticOpKind::MoeRoute,
                        vec![input],
                        vec![output],
                    )?;
                    vec![output]
                }
                Gemma4MoeGraphNodeKind::RoutedExpertsNvfp4 {
                    family,
                    only_selected_experts,
                    apply_scale_after_topk_renormalization,
                    ..
                } => {
                    if !*only_selected_experts || !*apply_scale_after_topk_renormalization {
                        return Err(Gemma4MoeExecutionError::invalid(
                            "expert scaling contract differs",
                        ));
                    }
                    let hidden = self.predecessor(graph_node_id, 0, 0)?;
                    let routes = self.predecessor(graph_node_id, 1, 0)?;
                    let blob = self.weight(&gemma4_moe_layer_blob_name(family.layer))?;
                    let output = self.workspace(
                        format!("{}.output", node.label()),
                        self.output_shape(graph_node_id)?,
                    )?;
                    self.generic_semantic(
                        graph_node_id,
                        SemanticOpKind::MoeExpert,
                        vec![hidden, routes, blob],
                        vec![output],
                    )?;
                    vec![output]
                }
                Gemma4MoeGraphNodeKind::Add { .. } => {
                    let inputs = vec![
                        self.predecessor(graph_node_id, 0, 0)?,
                        self.predecessor(graph_node_id, 1, 0)?,
                    ];
                    let output = self.workspace(
                        format!("{}.output", node.label()),
                        self.output_shape(graph_node_id)?,
                    )?;
                    self.generic_semantic(
                        graph_node_id,
                        SemanticOpKind::Add,
                        inputs,
                        vec![output],
                    )?;
                    vec![output]
                }
                Gemma4MoeGraphNodeKind::ScaleWeight { weight } => {
                    let input = self.predecessor(graph_node_id, 0, 0)?;
                    let output = self.workspace(
                        format!("{}.output", node.label()),
                        self.output_shape(graph_node_id)?,
                    )?;
                    self.generic_semantic(
                        graph_node_id,
                        SemanticOpKind::ScalarMul,
                        vec![input, self.weight(weight)?],
                        vec![output],
                    )?;
                    vec![output]
                }
                Gemma4MoeGraphNodeKind::TiedOutputProjection { embedding_weight } => {
                    let input = self.predecessor(graph_node_id, 0, 0)?;
                    let output = self.workspace(
                        format!("{}.output", node.label()),
                        self.output_shape(graph_node_id)?,
                    )?;
                    self.generic_semantic(
                        graph_node_id,
                        SemanticOpKind::Matmul,
                        vec![input, self.weight(embedding_weight)?],
                        vec![output],
                    )?;
                    vec![output]
                }
                Gemma4MoeGraphNodeKind::LogitSoftcap { cap_bits } => {
                    let input = self.predecessor(graph_node_id, 0, 0)?;
                    let scalar =
                        self.constant(node.label(), f32_to_bf16_rne(f32::from_bits(*cap_bits)), 1)?;
                    let output = self.workspace(
                        format!("{}.output", node.label()),
                        self.output_shape(graph_node_id)?,
                    )?;
                    self.generic_semantic(
                        graph_node_id,
                        SemanticOpKind::TanhSoftcap,
                        vec![input, scalar],
                        vec![output],
                    )?;
                    vec![output]
                }
                Gemma4MoeGraphNodeKind::Argmax => {
                    let input = self.predecessor(graph_node_id, 0, 0)?;
                    let output = self.workspace(
                        format!("{}.output", node.label()),
                        self.output_shape(graph_node_id)?,
                    )?;
                    self.generic_semantic(
                        graph_node_id,
                        SemanticOpKind::Argmax,
                        vec![input],
                        vec![output],
                    )?;
                    vec![output]
                }
            };
            self.outputs[graph_node_id] = output;
        }
        Ok(())
    }

    fn finish(self) -> Result<Gemma4MoeExecutionLayout, Gemma4MoeExecutionError> {
        let mut segments = Vec::new();
        let mut first = 0;
        for (index, node) in self.nodes.iter().enumerate() {
            if let Some(boundary) = node.boundary_after {
                segments.push(Gemma4MoeExecutionSegment {
                    first_node: first,
                    end_node: index + 1,
                    boundary,
                });
                first = index + 1;
            }
        }
        if first != self.nodes.len() || segments.len() != 2 {
            return Err(Gemma4MoeExecutionError::invalid(
                "execution boundaries do not form state-publication and readback segments",
            ));
        }
        let terminal_readback_tensor = *self
            .outputs
            .last()
            .and_then(|outputs| outputs.first())
            .ok_or_else(|| {
                Gemma4MoeExecutionError::invalid("terminal readback tensor is absent")
            })?;
        Ok(Gemma4MoeExecutionLayout {
            model_fingerprint: self.graph.model_fingerprint().to_owned(),
            source_container_identity: self.graph.source_container_identity().to_owned(),
            plan_digest: *self.plan.digest(),
            token_count: self.graph.token_count(),
            tensors: self.tensors,
            nodes: self.nodes,
            segments,
            transitions: plan_gemma4_moe_transitions(
                self.graph.start_position(),
                self.graph.token_count(),
            )?,
            terminal_readback_tensor,
            resident_weight_bytes: self.plan.total_destination_bytes,
            workspace_bytes: self.workspace_bytes,
        })
    }
}

pub fn build_gemma4_moe_execution_layout(
    graph: &Gemma4MoeGraph,
    plan: &WeightLoadPlan,
) -> Result<Gemma4MoeExecutionLayout, Gemma4MoeExecutionError> {
    let mut builder = LayoutBuilder::new(graph, plan)?;
    builder.lower()?;
    builder.finish()
}

fn copy_into(
    output: &mut [u8],
    range: [u64; 2],
    bytes: &[u8],
) -> Result<(), Gemma4MoeExecutionError> {
    let start = usize::try_from(range[0])
        .map_err(|_| Gemma4MoeExecutionError::invalid("blob offset exceeds usize"))?;
    let end = usize::try_from(range[1])
        .map_err(|_| Gemma4MoeExecutionError::invalid("blob offset exceeds usize"))?;
    let destination = output
        .get_mut(start..end)
        .ok_or_else(|| Gemma4MoeExecutionError::invalid("blob destination exceeds layout"))?;
    if destination.len() != bytes.len() {
        return Err(Gemma4MoeExecutionError::invalid(
            "blob source and destination lengths differ",
        ));
    }
    destination.copy_from_slice(bytes);
    Ok(())
}

pub fn pack_gemma4_moe_layer_blob(
    source: &dyn Gemma4MoeWeightSource,
    layer: u32,
) -> Result<Vec<u8>, Gemma4MoeExecutionError> {
    let mut blob = vec![
        0_u8;
        usize::try_from(GEMMA4_MOE_LAYER_BLOB_BYTES).map_err(|_| {
            Gemma4MoeExecutionError::invalid("layer blob exceeds usize")
        })?
    ];
    for input in gemma4_moe_layer_blob_pack_inputs(layer)? {
        let planes = source.read_expert_planes(
            u16::try_from(layer)
                .map_err(|_| Gemma4MoeExecutionError::invalid("layer exceeds u16"))?,
            input.expert,
            input.projection,
        )?;
        copy_into(&mut blob, input.value_destination, &planes.values)?;
        copy_into(
            &mut blob,
            input.block_scale_destination,
            &planes.block_scales,
        )?;
        copy_into(
            &mut blob,
            input.outer_scale_destination,
            &planes.outer_scale,
        )?;
        copy_into(
            &mut blob,
            input.input_scale_destination,
            &planes.input_scale,
        )?;
    }
    let scale_name = format!("model.language_model.layers.{layer}.router.per_expert_scale");
    let scales = source.read_direct_tensor(&scale_name)?;
    copy_into(
        &mut blob,
        gemma4_moe_per_expert_scale_destination(),
        &scales,
    )?;
    if GEMMA4_MOE_PER_EXPERT_SCALES_OFFSET + scales.len() as u64 != GEMMA4_MOE_LAYER_BLOB_BYTES {
        return Err(Gemma4MoeExecutionError::invalid(
            "per-expert scale bytes do not terminate the layer blob",
        ));
    }
    Ok(blob)
}

fn require_transfer_success(
    state: Result<ExecutionState, crate::ExecutionError>,
    label: &str,
) -> Result<(), Gemma4MoeExecutionError> {
    match state.map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))? {
        ExecutionState::Success => Ok(()),
        ExecutionState::Pending => Err(Gemma4MoeExecutionError::invalid(format!(
            "{label} remained pending"
        ))),
        ExecutionState::Failure => Err(Gemma4MoeExecutionError::invalid(format!(
            "{label} reported failure"
        ))),
    }
}

fn upload_bytes(
    session: &ExecutionSession,
    queue: &ExecutionQueue,
    buffer: &ExecutionBuffer,
    bytes: &[u8],
    timeout: Duration,
) -> Result<(), Gemma4MoeExecutionError> {
    if bytes.len() as u64 != buffer.size_bytes() {
        return Err(Gemma4MoeExecutionError::invalid(
            "resident upload bytes differ from allocation",
        ));
    }
    let limit = usize::try_from(
        session
            .max_transfer_bytes()
            .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?,
    )
    .map_err(|_| Gemma4MoeExecutionError::invalid("backend transfer limit exceeds usize"))?;
    let mut offset = 0_u64;
    for chunk in bytes.chunks(limit) {
        let length = chunk.len() as u64;
        let range = buffer
            .range(offset, length)
            .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
        let mut transfer = session
            .upload(queue, range, Arc::from(chunk))
            .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
        require_transfer_success(transfer.wait(timeout), "Gemma 4 MoE resident upload")?;
        offset += length;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gemma4MoeResidentAudit {
    resident_allocations: usize,
    direct_weight_allocations: usize,
    expert_blob_allocations: usize,
    individual_expert_allocations: usize,
    resident_bytes: u64,
}

impl Gemma4MoeResidentAudit {
    pub const fn resident_allocations(self) -> usize {
        self.resident_allocations
    }
    pub const fn direct_weight_allocations(self) -> usize {
        self.direct_weight_allocations
    }
    pub const fn expert_blob_allocations(self) -> usize {
        self.expert_blob_allocations
    }
    pub const fn individual_expert_allocations(self) -> usize {
        self.individual_expert_allocations
    }
    pub const fn resident_bytes(self) -> u64 {
        self.resident_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Gemma4MoePrefixIdentityV1 {
    model_fingerprint: String,
    source_container_identity: String,
    plan_digest: [u8; 32],
    config_digest: [u8; 32],
    state_capacity: u64,
}

/// One versioned, backend-neutral opaque KV layer in a Gemma 4 MoE image.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MoeKvStateImageV1 {
    descriptor: KvStateDescriptor,
    image: ExecutionStateImageV1,
}

impl Gemma4MoeKvStateImageV1 {
    pub const fn descriptor(&self) -> KvStateDescriptor {
        self.descriptor
    }

    pub const fn image(&self) -> &ExecutionStateImageV1 {
        &self.image
    }
}

/// Complete state of all 30 opaque Gemma 4 MoE KV layers.  Sliding layers
/// carry the backend's retained W+1 ring image while full-attention layers
/// carry their complete prefix; native state handles are never serialized.
#[derive(Clone, PartialEq)]
pub struct Gemma4MoeStateImageV1 {
    session_id: ExecutionSessionId,
    identity: Gemma4MoePrefixIdentityV1,
    committed_length: u64,
    kv_layers: BTreeMap<u32, Gemma4MoeKvStateImageV1>,
    cached_terminal_output: Option<Gemma4MoeExecutionOutput>,
}

impl fmt::Debug for Gemma4MoeStateImageV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Gemma4MoeStateImageV1")
            .field("session_id", &self.session_id)
            .field("identity", &"<redacted>")
            .field("committed_length", &self.committed_length)
            .field("kv_layer_count", &self.kv_layers.len())
            .field(
                "has_cached_terminal_output",
                &self.cached_terminal_output.is_some(),
            )
            .finish()
    }
}

impl Gemma4MoeStateImageV1 {
    pub const fn session_id(&self) -> ExecutionSessionId {
        self.session_id
    }

    pub const fn committed_length(&self) -> u64 {
        self.committed_length
    }

    pub fn model_fingerprint(&self) -> &str {
        &self.identity.model_fingerprint
    }

    pub fn source_container_identity(&self) -> &str {
        &self.identity.source_container_identity
    }

    pub fn plan_digest(&self) -> &[u8; 32] {
        &self.identity.plan_digest
    }

    pub fn config_digest(&self) -> &[u8; 32] {
        &self.identity.config_digest
    }

    pub const fn state_capacity(&self) -> u64 {
        self.identity.state_capacity
    }

    pub const fn kv_layers(&self) -> &BTreeMap<u32, Gemma4MoeKvStateImageV1> {
        &self.kv_layers
    }

    pub const fn cached_terminal_output(&self) -> Option<&Gemma4MoeExecutionOutput> {
        self.cached_terminal_output.as_ref()
    }

    pub fn without_terminal_output(mut self) -> Self {
        self.cached_terminal_output = None;
        self
    }

    pub fn kv_descriptor_digest(&self) -> [u8; 32] {
        gemma4_moe_kv_descriptor_digest(
            &self.identity,
            self.kv_layers
                .iter()
                .map(|(layer, image)| (*layer, image.descriptor)),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn to_checkpoint(
        &self,
        identity: CheckpointIdentity,
        token_history: &[u32],
        conversation: &[u8],
        sampler_state: &[u8],
        grammar_state: &[u8],
        stop_state: &[u8],
        absolute_position: u64,
        logical_position: u64,
        generation_state_version: u32,
    ) -> Result<SessionCheckpoint, Gemma4MoeExecutionError> {
        validate_gemma4_moe_state_image_topology(self)?;
        if token_history.len() as u64 != self.committed_length
            || logical_position != self.committed_length
            || absolute_position != logical_position
        {
            return Err(Gemma4MoeExecutionError::invalid(
                "checkpoint positions or token history differ from Gemma 4 MoE state",
            ));
        }
        if identity.model_lock_fingerprint != self.identity.model_fingerprint
            || identity.plan_digest != gemma4_moe_hex_digest(&self.identity.plan_digest)
            || identity.kv_descriptor_digest != self.kv_descriptor_digest()
            || self
                .kv_layers
                .values()
                .any(|layer| layer.descriptor.cache_encoding() != identity.kv_encoding)
        {
            return Err(Gemma4MoeExecutionError::invalid(
                "checkpoint identity differs from Gemma 4 MoE model, source/config, plan, or KV descriptors",
            ));
        }
        let mut state_layers = Vec::with_capacity(self.kv_layers.len());
        let mut state_planes = Vec::with_capacity(self.kv_layers.len() * 2);
        for layer in self.kv_layers.values() {
            if layer.image.metadata().published_length != logical_position {
                return Err(Gemma4MoeExecutionError::invalid(
                    "checkpoint KV length differs from Gemma 4 MoE logical position",
                ));
            }
            state_layers.push(layer.image.metadata().clone());
            state_planes.extend(layer.image.planes().iter().cloned());
        }
        SessionCheckpoint::new(
            identity,
            absolute_position,
            logical_position,
            generation_state_version,
            CheckpointPayload {
                token_history: token_history.to_vec(),
                conversation: conversation.to_vec(),
                state_layers,
                state_planes,
                sampler_state: sampler_state.to_vec(),
                grammar_state: grammar_state.to_vec(),
                stop_state: stop_state.to_vec(),
            },
        )
        .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn checkpoint(
        &self,
        identity: CheckpointIdentity,
        token_history: &[u32],
        conversation: &[u8],
        sampler_state: &[u8],
        grammar_state: &[u8],
        stop_state: &[u8],
        absolute_position: u64,
        logical_position: u64,
        generation_state_version: u32,
    ) -> Result<SessionCheckpoint, Gemma4MoeExecutionError> {
        self.to_checkpoint(
            identity,
            token_history,
            conversation,
            sampler_state,
            grammar_state,
            stop_state,
            absolute_position,
            logical_position,
            generation_state_version,
        )
    }
}

/// Aggregated, redacted ownership evidence for all 30 prefix forks.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Gemma4MoePrefixForkAuditV1 {
    kv_states: u32,
    sliding_states: u32,
    full_states: u32,
    shared_pages: u64,
    copied_bytes: u64,
    destination_owned_bytes: u64,
    cache_resident_bytes: u64,
}

impl Gemma4MoePrefixForkAuditV1 {
    pub const fn kv_states(self) -> u32 {
        self.kv_states
    }
    pub const fn sliding_states(self) -> u32 {
        self.sliding_states
    }
    pub const fn full_states(self) -> u32 {
        self.full_states
    }
    pub const fn shared_pages(self) -> u64 {
        self.shared_pages
    }
    pub const fn copied_bytes(self) -> u64 {
        self.copied_bytes
    }
    pub const fn destination_owned_bytes(self) -> u64 {
        self.destination_owned_bytes
    }
    pub const fn cache_resident_bytes(self) -> u64 {
        self.cache_resident_bytes
    }

    fn add(
        &mut self,
        audit: StateForkAuditV1,
        sliding: bool,
        physical: Option<KvPhysicalMemorySnapshot>,
        fallback_resident_bytes: u64,
    ) -> Result<(), Gemma4MoeExecutionError> {
        self.kv_states = self
            .kv_states
            .checked_add(1)
            .ok_or_else(|| Gemma4MoeExecutionError::invalid("KV fork count overflowed"))?;
        if sliding {
            self.sliding_states = self.sliding_states.checked_add(1).ok_or_else(|| {
                Gemma4MoeExecutionError::invalid("sliding KV fork count overflowed")
            })?;
        } else {
            self.full_states = self
                .full_states
                .checked_add(1)
                .ok_or_else(|| Gemma4MoeExecutionError::invalid("full KV fork count overflowed"))?;
        }
        self.shared_pages = self
            .shared_pages
            .checked_add(audit.shared_pages())
            .ok_or_else(|| Gemma4MoeExecutionError::invalid("shared page count overflowed"))?;
        self.copied_bytes = self
            .copied_bytes
            .checked_add(audit.copied_bytes())
            .ok_or_else(|| Gemma4MoeExecutionError::invalid("copied byte count overflowed"))?;
        self.destination_owned_bytes = self
            .destination_owned_bytes
            .checked_add(audit.destination_owned_bytes())
            .ok_or_else(|| Gemma4MoeExecutionError::invalid("owned byte count overflowed"))?;
        let resident_bytes = match audit.mode() {
            StateForkModeV1::SharedReadOnlyPages => physical
                .and_then(|snapshot| snapshot.committed_bytes_per_plane().checked_mul(2))
                .unwrap_or(fallback_resident_bytes),
            StateForkModeV1::DeviceCopy => audit.destination_owned_bytes(),
        };
        self.cache_resident_bytes = self
            .cache_resident_bytes
            .checked_add(resident_bytes)
            .ok_or_else(|| Gemma4MoeExecutionError::invalid("resident byte count overflowed"))?;
        Ok(())
    }
}

struct Gemma4MoePrefixStateInner {
    session: Arc<ExecutionSession>,
    identity: Gemma4MoePrefixIdentityV1,
    committed_length: u64,
    kv_states: BTreeMap<u32, KvState>,
    cached_terminal_output: Gemma4MoeExecutionOutput,
    fork_audit: Gemma4MoePrefixForkAuditV1,
}

/// Immutable, same-resident Gemma 4 MoE prefix owner. Requests always fork
/// its states and never mutate the prefix owner directly.
#[derive(Clone)]
pub struct Gemma4MoePrefixStateV1 {
    inner: Arc<Gemma4MoePrefixStateInner>,
}

impl fmt::Debug for Gemma4MoePrefixStateV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Gemma4MoePrefixStateV1")
            .field("session_id", &self.inner.session.id())
            .field("committed_length", &self.inner.committed_length)
            .field("state_capacity", &self.inner.identity.state_capacity)
            .field("fork_audit", &self.inner.fork_audit)
            .finish_non_exhaustive()
    }
}

impl Gemma4MoePrefixStateV1 {
    pub fn committed_length(&self) -> u64 {
        self.inner.committed_length
    }
    pub fn state_capacity(&self) -> u64 {
        self.inner.identity.state_capacity
    }
    pub fn fork_audit(&self) -> Gemma4MoePrefixForkAuditV1 {
        self.inner.fork_audit
    }
    pub fn model_fingerprint(&self) -> &str {
        &self.inner.identity.model_fingerprint
    }
    pub fn source_container_identity(&self) -> &str {
        &self.inner.identity.source_container_identity
    }
    pub fn plan_digest(&self) -> &[u8; 32] {
        &self.inner.identity.plan_digest
    }
    pub fn config_digest(&self) -> &[u8; 32] {
        &self.inner.identity.config_digest
    }
    pub fn cached_terminal_output(&self) -> &Gemma4MoeExecutionOutput {
        &self.inner.cached_terminal_output
    }
}

struct Gemma4MoeResidentInner {
    session: Arc<ExecutionSession>,
    queue: ExecutionQueue,
    config: Gemma4MoeConfig,
    source_container_identity: String,
    plan: WeightLoadPlan,
    buffers: BTreeMap<String, ExecutionBuffer>,
    audit: Gemma4MoeResidentAudit,
    completion_timeout: Duration,
}

#[derive(Clone)]
pub struct Gemma4MoeResidentModel {
    inner: Arc<Gemma4MoeResidentInner>,
}

impl fmt::Debug for Gemma4MoeResidentModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Gemma4MoeResidentModel")
            .field(
                "source_container_identity",
                &self.inner.source_container_identity,
            )
            .field("audit", &self.inner.audit)
            .finish_non_exhaustive()
    }
}

impl Gemma4MoeResidentModel {
    pub fn provision<S: Gemma4MoeWeightSource + 'static>(
        session: Arc<ExecutionSession>,
        source: Arc<S>,
        plan: WeightLoadPlan,
        completion_timeout: Duration,
    ) -> Result<Self, Gemma4MoeExecutionError> {
        let canonical = build_gemma4_moe_resident_weight_load_plan(source.as_ref())?;
        if !plan
            .has_valid_digest()
            .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?
            || plan != canonical
            || completion_timeout.is_zero()
        {
            return Err(Gemma4MoeExecutionError::invalid(
                "resident source and weight plan differ",
            ));
        }
        let queue = session
            .create_queue()
            .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
        let mut buffers = BTreeMap::new();
        let mut direct = 0_usize;
        let mut blobs = 0_usize;
        let mut resident_bytes = 0_u64;
        for entry in &plan.entries {
            if buffers.contains_key(&entry.tensor_name) {
                return Err(Gemma4MoeExecutionError::invalid(
                    "resident tensor would be uploaded twice",
                ));
            }
            if is_expert_source_tensor(&entry.tensor_name)
                || is_embedded_per_expert_scale(&entry.tensor_name)
            {
                return Err(Gemma4MoeExecutionError::invalid(
                    "individual expert residency is forbidden",
                ));
            }
            let bytes = if let Some(layer) = entry
                .tensor_name
                .strip_prefix(GEMMA4_MOE_LAYER_BLOB_PREFIX)
                .and_then(|layer| layer.parse::<u32>().ok())
            {
                blobs += 1;
                pack_gemma4_moe_layer_blob(source.as_ref(), layer)?
            } else {
                direct += 1;
                source.read_direct_tensor(&entry.tensor_name)?
            };
            let size = bytes.len() as u64;
            let buffer = session
                .allocate_with_category(size, AllocationCategory::ModelResident)
                .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
            upload_bytes(
                session.as_ref(),
                &queue,
                &buffer,
                &bytes,
                completion_timeout,
            )?;
            resident_bytes = resident_bytes
                .checked_add(size)
                .ok_or_else(|| Gemma4MoeExecutionError::invalid("resident bytes overflowed"))?;
            buffers.insert(entry.tensor_name.clone(), buffer);
        }
        if blobs != source.config().layer_count as usize
            || resident_bytes != plan.total_destination_bytes
        {
            return Err(Gemma4MoeExecutionError::invalid(
                "resident allocation accounting differs",
            ));
        }
        let audit = Gemma4MoeResidentAudit {
            resident_allocations: buffers.len(),
            direct_weight_allocations: direct,
            expert_blob_allocations: blobs,
            individual_expert_allocations: 0,
            resident_bytes,
        };
        Ok(Self {
            inner: Arc::new(Gemma4MoeResidentInner {
                session,
                queue,
                config: source.config().clone(),
                source_container_identity: source.source_container_identity().to_owned(),
                plan,
                buffers,
                audit,
                completion_timeout,
            }),
        })
    }

    pub fn audit(&self) -> Gemma4MoeResidentAudit {
        self.inner.audit
    }

    pub fn new_request(
        &self,
        graph: Gemma4MoeGraph,
    ) -> Result<Gemma4MoeExecutionRequest, Gemma4MoeExecutionError> {
        if graph.model_fingerprint() != GEMMA4_MOE_MODEL_FINGERPRINT
            || graph.source_container_identity() != self.inner.source_container_identity
        {
            return Err(Gemma4MoeExecutionError::invalid(
                "resident model and request graph identities differ",
            ));
        }
        let layout = build_gemma4_moe_execution_layout(&graph, &self.inner.plan)?;
        let provisioned = provision_gemma4_moe_request(&self.inner, &layout, &graph, None)?;
        Ok(Gemma4MoeExecutionRequest {
            _resident: Arc::clone(&self.inner),
            state: Gemma4MoeRequestState::fresh(&graph)?,
            layout,
            buffers: provisioned.buffers,
            _prepared_cache: provisioned.prepared_cache,
            prepared: provisioned.prepared,
            kv_states: provisioned.kv_states,
            transition_committed: false,
            poisoned: false,
            last_output: None,
        })
    }

    /// Creates a fresh request and transactionally imports all 30 opaque KV
    /// layers from a same-session image. The supplied graph must be a fresh
    /// start-position-zero graph; continuation is rebuilt as M=1 on demand.
    pub fn new_request_from_state_image(
        &self,
        image: &Gemma4MoeStateImageV1,
        graph: Gemma4MoeGraph,
    ) -> Result<Gemma4MoeExecutionRequest, Gemma4MoeExecutionError> {
        let mut request = self.new_request(graph)?;
        request.restore_state_image(image)?;
        Ok(request)
    }

    pub fn restore_request_from_state_image(
        &self,
        image: &Gemma4MoeStateImageV1,
        graph: Gemma4MoeGraph,
    ) -> Result<Gemma4MoeExecutionRequest, Gemma4MoeExecutionError> {
        self.new_request_from_state_image(image, graph)
    }

    pub fn request_from_state_image(
        &self,
        image: &Gemma4MoeStateImageV1,
        graph: Gemma4MoeGraph,
    ) -> Result<Gemma4MoeExecutionRequest, Gemma4MoeExecutionError> {
        self.new_request_from_state_image(image, graph)
    }

    /// Forks a same-session immutable prefix into new state identities. The
    /// source owner remains quiescent and is never reused as mutable state.
    pub fn new_request_from_prefix(
        &self,
        prefix: &Gemma4MoePrefixStateV1,
        graph: Gemma4MoeGraph,
    ) -> Result<Gemma4MoeExecutionRequest, Gemma4MoeExecutionError> {
        let mut request = self.new_request(graph)?;
        request.install_prefix(prefix)?;
        Ok(request)
    }

    pub fn request_from_prefix(
        &self,
        prefix: &Gemma4MoePrefixStateV1,
        graph: Gemma4MoeGraph,
    ) -> Result<Gemma4MoeExecutionRequest, Gemma4MoeExecutionError> {
        self.new_request_from_prefix(prefix, graph)
    }

    pub fn new_request_with_prefix(
        &self,
        graph: Gemma4MoeGraph,
        prefix: &Gemma4MoePrefixStateV1,
    ) -> Result<Gemma4MoeExecutionRequest, Gemma4MoeExecutionError> {
        self.new_request_from_prefix(prefix, graph)
    }

    /// Restores a portable checkpoint into a fresh request. Unlike raw state
    /// images, checkpoints may cross execution sessions after exact frontend,
    /// source/config, plan, and descriptor validation.
    pub fn new_request_from_checkpoint(
        &self,
        checkpoint: &SessionCheckpoint,
        graph: Gemma4MoeGraph,
        expected_identity: &CheckpointIdentity,
    ) -> Result<Gemma4MoeExecutionRequest, Gemma4MoeExecutionError> {
        let mut request = self.new_request(graph)?;
        request.restore_checkpoint(checkpoint, expected_identity)?;
        Ok(request)
    }

    pub fn restore_request_from_checkpoint(
        &self,
        checkpoint: &SessionCheckpoint,
        graph: Gemma4MoeGraph,
        expected_identity: &CheckpointIdentity,
    ) -> Result<Gemma4MoeExecutionRequest, Gemma4MoeExecutionError> {
        self.new_request_from_checkpoint(checkpoint, graph, expected_identity)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MoeOpaqueKvState {
    descriptor: Gemma4MoeKvDescriptor,
    committed_length: u64,
}

impl Gemma4MoeOpaqueKvState {
    pub const fn descriptor(&self) -> Gemma4MoeKvDescriptor {
        self.descriptor
    }
    pub const fn committed_length(&self) -> u64 {
        self.committed_length
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MoeRequestState {
    start_position: u64,
    expected_length: u64,
    layers: Vec<Gemma4MoeOpaqueKvState>,
}

impl Gemma4MoeRequestState {
    fn fresh(graph: &Gemma4MoeGraph) -> Result<Self, Gemma4MoeExecutionError> {
        let layers = graph
            .kv_descriptors()
            .map(|descriptor| Gemma4MoeOpaqueKvState {
                descriptor: *descriptor,
                committed_length: graph.start_position(),
            })
            .collect::<Vec<_>>();
        if layers.len() != graph.config().layer_count as usize
            || layers.iter().any(|state| {
                state.descriptor.dequant_scale_f32_bits != 1.0_f32.to_bits()
                    || state.descriptor.serialized_scale_tensor_count != 0
            })
        {
            return Err(Gemma4MoeExecutionError::invalid(
                "opaque static FP8 request-state contract differs",
            ));
        }
        Ok(Self {
            start_position: graph.start_position(),
            expected_length: graph.expected_length(),
            layers,
        })
    }

    pub const fn start_position(&self) -> u64 {
        self.start_position
    }
    pub const fn expected_length(&self) -> u64 {
        self.expected_length
    }
    pub fn layers(&self) -> &[Gemma4MoeOpaqueKvState] {
        &self.layers
    }
}

struct ProvisionedGemma4MoeRequest {
    buffers: Vec<ExecutionBuffer>,
    prepared_cache: Arc<PreparedSemanticCache>,
    prepared: Vec<Option<PreparedOperation>>,
    kv_states: Vec<KvState>,
}

fn provision_gemma4_moe_request(
    resident: &Arc<Gemma4MoeResidentInner>,
    layout: &Gemma4MoeExecutionLayout,
    graph: &Gemma4MoeGraph,
    retained_kv_states: Option<&[KvState]>,
) -> Result<ProvisionedGemma4MoeRequest, Gemma4MoeExecutionError> {
    if retained_kv_states.is_none() && graph.start_position() != 0 {
        return Err(Gemma4MoeExecutionError::invalid(
            "a fresh Gemma 4 MoE request must start at position zero",
        ));
    }
    let kv_states = if let Some(states) = retained_kv_states {
        states.to_vec()
    } else {
        let mut states = Vec::with_capacity(graph.config().layer_count as usize);
        for kv in graph.kv_descriptors() {
            let descriptor = if let Some(window) = kv.retention_window {
                KvStateDescriptor::new_with_static_fp8_sliding(
                    kv.layer,
                    kv.capacity,
                    kv.heads as usize,
                    kv.head_dim as usize,
                    window,
                )
            } else {
                KvStateDescriptor::new_with_static_fp8(
                    kv.layer,
                    kv.capacity,
                    kv.heads as usize,
                    kv.head_dim as usize,
                    1.0,
                    1.0,
                )
            }
            .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
            states.push(
                resident
                    .session
                    .create_kv_state(descriptor)
                    .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?,
            );
        }
        states
    };
    let mut buffers = Vec::with_capacity(layout.tensors.len());
    for tensor in &layout.tensors {
        let buffer = match tensor.backing() {
            Gemma4MoeTensorBacking::ResidentWeight(name) => {
                resident.buffers.get(name).cloned().ok_or_else(|| {
                    Gemma4MoeExecutionError::invalid(format!(
                        "resident request weight is absent: {name}"
                    ))
                })?
            }
            Gemma4MoeTensorBacking::ResidentExpertBlob { layer } => resident
                .buffers
                .get(&gemma4_moe_layer_blob_name(*layer))
                .cloned()
                .ok_or_else(|| {
                    Gemma4MoeExecutionError::invalid(format!(
                        "resident request expert blob is absent: {layer}"
                    ))
                })?,
            Gemma4MoeTensorBacking::Alias { tensor } => {
                buffers.get(*tensor).cloned().ok_or_else(|| {
                    Gemma4MoeExecutionError::invalid("request alias source is absent")
                })?
            }
            Gemma4MoeTensorBacking::TokenIds
            | Gemma4MoeTensorBacking::Positions
            | Gemma4MoeTensorBacking::ConstantBf16 { .. }
            | Gemma4MoeTensorBacking::Workspace => resident
                .session
                .allocate_with_category(tensor.view().end_offset(), AllocationCategory::Workspace)
                .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?,
        };
        if tensor.view().end_offset() > buffer.size_bytes() {
            return Err(Gemma4MoeExecutionError::invalid(
                "request tensor view exceeds its backing allocation",
            ));
        }
        if let Gemma4MoeTensorBacking::ConstantBf16 { bits, width } = tensor.backing() {
            let mut bytes = Vec::with_capacity(width * 2);
            for _ in 0..*width {
                bytes.extend_from_slice(&bits.to_le_bytes());
            }
            upload_bytes(
                resident.session.as_ref(),
                &resident.queue,
                &buffer,
                &bytes,
                resident.completion_timeout,
            )?;
        }
        buffers.push(buffer);
    }

    let cache = Arc::new(PreparedSemanticCache::default());
    let dynamic = PreparedDynamicIdentity::stateful(
        graph.token_count(),
        graph.start_position(),
        graph.expected_length(),
        0,
        0,
    );
    let mut prepared = Vec::with_capacity(layout.nodes.len());
    for node in &layout.nodes {
        let Gemma4MoeLowering::Semantic(descriptor) = node.lowering() else {
            prepared.push(None);
            continue;
        };
        let inputs = node
            .inputs()
            .iter()
            .map(|tensor| {
                resident
                    .session
                    .bind(
                        &buffers[*tensor],
                        layout.tensors[*tensor].view().clone(),
                        AccessMode::Read,
                    )
                    .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = node
            .outputs()
            .iter()
            .map(|tensor| {
                resident
                    .session
                    .bind(
                        &buffers[*tensor],
                        layout.tensors[*tensor].view().clone(),
                        AccessMode::Write,
                    )
                    .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        prepared.push(Some(
            cache
                .prepare(
                    resident.session.as_ref(),
                    descriptor.as_ref().clone(),
                    inputs,
                    outputs,
                    PreparedCachePolicy::Reusable(dynamic),
                )
                .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?,
        ));
    }
    Ok(ProvisionedGemma4MoeRequest {
        buffers,
        prepared_cache: cache,
        prepared,
        kv_states,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MoeExecutionOutput {
    token_ids: Vec<i32>,
    audit: PreparedExecutionAudit,
    committed_length: u64,
}

impl Gemma4MoeExecutionOutput {
    pub fn token_ids(&self) -> &[i32] {
        &self.token_ids
    }

    pub const fn audit(&self) -> &PreparedExecutionAudit {
        &self.audit
    }

    pub const fn committed_length(&self) -> u64 {
        self.committed_length
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MoePreparedAudit {
    semantic_node_count: usize,
    pending_attention_count: usize,
    segment_count: usize,
    boundary_count: usize,
    fallback_used: bool,
    terminal_readback_tensor: usize,
}

impl Gemma4MoePreparedAudit {
    pub const fn semantic_node_count(&self) -> usize {
        self.semantic_node_count
    }
    pub const fn pending_attention_count(&self) -> usize {
        self.pending_attention_count
    }
    pub const fn segment_count(&self) -> usize {
        self.segment_count
    }
    pub const fn boundary_count(&self) -> usize {
        self.boundary_count
    }
    pub const fn fallback_used(&self) -> bool {
        self.fallback_used
    }
    pub const fn terminal_readback_tensor(&self) -> usize {
        self.terminal_readback_tensor
    }
}

pub struct Gemma4MoeExecutionRequest {
    _resident: Arc<Gemma4MoeResidentInner>,
    state: Gemma4MoeRequestState,
    layout: Gemma4MoeExecutionLayout,
    buffers: Vec<ExecutionBuffer>,
    _prepared_cache: Arc<PreparedSemanticCache>,
    prepared: Vec<Option<PreparedOperation>>,
    kv_states: Vec<KvState>,
    transition_committed: bool,
    poisoned: bool,
    last_output: Option<Gemma4MoeExecutionOutput>,
}

impl Gemma4MoeExecutionRequest {
    pub const fn state(&self) -> &Gemma4MoeRequestState {
        &self.state
    }
    pub const fn layout(&self) -> &Gemma4MoeExecutionLayout {
        &self.layout
    }
    pub const fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    pub const fn transition_committed(&self) -> bool {
        self.transition_committed
    }

    /// Exports all 30 quiescent opaque KV states. Publication is rejected
    /// during a transition and after any fail-closed poisoning.
    pub fn state_image(&self) -> Result<Gemma4MoeStateImageV1, Gemma4MoeExecutionError> {
        if self.poisoned {
            return Err(Gemma4MoeExecutionError::invalid(
                "poisoned request state cannot be exported",
            ));
        }
        if !self.transition_committed {
            return Err(Gemma4MoeExecutionError::invalid(
                "state image export requires a completed quiescent transition",
            ));
        }
        let committed_length = self.committed_length()?;
        if committed_length == 0 {
            return Err(Gemma4MoeExecutionError::invalid(
                "state image export requires a non-empty prefix",
            ));
        }
        let capacity = self.state_capacity()?;
        let identity = gemma4_moe_prefix_identity(&self._resident, capacity);
        let mut kv_layers = BTreeMap::new();
        for state in &self.kv_states {
            let image = self
                ._resident
                .session
                .export_kv_state_image(state)
                .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
            validate_gemma4_moe_layer_image(
                &image,
                state.layer_id(),
                state.descriptor(),
                committed_length,
            )?;
            kv_layers.insert(
                state.layer_id(),
                Gemma4MoeKvStateImageV1 {
                    descriptor: state.descriptor(),
                    image,
                },
            );
        }
        let image = Gemma4MoeStateImageV1 {
            session_id: self._resident.session.id(),
            identity,
            committed_length,
            kv_layers,
            cached_terminal_output: self.last_output.clone(),
        };
        validate_gemma4_moe_state_image_topology(&image)?;
        Ok(image)
    }

    pub fn export_state_image(&self) -> Result<Gemma4MoeStateImageV1, Gemma4MoeExecutionError> {
        self.state_image()
    }

    pub fn save_state_image(&self) -> Result<Gemma4MoeStateImageV1, Gemma4MoeExecutionError> {
        self.state_image()
    }

    /// Captures the completed request directly into the common persistent
    /// checkpoint envelope. Terminal output remains request-local.
    #[allow(clippy::too_many_arguments)]
    pub fn checkpoint(
        &self,
        identity: CheckpointIdentity,
        token_history: &[u32],
        conversation: &[u8],
        sampler_state: &[u8],
        grammar_state: &[u8],
        stop_state: &[u8],
        absolute_position: u64,
        logical_position: u64,
        generation_state_version: u32,
    ) -> Result<SessionCheckpoint, Gemma4MoeExecutionError> {
        self.state_image()?.without_terminal_output().to_checkpoint(
            identity,
            token_history,
            conversation,
            sampler_state,
            grammar_state,
            stop_state,
            absolute_position,
            logical_position,
            generation_state_version,
        )
    }

    /// Creates an immutable same-resident prefix by forking every quiescent
    /// state into a distinct owner. A later failure drops all local children
    /// and leaves the source request untouched.
    pub fn publish_prefix(&self) -> Result<Gemma4MoePrefixStateV1, Gemma4MoeExecutionError> {
        if self.poisoned {
            return Err(Gemma4MoeExecutionError::invalid(
                "poisoned request state cannot be forked",
            ));
        }
        if !self.transition_committed {
            return Err(Gemma4MoeExecutionError::invalid(
                "prefix publication requires a completed quiescent transition",
            ));
        }
        let committed_length = self.committed_length()?;
        let cached_terminal_output = self.last_output.clone().ok_or_else(|| {
            Gemma4MoeExecutionError::invalid("prefix publication requires terminal output")
        })?;
        if committed_length == 0 || cached_terminal_output.committed_length != committed_length {
            return Err(Gemma4MoeExecutionError::invalid(
                "prefix terminal output differs from committed KV length",
            ));
        }
        let identity = gemma4_moe_prefix_identity(&self._resident, self.state_capacity()?);
        let mut kv_states = BTreeMap::new();
        let mut fork_audit = Gemma4MoePrefixForkAuditV1::default();
        for source in &self.kv_states {
            let descriptor = source.descriptor();
            let (forked, audit) = self
                ._resident
                .session
                .fork_kv_state(source, descriptor)
                .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
            if forked.id() == source.id() {
                return Err(Gemma4MoeExecutionError::invalid(
                    "prefix fork reused the mutable source state identity",
                ));
            }
            let snapshot = forked
                .snapshot(self._resident.session.as_ref())
                .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
            if snapshot.length() != committed_length {
                return Err(Gemma4MoeExecutionError::invalid(format!(
                    "forked layer {} length differs from prefix",
                    source.layer_id()
                )));
            }
            let fallback_resident_bytes = descriptor
                .resident_bytes_per_plane()
                .and_then(|bytes| bytes.checked_mul(2))
                .ok_or_else(|| {
                    Gemma4MoeExecutionError::invalid("prefix KV footprint overflowed")
                })?;
            fork_audit.add(
                audit,
                descriptor.sliding_window().is_some(),
                snapshot.physical_memory(),
                fallback_resident_bytes,
            )?;
            kv_states.insert(source.layer_id(), forked);
        }
        Ok(Gemma4MoePrefixStateV1 {
            inner: Arc::new(Gemma4MoePrefixStateInner {
                session: Arc::clone(&self._resident.session),
                identity,
                committed_length,
                kv_states,
                cached_terminal_output,
                fork_audit,
            }),
        })
    }

    pub fn prefix_state(&self) -> Result<Gemma4MoePrefixStateV1, Gemma4MoeExecutionError> {
        self.publish_prefix()
    }

    pub fn create_prefix_state(&self) -> Result<Gemma4MoePrefixStateV1, Gemma4MoeExecutionError> {
        self.publish_prefix()
    }

    /// Structural prepared audit. Runtime dispatch evidence is only returned
    /// by [`Self::execute`].
    pub fn prepared_audit(&self) -> Gemma4MoePreparedAudit {
        let pending_attention_count = self.layout.attention_hooks().count();
        Gemma4MoePreparedAudit {
            semantic_node_count: self.layout.nodes.len() - pending_attention_count,
            pending_attention_count,
            segment_count: self.layout.segments.len(),
            boundary_count: self.layout.segments.len(),
            fallback_used: false,
            terminal_readback_tensor: self.layout.terminal_readback_tensor,
        }
    }

    pub fn ensure_dispatchable(&self) -> Result<(), Gemma4MoeExecutionError> {
        if self.layout.transitions.len() != 1 {
            return Err(Gemma4MoeExecutionError::invalid(
                "a saturated sliding-ring request must be executed as the planned M=1 transition graphs",
            ));
        }
        if self.kv_states.len() != self.state.layers.len()
            || self.kv_states.len() != self.layout.attention_hooks().count()
            || self.prepared.len() != self.layout.nodes.len()
        {
            return Err(Gemma4MoeExecutionError::invalid(
                "prepared request resources differ from the execution layout",
            ));
        }
        for (index, node) in self.layout.nodes.iter().enumerate() {
            match node.lowering() {
                Gemma4MoeLowering::Semantic(_) if self.prepared[index].is_none() => {
                    return Err(Gemma4MoeExecutionError::invalid(
                        "prepared semantic node is absent",
                    ));
                }
                Gemma4MoeLowering::Semantic(_) => {}
                Gemma4MoeLowering::StaticFp8Attention(hook) => {
                    if self.prepared[index].is_some() {
                        return Err(Gemma4MoeExecutionError::invalid(
                            "opaque attention node unexpectedly owns a semantic preparation",
                        ));
                    }
                    let state = self.kv_states.get(hook.layer as usize).ok_or_else(|| {
                        Gemma4MoeExecutionError::invalid("opaque attention state is absent")
                    })?;
                    let descriptor = state.descriptor();
                    let expected_window = hook.kv.retention_window;
                    if descriptor.layer_id() != hook.layer
                        || descriptor.capacity() != hook.kv.capacity
                        || descriptor.layout().heads() != hook.kv.heads as usize
                        || descriptor.layout().head_dim() != hook.kv.head_dim as usize
                        || descriptor.static_fp8_scales() != Some((1.0, 1.0))
                        || descriptor.sliding_window() != expected_window
                        || hook.score_scale_bits != 1.0_f32.to_bits()
                        || hook.sliding_api_required != expected_window.is_some()
                    {
                        return Err(Gemma4MoeExecutionError::invalid(
                            "opaque static-FP8 attention descriptor differs from the host graph",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Runs one exact request transition. Sliding-window requests which cross
    /// saturation are rejected here and must be rebuilt from
    /// [`plan_gemma4_moe_transitions`] as the prescribed M=1 graphs.
    pub fn execute(
        &mut self,
        token_ids: &[i32],
    ) -> Result<Gemma4MoeExecutionOutput, Gemma4MoeExecutionError> {
        self.validate_token_ids(token_ids)?;
        self.ensure_dispatchable()?;
        if self.poisoned {
            return Err(Gemma4MoeExecutionError::invalid(
                "request is invalid after an unrecoverable transition failure",
            ));
        }
        if self.transition_committed {
            return Err(Gemma4MoeExecutionError::invalid(
                "current transition is already committed; use execute_next for decode",
            ));
        }
        match self.execute_transition(token_ids) {
            Ok(output) => {
                for layer in &mut self.state.layers {
                    layer.committed_length = self.state.expected_length;
                }
                self.transition_committed = true;
                self.last_output = Some(output.clone());
                Ok(output)
            }
            Err(execution_error) => {
                let recovery = self.recover_current_transition();
                self.poisoned = true;
                match recovery {
                    Ok(()) => Err(Gemma4MoeExecutionError::invalid(format!(
                        "transition failed ({execution_error}); KV state was rewound but the request was invalidated fail-closed"
                    ))),
                    Err(recovery_error) => Err(Gemma4MoeExecutionError::invalid(format!(
                        "transition failed ({execution_error}); recovery failed ({recovery_error}); request invalidated"
                    ))),
                }
            }
        }
    }

    fn execute_transition(
        &self,
        token_ids: &[i32],
    ) -> Result<Gemma4MoeExecutionOutput, Gemma4MoeExecutionError> {
        self.upload_request_inputs(token_ids)?;
        let resident = Arc::clone(&self._resident);
        let mut audit = ExecutionAuditAccumulator::new(1);
        for (index, node) in self.layout.nodes.iter().enumerate() {
            match node.lowering() {
                Gemma4MoeLowering::Semantic(_) => {
                    let prepared = self.prepared[index].as_ref().ok_or_else(|| {
                        Gemma4MoeExecutionError::invalid("prepared semantic node is absent")
                    })?;
                    let mut submission = resident
                        .session
                        .submit(prepared, &resident.queue)
                        .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
                    require_transfer_success(
                        submission.wait(resident.completion_timeout),
                        node.label(),
                    )?;
                    audit
                        .record_labeled(node.label(), submission.dispatch())
                        .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
                }
                Gemma4MoeLowering::StaticFp8Attention(hook) => {
                    let inputs = self.bind_node_tensors(node.inputs(), AccessMode::Read)?;
                    let outputs = self.bind_node_tensors(node.outputs(), AccessMode::Write)?;
                    if inputs.len() != 3 || outputs.len() != 1 {
                        return Err(Gemma4MoeExecutionError::invalid(
                            "opaque attention tensor bindings differ",
                        ));
                    }
                    let state = self
                        .kv_states
                        .get(hook.layer as usize)
                        .cloned()
                        .ok_or_else(|| {
                            Gemma4MoeExecutionError::invalid("opaque attention state is absent")
                        })?;
                    let snapshot = state
                        .snapshot(resident.session.as_ref())
                        .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
                    if snapshot.length() != self.state.start_position {
                        return Err(Gemma4MoeExecutionError::invalid(format!(
                            "{} KV length differs: expected {}, actual {}",
                            node.label(),
                            self.state.start_position,
                            snapshot.length()
                        )));
                    }
                    let mut append = resident
                        .session
                        .append_kv_state(
                            &state,
                            &resident.queue,
                            inputs[1].clone(),
                            inputs[2].clone(),
                            self.state.start_position,
                            self.state.start_position,
                        )
                        .map_err(|error| {
                            Gemma4MoeExecutionError::invalid(format!(
                                "{} static-FP8 KV append failed: {error}",
                                node.label()
                            ))
                        })?;
                    require_transfer_success(
                        append.wait(resident.completion_timeout),
                        &format!("{}.static_fp8_kv_append", node.label()),
                    )?;
                    audit
                        .record(append.dispatch())
                        .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
                    drop(append);
                    let descriptor = match hook.kv.retention_window {
                        Some(window) => CausalAttentionDescriptor::new_sliding_scaled(
                            self.state.start_position,
                            self.layout.token_count,
                            self.state.expected_length,
                            window,
                            f32::from_bits(hook.score_scale_bits),
                        ),
                        None => CausalAttentionDescriptor::new_scaled(
                            self.state.start_position,
                            self.layout.token_count,
                            self.state.expected_length,
                            f32::from_bits(hook.score_scale_bits),
                        ),
                    }
                    .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
                    let mut attention = resident
                        .session
                        .causal_attention(
                            &state,
                            &resident.queue,
                            inputs[0].clone(),
                            outputs[0].clone(),
                            descriptor,
                        )
                        .map_err(|error| {
                            Gemma4MoeExecutionError::invalid(format!(
                                "{} static-FP8 causal attention failed: {error}",
                                node.label()
                            ))
                        })?;
                    require_transfer_success(
                        attention.wait(resident.completion_timeout),
                        node.label(),
                    )?;
                    audit
                        .record(attention.dispatch())
                        .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
                }
            }
            if let Some(boundary) = node.boundary_after() {
                let boundary = match boundary {
                    Gemma4MoeExecutionBoundary::StatePublication => {
                        ExecutionBoundaryKind::StatePublication
                    }
                    Gemma4MoeExecutionBoundary::TerminalReadback => {
                        ExecutionBoundaryKind::TerminalReadback
                    }
                };
                audit
                    .record_boundary(boundary, true)
                    .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
            }
        }
        let token_ids = self.read_terminal_tokens()?;
        let audit = audit
            .snapshot()
            .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
        Ok(Gemma4MoeExecutionOutput {
            token_ids,
            audit,
            committed_length: self.state.expected_length,
        })
    }

    /// Rebinds workspace and prepared semantics for the next M=1 decode while
    /// retaining the exact same 30 opaque KV states.
    pub fn transition_decode(&mut self) -> Result<(), Gemma4MoeExecutionError> {
        if self.poisoned {
            return Err(Gemma4MoeExecutionError::invalid(
                "request is invalid; create a fresh request before decoding",
            ));
        }
        if !self.transition_committed {
            return Err(Gemma4MoeExecutionError::invalid(
                "the current transition must commit before decode rebinding",
            ));
        }
        let start_position = self.committed_length()?;
        let capacity = self
            .kv_states
            .first()
            .map(KvState::capacity)
            .ok_or_else(|| Gemma4MoeExecutionError::invalid("opaque KV states are absent"))?;
        if self
            .kv_states
            .iter()
            .any(|state| state.capacity() != capacity)
        {
            return Err(Gemma4MoeExecutionError::invalid(
                "opaque KV capacities differ across layers",
            ));
        }
        let resident = Arc::clone(&self._resident);
        let graph = build_gemma4_moe_graph_from_config_with_identity(
            &resident.config,
            1,
            start_position,
            capacity,
            GEMMA4_MOE_MODEL_FINGERPRINT,
            &resident.source_container_identity,
        )
        .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
        let layout = build_gemma4_moe_execution_layout(&graph, &resident.plan)?;
        if layout.transitions.len() != 1
            || layout.transitions[0].start_position != start_position
            || layout.transitions[0].token_count != 1
        {
            return Err(Gemma4MoeExecutionError::invalid(
                "decode transition does not satisfy the M=1 sliding-ring contract",
            ));
        }
        let provisioned =
            provision_gemma4_moe_request(&resident, &layout, &graph, Some(&self.kv_states))?;
        let state = Gemma4MoeRequestState::fresh(&graph)?;
        if provisioned
            .kv_states
            .iter()
            .zip(&self.kv_states)
            .any(|(next, current)| next.id() != current.id())
        {
            return Err(Gemma4MoeExecutionError::invalid(
                "decode rebinding replaced an opaque KV state",
            ));
        }

        // Publish the new transition only after all workspace bindings and
        // semantic preparations have succeeded.
        self.state = state;
        self.layout = layout;
        self.buffers = provisioned.buffers;
        self._prepared_cache = provisioned.prepared_cache;
        self.prepared = provisioned.prepared;
        self.transition_committed = false;
        Ok(())
    }

    /// Executes one continuation token on the same request-local KV states.
    pub fn execute_next(
        &mut self,
        token_ids: &[i32],
    ) -> Result<Gemma4MoeExecutionOutput, Gemma4MoeExecutionError> {
        if token_ids.len() != 1 || token_ids[0] < 0 || token_ids[0] >= 262_144 {
            return Err(Gemma4MoeExecutionError::invalid(
                "Gemma 4 MoE decode continuation must be one in-vocabulary token",
            ));
        }
        self.transition_decode()?;
        self.execute(token_ids)
    }

    /// Rewinds the most recently committed transition on every layer. This is
    /// only for a speculative result which has not been published to a client;
    /// it is not a retraction mechanism for an externally visible token. A
    /// failed rewind invalidates this request, and recovery then requires a
    /// new fresh request from the resident model.
    pub fn cancel_last_transition(&mut self) -> Result<(), Gemma4MoeExecutionError> {
        if self.poisoned {
            return Err(Gemma4MoeExecutionError::invalid(
                "request is already invalid",
            ));
        }
        if !self.transition_committed {
            return Err(Gemma4MoeExecutionError::invalid(
                "there is no committed transition to cancel",
            ));
        }
        if let Err(error) = self.recover_current_transition() {
            self.poisoned = true;
            return Err(Gemma4MoeExecutionError::invalid(format!(
                "transition cancellation failed ({error}); request invalidated"
            )));
        }
        self.transition_committed = false;
        self.last_output = None;
        Ok(())
    }

    fn committed_length(&self) -> Result<u64, Gemma4MoeExecutionError> {
        let committed = self
            .state
            .layers
            .first()
            .map(Gemma4MoeOpaqueKvState::committed_length)
            .ok_or_else(|| Gemma4MoeExecutionError::invalid("host KV state is absent"))?;
        if self
            .state
            .layers
            .iter()
            .any(|state| state.committed_length != committed)
        {
            return Err(Gemma4MoeExecutionError::invalid(
                "host KV committed lengths differ across layers",
            ));
        }
        for state in &self.kv_states {
            let snapshot = state
                .snapshot(self._resident.session.as_ref())
                .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
            if snapshot.length() != committed {
                return Err(Gemma4MoeExecutionError::invalid(format!(
                    "opaque KV length differs: expected {committed}, actual {}",
                    snapshot.length()
                )));
            }
        }
        Ok(committed)
    }

    fn state_capacity(&self) -> Result<u64, Gemma4MoeExecutionError> {
        let capacity = self
            .kv_states
            .first()
            .map(KvState::capacity)
            .ok_or_else(|| Gemma4MoeExecutionError::invalid("opaque KV states are absent"))?;
        if self
            .kv_states
            .iter()
            .any(|state| state.capacity() != capacity)
        {
            return Err(Gemma4MoeExecutionError::invalid(
                "opaque KV capacities differ across layers",
            ));
        }
        Ok(capacity)
    }

    fn ensure_fresh_restore_destination(&self) -> Result<(), Gemma4MoeExecutionError> {
        if self.poisoned
            || self.transition_committed
            || self.last_output.is_some()
            || self.state.start_position != 0
            || self.committed_length()? != 0
        {
            return Err(Gemma4MoeExecutionError::invalid(
                "state restore requires a fresh, unpoisoned start-position-zero request",
            ));
        }
        Ok(())
    }

    fn validate_state_image_identity(
        &self,
        image: &Gemma4MoeStateImageV1,
    ) -> Result<(), Gemma4MoeExecutionError> {
        if image.session_id != self._resident.session.id() {
            return Err(Gemma4MoeExecutionError::invalid(
                "state image belongs to a different execution session",
            ));
        }
        let expected_identity = gemma4_moe_prefix_identity(&self._resident, self.state_capacity()?);
        if image.identity != expected_identity {
            return Err(Gemma4MoeExecutionError::invalid(
                "state image model, source/config, plan, or capacity identity differs",
            ));
        }
        self.validate_state_image_layers(image)
    }

    fn validate_state_image_layers(
        &self,
        image: &Gemma4MoeStateImageV1,
    ) -> Result<(), Gemma4MoeExecutionError> {
        validate_gemma4_moe_state_image_topology(image)?;
        if image.kv_layers.len() != self.kv_states.len()
            || image
                .kv_layers
                .keys()
                .copied()
                .ne(self.kv_states.iter().map(KvState::layer_id))
        {
            return Err(Gemma4MoeExecutionError::invalid(
                "state image layer topology differs from the fresh request",
            ));
        }
        for destination in &self.kv_states {
            let layer = destination.layer_id();
            let entry = image.kv_layers.get(&layer).ok_or_else(|| {
                Gemma4MoeExecutionError::invalid(format!("state image KV layer {layer} is absent"))
            })?;
            if entry.descriptor != destination.descriptor() {
                return Err(Gemma4MoeExecutionError::invalid(format!(
                    "state image KV layer {layer} descriptor differs"
                )));
            }
            validate_gemma4_moe_layer_image(
                &entry.image,
                layer,
                destination.descriptor(),
                image.committed_length,
            )?;
        }
        Ok(())
    }

    fn restore_state_image(
        &mut self,
        image: &Gemma4MoeStateImageV1,
    ) -> Result<(), Gemma4MoeExecutionError> {
        self.ensure_fresh_restore_destination()?;
        self.validate_state_image_identity(image)?;

        // The request is private to the factory until this method returns.
        // Publish no host scalar until all 30 adapter imports and snapshots
        // have completed exactly.
        for destination in &self.kv_states {
            let entry = image
                .kv_layers
                .get(&destination.layer_id())
                .expect("validated state-image topology");
            self._resident
                .session
                .import_kv_state_image(destination, &entry.image)
                .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
            let snapshot = destination
                .snapshot(self._resident.session.as_ref())
                .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
            if snapshot.length() != image.committed_length {
                return Err(Gemma4MoeExecutionError::invalid(format!(
                    "restored KV layer {} length differs",
                    destination.layer_id()
                )));
            }
        }
        self.publish_restored_boundary(
            image.committed_length,
            image.cached_terminal_output.clone(),
        );
        Ok(())
    }

    fn restore_checkpoint(
        &mut self,
        checkpoint: &SessionCheckpoint,
        expected_identity: &CheckpointIdentity,
    ) -> Result<(), Gemma4MoeExecutionError> {
        self.ensure_fresh_restore_destination()?;
        checkpoint
            .validate()
            .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
        if checkpoint.header.identity != *expected_identity {
            return Err(Gemma4MoeExecutionError::invalid(
                "checkpoint frontend identity differs from the restore caller",
            ));
        }
        let identity = gemma4_moe_prefix_identity(&self._resident, self.state_capacity()?);
        let descriptor_digest = gemma4_moe_kv_descriptor_digest(
            &identity,
            self.kv_states
                .iter()
                .map(|state| (state.layer_id(), state.descriptor())),
        );
        if expected_identity.model_lock_fingerprint != identity.model_fingerprint
            || expected_identity.plan_digest != gemma4_moe_hex_digest(&identity.plan_digest)
            || expected_identity.kv_encoding != KvCacheEncoding::Fp8E4M3FnStatic
            || expected_identity.kv_descriptor_digest != descriptor_digest
        {
            return Err(Gemma4MoeExecutionError::invalid(
                "checkpoint model, source/config, plan, or KV descriptor identity differs",
            ));
        }
        let committed_length = checkpoint.header.logical_position;
        if committed_length == 0
            || committed_length != checkpoint.header.token_count
            || checkpoint.header.absolute_position != committed_length
            || committed_length > identity.state_capacity
        {
            return Err(Gemma4MoeExecutionError::invalid(
                "checkpoint position or token history differs from Gemma 4 MoE state capacity",
            ));
        }
        let actual_keys = checkpoint
            .payload
            .state_layers
            .iter()
            .map(|layer| (layer.owner, layer.layer_id))
            .collect::<BTreeSet<_>>();
        let expected_keys = self
            .kv_states
            .iter()
            .map(|state| (StateOwnerKindV1::Kv, state.layer_id()))
            .collect::<BTreeSet<_>>();
        if actual_keys.len() != checkpoint.payload.state_layers.len()
            || actual_keys != expected_keys
        {
            return Err(Gemma4MoeExecutionError::invalid(
                "checkpoint layer topology differs from the fresh Gemma 4 MoE graph",
            ));
        }
        let mut kv_layers = BTreeMap::new();
        for destination in &self.kv_states {
            let layer = destination.layer_id();
            let metadata = checkpoint
                .payload
                .state_layers
                .iter()
                .find(|metadata| {
                    metadata.owner == StateOwnerKindV1::Kv && metadata.layer_id == layer
                })
                .ok_or_else(|| {
                    Gemma4MoeExecutionError::invalid(format!(
                        "checkpoint KV layer {layer} is absent"
                    ))
                })?;
            let planes = checkpoint
                .payload
                .state_planes
                .iter()
                .filter(|plane| plane.owner == StateOwnerKindV1::Kv && plane.layer_id == layer)
                .cloned()
                .collect::<Vec<_>>();
            let state_image = ExecutionStateImageV1::new(metadata.clone(), planes);
            validate_gemma4_moe_layer_image(
                &state_image,
                layer,
                destination.descriptor(),
                committed_length,
            )?;
            kv_layers.insert(
                layer,
                Gemma4MoeKvStateImageV1 {
                    descriptor: destination.descriptor(),
                    image: state_image,
                },
            );
        }
        let image = Gemma4MoeStateImageV1 {
            session_id: self._resident.session.id(),
            identity,
            committed_length,
            kv_layers,
            cached_terminal_output: None,
        };
        self.restore_state_image(&image)
    }

    fn install_prefix(
        &mut self,
        prefix: &Gemma4MoePrefixStateV1,
    ) -> Result<(), Gemma4MoeExecutionError> {
        self.ensure_fresh_restore_destination()?;
        if prefix.inner.session.id() != self._resident.session.id() {
            return Err(Gemma4MoeExecutionError::invalid(
                "prefix belongs to a different execution session",
            ));
        }
        let identity = gemma4_moe_prefix_identity(&self._resident, self.state_capacity()?);
        if prefix.inner.identity != identity
            || prefix.committed_length() == 0
            || prefix.committed_length() > identity.state_capacity
            || prefix.cached_terminal_output().committed_length() != prefix.committed_length()
            || prefix.inner.kv_states.len() != self.kv_states.len()
        {
            return Err(Gemma4MoeExecutionError::invalid(
                "prefix model/source/config/plan/capacity/length identity differs",
            ));
        }
        let mut forked_states = BTreeMap::new();
        for destination in &self.kv_states {
            let layer = destination.layer_id();
            let source = prefix.inner.kv_states.get(&layer).ok_or_else(|| {
                Gemma4MoeExecutionError::invalid(format!("prefix KV layer {layer} is absent"))
            })?;
            if source.descriptor() != destination.descriptor() {
                return Err(Gemma4MoeExecutionError::invalid(format!(
                    "prefix KV layer {layer} descriptor differs"
                )));
            }
            let (forked, _) = self
                ._resident
                .session
                .fork_kv_state(source, destination.descriptor())
                .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
            if forked.id() == source.id() || forked.id() == destination.id() {
                return Err(Gemma4MoeExecutionError::invalid(
                    "prefix installation did not create a distinct state identity",
                ));
            }
            let snapshot = forked
                .snapshot(self._resident.session.as_ref())
                .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
            if snapshot.length() != prefix.committed_length() {
                return Err(Gemma4MoeExecutionError::invalid(format!(
                    "installed prefix layer {layer} length differs"
                )));
            }
            forked_states.insert(layer, forked);
        }
        self.kv_states = forked_states.into_values().collect();
        self.publish_restored_boundary(
            prefix.committed_length(),
            Some(prefix.cached_terminal_output().clone()),
        );
        self.committed_length()?;
        Ok(())
    }

    fn publish_restored_boundary(
        &mut self,
        committed_length: u64,
        cached_terminal_output: Option<Gemma4MoeExecutionOutput>,
    ) {
        self.state.start_position = committed_length;
        self.state.expected_length = committed_length;
        for layer in &mut self.state.layers {
            layer.committed_length = committed_length;
        }
        self.transition_committed = true;
        self.last_output = cached_terminal_output;
    }

    fn recover_current_transition(&mut self) -> Result<(), Gemma4MoeExecutionError> {
        let start = self.state.start_position;
        let expected = self.state.expected_length;
        let mut first_error = None;
        for state in &self.kv_states {
            let snapshot = match state.snapshot(self._resident.session.as_ref()) {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    first_error.get_or_insert_with(|| error.to_string());
                    continue;
                }
            };
            match snapshot.length() {
                length if length == start => {}
                length if length == expected => {
                    if let Err(error) = self
                        ._resident
                        .session
                        .rewind_last_kv_state_transition(state, expected, start)
                    {
                        first_error.get_or_insert_with(|| error.to_string());
                    }
                }
                length => {
                    first_error.get_or_insert_with(|| {
                        format!(
                            "layer {} has non-transactional KV length {length} outside {start}/{expected}",
                            state.layer_id()
                        )
                    });
                }
            }
        }
        if let Some(error) = first_error {
            return Err(Gemma4MoeExecutionError::invalid(error));
        }
        for state in &self.kv_states {
            let snapshot = state
                .snapshot(self._resident.session.as_ref())
                .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
            if snapshot.length() != start {
                return Err(Gemma4MoeExecutionError::invalid(format!(
                    "layer {} did not recover to KV length {start}",
                    state.layer_id()
                )));
            }
        }
        for layer in &mut self.state.layers {
            layer.committed_length = start;
        }
        Ok(())
    }

    fn validate_token_ids(&self, token_ids: &[i32]) -> Result<(), Gemma4MoeExecutionError> {
        if token_ids.len() as u64 != self.layout.token_count
            || token_ids
                .iter()
                .any(|token| *token < 0 || *token >= 262_144)
        {
            return Err(Gemma4MoeExecutionError::invalid(
                "request token IDs differ from the exact Gemma vocabulary/layout",
            ));
        }
        Ok(())
    }

    fn upload_request_inputs(&self, token_ids: &[i32]) -> Result<(), Gemma4MoeExecutionError> {
        let token_tensor = self
            .layout
            .tensors
            .iter()
            .find(|tensor| tensor.backing == Gemma4MoeTensorBacking::TokenIds)
            .ok_or_else(|| Gemma4MoeExecutionError::invalid("token tensor is absent"))?;
        let position_tensor = self
            .layout
            .tensors
            .iter()
            .find(|tensor| tensor.backing == Gemma4MoeTensorBacking::Positions)
            .ok_or_else(|| Gemma4MoeExecutionError::invalid("position tensor is absent"))?;
        let token_bytes = token_ids
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect::<Vec<_>>();
        let positions = (0..self.layout.token_count)
            .map(|row| {
                self.state
                    .start_position
                    .checked_add(row)
                    .and_then(|position| i32::try_from(position).ok())
                    .ok_or_else(|| Gemma4MoeExecutionError::invalid("request position exceeds i32"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let position_bytes = positions
            .iter()
            .flat_map(|position| position.to_le_bytes())
            .collect::<Vec<_>>();
        upload_bytes(
            self._resident.session.as_ref(),
            &self._resident.queue,
            &self.buffers[token_tensor.id],
            &token_bytes,
            self._resident.completion_timeout,
        )?;
        upload_bytes(
            self._resident.session.as_ref(),
            &self._resident.queue,
            &self.buffers[position_tensor.id],
            &position_bytes,
            self._resident.completion_timeout,
        )
    }

    fn bind_node_tensors(
        &self,
        tensors: &[usize],
        access: AccessMode,
    ) -> Result<Vec<OwnedTensorBinding>, Gemma4MoeExecutionError> {
        tensors
            .iter()
            .map(|tensor| {
                self._resident
                    .session
                    .bind(
                        &self.buffers[*tensor],
                        self.layout.tensors[*tensor].view.clone(),
                        access,
                    )
                    .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))
            })
            .collect()
    }

    fn read_terminal_tokens(&self) -> Result<Vec<i32>, Gemma4MoeExecutionError> {
        let tensor = &self.layout.tensors[self.layout.terminal_readback_tensor];
        let source = self.buffers[tensor.id]
            .range(tensor.view.byte_offset(), tensor.view.payload_bytes())
            .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
        let mut readback = self
            ._resident
            .session
            .readback(&self._resident.queue, source)
            .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
        require_transfer_success(
            readback.wait(self._resident.completion_timeout),
            "Gemma 4 MoE terminal readback",
        )?;
        let mut bytes = vec![0_u8; tensor.view.payload_bytes() as usize];
        readback
            .read_into(&mut bytes)
            .map_err(|error| Gemma4MoeExecutionError::invalid(error.to_string()))?;
        if bytes.len() % 4 != 0 {
            return Err(Gemma4MoeExecutionError::invalid(
                "terminal token byte count is not divisible by four",
            ));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|bytes| i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect())
    }
}

fn gemma4_moe_prefix_identity(
    resident: &Gemma4MoeResidentInner,
    state_capacity: u64,
) -> Gemma4MoePrefixIdentityV1 {
    Gemma4MoePrefixIdentityV1 {
        model_fingerprint: GEMMA4_MOE_MODEL_FINGERPRINT.to_owned(),
        source_container_identity: resident.source_container_identity.clone(),
        plan_digest: *resident.plan.digest(),
        config_digest: gemma4_moe_config_digest(&resident.config),
        state_capacity,
    }
}

fn gemma4_moe_config_digest(config: &Gemma4MoeConfig) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"sllm-gemma4-moe-config-v1");
    for value in [
        config.hidden_size,
        config.layer_count,
        config.attention_heads,
        config.sliding_kv_heads,
        config.full_kv_heads,
        config.sliding_head_dim,
        config.full_head_dim,
        config.sliding_window,
        config.max_position_embeddings,
        config.vocab_size,
        config.dense_intermediate_size,
        config.expert_count,
        config.selected_expert_count,
        config.expert_intermediate_size,
    ] {
        digest.update(value.to_le_bytes());
    }
    digest.update((config.layer_types.len() as u64).to_le_bytes());
    for layer_type in &config.layer_types {
        digest.update([match layer_type {
            crate::Gemma4LayerType::SlidingAttention => 1,
            crate::Gemma4LayerType::FullAttention => 2,
        }]);
    }
    digest.finalize().into()
}

fn gemma4_moe_kv_descriptor_digest(
    identity: &Gemma4MoePrefixIdentityV1,
    descriptors: impl IntoIterator<Item = (u32, KvStateDescriptor)>,
) -> [u8; 32] {
    let mut ordered = descriptors.into_iter().collect::<Vec<_>>();
    ordered.sort_by_key(|(layer, _)| *layer);
    let mut digest = Sha256::new();
    digest.update(b"sllm-gemma4-moe-kv-descriptor-v1");
    digest.update(identity.source_container_identity.as_bytes());
    digest.update(identity.config_digest);
    digest.update((ordered.len() as u64).to_le_bytes());
    for (layer, descriptor) in ordered {
        digest.update(layer.to_le_bytes());
        digest.update(descriptor.layer_id().to_le_bytes());
        digest.update(descriptor.capacity().to_le_bytes());
        digest.update((descriptor.layout().heads() as u64).to_le_bytes());
        digest.update((descriptor.layout().head_dim() as u64).to_le_bytes());
        digest.update([match descriptor.cache_encoding() {
            KvCacheEncoding::Fp16 => 0,
            KvCacheEncoding::Fp8E4M3Fn => 1,
            KvCacheEncoding::Fp8E4M3FnStatic => 2,
            KvCacheEncoding::Nvfp4 => 3,
            KvCacheEncoding::Fp8E4M3Block16 => 4,
            KvCacheEncoding::Fp8E5M2Block16 => 5,
            KvCacheEncoding::Mxfp8E4 => 6,
            KvCacheEncoding::Mxfp8E5 => 7,
        }]);
        if let Some((key, value)) = descriptor.static_fp8_scales() {
            digest.update([1]);
            digest.update(key.to_bits().to_le_bytes());
            digest.update(value.to_bits().to_le_bytes());
        } else {
            digest.update([0]);
        }
        if let Some(window) = descriptor.sliding_window() {
            digest.update([1]);
            digest.update(window.to_le_bytes());
        } else {
            digest.update([0]);
        }
    }
    digest.finalize().into()
}

fn gemma4_moe_hex_digest(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(71);
    output.push_str("sha256:");
    for byte in bytes {
        use std::fmt::Write as _;
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn validate_gemma4_moe_state_image_topology(
    image: &Gemma4MoeStateImageV1,
) -> Result<(), Gemma4MoeExecutionError> {
    if image.committed_length == 0 || image.committed_length > image.identity.state_capacity {
        return Err(Gemma4MoeExecutionError::invalid(
            "state image length is empty or exceeds capacity",
        ));
    }
    if image.kv_layers.len() != 30
        || image.kv_layers.keys().copied().ne(0_u32..30)
        || image
            .cached_terminal_output
            .as_ref()
            .is_some_and(|output| output.committed_length != image.committed_length)
    {
        return Err(Gemma4MoeExecutionError::invalid(
            "state image layer topology or terminal output length differs",
        ));
    }
    for (&layer, entry) in &image.kv_layers {
        if entry.descriptor.layer_id() != layer
            || entry.descriptor.capacity() != image.identity.state_capacity
            || entry.descriptor.cache_encoding() != KvCacheEncoding::Fp8E4M3FnStatic
            || entry.descriptor.static_fp8_scales() != Some((1.0, 1.0))
        {
            return Err(Gemma4MoeExecutionError::invalid(format!(
                "state image KV layer {layer} descriptor is not unit-scale static FP8"
            )));
        }
        validate_gemma4_moe_layer_image(
            &entry.image,
            layer,
            entry.descriptor,
            image.committed_length,
        )?;
    }
    Ok(())
}

fn validate_gemma4_moe_layer_image(
    image: &ExecutionStateImageV1,
    layer: u32,
    descriptor: KvStateDescriptor,
    expected_length: u64,
) -> Result<(), Gemma4MoeExecutionError> {
    let metadata = image.metadata();
    if metadata.owner != StateOwnerKindV1::Kv
        || metadata.layer_id != layer
        || metadata.published_length != expected_length
        || metadata.active_slot.is_some()
        || expected_length > descriptor.capacity()
    {
        return Err(Gemma4MoeExecutionError::invalid(format!(
            "state image KV layer {layer} metadata differs"
        )));
    }
    if image.planes().len() != 2 {
        return Err(Gemma4MoeExecutionError::invalid(format!(
            "state image KV layer {layer} plane topology differs"
        )));
    }
    let mut seen_key = false;
    let mut seen_value = false;
    for plane in image.planes() {
        if plane.owner != StateOwnerKindV1::Kv || plane.layer_id != layer || plane.bytes.is_empty()
        {
            return Err(Gemma4MoeExecutionError::invalid(format!(
                "state image KV layer {layer} contains an invalid plane"
            )));
        }
        match plane.plane {
            StatePlaneKindV1::KvKey if !seen_key => seen_key = true,
            StatePlaneKindV1::KvValue if !seen_value => seen_value = true,
            _ => {
                return Err(Gemma4MoeExecutionError::invalid(format!(
                    "state image KV layer {layer} contains a duplicate or unexpected plane"
                )));
            }
        }
    }
    if !seen_key || !seen_value {
        return Err(Gemma4MoeExecutionError::invalid(format!(
            "state image KV layer {layer} is missing K or V"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AdapterResource, BoundSemanticOp, BufferRange, DispatchEvidence, ExecutionAdapterAccess,
        ExecutionCausalAttentionSubmissionAdapter, ExecutionError,
        ExecutionKvStateSubmissionAdapter, ExecutionReadbackAdapter, ExecutionSessionAdapter,
        ExecutionSubmissionAdapter, ExecutionTransferAdapter, Gemma4LayerType, KvMemoryKind,
        KvStateAppendRequest, KvStateId, KvStateSnapshot, OpaqueStatePlane, PrepareSupport,
        ShutdownReport, StateLayerMetadataV1,
    };
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    fn config() -> Gemma4MoeConfig {
        Gemma4MoeConfig {
            hidden_size: 2_816,
            layer_count: 30,
            attention_heads: 16,
            sliding_kv_heads: 8,
            full_kv_heads: 2,
            sliding_head_dim: 256,
            full_head_dim: 512,
            sliding_window: 1_024,
            max_position_embeddings: 262_144,
            vocab_size: 262_144,
            dense_intermediate_size: 2_112,
            expert_count: 128,
            selected_expert_count: 8,
            expert_intermediate_size: 704,
            layer_types: (0..30)
                .map(|layer| {
                    if (layer + 1) % 6 == 0 {
                        Gemma4LayerType::FullAttention
                    } else {
                        Gemma4LayerType::SlidingAttention
                    }
                })
                .collect(),
        }
    }

    struct CatalogOnlySource {
        config: Gemma4MoeConfig,
        direct: Vec<Gemma4MoeTensorPlane>,
    }

    impl CatalogOnlySource {
        fn new() -> Self {
            let config = config();
            let mut offset = 0_u64;
            let direct = expected_gemma4_moe_text_tensor_catalog(&config)
                .unwrap()
                .into_iter()
                .filter(|spec| !is_expert_source_tensor(&spec.name))
                .map(|spec| {
                    let bytes = bytes_for_spec(&spec).unwrap();
                    let plane = Gemma4MoeTensorPlane {
                        source_file: "fixture.safetensors".to_owned(),
                        source_name: spec.name,
                        dtype: "BF16".to_owned(),
                        shape: spec.stored_shape,
                        absolute_byte_range: [offset, offset + bytes],
                    };
                    offset += bytes;
                    plane
                })
                .collect();
            Self { config, direct }
        }
    }

    impl Gemma4MoeWeightSource for CatalogOnlySource {
        fn config(&self) -> &Gemma4MoeConfig {
            &self.config
        }
        fn repository(&self) -> &str {
            GEMMA4_MOE_REPOSITORY
        }
        fn resolved_revision(&self) -> &str {
            GEMMA4_MOE_REVISION
        }
        fn source_container_identity(&self) -> &str {
            GEMMA4_MOE_MODEL_FINGERPRINT
        }
        fn direct_tensors(&self) -> &[Gemma4MoeTensorPlane] {
            &self.direct
        }
        fn read_direct_tensor(&self, _logical_name: &str) -> Result<Vec<u8>, Gemma4MoeModelError> {
            unreachable!()
        }
        fn read_expert_planes(
            &self,
            _layer: u16,
            _expert: u16,
            _projection: Gemma4MoeExpertProjection,
        ) -> Result<Gemma4MoeExpertPlanes, Gemma4MoeModelError> {
            unreachable!()
        }
    }

    #[derive(Clone, Copy)]
    struct TestKvEntry {
        id: KvStateId,
        descriptor: KvStateDescriptor,
        length: u64,
    }

    #[derive(Default)]
    struct TestExecutionAdapter {
        states: Arc<Mutex<Vec<TestKvEntry>>>,
        fail_prepare: AtomicBool,
        attention_calls: AtomicUsize,
        fail_attention_call: AtomicUsize,
        import_calls: AtomicUsize,
        fail_import_call: AtomicUsize,
        fork_calls: AtomicUsize,
        fail_fork_call: AtomicUsize,
        append_ranges: Mutex<Vec<(u32, u64, u64)>>,
    }

    impl TestExecutionAdapter {
        fn evidence(dispatch_id: u64, symbol: &str) -> DispatchEvidence {
            DispatchEvidence {
                abi_version: 1,
                info_version: 1,
                dispatch_id,
                dispatch_count: 1,
                kernel_id: 1,
                workgroup_size_x: 1,
                grid_size_x: 1,
                row_count: 1,
                normalized_size: 1,
                backend: 1,
                fallback_allowed: false,
                fallback_used: false,
                kernel_symbol: symbol.to_owned(),
                device_symbol: "test-gemma4-moe".to_owned(),
                target: "test-gemma4-moe".to_owned(),
            }
        }

        fn lengths(&self) -> Vec<u64> {
            self.states
                .lock()
                .expect("test state lock")
                .iter()
                .map(|state| state.length)
                .collect()
        }

        fn append_ranges(&self) -> Vec<(u32, u64, u64)> {
            self.append_ranges
                .lock()
                .expect("append-range lock")
                .clone()
        }
    }

    struct TestSemanticSubmission;
    struct TestTransfer;
    struct TestReadback {
        bytes: Vec<u8>,
    }
    struct TestAttentionSubmission;
    struct TestKvSubmission {
        states: Arc<Mutex<Vec<TestKvEntry>>>,
        request: KvStateAppendRequest,
        complete: bool,
    }

    impl TestKvSubmission {
        fn finish(&mut self) -> Result<ExecutionState, ExecutionError> {
            if !self.complete {
                let mut states = self.states.lock().map_err(|_| ExecutionError::Busy)?;
                let state = states
                    .iter_mut()
                    .find(|state| state.id == self.request.state_id())
                    .ok_or(ExecutionError::WrongKvState {
                        expected: self.request.state_id(),
                        actual: KvStateId::new(1),
                    })?;
                if state.length != self.request.expected_length() {
                    return Err(ExecutionError::StaleKvLength {
                        expected: self.request.expected_length(),
                        actual: state.length,
                    });
                }
                state.length = self.request.end_position();
                self.complete = true;
            }
            Ok(ExecutionState::Success)
        }
    }

    impl ExecutionKvStateSubmissionAdapter for TestKvSubmission {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            self.finish()
        }
        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            self.finish()
        }
    }

    impl ExecutionCausalAttentionSubmissionAdapter for TestAttentionSubmission {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }
        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }
    }

    impl ExecutionSubmissionAdapter for TestSemanticSubmission {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }
        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }
        fn start_output_readback(
            &mut self,
            _access: &ExecutionAdapterAccess<'_>,
            output: &OwnedTensorBinding,
        ) -> Result<Box<dyn ExecutionReadbackAdapter>, ExecutionError> {
            Ok(Box::new(TestReadback {
                bytes: vec![0; output.view().payload_bytes() as usize],
            }))
        }
    }

    impl ExecutionTransferAdapter for TestTransfer {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }
        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }
    }

    impl ExecutionReadbackAdapter for TestReadback {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }
        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }
        fn read_into(&mut self, destination: &mut [u8]) -> Result<u64, ExecutionError> {
            if destination.len() != self.bytes.len() {
                return Err(ExecutionError::InvalidRange {
                    reason: "test readback size differs".to_owned(),
                });
            }
            destination.copy_from_slice(&self.bytes);
            Ok(destination.len() as u64)
        }
    }

    impl ExecutionSessionAdapter for TestExecutionAdapter {
        fn max_transfer_bytes(&self) -> u64 {
            1 << 30
        }

        fn supports(&self, _descriptor: &SemanticOpDescriptor) -> PrepareSupport {
            PrepareSupport::Supported
        }

        fn create_queue(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
        ) -> Result<AdapterResource, ExecutionError> {
            Ok(AdapterResource::new(()))
        }

        fn allocate(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _size_bytes: u64,
        ) -> Result<AdapterResource, ExecutionError> {
            Ok(AdapterResource::new(()))
        }

        fn prepare(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _operation: &BoundSemanticOp,
        ) -> Result<AdapterResource, ExecutionError> {
            if self.fail_prepare.load(Ordering::Relaxed) {
                return Err(ExecutionError::BackendStatus {
                    status: 91,
                    diagnostic: "injected prepare failure".to_owned(),
                });
            }
            Ok(AdapterResource::new(()))
        }

        fn submit(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _prepared: &PreparedOperation,
            _queue: &ExecutionQueue,
        ) -> Result<(Box<dyn ExecutionSubmissionAdapter>, DispatchEvidence), ExecutionError>
        {
            Ok((
                Box::new(TestSemanticSubmission),
                Self::evidence(1, "test.semantic"),
            ))
        }

        fn upload(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _queue: &ExecutionQueue,
            _destination: &BufferRange,
            _bytes: Arc<[u8]>,
        ) -> Result<Box<dyn ExecutionTransferAdapter>, ExecutionError> {
            Ok(Box::new(TestTransfer))
        }

        fn readback(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _queue: &ExecutionQueue,
            source: &BufferRange,
        ) -> Result<Box<dyn ExecutionReadbackAdapter>, ExecutionError> {
            Ok(Box::new(TestReadback {
                bytes: vec![0; source.size_bytes() as usize],
            }))
        }

        fn shutdown(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _deadline: Duration,
        ) -> Result<ShutdownReport, ExecutionError> {
            Ok(ShutdownReport {
                retryable_cleanup: 0,
                durable_quarantine: 0,
            })
        }

        fn create_kv_state(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            state_id: KvStateId,
            descriptor: KvStateDescriptor,
        ) -> Result<AdapterResource, ExecutionError> {
            self.states
                .lock()
                .map_err(|_| ExecutionError::Busy)?
                .push(TestKvEntry {
                    id: state_id,
                    descriptor,
                    length: 0,
                });
            Ok(AdapterResource::new(()))
        }

        fn kv_state_snapshot(
            &self,
            access: &ExecutionAdapterAccess<'_>,
            state: &KvState,
        ) -> Result<KvStateSnapshot, ExecutionError> {
            let states = self.states.lock().map_err(|_| ExecutionError::Busy)?;
            let entry = states.iter().find(|entry| entry.id == state.id()).ok_or(
                ExecutionError::WrongKvState {
                    expected: state.id(),
                    actual: KvStateId::new(1),
                },
            )?;
            let physical = if let Some(window) = entry.descriptor.sliding_window() {
                let retained_length = entry.length.min(window);
                KvPhysicalMemorySnapshot::new_with_retention(
                    KvMemoryKind::VirtualContiguous,
                    entry.descriptor.capacity(),
                    entry.length,
                    1,
                    1,
                    window.checked_add(1).expect("test window capacity"),
                    retained_length,
                    entry.length.saturating_sub(window),
                    retained_length,
                )
            } else {
                KvPhysicalMemorySnapshot::new(
                    entry.descriptor.capacity(),
                    entry.length,
                    1,
                    1,
                    entry.descriptor.capacity(),
                    entry.length,
                )
            }
            .map_err(|error| ExecutionError::InvalidRequest {
                reason: error.to_string(),
            })?;
            KvStateSnapshot::new_with_physical_memory(
                access.session_id(),
                entry.id,
                entry.descriptor,
                entry.length,
                physical,
            )
            .map_err(|error| ExecutionError::InvalidRequest {
                reason: error.to_string(),
            })
        }

        fn fork_kv_state(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            source: &KvState,
            destination_id: KvStateId,
            destination_descriptor: KvStateDescriptor,
        ) -> Result<(AdapterResource, StateForkAuditV1), ExecutionError> {
            let call = self.fork_calls.fetch_add(1, Ordering::Relaxed) + 1;
            if self.fail_fork_call.load(Ordering::Relaxed) == call {
                return Err(ExecutionError::BackendStatus {
                    status: 94,
                    diagnostic: "injected KV fork failure".to_owned(),
                });
            }
            let mut states = self.states.lock().map_err(|_| ExecutionError::Busy)?;
            let length = states
                .iter()
                .find(|entry| entry.id == source.id())
                .ok_or(ExecutionError::WrongKvState {
                    expected: source.id(),
                    actual: KvStateId::new(1),
                })?
                .length;
            states.push(TestKvEntry {
                id: destination_id,
                descriptor: destination_descriptor,
                length,
            });
            let retained = destination_descriptor
                .sliding_window()
                .map_or(length, |window| length.min(window));
            let audit = StateForkAuditV1::new(
                StateForkModeV1::SharedReadOnlyPages,
                length,
                retained.max(1),
                0,
                0,
            )
            .map_err(|error| ExecutionError::InvalidRequest {
                reason: error.to_string(),
            })?;
            Ok((AdapterResource::new(()), audit))
        }

        fn export_kv_state_image(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            state: &KvState,
        ) -> Result<ExecutionStateImageV1, ExecutionError> {
            let states = self.states.lock().map_err(|_| ExecutionError::Busy)?;
            let entry = states.iter().find(|entry| entry.id == state.id()).ok_or(
                ExecutionError::WrongKvState {
                    expected: state.id(),
                    actual: KvStateId::new(1),
                },
            )?;
            let retained_start = entry
                .descriptor
                .sliding_window()
                .map_or(0, |window| entry.length.saturating_sub(window));
            let retained_length = entry
                .descriptor
                .sliding_window()
                .map_or(entry.length, |window| entry.length.min(window));
            let mut bytes = retained_start.to_le_bytes().to_vec();
            bytes.extend_from_slice(&retained_length.to_le_bytes());
            Ok(ExecutionStateImageV1::new(
                StateLayerMetadataV1 {
                    owner: StateOwnerKindV1::Kv,
                    layer_id: state.layer_id(),
                    published_length: entry.length,
                    generation: 1,
                    active_slot: None,
                },
                [StatePlaneKindV1::KvKey, StatePlaneKindV1::KvValue]
                    .into_iter()
                    .map(|plane| OpaqueStatePlane {
                        owner: StateOwnerKindV1::Kv,
                        layer_id: state.layer_id(),
                        plane,
                        bytes: bytes.clone(),
                    })
                    .collect(),
            ))
        }

        fn import_kv_state_image(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            state: &KvState,
            image: &ExecutionStateImageV1,
        ) -> Result<(), ExecutionError> {
            let call = self.import_calls.fetch_add(1, Ordering::Relaxed) + 1;
            if self.fail_import_call.load(Ordering::Relaxed) == call {
                return Err(ExecutionError::BackendStatus {
                    status: 95,
                    diagnostic: "injected KV image import failure".to_owned(),
                });
            }
            let mut states = self.states.lock().map_err(|_| ExecutionError::Busy)?;
            let entry = states
                .iter_mut()
                .find(|entry| entry.id == state.id())
                .ok_or(ExecutionError::WrongKvState {
                    expected: state.id(),
                    actual: KvStateId::new(1),
                })?;
            entry.length = image.metadata().published_length;
            Ok(())
        }

        fn rewind_last_kv_state_transition(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            state: &KvState,
            expected_length: u64,
            rewind_length: u64,
        ) -> Result<(), ExecutionError> {
            let mut states = self.states.lock().map_err(|_| ExecutionError::Busy)?;
            let entry = states
                .iter_mut()
                .find(|entry| entry.id == state.id())
                .ok_or(ExecutionError::WrongKvState {
                    expected: state.id(),
                    actual: KvStateId::new(1),
                })?;
            if entry.length != expected_length {
                return Err(ExecutionError::StaleKvLength {
                    expected: expected_length,
                    actual: entry.length,
                });
            }
            entry.length = rewind_length;
            Ok(())
        }

        fn append_kv_state(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            state: &KvState,
            _queue: &ExecutionQueue,
            _key: &OwnedTensorBinding,
            _value: &OwnedTensorBinding,
            request: &KvStateAppendRequest,
        ) -> Result<(Box<dyn ExecutionKvStateSubmissionAdapter>, DispatchEvidence), ExecutionError>
        {
            if state.id() != request.state_id() {
                return Err(ExecutionError::WrongKvState {
                    expected: state.id(),
                    actual: request.state_id(),
                });
            }
            self.append_ranges
                .lock()
                .map_err(|_| ExecutionError::Busy)?
                .push((
                    state.layer_id(),
                    request.start_position(),
                    request.token_count(),
                ));
            Ok((
                Box::new(TestKvSubmission {
                    states: Arc::clone(&self.states),
                    request: *request,
                    complete: false,
                }),
                Self::evidence(2, "test.static_fp8_append"),
            ))
        }

        fn execute_causal_attention(
            &self,
            access: &ExecutionAdapterAccess<'_>,
            state: &KvState,
            _queue: &ExecutionQueue,
            _query: &OwnedTensorBinding,
            _output: &OwnedTensorBinding,
            descriptor: CausalAttentionDescriptor,
        ) -> Result<
            (
                Box<dyn ExecutionCausalAttentionSubmissionAdapter>,
                DispatchEvidence,
            ),
            ExecutionError,
        > {
            let call = self.attention_calls.fetch_add(1, Ordering::Relaxed) + 1;
            if self.fail_attention_call.load(Ordering::Relaxed) == call {
                return Err(ExecutionError::BackendStatus {
                    status: 92,
                    diagnostic: "injected attention failure".to_owned(),
                });
            }
            let snapshot = self.kv_state_snapshot(access, state)?;
            if snapshot.length() != descriptor.expected_kv_length() {
                return Err(ExecutionError::StaleKvLength {
                    expected: descriptor.expected_kv_length(),
                    actual: snapshot.length(),
                });
            }
            Ok((
                Box::new(TestAttentionSubmission),
                Self::evidence(3, "test.static_fp8_attention"),
            ))
        }
    }

    fn test_request(
        token_count: u64,
        state_capacity: u64,
    ) -> (
        Gemma4MoeExecutionRequest,
        Arc<TestExecutionAdapter>,
        Gemma4MoeResidentModel,
    ) {
        let source = CatalogOnlySource::new();
        let plan = build_gemma4_moe_resident_weight_load_plan(&source).unwrap();
        let graph = crate::gemma4_moe_graph::build_gemma4_moe_graph_from_config(
            &source.config,
            token_count,
            0,
            state_capacity,
        )
        .unwrap();
        let layout = build_gemma4_moe_execution_layout(&graph, &plan).unwrap();
        let adapter = Arc::new(TestExecutionAdapter::default());
        let session = Arc::new(ExecutionSession::new("test-gemma4-moe", adapter.clone()));
        let queue = session.create_queue().unwrap();
        let buffers = plan
            .entries
            .iter()
            .map(|entry| {
                let tensor = layout
                    .tensors
                    .iter()
                    .find(|tensor| tensor.name == entry.tensor_name)
                    .unwrap();
                (
                    entry.tensor_name.clone(),
                    session
                        .allocate_with_category(
                            tensor.view.end_offset(),
                            AllocationCategory::ModelResident,
                        )
                        .unwrap(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let inner = Arc::new(Gemma4MoeResidentInner {
            session,
            queue,
            config: source.config.clone(),
            source_container_identity: GEMMA4_MOE_MODEL_FINGERPRINT.to_owned(),
            plan,
            buffers,
            audit: Gemma4MoeResidentAudit {
                resident_allocations: 597,
                direct_weight_allocations: 567,
                expert_blob_allocations: 30,
                individual_expert_allocations: 0,
                resident_bytes: crate::GEMMA4_MOE_TEXT_RESIDENT_BYTES,
            },
            completion_timeout: Duration::from_secs(1),
        });
        let model = Gemma4MoeResidentModel {
            inner: Arc::clone(&inner),
        };
        (model.new_request(graph).unwrap(), adapter, model)
    }

    fn fresh_graph(token_count: u64, state_capacity: u64) -> Gemma4MoeGraph {
        crate::gemma4_moe_graph::build_gemma4_moe_graph_from_config(
            &config(),
            token_count,
            0,
            state_capacity,
        )
        .unwrap()
    }

    fn checkpoint_identity(image: &Gemma4MoeStateImageV1, tokens: &[u32]) -> CheckpointIdentity {
        CheckpointIdentity::for_tokens(
            image.model_fingerprint(),
            image.source_container_identity(),
            "no-adapter",
            "test-renderer",
            "test-tokenizer",
            "test-gemma4-moe",
            gemma4_moe_hex_digest(image.plan_digest()),
            tokens,
            KvCacheEncoding::Fp8E4M3FnStatic,
            image.kv_descriptor_digest(),
            *image.config_digest(),
        )
        .unwrap()
    }

    #[test]
    fn state_image_mixes_retained_sliding_and_full_layers_and_resumes_without_reappend() {
        let (mut source, adapter, model) = test_request(1_017, 2_048);
        source.execute(&vec![0; 1_017]).unwrap();
        for _ in 0..8 {
            source.execute_next(&[0]).unwrap();
        }
        let source_ids = source.kv_states.iter().map(KvState::id).collect::<Vec<_>>();
        let image = source.state_image().unwrap();
        assert_eq!(image.committed_length(), 1_025);
        assert_eq!(image.kv_layers().len(), 30);
        for (&layer, entry) in image.kv_layers() {
            let bytes = &entry.image().planes()[0].bytes;
            let retained_start = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
            let retained_length = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
            if (layer + 1) % 6 == 0 {
                assert_eq!(entry.descriptor().sliding_window(), None);
                assert_eq!((retained_start, retained_length), (0, 1_025));
            } else {
                assert_eq!(entry.descriptor().sliding_window(), Some(1_024));
                assert_eq!((retained_start, retained_length), (1, 1_024));
            }
        }

        let append_count_before_restore = adapter.append_ranges().len();
        let mut restored = model
            .new_request_from_state_image(&image, fresh_graph(1, 2_048))
            .unwrap();
        assert_eq!(adapter.append_ranges().len(), append_count_before_restore);
        assert!(restored.transition_committed());
        assert_eq!(restored.state().start_position(), 1_025);
        assert_eq!(restored.state().expected_length(), 1_025);
        assert!(
            restored
                .state()
                .layers()
                .iter()
                .all(|layer| layer.committed_length() == 1_025)
        );
        let restored_ids = restored
            .kv_states
            .iter()
            .map(KvState::id)
            .collect::<Vec<_>>();
        assert!(
            restored_ids
                .iter()
                .zip(source_ids)
                .all(|(restored, source)| *restored != source)
        );

        restored.execute_next(&[1]).unwrap();
        let ranges = adapter.append_ranges();
        assert_eq!(ranges.len(), append_count_before_restore + 30);
        assert!(
            ranges[ranges.len() - 30..]
                .iter()
                .all(|(_, start, count)| (*start, *count) == (1_025, 1))
        );
        assert!(
            restored
                .state
                .layers
                .iter()
                .all(|layer| layer.committed_length == 1_026)
        );
    }

    #[test]
    fn prefix_forks_all_layers_with_distinct_ids_and_restores_a_quiescent_boundary() {
        let (mut source, adapter, model) = test_request(7, 2_048);
        source.execute(&[0; 7]).unwrap();
        let source_ids = source.kv_states.iter().map(KvState::id).collect::<Vec<_>>();
        let prefix = source.publish_prefix().unwrap();
        let audit = prefix.fork_audit();
        assert_eq!(audit.kv_states(), 30);
        assert_eq!(audit.sliding_states(), 25);
        assert_eq!(audit.full_states(), 5);
        assert!(audit.shared_pages() >= 30);
        assert_eq!(audit.copied_bytes(), 0);
        assert_eq!(audit.destination_owned_bytes(), 0);
        assert!(audit.cache_resident_bytes() > 0);
        let prefix_ids = prefix
            .inner
            .kv_states
            .values()
            .map(KvState::id)
            .collect::<Vec<_>>();
        assert!(
            prefix_ids
                .iter()
                .zip(&source_ids)
                .all(|(prefix, source)| prefix != source)
        );

        let mut child = model
            .new_request_from_prefix(&prefix, fresh_graph(1, 2_048))
            .unwrap();
        let child_ids = child.kv_states.iter().map(KvState::id).collect::<Vec<_>>();
        assert!(
            child_ids
                .iter()
                .zip(&prefix_ids)
                .zip(&source_ids)
                .all(|((child, prefix), source)| child != prefix && child != source)
        );
        assert_eq!(child.committed_length().unwrap(), 7);
        assert_eq!(source.committed_length().unwrap(), 7);
        child.execute_next(&[1]).unwrap();
        assert_eq!(child.committed_length().unwrap(), 8);
        assert_eq!(source.committed_length().unwrap(), 7);
        assert_eq!(prefix.committed_length(), 7);
        assert_eq!(adapter.fork_calls.load(Ordering::Relaxed), 60);
    }

    #[test]
    fn checkpoint_roundtrip_validates_identity_and_resumes_at_image_length() {
        let (mut source, adapter, model) = test_request(5, 2_048);
        source.execute(&[0; 5]).unwrap();
        let image = source.state_image().unwrap().without_terminal_output();
        let tokens = [0_u32; 5];
        let identity = checkpoint_identity(&image, &tokens);
        let checkpoint = image
            .to_checkpoint(
                identity.clone(),
                &tokens,
                b"conversation",
                b"sampler",
                b"grammar",
                b"stop",
                5,
                5,
                1,
            )
            .unwrap();
        let decoded = SessionCheckpoint::decode(&checkpoint.encode().unwrap()).unwrap();
        let append_count = adapter.append_ranges().len();
        let mut restored = model
            .new_request_from_checkpoint(&decoded, fresh_graph(1, 2_048), &identity)
            .unwrap();
        assert_eq!(adapter.append_ranges().len(), append_count);
        assert_eq!(restored.committed_length().unwrap(), 5);
        assert!(restored.last_output.is_none());
        restored.execute_next(&[2]).unwrap();
        assert!(
            adapter.append_ranges()[append_count..]
                .iter()
                .all(|(_, start, count)| (*start, *count) == (5, 1))
        );

        let mut wrong = identity.clone();
        wrong.kv_descriptor_digest[0] ^= 1;
        let mut wrong_checkpoint = decoded.clone();
        wrong_checkpoint.header.identity = wrong.clone();
        assert!(
            model
                .new_request_from_checkpoint(&wrong_checkpoint, fresh_graph(1, 2_048), &wrong)
                .is_err()
        );
    }

    #[test]
    fn reuse_rejects_transition_and_poison_and_recovers_fresh_after_atomic_failures() {
        let (mut source, adapter, model) = test_request(3, 2_048);
        assert!(source.state_image().is_err());
        source.execute(&[0; 3]).unwrap();
        let image = source.state_image().unwrap();
        source.transition_decode().unwrap();
        assert!(source.state_image().is_err());
        assert!(source.publish_prefix().is_err());

        adapter.fail_import_call.store(7, Ordering::Relaxed);
        assert!(
            model
                .new_request_from_state_image(&image, fresh_graph(1, 2_048))
                .is_err()
        );
        assert_eq!(source.committed_length().unwrap(), 3);
        adapter.fail_import_call.store(0, Ordering::Relaxed);
        let restored = model
            .new_request_from_state_image(&image, fresh_graph(1, 2_048))
            .unwrap();
        assert_eq!(restored.committed_length().unwrap(), 3);

        let (mut poisoned, poison_adapter, _poison_model) = test_request(1, 2_048);
        poison_adapter
            .fail_attention_call
            .store(1, Ordering::Relaxed);
        assert!(poisoned.execute(&[0]).is_err());
        assert!(poisoned.is_poisoned());
        assert!(poisoned.state_image().is_err());
        assert!(poisoned.publish_prefix().is_err());

        let (mut fork_source, fork_adapter, _fork_model) = test_request(2, 2_048);
        fork_source.execute(&[0; 2]).unwrap();
        fork_adapter.fail_fork_call.store(11, Ordering::Relaxed);
        assert!(fork_source.publish_prefix().is_err());
        assert_eq!(fork_source.committed_length().unwrap(), 2);
        fork_adapter.fail_fork_call.store(0, Ordering::Relaxed);
        assert!(fork_source.publish_prefix().is_ok());
    }

    #[test]
    fn load_plan_has_one_blob_per_layer_and_no_individual_expert_residency() {
        let source = CatalogOnlySource::new();
        let plan = build_gemma4_moe_resident_weight_load_plan(&source).unwrap();
        assert_eq!(
            plan.entries
                .iter()
                .filter(|entry| entry.tensor_name.starts_with(GEMMA4_MOE_LAYER_BLOB_PREFIX))
                .count(),
            30
        );
        assert!(
            plan.entries
                .iter()
                .all(|entry| !is_expert_source_tensor(&entry.tensor_name))
        );
        assert!(
            plan.entries
                .iter()
                .all(|entry| !is_embedded_per_expert_scale(&entry.tensor_name))
        );
        assert_eq!(plan.entries.len(), 597);
        assert_eq!(
            plan.total_destination_bytes,
            crate::GEMMA4_MOE_TEXT_RESIDENT_BYTES
        );
        assert!(plan.has_valid_digest().unwrap());
    }

    #[test]
    fn graph_and_plan_require_exact_container_identity_even_with_valid_digest() {
        let source = CatalogOnlySource::new();
        let plan = build_gemma4_moe_resident_weight_load_plan(&source).unwrap();
        let graph = crate::gemma4_moe_graph::build_gemma4_moe_graph_from_config(
            &source.config,
            1,
            0,
            2_048,
        )
        .unwrap();
        let mut entries = plan.entries.clone();
        for entry in &mut entries {
            entry.locked_file_sha256 = "different-container".to_owned();
        }
        let wrong_container_plan = WeightLoadPlan::from_verified_entries(
            VerifiedWeightPlanMetadata {
                schema_version: plan.schema_version.clone(),
                repo_id: plan.repo_id.clone(),
                resolved_revision: plan.resolved_revision.clone(),
                lock_fingerprint: plan.lock_fingerprint.clone(),
                tied_embeddings: plan.tied_embeddings,
                chunk_size: plan.chunk_size,
                total_destination_bytes: plan.total_destination_bytes,
            },
            entries,
        )
        .unwrap();
        assert!(wrong_container_plan.has_valid_digest().unwrap());
        assert!(build_gemma4_moe_execution_layout(&graph, &wrong_container_plan).is_err());
    }

    #[test]
    fn blob_manifest_is_exact_and_terminates_with_per_expert_scales() {
        for layer in [0, 29] {
            let inputs = gemma4_moe_layer_blob_pack_inputs(layer).unwrap();
            assert_eq!(inputs.len(), 128 * 3);
            assert_eq!(inputs[0].expert, 0);
            assert_eq!(inputs.last().unwrap().expert, 127);
            assert_eq!(
                inputs.last().unwrap().input_scale_destination[1],
                GEMMA4_MOE_PER_EXPERT_SCALES_OFFSET
            );
            assert_eq!(
                gemma4_moe_per_expert_scale_destination(),
                [
                    GEMMA4_MOE_PER_EXPERT_SCALES_OFFSET,
                    GEMMA4_MOE_LAYER_BLOB_BYTES
                ]
            );
        }
    }

    #[test]
    fn token_one_and_three_layouts_lower_every_host_node_without_bf16_attention() {
        let source = CatalogOnlySource::new();
        let plan = build_gemma4_moe_resident_weight_load_plan(&source).unwrap();
        for tokens in [1, 3] {
            let graph = crate::gemma4_moe_graph::build_gemma4_moe_graph_from_config(
                &source.config,
                tokens,
                0,
                2_048,
            )
            .unwrap();
            let layout = build_gemma4_moe_execution_layout(&graph, &plan).unwrap();
            assert_eq!(layout.token_count(), tokens);
            assert_eq!(layout.attention_hooks().count(), 30);
            assert_eq!(layout.segments().len(), 2);
            assert_eq!(layout.transitions().len(), 1);
            let kinds = layout
                .nodes()
                .iter()
                .filter_map(|node| match node.lowering() {
                    Gemma4MoeLowering::Semantic(descriptor) => {
                        Some((node.label().to_owned(), descriptor.kind()))
                    }
                    Gemma4MoeLowering::StaticFp8Attention(_) => None,
                })
                .collect::<Vec<_>>();
            let positions = [
                ("layer.0.router_norm", SemanticOpKind::RmsNorm),
                (
                    "layer.0.router_root_scale.broadcast",
                    SemanticOpKind::BroadcastMul,
                ),
                (
                    "layer.0.router_root_scale.root_scalar",
                    SemanticOpKind::ScalarMul,
                ),
                ("layer.0.router_projection", SemanticOpKind::Matmul),
                ("layer.0.stable_topk_router", SemanticOpKind::MoeRoute),
                (
                    "layer.0.pre_routed_feedforward_norm",
                    SemanticOpKind::RmsNorm,
                ),
                ("layer.0.routed_experts_nvfp4", SemanticOpKind::MoeExpert),
            ]
            .map(|expected| {
                kinds
                    .iter()
                    .position(|actual| actual.0 == expected.0 && actual.1 == expected.1)
                    .unwrap()
            });
            assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(layout.nodes().iter().all(|node| match node.lowering() {
                Gemma4MoeLowering::StaticFp8Attention(hook) => {
                    hook.kv().dequant_scale_f32_bits == 1.0_f32.to_bits()
                        && hook.kv().serialized_scale_tensor_count == 0
                        && hook.score_scale_bits() == 1.0_f32.to_bits()
                }
                Gemma4MoeLowering::Semantic(descriptor) => {
                    descriptor.kind() != SemanticOpKind::CausalAttention
                }
            }));
        }
    }

    #[test]
    fn prefill_to_seventeen_decode_layouts_are_m1_and_keep_exact_attention_contract() {
        let source = CatalogOnlySource::new();
        let plan = build_gemma4_moe_resident_weight_load_plan(&source).unwrap();
        let prefill_length = 1_017;
        for decode_index in 0..17 {
            let start = prefill_length + decode_index;
            let graph = crate::gemma4_moe_graph::build_gemma4_moe_graph_from_config(
                &source.config,
                1,
                start,
                2_048,
            )
            .unwrap();
            let layout = build_gemma4_moe_execution_layout(&graph, &plan).unwrap();
            assert_eq!(layout.transitions().len(), 1);
            assert_eq!(layout.transitions()[0].start_position(), start);
            assert_eq!(layout.transitions()[0].token_count(), 1);
            assert_eq!(layout.attention_hooks().count(), 30);
            assert!(layout.attention_hooks().all(|hook| {
                hook.score_scale_bits() == 1.0_f32.to_bits()
                    && hook.kv().dequant_scale_f32_bits == 1.0_f32.to_bits()
                    && hook.kv().serialized_scale_tensor_count == 0
            }));
        }
    }

    #[test]
    fn prefill_seventeen_then_decode_seventeen_retains_all_state_ids() {
        let (mut request, adapter, _model) = test_request(17, 2_048);
        let state_ids = request
            .kv_states
            .iter()
            .map(KvState::id)
            .collect::<Vec<_>>();
        request.execute(&[0; 17]).unwrap();
        for expected in 18..=34 {
            request.execute_next(&[0]).unwrap();
            assert_eq!(
                request
                    .kv_states
                    .iter()
                    .map(KvState::id)
                    .collect::<Vec<_>>(),
                state_ids
            );
            assert!(adapter.lengths().iter().all(|length| *length == expected));
        }
    }

    #[test]
    fn decode_m1_crosses_sliding_saturation_from_1017_without_replacing_state() {
        let (mut request, adapter, _model) = test_request(1_017, 2_048);
        let state_ids = request
            .kv_states
            .iter()
            .map(KvState::id)
            .collect::<Vec<_>>();
        request.execute(&vec![0; 1_017]).unwrap();
        for expected in 1_018..=1_034 {
            request.execute_next(&[0]).unwrap();
            assert_eq!(request.layout.transitions.len(), 1);
            assert_eq!(request.layout.transitions[0].token_count, 1);
            assert_eq!(
                request
                    .kv_states
                    .iter()
                    .map(KvState::id)
                    .collect::<Vec<_>>(),
                state_ids
            );
            assert!(adapter.lengths().iter().all(|length| *length == expected));
        }
    }

    #[test]
    fn decode_prepare_failure_keeps_committed_layout_and_state_unchanged() {
        let (mut request, adapter, _model) = test_request(17, 2_048);
        request.execute(&[0; 17]).unwrap();
        let old_ids = request
            .kv_states
            .iter()
            .map(KvState::id)
            .collect::<Vec<_>>();
        let old_digest = request.layout.plan_digest;
        let old_token_count = request.layout.token_count;
        let old_expected_length = request.state.expected_length;
        adapter.fail_prepare.store(true, Ordering::Relaxed);
        assert!(request.transition_decode().is_err());
        assert_eq!(request.layout.plan_digest, old_digest);
        assert_eq!(request.layout.token_count, old_token_count);
        assert_eq!(request.state.expected_length, old_expected_length);
        assert!(request.transition_committed());
        assert!(!request.is_poisoned());
        assert_eq!(
            request
                .kv_states
                .iter()
                .map(KvState::id)
                .collect::<Vec<_>>(),
            old_ids
        );
        assert!(adapter.lengths().iter().all(|length| *length == 17));
    }

    #[test]
    fn partial_dispatch_failure_poisoned_request_rejects_reuse_and_fresh_request_recovers() {
        let (mut request, adapter, model) = test_request(17, 2_048);
        request.execute(&[0; 17]).unwrap();
        let poisoned_ids = request
            .kv_states
            .iter()
            .map(KvState::id)
            .collect::<Vec<_>>();
        let fail_call = adapter.attention_calls.load(Ordering::Relaxed) + 2;
        adapter
            .fail_attention_call
            .store(fail_call, Ordering::Relaxed);
        assert!(request.execute_next(&[0]).is_err());
        assert!(request.is_poisoned());
        assert!(adapter.lengths()[..30].iter().all(|length| *length == 17));
        assert!(request.execute(&[0]).is_err());

        adapter.fail_attention_call.store(0, Ordering::Relaxed);
        let graph = crate::gemma4_moe_graph::build_gemma4_moe_graph_from_config(
            &model.inner.config,
            17,
            0,
            2_048,
        )
        .unwrap();
        let mut fresh = model.new_request(graph).unwrap();
        let fresh_ids = fresh.kv_states.iter().map(KvState::id).collect::<Vec<_>>();
        assert!(fresh_ids.iter().all(|id| !poisoned_ids.contains(id)));
        fresh.execute(&[0; 17]).unwrap();
        assert!(!fresh.is_poisoned());
        assert!(fresh.transition_committed());
    }

    #[test]
    fn cancel_unpublished_transition_rewinds_all_thirty_layers() {
        let (mut request, adapter, _model) = test_request(17, 2_048);
        request.execute(&[0; 17]).unwrap();
        request.execute_next(&[0]).unwrap();
        assert!(adapter.lengths().iter().all(|length| *length == 18));
        request.cancel_last_transition().unwrap();
        assert!(adapter.lengths().iter().all(|length| *length == 17));
        assert!(
            request
                .state
                .layers
                .iter()
                .all(|state| state.committed_length == 17)
        );
        assert!(!request.transition_committed());
        assert!(!request.is_poisoned());
    }

    #[test]
    fn sliding_ring_transition_plan_chunks_before_saturation_then_uses_m1() {
        assert_eq!(
            plan_gemma4_moe_transitions(0, 3).unwrap(),
            vec![Gemma4MoeTransitionSegment {
                start_position: 0,
                token_count: 3,
                expected_length: 3,
                saturated_sliding_ring: false,
            }]
        );
        let crossing = plan_gemma4_moe_transitions(1_022, 5).unwrap();
        assert_eq!(
            crossing
                .iter()
                .map(|segment| segment.token_count())
                .collect::<Vec<_>>(),
            vec![2, 1, 1, 1]
        );
        assert_eq!(crossing[0].expected_length(), 1_024);
        assert!(
            crossing
                .iter()
                .all(|segment| segment.saturated_sliding_ring())
        );
        assert!(
            plan_gemma4_moe_transitions(1_024, 3)
                .unwrap()
                .iter()
                .all(|segment| segment.token_count() == 1)
        );
    }
}
