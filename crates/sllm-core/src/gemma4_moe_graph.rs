//! Exact, host-only graph plan for the reviewed Gemma 4 26B-A4B text model.
//!
//! The existing Qwen sparse-MoE provider has a different router, expert count,
//! shared-expert topology, and weight encoding.  This module therefore records
//! a Gemma-specific semantic graph and resident tensor catalog.  It does not
//! claim that the graph is executable until a backend registers the explicit
//! [`Gemma4MoeProviderContract::Nvfp4RoutedExpertsV1`] contract.

use crate::Gemma4LayerType;
use crate::QuantizedTensorEncoding;
use crate::gemma4_moe::{
    GEMMA4_MOE_MODEL_FINGERPRINT, GEMMA4_MOE_TEXT_TENSOR_COUNT, Gemma4MoeConfig,
    Gemma4MoeExpertProjection, Gemma4MoeRecipe, VerifiedGemma4Moe, VerifiedGgufGemma4Moe,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

const MODEL_PREFIX: &str = "model.language_model";
const GRAPH_RUNTIME_MAX_CONTEXT_TOKENS: u64 = u32::MAX as u64;
const REVIEWED_RMS_NORM_EPSILON_BITS: u32 = 1.0e-6_f32.to_bits();
const REVIEWED_NVFP4_BLOCK_SIZE: u32 = 16;

const fn reviewed_graph_layer_schedule() -> [Gemma4LayerType; 30] {
    use Gemma4LayerType::{FullAttention as F, SlidingAttention as S};
    [
        S, S, S, S, S, F, S, S, S, S, S, F, S, S, S, S, S, F, S, S, S, S, S, F, S, S, S, S, S, F,
    ]
}

fn validate_identity_config(config: &Gemma4MoeConfig) -> Result<(), Gemma4MoeGraphError> {
    if config.hidden_size != 2_816
        || config.layer_count != 30
        || config.attention_heads != 16
        || config.sliding_kv_heads != 8
        || config.full_kv_heads != 2
        || config.sliding_head_dim != 256
        || config.full_head_dim != 512
        || config.sliding_window != 1_024
        || config.max_position_embeddings != 262_144
        || config.vocab_size != 262_144
        || config.dense_intermediate_size != 2_112
        || config.expert_count != 128
        || config.selected_expert_count != 8
        || config.expert_intermediate_size != 704
        || config.layer_types.as_slice() != reviewed_graph_layer_schedule().as_slice()
    {
        return Err(Gemma4MoeGraphError::UnreviewedConfig);
    }
    Ok(())
}

fn validate_identity_recipe(recipe: &Gemma4MoeRecipe) -> Result<(), Gemma4MoeGraphError> {
    if recipe.encoding != QuantizedTensorEncoding::Nvfp4E2M1Block16E4M3FnF32Outer
        || recipe.block_size != REVIEWED_NVFP4_BLOCK_SIZE
        || recipe.value_format != "E2M1"
        || recipe.block_scale_format != "E4M3FN"
        || recipe.outer_scale_format != "F32"
        || recipe.input_scale_format != "F32"
        || recipe.activation_dynamic
        || recipe.kv_cache_format != "FP8"
        || recipe.kv_cache_scale_source != "modelopt-fp8-cast-constant-amax-448"
        || recipe.kv_cache_dequant_scale_f32_bits != 1.0_f32.to_bits()
        || recipe.kv_cache_scale_tensor_count != 0
        || recipe.producer != "modelopt@0.43.0rc2.dev91+gc79ebc014"
    {
        return Err(Gemma4MoeGraphError::UnreviewedRecipe);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Gemma4MoeGraphTensorDtype {
    Bf16,
    F32,
    Fp8E4M3,
    U8,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Gemma4MoeGraphTensorEncoding {
    Plain,
    /// Two NVFP4 values packed into each U8, with block size 16 along K.
    Nvfp4PackedBlock16,
    /// One E4M3 block scale per block of 16 logical K values.
    Nvfp4BlockScaleE4M3,
    Nvfp4InputScaleF32,
    Nvfp4TensorScaleF32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MoeGraphTensorSpec {
    pub name: String,
    pub dtype: Gemma4MoeGraphTensorDtype,
    pub stored_shape: Vec<u64>,
    /// Unpacked matrix shape for packed weights; equal to `stored_shape` for
    /// ordinary tensors and scale planes.
    pub logical_shape: Vec<u64>,
    pub encoding: Gemma4MoeGraphTensorEncoding,
}

impl Gemma4MoeGraphTensorSpec {
    fn plain(
        name: impl Into<String>,
        dtype: Gemma4MoeGraphTensorDtype,
        shape: impl Into<Vec<u64>>,
    ) -> Self {
        let shape = shape.into();
        Self {
            name: name.into(),
            dtype,
            stored_shape: shape.clone(),
            logical_shape: shape,
            encoding: Gemma4MoeGraphTensorEncoding::Plain,
        }
    }

    fn encoded(
        name: impl Into<String>,
        dtype: Gemma4MoeGraphTensorDtype,
        stored_shape: impl Into<Vec<u64>>,
        logical_shape: impl Into<Vec<u64>>,
        encoding: Gemma4MoeGraphTensorEncoding,
    ) -> Self {
        Self {
            name: name.into(),
            dtype,
            stored_shape: stored_shape.into(),
            logical_shape: logical_shape.into(),
            encoding,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MoeGraphExpertTensorFamily {
    pub layer: u32,
    pub expert_count: u32,
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub block_size: u32,
}

impl Gemma4MoeGraphExpertTensorFamily {
    pub fn tensor_specs(&self) -> Vec<Gemma4MoeGraphTensorSpec> {
        let mut tensors = Vec::with_capacity(self.expert_count as usize * 12);
        for expert in 0..self.expert_count {
            for projection in [
                Gemma4MoeExpertProjection::Down,
                Gemma4MoeExpertProjection::Gate,
                Gemma4MoeExpertProjection::Up,
            ] {
                let prefix = format!(
                    "{MODEL_PREFIX}.layers.{}.experts.{expert}.{}",
                    self.layer,
                    projection.source_stem()
                );
                let [rows, logical_k] = projection.logical_shape();
                let value_shape = projection.value_shape();
                let block_scale_shape = projection.block_scale_shape();
                debug_assert_eq!(logical_k % 2, 0);
                debug_assert_eq!(logical_k % u64::from(self.block_size), 0);
                tensors.push(Gemma4MoeGraphTensorSpec::encoded(
                    format!("{prefix}.input_scale"),
                    Gemma4MoeGraphTensorDtype::F32,
                    vec![],
                    vec![],
                    Gemma4MoeGraphTensorEncoding::Nvfp4InputScaleF32,
                ));
                tensors.push(Gemma4MoeGraphTensorSpec::encoded(
                    format!("{prefix}.weight"),
                    Gemma4MoeGraphTensorDtype::U8,
                    value_shape,
                    vec![rows, logical_k],
                    Gemma4MoeGraphTensorEncoding::Nvfp4PackedBlock16,
                ));
                tensors.push(Gemma4MoeGraphTensorSpec::encoded(
                    format!("{prefix}.weight_scale"),
                    Gemma4MoeGraphTensorDtype::Fp8E4M3,
                    block_scale_shape,
                    block_scale_shape,
                    Gemma4MoeGraphTensorEncoding::Nvfp4BlockScaleE4M3,
                ));
                tensors.push(Gemma4MoeGraphTensorSpec::encoded(
                    format!("{prefix}.weight_scale_2"),
                    Gemma4MoeGraphTensorDtype::F32,
                    vec![],
                    vec![],
                    Gemma4MoeGraphTensorEncoding::Nvfp4TensorScaleF32,
                ));
            }
        }
        tensors
    }
}

pub fn expected_gemma4_moe_text_tensor_catalog(
    config: &Gemma4MoeConfig,
) -> Result<Vec<Gemma4MoeGraphTensorSpec>, Gemma4MoeGraphError> {
    validate_identity_config(config)?;
    let hidden_size = u64::from(config.hidden_size);
    let vocab_size = u64::from(config.vocab_size);
    let dense_intermediate_size = u64::from(config.dense_intermediate_size);
    let expert_intermediate_size = u64::from(config.expert_intermediate_size);
    let mut tensors = Vec::with_capacity(GEMMA4_MOE_TEXT_TENSOR_COUNT);
    tensors.push(Gemma4MoeGraphTensorSpec::plain(
        format!("{MODEL_PREFIX}.embed_tokens.weight"),
        Gemma4MoeGraphTensorDtype::Bf16,
        vec![vocab_size, hidden_size],
    ));
    for (layer, layer_type) in config.layer_types.iter().copied().enumerate() {
        let layer = u32::try_from(layer).expect("reviewed layer count fits u32");
        let prefix = format!("{MODEL_PREFIX}.layers.{layer}");
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm",
            "post_feedforward_layernorm_1",
            "post_feedforward_layernorm_2",
        ] {
            tensors.push(Gemma4MoeGraphTensorSpec::plain(
                format!("{prefix}.{norm}.weight"),
                Gemma4MoeGraphTensorDtype::Bf16,
                vec![hidden_size],
            ));
        }
        tensors.push(Gemma4MoeGraphTensorSpec::plain(
            format!("{prefix}.layer_scalar"),
            Gemma4MoeGraphTensorDtype::Bf16,
            vec![1],
        ));

        let (head_dim, kv_heads) = match layer_type {
            Gemma4LayerType::SlidingAttention => (config.sliding_head_dim, config.sliding_kv_heads),
            Gemma4LayerType::FullAttention => (config.full_head_dim, config.full_kv_heads),
        };
        let q_width = u64::from(config.attention_heads) * u64::from(head_dim);
        let kv_width = u64::from(kv_heads) * u64::from(head_dim);
        tensors.push(Gemma4MoeGraphTensorSpec::plain(
            format!("{prefix}.self_attn.q_proj.weight"),
            Gemma4MoeGraphTensorDtype::Bf16,
            vec![q_width, hidden_size],
        ));
        tensors.push(Gemma4MoeGraphTensorSpec::plain(
            format!("{prefix}.self_attn.k_proj.weight"),
            Gemma4MoeGraphTensorDtype::Bf16,
            vec![kv_width, hidden_size],
        ));
        if layer_type == Gemma4LayerType::SlidingAttention {
            tensors.push(Gemma4MoeGraphTensorSpec::plain(
                format!("{prefix}.self_attn.v_proj.weight"),
                Gemma4MoeGraphTensorDtype::Bf16,
                vec![kv_width, hidden_size],
            ));
        }
        tensors.push(Gemma4MoeGraphTensorSpec::plain(
            format!("{prefix}.self_attn.o_proj.weight"),
            Gemma4MoeGraphTensorDtype::Bf16,
            vec![hidden_size, q_width],
        ));
        tensors.push(Gemma4MoeGraphTensorSpec::plain(
            format!("{prefix}.self_attn.q_norm.weight"),
            Gemma4MoeGraphTensorDtype::Bf16,
            vec![u64::from(head_dim)],
        ));
        tensors.push(Gemma4MoeGraphTensorSpec::plain(
            format!("{prefix}.self_attn.k_norm.weight"),
            Gemma4MoeGraphTensorDtype::Bf16,
            vec![u64::from(head_dim)],
        ));

        for projection in ["gate_proj", "up_proj"] {
            tensors.push(Gemma4MoeGraphTensorSpec::plain(
                format!("{prefix}.mlp.{projection}.weight"),
                Gemma4MoeGraphTensorDtype::Bf16,
                vec![dense_intermediate_size, hidden_size],
            ));
        }
        tensors.push(Gemma4MoeGraphTensorSpec::plain(
            format!("{prefix}.mlp.down_proj.weight"),
            Gemma4MoeGraphTensorDtype::Bf16,
            vec![hidden_size, dense_intermediate_size],
        ));
        tensors.push(Gemma4MoeGraphTensorSpec::plain(
            format!("{prefix}.router.proj.weight"),
            Gemma4MoeGraphTensorDtype::Bf16,
            vec![u64::from(config.expert_count), hidden_size],
        ));
        tensors.push(Gemma4MoeGraphTensorSpec::plain(
            format!("{prefix}.router.scale"),
            Gemma4MoeGraphTensorDtype::Bf16,
            vec![hidden_size],
        ));
        tensors.push(Gemma4MoeGraphTensorSpec::plain(
            format!("{prefix}.router.per_expert_scale"),
            Gemma4MoeGraphTensorDtype::Bf16,
            vec![u64::from(config.expert_count)],
        ));
        tensors.extend(
            Gemma4MoeGraphExpertTensorFamily {
                layer,
                expert_count: config.expert_count,
                hidden_size,
                intermediate_size: expert_intermediate_size,
                block_size: REVIEWED_NVFP4_BLOCK_SIZE,
            }
            .tensor_specs(),
        );
    }
    tensors.push(Gemma4MoeGraphTensorSpec::plain(
        format!("{MODEL_PREFIX}.norm.weight"),
        Gemma4MoeGraphTensorDtype::Bf16,
        vec![hidden_size],
    ));
    validate_tensor_catalog(&tensors)?;
    if tensors.len() != GEMMA4_MOE_TEXT_TENSOR_COUNT {
        return Err(Gemma4MoeGraphError::InvalidTensorCount {
            expected: GEMMA4_MOE_TEXT_TENSOR_COUNT,
            actual: tensors.len(),
        });
    }
    Ok(tensors)
}

fn validate_tensor_catalog(
    tensors: &[Gemma4MoeGraphTensorSpec],
) -> Result<(), Gemma4MoeGraphError> {
    let mut names = BTreeSet::new();
    for tensor in tensors {
        if tensor.name.is_empty()
            || tensor.stored_shape.contains(&0)
            || tensor.logical_shape.contains(&0)
            || !names.insert(tensor.name.clone())
        {
            return Err(Gemma4MoeGraphError::InvalidTensorCatalog);
        }
        match tensor.encoding {
            Gemma4MoeGraphTensorEncoding::Nvfp4PackedBlock16
                if tensor.dtype != Gemma4MoeGraphTensorDtype::U8
                    || tensor.stored_shape.len() != 2
                    || tensor.logical_shape.len() != 2
                    || tensor.logical_shape[0] != tensor.stored_shape[0]
                    || tensor.logical_shape[1] != tensor.stored_shape[1] * 2 =>
            {
                return Err(Gemma4MoeGraphError::InvalidTensorCatalog);
            }
            Gemma4MoeGraphTensorEncoding::Nvfp4BlockScaleE4M3
                if tensor.dtype != Gemma4MoeGraphTensorDtype::Fp8E4M3 =>
            {
                return Err(Gemma4MoeGraphError::InvalidTensorCatalog);
            }
            Gemma4MoeGraphTensorEncoding::Nvfp4InputScaleF32
            | Gemma4MoeGraphTensorEncoding::Nvfp4TensorScaleF32
                if tensor.dtype != Gemma4MoeGraphTensorDtype::F32
                    || !tensor.stored_shape.is_empty() =>
            {
                return Err(Gemma4MoeGraphError::InvalidTensorCatalog);
            }
            _ => {}
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Gemma4MoeGraphBindingClass {
    TokenRows,
    PositionAndKv,
    RoutingWorkspace,
    TerminalOutput,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Gemma4MoeExecutionBoundary {
    StatePublication,
    TerminalReadback,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Gemma4MoeNormRole {
    Input,
    Query,
    Key,
    ValueUnitScale,
    PostAttention,
    PreSharedFeedforward,
    PostSharedFeedforward,
    RouterScaleLess,
    PreRoutedFeedforward,
    PostRoutedFeedforward,
    PostCombinedFeedforward,
    Final,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Gemma4MoeRmsScaleMode {
    Direct,
    NoAffineScale,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Gemma4MoeLinearRole {
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionO,
    SharedMlpGate,
    SharedMlpUp,
    SharedMlpDown,
    RouterProjection,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Gemma4MoeAddRole {
    AttentionResidual,
    SharedAndRoutedBranches,
    FeedforwardResidual,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Gemma4MoeProviderContract {
    /// Gemma router plus independently normalized dense shared and routed
    /// branches.  Expert matrices use the four-plane NVFP4 artifact recipe.
    Nvfp4RoutedExpertsV1,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Gemma4MoeRouteWeightSemantics {
    /// Full-softmax top-k weights renormalized to one.  The learned expert
    /// scale has not been applied and belongs to the routed down-combine.
    TopKRenormalizedWithoutExpertScale,
}

impl Gemma4MoeProviderContract {
    /// The existing Qwen provider uses 256 experts, a sigmoid router, a fused
    /// shared expert, and MXFP4 weights, so it is never a compatible provider.
    pub const fn is_qwen_sparse_moe_compatible(self) -> bool {
        false
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Gemma4MoeRopeType {
    Default,
    Proportional,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Gemma4MoeAttentionDescriptor {
    pub layer_type: Gemma4LayerType,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub q_width: u64,
    pub kv_width: u64,
    pub scaling_bits: u32,
    pub sliding_window: Option<u64>,
    pub k_equals_v_before_norm: bool,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Gemma4MoeRopeDescriptor {
    pub rope_type: Gemma4MoeRopeType,
    pub theta: u64,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Gemma4MoeKvStorageFormat {
    Fp8E4M3Fn,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Gemma4MoeKvScaleSource {
    /// ModelOpt 0.43 `fp8_cast` uses constant amax 448.  The resulting
    /// dequantization scale is exactly one and no K/V scale tensor is stored.
    ImplicitUnitModelOptFp8CastAmax448,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Gemma4MoeKvDescriptor {
    pub layer: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub capacity: u64,
    pub retention_window: Option<u64>,
    pub k_equals_v_before_norm: bool,
    pub storage_format: Gemma4MoeKvStorageFormat,
    pub scale_source: Gemma4MoeKvScaleSource,
    pub dequant_scale_f32_bits: u32,
    pub serialized_scale_tensor_count: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Gemma4MoeValueShape {
    Dense {
        rows: u64,
        width: u64,
    },
    QueryAndKey {
        rows: u64,
        query_width: u64,
        key_width: u64,
    },
    Routes {
        rows: u64,
        expert_count: u32,
        top_k: u32,
    },
    TokenIndices {
        rows: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Gemma4MoeGraphNodeKind {
    Embedding {
        weight: String,
    },
    ScaleConstant {
        value_bits: u32,
    },
    RmsNorm {
        role: Gemma4MoeNormRole,
        epsilon_bits: u32,
        scale_mode: Gemma4MoeRmsScaleMode,
        weight: Option<String>,
    },
    Linear {
        role: Gemma4MoeLinearRole,
        weight: String,
        input_features: u64,
        output_features: u64,
    },
    Rotary(Gemma4MoeRopeDescriptor),
    CausalAttention(Gemma4MoeAttentionDescriptor),
    GeluTanhMul,
    RouterRootScale {
        scale_weight: String,
        hidden_root_reciprocal_bits: u32,
    },
    StableTopKRouter {
        expert_count: u32,
        top_k: u32,
        full_softmax: bool,
        renormalize_selected_weights: bool,
        stable_tie_break_by_lower_expert: bool,
        output_weight_semantics: Gemma4MoeRouteWeightSemantics,
    },
    RoutedExpertsNvfp4 {
        family: Gemma4MoeGraphExpertTensorFamily,
        per_expert_scale_weight: String,
        provider_contract: Gemma4MoeProviderContract,
        only_selected_experts: bool,
        apply_scale_after_topk_renormalization: bool,
    },
    Add {
        role: Gemma4MoeAddRole,
    },
    ScaleWeight {
        weight: String,
    },
    TiedOutputProjection {
        embedding_weight: String,
    },
    LogitSoftcap {
        cap_bits: u32,
    },
    Argmax,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MoeGraphNode {
    id: usize,
    label: String,
    layer: Option<u32>,
    kind: Gemma4MoeGraphNodeKind,
    predecessors: Vec<usize>,
    output_shape: Gemma4MoeValueShape,
    binding_class: Gemma4MoeGraphBindingClass,
    boundary_after: Option<Gemma4MoeExecutionBoundary>,
}

impl Gemma4MoeGraphNode {
    pub const fn id(&self) -> usize {
        self.id
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub const fn layer(&self) -> Option<u32> {
        self.layer
    }

    pub const fn kind(&self) -> &Gemma4MoeGraphNodeKind {
        &self.kind
    }

    pub fn predecessors(&self) -> &[usize] {
        &self.predecessors
    }

    pub const fn output_shape(&self) -> &Gemma4MoeValueShape {
        &self.output_shape
    }

    pub const fn binding_class(&self) -> Gemma4MoeGraphBindingClass {
        self.binding_class
    }

    pub const fn boundary_after(&self) -> Option<Gemma4MoeExecutionBoundary> {
        self.boundary_after
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MoeLayerPlan {
    pub layer: u32,
    pub layer_type: Gemma4LayerType,
    pub attention: Gemma4MoeAttentionDescriptor,
    pub rope: Gemma4MoeRopeDescriptor,
    pub kv: Gemma4MoeKvDescriptor,
    pub first_node: usize,
    pub last_node: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MoeGraph {
    model_fingerprint: String,
    source_container_identity: String,
    config: Gemma4MoeConfig,
    token_count: u64,
    start_position: u64,
    expected_length: u64,
    state_capacity: u64,
    nodes: Vec<Gemma4MoeGraphNode>,
    layers: Vec<Gemma4MoeLayerPlan>,
    resident_tensors: Vec<Gemma4MoeGraphTensorSpec>,
}

impl Gemma4MoeGraph {
    pub fn model_fingerprint(&self) -> &str {
        &self.model_fingerprint
    }

    pub fn source_container_identity(&self) -> &str {
        &self.source_container_identity
    }

    pub const fn config(&self) -> &Gemma4MoeConfig {
        &self.config
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

    pub fn nodes(&self) -> &[Gemma4MoeGraphNode] {
        &self.nodes
    }

    pub fn layers(&self) -> &[Gemma4MoeLayerPlan] {
        &self.layers
    }

    pub fn kv_descriptors(&self) -> impl ExactSizeIterator<Item = &Gemma4MoeKvDescriptor> {
        self.layers.iter().map(|layer| &layer.kv)
    }

    pub fn resident_tensors(&self) -> &[Gemma4MoeGraphTensorSpec] {
        &self.resident_tensors
    }

    /// Exact pre-allocation comparison against a parsed text tensor catalog.
    pub fn validate_resident_catalog(
        &self,
        actual: &[Gemma4MoeGraphTensorSpec],
    ) -> Result<(), Gemma4MoeGraphError> {
        let expected = self
            .resident_tensors
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor))
            .collect::<BTreeMap<_, _>>();
        let mut seen = BTreeMap::new();
        for tensor in actual {
            if seen.insert(tensor.name.as_str(), tensor).is_some() {
                return Err(Gemma4MoeGraphError::DuplicateTensor(tensor.name.clone()));
            }
        }
        for (name, tensor) in &expected {
            let Some(actual_tensor) = seen.get(name) else {
                return Err(Gemma4MoeGraphError::MissingTensor((*name).to_owned()));
            };
            if **actual_tensor != **tensor {
                return Err(Gemma4MoeGraphError::TensorMismatch((*name).to_owned()));
            }
        }
        if let Some(extra) = seen.keys().find(|name| !expected.contains_key(**name)) {
            return Err(Gemma4MoeGraphError::ExtraTensor((*extra).to_owned()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Gemma4MoeGraphError {
    UnreviewedConfig,
    UnreviewedRecipe,
    ZeroTokenCount,
    ZeroStateCapacity,
    PositionOverflow,
    LengthOutOfBounds,
    InvalidTopology,
    InvalidTensorCatalog,
    InvalidTensorCount { expected: usize, actual: usize },
    MissingTensor(String),
    ExtraTensor(String),
    DuplicateTensor(String),
    TensorMismatch(String),
}

impl fmt::Display for Gemma4MoeGraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnreviewedConfig => formatter.write_str("unreviewed Gemma 4 MoE config"),
            Self::UnreviewedRecipe => formatter.write_str("unreviewed Gemma 4 MoE recipe"),
            Self::ZeroTokenCount => formatter.write_str("Gemma 4 MoE token count must be non-zero"),
            Self::ZeroStateCapacity => {
                formatter.write_str("Gemma 4 MoE state capacity must be non-zero")
            }
            Self::PositionOverflow => formatter.write_str("Gemma 4 MoE position range overflowed"),
            Self::LengthOutOfBounds => formatter.write_str(
                "Gemma 4 MoE position range exceeds request-state or u32 execution capacity",
            ),
            Self::InvalidTopology => formatter.write_str("invalid Gemma 4 MoE graph topology"),
            Self::InvalidTensorCatalog => formatter.write_str("invalid Gemma 4 MoE tensor catalog"),
            Self::InvalidTensorCount { expected, actual } => write!(
                formatter,
                "Gemma 4 MoE text tensor count differs: expected {expected}, got {actual}"
            ),
            Self::MissingTensor(name) => write!(formatter, "missing Gemma 4 MoE tensor: {name}"),
            Self::ExtraTensor(name) => write!(formatter, "extra Gemma 4 MoE tensor: {name}"),
            Self::DuplicateTensor(name) => {
                write!(formatter, "duplicate Gemma 4 MoE tensor: {name}")
            }
            Self::TensorMismatch(name) => {
                write!(formatter, "Gemma 4 MoE tensor contract mismatch: {name}")
            }
        }
    }
}

impl std::error::Error for Gemma4MoeGraphError {}

pub fn build_gemma4_moe_graph(
    verified: &VerifiedGemma4Moe,
    token_count: u64,
    start_position: u64,
    state_capacity: u64,
) -> Result<Gemma4MoeGraph, Gemma4MoeGraphError> {
    validate_identity_recipe(verified.recipe())?;
    build_gemma4_moe_graph_from_config_with_identity(
        verified.config(),
        token_count,
        start_position,
        state_capacity,
        GEMMA4_MOE_MODEL_FINGERPRINT,
        GEMMA4_MOE_MODEL_FINGERPRINT,
    )
}

/// Builds the same exact semantic graph from a verified, canonical GGUF
/// container. The semantic fingerprint remains the reviewed model identity;
/// the GGUF digest is retained separately so resident provisioning cannot
/// accidentally cross container instances.
pub fn build_gemma4_moe_gguf_graph(
    verified: &VerifiedGgufGemma4Moe,
    token_count: u64,
    start_position: u64,
    state_capacity: u64,
) -> Result<Gemma4MoeGraph, Gemma4MoeGraphError> {
    build_gemma4_moe_graph_from_config_with_identity(
        verified.config(),
        token_count,
        start_position,
        state_capacity,
        GEMMA4_MOE_MODEL_FINGERPRINT,
        verified.file_sha256(),
    )
}

#[cfg(test)]
pub(crate) fn build_gemma4_moe_graph_from_config(
    config: &Gemma4MoeConfig,
    token_count: u64,
    start_position: u64,
    state_capacity: u64,
) -> Result<Gemma4MoeGraph, Gemma4MoeGraphError> {
    build_gemma4_moe_graph_from_config_with_identity(
        config,
        token_count,
        start_position,
        state_capacity,
        GEMMA4_MOE_MODEL_FINGERPRINT,
        GEMMA4_MOE_MODEL_FINGERPRINT,
    )
}

pub(crate) fn build_gemma4_moe_graph_from_config_with_identity(
    config: &Gemma4MoeConfig,
    token_count: u64,
    start_position: u64,
    state_capacity: u64,
    model_fingerprint: &str,
    source_container_identity: &str,
) -> Result<Gemma4MoeGraph, Gemma4MoeGraphError> {
    validate_identity_config(config)?;
    if token_count == 0 {
        return Err(Gemma4MoeGraphError::ZeroTokenCount);
    }
    if state_capacity == 0 {
        return Err(Gemma4MoeGraphError::ZeroStateCapacity);
    }
    let expected_length = start_position
        .checked_add(token_count)
        .ok_or(Gemma4MoeGraphError::PositionOverflow)?;
    if expected_length > state_capacity || state_capacity > GRAPH_RUNTIME_MAX_CONTEXT_TOKENS {
        return Err(Gemma4MoeGraphError::LengthOutOfBounds);
    }

    let resident_tensors = expected_gemma4_moe_text_tensor_catalog(config)?;
    let mut nodes = Vec::new();
    let rows = token_count;
    let hidden_size = u64::from(config.hidden_size);
    let dense_intermediate_size = u64::from(config.dense_intermediate_size);
    let expert_intermediate_size = u64::from(config.expert_intermediate_size);
    let vocab_size = u64::from(config.vocab_size);
    let hidden_shape = dense(rows, hidden_size);
    let embedding_weight = format!("{MODEL_PREFIX}.embed_tokens.weight");
    let embedding = push_node(
        &mut nodes,
        "embedding",
        None,
        Gemma4MoeGraphNodeKind::Embedding {
            weight: embedding_weight.clone(),
        },
        vec![],
        hidden_shape.clone(),
        Gemma4MoeGraphBindingClass::TokenRows,
        None,
    );
    let mut hidden = push_node(
        &mut nodes,
        "embedding_scale",
        None,
        Gemma4MoeGraphNodeKind::ScaleConstant {
            value_bits: (config.hidden_size as f32).sqrt().to_bits(),
        },
        vec![embedding],
        hidden_shape.clone(),
        Gemma4MoeGraphBindingClass::TokenRows,
        None,
    );

    let mut layers = Vec::with_capacity(config.layer_count as usize);
    for (layer_index, layer_type) in config.layer_types.iter().copied().enumerate() {
        let layer = u32::try_from(layer_index).expect("reviewed layer count fits u32");
        let first_node = nodes.len();
        let prefix = format!("{MODEL_PREFIX}.layers.{layer}");
        let input_norm = norm_node(
            &mut nodes,
            layer,
            "input_norm",
            Gemma4MoeNormRole::Input,
            format!("{prefix}.input_layernorm.weight"),
            hidden,
            hidden_shape.clone(),
            REVIEWED_RMS_NORM_EPSILON_BITS,
        );
        let (head_dim, kv_heads, rope_type, theta, rotary_dim, window, k_equals_v) =
            match layer_type {
                Gemma4LayerType::SlidingAttention => (
                    config.sliding_head_dim,
                    config.sliding_kv_heads,
                    Gemma4MoeRopeType::Default,
                    10_000,
                    config.sliding_head_dim,
                    Some(u64::from(config.sliding_window)),
                    false,
                ),
                Gemma4LayerType::FullAttention => (
                    config.full_head_dim,
                    config.full_kv_heads,
                    Gemma4MoeRopeType::Proportional,
                    1_000_000,
                    config.full_head_dim / 4,
                    None,
                    true,
                ),
            };
        let q_width = u64::from(config.attention_heads) * u64::from(head_dim);
        let kv_width = u64::from(kv_heads) * u64::from(head_dim);
        let q = linear_node(
            &mut nodes,
            layer,
            "q_proj",
            Gemma4MoeLinearRole::AttentionQ,
            format!("{prefix}.self_attn.q_proj.weight"),
            hidden_size,
            q_width,
            input_norm,
            dense(rows, q_width),
        );
        let k = linear_node(
            &mut nodes,
            layer,
            "k_proj",
            Gemma4MoeLinearRole::AttentionK,
            format!("{prefix}.self_attn.k_proj.weight"),
            hidden_size,
            kv_width,
            input_norm,
            dense(rows, kv_width),
        );
        let v = if layer_type == Gemma4LayerType::SlidingAttention {
            linear_node(
                &mut nodes,
                layer,
                "v_proj",
                Gemma4MoeLinearRole::AttentionV,
                format!("{prefix}.self_attn.v_proj.weight"),
                hidden_size,
                kv_width,
                input_norm,
                dense(rows, kv_width),
            )
        } else {
            k
        };
        let q_norm = norm_node(
            &mut nodes,
            layer,
            "q_norm",
            Gemma4MoeNormRole::Query,
            format!("{prefix}.self_attn.q_norm.weight"),
            q,
            dense(rows, q_width),
            REVIEWED_RMS_NORM_EPSILON_BITS,
        );
        let k_norm = norm_node(
            &mut nodes,
            layer,
            "k_norm",
            Gemma4MoeNormRole::Key,
            format!("{prefix}.self_attn.k_norm.weight"),
            k,
            dense(rows, kv_width),
            REVIEWED_RMS_NORM_EPSILON_BITS,
        );
        let v_norm = push_node(
            &mut nodes,
            format!("layer.{layer}.v_norm"),
            Some(layer),
            Gemma4MoeGraphNodeKind::RmsNorm {
                role: Gemma4MoeNormRole::ValueUnitScale,
                epsilon_bits: REVIEWED_RMS_NORM_EPSILON_BITS,
                scale_mode: Gemma4MoeRmsScaleMode::NoAffineScale,
                weight: None,
            },
            vec![v],
            dense(rows, kv_width),
            Gemma4MoeGraphBindingClass::TokenRows,
            None,
        );
        let rope = Gemma4MoeRopeDescriptor {
            rope_type,
            theta,
            head_dim,
            rotary_dim,
            q_heads: config.attention_heads,
            kv_heads,
        };
        let rotary = push_node(
            &mut nodes,
            format!("layer.{layer}.rotary"),
            Some(layer),
            Gemma4MoeGraphNodeKind::Rotary(rope),
            vec![q_norm, k_norm],
            Gemma4MoeValueShape::QueryAndKey {
                rows,
                query_width: q_width,
                key_width: kv_width,
            },
            Gemma4MoeGraphBindingClass::PositionAndKv,
            None,
        );
        let attention_descriptor = Gemma4MoeAttentionDescriptor {
            layer_type,
            q_heads: config.attention_heads,
            kv_heads,
            head_dim,
            q_width,
            kv_width,
            scaling_bits: 1.0_f32.to_bits(),
            sliding_window: window,
            k_equals_v_before_norm: k_equals_v,
        };
        let attention = push_node(
            &mut nodes,
            format!("layer.{layer}.attention"),
            Some(layer),
            Gemma4MoeGraphNodeKind::CausalAttention(attention_descriptor),
            vec![rotary, v_norm],
            dense(rows, q_width),
            Gemma4MoeGraphBindingClass::PositionAndKv,
            None,
        );
        let attention_output = linear_node(
            &mut nodes,
            layer,
            "o_proj",
            Gemma4MoeLinearRole::AttentionO,
            format!("{prefix}.self_attn.o_proj.weight"),
            q_width,
            hidden_size,
            attention,
            hidden_shape.clone(),
        );
        let post_attention = norm_node(
            &mut nodes,
            layer,
            "post_attention_norm",
            Gemma4MoeNormRole::PostAttention,
            format!("{prefix}.post_attention_layernorm.weight"),
            attention_output,
            hidden_shape.clone(),
            REVIEWED_RMS_NORM_EPSILON_BITS,
        );
        let attention_residual = push_node(
            &mut nodes,
            format!("layer.{layer}.attention_residual"),
            Some(layer),
            Gemma4MoeGraphNodeKind::Add {
                role: Gemma4MoeAddRole::AttentionResidual,
            },
            vec![hidden, post_attention],
            hidden_shape.clone(),
            Gemma4MoeGraphBindingClass::TokenRows,
            None,
        );

        // Dense MLP is the shared branch.  It has its own pre/post norms.
        let pre_shared = norm_node(
            &mut nodes,
            layer,
            "pre_shared_feedforward_norm",
            Gemma4MoeNormRole::PreSharedFeedforward,
            format!("{prefix}.pre_feedforward_layernorm.weight"),
            attention_residual,
            hidden_shape.clone(),
            REVIEWED_RMS_NORM_EPSILON_BITS,
        );
        let shared_gate = linear_node(
            &mut nodes,
            layer,
            "shared_mlp_gate",
            Gemma4MoeLinearRole::SharedMlpGate,
            format!("{prefix}.mlp.gate_proj.weight"),
            hidden_size,
            dense_intermediate_size,
            pre_shared,
            dense(rows, dense_intermediate_size),
        );
        let shared_up = linear_node(
            &mut nodes,
            layer,
            "shared_mlp_up",
            Gemma4MoeLinearRole::SharedMlpUp,
            format!("{prefix}.mlp.up_proj.weight"),
            hidden_size,
            dense_intermediate_size,
            pre_shared,
            dense(rows, dense_intermediate_size),
        );
        let shared_activated = push_node(
            &mut nodes,
            format!("layer.{layer}.shared_gelu_tanh_mul"),
            Some(layer),
            Gemma4MoeGraphNodeKind::GeluTanhMul,
            vec![shared_gate, shared_up],
            dense(rows, dense_intermediate_size),
            Gemma4MoeGraphBindingClass::TokenRows,
            None,
        );
        let shared_down = linear_node(
            &mut nodes,
            layer,
            "shared_mlp_down",
            Gemma4MoeLinearRole::SharedMlpDown,
            format!("{prefix}.mlp.down_proj.weight"),
            dense_intermediate_size,
            hidden_size,
            shared_activated,
            hidden_shape.clone(),
        );
        let post_shared = norm_node(
            &mut nodes,
            layer,
            "post_shared_feedforward_norm",
            Gemma4MoeNormRole::PostSharedFeedforward,
            format!("{prefix}.post_feedforward_layernorm_1.weight"),
            shared_down,
            hidden_shape.clone(),
            REVIEWED_RMS_NORM_EPSILON_BITS,
        );

        // Router consumes the residual before either feedforward branch.
        let router_norm = push_node(
            &mut nodes,
            format!("layer.{layer}.router_norm"),
            Some(layer),
            Gemma4MoeGraphNodeKind::RmsNorm {
                role: Gemma4MoeNormRole::RouterScaleLess,
                epsilon_bits: REVIEWED_RMS_NORM_EPSILON_BITS,
                scale_mode: Gemma4MoeRmsScaleMode::NoAffineScale,
                weight: None,
            },
            vec![attention_residual],
            hidden_shape.clone(),
            Gemma4MoeGraphBindingClass::RoutingWorkspace,
            None,
        );
        let router_scaled = push_node(
            &mut nodes,
            format!("layer.{layer}.router_root_scale"),
            Some(layer),
            Gemma4MoeGraphNodeKind::RouterRootScale {
                scale_weight: format!("{prefix}.router.scale"),
                hidden_root_reciprocal_bits: (hidden_size as f32).powf(-0.5).to_bits(),
            },
            vec![router_norm],
            hidden_shape.clone(),
            Gemma4MoeGraphBindingClass::RoutingWorkspace,
            None,
        );
        let router_logits = linear_node(
            &mut nodes,
            layer,
            "router_projection",
            Gemma4MoeLinearRole::RouterProjection,
            format!("{prefix}.router.proj.weight"),
            hidden_size,
            u64::from(config.expert_count),
            router_scaled,
            dense(rows, u64::from(config.expert_count)),
        );
        let routes = push_node(
            &mut nodes,
            format!("layer.{layer}.stable_topk_router"),
            Some(layer),
            Gemma4MoeGraphNodeKind::StableTopKRouter {
                expert_count: config.expert_count,
                top_k: config.selected_expert_count,
                full_softmax: true,
                renormalize_selected_weights: true,
                stable_tie_break_by_lower_expert: true,
                output_weight_semantics:
                    Gemma4MoeRouteWeightSemantics::TopKRenormalizedWithoutExpertScale,
            },
            vec![router_logits],
            Gemma4MoeValueShape::Routes {
                rows,
                expert_count: config.expert_count,
                top_k: config.selected_expert_count,
            },
            Gemma4MoeGraphBindingClass::RoutingWorkspace,
            None,
        );
        let pre_routed = norm_node(
            &mut nodes,
            layer,
            "pre_routed_feedforward_norm",
            Gemma4MoeNormRole::PreRoutedFeedforward,
            format!("{prefix}.pre_feedforward_layernorm_2.weight"),
            attention_residual,
            hidden_shape.clone(),
            REVIEWED_RMS_NORM_EPSILON_BITS,
        );
        let routed = push_node(
            &mut nodes,
            format!("layer.{layer}.routed_experts_nvfp4"),
            Some(layer),
            Gemma4MoeGraphNodeKind::RoutedExpertsNvfp4 {
                family: Gemma4MoeGraphExpertTensorFamily {
                    layer,
                    expert_count: config.expert_count,
                    hidden_size,
                    intermediate_size: expert_intermediate_size,
                    block_size: REVIEWED_NVFP4_BLOCK_SIZE,
                },
                per_expert_scale_weight: format!("{prefix}.router.per_expert_scale"),
                provider_contract: Gemma4MoeProviderContract::Nvfp4RoutedExpertsV1,
                only_selected_experts: true,
                apply_scale_after_topk_renormalization: true,
            },
            vec![pre_routed, routes],
            hidden_shape.clone(),
            Gemma4MoeGraphBindingClass::RoutingWorkspace,
            None,
        );
        let post_routed = norm_node(
            &mut nodes,
            layer,
            "post_routed_feedforward_norm",
            Gemma4MoeNormRole::PostRoutedFeedforward,
            format!("{prefix}.post_feedforward_layernorm_2.weight"),
            routed,
            hidden_shape.clone(),
            REVIEWED_RMS_NORM_EPSILON_BITS,
        );
        let combined = push_node(
            &mut nodes,
            format!("layer.{layer}.combine_shared_and_routed"),
            Some(layer),
            Gemma4MoeGraphNodeKind::Add {
                role: Gemma4MoeAddRole::SharedAndRoutedBranches,
            },
            vec![post_shared, post_routed],
            hidden_shape.clone(),
            Gemma4MoeGraphBindingClass::TokenRows,
            None,
        );
        let post_combined = norm_node(
            &mut nodes,
            layer,
            "post_combined_feedforward_norm",
            Gemma4MoeNormRole::PostCombinedFeedforward,
            format!("{prefix}.post_feedforward_layernorm.weight"),
            combined,
            hidden_shape.clone(),
            REVIEWED_RMS_NORM_EPSILON_BITS,
        );
        let feedforward_residual = push_node(
            &mut nodes,
            format!("layer.{layer}.feedforward_residual"),
            Some(layer),
            Gemma4MoeGraphNodeKind::Add {
                role: Gemma4MoeAddRole::FeedforwardResidual,
            },
            vec![attention_residual, post_combined],
            hidden_shape.clone(),
            Gemma4MoeGraphBindingClass::TokenRows,
            None,
        );
        hidden = push_node(
            &mut nodes,
            format!("layer.{layer}.layer_scalar"),
            Some(layer),
            Gemma4MoeGraphNodeKind::ScaleWeight {
                weight: format!("{prefix}.layer_scalar"),
            },
            vec![feedforward_residual],
            hidden_shape.clone(),
            Gemma4MoeGraphBindingClass::TokenRows,
            (layer_index + 1 == config.layer_count as usize)
                .then_some(Gemma4MoeExecutionBoundary::StatePublication),
        );

        let kv = Gemma4MoeKvDescriptor {
            layer,
            heads: kv_heads,
            head_dim,
            capacity: state_capacity,
            retention_window: window,
            k_equals_v_before_norm: k_equals_v,
            storage_format: Gemma4MoeKvStorageFormat::Fp8E4M3Fn,
            scale_source: Gemma4MoeKvScaleSource::ImplicitUnitModelOptFp8CastAmax448,
            dequant_scale_f32_bits: 1.0_f32.to_bits(),
            serialized_scale_tensor_count: 0,
        };
        layers.push(Gemma4MoeLayerPlan {
            layer,
            layer_type,
            attention: attention_descriptor,
            rope,
            kv,
            first_node,
            last_node: hidden,
        });
    }

    let final_norm = norm_node_without_layer(
        &mut nodes,
        "final_norm",
        Gemma4MoeNormRole::Final,
        format!("{MODEL_PREFIX}.norm.weight"),
        hidden,
        hidden_shape,
        REVIEWED_RMS_NORM_EPSILON_BITS,
    );
    let logits = push_node(
        &mut nodes,
        "logits",
        None,
        Gemma4MoeGraphNodeKind::TiedOutputProjection { embedding_weight },
        vec![final_norm],
        dense(rows, vocab_size),
        Gemma4MoeGraphBindingClass::TerminalOutput,
        None,
    );
    let softcapped = push_node(
        &mut nodes,
        "logit_softcap",
        None,
        Gemma4MoeGraphNodeKind::LogitSoftcap {
            cap_bits: 30.0_f32.to_bits(),
        },
        vec![logits],
        dense(rows, vocab_size),
        Gemma4MoeGraphBindingClass::TerminalOutput,
        None,
    );
    push_node(
        &mut nodes,
        "argmax",
        None,
        Gemma4MoeGraphNodeKind::Argmax,
        vec![softcapped],
        Gemma4MoeValueShape::TokenIndices { rows },
        Gemma4MoeGraphBindingClass::TerminalOutput,
        Some(Gemma4MoeExecutionBoundary::TerminalReadback),
    );

    validate_graph(&nodes, &layers, config)?;
    Ok(Gemma4MoeGraph {
        model_fingerprint: model_fingerprint.to_owned(),
        source_container_identity: source_container_identity.to_owned(),
        config: config.clone(),
        token_count,
        start_position,
        expected_length,
        state_capacity,
        nodes,
        layers,
        resident_tensors,
    })
}

fn dense(rows: u64, width: u64) -> Gemma4MoeValueShape {
    Gemma4MoeValueShape::Dense { rows, width }
}

#[allow(clippy::too_many_arguments)]
fn push_node(
    nodes: &mut Vec<Gemma4MoeGraphNode>,
    label: impl Into<String>,
    layer: Option<u32>,
    kind: Gemma4MoeGraphNodeKind,
    predecessors: Vec<usize>,
    output_shape: Gemma4MoeValueShape,
    binding_class: Gemma4MoeGraphBindingClass,
    boundary_after: Option<Gemma4MoeExecutionBoundary>,
) -> usize {
    let id = nodes.len();
    nodes.push(Gemma4MoeGraphNode {
        id,
        label: label.into(),
        layer,
        kind,
        predecessors,
        output_shape,
        binding_class,
        boundary_after,
    });
    id
}

#[allow(clippy::too_many_arguments)]
fn linear_node(
    nodes: &mut Vec<Gemma4MoeGraphNode>,
    layer: u32,
    label: &str,
    role: Gemma4MoeLinearRole,
    weight: String,
    input_features: u64,
    output_features: u64,
    predecessor: usize,
    output_shape: Gemma4MoeValueShape,
) -> usize {
    push_node(
        nodes,
        format!("layer.{layer}.{label}"),
        Some(layer),
        Gemma4MoeGraphNodeKind::Linear {
            role,
            weight,
            input_features,
            output_features,
        },
        vec![predecessor],
        output_shape,
        if role == Gemma4MoeLinearRole::RouterProjection {
            Gemma4MoeGraphBindingClass::RoutingWorkspace
        } else {
            Gemma4MoeGraphBindingClass::TokenRows
        },
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn norm_node(
    nodes: &mut Vec<Gemma4MoeGraphNode>,
    layer: u32,
    label: &str,
    role: Gemma4MoeNormRole,
    weight: String,
    predecessor: usize,
    output_shape: Gemma4MoeValueShape,
    epsilon_bits: u32,
) -> usize {
    push_node(
        nodes,
        format!("layer.{layer}.{label}"),
        Some(layer),
        Gemma4MoeGraphNodeKind::RmsNorm {
            role,
            epsilon_bits,
            scale_mode: Gemma4MoeRmsScaleMode::Direct,
            weight: Some(weight),
        },
        vec![predecessor],
        output_shape,
        Gemma4MoeGraphBindingClass::TokenRows,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn norm_node_without_layer(
    nodes: &mut Vec<Gemma4MoeGraphNode>,
    label: &str,
    role: Gemma4MoeNormRole,
    weight: String,
    predecessor: usize,
    output_shape: Gemma4MoeValueShape,
    epsilon_bits: u32,
) -> usize {
    push_node(
        nodes,
        label,
        None,
        Gemma4MoeGraphNodeKind::RmsNorm {
            role,
            epsilon_bits,
            scale_mode: Gemma4MoeRmsScaleMode::Direct,
            weight: Some(weight),
        },
        vec![predecessor],
        output_shape,
        Gemma4MoeGraphBindingClass::TokenRows,
        None,
    )
}

fn validate_graph(
    nodes: &[Gemma4MoeGraphNode],
    layers: &[Gemma4MoeLayerPlan],
    config: &Gemma4MoeConfig,
) -> Result<(), Gemma4MoeGraphError> {
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
        || layers.len() != config.layer_count as usize
        || layers.iter().enumerate().any(|(index, layer)| {
            layer.layer as usize != index
                || layer.layer_type != config.layer_types[index]
                || layer.first_node > layer.last_node
                || layer.last_node >= nodes.len()
        })
    {
        return Err(Gemma4MoeGraphError::InvalidTopology);
    }
    let state_boundaries = nodes
        .iter()
        .filter(|node| node.boundary_after == Some(Gemma4MoeExecutionBoundary::StatePublication))
        .count();
    let terminal_boundaries = nodes
        .iter()
        .filter(|node| node.boundary_after == Some(Gemma4MoeExecutionBoundary::TerminalReadback))
        .count();
    let routed_nodes = nodes
        .iter()
        .filter(|node| {
            matches!(
                node.kind,
                Gemma4MoeGraphNodeKind::RoutedExpertsNvfp4 {
                    family: Gemma4MoeGraphExpertTensorFamily {
                        expert_count: 128,
                        hidden_size: 2_816,
                        intermediate_size: 704,
                        block_size: 16,
                        ..
                    },
                    provider_contract: Gemma4MoeProviderContract::Nvfp4RoutedExpertsV1,
                    only_selected_experts: true,
                    apply_scale_after_topk_renormalization: true,
                    ..
                }
            )
        })
        .count();
    let router_nodes = nodes
        .iter()
        .filter(|node| {
            matches!(
                node.kind,
                Gemma4MoeGraphNodeKind::StableTopKRouter {
                    expert_count: 128,
                    top_k: 8,
                    full_softmax: true,
                    renormalize_selected_weights: true,
                    stable_tie_break_by_lower_expert: true,
                    output_weight_semantics:
                        Gemma4MoeRouteWeightSemantics::TopKRenormalizedWithoutExpertScale,
                    ..
                }
            )
        })
        .count();
    if state_boundaries != 1
        || terminal_boundaries != 1
        || routed_nodes != config.layer_count as usize
        || router_nodes != config.layer_count as usize
    {
        return Err(Gemma4MoeGraphError::InvalidTopology);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reviewed_config() -> Gemma4MoeConfig {
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
            layer_types: reviewed_graph_layer_schedule().to_vec(),
        }
    }

    fn reviewed_recipe() -> Gemma4MoeRecipe {
        Gemma4MoeRecipe {
            encoding: QuantizedTensorEncoding::Nvfp4E2M1Block16E4M3FnF32Outer,
            block_size: 16,
            value_format: "E2M1",
            block_scale_format: "E4M3FN",
            outer_scale_format: "F32",
            input_scale_format: "F32",
            activation_dynamic: false,
            kv_cache_format: "FP8",
            kv_cache_scale_source: "modelopt-fp8-cast-constant-amax-448",
            kv_cache_dequant_scale_f32_bits: 1.0_f32.to_bits(),
            kv_cache_scale_tensor_count: 0,
            producer: "modelopt@0.43.0rc2.dev91+gc79ebc014".to_owned(),
        }
    }

    fn tensor<'a>(
        catalog: &'a [Gemma4MoeGraphTensorSpec],
        name: &str,
    ) -> &'a Gemma4MoeGraphTensorSpec {
        catalog
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap_or_else(|| panic!("missing fixture tensor {name}"))
    }

    #[test]
    fn reviewed_config_has_five_sliding_layers_then_one_full_layer() {
        let config = reviewed_config();
        validate_identity_config(&config).unwrap();
        validate_identity_recipe(&reviewed_recipe()).unwrap();
        assert_eq!(config.layer_types.len(), 30);
        assert_eq!(
            config
                .layer_types
                .iter()
                .filter(|kind| **kind == Gemma4LayerType::SlidingAttention)
                .count(),
            25
        );
        assert_eq!(
            config
                .layer_types
                .iter()
                .enumerate()
                .filter_map(
                    |(layer, kind)| (*kind == Gemma4LayerType::FullAttention).then_some(layer)
                )
                .collect::<Vec<_>>(),
            vec![5, 11, 17, 23, 29]
        );

        let mut unreviewed = config;
        unreviewed.selected_expert_count = 4;
        assert_eq!(
            validate_identity_config(&unreviewed),
            Err(Gemma4MoeGraphError::UnreviewedConfig)
        );
        let mut unreviewed_recipe = reviewed_recipe();
        unreviewed_recipe.kv_cache_scale_tensor_count = 1;
        assert_eq!(
            validate_identity_recipe(&unreviewed_recipe),
            Err(Gemma4MoeGraphError::UnreviewedRecipe)
        );
    }

    #[test]
    fn text_catalog_matches_exact_attention_and_nvfp4_shapes() {
        let config = reviewed_config();
        let catalog = expected_gemma4_moe_text_tensor_catalog(&config).unwrap();
        assert_eq!(catalog.len(), GEMMA4_MOE_TEXT_TENSOR_COUNT);
        assert_eq!(
            tensor(
                &catalog,
                "model.language_model.layers.0.self_attn.q_proj.weight"
            )
            .stored_shape,
            vec![4_096, 2_816]
        );
        assert_eq!(
            tensor(
                &catalog,
                "model.language_model.layers.0.self_attn.v_proj.weight"
            )
            .stored_shape,
            vec![2_048, 2_816]
        );
        assert_eq!(
            tensor(
                &catalog,
                "model.language_model.layers.5.self_attn.q_proj.weight"
            )
            .stored_shape,
            vec![8_192, 2_816]
        );
        assert!(
            catalog.iter().all(
                |tensor| tensor.name != "model.language_model.layers.5.self_attn.v_proj.weight"
            )
        );

        let packed = tensor(
            &catalog,
            "model.language_model.layers.0.experts.127.gate_proj.weight",
        );
        assert_eq!(packed.dtype, Gemma4MoeGraphTensorDtype::U8);
        assert_eq!(packed.stored_shape, vec![704, 1_408]);
        assert_eq!(packed.logical_shape, vec![704, 2_816]);
        assert_eq!(
            packed.encoding,
            Gemma4MoeGraphTensorEncoding::Nvfp4PackedBlock16
        );
        let block_scale = tensor(
            &catalog,
            "model.language_model.layers.0.experts.0.down_proj.weight_scale",
        );
        assert_eq!(block_scale.dtype, Gemma4MoeGraphTensorDtype::Fp8E4M3);
        assert_eq!(block_scale.stored_shape, vec![2_816, 44]);
        assert_eq!(
            block_scale.encoding,
            Gemma4MoeGraphTensorEncoding::Nvfp4BlockScaleE4M3
        );
        assert!(
            catalog
                .iter()
                .all(|tensor| !tensor.name.starts_with("model.embed_vision"))
        );
    }

    #[test]
    fn graph_models_separate_attention_geometries_and_branch_order() {
        let config = reviewed_config();
        let graph = build_gemma4_moe_graph_from_config(&config, 17, 1_007, 2_048).unwrap();
        assert_eq!(graph.expected_length(), 1_024);
        let sliding = &graph.layers()[0];
        assert_eq!(sliding.attention.q_width, 4_096);
        assert_eq!(sliding.attention.kv_width, 2_048);
        assert_eq!(sliding.attention.sliding_window, Some(1_024));
        assert!(!sliding.attention.k_equals_v_before_norm);
        assert_eq!(sliding.rope.rotary_dim, 256);
        assert_eq!(
            sliding.kv.storage_format,
            Gemma4MoeKvStorageFormat::Fp8E4M3Fn
        );
        assert_eq!(
            sliding.kv.scale_source,
            Gemma4MoeKvScaleSource::ImplicitUnitModelOptFp8CastAmax448
        );
        assert_eq!(f32::from_bits(sliding.kv.dequant_scale_f32_bits), 1.0);
        assert_eq!(sliding.kv.serialized_scale_tensor_count, 0);

        let full = &graph.layers()[5];
        assert_eq!(full.attention.q_width, 8_192);
        assert_eq!(full.attention.kv_width, 1_024);
        assert_eq!(full.attention.sliding_window, None);
        assert!(full.attention.k_equals_v_before_norm);
        assert_eq!(full.rope.rotary_dim, 128);
        assert_eq!(full.kv.serialized_scale_tensor_count, 0);
        assert!(graph.resident_tensors().iter().all(|tensor| {
            !tensor.name.ends_with(".k_scale") && !tensor.name.ends_with(".v_scale")
        }));

        let combined = graph
            .nodes()
            .iter()
            .find(|node| node.label() == "layer.0.combine_shared_and_routed")
            .unwrap();
        assert!(matches!(
            combined.kind(),
            Gemma4MoeGraphNodeKind::Add {
                role: Gemma4MoeAddRole::SharedAndRoutedBranches
            }
        ));
        let predecessor_labels = combined
            .predecessors()
            .iter()
            .map(|id| graph.nodes()[*id].label())
            .collect::<Vec<_>>();
        assert_eq!(
            predecessor_labels,
            vec![
                "layer.0.post_shared_feedforward_norm",
                "layer.0.post_routed_feedforward_norm"
            ]
        );
        let router = graph
            .nodes()
            .iter()
            .find(|node| node.label() == "layer.0.stable_topk_router")
            .unwrap();
        assert!(matches!(
            router.kind(),
            Gemma4MoeGraphNodeKind::StableTopKRouter {
                renormalize_selected_weights: true,
                output_weight_semantics:
                    Gemma4MoeRouteWeightSemantics::TopKRenormalizedWithoutExpertScale,
                ..
            }
        ));
        let routed = graph
            .nodes()
            .iter()
            .find(|node| node.label() == "layer.0.routed_experts_nvfp4")
            .unwrap();
        let Gemma4MoeGraphNodeKind::RoutedExpertsNvfp4 {
            provider_contract,
            only_selected_experts,
            per_expert_scale_weight,
            apply_scale_after_topk_renormalization,
            ..
        } = routed.kind()
        else {
            panic!("wrong routed node kind")
        };
        assert!(*only_selected_experts);
        assert!(*apply_scale_after_topk_renormalization);
        assert_eq!(
            per_expert_scale_weight,
            "model.language_model.layers.0.router.per_expert_scale"
        );
        assert!(!provider_contract.is_qwen_sparse_moe_compatible());
    }

    #[test]
    fn graph_covers_non_aligned_token_and_window_boundaries() {
        let config = reviewed_config();
        for tokens in [1, 3, 7, 8, 17, 31, 32, 33] {
            let graph = build_gemma4_moe_graph_from_config(&config, tokens, 0, 64).unwrap();
            assert_eq!(graph.expected_length(), tokens);
            assert_eq!(graph.token_count(), tokens);
        }
        for expected in [1_023, 1_024, 1_025] {
            let graph =
                build_gemma4_moe_graph_from_config(&config, 1, expected - 1, 2_048).unwrap();
            assert_eq!(graph.expected_length(), expected);
            assert_eq!(graph.layers()[0].kv.retention_window, Some(1_024));
        }
    }

    #[test]
    fn graph_fails_closed_on_lengths_and_catalog_drift() {
        let config = reviewed_config();
        assert_eq!(
            build_gemma4_moe_graph_from_config(&config, 0, 0, 1),
            Err(Gemma4MoeGraphError::ZeroTokenCount)
        );
        assert_eq!(
            build_gemma4_moe_graph_from_config(&config, 1, 0, 0),
            Err(Gemma4MoeGraphError::ZeroStateCapacity)
        );
        assert_eq!(
            build_gemma4_moe_graph_from_config(&config, 2, u64::MAX, u64::MAX),
            Err(Gemma4MoeGraphError::PositionOverflow)
        );
        assert_eq!(
            build_gemma4_moe_graph_from_config(&config, 2, 7, 8),
            Err(Gemma4MoeGraphError::LengthOutOfBounds)
        );

        let graph = build_gemma4_moe_graph_from_config(&config, 1, 0, 8).unwrap();
        let mut actual = graph.resident_tensors().to_vec();
        actual[0].stored_shape[0] -= 1;
        assert!(matches!(
            graph.validate_resident_catalog(&actual),
            Err(Gemma4MoeGraphError::TensorMismatch(_))
        ));
        let mut missing = graph.resident_tensors().to_vec();
        let missing_name = missing.pop().unwrap().name;
        assert_eq!(
            graph.validate_resident_catalog(&missing),
            Err(Gemma4MoeGraphError::MissingTensor(missing_name))
        );
    }
}
