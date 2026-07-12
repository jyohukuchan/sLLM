// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Backend-independent attention-stack contract fixtures.
//!
//! These fixtures exercise graph, weight-binding, state-schema, and structural
//! adapter-admission composition. They are not production package adapters:
//! bindings intentionally contain no payload hashes or canonical payload
//! evidence. Embeddings, MLPs, final normalization, the LM head, and
//! source-specific Qwen3.5 details such as query gates or checkpoint reorderings
//! are outside their scope and require later verified adapter composition.
//! Graph values, weights, and logical states use F32 as an oracle/contract
//! representation; this does not claim a production checkpoint storage format.
//! Delta layers fix the semantic order `depthwise convolution -> SiLU -> split`.
//! Counts are bounded before bulk reservation, but small identifier, shape, and
//! node-edge vectors still use ordinary Rust allocation; this is not a complete
//! fallible-allocation or process-OOM guarantee.

use crate::{
    adapter_admission::{
        AdapterAdmissionError, AdapterAdmissionSpec, CanonicalWeightUse, CanonicalizationRecipe,
        NodeLayerBinding, StructurallyAdmittedAdapter, validate_structural_adapter_admission,
    },
    model_graph::{
        ActivationKind, GraphNode, GraphNodeKind, GraphValue, ModelGraph, NodeId,
        NormalizationAffine, NormalizationAxis, NormalizationKind, NumericalFormat, PositiveF32,
        RotaryPairing, StateId, TensorLayout, TensorSpec, ValueId, WeightBinding, WeightBindings,
        WeightId, WeightSpec,
    },
    sq8_layer_oracle::{
        QWEN3_14B_HEAD_DIM, QWEN3_14B_HIDDEN_SIZE, QWEN3_14B_KV_HEADS, QWEN3_14B_Q_HEADS,
        QWEN3_14B_RMS_NORM_EPSILON, QWEN3_14B_ROPE_THETA, QWEN3_14B_VALUE_DIM,
    },
    sq8_stack_runtime::QWEN3_14B_SQ8_STACK_LAYERS,
    state_schema::{
        SnapshotRestorePolicy, StateExecutionProtocol, StateInitialization, StateKind, StateLayout,
        StateOwnership, StateResetProtocol, StateSchema, StateSpec, StateTransactionContract,
    },
};

const MAX_FIXTURE_TOKENS: usize = 1_024;
const FIXTURE_DECLARATION_LIMIT: usize = 4_096;
// Contract-only paged-cache geometry: 2048 logical token slots per dense layer.
const FIXTURE_KV_BLOCK_SIZE: usize = 16;
const FIXTURE_KV_CACHE_BLOCKS: usize = 128;
const QWEN35_HIDDEN_SIZE: usize = 4_096;
const QWEN35_LAYERS: usize = 32;
const FIXTURE_RMS_EPSILON: f32 = 1.0e-6;

/// Owned four-way attention-stack contract fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterContractFixture {
    /// Backend-independent graph contract.
    pub graph: ModelGraph,
    /// Structural logical-to-physical fixture bindings without payload hashes.
    pub bindings: WeightBindings,
    /// Request-layer state contract.
    pub states: StateSchema,
    /// Exact layer and logical-weight occurrence admission claim.
    pub admission: AdapterAdmissionSpec,
}

impl AdapterContractFixture {
    /// Revalidates the exact four owned components and returns an instance-bound proof.
    pub fn validate_structural(
        &self,
    ) -> Result<StructurallyAdmittedAdapter<'_>, AdapterAdmissionError> {
        validate_structural_adapter_admission(
            &self.graph,
            &self.bindings,
            &self.states,
            &self.admission,
        )
    }
}

/// Geometry for one explicit dense grouped-query attention layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseGqaGeometry {
    /// Number of query heads.
    pub q_heads: usize,
    /// Number of shared key/value heads.
    pub kv_heads: usize,
    /// Key/query coordinates per head.
    pub head_dim: usize,
    /// Value coordinates per head.
    pub value_dim: usize,
    /// Even rotary prefix width per head.
    pub rotary_dim: usize,
    /// Finite positive rotary frequency base.
    pub rotary_base: f32,
    /// Finite positive epsilon for the headwise query/key RMS normalizations.
    pub qk_norm_epsilon: PositiveF32,
}

/// Geometry for one explicit gated delta-rule linear-attention layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaRuleGeometry {
    /// Number of query/key heads.
    pub key_heads: usize,
    /// Number of value and recurrent-state heads.
    pub value_heads: usize,
    /// Query/key coordinates per head.
    pub key_dim: usize,
    /// Value coordinates per head.
    pub value_dim: usize,
    /// Causal depthwise convolution kernel width.
    pub kernel_size: usize,
}

#[derive(Debug, Clone, Copy)]
enum LayerContract {
    Dense(DenseGqaGeometry),
    Delta(DeltaRuleGeometry),
}

#[derive(Debug, Clone, Copy)]
struct PredictedCounts {
    nodes: usize,
    values: usize,
    weights: usize,
    states: usize,
    weight_edges: usize,
    state_edges: usize,
    layer_bindings: usize,
}

struct FixtureBuilder {
    tokens: usize,
    hidden: usize,
    graph_id: &'static str,
    values: Vec<GraphValue>,
    weights: Vec<WeightSpec>,
    nodes: Vec<GraphNode>,
    bindings: Vec<WeightBinding>,
    states: Vec<StateSpec>,
    node_layers: Vec<NodeLayerBinding>,
    weight_uses: Vec<CanonicalWeightUse>,
    positions: ValueId,
}

/// Builds the 40-layer Qwen3-14B dense-attention stack contract fixture.
pub fn build_qwen3_dense_attention_fixture(
    tokens: usize,
) -> Result<AdapterContractFixture, String> {
    validate_tokens(tokens)?;
    let geometry = DenseGqaGeometry {
        q_heads: QWEN3_14B_Q_HEADS,
        kv_heads: QWEN3_14B_KV_HEADS,
        head_dim: QWEN3_14B_HEAD_DIM,
        value_dim: QWEN3_14B_VALUE_DIM,
        rotary_dim: QWEN3_14B_HEAD_DIM,
        rotary_base: QWEN3_14B_ROPE_THETA,
        qk_norm_epsilon: PositiveF32::new(
            QWEN3_14B_RMS_NORM_EPSILON,
            "Qwen3 query/key norm epsilon",
        )?,
    };
    let mut layers = try_vec(QWEN3_14B_SQ8_STACK_LAYERS, "Qwen3 dense layer contracts")?;
    for _ in 0..QWEN3_14B_SQ8_STACK_LAYERS {
        layers.push(LayerContract::Dense(geometry));
    }
    build_fixture(
        "qwen3-dense-attention-contract",
        tokens,
        QWEN3_14B_HIDDEN_SIZE,
        &layers,
        QWEN3_14B_RMS_NORM_EPSILON,
    )
}

/// Builds the 32-layer Qwen3.5 hybrid attention-stack contract fixture.
///
/// The required dense geometry is an explicit caller input because verified
/// package evidence, rather than this fixture, must establish its source model.
pub fn build_qwen35_hybrid_attention_fixture(
    tokens: usize,
    dense_geometry: DenseGqaGeometry,
) -> Result<AdapterContractFixture, String> {
    validate_tokens(tokens)?;
    validate_dense_geometry(QWEN35_HIDDEN_SIZE, dense_geometry)?;
    let delta = DeltaRuleGeometry {
        key_heads: 16,
        value_heads: 32,
        key_dim: 128,
        value_dim: 128,
        kernel_size: 4,
    };
    validate_delta_geometry(delta)?;
    let mut layers = try_vec(QWEN35_LAYERS, "hybrid layer contracts")?;
    for index in 0..QWEN35_LAYERS {
        layers.push(if index % 4 == 3 {
            LayerContract::Dense(dense_geometry)
        } else {
            LayerContract::Delta(delta)
        });
    }
    build_fixture(
        "qwen35-hybrid-attention-contract",
        tokens,
        QWEN35_HIDDEN_SIZE,
        &layers,
        FIXTURE_RMS_EPSILON,
    )
}

fn build_fixture(
    graph_id: &'static str,
    tokens: usize,
    hidden: usize,
    layers: &[LayerContract],
    rms_epsilon: f32,
) -> Result<AdapterContractFixture, String> {
    validate_tokens(tokens)?;
    if hidden == 0 || layers.is_empty() {
        return Err("fixture hidden width and layer count must be nonzero".into());
    }
    for layer in layers {
        match layer {
            LayerContract::Dense(geometry) => validate_dense_geometry(hidden, *geometry)?,
            LayerContract::Delta(geometry) => validate_delta_geometry(*geometry)?,
        }
    }
    let counts = predict_counts(layers)?;
    validate_predicted_counts(counts)?;
    let epsilon = PositiveF32::new(rms_epsilon, "fixture RMS epsilon")?;
    let mut builder = FixtureBuilder::new(graph_id, tokens, hidden, counts)?;
    let mut hidden_value = builder.initial_hidden()?;
    for (layer_index, layer) in layers.iter().enumerate() {
        hidden_value = match layer {
            LayerContract::Dense(geometry) => {
                builder.add_dense_layer(layer_index, &hidden_value, *geometry, epsilon)?
            }
            LayerContract::Delta(geometry) => {
                builder.add_delta_layer(layer_index, &hidden_value, *geometry, epsilon)?
            }
        };
    }
    builder.finish(hidden_value)
}

impl FixtureBuilder {
    fn new(
        graph_id: &'static str,
        tokens: usize,
        hidden: usize,
        counts: PredictedCounts,
    ) -> Result<Self, String> {
        let mut values = try_vec(counts.values, "fixture values")?;
        let initial = value_id("hidden-0")?;
        values.push(GraphValue {
            id: initial,
            tensor: activation_tensor(tokens, hidden)?,
        });
        let positions = value_id("positions")?;
        values.push(GraphValue {
            id: positions.clone(),
            tensor: TensorSpec::new(vec![tokens], NumericalFormat::U64, TensorLayout::RowMajor)?,
        });
        Ok(Self {
            tokens,
            hidden,
            graph_id,
            values,
            weights: try_vec(counts.weights, "fixture weights")?,
            nodes: try_vec(counts.nodes, "fixture nodes")?,
            bindings: try_vec(counts.weights, "fixture bindings")?,
            states: try_vec(counts.states, "fixture states")?,
            node_layers: try_vec(counts.layer_bindings, "fixture layer bindings")?,
            weight_uses: try_vec(counts.weight_edges, "fixture weight occurrences")?,
            positions,
        })
    }

    fn initial_hidden(&self) -> Result<ValueId, String> {
        value_id("hidden-0")
    }

    fn add_dense_layer(
        &mut self,
        layer: usize,
        input: &ValueId,
        geometry: DenseGqaGeometry,
        epsilon: PositiveF32,
    ) -> Result<ValueId, String> {
        let q_width = checked_mul(geometry.q_heads, geometry.head_dim, "dense query width")?;
        let k_width = checked_mul(geometry.kv_heads, geometry.head_dim, "dense key width")?;
        let v_width = checked_mul(geometry.kv_heads, geometry.value_dim, "dense value width")?;
        let context_width =
            checked_mul(geometry.q_heads, geometry.value_dim, "dense context width")?;
        let prefix = format!("l{layer:02}");

        let norm = self.add_value(&format!("{prefix}-pre-norm"), self.hidden)?;
        let norm_weight = self.add_weight(layer, "pre-norm-scale", &[self.hidden])?;
        self.add_node(
            layer,
            "pre-norm",
            vec![input.clone()],
            vec![norm.clone()],
            vec![norm_weight],
            vec![],
            GraphNodeKind::Norm {
                epsilon,
                kind: NormalizationKind::Rms,
                affine: NormalizationAffine::Scale,
                axis: NormalizationAxis::Last,
            },
        )?;

        let query = self.add_linear(layer, "q-proj", &norm, q_width)?;
        let key = self.add_linear(layer, "k-proj", &norm, k_width)?;
        let value = self.add_linear(layer, "v-proj", &norm, v_width)?;

        let query_norm = self.add_value(&format!("{prefix}-q-norm"), q_width)?;
        let query_norm_weight = self.add_weight(layer, "q-norm-scale", &[geometry.head_dim])?;
        self.add_node(
            layer,
            "q-norm",
            vec![query],
            vec![query_norm.clone()],
            vec![query_norm_weight],
            vec![],
            GraphNodeKind::Norm {
                epsilon: geometry.qk_norm_epsilon,
                kind: NormalizationKind::Rms,
                affine: NormalizationAffine::Scale,
                axis: NormalizationAxis::GroupedLast {
                    groups: geometry.q_heads,
                    group_width: geometry.head_dim,
                },
            },
        )?;
        let key_norm = self.add_value(&format!("{prefix}-k-norm"), k_width)?;
        let key_norm_weight = self.add_weight(layer, "k-norm-scale", &[geometry.head_dim])?;
        self.add_node(
            layer,
            "k-norm",
            vec![key],
            vec![key_norm.clone()],
            vec![key_norm_weight],
            vec![],
            GraphNodeKind::Norm {
                epsilon: geometry.qk_norm_epsilon,
                kind: NormalizationKind::Rms,
                affine: NormalizationAffine::Scale,
                axis: NormalizationAxis::GroupedLast {
                    groups: geometry.kv_heads,
                    group_width: geometry.head_dim,
                },
            },
        )?;

        let rotary_base = PositiveF32::new(geometry.rotary_base, "dense rotary base")?;
        let query_rotary = self.add_value(&format!("{prefix}-q-rope"), q_width)?;
        self.add_node(
            layer,
            "q-rope",
            vec![query_norm, self.positions.clone()],
            vec![query_rotary.clone()],
            vec![],
            vec![],
            GraphNodeKind::RotaryPosition {
                heads: geometry.q_heads,
                head_dim: geometry.head_dim,
                rotary_dim: geometry.rotary_dim,
                base: rotary_base,
                pairing: RotaryPairing::SplitHalf,
            },
        )?;
        let key_rotary = self.add_value(&format!("{prefix}-k-rope"), k_width)?;
        self.add_node(
            layer,
            "k-rope",
            vec![key_norm, self.positions.clone()],
            vec![key_rotary.clone()],
            vec![],
            vec![],
            GraphNodeKind::RotaryPosition {
                heads: geometry.kv_heads,
                head_dim: geometry.head_dim,
                rotary_dim: geometry.rotary_dim,
                base: rotary_base,
                pairing: RotaryPairing::SplitHalf,
            },
        )?;

        let state = self.add_paged_kv_state(layer, geometry)?;
        let context = self.add_value(&format!("{prefix}-context"), context_width)?;
        self.add_node(
            layer,
            "gqa-core",
            vec![query_rotary, key_rotary, value],
            vec![context.clone()],
            vec![],
            vec![state],
            GraphNodeKind::CausalGqaAttentionCore {
                q_heads: geometry.q_heads,
                kv_heads: geometry.kv_heads,
                head_dim: geometry.head_dim,
                value_dim: geometry.value_dim,
                softmax_scale: PositiveF32::new(
                    1.0 / (geometry.head_dim as f32).sqrt(),
                    "dense softmax scale",
                )?,
            },
        )?;
        let projected = self.add_linear(layer, "o-proj", &context, self.hidden)?;
        let output = self.add_value(&format!("hidden-{}", layer + 1), self.hidden)?;
        self.add_node(
            layer,
            "residual",
            vec![input.clone(), projected],
            vec![output.clone()],
            vec![],
            vec![],
            GraphNodeKind::Residual,
        )?;
        Ok(output)
    }

    fn add_delta_layer(
        &mut self,
        layer: usize,
        input: &ValueId,
        geometry: DeltaRuleGeometry,
        epsilon: PositiveF32,
    ) -> Result<ValueId, String> {
        let q_width = checked_mul(geometry.key_heads, geometry.key_dim, "delta query width")?;
        let k_width = q_width;
        let v_width = checked_mul(
            geometry.value_heads,
            geometry.value_dim,
            "delta value width",
        )?;
        let qkv_width = checked_add(
            checked_add(q_width, k_width, "delta query-key width")?,
            v_width,
            "delta qkv width",
        )?;
        let prefix = format!("l{layer:02}");

        let norm = self.add_value(&format!("{prefix}-pre-norm"), self.hidden)?;
        let norm_weight = self.add_weight(layer, "pre-norm-scale", &[self.hidden])?;
        self.add_node(
            layer,
            "pre-norm",
            vec![input.clone()],
            vec![norm.clone()],
            vec![norm_weight],
            vec![],
            GraphNodeKind::Norm {
                epsilon,
                kind: NormalizationKind::Rms,
                affine: NormalizationAffine::Scale,
                axis: NormalizationAxis::Last,
            },
        )?;
        let qkv = self.add_linear(layer, "qkv-proj", &norm, qkv_width)?;
        let convolved = self.add_value(&format!("{prefix}-qkv-conv"), qkv_width)?;
        let kernel =
            self.add_weight(layer, "conv-kernel", &[qkv_width, 1, geometry.kernel_size])?;
        let conv_state = self.add_conv_state(layer, qkv_width, geometry.kernel_size)?;
        self.add_node(
            layer,
            "causal-conv",
            vec![qkv],
            vec![convolved.clone()],
            vec![kernel],
            vec![conv_state],
            GraphNodeKind::CausalDepthwiseConv1d {
                channels: qkv_width,
                kernel_size: geometry.kernel_size,
            },
        )?;

        let activated = self.add_value(&format!("{prefix}-qkv-silu"), qkv_width)?;
        self.add_node(
            layer,
            "qkv-silu",
            vec![convolved],
            vec![activated.clone()],
            vec![],
            vec![],
            GraphNodeKind::Activation {
                kind: ActivationKind::Silu,
            },
        )?;

        let query = self.add_value(&format!("{prefix}-q"), q_width)?;
        let key = self.add_value(&format!("{prefix}-k"), k_width)?;
        let value = self.add_value(&format!("{prefix}-v"), v_width)?;
        self.add_node(
            layer,
            "qkv-split",
            vec![activated],
            vec![query.clone(), key.clone(), value.clone()],
            vec![],
            vec![],
            GraphNodeKind::LastAxisSplit {
                segment_widths: vec![q_width, k_width, v_width],
            },
        )?;

        let l2_epsilon = PositiveF32::new(FIXTURE_RMS_EPSILON, "fixture L2 epsilon")?;
        let query_norm = self.add_value(&format!("{prefix}-q-l2"), q_width)?;
        self.add_node(
            layer,
            "q-l2",
            vec![query],
            vec![query_norm.clone()],
            vec![],
            vec![],
            GraphNodeKind::Norm {
                epsilon: l2_epsilon,
                kind: NormalizationKind::L2,
                affine: NormalizationAffine::FixedScale(PositiveF32::new(
                    1.0 / (geometry.key_dim as f32).sqrt(),
                    "delta query fixed scale",
                )?),
                axis: NormalizationAxis::GroupedLast {
                    groups: geometry.key_heads,
                    group_width: geometry.key_dim,
                },
            },
        )?;
        let key_norm = self.add_value(&format!("{prefix}-k-l2"), k_width)?;
        self.add_node(
            layer,
            "k-l2",
            vec![key],
            vec![key_norm.clone()],
            vec![],
            vec![],
            GraphNodeKind::Norm {
                epsilon: l2_epsilon,
                kind: NormalizationKind::L2,
                affine: NormalizationAffine::None,
                axis: NormalizationAxis::GroupedLast {
                    groups: geometry.key_heads,
                    group_width: geometry.key_dim,
                },
            },
        )?;

        let z = self.add_linear(layer, "z-proj", &norm, v_width)?;
        let decay_control = self.add_linear(layer, "decay-control", &norm, geometry.value_heads)?;
        let update_control =
            self.add_linear(layer, "update-control", &norm, geometry.value_heads)?;
        let log_decay = self.add_value(&format!("{prefix}-log-decay"), geometry.value_heads)?;
        let update_rate = self.add_value(&format!("{prefix}-update-rate"), geometry.value_heads)?;
        let log_rate = self.add_weight(layer, "log-rate", &[geometry.value_heads])?;
        let time_bias = self.add_weight(layer, "time-bias", &[geometry.value_heads])?;
        self.add_node(
            layer,
            "decay-parameters",
            vec![decay_control, update_control],
            vec![log_decay.clone(), update_rate.clone()],
            vec![log_rate, time_bias],
            vec![],
            GraphNodeKind::GatedDecayParameters {
                channels: geometry.value_heads,
            },
        )?;

        let scan_state = self.add_recurrent_state(layer, geometry)?;
        let context = self.add_value(&format!("{prefix}-scan-context"), v_width)?;
        self.add_node(
            layer,
            "delta-scan",
            vec![query_norm, key_norm, value, log_decay, update_rate],
            vec![context.clone()],
            vec![],
            vec![scan_state],
            GraphNodeKind::GatedDeltaRuleScan {
                key_heads: geometry.key_heads,
                value_heads: geometry.value_heads,
                key_dim: geometry.key_dim,
                value_dim: geometry.value_dim,
            },
        )?;

        let context_norm = self.add_value(&format!("{prefix}-context-norm"), v_width)?;
        let context_scale = self.add_weight(layer, "context-norm-scale", &[geometry.value_dim])?;
        self.add_node(
            layer,
            "context-norm",
            vec![context],
            vec![context_norm.clone()],
            vec![context_scale],
            vec![],
            GraphNodeKind::Norm {
                epsilon,
                kind: NormalizationKind::Rms,
                affine: NormalizationAffine::Scale,
                axis: NormalizationAxis::GroupedLast {
                    groups: geometry.value_heads,
                    group_width: geometry.value_dim,
                },
            },
        )?;
        let gated = self.add_value(&format!("{prefix}-gated"), v_width)?;
        self.add_node(
            layer,
            "gated-multiply",
            vec![context_norm, z],
            vec![gated.clone()],
            vec![],
            vec![],
            GraphNodeKind::GatedMultiply {
                activation: ActivationKind::Silu,
            },
        )?;
        let projected = self.add_linear(layer, "o-proj", &gated, self.hidden)?;
        let output = self.add_value(&format!("hidden-{}", layer + 1), self.hidden)?;
        self.add_node(
            layer,
            "residual",
            vec![input.clone(), projected],
            vec![output.clone()],
            vec![],
            vec![],
            GraphNodeKind::Residual,
        )?;
        Ok(output)
    }

    fn add_linear(
        &mut self,
        layer: usize,
        role: &str,
        input: &ValueId,
        output_width: usize,
    ) -> Result<ValueId, String> {
        let output = self.add_value(&format!("l{layer:02}-{role}"), output_width)?;
        let weight = self.add_weight(layer, role, &[output_width, self.value_width(input)?])?;
        self.add_node(
            layer,
            role,
            vec![input.clone()],
            vec![output.clone()],
            vec![weight],
            vec![],
            GraphNodeKind::Linear { has_bias: false },
        )?;
        Ok(output)
    }

    fn add_value(&mut self, name: &str, width: usize) -> Result<ValueId, String> {
        let id = value_id(name)?;
        self.values.push(GraphValue {
            id: id.clone(),
            tensor: activation_tensor(self.tokens, width)?,
        });
        Ok(id)
    }

    fn value_width(&self, id: &ValueId) -> Result<usize, String> {
        self.values
            .iter()
            .find(|value| &value.id == id)
            .and_then(|value| value.tensor.shape.last().copied())
            .ok_or_else(|| "fixture linear input is undeclared".to_string())
    }

    fn add_weight(
        &mut self,
        layer: usize,
        role: &str,
        shape: &[usize],
    ) -> Result<WeightId, String> {
        let id = weight_id(&format!("l{layer:02}-{role}-weight"))?;
        let tensor = TensorSpec::new(shape.to_vec(), NumericalFormat::F32, TensorLayout::RowMajor)?;
        self.weights.push(WeightSpec {
            id: id.clone(),
            tensor: tensor.clone(),
        });
        self.bindings.push(WeightBinding {
            logical_id: id.clone(),
            physical_tensor_name: format!("fixture.l{layer:02}.{role}"),
            tensor,
            content_sha256: None,
        });
        Ok(id)
    }

    fn add_node(
        &mut self,
        layer: usize,
        role: &str,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
        weights: Vec<WeightId>,
        states: Vec<StateId>,
        kind: GraphNodeKind,
    ) -> Result<(), String> {
        let id = node_id(&format!("l{layer:02}-{role}"))?;
        if !states.is_empty() {
            self.node_layers.push(NodeLayerBinding {
                node_id: id.clone(),
                layer_index: layer,
            });
        }
        for (weight_slot, logical_id) in weights.iter().enumerate() {
            let shape = self
                .weights
                .iter()
                .find(|weight| &weight.id == logical_id)
                .map(|weight| weight.tensor.shape.clone())
                .ok_or_else(|| "fixture node weight is undeclared".to_string())?;
            self.weight_uses.push(CanonicalWeightUse {
                node_id: id.clone(),
                weight_slot,
                logical_id: logical_id.clone(),
                recipe: CanonicalizationRecipe {
                    source_shape: shape,
                    steps: vec![],
                },
            });
        }
        self.nodes.push(GraphNode {
            id,
            inputs,
            outputs,
            weights,
            states,
            kind,
        });
        Ok(())
    }

    fn add_paged_kv_state(
        &mut self,
        layer: usize,
        geometry: DenseGqaGeometry,
    ) -> Result<StateId, String> {
        let id = state_id(&format!("l{layer:02}-paged-kv"))?;
        self.states.push(StateSpec {
            id: id.clone(),
            kind: StateKind::PagedKv,
            ownership: StateOwnership::RequestLayer { layer_index: layer },
            format: NumericalFormat::F32,
            layout: StateLayout::PagedKv {
                block_size: FIXTURE_KV_BLOCK_SIZE,
                cache_blocks: FIXTURE_KV_CACHE_BLOCKS,
                q_heads: geometry.q_heads,
                kv_heads: geometry.kv_heads,
                head_dim: geometry.head_dim,
                value_dim: geometry.value_dim,
            },
            transaction: transactional_state(),
        });
        Ok(id)
    }

    fn add_conv_state(
        &mut self,
        layer: usize,
        channels: usize,
        kernel_size: usize,
    ) -> Result<StateId, String> {
        let id = state_id(&format!("l{layer:02}-conv-history"))?;
        self.states.push(StateSpec {
            id: id.clone(),
            kind: StateKind::ConvolutionHistory,
            ownership: StateOwnership::RequestLayer { layer_index: layer },
            format: NumericalFormat::F32,
            layout: StateLayout::ConvolutionHistory {
                channels,
                history_tokens: kernel_size - 1,
            },
            transaction: transactional_state(),
        });
        Ok(id)
    }

    fn add_recurrent_state(
        &mut self,
        layer: usize,
        geometry: DeltaRuleGeometry,
    ) -> Result<StateId, String> {
        let id = state_id(&format!("l{layer:02}-recurrent-bank"))?;
        self.states.push(StateSpec {
            id: id.clone(),
            kind: StateKind::Recurrent,
            ownership: StateOwnership::RequestLayer { layer_index: layer },
            format: NumericalFormat::F32,
            layout: StateLayout::RecurrentBank {
                instances: geometry.value_heads,
                rows: geometry.key_dim,
                cols: geometry.value_dim,
            },
            transaction: transactional_state(),
        });
        Ok(id)
    }

    fn finish(self, output: ValueId) -> Result<AdapterContractFixture, String> {
        let graph = ModelGraph {
            graph_id: self.graph_id.into(),
            inputs: vec![value_id("hidden-0")?, self.positions],
            outputs: vec![output],
            values: self.values,
            weights: self.weights,
            nodes: self.nodes,
        };
        let bindings = WeightBindings {
            bindings: self.bindings,
        };
        let states = StateSchema::new(format!("{}-states", self.graph_id), self.states)?;
        let admission = AdapterAdmissionSpec {
            admission_id: format!("{}-admission", self.graph_id),
            graph_id: graph.graph_id.clone(),
            node_layers: self.node_layers,
            weight_uses: self.weight_uses,
        };
        graph
            .validate_with_bindings(&bindings)
            .map_err(|_| "fixture graph and bindings failed validation".to_string())?;
        states
            .validate_against_graph(&graph)
            .map_err(|_| "fixture state composition failed validation".to_string())?;
        let fixture = AdapterContractFixture {
            graph,
            bindings,
            states,
            admission,
        };
        drop(
            fixture
                .validate_structural()
                .map_err(|_| "fixture structural admission failed".to_string())?,
        );
        Ok(fixture)
    }
}

fn predict_counts(layers: &[LayerContract]) -> Result<PredictedCounts, String> {
    let mut counts = PredictedCounts {
        nodes: 0,
        values: 2,
        weights: 0,
        states: 0,
        weight_edges: 0,
        state_edges: 0,
        layer_bindings: 0,
    };
    for layer in layers {
        let increment = match layer {
            LayerContract::Dense(_) => PredictedCounts {
                nodes: 11,
                values: 11,
                weights: 7,
                states: 1,
                weight_edges: 7,
                state_edges: 1,
                layer_bindings: 1,
            },
            LayerContract::Delta(_) => PredictedCounts {
                nodes: 16,
                values: 20,
                weights: 10,
                states: 2,
                weight_edges: 10,
                state_edges: 2,
                layer_bindings: 2,
            },
        };
        counts.nodes = checked_add(counts.nodes, increment.nodes, "fixture node count")?;
        counts.values = checked_add(counts.values, increment.values, "fixture value count")?;
        counts.weights = checked_add(counts.weights, increment.weights, "fixture weight count")?;
        counts.states = checked_add(counts.states, increment.states, "fixture state count")?;
        counts.weight_edges = checked_add(
            counts.weight_edges,
            increment.weight_edges,
            "fixture weight edge count",
        )?;
        counts.state_edges = checked_add(
            counts.state_edges,
            increment.state_edges,
            "fixture state edge count",
        )?;
        counts.layer_bindings = checked_add(
            counts.layer_bindings,
            increment.layer_bindings,
            "fixture layer binding count",
        )?;
    }
    Ok(counts)
}

fn validate_predicted_counts(counts: PredictedCounts) -> Result<(), String> {
    if [
        counts.nodes,
        counts.values,
        counts.weights,
        counts.states,
        counts.weight_edges,
        counts.state_edges,
        counts.layer_bindings,
    ]
    .into_iter()
    .any(|count| count > FIXTURE_DECLARATION_LIMIT)
    {
        return Err("fixture declaration or edge prediction exceeds 4096".into());
    }
    Ok(())
}

fn validate_tokens(tokens: usize) -> Result<(), String> {
    if !(1..=MAX_FIXTURE_TOKENS).contains(&tokens) {
        return Err("fixture token count must be in 1..=1024".into());
    }
    Ok(())
}

fn validate_dense_geometry(hidden: usize, geometry: DenseGqaGeometry) -> Result<(), String> {
    if geometry.q_heads == 0
        || geometry.kv_heads == 0
        || geometry.head_dim == 0
        || geometry.value_dim == 0
        || geometry.rotary_dim == 0
    {
        return Err("dense geometry dimensions must be nonzero".into());
    }
    if geometry.q_heads % geometry.kv_heads != 0 {
        return Err("dense q_heads must be divisible by kv_heads".into());
    }
    if geometry.rotary_dim % 2 != 0 || geometry.rotary_dim > geometry.head_dim {
        return Err("dense rotary_dim must be even and not exceed head_dim".into());
    }
    PositiveF32::new(geometry.rotary_base, "dense rotary base")?;
    geometry
        .qk_norm_epsilon
        .validate("dense query/key norm epsilon")?;
    let q_width = checked_mul(geometry.q_heads, geometry.head_dim, "dense query width")?;
    if q_width != hidden {
        return Err("dense query width must equal fixture hidden width".into());
    }
    checked_mul(geometry.kv_heads, geometry.head_dim, "dense key width")?;
    checked_mul(geometry.kv_heads, geometry.value_dim, "dense value width")?;
    checked_mul(geometry.q_heads, geometry.value_dim, "dense context width")?;
    Ok(())
}

fn validate_delta_geometry(geometry: DeltaRuleGeometry) -> Result<(), String> {
    if geometry.key_heads == 0
        || geometry.value_heads == 0
        || geometry.key_dim == 0
        || geometry.value_dim == 0
        || geometry.kernel_size < 2
    {
        return Err(
            "delta geometry dimensions must be nonzero and kernel_size at least two".into(),
        );
    }
    if geometry.value_heads % geometry.key_heads != 0 {
        return Err("delta value_heads must be divisible by key_heads".into());
    }
    let q_width = checked_mul(geometry.key_heads, geometry.key_dim, "delta query width")?;
    let v_width = checked_mul(
        geometry.value_heads,
        geometry.value_dim,
        "delta value width",
    )?;
    let qkv = checked_add(
        checked_add(q_width, q_width, "delta query-key width")?,
        v_width,
        "delta qkv width",
    )?;
    checked_mul(
        qkv,
        geometry.kernel_size,
        "delta convolution kernel elements",
    )?;
    Ok(())
}

fn activation_tensor(tokens: usize, width: usize) -> Result<TensorSpec, String> {
    TensorSpec::new(
        vec![tokens, width],
        NumericalFormat::F32,
        TensorLayout::TokensHidden,
    )
}

fn transactional_state() -> StateTransactionContract {
    StateTransactionContract::Transactional {
        initialization: StateInitialization::Zeroed,
        execution: StateExecutionProtocol::PrepareExecuteCommit,
        reset: StateResetProtocol::Required,
        snapshot_restore: SnapshotRestorePolicy::Required,
    }
}

fn checked_add(left: usize, right: usize, label: &'static str) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("{label} overflows usize"))
}

fn checked_mul(left: usize, right: usize, label: &'static str) -> Result<usize, String> {
    let product = left
        .checked_mul(right)
        .ok_or_else(|| format!("{label} overflows usize"))?;
    if (product as u128) > u128::from(crate::model_graph::MAX_TENSOR_LOGICAL_ELEMENTS) {
        return Err(format!("{label} exceeds logical element limit"));
    }
    Ok(product)
}

fn try_vec<T>(capacity: usize, label: &'static str) -> Result<Vec<T>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| format!("{label} allocation failed"))?;
    Ok(values)
}

fn node_id(value: &str) -> Result<NodeId, String> {
    NodeId::new(value)
}

fn value_id(value: &str) -> Result<ValueId, String> {
    ValueId::new(value)
}

fn weight_id(value: &str) -> Result<WeightId, String> {
    WeightId::new(value)
}

fn state_id(value: &str) -> Result<StateId, String> {
    StateId::new(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter_admission::CanonicalTransform;

    fn hybrid_dense_geometry() -> DenseGqaGeometry {
        DenseGqaGeometry {
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            value_dim: 128,
            rotary_dim: 128,
            rotary_base: 1_000_000.0,
            qk_norm_epsilon: PositiveF32::new(1.0e-5, "hybrid q/k epsilon").unwrap(),
        }
    }

    fn layer_nodes(fixture: &AdapterContractFixture, layer: usize) -> Vec<&GraphNode> {
        let prefix = format!("l{layer:02}-");
        fixture
            .graph
            .nodes
            .iter()
            .filter(|node| node.id.as_str().starts_with(&prefix))
            .collect()
    }

    fn assert_norm_epsilon(node: &GraphNode, expected: f32) {
        let GraphNodeKind::Norm { epsilon, .. } = &node.kind else {
            panic!("expected normalization node")
        };
        assert_eq!(epsilon.get().to_bits(), expected.to_bits());
    }

    fn assert_dense_layer_contract(
        fixture: &AdapterContractFixture,
        layer: usize,
        qk_epsilon: f32,
    ) {
        let nodes = layer_nodes(fixture, layer);
        let expected = [
            "pre-norm", "q-proj", "k-proj", "v-proj", "q-norm", "k-norm", "q-rope", "k-rope",
            "gqa-core", "o-proj", "residual",
        ];
        assert_eq!(nodes.len(), expected.len());
        for (node, role) in nodes.iter().zip(expected) {
            assert_eq!(node.id.as_str(), format!("l{layer:02}-{role}"));
        }
        assert_norm_epsilon(nodes[0], FIXTURE_RMS_EPSILON);
        assert_eq!(
            nodes[0].inputs,
            vec![value_id(&format!("hidden-{layer}")).unwrap()]
        );
        let GraphNodeKind::CausalGqaAttentionCore {
            q_heads, kv_heads, ..
        } = &nodes[8].kind
        else {
            panic!("expected GQA core")
        };
        for (node, heads) in [(nodes[4], *q_heads), (nodes[5], *kv_heads)] {
            let GraphNodeKind::Norm {
                kind: NormalizationKind::Rms,
                affine: NormalizationAffine::Scale,
                axis:
                    NormalizationAxis::GroupedLast {
                        groups,
                        group_width: 128,
                    },
                ..
            } = &node.kind
            else {
                panic!("expected headwise RMS scale")
            };
            assert_eq!(*groups, heads);
            assert_norm_epsilon(node, qk_epsilon);
        }
        assert!(matches!(
            nodes[6].kind,
            GraphNodeKind::RotaryPosition { .. }
        ));
        assert!(matches!(
            nodes[7].kind,
            GraphNodeKind::RotaryPosition { .. }
        ));
        assert!(matches!(
            nodes[8].kind,
            GraphNodeKind::CausalGqaAttentionCore { .. }
        ));
        assert_eq!(nodes[1].inputs, nodes[0].outputs);
        assert_eq!(nodes[2].inputs, nodes[0].outputs);
        assert_eq!(nodes[3].inputs, nodes[0].outputs);
        assert_eq!(nodes[4].inputs, nodes[1].outputs);
        assert_eq!(nodes[5].inputs, nodes[2].outputs);
        assert_eq!(nodes[6].inputs[0], nodes[4].outputs[0]);
        assert_eq!(nodes[7].inputs[0], nodes[5].outputs[0]);
        assert_eq!(nodes[6].inputs[1], value_id("positions").unwrap());
        assert_eq!(nodes[7].inputs[1], value_id("positions").unwrap());
        assert_eq!(
            nodes[8].inputs,
            vec![
                nodes[6].outputs[0].clone(),
                nodes[7].outputs[0].clone(),
                nodes[3].outputs[0].clone(),
            ]
        );
        assert_eq!(nodes[8].states.len(), 1);
        let state = fixture
            .states
            .entries
            .iter()
            .find(|state| state.id == nodes[8].states[0])
            .unwrap();
        assert_eq!(state.kind, StateKind::PagedKv);
        assert_eq!(
            state.ownership,
            StateOwnership::RequestLayer { layer_index: layer }
        );
        assert_eq!(nodes[9].inputs, nodes[8].outputs);
        assert_eq!(nodes[10].inputs[0], nodes[0].inputs[0]);
        assert_eq!(nodes[10].inputs[1], nodes[9].outputs[0]);
        assert_eq!(
            nodes[0].weights[0].as_str(),
            format!("l{layer:02}-pre-norm-scale-weight")
        );
        assert_eq!(
            nodes[1].weights[0].as_str(),
            format!("l{layer:02}-q-proj-weight")
        );
        assert_eq!(
            nodes[2].weights[0].as_str(),
            format!("l{layer:02}-k-proj-weight")
        );
        assert_eq!(
            nodes[3].weights[0].as_str(),
            format!("l{layer:02}-v-proj-weight")
        );
        assert_eq!(
            nodes[4].weights[0].as_str(),
            format!("l{layer:02}-q-norm-scale-weight")
        );
        assert_eq!(
            nodes[5].weights[0].as_str(),
            format!("l{layer:02}-k-norm-scale-weight")
        );
        assert_eq!(
            nodes[9].weights[0].as_str(),
            format!("l{layer:02}-o-proj-weight")
        );
    }

    fn assert_delta_layer_contract(fixture: &AdapterContractFixture, layer: usize) {
        let nodes = layer_nodes(fixture, layer);
        let expected = [
            "pre-norm",
            "qkv-proj",
            "causal-conv",
            "qkv-silu",
            "qkv-split",
            "q-l2",
            "k-l2",
            "z-proj",
            "decay-control",
            "update-control",
            "decay-parameters",
            "delta-scan",
            "context-norm",
            "gated-multiply",
            "o-proj",
            "residual",
        ];
        assert_eq!(nodes.len(), expected.len());
        for (node, role) in nodes.iter().zip(expected) {
            assert_eq!(node.id.as_str(), format!("l{layer:02}-{role}"));
        }
        assert_norm_epsilon(nodes[0], FIXTURE_RMS_EPSILON);
        assert_eq!(nodes[1].inputs, nodes[0].outputs);
        assert_eq!(nodes[2].inputs, nodes[1].outputs);
        assert_eq!(nodes[2].outputs, nodes[3].inputs);
        assert!(matches!(
            nodes[3].kind,
            GraphNodeKind::Activation {
                kind: ActivationKind::Silu
            }
        ));
        assert_eq!(nodes[3].outputs, nodes[4].inputs);
        assert_eq!(nodes[5].inputs, vec![nodes[4].outputs[0].clone()]);
        assert_eq!(nodes[6].inputs, vec![nodes[4].outputs[1].clone()]);
        assert_eq!(nodes[7].inputs, nodes[0].outputs);
        assert_eq!(nodes[8].inputs, nodes[0].outputs);
        assert_eq!(nodes[9].inputs, nodes[0].outputs);
        let GraphNodeKind::Norm {
            kind: NormalizationKind::L2,
            affine: NormalizationAffine::FixedScale(scale),
            axis:
                NormalizationAxis::GroupedLast {
                    groups: 16,
                    group_width: 128,
                },
            ..
        } = &nodes[5].kind
        else {
            panic!("expected fixed-scale query L2")
        };
        assert_eq!(
            scale.get().to_bits(),
            (1.0_f32 / 128.0_f32.sqrt()).to_bits()
        );
        assert!(matches!(
            nodes[6].kind,
            GraphNodeKind::Norm {
                kind: NormalizationKind::L2,
                affine: NormalizationAffine::None,
                axis: NormalizationAxis::GroupedLast {
                    groups: 16,
                    group_width: 128,
                },
                ..
            }
        ));
        assert_eq!(
            nodes[10]
                .weights
                .iter()
                .map(|weight| weight.as_str())
                .collect::<Vec<_>>(),
            vec![
                format!("l{layer:02}-log-rate-weight"),
                format!("l{layer:02}-time-bias-weight")
            ]
        );
        assert_eq!(
            nodes[10].inputs,
            vec![nodes[8].outputs[0].clone(), nodes[9].outputs[0].clone()]
        );
        assert_eq!(nodes[11].states.len(), 1);
        let state = fixture
            .states
            .entries
            .iter()
            .find(|state| state.id == nodes[11].states[0])
            .unwrap();
        assert_eq!(
            state.ownership,
            StateOwnership::RequestLayer { layer_index: layer }
        );
        assert_eq!(
            nodes[11].inputs,
            vec![
                nodes[5].outputs[0].clone(),
                nodes[6].outputs[0].clone(),
                nodes[4].outputs[2].clone(),
                nodes[10].outputs[0].clone(),
                nodes[10].outputs[1].clone(),
            ]
        );
        let conv_state = fixture
            .states
            .entries
            .iter()
            .find(|state| state.id == nodes[2].states[0])
            .unwrap();
        assert_eq!(conv_state.kind, StateKind::ConvolutionHistory);
        assert_eq!(
            conv_state.ownership,
            StateOwnership::RequestLayer { layer_index: layer }
        );
        assert_eq!(nodes[11].outputs, nodes[12].inputs);
        assert!(matches!(
            nodes[12].kind,
            GraphNodeKind::Norm {
                kind: NormalizationKind::Rms,
                affine: NormalizationAffine::Scale,
                axis: NormalizationAxis::GroupedLast {
                    groups: 32,
                    group_width: 128,
                },
                ..
            }
        ));
        assert_norm_epsilon(nodes[12], FIXTURE_RMS_EPSILON);
        assert_eq!(nodes[12].outputs[0], nodes[13].inputs[0]);
        assert_eq!(nodes[7].outputs[0], nodes[13].inputs[1]);
        assert!(matches!(
            nodes[13].kind,
            GraphNodeKind::GatedMultiply {
                activation: ActivationKind::Silu
            }
        ));
        assert_eq!(nodes[13].outputs, nodes[14].inputs);
        assert_eq!(nodes[15].inputs[0], nodes[0].inputs[0]);
        assert_eq!(nodes[14].outputs[0], nodes[15].inputs[1]);
    }

    fn assert_identity_weight_recipes(fixture: &AdapterContractFixture) {
        for use_record in &fixture.admission.weight_uses {
            let logical = fixture
                .graph
                .weights
                .iter()
                .find(|weight| weight.id == use_record.logical_id)
                .unwrap();
            assert_eq!(use_record.recipe.source_shape, logical.tensor.shape);
            assert!(use_record.recipe.steps.is_empty());
        }
    }

    fn assert_below_admission_limits(fixture: &AdapterContractFixture) {
        assert!(fixture.graph.values.len() < FIXTURE_DECLARATION_LIMIT);
        assert!(fixture.graph.weights.len() < FIXTURE_DECLARATION_LIMIT);
        assert!(fixture.graph.nodes.len() < FIXTURE_DECLARATION_LIMIT);
        assert!(fixture.bindings.bindings.len() < FIXTURE_DECLARATION_LIMIT);
        assert!(fixture.states.entries.len() < FIXTURE_DECLARATION_LIMIT);
        assert!(fixture.admission.node_layers.len() < FIXTURE_DECLARATION_LIMIT);
        assert!(fixture.admission.weight_uses.len() < FIXTURE_DECLARATION_LIMIT);
        assert_eq!(fixture.graph.weights.len(), fixture.bindings.bindings.len());
        assert!(
            fixture
                .bindings
                .bindings
                .iter()
                .all(|binding| binding.content_sha256.is_none())
        );
        let weight_edges: usize = fixture
            .graph
            .nodes
            .iter()
            .map(|node| node.weights.len())
            .sum();
        let state_edges: usize = fixture
            .graph
            .nodes
            .iter()
            .map(|node| node.states.len())
            .sum();
        assert!(weight_edges < FIXTURE_DECLARATION_LIMIT);
        assert!(state_edges < FIXTURE_DECLARATION_LIMIT);
        assert_eq!(weight_edges, fixture.admission.weight_uses.len());
        assert_eq!(state_edges, fixture.admission.node_layers.len());
    }

    #[test]
    fn qwen3_dense_stack_has_40_exact_stateful_attention_layers() {
        for tokens in [1, 16] {
            let fixture = build_qwen3_dense_attention_fixture(tokens).unwrap();
            assert_eq!(fixture.graph.nodes.len(), 40 * 11);
            assert_eq!(fixture.states.entries.len(), 40);
            assert_eq!(fixture.admission.node_layers.len(), 40);
            assert!(fixture.states.entries.iter().all(|state| matches!(
                state.layout,
                StateLayout::PagedKv {
                    block_size: FIXTURE_KV_BLOCK_SIZE,
                    cache_blocks: FIXTURE_KV_CACHE_BLOCKS,
                    ..
                }
            )));
            assert_eq!(
                fixture.graph.nodes[0].inputs,
                vec![value_id("hidden-0").unwrap()]
            );
            assert!(matches!(
                fixture.graph.nodes[0].kind,
                GraphNodeKind::Norm { .. }
            ));
            let last = fixture.graph.nodes.last().unwrap();
            assert!(matches!(last.kind, GraphNodeKind::Residual));
            assert_eq!(last.outputs, fixture.graph.outputs);
            assert_eq!(fixture.graph.outputs, vec![value_id("hidden-40").unwrap()]);
            for layer in 0..40 {
                assert_dense_layer_contract(&fixture, layer, 1.0e-6);
            }
            assert_identity_weight_recipes(&fixture);
            fixture.validate_structural().unwrap();
            assert_below_admission_limits(&fixture);
        }
    }

    #[test]
    fn qwen35_hybrid_stack_has_exact_three_linear_then_self_pattern() {
        for tokens in [1, 16] {
            let fixture =
                build_qwen35_hybrid_attention_fixture(tokens, hybrid_dense_geometry()).unwrap();
            let dense = fixture
                .graph
                .nodes
                .iter()
                .filter(|node| matches!(node.kind, GraphNodeKind::CausalGqaAttentionCore { .. }))
                .count();
            let scans = fixture
                .graph
                .nodes
                .iter()
                .filter(|node| matches!(node.kind, GraphNodeKind::GatedDeltaRuleScan { .. }))
                .count();
            assert_eq!((scans, dense), (24, 8));
            assert_eq!(fixture.graph.nodes.len(), 24 * 16 + 8 * 11);
            assert_eq!(fixture.states.entries.len(), 24 * 2 + 8);
            assert_eq!(fixture.admission.node_layers.len(), 24 * 2 + 8);
            for layer in 0..32 {
                let prefix = format!("l{layer:02}-");
                let layer_nodes: Vec<_> = fixture
                    .graph
                    .nodes
                    .iter()
                    .filter(|node| node.id.as_str().starts_with(&prefix))
                    .collect();
                assert_eq!(
                    layer_nodes.first().unwrap().inputs,
                    vec![value_id(&format!("hidden-{layer}")).unwrap()]
                );
                assert_eq!(
                    layer_nodes.last().unwrap().outputs,
                    vec![value_id(&format!("hidden-{}", layer + 1)).unwrap()]
                );
                if layer % 4 == 3 {
                    assert_dense_layer_contract(&fixture, layer, 1.0e-5);
                    assert!(
                        layer_nodes
                            .iter()
                            .any(|node| matches!(node.kind, GraphNodeKind::RotaryPosition { .. }))
                    );
                    assert!(layer_nodes.iter().any(|node| matches!(
                        node.kind,
                        GraphNodeKind::CausalGqaAttentionCore { .. }
                    )));
                } else {
                    assert_delta_layer_contract(&fixture, layer);
                    assert!(layer_nodes.iter().any(|node| matches!(
                        node.kind,
                        GraphNodeKind::CausalDepthwiseConv1d { .. }
                    )));
                    assert!(
                        layer_nodes.iter().any(|node| matches!(
                            node.kind,
                            GraphNodeKind::GatedDeltaRuleScan { .. }
                        ))
                    );
                    assert!(layer_nodes.iter().any(|node| matches!(
                        node.kind,
                        GraphNodeKind::GatedDecayParameters { .. }
                    )));
                    assert!(layer_nodes.iter().any(|node| matches!(
                        node.kind,
                        GraphNodeKind::GatedMultiply {
                            activation: ActivationKind::Silu
                        }
                    )));
                    assert_eq!(
                        layer_nodes
                            .iter()
                            .filter(|node| matches!(
                                node.kind,
                                GraphNodeKind::Norm {
                                    kind: NormalizationKind::L2,
                                    ..
                                }
                            ))
                            .count(),
                        2
                    );
                }
            }
            assert_eq!(
                fixture.graph.nodes[0].inputs,
                vec![value_id("hidden-0").unwrap()]
            );
            assert_eq!(fixture.graph.outputs, vec![value_id("hidden-32").unwrap()]);
            assert_eq!(
                fixture.graph.nodes.last().unwrap().outputs,
                fixture.graph.outputs
            );
            assert_identity_weight_recipes(&fixture);
            fixture.validate_structural().unwrap();
            assert_below_admission_limits(&fixture);
        }
    }

    #[test]
    fn invalid_tokens_and_dense_geometry_fail_before_fixture_allocation() {
        assert!(PositiveF32::new(0.0, "q/k epsilon").is_err());
        assert!(PositiveF32::new(f32::NAN, "q/k epsilon").is_err());
        for tokens in [0, 1_025] {
            assert!(build_qwen3_dense_attention_fixture(tokens).is_err());
            assert!(
                build_qwen35_hybrid_attention_fixture(tokens, hybrid_dense_geometry()).is_err()
            );
        }
        for geometry in [
            DenseGqaGeometry {
                q_heads: 0,
                ..hybrid_dense_geometry()
            },
            DenseGqaGeometry {
                q_heads: 32,
                kv_heads: 7,
                ..hybrid_dense_geometry()
            },
            DenseGqaGeometry {
                rotary_dim: 127,
                ..hybrid_dense_geometry()
            },
            DenseGqaGeometry {
                rotary_dim: 130,
                ..hybrid_dense_geometry()
            },
            DenseGqaGeometry {
                rotary_base: 0.0,
                ..hybrid_dense_geometry()
            },
            DenseGqaGeometry {
                q_heads: usize::MAX,
                kv_heads: 1,
                head_dim: 2,
                ..hybrid_dense_geometry()
            },
        ] {
            assert!(build_qwen35_hybrid_attention_fixture(1, geometry).is_err());
        }
    }

    #[test]
    fn mutated_fixture_state_and_weight_claims_fail_closed() {
        let fixture = build_qwen3_dense_attention_fixture(1).unwrap();

        let mut wrong_layer = fixture.clone();
        wrong_layer.states.entries[0].ownership = StateOwnership::RequestLayer { layer_index: 999 };
        assert!(wrong_layer.validate_structural().is_err());

        let mut duplicate_state = fixture.clone();
        let core_indices: Vec<_> = duplicate_state
            .graph
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| matches!(node.kind, GraphNodeKind::CausalGqaAttentionCore { .. }))
            .map(|(index, _)| index)
            .collect();
        let first_state = duplicate_state.graph.nodes[core_indices[0]].states[0].clone();
        duplicate_state.graph.nodes[core_indices[1]].states[0] = first_state;
        assert!(duplicate_state.validate_structural().is_err());

        let mut wrong_slot = fixture.clone();
        wrong_slot.admission.weight_uses[0].weight_slot = usize::MAX;
        assert!(wrong_slot.validate_structural().is_err());

        let mut wrong_recipe = fixture.clone();
        wrong_recipe.admission.weight_uses[0].recipe.steps =
            vec![CanonicalTransform::Reshape { shape: vec![1] }];
        assert!(wrong_recipe.validate_structural().is_err());
    }
}
