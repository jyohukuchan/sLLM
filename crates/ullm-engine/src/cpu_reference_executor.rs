// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Deterministic single-thread CPU reference execution for a supported ModelGraph subset.
//!
//! This is a correctness oracle, not a serving or performance executor. It owns no
//! state handles, never mutates caller-owned tensors, and contains no model or backend
//! selection. Unsupported semantics fail before execution begins.
//!
//! Host payloads are canonical contiguous last-axis-major arrays: weights and token
//! indices must be [`TensorLayout::RowMajor`], and F32 values may be `RowMajor` or
//! `TokensHidden`. `PackedRagged` and custom layouts carry no offset or stride metadata
//! here, so they are rejected rather than interpreted incorrectly. `ActivationKind::Gelu`
//! is also rejected because `ModelGraph` does not yet distinguish exact GELU from a tanh
//! approximation.
//!
//! TODO(P1-B2): integrate a typed partial-failure trace with
//! `ProductionExecutionTrace`. This P1-B1 API returns no trace on error.

use std::collections::{BTreeMap, BTreeSet};

use crate::model_graph::{
    ActivationKind, GraphNode, GraphNodeKind, ModelGraph, NumericalFormat, TensorLayout,
    TensorSpec, ValueId, WeightId,
};

/// Maximum elements allowed in one CPU reference tensor.
pub const MAX_CPU_REFERENCE_TENSOR_ELEMENTS: usize = 1_048_576;
/// Maximum elements across bound payloads and cumulative output/temporary allocations.
pub const MAX_CPU_REFERENCE_TOTAL_ELEMENTS: usize = 8_388_608;
/// Maximum scalar reference work units in one execution preflight.
///
/// Matrix multiply-accumulate operations and elementwise/copy operations each count as
/// one work unit. This guard keeps the deterministic reference path from accidentally
/// becoming a long-running production kernel.
pub const MAX_CPU_REFERENCE_WORK_UNITS: u64 = 50_000_000;

/// Dense host tensor payloads for the CPU reference path.
///
/// Payload bytes are canonical contiguous logical-element order. Execution accepts F32
/// values only as `RowMajor` or `TokensHidden`, and accepts token/index tensors only as
/// `RowMajor`; constructors preserve other validated graph layouts so that execution can
/// reject them explicitly instead of silently reinterpreting the payload.
#[derive(Debug, Clone, PartialEq)]
pub enum HostTensor {
    /// Dense F32 values.
    F32 {
        /// Logical dimensions.
        shape: Vec<usize>,
        /// Logical layout.
        layout: TensorLayout,
        /// Canonical contiguous values in logical element order.
        data: Vec<f32>,
    },
    /// Unsigned 32-bit token or index values.
    U32 {
        /// Logical dimensions.
        shape: Vec<usize>,
        /// Logical layout.
        layout: TensorLayout,
        /// Canonical contiguous values in logical element order.
        data: Vec<u32>,
    },
    /// Unsigned 64-bit token or index values.
    U64 {
        /// Logical dimensions.
        shape: Vec<usize>,
        /// Logical layout.
        layout: TensorLayout,
        /// Canonical contiguous values in logical element order.
        data: Vec<u64>,
    },
}

impl HostTensor {
    /// Creates a checked F32 tensor without any implicit conversion.
    pub fn f32(shape: Vec<usize>, layout: TensorLayout, data: Vec<f32>) -> Result<Self, String> {
        validate_host_shape_and_len(&shape, data.len(), "F32 host tensor")?;
        layout.validate()?;
        Ok(Self::F32 {
            shape,
            layout,
            data,
        })
    }

    /// Creates a checked U32 tensor for token or index input.
    pub fn u32(shape: Vec<usize>, layout: TensorLayout, data: Vec<u32>) -> Result<Self, String> {
        validate_host_shape_and_len(&shape, data.len(), "U32 host tensor")?;
        layout.validate()?;
        Ok(Self::U32 {
            shape,
            layout,
            data,
        })
    }

    /// Creates a checked U64 tensor for token or index input.
    pub fn u64(shape: Vec<usize>, layout: TensorLayout, data: Vec<u64>) -> Result<Self, String> {
        validate_host_shape_and_len(&shape, data.len(), "U64 host tensor")?;
        layout.validate()?;
        Ok(Self::U64 {
            shape,
            layout,
            data,
        })
    }

    /// Returns this tensor's numerical format.
    pub fn format(&self) -> NumericalFormat {
        match self {
            Self::F32 { .. } => NumericalFormat::F32,
            Self::U32 { .. } => NumericalFormat::U32,
            Self::U64 { .. } => NumericalFormat::U64,
        }
    }

    /// Returns this tensor's shape.
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32 { shape, .. } | Self::U32 { shape, .. } | Self::U64 { shape, .. } => shape,
        }
    }

    /// Returns this tensor's layout.
    pub fn layout(&self) -> &TensorLayout {
        match self {
            Self::F32 { layout, .. } | Self::U32 { layout, .. } | Self::U64 { layout, .. } => {
                layout
            }
        }
    }

    /// Revalidates shape and payload length after construction.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::F32 {
                shape,
                layout,
                data,
            } => {
                validate_host_shape_and_len(shape, data.len(), "F32 host tensor")?;
                layout.validate()
            }
            Self::U32 {
                shape,
                layout,
                data,
            } => {
                validate_host_shape_and_len(shape, data.len(), "U32 host tensor")?;
                layout.validate()
            }
            Self::U64 {
                shape,
                layout,
                data,
            } => {
                validate_host_shape_and_len(shape, data.len(), "U64 host tensor")?;
                layout.validate()
            }
        }
    }

    fn f32_parts(&self) -> Result<(&[usize], &TensorLayout, &[f32]), String> {
        match self {
            Self::F32 {
                shape,
                layout,
                data,
            } => Ok((shape, layout, data)),
            _ => Err("CPU reference operation requires an F32 tensor".into()),
        }
    }

    fn token_parts(&self) -> Result<(&[usize], &TensorLayout, TokenValues<'_>), String> {
        match self {
            Self::U32 {
                shape,
                layout,
                data,
            } => Ok((shape, layout, TokenValues::U32(data))),
            Self::U64 {
                shape,
                layout,
                data,
            } => Ok((shape, layout, TokenValues::U64(data))),
            _ => Err("CPU reference embedding requires U32 or U64 token indices".into()),
        }
    }
}

enum TokenValues<'a> {
    U32(&'a [u32]),
    U64(&'a [u64]),
}

impl TokenValues<'_> {
    fn len(&self) -> usize {
        match self {
            Self::U32(values) => values.len(),
            Self::U64(values) => values.len(),
        }
    }

    fn get_usize(&self, index: usize) -> Result<usize, String> {
        match self {
            Self::U32(values) => Ok(values[index] as usize),
            Self::U64(values) => usize::try_from(values[index])
                .map_err(|_| "U64 token index does not fit usize".to_string()),
        }
    }
}

/// Deterministic trace and graph outputs from one CPU reference execution.
#[derive(Debug, Clone, PartialEq)]
pub struct CpuReferenceExecution {
    /// Node IDs that completed in topological order.
    pub executed_node_ids: Vec<crate::model_graph::NodeId>,
    /// Final graph outputs only, keyed by stable ValueId.
    pub outputs: BTreeMap<ValueId, HostTensor>,
}

/// Stateless deterministic CPU reference executor.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CpuReferenceExecutor;

impl CpuReferenceExecutor {
    /// Executes the supported stateless F32 graph subset.
    ///
    /// Inputs and logical weights must be exact maps: missing and extra entries are
    /// rejected so that a mistaken binding cannot be hidden by unused data.
    pub fn execute(
        &self,
        graph: &ModelGraph,
        inputs: BTreeMap<ValueId, HostTensor>,
        weights: BTreeMap<WeightId, HostTensor>,
    ) -> Result<CpuReferenceExecution, String> {
        graph.validate()?;
        validate_exact_keys(&inputs, &graph.inputs, "graph input")?;
        let mut graph_weight_ids = Vec::new();
        graph_weight_ids
            .try_reserve(graph.weights.len())
            .map_err(|_| "CPU reference logical weight ID allocation failed")?;
        graph_weight_ids.extend(graph.weights.iter().map(|weight| weight.id.clone()));
        validate_exact_keys(&weights, &graph_weight_ids, "logical weight")?;

        let value_specs = graph
            .values
            .iter()
            .map(|value| (value.id.clone(), &value.tensor))
            .collect::<BTreeMap<_, _>>();
        let weight_specs = graph
            .weights
            .iter()
            .map(|weight| (weight.id.clone(), &weight.tensor))
            .collect::<BTreeMap<_, _>>();
        let resource_plan = preflight_graph(graph, &value_specs, &weight_specs)?;

        for input_id in &graph.inputs {
            let tensor = inputs
                .get(input_id)
                .ok_or_else(|| format!("missing graph input {}", input_id.as_str()))?;
            validate_tensor_against(
                tensor,
                value_specs
                    .get(input_id)
                    .ok_or_else(|| format!("missing graph input spec {}", input_id.as_str()))?,
                &format!("graph input {}", input_id.as_str()),
            )?;
            validate_cpu_input_layout(
                value_specs
                    .get(input_id)
                    .ok_or_else(|| format!("missing graph input spec {}", input_id.as_str()))?,
                &format!("graph input {}", input_id.as_str()),
            )?;
        }
        for (weight_id, tensor) in &weights {
            let spec = weight_specs
                .get(weight_id)
                .ok_or_else(|| format!("unknown logical weight {}", weight_id.as_str()))?;
            if spec.format != NumericalFormat::F32 {
                return Err(format!(
                    "CPU reference does not materialize non-F32 logical weight {} with format {}",
                    weight_id.as_str(),
                    spec.format.as_str()
                ));
            }
            validate_tensor_against(
                tensor,
                spec,
                &format!("logical weight {}", weight_id.as_str()),
            )?;
            validate_cpu_weight_layout(spec, &format!("logical weight {}", weight_id.as_str()))?;
        }

        let initial_payload_elements = checked_payload_elements(&inputs, "graph input payload")?
            .checked_add(checked_payload_elements(
                &weights,
                "logical weight payload",
            )?)
            .ok_or_else(|| {
                "CPU reference initial payload element count overflows usize".to_string()
            })?;
        let planned_total_elements = checked_total_element_budget(
            initial_payload_elements,
            resource_plan.execution_elements,
            MAX_CPU_REFERENCE_TOTAL_ELEMENTS,
        )?;

        let mut values = inputs;
        let mut allocated_elements = initial_payload_elements;
        let mut executed_node_ids = Vec::new();
        executed_node_ids
            .try_reserve(graph.nodes.len())
            .map_err(|_| "CPU reference execution trace allocation failed")?;
        for node in &graph.nodes {
            let node_outputs = self.execute_node(
                node,
                &values,
                &weights,
                &value_specs,
                &weight_specs,
                &mut allocated_elements,
            )?;
            for (value_id, tensor) in node_outputs {
                if values.contains_key(&value_id) {
                    return Err(node_error(
                        node,
                        &format!("would overwrite existing value {}", value_id.as_str()),
                    ));
                }
                values.insert(value_id, tensor);
            }
            executed_node_ids.push(node.id.clone());
        }

        if allocated_elements != planned_total_elements {
            return Err("CPU reference allocation plan disagrees with execution accounting".into());
        }

        let mut outputs = BTreeMap::new();
        for output_id in &graph.outputs {
            let output = values.remove(output_id).ok_or_else(|| {
                format!(
                    "CPU reference final output {} is unavailable",
                    output_id.as_str()
                )
            })?;
            outputs.insert(output_id.clone(), output);
        }
        Ok(CpuReferenceExecution {
            executed_node_ids,
            outputs,
        })
    }

    fn execute_node(
        &self,
        node: &GraphNode,
        values: &BTreeMap<ValueId, HostTensor>,
        weights: &BTreeMap<WeightId, HostTensor>,
        value_specs: &BTreeMap<ValueId, &TensorSpec>,
        weight_specs: &BTreeMap<WeightId, &TensorSpec>,
        allocated_elements: &mut usize,
    ) -> Result<Vec<(ValueId, HostTensor)>, String> {
        match &node.kind {
            GraphNodeKind::Embedding {
                vocab_size,
                hidden_size,
            } => {
                let input = node_input(node, values, 0)?;
                let weight = node_weight(node, weights, weight_specs, 0)?;
                let output_spec = node_output_spec(node, value_specs, 0)?;
                let output = embedding_f32(
                    input,
                    weight,
                    *vocab_size,
                    *hidden_size,
                    output_spec,
                    allocated_elements,
                )
                .map_err(|error| node_error(node, &error))?;
                Ok(vec![(node.outputs[0].clone(), output)])
            }
            GraphNodeKind::Linear { has_bias } => {
                let input = node_input(node, values, 0)?;
                let matrix = node_weight(node, weights, weight_specs, 0)?;
                let bias = if *has_bias {
                    Some(node_weight(node, weights, weight_specs, 1)?)
                } else {
                    None
                };
                let output_spec = node_output_spec(node, value_specs, 0)?;
                let output = linear_f32(input, matrix, bias, output_spec, allocated_elements)
                    .map_err(|error| node_error(node, &error))?;
                Ok(vec![(node.outputs[0].clone(), output)])
            }
            GraphNodeKind::FusedLinearGroup { output_count } => {
                let input = node_input(node, values, 0)?;
                let mut outputs = Vec::new();
                outputs.try_reserve(*output_count).map_err(|_| {
                    node_error(node, "fused linear output metadata allocation failed")
                })?;
                for index in 0..*output_count {
                    let matrix = node_weight(node, weights, weight_specs, index)?;
                    let output_spec = node_output_spec(node, value_specs, index)?;
                    let output = linear_f32(input, matrix, None, output_spec, allocated_elements)
                        .map_err(|error| node_error(node, &error))?;
                    outputs.push((node.outputs[index].clone(), output));
                }
                Ok(outputs)
            }
            GraphNodeKind::Activation { kind } => {
                let input = node_input(node, values, 0)?;
                let output_spec = node_output_spec(node, value_specs, 0)?;
                let output = activation_f32(input, kind, output_spec, allocated_elements)
                    .map_err(|error| node_error(node, &error))?;
                Ok(vec![(node.outputs[0].clone(), output)])
            }
            GraphNodeKind::GatedMlp {
                intermediate_size,
                activation,
            } => {
                let input = node_input(node, values, 0)?;
                let gate = node_weight(node, weights, weight_specs, 0)?;
                let up = node_weight(node, weights, weight_specs, 1)?;
                let down = node_weight(node, weights, weight_specs, 2)?;
                let output_spec = node_output_spec(node, value_specs, 0)?;
                if node.weights.len() != 3 {
                    return Err(node_error(
                        node,
                        "CPU reference gated MLP requires exactly gate, up, and down weights",
                    ));
                }
                let output = gated_mlp_f32(
                    input,
                    gate,
                    up,
                    down,
                    *intermediate_size,
                    activation,
                    output_spec,
                    allocated_elements,
                )
                .map_err(|error| node_error(node, &error))?;
                Ok(vec![(node.outputs[0].clone(), output)])
            }
            GraphNodeKind::Residual => {
                let left = node_input(node, values, 0)?;
                let right = node_input(node, values, 1)?;
                let output_spec = node_output_spec(node, value_specs, 0)?;
                let output = residual_f32(left, right, output_spec, allocated_elements)
                    .map_err(|error| node_error(node, &error))?;
                Ok(vec![(node.outputs[0].clone(), output)])
            }
            GraphNodeKind::LmHead { .. } => {
                let input = node_input(node, values, 0)?;
                let matrix = node_weight(node, weights, weight_specs, 0)?;
                let output_spec = node_output_spec(node, value_specs, 0)?;
                let output = linear_f32(input, matrix, None, output_spec, allocated_elements)
                    .map_err(|error| node_error(node, &error))?;
                Ok(vec![(node.outputs[0].clone(), output)])
            }
            GraphNodeKind::Norm { .. } | GraphNodeKind::FinalNorm { .. } => Err(node_error(
                node,
                "unsupported CPU reference node: normalization semantic is not yet specified as RMSNorm or another contract",
            )),
            GraphNodeKind::RotaryPosition { .. }
            | GraphNodeKind::DenseAttention { .. }
            | GraphNodeKind::RecurrentAttention { .. }
            | GraphNodeKind::Sampling { .. } => Err(node_error(
                node,
                &format!(
                    "unsupported CPU reference node kind {}",
                    node_kind_name(&node.kind)
                ),
            )),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct ResourcePlan {
    execution_elements: usize,
    work_units: u64,
}

fn preflight_graph(
    graph: &ModelGraph,
    value_specs: &BTreeMap<ValueId, &TensorSpec>,
    weight_specs: &BTreeMap<WeightId, &TensorSpec>,
) -> Result<ResourcePlan, String> {
    let mut plan = ResourcePlan::default();
    for node in &graph.nodes {
        if !node.states.is_empty() {
            return Err(node_error(
                node,
                "stateful execution is unsupported by the stateless CPU reference",
            ));
        }
        match &node.kind {
            GraphNodeKind::Embedding { .. } => {
                let input = node_input_spec(node, value_specs, 0)?;
                let output = node_output_spec(node, value_specs, 0)?;
                let weight = node_weight_spec(node, weight_specs, 0)?;
                validate_cpu_token_spec(input).map_err(|error| node_error(node, &error))?;
                validate_cpu_embedding_output_spec(output)
                    .map_err(|error| node_error(node, &error))?;
                validate_cpu_weight_layout(weight, "embedding weight")
                    .map_err(|error| node_error(node, &error))?;
                reserve_output_elements(&mut plan, node, output)?;
                let output_elements = spec_elements(output, "embedding output")
                    .map_err(|error| node_error(node, &error))?;
                add_work_units(
                    &mut plan,
                    node,
                    u64::try_from(output_elements)
                        .map_err(|_| node_error(node, "embedding output elements exceed u64"))?,
                )?;
            }
            GraphNodeKind::Linear { has_bias } => {
                let input = node_input_spec(node, value_specs, 0)?;
                let output = node_output_spec(node, value_specs, 0)?;
                validate_cpu_value_spec(input, "linear input")
                    .map_err(|error| node_error(node, &error))?;
                validate_cpu_value_spec(output, "linear output")
                    .map_err(|error| node_error(node, &error))?;
                require_matching_value_layout(
                    node,
                    "linear input",
                    input,
                    "linear output",
                    output,
                )?;
                validate_cpu_weight_layout(
                    node_weight_spec(node, weight_specs, 0)?,
                    "linear weight",
                )
                .map_err(|error| node_error(node, &error))?;
                if *has_bias {
                    validate_cpu_weight_layout(
                        node_weight_spec(node, weight_specs, 1)?,
                        "linear bias",
                    )
                    .map_err(|error| node_error(node, &error))?;
                }
                plan_linear(&mut plan, node, input, output)?;
            }
            GraphNodeKind::FusedLinearGroup { output_count } => {
                let input = node_input_spec(node, value_specs, 0)?;
                validate_cpu_value_spec(input, "fused linear input")
                    .map_err(|error| node_error(node, &error))?;
                for index in 0..*output_count {
                    let output = node_output_spec(node, value_specs, index)?;
                    validate_cpu_value_spec(output, "fused linear output")
                        .map_err(|error| node_error(node, &error))?;
                    require_matching_value_layout(
                        node,
                        "fused linear input",
                        input,
                        "fused linear output",
                        output,
                    )?;
                    validate_cpu_weight_layout(
                        node_weight_spec(node, weight_specs, index)?,
                        "fused linear weight",
                    )
                    .map_err(|error| node_error(node, &error))?;
                    plan_linear(&mut plan, node, input, output)?;
                }
            }
            GraphNodeKind::Activation { kind } => {
                let input = node_input_spec(node, value_specs, 0)?;
                let output = node_output_spec(node, value_specs, 0)?;
                validate_cpu_value_spec(input, "activation input")
                    .map_err(|error| node_error(node, &error))?;
                validate_cpu_value_spec(output, "activation output")
                    .map_err(|error| node_error(node, &error))?;
                require_matching_value_layout(
                    node,
                    "activation input",
                    input,
                    "activation output",
                    output,
                )?;
                validate_cpu_activation(kind).map_err(|error| node_error(node, &error))?;
                reserve_output_elements(&mut plan, node, output)?;
                let output_elements = spec_elements(output, "activation output")
                    .map_err(|error| node_error(node, &error))?;
                add_work_units(
                    &mut plan,
                    node,
                    u64::try_from(output_elements)
                        .map_err(|_| node_error(node, "activation output elements exceed u64"))?,
                )?;
            }
            GraphNodeKind::GatedMlp {
                intermediate_size,
                activation,
            } => {
                let input = node_input_spec(node, value_specs, 0)?;
                let output = node_output_spec(node, value_specs, 0)?;
                validate_cpu_value_spec(input, "gated MLP input")
                    .map_err(|error| node_error(node, &error))?;
                validate_cpu_value_spec(output, "gated MLP output")
                    .map_err(|error| node_error(node, &error))?;
                require_matching_value_layout(
                    node,
                    "gated MLP input",
                    input,
                    "gated MLP output",
                    output,
                )?;
                for index in 0..3 {
                    validate_cpu_weight_layout(
                        node_weight_spec(node, weight_specs, index)?,
                        "gated MLP weight",
                    )
                    .map_err(|error| node_error(node, &error))?;
                }
                validate_cpu_activation(activation).map_err(|error| node_error(node, &error))?;
                plan_gated_mlp(&mut plan, node, input, output, *intermediate_size)?;
            }
            GraphNodeKind::Residual => {
                let left = node_input_spec(node, value_specs, 0)?;
                let right = node_input_spec(node, value_specs, 1)?;
                let output = node_output_spec(node, value_specs, 0)?;
                for (label, spec) in [
                    ("residual left input", left),
                    ("residual right input", right),
                    ("residual output", output),
                ] {
                    validate_cpu_value_spec(spec, label)
                        .map_err(|error| node_error(node, &error))?;
                }
                require_matching_value_layout(
                    node,
                    "residual left input",
                    left,
                    "residual right input",
                    right,
                )?;
                require_matching_value_layout(
                    node,
                    "residual left input",
                    left,
                    "residual output",
                    output,
                )?;
                reserve_output_elements(&mut plan, node, output)?;
                let output_elements = spec_elements(output, "residual output")
                    .map_err(|error| node_error(node, &error))?;
                add_work_units(
                    &mut plan,
                    node,
                    u64::try_from(output_elements)
                        .map_err(|_| node_error(node, "residual output elements exceed u64"))?,
                )?;
            }
            GraphNodeKind::LmHead { .. } => {
                let input = node_input_spec(node, value_specs, 0)?;
                let output = node_output_spec(node, value_specs, 0)?;
                validate_cpu_value_spec(input, "LM head input")
                    .map_err(|error| node_error(node, &error))?;
                validate_cpu_value_spec(output, "LM head output")
                    .map_err(|error| node_error(node, &error))?;
                require_matching_value_layout(
                    node,
                    "LM head input",
                    input,
                    "LM head output",
                    output,
                )?;
                validate_cpu_weight_layout(
                    node_weight_spec(node, weight_specs, 0)?,
                    "LM head weight",
                )
                .map_err(|error| node_error(node, &error))?;
                plan_linear(&mut plan, node, input, output)?;
            }
            GraphNodeKind::Norm { .. } | GraphNodeKind::FinalNorm { .. } => {
                return Err(node_error(
                    node,
                    "unsupported CPU reference node: normalization semantic is not yet specified as RMSNorm or another contract",
                ));
            }
            GraphNodeKind::RotaryPosition { .. }
            | GraphNodeKind::DenseAttention { .. }
            | GraphNodeKind::RecurrentAttention { .. }
            | GraphNodeKind::Sampling { .. } => {
                return Err(node_error(
                    node,
                    &format!(
                        "unsupported CPU reference node kind {}",
                        node_kind_name(&node.kind)
                    ),
                ));
            }
        }
    }
    Ok(plan)
}

fn node_input_spec<'a>(
    node: &GraphNode,
    value_specs: &BTreeMap<ValueId, &'a TensorSpec>,
    index: usize,
) -> Result<&'a TensorSpec, String> {
    let id = node
        .inputs
        .get(index)
        .ok_or_else(|| node_error(node, "missing declared input"))?;
    value_specs
        .get(id)
        .copied()
        .ok_or_else(|| node_error(node, &format!("input {} has no spec", id.as_str())))
}

fn node_weight_spec<'a>(
    node: &GraphNode,
    weight_specs: &BTreeMap<WeightId, &'a TensorSpec>,
    index: usize,
) -> Result<&'a TensorSpec, String> {
    let id = node
        .weights
        .get(index)
        .ok_or_else(|| node_error(node, "missing declared weight"))?;
    weight_specs
        .get(id)
        .copied()
        .ok_or_else(|| node_error(node, &format!("weight {} has no spec", id.as_str())))
}

fn validate_cpu_input_layout(spec: &TensorSpec, label: &str) -> Result<(), String> {
    match spec.format {
        NumericalFormat::F32 => validate_cpu_value_spec(spec, label),
        NumericalFormat::U32 | NumericalFormat::U64 => validate_cpu_token_spec(spec),
        _ => Err(format!(
            "{label} format {} is unsupported",
            spec.format.as_str()
        )),
    }
}

fn validate_cpu_value_spec(spec: &TensorSpec, label: &str) -> Result<(), String> {
    if spec.format != NumericalFormat::F32 {
        return Err(format!(
            "{label} requires F32, got {}",
            spec.format.as_str()
        ));
    }
    match spec.layout {
        TensorLayout::RowMajor | TensorLayout::TokensHidden => Ok(()),
        _ => Err(format!(
            "{label} layout is unsupported without explicit stride or packed offsets"
        )),
    }
}

fn require_matching_value_layout(
    node: &GraphNode,
    left_label: &str,
    left: &TensorSpec,
    right_label: &str,
    right: &TensorSpec,
) -> Result<(), String> {
    if left.layout != right.layout {
        return Err(node_error(
            node,
            &format!("{left_label} and {right_label} must use the same layout"),
        ));
    }
    Ok(())
}

fn validate_cpu_token_spec(spec: &TensorSpec) -> Result<(), String> {
    if !matches!(spec.format, NumericalFormat::U32 | NumericalFormat::U64) {
        return Err(format!(
            "embedding token input requires U32 or U64, got {}",
            spec.format.as_str()
        ));
    }
    if spec.layout != TensorLayout::RowMajor {
        return Err("embedding token input requires RowMajor layout".into());
    }
    Ok(())
}

fn validate_cpu_embedding_output_spec(spec: &TensorSpec) -> Result<(), String> {
    if spec.format != NumericalFormat::F32 || spec.layout != TensorLayout::TokensHidden {
        return Err("embedding output requires F32 TokensHidden layout".into());
    }
    Ok(())
}

fn validate_cpu_weight_layout(spec: &TensorSpec, label: &str) -> Result<(), String> {
    if spec.format != NumericalFormat::F32 {
        return Err(format!(
            "{label} requires F32, got {}",
            spec.format.as_str()
        ));
    }
    if spec.layout != TensorLayout::RowMajor {
        return Err(format!("{label} requires RowMajor layout"));
    }
    Ok(())
}

fn validate_cpu_activation(kind: &ActivationKind) -> Result<(), String> {
    match kind {
        ActivationKind::Silu | ActivationKind::Relu => Ok(()),
        ActivationKind::Gelu => {
            Err("GELU is unsupported until its exact or tanh contract is explicit".into())
        }
        ActivationKind::Custom(name) => Err(format!("unsupported custom activation {name}")),
    }
}

fn spec_elements(spec: &TensorSpec, label: &str) -> Result<usize, String> {
    let elements = spec.element_count()?;
    if elements > MAX_CPU_REFERENCE_TENSOR_ELEMENTS {
        return Err(format!(
            "{label} element count {elements} exceeds CPU reference limit {MAX_CPU_REFERENCE_TENSOR_ELEMENTS}"
        ));
    }
    Ok(elements)
}

fn reserve_output_elements(
    plan: &mut ResourcePlan,
    node: &GraphNode,
    output: &TensorSpec,
) -> Result<(), String> {
    let elements =
        spec_elements(output, "CPU reference output").map_err(|error| node_error(node, &error))?;
    plan.execution_elements = plan
        .execution_elements
        .checked_add(elements)
        .ok_or_else(|| node_error(node, "planned element count overflows usize"))?;
    if plan.execution_elements > MAX_CPU_REFERENCE_TOTAL_ELEMENTS {
        return Err(node_error(
            node,
            "cumulative output and temporary element budget is exceeded",
        ));
    }
    Ok(())
}

fn reserve_temporary_elements(
    plan: &mut ResourcePlan,
    node: &GraphNode,
    elements: usize,
) -> Result<(), String> {
    plan.execution_elements = plan
        .execution_elements
        .checked_add(elements)
        .ok_or_else(|| node_error(node, "planned temporary element count overflows usize"))?;
    if plan.execution_elements > MAX_CPU_REFERENCE_TOTAL_ELEMENTS {
        return Err(node_error(
            node,
            "cumulative output and temporary element budget is exceeded",
        ));
    }
    Ok(())
}

fn add_work_units(plan: &mut ResourcePlan, node: &GraphNode, units: u64) -> Result<(), String> {
    plan.work_units = plan
        .work_units
        .checked_add(units)
        .ok_or_else(|| node_error(node, "work-unit count overflows u64"))?;
    if plan.work_units > MAX_CPU_REFERENCE_WORK_UNITS {
        return Err(node_error(
            node,
            &format!(
                "work-unit budget {MAX_CPU_REFERENCE_WORK_UNITS} is exceeded by {}",
                plan.work_units
            ),
        ));
    }
    Ok(())
}

fn plan_linear(
    plan: &mut ResourcePlan,
    node: &GraphNode,
    input: &TensorSpec,
    output: &TensorSpec,
) -> Result<(), String> {
    reserve_output_elements(plan, node, output)?;
    let input_features = output_feature_width(input).map_err(|error| node_error(node, &error))?;
    let output_features = output_feature_width(output).map_err(|error| node_error(node, &error))?;
    let rows = spec_elements(input, "linear input")
        .map_err(|error| node_error(node, &error))?
        .checked_div(input_features)
        .ok_or_else(|| node_error(node, "linear input feature width is zero"))?;
    let macs = u64::try_from(rows)
        .map_err(|_| node_error(node, "linear row count exceeds u64"))?
        .checked_mul(
            u64::try_from(input_features)
                .map_err(|_| node_error(node, "linear input width exceeds u64"))?,
        )
        .and_then(|value| value.checked_mul(u64::try_from(output_features).ok()?))
        .ok_or_else(|| node_error(node, "linear MAC count overflows u64"))?;
    add_work_units(plan, node, macs)?;
    add_work_units(
        plan,
        node,
        u64::try_from(
            spec_elements(output, "linear output").map_err(|error| node_error(node, &error))?,
        )
        .map_err(|_| node_error(node, "linear output element count exceeds u64"))?,
    )
}

fn plan_gated_mlp(
    plan: &mut ResourcePlan,
    node: &GraphNode,
    input: &TensorSpec,
    output: &TensorSpec,
    intermediate_size: usize,
) -> Result<(), String> {
    let input_features = output_feature_width(input).map_err(|error| node_error(node, &error))?;
    let output_features = output_feature_width(output).map_err(|error| node_error(node, &error))?;
    let rows = spec_elements(input, "gated MLP input")
        .map_err(|error| node_error(node, &error))?
        .checked_div(input_features)
        .ok_or_else(|| node_error(node, "gated MLP input feature width is zero"))?;
    let intermediate_elements = rows
        .checked_mul(intermediate_size)
        .ok_or_else(|| node_error(node, "gated MLP intermediate element count overflows usize"))?;
    reserve_temporary_elements(plan, node, intermediate_elements)?;
    reserve_temporary_elements(plan, node, intermediate_elements)?;
    reserve_temporary_elements(plan, node, intermediate_elements)?;
    reserve_output_elements(plan, node, output)?;
    let rows =
        u64::try_from(rows).map_err(|_| node_error(node, "gated MLP row count exceeds u64"))?;
    let input_features = u64::try_from(input_features)
        .map_err(|_| node_error(node, "gated MLP input width exceeds u64"))?;
    let intermediate_size = u64::try_from(intermediate_size)
        .map_err(|_| node_error(node, "gated MLP intermediate width exceeds u64"))?;
    let output_features = u64::try_from(output_features)
        .map_err(|_| node_error(node, "gated MLP output width exceeds u64"))?;
    let gate_or_up_macs = rows
        .checked_mul(input_features)
        .and_then(|value| value.checked_mul(intermediate_size))
        .ok_or_else(|| node_error(node, "gated MLP MAC count overflows u64"))?;
    let down_macs = rows
        .checked_mul(intermediate_size)
        .and_then(|value| value.checked_mul(output_features))
        .ok_or_else(|| node_error(node, "gated MLP MAC count overflows u64"))?;
    add_work_units(plan, node, gate_or_up_macs)?;
    add_work_units(plan, node, gate_or_up_macs)?;
    add_work_units(plan, node, down_macs)?;
    add_work_units(
        plan,
        node,
        u64::try_from(intermediate_elements)
            .map_err(|_| node_error(node, "gated MLP intermediate elements exceed u64"))?
            .checked_mul(2)
            .ok_or_else(|| node_error(node, "gated MLP elementwise work overflows u64"))?,
    )?;
    add_work_units(
        plan,
        node,
        u64::try_from(
            spec_elements(output, "gated MLP output").map_err(|error| node_error(node, &error))?,
        )
        .map_err(|_| node_error(node, "gated MLP output elements exceed u64"))?,
    )
}

fn node_input<'a>(
    node: &GraphNode,
    values: &'a BTreeMap<ValueId, HostTensor>,
    index: usize,
) -> Result<&'a HostTensor, String> {
    let id = node
        .inputs
        .get(index)
        .ok_or_else(|| node_error(node, "missing declared input"))?;
    values
        .get(id)
        .ok_or_else(|| node_error(node, &format!("input {} is unavailable", id.as_str())))
}

fn node_weight<'a>(
    node: &GraphNode,
    weights: &'a BTreeMap<WeightId, HostTensor>,
    weight_specs: &BTreeMap<WeightId, &TensorSpec>,
    index: usize,
) -> Result<&'a HostTensor, String> {
    let id = node
        .weights
        .get(index)
        .ok_or_else(|| node_error(node, "missing declared weight"))?;
    let spec = weight_specs
        .get(id)
        .ok_or_else(|| node_error(node, &format!("unknown weight {}", id.as_str())))?;
    if spec.format != NumericalFormat::F32 {
        return Err(node_error(
            node,
            &format!(
                "unsupported quantized weight {} with format {}",
                id.as_str(),
                spec.format.as_str()
            ),
        ));
    }
    weights
        .get(id)
        .ok_or_else(|| node_error(node, &format!("weight {} is unavailable", id.as_str())))
}

fn node_output_spec<'a>(
    node: &GraphNode,
    value_specs: &BTreeMap<ValueId, &'a TensorSpec>,
    index: usize,
) -> Result<&'a TensorSpec, String> {
    let id = node
        .outputs
        .get(index)
        .ok_or_else(|| node_error(node, "missing declared output"))?;
    value_specs
        .get(id)
        .copied()
        .ok_or_else(|| node_error(node, &format!("output {} has no spec", id.as_str())))
}

fn embedding_f32(
    tokens: &HostTensor,
    embedding: &HostTensor,
    vocab_size: usize,
    hidden_size: usize,
    output_spec: &TensorSpec,
    allocated_elements: &mut usize,
) -> Result<HostTensor, String> {
    let (token_shape, _token_layout, token_values) = tokens.token_parts()?;
    if token_values.len() != checked_elements(token_shape, "embedding token shape")? {
        return Err("embedding token payload length does not match shape".into());
    }
    let (weight_shape, _, weight_data) = embedding.f32_parts()?;
    if weight_shape != [vocab_size, hidden_size] {
        return Err(format!(
            "embedding weight shape {:?} must equal [{vocab_size}, {hidden_size}]",
            weight_shape
        ));
    }
    let mut output_shape = token_shape.to_vec();
    output_shape.push(hidden_size);
    require_f32_shape(output_spec, &output_shape, "embedding")?;
    let output_elements = checked_elements(&output_shape, "embedding output shape")?;
    let mut output = allocate_f32(output_elements, allocated_elements, "embedding output")?;
    for token_index in 0..token_values.len() {
        let token = token_values.get_usize(token_index)?;
        if token >= vocab_size {
            return Err(format!(
                "embedding token index {token} exceeds vocab size {vocab_size}"
            ));
        }
        let source_start = token
            .checked_mul(hidden_size)
            .ok_or_else(|| "embedding source offset overflows usize".to_string())?;
        let destination_start = token_index
            .checked_mul(hidden_size)
            .ok_or_else(|| "embedding destination offset overflows usize".to_string())?;
        output[destination_start..destination_start + hidden_size]
            .copy_from_slice(&weight_data[source_start..source_start + hidden_size]);
    }
    HostTensor::f32(output_shape, output_spec.layout.clone(), output)
}

fn linear_f32(
    input: &HostTensor,
    matrix: &HostTensor,
    bias: Option<&HostTensor>,
    output_spec: &TensorSpec,
    allocated_elements: &mut usize,
) -> Result<HostTensor, String> {
    let (input_shape, input_layout, input_data) = input.f32_parts()?;
    let input_features = *input_shape
        .last()
        .ok_or_else(|| "linear input has no feature axis".to_string())?;
    let rows = input_data
        .len()
        .checked_div(input_features)
        .ok_or_else(|| "linear input feature axis is zero".to_string())?;
    let (weight_shape, _, weight_data) = matrix.f32_parts()?;
    if weight_shape.len() != 2 {
        return Err("linear weight must have shape [out, in]".into());
    }
    let output_features = weight_shape[0];
    if weight_shape[1] != input_features {
        return Err(format!(
            "linear weight input width {} does not match input feature width {input_features}",
            weight_shape[1]
        ));
    }
    let mut output_shape = input_shape.to_vec();
    *output_shape
        .last_mut()
        .ok_or_else(|| "linear input has no feature axis".to_string())? = output_features;
    require_f32_output_spec(output_spec, &output_shape, input_layout, "linear")?;
    let bias_data = if let Some(bias) = bias {
        let (shape, _, data) = bias.f32_parts()?;
        if shape != [output_features] {
            return Err(format!(
                "linear bias shape {:?} must equal [{output_features}]",
                shape
            ));
        }
        Some(data)
    } else {
        None
    };
    let output_elements = rows
        .checked_mul(output_features)
        .ok_or_else(|| "linear output element count overflows usize".to_string())?;
    let mut output = allocate_f32(output_elements, allocated_elements, "linear output")?;
    for row in 0..rows {
        let input_start = row
            .checked_mul(input_features)
            .ok_or_else(|| "linear input row offset overflows usize".to_string())?;
        let output_start = row
            .checked_mul(output_features)
            .ok_or_else(|| "linear output row offset overflows usize".to_string())?;
        for output_feature in 0..output_features {
            let weight_start = output_feature
                .checked_mul(input_features)
                .ok_or_else(|| "linear weight row offset overflows usize".to_string())?;
            let mut accumulator = bias_data.map_or(0.0, |values| values[output_feature]);
            for input_feature in 0..input_features {
                accumulator += input_data[input_start + input_feature]
                    * weight_data[weight_start + input_feature];
            }
            output[output_start + output_feature] = accumulator;
        }
    }
    HostTensor::f32(output_shape, output_spec.layout.clone(), output)
}

fn activation_f32(
    input: &HostTensor,
    kind: &ActivationKind,
    output_spec: &TensorSpec,
    allocated_elements: &mut usize,
) -> Result<HostTensor, String> {
    let (shape, layout, input_data) = input.f32_parts()?;
    require_f32_output_spec(output_spec, shape, layout, "activation")?;
    let mut output = allocate_f32(input_data.len(), allocated_elements, "activation output")?;
    for (destination, source) in output.iter_mut().zip(input_data) {
        *destination = activate(*source, kind)?;
    }
    HostTensor::f32(shape.to_vec(), output_spec.layout.clone(), output)
}

fn gated_mlp_f32(
    input: &HostTensor,
    gate: &HostTensor,
    up: &HostTensor,
    down: &HostTensor,
    intermediate_size: usize,
    activation: &ActivationKind,
    output_spec: &TensorSpec,
    allocated_elements: &mut usize,
) -> Result<HostTensor, String> {
    let (input_shape, input_layout, _) = input.f32_parts()?;
    let input_features = *input_shape
        .last()
        .ok_or_else(|| "gated MLP input has no feature axis".to_string())?;
    let intermediate_shape = replaced_last_axis(input_shape, intermediate_size)?;
    let intermediate_spec = TensorSpec::new(
        intermediate_shape,
        NumericalFormat::F32,
        input_layout.clone(),
    )?;
    let gate_values = linear_f32(input, gate, None, &intermediate_spec, allocated_elements)?;
    let up_values = linear_f32(input, up, None, &intermediate_spec, allocated_elements)?;
    let (_, _, gate_data) = gate_values.f32_parts()?;
    let (_, _, up_data) = up_values.f32_parts()?;
    let mut activated = allocate_f32(gate_data.len(), allocated_elements, "gated MLP activation")?;
    for index in 0..activated.len() {
        activated[index] = activate(gate_data[index], activation)? * up_data[index];
    }
    let activated_tensor = HostTensor::f32(
        intermediate_spec.shape.clone(),
        intermediate_spec.layout.clone(),
        activated,
    )?;
    let (down_shape, _, _) = down.f32_parts()?;
    if down_shape.len() != 2
        || down_shape[0] != output_feature_width(output_spec)?
        || down_shape[1] != intermediate_size
    {
        return Err("gated MLP down weight must have shape [out, intermediate]".into());
    }
    let output = linear_f32(
        &activated_tensor,
        down,
        None,
        output_spec,
        allocated_elements,
    )?;
    if input_features == 0 {
        return Err("gated MLP input feature width must be nonzero".into());
    }
    Ok(output)
}

fn residual_f32(
    left: &HostTensor,
    right: &HostTensor,
    output_spec: &TensorSpec,
    allocated_elements: &mut usize,
) -> Result<HostTensor, String> {
    let (left_shape, left_layout, left_data) = left.f32_parts()?;
    let (right_shape, right_layout, right_data) = right.f32_parts()?;
    if left_shape != right_shape || left_layout != right_layout {
        return Err("residual inputs must have identical shape and layout".into());
    }
    require_f32_output_spec(output_spec, left_shape, left_layout, "residual")?;
    let mut output = allocate_f32(left_data.len(), allocated_elements, "residual output")?;
    for index in 0..output.len() {
        output[index] = left_data[index] + right_data[index];
    }
    HostTensor::f32(left_shape.to_vec(), output_spec.layout.clone(), output)
}

fn activate(value: f32, kind: &ActivationKind) -> Result<f32, String> {
    match kind {
        ActivationKind::Silu => Ok(value / (1.0 + (-value).exp())),
        ActivationKind::Gelu => {
            Err("GELU is unsupported until its exact or tanh contract is explicit".into())
        }
        ActivationKind::Relu => Ok(value.max(0.0)),
        ActivationKind::Custom(name) => Err(format!("unsupported custom activation {name}")),
    }
}

fn require_f32_output_spec(
    spec: &TensorSpec,
    shape: &[usize],
    layout: &TensorLayout,
    operation: &str,
) -> Result<(), String> {
    require_f32_shape(spec, shape, operation)?;
    if &spec.layout != layout {
        return Err(format!(
            "CPU reference {operation} output spec shape/layout does not match computed output"
        ));
    }
    Ok(())
}

fn require_f32_shape(spec: &TensorSpec, shape: &[usize], operation: &str) -> Result<(), String> {
    if spec.format != NumericalFormat::F32 {
        return Err(format!(
            "CPU reference {operation} output requires F32, got {}",
            spec.format.as_str()
        ));
    }
    if spec.shape != shape {
        return Err(format!(
            "CPU reference {operation} output spec shape does not match computed output"
        ));
    }
    ensure_reference_element_limit(shape, &format!("{operation} output"))?;
    Ok(())
}

fn output_feature_width(spec: &TensorSpec) -> Result<usize, String> {
    spec.shape
        .last()
        .copied()
        .ok_or_else(|| "output spec has no feature axis".to_string())
}

fn replaced_last_axis(shape: &[usize], value: usize) -> Result<Vec<usize>, String> {
    if value == 0 {
        return Err("replacement feature axis must be nonzero".into());
    }
    let mut output = shape.to_vec();
    let Some(last) = output.last_mut() else {
        return Err("tensor shape has no feature axis".into());
    };
    *last = value;
    Ok(output)
}

fn validate_tensor_against(
    tensor: &HostTensor,
    spec: &TensorSpec,
    label: &str,
) -> Result<(), String> {
    tensor.validate()?;
    if tensor.format() != spec.format
        || tensor.shape() != spec.shape
        || tensor.layout() != &spec.layout
    {
        return Err(format!(
            "{label} host payload does not match declared shape, format, or layout"
        ));
    }
    Ok(())
}

fn validate_exact_keys<K, V>(
    map: &BTreeMap<K, V>,
    expected: &[K],
    label: &str,
) -> Result<(), String>
where
    K: Ord + Clone + std::fmt::Debug,
{
    let expected = expected.iter().cloned().collect::<BTreeSet<_>>();
    let actual = map.keys().cloned().collect::<BTreeSet<_>>();
    if expected != actual {
        let missing = expected.difference(&actual).collect::<Vec<_>>();
        let extra = actual.difference(&expected).collect::<Vec<_>>();
        return Err(format!(
            "{label} map mismatch: missing={missing:?} extra={extra:?}"
        ));
    }
    Ok(())
}

fn validate_host_shape_and_len(
    shape: &[usize],
    data_len: usize,
    label: &str,
) -> Result<(), String> {
    let elements = ensure_reference_element_limit(shape, label)?;
    if elements != data_len {
        return Err(format!(
            "{label} data length {data_len} does not match shape element count {elements}"
        ));
    }
    Ok(())
}

fn checked_payload_elements<K>(
    payloads: &BTreeMap<K, HostTensor>,
    label: &str,
) -> Result<usize, String>
where
    K: Ord,
{
    payloads.iter().try_fold(0_usize, |total, (_, tensor)| {
        let elements = checked_elements(tensor.shape(), label)?;
        total
            .checked_add(elements)
            .ok_or_else(|| format!("{label} element count overflows usize"))
    })
}

fn checked_total_element_budget(
    initial_elements: usize,
    execution_elements: usize,
    limit: usize,
) -> Result<usize, String> {
    let total = initial_elements
        .checked_add(execution_elements)
        .ok_or_else(|| "CPU reference total element count overflows usize".to_string())?;
    if total > limit {
        return Err(format!(
            "CPU reference total element budget {limit} is exceeded by {total}"
        ));
    }
    Ok(total)
}

fn ensure_reference_element_limit(shape: &[usize], label: &str) -> Result<usize, String> {
    let spec = TensorSpec::new(shape.to_vec(), NumericalFormat::F32, TensorLayout::RowMajor)?;
    let elements = spec.element_count()?;
    if elements > MAX_CPU_REFERENCE_TENSOR_ELEMENTS {
        return Err(format!(
            "{label} element count {elements} exceeds CPU reference limit {MAX_CPU_REFERENCE_TENSOR_ELEMENTS}"
        ));
    }
    Ok(elements)
}

fn checked_elements(shape: &[usize], label: &str) -> Result<usize, String> {
    let elements = ensure_reference_element_limit(shape, label)?;
    Ok(elements)
}

fn allocate_f32(
    elements: usize,
    allocated_elements: &mut usize,
    label: &str,
) -> Result<Vec<f32>, String> {
    let next_total = allocated_elements
        .checked_add(elements)
        .ok_or_else(|| "CPU reference allocation count overflows usize".to_string())?;
    if next_total > MAX_CPU_REFERENCE_TOTAL_ELEMENTS {
        return Err(format!(
            "{label} would exceed CPU reference total allocation limit {MAX_CPU_REFERENCE_TOTAL_ELEMENTS}"
        ));
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .map_err(|_| format!("{label} allocation failed"))?;
    output.resize(elements, 0.0);
    *allocated_elements = next_total;
    Ok(output)
}

fn node_error(node: &GraphNode, message: &str) -> String {
    format!(
        "CPU reference node {} ({}) failed: {message}",
        node.id.as_str(),
        node_kind_name(&node.kind)
    )
}

fn node_kind_name(kind: &GraphNodeKind) -> &'static str {
    match kind {
        GraphNodeKind::Embedding { .. } => "Embedding",
        GraphNodeKind::Norm { .. } => "Norm",
        GraphNodeKind::Linear { .. } => "Linear",
        GraphNodeKind::FusedLinearGroup { .. } => "FusedLinearGroup",
        GraphNodeKind::RotaryPosition { .. } => "RotaryPosition",
        GraphNodeKind::DenseAttention { .. } => "DenseAttention",
        GraphNodeKind::RecurrentAttention { .. } => "RecurrentAttention",
        GraphNodeKind::Activation { .. } => "Activation",
        GraphNodeKind::GatedMlp { .. } => "GatedMlp",
        GraphNodeKind::Residual => "Residual",
        GraphNodeKind::FinalNorm { .. } => "FinalNorm",
        GraphNodeKind::LmHead { .. } => "LmHead",
        GraphNodeKind::Sampling { .. } => "Sampling",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_graph::{
        ActivationKind, GraphNode, GraphNodeKind, NodeId, PositiveF32, StateId, WeightSpec,
    };

    fn value_id(name: &str) -> ValueId {
        ValueId::new(name).unwrap()
    }

    fn weight_id(name: &str) -> WeightId {
        WeightId::new(name).unwrap()
    }

    fn node_id(name: &str) -> NodeId {
        NodeId::new(name).unwrap()
    }

    fn spec(shape: &[usize], format: NumericalFormat, layout: TensorLayout) -> TensorSpec {
        TensorSpec::new(shape.to_vec(), format, layout).unwrap()
    }

    fn value(
        name: &str,
        shape: &[usize],
        format: NumericalFormat,
        layout: TensorLayout,
    ) -> crate::model_graph::GraphValue {
        crate::model_graph::GraphValue {
            id: value_id(name),
            tensor: spec(shape, format, layout),
        }
    }

    fn weight(name: &str, shape: &[usize]) -> WeightSpec {
        WeightSpec {
            id: weight_id(name),
            tensor: spec(shape, NumericalFormat::F32, TensorLayout::RowMajor),
        }
    }

    fn executor() -> CpuReferenceExecutor {
        CpuReferenceExecutor
    }

    fn f32(shape: &[usize], layout: TensorLayout, data: &[f32]) -> HostTensor {
        HostTensor::f32(shape.to_vec(), layout, data.to_vec()).unwrap()
    }

    fn u32_tokens(data: &[u32]) -> HostTensor {
        HostTensor::u32(vec![data.len()], TensorLayout::RowMajor, data.to_vec()).unwrap()
    }

    fn map<T, I>(entries: I) -> BTreeMap<T, HostTensor>
    where
        T: Ord,
        I: IntoIterator<Item = (T, HostTensor)>,
    {
        entries.into_iter().collect()
    }

    #[test]
    fn embedding_then_lm_head_executes_exactly() {
        let graph = ModelGraph {
            graph_id: "embedding-linear".into(),
            inputs: vec![value_id("tokens")],
            outputs: vec![value_id("out")],
            values: vec![
                value("tokens", &[2], NumericalFormat::U32, TensorLayout::RowMajor),
                value(
                    "embedded",
                    &[2, 2],
                    NumericalFormat::F32,
                    TensorLayout::TokensHidden,
                ),
                value(
                    "out",
                    &[2, 1],
                    NumericalFormat::F32,
                    TensorLayout::TokensHidden,
                ),
            ],
            weights: vec![weight("embedding", &[3, 2]), weight("linear", &[1, 2])],
            nodes: vec![
                GraphNode {
                    id: node_id("embedding"),
                    inputs: vec![value_id("tokens")],
                    outputs: vec![value_id("embedded")],
                    weights: vec![weight_id("embedding")],
                    states: vec![],
                    kind: GraphNodeKind::Embedding {
                        vocab_size: 3,
                        hidden_size: 2,
                    },
                },
                GraphNode {
                    id: node_id("lm-head"),
                    inputs: vec![value_id("embedded")],
                    outputs: vec![value_id("out")],
                    weights: vec![weight_id("linear")],
                    states: vec![],
                    kind: GraphNodeKind::LmHead { vocab_size: 1 },
                },
            ],
        };
        let run = executor()
            .execute(
                &graph,
                map([(value_id("tokens"), u32_tokens(&[2, 0]))]),
                map([
                    (
                        weight_id("embedding"),
                        f32(
                            &[3, 2],
                            TensorLayout::RowMajor,
                            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        ),
                    ),
                    (
                        weight_id("linear"),
                        f32(&[1, 2], TensorLayout::RowMajor, &[2.0, 1.0]),
                    ),
                ]),
            )
            .unwrap();
        assert_eq!(
            run.executed_node_ids,
            vec![node_id("embedding"), node_id("lm-head")]
        );
        assert_f32(&run.outputs[&value_id("out")], &[16.0, 4.0], 1e-6);
    }

    #[test]
    fn linear_activation_and_residual_executes() {
        let graph = ModelGraph {
            graph_id: "linear-activation-residual".into(),
            inputs: vec![value_id("input"), value_id("skip")],
            outputs: vec![value_id("out")],
            values: vec![
                value(
                    "input",
                    &[1, 2],
                    NumericalFormat::F32,
                    TensorLayout::TokensHidden,
                ),
                value(
                    "skip",
                    &[1, 2],
                    NumericalFormat::F32,
                    TensorLayout::TokensHidden,
                ),
                value(
                    "linear",
                    &[1, 2],
                    NumericalFormat::F32,
                    TensorLayout::TokensHidden,
                ),
                value(
                    "active",
                    &[1, 2],
                    NumericalFormat::F32,
                    TensorLayout::TokensHidden,
                ),
                value(
                    "out",
                    &[1, 2],
                    NumericalFormat::F32,
                    TensorLayout::TokensHidden,
                ),
            ],
            weights: vec![
                weight("linear-weight", &[2, 2]),
                weight("linear-bias", &[2]),
            ],
            nodes: vec![
                GraphNode {
                    id: node_id("linear"),
                    inputs: vec![value_id("input")],
                    outputs: vec![value_id("linear")],
                    weights: vec![weight_id("linear-weight"), weight_id("linear-bias")],
                    states: vec![],
                    kind: GraphNodeKind::Linear { has_bias: true },
                },
                GraphNode {
                    id: node_id("relu"),
                    inputs: vec![value_id("linear")],
                    outputs: vec![value_id("active")],
                    weights: vec![],
                    states: vec![],
                    kind: GraphNodeKind::Activation {
                        kind: ActivationKind::Relu,
                    },
                },
                GraphNode {
                    id: node_id("residual"),
                    inputs: vec![value_id("active"), value_id("skip")],
                    outputs: vec![value_id("out")],
                    weights: vec![],
                    states: vec![],
                    kind: GraphNodeKind::Residual,
                },
            ],
        };
        let run = executor()
            .execute(
                &graph,
                map([
                    (
                        value_id("input"),
                        f32(&[1, 2], TensorLayout::TokensHidden, &[2.0, -3.0]),
                    ),
                    (
                        value_id("skip"),
                        f32(&[1, 2], TensorLayout::TokensHidden, &[1.0, 1.0]),
                    ),
                ]),
                map([
                    (
                        weight_id("linear-weight"),
                        f32(&[2, 2], TensorLayout::RowMajor, &[1.0, 0.0, 0.0, 1.0]),
                    ),
                    (
                        weight_id("linear-bias"),
                        f32(&[2], TensorLayout::RowMajor, &[1.0, 1.0]),
                    ),
                ]),
            )
            .unwrap();
        assert_f32(&run.outputs[&value_id("out")], &[4.0, 1.0], 1e-6);
    }

    #[test]
    fn nonsquare_asymmetric_linear_uses_out_by_in_weight_order() {
        let graph = ModelGraph {
            graph_id: "nonsquare-linear".into(),
            inputs: vec![value_id("input")],
            outputs: vec![value_id("out")],
            values: vec![
                value(
                    "input",
                    &[1, 3],
                    NumericalFormat::F32,
                    TensorLayout::TokensHidden,
                ),
                value(
                    "out",
                    &[1, 2],
                    NumericalFormat::F32,
                    TensorLayout::TokensHidden,
                ),
            ],
            weights: vec![weight("projection", &[2, 3])],
            nodes: vec![GraphNode {
                id: node_id("projection"),
                inputs: vec![value_id("input")],
                outputs: vec![value_id("out")],
                weights: vec![weight_id("projection")],
                states: vec![],
                kind: GraphNodeKind::Linear { has_bias: false },
            }],
        };
        let run = executor()
            .execute(
                &graph,
                map([(
                    value_id("input"),
                    f32(&[1, 3], TensorLayout::TokensHidden, &[2.0, 3.0, 5.0]),
                )]),
                map([(
                    weight_id("projection"),
                    f32(
                        &[2, 3],
                        TensorLayout::RowMajor,
                        &[1.0, 2.0, 4.0, -3.0, 5.0, 7.0],
                    ),
                )]),
            )
            .unwrap();
        assert_f32(&run.outputs[&value_id("out")], &[28.0, 44.0], 1e-6);
    }

    #[test]
    fn fused_linear_and_gated_mlp_execute() {
        let graph = ModelGraph {
            graph_id: "fused-gated".into(),
            inputs: vec![value_id("input")],
            outputs: vec![value_id("out")],
            values: vec![
                value(
                    "input",
                    &[1, 2],
                    NumericalFormat::F32,
                    TensorLayout::TokensHidden,
                ),
                value(
                    "fused-a",
                    &[1, 1],
                    NumericalFormat::F32,
                    TensorLayout::TokensHidden,
                ),
                value(
                    "fused-b",
                    &[1, 1],
                    NumericalFormat::F32,
                    TensorLayout::TokensHidden,
                ),
                value(
                    "out",
                    &[1, 2],
                    NumericalFormat::F32,
                    TensorLayout::TokensHidden,
                ),
            ],
            weights: vec![
                weight("fused-a-weight", &[1, 2]),
                weight("fused-b-weight", &[1, 2]),
                weight("gate", &[2, 2]),
                weight("up", &[2, 2]),
                weight("down", &[2, 2]),
            ],
            nodes: vec![
                GraphNode {
                    id: node_id("fused"),
                    inputs: vec![value_id("input")],
                    outputs: vec![value_id("fused-a"), value_id("fused-b")],
                    weights: vec![weight_id("fused-a-weight"), weight_id("fused-b-weight")],
                    states: vec![],
                    kind: GraphNodeKind::FusedLinearGroup { output_count: 2 },
                },
                GraphNode {
                    id: node_id("gated"),
                    inputs: vec![value_id("input")],
                    outputs: vec![value_id("out")],
                    weights: vec![weight_id("gate"), weight_id("up"), weight_id("down")],
                    states: vec![],
                    kind: GraphNodeKind::GatedMlp {
                        intermediate_size: 2,
                        activation: ActivationKind::Relu,
                    },
                },
            ],
        };
        let run = executor()
            .execute(
                &graph,
                map([(
                    value_id("input"),
                    f32(&[1, 2], TensorLayout::TokensHidden, &[1.0, 2.0]),
                )]),
                map([
                    (
                        weight_id("fused-a-weight"),
                        f32(&[1, 2], TensorLayout::RowMajor, &[1.0, 0.0]),
                    ),
                    (
                        weight_id("fused-b-weight"),
                        f32(&[1, 2], TensorLayout::RowMajor, &[0.0, 1.0]),
                    ),
                    (
                        weight_id("gate"),
                        f32(&[2, 2], TensorLayout::RowMajor, &[1.0, 0.0, 0.0, 1.0]),
                    ),
                    (
                        weight_id("up"),
                        f32(&[2, 2], TensorLayout::RowMajor, &[2.0, 0.0, 0.0, 3.0]),
                    ),
                    (
                        weight_id("down"),
                        f32(&[2, 2], TensorLayout::RowMajor, &[1.0, 0.0, 0.0, 1.0]),
                    ),
                ]),
            )
            .unwrap();
        assert_f32(&run.outputs[&value_id("out")], &[2.0, 12.0], 1e-6);
    }

    #[test]
    fn missing_extra_and_shape_mismatched_bindings_are_rejected() {
        let graph = simple_linear_graph();
        let missing = executor().execute(
            &graph,
            BTreeMap::new(),
            map([(
                weight_id("weight"),
                f32(&[1, 1], TensorLayout::RowMajor, &[1.0]),
            )]),
        );
        assert!(missing.unwrap_err().contains("graph input map mismatch"));

        let extra = executor().execute(
            &graph,
            map([
                (
                    value_id("input"),
                    f32(&[1, 1], TensorLayout::TokensHidden, &[1.0]),
                ),
                (
                    value_id("extra"),
                    f32(&[1, 1], TensorLayout::TokensHidden, &[1.0]),
                ),
            ]),
            map([(
                weight_id("weight"),
                f32(&[1, 1], TensorLayout::RowMajor, &[1.0]),
            )]),
        );
        assert!(extra.unwrap_err().contains("graph input map mismatch"));

        let mismatched = executor().execute(
            &graph,
            map([(
                value_id("input"),
                f32(&[1, 2], TensorLayout::TokensHidden, &[1.0, 2.0]),
            )]),
            map([(
                weight_id("weight"),
                f32(&[1, 1], TensorLayout::RowMajor, &[1.0]),
            )]),
        );
        assert!(mismatched.unwrap_err().contains("does not match"));

        let mut quantized = simple_linear_graph();
        quantized.weights[0].tensor.format = NumericalFormat::Aq4_0;
        let quantized_error = executor()
            .execute(
                &quantized,
                map([(
                    value_id("input"),
                    f32(&[1, 1], TensorLayout::TokensHidden, &[1.0]),
                )]),
                map([(
                    weight_id("weight"),
                    f32(&[1, 1], TensorLayout::RowMajor, &[1.0]),
                )]),
            )
            .unwrap_err();
        assert!(quantized_error.contains("node linear (Linear)"));
        assert!(quantized_error.contains("requires F32"));
    }

    #[test]
    fn silu_runs_but_gelu_and_custom_activation_contracts_are_explicit() {
        assert!((activate(1.0, &ActivationKind::Silu).unwrap() - 0.731_058_6).abs() < 1e-6);
        assert!(
            activate(1.0, &ActivationKind::Gelu)
                .unwrap_err()
                .contains("GELU is unsupported")
        );
        assert!(
            activate(1.0, &ActivationKind::Custom("future".into()))
                .unwrap_err()
                .contains("unsupported custom activation")
        );
    }

    #[test]
    fn packed_or_noncanonical_layouts_are_rejected_before_execution() {
        let mut packed_values = simple_linear_graph();
        for value in &mut packed_values.values {
            value.tensor.layout = TensorLayout::PackedRagged;
        }
        let packed_error = executor()
            .execute(
                &packed_values,
                map([(
                    value_id("input"),
                    f32(&[1, 1], TensorLayout::PackedRagged, &[1.0]),
                )]),
                map([(
                    weight_id("weight"),
                    f32(&[1, 1], TensorLayout::RowMajor, &[1.0]),
                )]),
            )
            .unwrap_err();
        assert!(packed_error.contains("node linear (Linear)"));
        assert!(packed_error.contains("layout is unsupported"));

        let mut non_row_major_weight = simple_linear_graph();
        non_row_major_weight.weights[0].tensor.layout = TensorLayout::TokensHidden;
        let weight_error = executor()
            .execute(
                &non_row_major_weight,
                map([(
                    value_id("input"),
                    f32(&[1, 1], TensorLayout::TokensHidden, &[1.0]),
                )]),
                map([(
                    weight_id("weight"),
                    f32(&[1, 1], TensorLayout::TokensHidden, &[1.0]),
                )]),
            )
            .unwrap_err();
        assert!(weight_error.contains("node linear (Linear)"));
        assert!(weight_error.contains("requires RowMajor layout"));

        assert!(
            validate_cpu_token_spec(&spec(
                &[1],
                NumericalFormat::U32,
                TensorLayout::TokensHidden,
            ))
            .unwrap_err()
            .contains("requires RowMajor layout")
        );
        assert!(
            validate_cpu_embedding_output_spec(&spec(
                &[1, 2],
                NumericalFormat::F32,
                TensorLayout::RowMajor,
            ))
            .unwrap_err()
            .contains("requires F32 TokensHidden layout")
        );
        assert!(
            validate_cpu_value_spec(
                &spec(
                    &[1, 1],
                    NumericalFormat::F32,
                    TensorLayout::custom("strided").unwrap(),
                ),
                "test value",
            )
            .unwrap_err()
            .contains("layout is unsupported")
        );
    }

    #[test]
    fn preflight_rejects_supported_but_mismatched_value_layouts() {
        let mut graph = simple_linear_graph();
        graph.values[0].tensor.layout = TensorLayout::RowMajor;
        let error = executor()
            .execute(
                &graph,
                map([(
                    value_id("input"),
                    f32(&[1, 1], TensorLayout::RowMajor, &[1.0]),
                )]),
                map([(
                    weight_id("weight"),
                    f32(&[1, 1], TensorLayout::RowMajor, &[1.0]),
                )]),
            )
            .unwrap_err();
        assert!(error.contains("node linear (Linear)"));
        assert!(error.contains("linear input and linear output must use the same layout"));
    }

    #[test]
    fn gelu_preflight_rejects_the_graph_before_any_node_execution() {
        let mut graph = simple_linear_graph();
        graph.values.push(value(
            "gelu-out",
            &[1, 1],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
        ));
        graph.outputs = vec![value_id("gelu-out")];
        graph.nodes.push(GraphNode {
            id: node_id("gelu"),
            inputs: vec![value_id("out")],
            outputs: vec![value_id("gelu-out")],
            weights: vec![],
            states: vec![],
            kind: GraphNodeKind::Activation {
                kind: ActivationKind::Gelu,
            },
        });
        let error = executor()
            .execute(
                &graph,
                map([(
                    value_id("input"),
                    f32(&[1, 1], TensorLayout::TokensHidden, &[1.0]),
                )]),
                map([(
                    weight_id("weight"),
                    f32(&[1, 1], TensorLayout::RowMajor, &[1.0]),
                )]),
            )
            .unwrap_err();
        assert!(error.contains("node gelu (Activation)"));
        assert!(error.contains("GELU is unsupported"));
    }

    #[test]
    fn resource_preflight_rejects_aggregate_elements_and_billion_scale_work() {
        assert!(
            checked_total_element_budget(7, 2, 8)
                .unwrap_err()
                .contains("total element budget 8 is exceeded by 9")
        );

        let graph = ModelGraph {
            graph_id: "billion-mac".into(),
            inputs: vec![value_id("input")],
            outputs: vec![value_id("out")],
            values: vec![
                value(
                    "input",
                    &[1, 50_000],
                    NumericalFormat::F32,
                    TensorLayout::TokensHidden,
                ),
                value(
                    "out",
                    &[1, 20_000],
                    NumericalFormat::F32,
                    TensorLayout::TokensHidden,
                ),
            ],
            weights: vec![weight("weight", &[20_000, 50_000])],
            nodes: vec![GraphNode {
                id: node_id("large-linear"),
                inputs: vec![value_id("input")],
                outputs: vec![value_id("out")],
                weights: vec![weight_id("weight")],
                states: vec![],
                kind: GraphNodeKind::Linear { has_bias: false },
            }],
        };
        graph.validate().unwrap();
        let value_specs = graph
            .values
            .iter()
            .map(|value| (value.id.clone(), &value.tensor))
            .collect();
        let weight_specs = graph
            .weights
            .iter()
            .map(|weight| (weight.id.clone(), &weight.tensor))
            .collect();
        let error = preflight_graph(&graph, &value_specs, &weight_specs).unwrap_err();
        assert!(error.contains("node large-linear (Linear)"));
        assert!(error.contains("work-unit budget"));
    }

    #[test]
    fn directly_constructed_host_tensor_is_revalidated_before_execute() {
        let malformed = HostTensor::F32 {
            shape: vec![2],
            layout: TensorLayout::RowMajor,
            data: vec![1.0],
        };
        assert!(
            malformed
                .validate()
                .unwrap_err()
                .contains("data length 1 does not match shape element count 2")
        );

        let error = executor()
            .execute(
                &simple_linear_graph(),
                map([(
                    value_id("input"),
                    HostTensor::F32 {
                        shape: vec![1, 1],
                        layout: TensorLayout::TokensHidden,
                        data: vec![],
                    },
                )]),
                map([(
                    weight_id("weight"),
                    f32(&[1, 1], TensorLayout::RowMajor, &[1.0]),
                )]),
            )
            .unwrap_err();
        assert!(
            error.contains("F32 host tensor data length 0 does not match shape element count 1")
        );
    }

    #[test]
    fn unsupported_stateful_node_reports_node_id_and_kind() {
        let mut graph = simple_linear_graph();
        graph.nodes[0].kind = GraphNodeKind::DenseAttention {
            q_heads: 1,
            kv_heads: 1,
            head_dim: 1,
            value_dim: 1,
            softmax_scale: PositiveF32::new(1.0, "scale").unwrap(),
        };
        graph.nodes[0].weights = vec![
            weight_id("wq"),
            weight_id("wk"),
            weight_id("wv"),
            weight_id("wo"),
        ];
        graph.nodes[0].states = vec![StateId::new("kv").unwrap()];
        graph.weights = vec![
            weight("wq", &[1, 1]),
            weight("wk", &[1, 1]),
            weight("wv", &[1, 1]),
            weight("wo", &[1, 1]),
        ];
        let error = executor()
            .execute(
                &graph,
                map([(
                    value_id("input"),
                    f32(&[1, 1], TensorLayout::TokensHidden, &[1.0]),
                )]),
                map([
                    (
                        weight_id("wq"),
                        f32(&[1, 1], TensorLayout::RowMajor, &[1.0]),
                    ),
                    (
                        weight_id("wk"),
                        f32(&[1, 1], TensorLayout::RowMajor, &[1.0]),
                    ),
                    (
                        weight_id("wv"),
                        f32(&[1, 1], TensorLayout::RowMajor, &[1.0]),
                    ),
                    (
                        weight_id("wo"),
                        f32(&[1, 1], TensorLayout::RowMajor, &[1.0]),
                    ),
                ]),
            )
            .unwrap_err();
        assert!(error.contains("node linear (DenseAttention)"));
        assert!(error.contains("unsupported"));
    }

    #[test]
    fn huge_shape_is_rejected_before_cpu_reference_allocation() {
        let error = HostTensor::f32(
            vec![MAX_CPU_REFERENCE_TENSOR_ELEMENTS + 1],
            TensorLayout::RowMajor,
            Vec::new(),
        )
        .unwrap_err();
        assert!(error.contains("exceeds CPU reference limit"));
    }

    fn simple_linear_graph() -> ModelGraph {
        ModelGraph {
            graph_id: "simple-linear".into(),
            inputs: vec![value_id("input")],
            outputs: vec![value_id("out")],
            values: vec![
                value(
                    "input",
                    &[1, 1],
                    NumericalFormat::F32,
                    TensorLayout::TokensHidden,
                ),
                value(
                    "out",
                    &[1, 1],
                    NumericalFormat::F32,
                    TensorLayout::TokensHidden,
                ),
            ],
            weights: vec![weight("weight", &[1, 1])],
            nodes: vec![GraphNode {
                id: node_id("linear"),
                inputs: vec![value_id("input")],
                outputs: vec![value_id("out")],
                weights: vec![weight_id("weight")],
                states: vec![],
                kind: GraphNodeKind::Linear { has_bias: false },
            }],
        }
    }

    fn assert_f32(tensor: &HostTensor, expected: &[f32], tolerance: f32) {
        let (_, _, actual) = tensor.f32_parts().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (*actual - *expected).abs() <= tolerance,
                "actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }
}
