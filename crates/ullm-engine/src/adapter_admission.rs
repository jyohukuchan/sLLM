// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Generic structural admission before backend-specific adapter verification.
//!
//! This gate proves only graph/state layer mapping, logical weight occurrence
//! mapping, and shape-transform structure. It does not read source bytes, run a
//! transform, verify content hashes, bind an upload identity, or authorize a
//! production allocation/upload. A later verified-evidence gate must establish
//! those properties from canonical payload evidence.

use std::{cmp::Ordering, fmt};

use crate::{
    model_graph::{
        MAX_TENSOR_LOGICAL_ELEMENTS, MAX_TENSOR_RANK, ModelGraph, NodeId, WeightBindings, WeightId,
        validate_identifier,
    },
    state_schema::{StateOwnership, StateSchema, StateSpec},
};

/// Maximum node-layer or weight-use records accepted by one admission spec.
pub const MAX_ADAPTER_ADMISSION_RECORDS: usize = 4_096;
/// Maximum canonical shape transforms accepted for one logical weight use.
pub const MAX_CANONICAL_TRANSFORM_STEPS: usize = 64;
/// Maximum UTF-8 bytes retained in a structural admission error.
pub const MAX_ADAPTER_ADMISSION_ERROR_MESSAGE_BYTES: usize = 1_024;

/// Low-cardinality structural admission failure class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterAdmissionErrorClass {
    /// A graph, schema, mapping, occurrence, or recipe is inconsistent.
    Invalid,
    /// A declared limit, checked count, or fallible allocation was exceeded.
    Resource,
}

/// Bounded structural admission error without tensor or token payload content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterAdmissionError {
    class: AdapterAdmissionErrorClass,
    message: &'static str,
}

impl AdapterAdmissionError {
    /// Returns the stable failure class.
    pub const fn class(&self) -> AdapterAdmissionErrorClass {
        self.class
    }

    /// Returns the bounded UTF-8 diagnostic.
    pub fn message(&self) -> &str {
        self.message
    }

    fn new(class: AdapterAdmissionErrorClass, message: &'static str) -> Self {
        let message = if message.len() <= MAX_ADAPTER_ADMISSION_ERROR_MESSAGE_BYTES {
            message
        } else {
            "adapter admission diagnostic exceeded its bounded limit"
        };
        Self { class, message }
    }
}

impl fmt::Display for AdapterAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}: {}", self.class, self.message)
    }
}

impl std::error::Error for AdapterAdmissionError {}

/// Binds one graph node to the adapter's explicit decoder-layer index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeLayerBinding {
    /// Exact graph node identity.
    pub node_id: NodeId,
    /// Decoder-layer index shared by every RequestLayer state on the node.
    pub layer_index: usize,
}

/// One canonical source-shape transformation pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalizationRecipe {
    /// Logical source shape before any transform. No source bytes are admitted.
    pub source_shape: Vec<usize>,
    /// Ordered structural shape transforms. An empty list is identity.
    pub steps: Vec<CanonicalTransform>,
}

/// One generic canonical shape transform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalTransform {
    /// Reinterprets the current shape with an equal logical element count.
    Reshape { shape: Vec<usize> },
    /// Permutes every axis exactly once.
    ///
    /// Output axis `i` is input axis `axes[i]`.
    PermuteAxes { axes: Vec<usize> },
    /// Keeps a checked contiguous range on one logical axis.
    Slice {
        axis: usize,
        start: usize,
        length: usize,
    },
}

/// Maps one exact node weight occurrence to its logical weight and shape recipe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalWeightUse {
    /// Exact graph node identity.
    pub node_id: NodeId,
    /// Zero-based slot in `GraphNode::weights`.
    pub weight_slot: usize,
    /// Exact logical weight expected at that occurrence.
    pub logical_id: WeightId,
    /// Source-to-logical structural shape recipe.
    pub recipe: CanonicalizationRecipe,
}

/// Complete structural adapter claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterAdmissionSpec {
    /// Stable structural admission identity.
    pub admission_id: String,
    /// Exact graph identity this claim targets.
    pub graph_id: String,
    /// Exact set of nodes that reference RequestLayer state.
    pub node_layers: Vec<NodeLayerBinding>,
    /// Exact set of every `(node, weight_slot)` occurrence.
    pub weight_uses: Vec<CanonicalWeightUse>,
}

/// Opaque, instance-bound proof that generic structural checks completed.
///
/// The proof borrows the exact graph, bindings, state schema, and claim that
/// were validated. Safe Rust therefore cannot mutate those inputs while the
/// proof is live or replay the proof for another instance.
///
/// This token is not a production allocator/uploader capability. It does not
/// prove source bytes, a package hash, transform execution, canonical values,
/// numerical format conversion, physical layout, or upload identity. Consumers
/// must require a later verified-evidence token before admitting payloads.
pub struct StructurallyAdmittedAdapter<'a> {
    graph: &'a ModelGraph,
    bindings: &'a WeightBindings,
    states: &'a StateSchema,
    spec: &'a AdapterAdmissionSpec,
}

/// Exact read-only weight occurrence resolved from one admitted input set.
pub(crate) struct ResolvedAdmittedWeight<'a> {
    pub(crate) graph: &'a ModelGraph,
    pub(crate) spec: &'a AdapterAdmissionSpec,
    pub(crate) node_id: &'a NodeId,
    pub(crate) use_record: &'a CanonicalWeightUse,
    pub(crate) weight: &'a crate::model_graph::WeightSpec,
    pub(crate) binding: &'a crate::model_graph::WeightBinding,
}

impl fmt::Debug for StructurallyAdmittedAdapter<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StructurallyAdmittedAdapter")
            .field("admission_id", &self.admission_id())
            .field("graph_id", &self.graph_id())
            .finish_non_exhaustive()
    }
}

impl StructurallyAdmittedAdapter<'_> {
    /// Returns the structural admission identity.
    pub fn admission_id(&self) -> &str {
        &self.spec.admission_id
    }

    /// Returns the exact graph identity.
    pub fn graph_id(&self) -> &str {
        &self.spec.graph_id
    }

    /// Returns whether these are the exact validated input instances.
    pub fn matches_inputs(
        &self,
        graph: &ModelGraph,
        bindings: &WeightBindings,
        states: &StateSchema,
        spec: &AdapterAdmissionSpec,
    ) -> bool {
        std::ptr::eq(self.graph, graph)
            && std::ptr::eq(self.bindings, bindings)
            && std::ptr::eq(self.states, states)
            && std::ptr::eq(self.spec, spec)
    }

    /// Resolves one exact, already-admitted logical weight occurrence.
    pub(crate) fn resolve_weight_occurrence(
        &self,
        node_id: &NodeId,
        weight_slot: usize,
    ) -> Option<ResolvedAdmittedWeight<'_>> {
        let node = self.graph.nodes.iter().find(|node| &node.id == node_id)?;
        let logical_id = node.weights.get(weight_slot)?;
        let use_record = self
            .spec
            .weight_uses
            .iter()
            .find(|record| &record.node_id == node_id && record.weight_slot == weight_slot)?;
        if &use_record.logical_id != logical_id {
            return None;
        }
        let weight = self
            .graph
            .weights
            .iter()
            .find(|weight| &weight.id == logical_id)?;
        let binding = self
            .bindings
            .bindings
            .iter()
            .find(|binding| &binding.logical_id == logical_id)?;
        Some(ResolvedAdmittedWeight {
            graph: self.graph,
            spec: self.spec,
            node_id: &node.id,
            use_record,
            weight,
            binding,
        })
    }
}

struct ExpectedWeightUse<'a> {
    logical_id: &'a WeightId,
}

struct RequestStateUse<'a> {
    state: &'a StateSpec,
}

/// Validates generic graph/state/weight structure without admitting payloads.
pub fn validate_structural_adapter_admission<'a>(
    graph: &'a ModelGraph,
    bindings: &'a WeightBindings,
    states: &'a StateSchema,
    spec: &'a AdapterAdmissionSpec,
) -> Result<StructurallyAdmittedAdapter<'a>, AdapterAdmissionError> {
    validate_counts_before_allocation(graph, bindings, states, spec)?;
    // The existing graph/schema validators use ordinary internal collections.
    // This gate bounds every declaration and edge set to at most 4096 first;
    // their internal allocations are not themselves fallible.
    graph
        .validate_with_bindings(bindings)
        .map_err(|_| invalid("graph and logical weight bindings are invalid"))?;
    states
        .validate_against_graph(graph)
        .map_err(|_| invalid("state schema and graph composition are invalid"))?;
    validate_identifier(&spec.admission_id, "adapter admission ID")
        .map_err(|_| invalid("adapter admission ID is invalid"))?;
    validate_identifier(&spec.graph_id, "adapter graph ID")
        .map_err(|_| invalid("adapter graph ID is invalid"))?;
    if spec.graph_id != graph.graph_id {
        return Err(invalid("adapter graph ID does not match graph"));
    }

    validate_node_layers(graph, states, &spec.node_layers)?;
    validate_weight_uses(graph, &spec.weight_uses)?;

    Ok(StructurallyAdmittedAdapter {
        graph,
        bindings,
        states,
        spec,
    })
}

fn validate_counts_before_allocation(
    graph: &ModelGraph,
    bindings: &WeightBindings,
    states: &StateSchema,
    spec: &AdapterAdmissionSpec,
) -> Result<(), AdapterAdmissionError> {
    if graph.inputs.len() > MAX_ADAPTER_ADMISSION_RECORDS
        || graph.outputs.len() > MAX_ADAPTER_ADMISSION_RECORDS
        || graph.values.len() > MAX_ADAPTER_ADMISSION_RECORDS
        || graph.nodes.len() > MAX_ADAPTER_ADMISSION_RECORDS
        || graph.weights.len() > MAX_ADAPTER_ADMISSION_RECORDS
        || bindings.bindings.len() > MAX_ADAPTER_ADMISSION_RECORDS
        || states.entries.len() > MAX_ADAPTER_ADMISSION_RECORDS
    {
        return Err(resource(
            "adapter declaration count exceeds structural limit",
        ));
    }
    if spec.node_layers.len() > MAX_ADAPTER_ADMISSION_RECORDS
        || spec.weight_uses.len() > MAX_ADAPTER_ADMISSION_RECORDS
    {
        return Err(resource("adapter admission record count exceeds limit"));
    }
    let state_edges = graph
        .nodes
        .iter()
        .try_fold(0_usize, |total, node| total.checked_add(node.states.len()));
    let weight_edges = graph
        .nodes
        .iter()
        .try_fold(0_usize, |total, node| total.checked_add(node.weights.len()));
    let Some(state_edges) = state_edges else {
        return Err(resource("graph state edge count overflows usize"));
    };
    let Some(weight_edges) = weight_edges else {
        return Err(resource("graph weight edge count overflows usize"));
    };
    if state_edges > MAX_ADAPTER_ADMISSION_RECORDS || weight_edges > MAX_ADAPTER_ADMISSION_RECORDS {
        return Err(resource(
            "graph adapter edge count exceeds structural limit",
        ));
    }
    Ok(())
}

fn validate_node_layers(
    graph: &ModelGraph,
    states: &StateSchema,
    records: &[NodeLayerBinding],
) -> Result<(), AdapterAdmissionError> {
    let states_by_id = try_sorted_refs(&states.entries, |left, right| left.id.cmp(&right.id))?;
    let nodes_by_id = try_sorted_refs(&graph.nodes, |left, right| left.id.cmp(&right.id))?;
    let mut uses = try_vec_with_capacity(
        graph.nodes.iter().map(|node| node.states.len()).sum(),
        "request state use workspace",
    )?;
    let mut expected_nodes = try_vec_with_capacity(graph.nodes.len(), "node layer workspace")?;

    for node in &graph.nodes {
        let mut layer = None;
        for state_id in &node.states {
            let index = states_by_id
                .binary_search_by(|state| state.id.cmp(state_id))
                .map_err(|_| invalid("graph state is absent from state schema"))?;
            let state = states_by_id[index];
            if let StateOwnership::RequestLayer { layer_index } = state.ownership {
                if let Some(existing) = layer {
                    if existing != layer_index {
                        return Err(invalid("one node references different request layers"));
                    }
                } else {
                    layer = Some(layer_index);
                }
                uses.push(RequestStateUse { state });
            }
        }
        if let Some(layer_index) = layer {
            expected_nodes.push((&node.id, layer_index));
        }
    }

    uses.sort_unstable_by(|left, right| left.state.id.cmp(&right.state.id));
    if uses
        .windows(2)
        .any(|window| window[0].state.id == window[1].state.id)
    {
        return Err(invalid(
            "one RequestLayer state is referenced by multiple nodes",
        ));
    }
    let mut request_states =
        try_vec_with_capacity(states.entries.len(), "request state workspace")?;
    for state in &states_by_id {
        if matches!(state.ownership, StateOwnership::RequestLayer { .. }) {
            request_states.push(*state);
        }
    }
    if uses.len() != request_states.len()
        || request_states.iter().any(|state| {
            uses.binary_search_by(|usage| usage.state.id.cmp(&state.id))
                .is_err()
        })
    {
        return Err(invalid("RequestLayer state occurrence set is incomplete"));
    }

    let actual = try_sorted_refs(records, |left, right| left.node_id.cmp(&right.node_id))?;
    if actual
        .windows(2)
        .any(|window| window[0].node_id == window[1].node_id)
    {
        return Err(invalid("node-layer binding is duplicated"));
    }
    for record in records {
        record
            .node_id
            .validate()
            .map_err(|_| invalid("node-layer binding node ID is invalid"))?;
        if nodes_by_id
            .binary_search_by(|node| node.id.cmp(&record.node_id))
            .is_err()
        {
            return Err(invalid("node-layer binding references unknown node"));
        }
    }
    expected_nodes.sort_unstable_by(|left, right| left.0.cmp(right.0));
    if actual.len() != expected_nodes.len() {
        return Err(invalid("node-layer binding set is not exact"));
    }
    for (node_id, expected_layer) in expected_nodes {
        let index = actual
            .binary_search_by(|record| record.node_id.cmp(node_id))
            .map_err(|_| invalid("node-layer binding is missing"))?;
        if actual[index].layer_index != expected_layer {
            return Err(invalid("node-layer binding has wrong layer index"));
        }
    }
    Ok(())
}

fn validate_weight_uses(
    graph: &ModelGraph,
    records: &[CanonicalWeightUse],
) -> Result<(), AdapterAdmissionError> {
    let weights_by_id = try_sorted_refs(&graph.weights, |left, right| left.id.cmp(&right.id))?;
    let nodes_by_id = try_sorted_refs(&graph.nodes, |left, right| left.id.cmp(&right.id))?;
    let total_edges = graph.nodes.iter().map(|node| node.weights.len()).sum();
    let mut expected = try_vec_with_capacity(total_edges, "weight occurrence workspace")?;
    for node in &graph.nodes {
        for logical_id in &node.weights {
            weights_by_id
                .binary_search_by(|weight| weight.id.cmp(logical_id))
                .map_err(|_| invalid("node references undeclared logical weight"))?;
            expected.push(ExpectedWeightUse { logical_id });
        }
    }
    let mut expected_by_logical = try_vec_with_capacity(expected.len(), "weight use workspace")?;
    expected_by_logical.extend(expected.iter());
    expected_by_logical.sort_unstable_by(|left, right| left.logical_id.cmp(right.logical_id));
    for weight in &graph.weights {
        if expected_by_logical
            .binary_search_by(|usage| usage.logical_id.cmp(&weight.id))
            .is_err()
        {
            return Err(invalid("logical WeightSpec is not referenced by any node"));
        }
    }

    let actual = try_sorted_refs(records, compare_weight_records)?;
    if actual.windows(2).any(|window| {
        window[0].node_id == window[1].node_id && window[0].weight_slot == window[1].weight_slot
    }) {
        return Err(invalid("canonical weight occurrence is duplicated"));
    }
    if actual.len() != expected.len() {
        return Err(invalid("canonical weight occurrence set is not exact"));
    }
    for record in records {
        record
            .node_id
            .validate()
            .map_err(|_| invalid("canonical weight use node ID is invalid"))?;
        record
            .logical_id
            .validate()
            .map_err(|_| invalid("canonical weight logical ID is invalid"))?;
        let node_index = nodes_by_id
            .binary_search_by(|node| node.id.cmp(&record.node_id))
            .map_err(|_| invalid("canonical weight use references unknown node"))?;
        let node = nodes_by_id[node_index];
        let Some(expected_id) = node.weights.get(record.weight_slot) else {
            return Err(invalid("canonical weight use slot is out of range"));
        };
        if expected_id != &record.logical_id {
            return Err(invalid(
                "canonical weight use logical ID does not match slot",
            ));
        }
        let weight_index = weights_by_id
            .binary_search_by(|weight| weight.id.cmp(expected_id))
            .map_err(|_| invalid("canonical weight logical ID is undeclared"))?;
        validate_recipe(&record.recipe, &weights_by_id[weight_index].tensor.shape)?;
    }
    let by_logical = try_sorted_refs(records, |left, right| {
        left.logical_id.cmp(&right.logical_id)
    })?;
    if by_logical.windows(2).any(|window| {
        window[0].logical_id == window[1].logical_id && window[0].recipe != window[1].recipe
    }) {
        return Err(invalid(
            "one logical weight uses different canonical recipes",
        ));
    }
    Ok(())
}

fn compare_weight_records(left: &CanonicalWeightUse, right: &CanonicalWeightUse) -> Ordering {
    left.node_id
        .cmp(&right.node_id)
        .then_with(|| left.weight_slot.cmp(&right.weight_slot))
}

fn validate_recipe(
    recipe: &CanonicalizationRecipe,
    logical_shape: &[usize],
) -> Result<(), AdapterAdmissionError> {
    if recipe.steps.len() > MAX_CANONICAL_TRANSFORM_STEPS {
        return Err(resource("canonical transform step count exceeds limit"));
    }
    validate_shape(&recipe.source_shape)?;
    let mut current = try_clone_shape(&recipe.source_shape)?;
    for step in &recipe.steps {
        match step {
            CanonicalTransform::Reshape { shape } => {
                validate_shape(shape)?;
                if checked_elements(shape)? != checked_elements(&current)? {
                    return Err(invalid("reshape changes logical element count"));
                }
                current = try_clone_shape(shape)?;
            }
            CanonicalTransform::PermuteAxes { axes } => {
                if axes.len() != current.len() {
                    return Err(invalid("axis permutation must cover every rank axis"));
                }
                let mut seen = [false; MAX_TENSOR_RANK];
                for axis in axes {
                    if *axis >= current.len() {
                        return Err(invalid("axis permutation index is out of range"));
                    }
                    if seen[*axis] {
                        return Err(invalid("axis permutation contains duplicate axis"));
                    }
                    seen[*axis] = true;
                }
                let mut output = try_vec_with_capacity(current.len(), "permutation shape")?;
                for axis in axes {
                    output.push(current[*axis]);
                }
                current = output;
            }
            CanonicalTransform::Slice {
                axis,
                start,
                length,
            } => {
                if *axis >= current.len() {
                    return Err(invalid("slice axis is out of range"));
                }
                if *length == 0 {
                    return Err(invalid("slice length must be nonzero"));
                }
                let end = start
                    .checked_add(*length)
                    .ok_or_else(|| invalid("slice end overflows usize"))?;
                if end > current[*axis] {
                    return Err(invalid("slice range exceeds source axis"));
                }
                current[*axis] = *length;
            }
        }
    }
    if current != logical_shape {
        return Err(invalid(
            "canonical recipe final shape does not match logical weight",
        ));
    }
    Ok(())
}

fn validate_shape(shape: &[usize]) -> Result<(), AdapterAdmissionError> {
    if shape.is_empty() || shape.len() > MAX_TENSOR_RANK {
        return Err(invalid("canonical shape rank is outside supported range"));
    }
    if shape.iter().any(|dimension| *dimension == 0) {
        return Err(invalid("canonical shape dimensions must be nonzero"));
    }
    checked_elements(shape)?;
    Ok(())
}

fn checked_elements(shape: &[usize]) -> Result<u64, AdapterAdmissionError> {
    let mut elements = 1_u64;
    for dimension in shape {
        let dimension = u64::try_from(*dimension)
            .map_err(|_| resource("canonical shape dimension exceeds u64"))?;
        elements = elements
            .checked_mul(dimension)
            .ok_or_else(|| resource("canonical shape element count overflows u64"))?;
        if elements > MAX_TENSOR_LOGICAL_ELEMENTS {
            return Err(resource("canonical shape exceeds logical element limit"));
        }
    }
    Ok(elements)
}

fn try_sorted_refs<T, F>(values: &[T], compare: F) -> Result<Vec<&T>, AdapterAdmissionError>
where
    F: Fn(&T, &T) -> Ordering,
{
    let mut refs = try_vec_with_capacity(values.len(), "adapter validation references")?;
    refs.extend(values.iter());
    refs.sort_unstable_by(|left, right| compare(left, right));
    Ok(refs)
}

fn try_vec_with_capacity<T>(
    capacity: usize,
    label: &'static str,
) -> Result<Vec<T>, AdapterAdmissionError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| resource(label))?;
    Ok(values)
}

fn try_clone_shape(shape: &[usize]) -> Result<Vec<usize>, AdapterAdmissionError> {
    let mut result = try_vec_with_capacity(shape.len(), "canonical shape workspace")?;
    result.extend_from_slice(shape);
    Ok(result)
}

fn invalid(message: &'static str) -> AdapterAdmissionError {
    AdapterAdmissionError::new(AdapterAdmissionErrorClass::Invalid, message)
}

fn resource(message: &'static str) -> AdapterAdmissionError {
    AdapterAdmissionError::new(AdapterAdmissionErrorClass::Resource, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model_graph::{
            GraphNode, GraphNodeKind, GraphValue, NumericalFormat, PositiveF32, TensorLayout,
            TensorSpec, ValueId, WeightBinding, WeightSpec,
        },
        state_schema::{
            SnapshotRestorePolicy, StateExecutionProtocol, StateInitialization, StateKind,
            StateLayout, StateOwnership, StateResetProtocol, StateSpec, StateTransactionContract,
        },
    };

    fn node_id(value: &str) -> NodeId {
        NodeId::new(value).unwrap()
    }

    fn value_id(value: &str) -> ValueId {
        ValueId::new(value).unwrap()
    }

    fn weight_id(value: &str) -> WeightId {
        WeightId::new(value).unwrap()
    }

    fn state_id(value: &str) -> crate::model_graph::StateId {
        crate::model_graph::StateId::new(value).unwrap()
    }

    fn tensor(shape: &[usize], layout: TensorLayout) -> TensorSpec {
        TensorSpec::new(shape.to_vec(), NumericalFormat::F32, layout).unwrap()
    }

    fn graph_value(name: &str, shape: &[usize]) -> GraphValue {
        GraphValue {
            id: value_id(name),
            tensor: tensor(shape, TensorLayout::TokensHidden),
        }
    }

    fn weight_spec(name: &str, shape: &[usize]) -> WeightSpec {
        WeightSpec {
            id: weight_id(name),
            tensor: tensor(shape, TensorLayout::RowMajor),
        }
    }

    fn transactional() -> StateTransactionContract {
        StateTransactionContract::Transactional {
            initialization: StateInitialization::Zeroed,
            execution: StateExecutionProtocol::PrepareExecuteCommit,
            reset: StateResetProtocol::Required,
            snapshot_restore: SnapshotRestorePolicy::Optional,
        }
    }

    fn fixture() -> (
        ModelGraph,
        WeightBindings,
        StateSchema,
        AdapterAdmissionSpec,
    ) {
        let values = vec![
            graph_value("conv-in", &[1, 2]),
            graph_value("conv-out", &[1, 2]),
            graph_value("scan-q", &[1, 1]),
            graph_value("scan-k", &[1, 1]),
            graph_value("scan-v", &[1, 2]),
            graph_value("scan-decay", &[1, 2]),
            graph_value("scan-update", &[1, 2]),
            graph_value("scan-out", &[1, 2]),
            graph_value("dense-in", &[1, 2]),
            graph_value("dense-out", &[1, 2]),
            graph_value("linear-a-in", &[1, 2]),
            graph_value("linear-a-out", &[1, 2]),
            graph_value("linear-b-in", &[1, 2]),
            graph_value("linear-b-out", &[1, 2]),
        ];
        let weights = vec![
            weight_spec("conv-weight", &[2, 1, 3]),
            weight_spec("dense-q", &[2, 2]),
            weight_spec("dense-k", &[2, 2]),
            weight_spec("dense-v", &[2, 2]),
            weight_spec("dense-o", &[2, 2]),
            weight_spec("shared-linear", &[2, 2]),
        ];
        let nodes = vec![
            GraphNode {
                id: node_id("conv"),
                inputs: vec![value_id("conv-in")],
                outputs: vec![value_id("conv-out")],
                weights: vec![weight_id("conv-weight")],
                states: vec![state_id("conv-state")],
                kind: GraphNodeKind::CausalDepthwiseConv1d {
                    channels: 2,
                    kernel_size: 3,
                },
            },
            GraphNode {
                id: node_id("scan"),
                inputs: vec![
                    value_id("scan-q"),
                    value_id("scan-k"),
                    value_id("scan-v"),
                    value_id("scan-decay"),
                    value_id("scan-update"),
                ],
                outputs: vec![value_id("scan-out")],
                weights: vec![],
                states: vec![state_id("scan-state")],
                kind: GraphNodeKind::GatedDeltaRuleScan {
                    key_heads: 1,
                    value_heads: 2,
                    key_dim: 1,
                    value_dim: 1,
                },
            },
            GraphNode {
                id: node_id("dense"),
                inputs: vec![value_id("dense-in")],
                outputs: vec![value_id("dense-out")],
                weights: vec![
                    weight_id("dense-q"),
                    weight_id("dense-k"),
                    weight_id("dense-v"),
                    weight_id("dense-o"),
                ],
                states: vec![state_id("kv-state")],
                kind: GraphNodeKind::DenseAttention {
                    q_heads: 1,
                    kv_heads: 1,
                    head_dim: 2,
                    value_dim: 2,
                    softmax_scale: PositiveF32::new(1.0, "scale").unwrap(),
                },
            },
            GraphNode {
                id: node_id("linear-a"),
                inputs: vec![value_id("linear-a-in")],
                outputs: vec![value_id("linear-a-out")],
                weights: vec![weight_id("shared-linear")],
                states: vec![],
                kind: GraphNodeKind::Linear { has_bias: false },
            },
            GraphNode {
                id: node_id("linear-b"),
                inputs: vec![value_id("linear-b-in")],
                outputs: vec![value_id("linear-b-out")],
                weights: vec![weight_id("shared-linear")],
                states: vec![],
                kind: GraphNodeKind::Linear { has_bias: false },
            },
        ];
        let graph = ModelGraph {
            graph_id: "adapter-graph".into(),
            inputs: [
                "conv-in",
                "scan-q",
                "scan-k",
                "scan-v",
                "scan-decay",
                "scan-update",
                "dense-in",
                "linear-a-in",
                "linear-b-in",
            ]
            .into_iter()
            .map(value_id)
            .collect(),
            outputs: [
                "conv-out",
                "scan-out",
                "dense-out",
                "linear-a-out",
                "linear-b-out",
            ]
            .into_iter()
            .map(value_id)
            .collect(),
            values,
            weights,
            nodes,
        };
        let bindings = WeightBindings {
            bindings: graph
                .weights
                .iter()
                .map(|weight| WeightBinding {
                    logical_id: weight.id.clone(),
                    physical_tensor_name: format!("package.{}", weight.id.as_str()),
                    tensor: weight.tensor.clone(),
                    content_sha256: None,
                })
                .collect(),
        };
        let states = StateSchema::new(
            "adapter-states",
            vec![
                StateSpec {
                    id: state_id("conv-state"),
                    kind: StateKind::ConvolutionHistory,
                    ownership: StateOwnership::RequestLayer { layer_index: 0 },
                    format: NumericalFormat::F32,
                    layout: StateLayout::ConvolutionHistory {
                        channels: 2,
                        history_tokens: 2,
                    },
                    transaction: transactional(),
                },
                StateSpec {
                    id: state_id("scan-state"),
                    kind: StateKind::Recurrent,
                    ownership: StateOwnership::RequestLayer { layer_index: 0 },
                    format: NumericalFormat::F32,
                    layout: StateLayout::RecurrentBank {
                        instances: 2,
                        rows: 1,
                        cols: 1,
                    },
                    transaction: transactional(),
                },
                StateSpec {
                    id: state_id("kv-state"),
                    kind: StateKind::PagedKv,
                    ownership: StateOwnership::RequestLayer { layer_index: 1 },
                    format: NumericalFormat::F32,
                    layout: StateLayout::PagedKv {
                        block_size: 1,
                        cache_blocks: 1,
                        q_heads: 1,
                        kv_heads: 1,
                        head_dim: 2,
                        value_dim: 2,
                    },
                    transaction: transactional(),
                },
                StateSpec {
                    id: state_id("position"),
                    kind: StateKind::PositionCacheLength,
                    ownership: StateOwnership::Request,
                    format: NumericalFormat::U64,
                    layout: StateLayout::PositionCacheLength,
                    transaction: transactional(),
                },
            ],
        )
        .unwrap();

        let weight_uses = graph
            .nodes
            .iter()
            .flat_map(|node| {
                node.weights
                    .iter()
                    .enumerate()
                    .map(|(weight_slot, logical_id)| {
                        let shape = graph
                            .weights
                            .iter()
                            .find(|weight| &weight.id == logical_id)
                            .unwrap()
                            .tensor
                            .shape
                            .clone();
                        CanonicalWeightUse {
                            node_id: node.id.clone(),
                            weight_slot,
                            logical_id: logical_id.clone(),
                            recipe: CanonicalizationRecipe {
                                source_shape: shape,
                                steps: vec![],
                            },
                        }
                    })
            })
            .collect();
        let spec = AdapterAdmissionSpec {
            admission_id: "adapter-structural-v1".into(),
            graph_id: graph.graph_id.clone(),
            node_layers: vec![
                NodeLayerBinding {
                    node_id: node_id("conv"),
                    layer_index: 0,
                },
                NodeLayerBinding {
                    node_id: node_id("scan"),
                    layer_index: 0,
                },
                NodeLayerBinding {
                    node_id: node_id("dense"),
                    layer_index: 1,
                },
            ],
            weight_uses,
        };
        (graph, bindings, states, spec)
    }

    fn admit<'a>(
        graph: &'a ModelGraph,
        bindings: &'a WeightBindings,
        states: &'a StateSchema,
        spec: &'a AdapterAdmissionSpec,
    ) -> Result<StructurallyAdmittedAdapter<'a>, AdapterAdmissionError> {
        validate_structural_adapter_admission(graph, bindings, states, spec)
    }

    #[test]
    fn valid_layer_mapping_and_weight_occurrences_are_structurally_admitted() {
        let (graph, bindings, states, spec) = fixture();
        let other_graph = graph.clone();
        let token = admit(&graph, &bindings, &states, &spec).unwrap();
        assert_eq!(token.admission_id(), "adapter-structural-v1");
        assert_eq!(token.graph_id(), "adapter-graph");
        assert!(token.matches_inputs(&graph, &bindings, &states, &spec));
        assert!(!token.matches_inputs(&other_graph, &bindings, &states, &spec));
        assert_eq!(
            spec.weight_uses
                .iter()
                .filter(|usage| usage.logical_id == weight_id("shared-linear"))
                .count(),
            2
        );
    }

    #[test]
    fn repeated_logical_weight_requires_one_identical_recipe() {
        let (graph, bindings, states, spec) = fixture();
        assert!(admit(&graph, &bindings, &states, &spec).is_ok());

        let mut different = spec.clone();
        let shared = different
            .weight_uses
            .iter_mut()
            .filter(|usage| usage.logical_id == weight_id("shared-linear"))
            .nth(1)
            .unwrap();
        shared.recipe = CanonicalizationRecipe {
            source_shape: vec![4],
            steps: vec![CanonicalTransform::Reshape { shape: vec![2, 2] }],
        };
        let error = admit(&graph, &bindings, &states, &different).unwrap_err();
        assert_eq!(error.class(), AdapterAdmissionErrorClass::Invalid);
        assert_eq!(
            error.message(),
            "one logical weight uses different canonical recipes"
        );
    }

    #[test]
    fn node_layer_mapping_rejects_missing_extra_duplicate_wrong_and_unknown() {
        let (graph, bindings, states, spec) = fixture();
        let mut missing = spec.clone();
        missing.node_layers.pop();
        assert_eq!(
            admit(&graph, &bindings, &states, &missing)
                .unwrap_err()
                .class(),
            AdapterAdmissionErrorClass::Invalid
        );
        let mut extra = spec.clone();
        extra.node_layers.push(NodeLayerBinding {
            node_id: node_id("linear-a"),
            layer_index: 0,
        });
        assert!(admit(&graph, &bindings, &states, &extra).is_err());
        let mut duplicate = spec.clone();
        duplicate.node_layers.push(duplicate.node_layers[0].clone());
        assert!(admit(&graph, &bindings, &states, &duplicate).is_err());
        let mut wrong = spec.clone();
        wrong.node_layers[0].layer_index = 9;
        assert!(admit(&graph, &bindings, &states, &wrong).is_err());
        let mut unknown = spec.clone();
        unknown.node_layers[0].node_id = node_id("unknown");
        assert!(admit(&graph, &bindings, &states, &unknown).is_err());
    }

    #[test]
    fn node_layer_mapping_rejects_shared_state_and_request_layer_orphan() {
        let (mut graph, bindings, states, spec) = fixture();
        graph.nodes[3].kind = GraphNodeKind::DenseAttention {
            q_heads: 1,
            kv_heads: 1,
            head_dim: 2,
            value_dim: 2,
            softmax_scale: PositiveF32::new(1.0, "scale").unwrap(),
        };
        graph.nodes[3].weights = vec![
            weight_id("dense-q"),
            weight_id("dense-k"),
            weight_id("dense-v"),
            weight_id("dense-o"),
        ];
        graph.nodes[3].states = vec![state_id("kv-state")];
        assert!(admit(&graph, &bindings, &states, &spec).is_err());

        let (graph, bindings, mut states, spec) = fixture();
        states.entries.push(StateSpec {
            id: state_id("orphan"),
            kind: StateKind::Recurrent,
            ownership: StateOwnership::RequestLayer { layer_index: 2 },
            format: NumericalFormat::F32,
            layout: StateLayout::Recurrent { rows: 1, cols: 1 },
            transaction: transactional(),
        });
        let layer_error = validate_node_layers(&graph, &states, &spec.node_layers).unwrap_err();
        assert_eq!(
            layer_error.message(),
            "RequestLayer state occurrence set is incomplete"
        );
        assert!(admit(&graph, &bindings, &states, &spec).is_err());
    }

    #[test]
    fn weight_occurrences_reject_missing_extra_duplicate_slot_and_logical_swap() {
        let (graph, bindings, states, spec) = fixture();
        let mut missing = spec.clone();
        missing.weight_uses.pop();
        assert!(admit(&graph, &bindings, &states, &missing).is_err());
        let mut duplicate = spec.clone();
        duplicate.weight_uses.push(duplicate.weight_uses[0].clone());
        assert!(admit(&graph, &bindings, &states, &duplicate).is_err());
        let mut extra = spec.clone();
        let mut extra_record = extra.weight_uses[0].clone();
        extra_record.node_id = node_id("scan");
        extra_record.weight_slot = 0;
        extra.weight_uses.push(extra_record);
        assert!(admit(&graph, &bindings, &states, &extra).is_err());
        let mut slot = spec.clone();
        slot.weight_uses[0].weight_slot = usize::MAX;
        assert!(admit(&graph, &bindings, &states, &slot).is_err());
        let mut swap = spec.clone();
        let q = swap
            .weight_uses
            .iter()
            .position(|usage| usage.logical_id == weight_id("dense-q"))
            .unwrap();
        swap.weight_uses[q].logical_id = weight_id("dense-k");
        assert!(admit(&graph, &bindings, &states, &swap).is_err());
    }

    #[test]
    fn unreferenced_weight_spec_is_rejected_even_with_binding() {
        let (mut graph, mut bindings, states, spec) = fixture();
        let unused = weight_spec("unused", &[2, 2]);
        bindings.bindings.push(WeightBinding {
            logical_id: unused.id.clone(),
            physical_tensor_name: "package.unused".into(),
            tensor: unused.tensor.clone(),
            content_sha256: Some("a".repeat(64)),
        });
        graph.weights.push(unused);
        assert!(admit(&graph, &bindings, &states, &spec).is_err());
    }

    #[test]
    fn canonical_recipes_accept_identity_pipeline_and_slice() {
        validate_recipe(
            &CanonicalizationRecipe {
                source_shape: vec![2, 2],
                steps: vec![],
            },
            &[2, 2],
        )
        .unwrap();
        validate_recipe(
            &CanonicalizationRecipe {
                source_shape: vec![2, 3, 4],
                steps: vec![
                    CanonicalTransform::Reshape { shape: vec![6, 4] },
                    CanonicalTransform::PermuteAxes { axes: vec![1, 0] },
                    CanonicalTransform::Reshape { shape: vec![2, 12] },
                ],
            },
            &[2, 12],
        )
        .unwrap();
        validate_recipe(
            &CanonicalizationRecipe {
                source_shape: vec![4, 8],
                steps: vec![CanonicalTransform::Slice {
                    axis: 1,
                    start: 2,
                    length: 4,
                }],
            },
            &[4, 4],
        )
        .unwrap();
    }

    #[test]
    fn canonical_recipes_reject_shape_axis_slice_and_final_errors() {
        for shape in [
            vec![],
            vec![1, 0],
            vec![1; MAX_TENSOR_RANK + 1],
            vec![usize::MAX, 2],
        ] {
            assert!(
                validate_recipe(
                    &CanonicalizationRecipe {
                        source_shape: shape,
                        steps: vec![],
                    },
                    &[1]
                )
                .is_err()
            );
        }
        assert!(
            validate_recipe(
                &CanonicalizationRecipe {
                    source_shape: vec![2, 3],
                    steps: vec![CanonicalTransform::Reshape { shape: vec![5] }],
                },
                &[5]
            )
            .is_err()
        );
        for axes in [vec![0], vec![0, 0], vec![0, 2]] {
            assert!(
                validate_recipe(
                    &CanonicalizationRecipe {
                        source_shape: vec![2, 3],
                        steps: vec![CanonicalTransform::PermuteAxes { axes }],
                    },
                    &[2, 3]
                )
                .is_err()
            );
        }
        for step in [
            CanonicalTransform::Slice {
                axis: 2,
                start: 0,
                length: 1,
            },
            CanonicalTransform::Slice {
                axis: 1,
                start: usize::MAX,
                length: 2,
            },
            CanonicalTransform::Slice {
                axis: 1,
                start: 2,
                length: 2,
            },
            CanonicalTransform::Slice {
                axis: 1,
                start: 0,
                length: 0,
            },
        ] {
            assert!(
                validate_recipe(
                    &CanonicalizationRecipe {
                        source_shape: vec![2, 3],
                        steps: vec![step],
                    },
                    &[2, 1]
                )
                .is_err()
            );
        }
        assert!(
            validate_recipe(
                &CanonicalizationRecipe {
                    source_shape: vec![2, 3],
                    steps: vec![],
                },
                &[3, 2]
            )
            .is_err()
        );
    }

    #[test]
    fn canonical_source_element_limit_is_inclusive_and_checked_before_slice() {
        let maximum = usize::try_from(MAX_TENSOR_LOGICAL_ELEMENTS).unwrap();
        validate_recipe(
            &CanonicalizationRecipe {
                source_shape: vec![maximum],
                steps: vec![],
            },
            &[maximum],
        )
        .unwrap();

        let above = maximum.checked_add(1).unwrap();
        let error = validate_recipe(
            &CanonicalizationRecipe {
                source_shape: vec![above],
                steps: vec![],
            },
            &[above],
        )
        .unwrap_err();
        assert_eq!(error.class(), AdapterAdmissionErrorClass::Resource);

        let sliced = validate_recipe(
            &CanonicalizationRecipe {
                source_shape: vec![above],
                steps: vec![CanonicalTransform::Slice {
                    axis: 0,
                    start: 0,
                    length: 1,
                }],
            },
            &[1],
        )
        .unwrap_err();
        assert_eq!(sliced.class(), AdapterAdmissionErrorClass::Resource);

        // Reshape preserves element count, so no valid recipe can exceed the
        // limit only at an intermediate reshape while its endpoints stay valid.
    }

    #[test]
    fn adapter_prevalidation_bounds_existing_validator_inputs_and_allows_40_layers() {
        let (mut graph, bindings, states, spec) = fixture();
        let template = graph.nodes.last().unwrap().clone();
        while graph.nodes.len() < 40 {
            let mut node = template.clone();
            node.id = node_id(&format!("qwen-layer-{}", graph.nodes.len()));
            graph.nodes.push(node);
        }
        validate_counts_before_allocation(&graph, &bindings, &states, &spec).unwrap();

        let (mut excessive_graph, bindings, states, spec) = fixture();
        excessive_graph.values =
            vec![excessive_graph.values[0].clone(); MAX_ADAPTER_ADMISSION_RECORDS + 1];
        assert_eq!(
            validate_counts_before_allocation(&excessive_graph, &bindings, &states, &spec)
                .unwrap_err()
                .class(),
            AdapterAdmissionErrorClass::Resource
        );

        let (graph, mut excessive_bindings, states, spec) = fixture();
        excessive_bindings.bindings =
            vec![excessive_bindings.bindings[0].clone(); MAX_ADAPTER_ADMISSION_RECORDS + 1];
        assert_eq!(
            validate_counts_before_allocation(&graph, &excessive_bindings, &states, &spec)
                .unwrap_err()
                .class(),
            AdapterAdmissionErrorClass::Resource
        );

        let (mut graph, bindings, states, spec) = fixture();
        graph.weights = vec![graph.weights[0].clone(); MAX_ADAPTER_ADMISSION_RECORDS + 1];
        assert_eq!(
            validate_counts_before_allocation(&graph, &bindings, &states, &spec)
                .unwrap_err()
                .class(),
            AdapterAdmissionErrorClass::Resource
        );

        let (mut graph, bindings, states, spec) = fixture();
        graph.nodes = vec![graph.nodes[0].clone(); MAX_ADAPTER_ADMISSION_RECORDS + 1];
        assert_eq!(
            validate_counts_before_allocation(&graph, &bindings, &states, &spec)
                .unwrap_err()
                .class(),
            AdapterAdmissionErrorClass::Resource
        );

        let (graph, bindings, mut states, spec) = fixture();
        states.entries = vec![states.entries[0].clone(); MAX_ADAPTER_ADMISSION_RECORDS + 1];
        assert_eq!(
            validate_counts_before_allocation(&graph, &bindings, &states, &spec)
                .unwrap_err()
                .class(),
            AdapterAdmissionErrorClass::Resource
        );

        let (mut graph, bindings, states, spec) = fixture();
        graph.nodes[0].weights =
            vec![graph.weights[0].id.clone(); MAX_ADAPTER_ADMISSION_RECORDS + 1];
        assert_eq!(
            validate_counts_before_allocation(&graph, &bindings, &states, &spec)
                .unwrap_err()
                .class(),
            AdapterAdmissionErrorClass::Resource
        );

        let (mut graph, bindings, states, spec) = fixture();
        graph.nodes[0].states =
            vec![states.entries[0].id.clone(); MAX_ADAPTER_ADMISSION_RECORDS + 1];
        assert_eq!(
            validate_counts_before_allocation(&graph, &bindings, &states, &spec)
                .unwrap_err()
                .class(),
            AdapterAdmissionErrorClass::Resource
        );
    }

    #[test]
    fn record_and_transform_limits_are_resource_errors() {
        let (graph, bindings, states, mut spec) = fixture();
        spec.node_layers = vec![
            NodeLayerBinding {
                node_id: node_id("conv"),
                layer_index: 0,
            };
            MAX_ADAPTER_ADMISSION_RECORDS + 1
        ];
        assert_eq!(
            admit(&graph, &bindings, &states, &spec)
                .unwrap_err()
                .class(),
            AdapterAdmissionErrorClass::Resource
        );
        let recipe = CanonicalizationRecipe {
            source_shape: vec![1],
            steps: vec![
                CanonicalTransform::Reshape { shape: vec![1] };
                MAX_CANONICAL_TRANSFORM_STEPS + 1
            ],
        };
        assert_eq!(
            validate_recipe(&recipe, &[1]).unwrap_err().class(),
            AdapterAdmissionErrorClass::Resource
        );
    }
}
