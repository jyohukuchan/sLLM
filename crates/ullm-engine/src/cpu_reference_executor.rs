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
//! `TokensHidden`. RoPE additionally accepts `PackedRagged` because its explicit positions
//! and token-independent math do not require ragged offsets. Other operations reject
//! `PackedRagged` and custom layouts rather than interpreting missing offset or stride
//! metadata incorrectly. `ActivationKind::Gelu`
//! is also rejected because `ModelGraph` does not yet distinguish exact GELU from a tanh
//! approximation. F32 host payloads must contain only finite values: this executor
//! fail-closes NaN and infinity at execution admission, and checks generated F32 outputs
//! before exposing them.
//!
//! This module records a bounded in-memory CPU trace, but does not generate a
//! `ProductionExecutionTrace` JSON artifact. P2's resolved-registry trace collector owns
//! that conversion because it adds implementation selection, phase, workspace, identity,
//! and fallback evidence that does not belong to this stateless executor.

use std::collections::BTreeMap;
use std::fmt;

use crate::model_graph::{
    ActivationKind, GraphNode, GraphNodeKind, MAX_GRAPH_DECLARATIONS, MAX_GRAPH_ENDPOINTS,
    ModelGraph, NormalizationAffine, NormalizationAxis, NormalizationKind, NumericalFormat,
    RotaryPairing, TensorLayout, TensorSpec, ValueId, WeightId,
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
/// Largest unsigned token position that converts to F32 without rounding.
pub const MAX_EXACT_F32_INTEGER: u64 = 16_777_216;
/// Maximum completed node records retained in a typed CPU reference trace.
pub const MAX_CPU_REFERENCE_TRACE_NODES: usize = 4_096;
/// Maximum UTF-8 bytes retained in a typed CPU reference failure message.
pub const MAX_CPU_REFERENCE_FAILURE_MESSAGE_BYTES: usize = 1_024;

/// Dense host tensor payloads for the CPU reference path.
///
/// Payload bytes are canonical contiguous logical-element order. Execution accepts F32
/// values only as `RowMajor` or `TokensHidden` for ordinary operations, and accepts
/// token/index tensors only as `RowMajor`; the RoPE operation additionally accepts
/// `PackedRagged` values. Constructors preserve other validated graph layouts so that
/// execution can reject them explicitly instead of silently reinterpreting the payload.
/// Shape and layout validation do not imply numerical admission: CPU reference execution
/// rejects every non-finite F32 payload rather than propagating NaN or infinity.
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

    fn get_position_f32(&self, index: usize) -> Result<f32, String> {
        let value = match self {
            Self::U32(values) => u64::from(values[index]),
            Self::U64(values) => values[index],
        };
        if value > MAX_EXACT_F32_INTEGER {
            return Err(format!(
                "rotary position at row {index} exceeds exact F32 integer limit {MAX_EXACT_F32_INTEGER}"
            ));
        }
        Ok(value as f32)
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

/// Stable high-level classification of a failed CPU reference execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuReferenceFailureClass {
    /// Graph, binding, input map, or host payload validation failed.
    InvalidInput,
    /// The graph requests a semantic, format, or layout this executor cannot run.
    Unsupported,
    /// A checked element, work, metadata, or allocation resource limit was reached.
    Resource,
    /// A numerical contract such as finiteness failed.
    Numerical,
    /// An executor invariant failed after valid admission.
    Internal,
}

/// Attribute-free semantic node kind retained in a bounded execution trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuReferenceNodeKind {
    Embedding,
    Norm,
    Linear,
    FusedLinearGroup,
    GroupedLastSplit,
    LastAxisSplit,
    RotaryPosition,
    CausalGqaAttentionCore,
    DenseAttention,
    RecurrentAttention,
    CausalDepthwiseConv1d,
    GatedDecayParameters,
    GatedDeltaRuleScan,
    Activation,
    GatedMultiply,
    GatedMlp,
    Residual,
    FinalNorm,
    LmHead,
    Sampling,
}

impl From<&GraphNodeKind> for CpuReferenceNodeKind {
    fn from(kind: &GraphNodeKind) -> Self {
        match kind {
            GraphNodeKind::Embedding { .. } => Self::Embedding,
            GraphNodeKind::Norm { .. } => Self::Norm,
            GraphNodeKind::Linear { .. } => Self::Linear,
            GraphNodeKind::FusedLinearGroup { .. } => Self::FusedLinearGroup,
            GraphNodeKind::GroupedLastSplit { .. } => Self::GroupedLastSplit,
            GraphNodeKind::LastAxisSplit { .. } => Self::LastAxisSplit,
            GraphNodeKind::RotaryPosition { .. } => Self::RotaryPosition,
            GraphNodeKind::CausalGqaAttentionCore { .. } => Self::CausalGqaAttentionCore,
            GraphNodeKind::DenseAttention { .. } => Self::DenseAttention,
            GraphNodeKind::RecurrentAttention { .. } => Self::RecurrentAttention,
            GraphNodeKind::CausalDepthwiseConv1d { .. } => Self::CausalDepthwiseConv1d,
            GraphNodeKind::GatedDecayParameters { .. } => Self::GatedDecayParameters,
            GraphNodeKind::GatedDeltaRuleScan { .. } => Self::GatedDeltaRuleScan,
            GraphNodeKind::Activation { .. } => Self::Activation,
            GraphNodeKind::GatedMultiply { .. } => Self::GatedMultiply,
            GraphNodeKind::GatedMlp { .. } => Self::GatedMlp,
            GraphNodeKind::Residual => Self::Residual,
            GraphNodeKind::FinalNorm { .. } => Self::FinalNorm,
            GraphNodeKind::LmHead { .. } => Self::LmHead,
            GraphNodeKind::Sampling { .. } => Self::Sampling,
        }
    }
}

impl CpuReferenceNodeKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Embedding => "Embedding",
            Self::Norm => "Norm",
            Self::Linear => "Linear",
            Self::FusedLinearGroup => "FusedLinearGroup",
            Self::GroupedLastSplit => "GroupedLastSplit",
            Self::LastAxisSplit => "LastAxisSplit",
            Self::RotaryPosition => "RotaryPosition",
            Self::CausalGqaAttentionCore => "CausalGqaAttentionCore",
            Self::DenseAttention => "DenseAttention",
            Self::RecurrentAttention => "RecurrentAttention",
            Self::CausalDepthwiseConv1d => "CausalDepthwiseConv1d",
            Self::GatedDecayParameters => "GatedDecayParameters",
            Self::GatedDeltaRuleScan => "GatedDeltaRuleScan",
            Self::Activation => "Activation",
            Self::GatedMultiply => "GatedMultiply",
            Self::GatedMlp => "GatedMlp",
            Self::Residual => "Residual",
            Self::FinalNorm => "FinalNorm",
            Self::LmHead => "LmHead",
            Self::Sampling => "Sampling",
        }
    }
}

/// Stable node identity and semantic kind for a completed or failed node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuReferenceNodeRef {
    pub id: crate::model_graph::NodeId,
    pub kind: CpuReferenceNodeKind,
}

impl CpuReferenceNodeRef {
    fn from_node(node: &GraphNode) -> Self {
        Self {
            id: node.id.clone(),
            kind: CpuReferenceNodeKind::from(&node.kind),
        }
    }
}

/// Bounded ordered completion evidence shared by successful and failed executions.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CpuReferenceExecutionTrace {
    /// Exact count of nodes that completed successfully.
    pub completed_node_count: usize,
    /// Ordered prefix of completed nodes, bounded by [`MAX_CPU_REFERENCE_TRACE_NODES`].
    pub completed_nodes: Vec<CpuReferenceNodeRef>,
    /// Whether node IDs after the retained ordered prefix were omitted.
    pub completed_nodes_truncated: bool,
}

/// Typed terminal failure with bounded diagnostic context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuReferenceExecutionFailure {
    pub class: CpuReferenceFailureClass,
    /// `None` only when no graph node can be identified, such as map validation.
    pub failed_node: Option<CpuReferenceNodeRef>,
    pub trace: CpuReferenceExecutionTrace,
    /// Low-cardinality static diagnostic code for later registry-trace conversion.
    pub reason_code: &'static str,
    /// UTF-8-safe bounded diagnostic text without prompt or token content.
    pub message: String,
}

impl fmt::Display for CpuReferenceExecutionFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(node) = &self.failed_node {
            return write!(
                formatter,
                "CPU reference node {} ({}) failed: {}",
                node.id.as_str(),
                node.kind.as_str(),
                self.message
            );
        }
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CpuReferenceExecutionFailure {}

/// Successful typed CPU reference result with outputs and bounded completion evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct CpuReferenceTracedExecution {
    pub outputs: BTreeMap<ValueId, HostTensor>,
    pub trace: CpuReferenceExecutionTrace,
}

#[derive(Debug, Clone)]
struct CpuReferenceFault {
    class: CpuReferenceFailureClass,
    failed_node: Option<CpuReferenceNodeRef>,
    reason_code: &'static str,
    message: String,
}

impl CpuReferenceFault {
    fn global(
        class: CpuReferenceFailureClass,
        reason_code: &'static str,
        message: impl Into<String>,
    ) -> Self {
        Self {
            class,
            failed_node: None,
            reason_code,
            message: bounded_failure_message(message.into()),
        }
    }

    fn node(
        class: CpuReferenceFailureClass,
        node: &GraphNode,
        reason_code: &'static str,
        message: impl Into<String>,
    ) -> Self {
        let message = message.into();
        let prefix = node_error_prefix(node);
        let message = message.strip_prefix(&prefix).unwrap_or(&message).to_owned();
        Self {
            class,
            failed_node: Some(CpuReferenceNodeRef::from_node(node)),
            reason_code,
            message: bounded_failure_message(message),
        }
    }

    fn into_failure(self, trace: CpuReferenceExecutionTrace) -> CpuReferenceExecutionFailure {
        CpuReferenceExecutionFailure {
            class: self.class,
            failed_node: self.failed_node,
            trace,
            reason_code: self.reason_code,
            message: self.message,
        }
    }
}

fn bounded_failure_message(mut message: String) -> String {
    if message.len() <= MAX_CPU_REFERENCE_FAILURE_MESSAGE_BYTES {
        return message;
    }
    let mut end = MAX_CPU_REFERENCE_FAILURE_MESSAGE_BYTES;
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    message.truncate(end);
    message
}

trait CompletedNodeSink {
    fn record_completed(&mut self, node: &GraphNode) -> Result<(), CpuReferenceFault>;
}

#[derive(Debug)]
struct BoundedCompletedNodes {
    trace: CpuReferenceExecutionTrace,
    limit: usize,
}

impl BoundedCompletedNodes {
    fn new() -> Self {
        Self::with_limit(MAX_CPU_REFERENCE_TRACE_NODES)
    }

    fn with_limit(limit: usize) -> Self {
        Self {
            trace: CpuReferenceExecutionTrace::default(),
            limit,
        }
    }

    fn into_trace(self) -> CpuReferenceExecutionTrace {
        self.trace
    }
}

impl CompletedNodeSink for BoundedCompletedNodes {
    fn record_completed(&mut self, node: &GraphNode) -> Result<(), CpuReferenceFault> {
        // ModelGraph validation bounds the node count, so saturation is unreachable
        // in admitted executions while still keeping trace collection panic-free.
        self.trace.completed_node_count = self.trace.completed_node_count.saturating_add(1);
        if self.trace.completed_nodes_truncated || self.trace.completed_nodes.len() >= self.limit {
            self.trace.completed_nodes_truncated = true;
            return Ok(());
        }
        if self.trace.completed_nodes.try_reserve(1).is_err() {
            self.trace.completed_nodes_truncated = true;
            return Ok(());
        }
        self.trace
            .completed_nodes
            .push(CpuReferenceNodeRef::from_node(node));
        Ok(())
    }
}

#[derive(Debug, Default)]
struct LegacyCompletedNodes {
    ids: Vec<crate::model_graph::NodeId>,
}

impl CompletedNodeSink for LegacyCompletedNodes {
    fn record_completed(&mut self, node: &GraphNode) -> Result<(), CpuReferenceFault> {
        self.ids.try_reserve(1).map_err(|_| {
            CpuReferenceFault::node(
                CpuReferenceFailureClass::Resource,
                node,
                "trace_metadata_allocation",
                "CPU reference execution trace allocation failed",
            )
        })?;
        self.ids.push(node.id.clone());
        Ok(())
    }
}

/// Stateless deterministic CPU reference executor.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CpuReferenceExecutor;

impl CpuReferenceExecutor {
    /// Compatibility wrapper that retains the existing result type and string-error surface.
    ///
    /// Successful results retain [`CpuReferenceExecution::executed_node_ids`] and
    /// [`CpuReferenceExecution::outputs`]. Failures are rendered as the legacy `String`
    /// surface. This wrapper applies the current executor contract, including fail-closed
    /// rejection of non-finite F32 payloads; new integrations should use
    /// [`Self::execute_traced`] for the typed class, failed-node identity, and bounded
    /// completion evidence.
    pub fn execute(
        &self,
        graph: &ModelGraph,
        inputs: BTreeMap<ValueId, HostTensor>,
        weights: BTreeMap<WeightId, HostTensor>,
    ) -> Result<CpuReferenceExecution, String> {
        let mut completed = LegacyCompletedNodes::default();
        let outputs = self
            .execute_core(graph, inputs, weights, &mut completed)
            .map_err(|fault| {
                fault
                    .into_failure(CpuReferenceExecutionTrace::default())
                    .to_string()
            })?;
        Ok(CpuReferenceExecution {
            executed_node_ids: completed.ids,
            outputs,
        })
    }

    /// Executes with typed terminal failure information and bounded completion evidence.
    pub fn execute_traced(
        &self,
        graph: &ModelGraph,
        inputs: BTreeMap<ValueId, HostTensor>,
        weights: BTreeMap<WeightId, HostTensor>,
    ) -> Result<CpuReferenceTracedExecution, CpuReferenceExecutionFailure> {
        let mut completed = BoundedCompletedNodes::new();
        match self.execute_core(graph, inputs, weights, &mut completed) {
            Ok(outputs) => Ok(CpuReferenceTracedExecution {
                outputs,
                trace: completed.into_trace(),
            }),
            Err(fault) => Err(fault.into_failure(completed.into_trace())),
        }
    }

    fn execute_core<S: CompletedNodeSink>(
        &self,
        graph: &ModelGraph,
        inputs: BTreeMap<ValueId, HostTensor>,
        weights: BTreeMap<WeightId, HostTensor>,
        completed: &mut S,
    ) -> Result<BTreeMap<ValueId, HostTensor>, CpuReferenceFault> {
        graph.validate().map_err(|message| {
            CpuReferenceFault::global(
                CpuReferenceFailureClass::InvalidInput,
                "graph_validation",
                message,
            )
        })?;
        validate_binding_map_admission(&inputs, MAX_GRAPH_ENDPOINTS, "graph input").map_err(
            |message| {
                CpuReferenceFault::global(
                    CpuReferenceFailureClass::InvalidInput,
                    "input_map",
                    message,
                )
            },
        )?;
        validate_binding_map_admission(&weights, MAX_GRAPH_DECLARATIONS, "logical weight")
            .map_err(|message| {
                CpuReferenceFault::global(
                    CpuReferenceFailureClass::InvalidInput,
                    "weight_map",
                    message,
                )
            })?;

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
        let graph_input_ids = graph
            .inputs
            .iter()
            .map(|input_id| (input_id.clone(), ()))
            .collect::<BTreeMap<_, _>>();
        let resource_plan = preflight_graph(graph, &value_specs, &weight_specs)?;
        validate_exact_keys(&inputs, &graph_input_ids, "graph input").map_err(|message| {
            CpuReferenceFault::global(CpuReferenceFailureClass::InvalidInput, "input_map", message)
        })?;
        validate_exact_keys(&weights, &weight_specs, "logical weight").map_err(|message| {
            CpuReferenceFault::global(
                CpuReferenceFailureClass::InvalidInput,
                "weight_map",
                message,
            )
        })?;

        for input_id in &graph.inputs {
            let tensor = inputs.get(input_id).ok_or_else(|| {
                CpuReferenceFault::global(
                    CpuReferenceFailureClass::InvalidInput,
                    "input_map",
                    format!("missing graph input {}", input_id.as_str()),
                )
            })?;
            validate_tensor_against(
                tensor,
                value_specs.get(input_id).ok_or_else(|| {
                    CpuReferenceFault::global(
                        CpuReferenceFailureClass::Internal,
                        "value_spec_lookup",
                        format!("missing graph input spec {}", input_id.as_str()),
                    )
                })?,
                &format!("graph input {}", input_id.as_str()),
            )
            .map_err(|message| {
                CpuReferenceFault::global(
                    CpuReferenceFailureClass::InvalidInput,
                    "input_payload",
                    message,
                )
            })?;
            validate_finite_host_tensor(tensor, &format!("graph input {}", input_id.as_str()))
                .map_err(|message| {
                    CpuReferenceFault::global(
                        CpuReferenceFailureClass::Numerical,
                        "nonfinite_input",
                        message,
                    )
                })?;
            validate_cpu_input_layout(
                value_specs.get(input_id).ok_or_else(|| {
                    CpuReferenceFault::global(
                        CpuReferenceFailureClass::Internal,
                        "value_spec_lookup",
                        format!("missing graph input spec {}", input_id.as_str()),
                    )
                })?,
                &format!("graph input {}", input_id.as_str()),
            )
            .map_err(|message| {
                CpuReferenceFault::global(
                    CpuReferenceFailureClass::Unsupported,
                    "input_layout",
                    message,
                )
            })?;
        }
        for (weight_id, tensor) in &weights {
            let spec = weight_specs.get(weight_id).ok_or_else(|| {
                CpuReferenceFault::global(
                    CpuReferenceFailureClass::InvalidInput,
                    "weight_map",
                    format!("unknown logical weight {}", weight_id.as_str()),
                )
            })?;
            if spec.format != NumericalFormat::F32 {
                return Err(CpuReferenceFault::global(
                    CpuReferenceFailureClass::Unsupported,
                    "weight_format",
                    format!(
                        "CPU reference does not materialize non-F32 logical weight {} with format {}",
                        weight_id.as_str(),
                        spec.format.as_str()
                    ),
                ));
            }
            validate_tensor_against(
                tensor,
                spec,
                &format!("logical weight {}", weight_id.as_str()),
            )
            .map_err(|message| {
                CpuReferenceFault::global(
                    CpuReferenceFailureClass::InvalidInput,
                    "weight_payload",
                    message,
                )
            })?;
            validate_finite_host_tensor(tensor, &format!("logical weight {}", weight_id.as_str()))
                .map_err(|message| {
                    CpuReferenceFault::global(
                        CpuReferenceFailureClass::Numerical,
                        "nonfinite_weight",
                        message,
                    )
                })?;
            validate_cpu_weight_layout(spec, &format!("logical weight {}", weight_id.as_str()))
                .map_err(|message| {
                    CpuReferenceFault::global(
                        CpuReferenceFailureClass::Unsupported,
                        "weight_layout",
                        message,
                    )
                })?;
        }

        let initial_payload_elements = checked_payload_elements(&inputs, "graph input payload")
            .map_err(|message| {
                CpuReferenceFault::global(
                    CpuReferenceFailureClass::InvalidInput,
                    "input_payload",
                    message,
                )
            })?
            .checked_add(
                checked_payload_elements(&weights, "logical weight payload").map_err(
                    |message| {
                        CpuReferenceFault::global(
                            CpuReferenceFailureClass::InvalidInput,
                            "weight_payload",
                            message,
                        )
                    },
                )?,
            )
            .ok_or_else(|| {
                CpuReferenceFault::global(
                    CpuReferenceFailureClass::Resource,
                    "element_budget",
                    "CPU reference initial payload element count overflows usize",
                )
            })?;
        let planned_total_elements = checked_total_element_budget(
            initial_payload_elements,
            resource_plan.execution_elements,
            MAX_CPU_REFERENCE_TOTAL_ELEMENTS,
        )
        .map_err(|message| {
            CpuReferenceFault::global(
                CpuReferenceFailureClass::Resource,
                "element_budget",
                message,
            )
        })?;

        let mut values = inputs;
        let mut allocated_elements = initial_payload_elements;
        let mut runtime = RuntimeExecutionContext::default();
        for node in &graph.nodes {
            let node_outputs = self.execute_node(
                node,
                &values,
                &weights,
                &value_specs,
                &weight_specs,
                &mut allocated_elements,
                &mut runtime,
            )?;
            for (value_id, tensor) in node_outputs {
                validate_finite_host_tensor(&tensor, &format!("node output {}", value_id.as_str()))
                    .map_err(|message| {
                        CpuReferenceFault::node(
                            CpuReferenceFailureClass::Numerical,
                            node,
                            "nonfinite_output",
                            message,
                        )
                    })?;
                if values.contains_key(&value_id) {
                    return Err(CpuReferenceFault::node(
                        CpuReferenceFailureClass::Internal,
                        node,
                        "output_overwrite",
                        format!("would overwrite existing value {}", value_id.as_str()),
                    ));
                }
                values.insert(value_id, tensor);
            }
            completed.record_completed(node)?;
        }

        if allocated_elements != planned_total_elements {
            return Err(CpuReferenceFault::global(
                CpuReferenceFailureClass::Internal,
                "allocation_plan_mismatch",
                "CPU reference allocation plan disagrees with execution accounting",
            ));
        }

        let mut outputs = BTreeMap::new();
        for output_id in &graph.outputs {
            let output = values.remove(output_id).ok_or_else(|| {
                CpuReferenceFault::global(
                    CpuReferenceFailureClass::Internal,
                    "output_unavailable",
                    format!(
                        "CPU reference final output {} is unavailable",
                        output_id.as_str()
                    ),
                )
            })?;
            outputs.insert(output_id.clone(), output);
        }
        Ok(outputs)
    }

    fn execute_node(
        &self,
        node: &GraphNode,
        values: &BTreeMap<ValueId, HostTensor>,
        weights: &BTreeMap<WeightId, HostTensor>,
        value_specs: &BTreeMap<ValueId, &TensorSpec>,
        weight_specs: &BTreeMap<WeightId, &TensorSpec>,
        allocated_elements: &mut usize,
        runtime: &mut RuntimeExecutionContext,
    ) -> Result<Vec<(ValueId, HostTensor)>, CpuReferenceFault> {
        match &node.kind {
            GraphNodeKind::Embedding {
                vocab_size,
                hidden_size,
            } => {
                let input = node_internal(node, node_input(node, values, 0))?;
                let weight = node_internal(node, node_weight(node, weights, weight_specs, 0))?;
                let output_spec = node_internal(node, node_output_spec(node, value_specs, 0))?;
                let output = embedding_f32(
                    input,
                    weight,
                    *vocab_size,
                    *hidden_size,
                    output_spec,
                    allocated_elements,
                    runtime,
                )
                .map_err(|error| runtime_node_fault(node, runtime, error))?;
                Ok(vec![(node.outputs[0].clone(), output)])
            }
            GraphNodeKind::Linear { has_bias } => {
                let input = node_internal(node, node_input(node, values, 0))?;
                let matrix = node_internal(node, node_weight(node, weights, weight_specs, 0))?;
                let bias = if *has_bias {
                    Some(node_internal(
                        node,
                        node_weight(node, weights, weight_specs, 1),
                    )?)
                } else {
                    None
                };
                let output_spec = node_internal(node, node_output_spec(node, value_specs, 0))?;
                let output = linear_f32(
                    input,
                    matrix,
                    bias,
                    output_spec,
                    allocated_elements,
                    runtime,
                )
                .map_err(|error| runtime_node_fault(node, runtime, error))?;
                Ok(vec![(node.outputs[0].clone(), output)])
            }
            GraphNodeKind::FusedLinearGroup { output_count } => {
                let input = node_internal(node, node_input(node, values, 0))?;
                let mut outputs = Vec::new();
                outputs.try_reserve(*output_count).map_err(|_| {
                    CpuReferenceFault::node(
                        CpuReferenceFailureClass::Resource,
                        node,
                        "metadata_allocation",
                        "fused linear output metadata allocation failed",
                    )
                })?;
                for index in 0..*output_count {
                    let matrix =
                        node_internal(node, node_weight(node, weights, weight_specs, index))?;
                    let output_spec =
                        node_internal(node, node_output_spec(node, value_specs, index))?;
                    let output = linear_f32(
                        input,
                        matrix,
                        None,
                        output_spec,
                        allocated_elements,
                        runtime,
                    )
                    .map_err(|error| runtime_node_fault(node, runtime, error))?;
                    outputs.push((node.outputs[index].clone(), output));
                }
                Ok(outputs)
            }
            GraphNodeKind::GroupedLastSplit {
                groups,
                segment_widths,
            } => {
                let input = node_internal(node, node_input(node, values, 0))?;
                let segment_total = node_internal(
                    node,
                    cpu_grouped_last_split_geometry(*groups, segment_widths),
                )?;
                let mut outputs = Vec::new();
                outputs.try_reserve(segment_widths.len()).map_err(|_| {
                    CpuReferenceFault::node(
                        CpuReferenceFailureClass::Resource,
                        node,
                        "metadata_allocation",
                        "grouped last split output metadata allocation failed",
                    )
                })?;
                let mut segment_offset = 0_usize;
                for (index, segment_width) in segment_widths.iter().copied().enumerate() {
                    let output_spec =
                        node_internal(node, node_output_spec(node, value_specs, index))?;
                    let output = grouped_last_split_output_f32(
                        input,
                        *groups,
                        segment_total,
                        segment_offset,
                        segment_width,
                        output_spec,
                        allocated_elements,
                        runtime,
                    )
                    .map_err(|error| runtime_node_fault(node, runtime, error))?;
                    outputs.push((node.outputs[index].clone(), output));
                    segment_offset =
                        segment_offset.checked_add(segment_width).ok_or_else(|| {
                            CpuReferenceFault::node(
                                CpuReferenceFailureClass::Internal,
                                node,
                                "runtime_invariant",
                                "grouped last split segment offset overflows usize",
                            )
                        })?;
                }
                Ok(outputs)
            }
            GraphNodeKind::LastAxisSplit { segment_widths } => {
                let input = node_internal(node, node_input(node, values, 0))?;
                let segment_total =
                    node_internal(node, cpu_grouped_last_split_geometry(1, segment_widths))?;
                let mut outputs = Vec::new();
                outputs.try_reserve(segment_widths.len()).map_err(|_| {
                    CpuReferenceFault::node(
                        CpuReferenceFailureClass::Resource,
                        node,
                        "metadata_allocation",
                        "last axis split output metadata allocation failed",
                    )
                })?;
                let mut offset = 0_usize;
                for (index, width) in segment_widths.iter().copied().enumerate() {
                    let spec = node_internal(node, node_output_spec(node, value_specs, index))?;
                    let output = grouped_last_split_output_f32(
                        input,
                        1,
                        segment_total,
                        offset,
                        width,
                        spec,
                        allocated_elements,
                        runtime,
                    )
                    .map_err(|error| runtime_node_fault(node, runtime, error))?;
                    outputs.push((node.outputs[index].clone(), output));
                    offset = offset.checked_add(width).ok_or_else(|| {
                        CpuReferenceFault::node(
                            CpuReferenceFailureClass::Internal,
                            node,
                            "runtime_invariant",
                            "last axis split offset overflows usize",
                        )
                    })?;
                }
                Ok(outputs)
            }
            GraphNodeKind::Activation { kind } => {
                let input = node_internal(node, node_input(node, values, 0))?;
                let output_spec = node_internal(node, node_output_spec(node, value_specs, 0))?;
                let output = activation_f32(input, kind, output_spec, allocated_elements, runtime)
                    .map_err(|error| runtime_node_fault(node, runtime, error))?;
                Ok(vec![(node.outputs[0].clone(), output)])
            }
            GraphNodeKind::GatedMultiply { activation } => {
                let value_input = node_internal(node, node_input(node, values, 0))?;
                let gate = node_internal(node, node_input(node, values, 1))?;
                let output_spec = node_internal(node, node_output_spec(node, value_specs, 0))?;
                let output = gated_multiply_f32(
                    value_input,
                    gate,
                    activation,
                    output_spec,
                    allocated_elements,
                    runtime,
                )
                .map_err(|error| runtime_node_fault(node, runtime, error))?;
                Ok(vec![(node.outputs[0].clone(), output)])
            }
            GraphNodeKind::GatedMlp {
                intermediate_size,
                activation,
            } => {
                let input = node_internal(node, node_input(node, values, 0))?;
                let gate = node_internal(node, node_weight(node, weights, weight_specs, 0))?;
                let up = node_internal(node, node_weight(node, weights, weight_specs, 1))?;
                let down = node_internal(node, node_weight(node, weights, weight_specs, 2))?;
                let output_spec = node_internal(node, node_output_spec(node, value_specs, 0))?;
                if node.weights.len() != 3 {
                    return Err(CpuReferenceFault::node(
                        CpuReferenceFailureClass::Internal,
                        node,
                        "node_arity",
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
                    runtime,
                )
                .map_err(|error| runtime_node_fault(node, runtime, error))?;
                Ok(vec![(node.outputs[0].clone(), output)])
            }
            GraphNodeKind::Residual => {
                let left = node_internal(node, node_input(node, values, 0))?;
                let right = node_internal(node, node_input(node, values, 1))?;
                let output_spec = node_internal(node, node_output_spec(node, value_specs, 0))?;
                let output = residual_f32(left, right, output_spec, allocated_elements, runtime)
                    .map_err(|error| runtime_node_fault(node, runtime, error))?;
                Ok(vec![(node.outputs[0].clone(), output)])
            }
            GraphNodeKind::LmHead { .. } => {
                let input = node_internal(node, node_input(node, values, 0))?;
                let matrix = node_internal(node, node_weight(node, weights, weight_specs, 0))?;
                let output_spec = node_internal(node, node_output_spec(node, value_specs, 0))?;
                let output = linear_f32(
                    input,
                    matrix,
                    None,
                    output_spec,
                    allocated_elements,
                    runtime,
                )
                .map_err(|error| runtime_node_fault(node, runtime, error))?;
                Ok(vec![(node.outputs[0].clone(), output)])
            }
            GraphNodeKind::Norm {
                epsilon,
                kind,
                affine,
                axis,
            }
            | GraphNodeKind::FinalNorm {
                epsilon,
                kind,
                affine,
                axis,
            } => {
                validate_cpu_rms_normalization(*kind, *axis).map_err(|message| {
                    CpuReferenceFault::node(
                        CpuReferenceFailureClass::Unsupported,
                        node,
                        "normalization_kind",
                        message,
                    )
                })?;
                let input = node_internal(node, node_input(node, values, 0))?;
                let scale = if matches!(
                    affine,
                    NormalizationAffine::Scale
                        | NormalizationAffine::UnitOffsetScale
                        | NormalizationAffine::ScaleAndBias
                ) {
                    Some(node_internal(
                        node,
                        node_weight(node, weights, weight_specs, 0),
                    )?)
                } else {
                    None
                };
                let bias = if *affine == NormalizationAffine::ScaleAndBias {
                    Some(node_internal(
                        node,
                        node_weight(node, weights, weight_specs, 1),
                    )?)
                } else {
                    None
                };
                let output_spec = node_internal(node, node_output_spec(node, value_specs, 0))?;
                let output = rms_norm_f32(
                    input,
                    scale,
                    bias,
                    *kind,
                    *affine,
                    *axis,
                    epsilon.get(),
                    output_spec,
                    allocated_elements,
                    runtime,
                )
                .map_err(|error| runtime_node_fault(node, runtime, error))?;
                Ok(vec![(node.outputs[0].clone(), output)])
            }
            GraphNodeKind::RotaryPosition {
                heads,
                head_dim,
                rotary_dim,
                base,
                pairing,
            } => {
                let values_input = node_internal(node, node_input(node, values, 0))?;
                let positions = node_internal(node, node_input(node, values, 1))?;
                let output_spec = node_internal(node, node_output_spec(node, value_specs, 0))?;
                let output = rotary_f32(
                    values_input,
                    positions,
                    *heads,
                    *head_dim,
                    *rotary_dim,
                    base.get(),
                    *pairing,
                    output_spec,
                    allocated_elements,
                    runtime,
                )
                .map_err(|error| runtime_node_fault(node, runtime, error))?;
                Ok(vec![(node.outputs[0].clone(), output)])
            }
            GraphNodeKind::CausalGqaAttentionCore {
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
            } => {
                let query = node_internal(node, node_input(node, values, 0))?;
                let key = node_internal(node, node_input(node, values, 1))?;
                let value = node_internal(node, node_input(node, values, 2))?;
                let output_spec = node_internal(node, node_output_spec(node, value_specs, 0))?;
                let output = causal_gqa_attention_f32(
                    query,
                    key,
                    value,
                    *q_heads,
                    *kv_heads,
                    *head_dim,
                    *value_dim,
                    softmax_scale.get(),
                    output_spec,
                    allocated_elements,
                    runtime,
                )
                .map_err(|error| runtime_node_fault(node, runtime, error))?;
                Ok(vec![(node.outputs[0].clone(), output)])
            }
            GraphNodeKind::CausalDepthwiseConv1d {
                channels,
                kernel_size,
            } => {
                let input = node_internal(node, node_input(node, values, 0))?;
                let kernel = node_internal(node, node_weight(node, weights, weight_specs, 0))?;
                let spec = node_internal(node, node_output_spec(node, value_specs, 0))?;
                let output = causal_depthwise_conv1d_f32(
                    input,
                    kernel,
                    *channels,
                    *kernel_size,
                    spec,
                    allocated_elements,
                    runtime,
                )
                .map_err(|error| runtime_node_fault(node, runtime, error))?;
                Ok(vec![(node.outputs[0].clone(), output)])
            }
            GraphNodeKind::GatedDecayParameters { channels } => {
                let decay = node_internal(node, node_input(node, values, 0))?;
                let update = node_internal(node, node_input(node, values, 1))?;
                let log_rate = node_internal(node, node_weight(node, weights, weight_specs, 0))?;
                let time_bias = node_internal(node, node_weight(node, weights, weight_specs, 1))?;
                let log_spec = node_internal(node, node_output_spec(node, value_specs, 0))?;
                let update_spec = node_internal(node, node_output_spec(node, value_specs, 1))?;
                let (log_decay, update_rate) = gated_decay_parameters_f32(
                    decay,
                    update,
                    log_rate,
                    time_bias,
                    *channels,
                    log_spec,
                    update_spec,
                    allocated_elements,
                    runtime,
                )
                .map_err(|error| runtime_node_fault(node, runtime, error))?;
                Ok(vec![
                    (node.outputs[0].clone(), log_decay),
                    (node.outputs[1].clone(), update_rate),
                ])
            }
            GraphNodeKind::GatedDeltaRuleScan {
                key_heads,
                value_heads,
                key_dim,
                value_dim,
            } => {
                let query = node_internal(node, node_input(node, values, 0))?;
                let key = node_internal(node, node_input(node, values, 1))?;
                let value = node_internal(node, node_input(node, values, 2))?;
                let decay = node_internal(node, node_input(node, values, 3))?;
                let update = node_internal(node, node_input(node, values, 4))?;
                let spec = node_internal(node, node_output_spec(node, value_specs, 0))?;
                let (output, _) = gated_delta_rule_scan_f32(
                    query,
                    key,
                    value,
                    decay,
                    update,
                    *key_heads,
                    *value_heads,
                    *key_dim,
                    *value_dim,
                    None,
                    spec,
                    allocated_elements,
                    runtime,
                )
                .map_err(|error| runtime_node_fault(node, runtime, error))?;
                Ok(vec![(node.outputs[0].clone(), output)])
            }
            GraphNodeKind::DenseAttention { .. }
            | GraphNodeKind::RecurrentAttention { .. }
            | GraphNodeKind::Sampling { .. } => Err(CpuReferenceFault::node(
                CpuReferenceFailureClass::Unsupported,
                node,
                "unsupported_node",
                &format!(
                    "unsupported CPU reference node kind {}",
                    node_kind_name(&node.kind)
                ),
            )),
        }
    }
}

#[derive(Debug, Default)]
struct RuntimeExecutionContext {
    allocation_failed: bool,
    numerical_failed: bool,
    unsupported_failure_reason: Option<&'static str>,
}

fn node_internal<T>(node: &GraphNode, result: Result<T, String>) -> Result<T, CpuReferenceFault> {
    result.map_err(|message| {
        CpuReferenceFault::node(
            CpuReferenceFailureClass::Internal,
            node,
            "runtime_invariant",
            message,
        )
    })
}

fn runtime_node_fault(
    node: &GraphNode,
    runtime: &RuntimeExecutionContext,
    message: String,
) -> CpuReferenceFault {
    if runtime.allocation_failed {
        CpuReferenceFault::node(
            CpuReferenceFailureClass::Resource,
            node,
            "allocation",
            message,
        )
    } else if runtime.numerical_failed {
        CpuReferenceFault::node(
            CpuReferenceFailureClass::Numerical,
            node,
            "runtime_numerical",
            message,
        )
    } else if let Some(reason_code) = runtime.unsupported_failure_reason {
        CpuReferenceFault::node(
            CpuReferenceFailureClass::Unsupported,
            node,
            reason_code,
            message,
        )
    } else {
        CpuReferenceFault::node(
            CpuReferenceFailureClass::InvalidInput,
            node,
            "runtime_input",
            message,
        )
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
) -> Result<ResourcePlan, CpuReferenceFault> {
    let mut plan = ResourcePlan::default();
    for node in &graph.nodes {
        if !node.states.is_empty() {
            return Err(CpuReferenceFault::node(
                CpuReferenceFailureClass::Unsupported,
                node,
                "stateful_node",
                "stateful execution is unsupported by the stateless CPU reference",
            ));
        }
        match &node.kind {
            GraphNodeKind::Embedding { .. } => {
                let input = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_input_spec(node, value_specs, 0),
                )?;
                let output = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_output_spec(node, value_specs, 0),
                )?;
                let weight = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_weight_spec(node, weight_specs, 0),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "token_contract",
                    validate_cpu_token_spec(input),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "embedding_layout",
                    validate_cpu_embedding_output_spec(output),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "weight_layout",
                    validate_cpu_weight_layout(weight, "embedding weight"),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "element_budget",
                    reserve_output_elements(&mut plan, node, output),
                )?;
                let output_elements = preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "element_budget",
                    spec_elements(output, "embedding output"),
                )?;
                let work = u64::try_from(output_elements).map_err(|_| {
                    CpuReferenceFault::node(
                        CpuReferenceFailureClass::Resource,
                        node,
                        "work_budget",
                        "embedding output elements exceed u64",
                    )
                })?;
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "work_budget",
                    add_work_units(&mut plan, node, work),
                )?;
            }
            GraphNodeKind::Linear { has_bias } => {
                let input = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_input_spec(node, value_specs, 0),
                )?;
                let output = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_output_spec(node, value_specs, 0),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "value_layout",
                    validate_cpu_value_spec(input, "linear input"),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "value_layout",
                    validate_cpu_value_spec(output, "linear output"),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "layout_mismatch",
                    require_matching_value_layout(
                        node,
                        "linear input",
                        input,
                        "linear output",
                        output,
                    ),
                )?;
                let weight = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_weight_spec(node, weight_specs, 0),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "weight_layout",
                    validate_cpu_weight_layout(weight, "linear weight"),
                )?;
                if *has_bias {
                    let bias = preflight_result(
                        CpuReferenceFailureClass::Internal,
                        node,
                        "node_contract",
                        node_weight_spec(node, weight_specs, 1),
                    )?;
                    preflight_result(
                        CpuReferenceFailureClass::Unsupported,
                        node,
                        "weight_layout",
                        validate_cpu_weight_layout(bias, "linear bias"),
                    )?;
                }
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "work_budget",
                    plan_linear(&mut plan, node, input, output),
                )?;
            }
            GraphNodeKind::FusedLinearGroup { output_count } => {
                let input = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_input_spec(node, value_specs, 0),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "value_layout",
                    validate_cpu_value_spec(input, "fused linear input"),
                )?;
                for index in 0..*output_count {
                    let output = preflight_result(
                        CpuReferenceFailureClass::Internal,
                        node,
                        "node_contract",
                        node_output_spec(node, value_specs, index),
                    )?;
                    preflight_result(
                        CpuReferenceFailureClass::Unsupported,
                        node,
                        "value_layout",
                        validate_cpu_value_spec(output, "fused linear output"),
                    )?;
                    preflight_result(
                        CpuReferenceFailureClass::Unsupported,
                        node,
                        "layout_mismatch",
                        require_matching_value_layout(
                            node,
                            "fused linear input",
                            input,
                            "fused linear output",
                            output,
                        ),
                    )?;
                    let weight = preflight_result(
                        CpuReferenceFailureClass::Internal,
                        node,
                        "node_contract",
                        node_weight_spec(node, weight_specs, index),
                    )?;
                    preflight_result(
                        CpuReferenceFailureClass::Unsupported,
                        node,
                        "weight_layout",
                        validate_cpu_weight_layout(weight, "fused linear weight"),
                    )?;
                    preflight_result(
                        CpuReferenceFailureClass::Resource,
                        node,
                        "work_budget",
                        plan_linear(&mut plan, node, input, output),
                    )?;
                }
            }
            GraphNodeKind::GroupedLastSplit {
                groups,
                segment_widths,
            } => {
                let input = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_input_spec(node, value_specs, 0),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "value_layout",
                    validate_cpu_value_spec(input, "grouped last split input"),
                )?;
                let segment_total = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    cpu_grouped_last_split_geometry(*groups, segment_widths),
                )?;
                let mut copied_elements = 0_usize;
                for (index, segment_width) in segment_widths.iter().copied().enumerate() {
                    let output = preflight_result(
                        CpuReferenceFailureClass::Internal,
                        node,
                        "node_contract",
                        node_output_spec(node, value_specs, index),
                    )?;
                    preflight_result(
                        CpuReferenceFailureClass::Unsupported,
                        node,
                        "value_layout",
                        validate_cpu_value_spec(output, "grouped last split output"),
                    )?;
                    preflight_result(
                        CpuReferenceFailureClass::Unsupported,
                        node,
                        "layout_mismatch",
                        require_matching_value_layout(
                            node,
                            "grouped last split input",
                            input,
                            "grouped last split output",
                            output,
                        ),
                    )?;
                    preflight_result(
                        CpuReferenceFailureClass::Internal,
                        node,
                        "node_contract",
                        validate_cpu_grouped_last_split_output(
                            input,
                            output,
                            *groups,
                            segment_total,
                            segment_width,
                            index,
                        ),
                    )?;
                    preflight_result(
                        CpuReferenceFailureClass::Resource,
                        node,
                        "element_budget",
                        reserve_output_elements(&mut plan, node, output),
                    )?;
                    copied_elements = copied_elements
                        .checked_add(preflight_result(
                            CpuReferenceFailureClass::Resource,
                            node,
                            "element_budget",
                            spec_elements(output, "grouped last split output"),
                        )?)
                        .ok_or_else(|| {
                            CpuReferenceFault::node(
                                CpuReferenceFailureClass::Resource,
                                node,
                                "work_budget",
                                "grouped last split copied element count overflows usize",
                            )
                        })?;
                }
                let work = u64::try_from(copied_elements).map_err(|_| {
                    CpuReferenceFault::node(
                        CpuReferenceFailureClass::Resource,
                        node,
                        "work_budget",
                        "grouped last split copied element count exceeds u64",
                    )
                })?;
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "work_budget",
                    add_work_units(&mut plan, node, work),
                )?;
            }
            GraphNodeKind::LastAxisSplit { segment_widths } => {
                let input = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_input_spec(node, value_specs, 0),
                )?;
                // PackedRagged is graph-valid because splitting is local to each
                // final axis. This executor has no execution-batch packed offsets,
                // so its CPU capability remains F32 RowMajor/TokensHidden only.
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "value_layout",
                    validate_cpu_value_spec(input, "last axis split input"),
                )?;
                let total = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    cpu_grouped_last_split_geometry(1, segment_widths),
                )?;
                let mut copied = 0_usize;
                for (index, width) in segment_widths.iter().copied().enumerate() {
                    let output = preflight_result(
                        CpuReferenceFailureClass::Internal,
                        node,
                        "node_contract",
                        node_output_spec(node, value_specs, index),
                    )?;
                    preflight_result(
                        CpuReferenceFailureClass::Unsupported,
                        node,
                        "value_layout",
                        validate_cpu_value_spec(output, "last axis split output"),
                    )?;
                    preflight_result(
                        CpuReferenceFailureClass::Unsupported,
                        node,
                        "layout_mismatch",
                        require_matching_value_layout(
                            node,
                            "last axis split input",
                            input,
                            "last axis split output",
                            output,
                        ),
                    )?;
                    preflight_result(
                        CpuReferenceFailureClass::Internal,
                        node,
                        "node_contract",
                        validate_cpu_grouped_last_split_output(
                            input, output, 1, total, width, index,
                        ),
                    )?;
                    preflight_result(
                        CpuReferenceFailureClass::Resource,
                        node,
                        "element_budget",
                        reserve_output_elements(&mut plan, node, output),
                    )?;
                    copied = copied
                        .checked_add(preflight_result(
                            CpuReferenceFailureClass::Resource,
                            node,
                            "element_budget",
                            spec_elements(output, "last axis split output"),
                        )?)
                        .ok_or_else(|| {
                            CpuReferenceFault::node(
                                CpuReferenceFailureClass::Resource,
                                node,
                                "work_budget",
                                "last axis split copied element count overflows usize",
                            )
                        })?;
                }
                let work = u64::try_from(copied).map_err(|_| {
                    CpuReferenceFault::node(
                        CpuReferenceFailureClass::Resource,
                        node,
                        "work_budget",
                        "last axis split work exceeds u64",
                    )
                })?;
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "work_budget",
                    add_work_units(&mut plan, node, work),
                )?;
            }
            GraphNodeKind::Activation { kind } => {
                let input = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_input_spec(node, value_specs, 0),
                )?;
                let output = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_output_spec(node, value_specs, 0),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "value_layout",
                    validate_cpu_value_spec(input, "activation input"),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "value_layout",
                    validate_cpu_value_spec(output, "activation output"),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "layout_mismatch",
                    require_matching_value_layout(
                        node,
                        "activation input",
                        input,
                        "activation output",
                        output,
                    ),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "activation",
                    validate_cpu_activation(kind),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "element_budget",
                    reserve_output_elements(&mut plan, node, output),
                )?;
                let elements = preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "element_budget",
                    spec_elements(output, "activation output"),
                )?;
                let work = u64::try_from(elements).map_err(|_| {
                    CpuReferenceFault::node(
                        CpuReferenceFailureClass::Resource,
                        node,
                        "work_budget",
                        "activation output elements exceed u64",
                    )
                })?;
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "work_budget",
                    add_work_units(&mut plan, node, work),
                )?;
            }
            GraphNodeKind::GatedMultiply { activation } => {
                let value_input = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_input_spec(node, value_specs, 0),
                )?;
                let gate = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_input_spec(node, value_specs, 1),
                )?;
                let output = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_output_spec(node, value_specs, 0),
                )?;
                for (label, spec) in [
                    ("gated multiply value", value_input),
                    ("gated multiply gate", gate),
                    ("gated multiply output", output),
                ] {
                    preflight_result(
                        CpuReferenceFailureClass::Unsupported,
                        node,
                        "value_layout",
                        validate_cpu_value_spec(spec, label),
                    )?;
                }
                for (right_label, right) in [
                    ("gated multiply gate", gate),
                    ("gated multiply output", output),
                ] {
                    preflight_result(
                        CpuReferenceFailureClass::Unsupported,
                        node,
                        "layout_mismatch",
                        require_matching_value_layout(
                            node,
                            "gated multiply value",
                            value_input,
                            right_label,
                            right,
                        ),
                    )?;
                }
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "activation",
                    validate_cpu_gated_multiply_activation(activation),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    validate_cpu_gated_multiply_contract(value_input, gate, output),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "element_budget",
                    reserve_output_elements(&mut plan, node, output),
                )?;
                let elements = preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "element_budget",
                    spec_elements(output, "gated multiply output"),
                )?;
                let work = u64::try_from(elements)
                    .ok()
                    .and_then(|elements| elements.checked_mul(24))
                    .ok_or_else(|| {
                        CpuReferenceFault::node(
                            CpuReferenceFailureClass::Resource,
                            node,
                            "work_budget",
                            "gated multiply work-unit count overflows u64",
                        )
                    })?;
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "work_budget",
                    add_work_units(&mut plan, node, work),
                )?;
            }
            GraphNodeKind::GatedMlp {
                intermediate_size,
                activation,
            } => {
                let input = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_input_spec(node, value_specs, 0),
                )?;
                let output = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_output_spec(node, value_specs, 0),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "value_layout",
                    validate_cpu_value_spec(input, "gated MLP input"),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "value_layout",
                    validate_cpu_value_spec(output, "gated MLP output"),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "layout_mismatch",
                    require_matching_value_layout(
                        node,
                        "gated MLP input",
                        input,
                        "gated MLP output",
                        output,
                    ),
                )?;
                for index in 0..3 {
                    let weight = preflight_result(
                        CpuReferenceFailureClass::Internal,
                        node,
                        "node_contract",
                        node_weight_spec(node, weight_specs, index),
                    )?;
                    preflight_result(
                        CpuReferenceFailureClass::Unsupported,
                        node,
                        "weight_layout",
                        validate_cpu_weight_layout(weight, "gated MLP weight"),
                    )?;
                }
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "activation",
                    validate_cpu_activation(activation),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "work_budget",
                    plan_gated_mlp(&mut plan, node, input, output, *intermediate_size),
                )?;
            }
            GraphNodeKind::Residual => {
                let left = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_input_spec(node, value_specs, 0),
                )?;
                let right = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_input_spec(node, value_specs, 1),
                )?;
                let output = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_output_spec(node, value_specs, 0),
                )?;
                for (label, spec) in [
                    ("residual left input", left),
                    ("residual right input", right),
                    ("residual output", output),
                ] {
                    preflight_result(
                        CpuReferenceFailureClass::Unsupported,
                        node,
                        "value_layout",
                        validate_cpu_value_spec(spec, label),
                    )?;
                }
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "layout_mismatch",
                    require_matching_value_layout(
                        node,
                        "residual left input",
                        left,
                        "residual right input",
                        right,
                    ),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "layout_mismatch",
                    require_matching_value_layout(
                        node,
                        "residual left input",
                        left,
                        "residual output",
                        output,
                    ),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "element_budget",
                    reserve_output_elements(&mut plan, node, output),
                )?;
                let elements = preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "element_budget",
                    spec_elements(output, "residual output"),
                )?;
                let work = u64::try_from(elements).map_err(|_| {
                    CpuReferenceFault::node(
                        CpuReferenceFailureClass::Resource,
                        node,
                        "work_budget",
                        "residual output elements exceed u64",
                    )
                })?;
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "work_budget",
                    add_work_units(&mut plan, node, work),
                )?;
            }
            GraphNodeKind::LmHead { .. } => {
                let input = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_input_spec(node, value_specs, 0),
                )?;
                let output = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_output_spec(node, value_specs, 0),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "value_layout",
                    validate_cpu_value_spec(input, "LM head input"),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "value_layout",
                    validate_cpu_value_spec(output, "LM head output"),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "layout_mismatch",
                    require_matching_value_layout(
                        node,
                        "LM head input",
                        input,
                        "LM head output",
                        output,
                    ),
                )?;
                let weight = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_weight_spec(node, weight_specs, 0),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "weight_layout",
                    validate_cpu_weight_layout(weight, "LM head weight"),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "work_budget",
                    plan_linear(&mut plan, node, input, output),
                )?;
            }
            GraphNodeKind::Norm {
                kind, affine, axis, ..
            }
            | GraphNodeKind::FinalNorm {
                kind, affine, axis, ..
            } => {
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "normalization_kind",
                    validate_cpu_rms_normalization(*kind, *axis),
                )?;
                let input = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_input_spec(node, value_specs, 0),
                )?;
                let output = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_output_spec(node, value_specs, 0),
                )?;
                let scale = if matches!(
                    affine,
                    NormalizationAffine::Scale
                        | NormalizationAffine::UnitOffsetScale
                        | NormalizationAffine::ScaleAndBias
                ) {
                    Some(preflight_result(
                        CpuReferenceFailureClass::Internal,
                        node,
                        "node_contract",
                        node_weight_spec(node, weight_specs, 0),
                    )?)
                } else {
                    None
                };
                let bias = if *affine == NormalizationAffine::ScaleAndBias {
                    Some(preflight_result(
                        CpuReferenceFailureClass::Internal,
                        node,
                        "node_contract",
                        node_weight_spec(node, weight_specs, 1),
                    )?)
                } else {
                    None
                };
                for (label, spec) in [
                    ("RMS normalization input", input),
                    ("RMS normalization output", output),
                ] {
                    preflight_result(
                        CpuReferenceFailureClass::Unsupported,
                        node,
                        "value_layout",
                        validate_cpu_value_spec(spec, label),
                    )?;
                }
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "layout_mismatch",
                    require_matching_value_layout(
                        node,
                        "RMS normalization input",
                        input,
                        "RMS normalization output",
                        output,
                    ),
                )?;
                if let Some(scale) = scale {
                    preflight_result(
                        CpuReferenceFailureClass::Unsupported,
                        node,
                        "weight_layout",
                        validate_cpu_weight_layout(scale, "normalization scale"),
                    )?;
                }
                if let Some(bias) = bias {
                    preflight_result(
                        CpuReferenceFailureClass::Unsupported,
                        node,
                        "weight_layout",
                        validate_cpu_weight_layout(bias, "RMS normalization bias"),
                    )?;
                }
                preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    validate_cpu_rms_norm_contract(
                        input, output, scale, bias, *kind, *affine, *axis,
                    ),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "work_budget",
                    plan_rms_norm(&mut plan, node, input, output, *kind, *affine, *axis),
                )?;
            }
            GraphNodeKind::RotaryPosition {
                heads,
                head_dim,
                rotary_dim,
                base,
                ..
            } => {
                let values = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_input_spec(node, value_specs, 0),
                )?;
                let positions = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_input_spec(node, value_specs, 1),
                )?;
                let output = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_output_spec(node, value_specs, 0),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "rotary_values",
                    validate_cpu_rotary_value_spec(values, "rotary values input"),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "rotary_output",
                    validate_cpu_rotary_value_spec(output, "rotary output"),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "rotary_positions",
                    validate_cpu_rotary_positions_spec(positions),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "rotary_contract",
                    validate_cpu_rotary_contract(
                        values,
                        positions,
                        output,
                        *heads,
                        *head_dim,
                        *rotary_dim,
                        base.get(),
                    ),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "work_budget",
                    plan_rotary(
                        &mut plan,
                        node,
                        values,
                        output,
                        *heads,
                        *head_dim,
                        *rotary_dim,
                    ),
                )?;
            }
            GraphNodeKind::CausalGqaAttentionCore {
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
            } => {
                let query = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_input_spec(node, value_specs, 0),
                )?;
                let key = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_input_spec(node, value_specs, 1),
                )?;
                let value = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_input_spec(node, value_specs, 2),
                )?;
                let output = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_output_spec(node, value_specs, 0),
                )?;
                for (label, spec) in [
                    ("causal GQA query", query),
                    ("causal GQA key", key),
                    ("causal GQA value", value),
                    ("causal GQA context", output),
                ] {
                    preflight_result(
                        CpuReferenceFailureClass::Unsupported,
                        node,
                        "value_layout",
                        validate_cpu_value_spec(spec, label),
                    )?;
                }
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "layout_mismatch",
                    require_matching_value_layout(
                        node,
                        "causal GQA query",
                        query,
                        "causal GQA key",
                        key,
                    ),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "layout_mismatch",
                    require_matching_value_layout(
                        node,
                        "causal GQA query",
                        query,
                        "causal GQA value",
                        value,
                    ),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "layout_mismatch",
                    require_matching_value_layout(
                        node,
                        "causal GQA query",
                        query,
                        "causal GQA context",
                        output,
                    ),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    validate_cpu_causal_gqa_contract(
                        query,
                        key,
                        value,
                        output,
                        *q_heads,
                        *kv_heads,
                        *head_dim,
                        *value_dim,
                        softmax_scale.get(),
                    ),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "element_budget",
                    reserve_output_elements(&mut plan, node, output),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "work_budget",
                    plan_causal_gqa(
                        &mut plan,
                        node,
                        query,
                        output,
                        *q_heads,
                        *kv_heads,
                        *head_dim,
                        *value_dim,
                        softmax_scale.get(),
                    ),
                )?;
            }
            GraphNodeKind::CausalDepthwiseConv1d {
                channels,
                kernel_size,
            } => {
                let input = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_input_spec(node, value_specs, 0),
                )?;
                let output = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_output_spec(node, value_specs, 0),
                )?;
                let kernel = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_weight_spec(node, weight_specs, 0),
                )?;
                for (label, spec) in [
                    ("causal depthwise conv input", input),
                    ("causal depthwise conv output", output),
                ] {
                    preflight_result(
                        CpuReferenceFailureClass::Unsupported,
                        node,
                        "value_layout",
                        validate_cpu_value_spec(spec, label),
                    )?;
                }
                preflight_result(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "weight_layout",
                    validate_cpu_weight_layout(kernel, "causal depthwise conv kernel"),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    validate_cpu_conv_contract(input, output, kernel, *channels, *kernel_size),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "element_budget",
                    reserve_output_elements(&mut plan, node, output),
                )?;
                let elements = preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "element_budget",
                    spec_elements(output, "causal depthwise conv output"),
                )?;
                let work =
                    checked_work_product(elements, *kernel_size, 2, "causal depthwise conv work")
                        .map_err(|m| {
                        CpuReferenceFault::node(
                            CpuReferenceFailureClass::Resource,
                            node,
                            "work_budget",
                            m,
                        )
                    })?;
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "work_budget",
                    add_work_units(&mut plan, node, work),
                )?;
            }
            GraphNodeKind::GatedDecayParameters { channels } => {
                let decay = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_input_spec(node, value_specs, 0),
                )?;
                let update = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_input_spec(node, value_specs, 1),
                )?;
                let log_output = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_output_spec(node, value_specs, 0),
                )?;
                let update_output = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_output_spec(node, value_specs, 1),
                )?;
                let log_rate = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_weight_spec(node, weight_specs, 0),
                )?;
                let time_bias = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_weight_spec(node, weight_specs, 1),
                )?;
                for (label, spec) in [
                    ("gated decay control", decay),
                    ("gated update control", update),
                    ("gated log decay", log_output),
                    ("gated update rate", update_output),
                ] {
                    preflight_result(
                        CpuReferenceFailureClass::Unsupported,
                        node,
                        "value_layout",
                        validate_cpu_value_spec(spec, label),
                    )?;
                }
                for (label, spec) in [("gated log rate", log_rate), ("gated time bias", time_bias)]
                {
                    preflight_result(
                        CpuReferenceFailureClass::Unsupported,
                        node,
                        "weight_layout",
                        validate_cpu_weight_layout(spec, label),
                    )?;
                }
                preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    validate_cpu_decay_contract(
                        decay,
                        update,
                        log_output,
                        update_output,
                        log_rate,
                        time_bias,
                        *channels,
                    ),
                )?;
                for output in [log_output, update_output] {
                    preflight_result(
                        CpuReferenceFailureClass::Resource,
                        node,
                        "element_budget",
                        reserve_output_elements(&mut plan, node, output),
                    )?;
                }
                let elements = preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "element_budget",
                    spec_elements(log_output, "gated decay output"),
                )?;
                let work = u64::try_from(elements)
                    .ok()
                    .and_then(|v| v.checked_mul(48))
                    .ok_or_else(|| {
                        CpuReferenceFault::node(
                            CpuReferenceFailureClass::Resource,
                            node,
                            "work_budget",
                            "gated decay work overflows u64",
                        )
                    })?;
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "work_budget",
                    add_work_units(&mut plan, node, work),
                )?;
            }
            GraphNodeKind::GatedDeltaRuleScan {
                key_heads,
                value_heads,
                key_dim,
                value_dim,
            } => {
                let specs = [
                    preflight_result(
                        CpuReferenceFailureClass::Internal,
                        node,
                        "node_contract",
                        node_input_spec(node, value_specs, 0),
                    )?,
                    preflight_result(
                        CpuReferenceFailureClass::Internal,
                        node,
                        "node_contract",
                        node_input_spec(node, value_specs, 1),
                    )?,
                    preflight_result(
                        CpuReferenceFailureClass::Internal,
                        node,
                        "node_contract",
                        node_input_spec(node, value_specs, 2),
                    )?,
                    preflight_result(
                        CpuReferenceFailureClass::Internal,
                        node,
                        "node_contract",
                        node_input_spec(node, value_specs, 3),
                    )?,
                    preflight_result(
                        CpuReferenceFailureClass::Internal,
                        node,
                        "node_contract",
                        node_input_spec(node, value_specs, 4),
                    )?,
                ];
                let output = preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    node_output_spec(node, value_specs, 0),
                )?;
                for spec in specs.iter().copied().chain(std::iter::once(output)) {
                    preflight_result(
                        CpuReferenceFailureClass::Unsupported,
                        node,
                        "value_layout",
                        validate_cpu_value_spec(spec, "gated delta-rule tensor"),
                    )?;
                }
                preflight_result(
                    CpuReferenceFailureClass::Internal,
                    node,
                    "node_contract",
                    validate_cpu_scan_contract(
                        &specs,
                        output,
                        *key_heads,
                        *value_heads,
                        *key_dim,
                        *value_dim,
                    ),
                )?;
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "element_budget",
                    reserve_output_elements(&mut plan, node, output),
                )?;
                let state_elements = value_heads
                    .checked_mul(*key_dim)
                    .and_then(|v| v.checked_mul(*value_dim))
                    .ok_or_else(|| {
                        CpuReferenceFault::node(
                            CpuReferenceFailureClass::Resource,
                            node,
                            "element_budget",
                            "scan state size overflows usize",
                        )
                    })?;
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "element_budget",
                    reserve_temporary_elements(&mut plan, node, state_elements),
                )?;
                let output_elements = preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "element_budget",
                    spec_elements(output, "scan output"),
                )?;
                let per_value = key_dim
                    .checked_mul(8)
                    .and_then(|v| v.checked_add(40))
                    .ok_or_else(|| {
                        CpuReferenceFault::node(
                            CpuReferenceFailureClass::Resource,
                            node,
                            "work_budget",
                            "scan work factor overflows usize",
                        )
                    })?;
                let work = checked_work_product(output_elements, per_value, 1, "scan work")
                    .map_err(|m| {
                        CpuReferenceFault::node(
                            CpuReferenceFailureClass::Resource,
                            node,
                            "work_budget",
                            m,
                        )
                    })?;
                preflight_result(
                    CpuReferenceFailureClass::Resource,
                    node,
                    "work_budget",
                    add_work_units(&mut plan, node, work),
                )?;
            }
            GraphNodeKind::DenseAttention { .. }
            | GraphNodeKind::RecurrentAttention { .. }
            | GraphNodeKind::Sampling { .. } => {
                return Err(CpuReferenceFault::node(
                    CpuReferenceFailureClass::Unsupported,
                    node,
                    "unsupported_node",
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

fn preflight_result<T>(
    class: CpuReferenceFailureClass,
    node: &GraphNode,
    reason_code: &'static str,
    result: Result<T, String>,
) -> Result<T, CpuReferenceFault> {
    result.map_err(|message| CpuReferenceFault::node(class, node, reason_code, message))
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
        NumericalFormat::F32 => match spec.layout {
            TensorLayout::RowMajor | TensorLayout::TokensHidden | TensorLayout::PackedRagged => {
                Ok(())
            }
            _ => Err(format!(
                "{label} layout is unsupported without explicit stride or packed offsets"
            )),
        },
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

fn validate_cpu_rotary_value_spec(spec: &TensorSpec, label: &str) -> Result<(), String> {
    if spec.format != NumericalFormat::F32 {
        return Err(format!(
            "{label} requires F32 for the CPU reference, got {}",
            spec.format.as_str()
        ));
    }
    match spec.layout {
        TensorLayout::RowMajor | TensorLayout::TokensHidden | TensorLayout::PackedRagged => Ok(()),
        _ => Err(format!(
            "{label} layout is unsupported without explicit stride or packed offsets"
        )),
    }
}

fn validate_cpu_rotary_contract(
    values: &TensorSpec,
    positions: &TensorSpec,
    output: &TensorSpec,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    base: f32,
) -> Result<(), String> {
    if values.shape.len() < 2 {
        return Err("rotary values input must have rank at least 2".into());
    }
    let hidden = heads
        .checked_mul(head_dim)
        .ok_or_else(|| "rotary heads * head_dim overflows usize".to_string())?;
    if values.shape.last().copied() != Some(hidden) {
        return Err(format!(
            "rotary values feature width must equal heads * head_dim {hidden}"
        ));
    }
    if output.shape != values.shape
        || output.format != values.format
        || output.layout != values.layout
    {
        return Err("rotary output must match values shape, format, and layout".into());
    }
    if positions.shape.as_slice() != &values.shape[..values.shape.len() - 1] {
        return Err(
            "rotary positions shape must match values shape without the final feature axis".into(),
        );
    }
    validate_cpu_rotary_positions_spec(positions)?;
    if heads == 0 || head_dim == 0 || rotary_dim == 0 {
        return Err("rotary heads, head_dim, and rotary_dim must be nonzero".into());
    }
    if rotary_dim % 2 != 0 || rotary_dim > head_dim {
        return Err("rotary_dim must be even and no greater than head_dim".into());
    }
    if !base.is_finite() || base <= 0.0 {
        return Err("rotary base must be finite and positive".into());
    }
    Ok(())
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

fn validate_cpu_rotary_positions_spec(spec: &TensorSpec) -> Result<(), String> {
    if !matches!(spec.format, NumericalFormat::U32 | NumericalFormat::U64) {
        return Err("rotary positions input requires U32 or U64".into());
    }
    if spec.layout != TensorLayout::RowMajor {
        return Err("rotary positions input requires RowMajor layout".into());
    }
    Ok(())
}

fn validate_cpu_causal_gqa_contract(
    query: &TensorSpec,
    key: &TensorSpec,
    value: &TensorSpec,
    output: &TensorSpec,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
) -> Result<(), String> {
    let rank = query.shape.len();
    if rank < 2 || key.shape.len() < 2 || value.shape.len() < 2 || output.shape.len() < 2 {
        return Err("causal GQA tensors must have rank at least 2".into());
    }
    if key.shape.len() != rank || value.shape.len() != rank || output.shape.len() != rank {
        return Err("causal GQA tensors must have equal rank".into());
    }
    if key.shape[..rank - 1] != query.shape[..rank - 1]
        || value.shape[..rank - 1] != query.shape[..rank - 1]
        || output.shape[..rank - 1] != query.shape[..rank - 1]
    {
        return Err("causal GQA tensors must have matching leading and token shapes".into());
    }
    if q_heads == 0 || kv_heads == 0 || head_dim == 0 || value_dim == 0 {
        return Err("causal GQA geometry must be nonzero".into());
    }
    if q_heads % kv_heads != 0 {
        return Err("causal GQA q_heads must divide evenly by kv_heads".into());
    }
    let query_width = q_heads
        .checked_mul(head_dim)
        .ok_or_else(|| "causal GQA query width overflows usize".to_string())?;
    let key_width = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| "causal GQA key width overflows usize".to_string())?;
    let value_width = kv_heads
        .checked_mul(value_dim)
        .ok_or_else(|| "causal GQA value width overflows usize".to_string())?;
    let context_width = q_heads
        .checked_mul(value_dim)
        .ok_or_else(|| "causal GQA context width overflows usize".to_string())?;
    if query.shape.last().copied() != Some(query_width) {
        return Err("causal GQA query feature width does not match q_heads * head_dim".into());
    }
    if key.shape.last().copied() != Some(key_width) {
        return Err("causal GQA key feature width does not match kv_heads * head_dim".into());
    }
    if value.shape.last().copied() != Some(value_width) {
        return Err("causal GQA value feature width does not match kv_heads * value_dim".into());
    }
    if output.shape.last().copied() != Some(context_width) {
        return Err("causal GQA context feature width does not match q_heads * value_dim".into());
    }
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err("causal GQA softmax scale must be finite and positive".into());
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
        ActivationKind::Sigmoid => {
            Err("Sigmoid is supported only by GatedMultiply in the CPU reference".into())
        }
        ActivationKind::Silu | ActivationKind::Relu => Ok(()),
        ActivationKind::Gelu => {
            Err("GELU is unsupported until its exact or tanh contract is explicit".into())
        }
        ActivationKind::Custom(name) => Err(format!("unsupported custom activation {name}")),
    }
}

fn validate_cpu_gated_multiply_activation(kind: &ActivationKind) -> Result<(), String> {
    if kind == &ActivationKind::Sigmoid {
        Ok(())
    } else {
        Err("gated multiply CPU reference requires Sigmoid activation".into())
    }
}

fn cpu_grouped_last_split_geometry(
    groups: usize,
    segment_widths: &[usize],
) -> Result<usize, String> {
    if groups == 0 {
        return Err("grouped last split groups must be nonzero".into());
    }
    if segment_widths.len() < 2 {
        return Err("grouped last split requires at least two segments".into());
    }
    if segment_widths.len() > MAX_GRAPH_ENDPOINTS {
        return Err("grouped last split segment count exceeds graph endpoint limit".into());
    }
    let mut segment_total = 0_usize;
    for (index, segment_width) in segment_widths.iter().copied().enumerate() {
        if segment_width == 0 {
            return Err(format!(
                "grouped last split segment width {index} must be nonzero"
            ));
        }
        segment_total = segment_total
            .checked_add(segment_width)
            .ok_or_else(|| "grouped last split segment width sum overflows usize".to_string())?;
    }
    groups
        .checked_mul(segment_total)
        .ok_or_else(|| "grouped last split input width overflows usize".to_string())?;
    Ok(segment_total)
}

fn validate_cpu_grouped_last_split_output(
    input: &TensorSpec,
    output: &TensorSpec,
    groups: usize,
    segment_total: usize,
    segment_width: usize,
    output_index: usize,
) -> Result<(), String> {
    let rank = input.shape.len();
    if rank == 0 || output.shape.len() != rank {
        return Err(format!(
            "grouped last split output {output_index} must preserve nonzero input rank"
        ));
    }
    if output.shape[..rank - 1] != input.shape[..rank - 1] {
        return Err(format!(
            "grouped last split output {output_index} must preserve leading dimensions"
        ));
    }
    let input_width = groups
        .checked_mul(segment_total)
        .ok_or_else(|| "grouped last split input width overflows usize".to_string())?;
    if input.shape.last().copied() != Some(input_width) {
        return Err(format!(
            "grouped last split input final width must be {input_width}"
        ));
    }
    let output_width = groups
        .checked_mul(segment_width)
        .ok_or_else(|| "grouped last split output width overflows usize".to_string())?;
    if output.shape.last().copied() != Some(output_width) {
        return Err(format!(
            "grouped last split output {output_index} final width must be {output_width}"
        ));
    }
    Ok(())
}

fn validate_cpu_gated_multiply_contract(
    value_input: &TensorSpec,
    gate: &TensorSpec,
    output: &TensorSpec,
) -> Result<(), String> {
    if value_input.shape != gate.shape || value_input.shape != output.shape {
        return Err("gated multiply value, gate, and output shapes must match".into());
    }
    Ok(())
}

fn checked_work_product(
    elements: usize,
    factor: usize,
    multiplier: usize,
    label: &str,
) -> Result<u64, String> {
    let work = elements
        .checked_mul(factor)
        .and_then(|v| v.checked_mul(multiplier))
        .ok_or_else(|| format!("{label} overflows usize"))?;
    u64::try_from(work).map_err(|_| format!("{label} exceeds u64"))
}

fn validate_cpu_conv_contract(
    input: &TensorSpec,
    output: &TensorSpec,
    kernel: &TensorSpec,
    channels: usize,
    kernel_size: usize,
) -> Result<(), String> {
    if channels == 0 || kernel_size == 0 || input.shape.len() < 2 {
        return Err("invalid causal depthwise conv geometry".into());
    }
    if input.shape != output.shape || input.layout != output.layout || input.format != output.format
    {
        return Err("causal depthwise conv input/output mismatch".into());
    }
    if input.shape.last().copied() != Some(channels)
        || kernel.shape.as_slice() != [channels, 1, kernel_size]
    {
        return Err("causal depthwise conv shape mismatch".into());
    }
    Ok(())
}

fn validate_cpu_decay_contract(
    decay: &TensorSpec,
    update: &TensorSpec,
    log_output: &TensorSpec,
    update_output: &TensorSpec,
    log_rate: &TensorSpec,
    time_bias: &TensorSpec,
    channels: usize,
) -> Result<(), String> {
    if channels == 0 || decay.shape.len() < 2 || decay.shape.last().copied() != Some(channels) {
        return Err("invalid gated decay geometry".into());
    }
    for spec in [update, log_output, update_output] {
        if spec.shape != decay.shape || spec.layout != decay.layout || spec.format != decay.format {
            return Err("gated decay tensor metadata mismatch".into());
        }
    }
    if log_rate.shape.as_slice() != [channels] || time_bias.shape.as_slice() != [channels] {
        return Err("gated decay weight shape mismatch".into());
    }
    Ok(())
}

fn validate_cpu_scan_contract(
    specs: &[&TensorSpec; 5],
    output: &TensorSpec,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> Result<(), String> {
    if key_heads == 0
        || value_heads == 0
        || key_dim == 0
        || value_dim == 0
        || value_heads % key_heads != 0
    {
        return Err("invalid gated delta-rule geometry".into());
    }
    let rank = specs[0].shape.len();
    if rank < 2 {
        return Err("gated delta-rule tensors require rank at least 2".into());
    }
    for spec in specs.iter().copied().chain(std::iter::once(output)) {
        if spec.shape.len() != rank
            || spec.shape[..rank - 1] != specs[0].shape[..rank - 1]
            || spec.layout != specs[0].layout
            || spec.format != specs[0].format
        {
            return Err("gated delta-rule tensor metadata mismatch".into());
        }
    }
    let kw = key_heads
        .checked_mul(key_dim)
        .ok_or_else(|| "scan key width overflow".to_string())?;
    let vw = value_heads
        .checked_mul(value_dim)
        .ok_or_else(|| "scan value width overflow".to_string())?;
    if specs[0].shape[rank - 1] != kw
        || specs[1].shape[rank - 1] != kw
        || specs[2].shape[rank - 1] != vw
        || output.shape[rank - 1] != vw
        || specs[3].shape[rank - 1] != value_heads
        || specs[4].shape[rank - 1] != value_heads
    {
        return Err("gated delta-rule feature width mismatch".into());
    }
    Ok(())
}

fn validate_cpu_rms_normalization(
    kind: NormalizationKind,
    axis: NormalizationAxis,
) -> Result<(), String> {
    match (kind, axis) {
        (
            NormalizationKind::Rms | NormalizationKind::L2,
            NormalizationAxis::Last | NormalizationAxis::GroupedLast { .. },
        ) => Ok(()),
        (
            NormalizationKind::Layer,
            NormalizationAxis::Last | NormalizationAxis::GroupedLast { .. },
        ) => Err("Layer normalization is not in the P1-B2 CPU subset".into()),
    }
}

fn validate_cpu_rms_norm_contract(
    input: &TensorSpec,
    output: &TensorSpec,
    scale: Option<&TensorSpec>,
    bias: Option<&TensorSpec>,
    kind: NormalizationKind,
    affine: NormalizationAffine,
    axis: NormalizationAxis,
) -> Result<(), String> {
    if input.shape != output.shape {
        return Err("RMS normalization input and output shapes must match".into());
    }
    let (_, group_width) = rms_norm_group_geometry(&input.shape, axis)?;
    if let Some(scale) = scale {
        validate_cpu_rms_norm_weight_shape(scale, group_width, "scale")?;
    }
    match (kind, affine, scale, bias) {
        (
            NormalizationKind::L2,
            NormalizationAffine::None | NormalizationAffine::FixedScale(_),
            None,
            None,
        ) => Ok(()),
        (
            NormalizationKind::Rms | NormalizationKind::Layer,
            NormalizationAffine::ScaleAndBias,
            Some(_),
            Some(bias),
        ) => validate_cpu_rms_norm_weight_shape(bias, group_width, "bias"),
        (
            NormalizationKind::Rms | NormalizationKind::Layer,
            NormalizationAffine::Scale | NormalizationAffine::UnitOffsetScale,
            Some(_),
            None,
        ) => Ok(()),
        _ => Err("normalization kind, affine, and weights are inconsistent".into()),
    }
}

fn rms_norm_group_geometry(
    shape: &[usize],
    axis: NormalizationAxis,
) -> Result<(usize, usize), String> {
    let final_width = shape
        .last()
        .copied()
        .ok_or_else(|| "RMS normalization input has no final axis".to_string())?;
    if final_width == 0 {
        return Err("RMS normalization final-axis width must be nonzero".into());
    }
    match axis {
        NormalizationAxis::Last => Ok((1, final_width)),
        NormalizationAxis::GroupedLast {
            groups,
            group_width,
        } => {
            if groups == 0 || group_width == 0 {
                return Err(
                    "grouped RMS normalization groups and group_width must be nonzero".into(),
                );
            }
            let grouped_width = groups.checked_mul(group_width).ok_or_else(|| {
                "grouped RMS normalization groups * group_width overflows usize".to_string()
            })?;
            if grouped_width != final_width {
                return Err(format!(
                    "grouped RMS normalization width {grouped_width} does not match final-axis width {final_width}"
                ));
            }
            Ok((groups, group_width))
        }
    }
}

fn validate_cpu_rms_norm_weight_shape(
    weight: &TensorSpec,
    width: usize,
    label: &str,
) -> Result<(), String> {
    if weight.shape.len() != 1 || weight.shape[0] != width {
        return Err(format!(
            "RMS normalization {label} must have shape [{width}]"
        ));
    }
    Ok(())
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

fn plan_rms_norm(
    plan: &mut ResourcePlan,
    node: &GraphNode,
    input: &TensorSpec,
    output: &TensorSpec,
    kind: NormalizationKind,
    affine: NormalizationAffine,
    axis: NormalizationAxis,
) -> Result<(), String> {
    let input_elements = spec_elements(input, "RMS normalization input")
        .map_err(|error| node_error(node, &error))?;
    let output_elements = spec_elements(output, "RMS normalization output")
        .map_err(|error| node_error(node, &error))?;
    if input_elements != output_elements || input.shape != output.shape {
        return Err(node_error(
            node,
            "RMS normalization input and output element counts must match",
        ));
    }
    let (_, group_width) =
        rms_norm_group_geometry(&input.shape, axis).map_err(|error| node_error(node, &error))?;
    let units = input_elements
        .checked_div(group_width)
        .ok_or_else(|| node_error(node, "RMS normalization unit count overflows usize"))?;
    if units
        .checked_mul(group_width)
        .filter(|elements| *elements == input_elements)
        .is_none()
    {
        return Err(node_error(
            node,
            "RMS normalization input element count is not divisible by group_width",
        ));
    }
    reserve_output_elements(plan, node, output)?;

    // Account for the scalar square and reduction add, then the output normalization
    // and affine scale. Divide, epsilon add, sqrt, and reciprocal occur once per
    // normalization unit (one row for Last, or one contiguous group for GroupedLast).
    // UnitOffsetScale additionally charges the unit-offset add; ScaleAndBias charges
    // the bias add.
    let work_per_element = match (kind, affine) {
        (NormalizationKind::L2, NormalizationAffine::None) => 3_u64,
        (NormalizationKind::L2, NormalizationAffine::FixedScale(_)) => 4_u64,
        (NormalizationKind::Rms | NormalizationKind::Layer, NormalizationAffine::Scale) => 4_u64,
        (
            NormalizationKind::Rms | NormalizationKind::Layer,
            NormalizationAffine::UnitOffsetScale | NormalizationAffine::ScaleAndBias,
        ) => 5_u64,
        _ => {
            return Err(node_error(
                node,
                "invalid normalization kind and affine combination",
            ));
        }
    };
    let element_work = u64::try_from(output_elements)
        .map_err(|_| node_error(node, "RMS normalization element count exceeds u64"))?
        .checked_mul(work_per_element)
        .ok_or_else(|| {
            node_error(
                node,
                "RMS normalization element work-unit count overflows u64",
            )
        })?;
    let work_per_unit = if kind == NormalizationKind::L2 { 3 } else { 4 };
    let unit_work = u64::try_from(units)
        .map_err(|_| node_error(node, "RMS normalization unit count exceeds u64"))?
        .checked_mul(work_per_unit)
        .ok_or_else(|| node_error(node, "RMS normalization unit work-unit count overflows u64"))?;
    let work = element_work
        .checked_add(unit_work)
        .ok_or_else(|| node_error(node, "RMS normalization work-unit count overflows u64"))?;
    add_work_units(plan, node, work)
}

fn plan_rotary(
    plan: &mut ResourcePlan,
    node: &GraphNode,
    values: &TensorSpec,
    output: &TensorSpec,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<(), String> {
    if heads == 0 || head_dim == 0 || rotary_dim == 0 || rotary_dim % 2 != 0 {
        return Err(node_error(
            node,
            "rotary heads, head_dim, and rotary_dim must be nonzero with an even rotary_dim",
        ));
    }
    if rotary_dim > head_dim {
        return Err(node_error(node, "rotary_dim must not exceed head_dim"));
    }
    reserve_output_elements(plan, node, output)?;
    let hidden = heads
        .checked_mul(head_dim)
        .ok_or_else(|| node_error(node, "rotary heads * head_dim overflows usize"))?;
    let value_elements =
        spec_elements(values, "rotary values input").map_err(|error| node_error(node, &error))?;
    let rows =
        leading_rows(&values.shape, "rotary values").map_err(|error| node_error(node, &error))?;
    if rows.checked_mul(hidden) != Some(value_elements) {
        return Err(node_error(
            node,
            "rotary values element count is not divisible by head width",
        ));
    }
    let pair_count = rows
        .checked_mul(heads)
        .and_then(|count| count.checked_mul(rotary_dim / 2))
        .ok_or_else(|| node_error(node, "rotary pair count overflows usize"))?;
    let suffix_count = rows
        .checked_mul(heads)
        .and_then(|count| count.checked_mul(head_dim - rotary_dim))
        .ok_or_else(|| node_error(node, "rotary suffix element count overflows usize"))?;
    let row_count =
        u64::try_from(rows).map_err(|_| node_error(node, "rotary row count exceeds u64"))?;
    let pair_work = u64::try_from(pair_count)
        .map_err(|_| node_error(node, "rotary pair count exceeds u64"))?
        .checked_mul(32)
        .ok_or_else(|| node_error(node, "rotary pair work-unit count overflows u64"))?;
    let suffix_work = u64::try_from(suffix_count)
        .map_err(|_| node_error(node, "rotary suffix count exceeds u64"))?;
    let work = pair_work
        .checked_add(suffix_work)
        .and_then(|units| units.checked_add(row_count))
        .ok_or_else(|| node_error(node, "rotary work-unit count overflows u64"))?;
    add_work_units(plan, node, work)
}

fn plan_causal_gqa(
    plan: &mut ResourcePlan,
    node: &GraphNode,
    query: &TensorSpec,
    output: &TensorSpec,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
) -> Result<(), String> {
    let rank = query.shape.len();
    if rank < 2 {
        return Err(node_error(
            node,
            "causal GQA query must have rank at least 2",
        ));
    }
    let sequences =
        causal_sequence_count(&query.shape).map_err(|error| node_error(node, &error))?;
    let tokens = query.shape[rank - 2];
    let triangular = checked_triangular_count(tokens).map_err(|error| node_error(node, &error))?;
    let score_count = sequences
        .checked_mul(q_heads)
        .and_then(|count| count.checked_mul(triangular))
        .ok_or_else(|| node_error(node, "causal GQA score count overflows usize"))?;
    let score_count = u64::try_from(score_count)
        .map_err(|_| node_error(node, "causal GQA score count exceeds u64"))?;
    let head_dim =
        u64::try_from(head_dim).map_err(|_| node_error(node, "causal GQA head_dim exceeds u64"))?;
    let value_dim = u64::try_from(value_dim)
        .map_err(|_| node_error(node, "causal GQA value_dim exceeds u64"))?;
    let score_work = head_dim
        .checked_mul(2)
        .and_then(|work| work.checked_add(1))
        .and_then(|work| work.checked_mul(3))
        .and_then(|work| work.checked_add(1))
        .and_then(|work| work.checked_add(17))
        .and_then(|work| work.checked_add(18))
        .and_then(|work| work.checked_add(value_dim.checked_mul(2)?))
        .ok_or_else(|| node_error(node, "causal GQA score work count overflows u64"))?;
    let score_work = score_count
        .checked_mul(score_work)
        .ok_or_else(|| node_error(node, "causal GQA work count overflows u64"))?;
    let output_work = u64::try_from(spec_elements(output, "causal GQA context")?)
        .map_err(|_| node_error(node, "causal GQA output work count exceeds u64"))?;
    let work = score_work
        .checked_add(output_work)
        .ok_or_else(|| node_error(node, "causal GQA work count overflows u64"))?;
    let _ = (kv_heads, softmax_scale);
    add_work_units(plan, node, work)
}

fn causal_sequence_count(shape: &[usize]) -> Result<usize, String> {
    if shape.len() < 2 {
        return Err("causal GQA shape must have rank at least 2".into());
    }
    shape[..shape.len() - 2]
        .iter()
        .try_fold(1_usize, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or_else(|| "causal GQA sequence count overflows usize".to_string())
        })
}

fn checked_triangular_count(tokens: usize) -> Result<usize, String> {
    let half = tokens / 2;
    if tokens % 2 == 0 {
        half.checked_mul(
            tokens
                .checked_add(1)
                .ok_or_else(|| "causal GQA triangular token count overflows usize".to_string())?,
        )
    } else {
        tokens.checked_mul(
            half.checked_add(1)
                .ok_or_else(|| "causal GQA triangular token count overflows usize".to_string())?,
        )
    }
    .ok_or_else(|| "causal GQA triangular token count overflows usize".to_string())
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
    runtime: &mut RuntimeExecutionContext,
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
    let mut output = allocate_f32(
        output_elements,
        allocated_elements,
        runtime,
        "embedding output",
    )?;
    for token_index in 0..token_values.len() {
        let token = token_values.get_usize(token_index)?;
        if token >= vocab_size {
            return Err(format!(
                "embedding token position {token_index} is outside vocabulary size {vocab_size}"
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
    runtime: &mut RuntimeExecutionContext,
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
    let mut output = allocate_f32(
        output_elements,
        allocated_elements,
        runtime,
        "linear output",
    )?;
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

fn rotary_f32(
    values: &HostTensor,
    positions: &HostTensor,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    base: f32,
    pairing: RotaryPairing,
    output_spec: &TensorSpec,
    allocated_elements: &mut usize,
    runtime: &mut RuntimeExecutionContext,
) -> Result<HostTensor, String> {
    let (values_shape, values_layout, values_data) = values.f32_parts()?;
    if values_shape.len() < 2 {
        return Err("rotary values input must have rank at least 2".into());
    }
    if !matches!(
        values_layout,
        TensorLayout::RowMajor | TensorLayout::TokensHidden | TensorLayout::PackedRagged
    ) {
        return Err("rotary values input layout is unsupported".into());
    }
    let hidden = heads
        .checked_mul(head_dim)
        .ok_or_else(|| "rotary heads * head_dim overflows usize".to_string())?;
    if hidden == 0 || values_shape.last().copied() != Some(hidden) {
        return Err("rotary values feature width does not match heads * head_dim".into());
    }
    if rotary_dim == 0 || rotary_dim % 2 != 0 || rotary_dim > head_dim {
        return Err("rotary_dim must be even, nonzero, and no greater than head_dim".into());
    }
    if !base.is_finite() || base <= 0.0 {
        runtime.numerical_failed = true;
        return Err("rotary base must be finite and positive".into());
    }
    require_f32_output_spec(output_spec, values_shape, values_layout, "rotary")?;

    let (positions_shape, positions_layout, position_values) = positions.token_parts()?;
    if positions_layout != &TensorLayout::RowMajor {
        return Err("rotary positions input requires RowMajor layout".into());
    }
    if positions_shape != &values_shape[..values_shape.len() - 1] {
        return Err(
            "rotary positions shape must match values shape without the final feature axis".into(),
        );
    }
    let rows = leading_rows(values_shape, "rotary values")?;
    if rows.checked_mul(hidden) != Some(values_data.len()) {
        return Err("rotary values row count is invalid".into());
    }
    if position_values.len() != rows {
        return Err("rotary positions payload length does not match values rows".into());
    }
    for row in 0..rows {
        position_values.get_position_f32(row).map_err(|error| {
            runtime.unsupported_failure_reason = Some("position_precision");
            error
        })?;
    }
    let mut output = allocate_f32(
        values_data.len(),
        allocated_elements,
        runtime,
        "rotary output",
    )?;
    let half = rotary_dim / 2;
    for row in 0..rows {
        let position = position_values.get_position_f32(row).map_err(|error| {
            runtime.unsupported_failure_reason = Some("position_precision");
            error
        })?;
        let row_start = row
            .checked_mul(hidden)
            .ok_or_else(|| "rotary row offset overflows usize".to_string())?;
        for head in 0..heads {
            let head_offset = head
                .checked_mul(head_dim)
                .and_then(|offset| row_start.checked_add(offset))
                .ok_or_else(|| "rotary head offset overflows usize".to_string())?;
            for index in 0..half {
                let (a, b) = match pairing {
                    RotaryPairing::SplitHalf => {
                        let b = half
                            .checked_add(index)
                            .ok_or_else(|| "rotary split-half index overflows usize".to_string())?;
                        (index, b)
                    }
                    RotaryPairing::Interleaved => {
                        let a = index.checked_mul(2).ok_or_else(|| {
                            "rotary interleaved index overflows usize".to_string()
                        })?;
                        let b = a.checked_add(1).ok_or_else(|| {
                            "rotary interleaved index overflows usize".to_string()
                        })?;
                        (a, b)
                    }
                };
                let exponent = (2.0_f32 * index as f32) / rotary_dim as f32;
                let frequency = base.powf(exponent);
                let theta = position / frequency;
                if !exponent.is_finite()
                    || !frequency.is_finite()
                    || frequency <= 0.0
                    || !theta.is_finite()
                {
                    runtime.numerical_failed = true;
                    return Err("rotary angle intermediate is non-finite".into());
                }
                let (sin_theta, cos_theta) = theta.sin_cos();
                if !sin_theta.is_finite() || !cos_theta.is_finite() {
                    runtime.numerical_failed = true;
                    return Err("rotary sine/cosine intermediate is non-finite".into());
                }
                let a_offset = head_offset
                    .checked_add(a)
                    .ok_or_else(|| "rotary first pair offset overflows usize".to_string())?;
                let b_offset = head_offset
                    .checked_add(b)
                    .ok_or_else(|| "rotary second pair offset overflows usize".to_string())?;
                let x_a = values_data[a_offset];
                let x_b = values_data[b_offset];
                let y_a = x_a * cos_theta - x_b * sin_theta;
                let y_b = x_a * sin_theta + x_b * cos_theta;
                if !y_a.is_finite() || !y_b.is_finite() {
                    runtime.numerical_failed = true;
                    return Err("rotary output is non-finite".into());
                }
                output[a_offset] = y_a;
                output[b_offset] = y_b;
            }
            let suffix_start = head_offset
                .checked_add(rotary_dim)
                .ok_or_else(|| "rotary suffix offset overflows usize".to_string())?;
            let suffix_end = head_offset
                .checked_add(head_dim)
                .ok_or_else(|| "rotary suffix end offset overflows usize".to_string())?;
            output[suffix_start..suffix_end]
                .copy_from_slice(&values_data[suffix_start..suffix_end]);
        }
    }
    HostTensor::f32(values_shape.to_vec(), output_spec.layout.clone(), output)
}

fn causal_gqa_attention_f32(
    query: &HostTensor,
    key: &HostTensor,
    value: &HostTensor,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    output_spec: &TensorSpec,
    allocated_elements: &mut usize,
    runtime: &mut RuntimeExecutionContext,
) -> Result<HostTensor, String> {
    let (query_shape, query_layout, query_data) = query.f32_parts()?;
    let (key_shape, key_layout, key_data) = key.f32_parts()?;
    let (value_shape, value_layout, value_data) = value.f32_parts()?;
    if !matches!(
        query_layout,
        TensorLayout::RowMajor | TensorLayout::TokensHidden
    ) || key_layout != query_layout
        || value_layout != query_layout
    {
        return Err(
            "causal GQA CPU reference requires matching RowMajor or TokensHidden layouts".into(),
        );
    }
    if query_shape.len() < 2
        || key_shape.len() != query_shape.len()
        || value_shape.len() != query_shape.len()
    {
        return Err("causal GQA tensors must have equal rank at least 2".into());
    }
    if key_shape[..query_shape.len() - 1] != query_shape[..query_shape.len() - 1]
        || value_shape[..query_shape.len() - 1] != query_shape[..query_shape.len() - 1]
    {
        return Err("causal GQA tensors must have matching leading and token shapes".into());
    }
    let query_width = q_heads
        .checked_mul(head_dim)
        .ok_or_else(|| "causal GQA query width overflows usize".to_string())?;
    let key_width = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| "causal GQA key width overflows usize".to_string())?;
    let value_width = kv_heads
        .checked_mul(value_dim)
        .ok_or_else(|| "causal GQA value width overflows usize".to_string())?;
    let context_width = q_heads
        .checked_mul(value_dim)
        .ok_or_else(|| "causal GQA context width overflows usize".to_string())?;
    if query_shape.last().copied() != Some(query_width)
        || key_shape.last().copied() != Some(key_width)
        || value_shape.last().copied() != Some(value_width)
    {
        return Err("causal GQA input feature widths do not match geometry".into());
    }
    if q_heads == 0 || kv_heads == 0 || head_dim == 0 || value_dim == 0 {
        return Err("causal GQA geometry must be nonzero".into());
    }
    if q_heads % kv_heads != 0 {
        return Err("causal GQA q_heads must divide evenly by kv_heads".into());
    }
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        runtime.numerical_failed = true;
        return Err("causal GQA softmax scale must be finite and positive".into());
    }
    let rank = query_shape.len();
    let tokens = query_shape[rank - 2];
    let sequences = causal_sequence_count(query_shape)?;
    let expected_output = {
        let mut shape = query_shape.to_vec();
        *shape
            .last_mut()
            .ok_or_else(|| "causal GQA output has no feature axis".to_string())? = context_width;
        shape
    };
    require_f32_output_spec(output_spec, &expected_output, query_layout, "causal GQA")?;
    let query_sequence_elements = tokens
        .checked_mul(query_width)
        .ok_or_else(|| "causal GQA query sequence size overflows usize".to_string())?;
    let key_sequence_elements = tokens
        .checked_mul(key_width)
        .ok_or_else(|| "causal GQA key sequence size overflows usize".to_string())?;
    let value_sequence_elements = tokens
        .checked_mul(value_width)
        .ok_or_else(|| "causal GQA value sequence size overflows usize".to_string())?;
    let output_sequence_elements = tokens
        .checked_mul(context_width)
        .ok_or_else(|| "causal GQA output sequence size overflows usize".to_string())?;
    if sequences.checked_mul(query_sequence_elements) != Some(query_data.len())
        || sequences.checked_mul(key_sequence_elements) != Some(key_data.len())
        || sequences.checked_mul(value_sequence_elements) != Some(value_data.len())
    {
        return Err("causal GQA payload lengths do not match sequence shapes".into());
    }
    let mut output = allocate_f32(
        sequences
            .checked_mul(output_sequence_elements)
            .ok_or_else(|| "causal GQA output element count overflows usize".to_string())?,
        allocated_elements,
        runtime,
        "causal GQA output",
    )?;
    let group_size = q_heads
        .checked_div(kv_heads)
        .ok_or_else(|| "causal GQA q_heads must divide evenly by kv_heads".to_string())?;
    for sequence in 0..sequences {
        let query_base = sequence
            .checked_mul(query_sequence_elements)
            .ok_or_else(|| "causal GQA query sequence offset overflows usize".to_string())?;
        let key_base = sequence
            .checked_mul(key_sequence_elements)
            .ok_or_else(|| "causal GQA key sequence offset overflows usize".to_string())?;
        let value_base = sequence
            .checked_mul(value_sequence_elements)
            .ok_or_else(|| "causal GQA value sequence offset overflows usize".to_string())?;
        let output_base = sequence
            .checked_mul(output_sequence_elements)
            .ok_or_else(|| "causal GQA output sequence offset overflows usize".to_string())?;
        for token in 0..tokens {
            let query_token_base =
                query_base
                    .checked_add(token.checked_mul(query_width).ok_or_else(|| {
                        "causal GQA query token offset overflows usize".to_string()
                    })?)
                    .ok_or_else(|| "causal GQA query token offset overflows usize".to_string())?;
            let output_token_base =
                output_base
                    .checked_add(token.checked_mul(context_width).ok_or_else(|| {
                        "causal GQA output token offset overflows usize".to_string()
                    })?)
                    .ok_or_else(|| "causal GQA output token offset overflows usize".to_string())?;
            for query_head in 0..q_heads {
                let kv_head = query_head / group_size;
                let query_start = query_token_base
                    .checked_add(query_head.checked_mul(head_dim).ok_or_else(|| {
                        "causal GQA query head offset overflows usize".to_string()
                    })?)
                    .ok_or_else(|| "causal GQA query head offset overflows usize".to_string())?;
                let query_slice = f32_slice(query_data, query_start, head_dim, "causal GQA query")?;
                let output_start = output_token_base
                    .checked_add(query_head.checked_mul(value_dim).ok_or_else(|| {
                        "causal GQA context head offset overflows usize".to_string()
                    })?)
                    .ok_or_else(|| "causal GQA context head offset overflows usize".to_string())?;
                let output_slice =
                    f32_slice_mut(&mut output, output_start, value_dim, "causal GQA context")?;
                let mut maximum = f32::NEG_INFINITY;
                for key_token in 0..=token {
                    let key_start =
                        causal_gqa_key_offset(key_base, key_token, key_width, kv_head, head_dim)?;
                    let key_slice = f32_slice(key_data, key_start, head_dim, "causal GQA key")?;
                    let score = causal_gqa_score(query_slice, key_slice, softmax_scale).map_err(
                        |error| {
                            runtime.numerical_failed = true;
                            error
                        },
                    )?;
                    if score > maximum {
                        maximum = score;
                    }
                }
                let mut denominator = 0.0_f32;
                for key_token in 0..=token {
                    let key_start =
                        causal_gqa_key_offset(key_base, key_token, key_width, kv_head, head_dim)?;
                    let key_slice = f32_slice(key_data, key_start, head_dim, "causal GQA key")?;
                    let score = causal_gqa_score(query_slice, key_slice, softmax_scale).map_err(
                        |error| {
                            runtime.numerical_failed = true;
                            error
                        },
                    )?;
                    let exponent = (score - maximum).exp();
                    if !exponent.is_finite() {
                        runtime.numerical_failed = true;
                        return Err("causal GQA softmax exponent is non-finite".into());
                    }
                    denominator += exponent;
                    if !denominator.is_finite() {
                        runtime.numerical_failed = true;
                        return Err("causal GQA softmax denominator is non-finite".into());
                    }
                }
                if !denominator.is_finite() || denominator <= 0.0 {
                    runtime.numerical_failed = true;
                    return Err("causal GQA softmax denominator is not positive".into());
                }
                for key_token in 0..=token {
                    let key_start =
                        causal_gqa_key_offset(key_base, key_token, key_width, kv_head, head_dim)?;
                    let value_start = value_base
                        .checked_add(key_token.checked_mul(value_width).ok_or_else(|| {
                            "causal GQA value token offset overflows usize".to_string()
                        })?)
                        .and_then(|offset| offset.checked_add(kv_head.checked_mul(value_dim)?))
                        .ok_or_else(|| "causal GQA value offset overflows usize".to_string())?;
                    let key_slice = f32_slice(key_data, key_start, head_dim, "causal GQA key")?;
                    let value_slice =
                        f32_slice(value_data, value_start, value_dim, "causal GQA value")?;
                    let score = causal_gqa_score(query_slice, key_slice, softmax_scale).map_err(
                        |error| {
                            runtime.numerical_failed = true;
                            error
                        },
                    )?;
                    let exponent = (score - maximum).exp();
                    let weight = exponent / denominator;
                    if !exponent.is_finite() || !weight.is_finite() {
                        runtime.numerical_failed = true;
                        return Err("causal GQA softmax weight is non-finite".into());
                    }
                    for value_index in 0..value_dim {
                        let product = weight * value_slice[value_index];
                        let updated = output_slice[value_index] + product;
                        if !product.is_finite() || !updated.is_finite() {
                            runtime.numerical_failed = true;
                            return Err("causal GQA context accumulation is non-finite".into());
                        }
                        output_slice[value_index] = updated;
                    }
                }
            }
        }
    }
    HostTensor::f32(expected_output, output_spec.layout.clone(), output)
}

fn causal_gqa_key_offset(
    key_base: usize,
    token: usize,
    key_width: usize,
    kv_head: usize,
    head_dim: usize,
) -> Result<usize, String> {
    let token_offset = token
        .checked_mul(key_width)
        .ok_or_else(|| "causal GQA key token offset overflows usize")?;
    let head_offset = kv_head
        .checked_mul(head_dim)
        .ok_or_else(|| "causal GQA key head offset overflows usize")?;
    let sequence_offset = key_base
        .checked_add(token_offset)
        .ok_or_else(|| "causal GQA key offset overflows usize")?;
    sequence_offset
        .checked_add(head_offset)
        .ok_or_else(|| "causal GQA key offset overflows usize".into())
}

fn causal_gqa_score(query: &[f32], key: &[f32], softmax_scale: f32) -> Result<f32, String> {
    if query.len() != key.len() {
        return Err("causal GQA query/key head widths do not match".into());
    }
    let mut dot = 0.0_f32;
    for (&query_value, &key_value) in query.iter().zip(key) {
        let product = query_value * key_value;
        if !product.is_finite() {
            return Err("causal GQA dot product is non-finite".into());
        }
        dot += product;
        if !dot.is_finite() {
            return Err("causal GQA dot accumulation is non-finite".into());
        }
    }
    let score = dot * softmax_scale;
    if !score.is_finite() {
        return Err("causal GQA score is non-finite".into());
    }
    Ok(score)
}

fn f32_slice<'a>(
    data: &'a [f32],
    start: usize,
    len: usize,
    label: &str,
) -> Result<&'a [f32], String> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| format!("{label} slice end overflows usize"))?;
    data.get(start..end)
        .ok_or_else(|| format!("{label} slice is outside payload"))
}

fn f32_slice_mut<'a>(
    data: &'a mut [f32],
    start: usize,
    len: usize,
    label: &str,
) -> Result<&'a mut [f32], String> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| format!("{label} mutable slice end overflows usize"))?;
    data.get_mut(start..end)
        .ok_or_else(|| format!("{label} mutable slice is outside payload"))
}

fn rms_norm_f32(
    input: &HostTensor,
    scale: Option<&HostTensor>,
    bias: Option<&HostTensor>,
    kind: NormalizationKind,
    affine: NormalizationAffine,
    axis: NormalizationAxis,
    epsilon: f32,
    output_spec: &TensorSpec,
    allocated_elements: &mut usize,
    runtime: &mut RuntimeExecutionContext,
) -> Result<HostTensor, String> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        runtime.numerical_failed = true;
        return Err("RMS normalization epsilon must be finite and positive".into());
    }
    let (input_shape, input_layout, input_data) = input.f32_parts()?;
    if !matches!(
        input_layout,
        TensorLayout::RowMajor | TensorLayout::TokensHidden
    ) {
        return Err("RMS normalization input layout is unsupported".into());
    }
    require_f32_output_spec(output_spec, input_shape, input_layout, "RMS normalization")?;
    let (_, group_width) = rms_norm_group_geometry(input_shape, axis)?;
    let units = input_data
        .len()
        .checked_div(group_width)
        .ok_or_else(|| "RMS normalization unit count overflows usize".to_string())?;
    if units
        .checked_mul(group_width)
        .filter(|elements| *elements == input_data.len())
        .is_none()
    {
        return Err("RMS normalization input element count is not divisible by group_width".into());
    }

    let scale_data = if let Some(scale) = scale {
        let (scale_shape, scale_layout, scale_data) = scale.f32_parts()?;
        if scale_layout != &TensorLayout::RowMajor || scale_shape != [group_width] {
            return Err(format!(
                "normalization scale must have RowMajor shape [{group_width}]"
            ));
        }
        Some(scale_data)
    } else {
        None
    };
    let bias_data = match (kind, affine, bias) {
        (
            NormalizationKind::L2,
            NormalizationAffine::None | NormalizationAffine::FixedScale(_),
            None,
        ) => None,
        (_, NormalizationAffine::ScaleAndBias, Some(bias)) => {
            let (bias_shape, bias_layout, bias_data) = bias.f32_parts()?;
            if bias_layout != &TensorLayout::RowMajor {
                return Err("RMS normalization bias requires RowMajor layout".into());
            }
            if bias_shape.len() != 1 || bias_shape[0] != group_width {
                return Err(format!(
                    "RMS normalization bias must have shape [{group_width}]"
                ));
            }
            Some(bias_data)
        }
        (_, NormalizationAffine::ScaleAndBias, None) => {
            return Err("RMS normalization ScaleAndBias requires a bias weight".into());
        }
        (_, NormalizationAffine::Scale | NormalizationAffine::UnitOffsetScale, None) => None,
        (_, NormalizationAffine::Scale | NormalizationAffine::UnitOffsetScale, Some(_)) => {
            return Err("RMS normalization affine contract has an unexpected bias weight".into());
        }
        _ => return Err("normalization kind and affine combination is invalid".into()),
    };

    let mut output = allocate_f32(
        input_data.len(),
        allocated_elements,
        runtime,
        "RMS normalization output",
    )?;
    let bias_data = bias_data.unwrap_or(&[]);
    let group_width_as_f32 = group_width as f32;
    for unit in 0..units {
        let start = unit
            .checked_mul(group_width)
            .ok_or_else(|| "RMS normalization unit offset overflows usize".to_string())?;
        let mut sum_of_squares = 0.0_f32;
        for index in 0..group_width {
            let value = input_data[start + index];
            let square = value * value;
            if !square.is_finite() {
                runtime.numerical_failed = true;
                return Err("RMS normalization square is non-finite".into());
            }
            sum_of_squares += square;
            if !sum_of_squares.is_finite() {
                runtime.numerical_failed = true;
                return Err("RMS normalization sum of squares is non-finite".into());
            }
        }
        let mean_plus_epsilon = if kind == NormalizationKind::L2 {
            sum_of_squares + epsilon
        } else {
            sum_of_squares / group_width_as_f32 + epsilon
        };
        if !mean_plus_epsilon.is_finite() || mean_plus_epsilon <= 0.0 {
            runtime.numerical_failed = true;
            return Err("RMS normalization mean plus epsilon is invalid".into());
        }
        let root_mean_square = mean_plus_epsilon.sqrt();
        if !root_mean_square.is_finite() || root_mean_square <= 0.0 {
            runtime.numerical_failed = true;
            return Err("RMS normalization root mean square is invalid".into());
        }
        let inverse_rms = 1.0_f32 / root_mean_square;
        if !inverse_rms.is_finite() || inverse_rms <= 0.0 {
            runtime.numerical_failed = true;
            return Err("RMS normalization inverse root mean square is invalid".into());
        }
        for index in 0..group_width {
            let normalized = input_data[start + index] * inverse_rms;
            if !normalized.is_finite() {
                runtime.numerical_failed = true;
                return Err("RMS normalization normalized value is non-finite".into());
            }
            let affine_value = match affine {
                NormalizationAffine::None => normalized,
                NormalizationAffine::FixedScale(scale) => normalized * scale.get(),
                NormalizationAffine::Scale => normalized * scale_data.unwrap_or(&[])[index],
                NormalizationAffine::UnitOffsetScale => {
                    normalized * (1.0_f32 + scale_data.unwrap_or(&[])[index])
                }
                NormalizationAffine::ScaleAndBias => {
                    normalized * scale_data.unwrap_or(&[])[index] + bias_data[index]
                }
            };
            if !affine_value.is_finite() {
                runtime.numerical_failed = true;
                return Err("RMS normalization affine output is non-finite".into());
            }
            output[start + index] = affine_value;
        }
    }
    HostTensor::f32(input_shape.to_vec(), output_spec.layout.clone(), output)
}

fn causal_depthwise_conv1d_f32(
    input: &HostTensor,
    kernel: &HostTensor,
    channels: usize,
    kernel_size: usize,
    output_spec: &TensorSpec,
    allocated: &mut usize,
    runtime: &mut RuntimeExecutionContext,
) -> Result<HostTensor, String> {
    let (shape, layout, data) = input.f32_parts()?;
    let (kernel_shape, kernel_layout, weights) = kernel.f32_parts()?;
    if !matches!(layout, TensorLayout::RowMajor | TensorLayout::TokensHidden)
        || kernel_layout != &TensorLayout::RowMajor
    {
        return Err("causal depthwise conv layout unsupported".into());
    }
    if shape.len() < 2
        || shape.last().copied() != Some(channels)
        || kernel_shape != [channels, 1, kernel_size]
    {
        return Err("causal depthwise conv runtime shape mismatch".into());
    }
    require_f32_output_spec(output_spec, shape, layout, "causal depthwise conv")?;
    let tokens = shape[shape.len() - 2];
    let sequences = causal_sequence_count(shape)?;
    let mut output = allocate_f32(
        data.len(),
        allocated,
        runtime,
        "causal depthwise conv output",
    )?;
    for sequence in 0..sequences {
        let base = sequence
            .checked_mul(tokens)
            .and_then(|v| v.checked_mul(channels))
            .ok_or_else(|| "conv sequence offset overflow".to_string())?;
        for token in 0..tokens {
            for channel in 0..channels {
                let mut sum = 0.0_f32;
                for j in 0..kernel_size {
                    let lag = kernel_size - 1 - j;
                    if token >= lag {
                        let x = data[base + (token - lag) * channels + channel];
                        let w = weights[channel * kernel_size + j];
                        let product = x * w;
                        if !product.is_finite() {
                            runtime.numerical_failed = true;
                            return Err("conv product non-finite".into());
                        }
                        sum += product;
                        if !sum.is_finite() {
                            runtime.numerical_failed = true;
                            return Err("conv accumulation non-finite".into());
                        }
                    }
                }
                output[base + token * channels + channel] = sum;
            }
        }
    }
    HostTensor::f32(shape.to_vec(), output_spec.layout.clone(), output)
}

fn stable_softplus(value: f32) -> Result<f32, String> {
    if !value.is_finite() {
        return Err("softplus input non-finite".into());
    }
    let result = if value > 0.0 {
        value + (-value).exp().ln_1p()
    } else {
        value.exp().ln_1p()
    };
    if result.is_finite() {
        Ok(result)
    } else {
        Err("softplus output non-finite".into())
    }
}

#[allow(clippy::too_many_arguments)]
fn gated_decay_parameters_f32(
    decay: &HostTensor,
    update: &HostTensor,
    log_rate: &HostTensor,
    time_bias: &HostTensor,
    channels: usize,
    log_spec: &TensorSpec,
    update_spec: &TensorSpec,
    allocated: &mut usize,
    runtime: &mut RuntimeExecutionContext,
) -> Result<(HostTensor, HostTensor), String> {
    let (shape, layout, decay_data) = decay.f32_parts()?;
    let (update_shape, update_layout, update_data) = update.f32_parts()?;
    let (_, rate_layout, rates) = log_rate.f32_parts()?;
    let (_, bias_layout, biases) = time_bias.f32_parts()?;
    if shape != update_shape
        || layout != update_layout
        || shape.last().copied() != Some(channels)
        || rate_layout != &TensorLayout::RowMajor
        || bias_layout != &TensorLayout::RowMajor
        || rates.len() != channels
        || biases.len() != channels
    {
        return Err("gated decay runtime contract mismatch".into());
    }
    require_f32_output_spec(log_spec, shape, layout, "gated log decay")?;
    require_f32_output_spec(update_spec, shape, layout, "gated update rate")?;
    let mut logs = allocate_f32(decay_data.len(), allocated, runtime, "gated log decay")?;
    let mut updates = allocate_f32(decay_data.len(), allocated, runtime, "gated update rate")?;
    for index in 0..decay_data.len() {
        let channel = index % channels;
        let control = decay_data[index] + biases[channel];
        let rate = rates[channel].exp();
        let soft = stable_softplus(control).map_err(|e| {
            runtime.numerical_failed = true;
            e
        })?;
        let log = -rate * soft;
        let beta = stable_sigmoid(update_data[index]).map_err(|e| {
            runtime.numerical_failed = true;
            e
        })?;
        if !control.is_finite() || !rate.is_finite() || !log.is_finite() || !beta.is_finite() {
            runtime.numerical_failed = true;
            return Err("gated decay intermediate non-finite".into());
        }
        logs[index] = log;
        updates[index] = beta;
    }
    Ok((
        HostTensor::f32(shape.to_vec(), log_spec.layout.clone(), logs)?,
        HostTensor::f32(shape.to_vec(), update_spec.layout.clone(), updates)?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_rule_scan_f32(
    query: &HostTensor,
    key: &HostTensor,
    value: &HostTensor,
    log_decay: &HostTensor,
    update_rate: &HostTensor,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    initial_state: Option<&[f32]>,
    output_spec: &TensorSpec,
    allocated: &mut usize,
    runtime: &mut RuntimeExecutionContext,
) -> Result<(HostTensor, Vec<f32>), String> {
    let (shape, layout, q) = query.f32_parts()?;
    let (_, kl, k) = key.f32_parts()?;
    let (_, vl, v) = value.f32_parts()?;
    let (_, dl, d) = log_decay.f32_parts()?;
    let (_, ul, beta) = update_rate.f32_parts()?;
    if kl != layout || vl != layout || dl != layout || ul != layout {
        return Err("scan layouts mismatch".into());
    }
    let tokens = shape[shape.len() - 2];
    let sequences = causal_sequence_count(shape)?;
    let kw = key_heads
        .checked_mul(key_dim)
        .ok_or_else(|| "scan key width overflow".to_string())?;
    let vw = value_heads
        .checked_mul(value_dim)
        .ok_or_else(|| "scan value width overflow".to_string())?;
    let state_len = value_heads
        .checked_mul(key_dim)
        .and_then(|x| x.checked_mul(value_dim))
        .ok_or_else(|| "scan state size overflow".to_string())?;
    let mut state = allocate_f32(state_len, allocated, runtime, "gated delta-rule state")?;
    if let Some(initial) = initial_state {
        if initial.len() != state_len {
            return Err("scan initial state size mismatch".into());
        }
        state.copy_from_slice(initial);
    }
    let mut output = allocate_f32(
        sequences
            .checked_mul(tokens)
            .and_then(|x| x.checked_mul(vw))
            .ok_or_else(|| "scan output size overflow".to_string())?,
        allocated,
        runtime,
        "gated delta-rule output",
    )?;
    require_f32_output_spec(
        output_spec,
        &replaced_last_axis(shape, vw)?,
        layout,
        "gated delta-rule scan",
    )?;
    let group = value_heads / key_heads;
    for seq in 0..sequences {
        if seq > 0 {
            state.fill(0.0);
        }
        let qbase = seq * tokens * kw;
        let vbase = seq * tokens * vw;
        let hbase = seq * tokens * value_heads;
        for t in 0..tokens {
            for hv in 0..value_heads {
                let kh = hv / group;
                let decay = d[hbase + t * value_heads + hv].exp();
                let b = beta[hbase + t * value_heads + hv];
                if !decay.is_finite() || !b.is_finite() {
                    runtime.numerical_failed = true;
                    return Err("scan control non-finite".into());
                }
                for vd in 0..value_dim {
                    let mut pred = 0.0;
                    for kd in 0..key_dim {
                        let si = (hv * key_dim + kd) * value_dim + vd;
                        state[si] *= decay;
                        pred += state[si] * k[qbase + t * kw + kh * key_dim + kd];
                    }
                    let residual = (v[vbase + t * vw + hv * value_dim + vd] - pred) * b;
                    let mut context = 0.0;
                    for kd in 0..key_dim {
                        let si = (hv * key_dim + kd) * value_dim + vd;
                        state[si] += k[qbase + t * kw + kh * key_dim + kd] * residual;
                        context += state[si] * q[qbase + t * kw + kh * key_dim + kd];
                    }
                    if !pred.is_finite() || !residual.is_finite() || !context.is_finite() {
                        runtime.numerical_failed = true;
                        return Err("scan intermediate non-finite".into());
                    }
                    output[vbase + t * vw + hv * value_dim + vd] = context;
                }
            }
        }
    }
    Ok((
        HostTensor::f32(
            replaced_last_axis(shape, vw)?,
            output_spec.layout.clone(),
            output,
        )?,
        state,
    ))
}

fn grouped_last_split_output_f32(
    input: &HostTensor,
    groups: usize,
    segment_total: usize,
    segment_offset: usize,
    segment_width: usize,
    output_spec: &TensorSpec,
    allocated_elements: &mut usize,
    runtime: &mut RuntimeExecutionContext,
) -> Result<HostTensor, String> {
    let (input_shape, input_layout, input_data) = input.f32_parts()?;
    if !matches!(
        input_layout,
        TensorLayout::RowMajor | TensorLayout::TokensHidden
    ) {
        return Err("grouped last split input layout is unsupported".into());
    }
    if groups == 0 || segment_total == 0 || segment_width == 0 {
        return Err("grouped last split geometry must be nonzero".into());
    }
    if segment_offset
        .checked_add(segment_width)
        .filter(|end| *end <= segment_total)
        .is_none()
    {
        return Err("grouped last split segment lies outside each input group".into());
    }
    let input_width = groups
        .checked_mul(segment_total)
        .ok_or_else(|| "grouped last split input width overflows usize".to_string())?;
    if input_shape.last().copied() != Some(input_width) {
        return Err(format!(
            "grouped last split input final width must be {input_width}"
        ));
    }
    let output_width = groups
        .checked_mul(segment_width)
        .ok_or_else(|| "grouped last split output width overflows usize".to_string())?;
    let expected_output_shape = replaced_last_axis(input_shape, output_width)?;
    require_f32_output_spec(
        output_spec,
        &expected_output_shape,
        input_layout,
        "grouped last split",
    )?;
    let rows = input_data
        .len()
        .checked_div(input_width)
        .ok_or_else(|| "grouped last split row count overflows usize".to_string())?;
    if rows.checked_mul(input_width) != Some(input_data.len()) {
        return Err("grouped last split payload is not divisible by input width".into());
    }
    let output_elements = rows
        .checked_mul(output_width)
        .ok_or_else(|| "grouped last split output element count overflows usize".to_string())?;
    let mut output = allocate_f32(
        output_elements,
        allocated_elements,
        runtime,
        "grouped last split output",
    )?;
    for row in 0..rows {
        let input_row = row
            .checked_mul(input_width)
            .ok_or_else(|| "grouped last split input row offset overflows usize".to_string())?;
        let output_row = row
            .checked_mul(output_width)
            .ok_or_else(|| "grouped last split output row offset overflows usize".to_string())?;
        for group in 0..groups {
            let source_start = input_row
                .checked_add(group.checked_mul(segment_total).ok_or_else(|| {
                    "grouped last split input group offset overflows usize".to_string()
                })?)
                .and_then(|offset| offset.checked_add(segment_offset))
                .ok_or_else(|| {
                    "grouped last split input segment offset overflows usize".to_string()
                })?;
            let destination_start = output_row
                .checked_add(group.checked_mul(segment_width).ok_or_else(|| {
                    "grouped last split output group offset overflows usize".to_string()
                })?)
                .ok_or_else(|| {
                    "grouped last split output segment offset overflows usize".to_string()
                })?;
            let source = f32_slice(
                input_data,
                source_start,
                segment_width,
                "grouped last split input segment",
            )?;
            let destination = f32_slice_mut(
                &mut output,
                destination_start,
                segment_width,
                "grouped last split output segment",
            )?;
            destination.copy_from_slice(source);
        }
    }
    HostTensor::f32(expected_output_shape, output_spec.layout.clone(), output)
}

fn gated_multiply_f32(
    value_input: &HostTensor,
    gate: &HostTensor,
    activation: &ActivationKind,
    output_spec: &TensorSpec,
    allocated_elements: &mut usize,
    runtime: &mut RuntimeExecutionContext,
) -> Result<HostTensor, String> {
    if activation != &ActivationKind::Sigmoid {
        runtime.unsupported_failure_reason = Some("activation");
        return Err("gated multiply requires Sigmoid activation".into());
    }
    let (value_shape, value_layout, value_data) = value_input.f32_parts()?;
    let (gate_shape, gate_layout, gate_data) = gate.f32_parts()?;
    if value_shape != gate_shape || value_layout != gate_layout {
        return Err("gated multiply value and gate must have identical shape and layout".into());
    }
    if !matches!(
        value_layout,
        TensorLayout::RowMajor | TensorLayout::TokensHidden
    ) {
        return Err("gated multiply input layout is unsupported".into());
    }
    require_f32_output_spec(output_spec, value_shape, value_layout, "gated multiply")?;
    if value_data.len() != gate_data.len() {
        return Err("gated multiply input payload lengths must match".into());
    }
    let mut output = allocate_f32(
        value_data.len(),
        allocated_elements,
        runtime,
        "gated multiply output",
    )?;
    for index in 0..output.len() {
        if !value_data[index].is_finite() || !gate_data[index].is_finite() {
            runtime.numerical_failed = true;
            return Err("gated multiply input is non-finite".into());
        }
        let activated = stable_sigmoid(gate_data[index]).map_err(|error| {
            runtime.numerical_failed = true;
            error
        })?;
        let result = value_data[index] * activated;
        if !result.is_finite() {
            runtime.numerical_failed = true;
            return Err("gated multiply output is non-finite".into());
        }
        output[index] = result;
    }
    HostTensor::f32(value_shape.to_vec(), output_spec.layout.clone(), output)
}

fn activation_f32(
    input: &HostTensor,
    kind: &ActivationKind,
    output_spec: &TensorSpec,
    allocated_elements: &mut usize,
    runtime: &mut RuntimeExecutionContext,
) -> Result<HostTensor, String> {
    let (shape, layout, input_data) = input.f32_parts()?;
    require_f32_output_spec(output_spec, shape, layout, "activation")?;
    let mut output = allocate_f32(
        input_data.len(),
        allocated_elements,
        runtime,
        "activation output",
    )?;
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
    runtime: &mut RuntimeExecutionContext,
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
    let gate_values = linear_f32(
        input,
        gate,
        None,
        &intermediate_spec,
        allocated_elements,
        runtime,
    )?;
    let up_values = linear_f32(
        input,
        up,
        None,
        &intermediate_spec,
        allocated_elements,
        runtime,
    )?;
    let (_, _, gate_data) = gate_values.f32_parts()?;
    let (_, _, up_data) = up_values.f32_parts()?;
    let mut activated = allocate_f32(
        gate_data.len(),
        allocated_elements,
        runtime,
        "gated MLP activation",
    )?;
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
        runtime,
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
    runtime: &mut RuntimeExecutionContext,
) -> Result<HostTensor, String> {
    let (left_shape, left_layout, left_data) = left.f32_parts()?;
    let (right_shape, right_layout, right_data) = right.f32_parts()?;
    if left_shape != right_shape || left_layout != right_layout {
        return Err("residual inputs must have identical shape and layout".into());
    }
    require_f32_output_spec(output_spec, left_shape, left_layout, "residual")?;
    let mut output = allocate_f32(
        left_data.len(),
        allocated_elements,
        runtime,
        "residual output",
    )?;
    for index in 0..output.len() {
        output[index] = left_data[index] + right_data[index];
    }
    HostTensor::f32(left_shape.to_vec(), output_spec.layout.clone(), output)
}

fn activate(value: f32, kind: &ActivationKind) -> Result<f32, String> {
    match kind {
        ActivationKind::Sigmoid => stable_sigmoid(value),
        ActivationKind::Silu => Ok(value / (1.0 + (-value).exp())),
        ActivationKind::Gelu => {
            Err("GELU is unsupported until its exact or tanh contract is explicit".into())
        }
        ActivationKind::Relu => Ok(value.max(0.0)),
        ActivationKind::Custom(name) => Err(format!("unsupported custom activation {name}")),
    }
}

fn stable_sigmoid(value: f32) -> Result<f32, String> {
    if !value.is_finite() {
        return Err("sigmoid input is non-finite".into());
    }
    let result = if value >= 0.0 {
        1.0_f32 / (1.0_f32 + (-value).exp())
    } else {
        let exponent = value.exp();
        exponent / (1.0_f32 + exponent)
    };
    if !result.is_finite() {
        return Err("sigmoid result is non-finite".into());
    }
    Ok(result)
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

fn leading_rows(shape: &[usize], label: &str) -> Result<usize, String> {
    if shape.len() < 2 {
        return Err(format!("{label} must have rank at least 2"));
    }
    shape[..shape.len() - 1]
        .iter()
        .try_fold(1_usize, |rows, dimension| {
            rows.checked_mul(*dimension)
                .ok_or_else(|| format!("{label} leading row count overflows usize"))
        })
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

fn validate_finite_host_tensor(tensor: &HostTensor, label: &str) -> Result<(), String> {
    if let HostTensor::F32 { data, .. } = tensor {
        if data.iter().any(|value| !value.is_finite()) {
            return Err(format!("{label} contains a non-finite F32 value"));
        }
    }
    Ok(())
}

const MAX_CPU_REFERENCE_KEY_DIAGNOSTIC_IDS: usize = 8;

trait CpuReferenceBindingId: Ord {
    fn validate_binding_id(&self) -> Result<(), String>;
    fn binding_id_text(&self) -> &str;
}

impl CpuReferenceBindingId for ValueId {
    fn validate_binding_id(&self) -> Result<(), String> {
        self.validate()
    }

    fn binding_id_text(&self) -> &str {
        self.as_str()
    }
}

impl CpuReferenceBindingId for WeightId {
    fn validate_binding_id(&self) -> Result<(), String> {
        self.validate()
    }

    fn binding_id_text(&self) -> &str {
        self.as_str()
    }
}

#[derive(Debug)]
struct BindingKeyDiagnostic<'a> {
    count: usize,
    first: [Option<&'a str>; MAX_CPU_REFERENCE_KEY_DIAGNOSTIC_IDS],
}

impl<'a> BindingKeyDiagnostic<'a> {
    fn new() -> Self {
        Self {
            count: 0,
            first: [None; MAX_CPU_REFERENCE_KEY_DIAGNOSTIC_IDS],
        }
    }

    fn record(&mut self, id: &'a str) {
        if self.count < self.first.len() {
            self.first[self.count] = Some(id);
        }
        self.count += 1;
    }
}

impl fmt::Display for BindingKeyDiagnostic<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("[")?;
        for (index, id) in self.first.iter().flatten().enumerate() {
            if index != 0 {
                formatter.write_str(", ")?;
            }
            write!(formatter, "{id:?}")?;
        }
        formatter.write_str("]")
    }
}

fn validate_binding_map_admission<K, V>(
    map: &BTreeMap<K, V>,
    limit: usize,
    label: &str,
) -> Result<(), String>
where
    K: CpuReferenceBindingId,
{
    if map.len() > limit {
        return Err(format!(
            "{label} map entry count {} exceeds limit {limit}",
            map.len()
        ));
    }
    for (index, id) in map.keys().enumerate() {
        id.validate_binding_id().map_err(|message| {
            format!("{label} map key at sorted position {index} is invalid: {message}")
        })?;
    }
    Ok(())
}

fn validate_exact_keys<K, V, E>(
    map: &BTreeMap<K, V>,
    expected: &BTreeMap<K, E>,
    label: &str,
) -> Result<(), String>
where
    K: CpuReferenceBindingId,
{
    let mut missing = BindingKeyDiagnostic::new();
    for id in expected.keys() {
        if !map.contains_key(id) {
            missing.record(id.binding_id_text());
        }
    }
    let mut extra = BindingKeyDiagnostic::new();
    for id in map.keys() {
        if !expected.contains_key(id) {
            extra.record(id.binding_id_text());
        }
    }
    if missing.count != 0 || extra.count != 0 {
        return Err(format!(
            "{label} map mismatch: missing_count={} missing_first={missing} extra_count={} extra_first={extra}",
            missing.count, extra.count,
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
    runtime: &mut RuntimeExecutionContext,
    label: &str,
) -> Result<Vec<f32>, String> {
    let next_total = allocated_elements.checked_add(elements).ok_or_else(|| {
        runtime.allocation_failed = true;
        "CPU reference allocation count overflows usize".to_string()
    })?;
    if next_total > MAX_CPU_REFERENCE_TOTAL_ELEMENTS {
        runtime.allocation_failed = true;
        return Err(format!(
            "{label} would exceed CPU reference total allocation limit {MAX_CPU_REFERENCE_TOTAL_ELEMENTS}"
        ));
    }
    let mut output = Vec::new();
    output.try_reserve_exact(elements).map_err(|_| {
        runtime.allocation_failed = true;
        format!("{label} allocation failed")
    })?;
    output.resize(elements, 0.0);
    *allocated_elements = next_total;
    Ok(output)
}

fn node_error(node: &GraphNode, message: &str) -> String {
    format!("{}{message}", node_error_prefix(node))
}

fn node_error_prefix(node: &GraphNode) -> String {
    format!(
        "CPU reference node {} ({}) failed: ",
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
        GraphNodeKind::GroupedLastSplit { .. } => "GroupedLastSplit",
        GraphNodeKind::LastAxisSplit { .. } => "LastAxisSplit",
        GraphNodeKind::RotaryPosition { .. } => "RotaryPosition",
        GraphNodeKind::CausalGqaAttentionCore { .. } => "CausalGqaAttentionCore",
        GraphNodeKind::DenseAttention { .. } => "DenseAttention",
        GraphNodeKind::RecurrentAttention { .. } => "RecurrentAttention",
        GraphNodeKind::CausalDepthwiseConv1d { .. } => "CausalDepthwiseConv1d",
        GraphNodeKind::GatedDecayParameters { .. } => "GatedDecayParameters",
        GraphNodeKind::GatedDeltaRuleScan { .. } => "GatedDeltaRuleScan",
        GraphNodeKind::Activation { .. } => "Activation",
        GraphNodeKind::GatedMultiply { .. } => "GatedMultiply",
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
        ActivationKind, GraphNode, GraphNodeKind, NodeId, NormalizationAffine, NormalizationAxis,
        NormalizationKind, PositiveF32, StateId, WeightSpec,
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
    fn traced_success_records_node_id_kind_order_and_legacy_wrapper_matches() {
        let graph = simple_linear_graph();
        let inputs = map([(
            value_id("input"),
            f32(&[1, 1], TensorLayout::TokensHidden, &[3.0]),
        )]);
        let weights = map([(
            weight_id("weight"),
            f32(&[1, 1], TensorLayout::RowMajor, &[2.0]),
        )]);
        let traced = executor()
            .execute_traced(&graph, inputs.clone(), weights.clone())
            .unwrap();
        assert_eq!(traced.trace.completed_node_count, 1);
        assert!(!traced.trace.completed_nodes_truncated);
        assert_eq!(
            traced.trace.completed_nodes,
            vec![CpuReferenceNodeRef {
                id: node_id("linear"),
                kind: CpuReferenceNodeKind::Linear,
            }]
        );
        let legacy = executor().execute(&graph, inputs, weights).unwrap();
        assert_eq!(legacy.executed_node_ids, vec![node_id("linear")]);
        assert_eq!(legacy.outputs, traced.outputs);
    }

    #[test]
    fn rotary_split_half_matches_oracle_and_position_zero_is_identity() {
        let graph = rotary_graph(
            &[2, 8],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            NumericalFormat::U32,
            2,
            4,
            4,
            10_000.0,
            RotaryPairing::SplitHalf,
        );
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0,
        ];
        let inputs = map([
            (
                value_id("values"),
                f32(&[2, 8], TensorLayout::TokensHidden, &input),
            ),
            (
                value_id("positions"),
                HostTensor::u32(vec![2], TensorLayout::RowMajor, vec![0, 1]).unwrap(),
            ),
        ]);
        let run = executor()
            .execute_traced(&graph, inputs.clone(), BTreeMap::new())
            .unwrap();
        let expected = rotary_oracle(&input, &[0, 1], 2, 4, 4, 10_000.0, RotaryPairing::SplitHalf);
        assert_f32(&run.outputs[&value_id("out")], &expected, 1e-6);
        let (_, _, output) = run.outputs[&value_id("out")].f32_parts().unwrap();
        assert_eq!(&output[..8], &input[..8]);
        assert_eq!(run.trace.completed_node_count, 1);
        let legacy = executor().execute(&graph, inputs, BTreeMap::new()).unwrap();
        assert_eq!(legacy.outputs, run.outputs);
    }

    #[test]
    fn rotary_interleaved_matches_independent_oracle() {
        let graph = rotary_graph(
            &[1, 4],
            NumericalFormat::F32,
            TensorLayout::RowMajor,
            NumericalFormat::U64,
            1,
            4,
            4,
            10_000.0,
            RotaryPairing::Interleaved,
        );
        let input = [1.0, 2.0, 3.0, 4.0];
        let run = executor()
            .execute(
                &graph,
                map([
                    (
                        value_id("values"),
                        f32(&[1, 4], TensorLayout::RowMajor, &input),
                    ),
                    (
                        value_id("positions"),
                        HostTensor::u64(vec![1], TensorLayout::RowMajor, vec![1]).unwrap(),
                    ),
                ]),
                BTreeMap::new(),
            )
            .unwrap();
        let expected = rotary_oracle(&input, &[1], 1, 4, 4, 10_000.0, RotaryPairing::Interleaved);
        assert_f32(&run.outputs[&value_id("out")], &expected, 1e-6);
    }

    #[test]
    fn rotary_literal_goldens_fix_pairing_signs_and_head_frequency_reset() {
        const COS_THETA: f32 = 0.5403023;
        const SIN_THETA: f32 = 0.84147096;
        let input = [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
        let positions = HostTensor::u32(vec![1], TensorLayout::RowMajor, vec![1]).unwrap();

        let split = rotary_graph(
            &[1, 8],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            NumericalFormat::U32,
            2,
            4,
            4,
            1.0,
            RotaryPairing::SplitHalf,
        );
        let split_run = executor()
            .execute(
                &split,
                map([
                    (
                        value_id("values"),
                        f32(&[1, 8], TensorLayout::TokensHidden, &input),
                    ),
                    (value_id("positions"), positions.clone()),
                ]),
                BTreeMap::new(),
            )
            .unwrap();
        assert_f32(
            &split_run.outputs[&value_id("out")],
            &[
                COS_THETA,
                -SIN_THETA,
                SIN_THETA,
                COS_THETA,
                2.0 * COS_THETA,
                -2.0 * SIN_THETA,
                2.0 * SIN_THETA,
                2.0 * COS_THETA,
            ],
            1e-6,
        );

        let interleaved = rotary_graph(
            &[1, 8],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            NumericalFormat::U32,
            2,
            4,
            4,
            1.0,
            RotaryPairing::Interleaved,
        );
        let interleaved_run = executor()
            .execute(
                &interleaved,
                map([
                    (
                        value_id("values"),
                        f32(&[1, 8], TensorLayout::TokensHidden, &input),
                    ),
                    (value_id("positions"), positions),
                ]),
                BTreeMap::new(),
            )
            .unwrap();
        assert_f32(
            &interleaved_run.outputs[&value_id("out")],
            &[
                COS_THETA,
                SIN_THETA,
                -SIN_THETA,
                COS_THETA,
                2.0 * COS_THETA,
                2.0 * SIN_THETA,
                -2.0 * SIN_THETA,
                2.0 * COS_THETA,
            ],
            1e-6,
        );
    }

    #[test]
    fn rotary_partial_suffix_is_exact_and_rank3_positions_are_row_specific() {
        let graph = rotary_graph(
            &[1, 8],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            NumericalFormat::U32,
            2,
            4,
            2,
            10_000.0,
            RotaryPairing::SplitHalf,
        );
        let input = [1.0, 2.0, 30.0, 40.0, 5.0, 6.0, 70.0, 80.0];
        let run = executor()
            .execute(
                &graph,
                map([
                    (
                        value_id("values"),
                        f32(&[1, 8], TensorLayout::TokensHidden, &input),
                    ),
                    (
                        value_id("positions"),
                        HostTensor::u32(vec![1], TensorLayout::RowMajor, vec![1]).unwrap(),
                    ),
                ]),
                BTreeMap::new(),
            )
            .unwrap();
        let (_, _, output) = run.outputs[&value_id("out")].f32_parts().unwrap();
        assert_eq!(&output[2..4], &input[2..4]);
        assert_eq!(&output[6..8], &input[6..8]);

        let rank3 = rotary_graph(
            &[2, 2, 8],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            NumericalFormat::U64,
            2,
            4,
            4,
            10_000.0,
            RotaryPairing::Interleaved,
        );
        let rank3_input = (1..=32).map(|value| value as f32).collect::<Vec<_>>();
        let positions = vec![0, 1, 2, 3];
        let rank3_run = executor()
            .execute(
                &rank3,
                map([
                    (
                        value_id("values"),
                        f32(&[2, 2, 8], TensorLayout::TokensHidden, &rank3_input),
                    ),
                    (
                        value_id("positions"),
                        HostTensor::u64(vec![2, 2], TensorLayout::RowMajor, positions.clone())
                            .unwrap(),
                    ),
                ]),
                BTreeMap::new(),
            )
            .unwrap();
        let expected = rotary_oracle(
            &rank3_input,
            &positions,
            2,
            4,
            4,
            10_000.0,
            RotaryPairing::Interleaved,
        );
        assert_f32(&rank3_run.outputs[&value_id("out")], &expected, 1e-6);
    }

    #[test]
    fn rotary_packed_ragged_succeeds_and_bf16_fails_preflight() {
        let packed = rotary_graph(
            &[2, 8],
            NumericalFormat::F32,
            TensorLayout::PackedRagged,
            NumericalFormat::U32,
            2,
            4,
            4,
            10_000.0,
            RotaryPairing::SplitHalf,
        );
        let packed_run = executor()
            .execute_traced(
                &packed,
                map([
                    (
                        value_id("values"),
                        f32(&[2, 8], TensorLayout::PackedRagged, &[1.0; 16]),
                    ),
                    (
                        value_id("positions"),
                        HostTensor::u32(vec![2], TensorLayout::RowMajor, vec![0, 1]).unwrap(),
                    ),
                ]),
                BTreeMap::new(),
            )
            .unwrap();
        assert_eq!(
            packed_run.outputs[&value_id("out")].layout(),
            &TensorLayout::PackedRagged
        );

        let bf16 = rotary_graph(
            &[1, 8],
            NumericalFormat::Bf16,
            TensorLayout::TokensHidden,
            NumericalFormat::U32,
            2,
            4,
            4,
            10_000.0,
            RotaryPairing::SplitHalf,
        );
        let error = executor()
            .execute_traced(&bf16, BTreeMap::new(), BTreeMap::new())
            .unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::Unsupported);
        assert_eq!(error.reason_code, "rotary_values");
        assert_eq!(error.trace.completed_node_count, 0);

        let fp16 = rotary_graph(
            &[1, 8],
            NumericalFormat::Fp16,
            TensorLayout::TokensHidden,
            NumericalFormat::U32,
            2,
            4,
            4,
            10_000.0,
            RotaryPairing::SplitHalf,
        );
        let fp16_error = executor()
            .execute_traced(&fp16, BTreeMap::new(), BTreeMap::new())
            .unwrap_err();
        assert_eq!(fp16_error.class, CpuReferenceFailureClass::Unsupported);
        assert_eq!(fp16_error.reason_code, "rotary_values");
        assert_eq!(fp16_error.trace.completed_node_count, 0);
    }

    #[test]
    fn rotary_position_precision_fails_without_value_disclosure_and_preserves_prefix() {
        let graph = rotary_graph(
            &[1, 8],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            NumericalFormat::U64,
            2,
            4,
            4,
            10_000.0,
            RotaryPairing::SplitHalf,
        );
        let secret_position = MAX_EXACT_F32_INTEGER + 1;
        let error = executor()
            .execute_traced(
                &graph,
                map([
                    (
                        value_id("values"),
                        f32(&[1, 8], TensorLayout::TokensHidden, &[1.0; 8]),
                    ),
                    (
                        value_id("positions"),
                        HostTensor::u64(vec![1], TensorLayout::RowMajor, vec![secret_position])
                            .unwrap(),
                    ),
                ]),
                BTreeMap::new(),
            )
            .unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::Unsupported);
        assert_eq!(error.reason_code, "position_precision");
        assert!(error.message.contains("row 0"));
        assert!(error.message.contains(&MAX_EXACT_F32_INTEGER.to_string()));
        assert!(!error.message.contains(&secret_position.to_string()));
        assert_eq!(error.trace.completed_node_count, 0);

        let mut prefixed = graph.clone();
        prefixed.inputs[0] = value_id("input");
        prefixed.values[0].id = value_id("input");
        prefixed.values.insert(
            1,
            value(
                "values",
                &[1, 8],
                NumericalFormat::F32,
                TensorLayout::TokensHidden,
            ),
        );
        let mut identity = vec![0.0_f32; 64];
        for index in 0..8 {
            identity[index * 8 + index] = 1.0;
        }
        prefixed.weights.push(weight("linear", &[8, 8]));
        let rotary_node = prefixed.nodes.pop().unwrap();
        prefixed.nodes.push(GraphNode {
            id: node_id("linear"),
            inputs: vec![value_id("input")],
            outputs: vec![value_id("values")],
            weights: vec![weight_id("linear")],
            states: vec![],
            kind: GraphNodeKind::Linear { has_bias: false },
        });
        prefixed.nodes.push(rotary_node);
        let prefixed_error = executor()
            .execute_traced(
                &prefixed,
                map([
                    (
                        value_id("input"),
                        f32(&[1, 8], TensorLayout::TokensHidden, &[1.0; 8]),
                    ),
                    (
                        value_id("positions"),
                        HostTensor::u64(vec![1], TensorLayout::RowMajor, vec![secret_position])
                            .unwrap(),
                    ),
                ]),
                map([(
                    weight_id("linear"),
                    f32(&[8, 8], TensorLayout::RowMajor, &identity),
                )]),
            )
            .unwrap_err();
        assert_eq!(prefixed_error.class, CpuReferenceFailureClass::Unsupported);
        assert_eq!(prefixed_error.reason_code, "position_precision");
        assert_eq!(prefixed_error.trace.completed_node_count, 1);
        assert_eq!(
            prefixed_error.trace.completed_nodes[0].id,
            node_id("linear")
        );
    }

    #[test]
    fn rotary_position_u32_and_u64_boundaries_are_explicit() {
        for format in [NumericalFormat::U32, NumericalFormat::U64] {
            let graph = rotary_graph(
                &[1, 4],
                NumericalFormat::F32,
                TensorLayout::TokensHidden,
                format.clone(),
                1,
                4,
                4,
                1.0,
                RotaryPairing::Interleaved,
            );
            let exact = MAX_EXACT_F32_INTEGER;
            let exact_position = match &format {
                NumericalFormat::U32 => {
                    HostTensor::u32(vec![1], TensorLayout::RowMajor, vec![exact as u32]).unwrap()
                }
                NumericalFormat::U64 => {
                    HostTensor::u64(vec![1], TensorLayout::RowMajor, vec![exact]).unwrap()
                }
                _ => unreachable!(),
            };
            executor()
                .execute(
                    &graph,
                    map([
                        (
                            value_id("values"),
                            f32(&[1, 4], TensorLayout::TokensHidden, &[1.0; 4]),
                        ),
                        (value_id("positions"), exact_position),
                    ]),
                    BTreeMap::new(),
                )
                .unwrap();

            let over = exact + 1;
            let over_position = match &format {
                NumericalFormat::U32 => {
                    HostTensor::u32(vec![1], TensorLayout::RowMajor, vec![over as u32]).unwrap()
                }
                NumericalFormat::U64 => {
                    HostTensor::u64(vec![1], TensorLayout::RowMajor, vec![over]).unwrap()
                }
                _ => unreachable!(),
            };
            let error = executor()
                .execute_traced(
                    &graph,
                    map([
                        (
                            value_id("values"),
                            f32(&[1, 4], TensorLayout::TokensHidden, &[1.0; 4]),
                        ),
                        (value_id("positions"), over_position),
                    ]),
                    BTreeMap::new(),
                )
                .unwrap_err();
            assert_eq!(error.class, CpuReferenceFailureClass::Unsupported);
            assert_eq!(error.reason_code, "position_precision");
            assert!(!error.message.contains(&over.to_string()));
        }
    }

    #[test]
    fn rotary_nonfinite_output_is_typed_numerical_failure() {
        let graph = rotary_graph(
            &[1, 4],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            NumericalFormat::U32,
            1,
            4,
            4,
            10_000.0,
            RotaryPairing::Interleaved,
        );
        let error = executor()
            .execute_traced(
                &graph,
                map([
                    (
                        value_id("values"),
                        f32(
                            &[1, 4],
                            TensorLayout::TokensHidden,
                            &[f32::MAX, f32::MAX, f32::MAX, f32::MAX],
                        ),
                    ),
                    (
                        value_id("positions"),
                        HostTensor::u32(vec![1], TensorLayout::RowMajor, vec![1]).unwrap(),
                    ),
                ]),
                BTreeMap::new(),
            )
            .unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::Numerical);
        assert_eq!(error.reason_code, "runtime_numerical");
        assert!(error.message.contains("rotary"));
        assert_eq!(error.trace.completed_node_count, 0);
    }

    #[test]
    fn rotary_work_budget_fails_in_preflight_without_execution_allocation() {
        const WIDTH: usize = 1_048_576;
        const NODES: usize = 3;
        let mut values = vec![value(
            "input",
            &[1, WIDTH],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
        )];
        let mut inputs = vec![value_id("input")];
        let mut nodes = Vec::new();
        let mut previous = value_id("input");
        for index in 0..NODES {
            let position_id = ValueId::new(format!("position-{index}")).unwrap();
            let output_id = ValueId::new(format!("out-{index}")).unwrap();
            inputs.push(position_id.clone());
            values.push(value(
                position_id.as_str(),
                &[1],
                NumericalFormat::U32,
                TensorLayout::RowMajor,
            ));
            values.push(value(
                output_id.as_str(),
                &[1, WIDTH],
                NumericalFormat::F32,
                TensorLayout::TokensHidden,
            ));
            nodes.push(GraphNode {
                id: NodeId::new(format!("rotary-{index}")).unwrap(),
                inputs: vec![previous.clone(), position_id],
                outputs: vec![output_id.clone()],
                weights: vec![],
                states: vec![],
                kind: GraphNodeKind::RotaryPosition {
                    heads: 1,
                    head_dim: WIDTH,
                    rotary_dim: WIDTH,
                    base: PositiveF32::new(10_000.0, "rotary base").unwrap(),
                    pairing: RotaryPairing::SplitHalf,
                },
            });
            previous = output_id;
        }
        let graph = ModelGraph {
            graph_id: "rotary-work-budget".into(),
            inputs,
            outputs: vec![previous],
            values,
            weights: vec![],
            nodes,
        };
        graph.validate().unwrap();
        let value_specs = graph
            .values
            .iter()
            .map(|value| (value.id.clone(), &value.tensor))
            .collect();
        let weight_specs = BTreeMap::new();
        let error = preflight_graph(&graph, &value_specs, &weight_specs).unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::Resource);
        assert_eq!(error.reason_code, "work_budget");
        assert_eq!(error.failed_node.as_ref().unwrap().id, node_id("rotary-2"));

        let execution_error = executor()
            .execute_traced(&graph, BTreeMap::new(), BTreeMap::new())
            .unwrap_err();
        assert_eq!(execution_error.class, CpuReferenceFailureClass::Resource);
        assert_eq!(execution_error.reason_code, "work_budget");
        assert_eq!(
            execution_error.failed_node.as_ref().unwrap().id,
            node_id("rotary-2")
        );
        assert_eq!(execution_error.trace.completed_node_count, 0);
    }

    #[test]
    fn causal_gqa_matches_literal_causal_grouped_query_attention() {
        let graph = causal_gqa_graph(
            &[2, 4],
            &[2, 2],
            &[2, 2],
            &[2, 4],
            TensorLayout::TokensHidden,
            vec![],
            4,
            2,
            1,
            1,
            1.0,
        );
        let inputs = map([
            (
                value_id("query"),
                f32(&[2, 4], TensorLayout::TokensHidden, &[0.0; 8]),
            ),
            (
                value_id("key"),
                f32(&[2, 2], TensorLayout::TokensHidden, &[1.0, 2.0, 3.0, 4.0]),
            ),
            (
                value_id("value"),
                f32(&[2, 2], TensorLayout::TokensHidden, &[1.0, 10.0, 3.0, 14.0]),
            ),
        ]);
        let traced = executor()
            .execute_traced(&graph, inputs.clone(), BTreeMap::new())
            .unwrap();
        assert_f32(
            &traced.outputs[&value_id("context")],
            &[1.0, 1.0, 10.0, 10.0, 2.0, 2.0, 12.0, 12.0],
            1e-6,
        );
        assert_eq!(
            traced.trace.completed_nodes[0].kind,
            CpuReferenceNodeKind::CausalGqaAttentionCore
        );
        let legacy = executor().execute(&graph, inputs, BTreeMap::new()).unwrap();
        assert_eq!(legacy.outputs, traced.outputs);
    }

    #[test]
    fn causal_gqa_stable_softmax_matches_literal_value() {
        let graph = causal_gqa_graph(
            &[2, 1],
            &[2, 1],
            &[2, 1],
            &[2, 1],
            TensorLayout::RowMajor,
            vec![],
            1,
            1,
            1,
            1,
            1.0,
        );
        let run = executor()
            .execute(
                &graph,
                map([
                    (
                        value_id("query"),
                        f32(&[2, 1], TensorLayout::RowMajor, &[0.0, 1.0]),
                    ),
                    (
                        value_id("key"),
                        f32(&[2, 1], TensorLayout::RowMajor, &[1.0, 2.0]),
                    ),
                    (
                        value_id("value"),
                        f32(&[2, 1], TensorLayout::RowMajor, &[2.0, 6.0]),
                    ),
                ]),
                BTreeMap::new(),
            )
            .unwrap();
        assert_f32(
            &run.outputs[&value_id("context")],
            &[2.0, 4.924_234_4],
            1e-5,
        );

        // At token 1 the literal scores are 100 and 200. Computing exp(score)
        // directly overflows f32, while max-subtracted softmax converges to the
        // second value. Keep this expected value independent of implementation
        // helpers so removing the stability shift makes the test fail.
        let overflow_without_shift = executor()
            .execute(
                &graph,
                map([
                    (
                        value_id("query"),
                        f32(&[2, 1], TensorLayout::RowMajor, &[0.0, 100.0]),
                    ),
                    (
                        value_id("key"),
                        f32(&[2, 1], TensorLayout::RowMajor, &[1.0, 2.0]),
                    ),
                    (
                        value_id("value"),
                        f32(&[2, 1], TensorLayout::RowMajor, &[2.0, 6.0]),
                    ),
                ]),
                BTreeMap::new(),
            )
            .unwrap();
        assert_f32(
            &overflow_without_shift.outputs[&value_id("context")],
            &[2.0, 6.0],
            1e-6,
        );
    }

    #[test]
    fn causal_gqa_rank3_sequences_are_independent_and_value_dim_can_differ() {
        let rank3 = causal_gqa_graph(
            &[2, 2, 1],
            &[2, 2, 1],
            &[2, 2, 1],
            &[2, 2, 1],
            TensorLayout::TokensHidden,
            vec![],
            1,
            1,
            1,
            1,
            1.0,
        );
        let rank3_run = executor()
            .execute(
                &rank3,
                map([
                    (
                        value_id("query"),
                        f32(&[2, 2, 1], TensorLayout::TokensHidden, &[0.0; 4]),
                    ),
                    (
                        value_id("key"),
                        f32(&[2, 2, 1], TensorLayout::TokensHidden, &[1.0; 4]),
                    ),
                    (
                        value_id("value"),
                        f32(
                            &[2, 2, 1],
                            TensorLayout::TokensHidden,
                            &[1.0, 3.0, 10.0, 30.0],
                        ),
                    ),
                ]),
                BTreeMap::new(),
            )
            .unwrap();
        assert_f32(
            &rank3_run.outputs[&value_id("context")],
            &[1.0, 2.0, 10.0, 20.0],
            1e-6,
        );

        let value_dim = causal_gqa_graph(
            &[2, 4],
            &[2, 2],
            &[2, 3],
            &[2, 6],
            TensorLayout::TokensHidden,
            vec![],
            2,
            1,
            2,
            3,
            1.0,
        );
        let value_dim_run = executor()
            .execute(
                &value_dim,
                map([
                    (
                        value_id("query"),
                        f32(&[2, 4], TensorLayout::TokensHidden, &[0.0; 8]),
                    ),
                    (
                        value_id("key"),
                        f32(&[2, 2], TensorLayout::TokensHidden, &[1.0; 4]),
                    ),
                    (
                        value_id("value"),
                        f32(
                            &[2, 3],
                            TensorLayout::TokensHidden,
                            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        ),
                    ),
                ]),
                BTreeMap::new(),
            )
            .unwrap();
        assert_f32(
            &value_dim_run.outputs[&value_id("context")],
            &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.5, 3.5, 4.5, 2.5, 3.5, 4.5],
            1e-6,
        );
    }

    #[test]
    fn causal_gqa_rejects_state_packed_and_non_f32_capabilities_before_payload() {
        let stateful = causal_gqa_graph(
            &[2, 1],
            &[2, 1],
            &[2, 1],
            &[2, 1],
            TensorLayout::TokensHidden,
            vec![StateId::new("kv").unwrap()],
            1,
            1,
            1,
            1,
            1.0,
        );
        let state_error = executor()
            .execute_traced(&stateful, BTreeMap::new(), BTreeMap::new())
            .unwrap_err();
        assert_eq!(state_error.class, CpuReferenceFailureClass::Unsupported);
        assert_eq!(state_error.reason_code, "stateful_node");
        assert_eq!(state_error.trace.completed_node_count, 0);
        assert_eq!(
            state_error.failed_node.unwrap().kind,
            CpuReferenceNodeKind::CausalGqaAttentionCore
        );

        let packed = causal_gqa_graph(
            &[2, 1],
            &[2, 1],
            &[2, 1],
            &[2, 1],
            TensorLayout::PackedRagged,
            vec![],
            1,
            1,
            1,
            1,
            1.0,
        );
        let packed_error = executor()
            .execute_traced(&packed, BTreeMap::new(), BTreeMap::new())
            .unwrap_err();
        assert_eq!(packed_error.class, CpuReferenceFailureClass::Unsupported);
        assert_eq!(packed_error.reason_code, "value_layout");

        for format in [NumericalFormat::Bf16, NumericalFormat::Fp16] {
            let mut graph = causal_gqa_graph(
                &[2, 1],
                &[2, 1],
                &[2, 1],
                &[2, 1],
                TensorLayout::TokensHidden,
                vec![],
                1,
                1,
                1,
                1,
                1.0,
            );
            for value in &mut graph.values {
                value.tensor.format = format.clone();
            }
            let error = executor()
                .execute_traced(&graph, BTreeMap::new(), BTreeMap::new())
                .unwrap_err();
            assert_eq!(error.class, CpuReferenceFailureClass::Unsupported);
            assert_eq!(error.reason_code, "value_layout");
        }
    }

    #[test]
    fn causal_gqa_work_budget_rejects_before_payload_and_max_dot_is_numerical() {
        let huge = causal_gqa_graph(
            &[10_000, 1],
            &[10_000, 1],
            &[10_000, 1],
            &[10_000, 1],
            TensorLayout::TokensHidden,
            vec![],
            1,
            1,
            1,
            1,
            1.0,
        );
        let work_error = executor()
            .execute_traced(&huge, BTreeMap::new(), BTreeMap::new())
            .unwrap_err();
        assert_eq!(work_error.class, CpuReferenceFailureClass::Resource);
        assert_eq!(work_error.reason_code, "work_budget");
        assert_eq!(work_error.failed_node.unwrap().id, node_id("causal-gqa"));
        assert_eq!(work_error.trace.completed_node_count, 0);

        let numerical = causal_gqa_graph(
            &[1, 1],
            &[1, 1],
            &[1, 1],
            &[1, 1],
            TensorLayout::TokensHidden,
            vec![],
            1,
            1,
            1,
            1,
            1.0,
        );
        let numerical_error = executor()
            .execute_traced(
                &numerical,
                map([
                    (
                        value_id("query"),
                        f32(&[1, 1], TensorLayout::TokensHidden, &[f32::MAX]),
                    ),
                    (
                        value_id("key"),
                        f32(&[1, 1], TensorLayout::TokensHidden, &[f32::MAX]),
                    ),
                    (
                        value_id("value"),
                        f32(&[1, 1], TensorLayout::TokensHidden, &[1.0]),
                    ),
                ]),
                BTreeMap::new(),
            )
            .unwrap_err();
        assert_eq!(numerical_error.class, CpuReferenceFailureClass::Numerical);
        assert_eq!(numerical_error.reason_code, "runtime_numerical");
        assert_eq!(
            numerical_error.failed_node.unwrap().id,
            node_id("causal-gqa")
        );
        assert_eq!(numerical_error.trace.completed_node_count, 0);
    }

    #[test]
    fn traced_invalid_map_and_payload_fail_without_node() {
        let graph = simple_linear_graph();
        let missing = executor()
            .execute_traced(
                &graph,
                BTreeMap::new(),
                map([(
                    weight_id("weight"),
                    f32(&[1, 1], TensorLayout::RowMajor, &[1.0]),
                )]),
            )
            .unwrap_err();
        assert_eq!(missing.class, CpuReferenceFailureClass::InvalidInput);
        assert_eq!(missing.reason_code, "input_map");
        assert!(missing.failed_node.is_none());
        assert_eq!(missing.trace.completed_node_count, 0);

        let malformed = executor()
            .execute_traced(
                &graph,
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
        assert_eq!(malformed.class, CpuReferenceFailureClass::InvalidInput);
        assert_eq!(malformed.reason_code, "input_payload");
        assert!(malformed.failed_node.is_none());

        let nonfinite = executor()
            .execute_traced(
                &graph,
                map([(
                    value_id("input"),
                    f32(&[1, 1], TensorLayout::TokensHidden, &[f32::NAN]),
                )]),
                map([(
                    weight_id("weight"),
                    f32(&[1, 1], TensorLayout::RowMajor, &[1.0]),
                )]),
            )
            .unwrap_err();
        assert_eq!(nonfinite.class, CpuReferenceFailureClass::Numerical);
        assert_eq!(nonfinite.reason_code, "nonfinite_input");
    }

    #[test]
    fn traced_unsupported_preflight_keeps_zero_completed_nodes() {
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
            .execute_traced(
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
        assert_eq!(error.class, CpuReferenceFailureClass::Unsupported);
        assert_eq!(error.reason_code, "activation");
        assert_eq!(error.trace.completed_node_count, 0);
        assert_eq!(
            error.failed_node.unwrap(),
            CpuReferenceNodeRef {
                id: node_id("gelu"),
                kind: CpuReferenceNodeKind::Activation,
            }
        );
    }

    #[test]
    fn traced_runtime_token_error_keeps_completed_prefix() {
        let graph = linear_then_embedding_graph();
        let secret_token = 3_333_333_333_u32;
        let inputs = map([
            (
                value_id("pre"),
                f32(&[1, 1], TensorLayout::TokensHidden, &[2.0]),
            ),
            (value_id("tokens"), u32_tokens(&[secret_token])),
        ]);
        let weights = map([
            (
                weight_id("linear-weight"),
                f32(&[1, 1], TensorLayout::RowMajor, &[1.0]),
            ),
            (
                weight_id("embedding"),
                f32(&[3, 1], TensorLayout::RowMajor, &[1.0, 2.0, 3.0]),
            ),
        ]);
        let error = executor()
            .execute_traced(&graph, inputs.clone(), weights.clone())
            .unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::InvalidInput);
        assert_eq!(error.reason_code, "runtime_input");
        assert_eq!(
            error.message,
            "embedding token position 0 is outside vocabulary size 3"
        );
        assert!(!error.message.contains(&secret_token.to_string()));
        assert!(!error.to_string().contains(&secret_token.to_string()));
        assert_eq!(
            error.failed_node.unwrap(),
            CpuReferenceNodeRef {
                id: node_id("embedding"),
                kind: CpuReferenceNodeKind::Embedding,
            }
        );
        assert_eq!(error.trace.completed_node_count, 1);
        assert_eq!(error.trace.completed_nodes[0].id, node_id("linear"));

        let legacy_error = executor().execute(&graph, inputs, weights).unwrap_err();
        assert!(legacy_error.contains("embedding token position 0 is outside vocabulary size 3"));
        assert!(!legacy_error.contains(&secret_token.to_string()));
    }

    #[test]
    fn binding_map_admission_rejects_unvalidated_or_excessive_keys_without_copying_them() {
        let graph = simple_linear_graph();
        let untrusted_id = "x".repeat(MAX_CPU_REFERENCE_FAILURE_MESSAGE_BYTES * 4);
        let error = executor()
            .execute_traced(
                &graph,
                map([(
                    ValueId(untrusted_id.clone()),
                    f32(&[1, 1], TensorLayout::TokensHidden, &[1.0]),
                )]),
                map([(
                    weight_id("weight"),
                    f32(&[1, 1], TensorLayout::RowMajor, &[1.0]),
                )]),
            )
            .unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::InvalidInput);
        assert_eq!(error.reason_code, "input_map");
        assert!(error.failed_node.is_none());
        assert!(error.message.len() <= MAX_CPU_REFERENCE_FAILURE_MESSAGE_BYTES);
        assert!(!error.message.contains(&untrusted_id));

        let mut excessive = BTreeMap::<ValueId, ()>::new();
        for index in 0..=MAX_GRAPH_ENDPOINTS {
            excessive.insert(ValueId::new(format!("input-{index}")).unwrap(), ());
        }
        let count_error =
            validate_binding_map_admission(&excessive, MAX_GRAPH_ENDPOINTS, "graph input")
                .unwrap_err();
        assert_eq!(
            count_error,
            format!(
                "graph input map entry count {} exceeds limit {MAX_GRAPH_ENDPOINTS}",
                MAX_GRAPH_ENDPOINTS + 1
            )
        );

        let expected = BTreeMap::<ValueId, ()>::new();
        let extras = (0..=MAX_CPU_REFERENCE_KEY_DIAGNOSTIC_IDS)
            .map(|index| (ValueId::new(format!("extra-{index}")).unwrap(), ()))
            .collect::<BTreeMap<_, _>>();
        let mismatch = validate_exact_keys(&extras, &expected, "graph input").unwrap_err();
        assert!(mismatch.contains("extra_count=9"));
        assert!(mismatch.contains("extra-7"));
        assert!(!mismatch.contains("extra-8"));
    }

    #[test]
    fn bounded_trace_and_failure_message_are_utf8_safe() {
        let graph = simple_linear_graph();
        let node = &graph.nodes[0];
        let mut trace = BoundedCompletedNodes::with_limit(1);
        trace.record_completed(node).unwrap();
        trace.record_completed(node).unwrap();
        let trace = trace.into_trace();
        assert_eq!(trace.completed_node_count, 2);
        assert_eq!(trace.completed_nodes.len(), 1);
        assert!(trace.completed_nodes_truncated);

        let message = bounded_failure_message("é".repeat(MAX_CPU_REFERENCE_FAILURE_MESSAGE_BYTES));
        assert!(message.len() <= MAX_CPU_REFERENCE_FAILURE_MESSAGE_BYTES);
        assert!(message.is_char_boundary(message.len()));
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
    fn sigmoid_standalone_and_gated_mlp_stay_unsupported_to_avoid_undercharged_work() {
        // These paths do not yet charge sigmoid exponentiation. Empty payload
        // maps fix capability rejection in preflight, before payload admission
        // or any output/temporary allocation can execute.
        let activation = ModelGraph {
            graph_id: "unsupported-standalone-sigmoid".into(),
            inputs: vec![value_id("input")],
            outputs: vec![value_id("out")],
            values: vec![
                value(
                    "input",
                    &[1, 2],
                    NumericalFormat::F32,
                    TensorLayout::RowMajor,
                ),
                value("out", &[1, 2], NumericalFormat::F32, TensorLayout::RowMajor),
            ],
            weights: vec![],
            nodes: vec![GraphNode {
                id: node_id("standalone-sigmoid"),
                inputs: vec![value_id("input")],
                outputs: vec![value_id("out")],
                weights: vec![],
                states: vec![],
                kind: GraphNodeKind::Activation {
                    kind: ActivationKind::Sigmoid,
                },
            }],
        };
        let activation_error = executor()
            .execute_traced(&activation, BTreeMap::new(), BTreeMap::new())
            .unwrap_err();
        assert_eq!(
            activation_error.class,
            CpuReferenceFailureClass::Unsupported
        );
        assert_eq!(activation_error.reason_code, "activation");
        assert_eq!(activation_error.trace.completed_node_count, 0);
        assert_eq!(
            activation_error.failed_node,
            Some(CpuReferenceNodeRef {
                id: node_id("standalone-sigmoid"),
                kind: CpuReferenceNodeKind::Activation,
            })
        );

        let gated_mlp = ModelGraph {
            graph_id: "unsupported-gated-mlp-sigmoid".into(),
            inputs: vec![value_id("input")],
            outputs: vec![value_id("out")],
            values: vec![
                value(
                    "input",
                    &[1, 2],
                    NumericalFormat::F32,
                    TensorLayout::RowMajor,
                ),
                value("out", &[1, 2], NumericalFormat::F32, TensorLayout::RowMajor),
            ],
            weights: vec![
                weight("gate", &[2, 2]),
                weight("up", &[2, 2]),
                weight("down", &[2, 2]),
            ],
            nodes: vec![GraphNode {
                id: node_id("gated-mlp-sigmoid"),
                inputs: vec![value_id("input")],
                outputs: vec![value_id("out")],
                weights: vec![weight_id("gate"), weight_id("up"), weight_id("down")],
                states: vec![],
                kind: GraphNodeKind::GatedMlp {
                    intermediate_size: 2,
                    activation: ActivationKind::Sigmoid,
                },
            }],
        };
        let gated_mlp_error = executor()
            .execute_traced(&gated_mlp, BTreeMap::new(), BTreeMap::new())
            .unwrap_err();
        assert_eq!(gated_mlp_error.class, CpuReferenceFailureClass::Unsupported);
        assert_eq!(gated_mlp_error.reason_code, "activation");
        assert_eq!(gated_mlp_error.trace.completed_node_count, 0);
        assert_eq!(
            gated_mlp_error.failed_node,
            Some(CpuReferenceNodeRef {
                id: node_id("gated-mlp-sigmoid"),
                kind: CpuReferenceNodeKind::GatedMlp,
            })
        );
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
        assert_eq!(error.class, CpuReferenceFailureClass::Resource);
        assert_eq!(error.reason_code, "work_budget");
        assert_eq!(
            error.failed_node.as_ref().unwrap().id,
            node_id("large-linear")
        );
        assert!(error.message.contains("work-unit budget"));
        let failure = error.into_failure(CpuReferenceExecutionTrace::default());
        assert_eq!(failure.trace.completed_node_count, 0);
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
    fn new_scan_conv_and_decay_nodes_execute_state_free_and_reject_stateful() {
        let conv = ModelGraph {
            graph_id: "unsupported-causal-depthwise-conv".into(),
            inputs: vec![value_id("conv-input")],
            outputs: vec![value_id("conv-output")],
            values: vec![
                value(
                    "conv-input",
                    &[1, 2, 2],
                    NumericalFormat::F32,
                    TensorLayout::RowMajor,
                ),
                value(
                    "conv-output",
                    &[1, 2, 2],
                    NumericalFormat::F32,
                    TensorLayout::RowMajor,
                ),
            ],
            weights: vec![weight("conv-kernel", &[2, 1, 3])],
            nodes: vec![GraphNode {
                id: node_id("conv"),
                inputs: vec![value_id("conv-input")],
                outputs: vec![value_id("conv-output")],
                weights: vec![weight_id("conv-kernel")],
                states: vec![],
                kind: GraphNodeKind::CausalDepthwiseConv1d {
                    channels: 2,
                    kernel_size: 3,
                },
            }],
        };
        let decay = ModelGraph {
            graph_id: "unsupported-gated-decay".into(),
            inputs: vec![value_id("decay-control"), value_id("update-control")],
            outputs: vec![value_id("log-decay"), value_id("update-rate")],
            values: [
                "decay-control",
                "update-control",
                "log-decay",
                "update-rate",
            ]
            .into_iter()
            .map(|name| {
                value(
                    name,
                    &[1, 2, 2],
                    NumericalFormat::F32,
                    TensorLayout::RowMajor,
                )
            })
            .collect(),
            weights: vec![weight("log-rate", &[2]), weight("time-bias", &[2])],
            nodes: vec![GraphNode {
                id: node_id("decay"),
                inputs: vec![value_id("decay-control"), value_id("update-control")],
                outputs: vec![value_id("log-decay"), value_id("update-rate")],
                weights: vec![weight_id("log-rate"), weight_id("time-bias")],
                states: vec![],
                kind: GraphNodeKind::GatedDecayParameters { channels: 2 },
            }],
        };
        let scan = ModelGraph {
            graph_id: "unsupported-gated-delta-scan".into(),
            inputs: vec![
                value_id("query"),
                value_id("key"),
                value_id("scan-value"),
                value_id("scan-log-decay"),
                value_id("scan-update-rate"),
            ],
            outputs: vec![value_id("context")],
            values: [
                "query",
                "key",
                "scan-value",
                "scan-log-decay",
                "scan-update-rate",
                "context",
            ]
            .into_iter()
            .map(|name| {
                value(
                    name,
                    &[1, 2, 2],
                    NumericalFormat::F32,
                    TensorLayout::RowMajor,
                )
            })
            .collect(),
            weights: vec![],
            nodes: vec![GraphNode {
                id: node_id("scan"),
                inputs: vec![
                    value_id("query"),
                    value_id("key"),
                    value_id("scan-value"),
                    value_id("scan-log-decay"),
                    value_id("scan-update-rate"),
                ],
                outputs: vec![value_id("context")],
                weights: vec![],
                states: vec![],
                kind: GraphNodeKind::GatedDeltaRuleScan {
                    key_heads: 1,
                    value_heads: 2,
                    key_dim: 2,
                    value_dim: 1,
                },
            }],
        };
        let mut stateful_conv = conv.clone();
        stateful_conv.nodes[0].states = vec![StateId::new("conv-history").unwrap()];
        let mut stateful_scan = scan.clone();
        stateful_scan.nodes[0].states = vec![StateId::new("scan-bank").unwrap()];

        let conv_run = executor()
            .execute_traced(
                &conv,
                map([(
                    value_id("conv-input"),
                    f32(&[1, 2, 2], TensorLayout::RowMajor, &[1.0, 10.0, 2.0, 20.0]),
                )]),
                map([(
                    weight_id("conv-kernel"),
                    f32(
                        &[2, 1, 3],
                        TensorLayout::RowMajor,
                        &[1.0, 2.0, 3.0, 1.0, 0.0, 1.0],
                    ),
                )]),
            )
            .unwrap();
        assert_f32(
            &conv_run.outputs[&value_id("conv-output")],
            &[3.0, 10.0, 8.0, 20.0],
            0.0,
        );
        assert_eq!(
            conv_run.trace.completed_nodes[0].kind,
            CpuReferenceNodeKind::CausalDepthwiseConv1d
        );

        let decay_run = executor()
            .execute(
                &decay,
                map([
                    (
                        value_id("decay-control"),
                        f32(&[1, 2, 2], TensorLayout::RowMajor, &[0.0, 0.0, 1.0, -1.0]),
                    ),
                    (
                        value_id("update-control"),
                        f32(
                            &[1, 2, 2],
                            TensorLayout::RowMajor,
                            &[0.0, 100.0, -100.0, 0.0],
                        ),
                    ),
                ]),
                map([
                    (
                        weight_id("log-rate"),
                        f32(&[2], TensorLayout::RowMajor, &[0.0, 0.0]),
                    ),
                    (
                        weight_id("time-bias"),
                        f32(&[2], TensorLayout::RowMajor, &[0.0, 0.0]),
                    ),
                ]),
            )
            .unwrap();
        assert_f32(
            &decay_run.outputs[&value_id("log-decay")],
            &[-0.693_147_2, -0.693_147_2, -1.313_261_6, -0.313_261_7],
            2e-6,
        );
        assert_f32(
            &decay_run.outputs[&value_id("update-rate")],
            &[0.5, 1.0, 0.0, 0.5],
            1e-6,
        );
        let (_, _, update_values) = decay_run.outputs[&value_id("update-rate")]
            .f32_parts()
            .unwrap();
        assert!(update_values[2] > 0.0 && update_values[2].is_finite());

        let scan_inputs = map([
            (
                value_id("query"),
                f32(&[1, 2, 2], TensorLayout::RowMajor, &[1.0, 0.0, 0.0, 1.0]),
            ),
            (
                value_id("key"),
                f32(&[1, 2, 2], TensorLayout::RowMajor, &[1.0, 0.0, 0.0, 1.0]),
            ),
            (
                value_id("scan-value"),
                f32(&[1, 2, 2], TensorLayout::RowMajor, &[2.0, 4.0, 3.0, 5.0]),
            ),
            (
                value_id("scan-log-decay"),
                f32(&[1, 2, 2], TensorLayout::RowMajor, &[0.0; 4]),
            ),
            (
                value_id("scan-update-rate"),
                f32(&[1, 2, 2], TensorLayout::RowMajor, &[1.0; 4]),
            ),
        ]);
        let scan_traced = executor()
            .execute_traced(&scan, scan_inputs.clone(), BTreeMap::new())
            .unwrap();
        assert_f32(
            &scan_traced.outputs[&value_id("context")],
            &[2.0, 4.0, 3.0, 5.0],
            1e-6,
        );
        assert_eq!(
            executor()
                .execute(&scan, scan_inputs, BTreeMap::new())
                .unwrap()
                .outputs,
            scan_traced.outputs
        );

        for (graph, expected_id, expected_kind) in [
            (
                stateful_conv,
                node_id("conv"),
                CpuReferenceNodeKind::CausalDepthwiseConv1d,
            ),
            (
                stateful_scan,
                node_id("scan"),
                CpuReferenceNodeKind::GatedDeltaRuleScan,
            ),
        ] {
            graph.validate().unwrap();
            let error = executor()
                .execute_traced(&graph, BTreeMap::new(), BTreeMap::new())
                .unwrap_err();
            assert_eq!(error.class, CpuReferenceFailureClass::Unsupported);
            assert_eq!(error.reason_code, "stateful_node");
            assert_eq!(error.trace.completed_node_count, 0);
            assert_eq!(
                error.failed_node,
                Some(CpuReferenceNodeRef {
                    id: expected_id,
                    kind: expected_kind,
                })
            );
        }
    }

    #[test]
    fn conv_state_free_k1_rank3_capabilities_and_failures() {
        let k1 = conv_graph(
            &[1, 2, 1],
            NumericalFormat::F32,
            TensorLayout::RowMajor,
            1,
            1,
        );
        let run = executor()
            .execute(
                &k1,
                map([(
                    value_id("x"),
                    f32(&[1, 2, 1], TensorLayout::RowMajor, &[2.0, 3.0]),
                )]),
                map([(
                    weight_id("kernel"),
                    f32(&[1, 1, 1], TensorLayout::RowMajor, &[4.0]),
                )]),
            )
            .unwrap();
        assert_f32(&run.outputs[&value_id("y")], &[8.0, 12.0], 0.0);
        let rank3 = conv_graph(
            &[2, 2, 1],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            1,
            2,
        );
        let run = executor()
            .execute(
                &rank3,
                map([(
                    value_id("x"),
                    f32(
                        &[2, 2, 1],
                        TensorLayout::TokensHidden,
                        &[1.0, 2.0, 10.0, 20.0],
                    ),
                )]),
                map([(
                    weight_id("kernel"),
                    f32(&[1, 1, 2], TensorLayout::RowMajor, &[1.0, 1.0]),
                )]),
            )
            .unwrap();
        assert_f32(&run.outputs[&value_id("y")], &[1.0, 3.0, 10.0, 30.0], 0.0);
        for graph in [
            conv_graph(
                &[1, 1, 1],
                NumericalFormat::Bf16,
                TensorLayout::RowMajor,
                1,
                1,
            ),
            conv_graph(
                &[1, 1, 1],
                NumericalFormat::F32,
                TensorLayout::PackedRagged,
                1,
                1,
            ),
        ] {
            let error = executor()
                .execute_traced(&graph, BTreeMap::new(), BTreeMap::new())
                .unwrap_err();
            assert_eq!(error.class, CpuReferenceFailureClass::Unsupported);
        }
        let numerical = conv_graph(
            &[1, 1, 1],
            NumericalFormat::F32,
            TensorLayout::RowMajor,
            1,
            1,
        );
        let error = executor()
            .execute_traced(
                &numerical,
                map([(
                    value_id("x"),
                    f32(&[1, 1, 1], TensorLayout::RowMajor, &[f32::MAX]),
                )]),
                map([(
                    weight_id("kernel"),
                    f32(&[1, 1, 1], TensorLayout::RowMajor, &[2.0]),
                )]),
            )
            .unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::Numerical);
        let large = MAX_CPU_REFERENCE_TENSOR_ELEMENTS + 1;
        let resource = conv_graph(
            &[1, large, 1],
            NumericalFormat::F32,
            TensorLayout::RowMajor,
            1,
            1,
        );
        let error = executor()
            .execute_traced(&resource, BTreeMap::new(), BTreeMap::new())
            .unwrap_err();
        assert_eq!(error.reason_code, "element_budget");
        let work = conv_graph(
            &[1, MAX_CPU_REFERENCE_TENSOR_ELEMENTS, 1],
            NumericalFormat::F32,
            TensorLayout::RowMajor,
            1,
            26,
        );
        let error = executor()
            .execute_traced(&work, BTreeMap::new(), BTreeMap::new())
            .unwrap_err();
        assert_eq!(error.reason_code, "work_budget");
    }

    #[test]
    fn decay_state_free_capabilities_overflow_and_resource_preflight() {
        let tokens_hidden = decay_graph(
            &[1, 1, 1],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            1,
        );
        let run = executor()
            .execute(
                &tokens_hidden,
                map([
                    (
                        value_id("dc"),
                        f32(&[1, 1, 1], TensorLayout::TokensHidden, &[0.0]),
                    ),
                    (
                        value_id("uc"),
                        f32(&[1, 1, 1], TensorLayout::TokensHidden, &[0.0]),
                    ),
                ]),
                map([
                    (weight_id("lr"), f32(&[1], TensorLayout::RowMajor, &[0.0])),
                    (weight_id("tb"), f32(&[1], TensorLayout::RowMajor, &[0.0])),
                ]),
            )
            .unwrap();
        assert_f32(
            &run.outputs[&value_id("ld")],
            &[-std::f32::consts::LN_2],
            1e-6,
        );
        assert_f32(&run.outputs[&value_id("ur")], &[0.5], 1e-6);

        for graph in [
            decay_graph(&[1, 1, 1], NumericalFormat::Fp16, TensorLayout::RowMajor, 1),
            decay_graph(
                &[1, 1, 1],
                NumericalFormat::F32,
                TensorLayout::PackedRagged,
                1,
            ),
        ] {
            let error = executor()
                .execute_traced(&graph, BTreeMap::new(), BTreeMap::new())
                .unwrap_err();
            assert_eq!(error.class, CpuReferenceFailureClass::Unsupported);
        }
        let graph = decay_graph(&[1, 1, 1], NumericalFormat::F32, TensorLayout::RowMajor, 1);
        let error = executor()
            .execute_traced(
                &graph,
                map([
                    (
                        value_id("dc"),
                        f32(&[1, 1, 1], TensorLayout::RowMajor, &[0.0]),
                    ),
                    (
                        value_id("uc"),
                        f32(&[1, 1, 1], TensorLayout::RowMajor, &[0.0]),
                    ),
                ]),
                map([
                    (weight_id("lr"), f32(&[1], TensorLayout::RowMajor, &[100.0])),
                    (weight_id("tb"), f32(&[1], TensorLayout::RowMajor, &[0.0])),
                ]),
            )
            .unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::Numerical);
        let large = MAX_CPU_REFERENCE_TENSOR_ELEMENTS + 1;
        let graph = decay_graph(
            &[1, large, 1],
            NumericalFormat::F32,
            TensorLayout::RowMajor,
            1,
        );
        let error = executor()
            .execute_traced(&graph, BTreeMap::new(), BTreeMap::new())
            .unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::Resource);
        let work = decay_graph(
            &[1, MAX_CPU_REFERENCE_TENSOR_ELEMENTS, 1],
            NumericalFormat::F32,
            TensorLayout::RowMajor,
            1,
        );
        let error = executor()
            .execute_traced(&work, BTreeMap::new(), BTreeMap::new())
            .unwrap_err();
        assert_eq!(error.reason_code, "work_budget");
    }

    #[test]
    fn scan_state_free_rank3_chunk_equivalence_capabilities_and_budgets() {
        let rank3 = scan_graph(
            &[2, 1],
            NumericalFormat::F32,
            TensorLayout::RowMajor,
            1,
            2,
            2,
            1,
        );
        let inputs = map([
            (
                value_id("q"),
                f32(&[2, 1, 2], TensorLayout::RowMajor, &[1.0, 1.0, 1.0, 1.0]),
            ),
            (
                value_id("k"),
                f32(&[2, 1, 2], TensorLayout::RowMajor, &[1.0, 0.5, 1.0, 0.5]),
            ),
            (
                value_id("v"),
                f32(&[2, 1, 2], TensorLayout::RowMajor, &[2.0, 4.0, 3.0, 5.0]),
            ),
            (
                value_id("d"),
                f32(&[2, 1, 2], TensorLayout::RowMajor, &[0.0; 4]),
            ),
            (
                value_id("b"),
                f32(&[2, 1, 2], TensorLayout::RowMajor, &[0.5; 4]),
            ),
        ]);
        let run = executor().execute(&rank3, inputs, BTreeMap::new()).unwrap();
        assert_f32(&run.outputs[&value_id("o")], &[1.5, 3.0, 2.25, 3.75], 1e-6);

        let tensor = |data: &[f32]| f32(&[1, 2, 1], TensorLayout::RowMajor, data);
        let q = tensor(&[1.0, 1.0]);
        let k = tensor(&[1.0, 1.0]);
        let v = tensor(&[2.0, 4.0]);
        let d = tensor(&[0.0, 0.0]);
        let b = tensor(&[1.0, 0.5]);
        let full_spec = spec(&[1, 2, 1], NumericalFormat::F32, TensorLayout::RowMajor);
        let mut allocated = 0;
        let mut runtime = RuntimeExecutionContext::default();
        let (full, full_state) = gated_delta_rule_scan_f32(
            &q,
            &k,
            &v,
            &d,
            &b,
            1,
            1,
            1,
            1,
            None,
            &full_spec,
            &mut allocated,
            &mut runtime,
        )
        .unwrap();
        let one = |x: f32| f32(&[1, 1, 1], TensorLayout::RowMajor, &[x]);
        let one_spec = spec(&[1, 1, 1], NumericalFormat::F32, TensorLayout::RowMajor);
        let mut a1 = 0;
        let mut r1 = RuntimeExecutionContext::default();
        let (first, s1) = gated_delta_rule_scan_f32(
            &one(1.0),
            &one(1.0),
            &one(2.0),
            &one(0.0),
            &one(1.0),
            1,
            1,
            1,
            1,
            None,
            &one_spec,
            &mut a1,
            &mut r1,
        )
        .unwrap();
        let mut a2 = 0;
        let mut r2 = RuntimeExecutionContext::default();
        let (second, s2) = gated_delta_rule_scan_f32(
            &one(1.0),
            &one(1.0),
            &one(4.0),
            &one(0.0),
            &one(0.5),
            1,
            1,
            1,
            1,
            Some(&s1),
            &one_spec,
            &mut a2,
            &mut r2,
        )
        .unwrap();
        let mut a_zero = 0;
        let mut r_zero = RuntimeExecutionContext::default();
        let (second_from_zero, zero_state) = gated_delta_rule_scan_f32(
            &one(1.0),
            &one(1.0),
            &one(4.0),
            &one(0.0),
            &one(0.5),
            1,
            1,
            1,
            1,
            None,
            &one_spec,
            &mut a_zero,
            &mut r_zero,
        )
        .unwrap();
        assert_f32(&full, &[2.0, 3.0], 1e-6);
        assert_f32(&first, &[2.0], 1e-6);
        assert_f32(&second, &[3.0], 1e-6);
        assert_eq!(s2, vec![3.0]);
        assert_f32(&second_from_zero, &[2.0], 1e-6);
        assert_eq!(zero_state, vec![2.0]);
        assert_ne!(second, second_from_zero);
        assert_eq!(full_state, s2);

        let golden_spec = spec(&[1, 1, 4], NumericalFormat::F32, TensorLayout::TokensHidden);
        let mut golden_allocated = 0;
        let mut golden_runtime = RuntimeExecutionContext::default();
        let (golden_output, golden_state) = gated_delta_rule_scan_f32(
            &f32(&[1, 1, 2], TensorLayout::TokensHidden, &[2.0, 3.0]),
            &f32(&[1, 1, 2], TensorLayout::TokensHidden, &[4.0, 5.0]),
            &f32(
                &[1, 1, 4],
                TensorLayout::TokensHidden,
                &[10.0, 20.0, 30.0, 40.0],
            ),
            &f32(
                &[1, 1, 4],
                TensorLayout::TokensHidden,
                &[-std::f32::consts::LN_2; 4],
            ),
            &f32(&[1, 1, 4], TensorLayout::TokensHidden, &[0.25; 4]),
            2,
            4,
            1,
            1,
            Some(&[1.0, 2.0, 3.0, 4.0]),
            &golden_spec,
            &mut golden_allocated,
            &mut golden_runtime,
        )
        .unwrap();
        // Each lane decays, predicts, applies the beta-scaled residual, then reads the
        // updated state. For hv0: 1/2 -> prediction 2 -> residual step 2 -> state
        // 1/2 + 4 * 2 = 8.5 -> context 2 * 8.5 = 17; the other lanes follow likewise.
        assert_f32(&golden_output, &[17.0, 34.0, 88.875, 118.5], 1e-6);
        assert_eq!(golden_state, vec![8.5, 17.0, 29.625, 39.5]);

        for graph in [
            scan_graph(
                &[1, 1],
                NumericalFormat::Bf16,
                TensorLayout::RowMajor,
                1,
                1,
                1,
                1,
            ),
            scan_graph(
                &[1, 1],
                NumericalFormat::F32,
                TensorLayout::PackedRagged,
                1,
                1,
                1,
                1,
            ),
        ] {
            let error = executor()
                .execute_traced(&graph, BTreeMap::new(), BTreeMap::new())
                .unwrap_err();
            assert_eq!(error.class, CpuReferenceFailureClass::Unsupported);
        }
        let overflow = scan_graph(
            &[1, 1],
            NumericalFormat::F32,
            TensorLayout::RowMajor,
            1,
            1,
            1,
            1,
        );
        let error = executor()
            .execute_traced(
                &overflow,
                map([
                    (value_id("q"), one(1.0)),
                    (value_id("k"), one(1.0)),
                    (value_id("v"), one(1.0)),
                    (value_id("d"), one(100.0)),
                    (value_id("b"), one(1.0)),
                ]),
                BTreeMap::new(),
            )
            .unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::Numerical);
        let work = scan_graph(
            &[1, 100_000],
            NumericalFormat::F32,
            TensorLayout::RowMajor,
            1,
            1,
            64,
            1,
        );
        let error = executor()
            .execute_traced(&work, BTreeMap::new(), BTreeMap::new())
            .unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::Resource);
        assert_eq!(error.reason_code, "work_budget");
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

    #[test]
    fn grouped_last_split_literal_preserves_group_order_and_segment_offsets() {
        let graph = grouped_last_split_graph(
            &[1, 18],
            &[&[1, 6], &[1, 3], &[1, 9]],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            3,
            vec![2, 1, 3],
        );
        let inputs = map([(
            value_id("split-input"),
            f32(
                &[1, 18],
                TensorLayout::TokensHidden,
                &[
                    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0, 17.0,
                ],
            ),
        )]);
        let traced = executor()
            .execute_traced(&graph, inputs.clone(), BTreeMap::new())
            .unwrap();
        assert_f32(
            &traced.outputs[&value_id("split-output-0")],
            &[0.0, 1.0, 6.0, 7.0, 12.0, 13.0],
            0.0,
        );
        assert_f32(
            &traced.outputs[&value_id("split-output-1")],
            &[2.0, 8.0, 14.0],
            0.0,
        );
        assert_f32(
            &traced.outputs[&value_id("split-output-2")],
            &[3.0, 4.0, 5.0, 9.0, 10.0, 11.0, 15.0, 16.0, 17.0],
            0.0,
        );
        assert_eq!(
            traced.trace.completed_nodes,
            vec![CpuReferenceNodeRef {
                id: node_id("split"),
                kind: CpuReferenceNodeKind::GroupedLastSplit,
            }]
        );
        let legacy = executor().execute(&graph, inputs, BTreeMap::new()).unwrap();
        assert_eq!(legacy.outputs, traced.outputs);
    }

    #[test]
    fn last_axis_split_literal_rank3_and_legacy_are_exact() {
        let mut graph = grouped_last_split_graph(
            &[1, 2, 6],
            &[&[1, 2, 2], &[1, 2, 1], &[1, 2, 3]],
            NumericalFormat::F32,
            TensorLayout::RowMajor,
            1,
            vec![2, 1, 3],
        );
        graph.nodes[0].kind = GraphNodeKind::LastAxisSplit {
            segment_widths: vec![2, 1, 3],
        };
        let inputs = map([(
            value_id("split-input"),
            f32(
                &[1, 2, 6],
                TensorLayout::RowMajor,
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            ),
        )]);
        let traced = executor()
            .execute_traced(&graph, inputs.clone(), BTreeMap::new())
            .unwrap();
        assert_f32(
            &traced.outputs[&value_id("split-output-0")],
            &[0.0, 1.0, 6.0, 7.0],
            0.0,
        );
        assert_f32(
            &traced.outputs[&value_id("split-output-1")],
            &[2.0, 8.0],
            0.0,
        );
        assert_f32(
            &traced.outputs[&value_id("split-output-2")],
            &[3.0, 4.0, 5.0, 9.0, 10.0, 11.0],
            0.0,
        );
        assert_eq!(
            traced.trace.completed_nodes[0].kind,
            CpuReferenceNodeKind::LastAxisSplit
        );
        assert_eq!(
            executor()
                .execute(&graph, inputs, BTreeMap::new())
                .unwrap()
                .outputs,
            traced.outputs
        );
    }

    #[test]
    fn last_axis_split_preflight_rejects_dtype_layout_and_resource() {
        for (format, layout) in [
            (NumericalFormat::Bf16, TensorLayout::RowMajor),
            (NumericalFormat::F32, TensorLayout::PackedRagged),
        ] {
            // PackedRagged remains graph-valid for this final-axis-local
            // semantic; only this CPU executor capability rejects it.
            let mut graph = grouped_last_split_graph(
                &[1, 2],
                &[&[1, 1], &[1, 1]],
                format,
                layout,
                1,
                vec![1, 1],
            );
            graph.nodes[0].kind = GraphNodeKind::LastAxisSplit {
                segment_widths: vec![1, 1],
            };
            let error = executor()
                .execute_traced(&graph, BTreeMap::new(), BTreeMap::new())
                .unwrap_err();
            assert_eq!(error.class, CpuReferenceFailureClass::Unsupported);
            assert_eq!(error.reason_code, "value_layout");
        }
        let large = MAX_CPU_REFERENCE_TENSOR_ELEMENTS + 1;
        let mut graph = grouped_last_split_graph(
            &[1, large + 1],
            &[&[1, large], &[1, 1]],
            NumericalFormat::F32,
            TensorLayout::RowMajor,
            1,
            vec![large, 1],
        );
        graph.nodes[0].kind = GraphNodeKind::LastAxisSplit {
            segment_widths: vec![large, 1],
        };
        let error = executor()
            .execute_traced(&graph, BTreeMap::new(), BTreeMap::new())
            .unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::Resource);
        assert_eq!(error.reason_code, "element_budget");

        let mut accounted = grouped_last_split_graph(
            &[1, 6],
            &[&[1, 2], &[1, 1], &[1, 3]],
            NumericalFormat::F32,
            TensorLayout::RowMajor,
            1,
            vec![2, 1, 3],
        );
        accounted.nodes[0].kind = GraphNodeKind::LastAxisSplit {
            segment_widths: vec![2, 1, 3],
        };
        let specs = accounted
            .values
            .iter()
            .map(|value| (value.id.clone(), &value.tensor))
            .collect();
        let plan = preflight_graph(&accounted, &specs, &BTreeMap::new()).unwrap();
        assert_eq!(plan.execution_elements, 6);
        assert_eq!(plan.work_units, 6);

        // Copy work equals output elements, so the stricter aggregate element
        // budget necessarily fires before the larger global work budget.
        const WIDTH: usize = MAX_CPU_REFERENCE_TENSOR_ELEMENTS;
        let output_shapes = [&[1, WIDTH][..]; 9];
        let mut aggregate = grouped_last_split_graph(
            &[1, WIDTH * 9],
            &output_shapes,
            NumericalFormat::F32,
            TensorLayout::RowMajor,
            1,
            vec![WIDTH; 9],
        );
        aggregate.nodes[0].kind = GraphNodeKind::LastAxisSplit {
            segment_widths: vec![WIDTH; 9],
        };
        aggregate.validate().unwrap();
        let error = executor()
            .execute_traced(&aggregate, BTreeMap::new(), BTreeMap::new())
            .unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::Resource);
        assert_eq!(error.reason_code, "element_budget");
        assert_eq!(error.trace.completed_node_count, 0);
    }

    #[test]
    fn grouped_last_split_rank3_keeps_rows_independent() {
        let graph = grouped_last_split_graph(
            &[1, 2, 6],
            &[&[1, 2, 2], &[1, 2, 4]],
            NumericalFormat::F32,
            TensorLayout::RowMajor,
            2,
            vec![1, 2],
        );
        let run = executor()
            .execute(
                &graph,
                map([(
                    value_id("split-input"),
                    f32(
                        &[1, 2, 6],
                        TensorLayout::RowMajor,
                        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                    ),
                )]),
                BTreeMap::new(),
            )
            .unwrap();
        assert_f32(
            &run.outputs[&value_id("split-output-0")],
            &[0.0, 3.0, 6.0, 9.0],
            0.0,
        );
        assert_f32(
            &run.outputs[&value_id("split-output-1")],
            &[1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0, 11.0],
            0.0,
        );
    }

    #[test]
    fn grouped_last_split_rejects_non_cpu_capabilities_before_payload() {
        for graph in [
            grouped_last_split_graph(
                &[1, 4],
                &[&[1, 2], &[1, 2]],
                NumericalFormat::Bf16,
                TensorLayout::RowMajor,
                2,
                vec![1, 1],
            ),
            grouped_last_split_graph(
                &[1, 4],
                &[&[1, 2], &[1, 2]],
                NumericalFormat::F32,
                TensorLayout::PackedRagged,
                2,
                vec![1, 1],
            ),
        ] {
            graph.validate().unwrap();
            let error = executor()
                .execute_traced(&graph, BTreeMap::new(), BTreeMap::new())
                .unwrap_err();
            assert_eq!(error.class, CpuReferenceFailureClass::Unsupported);
            assert_eq!(error.reason_code, "value_layout");
            assert_eq!(error.trace.completed_node_count, 0);
            assert_eq!(
                error.failed_node.as_ref().unwrap().kind,
                CpuReferenceNodeKind::GroupedLastSplit
            );
        }
    }

    #[test]
    fn grouped_last_split_preflight_accounts_outputs_and_rejects_resource_limit() {
        let graph = grouped_last_split_graph(
            &[2, 18],
            &[&[2, 6], &[2, 3], &[2, 9]],
            NumericalFormat::F32,
            TensorLayout::RowMajor,
            3,
            vec![2, 1, 3],
        );
        let value_specs = graph
            .values
            .iter()
            .map(|value| (value.id.clone(), &value.tensor))
            .collect();
        let plan = preflight_graph(&graph, &value_specs, &BTreeMap::new()).unwrap();
        assert_eq!(plan.execution_elements, 36);
        assert_eq!(plan.work_units, 36);

        let too_large = MAX_CPU_REFERENCE_TENSOR_ELEMENTS + 1;
        let resource = grouped_last_split_graph(
            &[1, too_large + 1],
            &[&[1, too_large], &[1, 1]],
            NumericalFormat::F32,
            TensorLayout::RowMajor,
            1,
            vec![too_large, 1],
        );
        resource.validate().unwrap();
        let error = executor()
            .execute_traced(&resource, BTreeMap::new(), BTreeMap::new())
            .unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::Resource);
        assert_eq!(error.reason_code, "element_budget");
        assert_eq!(error.trace.completed_node_count, 0);
    }

    #[test]
    fn gated_multiply_sigmoid_matches_literals_extremes_and_legacy() {
        let graph = gated_multiply_graph(&[1, 5], NumericalFormat::F32, TensorLayout::TokensHidden);
        let inputs = map([
            (
                value_id("multiply-value"),
                f32(
                    &[1, 5],
                    TensorLayout::TokensHidden,
                    &[2.0, -2.0, 4.0, -4.0, 1.0],
                ),
            ),
            (
                value_id("multiply-gate"),
                f32(
                    &[1, 5],
                    TensorLayout::TokensHidden,
                    &[0.0, 1.0, -1.0, 2.0, -2.0],
                ),
            ),
        ]);
        let traced = executor()
            .execute_traced(&graph, inputs.clone(), BTreeMap::new())
            .unwrap();
        assert_f32(
            &traced.outputs[&value_id("multiply-output")],
            &[1.0, -1.462_117_2, 1.075_765_7, -3.523_188_4, 0.119_202_92],
            2e-6,
        );
        assert_eq!(
            traced.trace.completed_nodes[0].kind,
            CpuReferenceNodeKind::GatedMultiply
        );
        let legacy = executor().execute(&graph, inputs, BTreeMap::new()).unwrap();
        assert_eq!(legacy.outputs, traced.outputs);

        let extremes = gated_multiply_graph(&[1, 2], NumericalFormat::F32, TensorLayout::RowMajor);
        let extreme_run = executor()
            .execute(
                &extremes,
                map([
                    (
                        value_id("multiply-value"),
                        f32(&[1, 2], TensorLayout::RowMajor, &[3.0, 3.0]),
                    ),
                    (
                        value_id("multiply-gate"),
                        f32(&[1, 2], TensorLayout::RowMajor, &[100.0, -100.0]),
                    ),
                ]),
                BTreeMap::new(),
            )
            .unwrap();
        assert_f32(
            &extreme_run.outputs[&value_id("multiply-output")],
            &[3.0, 0.0],
            1e-6,
        );
        let (_, _, extreme_values) = extreme_run.outputs[&value_id("multiply-output")]
            .f32_parts()
            .unwrap();
        assert!(extreme_values[1].is_finite());
        assert!(extreme_values[1] > 0.0);
    }

    #[test]
    fn gated_multiply_rejects_non_cpu_capabilities_and_nonfinite_input() {
        for graph in [
            gated_multiply_graph(&[1, 2], NumericalFormat::Fp16, TensorLayout::RowMajor),
            gated_multiply_graph(&[1, 2], NumericalFormat::F32, TensorLayout::PackedRagged),
        ] {
            graph.validate().unwrap();
            let error = executor()
                .execute_traced(&graph, BTreeMap::new(), BTreeMap::new())
                .unwrap_err();
            assert_eq!(error.class, CpuReferenceFailureClass::Unsupported);
            assert_eq!(error.reason_code, "value_layout");
            assert_eq!(error.trace.completed_node_count, 0);
            assert_eq!(
                error.failed_node.as_ref().unwrap().kind,
                CpuReferenceNodeKind::GatedMultiply
            );
        }

        let graph = gated_multiply_graph(&[1, 2], NumericalFormat::F32, TensorLayout::RowMajor);
        let error = executor()
            .execute_traced(
                &graph,
                map([
                    (
                        value_id("multiply-value"),
                        f32(&[1, 2], TensorLayout::RowMajor, &[1.0, f32::NAN]),
                    ),
                    (
                        value_id("multiply-gate"),
                        f32(&[1, 2], TensorLayout::RowMajor, &[0.0, 0.0]),
                    ),
                ]),
                BTreeMap::new(),
            )
            .unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::Numerical);
        assert_eq!(error.reason_code, "nonfinite_input");
        assert_eq!(error.trace.completed_node_count, 0);
        assert!(error.failed_node.is_none());
    }

    #[test]
    fn gated_multiply_preflight_rejects_element_and_work_budgets() {
        let too_large = MAX_CPU_REFERENCE_TENSOR_ELEMENTS + 1;
        let resource = gated_multiply_graph(
            &[1, too_large],
            NumericalFormat::F32,
            TensorLayout::RowMajor,
        );
        resource.validate().unwrap();
        let error = executor()
            .execute_traced(&resource, BTreeMap::new(), BTreeMap::new())
            .unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::Resource);
        assert_eq!(error.reason_code, "element_budget");

        let width = MAX_CPU_REFERENCE_TENSOR_ELEMENTS;
        let mut work =
            gated_multiply_graph(&[1, width], NumericalFormat::F32, TensorLayout::RowMajor);
        work.values.push(value(
            "multiply-output-2",
            &[1, width],
            NumericalFormat::F32,
            TensorLayout::RowMajor,
        ));
        work.outputs = vec![value_id("multiply-output-2")];
        work.nodes.push(GraphNode {
            id: node_id("gated-multiply-2"),
            inputs: vec![value_id("multiply-output"), value_id("multiply-gate")],
            outputs: vec![value_id("multiply-output-2")],
            weights: vec![],
            states: vec![],
            kind: GraphNodeKind::GatedMultiply {
                activation: ActivationKind::Sigmoid,
            },
        });
        work.validate().unwrap();
        let error = executor()
            .execute_traced(&work, BTreeMap::new(), BTreeMap::new())
            .unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::Resource);
        assert_eq!(error.reason_code, "work_budget");
        assert_eq!(
            error.failed_node.as_ref().unwrap().id,
            node_id("gated-multiply-2")
        );
        assert_eq!(error.trace.completed_node_count, 0);
    }

    #[test]
    fn rms_norm_scale_matches_hand_computation_and_legacy_wrapper() {
        let graph = rms_norm_graph(
            false,
            NormalizationAffine::Scale,
            TensorLayout::TokensHidden,
            TensorLayout::TokensHidden,
        );
        let inputs = map([(
            value_id("input"),
            f32(
                &[2, 3],
                TensorLayout::TokensHidden,
                &[3.0, 4.0, 0.0, 1.0, 2.0, 2.0],
            ),
        )]);
        let weights = map([(
            weight_id("scale"),
            f32(&[3], TensorLayout::RowMajor, &[1.0, 2.0, 0.5]),
        )]);

        let traced = executor()
            .execute_traced(&graph, inputs.clone(), weights.clone())
            .unwrap();
        assert_eq!(traced.trace.completed_node_count, 1);
        assert_eq!(
            traced.trace.completed_nodes[0].kind,
            CpuReferenceNodeKind::Norm
        );
        // Hand-computed from inv=1/sqrt(sum(x*x)/3 + 1e-5), then multiplied by scale.
        assert_f32(
            &traced.outputs[&value_id("out")],
            &[
                1.039_229_9,
                2.771_279_6,
                0.0,
                0.577_349_3,
                2.309_397_2,
                0.577_349_3,
            ],
            3e-5,
        );

        let legacy = executor().execute(&graph, inputs, weights).unwrap();
        assert_eq!(legacy.executed_node_ids, vec![node_id("norm")]);
        assert_eq!(legacy.outputs, traced.outputs);
    }

    #[test]
    fn grouped_rms_norm_uses_independent_groups_and_shared_scale() {
        let grouped = grouped_rms_norm_graph(
            &[1, 4],
            NormalizationAffine::Scale,
            TensorLayout::TokensHidden,
            2,
            2,
        );
        let inputs = map([(
            value_id("input"),
            f32(&[1, 4], TensorLayout::TokensHidden, &[3.0, 4.0, 5.0, 12.0]),
        )]);
        let grouped_weights = map([(
            weight_id("scale"),
            f32(&[2], TensorLayout::RowMajor, &[2.0, 0.5]),
        )]);
        let traced = executor()
            .execute_traced(&grouped, inputs.clone(), grouped_weights.clone())
            .unwrap();
        assert_f32(
            &traced.outputs[&value_id("out")],
            &[1.697_055_6, 0.565_685_2, 1.087_856_5, 0.652_713_9],
            3e-5,
        );
        let legacy = executor()
            .execute(&grouped, inputs.clone(), grouped_weights)
            .unwrap();
        assert_eq!(legacy.outputs, traced.outputs);

        let mut last = rms_norm_graph(
            false,
            NormalizationAffine::Scale,
            TensorLayout::TokensHidden,
            TensorLayout::TokensHidden,
        );
        for value in &mut last.values {
            value.tensor.shape = vec![1, 4];
        }
        last.weights[0].tensor.shape = vec![4];
        let last_run = executor()
            .execute(
                &last,
                inputs,
                map([(
                    weight_id("scale"),
                    f32(&[4], TensorLayout::RowMajor, &[2.0, 0.5, 2.0, 0.5]),
                )]),
            )
            .unwrap();
        assert_f32(
            &last_run.outputs[&value_id("out")],
            &[0.861_549_7, 0.287_183_23, 1.435_916_2, 0.861_549_7],
            3e-5,
        );
        assert_ne!(last_run.outputs, traced.outputs);
    }

    #[test]
    fn l2_normalization_literals_grouped_fixed_scale_rank3_and_legacy() {
        let graph = l2_norm_graph(
            &[1, 2],
            NormalizationAxis::Last,
            NormalizationAffine::None,
            TensorLayout::RowMajor,
        );
        let inputs = map([(
            value_id("l2-input"),
            f32(&[1, 2], TensorLayout::RowMajor, &[-3.0, 4.0]),
        )]);
        let traced = executor()
            .execute_traced(&graph, inputs.clone(), BTreeMap::new())
            .unwrap();
        assert_f32(
            &traced.outputs[&value_id("l2-output")],
            &[-0.599_999_9, 0.799_999_83],
            2e-6,
        );
        assert_eq!(
            executor()
                .execute(&graph, inputs, BTreeMap::new())
                .unwrap()
                .outputs,
            traced.outputs
        );

        let grouped = l2_norm_graph(
            &[1, 2, 4],
            NormalizationAxis::GroupedLast {
                groups: 2,
                group_width: 2,
            },
            NormalizationAffine::FixedScale(PositiveF32::new(2.0, "scale").unwrap()),
            TensorLayout::TokensHidden,
        );
        let run = executor()
            .execute(
                &grouped,
                map([(
                    value_id("l2-input"),
                    f32(
                        &[1, 2, 4],
                        TensorLayout::TokensHidden,
                        &[3.0, 4.0, 5.0, 12.0, 0.0, 2.0, 8.0, 6.0],
                    ),
                )]),
                BTreeMap::new(),
            )
            .unwrap();
        assert_f32(
            &run.outputs[&value_id("l2-output")],
            &[
                1.199_999_8,
                1.599_999_7,
                0.769_230_7,
                1.846_153_7,
                0.0,
                1.999_997_5,
                1.599_999_9,
                1.199_999_9,
            ],
            3e-6,
        );
    }

    #[test]
    fn l2_normalization_preflight_work_resource_and_numerical_are_typed() {
        for (affine, expected_work) in [
            (NormalizationAffine::None, 18_u64),
            (
                NormalizationAffine::FixedScale(PositiveF32::new(2.0, "scale").unwrap()),
                22_u64,
            ),
        ] {
            let graph = l2_norm_graph(
                &[2, 2],
                NormalizationAxis::Last,
                affine,
                TensorLayout::RowMajor,
            );
            let specs = graph
                .values
                .iter()
                .map(|value| (value.id.clone(), &value.tensor))
                .collect();
            let plan = preflight_graph(&graph, &specs, &BTreeMap::new()).unwrap();
            assert_eq!(plan.execution_elements, 4);
            assert_eq!(plan.work_units, expected_work);
        }
        let large = MAX_CPU_REFERENCE_TENSOR_ELEMENTS + 1;
        let resource = l2_norm_graph(
            &[1, large],
            NormalizationAxis::Last,
            NormalizationAffine::None,
            TensorLayout::RowMajor,
        );
        let error = executor()
            .execute_traced(&resource, BTreeMap::new(), BTreeMap::new())
            .unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::Resource);
        let numerical = l2_norm_graph(
            &[1, 2],
            NormalizationAxis::Last,
            NormalizationAffine::None,
            TensorLayout::RowMajor,
        );
        let error = executor()
            .execute_traced(
                &numerical,
                map([(
                    value_id("l2-input"),
                    f32(&[1, 2], TensorLayout::RowMajor, &[f32::MAX, 1.0]),
                )]),
                BTreeMap::new(),
            )
            .unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::Numerical);
        assert_eq!(error.reason_code, "runtime_numerical");
    }

    #[test]
    fn grouped_rms_norm_rank3_unit_offset_zero_weight_is_identity_affine() {
        let graph = grouped_rms_norm_graph(
            &[1, 2, 4],
            NormalizationAffine::UnitOffsetScale,
            TensorLayout::RowMajor,
            2,
            2,
        );
        let run = executor()
            .execute_traced(
                &graph,
                map([(
                    value_id("input"),
                    f32(
                        &[1, 2, 4],
                        TensorLayout::RowMajor,
                        &[3.0, 4.0, 5.0, 12.0, 0.0, 2.0, 8.0, 6.0],
                    ),
                )]),
                map([(
                    weight_id("scale"),
                    f32(&[2], TensorLayout::RowMajor, &[0.0, 0.0]),
                )]),
            )
            .unwrap();
        assert_f32(
            &run.outputs[&value_id("out")],
            &[
                0.848_527_8,
                1.131_370_4,
                0.543_928_3,
                1.305_427_9,
                0.0,
                1.414_21,
                1.131_370_7,
                0.848_528,
            ],
            3e-5,
        );
    }

    #[test]
    fn grouped_rms_norm_scale_and_bias_are_shared_across_groups() {
        let graph = grouped_rms_norm_graph(
            &[1, 4],
            NormalizationAffine::ScaleAndBias,
            TensorLayout::RowMajor,
            2,
            2,
        );
        let run = executor()
            .execute(
                &graph,
                map([(
                    value_id("input"),
                    f32(&[1, 4], TensorLayout::RowMajor, &[3.0, 4.0, 5.0, 12.0]),
                )]),
                map([
                    (
                        weight_id("scale"),
                        f32(&[2], TensorLayout::RowMajor, &[1.0, 2.0]),
                    ),
                    (
                        weight_id("bias"),
                        f32(&[2], TensorLayout::RowMajor, &[0.5, -1.0]),
                    ),
                ]),
            )
            .unwrap();
        assert_f32(
            &run.outputs[&value_id("out")],
            &[1.348_527_8, 1.262_740_8, 1.043_928_3, 1.610_855_8],
            3e-5,
        );
    }

    #[test]
    fn grouped_rms_norm_nonfinite_affine_output_is_numerical() {
        let graph = grouped_rms_norm_graph(
            &[1, 2],
            NormalizationAffine::Scale,
            TensorLayout::RowMajor,
            1,
            2,
        );
        let error = executor()
            .execute_traced(
                &graph,
                map([(
                    value_id("input"),
                    f32(&[1, 2], TensorLayout::RowMajor, &[2.0, 0.0]),
                )]),
                map([(
                    weight_id("scale"),
                    f32(&[2], TensorLayout::RowMajor, &[f32::MAX, 1.0]),
                )]),
            )
            .unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::Numerical);
        assert_eq!(error.reason_code, "runtime_numerical");
        assert_eq!(error.trace.completed_node_count, 0);
        assert_eq!(error.failed_node.as_ref().unwrap().id, node_id("norm"));
        assert!(error.message.contains("affine output is non-finite"));
    }

    #[test]
    fn grouped_rms_norm_invalid_axis_fails_graph_and_preflight_contract() {
        let mut graph = grouped_rms_norm_graph(
            &[1, 4],
            NormalizationAffine::Scale,
            TensorLayout::RowMajor,
            2,
            2,
        );
        if let GraphNodeKind::Norm { axis, .. } = &mut graph.nodes[0].kind {
            *axis = NormalizationAxis::GroupedLast {
                groups: 3,
                group_width: 2,
            };
        } else {
            panic!("grouped RMS normalization test graph must contain Norm");
        }
        assert!(graph.validate().unwrap_err().contains("final width"));

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
        assert_eq!(error.class, CpuReferenceFailureClass::Internal);
        assert_eq!(error.reason_code, "node_contract");
        assert_eq!(error.failed_node.as_ref().unwrap().id, node_id("norm"));
    }

    #[test]
    fn grouped_rms_norm_preflight_counts_each_group_as_a_normalization_unit() {
        for (affine, expected_work) in [
            (NormalizationAffine::Scale, 80_u64),
            (NormalizationAffine::UnitOffsetScale, 96_u64),
            (NormalizationAffine::ScaleAndBias, 96_u64),
        ] {
            let graph = grouped_rms_norm_graph(&[2, 8], affine, TensorLayout::RowMajor, 2, 4);
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
            let plan = preflight_graph(&graph, &value_specs, &weight_specs).unwrap();
            assert_eq!(plan.execution_elements, 16);
            assert_eq!(plan.work_units, expected_work);
        }
    }

    #[test]
    fn rms_norm_unit_offset_scale_with_zero_weight_is_not_zero() {
        let graph = rms_norm_graph(
            false,
            NormalizationAffine::UnitOffsetScale,
            TensorLayout::TokensHidden,
            TensorLayout::TokensHidden,
        );
        let run = executor()
            .execute_traced(
                &graph,
                map([(
                    value_id("input"),
                    f32(
                        &[2, 3],
                        TensorLayout::TokensHidden,
                        &[3.0, 4.0, 0.0, 1.0, 2.0, 2.0],
                    ),
                )]),
                map([(
                    weight_id("scale"),
                    f32(&[3], TensorLayout::RowMajor, &[0.0, 0.0, 0.0]),
                )]),
            )
            .unwrap();
        assert_f32(
            &run.outputs[&value_id("out")],
            &[
                1.039_229_9,
                1.385_639_8,
                0.0,
                0.577_349_3,
                1.154_698_6,
                1.154_698_6,
            ],
            3e-5,
        );
        let (_, _, values) = run.outputs[&value_id("out")].f32_parts().unwrap();
        assert!(values.iter().any(|value| value.abs() > 0.5));
    }

    #[test]
    fn rms_norm_scale_and_bias_final_norm_executes() {
        let graph = rms_norm_graph(
            true,
            NormalizationAffine::ScaleAndBias,
            TensorLayout::TokensHidden,
            TensorLayout::TokensHidden,
        );
        let run = executor()
            .execute_traced(
                &graph,
                map([(
                    value_id("input"),
                    f32(
                        &[2, 3],
                        TensorLayout::TokensHidden,
                        &[1.0, 2.0, 2.0, 3.0, 0.0, 4.0],
                    ),
                )]),
                map([
                    (
                        weight_id("scale"),
                        f32(&[3], TensorLayout::RowMajor, &[2.0, 0.5, -1.0]),
                    ),
                    (
                        weight_id("bias"),
                        f32(&[3], TensorLayout::RowMajor, &[0.5, -1.0, 2.0]),
                    ),
                ]),
            )
            .unwrap();
        assert_eq!(
            run.trace.completed_nodes,
            vec![CpuReferenceNodeRef {
                id: node_id("final-norm"),
                kind: CpuReferenceNodeKind::FinalNorm,
            }]
        );
        assert_f32(
            &run.outputs[&value_id("out")],
            &[
                1.654_698_6,
                -0.422_650_7,
                0.845_301_4,
                2.578_459_7,
                -1.0,
                0.614_360_2,
            ],
            3e-5,
        );
    }

    #[test]
    fn layer_normalization_is_typed_unsupported_before_execution() {
        let mut graph = rms_norm_graph(
            false,
            NormalizationAffine::Scale,
            TensorLayout::TokensHidden,
            TensorLayout::TokensHidden,
        );
        if let GraphNodeKind::Norm { kind, .. } = &mut graph.nodes[0].kind {
            *kind = NormalizationKind::Layer;
        } else {
            panic!("RMS normalization test graph must contain Norm");
        }
        let error = executor()
            .execute_traced(
                &graph,
                map([(
                    value_id("input"),
                    f32(&[2, 3], TensorLayout::TokensHidden, &[1.0; 6]),
                )]),
                map([(
                    weight_id("scale"),
                    f32(&[3], TensorLayout::RowMajor, &[1.0; 3]),
                )]),
            )
            .unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::Unsupported);
        assert_eq!(error.reason_code, "normalization_kind");
        assert_eq!(error.trace.completed_node_count, 0);
        assert_eq!(
            error.failed_node,
            Some(CpuReferenceNodeRef {
                id: node_id("norm"),
                kind: CpuReferenceNodeKind::Norm,
            })
        );
        assert!(
            error
                .message
                .contains("Layer normalization is not in the P1-B2 CPU subset")
        );
    }

    #[test]
    fn rms_norm_shape_is_graph_invalid_and_layout_mismatch_is_preflight_unsupported() {
        let mut shape_mismatch = rms_norm_graph(
            false,
            NormalizationAffine::Scale,
            TensorLayout::TokensHidden,
            TensorLayout::TokensHidden,
        );
        shape_mismatch.values[1].tensor.shape = vec![2, 2];
        assert!(
            shape_mismatch
                .validate()
                .unwrap_err()
                .contains("normalization input and output")
        );

        let layout_mismatch = rms_norm_graph(
            false,
            NormalizationAffine::Scale,
            TensorLayout::TokensHidden,
            TensorLayout::RowMajor,
        );
        let error = executor()
            .execute_traced(
                &layout_mismatch,
                map([(
                    value_id("input"),
                    f32(&[2, 3], TensorLayout::TokensHidden, &[1.0; 6]),
                )]),
                map([(
                    weight_id("scale"),
                    f32(&[3], TensorLayout::RowMajor, &[1.0; 3]),
                )]),
            )
            .unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::Unsupported);
        assert_eq!(error.reason_code, "layout_mismatch");
        assert_eq!(error.trace.completed_node_count, 0);
        assert_eq!(
            error.failed_node,
            Some(CpuReferenceNodeRef {
                id: node_id("norm"),
                kind: CpuReferenceNodeKind::Norm,
            })
        );
    }

    #[test]
    fn rms_norm_intermediate_overflow_is_numerical_and_keeps_the_completed_prefix() {
        let graph = ModelGraph {
            graph_id: "linear-then-rms-norm".into(),
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
                    "hidden",
                    &[1, 3],
                    NumericalFormat::F32,
                    TensorLayout::TokensHidden,
                ),
                value(
                    "out",
                    &[1, 3],
                    NumericalFormat::F32,
                    TensorLayout::TokensHidden,
                ),
            ],
            weights: vec![weight("linear", &[3, 3]), weight("scale", &[3])],
            nodes: vec![
                GraphNode {
                    id: node_id("linear"),
                    inputs: vec![value_id("input")],
                    outputs: vec![value_id("hidden")],
                    weights: vec![weight_id("linear")],
                    states: vec![],
                    kind: GraphNodeKind::Linear { has_bias: false },
                },
                GraphNode {
                    id: node_id("norm"),
                    inputs: vec![value_id("hidden")],
                    outputs: vec![value_id("out")],
                    weights: vec![weight_id("scale")],
                    states: vec![],
                    kind: GraphNodeKind::Norm {
                        epsilon: PositiveF32::new(1e-5, "epsilon").unwrap(),
                        kind: NormalizationKind::Rms,
                        affine: NormalizationAffine::Scale,
                        axis: NormalizationAxis::Last,
                    },
                },
            ],
        };
        let error = executor()
            .execute_traced(
                &graph,
                map([(
                    value_id("input"),
                    f32(&[1, 3], TensorLayout::TokensHidden, &[f32::MAX, 1.0, 1.0]),
                )]),
                map([
                    (
                        weight_id("linear"),
                        f32(
                            &[3, 3],
                            TensorLayout::RowMajor,
                            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        ),
                    ),
                    (
                        weight_id("scale"),
                        f32(&[3], TensorLayout::RowMajor, &[1.0, 1.0, 1.0]),
                    ),
                ]),
            )
            .unwrap_err();
        assert_eq!(error.class, CpuReferenceFailureClass::Numerical);
        assert_eq!(error.reason_code, "runtime_numerical");
        assert!(error.message.contains("square is non-finite"));
        assert_eq!(error.trace.completed_node_count, 1);
        assert_eq!(error.trace.completed_nodes[0].id, node_id("linear"));
        assert_eq!(
            error.failed_node,
            Some(CpuReferenceNodeRef {
                id: node_id("norm"),
                kind: CpuReferenceNodeKind::Norm,
            })
        );
    }

    #[test]
    fn rms_norm_preflight_charges_width_one_rows_above_the_old_scalar_only_budget() {
        for (affine, expected_work, old_scalar_only_work) in [
            (NormalizationAffine::Scale, 32_u64, 24_u64),
            (NormalizationAffine::UnitOffsetScale, 36_u64, 28_u64),
            (NormalizationAffine::ScaleAndBias, 36_u64, 28_u64),
        ] {
            let mut graph = rms_norm_graph(
                false,
                affine,
                TensorLayout::RowMajor,
                TensorLayout::RowMajor,
            );
            for value in &mut graph.values {
                value.tensor.shape = vec![4, 1];
            }
            for weight in &mut graph.weights {
                weight.tensor.shape = vec![1];
            }
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
            let plan = preflight_graph(&graph, &value_specs, &weight_specs).unwrap();
            assert_eq!(plan.execution_elements, 4);
            assert_eq!(plan.work_units, expected_work);
            assert!(plan.work_units > old_scalar_only_work);
        }
    }

    #[test]
    fn rms_norm_preflight_rejects_row_work_budget_without_executing_payloads() {
        const ROWS: usize = 1_000_000;
        const NODES: usize = 7;
        for axis in [
            NormalizationAxis::Last,
            NormalizationAxis::GroupedLast {
                groups: 1,
                group_width: 1,
            },
        ] {
            let mut values = Vec::new();
            values.push(value(
                "value-0",
                &[ROWS, 1],
                NumericalFormat::F32,
                TensorLayout::RowMajor,
            ));
            for index in 1..=NODES {
                values.push(value(
                    &format!("value-{index}"),
                    &[ROWS, 1],
                    NumericalFormat::F32,
                    TensorLayout::RowMajor,
                ));
            }
            let nodes = (0..NODES)
                .map(|index| GraphNode {
                    id: node_id(&format!("norm-{index}")),
                    inputs: vec![value_id(&format!("value-{index}"))],
                    outputs: vec![value_id(&format!("value-{}", index + 1))],
                    weights: vec![weight_id("scale")],
                    states: vec![],
                    kind: GraphNodeKind::Norm {
                        epsilon: PositiveF32::new(1e-5, "epsilon").unwrap(),
                        kind: NormalizationKind::Rms,
                        affine: NormalizationAffine::Scale,
                        axis,
                    },
                })
                .collect();
            let graph = ModelGraph {
                graph_id: "rms-work-budget".into(),
                inputs: vec![value_id("value-0")],
                outputs: vec![value_id(&format!("value-{NODES}"))],
                values,
                weights: vec![weight("scale", &[1])],
                nodes,
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
            assert_eq!(error.class, CpuReferenceFailureClass::Resource);
            assert_eq!(error.reason_code, "work_budget");
            assert_eq!(error.failed_node.as_ref().unwrap().id, node_id("norm-6"));
            assert!(error.message.contains("work-unit budget"));
        }
    }

    #[test]
    fn rms_norm_accepts_row_major_and_multidimensional_width_one() {
        let row_major = rms_norm_graph(
            false,
            NormalizationAffine::Scale,
            TensorLayout::RowMajor,
            TensorLayout::RowMajor,
        );
        let row_major_run = executor()
            .execute_traced(
                &row_major,
                map([(
                    value_id("input"),
                    f32(
                        &[2, 3],
                        TensorLayout::RowMajor,
                        &[1.0, 2.0, 2.0, 3.0, 4.0, 0.0],
                    ),
                )]),
                map([(
                    weight_id("scale"),
                    f32(&[3], TensorLayout::RowMajor, &[1.0, 1.0, 1.0]),
                )]),
            )
            .unwrap();
        assert_f32(
            &row_major_run.outputs[&value_id("out")],
            &[
                0.577_349_3,
                1.154_698_6,
                1.154_698_6,
                1.039_229_9,
                1.385_639_8,
                0.0,
            ],
            3e-5,
        );

        let mut width_one = rms_norm_graph(
            false,
            NormalizationAffine::Scale,
            TensorLayout::RowMajor,
            TensorLayout::RowMajor,
        );
        for value in &mut width_one.values {
            value.tensor.shape = vec![2, 2, 1];
        }
        width_one.weights[0].tensor.shape = vec![1];
        width_one.validate().unwrap();
        let width_one_run = executor()
            .execute_traced(
                &width_one,
                map([(
                    value_id("input"),
                    f32(&[2, 2, 1], TensorLayout::RowMajor, &[1.0, 2.0, 3.0, 4.0]),
                )]),
                map([(
                    weight_id("scale"),
                    f32(&[1], TensorLayout::RowMajor, &[2.0]),
                )]),
            )
            .unwrap();
        assert_f32(
            &width_one_run.outputs[&value_id("out")],
            &[1.999_99, 1.999_997_5, 1.999_998_9, 1.999_999_4],
            3e-5,
        );
    }

    fn rms_norm_graph(
        final_norm: bool,
        affine: NormalizationAffine,
        input_layout: TensorLayout,
        output_layout: TensorLayout,
    ) -> ModelGraph {
        let mut weights = vec![weight("scale", &[3])];
        let mut node_weights = vec![weight_id("scale")];
        if affine == NormalizationAffine::ScaleAndBias {
            weights.push(weight("bias", &[3]));
            node_weights.push(weight_id("bias"));
        }
        let epsilon = PositiveF32::new(1e-5, "epsilon").unwrap();
        let kind = if final_norm {
            GraphNodeKind::FinalNorm {
                epsilon,
                kind: NormalizationKind::Rms,
                affine,
                axis: NormalizationAxis::Last,
            }
        } else {
            GraphNodeKind::Norm {
                epsilon,
                kind: NormalizationKind::Rms,
                affine,
                axis: NormalizationAxis::Last,
            }
        };
        ModelGraph {
            graph_id: if final_norm {
                "final-rms-norm".into()
            } else {
                "rms-norm".into()
            },
            inputs: vec![value_id("input")],
            outputs: vec![value_id("out")],
            values: vec![
                value("input", &[2, 3], NumericalFormat::F32, input_layout),
                value("out", &[2, 3], NumericalFormat::F32, output_layout),
            ],
            weights,
            nodes: vec![GraphNode {
                id: node_id(if final_norm { "final-norm" } else { "norm" }),
                inputs: vec![value_id("input")],
                outputs: vec![value_id("out")],
                weights: node_weights,
                states: vec![],
                kind,
            }],
        }
    }

    fn grouped_rms_norm_graph(
        shape: &[usize],
        affine: NormalizationAffine,
        layout: TensorLayout,
        groups: usize,
        group_width: usize,
    ) -> ModelGraph {
        let mut weights = vec![weight("scale", &[group_width])];
        let mut node_weights = vec![weight_id("scale")];
        if affine == NormalizationAffine::ScaleAndBias {
            weights.push(weight("bias", &[group_width]));
            node_weights.push(weight_id("bias"));
        }
        ModelGraph {
            graph_id: "grouped-rms-norm".into(),
            inputs: vec![value_id("input")],
            outputs: vec![value_id("out")],
            values: vec![
                value("input", shape, NumericalFormat::F32, layout.clone()),
                value("out", shape, NumericalFormat::F32, layout),
            ],
            weights,
            nodes: vec![GraphNode {
                id: node_id("norm"),
                inputs: vec![value_id("input")],
                outputs: vec![value_id("out")],
                weights: node_weights,
                states: vec![],
                kind: GraphNodeKind::Norm {
                    epsilon: PositiveF32::new(1e-5, "epsilon").unwrap(),
                    kind: NormalizationKind::Rms,
                    affine,
                    axis: NormalizationAxis::GroupedLast {
                        groups,
                        group_width,
                    },
                },
            }],
        }
    }

    fn l2_norm_graph(
        shape: &[usize],
        axis: NormalizationAxis,
        affine: NormalizationAffine,
        layout: TensorLayout,
    ) -> ModelGraph {
        ModelGraph {
            graph_id: "l2-normalization".into(),
            inputs: vec![value_id("l2-input")],
            outputs: vec![value_id("l2-output")],
            values: vec![
                value("l2-input", shape, NumericalFormat::F32, layout.clone()),
                value("l2-output", shape, NumericalFormat::F32, layout),
            ],
            weights: vec![],
            nodes: vec![GraphNode {
                id: node_id("l2"),
                inputs: vec![value_id("l2-input")],
                outputs: vec![value_id("l2-output")],
                weights: vec![],
                states: vec![],
                kind: GraphNodeKind::Norm {
                    epsilon: PositiveF32::new(1e-5, "epsilon").unwrap(),
                    kind: NormalizationKind::L2,
                    affine,
                    axis,
                },
            }],
        }
    }

    fn grouped_last_split_graph(
        input_shape: &[usize],
        output_shapes: &[&[usize]],
        format: NumericalFormat,
        layout: TensorLayout,
        groups: usize,
        segment_widths: Vec<usize>,
    ) -> ModelGraph {
        let mut values = vec![value(
            "split-input",
            input_shape,
            format.clone(),
            layout.clone(),
        )];
        let outputs = output_shapes
            .iter()
            .enumerate()
            .map(|(index, shape)| {
                let name = format!("split-output-{index}");
                values.push(value(&name, shape, format.clone(), layout.clone()));
                value_id(&name)
            })
            .collect::<Vec<_>>();
        ModelGraph {
            graph_id: "grouped-last-split-reference".into(),
            inputs: vec![value_id("split-input")],
            outputs: outputs.clone(),
            values,
            weights: vec![],
            nodes: vec![GraphNode {
                id: node_id("split"),
                inputs: vec![value_id("split-input")],
                outputs,
                weights: vec![],
                states: vec![],
                kind: GraphNodeKind::GroupedLastSplit {
                    groups,
                    segment_widths,
                },
            }],
        }
    }

    fn gated_multiply_graph(
        shape: &[usize],
        format: NumericalFormat,
        layout: TensorLayout,
    ) -> ModelGraph {
        ModelGraph {
            graph_id: "gated-multiply-reference".into(),
            inputs: vec![value_id("multiply-value"), value_id("multiply-gate")],
            outputs: vec![value_id("multiply-output")],
            values: vec![
                value("multiply-value", shape, format.clone(), layout.clone()),
                value("multiply-gate", shape, format.clone(), layout.clone()),
                value("multiply-output", shape, format, layout),
            ],
            weights: vec![],
            nodes: vec![GraphNode {
                id: node_id("gated-multiply"),
                inputs: vec![value_id("multiply-value"), value_id("multiply-gate")],
                outputs: vec![value_id("multiply-output")],
                weights: vec![],
                states: vec![],
                kind: GraphNodeKind::GatedMultiply {
                    activation: ActivationKind::Sigmoid,
                },
            }],
        }
    }

    fn conv_graph(
        shape: &[usize],
        format: NumericalFormat,
        layout: TensorLayout,
        channels: usize,
        kernel_size: usize,
    ) -> ModelGraph {
        ModelGraph {
            graph_id: "conv-reference".into(),
            inputs: vec![value_id("x")],
            outputs: vec![value_id("y")],
            values: vec![
                value("x", shape, format.clone(), layout.clone()),
                value("y", shape, format.clone(), layout),
            ],
            weights: vec![WeightSpec {
                id: weight_id("kernel"),
                tensor: spec(&[channels, 1, kernel_size], format, TensorLayout::RowMajor),
            }],
            nodes: vec![GraphNode {
                id: node_id("conv-ref"),
                inputs: vec![value_id("x")],
                outputs: vec![value_id("y")],
                weights: vec![weight_id("kernel")],
                states: vec![],
                kind: GraphNodeKind::CausalDepthwiseConv1d {
                    channels,
                    kernel_size,
                },
            }],
        }
    }

    fn decay_graph(
        shape: &[usize],
        format: NumericalFormat,
        layout: TensorLayout,
        channels: usize,
    ) -> ModelGraph {
        ModelGraph {
            graph_id: "decay-reference".into(),
            inputs: vec![value_id("dc"), value_id("uc")],
            outputs: vec![value_id("ld"), value_id("ur")],
            values: vec![
                value("dc", shape, format.clone(), layout.clone()),
                value("uc", shape, format.clone(), layout.clone()),
                value("ld", shape, format.clone(), layout.clone()),
                value("ur", shape, format.clone(), layout),
            ],
            weights: vec![
                WeightSpec {
                    id: weight_id("lr"),
                    tensor: spec(&[channels], format.clone(), TensorLayout::RowMajor),
                },
                WeightSpec {
                    id: weight_id("tb"),
                    tensor: spec(&[channels], format, TensorLayout::RowMajor),
                },
            ],
            nodes: vec![GraphNode {
                id: node_id("decay-ref"),
                inputs: vec![value_id("dc"), value_id("uc")],
                outputs: vec![value_id("ld"), value_id("ur")],
                weights: vec![weight_id("lr"), weight_id("tb")],
                states: vec![],
                kind: GraphNodeKind::GatedDecayParameters { channels },
            }],
        }
    }

    fn scan_graph(
        shape_prefix: &[usize],
        format: NumericalFormat,
        layout: TensorLayout,
        kh: usize,
        vh: usize,
        kd: usize,
        vd: usize,
    ) -> ModelGraph {
        let mut key_shape = shape_prefix.to_vec();
        key_shape.push(kh * kd);
        let mut value_shape = shape_prefix.to_vec();
        value_shape.push(vh * vd);
        let mut head_shape = shape_prefix.to_vec();
        head_shape.push(vh);
        ModelGraph {
            graph_id: "scan-reference".into(),
            inputs: vec![
                value_id("q"),
                value_id("k"),
                value_id("v"),
                value_id("d"),
                value_id("b"),
            ],
            outputs: vec![value_id("o")],
            values: vec![
                value("q", &key_shape, format.clone(), layout.clone()),
                value("k", &key_shape, format.clone(), layout.clone()),
                value("v", &value_shape, format.clone(), layout.clone()),
                value("d", &head_shape, format.clone(), layout.clone()),
                value("b", &head_shape, format.clone(), layout.clone()),
                value("o", &value_shape, format, layout),
            ],
            weights: vec![],
            nodes: vec![GraphNode {
                id: node_id("scan-ref"),
                inputs: vec![
                    value_id("q"),
                    value_id("k"),
                    value_id("v"),
                    value_id("d"),
                    value_id("b"),
                ],
                outputs: vec![value_id("o")],
                weights: vec![],
                states: vec![],
                kind: GraphNodeKind::GatedDeltaRuleScan {
                    key_heads: kh,
                    value_heads: vh,
                    key_dim: kd,
                    value_dim: vd,
                },
            }],
        }
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

    fn rotary_graph(
        values_shape: &[usize],
        values_format: NumericalFormat,
        values_layout: TensorLayout,
        positions_format: NumericalFormat,
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        base: f32,
        pairing: RotaryPairing,
    ) -> ModelGraph {
        let positions_shape = values_shape[..values_shape.len() - 1].to_vec();
        ModelGraph {
            graph_id: "rotary-reference".into(),
            inputs: vec![value_id("values"), value_id("positions")],
            outputs: vec![value_id("out")],
            values: vec![
                value(
                    "values",
                    values_shape,
                    values_format.clone(),
                    values_layout.clone(),
                ),
                value(
                    "positions",
                    &positions_shape,
                    positions_format,
                    TensorLayout::RowMajor,
                ),
                value("out", values_shape, values_format, values_layout),
            ],
            weights: vec![],
            nodes: vec![GraphNode {
                id: node_id("rotary"),
                inputs: vec![value_id("values"), value_id("positions")],
                outputs: vec![value_id("out")],
                weights: vec![],
                states: vec![],
                kind: GraphNodeKind::RotaryPosition {
                    heads,
                    head_dim,
                    rotary_dim,
                    base: PositiveF32::new(base, "rotary base").unwrap(),
                    pairing,
                },
            }],
        }
    }

    fn rotary_oracle(
        input: &[f32],
        positions: &[u64],
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        base: f32,
        pairing: RotaryPairing,
    ) -> Vec<f32> {
        let hidden = heads * head_dim;
        let half = rotary_dim / 2;
        let mut expected = input.to_vec();
        for (row, position) in positions.iter().enumerate() {
            let row_start = row * hidden;
            for head in 0..heads {
                let head_start = row_start + head * head_dim;
                for index in 0..half {
                    let (a, b) = match pairing {
                        RotaryPairing::SplitHalf => (index, half + index),
                        RotaryPairing::Interleaved => (2 * index, 2 * index + 1),
                    };
                    let theta =
                        *position as f32 / base.powf((2.0 * index as f32) / rotary_dim as f32);
                    let (sin_theta, cos_theta) = theta.sin_cos();
                    let x_a = input[head_start + a];
                    let x_b = input[head_start + b];
                    expected[head_start + a] = x_a * cos_theta - x_b * sin_theta;
                    expected[head_start + b] = x_a * sin_theta + x_b * cos_theta;
                }
            }
        }
        expected
    }

    fn causal_gqa_graph(
        query_shape: &[usize],
        key_shape: &[usize],
        value_shape: &[usize],
        context_shape: &[usize],
        layout: TensorLayout,
        states: Vec<StateId>,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: f32,
    ) -> ModelGraph {
        ModelGraph {
            graph_id: "causal-gqa-reference".into(),
            inputs: vec![value_id("query"), value_id("key"), value_id("value")],
            outputs: vec![value_id("context")],
            values: vec![
                value("query", query_shape, NumericalFormat::F32, layout.clone()),
                value("key", key_shape, NumericalFormat::F32, layout.clone()),
                value("value", value_shape, NumericalFormat::F32, layout.clone()),
                value("context", context_shape, NumericalFormat::F32, layout),
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
                    softmax_scale: PositiveF32::new(softmax_scale, "softmax scale").unwrap(),
                },
            }],
        }
    }

    fn linear_then_embedding_graph() -> ModelGraph {
        ModelGraph {
            graph_id: "linear-then-embedding".into(),
            inputs: vec![value_id("pre"), value_id("tokens")],
            outputs: vec![value_id("embedded")],
            values: vec![
                value(
                    "pre",
                    &[1, 1],
                    NumericalFormat::F32,
                    TensorLayout::TokensHidden,
                ),
                value("tokens", &[1], NumericalFormat::U32, TensorLayout::RowMajor),
                value(
                    "linear-out",
                    &[1, 1],
                    NumericalFormat::F32,
                    TensorLayout::TokensHidden,
                ),
                value(
                    "embedded",
                    &[1, 1],
                    NumericalFormat::F32,
                    TensorLayout::TokensHidden,
                ),
            ],
            weights: vec![
                weight("linear-weight", &[1, 1]),
                weight("embedding", &[3, 1]),
            ],
            nodes: vec![
                GraphNode {
                    id: node_id("linear"),
                    inputs: vec![value_id("pre")],
                    outputs: vec![value_id("linear-out")],
                    weights: vec![weight_id("linear-weight")],
                    states: vec![],
                    kind: GraphNodeKind::Linear { has_bias: false },
                },
                GraphNode {
                    id: node_id("embedding"),
                    inputs: vec![value_id("tokens")],
                    outputs: vec![value_id("embedded")],
                    weights: vec![weight_id("embedding")],
                    states: vec![],
                    kind: GraphNodeKind::Embedding {
                        vocab_size: 3,
                        hidden_size: 1,
                    },
                },
            ],
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
