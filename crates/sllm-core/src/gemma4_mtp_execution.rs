//! Resident and request execution for the reviewed width-one Gemma 4 MTP
//! assistant.
//!
//! All assistant tensors are BF16 and separately resident.  Proposal calls
//! borrow [`crate::Gemma4MtpTargetKvLease`], so target KV is read-only for the
//! complete operation and no assistant KV allocation exists.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use crate::gemma4_execution::{Gemma4KvPlane, Gemma4MtpTargetKvLease};
use crate::gemma4_mtp_graph::{
    GEMMA4_MTP_BACKBONE_HIDDEN_SIZE, GEMMA4_MTP_HIDDEN_SIZE, GEMMA4_MTP_INTERMEDIATE_SIZE,
    GEMMA4_MTP_VOCAB_SIZE, Gemma4MtpGraph, Gemma4MtpGraphNodeKind, Gemma4MtpNormRole,
};
use crate::op::{RmsNormScaleMode, SemanticOpDescriptor, SemanticOpKind};
use crate::weights::{WeightConsumer, WeightLoadPlan, build_verified_gemma4_mtp_weight_load_plan};
use crate::{
    AccessMode, AllocationCategory, BoundSemanticOp, DType, Encoding, ExecutionBuffer,
    ExecutionQueue, ExecutionSession, ExecutionState, Gemma4MtpWeightSource, OwnedTensorBinding,
    TensorDType, TensorView,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MtpExecutionError(String);

impl Gemma4MtpExecutionError {
    fn invalid(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Gemma4MtpExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid Gemma 4 MTP execution: {}", self.0)
    }
}

impl std::error::Error for Gemma4MtpExecutionError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Gemma4MtpTensorBacking {
    AssistantWeight(String),
    TargetEmbedding,
    TargetSliding { plane: Gemma4KvPlane },
    TokenId,
    TargetHidden,
    Position,
    ConstantBf16 { bits: u16, width: usize },
    Workspace,
    Alias { tensor: usize },
    TerminalToken,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MtpExecutionTensor {
    id: usize,
    name: String,
    view: TensorView,
    backing: Gemma4MtpTensorBacking,
}

impl Gemma4MtpExecutionTensor {
    pub const fn id(&self) -> usize {
        self.id
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub const fn view(&self) -> &TensorView {
        &self.view
    }
    pub const fn backing(&self) -> &Gemma4MtpTensorBacking {
        &self.backing
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Gemma4MtpLowering {
    Semantic(Box<SemanticOpDescriptor>),
    OpaqueFullTargetAttention,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MtpExecutionNode {
    graph_node_id: usize,
    label: String,
    lowering: Gemma4MtpLowering,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
}

impl Gemma4MtpExecutionNode {
    pub const fn graph_node_id(&self) -> usize {
        self.graph_node_id
    }
    pub fn label(&self) -> &str {
        &self.label
    }
    pub const fn lowering(&self) -> &Gemma4MtpLowering {
        &self.lowering
    }
    pub fn inputs(&self) -> &[usize] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[usize] {
        &self.outputs
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MtpExecutionLayout {
    assistant_fingerprint: String,
    target_fingerprint: String,
    plan_digest: [u8; 32],
    target_kv_length: u64,
    tensors: Vec<Gemma4MtpExecutionTensor>,
    nodes: Vec<Gemma4MtpExecutionNode>,
    terminal_token_tensor: usize,
    draft_hidden_tensor: usize,
    target_hidden_tensor: usize,
    token_id_tensor: usize,
    position_tensor: usize,
    resident_bytes: u64,
    workspace_bytes: u64,
    assistant_kv_bytes: u64,
}

impl Gemma4MtpExecutionLayout {
    pub fn assistant_fingerprint(&self) -> &str {
        &self.assistant_fingerprint
    }
    pub fn target_fingerprint(&self) -> &str {
        &self.target_fingerprint
    }
    pub const fn plan_digest(&self) -> &[u8; 32] {
        &self.plan_digest
    }
    pub const fn target_kv_length(&self) -> u64 {
        self.target_kv_length
    }
    pub fn tensors(&self) -> &[Gemma4MtpExecutionTensor] {
        &self.tensors
    }
    pub fn nodes(&self) -> &[Gemma4MtpExecutionNode] {
        &self.nodes
    }
    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
    pub const fn workspace_bytes(&self) -> u64 {
        self.workspace_bytes
    }
    pub const fn assistant_kv_allocation_bytes(&self) -> u64 {
        self.assistant_kv_bytes
    }
}

fn contiguous(dtype: DType, shape: &[u64]) -> Result<TensorView, Gemma4MtpExecutionError> {
    let shape = shape
        .iter()
        .map(|dimension| usize::try_from(*dimension))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| Gemma4MtpExecutionError::invalid("tensor extent exceeds usize"))?;
    TensorView::contiguous(dtype, &shape)
        .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))
}

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let upper = bits >> 16;
    let lower = bits & 0xffff;
    (upper + u32::from(lower > 0x8000 || (lower == 0x8000 && upper & 1 != 0))) as u16
}

struct LayoutBuilder<'a> {
    graph: &'a Gemma4MtpGraph,
    plan: &'a WeightLoadPlan,
    tensors: Vec<Gemma4MtpExecutionTensor>,
    nodes: Vec<Gemma4MtpExecutionNode>,
    outputs: Vec<Vec<usize>>,
    weights: BTreeMap<String, usize>,
    target_embedding: usize,
    target_hidden: usize,
    token_id: usize,
    position: usize,
    concat: Option<usize>,
    workspace_bytes: u64,
}

impl<'a> LayoutBuilder<'a> {
    fn new(
        graph: &'a Gemma4MtpGraph,
        plan: &'a WeightLoadPlan,
    ) -> Result<Self, Gemma4MtpExecutionError> {
        if graph.assistant_fingerprint() != plan.lock_fingerprint
            || graph.weight_plan_digest() != plan.digest()
            || !plan
                .has_valid_digest()
                .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?
        {
            return Err(Gemma4MtpExecutionError::invalid(
                "assistant graph and weight plan identity differ",
            ));
        }
        let mut builder = Self {
            graph,
            plan,
            tensors: Vec::new(),
            nodes: Vec::new(),
            outputs: vec![Vec::new(); graph.nodes().len()],
            weights: BTreeMap::new(),
            target_embedding: usize::MAX,
            target_hidden: usize::MAX,
            token_id: usize::MAX,
            position: usize::MAX,
            concat: None,
            workspace_bytes: 0,
        };
        let mut names = BTreeSet::new();
        for entry in &plan.entries {
            if !names.insert(entry.tensor_name.as_str())
                || entry.dtype != TensorDType::Bf16
                || entry.destination_start.is_none()
            {
                return Err(Gemma4MtpExecutionError::invalid(
                    "assistant resident tensor metadata differs",
                ));
            }
            let view = contiguous(DType::Bf16, &entry.shape)?;
            let id = builder.push_tensor(
                entry.tensor_name.clone(),
                view,
                Gemma4MtpTensorBacking::AssistantWeight(entry.tensor_name.clone()),
            );
            builder.weights.insert(entry.tensor_name.clone(), id);
        }
        builder.target_embedding = builder.push_tensor(
            "target.embed_tokens.weight",
            contiguous(
                DType::Bf16,
                &[GEMMA4_MTP_VOCAB_SIZE, GEMMA4_MTP_BACKBONE_HIDDEN_SIZE],
            )?,
            Gemma4MtpTensorBacking::TargetEmbedding,
        );
        builder.target_hidden = builder.push_tensor(
            "request.target_hidden",
            contiguous(DType::Bf16, &[1, GEMMA4_MTP_BACKBONE_HIDDEN_SIZE])?,
            Gemma4MtpTensorBacking::TargetHidden,
        );
        builder.token_id = builder.push_tensor(
            "request.target_token_id",
            contiguous(DType::I32, &[1])?,
            Gemma4MtpTensorBacking::TokenId,
        );
        builder.position = builder.push_tensor(
            "request.target_position",
            contiguous(DType::I32, &[1])?,
            Gemma4MtpTensorBacking::Position,
        );
        Ok(builder)
    }

    fn push_tensor(
        &mut self,
        name: impl Into<String>,
        view: TensorView,
        backing: Gemma4MtpTensorBacking,
    ) -> usize {
        let id = self.tensors.len();
        self.tensors.push(Gemma4MtpExecutionTensor {
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
    ) -> Result<usize, Gemma4MtpExecutionError> {
        self.workspace_bytes = self
            .workspace_bytes
            .checked_add(view.end_offset())
            .ok_or_else(|| Gemma4MtpExecutionError::invalid("workspace bytes overflowed"))?;
        Ok(self.push_tensor(name, view, Gemma4MtpTensorBacking::Workspace))
    }

    fn alias(
        &mut self,
        name: impl Into<String>,
        tensor: usize,
        view: TensorView,
    ) -> Result<usize, Gemma4MtpExecutionError> {
        if view.end_offset() > self.tensors[tensor].view.end_offset() {
            return Err(Gemma4MtpExecutionError::invalid(
                "assistant alias exceeds its workspace owner",
            ));
        }
        Ok(self.push_tensor(name, view, Gemma4MtpTensorBacking::Alias { tensor }))
    }

    fn constant(
        &mut self,
        label: &str,
        bits: u16,
        width: usize,
    ) -> Result<usize, Gemma4MtpExecutionError> {
        Ok(self.push_tensor(
            format!("{label}.constant"),
            TensorView::contiguous(DType::Bf16, &[width])
                .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?,
            Gemma4MtpTensorBacking::ConstantBf16 { bits, width },
        ))
    }

    fn constant_with_view(
        &mut self,
        label: &str,
        bits: u16,
        view: TensorView,
    ) -> Result<usize, Gemma4MtpExecutionError> {
        if view.dtype() != DType::Bf16
            || view.encoding() != Encoding::Unquantized
            || !view.is_contiguous()
            || view.payload_bytes() == 0
            || view.payload_bytes() % 2 != 0
        {
            return Err(Gemma4MtpExecutionError::invalid(
                "assistant BF16 constant view differs",
            ));
        }
        let width = usize::try_from(view.payload_bytes() / 2)
            .map_err(|_| Gemma4MtpExecutionError::invalid("constant width exceeds usize"))?;
        Ok(self.push_tensor(
            format!("{label}.constant"),
            view,
            Gemma4MtpTensorBacking::ConstantBf16 { bits, width },
        ))
    }

    fn target_sliding(
        &mut self,
        label: &str,
        plane: Gemma4KvPlane,
    ) -> Result<usize, Gemma4MtpExecutionError> {
        Ok(self.push_tensor(
            format!("{label}.target_{plane:?}"),
            contiguous(DType::Bf16, &[self.graph.target_kv_length(), 8, 256])?,
            Gemma4MtpTensorBacking::TargetSliding { plane },
        ))
    }

    fn weight(&self, name: &str) -> Result<usize, Gemma4MtpExecutionError> {
        self.weights.get(name).copied().ok_or_else(|| {
            Gemma4MtpExecutionError::invalid(format!("assistant resident weight absent: {name}"))
        })
    }

    fn predecessor(
        &self,
        node: usize,
        predecessor: usize,
    ) -> Result<usize, Gemma4MtpExecutionError> {
        let predecessor = *self.graph.nodes()[node]
            .predecessors()
            .get(predecessor)
            .ok_or_else(|| Gemma4MtpExecutionError::invalid("graph predecessor absent"))?;
        self.outputs[predecessor]
            .first()
            .copied()
            .ok_or_else(|| Gemma4MtpExecutionError::invalid("predecessor output absent"))
    }

    fn views(&self, tensors: &[usize]) -> Vec<TensorView> {
        tensors
            .iter()
            .map(|tensor| self.tensors[*tensor].view.clone())
            .collect()
    }

    fn semantic(
        &mut self,
        graph_node_id: usize,
        descriptor: SemanticOpDescriptor,
        inputs: Vec<usize>,
        outputs: Vec<usize>,
    ) {
        self.nodes.push(Gemma4MtpExecutionNode {
            graph_node_id,
            label: self.graph.nodes()[graph_node_id].label().to_owned(),
            lowering: Gemma4MtpLowering::Semantic(Box::new(descriptor)),
            inputs,
            outputs,
        });
    }

    fn generic(
        &mut self,
        graph_node_id: usize,
        kind: SemanticOpKind,
        inputs: Vec<usize>,
        outputs: Vec<usize>,
    ) -> Result<(), Gemma4MtpExecutionError> {
        let descriptor = SemanticOpDescriptor::new(kind, self.views(&inputs), self.views(&outputs))
            .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
        self.semantic(graph_node_id, descriptor, inputs, outputs);
        Ok(())
    }

    fn matmul_widths(
        &self,
        layer: Option<u32>,
        consumer: WeightConsumer,
    ) -> Result<(u64, u64), Gemma4MtpExecutionError> {
        let widths = match consumer {
            WeightConsumer::Gemma4MtpPreProjection => (7_680, GEMMA4_MTP_HIDDEN_SIZE),
            WeightConsumer::Gemma4MtpPostProjection => {
                (GEMMA4_MTP_HIDDEN_SIZE, GEMMA4_MTP_BACKBONE_HIDDEN_SIZE)
            }
            WeightConsumer::EmbeddingAndTiedOutput => {
                (GEMMA4_MTP_HIDDEN_SIZE, GEMMA4_MTP_VOCAB_SIZE)
            }
            WeightConsumer::AttentionQ => match layer {
                Some(0..=2) => (GEMMA4_MTP_HIDDEN_SIZE, 4_096),
                Some(3) => (GEMMA4_MTP_HIDDEN_SIZE, 8_192),
                _ => return Err(Gemma4MtpExecutionError::invalid("Q layer differs")),
            },
            WeightConsumer::AttentionO => match layer {
                Some(0..=2) => (4_096, GEMMA4_MTP_HIDDEN_SIZE),
                Some(3) => (8_192, GEMMA4_MTP_HIDDEN_SIZE),
                _ => return Err(Gemma4MtpExecutionError::invalid("O layer differs")),
            },
            WeightConsumer::MlpGate | WeightConsumer::MlpUp => {
                (GEMMA4_MTP_HIDDEN_SIZE, GEMMA4_MTP_INTERMEDIATE_SIZE)
            }
            WeightConsumer::MlpDown => (GEMMA4_MTP_INTERMEDIATE_SIZE, GEMMA4_MTP_HIDDEN_SIZE),
            _ => return Err(Gemma4MtpExecutionError::invalid("linear role differs")),
        };
        Ok(widths)
    }

    fn lower(mut self) -> Result<Gemma4MtpExecutionLayout, Gemma4MtpExecutionError> {
        let mut terminal = None;
        let mut draft_hidden = None;
        for graph_node_id in 0..self.graph.nodes().len() {
            let node = &self.graph.nodes()[graph_node_id];
            let result = match node.kind() {
                Gemma4MtpGraphNodeKind::TargetEmbedding => {
                    let output = self.workspace(
                        format!("{}.output", node.label()),
                        contiguous(DType::Bf16, &[1, GEMMA4_MTP_BACKBONE_HIDDEN_SIZE])?,
                    )?;
                    self.generic(
                        graph_node_id,
                        SemanticOpKind::Embedding,
                        vec![self.target_embedding, self.token_id],
                        vec![output],
                    )?;
                    vec![output]
                }
                Gemma4MtpGraphNodeKind::ScaleConstant { value_bits } => {
                    let input = self.predecessor(graph_node_id, 0)?;
                    let scalar = self.constant(
                        node.label(),
                        f32_to_bf16_rne(f32::from_bits(*value_bits)),
                        1,
                    )?;
                    let output = self.workspace(
                        format!("{}.output", node.label()),
                        self.tensors[input].view.clone(),
                    )?;
                    self.generic(
                        graph_node_id,
                        SemanticOpKind::ScalarMul,
                        vec![input, scalar],
                        vec![output],
                    )?;
                    vec![output]
                }
                Gemma4MtpGraphNodeKind::CopyToConcat { destination_column } => {
                    let concat = match self.concat {
                        Some(concat) => concat,
                        None => {
                            let concat = self.workspace(
                                "request.concat",
                                contiguous(DType::Bf16, &[1, 7_680])?,
                            )?;
                            self.concat = Some(concat);
                            concat
                        }
                    };
                    let input = if *destination_column == 0 {
                        self.predecessor(graph_node_id, 0)?
                    } else if u64::from(*destination_column) == GEMMA4_MTP_BACKBONE_HIDDEN_SIZE {
                        self.target_hidden
                    } else {
                        return Err(Gemma4MtpExecutionError::invalid(
                            "concat destination column differs",
                        ));
                    };
                    let offset = u64::from(*destination_column) * 2;
                    let output_view = TensorView::new(
                        DType::Bf16,
                        Encoding::Unquantized,
                        &[1, GEMMA4_MTP_BACKBONE_HIDDEN_SIZE as usize],
                        &[GEMMA4_MTP_BACKBONE_HIDDEN_SIZE as usize, 1],
                        offset,
                    )
                    .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
                    let output =
                        self.alias(format!("{}.output", node.label()), concat, output_view)?;
                    self.generic(
                        graph_node_id,
                        SemanticOpKind::Copy,
                        vec![input],
                        vec![output],
                    )?;
                    vec![output]
                }
                Gemma4MtpGraphNodeKind::Matmul { consumer, weight } => {
                    let (input_width, output_width) =
                        self.matmul_widths(node.assistant_layer(), *consumer)?;
                    let input = if *consumer == WeightConsumer::Gemma4MtpPreProjection {
                        self.concat.ok_or_else(|| {
                            Gemma4MtpExecutionError::invalid("concat workspace absent")
                        })?
                    } else {
                        self.predecessor(graph_node_id, 0)?
                    };
                    let input = self.alias(
                        format!("{}.input", node.label()),
                        input,
                        contiguous(DType::Bf16, &[1, input_width])?,
                    )?;
                    let output = self.workspace(
                        format!("{}.output", node.label()),
                        contiguous(DType::Bf16, &[1, output_width])?,
                    )?;
                    self.generic(
                        graph_node_id,
                        SemanticOpKind::Matmul,
                        vec![input, self.weight(weight)?],
                        vec![output],
                    )?;
                    if *consumer == WeightConsumer::Gemma4MtpPostProjection {
                        draft_hidden = Some(output);
                    }
                    vec![output]
                }
                Gemma4MtpGraphNodeKind::RmsNorm {
                    role,
                    scale_mode,
                    epsilon_bits,
                    weight,
                } => {
                    let source = self.predecessor(graph_node_id, 0)?;
                    let view = if *role == Gemma4MtpNormRole::Query {
                        let layer = node.assistant_layer().ok_or_else(|| {
                            Gemma4MtpExecutionError::invalid("query norm layer absent")
                        })?;
                        let dim = if layer == 3 { 512 } else { 256 };
                        contiguous(DType::Bf16, &[16, dim])?
                    } else {
                        contiguous(DType::Bf16, &[1, GEMMA4_MTP_HIDDEN_SIZE])?
                    };
                    let input =
                        self.alias(format!("{}.input", node.label()), source, view.clone())?;
                    let output = self.workspace(format!("{}.output", node.label()), view)?;
                    let inputs = vec![input, self.weight(weight)?];
                    let descriptor = SemanticOpDescriptor::new_rms_norm(
                        self.views(&inputs),
                        self.views(&[output]),
                        f32::from_bits(*epsilon_bits),
                        match scale_mode {
                            RmsNormScaleMode::Direct => RmsNormScaleMode::Direct,
                            RmsNormScaleMode::OffsetOne => RmsNormScaleMode::OffsetOne,
                        },
                    )
                    .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
                    self.semantic(graph_node_id, descriptor, inputs, vec![output]);
                    vec![output]
                }
                Gemma4MtpGraphNodeKind::QueryRotary(rope) => {
                    let source = self.predecessor(graph_node_id, 0)?;
                    let q_view = contiguous(
                        DType::Bf16,
                        &[1, u64::from(rope.q_heads), u64::from(rope.head_dim)],
                    )?;
                    let k_view = contiguous(
                        DType::Bf16,
                        &[1, u64::from(rope.dummy_kv_heads), u64::from(rope.head_dim)],
                    )?;
                    let q = self.alias(format!("{}.q", node.label()), source, q_view.clone())?;
                    let k = self.constant_with_view(node.label(), 0, k_view.clone())?;
                    let q_out = self.workspace(format!("{}.q_out", node.label()), q_view)?;
                    let k_out = self.workspace(format!("{}.k_out", node.label()), k_view)?;
                    let inputs = vec![q, k, self.position];
                    let descriptor = SemanticOpDescriptor::new_rotary(
                        self.views(&inputs),
                        self.views(&[q_out, k_out]),
                        rope.semantic_contract()
                            .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?,
                    )
                    .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
                    self.semantic(graph_node_id, descriptor, inputs, vec![q_out, k_out]);
                    vec![q_out]
                }
                Gemma4MtpGraphNodeKind::SharedTargetAttention(attention) => {
                    let q = self.predecessor(graph_node_id, 0)?;
                    let output = self.workspace(
                        format!("{}.output", node.label()),
                        contiguous(
                            DType::Bf16,
                            &[
                                1,
                                u64::from(attention.q_heads),
                                u64::from(attention.head_dim),
                            ],
                        )?,
                    )?;
                    if attention.sliding_window.is_some() {
                        let key = self.target_sliding(node.label(), Gemma4KvPlane::Key)?;
                        let value = self.target_sliding(node.label(), Gemma4KvPlane::Value)?;
                        let inputs = vec![q, key, value];
                        let descriptor = SemanticOpDescriptor::new_causal_attention(
                            self.views(&inputs),
                            self.views(&[output]),
                            attention.semantic_contract().map_err(|error| {
                                Gemma4MtpExecutionError::invalid(error.to_string())
                            })?,
                        )
                        .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
                        self.semantic(graph_node_id, descriptor, inputs, vec![output]);
                    } else {
                        self.nodes.push(Gemma4MtpExecutionNode {
                            graph_node_id,
                            label: node.label().to_owned(),
                            lowering: Gemma4MtpLowering::OpaqueFullTargetAttention,
                            inputs: vec![q],
                            outputs: vec![output],
                        });
                    }
                    vec![output]
                }
                Gemma4MtpGraphNodeKind::GeluTanhMul => {
                    let inputs = vec![
                        self.predecessor(graph_node_id, 0)?,
                        self.predecessor(graph_node_id, 1)?,
                    ];
                    let output = self.workspace(
                        format!("{}.output", node.label()),
                        contiguous(DType::Bf16, &[1, GEMMA4_MTP_INTERMEDIATE_SIZE])?,
                    )?;
                    self.generic(
                        graph_node_id,
                        SemanticOpKind::GeluTanhMul,
                        inputs,
                        vec![output],
                    )?;
                    vec![output]
                }
                Gemma4MtpGraphNodeKind::Add => {
                    let inputs = vec![
                        self.predecessor(graph_node_id, 0)?,
                        self.predecessor(graph_node_id, 1)?,
                    ];
                    let output = self.workspace(
                        format!("{}.output", node.label()),
                        contiguous(DType::Bf16, &[1, GEMMA4_MTP_HIDDEN_SIZE])?,
                    )?;
                    self.generic(graph_node_id, SemanticOpKind::Add, inputs, vec![output])?;
                    vec![output]
                }
                Gemma4MtpGraphNodeKind::ScaleWeight { weight } => {
                    let input = self.predecessor(graph_node_id, 0)?;
                    let output = self.workspace(
                        format!("{}.output", node.label()),
                        contiguous(DType::Bf16, &[1, GEMMA4_MTP_HIDDEN_SIZE])?,
                    )?;
                    self.generic(
                        graph_node_id,
                        SemanticOpKind::ScalarMul,
                        vec![input, self.weight(weight)?],
                        vec![output],
                    )?;
                    vec![output]
                }
                Gemma4MtpGraphNodeKind::Argmax => {
                    let input = self.predecessor(graph_node_id, 0)?;
                    let output = self.push_tensor(
                        "terminal.token_id",
                        contiguous(DType::I32, &[1])?,
                        Gemma4MtpTensorBacking::TerminalToken,
                    );
                    self.generic(
                        graph_node_id,
                        SemanticOpKind::Argmax,
                        vec![input],
                        vec![output],
                    )?;
                    terminal = Some(output);
                    vec![output]
                }
            };
            self.outputs[graph_node_id] = result;
        }
        let terminal_token_tensor = terminal.ok_or_else(|| {
            Gemma4MtpExecutionError::invalid("assistant terminal token output absent")
        })?;
        let draft_hidden_tensor = draft_hidden.ok_or_else(|| {
            Gemma4MtpExecutionError::invalid("assistant post-projection output absent")
        })?;
        if self.nodes.len() != self.graph.nodes().len()
            || self.graph.assistant_kv_allocation_bytes() != 0
        {
            return Err(Gemma4MtpExecutionError::invalid(
                "assistant lowering count or KV contract differs",
            ));
        }
        Ok(Gemma4MtpExecutionLayout {
            assistant_fingerprint: self.graph.assistant_fingerprint().to_owned(),
            target_fingerprint: self.graph.target_fingerprint().to_owned(),
            plan_digest: *self.plan.digest(),
            target_kv_length: self.graph.target_kv_length(),
            tensors: self.tensors,
            nodes: self.nodes,
            terminal_token_tensor,
            draft_hidden_tensor,
            target_hidden_tensor: self.target_hidden,
            token_id_tensor: self.token_id,
            position_tensor: self.position,
            resident_bytes: self.plan.total_destination_bytes,
            workspace_bytes: self.workspace_bytes,
            assistant_kv_bytes: 0,
        })
    }
}

pub fn build_gemma4_mtp_execution_layout(
    graph: &Gemma4MtpGraph,
    plan: &WeightLoadPlan,
) -> Result<Gemma4MtpExecutionLayout, Gemma4MtpExecutionError> {
    LayoutBuilder::new(graph, plan)?.lower()
}

struct Gemma4MtpResidentInner {
    session: Arc<ExecutionSession>,
    queue: ExecutionQueue,
    assistant_fingerprint: String,
    target_fingerprint: String,
    plan: WeightLoadPlan,
    buffers: BTreeMap<String, ExecutionBuffer>,
    completion_timeout: Duration,
}

#[derive(Clone)]
pub struct Gemma4MtpResidentModel {
    inner: Arc<Gemma4MtpResidentInner>,
}

impl fmt::Debug for Gemma4MtpResidentModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Gemma4MtpResidentModel")
            .field("session_id", &self.inner.session.id())
            .field("assistant_fingerprint", &self.inner.assistant_fingerprint)
            .field("target_fingerprint", &self.inner.target_fingerprint)
            .field("resident_allocations", &self.inner.buffers.len())
            .finish_non_exhaustive()
    }
}

impl Gemma4MtpResidentModel {
    pub fn provision<S>(
        session: Arc<ExecutionSession>,
        lock: &crate::Gemma4MtpModelLock,
        source: &S,
        plan: WeightLoadPlan,
        completion_timeout: Duration,
    ) -> Result<Self, Gemma4MtpExecutionError>
    where
        S: Gemma4MtpWeightSource + ?Sized,
    {
        let canonical = build_verified_gemma4_mtp_weight_load_plan(lock, source)
            .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
        if plan != canonical || completion_timeout.is_zero() {
            return Err(Gemma4MtpExecutionError::invalid(
                "assistant source and resident plan differ",
            ));
        }
        let queue = session
            .create_queue()
            .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
        let mut buffers = BTreeMap::new();
        let mut resident_bytes = 0_u64;
        for entry in &plan.entries {
            let size = entry.source_range[1]
                .checked_sub(entry.source_range[0])
                .ok_or_else(|| Gemma4MtpExecutionError::invalid("weight range underflowed"))?;
            let buffer = session
                .allocate_with_category(size, AllocationCategory::ModelResident)
                .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
            for chunk in &entry.chunks {
                let source_offset = chunk
                    .source_offset
                    .checked_sub(entry.source_range[0])
                    .ok_or_else(|| Gemma4MtpExecutionError::invalid("chunk source underflowed"))?;
                let destination_offset = chunk
                    .destination_offset
                    .checked_sub(entry.destination_start.ok_or_else(|| {
                        Gemma4MtpExecutionError::invalid("weight destination absent")
                    })?)
                    .ok_or_else(|| {
                        Gemma4MtpExecutionError::invalid("chunk destination underflowed")
                    })?;
                let length = usize::try_from(chunk.byte_length)
                    .map_err(|_| Gemma4MtpExecutionError::invalid("weight chunk exceeds usize"))?;
                let bytes = source
                    .read_tensor_range(&entry.tensor_name, source_offset, length)
                    .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
                let range = buffer
                    .range(destination_offset, chunk.byte_length)
                    .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
                let mut transfer = session
                    .upload(&queue, range, Arc::from(bytes))
                    .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
                if transfer
                    .wait(completion_timeout)
                    .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?
                    != ExecutionState::Success
                {
                    return Err(Gemma4MtpExecutionError::invalid(
                        "assistant weight upload did not complete successfully",
                    ));
                }
            }
            resident_bytes = resident_bytes
                .checked_add(size)
                .ok_or_else(|| Gemma4MtpExecutionError::invalid("resident bytes overflowed"))?;
            if buffers.insert(entry.tensor_name.clone(), buffer).is_some() {
                return Err(Gemma4MtpExecutionError::invalid(
                    "assistant resident tensor duplicated",
                ));
            }
        }
        if resident_bytes != plan.total_destination_bytes
            || buffers.len() as u64 != crate::GEMMA4_MTP_TENSOR_COUNT
        {
            return Err(Gemma4MtpExecutionError::invalid(
                "assistant resident allocation accounting differs",
            ));
        }
        Ok(Self {
            inner: Arc::new(Gemma4MtpResidentInner {
                session,
                queue,
                assistant_fingerprint: source.lock_fingerprint().to_owned(),
                target_fingerprint: source.target_fingerprint().to_owned(),
                plan,
                buffers,
                completion_timeout,
            }),
        })
    }

    pub fn new_request(
        &self,
        graph: &Gemma4MtpGraph,
    ) -> Result<Gemma4MtpExecutionRequest, Gemma4MtpExecutionError> {
        if graph.assistant_fingerprint() != self.inner.assistant_fingerprint
            || graph.target_fingerprint() != self.inner.target_fingerprint
            || graph.weight_plan_digest() != self.inner.plan.digest()
        {
            return Err(Gemma4MtpExecutionError::invalid(
                "assistant resident and graph identity differ",
            ));
        }
        let layout = build_gemma4_mtp_execution_layout(graph, &self.inner.plan)?;
        let storage = provision_request_storage(&self.inner, &layout)?;
        Ok(Gemma4MtpExecutionRequest {
            resident: Arc::clone(&self.inner),
            graph: graph.clone(),
            layout,
            storage,
            cancelled: false,
        })
    }

    pub fn resident_bytes(&self) -> u64 {
        self.inner.plan.total_destination_bytes
    }
}

#[derive(Clone)]
enum MtpStorage {
    Owned(ExecutionBuffer),
    Target,
}

fn provision_request_storage(
    resident: &Gemma4MtpResidentInner,
    layout: &Gemma4MtpExecutionLayout,
) -> Result<Vec<MtpStorage>, Gemma4MtpExecutionError> {
    let mut storage = Vec::with_capacity(layout.tensors.len());
    for tensor in &layout.tensors {
        let entry = match tensor.backing() {
            Gemma4MtpTensorBacking::AssistantWeight(name) => {
                MtpStorage::Owned(resident.buffers.get(name).cloned().ok_or_else(|| {
                    Gemma4MtpExecutionError::invalid("assistant resident buffer absent")
                })?)
            }
            Gemma4MtpTensorBacking::TargetEmbedding
            | Gemma4MtpTensorBacking::TargetSliding { .. } => MtpStorage::Target,
            Gemma4MtpTensorBacking::Alias { tensor: source } => storage
                .get(*source)
                .cloned()
                .ok_or_else(|| Gemma4MtpExecutionError::invalid("alias source absent"))?,
            Gemma4MtpTensorBacking::TokenId
            | Gemma4MtpTensorBacking::TargetHidden
            | Gemma4MtpTensorBacking::Position
            | Gemma4MtpTensorBacking::ConstantBf16 { .. }
            | Gemma4MtpTensorBacking::Workspace
            | Gemma4MtpTensorBacking::TerminalToken => MtpStorage::Owned(
                resident
                    .session
                    .allocate_with_category(
                        tensor.view().end_offset(),
                        AllocationCategory::Workspace,
                    )
                    .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?,
            ),
        };
        if let (Gemma4MtpTensorBacking::ConstantBf16 { bits, width }, MtpStorage::Owned(buffer)) =
            (tensor.backing(), &entry)
        {
            let mut bytes = Vec::with_capacity(width * 2);
            for _ in 0..*width {
                bytes.extend_from_slice(&bits.to_le_bytes());
            }
            upload_exact(
                resident.session.as_ref(),
                &resident.queue,
                buffer,
                &bytes,
                resident.completion_timeout,
            )?;
        }
        storage.push(entry);
    }
    Ok(storage)
}

fn validate_dynamic_rebind(
    previous: &Gemma4MtpExecutionLayout,
    next: &Gemma4MtpExecutionLayout,
    storage: &[MtpStorage],
) -> Result<(), Gemma4MtpExecutionError> {
    if previous.assistant_fingerprint != next.assistant_fingerprint
        || previous.target_fingerprint != next.target_fingerprint
        || previous.plan_digest != next.plan_digest
        || previous.nodes.len() != next.nodes.len()
        || previous.tensors.len() != next.tensors.len()
        || storage.len() != next.tensors.len()
        || previous.terminal_token_tensor != next.terminal_token_tensor
        || previous.draft_hidden_tensor != next.draft_hidden_tensor
        || previous.target_hidden_tensor != next.target_hidden_tensor
        || previous.token_id_tensor != next.token_id_tensor
        || previous.position_tensor != next.position_tensor
        || previous.resident_bytes != next.resident_bytes
        || previous.workspace_bytes != next.workspace_bytes
        || next.assistant_kv_bytes != 0
    {
        return Err(Gemma4MtpExecutionError::invalid(
            "dynamic target rebind changed assistant-owned layout",
        ));
    }
    for ((previous, next), storage) in previous.tensors.iter().zip(&next.tensors).zip(storage) {
        if previous.id != next.id || previous.name != next.name {
            return Err(Gemma4MtpExecutionError::invalid(
                "dynamic target rebind changed tensor identity",
            ));
        }
        match (previous.backing(), next.backing(), storage) {
            (
                Gemma4MtpTensorBacking::TargetSliding {
                    plane: previous_plane,
                },
                Gemma4MtpTensorBacking::TargetSliding { plane: next_plane },
                MtpStorage::Target,
            ) if previous_plane == next_plane
                && previous.view.dtype() == DType::Bf16
                && next.view.dtype() == DType::Bf16
                && previous.view.shape().get(1..) == Some(&[8, 256][..])
                && next.view.shape().get(1..) == Some(&[8, 256][..]) => {}
            (
                Gemma4MtpTensorBacking::TargetEmbedding,
                Gemma4MtpTensorBacking::TargetEmbedding,
                MtpStorage::Target,
            ) if previous.view == next.view => {}
            (_, _, MtpStorage::Owned(_))
                if !matches!(
                    previous.backing,
                    Gemma4MtpTensorBacking::TargetEmbedding
                        | Gemma4MtpTensorBacking::TargetSliding { .. }
                ) && previous.backing == next.backing
                    && previous.view == next.view => {}
            _ => {
                return Err(Gemma4MtpExecutionError::invalid(
                    "dynamic target rebind changed an owned tensor or storage class",
                ));
            }
        }
    }
    Ok(())
}

pub struct Gemma4MtpExecutionRequest {
    resident: Arc<Gemma4MtpResidentInner>,
    graph: Gemma4MtpGraph,
    layout: Gemma4MtpExecutionLayout,
    storage: Vec<MtpStorage>,
    cancelled: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MtpExecutionAudit {
    target: String,
    submission_count: u64,
    kernel_dispatch_count: u64,
    fallback_used: bool,
    target_state_generation: u64,
    target_length: u64,
    assistant_kv_allocation_bytes: u64,
}

impl Gemma4MtpExecutionAudit {
    pub fn target(&self) -> &str {
        &self.target
    }
    pub const fn submission_count(&self) -> u64 {
        self.submission_count
    }
    pub const fn kernel_dispatch_count(&self) -> u64 {
        self.kernel_dispatch_count
    }
    pub const fn fallback_used(&self) -> bool {
        self.fallback_used
    }
    pub const fn target_state_generation(&self) -> u64 {
        self.target_state_generation
    }
    pub const fn target_length(&self) -> u64 {
        self.target_length
    }
    pub const fn assistant_kv_allocation_bytes(&self) -> u64 {
        self.assistant_kv_allocation_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MtpExecutionOutput {
    token_ids: [i32; 1],
    hidden_states_bf16: Vec<u16>,
    audit: Gemma4MtpExecutionAudit,
}

impl Gemma4MtpExecutionOutput {
    pub const fn token_id(&self) -> i32 {
        self.token_ids[0]
    }
    pub const fn token_ids(&self) -> &[i32] {
        &self.token_ids
    }
    /// One 3,840-wide BF16 post-projection row. It is draft feedback only;
    /// the assistant still allocates and owns no KV state.
    pub fn hidden_states_bf16(&self) -> Option<&[u16]> {
        Some(&self.hidden_states_bf16)
    }
    pub const fn audit(&self) -> &Gemma4MtpExecutionAudit {
        &self.audit
    }
}

impl Gemma4MtpExecutionRequest {
    pub const fn layout(&self) -> &Gemma4MtpExecutionLayout {
        &self.layout
    }

    fn retarget(
        &mut self,
        target_kv_length: u64,
        absolute_query_position: u64,
    ) -> Result<(), Gemma4MtpExecutionError> {
        if self.layout.target_kv_length == target_kv_length
            && self.layout_position() == absolute_query_position
        {
            return Ok(());
        }
        let graph = self
            .graph
            .with_target_snapshot(target_kv_length, absolute_query_position)
            .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
        let layout = build_gemma4_mtp_execution_layout(&graph, &self.resident.plan)?;
        validate_dynamic_rebind(&self.layout, &layout, &self.storage)?;
        self.graph = graph;
        self.layout = layout;
        Ok(())
    }

    /// Produces one greedy draft token. The target hidden row is the exact
    /// final-RMSNorm BF16 row returned by the same target transition that owns
    /// `target`; no assistant state is created or committed.
    pub fn propose(
        &mut self,
        target_token_id: i32,
        target_hidden_bf16: &[u16],
        target: &Gemma4MtpTargetKvLease<'_>,
    ) -> Result<Gemma4MtpExecutionOutput, Gemma4MtpExecutionError> {
        if self.cancelled
            || target.session_id() != self.resident.session.id()
            || target_hidden_bf16.len() != GEMMA4_MTP_BACKBONE_HIDDEN_SIZE as usize
            || target_token_id < 0
            || u64::try_from(target_token_id)
                .ok()
                .is_none_or(|token| token >= GEMMA4_MTP_VOCAB_SIZE)
        {
            return Err(Gemma4MtpExecutionError::invalid(
                "assistant proposal input or target lease differs",
            ));
        }
        target
            .verify_unchanged()
            .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
        let absolute_query_position = target
            .absolute_query_position()
            .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
        self.retarget(target.committed_length(), absolute_query_position)?;
        let token_bytes = target_token_id.to_le_bytes();
        self.upload_tensor(self.layout.token_id_tensor, &token_bytes)?;
        let hidden_bytes = target_hidden_bf16
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect::<Vec<_>>();
        self.upload_tensor(self.layout.target_hidden_tensor, &hidden_bytes)?;
        let position = i32::try_from(self.layout_position()).map_err(|_| {
            Gemma4MtpExecutionError::invalid("assistant absolute query position exceeds i32")
        })?;
        self.upload_tensor(self.layout.position_tensor, &position.to_le_bytes())?;

        let mut audit = Gemma4MtpExecutionAudit {
            target: String::new(),
            submission_count: 0,
            kernel_dispatch_count: 0,
            fallback_used: false,
            target_state_generation: target.state_generation(),
            target_length: target.committed_length(),
            assistant_kv_allocation_bytes: self.layout.assistant_kv_bytes,
        };
        let mut token = None;
        let mut draft_hidden = None;
        for node_index in 0..self.layout.nodes.len() {
            let node = &self.layout.nodes[node_index];
            match node.lowering() {
                Gemma4MtpLowering::Semantic(descriptor) => {
                    let inputs = node
                        .inputs()
                        .iter()
                        .map(|tensor| self.bind(*tensor, AccessMode::Read, target))
                        .collect::<Result<Vec<_>, _>>()?;
                    let outputs = node
                        .outputs()
                        .iter()
                        .map(|tensor| self.bind(*tensor, AccessMode::Write, target))
                        .collect::<Result<Vec<_>, _>>()?;
                    let bound = BoundSemanticOp::new(
                        Arc::new(descriptor.as_ref().clone()),
                        inputs,
                        outputs,
                    )
                    .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
                    let prepared = self
                        .resident
                        .session
                        .prepare(Arc::new(bound))
                        .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
                    let mut submission = self
                        .resident
                        .session
                        .submit(&prepared, &self.resident.queue)
                        .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
                    record_dispatch(&mut audit, submission.dispatch())?;
                    if submission
                        .wait(self.resident.completion_timeout)
                        .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?
                        != ExecutionState::Success
                    {
                        return Err(Gemma4MtpExecutionError::invalid(format!(
                            "assistant node did not complete: {}",
                            node.label()
                        )));
                    }
                    if node.outputs() == [self.layout.terminal_token_tensor] {
                        let mut readback = submission
                            .start_output_readback(0)
                            .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
                        if readback
                            .wait(self.resident.completion_timeout)
                            .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?
                            != ExecutionState::Success
                        {
                            return Err(Gemma4MtpExecutionError::invalid(
                                "assistant token readback did not complete",
                            ));
                        }
                        let mut bytes = [0_u8; 4];
                        readback
                            .read_into(&mut bytes)
                            .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
                        token = Some(i32::from_le_bytes(bytes));
                    } else if node.outputs() == [self.layout.draft_hidden_tensor] {
                        let mut readback = submission
                            .start_output_readback(0)
                            .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
                        if readback
                            .wait(self.resident.completion_timeout)
                            .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?
                            != ExecutionState::Success
                        {
                            return Err(Gemma4MtpExecutionError::invalid(
                                "assistant hidden readback did not complete",
                            ));
                        }
                        let mut bytes = vec![0_u8; GEMMA4_MTP_BACKBONE_HIDDEN_SIZE as usize * 2];
                        readback
                            .read_into(&mut bytes)
                            .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
                        draft_hidden = Some(
                            bytes
                                .chunks_exact(2)
                                .map(|word| u16::from_le_bytes([word[0], word[1]]))
                                .collect::<Vec<_>>(),
                        );
                    }
                }
                Gemma4MtpLowering::OpaqueFullTargetAttention => {
                    let query = self.bind(node.inputs()[0], AccessMode::Read, target)?;
                    let output = self.bind(node.outputs()[0], AccessMode::Write, target)?;
                    let mut submission = target
                        .submit_full_attention(query, output)
                        .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
                    record_dispatch(&mut audit, submission.dispatch())?;
                    if submission
                        .wait(self.resident.completion_timeout)
                        .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?
                        != ExecutionState::Success
                    {
                        return Err(Gemma4MtpExecutionError::invalid(
                            "assistant full target attention did not complete",
                        ));
                    }
                }
            }
        }
        target
            .verify_unchanged()
            .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
        let token_id = token.ok_or_else(|| {
            Gemma4MtpExecutionError::invalid("assistant terminal token was not published")
        })?;
        let hidden_states_bf16 = draft_hidden.ok_or_else(|| {
            Gemma4MtpExecutionError::invalid("assistant post-projection row was not published")
        })?;
        if token_id < 0
            || u64::try_from(token_id)
                .ok()
                .is_none_or(|id| id >= GEMMA4_MTP_VOCAB_SIZE)
        {
            return Err(Gemma4MtpExecutionError::invalid(
                "assistant token is nonfinite sentinel or outside vocabulary",
            ));
        }
        Ok(Gemma4MtpExecutionOutput {
            token_ids: [token_id],
            hidden_states_bf16,
            audit,
        })
    }

    /// Poisons this stateless proposal workspace. Dropping it releases every
    /// assistant workspace allocation; target state is neither cancelled nor
    /// modified by this operation.
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    fn layout_position(&self) -> u64 {
        // The graph already validated absolute_position >= logical tail.
        // Every one of the four assistant layers uses this same value.
        self.layout
            .nodes
            .iter()
            .find_map(|node| match node.lowering() {
                Gemma4MtpLowering::Semantic(descriptor)
                    if descriptor.kind() == SemanticOpKind::Rotary =>
                {
                    descriptor
                        .rotary_contract()
                        .map(|contract| u64::from(contract.start_position()))
                }
                _ => None,
            })
            .unwrap_or(self.layout.target_kv_length - 1)
    }

    fn upload_tensor(&self, tensor: usize, bytes: &[u8]) -> Result<(), Gemma4MtpExecutionError> {
        let buffer = self.owned(tensor)?;
        upload_exact(
            self.resident.session.as_ref(),
            &self.resident.queue,
            buffer,
            bytes,
            self.resident.completion_timeout,
        )
    }

    fn owned(&self, tensor: usize) -> Result<&ExecutionBuffer, Gemma4MtpExecutionError> {
        match self.storage.get(tensor) {
            Some(MtpStorage::Owned(buffer)) => Ok(buffer),
            _ => Err(Gemma4MtpExecutionError::invalid(
                "assistant tensor has no owned buffer",
            )),
        }
    }

    fn bind(
        &self,
        tensor: usize,
        access: AccessMode,
        target: &Gemma4MtpTargetKvLease<'_>,
    ) -> Result<OwnedTensorBinding, Gemma4MtpExecutionError> {
        let descriptor = self
            .layout
            .tensors
            .get(tensor)
            .ok_or_else(|| Gemma4MtpExecutionError::invalid("assistant tensor id is absent"))?;
        match descriptor.backing() {
            Gemma4MtpTensorBacking::TargetEmbedding => {
                if access != AccessMode::Read {
                    return Err(Gemma4MtpExecutionError::invalid(
                        "target embedding cannot be bound writable",
                    ));
                }
                target
                    .bind_target_embedding()
                    .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))
            }
            Gemma4MtpTensorBacking::TargetSliding { plane } => {
                if access != AccessMode::Read {
                    return Err(Gemma4MtpExecutionError::invalid(
                        "target sliding KV cannot be bound writable",
                    ));
                }
                target
                    .bind_sliding(*plane)
                    .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))
            }
            _ => self
                .resident
                .session
                .bind(self.owned(tensor)?, descriptor.view().clone(), access)
                .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string())),
        }
    }
}

fn record_dispatch(
    audit: &mut Gemma4MtpExecutionAudit,
    dispatch: &crate::DispatchEvidence,
) -> Result<(), Gemma4MtpExecutionError> {
    if dispatch.backend != 1 || dispatch.fallback_used {
        return Err(Gemma4MtpExecutionError::invalid(
            "assistant dispatch is not exact HIP/no-fallback",
        ));
    }
    if audit.target.is_empty() {
        audit.target = dispatch.target.clone();
    } else if audit.target != dispatch.target {
        return Err(Gemma4MtpExecutionError::invalid(
            "assistant dispatch targets differ",
        ));
    }
    audit.submission_count = audit
        .submission_count
        .checked_add(1)
        .ok_or_else(|| Gemma4MtpExecutionError::invalid("submission count overflowed"))?;
    audit.kernel_dispatch_count = audit
        .kernel_dispatch_count
        .checked_add(u64::from(dispatch.dispatch_count))
        .ok_or_else(|| Gemma4MtpExecutionError::invalid("dispatch count overflowed"))?;
    audit.fallback_used |= dispatch.fallback_used;
    Ok(())
}

fn upload_exact(
    session: &ExecutionSession,
    queue: &ExecutionQueue,
    buffer: &ExecutionBuffer,
    bytes: &[u8],
    timeout: Duration,
) -> Result<(), Gemma4MtpExecutionError> {
    if bytes.is_empty() || u64::try_from(bytes.len()).ok() != Some(buffer.size_bytes()) {
        return Err(Gemma4MtpExecutionError::invalid(
            "assistant upload byte length differs",
        ));
    }
    let range = buffer
        .range(0, buffer.size_bytes())
        .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
    let mut transfer = session
        .upload(queue, range, Arc::from(bytes))
        .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?;
    if transfer
        .wait(timeout)
        .map_err(|error| Gemma4MtpExecutionError::invalid(error.to_string()))?
        != ExecutionState::Success
    {
        return Err(Gemma4MtpExecutionError::invalid(
            "assistant upload did not complete successfully",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixtureSource {
        lock_fingerprint: String,
        target_fingerprint: String,
        config: crate::Gemma4MtpConfig,
        tensors: BTreeMap<String, crate::TensorDescriptor>,
    }

    impl Gemma4MtpWeightSource for FixtureSource {
        fn lock_fingerprint(&self) -> &str {
            &self.lock_fingerprint
        }

        fn target_fingerprint(&self) -> &str {
            &self.target_fingerprint
        }

        fn config(&self) -> &crate::Gemma4MtpConfig {
            &self.config
        }

        fn tensors(&self) -> &BTreeMap<String, crate::TensorDescriptor> {
            &self.tensors
        }

        fn read_tensor_range(
            &self,
            _name: &str,
            _tensor_offset: u64,
            _length: usize,
        ) -> Result<Vec<u8>, crate::ModelError> {
            Err(crate::ModelError::Invalid(
                "layout fixture does not read payload bytes".to_owned(),
            ))
        }
    }

    fn exact_layout_fixture() -> Gemma4MtpExecutionLayout {
        let lock = crate::parse_gemma4_mtp_model_lock(include_bytes!(
            "../../../docs/models/locks/gemma4-12b-it-assistant-bf16.json"
        ))
        .expect("tracked assistant lock");
        let source = FixtureSource {
            lock_fingerprint: lock.fingerprint().to_owned(),
            target_fingerprint: lock.target_fingerprint().to_owned(),
            config: crate::Gemma4MtpConfig {
                hidden_size: 1_024,
                backbone_hidden_size: 3_840,
                intermediate_size: 8_192,
                layer_count: 4,
                attention_heads: 16,
                kv_heads: 8,
                global_kv_heads: 1,
                head_dim: 256,
                global_head_dim: 512,
                sliding_window: 1_024,
                max_position_embeddings: 262_144,
                vocab_size: 262_144,
                layer_types: vec![
                    crate::Gemma4LayerType::SlidingAttention,
                    crate::Gemma4LayerType::SlidingAttention,
                    crate::Gemma4LayerType::SlidingAttention,
                    crate::Gemma4LayerType::FullAttention,
                ],
                draft_to_target_kv_layers: [46, 46, 46, 47],
            },
            tensors: crate::expected_gemma4_mtp_tensor_catalog().expect("exact tensor catalog"),
        };
        let plan = build_verified_gemma4_mtp_weight_load_plan(&lock, &source)
            .expect("exact assistant plan");
        let graph = crate::build_gemma4_mtp_graph(&lock, &source, &plan, 17, 16)
            .expect("exact assistant graph");
        build_gemma4_mtp_execution_layout(&graph, &plan).expect("exact assistant layout")
    }

    #[test]
    fn q_only_rotary_dummy_k_keeps_exact_three_dimensional_view() {
        let layout = exact_layout_fixture();
        let rotary = layout
            .nodes()
            .iter()
            .filter_map(|node| match node.lowering() {
                Gemma4MtpLowering::Semantic(descriptor)
                    if descriptor.kind() == SemanticOpKind::Rotary =>
                {
                    Some((node, descriptor.as_ref()))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(rotary.len(), 4);
        for (layer, (node, descriptor)) in rotary.into_iter().enumerate() {
            let head_dim = if layer == 3 { 512 } else { 256 };
            let kv_heads = if layer == 3 { 1 } else { 8 };
            assert_eq!(descriptor.inputs()[0].shape(), [1, 16, head_dim]);
            assert_eq!(descriptor.inputs()[1].shape(), [1, kv_heads, head_dim]);
            assert_eq!(descriptor.inputs()[2].shape(), [1]);
            assert_eq!(descriptor.outputs()[0].shape(), [1, 16, head_dim]);
            assert_eq!(descriptor.outputs()[1].shape(), [1, kv_heads, head_dim]);
            let dummy_k = node.inputs()[1];
            assert!(matches!(
                layout.tensors()[dummy_k].backing(),
                Gemma4MtpTensorBacking::ConstantBf16 { bits: 0, width }
                    if *width == kv_heads * head_dim
            ));
            assert_eq!(layout.tensors()[dummy_k].view(), &descriptor.inputs()[1]);
        }
    }

    #[test]
    fn width_one_output_exposes_token_and_post_projection_row() {
        let output = Gemma4MtpExecutionOutput {
            token_ids: [258_884],
            hidden_states_bf16: vec![0x3f80; GEMMA4_MTP_BACKBONE_HIDDEN_SIZE as usize],
            audit: Gemma4MtpExecutionAudit {
                target: "gfx1201".to_owned(),
                submission_count: 1,
                kernel_dispatch_count: 1,
                fallback_used: false,
                target_state_generation: 7,
                target_length: 65,
                assistant_kv_allocation_bytes: 0,
            },
        };
        assert_eq!(output.token_id(), 258_884);
        assert_eq!(output.token_ids(), [258_884]);
        assert_eq!(
            output.hidden_states_bf16().unwrap().len(),
            GEMMA4_MTP_BACKBONE_HIDDEN_SIZE as usize
        );
        assert_eq!(output.audit().assistant_kv_allocation_bytes(), 0);
        assert!(!output.audit().fallback_used());
    }

    #[test]
    fn bf16_constants_use_round_to_nearest_even() {
        assert_eq!(f32_to_bf16_rne(0.0), 0);
        assert_eq!(f32_to_bf16_rne(1.0), 0x3f80);
        assert_eq!(
            f32_to_bf16_rne((GEMMA4_MTP_BACKBONE_HIDDEN_SIZE as f32).sqrt()),
            0x4278
        );
    }
}
