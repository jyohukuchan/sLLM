// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Load-once, request-resettable Qwen3.5 AQ4 model runtime.
//!
//! This module is the ownership boundary between package-derived model geometry and request
//! execution. Resident weights and request state are loaded once; request reset never reloads or
//! clones a weight buffer.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::backend_operation_registry::{OperationExecutionRecord, OperationResolutionTrace};
use crate::execution_batch::ExecutionPhase;
use crate::execution_batch::WorkspacePlan;
use crate::loader::{
    PassthroughF32Data, effective_rmsnorm_weight_values, read_named_passthrough_f32,
};
use crate::qwen35_aq4_head_runtime::{
    PackageEmbeddingRuntime, PackageFinalNormRuntime, PackageLmHeadMode, PackageLmHeadRuntime,
    PackageTokenLogit, QWEN3_FINAL_NORM_TENSOR, package_embedding_shape,
};
use crate::qwen35_aq4_layer_runtime::{
    PackageLinearAttnComponentStepMs, PackageLinearAttnResidentStepLayer,
    PackageLinearAttnSequenceGeometry, PackageLinearAttnSequenceWorkspace,
    PackageSelfAttnComponentStepMs, PackageSelfAttnResidentStepLayer,
};
use crate::qwen35_package_contract::{
    PackageDecoderLayerKind, PackageManifestLayerEntry, package_manifest_layer_entries,
};

const QWEN35_LINEAR_PERSISTENT_STATE_BYTES: u64 = 2_228_224;
const QWEN35_SELF_PERSISTENT_STATE_BYTES: u64 = 33_554_432;
const QWEN35_REQUIRED_DEVICE_HEADROOM_BYTES: u64 = 512 * 1024 * 1024;
pub const QWEN35_AQ4_NATIVE_PREFILL_MAX_WIDTH: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35Aq4PrefillInvocation {
    pub layer_index: usize,
    pub execution_width: usize,
    pub phase: ExecutionPhase,
    pub records: [OperationExecutionRecord; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35Aq4PrefillChunkStep {
    pub execution_width: usize,
    pub invocations: Vec<Qwen35Aq4PrefillInvocation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35Aq4FailedPrefillInvocation {
    pub layer_index: usize,
    pub execution_width: usize,
    pub phase: ExecutionPhase,
    pub records: [Option<OperationExecutionRecord>; 2],
}

fn qwen35_model_workspace_plan(
    capacity_bytes: u64,
    resident_bytes: u64,
    retained_activation_bytes: u64,
    linear_layers: usize,
    self_layers: usize,
) -> Result<WorkspacePlan, String> {
    // `capacity_bytes` is the device property's totalGlobalMem. This is a conservative static
    // admission bound with 512 MiB headroom; it is not a reservation against currently free VRAM.
    let linear = u64::try_from(linear_layers)
        .map_err(|_| "linear layer count does not fit u64".to_string())?
        .checked_mul(QWEN35_LINEAR_PERSISTENT_STATE_BYTES)
        .ok_or_else(|| "linear persistent state bytes overflow".to_string())?;
    let self_attention = u64::try_from(self_layers)
        .map_err(|_| "self-attention layer count does not fit u64".to_string())?
        .checked_mul(QWEN35_SELF_PERSISTENT_STATE_BYTES)
        .ok_or_else(|| "self-attention persistent state bytes overflow".to_string())?;
    let plan = WorkspacePlan {
        capacity_bytes,
        resident_bytes,
        persistent_state_bytes: linear
            .checked_add(self_attention)
            .ok_or_else(|| "model persistent state bytes overflow".to_string())?,
        temporary_activation_bytes: retained_activation_bytes,
        // Registry state/IO estimates describe the same buffers counted above and the peak, so
        // they are audit metadata rather than an additional allocation here.
        operator_workspace_bytes: 0,
        required_headroom_bytes: QWEN35_REQUIRED_DEVICE_HEADROOM_BYTES,
    };
    plan.validate()?;
    Ok(plan)
}

fn qwen35_retained_activation_bytes(
    package_dir: &Path,
    layers: &[Qwen35Aq4LayerSpec],
    hidden: usize,
    vocab: usize,
) -> Result<u64, String> {
    let bundles = crate::package::list_tensor_payload_bundles(package_dir)?;
    let rows = |name: &str| -> Result<u64, String> {
        let bundle = bundles
            .iter()
            .find(|bundle| bundle.tensor_name == name)
            .ok_or_else(|| format!("missing tensor metadata {name}"))?;
        bundle
            .shape
            .first()
            .copied()
            .ok_or_else(|| format!("tensor {name} has no row dimension"))
    };
    let h = u64::try_from(hidden).map_err(|_| "hidden does not fit u64".to_string())?;
    let mut elements = 0_u64;
    let mut maximum_linear_intermediate = None::<u64>;
    for layer in layers {
        let prefix = format!("model.language_model.layers.{}", layer.layer_index);
        let intermediate = rows(&format!("{prefix}.mlp.gate_proj.weight"))?;
        let retained = match layer.kind {
            Qwen35Aq4LayerKind::LinearAttention => {
                maximum_linear_intermediate = Some(
                    maximum_linear_intermediate
                        .map_or(intermediate, |previous| previous.max(intermediate)),
                );
                9_u64
                    .checked_mul(h)
                    .and_then(|value| value.checked_add(2 * 8_192))
                    .and_then(|value| value.checked_add(2 * 2_048))
                    .and_then(|value| value.checked_add(2 * 32))
                    .and_then(|value| value.checked_add(intermediate))
            }
            Qwen35Aq4LayerKind::SelfAttention => {
                let q_rows = rows(&format!("{prefix}.self_attn.q_proj.weight"))?;
                let k_rows = rows(&format!("{prefix}.self_attn.k_proj.weight"))?;
                let v_rows = rows(&format!("{prefix}.self_attn.v_proj.weight"))?;
                let q = q_rows / 2;
                let attention = q;
                5_u64
                    .checked_mul(h)
                    .and_then(|value| value.checked_add(q_rows))
                    .and_then(|value| value.checked_add(3 * q))
                    .and_then(|value| value.checked_add(3 * k_rows))
                    .and_then(|value| value.checked_add(v_rows))
                    .and_then(|value| value.checked_add(2 * attention))
                    .and_then(|value| value.checked_add(intermediate))
            }
        }
        .ok_or_else(|| "retained layer activation elements overflow".to_string())?;
        elements = elements
            .checked_add(retained)
            .ok_or_else(|| "retained model activation elements overflow".to_string())?;
    }
    if let Some(intermediate) = maximum_linear_intermediate {
        // One model-wide arena: linear sequence intermediates plus two `[128,H]` ping-pong
        // buffers and one M1 splice row. No layer owns a duplicate of this allocation.
        let per_row = 10_u64
            .checked_mul(h)
            .and_then(|value| value.checked_add(2 * 8_192))
            .and_then(|value| value.checked_add(2 * 2_048))
            .and_then(|value| value.checked_add(4 * 32))
            .and_then(|value| value.checked_add(3 * intermediate))
            .ok_or_else(|| "shared prefill workspace row elements overflow".to_string())?;
        let shared = per_row
            .checked_mul(QWEN35_AQ4_NATIVE_PREFILL_MAX_WIDTH as u64)
            .and_then(|value| value.checked_add(2 * 128 * h))
            .and_then(|value| value.checked_add(h))
            .ok_or_else(|| "shared prefill workspace elements overflow".to_string())?;
        elements = elements
            .checked_add(shared)
            .ok_or_else(|| "retained prefill workspace elements overflow".to_string())?;
    }
    // Embedding output, final-norm weight/output, lm-head input/logits, and a conservative
    // per-vocabulary upper bound for top-1 partial value/index buffers.
    let vocab = u64::try_from(vocab).map_err(|_| "vocab does not fit u64".to_string())?;
    elements = elements
        .checked_add(3 * h)
        .and_then(|value| value.checked_add(5 * vocab))
        .ok_or_else(|| "global activation elements overflow".to_string())?;
    elements
        .checked_mul(4)
        .ok_or_else(|| "retained activation bytes overflow".to_string())
}

fn tensor_layer_index(name: &str) -> Option<usize> {
    let (_, suffix) = name.split_once(".layers.")?;
    suffix.split('.').next()?.parse().ok()
}

fn qwen35_package_resident_plan_bytes(
    package_dir: &Path,
    selected_layers: &[Qwen35Aq4LayerSpec],
) -> Result<u64, String> {
    let selected = selected_layers
        .iter()
        .map(|layer| layer.layer_index)
        .collect::<BTreeSet<_>>();
    let mut bytes = 0_u64;
    let mut component_codebooks = BTreeSet::new();
    for bundle in crate::package::list_tensor_payload_bundles(package_dir)? {
        let layer_index = tensor_layer_index(&bundle.tensor_name);
        if layer_index.is_some_and(|index| !selected.contains(&index)) {
            continue;
        }
        let component = layer_index
            .map(|index| format!("layer-{index}"))
            .unwrap_or_else(|| bundle.tensor_name.clone());
        bytes = bytes
            .checked_add(bundle.index_file.bytes)
            .and_then(|value| value.checked_add(bundle.scale_file.bytes))
            .ok_or_else(|| "AQ4 resident payload bytes overflow".to_string())?;
        if component_codebooks.insert((component, bundle.codebook_file.absolute_path.clone())) {
            bytes = bytes
                .checked_add(bundle.codebook_file.bytes)
                .ok_or_else(|| "AQ4 codebook resident bytes overflow".to_string())?;
        }
        let scale_format = bundle
            .scale_format
            .as_deref()
            .ok_or_else(|| format!("{} has no AQ4 scale format", bundle.tensor_name))?;
        let scale_table_bytes = crate::aq4_package_runtime::package_aq4_f32_allocation_bytes(
            u64::try_from(crate::aq::scale_values(scale_format)?.len())
                .map_err(|_| "AQ4 scale-table length does not fit u64".to_string())?,
        )?;
        bytes = bytes
            .checked_add(scale_table_bytes)
            .ok_or_else(|| "AQ4 resident bytes overflow".to_string())?;
        if !bundle.row_scale_overrides.is_empty() {
            let rows = *bundle
                .shape
                .first()
                .ok_or_else(|| format!("{} has no row dimension", bundle.tensor_name))?;
            bytes = bytes
                .checked_add(crate::aq4_package_runtime::package_aq4_f32_allocation_bytes(rows)?)
                .ok_or_else(|| "AQ4 resident bytes overflow".to_string())?;
        }
    }
    for bundle in crate::package::list_passthrough_payload_bundles(package_dir)? {
        if tensor_layer_index(&bundle.tensor_name).is_some_and(|index| !selected.contains(&index)) {
            continue;
        }
        bytes = bytes
            .checked_add(bundle.payload_bytes)
            .ok_or_else(|| "passthrough resident bytes overflow".to_string())?;
    }
    Ok(bytes)
}

/// Product context length used by the Qwen3.5 9B served-model contract.
pub const QWEN35_AQ4_CONTEXT_LENGTH: usize = 4096;
pub const QWEN35_AQ4_KV_BLOCK_SIZE: usize = 256;

pub type Qwen35Aq4LayerKind = PackageDecoderLayerKind;
pub type Qwen35Aq4LayerSpec = PackageManifestLayerEntry;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35Aq4SelfAttentionGeometry {
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub value_dim: usize,
    pub q_projection_layout: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35Aq4ModelGeometry {
    pub vocab: usize,
    pub hidden: usize,
    pub context_length: usize,
    pub block_size: usize,
    pub cache_blocks: usize,
    pub block_table: Vec<u32>,
    pub layers: Vec<Qwen35Aq4LayerSpec>,
    pub self_attention: Option<Qwen35Aq4SelfAttentionGeometry>,
}

impl Qwen35Aq4ModelGeometry {
    fn new(
        vocab: usize,
        hidden: usize,
        context_length: usize,
        block_size: usize,
        layers: Vec<Qwen35Aq4LayerSpec>,
    ) -> Result<Self, String> {
        if vocab == 0 || hidden == 0 || context_length == 0 || block_size == 0 {
            return Err("Qwen3.5 AQ4 model geometry must be nonzero".to_string());
        }
        if layers.is_empty() {
            return Err("Qwen3.5 AQ4 model requires at least one decoder layer".to_string());
        }
        let cache_blocks = context_length.div_ceil(block_size);
        let block_table = (0..cache_blocks)
            .map(|index| {
                u32::try_from(index)
                    .map_err(|_| format!("Qwen3.5 AQ4 KV block index {index} exceeds u32"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            vocab,
            hidden,
            context_length,
            block_size,
            cache_blocks,
            block_table,
            layers,
            self_attention: None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Qwen35Aq4ModelLoadConfig {
    pub package_dir: PathBuf,
    pub device_index: u32,
    pub expected_architecture: Option<String>,
    pub chunk_bytes: usize,
    pub context_length: usize,
    pub kv_block_size: usize,
    /// `None` loads every package-manifest layer. A selection must retain manifest order.
    pub layer_indices: Option<Vec<usize>>,
    pub lm_head_mode: PackageLmHeadMode,
    pub lm_head_chunk_rows: usize,
}

#[derive(Clone)]
pub struct Qwen35Aq4StackStep {
    pub final_layer_position: usize,
    pub layer_step_ms: Vec<f64>,
    pub linear_attention_components: Vec<Option<PackageLinearAttnComponentStepMs>>,
    pub self_attention_components: Vec<Option<PackageSelfAttnComponentStepMs>>,
    /// Exactly two registry-routed operations for every successfully completed layer.
    pub operation_executions: Vec<[OperationExecutionRecord; 2]>,
}

pub enum Qwen35Aq4ResidentLayer {
    LinearAttention(PackageLinearAttnResidentStepLayer),
    SelfAttention(PackageSelfAttnResidentStepLayer),
}

impl Qwen35Aq4ResidentLayer {
    pub fn kind(&self) -> Qwen35Aq4LayerKind {
        match self {
            Self::LinearAttention(_) => Qwen35Aq4LayerKind::LinearAttention,
            Self::SelfAttention(_) => Qwen35Aq4LayerKind::SelfAttention,
        }
    }

    pub fn output_buffer(&self) -> &ullm_runtime_sys::RuntimeBuffer {
        match self {
            Self::LinearAttention(layer) => layer.output_buffer(),
            Self::SelfAttention(layer) => layer.output_buffer(),
        }
    }

    pub fn operation_resolution_traces(&self) -> Vec<OperationResolutionTrace> {
        match self {
            Self::LinearAttention(layer) => layer.operation_resolution_traces(),
            Self::SelfAttention(layer) => layer.operation_resolution_traces(),
        }
    }

    fn step_from_device(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        input: &ullm_runtime_sys::RuntimeBuffer,
        rotary_dim: usize,
        rope_base: f32,
        rope_position: usize,
        cache_position: usize,
        phase: ExecutionPhase,
        label: &str,
    ) -> Result<(), String> {
        match self {
            Self::LinearAttention(layer) => {
                layer.step_from_device_to_device_for_phase(stream, input, phase, label)
            }
            Self::SelfAttention(layer) => layer.step_from_device_to_device_for_phase(
                stream,
                input,
                rotary_dim,
                rope_base,
                rope_position,
                cache_position,
                phase,
                label,
            ),
        }
    }

    fn reset_synchronized(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
    ) -> Result<(), String> {
        match self {
            Self::LinearAttention(layer) => layer.reset_request_state_synchronized(stream),
            Self::SelfAttention(layer) => layer.reset_request_state_synchronized(stream),
        }
    }

    fn take_components(
        &mut self,
    ) -> (
        Option<PackageLinearAttnComponentStepMs>,
        Option<PackageSelfAttnComponentStepMs>,
    ) {
        match self {
            Self::LinearAttention(layer) => (layer.take_last_component_step_ms(), None),
            Self::SelfAttention(layer) => (None, layer.take_last_component_step_ms()),
        }
    }

    fn take_operation_executions(&mut self) -> [Option<OperationExecutionRecord>; 2] {
        match self {
            Self::LinearAttention(layer) => layer.take_last_operation_executions(),
            Self::SelfAttention(layer) => layer.take_last_operation_executions(),
        }
    }

    fn mark_execution_failed(&mut self) {
        match self {
            Self::LinearAttention(layer) => layer.mark_request_execution_failed(),
            Self::SelfAttention(layer) => layer.mark_request_execution_failed(),
        }
    }
}

/// Owns one device context and every resident allocation required by one Qwen3.5 AQ4 model.
///
/// Rust drops fields in declaration order. GPU allocation holders are therefore declared before
/// `stream`, and `stream` before `context`; the context can never be destroyed while a holder or
/// stream still exists. `Drop` synchronizes outstanding work before that ordered destruction.
pub struct Qwen35Aq4ModelRuntime {
    // GPU allocation holders -- do not move these below stream/context.
    embedding: Option<PackageEmbeddingRuntime>,
    layers: Vec<Qwen35Aq4ResidentLayer>,
    final_norm: PassthroughF32Data,
    final_norm_runtime: Option<PackageFinalNormRuntime>,
    lm_head: PackageLmHeadRuntime,
    prefill_sequence_workspace: Option<PackageLinearAttnSequenceWorkspace>,
    prefill_ping_buffers: [ullm_runtime_sys::RuntimeBuffer; 2],
    prefill_row_input_buffer: ullm_runtime_sys::RuntimeBuffer,
    // Runtime handles -- stream must precede context for destruction order.
    stream: ullm_runtime_sys::RuntimeStream,
    _context: ullm_runtime_sys::RuntimeContext,
    package_dir: PathBuf,
    geometry: Qwen35Aq4ModelGeometry,
    device_name: String,
    backend: String,
    device_total_global_mem: u64,
    last_partial_operation_executions: Vec<[Option<OperationExecutionRecord>; 2]>,
    last_partial_prefill_invocations: Vec<Qwen35Aq4FailedPrefillInvocation>,
}

impl Qwen35Aq4ModelRuntime {
    pub fn load(config: Qwen35Aq4ModelLoadConfig) -> Result<Self, String> {
        if config.chunk_bytes == 0 {
            return Err("Qwen3.5 AQ4 load chunk bytes must be positive".to_string());
        }
        let path = package_path_text(&config.package_dir)?;
        let manifest_layers = package_manifest_layer_entries(&config.package_dir)?;
        let layers = select_manifest_layers(&manifest_layers, config.layer_indices.as_deref())?;
        let (vocab, hidden) = package_embedding_shape(path)?;
        let mut geometry = Qwen35Aq4ModelGeometry::new(
            vocab,
            hidden,
            config.context_length,
            config.kv_block_size,
            layers,
        )?;

        let mut context = ullm_runtime_sys::RuntimeContext::create(config.device_index)
            .map_err(|err| format!("failed to create Qwen3.5 AQ4 runtime context: {err}"))?;
        let info = context
            .device_info()
            .map_err(|err| format!("failed to query Qwen3.5 AQ4 runtime device: {err}"))?;
        if let Some(expected) = config.expected_architecture.as_deref() {
            crate::backend_operation_registry::require_device_architecture(&info, expected)
                .map_err(|error| format!("Qwen3.5 AQ4 {error}"))?;
        }
        let linear_layers = geometry
            .layers
            .iter()
            .filter(|layer| layer.kind == Qwen35Aq4LayerKind::LinearAttention)
            .count();
        let self_layers = geometry.layers.len() - linear_layers;
        let _workspace = qwen35_model_workspace_plan(
            info.total_global_mem,
            qwen35_package_resident_plan_bytes(&config.package_dir, &geometry.layers)?,
            qwen35_retained_activation_bytes(&config.package_dir, &geometry.layers, hidden, vocab)?,
            linear_layers,
            self_layers,
        )
        .map_err(|error| format!("Qwen3.5 AQ4 model workspace admission failed: {error}"))?;
        let mut stream = context
            .create_stream()
            .map_err(|err| format!("failed to create Qwen3.5 AQ4 runtime stream: {err}"))?;

        let mut resident_layers = Vec::with_capacity(geometry.layers.len());
        for spec in &geometry.layers {
            let layer = match spec.kind {
                Qwen35Aq4LayerKind::LinearAttention => Qwen35Aq4ResidentLayer::LinearAttention({
                    let layer = PackageLinearAttnResidentStepLayer::load(
                        &mut context,
                        &mut stream,
                        path,
                        config.chunk_bytes,
                        spec.layer_index,
                    )
                    .map_err(|err| {
                        format!(
                            "failed to load Qwen3.5 AQ4 linear layer {}: {err}",
                            spec.layer_index
                        )
                    })?;
                    if layer.hidden != hidden {
                        return Err(format!(
                            "Qwen3.5 AQ4 linear-attention layer {} hidden {} does not match embedding hidden {hidden}",
                            spec.layer_index, layer.hidden
                        ));
                    }
                    layer
                }),
                Qwen35Aq4LayerKind::SelfAttention => Qwen35Aq4ResidentLayer::SelfAttention({
                    let layer = PackageSelfAttnResidentStepLayer::load(
                        &mut context,
                        &mut stream,
                        path,
                        config.chunk_bytes,
                        spec.layer_index,
                        &geometry.block_table,
                        geometry.block_size,
                        geometry.cache_blocks,
                    )
                    .map_err(|err| {
                        format!(
                            "failed to load Qwen3.5 AQ4 self-attention layer {}: {err}",
                            spec.layer_index
                        )
                    })?;
                    if layer.hidden != hidden {
                        return Err(format!(
                            "Qwen3.5 AQ4 self-attention layer {} hidden {} does not match embedding hidden {hidden}",
                            spec.layer_index, layer.hidden
                        ));
                    }
                    let layer_geometry = Qwen35Aq4SelfAttentionGeometry {
                        q_heads: layer.q_heads,
                        kv_heads: layer.kv_heads,
                        head_dim: layer.head_dim,
                        value_dim: layer.value_dim,
                        q_projection_layout: layer.q_projection_layout.as_str(),
                    };
                    match &geometry.self_attention {
                        Some(previous) if previous != &layer_geometry => {
                            return Err(format!(
                                "Qwen3.5 AQ4 self-attention geometry changed at layer {}: previous={previous:?} current={layer_geometry:?}",
                                spec.layer_index
                            ));
                        }
                        None => geometry.self_attention = Some(layer_geometry),
                        Some(_) => {}
                    }
                    layer
                }),
            };
            resident_layers.push(layer);
        }

        let mut final_norm =
            read_named_passthrough_f32(path, QWEN3_FINAL_NORM_TENSOR, config.chunk_bytes)
                .map_err(|err| format!("failed to read Qwen3.5 final RMSNorm: {err}"))?;
        final_norm.values =
            effective_rmsnorm_weight_values(QWEN3_FINAL_NORM_TENSOR, &final_norm.values);
        if final_norm.values.len() != hidden {
            return Err(format!(
                "Qwen3.5 final RMSNorm length {} does not match hidden {hidden}",
                final_norm.values.len()
            ));
        }
        let lm_head = PackageLmHeadRuntime::load(
            config.lm_head_mode,
            &mut context,
            &mut stream,
            path,
            config.chunk_bytes,
            hidden,
            config.lm_head_chunk_rows,
        )?;
        let final_norm_runtime = if lm_head.supports_device_input() {
            Some(PackageFinalNormRuntime::load(
                &mut context,
                &mut stream,
                &final_norm,
                hidden,
            )?)
        } else {
            None
        };
        let embedding = if lm_head.supports_device_input() {
            PackageEmbeddingRuntime::load_if_available(
                &mut context,
                &mut stream,
                path,
                config.chunk_bytes,
                hidden,
            )?
        } else {
            None
        };
        let sequence_geometry = resident_layers
            .iter()
            .filter_map(|layer| match layer {
                Qwen35Aq4ResidentLayer::LinearAttention(layer) => {
                    Some(layer.sequence_geometry())
                }
                Qwen35Aq4ResidentLayer::SelfAttention(_) => None,
            })
            .try_fold(None, |previous: Option<PackageLinearAttnSequenceGeometry>, current| {
                if let Some(previous) = previous
                    && previous != current
                {
                    return Err(format!(
                        "Qwen3.5 AQ4 linear sequence geometry changed: previous={previous:?} current={current:?}"
                    ));
                }
                Ok(Some(current))
            })?;
        let prefill_sequence_workspace = sequence_geometry
            .map(|geometry| {
                PackageLinearAttnSequenceWorkspace::allocate(
                    &mut context,
                    QWEN35_AQ4_NATIVE_PREFILL_MAX_WIDTH,
                    geometry,
                )
            })
            .transpose()?;
        let max_prefill_elements = QWEN35_AQ4_NATIVE_PREFILL_MAX_WIDTH
            .checked_mul(hidden)
            .ok_or_else(|| "Qwen3.5 AQ4 prefill ping element count overflows".to_string())?;
        let prefill_ping_bytes = max_prefill_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Qwen3.5 AQ4 prefill ping byte count overflows".to_string())?;
        let prefill_ping_0 = context
            .alloc_buffer(prefill_ping_bytes)
            .map_err(|error| format!("failed to allocate Qwen3.5 AQ4 prefill ping 0: {error}"))?;
        let prefill_ping_1 = context
            .alloc_buffer(prefill_ping_bytes)
            .map_err(|error| format!("failed to allocate Qwen3.5 AQ4 prefill ping 1: {error}"))?;
        let prefill_ping_buffers = [prefill_ping_0, prefill_ping_1];
        let prefill_row_input_buffer = context
            .alloc_buffer(
                hidden
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| "Qwen3.5 AQ4 prefill row byte count overflows".to_string())?,
            )
            .map_err(|error| {
                format!("failed to allocate Qwen3.5 AQ4 prefill row input: {error}")
            })?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize Qwen3.5 AQ4 model load: {err}"))?;

        Ok(Self {
            embedding,
            layers: resident_layers,
            final_norm,
            final_norm_runtime,
            lm_head,
            prefill_sequence_workspace,
            prefill_ping_buffers,
            prefill_row_input_buffer,
            stream,
            _context: context,
            package_dir: config.package_dir,
            geometry,
            device_name: info.name,
            backend: info.backend.to_string(),
            device_total_global_mem: info.total_global_mem,
            last_partial_operation_executions: Vec::with_capacity(32),
            last_partial_prefill_invocations: Vec::new(),
        })
    }

    pub fn geometry(&self) -> &Qwen35Aq4ModelGeometry {
        &self.geometry
    }

    pub fn package_dir(&self) -> &Path {
        &self.package_dir
    }

    pub fn backend(&self) -> &str {
        &self.backend
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn device_total_global_mem(&self) -> u64 {
        self.device_total_global_mem
    }

    pub fn has_resident_embedding(&self) -> bool {
        self.embedding.is_some()
    }

    pub fn supports_device_logits(&self) -> bool {
        self.final_norm_runtime.is_some() && self.lm_head.supports_device_input()
    }

    /// Returns all load-time operation resolutions in decoder-layer order.
    pub fn operation_resolution_traces(&self) -> Vec<Vec<OperationResolutionTrace>> {
        self.layers
            .iter()
            .map(Qwen35Aq4ResidentLayer::operation_resolution_traces)
            .collect()
    }

    /// Gathers one token and dispatches it through the complete resident decoder stack.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_token(
        &mut self,
        token_id: usize,
        rotary_dim: usize,
        rope_base: f32,
        rope_position: usize,
        cache_position: usize,
        sync_each_layer_for_timing: bool,
        label: &str,
    ) -> Result<Qwen35Aq4StackStep, String> {
        self.dispatch_token_for_phase(
            token_id,
            rotary_dim,
            rope_base,
            rope_position,
            cache_position,
            ExecutionPhase::Decode,
            sync_each_layer_for_timing,
            label,
        )
    }

    /// Dispatches one token using operation plans resolved for the explicit scheduler phase.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_token_for_phase(
        &mut self,
        token_id: usize,
        rotary_dim: usize,
        rope_base: f32,
        rope_position: usize,
        cache_position: usize,
        phase: ExecutionPhase,
        sync_each_layer_for_timing: bool,
        label: &str,
    ) -> Result<Qwen35Aq4StackStep, String> {
        if cache_position >= self.geometry.context_length {
            return Err(format!(
                "{label} cache position {cache_position} exceeds context length {}",
                self.geometry.context_length
            ));
        }
        if let Some(attention) = &self.geometry.self_attention
            && (rotary_dim == 0 || rotary_dim > attention.head_dim || rotary_dim % 2 != 0)
        {
            return Err(format!(
                "{label} rotary dimension {rotary_dim} must be a positive even value at most {}",
                attention.head_dim
            ));
        }
        let embedding = self.embedding.as_mut().ok_or_else(|| {
            "Qwen3.5 AQ4 token dispatch requires a resident embedding".to_string()
        })?;
        embedding.gather_token(&mut self.stream, token_id, label)?;
        let input = embedding.output_buffer();
        self.last_partial_operation_executions.clear();
        self.last_partial_prefill_invocations.clear();
        dispatch_layer_stack(
            &mut self.layers,
            &mut self.stream,
            input,
            rotary_dim,
            rope_base,
            rope_position,
            cache_position,
            phase,
            sync_each_layer_for_timing,
            label,
            &mut self.last_partial_operation_executions,
        )
    }

    /// Executes one single-request prompt chunk. Linear-attention layers consume the complete
    /// `[M, H]` tensor while self-attention layers preserve causal KV order with an explicit M1
    /// row splice.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_prefill_chunk_for_phase(
        &mut self,
        token_ids: &[usize],
        rotary_dim: usize,
        rope_base: f32,
        absolute_start: usize,
        phase: ExecutionPhase,
        sync_each_layer_for_timing: bool,
        label: &str,
    ) -> Result<Qwen35Aq4PrefillChunkStep, String> {
        let sequence_len = token_ids.len();
        if !(2..=QWEN35_AQ4_NATIVE_PREFILL_MAX_WIDTH).contains(&sequence_len) {
            return Err(format!(
                "{label} native prefill width must be in 2..={QWEN35_AQ4_NATIVE_PREFILL_MAX_WIDTH}, got {sequence_len}"
            ));
        }
        let absolute_end = absolute_start
            .checked_add(sequence_len)
            .ok_or_else(|| format!("{label} position range overflows"))?;
        if absolute_end > self.geometry.context_length {
            return Err(format!(
                "{label} position range {absolute_start}..{absolute_end} exceeds context length {}",
                self.geometry.context_length
            ));
        }
        if let Some(attention) = &self.geometry.self_attention
            && (rotary_dim == 0 || rotary_dim > attention.head_dim || rotary_dim % 2 != 0)
        {
            return Err(format!(
                "{label} rotary dimension {rotary_dim} must be a positive even value at most {}",
                attention.head_dim
            ));
        }
        let hidden_bytes = self
            .geometry
            .hidden
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("{label} hidden byte count overflows"))?;
        let sequence_bytes = sequence_len
            .checked_mul(hidden_bytes)
            .ok_or_else(|| format!("{label} sequence byte count overflows"))?;
        let embedding = self.embedding.as_mut().ok_or_else(|| {
            "Qwen3.5 AQ4 native prefill requires a resident embedding".to_string()
        })?;
        for (row, token_id) in token_ids.iter().copied().enumerate() {
            embedding.gather_token(&mut self.stream, token_id, label)?;
            self.prefill_ping_buffers[0]
                .copy_from_buffer(
                    row.checked_mul(hidden_bytes)
                        .ok_or_else(|| format!("{label} embedding row offset overflows"))?,
                    embedding.output_buffer(),
                    0,
                    hidden_bytes,
                    Some(&mut self.stream),
                )
                .map_err(|error| {
                    format!("failed to scatter {label} embedding row {row}: {error}")
                })?;
        }

        self.last_partial_operation_executions.clear();
        self.last_partial_prefill_invocations.clear();
        let layer_count = self.layers.len();
        let mut invocations = Vec::with_capacity(layer_count * sequence_len);
        let mut current_ping = 0usize;
        for layer_position in 0..layer_count {
            let next_ping = 1 - current_ping;
            let layer_label = format!(
                "{label} layer {layer_position} positions {absolute_start}..{absolute_end}"
            );
            let (source, destination) = if current_ping == 0 {
                let (left, right) = self.prefill_ping_buffers.split_at_mut(1);
                (&left[0], &mut right[0])
            } else {
                let (left, right) = self.prefill_ping_buffers.split_at_mut(1);
                (&right[0], &mut left[0])
            };
            match &mut self.layers[layer_position] {
                Qwen35Aq4ResidentLayer::LinearAttention(layer) => {
                    let workspace = self.prefill_sequence_workspace.as_mut().ok_or_else(|| {
                        format!("{layer_label} has no shared linear sequence workspace")
                    })?;
                    let records = match layer.run_device_sequence_for_phase(
                        &mut self.stream,
                        source,
                        sequence_len,
                        phase,
                        workspace,
                        &layer_label,
                    ) {
                        Ok(records) => records,
                        Err(error) => {
                            let partial = layer.take_last_operation_executions();
                            self.last_partial_operation_executions.push(partial);
                            self.last_partial_prefill_invocations.push(
                                Qwen35Aq4FailedPrefillInvocation {
                                    layer_index: layer_position,
                                    execution_width: sequence_len,
                                    phase,
                                    records: partial,
                                },
                            );
                            return Err(error);
                        }
                    };
                    if let Err(error) = destination.copy_from_buffer(
                        0,
                        workspace.output_buffer(),
                        0,
                        sequence_bytes,
                        Some(&mut self.stream),
                    ) {
                        layer.mark_request_execution_failed();
                        self.last_partial_prefill_invocations.push(
                            Qwen35Aq4FailedPrefillInvocation {
                                layer_index: layer_position,
                                execution_width: sequence_len,
                                phase,
                                records: [Some(records[0]), Some(records[1])],
                            },
                        );
                        return Err(format!(
                            "failed to copy {layer_label} sequence output: {error}"
                        ));
                    }
                    if layer_position + 1 == layer_count {
                        if let Err(error) = layer.retain_last_sequence_row(
                            &mut self.stream,
                            workspace,
                            sequence_len,
                            &layer_label,
                        ) {
                            layer.mark_request_execution_failed();
                            self.last_partial_prefill_invocations.push(
                                Qwen35Aq4FailedPrefillInvocation {
                                    layer_index: layer_position,
                                    execution_width: sequence_len,
                                    phase,
                                    records: [Some(records[0]), Some(records[1])],
                                },
                            );
                            return Err(error);
                        }
                    }
                    self.last_partial_operation_executions
                        .push([Some(records[0]), Some(records[1])]);
                    self.last_partial_prefill_invocations
                        .push(Qwen35Aq4FailedPrefillInvocation {
                            layer_index: layer_position,
                            execution_width: sequence_len,
                            phase,
                            records: [Some(records[0]), Some(records[1])],
                        });
                    invocations.push(Qwen35Aq4PrefillInvocation {
                        layer_index: layer_position,
                        execution_width: sequence_len,
                        phase,
                        records,
                    });
                }
                Qwen35Aq4ResidentLayer::SelfAttention(layer) => {
                    for row in 0..sequence_len {
                        let row_offset = row
                            .checked_mul(hidden_bytes)
                            .ok_or_else(|| format!("{layer_label} row offset overflows"))?;
                        self.prefill_row_input_buffer
                            .copy_from_buffer(
                                0,
                                source,
                                row_offset,
                                hidden_bytes,
                                Some(&mut self.stream),
                            )
                            .map_err(|error| {
                                format!("failed to gather {layer_label} row {row}: {error}")
                            })?;
                        let position = absolute_start
                            .checked_add(row)
                            .ok_or_else(|| format!("{layer_label} row position overflows"))?;
                        let token_phase = if position == 0 {
                            ExecutionPhase::ColdPrefill
                        } else {
                            ExecutionPhase::CachedPrefixPrefill
                        };
                        let row_label = format!("{layer_label} row {row} position {position}");
                        if let Err(error) = layer.step_from_device_to_device_for_phase(
                            &mut self.stream,
                            &self.prefill_row_input_buffer,
                            rotary_dim,
                            rope_base,
                            position,
                            position,
                            token_phase,
                            &row_label,
                        ) {
                            let partial = layer.take_last_operation_executions();
                            self.last_partial_operation_executions.push(partial);
                            self.last_partial_prefill_invocations.push(
                                Qwen35Aq4FailedPrefillInvocation {
                                    layer_index: layer_position,
                                    execution_width: 1,
                                    phase: token_phase,
                                    records: partial,
                                },
                            );
                            return Err(error);
                        }
                        let [first, second] = layer.take_last_operation_executions();
                        let Some(first) = first else {
                            layer.mark_request_execution_failed();
                            return Err(format!("{row_label} did not record its first operation"));
                        };
                        let Some(second) = second else {
                            layer.mark_request_execution_failed();
                            return Err(format!("{row_label} did not record its second operation"));
                        };
                        let records = [first, second];
                        if let Err(error) = destination.copy_from_buffer(
                            row_offset,
                            layer.output_buffer(),
                            0,
                            hidden_bytes,
                            Some(&mut self.stream),
                        ) {
                            layer.mark_request_execution_failed();
                            self.last_partial_prefill_invocations.push(
                                Qwen35Aq4FailedPrefillInvocation {
                                    layer_index: layer_position,
                                    execution_width: 1,
                                    phase: token_phase,
                                    records: [Some(records[0]), Some(records[1])],
                                },
                            );
                            return Err(format!("failed to scatter {row_label} output: {error}"));
                        }
                        self.last_partial_operation_executions
                            .push([Some(records[0]), Some(records[1])]);
                        self.last_partial_prefill_invocations.push(
                            Qwen35Aq4FailedPrefillInvocation {
                                layer_index: layer_position,
                                execution_width: 1,
                                phase: token_phase,
                                records: [Some(records[0]), Some(records[1])],
                            },
                        );
                        invocations.push(Qwen35Aq4PrefillInvocation {
                            layer_index: layer_position,
                            execution_width: 1,
                            phase: token_phase,
                            records,
                        });
                    }
                }
            }
            if sync_each_layer_for_timing {
                if let Err(error) = self.stream.synchronize() {
                    for layer in &mut self.layers[..=layer_position] {
                        layer.mark_execution_failed();
                    }
                    return Err(format!("failed to synchronize {layer_label}: {error}"));
                }
            }
            current_ping = next_ping;
        }
        Ok(Qwen35Aq4PrefillChunkStep {
            execution_width: sequence_len,
            invocations,
        })
    }

    pub fn take_last_partial_operation_executions(
        &mut self,
    ) -> Vec<[Option<OperationExecutionRecord>; 2]> {
        std::mem::take(&mut self.last_partial_operation_executions)
    }

    pub fn take_last_partial_prefill_invocations(
        &mut self,
    ) -> Vec<Qwen35Aq4FailedPrefillInvocation> {
        std::mem::take(&mut self.last_partial_prefill_invocations)
    }

    pub fn top_logits_from_last_layer(
        &mut self,
        top_k: usize,
        label: &str,
    ) -> Result<Vec<PackageTokenLogit>, String> {
        let last = self
            .layers
            .last()
            .ok_or_else(|| "Qwen3.5 AQ4 model has no final layer".to_string())?;
        let final_norm = self.final_norm_runtime.as_mut().ok_or_else(|| {
            "Qwen3.5 AQ4 device logits require resident final RMSNorm".to_string()
        })?;
        final_norm.normalize_device(&mut self.stream, last.output_buffer(), label)?;
        self.lm_head.top_logits_from_device_buffer(
            &mut self.stream,
            final_norm.output_buffer(),
            top_k,
        )
    }

    pub fn final_norm(&self) -> &PassthroughF32Data {
        &self.final_norm
    }

    pub fn lm_head(&mut self) -> &mut PackageLmHeadRuntime {
        &mut self.lm_head
    }

    /// Synchronizes, clears all request-owned KV/conv/recurrent state, and retains all weights.
    pub fn reset_all_request_state_synchronized(&mut self) -> Result<(), String> {
        self.stream.synchronize().map_err(|err| {
            format!("failed to synchronize Qwen3.5 AQ4 model before request reset: {err}")
        })?;
        for (position, layer) in self.layers.iter_mut().enumerate() {
            layer
                .reset_synchronized(&mut self.stream)
                .map_err(|err| format!("failed to reset Qwen3.5 AQ4 layer {position}: {err}"))?;
        }
        self.stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize Qwen3.5 AQ4 model request reset: {err}"))
    }

    pub fn synchronize(&mut self) -> Result<(), String> {
        self.stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize Qwen3.5 AQ4 model runtime: {err}"))
    }

    pub fn mark_prefill_chunk_uncommitted(&mut self) {
        for layer in &mut self.layers {
            layer.mark_execution_failed();
        }
    }

    /// Explicitly synchronizes before the owner and its context are dropped.
    pub fn shutdown_synchronized(mut self) -> Result<(), String> {
        self.synchronize()
    }
}

impl Drop for Qwen35Aq4ModelRuntime {
    fn drop(&mut self) {
        // Destructors cannot return errors; serving code should call `shutdown_synchronized` when
        // it needs an observable shutdown result. This still prevents normal outstanding work
        // from racing the ordered GPU-holder -> stream -> context destruction below.
        let _ = self.stream.synchronize();
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_layer_stack(
    layers: &mut [Qwen35Aq4ResidentLayer],
    stream: &mut ullm_runtime_sys::RuntimeStream,
    first_input: &ullm_runtime_sys::RuntimeBuffer,
    rotary_dim: usize,
    rope_base: f32,
    rope_position: usize,
    cache_position: usize,
    phase: ExecutionPhase,
    sync_each_layer_for_timing: bool,
    label: &str,
    partial_operation_executions: &mut Vec<[Option<OperationExecutionRecord>; 2]>,
) -> Result<Qwen35Aq4StackStep, String> {
    if layers.is_empty() {
        return Err(format!("{label} requires at least one resident layer"));
    }
    let mut layer_step_ms = Vec::with_capacity(layers.len());
    let mut linear_attention_components = Vec::with_capacity(layers.len());
    let mut self_attention_components = Vec::with_capacity(layers.len());
    let mut operation_executions = Vec::with_capacity(layers.len());
    for position in 0..layers.len() {
        let started = std::time::Instant::now();
        let layer_label = format!("{label} layer {position} position {rope_position}");
        let step_result = if position == 0 {
            layers[0].step_from_device(
                stream,
                first_input,
                rotary_dim,
                rope_base,
                rope_position,
                cache_position,
                phase,
                &layer_label,
            )
        } else {
            let (previous, current) = layers.split_at_mut(position);
            current[0].step_from_device(
                stream,
                previous[position - 1].output_buffer(),
                rotary_dim,
                rope_base,
                rope_position,
                cache_position,
                phase,
                &layer_label,
            )
        };
        if let Err(error) = step_result {
            partial_operation_executions.push(layers[position].take_operation_executions());
            return Err(error);
        }
        if sync_each_layer_for_timing {
            stream
                .synchronize()
                .map_err(|err| format!("failed to synchronize {layer_label}: {err}"))?;
        }
        layer_step_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        let (linear, self_attention) = layers[position].take_components();
        linear_attention_components.push(linear);
        self_attention_components.push(self_attention);
        let [first, second] = layers[position].take_operation_executions();
        operation_executions.push([
            first.ok_or_else(|| format!("{layer_label} did not record its first operation"))?,
            second.ok_or_else(|| format!("{layer_label} did not record its second operation"))?,
        ]);
        partial_operation_executions.push([
            Some(operation_executions[position][0]),
            Some(operation_executions[position][1]),
        ]);
    }
    Ok(Qwen35Aq4StackStep {
        final_layer_position: layers.len() - 1,
        layer_step_ms,
        linear_attention_components,
        self_attention_components,
        operation_executions,
    })
}

fn select_manifest_layers(
    manifest: &[Qwen35Aq4LayerSpec],
    requested: Option<&[usize]>,
) -> Result<Vec<Qwen35Aq4LayerSpec>, String> {
    let Some(requested) = requested else {
        return Ok(manifest.to_vec());
    };
    if requested.is_empty() {
        return Err("Qwen3.5 AQ4 selected layer list is empty".to_string());
    }
    let requested_set = requested.iter().copied().collect::<BTreeSet<_>>();
    if requested_set.len() != requested.len() {
        return Err("Qwen3.5 AQ4 selected layer list contains duplicates".to_string());
    }
    let selected = manifest
        .iter()
        .copied()
        .filter(|entry| requested_set.contains(&entry.layer_index))
        .collect::<Vec<_>>();
    let selected_indices = selected
        .iter()
        .map(|entry| entry.layer_index)
        .collect::<Vec<_>>();
    if selected_indices != requested {
        return Err(format!(
            "Qwen3.5 AQ4 selected layers must exist and retain manifest order: requested={requested:?} manifest_selection={selected_indices:?}"
        ));
    }
    Ok(selected)
}

fn package_path_text(path: &Path) -> Result<&str, String> {
    path.to_str()
        .ok_or_else(|| "Qwen3.5 AQ4 package path is not valid UTF-8".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn specs() -> Vec<Qwen35Aq4LayerSpec> {
        vec![
            Qwen35Aq4LayerSpec {
                layer_index: 0,
                kind: Qwen35Aq4LayerKind::LinearAttention,
            },
            Qwen35Aq4LayerSpec {
                layer_index: 1,
                kind: Qwen35Aq4LayerKind::SelfAttention,
            },
            Qwen35Aq4LayerSpec {
                layer_index: 2,
                kind: Qwen35Aq4LayerKind::LinearAttention,
            },
        ]
    }

    #[test]
    fn aggregate_workspace_admission_is_checked_before_model_allocations() {
        let valid = qwen35_model_workspace_plan(8 << 30, 5 << 30, 128 << 20, 24, 8).unwrap();
        assert!(valid.persistent_state_bytes > 0);
        assert_eq!(valid.operator_workspace_bytes, 0);
        assert!(qwen35_model_workspace_plan(1 << 30, 900 << 20, 128 << 20, 24, 8).is_err());
        assert!(qwen35_model_workspace_plan(u64::MAX, 0, 0, usize::MAX, usize::MAX).is_err());
    }

    #[test]
    fn package_workspace_uses_real_manifest_selection_and_allocation_components() {
        let root = std::env::temp_dir().join(format!(
            "ullm-qwen35-workspace-fixture-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("tensors")).unwrap();
        fs::create_dir_all(root.join("codebooks")).unwrap();
        for (path, bytes) in [
            ("tensors/l0q.idx", 10),
            ("tensors/l0q.scale", 2),
            ("tensors/l0k.idx", 12),
            ("tensors/l0k.scale", 3),
            ("tensors/l1q.idx", 14),
            ("tensors/l1q.scale", 4),
            ("tensors/l2q.idx", 16),
            ("tensors/l2q.scale", 5),
            ("codebooks/shared.f32", 64),
            ("tensors/l0.raw", 7),
            ("tensors/l1.raw", 8),
            ("tensors/l2.raw", 9),
            ("tensors/embed.raw", 11),
            ("tensors/norm.raw", 12),
            ("tensors/head.raw", 13),
        ] {
            fs::write(root.join(path), vec![1_u8; bytes]).unwrap();
        }
        fs::write(
            root.join("manifest.json"),
            r#"{
              "tensors": [
                {"name":"model.layers.0.q.weight","shape":[2,4],"scale_format":"e4m3","elements":8,"groups":2,"index_file":"tensors/l0q.idx","scale_file":"tensors/l0q.scale","codebook_file":"codebooks/shared.f32"},
                {"name":"model.layers.0.k.weight","shape":[3,4],"scale_format":"e4m3","elements":12,"groups":3,"index_file":"tensors/l0k.idx","scale_file":"tensors/l0k.scale","codebook_file":"codebooks/shared.f32"},
                {"name":"model.layers.1.q.weight","shape":[2,4],"scale_format":"e4m3","elements":8,"groups":2,"index_file":"tensors/l1q.idx","scale_file":"tensors/l1q.scale","codebook_file":"codebooks/shared.f32"},
                {"name":"model.layers.2.q.weight","shape":[2,4],"scale_format":"e4m3","elements":8,"groups":2,"index_file":"tensors/l2q.idx","scale_file":"tensors/l2q.scale","codebook_file":"codebooks/shared.f32"}
              ],
              "passthrough_tensors": [
                {"name":"model.layers.0.input_layernorm.weight","shape":[3],"elements":3,"payload_bytes":7,"payload_file":"tensors/l0.raw"},
                {"name":"model.layers.1.input_layernorm.weight","shape":[4],"elements":4,"payload_bytes":8,"payload_file":"tensors/l1.raw"},
                {"name":"model.layers.2.input_layernorm.weight","shape":[4],"elements":4,"payload_bytes":9,"payload_file":"tensors/l2.raw"},
                {"name":"model.embed_tokens.weight","shape":[1],"elements":1,"payload_bytes":11,"payload_file":"tensors/embed.raw"},
                {"name":"model.norm.weight","shape":[1],"elements":1,"payload_bytes":12,"payload_file":"tensors/norm.raw"},
                {"name":"lm_head.weight","shape":[1],"elements":1,"payload_bytes":13,"payload_file":"tensors/head.raw"}
              ],
              "row_scale_overrides":{"schema_version":"row-scale-overrides-v0.1","entries":[{"tensor_name":"model.layers.0.q.weight","row_index":1,"scale":1.25}]}
            }"#,
        )
        .unwrap();
        let layer = |layer_index| Qwen35Aq4LayerSpec {
            layer_index,
            kind: Qwen35Aq4LayerKind::SelfAttention,
        };
        assert_eq!(crate::aq::scale_values("e4m3").unwrap().len(), 119);
        // layer 0: index+scale 27, one shared codebook 64, two tables 952,
        // row-scale allocation 8, passthrough 7; global embedding/norm/head 36.
        assert_eq!(
            qwen35_package_resident_plan_bytes(&root, &[layer(0)]).unwrap(),
            1_094
        );
        // The same physical codebook is one allocation per layer component, not one per package.
        assert_eq!(
            qwen35_package_resident_plan_bytes(&root, &[layer(0), layer(1)]).unwrap(),
            1_660
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn geometry_allocates_the_full_product_context() {
        let geometry = Qwen35Aq4ModelGeometry::new(
            248_320,
            4_096,
            QWEN35_AQ4_CONTEXT_LENGTH,
            QWEN35_AQ4_KV_BLOCK_SIZE,
            specs(),
        )
        .unwrap();
        assert_eq!(geometry.context_length, 4_096);
        assert_eq!(geometry.block_size, 256);
        assert_eq!(geometry.cache_blocks, 16);
        assert_eq!(geometry.block_table, (0_u32..16).collect::<Vec<_>>());

        let smaller = Qwen35Aq4ModelGeometry::new(100, 64, 513, 256, specs()).unwrap();
        assert_eq!(smaller.context_length, 513);
        assert_eq!(smaller.cache_blocks, 3);
        assert_eq!(smaller.block_table, vec![0, 1, 2]);
    }

    #[test]
    fn selection_requires_unique_manifest_order() {
        assert_eq!(
            select_manifest_layers(&specs(), Some(&[0, 2]))
                .unwrap()
                .iter()
                .map(|entry| entry.layer_index)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert!(select_manifest_layers(&specs(), Some(&[2, 0])).is_err());
        assert!(select_manifest_layers(&specs(), Some(&[1, 1])).is_err());
        assert!(select_manifest_layers(&specs(), Some(&[9])).is_err());
    }

    #[test]
    fn layer_kind_names_match_existing_cli_reports() {
        assert_eq!(Qwen35Aq4LayerKind::SelfAttention.as_str(), "self_attention");
        assert_eq!(
            Qwen35Aq4LayerKind::LinearAttention.as_str(),
            "linear_attention"
        );
    }
}
