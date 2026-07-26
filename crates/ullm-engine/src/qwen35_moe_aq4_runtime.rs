// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Decode-first resident executor for the text-only Qwen3.5 MoE AQ4_0 package.
//!
//! This deliberately composes the already-proven Qwen3.5 hybrid-attention
//! layers with the MoE runtime primitives.  It is not a second implementation
//! of attention: full attention continues to own paged KV, mRoPE and Q-output
//! gating, while linear attention continues to own convolution and recurrent
//! state.  The only new boundary is post-attention RMSNorm -> MoE -> residual.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::aq4_package_runtime::PackageResidentSharedBufferRegistry;
use crate::backend_operation_registry::require_device_architecture;
use crate::loader::{
    LoadOptions, WeightRegistry, effective_qwen35_rmsnorm_weight_values,
    load_named_passthrough_bf16_resident, materialize_config, read_named_passthrough_f32,
};
use crate::model_config::{
    DecoderLayerKind, ModelArchitectureKind, ResidentKvCacheMode, ResidentMlpDescriptor,
    ResidentModelDescriptor, ResidentRopeKind, RmsNormWeightConvention,
    load_model_config_from_package,
};
use crate::package::{
    TensorSelector, inspect_package, select_exact_passthrough_payload_bundle,
    select_tensor_payload_bundle,
};
use crate::qwen35_aq4_head_runtime::{
    PackageEmbeddingRuntime, PackageFinalNormRuntime, PackageLmHeadMode, PackageLmHeadRuntime,
    PackageTokenLogit, QWEN3_FINAL_NORM_TENSOR,
};
use crate::qwen35_aq4_layer_runtime::{
    PackageLinearAttnGeometry, PackageLinearAttnResidentStepInput,
    PackageLinearAttnResidentStepLayer, PackageSelfAttnResidentStepInput,
    PackageSelfAttnResidentStepLayer,
};

/// The pure-text Qwen3.5 mRoPE position rows are all the scalar text position.
/// The descriptor still validates the complete mRoPE contract before this
/// scalar decode bridge is admitted.
pub const QWEN35_MOE_TEXT_ROTARY_DIM: usize = 64;
pub const QWEN35_MOE_TEXT_ROPE_BASE: f32 = 10_000_000.0;
pub const QWEN35_MOE_DEFAULT_CONTEXT_LENGTH: usize = 262_144;
pub const QWEN35_MOE_DEFAULT_KV_BLOCK_SIZE: usize = 256;

#[derive(Debug, Clone)]
pub struct Qwen35MoeAq4ModelLoadConfig {
    pub package_dir: PathBuf,
    /// Runtime-visible device index.  GPU callers must make this the isolated
    /// R9700 index after setting HIP_VISIBLE_DEVICES.
    pub device_index: u32,
    pub expected_architecture: Option<String>,
    pub chunk_bytes: usize,
    pub context_length: usize,
    pub kv_block_size: usize,
    pub lm_head_chunk_rows: usize,
}

impl Qwen35MoeAq4ModelLoadConfig {
    pub fn production_sized(package_dir: impl Into<PathBuf>, device_index: u32) -> Self {
        Self {
            package_dir: package_dir.into(),
            device_index,
            expected_architecture: Some("gfx1201".to_string()),
            chunk_bytes: 64 * 1024 * 1024,
            context_length: QWEN35_MOE_DEFAULT_CONTEXT_LENGTH,
            kv_block_size: QWEN35_MOE_DEFAULT_KV_BLOCK_SIZE,
            lm_head_chunk_rows: 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35MoeRouteTrace {
    pub layer_index: usize,
    pub selected_expert_ids: Vec<i32>,
    pub routing_scores: Vec<f32>,
    pub boundary_tie_flags: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35MoeRouterVerification {
    pub layer_index: usize,
    pub runtime_selected_expert_ids: Vec<i32>,
    pub reference_selected_expert_ids: Vec<i32>,
    pub runtime_boundary_tie_flags: Vec<u32>,
    pub reference_boundary_tie_flags: Vec<u32>,
    /// `None` means a top-k boundary tie makes an ordering assertion invalid.
    pub strict_order_match: Option<bool>,
    pub routing_score_sum: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35MoeAq4Step {
    pub position: usize,
    pub top_logits: Vec<PackageTokenLogit>,
    pub routes: Vec<Qwen35MoeRouteTrace>,
    pub wall_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35MoeAq4Generation {
    pub prompt_token_ids: Vec<usize>,
    pub generated_token_ids: Vec<usize>,
    pub final_step: Qwen35MoeAq4Step,
    /// Sum of prompt-token dispatch times, excluding model load and routing
    /// verification read-back.
    pub prompt_wall_ms: f64,
    /// Sum of the greedy decode-token dispatch times.  This is the appropriate
    /// denominator for decode tok/s; `wall_ms` includes the prompt too.
    pub decode_wall_ms: f64,
    pub wall_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35MoeAq4Residency {
    pub declared_package_bytes: u64,
    pub device_total_global_mem_bytes: u64,
    pub context_length: usize,
    pub kv_block_size: usize,
    pub cache_blocks: usize,
    pub resident_expert_payload_bytes: u64,
    /// One reusable decode workspace for all serial MoE layers; it is not
    /// forty per-layer copies of staged/dequantized expert slabs.
    pub shared_moe_decode_workspace_bytes: u64,
}

struct RawBf16Matrix {
    tensor_name: String,
    rows: usize,
    cols: usize,
    buffer: Arc<ullm_runtime_sys::RuntimeBuffer>,
}

impl RawBf16Matrix {
    fn load(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        package_path: &str,
        tensor_name: String,
        chunk_bytes: usize,
    ) -> Result<Self, String> {
        let metadata = select_exact_passthrough_payload_bundle(package_path, &tensor_name)
            .map_err(|error| format!("failed to inspect raw BF16 {tensor_name}: {error}"))?;
        let [rows, cols] = metadata.shape.as_slice() else {
            return Err(format!(
                "raw BF16 {tensor_name} must be rank 2, got {:?}",
                metadata.shape
            ));
        };
        let rows = usize::try_from(*rows)
            .map_err(|_| format!("raw BF16 {tensor_name} rows exceed usize"))?;
        let cols = usize::try_from(*cols)
            .map_err(|_| format!("raw BF16 {tensor_name} cols exceed usize"))?;
        if rows == 0 || cols == 0 {
            return Err(format!("raw BF16 {tensor_name} has a zero dimension"));
        }
        let resident = load_named_passthrough_bf16_resident(
            context,
            stream,
            package_path,
            &tensor_name,
            &metadata.shape,
            chunk_bytes,
        )?;
        Ok(Self {
            tensor_name,
            rows,
            cols,
            buffer: Arc::new(resident.buffer),
        })
    }
}

struct Aq4MoeTensor {
    tensor_name: String,
    num_experts: usize,
    rows_per_expert: usize,
    cols: usize,
    group_size: usize,
    tensor_scale: f32,
    scale_values: Vec<f32>,
    index_buffer: Arc<ullm_runtime_sys::RuntimeBuffer>,
    scale_buffer: Arc<ullm_runtime_sys::RuntimeBuffer>,
    codebook_buffer: Arc<ullm_runtime_sys::RuntimeBuffer>,
    index_bytes_per_expert: usize,
    scale_bytes_per_expert: usize,
}

impl Aq4MoeTensor {
    fn load(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        registry: &mut WeightRegistry,
        package_path: &str,
        tensor_name: String,
        expected_experts: usize,
        expected_rows: usize,
        expected_cols: usize,
        chunk_bytes: usize,
    ) -> Result<Self, String> {
        let selector = TensorSelector::Name(tensor_name.clone());
        let bundle = select_tensor_payload_bundle(package_path, &selector)
            .map_err(|error| format!("failed to select AQ4_0 MoE tensor {tensor_name}: {error}"))?;
        if bundle.tensor_name != tensor_name {
            return Err(format!(
                "AQ4_0 MoE selector for {tensor_name} resolved to {}",
                bundle.tensor_name
            ));
        }
        let [experts, rows, cols] = bundle.shape.as_slice() else {
            return Err(format!(
                "AQ4_0 MoE tensor {tensor_name} must have [expert,row,col] shape, got {:?}",
                bundle.shape
            ));
        };
        let experts = usize::try_from(*experts)
            .map_err(|_| format!("AQ4_0 MoE tensor {tensor_name} expert count exceeds usize"))?;
        let rows = usize::try_from(*rows)
            .map_err(|_| format!("AQ4_0 MoE tensor {tensor_name} row count exceeds usize"))?;
        let cols = usize::try_from(*cols)
            .map_err(|_| format!("AQ4_0 MoE tensor {tensor_name} column count exceeds usize"))?;
        if (experts, rows, cols) != (expected_experts, expected_rows, expected_cols) {
            return Err(format!(
                "AQ4_0 MoE tensor {tensor_name} geometry [{experts},{rows},{cols}] differs from expected [{expected_experts},{expected_rows},{expected_cols}]"
            ));
        }
        let registry_index = registry
            .load_and_insert(
                context,
                stream,
                &bundle,
                LoadOptions {
                    chunk_bytes,
                    verify: true,
                },
            )
            .map_err(|error| {
                format!("failed to make AQ4_0 MoE tensor {tensor_name} resident: {error}")
            })?;
        let loaded = registry.get(registry_index).ok_or_else(|| {
            format!("resident AQ4_0 MoE tensor {tensor_name} disappeared from registry")
        })?;
        let materialize = materialize_config(loaded)
            .map_err(|error| format!("invalid AQ4_0 MoE tensor {tensor_name}: {error}"))?;
        let expected_elements = experts
            .checked_mul(rows)
            .and_then(|value| value.checked_mul(cols))
            .ok_or_else(|| format!("AQ4_0 MoE tensor {tensor_name} element count overflows"))?;
        if materialize.elements != expected_elements {
            return Err(format!(
                "AQ4_0 MoE tensor {tensor_name} materialized element count {} differs from {expected_elements}",
                materialize.elements
            ));
        }
        let index_total = usize::try_from(loaded.index.bytes)
            .map_err(|_| format!("AQ4_0 MoE tensor {tensor_name} index bytes exceed usize"))?;
        let scale_total = usize::try_from(loaded.scale.bytes)
            .map_err(|_| format!("AQ4_0 MoE tensor {tensor_name} scale bytes exceed usize"))?;
        if index_total % experts != 0 || scale_total % experts != 0 {
            return Err(format!(
                "AQ4_0 MoE tensor {tensor_name} payload does not divide into {experts} expert slabs"
            ));
        }
        let elements_per_expert = rows.checked_mul(cols).ok_or_else(|| {
            format!("AQ4_0 MoE tensor {tensor_name} expert element count overflows")
        })?;
        if elements_per_expert % 2 != 0 || elements_per_expert % materialize.group_size != 0 {
            return Err(format!(
                "AQ4_0 MoE tensor {tensor_name} expert slab is not nibble/group aligned"
            ));
        }
        Ok(Self {
            tensor_name,
            num_experts: experts,
            rows_per_expert: rows,
            cols,
            group_size: materialize.group_size,
            tensor_scale: materialize.tensor_scale,
            scale_values: materialize.scale_values,
            index_buffer: loaded.index.buffer.clone(),
            scale_buffer: loaded.scale.buffer.clone(),
            codebook_buffer: loaded.codebook.buffer.clone(),
            index_bytes_per_expert: index_total / experts,
            scale_bytes_per_expert: scale_total / experts,
        })
    }

    fn selected_elements(&self, top_k: usize) -> Result<usize, String> {
        top_k
            .checked_mul(self.rows_per_expert)
            .and_then(|value| value.checked_mul(self.cols))
            .ok_or_else(|| format!("{} selected AQ4_0 elements overflow", self.tensor_name))
    }

    #[allow(clippy::too_many_arguments)]
    fn stage_and_dequant_selected(
        &self,
        selected_expert_ids: &[i32],
        index_staging: &mut ullm_runtime_sys::RuntimeBuffer,
        scale_staging: &mut ullm_runtime_sys::RuntimeBuffer,
        dequantized: &mut ullm_runtime_sys::RuntimeBuffer,
        stream: &mut ullm_runtime_sys::RuntimeStream,
    ) -> Result<(), String> {
        if selected_expert_ids.is_empty() {
            return Err(format!("{} has no selected experts", self.tensor_name));
        }
        for (rank, expert_id) in selected_expert_ids.iter().copied().enumerate() {
            let expert = usize::try_from(expert_id).map_err(|_| {
                format!(
                    "{} selected expert id {expert_id} is negative",
                    self.tensor_name
                )
            })?;
            if expert >= self.num_experts {
                return Err(format!(
                    "{} selected expert id {expert} is outside 0..{}",
                    self.tensor_name, self.num_experts
                ));
            }
            let dst_index_offset = rank
                .checked_mul(self.index_bytes_per_expert)
                .ok_or_else(|| format!("{} staged index offset overflows", self.tensor_name))?;
            let src_index_offset = expert
                .checked_mul(self.index_bytes_per_expert)
                .ok_or_else(|| format!("{} source index offset overflows", self.tensor_name))?;
            index_staging
                .copy_from_buffer(
                    dst_index_offset,
                    self.index_buffer.as_ref(),
                    src_index_offset,
                    self.index_bytes_per_expert,
                    Some(stream),
                )
                .map_err(|error| {
                    format!("failed to stage {} index slab: {error}", self.tensor_name)
                })?;
            let dst_scale_offset = rank
                .checked_mul(self.scale_bytes_per_expert)
                .ok_or_else(|| format!("{} staged scale offset overflows", self.tensor_name))?;
            let src_scale_offset = expert
                .checked_mul(self.scale_bytes_per_expert)
                .ok_or_else(|| format!("{} source scale offset overflows", self.tensor_name))?;
            scale_staging
                .copy_from_buffer(
                    dst_scale_offset,
                    self.scale_buffer.as_ref(),
                    src_scale_offset,
                    self.scale_bytes_per_expert,
                    Some(stream),
                )
                .map_err(|error| {
                    format!("failed to stage {} scale slab: {error}", self.tensor_name)
                })?;
        }
        let elements = self.selected_elements(selected_expert_ids.len())?;
        ullm_runtime_sys::aq4_dequant_f32(
            index_staging,
            scale_staging,
            self.codebook_buffer.as_ref(),
            &self.scale_values,
            self.group_size,
            self.tensor_scale,
            elements,
            dequantized,
            Some(stream),
        )
        .map_err(|error| {
            format!(
                "failed to dequantize selected {} slabs: {error}",
                self.tensor_name
            )
        })
    }
}

struct MoEExecutionBuffers {
    routing_scores: ullm_runtime_sys::RuntimeBuffer,
    selected_expert_ids: ullm_runtime_sys::RuntimeBuffer,
    boundary_tie_flags: ullm_runtime_sys::RuntimeBuffer,
    local_expert_ids: ullm_runtime_sys::RuntimeBuffer,
    gathered_hidden: ullm_runtime_sys::RuntimeBuffer,
    gate_up_index_staging: ullm_runtime_sys::RuntimeBuffer,
    gate_up_scale_staging: ullm_runtime_sys::RuntimeBuffer,
    gate_up_dequantized: ullm_runtime_sys::RuntimeBuffer,
    gate_up_output: ullm_runtime_sys::RuntimeBuffer,
    activation: ullm_runtime_sys::RuntimeBuffer,
    down_index_staging: ullm_runtime_sys::RuntimeBuffer,
    down_scale_staging: ullm_runtime_sys::RuntimeBuffer,
    down_dequantized: ullm_runtime_sys::RuntimeBuffer,
    expert_output: ullm_runtime_sys::RuntimeBuffer,
    routed_output: ullm_runtime_sys::RuntimeBuffer,
    shared_output: ullm_runtime_sys::RuntimeBuffer,
    shared_gate: ullm_runtime_sys::RuntimeBuffer,
    shared_gated_output: ullm_runtime_sys::RuntimeBuffer,
    total_output: ullm_runtime_sys::RuntimeBuffer,
    allocated_bytes: usize,
}

struct MoELayer {
    layer_index: usize,
    hidden: usize,
    num_experts: usize,
    top_k: usize,
    intermediate: usize,
    shared_intermediate: usize,
    router: RawBf16Matrix,
    shared_expert_gate: RawBf16Matrix,
    gate_up: Aq4MoeTensor,
    down: Aq4MoeTensor,
    last_route: Option<Qwen35MoeRouteTrace>,
}

#[derive(Debug, Default)]
struct MoEExecutionBufferRequirements {
    top_k: usize,
    hidden: usize,
    gate_up_elements: usize,
    gate_up_output_elements: usize,
    activation_elements: usize,
    down_elements: usize,
    expert_output_elements: usize,
    gate_up_index_staging_bytes: usize,
    gate_up_scale_staging_bytes: usize,
    down_index_staging_bytes: usize,
    down_scale_staging_bytes: usize,
}

fn f32_bytes(elements: usize, label: &str) -> Result<usize, String> {
    elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{label} f32 byte count overflows"))
}

fn i32_bytes(elements: usize, label: &str) -> Result<usize, String> {
    elements
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| format!("{label} i32 byte count overflows"))
}

fn u32_bytes(elements: usize, label: &str) -> Result<usize, String> {
    elements
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| format!("{label} u32 byte count overflows"))
}

fn allocate_f32(
    context: &mut ullm_runtime_sys::RuntimeContext,
    elements: usize,
    label: &str,
) -> Result<ullm_runtime_sys::RuntimeBuffer, String> {
    context
        .alloc_buffer(f32_bytes(elements, label)?)
        .map_err(|error| format!("failed to allocate {label}: {error}"))
}

fn decode_f32(bytes: &[u8], label: &str) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err(format!("{label} F32 bytes are not aligned"));
    }
    Ok(bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 width is fixed")))
        .collect())
}

fn decode_i32(bytes: &[u8], label: &str) -> Result<Vec<i32>, String> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<i32>()) {
        return Err(format!("{label} i32 bytes are not aligned"));
    }
    Ok(bytes
        .chunks_exact(std::mem::size_of::<i32>())
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("i32 width is fixed")))
        .collect())
}

fn decode_u32(bytes: &[u8], label: &str) -> Result<Vec<u32>, String> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<u32>()) {
        return Err(format!("{label} u32 bytes are not aligned"));
    }
    Ok(bytes
        .chunks_exact(std::mem::size_of::<u32>())
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("u32 width is fixed")))
        .collect())
}

fn encode_i32(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

impl MoEExecutionBufferRequirements {
    fn include(&mut self, layer: &MoELayer) -> Result<(), String> {
        let assignments = layer.top_k;
        let gate_up_elements = layer.gate_up.selected_elements(assignments)?;
        let down_elements = layer.down.selected_elements(assignments)?;
        let gate_up_output_elements = assignments
            .checked_mul(layer.gate_up.rows_per_expert)
            .ok_or_else(|| "shared MoE gate/up output element count overflows".to_string())?;
        let activation_elements = assignments
            .checked_mul(layer.intermediate)
            .ok_or_else(|| "shared MoE activation element count overflows".to_string())?;
        let expert_output_elements = assignments
            .checked_mul(layer.hidden)
            .ok_or_else(|| "shared MoE expert output element count overflows".to_string())?;
        let gate_up_index_staging_bytes = assignments
            .checked_mul(layer.gate_up.index_bytes_per_expert)
            .ok_or_else(|| "shared MoE gate/up index staging bytes overflow".to_string())?;
        let gate_up_scale_staging_bytes = assignments
            .checked_mul(layer.gate_up.scale_bytes_per_expert)
            .ok_or_else(|| "shared MoE gate/up scale staging bytes overflow".to_string())?;
        let down_index_staging_bytes =
            assignments
                .checked_mul(layer.down.index_bytes_per_expert)
                .ok_or_else(|| "shared MoE down index staging bytes overflow".to_string())?;
        let down_scale_staging_bytes =
            assignments
                .checked_mul(layer.down.scale_bytes_per_expert)
                .ok_or_else(|| "shared MoE down scale staging bytes overflow".to_string())?;
        self.top_k = self.top_k.max(assignments);
        self.hidden = self.hidden.max(layer.hidden);
        self.gate_up_elements = self.gate_up_elements.max(gate_up_elements);
        self.gate_up_output_elements = self.gate_up_output_elements.max(gate_up_output_elements);
        self.activation_elements = self.activation_elements.max(activation_elements);
        self.down_elements = self.down_elements.max(down_elements);
        self.expert_output_elements = self.expert_output_elements.max(expert_output_elements);
        self.gate_up_index_staging_bytes = self
            .gate_up_index_staging_bytes
            .max(gate_up_index_staging_bytes);
        self.gate_up_scale_staging_bytes = self
            .gate_up_scale_staging_bytes
            .max(gate_up_scale_staging_bytes);
        self.down_index_staging_bytes = self.down_index_staging_bytes.max(down_index_staging_bytes);
        self.down_scale_staging_bytes = self.down_scale_staging_bytes.max(down_scale_staging_bytes);
        Ok(())
    }

    fn validate(&self) -> Result<(), String> {
        if self.top_k == 0
            || self.hidden == 0
            || self.gate_up_elements == 0
            || self.down_elements == 0
        {
            return Err("cannot allocate an empty shared MoE decode workspace".into());
        }
        Ok(())
    }
}

impl MoEExecutionBuffers {
    /// All decoder layers execute serially for batch-1 decode, so their routed
    /// expert staging/dequant/output buffers are one reusable workspace rather
    /// than forty simultaneous allocations.  This follows the package ledger's
    /// one active MoE gather/workspace reserve.
    fn allocate(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        requirements: &MoEExecutionBufferRequirements,
    ) -> Result<Self, String> {
        requirements.validate()?;
        let routing_scores_bytes = f32_bytes(requirements.top_k, "shared MoE routing scores")?;
        let selected_expert_ids_bytes =
            i32_bytes(requirements.top_k, "shared MoE selected expert IDs")?;
        let boundary_tie_flags_bytes = u32_bytes(1, "shared MoE boundary tie flags")?;
        let local_expert_ids_bytes = i32_bytes(requirements.top_k, "shared local MoE expert IDs")?;
        let gathered_hidden_bytes = f32_bytes(
            requirements.expert_output_elements,
            "shared MoE gathered hidden",
        )?;
        let gate_up_dequantized_bytes =
            f32_bytes(requirements.gate_up_elements, "shared MoE gate/up dequant")?;
        let gate_up_output_bytes = f32_bytes(
            requirements.gate_up_output_elements,
            "shared MoE gate/up output",
        )?;
        let activation_bytes =
            f32_bytes(requirements.activation_elements, "shared MoE activation")?;
        let down_dequantized_bytes =
            f32_bytes(requirements.down_elements, "shared MoE down dequant")?;
        let expert_output_bytes = f32_bytes(
            requirements.expert_output_elements,
            "shared MoE expert output",
        )?;
        let hidden_bytes = f32_bytes(requirements.hidden, "shared MoE hidden")?;
        let allocated_bytes = [
            routing_scores_bytes,
            selected_expert_ids_bytes,
            boundary_tie_flags_bytes,
            local_expert_ids_bytes,
            gathered_hidden_bytes,
            requirements.gate_up_index_staging_bytes,
            requirements.gate_up_scale_staging_bytes,
            gate_up_dequantized_bytes,
            gate_up_output_bytes,
            activation_bytes,
            requirements.down_index_staging_bytes,
            requirements.down_scale_staging_bytes,
            down_dequantized_bytes,
            expert_output_bytes,
            hidden_bytes,
            hidden_bytes,
            std::mem::size_of::<f32>(),
            hidden_bytes,
            hidden_bytes,
        ]
        .into_iter()
        .try_fold(0_usize, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or_else(|| "shared MoE decode workspace byte count overflows".to_string())
        })?;
        let local_ids = (0..requirements.top_k)
            .map(|rank| i32::try_from(rank).map_err(|_| "local expert ID exceeds i32".to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut local_expert_ids = context
            .alloc_buffer(local_expert_ids_bytes)
            .map_err(|error| format!("failed to allocate shared local MoE expert IDs: {error}"))?;
        local_expert_ids
            .copy_from_host(0, &encode_i32(&local_ids), Some(stream))
            .map_err(|error| format!("failed to upload shared local MoE expert IDs: {error}"))?;
        Ok(Self {
            routing_scores: context
                .alloc_buffer(routing_scores_bytes)
                .map_err(|error| {
                    format!("failed to allocate shared MoE routing scores: {error}")
                })?,
            selected_expert_ids: context.alloc_buffer(selected_expert_ids_bytes).map_err(
                |error| format!("failed to allocate shared MoE selected expert IDs: {error}"),
            )?,
            boundary_tie_flags: context
                .alloc_buffer(boundary_tie_flags_bytes)
                .map_err(|error| format!("failed to allocate shared MoE tie flags: {error}"))?,
            local_expert_ids,
            gathered_hidden: context
                .alloc_buffer(gathered_hidden_bytes)
                .map_err(|error| {
                    format!("failed to allocate shared MoE gathered hidden: {error}")
                })?,
            gate_up_index_staging: context
                .alloc_buffer(requirements.gate_up_index_staging_bytes)
                .map_err(|error| {
                    format!("failed to allocate shared MoE gate/up index staging: {error}")
                })?,
            gate_up_scale_staging: context
                .alloc_buffer(requirements.gate_up_scale_staging_bytes)
                .map_err(|error| {
                    format!("failed to allocate shared MoE gate/up scale staging: {error}")
                })?,
            gate_up_dequantized: context.alloc_buffer(gate_up_dequantized_bytes).map_err(
                |error| format!("failed to allocate shared MoE gate/up dequant: {error}"),
            )?,
            gate_up_output: context
                .alloc_buffer(gate_up_output_bytes)
                .map_err(|error| {
                    format!("failed to allocate shared MoE gate/up output: {error}")
                })?,
            activation: context
                .alloc_buffer(activation_bytes)
                .map_err(|error| format!("failed to allocate shared MoE activation: {error}"))?,
            down_index_staging: context
                .alloc_buffer(requirements.down_index_staging_bytes)
                .map_err(|error| {
                    format!("failed to allocate shared MoE down index staging: {error}")
                })?,
            down_scale_staging: context
                .alloc_buffer(requirements.down_scale_staging_bytes)
                .map_err(|error| {
                    format!("failed to allocate shared MoE down scale staging: {error}")
                })?,
            down_dequantized: context
                .alloc_buffer(down_dequantized_bytes)
                .map_err(|error| format!("failed to allocate shared MoE down dequant: {error}"))?,
            expert_output: context
                .alloc_buffer(expert_output_bytes)
                .map_err(|error| format!("failed to allocate shared MoE expert output: {error}"))?,
            routed_output: context
                .alloc_buffer(hidden_bytes)
                .map_err(|error| format!("failed to allocate shared MoE routed output: {error}"))?,
            shared_output: context
                .alloc_buffer(hidden_bytes)
                .map_err(|error| format!("failed to allocate shared MoE shared output: {error}"))?,
            shared_gate: context
                .alloc_buffer(std::mem::size_of::<f32>())
                .map_err(|error| format!("failed to allocate shared MoE shared gate: {error}"))?,
            shared_gated_output: context.alloc_buffer(hidden_bytes).map_err(|error| {
                format!("failed to allocate shared MoE gated shared output: {error}")
            })?,
            total_output: context
                .alloc_buffer(hidden_bytes)
                .map_err(|error| format!("failed to allocate shared MoE total output: {error}"))?,
            allocated_bytes,
        })
    }
}

impl MoELayer {
    #[allow(clippy::too_many_arguments)]
    fn load(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        registry: &mut WeightRegistry,
        package_path: &str,
        layer_index: usize,
        hidden: usize,
        num_experts: usize,
        top_k: usize,
        intermediate: usize,
        shared_intermediate: usize,
        chunk_bytes: usize,
    ) -> Result<Self, String> {
        if num_experts == 0 || top_k == 0 || top_k > num_experts || intermediate == 0 {
            return Err(format!(
                "layer {layer_index} has invalid MoE descriptor geometry"
            ));
        }
        let prefix = format!("model.language_model.layers.{layer_index}.mlp");
        let router = RawBf16Matrix::load(
            context,
            stream,
            package_path,
            format!("{prefix}.gate.weight"),
            chunk_bytes,
        )?;
        if (router.rows, router.cols) != (num_experts, hidden) {
            return Err(format!(
                "layer {layer_index} router shape [{},{}] differs from [{num_experts},{hidden}]",
                router.rows, router.cols
            ));
        }
        let shared_expert_gate = RawBf16Matrix::load(
            context,
            stream,
            package_path,
            format!("{prefix}.shared_expert_gate.weight"),
            chunk_bytes,
        )?;
        if (shared_expert_gate.rows, shared_expert_gate.cols) != (1, hidden) {
            return Err(format!(
                "layer {layer_index} shared-expert gate shape [{},{}] differs from [1,{hidden}]",
                shared_expert_gate.rows, shared_expert_gate.cols
            ));
        }
        let gate_rows = intermediate
            .checked_mul(2)
            .ok_or_else(|| format!("layer {layer_index} gate/up row count overflows"))?;
        let gate_up = Aq4MoeTensor::load(
            context,
            stream,
            registry,
            package_path,
            format!("{prefix}.experts.gate_up_proj"),
            num_experts,
            gate_rows,
            hidden,
            chunk_bytes,
        )?;
        let down = Aq4MoeTensor::load(
            context,
            stream,
            registry,
            package_path,
            format!("{prefix}.experts.down_proj"),
            num_experts,
            hidden,
            intermediate,
            chunk_bytes,
        )?;

        Ok(Self {
            layer_index,
            hidden,
            num_experts,
            top_k,
            intermediate,
            shared_intermediate,
            router,
            shared_expert_gate,
            gate_up,
            down,
            last_route: None,
        })
    }

    fn run_routed(
        &mut self,
        buffers: &mut MoEExecutionBuffers,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        post_normed: &ullm_runtime_sys::RuntimeBuffer,
        label: &str,
    ) -> Result<Qwen35MoeRouteTrace, String> {
        ullm_runtime_sys::moe_route_f32(
            post_normed,
            self.router.buffer.as_ref(),
            ullm_runtime_sys::MoeWeightDtype::Bf16,
            1,
            self.hidden,
            self.num_experts,
            self.top_k,
            &mut buffers.routing_scores,
            &mut buffers.selected_expert_ids,
            &mut buffers.boundary_tie_flags,
            Some(stream),
        )
        .map_err(|error| {
            format!(
                "failed to route {label} layer {}: {error}",
                self.layer_index
            )
        })?;

        let mut ids_bytes = vec![0_u8; i32_bytes(self.top_k, "MoE route IDs")?];
        buffers
            .selected_expert_ids
            .copy_to_host(0, &mut ids_bytes, Some(stream))
            .map_err(|error| format!("failed to copy {label} MoE route IDs: {error}"))?;
        let mut scores_bytes = vec![0_u8; f32_bytes(self.top_k, "MoE route scores")?];
        buffers
            .routing_scores
            .copy_to_host(0, &mut scores_bytes, Some(stream))
            .map_err(|error| format!("failed to copy {label} MoE route scores: {error}"))?;
        let mut ties_bytes = vec![0_u8; u32_bytes(1, "MoE route ties")?];
        buffers
            .boundary_tie_flags
            .copy_to_host(0, &mut ties_bytes, Some(stream))
            .map_err(|error| format!("failed to copy {label} MoE route ties: {error}"))?;
        stream.synchronize().map_err(|error| {
            format!("failed to synchronize {label} MoE route readback: {error}")
        })?;
        let route = Qwen35MoeRouteTrace {
            layer_index: self.layer_index,
            selected_expert_ids: decode_i32(&ids_bytes, "MoE route IDs")?,
            routing_scores: decode_f32(&scores_bytes, "MoE route scores")?,
            boundary_tie_flags: decode_u32(&ties_bytes, "MoE route ties")?,
        };
        if route
            .routing_scores
            .iter()
            .any(|score| !score.is_finite() || *score < 0.0)
        {
            return Err(format!(
                "{label} layer {} produced invalid routing scores",
                self.layer_index
            ));
        }
        self.gate_up.stage_and_dequant_selected(
            &route.selected_expert_ids,
            &mut buffers.gate_up_index_staging,
            &mut buffers.gate_up_scale_staging,
            &mut buffers.gate_up_dequantized,
            stream,
        )?;
        self.down.stage_and_dequant_selected(
            &route.selected_expert_ids,
            &mut buffers.down_index_staging,
            &mut buffers.down_scale_staging,
            &mut buffers.down_dequantized,
            stream,
        )?;
        ullm_runtime_sys::moe_gather_f32(
            post_normed,
            1,
            self.hidden,
            self.top_k,
            &mut buffers.gathered_hidden,
            Some(stream),
        )
        .map_err(|error| format!("failed to gather {label} MoE hidden: {error}"))?;
        ullm_runtime_sys::moe_decode_gemm_f32(
            &buffers.gate_up_dequantized,
            ullm_runtime_sys::MoeWeightDtype::F32,
            &buffers.local_expert_ids,
            &buffers.gathered_hidden,
            self.top_k,
            self.top_k,
            self.gate_up.rows_per_expert,
            self.hidden,
            &mut buffers.gate_up_output,
            Some(stream),
        )
        .map_err(|error| format!("failed to run {label} MoE gate/up decode GEMM: {error}"))?;
        ullm_runtime_sys::moe_gated_silu_f32(
            &buffers.gate_up_output,
            self.top_k,
            self.intermediate,
            &mut buffers.activation,
            Some(stream),
        )
        .map_err(|error| format!("failed to activate {label} MoE gate/up output: {error}"))?;
        ullm_runtime_sys::moe_decode_gemm_f32(
            &buffers.down_dequantized,
            ullm_runtime_sys::MoeWeightDtype::F32,
            &buffers.local_expert_ids,
            &buffers.activation,
            self.top_k,
            self.top_k,
            self.hidden,
            self.intermediate,
            &mut buffers.expert_output,
            Some(stream),
        )
        .map_err(|error| format!("failed to run {label} MoE down decode GEMM: {error}"))?;
        ullm_runtime_sys::moe_scatter_weighted_f32(
            &buffers.expert_output,
            &buffers.routing_scores,
            1,
            self.top_k,
            self.hidden,
            &mut buffers.routed_output,
            Some(stream),
        )
        .map_err(|error| format!("failed to scatter {label} MoE expert output: {error}"))?;
        self.last_route = Some(route.clone());
        Ok(route)
    }

    fn gate_shared_output(
        &mut self,
        buffers: &mut MoEExecutionBuffers,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        post_normed: &ullm_runtime_sys::RuntimeBuffer,
        label: &str,
    ) -> Result<(), String> {
        ullm_runtime_sys::matvec_bf16_f32(
            self.shared_expert_gate.buffer.as_ref(),
            post_normed,
            1,
            self.hidden,
            &mut buffers.shared_gate,
            Some(stream),
        )
        .map_err(|error| format!("failed to project {label} shared-expert gate: {error}"))?;
        ullm_runtime_sys::moe_sigmoid_gate_f32(
            &buffers.shared_gate,
            &buffers.shared_output,
            1,
            self.hidden,
            &mut buffers.shared_gated_output,
            Some(stream),
        )
        .map_err(|error| format!("failed to gate {label} shared-expert output: {error}"))?;
        ullm_runtime_sys::add_f32(
            &buffers.routed_output,
            &buffers.shared_gated_output,
            self.hidden,
            &mut buffers.total_output,
            Some(stream),
        )
        .map_err(|error| format!("failed to combine {label} routed/shared MoE outputs: {error}"))
    }

    fn verify_last_route(
        &self,
        package_path: &str,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        post_normed: &ullm_runtime_sys::RuntimeBuffer,
        chunk_bytes: usize,
    ) -> Result<Qwen35MoeRouterVerification, String> {
        let runtime = self
            .last_route
            .as_ref()
            .ok_or_else(|| format!("layer {} has no route to verify", self.layer_index))?;
        let mut hidden_bytes =
            vec![0_u8; f32_bytes(self.hidden, "MoE router verification hidden")?];
        post_normed
            .copy_to_host(0, &mut hidden_bytes, Some(stream))
            .map_err(|error| {
                format!(
                    "failed to copy layer {} router hidden: {error}",
                    self.layer_index
                )
            })?;
        stream.synchronize().map_err(|error| {
            format!(
                "failed to synchronize layer {} router hidden: {error}",
                self.layer_index
            )
        })?;
        let hidden = decode_f32(&hidden_bytes, "MoE router verification hidden")?;
        let router =
            read_named_passthrough_f32(package_path, &self.router.tensor_name, chunk_bytes)
                .map_err(|error| {
                    format!(
                        "failed to read layer {} router reference: {error}",
                        self.layer_index
                    )
                })?;
        let expected_elements = self
            .num_experts
            .checked_mul(self.hidden)
            .ok_or_else(|| "MoE router reference element count overflows".to_string())?;
        if router.values.len() != expected_elements {
            return Err(format!(
                "layer {} router reference has {} values, expected {expected_elements}",
                self.layer_index,
                router.values.len()
            ));
        }
        let reference = ullm_runtime_sys::moe_route_reference_with_weight_dtype_f32(
            ullm_runtime_sys::MoeShape {
                tokens: 1,
                hidden_size: self.hidden,
                num_experts: self.num_experts,
                top_k: self.top_k,
                intermediate_size: self.intermediate,
                shared_intermediate_size: self.shared_intermediate,
            },
            &hidden,
            &router.values,
            ullm_runtime_sys::MoeWeightDtype::Bf16,
        )?;
        let tie_free = runtime.boundary_tie_flags.iter().all(|flag| *flag == 0)
            && reference.boundary_tie_flags.iter().all(|flag| *flag == 0);
        Ok(Qwen35MoeRouterVerification {
            layer_index: self.layer_index,
            runtime_selected_expert_ids: runtime.selected_expert_ids.clone(),
            reference_selected_expert_ids: reference.selected_expert_ids.clone(),
            runtime_boundary_tie_flags: runtime.boundary_tie_flags.clone(),
            reference_boundary_tie_flags: reference.boundary_tie_flags,
            strict_order_match: tie_free
                .then(|| runtime.selected_expert_ids == reference.selected_expert_ids),
            routing_score_sum: runtime.routing_scores.iter().sum(),
        })
    }
}

enum Qwen35MoeResidentLayer {
    Linear {
        layer_index: usize,
        attention: PackageLinearAttnResidentStepLayer,
        moe: MoELayer,
    },
    Full {
        layer_index: usize,
        attention: PackageSelfAttnResidentStepLayer,
        moe: MoELayer,
    },
}

impl Qwen35MoeResidentLayer {
    fn moe(&self) -> &MoELayer {
        match self {
            Self::Linear { moe, .. } | Self::Full { moe, .. } => moe,
        }
    }

    fn layer_index(&self) -> usize {
        match self {
            Self::Linear { layer_index, .. } | Self::Full { layer_index, .. } => *layer_index,
        }
    }

    fn reset(&mut self, stream: &mut ullm_runtime_sys::RuntimeStream) -> Result<(), String> {
        match self {
            Self::Linear { attention, .. } => attention.reset_request_state_synchronized(stream),
            Self::Full { attention, .. } => attention.reset_request_state_synchronized(stream),
        }
    }

    fn run(
        &mut self,
        buffers: &mut MoEExecutionBuffers,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        input: &ullm_runtime_sys::RuntimeBuffer,
        rope_position: usize,
        cache_position: usize,
        rms_norm_epsilon: f32,
        label: &str,
    ) -> Result<Qwen35MoeRouteTrace, String> {
        match self {
            Self::Linear { attention, moe, .. } => {
                attention.run_device_step_through_post_norm_with_rms_epsilon(
                    stream,
                    PackageLinearAttnResidentStepInput::ExternalBuffer(input),
                    rms_norm_epsilon,
                    label,
                )?;
                let route =
                    moe.run_routed(buffers, stream, attention.post_normed_buffer(), label)?;
                attention.run_moe_shared_expert(stream, &mut buffers.shared_output, label)?;
                moe.gate_shared_output(buffers, stream, attention.post_normed_buffer(), label)?;
                attention.finish_external_mlp(stream, &buffers.total_output, label)?;
                Ok(route)
            }
            Self::Full { attention, moe, .. } => {
                attention.run_device_step_through_post_norm_with_rms_epsilon(
                    stream,
                    PackageSelfAttnResidentStepInput::ExternalBuffer(input),
                    QWEN35_MOE_TEXT_ROTARY_DIM,
                    QWEN35_MOE_TEXT_ROPE_BASE,
                    rope_position,
                    cache_position,
                    rms_norm_epsilon,
                    label,
                )?;
                let route =
                    moe.run_routed(buffers, stream, attention.post_normed_buffer(), label)?;
                attention.run_moe_shared_expert(stream, &mut buffers.shared_output, label)?;
                moe.gate_shared_output(buffers, stream, attention.post_normed_buffer(), label)?;
                attention.finish_external_mlp(stream, &buffers.total_output, label)?;
                Ok(route)
            }
        }
    }

    fn output_buffer(&self) -> &ullm_runtime_sys::RuntimeBuffer {
        match self {
            Self::Linear { attention, .. } => attention.output_buffer(),
            Self::Full { attention, .. } => attention.output_buffer(),
        }
    }

    fn verify_last_route(
        &self,
        package_path: &str,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        chunk_bytes: usize,
    ) -> Result<Qwen35MoeRouterVerification, String> {
        match self {
            Self::Linear { attention, moe, .. } => moe.verify_last_route(
                package_path,
                stream,
                attention.post_normed_buffer(),
                chunk_bytes,
            ),
            Self::Full { attention, moe, .. } => moe.verify_last_route(
                package_path,
                stream,
                attention.post_normed_buffer(),
                chunk_bytes,
            ),
        }
    }
}

/// A load-once, resettable Qwen3.5-35B-A3B AQ4_0 text runtime.
///
/// The context owns all buffers.  Fields holding runtime allocations must
/// precede stream/context so their destructors run first.
pub struct Qwen35MoeAq4Runtime {
    embedding: PackageEmbeddingRuntime,
    layers: Vec<Qwen35MoeResidentLayer>,
    final_norm: PackageFinalNormRuntime,
    lm_head: PackageLmHeadRuntime,
    ping_buffers: [ullm_runtime_sys::RuntimeBuffer; 2],
    moe_buffers: MoEExecutionBuffers,
    _expert_registry: WeightRegistry,
    stream: ullm_runtime_sys::RuntimeStream,
    _context: ullm_runtime_sys::RuntimeContext,
    package_dir: PathBuf,
    package_path: String,
    descriptor: ResidentModelDescriptor,
    context_length: usize,
    kv_block_size: usize,
    cache_blocks: usize,
    declared_package_bytes: u64,
    device_total_global_mem_bytes: u64,
    resident_expert_payload_bytes: u64,
    shared_moe_decode_workspace_bytes: u64,
    position: usize,
}

fn package_path_text(path: &Path) -> Result<&str, String> {
    path.to_str()
        .ok_or_else(|| format!("package path {} is not UTF-8", path.display()))
}

fn block_table(context_length: usize, kv_block_size: usize) -> Result<Vec<u32>, String> {
    if context_length == 0 || kv_block_size == 0 {
        return Err("Qwen3.5 MoE context length and KV block size must be positive".to_string());
    }
    let blocks = context_length.div_ceil(kv_block_size);
    (0..blocks)
        .map(|index| {
            u32::try_from(index)
                .map_err(|_| format!("Qwen3.5 MoE KV block index {index} exceeds u32"))
        })
        .collect()
}

fn moe_descriptor_geometry(
    layer: &crate::model_config::ResidentLayerDescriptor,
) -> Result<(usize, usize, usize, usize), String> {
    match &layer.mlp {
        ResidentMlpDescriptor::MoE {
            num_experts,
            experts_per_token,
            expert_intermediate_size,
            shared_expert_intermediate_size,
            activation,
        } if activation == "silu" => Ok((
            *num_experts,
            *experts_per_token,
            *expert_intermediate_size,
            *shared_expert_intermediate_size,
        )),
        ResidentMlpDescriptor::MoE { activation, .. } => Err(format!(
            "Qwen3.5 MoE layer {} activation must be silu, got {activation}",
            layer.layer_index
        )),
        ResidentMlpDescriptor::Dense { .. } => Err(format!(
            "Qwen3.5 MoE layer {} unexpectedly has a dense MLP descriptor",
            layer.layer_index
        )),
    }
}

fn validate_qwen35_moe_descriptor(descriptor: &ResidentModelDescriptor) -> Result<(), String> {
    if descriptor.architecture != ModelArchitectureKind::Qwen35MoeText {
        return Err(format!(
            "Qwen3.5 MoE AQ4_0 runtime requires {}, got {}",
            ModelArchitectureKind::Qwen35MoeText.architecture_name(),
            descriptor.architecture.architecture_name()
        ));
    }
    if descriptor.decoder.hidden_size == 0
        || descriptor.layers.is_empty()
        || descriptor.decoder.norm_weight_convention != RmsNormWeightConvention::OnePlusWeight
    {
        return Err(
            "Qwen3.5 MoE descriptor is missing decoder geometry or 1+weight RMSNorm".into(),
        );
    }
    for (position, layer) in descriptor.layers.iter().enumerate() {
        if layer.layer_index != position {
            return Err(format!(
                "Qwen3.5 MoE descriptor layer order is not contiguous at {position}"
            ));
        }
        let _ = moe_descriptor_geometry(layer)?;
        match layer.attention.kind {
            DecoderLayerKind::FullAttention => {
                let rope = layer.attention.rope.as_ref().ok_or_else(|| {
                    format!("Qwen3.5 MoE full layer {position} has no mRoPE descriptor")
                })?;
                if rope.kind != ResidentRopeKind::Mrope
                    || rope.rotary_dim != Some(QWEN35_MOE_TEXT_ROTARY_DIM)
                    || rope.theta.to_bits() != QWEN35_MOE_TEXT_ROPE_BASE.to_bits()
                    || !rope.mrope_interleaved
                    || rope.mrope_sections != [11, 11, 10]
                    || !layer.attention.q_norm
                    || !layer.attention.k_norm
                    || !layer.attention.output_gate
                    || layer.attention.kv_cache != ResidentKvCacheMode::Own
                {
                    return Err(format!(
                        "Qwen3.5 MoE full layer {position} does not match the inspected mRoPE/Q-gate/KV contract"
                    ));
                }
            }
            DecoderLayerKind::LinearAttention => {
                if layer.attention.kv_cache != ResidentKvCacheMode::LinearState
                    || layer.attention.linear_attention.is_none()
                {
                    return Err(format!(
                        "Qwen3.5 MoE linear layer {position} does not declare recurrent state"
                    ));
                }
            }
            DecoderLayerKind::SlidingAttention => {
                return Err(format!(
                    "Qwen3.5 MoE layer {position} uses unsupported sliding attention"
                ));
            }
        }
    }
    Ok(())
}

impl Qwen35MoeAq4Runtime {
    pub fn load(config: Qwen35MoeAq4ModelLoadConfig) -> Result<Self, String> {
        if config.chunk_bytes == 0 || config.context_length == 0 || config.kv_block_size == 0 {
            return Err("Qwen3.5 MoE AQ4_0 load config has a zero size".to_string());
        }
        let loaded = load_model_config_from_package(&config.package_dir)?;
        let descriptor = loaded.resident_descriptor()?;
        validate_qwen35_moe_descriptor(&descriptor)?;
        if config.context_length > descriptor.decoder.max_position_embeddings {
            return Err(format!(
                "Qwen3.5 MoE context length {} exceeds descriptor maximum {}",
                config.context_length, descriptor.decoder.max_position_embeddings
            ));
        }
        let package_path = package_path_text(&config.package_dir)?.to_owned();
        let block_table = block_table(config.context_length, config.kv_block_size)?;
        let cache_blocks = block_table.len();
        let summary = inspect_package(&config.package_dir)?;
        let declared_package_bytes = summary.referenced_file_bytes;
        let mut context =
            ullm_runtime_sys::RuntimeContext::create(config.device_index).map_err(|error| {
                format!("failed to create Qwen3.5 MoE AQ4_0 runtime context: {error}")
            })?;
        let device = context
            .device_info()
            .map_err(|error| format!("failed to query Qwen3.5 MoE AQ4_0 device: {error}"))?;
        if let Some(expected) = config.expected_architecture.as_deref() {
            require_device_architecture(&device, expected)
                .map_err(|error| format!("Qwen3.5 MoE AQ4_0 {error}"))?;
        }
        let mut stream = context
            .create_stream()
            .map_err(|error| format!("failed to create Qwen3.5 MoE AQ4_0 stream: {error}"))?;
        let hidden = descriptor.decoder.hidden_size;
        let mut expert_registry = WeightRegistry::new();
        let mut shared_buffers = PackageResidentSharedBufferRegistry::new();
        let mut layers = Vec::with_capacity(descriptor.layers.len());
        for layer_descriptor in &descriptor.layers {
            let layer_index = layer_descriptor.layer_index;
            let (num_experts, top_k, intermediate, shared_intermediate) =
                moe_descriptor_geometry(layer_descriptor)?;
            let moe = MoELayer::load(
                &mut context,
                &mut stream,
                &mut expert_registry,
                &package_path,
                layer_index,
                hidden,
                num_experts,
                top_k,
                intermediate,
                shared_intermediate,
                config.chunk_bytes,
            )?;
            let resident = match layer_descriptor.attention.kind {
                DecoderLayerKind::LinearAttention => Qwen35MoeResidentLayer::Linear {
                    layer_index,
                    attention:
                        PackageLinearAttnResidentStepLayer::load_moe_shared_with_registry_geometry(
                            &mut context,
                            &mut stream,
                            &mut expert_registry,
                            Some(&mut shared_buffers),
                            &package_path,
                            config.chunk_bytes,
                            layer_index,
                            None,
                            PackageLinearAttnGeometry {
                                hidden,
                                key_heads: layer_descriptor
                                    .attention
                                    .linear_attention
                                    .as_ref()
                                    .ok_or_else(|| {
                                        format!(
                                            "layer {layer_index} has no linear-attention descriptor"
                                        )
                                    })?
                                    .num_key_heads,
                                value_heads: layer_descriptor
                                    .attention
                                    .linear_attention
                                    .as_ref()
                                    .expect("checked linear-attention descriptor")
                                    .num_value_heads,
                                key_dim: layer_descriptor
                                    .attention
                                    .linear_attention
                                    .as_ref()
                                    .expect("checked linear-attention descriptor")
                                    .key_head_dim,
                                value_dim: layer_descriptor
                                    .attention
                                    .linear_attention
                                    .as_ref()
                                    .expect("checked linear-attention descriptor")
                                    .value_head_dim,
                                kernel_size: layer_descriptor
                                    .attention
                                    .linear_attention
                                    .as_ref()
                                    .expect("checked linear-attention descriptor")
                                    .conv_kernel_dim,
                            },
                        )
                        .map_err(|error| {
                            format!(
                                "failed to load Qwen3.5 MoE linear layer {layer_index}: {error}"
                            )
                        })?,
                    moe,
                },
                DecoderLayerKind::FullAttention => Qwen35MoeResidentLayer::Full {
                    layer_index,
                    attention: PackageSelfAttnResidentStepLayer::load_moe_shared_with_registry(
                        &mut context,
                        &mut stream,
                        &mut expert_registry,
                        Some(&mut shared_buffers),
                        &package_path,
                        config.chunk_bytes,
                        layer_index,
                        &block_table,
                        config.kv_block_size,
                        cache_blocks,
                        None,
                    )
                    .map_err(|error| {
                        format!(
                            "failed to load Qwen3.5 MoE full-attention layer {layer_index}: {error}"
                        )
                    })?,
                    moe,
                },
                DecoderLayerKind::SlidingAttention => unreachable!("descriptor was validated"),
            };
            layers.push(resident);
        }
        let mut moe_workspace_requirements = MoEExecutionBufferRequirements::default();
        for layer in &layers {
            moe_workspace_requirements.include(layer.moe())?;
        }
        let moe_buffers =
            MoEExecutionBuffers::allocate(&mut context, &mut stream, &moe_workspace_requirements)?;
        let shared_moe_decode_workspace_bytes = u64::try_from(moe_buffers.allocated_bytes)
            .map_err(|_| "shared MoE decode workspace bytes exceed u64".to_string())?;
        let mut final_norm =
            read_named_passthrough_f32(&package_path, QWEN3_FINAL_NORM_TENSOR, config.chunk_bytes)?;
        final_norm.values =
            effective_qwen35_rmsnorm_weight_values(QWEN3_FINAL_NORM_TENSOR, &final_norm.values);
        let final_norm =
            PackageFinalNormRuntime::load(&mut context, &mut stream, &final_norm, hidden)?;
        let embedding = PackageEmbeddingRuntime::load_if_available(
            &mut context,
            &mut stream,
            &package_path,
            config.chunk_bytes,
            hidden,
        )?
        .ok_or_else(|| {
            "Qwen3.5 MoE AQ4_0 package has no resident BF16/AQ4_0 embedding".to_string()
        })?;
        let lm_head = PackageLmHeadRuntime::load(
            PackageLmHeadMode::GpuResidentF32,
            &mut context,
            &mut stream,
            &package_path,
            config.chunk_bytes,
            hidden,
            config.lm_head_chunk_rows,
        )?;
        if !lm_head.supports_device_input() {
            return Err("Qwen3.5 MoE AQ4_0 runtime requires a resident device lm_head".into());
        }
        let ping_buffers = [
            allocate_f32(&mut context, hidden, "Qwen3.5 MoE ping 0")?,
            allocate_f32(&mut context, hidden, "Qwen3.5 MoE ping 1")?,
        ];
        stream
            .synchronize()
            .map_err(|error| format!("failed to synchronize Qwen3.5 MoE AQ4_0 load: {error}"))?;
        let resident_expert_payload_bytes = expert_registry.resident_payload_bytes();
        Ok(Self {
            embedding,
            layers,
            final_norm,
            lm_head,
            ping_buffers,
            moe_buffers,
            _expert_registry: expert_registry,
            stream,
            _context: context,
            package_dir: config.package_dir,
            package_path,
            descriptor,
            context_length: config.context_length,
            kv_block_size: config.kv_block_size,
            cache_blocks,
            declared_package_bytes,
            device_total_global_mem_bytes: device.total_global_mem,
            resident_expert_payload_bytes,
            shared_moe_decode_workspace_bytes,
            position: 0,
        })
    }

    pub fn descriptor(&self) -> &ResidentModelDescriptor {
        &self.descriptor
    }

    pub fn package_dir(&self) -> &Path {
        &self.package_dir
    }

    pub fn residency(&self) -> Qwen35MoeAq4Residency {
        Qwen35MoeAq4Residency {
            declared_package_bytes: self.declared_package_bytes,
            device_total_global_mem_bytes: self.device_total_global_mem_bytes,
            context_length: self.context_length,
            kv_block_size: self.kv_block_size,
            cache_blocks: self.cache_blocks,
            resident_expert_payload_bytes: self.resident_expert_payload_bytes,
            shared_moe_decode_workspace_bytes: self.shared_moe_decode_workspace_bytes,
        }
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn reset_request_state(&mut self) -> Result<(), String> {
        for layer in &mut self.layers {
            layer.reset(&mut self.stream)?;
        }
        self.position = 0;
        Ok(())
    }

    pub fn dispatch_token(
        &mut self,
        token_id: usize,
        top_k: usize,
        label: &str,
    ) -> Result<Qwen35MoeAq4Step, String> {
        if top_k == 0 {
            return Err("Qwen3.5 MoE top_k must be positive".into());
        }
        if self.position >= self.context_length {
            return Err(format!(
                "{label} position {} reaches configured context length {}",
                self.position, self.context_length
            ));
        }
        let started = Instant::now();
        self.embedding.gather_token_to_buffer(
            &mut self.stream,
            token_id,
            &mut self.ping_buffers[0],
            label,
        )?;
        let mut current = 0_usize;
        let mut routes = Vec::with_capacity(self.layers.len());
        for layer_position in 0..self.layers.len() {
            let next = 1 - current;
            let layer_label = format!(
                "{label} layer {}",
                self.layers[layer_position].layer_index()
            );
            let route = self.layers[layer_position].run(
                &mut self.moe_buffers,
                &mut self.stream,
                &self.ping_buffers[current],
                self.position,
                self.position,
                self.descriptor.decoder.rms_norm_epsilon,
                &layer_label,
            )?;
            let output = self.layers[layer_position].output_buffer();
            self.ping_buffers[next]
                .copy_from_buffer(
                    0,
                    output,
                    0,
                    f32_bytes(self.descriptor.decoder.hidden_size, "Qwen3.5 MoE ping copy")?,
                    Some(&mut self.stream),
                )
                .map_err(|error| {
                    format!("failed to copy {layer_label} output into ping buffer: {error}")
                })?;
            routes.push(route);
            current = next;
        }
        self.final_norm
            .normalize_device(&mut self.stream, &self.ping_buffers[current], label)?;
        let top_logits = self.lm_head.top_logits_from_device_buffer(
            &mut self.stream,
            self.final_norm.output_buffer(),
            top_k,
        )?;
        let position = self.position;
        self.position = self
            .position
            .checked_add(1)
            .ok_or_else(|| "Qwen3.5 MoE position overflows".to_string())?;
        Ok(Qwen35MoeAq4Step {
            position,
            top_logits,
            routes,
            wall_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }

    pub fn generate_greedy(
        &mut self,
        prompt_token_ids: &[usize],
        max_new_tokens: usize,
    ) -> Result<Qwen35MoeAq4Generation, String> {
        if prompt_token_ids.is_empty() {
            return Err("Qwen3.5 MoE generation requires at least one prompt token".into());
        }
        let started = Instant::now();
        self.reset_request_state()?;
        let mut final_step = None;
        let mut prompt_wall_ms = 0.0_f64;
        for token_id in prompt_token_ids {
            let step = self.dispatch_token(*token_id, 1, "Qwen3.5 MoE prompt")?;
            prompt_wall_ms += step.wall_ms;
            final_step = Some(step);
        }
        let mut generated_token_ids = Vec::with_capacity(max_new_tokens);
        let mut decode_wall_ms = 0.0_f64;
        for _ in 0..max_new_tokens {
            let step = final_step.as_ref().expect("prompt dispatched");
            let next = step
                .top_logits
                .first()
                .ok_or_else(|| "Qwen3.5 MoE lm_head returned no top token".to_string())?
                .token_id;
            generated_token_ids.push(next);
            let step = self.dispatch_token(next, 1, "Qwen3.5 MoE decode")?;
            decode_wall_ms += step.wall_ms;
            final_step = Some(step);
        }
        Ok(Qwen35MoeAq4Generation {
            prompt_token_ids: prompt_token_ids.to_vec(),
            generated_token_ids,
            final_step: final_step.expect("prompt dispatched"),
            prompt_wall_ms,
            decode_wall_ms,
            wall_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Independently recomputes the most recent token's router selection from
    /// the exact raw BF16 router and the post-norm hidden boundary.  Ties are
    /// reported rather than converted into an artificial pass/fail decision.
    pub fn verify_last_token_routes(&mut self) -> Result<Vec<Qwen35MoeRouterVerification>, String> {
        self.layers
            .iter()
            .map(|layer| {
                layer.verify_last_route(&self.package_path, &mut self.stream, 64 * 1024 * 1024)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_sized_config_keeps_the_declared_r9700_contract() {
        let config = Qwen35MoeAq4ModelLoadConfig::production_sized("/package", 0);
        assert_eq!(config.context_length, 262_144);
        assert_eq!(config.kv_block_size, 256);
        assert_eq!(config.expected_architecture.as_deref(), Some("gfx1201"));
    }

    #[test]
    fn cache_block_table_is_contiguous() -> Result<(), String> {
        assert_eq!(block_table(513, 256)?, vec![0, 1, 2]);
        assert!(block_table(0, 256).is_err());
        Ok(())
    }
}
