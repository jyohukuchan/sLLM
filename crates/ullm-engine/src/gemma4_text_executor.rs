// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic-only, non-quantized Gemma4 text execution.
//!
//! `Gemma4TextExecutor` deliberately consumes the inspected source checkpoint
//! directly.  Its original diagnostic mode streams BF16 safetensors weights
//! to the existing BF16 x F32 matvec primitive; its resident mode uploads the
//! complete BF16 source checkpoint once to the selected R9700 and keeps a
//! device-resident paged KV cache for the text decoder.  Activation-side math remains F32 in both modes so the
//! architecture trace stays useful for localizing a mismatch.  This is not an
//! SQ8_0/AQ4_0 conversion or a serving runtime.
//!
//! The implementation is intentionally fail-closed around the exact Hugging
//! Face branches needed by `google/gemma-4-E2B`: causal local/full attention,
//! direct-weight RMSNorm, PLE, tied embedding/head, and final logit soft-cap.

use crate::host_bytes::{decode_f32_le_values, encode_f32_to_bytes, encode_u32_to_bytes};
use crate::model_config::{
    DecoderLayerKind, Gemma4TextConfig, LoadedModelConfig, ResidentKvCacheMode,
    ResidentMlpDescriptor, ResidentModelDescriptor, ResidentRopeDescriptor, ResidentRopeKind,
    load_model_config_from_dir,
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;
use ullm_runtime_sys::{
    DeviceInfo, RuntimeBuffer, RuntimeContext, RuntimeStream, add_f32, bf16_row_f32, device_count,
    device_info, gelu_tanh_mul_f32, gemma_bf16_matmul_f32, gemma_proportional_rope_f32,
    matvec_bf16_f32, paged_decode_attn_f32, paged_kv_write_f32, rmsnorm_f32, rope_f32,
    segmented_rmsnorm_f32,
};

pub const GEMMA4_TEXT_MODEL_FILE: &str = "model.safetensors";
pub const GEMMA4_TEXT_WEIGHT_PREFIX: &str = "model.language_model.";
pub const GEMMA4_TEXT_EMBED_TOKENS: &str = "model.language_model.embed_tokens.weight";
pub const GEMMA4_TEXT_EMBED_TOKENS_PER_LAYER: &str =
    "model.language_model.embed_tokens_per_layer.weight";
pub const GEMMA4_TEXT_PER_LAYER_MODEL_PROJECTION: &str =
    "model.language_model.per_layer_model_projection.weight";
pub const GEMMA4_TEXT_PER_LAYER_PROJECTION_NORM: &str =
    "model.language_model.per_layer_projection_norm.weight";
pub const GEMMA4_TEXT_FINAL_NORM: &str = "model.language_model.norm.weight";
pub const GEMMA4_TEXT_REQUIRED_HIP_KERNEL_ENV: &str = "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL";
pub const GEMMA4_TEXT_REQUIRED_HIP_BF16_ROW_KERNEL_ENV: &str = "ULLM_REQUIRE_HIP_BF16_ROW_KERNEL";
pub const GEMMA4_TEXT_REQUIRED_HIP_RMSNORM_KERNEL_ENV: &str = "ULLM_REQUIRE_HIP_RMSNORM_KERNEL";
pub const GEMMA4_TEXT_REQUIRED_HIP_ADD_KERNEL_ENV: &str = "ULLM_REQUIRE_HIP_ADD_KERNEL";
pub const GEMMA4_TEXT_REQUIRED_HIP_ROPE_KERNEL_ENV: &str = "ULLM_REQUIRE_HIP_ROPE_KERNEL";
pub const GEMMA4_TEXT_REQUIRED_HIP_PROPORTIONAL_ROPE_KERNEL_ENV: &str =
    "ULLM_REQUIRE_HIP_GEMMA_PROPORTIONAL_ROPE_KERNEL";
pub const GEMMA4_TEXT_REQUIRED_HIP_PAGED_DECODE_ENV: &str =
    "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL";
pub const GEMMA4_TEXT_REQUIRED_HIP_PAGED_KV_WRITE_ENV: &str =
    "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL";

const SAFETENSORS_HEADER_LIMIT_BYTES: u64 = 128 * 1024 * 1024;
const RESIDENT_UPLOAD_CHUNK_BYTES: usize = 16 * 1024 * 1024;
const GEMMA4_DEVICE_KV_BLOCK_SIZE: usize = 1;
const GEMMA4_PREFILL_ACTIVATION_CHUNK_TOKENS: usize = 128;
const R9700_RUNTIME_NAME: &str = "AMD Radeon Graphics";
const R9700_MEMORY_BYTES_MIN: u64 = 30 * 1024 * 1024 * 1024;
const R9700_MEMORY_BYTES_MAX: u64 = 34 * 1024 * 1024 * 1024;
const GEMMA4_VALIDATE_DEVICE_MLP_ENV: &str = "ULLM_GEMMA4_VALIDATE_DEVICE_MLP";
const GEMMA4_VALIDATE_PROPORTIONAL_ROPE_ENV: &str = "ULLM_GEMMA4_VALIDATE_PROPORTIONAL_ROPE";
const GEMMA4_DISABLE_PLE_REGION_ENV: &str = "ULLM_GEMMA4_DISABLE_PLE_REGION";

fn record_elapsed_ns(slot: &mut u64, started: Instant) {
    *slot = slot.saturating_add(elapsed_ns(started));
}

fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gemma4TextDeviceIdentity {
    pub runtime_index: u32,
    pub device_id: i32,
    pub backend: String,
    pub name: String,
    pub gcn_arch_name: String,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub total_global_mem: u64,
}

impl Gemma4TextDeviceIdentity {
    fn from_runtime(runtime_index: u32, info: DeviceInfo) -> Self {
        Self {
            runtime_index,
            device_id: info.device_id,
            backend: info.backend,
            name: info.name,
            gcn_arch_name: info.gcn_arch_name,
            compute_major: info.compute_major,
            compute_minor: info.compute_minor,
            total_global_mem: info.total_global_mem,
        }
    }

    fn validate_r9700(&self) -> Result<(), String> {
        let arch = self.gcn_arch_name.split(':').next().unwrap_or_default();
        if self.backend != "hip"
            || self.name != R9700_RUNTIME_NAME
            || !arch.eq_ignore_ascii_case("gfx1201")
            || self.compute_major != 12
            || self.compute_minor != 0
            || !(R9700_MEMORY_BYTES_MIN..=R9700_MEMORY_BYTES_MAX).contains(&self.total_global_mem)
        {
            return Err(format!(
                "Gemma4TextExecutor requires the canonical R9700/gfx1201 HIP identity, got runtime_index={} backend={} name={} arch={} compute={}.{} memory={}",
                self.runtime_index,
                self.backend,
                self.name,
                self.gcn_arch_name,
                self.compute_major,
                self.compute_minor,
                self.total_global_mem,
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4TextTop1 {
    pub token_id: u32,
    pub logit: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4TextStepTrace {
    pub input_token_ids: Vec<u32>,
    /// Flattened `[tokens, hidden]` F32 scaled word embeddings.
    pub embedding: Vec<f32>,
    /// One flattened `[tokens, hidden]` F32 output per decoder layer.
    pub layer_outputs: Vec<Vec<f32>>,
    /// Flattened `[tokens, hidden]` F32 final RMSNorm output.
    pub final_norm: Vec<f32>,
    /// F32 final-token logits after Gemma4's final soft-cap.
    pub logits_last: Vec<f32>,
    pub top1: Gemma4TextTop1,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Gemma4ResidentMlpValidation {
    pub calls: u64,
    pub elements: u64,
    pub max_abs: f32,
    pub max_rel: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Gemma4ResidentRopeValidation {
    pub calls: u64,
    pub elements: u64,
    pub max_abs: f32,
    pub max_rel: f32,
    /// Aggregate error within the channels that are actually rotated.
    pub rotated_max_abs: f32,
    pub rotated_max_rel: f32,
    /// Aggregate error in the two unrotated channel spans.  This must be zero:
    /// the proportional RoPE kernel copies these values, including the two
    /// channels immediately adjacent to each active partial-pair span.
    pub unrotated_max_abs: f32,
    pub unrotated_max_rel: f32,
}

/// Byte-accounted resident-text execution plan for the inspected checkpoint.
///
/// `text_weight_bytes` is intentionally distinct from the resident checkpoint
/// byte count: this executor executes the causal text decoder only, while the
/// source file also contains vision/audio branches.  All branches are still
/// uploaded in resident mode; the PLE tensors are included in the text number
/// and exposed separately because `embed_tokens_per_layer.weight` is material
/// on E2B.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gemma4ResidentMemoryPlan {
    pub source_model_file_bytes: u64,
    pub source_payload_bytes: u64,
    /// All BF16 tensors from the one-file source checkpoint.  Resident mode
    /// uploads this complete set, including non-text branches, so the actual
    /// allocation is directly comparable to `model.safetensors`.
    pub resident_checkpoint_weight_bytes: u64,
    pub resident_checkpoint_tensor_count: usize,
    /// Text-decoder subset that this executor actually evaluates.
    pub text_weight_bytes: u64,
    pub text_tensor_count: usize,
    pub ple_weight_bytes: u64,
    pub unexecuted_multimodal_weight_bytes: u64,
    pub local_kv_source_layers: usize,
    pub full_kv_source_layers: usize,
    pub local_kv_capacity_tokens: usize,
    pub local_kv_bytes: u64,
    pub full_kv_bytes_per_token: u64,
    pub page_table_bytes_per_full_token: u64,
    pub device_transient_bytes: u64,
    pub max_context_tokens: usize,
}

/// Observed state of one device-resident K/V source cache.  Only the
/// non-sharing source layers own storage; shared layers are represented by
/// `Gemma4SharedKvSource` instead of carrying duplicate K/V allocations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gemma4ResidentKvLayerState {
    pub layer_index: usize,
    pub layer_kind: String,
    pub capacity_tokens: usize,
    pub cache_len: usize,
    pub absolute_len: usize,
    pub allocated_bytes: u64,
}

/// Explicit source selection for a KV-sharing decoder layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gemma4SharedKvSource {
    pub layer_index: usize,
    pub layer_kind: String,
    pub source_layer_index: usize,
}

/// Snapshot used by the resident validation driver.  It makes the window
/// length and the layer-15-and-later sharing topology auditable without
/// exposing mutable device buffers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gemma4ResidentKvCacheSnapshot {
    pub source_layers: Vec<Gemma4ResidentKvLayerState>,
    pub shared_layer_sources: Vec<Gemma4SharedKvSource>,
}

/// Logical lower-bound traffic attributed to resident execution operations.
///
/// These counters do not claim that every byte crossed HBM: cache reuse,
/// activation copies, allocator traffic, and page-table traffic are outside
/// the denominator.  They let benchmark evidence use the same explicit
/// accounting style as the existing SQ8_0 efficiency report without treating
/// a profiler range as throughput.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Gemma4ResidentLogicalBytes {
    pub bf16_weight_bytes: u64,
    pub kv_read_bytes: u64,
    pub kv_write_bytes: u64,
    pub matvec_calls: u64,
    pub bf16_row_reads: u64,
    pub attention_calls: u64,
}

/// Host-side timing of the resident Gemma4 executor's primitive boundaries.
///
/// All values are monotonically accumulated nanoseconds. `primitive_ns` is
/// inclusive: it contains the categories below plus the small residual spent
/// in primitive validation and bookkeeping. `executor_other_ns` is the time in
/// a token forward pass not spent in one of those primitive calls. This is a
/// diagnostic counter, not a throughput clock.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Gemma4ResidentPrimitiveHostProfile {
    pub primitive_ns: u64,
    pub input_encode_ns: u64,
    pub output_allocation_ns: u64,
    pub h2d_submit_ns: u64,
    pub kernel_submit_ns: u64,
    pub d2h_submit_ns: u64,
    pub stream_synchronize_ns: u64,
    pub output_decode_validate_ns: u64,
    pub kv_table_host_ns: u64,
    pub calls: u64,
}

impl Gemma4ResidentPrimitiveHostProfile {
    fn record(
        &mut self,
        primitive_ns: u64,
        input_encode_ns: u64,
        output_allocation_ns: u64,
        h2d_submit_ns: u64,
        kernel_submit_ns: u64,
        d2h_submit_ns: u64,
        stream_synchronize_ns: u64,
        output_decode_validate_ns: u64,
        kv_table_host_ns: u64,
    ) {
        self.primitive_ns = self.primitive_ns.saturating_add(primitive_ns);
        self.input_encode_ns = self.input_encode_ns.saturating_add(input_encode_ns);
        self.output_allocation_ns = self
            .output_allocation_ns
            .saturating_add(output_allocation_ns);
        self.h2d_submit_ns = self.h2d_submit_ns.saturating_add(h2d_submit_ns);
        self.kernel_submit_ns = self.kernel_submit_ns.saturating_add(kernel_submit_ns);
        self.d2h_submit_ns = self.d2h_submit_ns.saturating_add(d2h_submit_ns);
        self.stream_synchronize_ns = self
            .stream_synchronize_ns
            .saturating_add(stream_synchronize_ns);
        self.output_decode_validate_ns = self
            .output_decode_validate_ns
            .saturating_add(output_decode_validate_ns);
        self.kv_table_host_ns = self.kv_table_host_ns.saturating_add(kv_table_host_ns);
        self.calls = self.calls.saturating_add(1);
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Gemma4ResidentHostProfile {
    pub token_forward_ns: u64,
    pub primitive_ns: u64,
    pub executor_other_ns: u64,
    pub input_encode_ns: u64,
    pub output_allocation_ns: u64,
    pub buffer_ensure_ns: u64,
    pub buffer_allocate_ns: u64,
    pub h2d_submit_ns: u64,
    pub kernel_submit_ns: u64,
    pub d2h_submit_ns: u64,
    pub stream_synchronize_ns: u64,
    pub output_decode_validate_ns: u64,
    pub kv_table_host_ns: u64,
    pub matvec_calls: u64,
    pub row_calls: u64,
    pub attention_calls: u64,
    pub kv_write_calls: u64,
    /// Per-operation decomposition of the aggregate fields above.  This is
    /// deliberately separate so old consumers of the aggregate contract stay
    /// valid while the port order is measured from actual D2H/sync costs.
    pub matvec: Gemma4ResidentPrimitiveHostProfile,
    pub bf16_row: Gemma4ResidentPrimitiveHostProfile,
    pub attention: Gemma4ResidentPrimitiveHostProfile,
    pub kv_write: Gemma4ResidentPrimitiveHostProfile,
}

/// Diagnostic-only result from re-enabling the physical K/V projections of
/// layers that Gemma4 HF normally shares.  It is deliberately not a serving
/// option: its sole purpose is to make the source-cache selection auditable.
#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4UnsharedKvReference {
    pub generated_token_ids: Vec<u32>,
    pub top1_logits: Vec<f32>,
}

impl Gemma4ResidentLogicalBytes {
    pub fn total_bytes(self) -> Result<u64, String> {
        self.bf16_weight_bytes
            .checked_add(self.kv_read_bytes)
            .and_then(|bytes| bytes.checked_add(self.kv_write_bytes))
            .ok_or_else(|| "Gemma4 resident logical byte count overflows u64".to_string())
    }
}

impl Gemma4ResidentMemoryPlan {
    fn from_checkpoint(
        reader: &SafeTensorReader,
        descriptor: &ResidentModelDescriptor,
    ) -> Result<Self, String> {
        descriptor.require_gemma4_resident_bf16()?;
        let mut resident_checkpoint_weight_bytes = 0_u64;
        let mut resident_checkpoint_tensor_count = 0_usize;
        let mut text_weight_bytes = 0_u64;
        let mut text_tensor_count = 0_usize;
        let mut ple_weight_bytes = 0_u64;
        for (name, entry) in &reader.entries {
            if entry.dtype != "BF16" {
                return Err(format!(
                    "Gemma4 resident checkpoint requires BF16 source tensors, found {} for {name}",
                    entry.dtype
                ));
            }
            let bytes = entry
                .data_end
                .checked_sub(entry.data_start)
                .ok_or_else(|| format!("Gemma4 tensor {name} has invalid offsets"))?;
            resident_checkpoint_weight_bytes = resident_checkpoint_weight_bytes
                .checked_add(bytes)
                .ok_or_else(|| "Gemma4 resident checkpoint byte count overflows u64".to_string())?;
            resident_checkpoint_tensor_count = resident_checkpoint_tensor_count
                .checked_add(1)
                .ok_or_else(|| {
                    "Gemma4 resident checkpoint tensor count overflows usize".to_string()
                })?;
            if !name.starts_with(GEMMA4_TEXT_WEIGHT_PREFIX) {
                continue;
            }
            text_weight_bytes = text_weight_bytes
                .checked_add(bytes)
                .ok_or_else(|| "Gemma4 text weight byte count overflows u64".to_string())?;
            text_tensor_count = text_tensor_count
                .checked_add(1)
                .ok_or_else(|| "Gemma4 text tensor count overflows usize".to_string())?;
            let is_ple = name == GEMMA4_TEXT_EMBED_TOKENS_PER_LAYER
                || name == GEMMA4_TEXT_PER_LAYER_MODEL_PROJECTION
                || name == GEMMA4_TEXT_PER_LAYER_PROJECTION_NORM
                || name.contains(".per_layer_input_gate.")
                || name.contains(".per_layer_projection.")
                || name.contains(".post_per_layer_input_norm.");
            if is_ple {
                ple_weight_bytes = ple_weight_bytes
                    .checked_add(bytes)
                    .ok_or_else(|| "Gemma4 PLE weight byte count overflows u64".to_string())?;
            }
        }
        if resident_checkpoint_weight_bytes != reader.payload_bytes() {
            return Err(format!(
                "Gemma4 resident checkpoint BF16 payload accounting mismatch: tensors total {resident_checkpoint_weight_bytes}, safetensors payload {}",
                reader.payload_bytes()
            ));
        }
        let mut local_kv_source_layers = 0_usize;
        let mut full_kv_source_layers = 0_usize;
        let mut local_kv_capacity_tokens = 0_usize;
        let mut local_kv_bytes = 0_u64;
        let mut full_kv_bytes_per_token = 0_u64;
        let mut page_table_bytes_per_full_token = 0_u64;
        let f32_bytes = u64::try_from(std::mem::size_of::<f32>())
            .map_err(|_| "Gemma4 F32 byte width exceeds u64".to_string())?;
        let u32_bytes = u64::try_from(std::mem::size_of::<u32>())
            .map_err(|_| "Gemma4 U32 byte width exceeds u64".to_string())?;
        for layer in descriptor
            .layers
            .iter()
            .filter(|layer| matches!(layer.attention.kv_cache, ResidentKvCacheMode::Own))
        {
            let attention = &layer.attention;
            let kv_width = attention
                .kv_heads
                .checked_mul(attention.head_dim)
                .ok_or_else(|| "Gemma4 KV width overflows usize".to_string())?;
            let kv_width =
                u64::try_from(kv_width).map_err(|_| "Gemma4 KV width exceeds u64".to_string())?;
            match attention.kind {
                DecoderLayerKind::SlidingAttention => {
                    local_kv_source_layers =
                        local_kv_source_layers.checked_add(1).ok_or_else(|| {
                            "Gemma4 local KV source layer count overflows usize".to_string()
                        })?;
                    let capacity = attention.sliding_window.ok_or_else(|| {
                        format!(
                            "Gemma4 descriptor local layer {} has no sliding-window capacity",
                            layer.layer_index
                        )
                    })?;
                    local_kv_capacity_tokens = local_kv_capacity_tokens.max(capacity);
                    let capacity = u64::try_from(capacity)
                        .map_err(|_| "Gemma4 local KV capacity exceeds u64".to_string())?;
                    let elements = capacity
                        .checked_mul(kv_width)
                        .ok_or_else(|| "Gemma4 local KV element count overflows u64".to_string())?;
                    let cache_bytes = elements
                        .checked_mul(2)
                        .and_then(|elements| elements.checked_mul(f32_bytes))
                        .ok_or_else(|| "Gemma4 local KV byte count overflows u64".to_string())?;
                    // Local source caches have both identity write and ordered
                    // read page tables in addition to F32 K and V.
                    let table_bytes = capacity
                        .checked_mul(u32_bytes)
                        .and_then(|bytes| bytes.checked_mul(2))
                        .ok_or_else(|| {
                            "Gemma4 local KV table byte count overflows u64".to_string()
                        })?;
                    local_kv_bytes = local_kv_bytes
                        .checked_add(cache_bytes)
                        .and_then(|bytes| bytes.checked_add(table_bytes))
                        .ok_or_else(|| "Gemma4 local KV byte count overflows u64".to_string())?;
                }
                DecoderLayerKind::FullAttention => {
                    full_kv_source_layers =
                        full_kv_source_layers.checked_add(1).ok_or_else(|| {
                            "Gemma4 full KV source layer count overflows usize".to_string()
                        })?;
                    let bytes_per_token = kv_width
                        .checked_mul(2)
                        .and_then(|elements| elements.checked_mul(f32_bytes))
                        .ok_or_else(|| {
                            "Gemma4 full KV bytes-per-token overflows u64".to_string()
                        })?;
                    full_kv_bytes_per_token = full_kv_bytes_per_token
                        .checked_add(bytes_per_token)
                        .ok_or_else(|| {
                            "Gemma4 full KV bytes-per-token overflows u64".to_string()
                        })?;
                    page_table_bytes_per_full_token = page_table_bytes_per_full_token
                        .checked_add(u32_bytes)
                        .ok_or_else(|| {
                            "Gemma4 full KV page-table bytes-per-token overflows u64".to_string()
                        })?;
                }
                DecoderLayerKind::LinearAttention => {
                    return Err("Gemma4 resident plan does not support linear attention".into());
                }
            }
        }
        let max_projection_input = descriptor
            .layers
            .iter()
            .map(|layer| match &layer.mlp {
                ResidentMlpDescriptor::Dense {
                    intermediate_size, ..
                } => Ok(*intermediate_size),
                ResidentMlpDescriptor::MoE { .. } => Err(format!(
                    "Gemma4 resident plan cannot allocate MoE layer {}",
                    layer.layer_index
                )),
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .ok_or_else(|| "Gemma4 descriptor has no MLP width".to_string())?;
        let max_projection_output = descriptor.decoder.vocab_size;
        let ple = descriptor
            .layers
            .first()
            .and_then(|layer| layer.per_layer_embedding.as_ref())
            .ok_or_else(|| "Gemma4 descriptor has no PLE contract".to_string())?;
        let packed_ple = descriptor
            .layers
            .len()
            .checked_mul(ple.input_size)
            .ok_or_else(|| "Gemma4 packed PLE width overflows usize".to_string())?;
        let max_attention_width = descriptor
            .layers
            .iter()
            .map(|layer| {
                layer
                    .attention
                    .q_heads
                    .checked_mul(layer.attention.head_dim)
                    .ok_or_else(|| "Gemma4 maximum attention width overflows usize".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .ok_or_else(|| "Gemma4 descriptor has no attention width".to_string())?;
        let max_kv_width = descriptor
            .layers
            .iter()
            .map(|layer| {
                layer
                    .attention
                    .kv_heads
                    .checked_mul(layer.attention.head_dim)
                    .ok_or_else(|| "Gemma4 maximum KV width overflows usize".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .ok_or_else(|| "Gemma4 descriptor has no KV width".to_string())?;
        let device_transient_bytes = [
            max_projection_input,
            max_projection_output,
            packed_ple,
            max_kv_width,
            max_kv_width,
            max_attention_width,
            max_attention_width,
        ]
        .into_iter()
        .try_fold(0_u64, |total, elements| {
            u64::try_from(elements)
                .ok()
                .and_then(|elements| elements.checked_mul(f32_bytes))
                .and_then(|bytes| total.checked_add(bytes))
                .ok_or_else(|| "Gemma4 device transient byte count overflows u64".to_string())
        })?;
        Ok(Self {
            source_model_file_bytes: reader.file_bytes(),
            source_payload_bytes: reader.payload_bytes(),
            resident_checkpoint_weight_bytes,
            resident_checkpoint_tensor_count,
            text_weight_bytes,
            text_tensor_count,
            ple_weight_bytes,
            unexecuted_multimodal_weight_bytes: reader
                .payload_bytes()
                .checked_sub(text_weight_bytes)
                .ok_or_else(|| "Gemma4 source payload is smaller than text tensors".to_string())?,
            local_kv_source_layers,
            full_kv_source_layers,
            local_kv_capacity_tokens,
            local_kv_bytes,
            full_kv_bytes_per_token,
            page_table_bytes_per_full_token,
            device_transient_bytes,
            max_context_tokens: descriptor.decoder.max_position_embeddings,
        })
    }

    pub fn estimated_kv_bytes(&self, context_tokens: usize) -> Result<u64, String> {
        let context = u64::try_from(context_tokens)
            .map_err(|_| "Gemma4 context token count exceeds u64".to_string())?;
        self.local_kv_bytes
            .checked_add(
                self.full_kv_bytes_per_token
                    .checked_mul(context)
                    .ok_or_else(|| "Gemma4 full KV byte count overflows u64".to_string())?,
            )
            .and_then(|bytes| {
                self.page_table_bytes_per_full_token
                    .checked_mul(context)
                    .and_then(|tables| bytes.checked_add(tables))
            })
            .ok_or_else(|| "Gemma4 total KV byte count overflows u64".to_string())
    }

    pub fn estimated_device_bytes(&self, context_tokens: usize) -> Result<u64, String> {
        self.resident_checkpoint_weight_bytes
            .checked_add(self.estimated_kv_bytes(context_tokens)?)
            .and_then(|bytes| bytes.checked_add(self.device_transient_bytes))
            .ok_or_else(|| "Gemma4 resident device byte estimate overflows u64".to_string())
    }
}

#[derive(Debug, Clone)]
struct SafeTensorEntry {
    dtype: String,
    shape: Vec<usize>,
    data_start: u64,
    data_end: u64,
}

struct SafeTensorReader {
    path: PathBuf,
    file: File,
    file_bytes: u64,
    payload_start: u64,
    payload_bytes: u64,
    entries: BTreeMap<String, SafeTensorEntry>,
}

impl SafeTensorReader {
    fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let canonical_path = std::fs::canonicalize(path).map_err(|error| {
            format!(
                "failed to canonicalize safetensors {}: {error}",
                path.display()
            )
        })?;
        let mut file = File::open(&canonical_path).map_err(|error| {
            format!(
                "failed to open safetensors {}: {error}",
                canonical_path.display()
            )
        })?;
        let file_len = file
            .metadata()
            .map_err(|error| {
                format!(
                    "failed to stat safetensors {}: {error}",
                    canonical_path.display()
                )
            })?
            .len();
        if file_len < 8 {
            return Err(format!(
                "safetensors {} is shorter than its header length field",
                canonical_path.display()
            ));
        }
        let mut header_length_bytes = [0_u8; 8];
        file.read_exact(&mut header_length_bytes).map_err(|error| {
            format!(
                "failed to read safetensors header length {}: {error}",
                canonical_path.display()
            )
        })?;
        let header_length = u64::from_le_bytes(header_length_bytes);
        if header_length == 0 || header_length > SAFETENSORS_HEADER_LIMIT_BYTES {
            return Err(format!(
                "safetensors {} has invalid header length {header_length}",
                canonical_path.display()
            ));
        }
        let payload_start = 8_u64
            .checked_add(header_length)
            .ok_or_else(|| "safetensors header offset overflows u64".to_string())?;
        if payload_start > file_len {
            return Err(format!(
                "safetensors {} header extends beyond the file",
                canonical_path.display()
            ));
        }
        let header_size = usize::try_from(header_length)
            .map_err(|_| "safetensors header length exceeds host usize".to_string())?;
        let mut header_bytes = vec![0_u8; header_size];
        file.read_exact(&mut header_bytes).map_err(|error| {
            format!(
                "failed to read safetensors header {}: {error}",
                canonical_path.display()
            )
        })?;
        let header: Value = serde_json::from_slice(&header_bytes).map_err(|error| {
            format!(
                "failed to parse safetensors header {}: {error}",
                canonical_path.display()
            )
        })?;
        let object = header.as_object().ok_or_else(|| {
            format!(
                "safetensors header {} must be a JSON object",
                canonical_path.display()
            )
        })?;
        let payload_bytes = file_len
            .checked_sub(payload_start)
            .ok_or_else(|| "safetensors payload length underflow".to_string())?;
        let mut entries = BTreeMap::new();
        let mut intervals = Vec::new();
        for (name, value) in object {
            if name == "__metadata__" {
                continue;
            }
            let descriptor = value.as_object().ok_or_else(|| {
                format!(
                    "safetensors tensor {name} in {} must be an object",
                    canonical_path.display()
                )
            })?;
            let dtype = descriptor
                .get("dtype")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| format!("safetensors tensor {name} has no dtype"))?
                .to_string();
            let shape = descriptor
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("safetensors tensor {name} has no shape"))?
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let dimension = value.as_u64().ok_or_else(|| {
                        format!("safetensors tensor {name}.shape[{index}] is not an integer")
                    })?;
                    let dimension = usize::try_from(dimension).map_err(|_| {
                        format!("safetensors tensor {name}.shape[{index}] exceeds host usize")
                    })?;
                    if dimension == 0 {
                        return Err(format!("safetensors tensor {name}.shape[{index}] is zero"));
                    }
                    Ok(dimension)
                })
                .collect::<Result<Vec<_>, String>>()?;
            let offsets = descriptor
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("safetensors tensor {name} has no data_offsets"))?;
            if offsets.len() != 2 {
                return Err(format!(
                    "safetensors tensor {name} data_offsets must have two entries"
                ));
            }
            let data_start = offsets[0].as_u64().ok_or_else(|| {
                format!("safetensors tensor {name} data_offsets[0] is not an integer")
            })?;
            let data_end = offsets[1].as_u64().ok_or_else(|| {
                format!("safetensors tensor {name} data_offsets[1] is not an integer")
            })?;
            if data_start > data_end || data_end > payload_bytes {
                return Err(format!(
                    "safetensors tensor {name} offsets [{data_start},{data_end}) exceed payload {payload_bytes}"
                ));
            }
            intervals.push((data_start, data_end, name.clone()));
            if entries
                .insert(
                    name.clone(),
                    SafeTensorEntry {
                        dtype,
                        shape,
                        data_start,
                        data_end,
                    },
                )
                .is_some()
            {
                return Err(format!("duplicate safetensors tensor name {name}"));
            }
        }
        if entries.is_empty() {
            return Err(format!(
                "safetensors {} contains no tensors",
                canonical_path.display()
            ));
        }
        intervals.sort_unstable_by_key(|(start, _, _)| *start);
        let mut previous_end = 0_u64;
        for (start, end, name) in intervals {
            if start < previous_end {
                return Err(format!(
                    "safetensors tensor {name} overlaps a preceding payload"
                ));
            }
            previous_end = end;
        }
        Ok(Self {
            path: canonical_path,
            file,
            file_bytes: file_len,
            payload_start,
            payload_bytes,
            entries,
        })
    }

    fn file_bytes(&self) -> u64 {
        self.file_bytes
    }

    fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    fn matching_bf16_entries(
        &self,
        prefix: &str,
    ) -> Result<Vec<(String, SafeTensorEntry)>, String> {
        let mut result = Vec::new();
        for (name, entry) in &self.entries {
            if !name.starts_with(prefix) {
                continue;
            }
            if entry.dtype != "BF16" {
                return Err(format!(
                    "Gemma4 resident tensor {name} must be BF16, got {}",
                    entry.dtype
                ));
            }
            let elements = entry.shape.iter().try_fold(1_usize, |product, dimension| {
                product.checked_mul(*dimension).ok_or_else(|| {
                    format!("Gemma4 resident tensor {name} element count overflows usize")
                })
            })?;
            let expected_bytes = elements
                .checked_mul(std::mem::size_of::<u16>())
                .ok_or_else(|| {
                    format!("Gemma4 resident tensor {name} byte count overflows usize")
                })?;
            let actual_bytes = usize::try_from(
                entry
                    .data_end
                    .checked_sub(entry.data_start)
                    .ok_or_else(|| format!("Gemma4 resident tensor {name} has invalid offsets"))?,
            )
            .map_err(|_| format!("Gemma4 resident tensor {name} exceeds host usize"))?;
            if expected_bytes != actual_bytes {
                return Err(format!(
                    "Gemma4 resident tensor {name} payload length mismatch: expected {expected_bytes}, got {actual_bytes}"
                ));
            }
            result.push((name.clone(), entry.clone()));
        }
        if result.is_empty() {
            return Err(format!(
                "Gemma4 safetensors {} contains no BF16 tensors under {prefix}",
                self.path.display()
            ));
        }
        Ok(result)
    }

    fn read_bf16_chunk(
        &mut self,
        name: &str,
        entry: &SafeTensorEntry,
        offset: usize,
        bytes: usize,
    ) -> Result<Vec<u8>, String> {
        let total_bytes = usize::try_from(
            entry
                .data_end
                .checked_sub(entry.data_start)
                .ok_or_else(|| format!("Gemma4 tensor {name} has invalid offsets"))?,
        )
        .map_err(|_| format!("Gemma4 tensor {name} payload exceeds host usize"))?;
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| format!("Gemma4 tensor {name} chunk range overflows"))?;
        if end > total_bytes {
            return Err(format!(
                "Gemma4 tensor {name} chunk [{offset},{end}) exceeds {total_bytes} bytes"
            ));
        }
        let absolute_offset = self
            .payload_start
            .checked_add(entry.data_start)
            .and_then(|base| base.checked_add(u64::try_from(offset).ok()?))
            .ok_or_else(|| format!("Gemma4 tensor {name} chunk file offset overflows"))?;
        self.file
            .seek(SeekFrom::Start(absolute_offset))
            .map_err(|error| format!("failed to seek Gemma4 tensor {name} chunk: {error}"))?;
        let mut chunk = vec![0_u8; bytes];
        self.file
            .read_exact(&mut chunk)
            .map_err(|error| format!("failed to read Gemma4 tensor {name} chunk: {error}"))?;
        Ok(chunk)
    }

    fn require_bf16_shape(
        &self,
        name: &str,
        expected_shape: &[usize],
    ) -> Result<&SafeTensorEntry, String> {
        let entry = self.entries.get(name).ok_or_else(|| {
            format!(
                "Gemma4 safetensors {} is missing required tensor {name}",
                self.path.display()
            )
        })?;
        if entry.dtype != "BF16" {
            return Err(format!(
                "Gemma4 tensor {name} must be BF16, got {}",
                entry.dtype
            ));
        }
        if entry.shape != expected_shape {
            return Err(format!(
                "Gemma4 tensor {name} shape mismatch: expected {expected_shape:?}, got {:?}",
                entry.shape
            ));
        }
        let elements = expected_shape
            .iter()
            .try_fold(1_usize, |product, dimension| {
                product
                    .checked_mul(*dimension)
                    .ok_or_else(|| format!("Gemma4 tensor {name} element count overflows usize"))
            })?;
        let expected_bytes = elements
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| format!("Gemma4 tensor {name} byte count overflows usize"))?;
        let actual_bytes = entry
            .data_end
            .checked_sub(entry.data_start)
            .ok_or_else(|| format!("Gemma4 tensor {name} has invalid offsets"))?;
        if actual_bytes
            != u64::try_from(expected_bytes)
                .map_err(|_| format!("Gemma4 tensor {name} expected byte count exceeds u64"))?
        {
            return Err(format!(
                "Gemma4 tensor {name} payload length mismatch: expected {expected_bytes}, got {actual_bytes}"
            ));
        }
        Ok(entry)
    }

    fn read_bf16(&mut self, name: &str, expected_shape: &[usize]) -> Result<Vec<u8>, String> {
        let entry = self.require_bf16_shape(name, expected_shape)?.clone();
        let byte_count = usize::try_from(entry.data_end - entry.data_start)
            .map_err(|_| format!("Gemma4 tensor {name} payload exceeds host usize"))?;
        let absolute_offset = self
            .payload_start
            .checked_add(entry.data_start)
            .ok_or_else(|| format!("Gemma4 tensor {name} file offset overflows"))?;
        self.file
            .seek(SeekFrom::Start(absolute_offset))
            .map_err(|error| format!("failed to seek Gemma4 tensor {name}: {error}"))?;
        let mut bytes = vec![0_u8; byte_count];
        self.file
            .read_exact(&mut bytes)
            .map_err(|error| format!("failed to read Gemma4 tensor {name}: {error}"))?;
        Ok(bytes)
    }

    fn read_bf16_row(
        &mut self,
        name: &str,
        rows: usize,
        columns: usize,
        row_index: usize,
    ) -> Result<Vec<f32>, String> {
        if row_index >= rows {
            return Err(format!(
                "Gemma4 tensor {name} row {row_index} is outside 0..{rows}"
            ));
        }
        let entry = self.require_bf16_shape(name, &[rows, columns])?.clone();
        let row_bytes = columns
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| format!("Gemma4 tensor {name} row byte count overflows"))?;
        let row_offset = row_index
            .checked_mul(row_bytes)
            .ok_or_else(|| format!("Gemma4 tensor {name} row offset overflows"))?;
        let absolute_offset = self
            .payload_start
            .checked_add(entry.data_start)
            .and_then(|offset| offset.checked_add(u64::try_from(row_offset).ok()?))
            .ok_or_else(|| format!("Gemma4 tensor {name} row file offset overflows"))?;
        self.file
            .seek(SeekFrom::Start(absolute_offset))
            .map_err(|error| format!("failed to seek Gemma4 tensor {name} row: {error}"))?;
        let mut bytes = vec![0_u8; row_bytes];
        self.file
            .read_exact(&mut bytes)
            .map_err(|error| format!("failed to read Gemma4 tensor {name} row: {error}"))?;
        bf16_bytes_to_f32(&bytes, name)
    }

    fn read_bf16_vector(&mut self, name: &str, elements: usize) -> Result<Vec<f32>, String> {
        let bytes = self.read_bf16(name, &[elements])?;
        bf16_bytes_to_f32(&bytes, name)
    }
}

#[derive(Debug)]
struct ResidentBf16Tensor {
    shape: Vec<usize>,
    bytes: usize,
    buffer: RuntimeBuffer,
}

#[derive(Debug)]
struct ResidentGemma4Weights {
    tensors: BTreeMap<String, ResidentBf16Tensor>,
    bytes: u64,
}

impl ResidentGemma4Weights {
    fn upload_checkpoint(
        reader: &mut SafeTensorReader,
        runtime: &mut Bf16MatvecRuntime,
    ) -> Result<Self, String> {
        // An empty prefix is intentional: model.safetensors is a single BF16
        // checkpoint, and resident mode owns the complete source payload even
        // though this executor presently executes only the text branch.
        let entries = reader.matching_bf16_entries("")?;
        let mut tensors = BTreeMap::new();
        let mut total_bytes = 0_u64;
        for (name, entry) in entries {
            let bytes = usize::try_from(
                entry
                    .data_end
                    .checked_sub(entry.data_start)
                    .ok_or_else(|| format!("Gemma4 resident tensor {name} has invalid offsets"))?,
            )
            .map_err(|_| format!("Gemma4 resident tensor {name} exceeds host usize"))?;
            let buffer = runtime.upload_bf16_entry(reader, &name, &entry, bytes)?;
            total_bytes =
                total_bytes
                    .checked_add(u64::try_from(bytes).map_err(|_| {
                        format!("Gemma4 resident tensor {name} byte count exceeds u64")
                    })?)
                    .ok_or_else(|| {
                        "Gemma4 resident checkpoint weight byte count overflows u64".to_string()
                    })?;
            if tensors
                .insert(
                    name.clone(),
                    ResidentBf16Tensor {
                        shape: entry.shape,
                        bytes,
                        buffer,
                    },
                )
                .is_some()
            {
                return Err(format!("duplicate resident Gemma4 tensor {name}"));
            }
        }
        Ok(Self {
            tensors,
            bytes: total_bytes,
        })
    }

    fn tensor(&self, name: &str, expected_shape: &[usize]) -> Result<&ResidentBf16Tensor, String> {
        let tensor = self.tensors.get(name).ok_or_else(|| {
            format!("Gemma4 resident text weights are missing required tensor {name}")
        })?;
        if tensor.shape != expected_shape {
            return Err(format!(
                "Gemma4 resident tensor {name} shape mismatch: expected {expected_shape:?}, got {:?}",
                tensor.shape
            ));
        }
        let expected_bytes = expected_shape
            .iter()
            .try_fold(1_usize, |product, dimension| {
                product.checked_mul(*dimension).ok_or_else(|| {
                    format!("Gemma4 resident tensor {name} element count overflows usize")
                })
            })?
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| format!("Gemma4 resident tensor {name} byte count overflows usize"))?;
        if tensor.bytes != expected_bytes {
            return Err(format!(
                "Gemma4 resident tensor {name} byte length mismatch: expected {expected_bytes}, got {}",
                tensor.bytes
            ));
        }
        Ok(tensor)
    }

    fn bytes(&self) -> u64 {
        self.bytes
    }

    fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

enum Gemma4WeightStorage {
    Streamed(SafeTensorReader),
    Resident(ResidentGemma4Weights),
}

struct Bf16MatvecRuntime {
    context: RuntimeContext,
    stream: RuntimeStream,
    device: Gemma4TextDeviceIdentity,
    resident_input: Option<RuntimeBuffer>,
    resident_output: Option<RuntimeBuffer>,
    resident_row_output: Option<RuntimeBuffer>,
    mlp_gate: Option<RuntimeBuffer>,
    mlp_up: Option<RuntimeBuffer>,
    mlp_activated: Option<RuntimeBuffer>,
    mlp_pre_norm_weight: Option<RuntimeBuffer>,
    mlp_pre_normed: Option<RuntimeBuffer>,
    mlp_post_norm_weight: Option<RuntimeBuffer>,
    mlp_post_normed: Option<RuntimeBuffer>,
    ple_input: Option<RuntimeBuffer>,
    kv_key_input: Option<RuntimeBuffer>,
    kv_value_input: Option<RuntimeBuffer>,
    attention_input_norm_weight: Option<RuntimeBuffer>,
    attention_input_normed: Option<RuntimeBuffer>,
    attention_q_norm_weight: Option<RuntimeBuffer>,
    attention_k_norm_weight: Option<RuntimeBuffer>,
    attention_value_norm_weight: Option<RuntimeBuffer>,
    attention_post_norm_weight: Option<RuntimeBuffer>,
    attention_query: Option<RuntimeBuffer>,
    attention_query_normed: Option<RuntimeBuffer>,
    attention_key: Option<RuntimeBuffer>,
    attention_key_normed: Option<RuntimeBuffer>,
    attention_value: Option<RuntimeBuffer>,
    attention_value_normed: Option<RuntimeBuffer>,
    attention_output: Option<RuntimeBuffer>,
    resident_logical_bytes: Gemma4ResidentLogicalBytes,
    resident_host_profile: Gemma4ResidentHostProfile,
    mlp_validation: Gemma4ResidentMlpValidation,
    rope_validation: Gemma4ResidentRopeValidation,
}

impl Bf16MatvecRuntime {
    fn create() -> Result<Self, String> {
        for required in [
            GEMMA4_TEXT_REQUIRED_HIP_KERNEL_ENV,
            GEMMA4_TEXT_REQUIRED_HIP_BF16_ROW_KERNEL_ENV,
            GEMMA4_TEXT_REQUIRED_HIP_RMSNORM_KERNEL_ENV,
            GEMMA4_TEXT_REQUIRED_HIP_ADD_KERNEL_ENV,
            GEMMA4_TEXT_REQUIRED_HIP_ROPE_KERNEL_ENV,
            GEMMA4_TEXT_REQUIRED_HIP_PROPORTIONAL_ROPE_KERNEL_ENV,
        ] {
            if env::var(required).ok().as_deref() != Some("1") {
                return Err(format!(
                    "Gemma4TextExecutor requires {required}=1 to forbid host-staging fallback"
                ));
            }
        }
        let count = device_count()?;
        let mut candidates = Vec::new();
        for runtime_index in 0..count {
            let info = device_info(runtime_index).map_err(|error| {
                format!("failed to inspect runtime device {runtime_index}: {error}")
            })?;
            let identity = Gemma4TextDeviceIdentity::from_runtime(runtime_index, info);
            if identity.backend == "hip" {
                if identity.validate_r9700().is_ok() {
                    candidates.push(identity);
                } else if identity
                    .gcn_arch_name
                    .split(':')
                    .next()
                    .unwrap_or_default()
                    .eq_ignore_ascii_case("gfx1030")
                {
                    // Explicitly never select the passively cooled V620.
                    continue;
                }
            }
        }
        if candidates.len() != 1 {
            return Err(format!(
                "Gemma4TextExecutor requires exactly one selectable R9700/gfx1201, found {}",
                candidates.len()
            ));
        }
        let device = candidates.pop().expect("checked candidate count");
        let mut context = RuntimeContext::create(device.runtime_index)?;
        let context_device =
            Gemma4TextDeviceIdentity::from_runtime(device.runtime_index, context.device_info()?);
        context_device.validate_r9700()?;
        let stream = context.create_stream()?;
        Ok(Self {
            context,
            stream,
            device: context_device,
            resident_input: None,
            resident_output: None,
            resident_row_output: None,
            mlp_gate: None,
            mlp_up: None,
            mlp_activated: None,
            mlp_pre_norm_weight: None,
            mlp_pre_normed: None,
            mlp_post_norm_weight: None,
            mlp_post_normed: None,
            ple_input: None,
            kv_key_input: None,
            kv_value_input: None,
            attention_input_norm_weight: None,
            attention_input_normed: None,
            attention_q_norm_weight: None,
            attention_k_norm_weight: None,
            attention_value_norm_weight: None,
            attention_post_norm_weight: None,
            attention_query: None,
            attention_query_normed: None,
            attention_key: None,
            attention_key_normed: None,
            attention_value: None,
            attention_value_normed: None,
            attention_output: None,
            resident_logical_bytes: Gemma4ResidentLogicalBytes::default(),
            resident_host_profile: Gemma4ResidentHostProfile::default(),
            mlp_validation: Gemma4ResidentMlpValidation::default(),
            rope_validation: Gemma4ResidentRopeValidation::default(),
        })
    }

    fn device(&self) -> &Gemma4TextDeviceIdentity {
        &self.device
    }

    fn resident_transient_bytes(&self) -> Result<u64, String> {
        [
            &self.resident_input,
            &self.resident_output,
            &self.resident_row_output,
            &self.mlp_gate,
            &self.mlp_up,
            &self.mlp_activated,
            &self.mlp_pre_norm_weight,
            &self.mlp_pre_normed,
            &self.mlp_post_norm_weight,
            &self.mlp_post_normed,
            &self.ple_input,
            &self.kv_key_input,
            &self.kv_value_input,
            &self.attention_input_norm_weight,
            &self.attention_input_normed,
            &self.attention_q_norm_weight,
            &self.attention_k_norm_weight,
            &self.attention_value_norm_weight,
            &self.attention_post_norm_weight,
            &self.attention_query,
            &self.attention_query_normed,
            &self.attention_key,
            &self.attention_key_normed,
            &self.attention_value,
            &self.attention_value_normed,
            &self.attention_output,
        ]
        .into_iter()
        .try_fold(0_u64, |total, buffer| {
            let bytes = match buffer {
                Some(buffer) => u64::try_from(buffer.size()?)
                    .map_err(|_| "Gemma4 transient buffer size exceeds u64".to_string())?,
                None => 0,
            };
            total
                .checked_add(bytes)
                .ok_or_else(|| "Gemma4 transient allocation byte count overflows u64".to_string())
        })
    }

    fn resident_logical_bytes(&self) -> Gemma4ResidentLogicalBytes {
        self.resident_logical_bytes
    }

    fn reset_resident_logical_bytes(&mut self) {
        self.resident_logical_bytes = Gemma4ResidentLogicalBytes::default();
    }

    fn resident_host_profile(&self) -> Gemma4ResidentHostProfile {
        self.resident_host_profile
    }

    fn reset_resident_host_profile(&mut self) {
        self.resident_host_profile = Gemma4ResidentHostProfile::default();
    }

    fn mlp_validation(&self) -> Gemma4ResidentMlpValidation {
        self.mlp_validation
    }

    fn rope_validation(&self) -> Gemma4ResidentRopeValidation {
        self.rope_validation
    }

    fn account_resident_weight_read(
        &mut self,
        bytes: usize,
        is_matvec: bool,
    ) -> Result<(), String> {
        let bytes = u64::try_from(bytes)
            .map_err(|_| "Gemma4 resident weight read byte count exceeds u64".to_string())?;
        self.resident_logical_bytes.bf16_weight_bytes = self
            .resident_logical_bytes
            .bf16_weight_bytes
            .checked_add(bytes)
            .ok_or_else(|| "Gemma4 resident BF16 logical byte count overflows u64".to_string())?;
        if is_matvec {
            self.resident_logical_bytes.matvec_calls = self
                .resident_logical_bytes
                .matvec_calls
                .checked_add(1)
                .ok_or_else(|| "Gemma4 resident matvec count overflows u64".to_string())?;
        } else {
            self.resident_logical_bytes.bf16_row_reads = self
                .resident_logical_bytes
                .bf16_row_reads
                .checked_add(1)
                .ok_or_else(|| "Gemma4 resident BF16 row-read count overflows u64".to_string())?;
        }
        Ok(())
    }

    fn account_device_kv(
        &mut self,
        read_bytes: usize,
        write_bytes: usize,
        attention_call: bool,
    ) -> Result<(), String> {
        let read_bytes = u64::try_from(read_bytes)
            .map_err(|_| "Gemma4 resident KV read byte count exceeds u64".to_string())?;
        let write_bytes = u64::try_from(write_bytes)
            .map_err(|_| "Gemma4 resident KV write byte count exceeds u64".to_string())?;
        self.resident_logical_bytes.kv_read_bytes = self
            .resident_logical_bytes
            .kv_read_bytes
            .checked_add(read_bytes)
            .ok_or_else(|| {
                "Gemma4 resident KV read logical byte count overflows u64".to_string()
            })?;
        self.resident_logical_bytes.kv_write_bytes = self
            .resident_logical_bytes
            .kv_write_bytes
            .checked_add(write_bytes)
            .ok_or_else(|| {
                "Gemma4 resident KV write logical byte count overflows u64".to_string()
            })?;
        if attention_call {
            self.resident_logical_bytes.attention_calls = self
                .resident_logical_bytes
                .attention_calls
                .checked_add(1)
                .ok_or_else(|| "Gemma4 resident attention call count overflows u64".to_string())?;
        }
        Ok(())
    }

    fn ensure_buffer(
        context: &mut RuntimeContext,
        slot: &mut Option<RuntimeBuffer>,
        required_bytes: usize,
        label: &str,
        host_profile: &mut Gemma4ResidentHostProfile,
    ) -> Result<(), String> {
        let started = Instant::now();
        if required_bytes == 0 {
            return Err(format!("Gemma4 {label} buffer requires nonzero bytes"));
        }
        let needs_replacement = match slot.as_ref() {
            Some(buffer) => buffer.size()? < required_bytes,
            None => true,
        };
        if needs_replacement {
            let allocation_started = Instant::now();
            *slot =
                Some(context.alloc_buffer(required_bytes).map_err(|error| {
                    format!("failed to allocate Gemma4 {label} buffer: {error}")
                })?);
            record_elapsed_ns(&mut host_profile.buffer_allocate_ns, allocation_started);
        }
        record_elapsed_ns(&mut host_profile.buffer_ensure_ns, started);
        Ok(())
    }

    fn alloc_buffer(&mut self, bytes: usize, label: &str) -> Result<RuntimeBuffer, String> {
        self.context
            .alloc_buffer(bytes)
            .map_err(|error| format!("failed to allocate Gemma4 {label} buffer: {error}"))
    }

    fn upload_bf16_entry(
        &mut self,
        reader: &mut SafeTensorReader,
        name: &str,
        entry: &SafeTensorEntry,
        bytes: usize,
    ) -> Result<RuntimeBuffer, String> {
        if bytes == 0 {
            return Err(format!("Gemma4 resident tensor {name} has zero bytes"));
        }
        let mut destination = self.alloc_buffer(bytes, &format!("resident tensor {name}"))?;
        let mut offset = 0_usize;
        while offset < bytes {
            let remaining = bytes
                .checked_sub(offset)
                .ok_or_else(|| format!("Gemma4 resident tensor {name} upload underflow"))?;
            let chunk_bytes = remaining.min(RESIDENT_UPLOAD_CHUNK_BYTES);
            let chunk = reader.read_bf16_chunk(name, entry, offset, chunk_bytes)?;
            destination.copy_from_host(offset, &chunk, Some(&mut self.stream))?;
            // The host chunk is intentionally released before the next file read.
            // Synchronizing here makes that lifetime explicit for async HIP copies.
            self.stream.synchronize()?;
            offset = offset
                .checked_add(chunk_bytes)
                .ok_or_else(|| format!("Gemma4 resident tensor {name} upload offset overflows"))?;
        }
        Ok(destination)
    }

    fn matvec_resident(
        &mut self,
        matrix: &RuntimeBuffer,
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let operation_started = Instant::now();
        if input.len() != columns {
            return Err(format!(
                "resident BF16 matvec input width mismatch: expected {columns}, got {}",
                input.len()
            ));
        }
        let encode_started = Instant::now();
        let input_bytes = encode_f32_to_bytes(input);
        let input_encode_ns = elapsed_ns(encode_started);
        let output_bytes = rows
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "resident BF16 matvec output byte count overflows".to_string())?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.resident_input,
            input_bytes.len(),
            "resident matvec input",
            &mut self.resident_host_profile,
        )?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.resident_output,
            output_bytes,
            "resident matvec output",
            &mut self.resident_host_profile,
        )?;
        let (input_slot, output_slot) = (&mut self.resident_input, &mut self.resident_output);
        let input_buffer = input_slot
            .as_mut()
            .expect("resident input buffer was allocated");
        let output_buffer = output_slot
            .as_mut()
            .expect("resident output buffer was allocated");
        let h2d_started = Instant::now();
        input_buffer.copy_from_host(0, &input_bytes, Some(&mut self.stream))?;
        let h2d_submit_ns = elapsed_ns(h2d_started);
        let kernel_started = Instant::now();
        matvec_bf16_f32(
            matrix,
            input_buffer,
            rows,
            columns,
            output_buffer,
            Some(&mut self.stream),
        )?;
        let kernel_submit_ns = elapsed_ns(kernel_started);
        let output_allocation_started = Instant::now();
        let mut host_output = vec![0_u8; output_bytes];
        let output_allocation_ns = elapsed_ns(output_allocation_started);
        let d2h_started = Instant::now();
        output_buffer.copy_to_host(0, &mut host_output, Some(&mut self.stream))?;
        let d2h_submit_ns = elapsed_ns(d2h_started);
        let synchronize_started = Instant::now();
        self.stream.synchronize()?;
        let stream_synchronize_ns = elapsed_ns(synchronize_started);
        let decode_started = Instant::now();
        let output = decode_f32_le_values(&host_output);
        if output.len() != rows || output.iter().any(|value| !value.is_finite()) {
            return Err(
                "Gemma4 resident BF16 matvec returned non-finite or malformed F32 output".into(),
            );
        }
        let output_decode_validate_ns = elapsed_ns(decode_started);
        let matrix_bytes = rows
            .checked_mul(columns)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| {
                "Gemma4 resident BF16 matvec logical byte count overflows".to_string()
            })?;
        self.account_resident_weight_read(matrix_bytes, true)?;
        let primitive_ns = elapsed_ns(operation_started);
        let profile = &mut self.resident_host_profile;
        profile.input_encode_ns = profile.input_encode_ns.saturating_add(input_encode_ns);
        profile.output_allocation_ns = profile
            .output_allocation_ns
            .saturating_add(output_allocation_ns);
        profile.h2d_submit_ns = profile.h2d_submit_ns.saturating_add(h2d_submit_ns);
        profile.kernel_submit_ns = profile.kernel_submit_ns.saturating_add(kernel_submit_ns);
        profile.d2h_submit_ns = profile.d2h_submit_ns.saturating_add(d2h_submit_ns);
        profile.stream_synchronize_ns = profile
            .stream_synchronize_ns
            .saturating_add(stream_synchronize_ns);
        profile.output_decode_validate_ns = profile
            .output_decode_validate_ns
            .saturating_add(output_decode_validate_ns);
        profile.primitive_ns = profile.primitive_ns.saturating_add(primitive_ns);
        profile.matvec_calls = profile.matvec_calls.saturating_add(1);
        profile.matvec.record(
            primitive_ns,
            input_encode_ns,
            output_allocation_ns,
            h2d_submit_ns,
            kernel_submit_ns,
            d2h_submit_ns,
            stream_synchronize_ns,
            output_decode_validate_ns,
            0,
        );
        Ok(output)
    }

    /// Gemma-only row-major `[M, columns]` BF16-weight projection.  Decode
    /// never calls this method: it retains its established M=1 matvec route.
    fn gemma_matmul_resident(
        &mut self,
        matrix: &RuntimeBuffer,
        rows: usize,
        columns: usize,
        batch_count: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let expected_input = batch_count
            .checked_mul(columns)
            .ok_or_else(|| "Gemma BF16 matmul input element count overflows".to_string())?;
        if input.len() != expected_input {
            return Err(format!(
                "Gemma BF16 matmul input width mismatch: expected {expected_input}, got {}",
                input.len()
            ));
        }
        let output_elements = batch_count
            .checked_mul(rows)
            .ok_or_else(|| "Gemma BF16 matmul output element count overflows".to_string())?;
        let input_bytes = encode_f32_to_bytes(input);
        let output_bytes = output_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Gemma BF16 matmul output byte count overflows".to_string())?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.resident_input,
            input_bytes.len(),
            "Gemma batched matmul input",
            &mut self.resident_host_profile,
        )?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.resident_output,
            output_bytes,
            "Gemma batched matmul output",
            &mut self.resident_host_profile,
        )?;
        self.resident_input
            .as_mut()
            .expect("Gemma batched matmul input allocated")
            .copy_from_host(0, &input_bytes, Some(&mut self.stream))?;
        gemma_bf16_matmul_f32(
            matrix,
            self.resident_input
                .as_ref()
                .expect("Gemma batched matmul input allocated"),
            rows,
            columns,
            batch_count,
            self.resident_output
                .as_mut()
                .expect("Gemma batched matmul output allocated"),
            Some(&mut self.stream),
        )?;
        let mut host_output = vec![0_u8; output_bytes];
        self.resident_output
            .as_mut()
            .expect("Gemma batched matmul output allocated")
            .copy_to_host(0, &mut host_output, Some(&mut self.stream))?;
        self.stream.synchronize()?;
        let output = decode_f32_le_values(&host_output);
        if output.len() != output_elements || output.iter().any(|value| !value.is_finite()) {
            return Err(
                "Gemma BF16 batched matmul returned non-finite or malformed F32 output".into(),
            );
        }
        let matrix_bytes = rows
            .checked_mul(columns)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| "Gemma BF16 matmul logical byte count overflows".to_string())?;
        self.account_resident_weight_read(
            matrix_bytes.checked_mul(batch_count).ok_or_else(|| {
                "Gemma BF16 matmul aggregate logical byte count overflows".to_string()
            })?,
            true,
        )?;
        Ok(output)
    }

    fn bf16_row_resident(
        &mut self,
        matrix: &RuntimeBuffer,
        rows: usize,
        columns: usize,
        row_index: usize,
    ) -> Result<Vec<f32>, String> {
        let operation_started = Instant::now();
        let output_bytes = columns
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "resident BF16 row output byte count overflows".to_string())?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.resident_row_output,
            output_bytes,
            "resident BF16 row output",
            &mut self.resident_host_profile,
        )?;
        let output_buffer = self
            .resident_row_output
            .as_mut()
            .expect("resident row output buffer was allocated");
        let kernel_started = Instant::now();
        bf16_row_f32(
            matrix,
            rows,
            columns,
            row_index,
            output_buffer,
            Some(&mut self.stream),
        )?;
        let kernel_submit_ns = elapsed_ns(kernel_started);
        let output_allocation_started = Instant::now();
        let mut host_output = vec![0_u8; output_bytes];
        let output_allocation_ns = elapsed_ns(output_allocation_started);
        let d2h_started = Instant::now();
        output_buffer.copy_to_host(0, &mut host_output, Some(&mut self.stream))?;
        let d2h_submit_ns = elapsed_ns(d2h_started);
        let synchronize_started = Instant::now();
        self.stream.synchronize()?;
        let stream_synchronize_ns = elapsed_ns(synchronize_started);
        let decode_started = Instant::now();
        let output = decode_f32_le_values(&host_output);
        if output.len() != columns || output.iter().any(|value| !value.is_finite()) {
            return Err(
                "Gemma4 resident BF16 row returned non-finite or malformed F32 output".into(),
            );
        }
        let output_decode_validate_ns = elapsed_ns(decode_started);
        let row_bytes = columns
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| "Gemma4 resident BF16 row logical byte count overflows".to_string())?;
        self.account_resident_weight_read(row_bytes, false)?;
        let primitive_ns = elapsed_ns(operation_started);
        let profile = &mut self.resident_host_profile;
        profile.output_allocation_ns = profile
            .output_allocation_ns
            .saturating_add(output_allocation_ns);
        profile.kernel_submit_ns = profile.kernel_submit_ns.saturating_add(kernel_submit_ns);
        profile.d2h_submit_ns = profile.d2h_submit_ns.saturating_add(d2h_submit_ns);
        profile.stream_synchronize_ns = profile
            .stream_synchronize_ns
            .saturating_add(stream_synchronize_ns);
        profile.output_decode_validate_ns = profile
            .output_decode_validate_ns
            .saturating_add(output_decode_validate_ns);
        profile.primitive_ns = profile.primitive_ns.saturating_add(primitive_ns);
        profile.row_calls = profile.row_calls.saturating_add(1);
        profile.bf16_row.record(
            primitive_ns,
            0,
            output_allocation_ns,
            0,
            kernel_submit_ns,
            d2h_submit_ns,
            stream_synchronize_ns,
            output_decode_validate_ns,
            0,
        );
        Ok(output)
    }

    /// Executes the Gemma proportional/partial RoPE on a captured F32 head
    /// activation, then compares it with the unchanged host implementation.
    /// This is deliberately validation-only until the surrounding attention
    /// chain can retain the activation on the device.
    fn validate_gemma_proportional_rope(
        &mut self,
        values: &[f32],
        heads: usize,
        head_dim: usize,
        rope: &ResidentRopeDescriptor,
        position: usize,
    ) -> Result<(), String> {
        if env::var(GEMMA4_VALIDATE_PROPORTIONAL_ROPE_ENV)
            .ok()
            .as_deref()
            != Some("1")
        {
            return Ok(());
        }
        let ResidentRopeKind::Proportional = rope.kind else {
            return Ok(());
        };
        let partial = rope.partial_rotary_factor.unwrap_or(1.0);
        let rotary_dim = ((partial * head_dim as f32) / 2.0).floor() as usize * 2;
        if rotary_dim == 0 {
            return Ok(());
        }
        let expected_elements = heads
            .checked_mul(head_dim)
            .ok_or_else(|| "Gemma proportional RoPE validation shape overflows".to_string())?;
        if values.len() != expected_elements {
            return Err(
                "Gemma proportional RoPE validation input shape disagrees with heads".into(),
            );
        }
        let bytes = expected_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Gemma proportional RoPE validation byte count overflows".to_string())?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.attention_query,
            bytes,
            "Gemma proportional RoPE validation input",
            &mut self.resident_host_profile,
        )?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.attention_output,
            bytes,
            "Gemma proportional RoPE validation output",
            &mut self.resident_host_profile,
        )?;
        let input_bytes = encode_f32_to_bytes(values);
        self.attention_query
            .as_mut()
            .expect("Gemma proportional RoPE validation input allocated")
            .copy_from_host(0, &input_bytes, Some(&mut self.stream))?;
        gemma_proportional_rope_f32(
            self.attention_query
                .as_ref()
                .expect("Gemma proportional RoPE validation input allocated"),
            1,
            heads,
            head_dim,
            rotary_dim,
            position,
            rope.theta,
            self.attention_output
                .as_mut()
                .expect("Gemma proportional RoPE validation output allocated"),
            Some(&mut self.stream),
        )?;
        let mut output_bytes = vec![0_u8; bytes];
        self.attention_output
            .as_mut()
            .expect("Gemma proportional RoPE validation output allocated")
            .copy_to_host(0, &mut output_bytes, Some(&mut self.stream))?;
        self.stream.synchronize()?;
        let actual = decode_f32_le_values(&output_bytes);
        let mut expected = values.to_vec();
        apply_gemma4_rope_in_place(&mut expected, heads, head_dim, rope, position)?;
        let active_pairs = rotary_dim / 2;
        let half = head_dim / 2;
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let abs = (actual - expected).abs();
            let rel = abs / expected.abs().max(f32::MIN_POSITIVE);
            self.rope_validation.max_abs = self.rope_validation.max_abs.max(abs);
            self.rope_validation.max_rel = self.rope_validation.max_rel.max(rel);
            let channel = index % head_dim;
            if channel < active_pairs || (half..half + active_pairs).contains(&channel) {
                self.rope_validation.rotated_max_abs =
                    self.rope_validation.rotated_max_abs.max(abs);
                self.rope_validation.rotated_max_rel =
                    self.rope_validation.rotated_max_rel.max(rel);
            } else {
                self.rope_validation.unrotated_max_abs =
                    self.rope_validation.unrotated_max_abs.max(abs);
                self.rope_validation.unrotated_max_rel =
                    self.rope_validation.unrotated_max_rel.max(rel);
            }
        }
        self.rope_validation.calls = self.rope_validation.calls.saturating_add(1);
        self.rope_validation.elements = self.rope_validation.elements.saturating_add(
            u64::try_from(expected_elements).map_err(|_| {
                "Gemma proportional RoPE validation elements exceed u64".to_string()
            })?,
        );
        Ok(())
    }

    /// Dense Gemma4 MLP region: host input -> gate/up projections -> GELUTanh
    /// product -> down projection -> host output.  Every intermediate remains
    /// in these persistent device workspaces.
    fn dense_mlp_resident(
        &mut self,
        gate_matrix: &RuntimeBuffer,
        up_matrix: &RuntimeBuffer,
        down_matrix: &RuntimeBuffer,
        hidden: usize,
        intermediate: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if input.len() != hidden {
            return Err(format!(
                "Gemma4 dense MLP input width mismatch: expected {hidden}, got {}",
                input.len()
            ));
        }
        let input_bytes = encode_f32_to_bytes(input);
        let intermediate_bytes = intermediate
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Gemma4 dense MLP intermediate bytes overflow".to_string())?;
        let output_bytes = hidden
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Gemma4 dense MLP output bytes overflow".to_string())?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.resident_input,
            input_bytes.len(),
            "dense MLP input",
            &mut self.resident_host_profile,
        )?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.mlp_gate,
            intermediate_bytes,
            "dense MLP gate",
            &mut self.resident_host_profile,
        )?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.mlp_up,
            intermediate_bytes,
            "dense MLP up",
            &mut self.resident_host_profile,
        )?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.mlp_activated,
            intermediate_bytes,
            "dense MLP activated",
            &mut self.resident_host_profile,
        )?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.resident_output,
            output_bytes,
            "dense MLP output",
            &mut self.resident_host_profile,
        )?;
        let input_buffer = self
            .resident_input
            .as_mut()
            .expect("dense MLP input allocated");
        input_buffer.copy_from_host(0, &input_bytes, Some(&mut self.stream))?;
        matvec_bf16_f32(
            gate_matrix,
            input_buffer,
            intermediate,
            hidden,
            self.mlp_gate.as_mut().expect("dense MLP gate allocated"),
            Some(&mut self.stream),
        )?;
        matvec_bf16_f32(
            up_matrix,
            input_buffer,
            intermediate,
            hidden,
            self.mlp_up.as_mut().expect("dense MLP up allocated"),
            Some(&mut self.stream),
        )?;
        gelu_tanh_mul_f32(
            self.mlp_gate.as_ref().expect("dense MLP gate allocated"),
            self.mlp_up.as_ref().expect("dense MLP up allocated"),
            intermediate,
            self.mlp_activated
                .as_mut()
                .expect("dense MLP activated allocated"),
            Some(&mut self.stream),
        )?;
        matvec_bf16_f32(
            down_matrix,
            self.mlp_activated
                .as_ref()
                .expect("dense MLP activated allocated"),
            hidden,
            intermediate,
            self.resident_output
                .as_mut()
                .expect("dense MLP output allocated"),
            Some(&mut self.stream),
        )?;
        let mut host_output = vec![0_u8; output_bytes];
        self.resident_output
            .as_mut()
            .expect("dense MLP output allocated")
            .copy_to_host(0, &mut host_output, Some(&mut self.stream))?;
        self.stream.synchronize()?;
        let output = decode_f32_le_values(&host_output);
        if output.len() != hidden || output.iter().any(|value| !value.is_finite()) {
            return Err(
                "Gemma4 dense MLP resident region returned non-finite or malformed F32 output"
                    .into(),
            );
        }
        if env::var(GEMMA4_VALIDATE_DEVICE_MLP_ENV).ok().as_deref() == Some("1") {
            let gate = self.matvec_resident(gate_matrix, intermediate, hidden, input)?;
            let up = self.matvec_resident(up_matrix, intermediate, hidden, input)?;
            let mut activated = gelu_pytorch_tanh(&gate)?;
            multiply_in_place(&mut activated, &up, "Gemma4 MLP validation product")?;
            let reference = self.matvec_resident(down_matrix, hidden, intermediate, &activated)?;
            for (actual, expected) in output.iter().zip(reference.iter()) {
                let abs = (actual - expected).abs();
                let rel = abs / expected.abs().max(f32::MIN_POSITIVE);
                self.mlp_validation.max_abs = self.mlp_validation.max_abs.max(abs);
                self.mlp_validation.max_rel = self.mlp_validation.max_rel.max(rel);
            }
            self.mlp_validation.calls = self.mlp_validation.calls.saturating_add(1);
            self.mlp_validation.elements = self.mlp_validation.elements.saturating_add(
                u64::try_from(output.len())
                    .map_err(|_| "Gemma4 MLP validation elements exceed u64".to_string())?,
            );
        }
        let matrix_bytes = |rows: usize, cols: usize| {
            rows.checked_mul(cols)
                .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
                .ok_or_else(|| "Gemma4 dense MLP logical byte count overflows".to_string())
        };
        self.account_resident_weight_read(matrix_bytes(intermediate, hidden)?, true)?;
        self.account_resident_weight_read(matrix_bytes(intermediate, hidden)?, true)?;
        self.account_resident_weight_read(matrix_bytes(hidden, intermediate)?, true)?;
        Ok(output)
    }

    /// Extends the dense MLP region across both adjacent host boundaries:
    ///
    /// ```text
    /// host attention residual -> direct-BF16 pre-FF RMSNorm -> dense MLP
    ///     -> direct-BF16 post-FF RMSNorm -> residual add -> host MLP residual
    /// ```
    ///
    /// The norm weights remain in their checkpoint BF16 buffers.  `bf16_row_f32`
    /// converts each direct Gemma gamma into a device workspace; in particular,
    /// this deliberately does not apply Qwen's `weight + 1` convention.
    fn dense_mlp_norm_residual_resident(
        &mut self,
        gate_matrix: &RuntimeBuffer,
        up_matrix: &RuntimeBuffer,
        down_matrix: &RuntimeBuffer,
        pre_feedforward_weight: &RuntimeBuffer,
        post_feedforward_weight: &RuntimeBuffer,
        hidden: usize,
        intermediate: usize,
        epsilon: f32,
        attention_residual: &[f32],
    ) -> Result<Vec<f32>, String> {
        if attention_residual.len() != hidden {
            return Err(format!(
                "Gemma4 dense MLP residual width mismatch: expected {hidden}, got {}",
                attention_residual.len()
            ));
        }
        let input_bytes = encode_f32_to_bytes(attention_residual);
        let hidden_bytes = hidden
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Gemma4 dense MLP hidden bytes overflow".to_string())?;
        let intermediate_bytes = intermediate
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Gemma4 dense MLP intermediate bytes overflow".to_string())?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.resident_input,
            input_bytes.len(),
            "dense MLP residual input",
            &mut self.resident_host_profile,
        )?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.mlp_pre_norm_weight,
            hidden_bytes,
            "dense MLP pre-feedforward gamma",
            &mut self.resident_host_profile,
        )?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.mlp_pre_normed,
            hidden_bytes,
            "dense MLP pre-feedforward output",
            &mut self.resident_host_profile,
        )?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.mlp_gate,
            intermediate_bytes,
            "dense MLP gate",
            &mut self.resident_host_profile,
        )?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.mlp_up,
            intermediate_bytes,
            "dense MLP up",
            &mut self.resident_host_profile,
        )?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.mlp_activated,
            intermediate_bytes,
            "dense MLP activated",
            &mut self.resident_host_profile,
        )?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.resident_output,
            hidden_bytes,
            "dense MLP output",
            &mut self.resident_host_profile,
        )?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.mlp_post_norm_weight,
            hidden_bytes,
            "dense MLP post-feedforward gamma",
            &mut self.resident_host_profile,
        )?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.mlp_post_normed,
            hidden_bytes,
            "dense MLP post-feedforward output",
            &mut self.resident_host_profile,
        )?;

        self.resident_input
            .as_mut()
            .expect("dense MLP residual input allocated")
            .copy_from_host(0, &input_bytes, Some(&mut self.stream))?;
        bf16_row_f32(
            pre_feedforward_weight,
            1,
            hidden,
            0,
            self.mlp_pre_norm_weight
                .as_mut()
                .expect("dense MLP pre-feedforward gamma allocated"),
            Some(&mut self.stream),
        )?;
        rmsnorm_f32(
            self.resident_input
                .as_ref()
                .expect("dense MLP residual input allocated"),
            self.mlp_pre_norm_weight
                .as_ref()
                .expect("dense MLP pre-feedforward gamma allocated"),
            hidden,
            epsilon,
            self.mlp_pre_normed
                .as_mut()
                .expect("dense MLP pre-feedforward output allocated"),
            Some(&mut self.stream),
        )?;
        matvec_bf16_f32(
            gate_matrix,
            self.mlp_pre_normed
                .as_ref()
                .expect("dense MLP pre-feedforward output allocated"),
            intermediate,
            hidden,
            self.mlp_gate.as_mut().expect("dense MLP gate allocated"),
            Some(&mut self.stream),
        )?;
        matvec_bf16_f32(
            up_matrix,
            self.mlp_pre_normed
                .as_ref()
                .expect("dense MLP pre-feedforward output allocated"),
            intermediate,
            hidden,
            self.mlp_up.as_mut().expect("dense MLP up allocated"),
            Some(&mut self.stream),
        )?;
        gelu_tanh_mul_f32(
            self.mlp_gate.as_ref().expect("dense MLP gate allocated"),
            self.mlp_up.as_ref().expect("dense MLP up allocated"),
            intermediate,
            self.mlp_activated
                .as_mut()
                .expect("dense MLP activated allocated"),
            Some(&mut self.stream),
        )?;
        matvec_bf16_f32(
            down_matrix,
            self.mlp_activated
                .as_ref()
                .expect("dense MLP activated allocated"),
            hidden,
            intermediate,
            self.resident_output
                .as_mut()
                .expect("dense MLP output allocated"),
            Some(&mut self.stream),
        )?;
        bf16_row_f32(
            post_feedforward_weight,
            1,
            hidden,
            0,
            self.mlp_post_norm_weight
                .as_mut()
                .expect("dense MLP post-feedforward gamma allocated"),
            Some(&mut self.stream),
        )?;
        rmsnorm_f32(
            self.resident_output
                .as_ref()
                .expect("dense MLP output allocated"),
            self.mlp_post_norm_weight
                .as_ref()
                .expect("dense MLP post-feedforward gamma allocated"),
            hidden,
            epsilon,
            self.mlp_post_normed
                .as_mut()
                .expect("dense MLP post-feedforward output allocated"),
            Some(&mut self.stream),
        )?;
        add_f32(
            self.resident_input
                .as_ref()
                .expect("dense MLP residual input allocated"),
            self.mlp_post_normed
                .as_ref()
                .expect("dense MLP post-feedforward output allocated"),
            hidden,
            self.mlp_pre_normed
                .as_mut()
                .expect("dense MLP pre-feedforward output allocated"),
            Some(&mut self.stream),
        )?;
        let mut host_output = vec![0_u8; hidden_bytes];
        self.mlp_pre_normed
            .as_mut()
            .expect("dense MLP residual output allocated")
            .copy_to_host(0, &mut host_output, Some(&mut self.stream))?;
        self.stream.synchronize()?;
        let output = decode_f32_le_values(&host_output);
        if output.len() != hidden || output.iter().any(|value| !value.is_finite()) {
            return Err(
                "Gemma4 dense MLP norm-residual region returned non-finite or malformed F32 output"
                    .into(),
            );
        }

        if env::var(GEMMA4_VALIDATE_DEVICE_MLP_ENV).ok().as_deref() == Some("1") {
            let pre_weight = self.bf16_row_resident(pre_feedforward_weight, 1, hidden, 0)?;
            let feedforward_input = rms_norm(attention_residual, Some(&pre_weight), epsilon)?;
            let gate =
                self.matvec_resident(gate_matrix, intermediate, hidden, &feedforward_input)?;
            let up = self.matvec_resident(up_matrix, intermediate, hidden, &feedforward_input)?;
            let mut activated = gelu_pytorch_tanh(&gate)?;
            multiply_in_place(
                &mut activated,
                &up,
                "Gemma4 MLP norm-residual validation product",
            )?;
            let mlp = self.matvec_resident(down_matrix, hidden, intermediate, &activated)?;
            let post_weight = self.bf16_row_resident(post_feedforward_weight, 1, hidden, 0)?;
            let post_feedforward = rms_norm(&mlp, Some(&post_weight), epsilon)?;
            let reference = add_vectors(
                attention_residual,
                &post_feedforward,
                "Gemma4 MLP norm-residual validation add",
            )?;
            for (actual, expected) in output.iter().zip(reference.iter()) {
                let abs = (actual - expected).abs();
                let rel = abs / expected.abs().max(f32::MIN_POSITIVE);
                self.mlp_validation.max_abs = self.mlp_validation.max_abs.max(abs);
                self.mlp_validation.max_rel = self.mlp_validation.max_rel.max(rel);
            }
            self.mlp_validation.calls = self.mlp_validation.calls.saturating_add(1);
            self.mlp_validation.elements = self.mlp_validation.elements.saturating_add(
                u64::try_from(output.len())
                    .map_err(|_| "Gemma4 MLP validation elements exceed u64".to_string())?,
            );
        }
        let matrix_bytes = |rows: usize, cols: usize| {
            rows.checked_mul(cols)
                .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
                .ok_or_else(|| "Gemma4 dense MLP logical byte count overflows".to_string())
        };
        self.account_resident_weight_read(matrix_bytes(1, hidden)?, false)?;
        self.account_resident_weight_read(matrix_bytes(intermediate, hidden)?, true)?;
        self.account_resident_weight_read(matrix_bytes(intermediate, hidden)?, true)?;
        self.account_resident_weight_read(matrix_bytes(hidden, intermediate)?, true)?;
        self.account_resident_weight_read(matrix_bytes(1, hidden)?, false)?;
        Ok(output)
    }

    /// One complete per-layer-embedding update.  The PLE gate/projection and
    /// every activation-side operation between them stay on the device.  The
    /// only activation boundary is the completed layer output, retained for
    /// the adjacent attention region until that region is joined in a later
    /// growth step.
    fn ple_norm_residual_resident(
        &mut self,
        gate_matrix: &RuntimeBuffer,
        projection_matrix: &RuntimeBuffer,
        post_weight: &RuntimeBuffer,
        hidden: usize,
        ple_dim: usize,
        epsilon: f32,
        mlp_residual: &[f32],
        per_layer_input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if mlp_residual.len() != hidden || per_layer_input.len() != ple_dim {
            return Err(format!(
                "Gemma4 resident PLE shape mismatch: residual={} expected={hidden}, input={} expected={ple_dim}",
                mlp_residual.len(),
                per_layer_input.len()
            ));
        }
        let f32_bytes = std::mem::size_of::<f32>();
        let hidden_bytes = hidden
            .checked_mul(f32_bytes)
            .ok_or_else(|| "Gemma4 resident PLE hidden bytes overflow".to_string())?;
        let ple_bytes = ple_dim
            .checked_mul(f32_bytes)
            .ok_or_else(|| "Gemma4 resident PLE input bytes overflow".to_string())?;
        for (slot, bytes, label) in [
            (
                &mut self.resident_input,
                hidden_bytes,
                "resident PLE residual",
            ),
            (&mut self.ple_input, ple_bytes, "resident PLE input"),
            (&mut self.mlp_gate, ple_bytes, "resident PLE gate"),
            (
                &mut self.mlp_activated,
                ple_bytes,
                "resident PLE activated gate",
            ),
            (
                &mut self.resident_output,
                hidden_bytes,
                "resident PLE projection",
            ),
            (
                &mut self.mlp_post_norm_weight,
                hidden_bytes,
                "resident PLE post gamma",
            ),
            (
                &mut self.mlp_post_normed,
                hidden_bytes,
                "resident PLE post norm",
            ),
            (
                &mut self.mlp_pre_normed,
                hidden_bytes,
                "resident PLE residual output",
            ),
        ] {
            Self::ensure_buffer(
                &mut self.context,
                slot,
                bytes,
                label,
                &mut self.resident_host_profile,
            )?;
        }
        self.resident_input
            .as_mut()
            .expect("resident PLE residual allocated")
            .copy_from_host(
                0,
                &encode_f32_to_bytes(mlp_residual),
                Some(&mut self.stream),
            )?;
        self.ple_input
            .as_mut()
            .expect("resident PLE input allocated")
            .copy_from_host(
                0,
                &encode_f32_to_bytes(per_layer_input),
                Some(&mut self.stream),
            )?;
        matvec_bf16_f32(
            gate_matrix,
            self.resident_input
                .as_ref()
                .expect("resident PLE residual allocated"),
            ple_dim,
            hidden,
            self.mlp_gate.as_mut().expect("resident PLE gate allocated"),
            Some(&mut self.stream),
        )?;
        gelu_tanh_mul_f32(
            self.mlp_gate.as_ref().expect("resident PLE gate allocated"),
            self.ple_input
                .as_ref()
                .expect("resident PLE input allocated"),
            ple_dim,
            self.mlp_activated
                .as_mut()
                .expect("resident PLE activated gate allocated"),
            Some(&mut self.stream),
        )?;
        matvec_bf16_f32(
            projection_matrix,
            self.mlp_activated
                .as_ref()
                .expect("resident PLE activated gate allocated"),
            hidden,
            ple_dim,
            self.resident_output
                .as_mut()
                .expect("resident PLE projection allocated"),
            Some(&mut self.stream),
        )?;
        bf16_row_f32(
            post_weight,
            1,
            hidden,
            0,
            self.mlp_post_norm_weight
                .as_mut()
                .expect("resident PLE post gamma allocated"),
            Some(&mut self.stream),
        )?;
        rmsnorm_f32(
            self.resident_output
                .as_ref()
                .expect("resident PLE projection allocated"),
            self.mlp_post_norm_weight
                .as_ref()
                .expect("resident PLE post gamma allocated"),
            hidden,
            epsilon,
            self.mlp_post_normed
                .as_mut()
                .expect("resident PLE post norm allocated"),
            Some(&mut self.stream),
        )?;
        add_f32(
            self.resident_input
                .as_ref()
                .expect("resident PLE residual allocated"),
            self.mlp_post_normed
                .as_ref()
                .expect("resident PLE post norm allocated"),
            hidden,
            self.mlp_pre_normed
                .as_mut()
                .expect("resident PLE residual output allocated"),
            Some(&mut self.stream),
        )?;
        let mut bytes = vec![0_u8; hidden_bytes];
        self.mlp_pre_normed
            .as_mut()
            .expect("resident PLE residual output allocated")
            .copy_to_host(0, &mut bytes, Some(&mut self.stream))?;
        self.stream.synchronize()?;
        let output = decode_f32_le_values(&bytes);
        if output.len() != hidden || output.iter().any(|value| !value.is_finite()) {
            return Err("Gemma4 resident PLE returned non-finite or malformed F32 output".into());
        }
        let matrix_bytes = |rows: usize, cols: usize| {
            rows.checked_mul(cols)
                .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
                .ok_or_else(|| "Gemma4 resident PLE logical bytes overflow".to_string())
        };
        self.account_resident_weight_read(matrix_bytes(ple_dim, hidden)?, true)?;
        self.account_resident_weight_read(matrix_bytes(hidden, ple_dim)?, true)?;
        self.account_resident_weight_read(matrix_bytes(1, hidden)?, false)?;
        Ok(output)
    }

    fn matvec(
        &mut self,
        matrix: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if input.len() != columns {
            return Err(format!(
                "BF16 matvec input width mismatch: expected {columns}, got {}",
                input.len()
            ));
        }
        let expected_matrix_bytes = rows
            .checked_mul(columns)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| "BF16 matvec matrix byte count overflows".to_string())?;
        if matrix.len() != expected_matrix_bytes {
            return Err(format!(
                "BF16 matvec matrix byte length mismatch: expected {expected_matrix_bytes}, got {}",
                matrix.len()
            ));
        }
        let input_bytes = encode_f32_to_bytes(input);
        let output_bytes = rows
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "BF16 matvec output byte count overflows".to_string())?;
        let mut matrix_buffer = self.context.alloc_buffer(expected_matrix_bytes)?;
        let mut input_buffer = self.context.alloc_buffer(input_bytes.len())?;
        let mut output_buffer = self.context.alloc_buffer(output_bytes)?;
        matrix_buffer.copy_from_host(0, matrix, Some(&mut self.stream))?;
        input_buffer.copy_from_host(0, &input_bytes, Some(&mut self.stream))?;
        matvec_bf16_f32(
            &matrix_buffer,
            &input_buffer,
            rows,
            columns,
            &mut output_buffer,
            Some(&mut self.stream),
        )?;
        let mut host_output = vec![0_u8; output_bytes];
        output_buffer.copy_to_host(0, &mut host_output, Some(&mut self.stream))?;
        self.stream.synchronize()?;
        let output = decode_f32_le_values(&host_output);
        if output.len() != rows || output.iter().any(|value| !value.is_finite()) {
            return Err("Gemma4 BF16 matvec returned non-finite or malformed F32 output".into());
        }
        Ok(output)
    }
}

#[derive(Debug, Clone, Default)]
struct KvSequence {
    width: usize,
    key: Vec<f32>,
    value: Vec<f32>,
}

impl KvSequence {
    fn append(&mut self, key: &[f32], value: &[f32]) -> Result<(), String> {
        if key.len() != value.len() || key.is_empty() {
            return Err("Gemma4 KV append requires equal nonempty key/value widths".into());
        }
        if self.width == 0 {
            self.width = key.len();
        }
        if self.width != key.len() {
            return Err(format!(
                "Gemma4 KV cache width changed from {} to {}",
                self.width,
                key.len()
            ));
        }
        self.key.extend_from_slice(key);
        self.value.extend_from_slice(value);
        Ok(())
    }

    fn len(&self) -> Result<usize, String> {
        if self.width == 0
            || self.key.len() != self.value.len()
            || !self.key.len().is_multiple_of(self.width)
        {
            return Err("Gemma4 KV cache has invalid storage shape".into());
        }
        Ok(self.key.len() / self.width)
    }

    fn retain_last(&mut self, max_tokens: usize) -> Result<(), String> {
        if self.width == 0 {
            if self.key.is_empty() && self.value.is_empty() {
                // The first local token has no historical rows to evict.
                return Ok(());
            }
            return Err("Gemma4 empty-width KV cache has nonempty storage".into());
        }
        let length = self.len()?;
        if length <= max_tokens {
            return Ok(());
        }
        let drop_tokens = length
            .checked_sub(max_tokens)
            .ok_or_else(|| "Gemma4 KV cache truncation underflow".to_string())?;
        let drop_elements = drop_tokens
            .checked_mul(self.width)
            .ok_or_else(|| "Gemma4 KV cache truncation offset overflows".to_string())?;
        self.key.drain(..drop_elements);
        self.value.drain(..drop_elements);
        Ok(())
    }
}

#[derive(Debug)]
struct Gemma4KvCaches {
    per_layer: Vec<Option<KvSequence>>,
}

impl Gemma4KvCaches {
    fn new(layer_count: usize) -> Self {
        Self {
            per_layer: vec![None; layer_count],
        }
    }
}

/// Device-side F32 K/V storage for one non-sharing source layer.  Full
/// attention grows to the configured context limit; sliding attention is a
/// ring of exactly `sliding_window` entries.  A local cache keeps the current
/// full window until the next append so later shared layers can reuse the
/// source state exactly as HF's `shared_kv_states` branch does.
#[derive(Debug)]
struct Gemma4DeviceKvCache {
    layer_index: usize,
    layer_kind: DecoderLayerKind,
    capacity_tokens: usize,
    cache_len: usize,
    absolute_len: usize,
    kv_heads: usize,
    head_dim: usize,
    key: RuntimeBuffer,
    value: RuntimeBuffer,
    read_table: RuntimeBuffer,
    write_table: Option<RuntimeBuffer>,
}

impl Gemma4DeviceKvCache {
    fn is_sliding(&self) -> bool {
        matches!(self.layer_kind, DecoderLayerKind::SlidingAttention)
    }

    fn width(&self) -> Result<usize, String> {
        self.kv_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| "Gemma4 device KV width overflows usize".to_string())
    }

    fn write_position(&self) -> Result<usize, String> {
        if self.capacity_tokens == 0 {
            return Err("Gemma4 device KV cache has zero capacity".into());
        }
        if self.is_sliding() {
            Ok(self.absolute_len % self.capacity_tokens)
        } else if self.absolute_len >= self.capacity_tokens {
            Err(format!(
                "Gemma4 full KV cache for layer {} exceeds capacity {}",
                self.layer_index, self.capacity_tokens
            ))
        } else {
            Ok(self.absolute_len)
        }
    }

    fn record_append(&mut self, stream: &mut RuntimeStream) -> Result<(), String> {
        self.absolute_len = self
            .absolute_len
            .checked_add(1)
            .ok_or_else(|| "Gemma4 device KV absolute length overflows usize".to_string())?;
        self.cache_len = if self.is_sliding() {
            self.absolute_len.min(self.capacity_tokens)
        } else {
            self.absolute_len
        };
        if self.is_sliding() {
            let start = self
                .absolute_len
                .checked_sub(self.cache_len)
                .ok_or_else(|| "Gemma4 sliding KV start underflows usize".to_string())?;
            let mut table = Vec::with_capacity(self.cache_len);
            for relative in 0..self.cache_len {
                let logical = start.checked_add(relative).ok_or_else(|| {
                    "Gemma4 sliding KV logical position overflows usize".to_string()
                })?;
                table.push(u32::try_from(logical % self.capacity_tokens).map_err(|_| {
                    "Gemma4 sliding KV physical block index exceeds u32".to_string()
                })?);
            }
            let bytes = encode_u32_to_bytes(&table);
            self.read_table.copy_from_host(0, &bytes, Some(stream))?;
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.cache_len = 0;
        self.absolute_len = 0;
    }

    fn allocated_bytes(&self) -> Result<u64, String> {
        let key = u64::try_from(self.key.size()?)
            .map_err(|_| "Gemma4 device KV key allocation exceeds u64".to_string())?;
        let value = u64::try_from(self.value.size()?)
            .map_err(|_| "Gemma4 device KV value allocation exceeds u64".to_string())?;
        let read_table = u64::try_from(self.read_table.size()?)
            .map_err(|_| "Gemma4 device KV read table allocation exceeds u64".to_string())?;
        let write_table = self
            .write_table
            .as_ref()
            .map(|buffer| {
                u64::try_from(buffer.size()?)
                    .map_err(|_| "Gemma4 device KV write table allocation exceeds u64".to_string())
            })
            .transpose()?
            .unwrap_or(0);
        key.checked_add(value)
            .and_then(|bytes| bytes.checked_add(read_table))
            .and_then(|bytes| bytes.checked_add(write_table))
            .ok_or_else(|| "Gemma4 device KV allocation byte count overflows u64".to_string())
    }
}

#[derive(Debug)]
struct Gemma4DeviceKvCaches {
    per_layer: Vec<Option<Gemma4DeviceKvCache>>,
}

impl Gemma4DeviceKvCaches {
    fn new(
        descriptor: &ResidentModelDescriptor,
        runtime: &mut Bf16MatvecRuntime,
    ) -> Result<Self, String> {
        descriptor.require_gemma4_resident_bf16()?;
        for required in [
            GEMMA4_TEXT_REQUIRED_HIP_PAGED_DECODE_ENV,
            GEMMA4_TEXT_REQUIRED_HIP_PAGED_KV_WRITE_ENV,
        ] {
            if env::var(required).ok().as_deref() != Some("1") {
                return Err(format!(
                    "Gemma4 resident KV requires {required}=1 to forbid host-staging fallback"
                ));
            }
        }
        let mut per_layer = (0..descriptor.layers.len())
            .map(|_| None)
            .collect::<Vec<Option<Gemma4DeviceKvCache>>>();
        for layer in descriptor
            .layers
            .iter()
            .filter(|layer| matches!(layer.attention.kv_cache, ResidentKvCacheMode::Own))
        {
            let layer_index = layer.layer_index;
            let attention = &layer.attention;
            let layer_kind = attention.kind;
            let (head_dim, capacity_tokens) = match layer_kind {
                DecoderLayerKind::SlidingAttention => (
                    attention.head_dim,
                    attention.sliding_window.ok_or_else(|| {
                        format!("Gemma4 descriptor layer {layer_index} is missing a sliding window")
                    })?,
                ),
                DecoderLayerKind::FullAttention => (
                    attention.head_dim,
                    descriptor.decoder.max_position_embeddings,
                ),
                DecoderLayerKind::LinearAttention => {
                    return Err("Gemma4 resident KV does not support linear attention".into());
                }
            };
            if capacity_tokens == 0 {
                return Err(format!("Gemma4 layer {layer_index} has zero KV capacity"));
            }
            let kv_heads = attention.kv_heads;
            let width = kv_heads
                .checked_mul(head_dim)
                .ok_or_else(|| "Gemma4 device KV width overflows usize".to_string())?;
            let cache_bytes = capacity_tokens
                .checked_mul(width)
                .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| {
                    "Gemma4 device KV allocation byte count overflows usize".to_string()
                })?;
            let table_bytes = capacity_tokens
                .checked_mul(std::mem::size_of::<u32>())
                .ok_or_else(|| "Gemma4 device KV table byte count overflows usize".to_string())?;
            let mut key =
                runtime.alloc_buffer(cache_bytes, &format!("KV key layer {layer_index}"))?;
            let mut value =
                runtime.alloc_buffer(cache_bytes, &format!("KV value layer {layer_index}"))?;
            let mut read_table =
                runtime.alloc_buffer(table_bytes, &format!("KV read table layer {layer_index}"))?;
            let mut write_table = matches!(layer_kind, DecoderLayerKind::SlidingAttention)
                .then(|| {
                    runtime
                        .alloc_buffer(table_bytes, &format!("KV write table layer {layer_index}"))
                })
                .transpose()?;
            let identity = (0..capacity_tokens)
                .map(|index| {
                    u32::try_from(index)
                        .map_err(|_| "Gemma4 KV table index exceeds u32".to_string())
                })
                .collect::<Result<Vec<_>, String>>()?;
            let identity_bytes = encode_u32_to_bytes(&identity);
            key.zero(0, cache_bytes, Some(&mut runtime.stream))?;
            value.zero(0, cache_bytes, Some(&mut runtime.stream))?;
            read_table.copy_from_host(0, &identity_bytes, Some(&mut runtime.stream))?;
            if let Some(write) = write_table.as_mut() {
                write.copy_from_host(0, &identity_bytes, Some(&mut runtime.stream))?;
            }
            runtime.stream.synchronize()?;
            per_layer[layer_index] = Some(Gemma4DeviceKvCache {
                layer_index,
                layer_kind,
                capacity_tokens,
                cache_len: 0,
                absolute_len: 0,
                kv_heads,
                head_dim,
                key,
                value,
                read_table,
                write_table,
            });
        }
        Ok(Self { per_layer })
    }

    fn reset(&mut self) {
        for cache in self.per_layer.iter_mut().flatten() {
            cache.reset();
        }
    }

    fn cache(&self, layer_index: usize) -> Result<&Gemma4DeviceKvCache, String> {
        self.per_layer
            .get(layer_index)
            .and_then(Option::as_ref)
            .ok_or_else(|| format!("Gemma4 device KV cache has no source layer {layer_index}"))
    }

    fn cache_mut(&mut self, layer_index: usize) -> Result<&mut Gemma4DeviceKvCache, String> {
        self.per_layer
            .get_mut(layer_index)
            .and_then(Option::as_mut)
            .ok_or_else(|| format!("Gemma4 device KV cache has no source layer {layer_index}"))
    }

    fn allocated_bytes(&self) -> Result<u64, String> {
        self.per_layer
            .iter()
            .flatten()
            .try_fold(0_u64, |total, cache| {
                total.checked_add(cache.allocated_bytes()?).ok_or_else(|| {
                    "Gemma4 device KV aggregate allocation overflows u64".to_string()
                })
            })
    }

    fn source_layer_states(&self) -> Result<Vec<Gemma4ResidentKvLayerState>, String> {
        self.per_layer
            .iter()
            .flatten()
            .map(|cache| {
                Ok(Gemma4ResidentKvLayerState {
                    layer_index: cache.layer_index,
                    layer_kind: cache.layer_kind.as_str().to_string(),
                    capacity_tokens: cache.capacity_tokens,
                    cache_len: cache.cache_len,
                    absolute_len: cache.absolute_len,
                    allocated_bytes: cache.allocated_bytes()?,
                })
            })
            .collect()
    }
}

impl Bf16MatvecRuntime {
    /// One complete Gemma attention/residual segment.  The only activation
    /// transfers are the residual entering and leaving this method; all
    /// normalization, projections, proportional RoPE, KV write, attention,
    /// output projection, and post-attention residual remain on the device.
    #[allow(clippy::too_many_arguments)]
    fn attention_norm_residual_resident(
        &mut self,
        residual: &[f32],
        input_weight: &RuntimeBuffer,
        q_matrix: &RuntimeBuffer,
        k_matrix: Option<&RuntimeBuffer>,
        v_matrix: Option<&RuntimeBuffer>,
        o_matrix: &RuntimeBuffer,
        q_norm_weight: &RuntimeBuffer,
        k_norm_weight: Option<&RuntimeBuffer>,
        post_weight: &RuntimeBuffer,
        hidden: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rope: &ResidentRopeDescriptor,
        position: usize,
        epsilon: f32,
        own_cache: Option<&mut Gemma4DeviceKvCache>,
        shared_cache: Option<&Gemma4DeviceKvCache>,
    ) -> Result<Vec<f32>, String> {
        if residual.len() != hidden {
            return Err(format!(
                "Gemma4 resident attention residual width mismatch: expected {hidden}, got {}",
                residual.len()
            ));
        }
        let owns_cache = own_cache.is_some();
        let q_width = q_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "Gemma4 resident Q width overflows".to_string())?;
        let kv_width = kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "Gemma4 resident KV width overflows".to_string())?;
        let rotary_dim = match rope.kind {
            ResidentRopeKind::Proportional => {
                ((rope.partial_rotary_factor.unwrap_or(1.0) * head_dim as f32) / 2.0).floor()
                    as usize
                    * 2
            }
            ResidentRopeKind::Default => rope.rotary_dim.unwrap_or(head_dim),
            ResidentRopeKind::Mrope => {
                return Err("Gemma resident attention does not support mRoPE".into());
            }
        };
        if rotary_dim == 0 {
            return Err("Gemma resident attention RoPE has zero rotary width".into());
        }
        if own_cache.is_some()
            != (k_matrix.is_some() && v_matrix.is_some() && k_norm_weight.is_some())
        {
            return Err(
                "Gemma resident attention K/V projection/cache contract is inconsistent".into(),
            );
        }
        if own_cache.is_none() && shared_cache.is_none() {
            return Err("Gemma resident attention needs an own or shared device KV cache".into());
        }
        let hidden_bytes = hidden
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Gemma resident attention hidden bytes overflow".to_string())?;
        let q_bytes = q_width
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Gemma resident attention Q bytes overflow".to_string())?;
        let kv_bytes = kv_width
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Gemma resident attention KV bytes overflow".to_string())?;
        for (slot, bytes, label) in [
            (
                &mut self.resident_input,
                hidden_bytes,
                "resident attention residual",
            ),
            (
                &mut self.attention_input_norm_weight,
                hidden_bytes,
                "resident attention input gamma",
            ),
            (
                &mut self.attention_input_normed,
                hidden_bytes,
                "resident attention input norm",
            ),
            (
                &mut self.attention_q_norm_weight,
                head_dim * 4,
                "resident attention Q gamma",
            ),
            (
                &mut self.attention_post_norm_weight,
                hidden_bytes,
                "resident attention post gamma",
            ),
            (&mut self.attention_query, q_bytes, "resident attention Q"),
            (
                &mut self.attention_query_normed,
                q_bytes,
                "resident attention normalized Q",
            ),
            (
                &mut self.attention_output,
                q_bytes,
                "resident attention output",
            ),
            (
                &mut self.resident_output,
                hidden_bytes,
                "resident attention O projection",
            ),
            (
                &mut self.mlp_pre_normed,
                hidden_bytes,
                "resident attention residual output",
            ),
        ] {
            Self::ensure_buffer(
                &mut self.context,
                slot,
                bytes,
                label,
                &mut self.resident_host_profile,
            )?;
        }
        if owns_cache {
            for (slot, bytes, label) in [
                (
                    &mut self.attention_k_norm_weight,
                    head_dim * 4,
                    "resident attention K gamma",
                ),
                (
                    &mut self.attention_value_norm_weight,
                    head_dim * 4,
                    "resident attention V gamma",
                ),
                (&mut self.attention_key, kv_bytes, "resident attention K"),
                (
                    &mut self.attention_key_normed,
                    kv_bytes,
                    "resident attention normalized K",
                ),
                (&mut self.attention_value, kv_bytes, "resident attention V"),
                (
                    &mut self.attention_value_normed,
                    kv_bytes,
                    "resident attention normalized V",
                ),
            ] {
                Self::ensure_buffer(
                    &mut self.context,
                    slot,
                    bytes,
                    label,
                    &mut self.resident_host_profile,
                )?;
            }
        }
        self.resident_input
            .as_mut()
            .expect("resident attention residual allocated")
            .copy_from_host(0, &encode_f32_to_bytes(residual), Some(&mut self.stream))?;
        bf16_row_f32(
            input_weight,
            1,
            hidden,
            0,
            self.attention_input_norm_weight.as_mut().unwrap(),
            Some(&mut self.stream),
        )?;
        rmsnorm_f32(
            self.resident_input.as_ref().unwrap(),
            self.attention_input_norm_weight.as_ref().unwrap(),
            hidden,
            epsilon,
            self.attention_input_normed.as_mut().unwrap(),
            Some(&mut self.stream),
        )?;
        matvec_bf16_f32(
            q_matrix,
            self.attention_input_normed.as_ref().unwrap(),
            q_width,
            hidden,
            self.attention_query.as_mut().unwrap(),
            Some(&mut self.stream),
        )?;
        bf16_row_f32(
            q_norm_weight,
            1,
            head_dim,
            0,
            self.attention_q_norm_weight.as_mut().unwrap(),
            Some(&mut self.stream),
        )?;
        segmented_rmsnorm_f32(
            self.attention_query.as_ref().unwrap(),
            self.attention_q_norm_weight.as_ref().unwrap(),
            q_heads,
            head_dim,
            epsilon,
            self.attention_query_normed.as_mut().unwrap(),
            Some(&mut self.stream),
        )?;
        match rope.kind {
            ResidentRopeKind::Proportional => gemma_proportional_rope_f32(
                self.attention_query_normed.as_ref().unwrap(),
                1,
                q_heads,
                head_dim,
                rotary_dim,
                position,
                rope.theta,
                self.attention_query.as_mut().unwrap(),
                Some(&mut self.stream),
            )?,
            ResidentRopeKind::Default => rope_f32(
                self.attention_query_normed.as_ref().unwrap(),
                1,
                q_heads,
                head_dim,
                rotary_dim,
                position,
                rope.theta,
                self.attention_query.as_mut().unwrap(),
                Some(&mut self.stream),
            )?,
            ResidentRopeKind::Mrope => unreachable!(),
        }

        if let Some(cache) = own_cache {
            matvec_bf16_f32(
                k_matrix.unwrap(),
                self.attention_input_normed.as_ref().unwrap(),
                kv_width,
                hidden,
                self.attention_key.as_mut().unwrap(),
                Some(&mut self.stream),
            )?;
            matvec_bf16_f32(
                v_matrix.unwrap(),
                self.attention_input_normed.as_ref().unwrap(),
                kv_width,
                hidden,
                self.attention_value.as_mut().unwrap(),
                Some(&mut self.stream),
            )?;
            bf16_row_f32(
                k_norm_weight.unwrap(),
                1,
                head_dim,
                0,
                self.attention_k_norm_weight.as_mut().unwrap(),
                Some(&mut self.stream),
            )?;
            self.attention_value_norm_weight
                .as_mut()
                .unwrap()
                .copy_from_host(
                    0,
                    &encode_f32_to_bytes(&vec![1.0_f32; head_dim]),
                    Some(&mut self.stream),
                )?;
            segmented_rmsnorm_f32(
                self.attention_key.as_ref().unwrap(),
                self.attention_k_norm_weight.as_ref().unwrap(),
                kv_heads,
                head_dim,
                epsilon,
                self.attention_key_normed.as_mut().unwrap(),
                Some(&mut self.stream),
            )?;
            match rope.kind {
                ResidentRopeKind::Proportional => gemma_proportional_rope_f32(
                    self.attention_key_normed.as_ref().unwrap(),
                    1,
                    kv_heads,
                    head_dim,
                    rotary_dim,
                    position,
                    rope.theta,
                    self.attention_key.as_mut().unwrap(),
                    Some(&mut self.stream),
                )?,
                ResidentRopeKind::Default => rope_f32(
                    self.attention_key_normed.as_ref().unwrap(),
                    1,
                    kv_heads,
                    head_dim,
                    rotary_dim,
                    position,
                    rope.theta,
                    self.attention_key.as_mut().unwrap(),
                    Some(&mut self.stream),
                )?,
                ResidentRopeKind::Mrope => unreachable!(),
            }
            segmented_rmsnorm_f32(
                self.attention_value.as_ref().unwrap(),
                self.attention_value_norm_weight.as_ref().unwrap(),
                kv_heads,
                head_dim,
                epsilon,
                self.attention_value_normed.as_mut().unwrap(),
                Some(&mut self.stream),
            )?;
            let write_position = cache.write_position()?;
            let write_table = cache.write_table.as_ref().unwrap_or(&cache.read_table);
            paged_kv_write_f32(
                self.attention_key.as_ref().unwrap(),
                self.attention_value_normed.as_ref().unwrap(),
                write_table,
                write_position,
                GEMMA4_DEVICE_KV_BLOCK_SIZE,
                cache.capacity_tokens,
                cache.kv_heads,
                cache.head_dim,
                cache.head_dim,
                &mut cache.key,
                &mut cache.value,
                Some(&mut self.stream),
            )?;
            cache.record_append(&mut self.stream)?;
            self.account_device_kv(
                0,
                kv_bytes
                    .checked_mul(2)
                    .ok_or_else(|| "Gemma resident KV write bytes overflow".to_string())?,
                false,
            )?;
            self.attention_device_resident(cache, q_heads, kv_heads, head_dim)?;
        } else {
            self.attention_device_resident(shared_cache.unwrap(), q_heads, kv_heads, head_dim)?;
        }
        matvec_bf16_f32(
            o_matrix,
            self.attention_output.as_ref().unwrap(),
            hidden,
            q_width,
            self.resident_output.as_mut().unwrap(),
            Some(&mut self.stream),
        )?;
        bf16_row_f32(
            post_weight,
            1,
            hidden,
            0,
            self.attention_post_norm_weight.as_mut().unwrap(),
            Some(&mut self.stream),
        )?;
        rmsnorm_f32(
            self.resident_output.as_ref().unwrap(),
            self.attention_post_norm_weight.as_ref().unwrap(),
            hidden,
            epsilon,
            self.attention_input_normed.as_mut().unwrap(),
            Some(&mut self.stream),
        )?;
        add_f32(
            self.resident_input.as_ref().unwrap(),
            self.attention_input_normed.as_ref().unwrap(),
            hidden,
            self.mlp_pre_normed.as_mut().unwrap(),
            Some(&mut self.stream),
        )?;
        let mut bytes = vec![0_u8; hidden_bytes];
        self.mlp_pre_normed.as_mut().unwrap().copy_to_host(
            0,
            &mut bytes,
            Some(&mut self.stream),
        )?;
        self.stream.synchronize()?;
        let output = decode_f32_le_values(&bytes);
        if output.iter().any(|value| !value.is_finite()) {
            return Err("Gemma resident attention region produced non-finite output".into());
        }
        let matrix_bytes = |rows: usize, cols: usize| {
            rows.checked_mul(cols)
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| "Gemma resident attention logical bytes overflow".to_string())
        };
        self.account_resident_weight_read(matrix_bytes(1, hidden)?, false)?;
        self.account_resident_weight_read(matrix_bytes(q_width, hidden)?, true)?;
        if owns_cache {
            self.account_resident_weight_read(matrix_bytes(kv_width, hidden)?, true)?;
            self.account_resident_weight_read(matrix_bytes(kv_width, hidden)?, true)?;
            self.account_resident_weight_read(matrix_bytes(1, head_dim)?, false)?;
        }
        self.account_resident_weight_read(matrix_bytes(1, head_dim)?, false)?;
        self.account_resident_weight_read(matrix_bytes(hidden, q_width)?, true)?;
        self.account_resident_weight_read(matrix_bytes(1, hidden)?, false)?;
        Ok(output)
    }

    fn attention_device_resident(
        &mut self,
        cache: &Gemma4DeviceKvCache,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<(), String> {
        if cache.cache_len == 0 {
            return Err("Gemma resident attention cache is empty".into());
        }
        paged_decode_attn_f32(
            self.attention_query.as_ref().unwrap(),
            &cache.key,
            &cache.value,
            &cache.read_table,
            cache.cache_len,
            GEMMA4_DEVICE_KV_BLOCK_SIZE,
            cache.capacity_tokens,
            q_heads,
            kv_heads,
            head_dim,
            head_dim,
            1.0,
            self.attention_output.as_mut().unwrap(),
            Some(&mut self.stream),
        )?;
        let read = cache
            .cache_len
            .checked_mul(cache.width()?)
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| "Gemma resident KV read bytes overflow".to_string())?;
        self.account_device_kv(read, 0, true)
    }

    fn append_device_kv(
        &mut self,
        cache: &mut Gemma4DeviceKvCache,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), String> {
        let operation_started = Instant::now();
        let width = cache.width()?;
        if key.len() != width || value.len() != width {
            return Err(format!(
                "Gemma4 device KV append width mismatch for layer {}: key={} value={} expected={width}",
                cache.layer_index,
                key.len(),
                value.len(),
            ));
        }
        if key
            .iter()
            .chain(value)
            .any(|component| !component.is_finite())
        {
            return Err(format!(
                "Gemma4 device KV append for layer {} contains non-finite values",
                cache.layer_index
            ));
        }
        let encode_started = Instant::now();
        let key_bytes = encode_f32_to_bytes(key);
        let value_bytes = encode_f32_to_bytes(value);
        let input_encode_ns = elapsed_ns(encode_started);
        Self::ensure_buffer(
            &mut self.context,
            &mut self.kv_key_input,
            key_bytes.len(),
            "device KV key staging",
            &mut self.resident_host_profile,
        )?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.kv_value_input,
            value_bytes.len(),
            "device KV value staging",
            &mut self.resident_host_profile,
        )?;
        let position = cache.write_position()?;
        let (key_slot, value_slot) = (&mut self.kv_key_input, &mut self.kv_value_input);
        let key_staging = key_slot
            .as_mut()
            .expect("device KV key staging buffer was allocated");
        let value_staging = value_slot
            .as_mut()
            .expect("device KV value staging buffer was allocated");
        let h2d_started = Instant::now();
        key_staging.copy_from_host(0, &key_bytes, Some(&mut self.stream))?;
        value_staging.copy_from_host(0, &value_bytes, Some(&mut self.stream))?;
        let h2d_submit_ns = elapsed_ns(h2d_started);
        let (key_cache, value_cache, read_table, write_table) = (
            &mut cache.key,
            &mut cache.value,
            &cache.read_table,
            &cache.write_table,
        );
        let write_table = write_table.as_ref().unwrap_or(read_table);
        let kernel_started = Instant::now();
        paged_kv_write_f32(
            key_staging,
            value_staging,
            write_table,
            position,
            GEMMA4_DEVICE_KV_BLOCK_SIZE,
            cache.capacity_tokens,
            cache.kv_heads,
            cache.head_dim,
            cache.head_dim,
            key_cache,
            value_cache,
            Some(&mut self.stream),
        )?;
        let kernel_submit_ns = elapsed_ns(kernel_started);
        let table_started = Instant::now();
        cache.record_append(&mut self.stream)?;
        let kv_table_host_ns = elapsed_ns(table_started);
        let write_bytes = width
            .checked_mul(2)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "Gemma4 resident KV write logical byte count overflows".to_string())?;
        self.account_device_kv(0, write_bytes, false)?;
        let primitive_ns = elapsed_ns(operation_started);
        let profile = &mut self.resident_host_profile;
        profile.input_encode_ns = profile.input_encode_ns.saturating_add(input_encode_ns);
        profile.h2d_submit_ns = profile.h2d_submit_ns.saturating_add(h2d_submit_ns);
        profile.kernel_submit_ns = profile.kernel_submit_ns.saturating_add(kernel_submit_ns);
        profile.kv_table_host_ns = profile.kv_table_host_ns.saturating_add(kv_table_host_ns);
        profile.primitive_ns = profile.primitive_ns.saturating_add(primitive_ns);
        profile.kv_write_calls = profile.kv_write_calls.saturating_add(1);
        profile.kv_write.record(
            primitive_ns,
            input_encode_ns,
            0,
            h2d_submit_ns,
            kernel_submit_ns,
            0,
            0,
            0,
            kv_table_host_ns,
        );
        Ok(())
    }

    fn device_attention(
        &mut self,
        cache: &Gemma4DeviceKvCache,
        query: &[f32],
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>, String> {
        let operation_started = Instant::now();
        if cache.cache_len == 0 {
            return Err(format!(
                "Gemma4 device attention layer {} has no KV entries",
                cache.layer_index
            ));
        }
        let expected_query = q_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "Gemma4 device attention query width overflows usize".to_string())?;
        if query.len() != expected_query {
            return Err(format!(
                "Gemma4 device attention query width mismatch: expected {expected_query}, got {}",
                query.len()
            ));
        }
        if kv_heads != cache.kv_heads || head_dim != cache.head_dim {
            return Err(format!(
                "Gemma4 device attention geometry disagrees with layer {} cache",
                cache.layer_index
            ));
        }
        let encode_started = Instant::now();
        let query_bytes = encode_f32_to_bytes(query);
        let input_encode_ns = elapsed_ns(encode_started);
        let output_bytes = expected_query
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                "Gemma4 device attention output byte count overflows usize".to_string()
            })?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.attention_query,
            query_bytes.len(),
            "device attention query",
            &mut self.resident_host_profile,
        )?;
        Self::ensure_buffer(
            &mut self.context,
            &mut self.attention_output,
            output_bytes,
            "device attention output",
            &mut self.resident_host_profile,
        )?;
        let (query_slot, output_slot) = (&mut self.attention_query, &mut self.attention_output);
        let query_buffer = query_slot
            .as_mut()
            .expect("device attention query buffer was allocated");
        let output_buffer = output_slot
            .as_mut()
            .expect("device attention output buffer was allocated");
        let h2d_started = Instant::now();
        query_buffer.copy_from_host(0, &query_bytes, Some(&mut self.stream))?;
        let h2d_submit_ns = elapsed_ns(h2d_started);
        let kernel_started = Instant::now();
        paged_decode_attn_f32(
            query_buffer,
            &cache.key,
            &cache.value,
            &cache.read_table,
            cache.cache_len,
            GEMMA4_DEVICE_KV_BLOCK_SIZE,
            cache.capacity_tokens,
            q_heads,
            kv_heads,
            head_dim,
            head_dim,
            1.0,
            output_buffer,
            Some(&mut self.stream),
        )?;
        let kernel_submit_ns = elapsed_ns(kernel_started);
        let output_allocation_started = Instant::now();
        let mut host_output = vec![0_u8; output_bytes];
        let output_allocation_ns = elapsed_ns(output_allocation_started);
        let d2h_started = Instant::now();
        output_buffer.copy_to_host(0, &mut host_output, Some(&mut self.stream))?;
        let d2h_submit_ns = elapsed_ns(d2h_started);
        let synchronize_started = Instant::now();
        self.stream.synchronize()?;
        let stream_synchronize_ns = elapsed_ns(synchronize_started);
        let decode_started = Instant::now();
        let output = decode_f32_le_values(&host_output);
        if output.len() != expected_query || output.iter().any(|value| !value.is_finite()) {
            return Err(
                "Gemma4 device attention returned non-finite or malformed F32 output".into(),
            );
        }
        let output_decode_validate_ns = elapsed_ns(decode_started);
        let read_bytes = cache
            .cache_len
            .checked_mul(cache.width()?)
            .and_then(|elements| elements.checked_mul(2))
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "Gemma4 resident KV read logical byte count overflows".to_string())?;
        self.account_device_kv(read_bytes, 0, true)?;
        let primitive_ns = elapsed_ns(operation_started);
        let profile = &mut self.resident_host_profile;
        profile.input_encode_ns = profile.input_encode_ns.saturating_add(input_encode_ns);
        profile.output_allocation_ns = profile
            .output_allocation_ns
            .saturating_add(output_allocation_ns);
        profile.h2d_submit_ns = profile.h2d_submit_ns.saturating_add(h2d_submit_ns);
        profile.kernel_submit_ns = profile.kernel_submit_ns.saturating_add(kernel_submit_ns);
        profile.d2h_submit_ns = profile.d2h_submit_ns.saturating_add(d2h_submit_ns);
        profile.stream_synchronize_ns = profile
            .stream_synchronize_ns
            .saturating_add(stream_synchronize_ns);
        profile.output_decode_validate_ns = profile
            .output_decode_validate_ns
            .saturating_add(output_decode_validate_ns);
        profile.primitive_ns = profile.primitive_ns.saturating_add(primitive_ns);
        profile.attention_calls = profile.attention_calls.saturating_add(1);
        profile.attention.record(
            primitive_ns,
            input_encode_ns,
            output_allocation_ns,
            h2d_submit_ns,
            kernel_submit_ns,
            d2h_submit_ns,
            stream_synchronize_ns,
            output_decode_validate_ns,
            0,
        );
        Ok(output)
    }
}

enum Gemma4KvStorage {
    Host(Gemma4KvCaches),
    Device(Gemma4DeviceKvCaches),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gemma4KvSharingMode {
    /// Match HF: layers 15+ read the final non-sharing local/full source.
    SourceCache,
    /// Diagnostic reference only: execute the physical K/V projection of
    /// every shared layer and retain that layer's independent host cache.
    ReprojectPhysical,
}

/// One causal Gemma4 text sequence over source BF16 weights and F32
/// activations. `load` creates a new R9700 HIP context; callers should call
/// `reset` before an unrelated request.
pub struct Gemma4TextExecutor {
    source_model_dir: PathBuf,
    config_sha256: String,
    resident_descriptor: ResidentModelDescriptor,
    config: Gemma4TextConfig,
    weights: Gemma4WeightStorage,
    matvec: Bf16MatvecRuntime,
    caches: Gemma4KvStorage,
    kv_sharing_mode: Gemma4KvSharingMode,
    resident_memory_plan: Option<Gemma4ResidentMemoryPlan>,
    position: usize,
}

impl Gemma4TextExecutor {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, String> {
        let loaded = load_model_config_from_dir(model_dir)?;
        Self::from_loaded_config(loaded)
    }

    pub fn from_loaded_config(loaded: LoadedModelConfig) -> Result<Self, String> {
        let resident_descriptor = loaded.resident_descriptor()?;
        resident_descriptor.require_gemma4_resident_bf16()?;
        let config = loaded.require_gemma4_text_executor()?.clone();
        let model_path = loaded.source_model_dir.join(GEMMA4_TEXT_MODEL_FILE);
        let weights = SafeTensorReader::open(&model_path)?;
        validate_checkpoint_contract(&weights, &resident_descriptor)?;
        let matvec = Bf16MatvecRuntime::create()?;
        Ok(Self {
            source_model_dir: loaded.source_model_dir,
            config_sha256: loaded.config_sha256,
            resident_descriptor,
            caches: Gemma4KvStorage::Host(Gemma4KvCaches::new(config.decoder.num_hidden_layers)),
            config,
            weights: Gemma4WeightStorage::Streamed(weights),
            matvec,
            kv_sharing_mode: Gemma4KvSharingMode::SourceCache,
            resident_memory_plan: None,
            position: 0,
        })
    }

    /// Loads every BF16 tensor from `model.safetensors` once onto the R9700
    /// and allocates device K/V caches for the maximum source-config context.
    /// Execution remains text-only, but vision/audio tensors are resident too
    /// so the allocation faithfully represents the complete checkpoint.
    pub fn load_resident(model_dir: impl AsRef<Path>) -> Result<Self, String> {
        let loaded = load_model_config_from_dir(model_dir)?;
        Self::from_loaded_config_resident(loaded)
    }

    pub fn from_loaded_config_resident(loaded: LoadedModelConfig) -> Result<Self, String> {
        let resident_descriptor = loaded.resident_descriptor()?;
        resident_descriptor.require_gemma4_resident_bf16()?;
        let config = loaded.require_gemma4_text_executor()?.clone();
        let model_path = loaded.source_model_dir.join(GEMMA4_TEXT_MODEL_FILE);
        let mut source_weights = SafeTensorReader::open(&model_path)?;
        validate_checkpoint_contract(&source_weights, &resident_descriptor)?;
        let memory_plan =
            Gemma4ResidentMemoryPlan::from_checkpoint(&source_weights, &resident_descriptor)?;
        let mut matvec = Bf16MatvecRuntime::create()?;
        let resident_weights =
            ResidentGemma4Weights::upload_checkpoint(&mut source_weights, &mut matvec)?;
        if resident_weights.bytes() != memory_plan.resident_checkpoint_weight_bytes
            || resident_weights.tensor_count() != memory_plan.resident_checkpoint_tensor_count
        {
            return Err(format!(
                "Gemma4 resident upload accounting mismatch: uploaded {} bytes / {} tensors, plan has {} bytes / {} tensors",
                resident_weights.bytes(),
                resident_weights.tensor_count(),
                memory_plan.resident_checkpoint_weight_bytes,
                memory_plan.resident_checkpoint_tensor_count,
            ));
        }
        let device_caches = Gemma4DeviceKvCaches::new(&resident_descriptor, &mut matvec)?;
        let actual_kv_bytes = device_caches.allocated_bytes()?;
        let planned_kv_bytes = memory_plan.estimated_kv_bytes(config.max_position_embeddings)?;
        if actual_kv_bytes != planned_kv_bytes {
            return Err(format!(
                "Gemma4 resident KV accounting mismatch: allocated {actual_kv_bytes} bytes, plan has {planned_kv_bytes}"
            ));
        }
        Ok(Self {
            source_model_dir: loaded.source_model_dir,
            config_sha256: loaded.config_sha256,
            resident_descriptor,
            caches: Gemma4KvStorage::Device(device_caches),
            config,
            weights: Gemma4WeightStorage::Resident(resident_weights),
            matvec,
            kv_sharing_mode: Gemma4KvSharingMode::SourceCache,
            resident_memory_plan: Some(memory_plan),
            position: 0,
        })
    }

    pub fn config(&self) -> &Gemma4TextConfig {
        &self.config
    }

    /// The config-derived topology actually selected for this executor.  The
    /// legacy config accessor remains for callers that need source metadata;
    /// execution semantics below read this descriptor for layer geometry.
    pub fn resident_descriptor(&self) -> &ResidentModelDescriptor {
        &self.resident_descriptor
    }

    pub fn source_model_dir(&self) -> &Path {
        &self.source_model_dir
    }

    pub fn config_sha256(&self) -> &str {
        &self.config_sha256
    }

    pub fn device(&self) -> &Gemma4TextDeviceIdentity {
        self.matvec.device()
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn is_resident(&self) -> bool {
        matches!(self.weights, Gemma4WeightStorage::Resident(_))
    }

    pub fn resident_memory_plan(&self) -> Option<&Gemma4ResidentMemoryPlan> {
        self.resident_memory_plan.as_ref()
    }

    pub fn resident_weight_bytes(&self) -> Option<u64> {
        match &self.weights {
            Gemma4WeightStorage::Streamed(_) => None,
            Gemma4WeightStorage::Resident(weights) => Some(weights.bytes()),
        }
    }

    pub fn device_kv_bytes(&self) -> Result<Option<u64>, String> {
        match &self.caches {
            Gemma4KvStorage::Host(_) => Ok(None),
            Gemma4KvStorage::Device(caches) => caches.allocated_bytes().map(Some),
        }
    }

    /// Actual temporary device buffers allocated so far.  Unlike the memory
    /// plan this grows lazily as the first prefill/decode reaches each
    /// projection and attention shape.
    pub fn device_transient_bytes(&self) -> Result<Option<u64>, String> {
        if self.is_resident() {
            self.matvec.resident_transient_bytes().map(Some)
        } else {
            Ok(None)
        }
    }

    pub fn resident_mlp_validation(&self) -> Gemma4ResidentMlpValidation {
        self.matvec.mlp_validation()
    }

    pub fn resident_rope_validation(&self) -> Gemma4ResidentRopeValidation {
        self.matvec.rope_validation()
    }

    /// Actual bytes held by text weights, K/V storage, and the lazily
    /// allocated temporary buffers.  It intentionally excludes HIP runtime
    /// allocator/context overhead that is not exposed through RuntimeBuffer.
    pub fn resident_device_allocation_bytes(&self) -> Result<Option<u64>, String> {
        let Some(weight_bytes) = self.resident_weight_bytes() else {
            return Ok(None);
        };
        let kv_bytes = self
            .device_kv_bytes()?
            .ok_or_else(|| "resident Gemma4 executor has no device KV cache".to_string())?;
        let transient_bytes = self
            .device_transient_bytes()?
            .ok_or_else(|| "resident Gemma4 executor has no transient accounting".to_string())?;
        weight_bytes
            .checked_add(kv_bytes)
            .and_then(|bytes| bytes.checked_add(transient_bytes))
            .map(Some)
            .ok_or_else(|| "Gemma4 resident allocation byte count overflows u64".to_string())
    }

    /// Logical lower-bound stream accounting since the most recent reset of
    /// the counters.  It is available only for the resident execution path.
    pub fn resident_logical_bytes(&self) -> Option<Gemma4ResidentLogicalBytes> {
        self.is_resident()
            .then(|| self.matvec.resident_logical_bytes())
    }

    pub fn reset_resident_logical_bytes(&mut self) {
        self.matvec.reset_resident_logical_bytes();
    }

    /// Host-side primitive accounting since the most recent reset. Available
    /// only while the source BF16 checkpoint is resident on device.
    pub fn resident_host_profile(&self) -> Option<Gemma4ResidentHostProfile> {
        self.is_resident()
            .then(|| self.matvec.resident_host_profile())
    }

    pub fn reset_resident_host_profile(&mut self) {
        self.matvec.reset_resident_host_profile();
    }

    /// Immutable device-cache state, including the explicit layer-to-source
    /// mapping used by Gemma4's shared K/V attention layers.
    pub fn resident_kv_cache_snapshot(
        &self,
    ) -> Result<Option<Gemma4ResidentKvCacheSnapshot>, String> {
        let Gemma4KvStorage::Device(caches) = &self.caches else {
            return Ok(None);
        };
        let shared_layer_sources = self
            .resident_descriptor
            .layers
            .iter()
            .filter_map(|layer| match layer.attention.kv_cache {
                ResidentKvCacheMode::SharedFrom { source_layer_index } => {
                    Some(Gemma4SharedKvSource {
                        layer_index: layer.layer_index,
                        layer_kind: layer.attention.kind.as_str().to_string(),
                        source_layer_index,
                    })
                }
                ResidentKvCacheMode::Own | ResidentKvCacheMode::LinearState => None,
            })
            .collect();
        Ok(Some(Gemma4ResidentKvCacheSnapshot {
            source_layers: caches.source_layer_states()?,
            shared_layer_sources,
        }))
    }

    /// Runs the deliberately-wrong "no KV sharing" topology against the
    /// same resident BF16 weights, then restores the normal device cache.
    ///
    /// Gemma4 HF ignores the physical K/V modules for shared layers, so this
    /// is not a candidate execution path.  It is a focused diagnostic: a
    /// difference from the normal trace demonstrates that layers 15+ are not
    /// silently using their checkpoint K/V tensors instead of sources 13/14.
    pub fn unshared_kv_reference_generation(
        &mut self,
        initial_token_ids: &[u32],
        new_tokens: usize,
    ) -> Result<Gemma4UnsharedKvReference, String> {
        if !self.is_resident() {
            return Err("unshared Gemma4 K/V reference requires resident BF16 weights".into());
        }
        if new_tokens == 0 {
            return Err("unshared Gemma4 K/V reference needs at least one generated token".into());
        }
        let saved_caches = std::mem::replace(
            &mut self.caches,
            Gemma4KvStorage::Host(Gemma4KvCaches::new(self.resident_descriptor.layers.len())),
        );
        let saved_mode = std::mem::replace(
            &mut self.kv_sharing_mode,
            Gemma4KvSharingMode::ReprojectPhysical,
        );
        let saved_position = self.position;
        let saved_logical_bytes = self.matvec.resident_logical_bytes();
        self.position = 0;
        self.matvec.reset_resident_logical_bytes();
        let result = (|| {
            let mut generated_token_ids = Vec::with_capacity(new_tokens);
            let mut top1_logits = Vec::with_capacity(new_tokens);
            let trace = self.prefill(initial_token_ids)?;
            generated_token_ids.push(trace.top1.token_id);
            top1_logits.push(trace.top1.logit);
            for _ in 1..new_tokens {
                let input = *generated_token_ids.last().ok_or_else(|| {
                    "unshared Gemma4 K/V reference lost its previous token".to_string()
                })?;
                let trace = self.decode(input)?;
                generated_token_ids.push(trace.top1.token_id);
                top1_logits.push(trace.top1.logit);
            }
            Ok(Gemma4UnsharedKvReference {
                generated_token_ids,
                top1_logits,
            })
        })();
        self.caches = saved_caches;
        self.kv_sharing_mode = saved_mode;
        self.position = saved_position;
        self.matvec.resident_logical_bytes = saved_logical_bytes;
        result
    }

    pub fn reset(&mut self) {
        match &mut self.caches {
            Gemma4KvStorage::Host(caches) => {
                *caches = Gemma4KvCaches::new(self.resident_descriptor.layers.len());
            }
            Gemma4KvStorage::Device(caches) => caches.reset(),
        }
        self.position = 0;
    }

    /// Prefills a sequence of M=N input tokens.  The current generic
    /// BF16×F32 primitive is a matvec primitive, so projection launches are
    /// issued in causal token order; the public operation and K/V transition
    /// are nevertheless a prefill of the complete supplied sequence.
    pub fn prefill(&mut self, input_token_ids: &[u32]) -> Result<Gemma4TextStepTrace, String> {
        self.execute_step(input_token_ids)
    }

    /// Runs one M=1 decode token against the retained cache.
    pub fn decode(&mut self, token_id: u32) -> Result<Gemma4TextStepTrace, String> {
        self.execute_step(&[token_id])
    }

    fn layer_descriptor(
        &self,
        layer_index: usize,
    ) -> Result<&crate::model_config::ResidentLayerDescriptor, String> {
        self.resident_descriptor.layer(layer_index)
    }

    fn ple_descriptor(
        &self,
    ) -> Result<&crate::model_config::ResidentPerLayerEmbeddingDescriptor, String> {
        self.resident_descriptor
            .layers
            .first()
            .and_then(|layer| layer.per_layer_embedding.as_ref())
            .ok_or_else(|| "Gemma4 resident descriptor has no per-layer embedding contract".into())
    }

    /// Executes one causal input chunk. The chunk is evaluated token-by-token
    /// to preserve a simple explicit KV cache, but returned rows are laid out
    /// exactly as `[batch=1, tokens, width]` for the trace writer.
    pub fn execute_step(&mut self, input_token_ids: &[u32]) -> Result<Gemma4TextStepTrace, String> {
        // This first prefill activation increment only batches the PLE model
        // projection.  It is deliberately unavailable to decode so the M=1
        // resident route remains exactly as it was before this change.
        if input_token_ids.len() > 1 && matches!(self.weights, Gemma4WeightStorage::Resident(_)) {
            return self.execute_prefill_with_batched_ple(input_token_ids);
        }
        let token_started = Instant::now();
        let primitive_before = self.matvec.resident_host_profile().primitive_ns;
        if input_token_ids.is_empty() {
            return Err("Gemma4TextExecutor input token list must be nonempty".into());
        }
        let next_position = self
            .position
            .checked_add(input_token_ids.len())
            .ok_or_else(|| "Gemma4 sequence position overflows usize".to_string())?;
        if next_position > self.resident_descriptor.decoder.max_position_embeddings {
            return Err(format!(
                "Gemma4 input reaches position {next_position}, beyond max_position_embeddings={}",
                self.resident_descriptor.decoder.max_position_embeddings
            ));
        }
        let hidden = self.resident_descriptor.decoder.hidden_size;
        let layers = self.resident_descriptor.layers.len();
        let ple_vocabulary = self.ple_descriptor()?.vocabulary_size;
        let mut embedding = Vec::with_capacity(
            input_token_ids
                .len()
                .checked_mul(hidden)
                .ok_or_else(|| "Gemma4 embedding trace allocation overflows".to_string())?,
        );
        let mut layer_outputs = (0..layers)
            .map(|_| Vec::with_capacity(input_token_ids.len() * hidden))
            .collect::<Vec<_>>();
        let mut final_norm = Vec::with_capacity(input_token_ids.len() * hidden);
        let mut final_token_hidden = None;
        for token_id in input_token_ids {
            let token_index = usize::try_from(*token_id)
                .map_err(|_| "Gemma4 token ID does not fit usize".to_string())?;
            if token_index >= self.resident_descriptor.decoder.vocab_size {
                return Err(format!(
                    "Gemma4 token ID {token_id} is outside vocabulary 0..{}",
                    self.resident_descriptor.decoder.vocab_size
                ));
            }
            if token_index >= ple_vocabulary {
                return Err(format!(
                    "Gemma4 token ID {token_id} is outside PLE vocabulary 0..{}",
                    ple_vocabulary
                ));
            }
            let token = self.forward_token(*token_id)?;
            embedding.extend_from_slice(&token.embedding);
            for (layer_index, output) in token.layer_outputs.iter().enumerate() {
                layer_outputs[layer_index].extend_from_slice(output);
            }
            final_norm.extend_from_slice(&token.final_norm);
            final_token_hidden = Some(token.final_norm);
        }
        let final_token_hidden = final_token_hidden.expect("checked nonempty input");
        let logits_last = self.project_tied_logits(&final_token_hidden)?;
        let top1 = top1_from_logits(&logits_last)?;
        let trace = Gemma4TextStepTrace {
            input_token_ids: input_token_ids.to_vec(),
            embedding,
            layer_outputs,
            final_norm,
            logits_last,
            top1,
        };
        let token_forward_ns = elapsed_ns(token_started);
        let primitive_ns = self
            .matvec
            .resident_host_profile()
            .primitive_ns
            .saturating_sub(primitive_before);
        let profile = &mut self.matvec.resident_host_profile;
        profile.token_forward_ns = profile.token_forward_ns.saturating_add(token_forward_ns);
        profile.executor_other_ns = profile
            .executor_other_ns
            .saturating_add(token_forward_ns.saturating_sub(primitive_ns));
        Ok(trace)
    }

    fn execute_prefill_with_batched_ple(
        &mut self,
        input_token_ids: &[u32],
    ) -> Result<Gemma4TextStepTrace, String> {
        let token_started = Instant::now();
        let primitive_before = self.matvec.resident_host_profile().primitive_ns;
        if input_token_ids.is_empty() {
            return Err("Gemma4TextExecutor input token list must be nonempty".into());
        }
        let next_position = self
            .position
            .checked_add(input_token_ids.len())
            .ok_or_else(|| "Gemma4 sequence position overflows usize".to_string())?;
        if next_position > self.resident_descriptor.decoder.max_position_embeddings {
            return Err(format!(
                "Gemma4 input reaches position {next_position}, beyond max_position_embeddings={}",
                self.resident_descriptor.decoder.max_position_embeddings
            ));
        }
        let hidden = self.resident_descriptor.decoder.hidden_size;
        let layers = self.resident_descriptor.layers.len();
        let mut embedding = Vec::with_capacity(input_token_ids.len() * hidden);
        let mut layer_outputs = (0..layers)
            .map(|_| Vec::with_capacity(input_token_ids.len() * hidden))
            .collect::<Vec<_>>();
        let mut final_norm = Vec::with_capacity(input_token_ids.len() * hidden);
        let mut final_token_hidden = None;

        for token_chunk in input_token_ids.chunks(GEMMA4_PREFILL_ACTIVATION_CHUNK_TOKENS) {
            let (chunk_embeddings, per_layer_inputs) =
                self.prefill_ple_inputs_batched(token_chunk)?;
            let ple_width = self
                .resident_descriptor
                .layers
                .len()
                .checked_mul(self.ple_descriptor()?.input_size)
                .ok_or_else(|| "Gemma4 batched PLE width overflows".to_string())?;
            for (token_offset, token_id) in token_chunk.iter().enumerate() {
                let embedding_start = token_offset
                    .checked_mul(hidden)
                    .ok_or_else(|| "Gemma4 batched embedding offset overflows".to_string())?;
                let ple_start = token_offset
                    .checked_mul(ple_width)
                    .ok_or_else(|| "Gemma4 batched PLE offset overflows".to_string())?;
                let token = self.forward_token_prefill_from_ple(
                    *token_id,
                    &chunk_embeddings[embedding_start..embedding_start + hidden],
                    &per_layer_inputs[ple_start..ple_start + ple_width],
                )?;
                embedding.extend_from_slice(&token.embedding);
                for (layer_index, output) in token.layer_outputs.iter().enumerate() {
                    layer_outputs[layer_index].extend_from_slice(output);
                }
                final_norm.extend_from_slice(&token.final_norm);
                final_token_hidden = Some(token.final_norm);
            }
        }
        let final_token_hidden = final_token_hidden.expect("checked nonempty input");
        let logits_last = self.project_tied_logits(&final_token_hidden)?;
        let top1 = top1_from_logits(&logits_last)?;
        let trace = Gemma4TextStepTrace {
            input_token_ids: input_token_ids.to_vec(),
            embedding,
            layer_outputs,
            final_norm,
            logits_last,
            top1,
        };
        let token_forward_ns = elapsed_ns(token_started);
        let primitive_ns = self
            .matvec
            .resident_host_profile()
            .primitive_ns
            .saturating_sub(primitive_before);
        let profile = &mut self.matvec.resident_host_profile;
        profile.token_forward_ns = profile.token_forward_ns.saturating_add(token_forward_ns);
        profile.executor_other_ns = profile
            .executor_other_ns
            .saturating_add(token_forward_ns.saturating_sub(primitive_ns));
        Ok(trace)
    }

    /// Computes all token-dependent PLE inputs for one prefill chunk before
    /// layer execution.  PLE has no K/V dependency, so this cannot affect the
    /// causal cache transition; each subsequent token still takes the existing
    /// M=1 attention route with its own cache length.
    fn prefill_ple_inputs_batched(
        &mut self,
        token_ids: &[u32],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let decoder = self.resident_descriptor.decoder.clone();
        let ple = self.ple_descriptor()?.clone();
        let layers = self.resident_descriptor.layers.len();
        let hidden = decoder.hidden_size;
        let packed_width = layers
            .checked_mul(ple.input_size)
            .ok_or_else(|| "Gemma4 batched PLE packed width overflows".to_string())?;
        let embedding_scale = self.resident_descriptor.embedding.scale.ok_or_else(|| {
            "Gemma4 resident descriptor is missing the token embedding scale".to_string()
        })?;
        let mut embeddings = Vec::with_capacity(token_ids.len() * hidden);
        let mut token_identity = Vec::with_capacity(token_ids.len() * packed_width);
        for token_id in token_ids {
            let token_index = usize::try_from(*token_id)
                .map_err(|_| "Gemma4 token ID does not fit usize".to_string())?;
            if token_index >= decoder.vocab_size || token_index >= ple.vocabulary_size {
                return Err(format!("Gemma4 token ID {token_id} is outside vocabulary"));
            }
            let mut token_embedding = self.read_weight_row(
                GEMMA4_TEXT_EMBED_TOKENS,
                decoder.vocab_size,
                hidden,
                token_index,
            )?;
            scale_in_place(
                &mut token_embedding,
                embedding_scale,
                "Gemma4 embedding scale",
            )?;
            embeddings.extend_from_slice(&token_embedding);
            let mut identity_row = self.read_weight_row(
                GEMMA4_TEXT_EMBED_TOKENS_PER_LAYER,
                ple.vocabulary_size,
                packed_width,
                token_index,
            )?;
            scale_in_place(
                &mut identity_row,
                ple.token_embedding_scale,
                "Gemma4 PLE token embedding scale",
            )?;
            token_identity.extend_from_slice(&identity_row);
        }
        let matrix = match &self.weights {
            Gemma4WeightStorage::Resident(weights) => weights.tensor(
                GEMMA4_TEXT_PER_LAYER_MODEL_PROJECTION,
                &[packed_width, hidden],
            )?,
            Gemma4WeightStorage::Streamed(_) => {
                return Err("Gemma4 batched PLE requires resident weights".into());
            }
        };
        let mut projected = self.matvec.gemma_matmul_resident(
            &matrix.buffer,
            packed_width,
            hidden,
            token_ids.len(),
            &embeddings,
        )?;
        scale_in_place(
            &mut projected,
            ple.model_projection_scale,
            "Gemma4 PLE model projection scale",
        )?;
        let norm_weight =
            self.read_weight_vector(GEMMA4_TEXT_PER_LAYER_PROJECTION_NORM, ple.input_size)?;
        for (token_index, identity) in token_identity.chunks_exact(packed_width).enumerate() {
            let token_start = token_index
                .checked_mul(packed_width)
                .ok_or_else(|| "Gemma4 batched PLE token offset overflows".to_string())?;
            for layer_index in 0..layers {
                let start = token_start
                    .checked_add(layer_index * ple.input_size)
                    .ok_or_else(|| "Gemma4 batched PLE slice offset overflows".to_string())?;
                let end = start
                    .checked_add(ple.input_size)
                    .ok_or_else(|| "Gemma4 batched PLE slice end overflows".to_string())?;
                let normalized = rms_norm(
                    &projected[start..end],
                    Some(&norm_weight),
                    decoder.rms_norm_epsilon,
                )?;
                for (index, value) in projected[start..end].iter_mut().enumerate() {
                    *value = (normalized[index] + identity[layer_index * ple.input_size + index])
                        * ple.residual_combine_scale;
                }
            }
        }
        finite_slice(&projected, "Gemma4 batched PLE")?;
        Ok((embeddings, projected))
    }

    fn forward_token_prefill_from_ple(
        &mut self,
        token_id: u32,
        embedding: &[f32],
        per_layer_inputs: &[f32],
    ) -> Result<TokenForward, String> {
        let decoder = self.resident_descriptor.decoder.clone();
        let ple = self.ple_descriptor()?.clone();
        let hidden = decoder.hidden_size;
        let layers = self.resident_descriptor.layers.len();
        let ple_dim = ple.input_size;
        if embedding.len() != hidden || per_layer_inputs.len() != layers * ple_dim {
            return Err("Gemma4 batched prefill PLE input shape is invalid".into());
        }
        let mut hidden_states = embedding.to_vec();
        let mut layer_outputs = Vec::with_capacity(layers);
        let position = self.position;
        for layer_index in 0..layers {
            let ple_start = layer_index
                .checked_mul(ple_dim)
                .ok_or_else(|| "Gemma4 batched PLE layer offset overflows".to_string())?;
            let ple_end = ple_start
                .checked_add(ple_dim)
                .ok_or_else(|| "Gemma4 batched PLE layer end overflows".to_string())?;
            hidden_states = self.forward_layer(
                layer_index,
                &hidden_states,
                &per_layer_inputs[ple_start..ple_end],
                position,
            )?;
            layer_outputs.push(hidden_states.clone());
        }
        let final_weight = self.read_weight_vector(GEMMA4_TEXT_FINAL_NORM, hidden)?;
        let final_norm = rms_norm(
            &hidden_states,
            Some(&final_weight),
            decoder.rms_norm_epsilon,
        )?;
        self.position = self
            .position
            .checked_add(1)
            .ok_or_else(|| "Gemma4 sequence position overflows usize".to_string())?;
        let _ = token_id;
        Ok(TokenForward {
            embedding: embedding.to_vec(),
            layer_outputs,
            final_norm,
        })
    }

    fn forward_token(&mut self, token_id: u32) -> Result<TokenForward, String> {
        let token_index = usize::try_from(token_id)
            .map_err(|_| "Gemma4 token ID does not fit usize".to_string())?;
        let decoder = self.resident_descriptor.decoder.clone();
        let ple = self.ple_descriptor()?.clone();
        let hidden = decoder.hidden_size;
        let layers = self.resident_descriptor.layers.len();
        let ple_dim = ple.input_size;
        let embedding_scale = self.resident_descriptor.embedding.scale.ok_or_else(|| {
            "Gemma4 resident descriptor is missing the token embedding scale".to_string()
        })?;
        let mut embedding = self.read_weight_row(
            GEMMA4_TEXT_EMBED_TOKENS,
            decoder.vocab_size,
            hidden,
            token_index,
        )?;
        scale_in_place(&mut embedding, embedding_scale, "Gemma4 embedding scale")?;
        let per_layer_inputs = self.compute_ple(token_index, &embedding)?;
        if per_layer_inputs.len() != layers * ple_dim {
            return Err("Gemma4 PLE result has an invalid packed width".into());
        }
        let mut hidden_states = embedding.clone();
        let mut layer_outputs = Vec::with_capacity(layers);
        let position = self.position;
        for layer_index in 0..layers {
            let ple_start = layer_index
                .checked_mul(ple_dim)
                .ok_or_else(|| "Gemma4 PLE layer offset overflows".to_string())?;
            let ple_end = ple_start
                .checked_add(ple_dim)
                .ok_or_else(|| "Gemma4 PLE layer end overflows".to_string())?;
            hidden_states = self.forward_layer(
                layer_index,
                &hidden_states,
                &per_layer_inputs[ple_start..ple_end],
                position,
            )?;
            layer_outputs.push(hidden_states.clone());
        }
        let final_weight = self.read_weight_vector(GEMMA4_TEXT_FINAL_NORM, hidden)?;
        let final_norm = rms_norm(
            &hidden_states,
            Some(&final_weight),
            decoder.rms_norm_epsilon,
        )?;
        self.position = self
            .position
            .checked_add(1)
            .ok_or_else(|| "Gemma4 sequence position overflows usize".to_string())?;
        Ok(TokenForward {
            embedding,
            layer_outputs,
            final_norm,
        })
    }

    fn compute_ple(&mut self, token_index: usize, embedding: &[f32]) -> Result<Vec<f32>, String> {
        let decoder = self.resident_descriptor.decoder.clone();
        let ple = self.ple_descriptor()?.clone();
        let layers = self.resident_descriptor.layers.len();
        let ple_dim = ple.input_size;
        let packed_width = layers
            .checked_mul(ple_dim)
            .ok_or_else(|| "Gemma4 PLE packed width overflows".to_string())?;
        let mut token_identity = self.read_weight_row(
            GEMMA4_TEXT_EMBED_TOKENS_PER_LAYER,
            ple.vocabulary_size,
            packed_width,
            token_index,
        )?;
        scale_in_place(
            &mut token_identity,
            ple.token_embedding_scale,
            "Gemma4 PLE token embedding scale",
        )?;
        let mut projected = self.matmul_named(
            GEMMA4_TEXT_PER_LAYER_MODEL_PROJECTION,
            packed_width,
            decoder.hidden_size,
            embedding,
        )?;
        scale_in_place(
            &mut projected,
            ple.model_projection_scale,
            "Gemma4 PLE model projection scale",
        )?;
        let norm_weight =
            self.read_weight_vector(GEMMA4_TEXT_PER_LAYER_PROJECTION_NORM, ple_dim)?;
        for layer_index in 0..layers {
            let start = layer_index
                .checked_mul(ple_dim)
                .ok_or_else(|| "Gemma4 PLE slice offset overflows".to_string())?;
            let end = start
                .checked_add(ple_dim)
                .ok_or_else(|| "Gemma4 PLE slice end overflows".to_string())?;
            let normalized = rms_norm(
                &projected[start..end],
                Some(&norm_weight),
                decoder.rms_norm_epsilon,
            )?;
            for (index, value) in projected[start..end].iter_mut().enumerate() {
                *value = (normalized[index] + token_identity[start + index])
                    * ple.residual_combine_scale;
            }
        }
        finite_slice(&projected, "Gemma4 PLE")?;
        Ok(projected)
    }

    fn forward_layer(
        &mut self,
        layer_index: usize,
        hidden_states: &[f32],
        per_layer_input: &[f32],
        position: usize,
    ) -> Result<Vec<f32>, String> {
        let decoder = self.resident_descriptor.decoder.clone();
        let layer = self.layer_descriptor(layer_index)?.clone();
        let ple = layer
            .per_layer_embedding
            .as_ref()
            .ok_or_else(|| format!("Gemma4 descriptor layer {layer_index} is missing PLE"))?;
        let hidden = decoder.hidden_size;
        if hidden_states.len() != hidden {
            return Err(format!(
                "Gemma4 layer {layer_index} input width mismatch: expected {hidden}, got {}",
                hidden_states.len()
            ));
        }
        if per_layer_input.len() != ple.input_size {
            return Err(format!(
                "Gemma4 layer {layer_index} PLE width mismatch: expected {}, got {}",
                ple.input_size,
                per_layer_input.len()
            ));
        }
        let attention_residual = if env::var("ULLM_GEMMA4_DISABLE_ATTENTION_REGION")
            .ok()
            .as_deref()
            != Some("1")
            && matches!(self.weights, Gemma4WeightStorage::Resident(_))
            && matches!(self.caches, Gemma4KvStorage::Device(_))
        {
            self.forward_attention_norm_resident(layer_index, hidden_states, position)?
        } else {
            let residual_attention = hidden_states.to_vec();
            let input_norm_weight = self
                .read_weight_vector(&layer_tensor(layer_index, "input_layernorm.weight"), hidden)?;
            let input_norm = rms_norm(
                hidden_states,
                Some(&input_norm_weight),
                decoder.rms_norm_epsilon,
            )?;
            let attention = self.forward_attention(layer_index, &input_norm, position)?;
            let post_attention_weight = self.read_weight_vector(
                &layer_tensor(layer_index, "post_attention_layernorm.weight"),
                hidden,
            )?;
            let post_attention = rms_norm(
                &attention,
                Some(&post_attention_weight),
                decoder.rms_norm_epsilon,
            )?;
            add_vectors(
                &residual_attention,
                &post_attention,
                "Gemma4 attention residual",
            )?
        };

        let mlp_residual = if let Gemma4WeightStorage::Resident(weights) = &self.weights {
            let intermediate = match layer.mlp {
                ResidentMlpDescriptor::Dense {
                    intermediate_size, ..
                } => intermediate_size,
                ResidentMlpDescriptor::MoE { .. } => {
                    return Err(format!(
                        "Gemma4 resident executor cannot execute MoE MLP at layer {layer_index}"
                    ));
                }
            };
            let gate = weights.tensor(
                &layer_tensor(layer_index, "mlp.gate_proj.weight"),
                &[intermediate, hidden],
            )?;
            let up = weights.tensor(
                &layer_tensor(layer_index, "mlp.up_proj.weight"),
                &[intermediate, hidden],
            )?;
            let down = weights.tensor(
                &layer_tensor(layer_index, "mlp.down_proj.weight"),
                &[hidden, intermediate],
            )?;
            let pre_feedforward_weight = weights.tensor(
                &layer_tensor(layer_index, "pre_feedforward_layernorm.weight"),
                &[hidden],
            )?;
            let post_feedforward_weight = weights.tensor(
                &layer_tensor(layer_index, "post_feedforward_layernorm.weight"),
                &[hidden],
            )?;
            self.matvec.dense_mlp_norm_residual_resident(
                &gate.buffer,
                &up.buffer,
                &down.buffer,
                &pre_feedforward_weight.buffer,
                &post_feedforward_weight.buffer,
                hidden,
                intermediate,
                decoder.rms_norm_epsilon,
                &attention_residual,
            )?
        } else {
            let residual_mlp = attention_residual.clone();
            let pre_feedforward_weight = self.read_weight_vector(
                &layer_tensor(layer_index, "pre_feedforward_layernorm.weight"),
                hidden,
            )?;
            let feedforward_input = rms_norm(
                &attention_residual,
                Some(&pre_feedforward_weight),
                decoder.rms_norm_epsilon,
            )?;
            let mlp = self.forward_mlp(layer_index, &feedforward_input)?;
            let post_feedforward_weight = self.read_weight_vector(
                &layer_tensor(layer_index, "post_feedforward_layernorm.weight"),
                hidden,
            )?;
            let post_feedforward = rms_norm(
                &mlp,
                Some(&post_feedforward_weight),
                decoder.rms_norm_epsilon,
            )?;
            add_vectors(&residual_mlp, &post_feedforward, "Gemma4 MLP residual")?
        };

        let mut output = if env::var(GEMMA4_DISABLE_PLE_REGION_ENV).ok().as_deref() != Some("1") {
            if let Gemma4WeightStorage::Resident(weights) = &self.weights {
                let gate = weights.tensor(
                    &layer_tensor(layer_index, "per_layer_input_gate.weight"),
                    &[ple.input_size, hidden],
                )?;
                let projection = weights.tensor(
                    &layer_tensor(layer_index, "per_layer_projection.weight"),
                    &[hidden, ple.input_size],
                )?;
                let post_weight = weights.tensor(
                    &layer_tensor(layer_index, "post_per_layer_input_norm.weight"),
                    &[hidden],
                )?;
                self.matvec.ple_norm_residual_resident(
                    &gate.buffer,
                    &projection.buffer,
                    &post_weight.buffer,
                    hidden,
                    ple.input_size,
                    decoder.rms_norm_epsilon,
                    &mlp_residual,
                    per_layer_input,
                )?
            } else {
                self.forward_ple_host(
                    layer_index,
                    hidden,
                    ple.input_size,
                    decoder.rms_norm_epsilon,
                    &mlp_residual,
                    per_layer_input,
                )?
            }
        } else {
            self.forward_ple_host(
                layer_index,
                hidden,
                ple.input_size,
                decoder.rms_norm_epsilon,
                &mlp_residual,
                per_layer_input,
            )?
        };
        let layer_scalar =
            self.read_weight_vector(&layer_tensor(layer_index, "layer_scalar"), 1)?[0];
        scale_in_place(&mut output, layer_scalar, "Gemma4 layer scalar")?;
        Ok(output)
    }

    fn forward_ple_host(
        &mut self,
        layer_index: usize,
        hidden: usize,
        ple_dim: usize,
        epsilon: f32,
        mlp_residual: &[f32],
        per_layer_input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let ple_gate = self.matmul_named(
            &layer_tensor(layer_index, "per_layer_input_gate.weight"),
            ple_dim,
            hidden,
            mlp_residual,
        )?;
        let mut ple_product = gelu_pytorch_tanh(&ple_gate)?;
        multiply_in_place(&mut ple_product, per_layer_input, "Gemma4 PLE gate product")?;
        let ple_projection = self.matmul_named(
            &layer_tensor(layer_index, "per_layer_projection.weight"),
            hidden,
            ple_dim,
            &ple_product,
        )?;
        let post_ple_weight = self.read_weight_vector(
            &layer_tensor(layer_index, "post_per_layer_input_norm.weight"),
            hidden,
        )?;
        let post_ple = rms_norm(&ple_projection, Some(&post_ple_weight), epsilon)?;
        add_vectors(mlp_residual, &post_ple, "Gemma4 PLE residual")
    }

    fn forward_attention_norm_resident(
        &mut self,
        layer_index: usize,
        hidden_states: &[f32],
        position: usize,
    ) -> Result<Vec<f32>, String> {
        let decoder = self.resident_descriptor.decoder.clone();
        let layer = self.layer_descriptor(layer_index)?.clone();
        let attention = layer.attention;
        let rope = attention
            .rope
            .clone()
            .ok_or_else(|| format!("Gemma4 descriptor layer {layer_index} is missing RoPE"))?;
        let hidden = decoder.hidden_size;
        let q_width = attention
            .q_heads
            .checked_mul(attention.head_dim)
            .ok_or_else(|| "Gemma resident Q width overflows".to_string())?;
        let kv_width = attention
            .kv_heads
            .checked_mul(attention.head_dim)
            .ok_or_else(|| "Gemma resident KV width overflows".to_string())?;
        let weights = match &self.weights {
            Gemma4WeightStorage::Resident(weights) => weights,
            Gemma4WeightStorage::Streamed(_) => {
                return Err("Gemma resident attention region requires resident weights".into());
            }
        };
        let input_weight = &weights
            .tensor(
                &layer_tensor(layer_index, "input_layernorm.weight"),
                &[hidden],
            )?
            .buffer;
        let q_matrix = &weights
            .tensor(
                &layer_tensor(layer_index, "self_attn.q_proj.weight"),
                &[q_width, hidden],
            )?
            .buffer;
        let o_matrix = &weights
            .tensor(
                &layer_tensor(layer_index, "self_attn.o_proj.weight"),
                &[hidden, q_width],
            )?
            .buffer;
        let q_norm_weight = &weights
            .tensor(
                &layer_tensor(layer_index, "self_attn.q_norm.weight"),
                &[attention.head_dim],
            )?
            .buffer;
        let post_weight = &weights
            .tensor(
                &layer_tensor(layer_index, "post_attention_layernorm.weight"),
                &[hidden],
            )?
            .buffer;
        let shared_source = match attention.kv_cache {
            ResidentKvCacheMode::Own => None,
            ResidentKvCacheMode::SharedFrom { source_layer_index } => Some(source_layer_index),
            ResidentKvCacheMode::LinearState => {
                return Err("Gemma resident attention does not support linear state".into());
            }
        };
        let (k_matrix, v_matrix, k_norm_weight) = if shared_source.is_none() {
            (
                Some(
                    &weights
                        .tensor(
                            &layer_tensor(layer_index, "self_attn.k_proj.weight"),
                            &[kv_width, hidden],
                        )?
                        .buffer,
                ),
                Some(
                    &weights
                        .tensor(
                            &layer_tensor(layer_index, "self_attn.v_proj.weight"),
                            &[kv_width, hidden],
                        )?
                        .buffer,
                ),
                Some(
                    &weights
                        .tensor(
                            &layer_tensor(layer_index, "self_attn.k_norm.weight"),
                            &[attention.head_dim],
                        )?
                        .buffer,
                ),
            )
        } else {
            (None, None, None)
        };
        let caches = match &mut self.caches {
            Gemma4KvStorage::Device(caches) => caches,
            Gemma4KvStorage::Host(_) => {
                return Err("Gemma resident attention region requires device KV".into());
            }
        };
        if let Some(source) = shared_source {
            let cache = caches.cache(source)?;
            self.matvec.attention_norm_residual_resident(
                hidden_states,
                input_weight,
                q_matrix,
                None,
                None,
                o_matrix,
                q_norm_weight,
                None,
                post_weight,
                hidden,
                attention.q_heads,
                attention.kv_heads,
                attention.head_dim,
                &rope,
                position,
                decoder.rms_norm_epsilon,
                None,
                Some(cache),
            )
        } else {
            let cache = caches.cache_mut(layer_index)?;
            self.matvec.attention_norm_residual_resident(
                hidden_states,
                input_weight,
                q_matrix,
                k_matrix,
                v_matrix,
                o_matrix,
                q_norm_weight,
                k_norm_weight,
                post_weight,
                hidden,
                attention.q_heads,
                attention.kv_heads,
                attention.head_dim,
                &rope,
                position,
                decoder.rms_norm_epsilon,
                Some(cache),
                None,
            )
        }
    }

    fn forward_attention(
        &mut self,
        layer_index: usize,
        hidden_states: &[f32],
        position: usize,
    ) -> Result<Vec<f32>, String> {
        let decoder = self.resident_descriptor.decoder.clone();
        let layer = self.layer_descriptor(layer_index)?.clone();
        let attention = layer.attention;
        let layer_kind = attention.kind;
        if matches!(layer_kind, DecoderLayerKind::LinearAttention) {
            return Err("Gemma4TextExecutor does not implement linear attention".into());
        }
        let head_dim = attention.head_dim;
        let q_heads = attention.q_heads;
        let kv_heads = attention.kv_heads;
        let rope = attention
            .rope
            .clone()
            .ok_or_else(|| format!("Gemma4 descriptor layer {layer_index} is missing RoPE"))?;
        let q_width = q_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "Gemma4 q width overflows".to_string())?;
        let kv_width = kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "Gemma4 KV width overflows".to_string())?;
        let hidden = decoder.hidden_size;
        let q_raw = self.matmul_named(
            &layer_tensor(layer_index, "self_attn.q_proj.weight"),
            q_width,
            hidden,
            hidden_states,
        )?;
        let q_norm_weight = self.read_weight_vector(
            &layer_tensor(layer_index, "self_attn.q_norm.weight"),
            head_dim,
        )?;
        let mut query = rms_norm_heads(
            &q_raw,
            q_heads,
            head_dim,
            Some(&q_norm_weight),
            decoder.rms_norm_epsilon,
        )?;
        self.matvec
            .validate_gemma_proportional_rope(&query, q_heads, head_dim, &rope, position)?;
        apply_gemma4_rope_in_place(&mut query, q_heads, head_dim, &rope, position)?;

        let shared_source_layer = match (attention.kv_cache, self.kv_sharing_mode) {
            (
                ResidentKvCacheMode::SharedFrom { source_layer_index },
                Gemma4KvSharingMode::SourceCache,
            ) => Some(source_layer_index),
            (ResidentKvCacheMode::Own, _) => None,
            (ResidentKvCacheMode::SharedFrom { .. }, Gemma4KvSharingMode::ReprojectPhysical) => {
                None
            }
            (ResidentKvCacheMode::LinearState, _) => {
                return Err("Gemma4TextExecutor cannot use linear-attention state".into());
            }
        };
        let source_kv = if shared_source_layer.is_some() {
            None
        } else {
            let key_raw = self.matmul_named(
                &layer_tensor(layer_index, "self_attn.k_proj.weight"),
                kv_width,
                hidden,
                hidden_states,
            )?;
            let value_raw = self.matmul_named(
                &layer_tensor(layer_index, "self_attn.v_proj.weight"),
                kv_width,
                hidden,
                hidden_states,
            )?;
            let k_norm_weight = self.read_weight_vector(
                &layer_tensor(layer_index, "self_attn.k_norm.weight"),
                head_dim,
            )?;
            let mut key = rms_norm_heads(
                &key_raw,
                kv_heads,
                head_dim,
                Some(&k_norm_weight),
                decoder.rms_norm_epsilon,
            )?;
            self.matvec
                .validate_gemma_proportional_rope(&key, kv_heads, head_dim, &rope, position)?;
            apply_gemma4_rope_in_place(&mut key, kv_heads, head_dim, &rope, position)?;
            let value = rms_norm_heads(
                &value_raw,
                kv_heads,
                head_dim,
                None,
                decoder.rms_norm_epsilon,
            )?;
            Some((key, value))
        };
        let attention_output = match &mut self.caches {
            Gemma4KvStorage::Host(caches) => {
                let cache = if let Some(source_layer) = shared_source_layer {
                    caches
                        .per_layer
                        .get(source_layer)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            format!(
                                "Gemma4 layer {layer_index} needs shared {} K/V from source layer {source_layer} before it ran",
                                layer_kind.as_str()
                            )
                        })?
                } else {
                    let (key, value) = source_kv.as_ref().ok_or_else(|| {
                        format!("Gemma4 layer {layer_index} has no freshly projected K/V")
                    })?;
                    let layer_cache = caches
                        .per_layer
                        .get_mut(layer_index)
                        .ok_or_else(|| format!("Gemma4 KV cache has no layer {layer_index}"))?
                        .get_or_insert_with(KvSequence::default);
                    if let Some(window) = attention.sliding_window {
                        let retained = window.checked_sub(1).ok_or_else(|| {
                            "Gemma4 descriptor sliding window must be nonzero".to_string()
                        })?;
                        // Retain W-1 historical rows before appending this token.  After
                        // append the source cache has W rows and is deliberately left
                        // intact for later shared local-attention layers of this token.
                        layer_cache.retain_last(retained)?;
                    }
                    layer_cache.append(key, value)?;
                    layer_cache
                };
                causal_attention(
                    &query,
                    cache,
                    q_heads,
                    kv_heads,
                    head_dim,
                    attention.sliding_window,
                )?
            }
            Gemma4KvStorage::Device(caches) => {
                if let Some(source_layer) = shared_source_layer {
                    let cache = caches.cache(source_layer)?;
                    self.matvec
                        .device_attention(cache, &query, q_heads, kv_heads, head_dim)?
                } else {
                    let (key, value) = source_kv.as_ref().ok_or_else(|| {
                        format!("Gemma4 layer {layer_index} has no freshly projected K/V")
                    })?;
                    let cache = caches.cache_mut(layer_index)?;
                    self.matvec.append_device_kv(cache, key, value)?;
                    let cache = caches.cache(layer_index)?;
                    self.matvec
                        .device_attention(cache, &query, q_heads, kv_heads, head_dim)?
                }
            }
        };
        self.matmul_named(
            &layer_tensor(layer_index, "self_attn.o_proj.weight"),
            hidden,
            q_width,
            &attention_output,
        )
    }

    fn forward_mlp(&mut self, layer_index: usize, input: &[f32]) -> Result<Vec<f32>, String> {
        let layer = self.layer_descriptor(layer_index)?.clone();
        let intermediate = match layer.mlp {
            ResidentMlpDescriptor::Dense {
                intermediate_size, ..
            } => intermediate_size,
            ResidentMlpDescriptor::MoE { .. } => {
                return Err(format!(
                    "Gemma4 resident executor cannot execute MoE MLP at layer {layer_index}"
                ));
            }
        };
        let hidden = self.resident_descriptor.decoder.hidden_size;
        if let Gemma4WeightStorage::Resident(weights) = &self.weights {
            let gate = weights.tensor(
                &layer_tensor(layer_index, "mlp.gate_proj.weight"),
                &[intermediate, hidden],
            )?;
            let up = weights.tensor(
                &layer_tensor(layer_index, "mlp.up_proj.weight"),
                &[intermediate, hidden],
            )?;
            let down = weights.tensor(
                &layer_tensor(layer_index, "mlp.down_proj.weight"),
                &[hidden, intermediate],
            )?;
            return self.matvec.dense_mlp_resident(
                &gate.buffer,
                &up.buffer,
                &down.buffer,
                hidden,
                intermediate,
                input,
            );
        }
        let gate = self.matmul_named(
            &layer_tensor(layer_index, "mlp.gate_proj.weight"),
            intermediate,
            hidden,
            input,
        )?;
        let up = self.matmul_named(
            &layer_tensor(layer_index, "mlp.up_proj.weight"),
            intermediate,
            hidden,
            input,
        )?;
        let mut activated = gelu_pytorch_tanh(&gate)?;
        multiply_in_place(&mut activated, &up, "Gemma4 gated MLP product")?;
        self.matmul_named(
            &layer_tensor(layer_index, "mlp.down_proj.weight"),
            hidden,
            intermediate,
            &activated,
        )
    }

    fn project_tied_logits(&mut self, final_hidden: &[f32]) -> Result<Vec<f32>, String> {
        let output = self.resident_descriptor.output.clone();
        if !output.tied_to_embedding || !self.resident_descriptor.embedding.tied_to_output {
            return Err("Gemma4 resident descriptor does not tie embedding and output head".into());
        }
        let cap = output.logit_soft_cap.ok_or_else(|| {
            "Gemma4 resident descriptor is missing final-logit soft-cap".to_string()
        })?;
        let mut logits = self.matmul_named(
            GEMMA4_TEXT_EMBED_TOKENS,
            self.resident_descriptor.decoder.vocab_size,
            self.resident_descriptor.decoder.hidden_size,
            final_hidden,
        )?;
        for value in &mut logits {
            *value = (*value / cap).tanh() * cap;
        }
        finite_slice(&logits, "Gemma4 final soft-capped logits")?;
        Ok(logits)
    }

    fn read_weight_row(
        &mut self,
        name: &str,
        rows: usize,
        columns: usize,
        row_index: usize,
    ) -> Result<Vec<f32>, String> {
        match &mut self.weights {
            Gemma4WeightStorage::Streamed(weights) => {
                weights.read_bf16_row(name, rows, columns, row_index)
            }
            Gemma4WeightStorage::Resident(weights) => {
                let tensor = weights.tensor(name, &[rows, columns])?;
                self.matvec
                    .bf16_row_resident(&tensor.buffer, rows, columns, row_index)
            }
        }
    }

    fn read_weight_vector(&mut self, name: &str, elements: usize) -> Result<Vec<f32>, String> {
        match &mut self.weights {
            Gemma4WeightStorage::Streamed(weights) => weights.read_bf16_vector(name, elements),
            Gemma4WeightStorage::Resident(weights) => {
                let tensor = weights.tensor(name, &[elements])?;
                self.matvec
                    .bf16_row_resident(&tensor.buffer, 1, elements, 0)
            }
        }
    }

    fn matmul_named(
        &mut self,
        name: &str,
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        finite_slice(input, &format!("Gemma4 matvec input {name}"))?;
        match &mut self.weights {
            Gemma4WeightStorage::Streamed(weights) => {
                let matrix = weights.read_bf16(name, &[rows, columns])?;
                self.matvec.matvec(&matrix, rows, columns, input)
            }
            Gemma4WeightStorage::Resident(weights) => {
                let tensor = weights.tensor(name, &[rows, columns])?;
                self.matvec
                    .matvec_resident(&tensor.buffer, rows, columns, input)
            }
        }
    }
}

#[derive(Debug)]
struct TokenForward {
    embedding: Vec<f32>,
    layer_outputs: Vec<Vec<f32>>,
    final_norm: Vec<f32>,
}

fn validate_checkpoint_contract(
    weights: &SafeTensorReader,
    descriptor: &ResidentModelDescriptor,
) -> Result<(), String> {
    descriptor.require_gemma4_resident_bf16()?;
    let hidden = descriptor.decoder.hidden_size;
    let layers = descriptor.layers.len();
    let ple = descriptor
        .layers
        .first()
        .and_then(|layer| layer.per_layer_embedding.as_ref())
        .ok_or_else(|| "Gemma4 descriptor has no PLE contract".to_string())?;
    let ple_dim = ple.input_size;
    let packed_ple = layers
        .checked_mul(ple_dim)
        .ok_or_else(|| "Gemma4 PLE packed width overflows".to_string())?;
    weights.require_bf16_shape(
        GEMMA4_TEXT_EMBED_TOKENS,
        &[descriptor.decoder.vocab_size, hidden],
    )?;
    weights.require_bf16_shape(
        GEMMA4_TEXT_EMBED_TOKENS_PER_LAYER,
        &[ple.vocabulary_size, packed_ple],
    )?;
    weights.require_bf16_shape(
        GEMMA4_TEXT_PER_LAYER_MODEL_PROJECTION,
        &[packed_ple, hidden],
    )?;
    weights.require_bf16_shape(GEMMA4_TEXT_PER_LAYER_PROJECTION_NORM, &[ple_dim])?;
    weights.require_bf16_shape(GEMMA4_TEXT_FINAL_NORM, &[hidden])?;

    for layer in &descriptor.layers {
        let layer_index = layer.layer_index;
        let attention = &layer.attention;
        if matches!(attention.kind, DecoderLayerKind::LinearAttention) {
            return Err("Gemma4 checkpoint has unsupported linear-attention layer".into());
        }
        if layer
            .per_layer_embedding
            .as_ref()
            .filter(|candidate| *candidate == ple)
            .is_none()
        {
            return Err(format!(
                "Gemma4 descriptor layer {layer_index} has a PLE contract different from layer 0"
            ));
        }
        let q_width = attention
            .q_heads
            .checked_mul(attention.head_dim)
            .ok_or_else(|| "Gemma4 q width overflows".to_string())?;
        let kv_width = attention
            .kv_heads
            .checked_mul(attention.value_dim)
            .ok_or_else(|| "Gemma4 KV width overflows".to_string())?;
        for norm in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "post_per_layer_input_norm.weight",
        ] {
            weights.require_bf16_shape(&layer_tensor(layer_index, norm), &[hidden])?;
        }
        weights.require_bf16_shape(&layer_tensor(layer_index, "layer_scalar"), &[1])?;
        weights.require_bf16_shape(
            &layer_tensor(layer_index, "self_attn.q_proj.weight"),
            &[q_width, hidden],
        )?;
        weights.require_bf16_shape(
            &layer_tensor(layer_index, "self_attn.q_norm.weight"),
            &[attention.head_dim],
        )?;
        weights.require_bf16_shape(
            &layer_tensor(layer_index, "self_attn.o_proj.weight"),
            &[hidden, q_width],
        )?;
        // HF ignores the physical K/V tensors of shared layers.  Require the
        // tensors only for layers whose descriptor says they own K/V.
        if matches!(attention.kv_cache, ResidentKvCacheMode::Own) {
            weights.require_bf16_shape(
                &layer_tensor(layer_index, "self_attn.k_proj.weight"),
                &[kv_width, hidden],
            )?;
            weights.require_bf16_shape(
                &layer_tensor(layer_index, "self_attn.v_proj.weight"),
                &[kv_width, hidden],
            )?;
            weights.require_bf16_shape(
                &layer_tensor(layer_index, "self_attn.k_norm.weight"),
                &[attention.head_dim],
            )?;
        }
        let intermediate = match &layer.mlp {
            ResidentMlpDescriptor::Dense {
                intermediate_size, ..
            } => *intermediate_size,
            ResidentMlpDescriptor::MoE { .. } => {
                return Err(format!(
                    "Gemma4 descriptor layer {layer_index} unexpectedly selected an MoE MLP"
                ));
            }
        };
        weights.require_bf16_shape(
            &layer_tensor(layer_index, "mlp.gate_proj.weight"),
            &[intermediate, hidden],
        )?;
        weights.require_bf16_shape(
            &layer_tensor(layer_index, "mlp.up_proj.weight"),
            &[intermediate, hidden],
        )?;
        weights.require_bf16_shape(
            &layer_tensor(layer_index, "mlp.down_proj.weight"),
            &[hidden, intermediate],
        )?;
        weights.require_bf16_shape(
            &layer_tensor(layer_index, "per_layer_input_gate.weight"),
            &[ple_dim, hidden],
        )?;
        weights.require_bf16_shape(
            &layer_tensor(layer_index, "per_layer_projection.weight"),
            &[hidden, ple_dim],
        )?;
    }
    Ok(())
}

fn layer_tensor(layer_index: usize, suffix: &str) -> String {
    format!("{GEMMA4_TEXT_WEIGHT_PREFIX}layers.{layer_index}.{suffix}")
}

fn bf16_bytes_to_f32(bytes: &[u8], label: &str) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<u16>()) {
        return Err(format!("{label} BF16 payload has an odd byte length"));
    }
    let values = bytes
        .chunks_exact(std::mem::size_of::<u16>())
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f32::from_bits(u32::from(bits) << 16)
        })
        .collect::<Vec<_>>();
    finite_slice(&values, label)?;
    Ok(values)
}

fn finite_slice(values: &[f32], label: &str) -> Result<(), String> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "{label} contains non-finite value at {index}: {value}"
        ));
    }
    Ok(())
}

fn scale_in_place(values: &mut [f32], scale: f32, label: &str) -> Result<(), String> {
    if !scale.is_finite() {
        return Err(format!("{label} is non-finite: {scale}"));
    }
    for value in values.iter_mut() {
        *value *= scale;
    }
    finite_slice(values, label)
}

fn add_vectors(left: &[f32], right: &[f32], label: &str) -> Result<Vec<f32>, String> {
    if left.len() != right.len() {
        return Err(format!(
            "{label} width mismatch: left={} right={}",
            left.len(),
            right.len()
        ));
    }
    let output = left
        .iter()
        .zip(right)
        .map(|(left, right)| left + right)
        .collect::<Vec<_>>();
    finite_slice(&output, label)?;
    Ok(output)
}

fn multiply_in_place(values: &mut [f32], other: &[f32], label: &str) -> Result<(), String> {
    if values.len() != other.len() {
        return Err(format!(
            "{label} width mismatch: left={} right={}",
            values.len(),
            other.len()
        ));
    }
    for (value, other) in values.iter_mut().zip(other) {
        *value *= *other;
    }
    finite_slice(values, label)
}

fn rms_norm(values: &[f32], weight: Option<&[f32]>, epsilon: f32) -> Result<Vec<f32>, String> {
    if values.is_empty() || !epsilon.is_finite() || epsilon <= 0.0 {
        return Err("Gemma4 RMSNorm needs nonempty input and finite positive epsilon".into());
    }
    if let Some(weight) = weight
        && weight.len() != values.len()
    {
        return Err(format!(
            "Gemma4 RMSNorm weight width mismatch: values={} weight={}",
            values.len(),
            weight.len()
        ));
    }
    finite_slice(values, "Gemma4 RMSNorm input")?;
    if let Some(weight) = weight {
        finite_slice(weight, "Gemma4 RMSNorm weight")?;
    }
    let mut mean_squared = 0.0_f32;
    for value in values {
        mean_squared += value * value;
    }
    mean_squared = mean_squared / values.len() as f32 + epsilon;
    let inverse = mean_squared.powf(-0.5);
    if !inverse.is_finite() {
        return Err("Gemma4 RMSNorm inverse RMS is non-finite".into());
    }
    let output: Vec<f32> = match weight {
        Some(weight) => values
            .iter()
            .zip(weight)
            .map(|(value, weight)| value * inverse * weight)
            .collect(),
        None => values.iter().map(|value| value * inverse).collect(),
    };
    finite_slice(&output, "Gemma4 RMSNorm output")?;
    Ok(output)
}

fn rms_norm_heads(
    values: &[f32],
    heads: usize,
    head_dim: usize,
    weight: Option<&[f32]>,
    epsilon: f32,
) -> Result<Vec<f32>, String> {
    let expected = heads
        .checked_mul(head_dim)
        .ok_or_else(|| "Gemma4 RMSNorm head shape overflows".to_string())?;
    if values.len() != expected {
        return Err(format!(
            "Gemma4 head RMSNorm value width mismatch: expected {expected}, got {}",
            values.len()
        ));
    }
    if let Some(weight) = weight
        && weight.len() != head_dim
    {
        return Err(format!(
            "Gemma4 head RMSNorm weight width mismatch: expected {head_dim}, got {}",
            weight.len()
        ));
    }
    let mut output = Vec::with_capacity(expected);
    for head in values.chunks_exact(head_dim) {
        output.extend_from_slice(&rms_norm(head, weight, epsilon)?);
    }
    Ok(output)
}

fn gelu_pytorch_tanh(values: &[f32]) -> Result<Vec<f32>, String> {
    finite_slice(values, "Gemma4 GELU input")?;
    // Match Transformers' GELUTanh.forward operation order and its literal
    // 0.7978845608 coefficient, rather than substituting a mathematically
    // equivalent sqrt(2/pi) expression with a different F32 rounding path.
    let coefficient = 0.797_884_560_8_f32;
    let output = values
        .iter()
        .map(|value| {
            0.5 * value * (1.0 + (value * coefficient * (1.0 + 0.044_715 * value * value)).tanh())
        })
        .collect::<Vec<_>>();
    finite_slice(&output, "Gemma4 GELU output")?;
    Ok(output)
}

fn apply_gemma4_rope_in_place(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    rope: &ResidentRopeDescriptor,
    position: usize,
) -> Result<(), String> {
    if !head_dim.is_multiple_of(2) {
        return Err(format!("Gemma4 RoPE head_dim must be even, got {head_dim}"));
    }
    let expected = heads
        .checked_mul(head_dim)
        .ok_or_else(|| "Gemma4 RoPE shape overflows".to_string())?;
    if values.len() != expected {
        return Err(format!(
            "Gemma4 RoPE input width mismatch: expected {expected}, got {}",
            values.len()
        ));
    }
    let half = head_dim / 2;
    let active_pairs = match rope.kind {
        ResidentRopeKind::Default => rope.rotary_dim.unwrap_or(head_dim) / 2,
        ResidentRopeKind::Proportional => {
            let partial = rope.partial_rotary_factor.unwrap_or(1.0);
            ((partial * head_dim as f32) / 2.0).floor() as usize
        }
        ResidentRopeKind::Mrope => {
            return Err("Gemma4 resident executor does not implement mRoPE".into());
        }
    };
    if active_pairs > half {
        return Err(format!(
            "Gemma4 RoPE active pair count {active_pairs} exceeds head half-width {half}"
        ));
    }
    let mut cosine = Vec::with_capacity(half);
    let mut sine = Vec::with_capacity(half);
    for pair in 0..half {
        let inverse_frequency = if pair < active_pairs {
            let exponent = (2 * pair) as f32 / head_dim as f32;
            rope.theta.powf(exponent).recip()
        } else {
            0.0
        };
        let angle = inverse_frequency * position as f32;
        cosine.push(angle.cos());
        sine.push(angle.sin());
    }
    for head in values.chunks_exact_mut(head_dim) {
        for pair in 0..half {
            let first = head[pair];
            let second = head[half + pair];
            head[pair] = first * cosine[pair] - second * sine[pair];
            head[half + pair] = second * cosine[pair] + first * sine[pair];
        }
    }
    finite_slice(values, "Gemma4 RoPE output")
}

fn causal_attention(
    query: &[f32],
    cache: &KvSequence,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
) -> Result<Vec<f32>, String> {
    if q_heads == 0 || kv_heads == 0 || head_dim == 0 || !q_heads.is_multiple_of(kv_heads) {
        return Err("Gemma4 attention has invalid head geometry".into());
    }
    let expected_q_width = q_heads
        .checked_mul(head_dim)
        .ok_or_else(|| "Gemma4 attention query width overflows".to_string())?;
    let expected_kv_width = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| "Gemma4 attention KV width overflows".to_string())?;
    if query.len() != expected_q_width || cache.width != expected_kv_width {
        return Err(format!(
            "Gemma4 attention width mismatch: query={} expected_q={} cache={} expected_kv={}",
            query.len(),
            expected_q_width,
            cache.width,
            expected_kv_width
        ));
    }
    let cache_len = cache.len()?;
    if cache_len == 0 {
        return Err("Gemma4 attention has no causal KV entries".into());
    }
    let first_key = match sliding_window {
        Some(window) => {
            if window == 0 {
                return Err("Gemma4 sliding attention window must be nonzero".into());
            }
            cache_len.saturating_sub(window)
        }
        None => 0,
    };
    let key_count = cache_len
        .checked_sub(first_key)
        .ok_or_else(|| "Gemma4 attention key range underflow".to_string())?;
    let groups = q_heads / kv_heads;
    let mut output = vec![0.0_f32; expected_q_width];
    for q_head in 0..q_heads {
        let query_head = &query[q_head * head_dim..(q_head + 1) * head_dim];
        let kv_head = q_head / groups;
        let mut scores = Vec::with_capacity(key_count);
        let mut maximum = f32::NEG_INFINITY;
        for key_index in first_key..cache_len {
            let base = key_index
                .checked_mul(cache.width)
                .and_then(|offset| offset.checked_add(kv_head * head_dim))
                .ok_or_else(|| "Gemma4 attention key offset overflows".to_string())?;
            let mut score = 0.0_f32;
            for channel in 0..head_dim {
                score += query_head[channel] * cache.key[base + channel];
            }
            maximum = maximum.max(score);
            scores.push(score);
        }
        let mut denominator = 0.0_f32;
        for score in &mut scores {
            *score = (*score - maximum).exp();
            denominator += *score;
        }
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err("Gemma4 attention softmax denominator is invalid".into());
        }
        let output_head = &mut output[q_head * head_dim..(q_head + 1) * head_dim];
        for (relative_key, probability) in scores.into_iter().enumerate() {
            let key_index = first_key + relative_key;
            let base = key_index
                .checked_mul(cache.width)
                .and_then(|offset| offset.checked_add(kv_head * head_dim))
                .ok_or_else(|| "Gemma4 attention value offset overflows".to_string())?;
            let probability = probability / denominator;
            for channel in 0..head_dim {
                output_head[channel] += probability * cache.value[base + channel];
            }
        }
    }
    finite_slice(&output, "Gemma4 attention output")?;
    Ok(output)
}

fn top1_from_logits(logits: &[f32]) -> Result<Gemma4TextTop1, String> {
    let Some((&first, rest)) = logits.split_first() else {
        return Err("Gemma4 top-1 needs nonempty logits".into());
    };
    if !first.is_finite() {
        return Err("Gemma4 top-1 first logit is non-finite".into());
    }
    let mut token_id = 0_usize;
    let mut logit = first;
    for (index, value) in rest.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(format!("Gemma4 top-1 logit at {} is non-finite", index + 1));
        }
        // PyTorch argmax returns the first maximal index; retain that tie rule.
        if value > logit {
            token_id = index + 1;
            logit = value;
        }
    }
    Ok(Gemma4TextTop1 {
        token_id: u32::try_from(token_id)
            .map_err(|_| "Gemma4 vocabulary index does not fit u32".to_string())?,
        logit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_conversion_preserves_expected_values() {
        let values = bf16_bytes_to_f32(&[0x80, 0x3f, 0x20, 0xc0], "test").unwrap();
        assert_eq!(values, vec![1.0, -2.5]);
    }

    #[test]
    fn direct_rms_norm_does_not_add_one_to_weights() {
        let output = rms_norm(&[3.0, 4.0], Some(&[2.0, 3.0]), 0.0 + 1e-6).unwrap();
        let inverse = (12.5_f32 + 1e-6).powf(-0.5);
        assert!((output[0] - 3.0 * inverse * 2.0).abs() < 1e-6);
        assert!((output[1] - 4.0 * inverse * 3.0).abs() < 1e-6);
    }

    #[test]
    fn proportional_rope_leaves_unrotated_channels_unchanged() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        let rope = ResidentRopeDescriptor {
            kind: ResidentRopeKind::Proportional,
            theta: 1_000_000.0,
            rotary_dim: None,
            partial_rotary_factor: Some(0.25),
            mrope_interleaved: false,
            mrope_sections: Vec::new(),
        };
        apply_gemma4_rope_in_place(&mut values, 1, 4, &rope, 7).unwrap();
        // 25% of a four-wide head yields zero active pairs under HF's
        // `int(partial * head_dim // 2)` rule.
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn sliding_attention_keeps_only_the_recent_window() {
        let mut cache = KvSequence {
            width: 1,
            key: vec![1.0, 1.0, 1.0],
            value: vec![1.0, 2.0, 5.0],
        };
        cache.append(&[1.0], &[9.0]).unwrap();
        let output = causal_attention(&[1.0], &cache, 1, 1, 1, Some(2)).unwrap();
        assert!(output[0] > 6.0);
        assert!(output[0] < 9.1);
    }

    #[test]
    fn empty_sliding_cache_can_start_without_an_eviction() {
        let mut cache = KvSequence::default();
        cache.retain_last(511).unwrap();
        cache.append(&[1.0], &[2.0]).unwrap();
        assert_eq!(cache.len().unwrap(), 1);
    }
}
