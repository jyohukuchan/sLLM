// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use crate::decode_runner::{
    Qwen3DecoderLayerDecodeBatchOutput, Qwen3DecoderLayerDecodeInputLayout,
    Qwen3DecoderLayerDecodeSequenceView, Qwen3DecoderLayerStackRequestDecodeRunner,
    qwen3_decoder_layer_decode_batch_inputs_from_sequences,
    qwen3_decoder_layer_prefill_input_from_sequence,
};
use crate::decoder::{
    PagedDecodeShape, Qwen3DecoderLayerRuntimeWeights, Qwen3MlpRuntimeWeights,
    Qwen3PostAttentionRuntimeWeights, Qwen3SelfAttnRuntimeShape, Qwen3SelfAttnRuntimeWeights,
    qwen3_self_attn_runtime_shape,
};
use crate::host_bytes::encode_f32_to_bytes;
use crate::loader::{
    PassthroughF32Data, WeightRegistry, effective_rmsnorm_weight_values,
    materialize_selected_aq4_matrix, read_named_passthrough_f32,
};
use crate::scheduler::{RequestId, SchedulerDecodeRequest, SchedulerState};
use crate::sq::{SqFp8Artifact, materialize_named_sq_fp8_tensor_to_runtime_f32};
use ullm_runtime_sys::{RuntimeBuffer, RuntimeContext, RuntimeStream};

pub struct Qwen3PackageDecoderLayerRuntime {
    pub layer_index: usize,
    pub input_norm_tensor: String,
    pub q_tensor: String,
    pub k_tensor: String,
    pub v_tensor: String,
    pub o_tensor: String,
    pub q_norm_tensor: String,
    pub k_norm_tensor: String,
    pub post_norm_tensor: String,
    pub gate_tensor: String,
    pub up_tensor: String,
    pub down_tensor: String,
    pub input_norm: PassthroughF32Data,
    pub q_norm: PassthroughF32Data,
    pub k_norm: PassthroughF32Data,
    pub post_norm: PassthroughF32Data,
    pub weights: Qwen3DecoderLayerRuntimeWeights,
    pub runtime_shape: Qwen3SelfAttnRuntimeShape,
}

pub struct Qwen3PackageModelRuntime {
    pub layers: Vec<Qwen3PackageDecoderLayerRuntime>,
    pub hidden: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub value_dim: usize,
    pub softmax_scale: f32,
    pub mlp_epsilon: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Qwen3PackageSqOverlay<'a> {
    pub artifact: &'a SqFp8Artifact,
    pub row_chunk: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen3PackageModelDecodePlan {
    pub decode_shape: PagedDecodeShape,
    pub q_token_elements: usize,
    pub k_token_elements: usize,
    pub v_token_elements: usize,
    pub attention_elements: usize,
    pub hidden: usize,
}

impl Qwen3PackageModelDecodePlan {
    pub fn from_model(
        model: &Qwen3PackageModelRuntime,
        block_size: usize,
        cache_blocks: usize,
    ) -> Result<Self, String> {
        Self::from_shape(
            model.hidden,
            model.q_heads,
            model.kv_heads,
            model.head_dim,
            model.value_dim,
            block_size,
            cache_blocks,
        )
    }

    pub fn from_shape(
        hidden: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        block_size: usize,
        cache_blocks: usize,
    ) -> Result<Self, String> {
        if hidden == 0 {
            return Err("Qwen3 package model decode plan hidden must be greater than zero".into());
        }
        let decode_shape = PagedDecodeShape {
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
        };
        decode_shape.validate()?;
        Ok(Self {
            decode_shape,
            q_token_elements: decode_shape.q_elements()?,
            k_token_elements: decode_shape.k_token_elements()?,
            v_token_elements: decode_shape.v_token_elements()?,
            attention_elements: decode_shape.output_elements()?,
            hidden,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen3PackageModelStackRequest<'a> {
    pub request_id: RequestId,
    pub block_table: &'a [u32],
}

pub fn qwen3_package_model_stack_runner<'weights, 'requests>(
    model: &'weights Qwen3PackageModelRuntime,
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    decode_plan: Qwen3PackageModelDecodePlan,
    layer_requests: &[Vec<Qwen3PackageModelStackRequest<'requests>>],
) -> Result<Qwen3DecoderLayerStackRequestDecodeRunner<'weights>, String> {
    if layer_requests.len() != model.layers.len() {
        return Err(format!(
            "Qwen3 package model stack runner setup has {} layers but {} layer request sets",
            model.layers.len(),
            layer_requests.len()
        ));
    }

    let mut layer_runner = Qwen3DecoderLayerStackRequestDecodeRunner::new();
    for (layer_position, (layer, requests)) in
        model.layers.iter().zip(layer_requests.iter()).enumerate()
    {
        let stack_layer_index = layer_runner.push_layer();
        if stack_layer_index != layer_position {
            return Err(format!(
                "Qwen3 package model stack layer index {stack_layer_index} did not match layer position {layer_position}"
            ));
        }
        for request in requests {
            layer_runner.insert_request(
                stack_layer_index,
                context,
                stream,
                request.request_id,
                &layer.weights,
                decode_plan.decode_shape,
                request.block_table.to_vec(),
                model.softmax_scale,
                model.mlp_epsilon,
            )?;
        }
    }
    Ok(layer_runner)
}

pub fn qwen3_package_model_run_prefill_step_from_sequence(
    runner: &mut Qwen3DecoderLayerStackRequestDecodeRunner<'_>,
    layer_index: usize,
    stream: &mut RuntimeStream,
    sequence: Qwen3DecoderLayerDecodeSequenceView<'_>,
    timestep: usize,
    decode_plan: Qwen3PackageModelDecodePlan,
    label: &str,
) -> Result<Qwen3DecoderLayerDecodeBatchOutput, String> {
    let input_layout = Qwen3DecoderLayerDecodeInputLayout {
        q_token_elements: decode_plan.q_token_elements,
        k_token_elements: decode_plan.k_token_elements,
        v_token_elements: decode_plan.v_token_elements,
        attention_elements: decode_plan.attention_elements,
        hidden: decode_plan.hidden,
    };
    let input =
        qwen3_decoder_layer_prefill_input_from_sequence(sequence, timestep, input_layout, label)?;
    runner.run_prefill_step(layer_index, stream, input)
}

pub fn qwen3_package_model_run_ready_batch_from_sequences<'a>(
    runner: &mut Qwen3DecoderLayerStackRequestDecodeRunner<'_>,
    stream: &mut RuntimeStream,
    scheduler: &mut SchedulerState,
    ready_batch: &[SchedulerDecodeRequest],
    decode_plan: Qwen3PackageModelDecodePlan,
    layer_sequences: &[&[Qwen3DecoderLayerDecodeSequenceView<'a>]],
    label: &str,
) -> Result<Vec<Vec<Qwen3DecoderLayerDecodeBatchOutput>>, String> {
    let input_layout = Qwen3DecoderLayerDecodeInputLayout {
        q_token_elements: decode_plan.q_token_elements,
        k_token_elements: decode_plan.k_token_elements,
        v_token_elements: decode_plan.v_token_elements,
        attention_elements: decode_plan.attention_elements,
        hidden: decode_plan.hidden,
    };
    let mut layer_inputs = Vec::with_capacity(layer_sequences.len());
    for (layer_position, sequences) in layer_sequences.iter().enumerate() {
        let inputs = qwen3_decoder_layer_decode_batch_inputs_from_sequences(
            ready_batch,
            sequences,
            input_layout,
            &format!("{label} layer {layer_position}"),
        )?;
        layer_inputs.push(inputs);
    }

    let layer_input_refs = layer_inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
    runner.run_ready_batch_across_layers(stream, scheduler, ready_batch, &layer_input_refs)
}

impl Qwen3PackageModelRuntime {
    pub fn load(
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        path: &str,
        chunk_bytes: usize,
        layer_indices: &[usize],
    ) -> Result<Self, String> {
        Self::load_with_sq_overlay(context, stream, path, chunk_bytes, layer_indices, None)
    }

    pub fn load_with_sq_overlay(
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        path: &str,
        chunk_bytes: usize,
        layer_indices: &[usize],
        sq_overlay: Option<&Qwen3PackageSqOverlay<'_>>,
    ) -> Result<Self, String> {
        if layer_indices.is_empty() {
            return Err(
                "Qwen3 package model runtime requires at least one layer index".to_string(),
            );
        }
        let mut unique_layers = BTreeSet::new();
        for &layer_index in layer_indices {
            if !unique_layers.insert(layer_index) {
                return Err(format!(
                    "Qwen3 package model runtime layer index {layer_index} is duplicated"
                ));
            }
        }

        let mut layers = Vec::with_capacity(layer_indices.len());
        for &layer_index in layer_indices {
            layers.push(
                qwen3_package_decoder_layer_runtime_from_package_with_sq_overlay(
                    context,
                    stream,
                    path,
                    chunk_bytes,
                    layer_index,
                    sq_overlay,
                )?,
            );
        }

        let first = layers
            .first()
            .ok_or_else(|| "Qwen3 package model runtime loaded no layers".to_string())?;
        let hidden = first.runtime_shape.hidden;
        let q_heads = first.runtime_shape.q_heads;
        let kv_heads = first.runtime_shape.kv_heads;
        let head_dim = first.runtime_shape.head_dim;
        let value_dim = first.runtime_shape.value_dim;
        for layer in &layers {
            if layer.runtime_shape.hidden != hidden
                || layer.runtime_shape.q_heads != q_heads
                || layer.runtime_shape.kv_heads != kv_heads
                || layer.runtime_shape.head_dim != head_dim
                || layer.runtime_shape.value_dim != value_dim
            {
                return Err(format!(
                    "Qwen3 package model runtime layer {} shape mismatch: hidden={} q_heads={} kv_heads={} head_dim={} value_dim={}",
                    layer.layer_index,
                    layer.runtime_shape.hidden,
                    layer.runtime_shape.q_heads,
                    layer.runtime_shape.kv_heads,
                    layer.runtime_shape.head_dim,
                    layer.runtime_shape.value_dim
                ));
            }
            if layer.q_norm.values.len() != head_dim || layer.k_norm.values.len() != head_dim {
                return Err(format!(
                    "Qwen3 package model runtime layer {} q/k norm length mismatch: q={} k={} head_dim={head_dim}",
                    layer.layer_index,
                    layer.q_norm.values.len(),
                    layer.k_norm.values.len()
                ));
            }
            if layer.input_norm.values.len() != hidden {
                return Err(format!(
                    "Qwen3 package model runtime layer {} input norm length mismatch: input_norm={} hidden={hidden}",
                    layer.layer_index,
                    layer.input_norm.values.len()
                ));
            }
        }

        Ok(Self {
            layers,
            hidden,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale: 1.0_f32 / (head_dim as f32).sqrt(),
            mlp_epsilon: 1e-5_f32,
        })
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn layer_indices(&self) -> Vec<usize> {
        self.layers.iter().map(|layer| layer.layer_index).collect()
    }

    pub fn default_rotary_dim(&self) -> Result<usize, String> {
        let candidate = if self.head_dim >= 4 {
            self.head_dim / 4
        } else {
            self.head_dim
        };
        let rotary_dim = candidate - (candidate % 2);
        if rotary_dim == 0 {
            return Err(format!(
                "default rotary_dim is zero for head_dim={}",
                self.head_dim
            ));
        }
        Ok(rotary_dim)
    }

    pub fn decode_shape(&self, block_size: usize, cache_blocks: usize) -> PagedDecodeShape {
        PagedDecodeShape {
            block_size,
            cache_blocks,
            q_heads: self.q_heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            value_dim: self.value_dim,
        }
    }

    pub fn tensor_names_by_layer<F>(&self, mut select: F) -> Vec<String>
    where
        F: FnMut(&Qwen3PackageDecoderLayerRuntime) -> &str,
    {
        self.layers
            .iter()
            .map(|layer| select(layer).to_string())
            .collect()
    }

    pub fn q_norm_dtypes(&self) -> Vec<String> {
        self.layers
            .iter()
            .map(|layer| layer.q_norm.dtype.clone())
            .collect()
    }

    pub fn input_norm_dtypes(&self) -> Vec<String> {
        self.layers
            .iter()
            .map(|layer| layer.input_norm.dtype.clone())
            .collect()
    }

    pub fn k_norm_dtypes(&self) -> Vec<String> {
        self.layers
            .iter()
            .map(|layer| layer.k_norm.dtype.clone())
            .collect()
    }

    pub fn post_norm_dtypes(&self) -> Vec<String> {
        self.layers
            .iter()
            .map(|layer| layer.post_norm.dtype.clone())
            .collect()
    }

    pub fn intermediates(&self) -> Vec<usize> {
        self.layers
            .iter()
            .map(|layer| layer.weights.post_attention.intermediate)
            .collect()
    }
}

pub fn qwen3_package_decoder_layer_runtime_from_package(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    path: &str,
    chunk_bytes: usize,
    layer_index: usize,
) -> Result<Qwen3PackageDecoderLayerRuntime, String> {
    qwen3_package_decoder_layer_runtime_from_package_with_sq_overlay(
        context,
        stream,
        path,
        chunk_bytes,
        layer_index,
        None,
    )
}

pub fn qwen3_package_decoder_layer_runtime_from_package_with_sq_overlay(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    path: &str,
    chunk_bytes: usize,
    layer_index: usize,
    sq_overlay: Option<&Qwen3PackageSqOverlay<'_>>,
) -> Result<Qwen3PackageDecoderLayerRuntime, String> {
    let input_norm_tensor =
        format!("model.language_model.layers.{layer_index}.input_layernorm.weight");
    let q_tensor = format!("model.language_model.layers.{layer_index}.self_attn.q_proj.weight");
    let k_tensor = format!("model.language_model.layers.{layer_index}.self_attn.k_proj.weight");
    let v_tensor = format!("model.language_model.layers.{layer_index}.self_attn.v_proj.weight");
    let o_tensor = format!("model.language_model.layers.{layer_index}.self_attn.o_proj.weight");
    let q_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.q_norm.weight");
    let k_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.k_norm.weight");
    let post_norm_tensor =
        format!("model.language_model.layers.{layer_index}.post_attention_layernorm.weight");
    let gate_tensor = format!("model.language_model.layers.{layer_index}.mlp.gate_proj.weight");
    let up_tensor = format!("model.language_model.layers.{layer_index}.mlp.up_proj.weight");
    let down_tensor = format!("model.language_model.layers.{layer_index}.mlp.down_proj.weight");

    let mut input_norm = read_named_passthrough_f32(path, &input_norm_tensor, chunk_bytes)?;
    input_norm.values = effective_rmsnorm_weight_values(&input_norm_tensor, &input_norm.values);
    let mut q_norm = read_named_passthrough_f32(path, &q_norm_tensor, chunk_bytes)?;
    q_norm.values = effective_rmsnorm_weight_values(&q_norm_tensor, &q_norm.values);
    let mut k_norm = read_named_passthrough_f32(path, &k_norm_tensor, chunk_bytes)?;
    k_norm.values = effective_rmsnorm_weight_values(&k_norm_tensor, &k_norm.values);
    let mut post_norm = read_named_passthrough_f32(path, &post_norm_tensor, chunk_bytes)?;
    post_norm.values = effective_rmsnorm_weight_values(&post_norm_tensor, &post_norm.values);
    let weights = qwen3_decoder_layer_runtime_weights_from_package_with_sq_overlay(
        context,
        stream,
        path,
        chunk_bytes,
        &q_tensor,
        &k_tensor,
        &v_tensor,
        &o_tensor,
        &q_norm,
        &k_norm,
        &post_norm,
        &gate_tensor,
        &up_tensor,
        &down_tensor,
        sq_overlay,
    )?;
    let runtime_shape = qwen3_self_attn_runtime_shape(&weights.self_attn)?;
    if weights.post_attention.hidden != runtime_shape.hidden {
        return Err(format!(
            "Qwen3 package decoder layer {layer_index} hidden mismatch: self_attn={} post_attention={}",
            runtime_shape.hidden, weights.post_attention.hidden
        ));
    }
    Ok(Qwen3PackageDecoderLayerRuntime {
        layer_index,
        input_norm_tensor,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        q_norm_tensor,
        k_norm_tensor,
        post_norm_tensor,
        gate_tensor,
        up_tensor,
        down_tensor,
        input_norm,
        q_norm,
        k_norm,
        post_norm,
        weights,
        runtime_shape,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn qwen3_self_attn_runtime_weights_from_package(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    path: &str,
    chunk_bytes: usize,
    q_tensor: &str,
    k_tensor: &str,
    v_tensor: &str,
    o_tensor: &str,
    q_norm: &PassthroughF32Data,
    k_norm: &PassthroughF32Data,
) -> Result<Qwen3SelfAttnRuntimeWeights, String> {
    qwen3_self_attn_runtime_weights_from_package_with_sq_overlay(
        context,
        stream,
        path,
        chunk_bytes,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        q_norm,
        k_norm,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen3_self_attn_runtime_weights_from_package_with_sq_overlay(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    path: &str,
    chunk_bytes: usize,
    q_tensor: &str,
    k_tensor: &str,
    v_tensor: &str,
    o_tensor: &str,
    q_norm: &PassthroughF32Data,
    k_norm: &PassthroughF32Data,
    sq_overlay: Option<&Qwen3PackageSqOverlay<'_>>,
) -> Result<Qwen3SelfAttnRuntimeWeights, String> {
    let head_dim = q_norm.values.len();
    if head_dim == 0 || k_norm.values.len() != head_dim {
        return Err(format!(
            "self-attn q/k norm head dims must be nonzero and equal: q_head_dim={} k_head_dim={}",
            head_dim,
            k_norm.values.len()
        ));
    }

    let mut registry = WeightRegistry::new();
    let (q_rows, q_cols, q_matrix) = materialize_package_projection_matrix(
        context,
        stream,
        &mut registry,
        path,
        q_tensor,
        chunk_bytes,
        sq_overlay,
    )?;
    let (k_rows, k_cols, k_matrix) = materialize_package_projection_matrix(
        context,
        stream,
        &mut registry,
        path,
        k_tensor,
        chunk_bytes,
        sq_overlay,
    )?;
    let (v_rows, v_cols, v_matrix) = materialize_package_projection_matrix(
        context,
        stream,
        &mut registry,
        path,
        v_tensor,
        chunk_bytes,
        sq_overlay,
    )?;
    let (o_rows, o_cols, o_matrix) = materialize_package_projection_matrix(
        context,
        stream,
        &mut registry,
        path,
        o_tensor,
        chunk_bytes,
        sq_overlay,
    )?;

    if q_cols != k_cols || q_cols != v_cols {
        return Err(format!(
            "self-attn q/k/v projection hidden sizes differ: q_cols={q_cols}, k_cols={k_cols}, v_cols={v_cols}"
        ));
    }
    if o_rows != q_cols {
        return Err(format!(
            "self-attn o projection output hidden size mismatch: o_rows={o_rows}, q_cols={q_cols}"
        ));
    }
    if k_rows % head_dim != 0 {
        return Err(format!(
            "k rows must be a multiple of head_dim: k_rows={k_rows}, head_dim={head_dim}"
        ));
    }
    let kv_heads = k_rows / head_dim;
    if kv_heads == 0 {
        return Err("kv_heads must be greater than zero".to_string());
    }
    if v_rows % kv_heads != 0 {
        return Err(format!(
            "v rows must be a multiple of kv_heads: v_rows={v_rows}, kv_heads={kv_heads}"
        ));
    }
    let value_dim = v_rows / kv_heads;

    Ok(Qwen3SelfAttnRuntimeWeights {
        q_rows,
        q_cols,
        k_rows,
        v_rows,
        o_rows,
        o_cols,
        head_dim,
        kv_heads,
        value_dim,
        q_matrix,
        k_matrix,
        v_matrix,
        o_matrix,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn qwen3_post_attention_runtime_weights_from_package(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    path: &str,
    chunk_bytes: usize,
    hidden: usize,
    post_norm: &PassthroughF32Data,
    gate_tensor: &str,
    up_tensor: &str,
    down_tensor: &str,
) -> Result<Qwen3PostAttentionRuntimeWeights, String> {
    qwen3_post_attention_runtime_weights_from_package_with_sq_overlay(
        context,
        stream,
        path,
        chunk_bytes,
        hidden,
        post_norm,
        gate_tensor,
        up_tensor,
        down_tensor,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen3_post_attention_runtime_weights_from_package_with_sq_overlay(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    path: &str,
    chunk_bytes: usize,
    hidden: usize,
    post_norm: &PassthroughF32Data,
    gate_tensor: &str,
    up_tensor: &str,
    down_tensor: &str,
    sq_overlay: Option<&Qwen3PackageSqOverlay<'_>>,
) -> Result<Qwen3PostAttentionRuntimeWeights, String> {
    if post_norm.values.len() != hidden {
        return Err(format!(
            "post RMSNorm length must match hidden={hidden}: len={}",
            post_norm.values.len()
        ));
    }
    let post_norm_weight_bytes = encode_f32_to_bytes(&post_norm.values);
    let mut post_norm_weight_buffer = context
        .alloc_buffer(post_norm_weight_bytes.len())
        .map_err(|err| format!("failed to allocate post RMSNorm weight buffer: {err}"))?;
    post_norm_weight_buffer
        .copy_from_host(0, &post_norm_weight_bytes, Some(stream))
        .map_err(|err| format!("failed to copy post RMSNorm weight into runtime buffer: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after post RMSNorm weight copy: {err}"))?;

    let mut registry = WeightRegistry::new();
    let (gate_rows, gate_cols, gate_matrix) = materialize_package_projection_matrix(
        context,
        stream,
        &mut registry,
        path,
        gate_tensor,
        chunk_bytes,
        sq_overlay,
    )?;
    let (up_rows, up_cols, up_matrix) = materialize_package_projection_matrix(
        context,
        stream,
        &mut registry,
        path,
        up_tensor,
        chunk_bytes,
        sq_overlay,
    )?;
    let (down_rows, down_cols, down_matrix) = materialize_package_projection_matrix(
        context,
        stream,
        &mut registry,
        path,
        down_tensor,
        chunk_bytes,
        sq_overlay,
    )?;
    if gate_rows != up_rows || gate_cols != up_cols || gate_cols != hidden {
        return Err(format!(
            "MLP gate/up shape mismatch: gate=[{gate_rows},{gate_cols}] up=[{up_rows},{up_cols}] hidden={hidden}"
        ));
    }
    if down_rows != hidden || down_cols != gate_rows {
        return Err(format!(
            "MLP down shape mismatch: expected [{hidden},{gate_rows}], got [{down_rows},{down_cols}]"
        ));
    }
    let intermediate = gate_rows;

    Ok(Qwen3PostAttentionRuntimeWeights {
        hidden,
        intermediate,
        post_norm_weight: post_norm_weight_buffer,
        mlp: Qwen3MlpRuntimeWeights {
            gate_rows,
            gate_cols,
            gate_matrix,
            up_matrix,
            down_matrix,
        },
    })
}

#[allow(clippy::too_many_arguments)]
pub fn qwen3_decoder_layer_runtime_weights_from_package(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    path: &str,
    chunk_bytes: usize,
    q_tensor: &str,
    k_tensor: &str,
    v_tensor: &str,
    o_tensor: &str,
    q_norm: &PassthroughF32Data,
    k_norm: &PassthroughF32Data,
    post_norm: &PassthroughF32Data,
    gate_tensor: &str,
    up_tensor: &str,
    down_tensor: &str,
) -> Result<Qwen3DecoderLayerRuntimeWeights, String> {
    qwen3_decoder_layer_runtime_weights_from_package_with_sq_overlay(
        context,
        stream,
        path,
        chunk_bytes,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        q_norm,
        k_norm,
        post_norm,
        gate_tensor,
        up_tensor,
        down_tensor,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn qwen3_decoder_layer_runtime_weights_from_package_with_sq_overlay(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    path: &str,
    chunk_bytes: usize,
    q_tensor: &str,
    k_tensor: &str,
    v_tensor: &str,
    o_tensor: &str,
    q_norm: &PassthroughF32Data,
    k_norm: &PassthroughF32Data,
    post_norm: &PassthroughF32Data,
    gate_tensor: &str,
    up_tensor: &str,
    down_tensor: &str,
    sq_overlay: Option<&Qwen3PackageSqOverlay<'_>>,
) -> Result<Qwen3DecoderLayerRuntimeWeights, String> {
    let self_attn = qwen3_self_attn_runtime_weights_from_package_with_sq_overlay(
        context,
        stream,
        path,
        chunk_bytes,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        q_norm,
        k_norm,
        sq_overlay,
    )?;
    let post_attention = qwen3_post_attention_runtime_weights_from_package_with_sq_overlay(
        context,
        stream,
        path,
        chunk_bytes,
        self_attn.q_cols,
        post_norm,
        gate_tensor,
        up_tensor,
        down_tensor,
        sq_overlay,
    )?;

    Ok(Qwen3DecoderLayerRuntimeWeights {
        self_attn,
        post_attention,
    })
}

fn materialize_package_projection_matrix(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    registry: &mut WeightRegistry,
    package_path: &str,
    tensor_name: &str,
    chunk_bytes: usize,
    sq_overlay: Option<&Qwen3PackageSqOverlay<'_>>,
) -> Result<(usize, usize, RuntimeBuffer), String> {
    if let Some(overlay) = sq_overlay {
        match materialize_named_sq_fp8_tensor_to_runtime_f32(
            context,
            stream,
            overlay.artifact,
            tensor_name,
            overlay.row_chunk,
        ) {
            Ok(Some(materialized)) => {
                return Ok((materialized.rows, materialized.cols, materialized.buffer));
            }
            Ok(None) => {}
            Err(err) => {
                return Err(format!(
                    "failed to materialize SQ FP8 overlay tensor {tensor_name}: {err}"
                ));
            }
        }
    }
    materialize_selected_aq4_matrix(
        context,
        stream,
        registry,
        package_path,
        tensor_name,
        chunk_bytes,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::Request;

    fn passthrough(values: Vec<f32>) -> PassthroughF32Data {
        PassthroughF32Data {
            shape: vec![values.len() as u64],
            dtype: "F32".to_string(),
            values,
        }
    }

    #[test]
    fn self_attn_loader_rejects_mismatched_qk_norm_before_package_io() {
        let mut context = RuntimeContext::create(0).expect("create CPU runtime context");
        let mut stream = context.create_stream().expect("create CPU runtime stream");
        let q_norm = passthrough(vec![1.0, 1.0]);
        let k_norm = passthrough(vec![1.0]);

        let err = match qwen3_self_attn_runtime_weights_from_package(
            &mut context,
            &mut stream,
            "/path/that/should/not/be/read",
            1024,
            "q",
            "k",
            "v",
            "o",
            &q_norm,
            &k_norm,
        ) {
            Ok(_) => panic!("mismatched q/k norm must fail before package IO"),
            Err(err) => err,
        };

        assert!(err.contains("q/k norm head dims"));
    }

    #[test]
    fn post_attention_loader_rejects_mismatched_post_norm_before_package_io() {
        let mut context = RuntimeContext::create(0).expect("create CPU runtime context");
        let mut stream = context.create_stream().expect("create CPU runtime stream");
        let post_norm = passthrough(vec![1.0, 1.0]);

        let err = match qwen3_post_attention_runtime_weights_from_package(
            &mut context,
            &mut stream,
            "/path/that/should/not/be/read",
            1024,
            3,
            &post_norm,
            "gate",
            "up",
            "down",
        ) {
            Ok(_) => panic!("mismatched post norm must fail before package IO"),
            Err(err) => err,
        };

        assert!(err.contains("post RMSNorm length"));
    }

    #[test]
    fn package_model_decode_plan_from_shape_derives_token_elements() {
        let plan = Qwen3PackageModelDecodePlan::from_shape(16, 4, 2, 4, 3, 8, 5)
            .expect("decode plan from shape");

        assert_eq!(plan.decode_shape.block_size, 8);
        assert_eq!(plan.decode_shape.cache_blocks, 5);
        assert_eq!(plan.q_token_elements, 16);
        assert_eq!(plan.k_token_elements, 8);
        assert_eq!(plan.v_token_elements, 6);
        assert_eq!(plan.attention_elements, 12);
        assert_eq!(plan.hidden, 16);
    }

    #[test]
    fn package_model_decode_plan_rejects_invalid_shape() {
        let err = match Qwen3PackageModelDecodePlan::from_shape(16, 3, 2, 4, 3, 8, 5) {
            Ok(_) => panic!("non-divisible q/kv heads must fail"),
            Err(err) => err,
        };

        assert!(err.contains("q_heads must be a multiple of kv_heads"));
    }

    #[test]
    fn package_model_stack_runner_rejects_layer_request_count_mismatch_before_layer_access() {
        let mut context = RuntimeContext::create(0).expect("create CPU runtime context");
        let mut stream = context.create_stream().expect("create CPU runtime stream");
        let model = Qwen3PackageModelRuntime {
            layers: Vec::new(),
            hidden: 1,
            q_heads: 1,
            kv_heads: 1,
            head_dim: 1,
            value_dim: 1,
            softmax_scale: 1.0,
            mlp_epsilon: 1e-5,
        };
        let decode_plan = Qwen3PackageModelDecodePlan::from_model(&model, 1, 1)
            .expect("decode plan for empty test model metadata");
        let layer_requests = vec![Vec::new()];

        let err = match qwen3_package_model_stack_runner(
            &model,
            &mut context,
            &mut stream,
            decode_plan,
            &layer_requests,
        ) {
            Ok(_) => panic!("layer request count mismatch must fail before layer access"),
            Err(err) => err,
        };

        assert!(err.contains("0 layers but 1 layer request sets"));
    }

    #[test]
    fn package_model_ready_batch_from_sequences_builds_inputs_before_stack_run() {
        let mut context = RuntimeContext::create(0).expect("create CPU runtime context");
        let mut stream = context.create_stream().expect("create CPU runtime stream");
        let decode_plan = Qwen3PackageModelDecodePlan::from_shape(4, 2, 1, 2, 2, 2, 2)
            .expect("decode plan from small shape");
        let mut scheduler = SchedulerState::with_block_size(2, 2);
        scheduler.enqueue(Request::new(61, 1, 1));
        scheduler
            .pop_prefill_batch_with_allocation(1)
            .expect("allocation should succeed");
        scheduler
            .complete_prefill(RequestId(61))
            .expect("prefill completion should succeed");
        let ready = scheduler
            .ready_decode_batch(1)
            .expect("ready batch should be generated");
        let q = vec![0.125_f32; 2 * decode_plan.q_token_elements];
        let k = vec![0.25_f32; 2 * decode_plan.k_token_elements];
        let v = vec![0.375_f32; 2 * decode_plan.v_token_elements];
        let gate = vec![0.5_f32; 2 * decode_plan.attention_elements];
        let residual = vec![0.625_f32; 2 * decode_plan.hidden];
        let sequence = Qwen3DecoderLayerDecodeSequenceView {
            request_id: RequestId(61),
            q_sequence: &q,
            k_sequence: &k,
            v_sequence: &v,
            output_gate_sequence: Some(&gate),
            residual_sequence: &residual,
        };
        let layer_sequences = vec![vec![sequence]];
        let layer_sequence_refs = layer_sequences
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>();
        let mut runner = Qwen3DecoderLayerStackRequestDecodeRunner::new();

        let err = qwen3_package_model_run_ready_batch_from_sequences(
            &mut runner,
            &mut stream,
            &mut scheduler,
            &ready,
            decode_plan,
            &layer_sequence_refs,
            "package model ready batch test",
        )
        .expect_err("empty stack runner should reject layer input");

        assert!(err.contains("0 layers but 1 layer input batches"), "{err}");
        let active = scheduler
            .active_request(RequestId(61))
            .expect("request should remain active after failed stack run");
        assert_eq!(active.cached_tokens, 1);
        assert_eq!(active.generated_tokens, 0);
    }

    #[test]
    fn package_model_prefill_step_from_sequence_builds_input_before_stack_run() {
        let mut context = RuntimeContext::create(0).expect("create CPU runtime context");
        let mut stream = context.create_stream().expect("create CPU runtime stream");
        let decode_plan = Qwen3PackageModelDecodePlan::from_shape(4, 2, 1, 2, 2, 2, 2)
            .expect("decode plan from small shape");
        let q = vec![0.125_f32; decode_plan.q_token_elements];
        let k = vec![0.25_f32; decode_plan.k_token_elements];
        let v = vec![0.375_f32; decode_plan.v_token_elements];
        let gate = vec![0.5_f32; decode_plan.attention_elements];
        let residual = vec![0.625_f32; decode_plan.hidden];
        let sequence = Qwen3DecoderLayerDecodeSequenceView {
            request_id: RequestId(71),
            q_sequence: &q,
            k_sequence: &k,
            v_sequence: &v,
            output_gate_sequence: Some(&gate),
            residual_sequence: &residual,
        };
        let mut runner = Qwen3DecoderLayerStackRequestDecodeRunner::new();

        let err = qwen3_package_model_run_prefill_step_from_sequence(
            &mut runner,
            0,
            &mut stream,
            sequence,
            0,
            decode_plan,
            "package model prefill test",
        )
        .expect_err("empty stack runner should reject prefill after input construction");

        assert!(
            err.contains("decoder layer index 0 is out of bounds"),
            "{err}"
        );
    }

    #[test]
    fn package_model_runtime_rejects_empty_layer_list_before_package_io() {
        let mut context = RuntimeContext::create(0).expect("create CPU runtime context");
        let mut stream = context.create_stream().expect("create CPU runtime stream");

        let err = match Qwen3PackageModelRuntime::load(
            &mut context,
            &mut stream,
            "/path/that/should/not/be/read",
            1024,
            &[],
        ) {
            Ok(_) => panic!("empty layer list must fail before package IO"),
            Err(err) => err,
        };

        assert!(err.contains("at least one layer index"));
    }

    #[test]
    fn package_model_runtime_rejects_duplicate_layers_before_package_io() {
        let mut context = RuntimeContext::create(0).expect("create CPU runtime context");
        let mut stream = context.create_stream().expect("create CPU runtime stream");

        let err = match Qwen3PackageModelRuntime::load(
            &mut context,
            &mut stream,
            "/path/that/should/not/be/read",
            1024,
            &[3, 3],
        ) {
            Ok(_) => panic!("duplicate layer list must fail before package IO"),
            Err(err) => err,
        };

        assert!(err.contains("duplicated"));
    }
}
