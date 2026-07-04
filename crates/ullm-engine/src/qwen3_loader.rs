// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use crate::decoder::{
    PagedDecodeShape, Qwen3DecoderLayerRuntimeWeights, Qwen3MlpRuntimeWeights,
    Qwen3PostAttentionRuntimeWeights, Qwen3SelfAttnRuntimeShape, Qwen3SelfAttnRuntimeWeights,
    qwen3_self_attn_runtime_shape,
};
use crate::host_bytes::encode_f32_to_bytes;
use crate::loader::{
    PassthroughF32Data, WeightRegistry, materialize_selected_aq4_matrix, read_named_passthrough_f32,
};
use ullm_runtime_sys::{RuntimeContext, RuntimeStream};

pub struct Qwen3PackageDecoderLayerRuntime {
    pub layer_index: usize,
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

impl Qwen3PackageModelRuntime {
    pub fn load(
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        path: &str,
        chunk_bytes: usize,
        layer_indices: &[usize],
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
            layers.push(qwen3_package_decoder_layer_runtime_from_package(
                context,
                stream,
                path,
                chunk_bytes,
                layer_index,
            )?);
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

    let q_norm = read_named_passthrough_f32(path, &q_norm_tensor, chunk_bytes)?;
    let k_norm = read_named_passthrough_f32(path, &k_norm_tensor, chunk_bytes)?;
    let post_norm = read_named_passthrough_f32(path, &post_norm_tensor, chunk_bytes)?;
    let weights = qwen3_decoder_layer_runtime_weights_from_package(
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
    let head_dim = q_norm.values.len();
    if head_dim == 0 || k_norm.values.len() != head_dim {
        return Err(format!(
            "self-attn q/k norm head dims must be nonzero and equal: q_head_dim={} k_head_dim={}",
            head_dim,
            k_norm.values.len()
        ));
    }

    let mut registry = WeightRegistry::new();
    let (q_rows, q_cols, q_matrix) = materialize_selected_aq4_matrix(
        context,
        stream,
        &mut registry,
        path,
        q_tensor,
        chunk_bytes,
    )?;
    let (k_rows, k_cols, k_matrix) = materialize_selected_aq4_matrix(
        context,
        stream,
        &mut registry,
        path,
        k_tensor,
        chunk_bytes,
    )?;
    let (v_rows, v_cols, v_matrix) = materialize_selected_aq4_matrix(
        context,
        stream,
        &mut registry,
        path,
        v_tensor,
        chunk_bytes,
    )?;
    let (o_rows, o_cols, o_matrix) = materialize_selected_aq4_matrix(
        context,
        stream,
        &mut registry,
        path,
        o_tensor,
        chunk_bytes,
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
    let (gate_rows, gate_cols, gate_matrix) = materialize_selected_aq4_matrix(
        context,
        stream,
        &mut registry,
        path,
        gate_tensor,
        chunk_bytes,
    )?;
    let (up_rows, up_cols, up_matrix) = materialize_selected_aq4_matrix(
        context,
        stream,
        &mut registry,
        path,
        up_tensor,
        chunk_bytes,
    )?;
    let (down_rows, down_cols, down_matrix) = materialize_selected_aq4_matrix(
        context,
        stream,
        &mut registry,
        path,
        down_tensor,
        chunk_bytes,
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
    let self_attn = qwen3_self_attn_runtime_weights_from_package(
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
    )?;
    let post_attention = qwen3_post_attention_runtime_weights_from_package(
        context,
        stream,
        path,
        chunk_bytes,
        self_attn.q_cols,
        post_norm,
        gate_tensor,
        up_tensor,
        down_tensor,
    )?;

    Ok(Qwen3DecoderLayerRuntimeWeights {
        self_attn,
        post_attention,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
