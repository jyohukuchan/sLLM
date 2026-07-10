// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Narrow decoder-step state for runtime paged K/V cache writes and paged decode.
//!
//! Logical token position `t` maps to physical cache slot
//! `block_table[t / block_size] * block_size + (t % block_size)`.
//! `read_cache_to_host` returns the physical cache layout, not logical order.

use ullm_runtime_sys::{RuntimeBuffer, RuntimeContext, RuntimeStream};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedDecodeShape {
    pub block_size: usize,
    pub cache_blocks: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub value_dim: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PagedKvCacheReadback {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PagedDecodeStepOutput {
    pub cache_position: usize,
    pub cache_len: usize,
    pub output: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3SelfAttnDecodeStepOutput {
    pub cache_position: usize,
    pub cache_len: usize,
    pub attention_output: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3SelfAttnBlockStepOutput {
    pub cache_position: usize,
    pub cache_len: usize,
    pub attention_output: Vec<f32>,
    pub attention_projection_input: Vec<f32>,
    pub projected_output: Vec<f32>,
    pub block_output: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3DecoderLayerStepOutput {
    pub cache_position: usize,
    pub cache_len: usize,
    pub attention_output: Vec<f32>,
    pub attention_projection_input: Vec<f32>,
    pub projected_output: Vec<f32>,
    pub block_output: Vec<f32>,
    pub post_normed: Vec<f32>,
    pub mlp_output: Vec<f32>,
    pub layer_output: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3DecoderLayerOutputStep {
    pub cache_position: usize,
    pub cache_len: usize,
    pub layer_output: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PagedDecodeDeviceStepOutput {
    pub(crate) cache_position: usize,
    pub(crate) cache_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Qwen3SelfAttnBlockDeviceStepOutput {
    cache_position: usize,
    cache_len: usize,
    gated_projection_input: bool,
}

#[derive(Debug)]
pub struct PagedDecodeState {
    shape: PagedDecodeShape,
    block_table: Vec<u32>,
    written_len: usize,
    block_table_buffer: RuntimeBuffer,
    q_buffer: RuntimeBuffer,
    k_token_buffer: RuntimeBuffer,
    v_token_buffer: RuntimeBuffer,
    k_cache_buffer: RuntimeBuffer,
    v_cache_buffer: RuntimeBuffer,
    output_buffer: RuntimeBuffer,
}

#[derive(Debug)]
pub struct Qwen3SelfAttnDecodeState {
    state: PagedDecodeState,
    softmax_scale: f32,
}

#[derive(Debug)]
pub struct Qwen3SelfAttnBlockStepState {
    decode: Qwen3SelfAttnDecodeState,
    hidden: usize,
    attention_elements: usize,
    gate_buffer: RuntimeBuffer,
    projection_input_buffer: RuntimeBuffer,
    projected_buffer: RuntimeBuffer,
    residual_buffer: RuntimeBuffer,
    block_buffer: RuntimeBuffer,
}

#[derive(Debug)]
pub struct Qwen3DecoderLayerStepState {
    block_state: Qwen3SelfAttnBlockStepState,
    intermediate: usize,
    mlp_epsilon: f32,
    post_normed_buffer: RuntimeBuffer,
    gate_buffer: RuntimeBuffer,
    up_buffer: RuntimeBuffer,
    activated_buffer: RuntimeBuffer,
    mlp_output_buffer: RuntimeBuffer,
    layer_output_buffer: RuntimeBuffer,
}

pub struct Qwen3SelfAttnRuntimeWeights {
    pub q_rows: usize,
    pub q_cols: usize,
    pub k_rows: usize,
    pub v_rows: usize,
    pub o_rows: usize,
    pub o_cols: usize,
    pub head_dim: usize,
    pub kv_heads: usize,
    pub value_dim: usize,
    pub q_matrix: RuntimeBuffer,
    pub k_matrix: RuntimeBuffer,
    pub v_matrix: RuntimeBuffer,
    pub o_matrix: RuntimeBuffer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3SelfAttnRuntimeShape {
    pub hidden: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub value_dim: usize,
    pub attention_width: usize,
    pub q_projection_layout: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3SelfAttnProjectedSequence {
    pub q_projected: Vec<f32>,
    pub k_projected: Vec<f32>,
    pub v_projected: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3SelfAttnRuntimePreparedSequence {
    pub q_query: Vec<f32>,
    pub k_projected: Vec<f32>,
    pub q_normed: Vec<f32>,
    pub k_normed: Vec<f32>,
    pub q_rope: Vec<f32>,
    pub k_rope: Vec<f32>,
    pub v_projected: Vec<f32>,
    pub q_gate: Option<Vec<f32>>,
    pub attention_output: Vec<f32>,
    pub shape: Qwen3SelfAttnRuntimeShape,
    pub softmax_scale: f32,
    pub q_projection_layout: &'static str,
    pub q_gate_elements: usize,
    pub output_gate_layout: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3SelfAttnRuntimePreparedSequenceForPagedDecode {
    pub residual_sequence: Vec<f32>,
    pub prepared: Qwen3SelfAttnRuntimePreparedSequence,
    pub paged_k_cache: Vec<f32>,
    pub paged_v_cache: Vec<f32>,
    pub paged_block_table: Vec<u32>,
    pub paged_block_size: usize,
    pub paged_cache_blocks: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3SelfAttnBlockSequenceOutput {
    pub attention_output: Vec<f32>,
    pub attention_projection_input: Vec<f32>,
    pub projected_output: Vec<f32>,
    pub block_output: Vec<f32>,
    pub paged_cache: PagedKvCacheReadback,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3DecoderLayerSequenceOutput {
    pub attention_output: Vec<f32>,
    pub attention_projection_input: Vec<f32>,
    pub projected_output: Vec<f32>,
    pub block_output: Vec<f32>,
    pub post_normed: Vec<f32>,
    pub mlp_output: Vec<f32>,
    pub layer_output: Vec<f32>,
    pub paged_cache: PagedKvCacheReadback,
}

#[allow(clippy::too_many_arguments)]
pub fn qwen3_self_attn_block_sequence_to_host_f32(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    shape: PagedDecodeShape,
    block_table: &[u32],
    hidden: usize,
    softmax_scale: f32,
    o_projection_matrix: &RuntimeBuffer,
    q_sequence: &[f32],
    k_sequence: &[f32],
    v_sequence: &[f32],
    output_gate_sequence: Option<&[f32]>,
    residual_sequence: &[f32],
    sequence_len: usize,
) -> Result<Qwen3SelfAttnBlockSequenceOutput, String> {
    shape.validate()?;
    if sequence_len == 0 {
        return Err("self-attn block sequence length must be greater than zero".to_string());
    }
    if hidden == 0 {
        return Err("self-attn block hidden size must be greater than zero".to_string());
    }
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err(
            "self-attn block softmax_scale must be finite and greater than zero".to_string(),
        );
    }

    let q_token_elements = shape.q_elements()?;
    let k_token_elements = shape.k_token_elements()?;
    let v_token_elements = shape.v_token_elements()?;
    let attention_elements = shape.output_elements()?;

    let expected_q_len = q_token_elements
        .checked_mul(sequence_len)
        .ok_or_else(|| "self-attn block q sequence length overflows".to_string())?;
    let expected_k_len = k_token_elements
        .checked_mul(sequence_len)
        .ok_or_else(|| "self-attn block k sequence length overflows".to_string())?;
    let expected_v_len = v_token_elements
        .checked_mul(sequence_len)
        .ok_or_else(|| "self-attn block v sequence length overflows".to_string())?;
    let expected_residual_len = hidden
        .checked_mul(sequence_len)
        .ok_or_else(|| "self-attn block residual sequence length overflows".to_string())?;
    let expected_attention_len = attention_elements
        .checked_mul(sequence_len)
        .ok_or_else(|| "self-attn block attention sequence length overflows".to_string())?;
    if q_sequence.len() != expected_q_len {
        return Err(format!(
            "self-attn block q sequence length {} does not match sequence_len={sequence_len} q_token_elements={q_token_elements}",
            q_sequence.len()
        ));
    }
    if k_sequence.len() != expected_k_len {
        return Err(format!(
            "self-attn block k sequence length {} does not match sequence_len={sequence_len} k_token_elements={k_token_elements}",
            k_sequence.len()
        ));
    }
    if v_sequence.len() != expected_v_len {
        return Err(format!(
            "self-attn block v sequence length {} does not match sequence_len={sequence_len} v_token_elements={v_token_elements}",
            v_sequence.len()
        ));
    }
    if residual_sequence.len() != expected_residual_len {
        return Err(format!(
            "self-attn block residual sequence length {} does not match sequence_len={sequence_len} hidden={hidden}",
            residual_sequence.len()
        ));
    }
    if let Some(gate_sequence) = output_gate_sequence {
        if gate_sequence.len() != expected_attention_len {
            return Err(format!(
                "self-attn block output gate length {} does not match sequence_len={sequence_len} attention_elements={attention_elements}",
                gate_sequence.len()
            ));
        }
    }

    let mut state = Qwen3SelfAttnBlockStepState::new(
        context,
        stream,
        shape,
        block_table.to_vec(),
        hidden,
        softmax_scale,
    )
    .map_err(|err| format!("failed to create self-attn block sequence state: {err}"))?;

    let mut attention_output = Vec::with_capacity(expected_attention_len);
    let mut attention_projection_input = Vec::with_capacity(expected_attention_len);
    let mut projected_output = Vec::with_capacity(expected_residual_len);
    let mut block_output = Vec::with_capacity(expected_residual_len);

    for timestep in 0..sequence_len {
        let q_start = timestep
            .checked_mul(q_token_elements)
            .ok_or_else(|| "self-attn block q slice start overflows".to_string())?;
        let q_end = q_start
            .checked_add(q_token_elements)
            .ok_or_else(|| "self-attn block q slice end overflows".to_string())?;
        let k_start = timestep
            .checked_mul(k_token_elements)
            .ok_or_else(|| "self-attn block k slice start overflows".to_string())?;
        let k_end = k_start
            .checked_add(k_token_elements)
            .ok_or_else(|| "self-attn block k slice end overflows".to_string())?;
        let v_start = timestep
            .checked_mul(v_token_elements)
            .ok_or_else(|| "self-attn block v slice start overflows".to_string())?;
        let v_end = v_start
            .checked_add(v_token_elements)
            .ok_or_else(|| "self-attn block v slice end overflows".to_string())?;
        let residual_start = timestep
            .checked_mul(hidden)
            .ok_or_else(|| "self-attn block residual slice start overflows".to_string())?;
        let residual_end = residual_start
            .checked_add(hidden)
            .ok_or_else(|| "self-attn block residual slice end overflows".to_string())?;
        let output_gate = output_gate_sequence.map(|gate| {
            let gate_start = timestep * attention_elements;
            let gate_end = gate_start + attention_elements;
            &gate[gate_start..gate_end]
        });

        let step = state
            .step(
                stream,
                o_projection_matrix,
                &q_sequence[q_start..q_end],
                &k_sequence[k_start..k_end],
                &v_sequence[v_start..v_end],
                output_gate,
                &residual_sequence[residual_start..residual_end],
            )
            .map_err(|err| {
                format!("failed to run self-attn block sequence step {timestep}: {err}")
            })?;
        if step.cache_position != timestep {
            return Err(format!(
                "self-attn block sequence step wrote position {}, expected {timestep}",
                step.cache_position
            ));
        }
        if step.cache_len != timestep + 1 {
            return Err(format!(
                "self-attn block sequence step reported cache_len {}, expected {}",
                step.cache_len,
                timestep + 1
            ));
        }

        attention_output.extend_from_slice(&step.attention_output);
        attention_projection_input.extend_from_slice(&step.attention_projection_input);
        projected_output.extend_from_slice(&step.projected_output);
        block_output.extend_from_slice(&step.block_output);
    }

    let paged_cache = state
        .read_cache_to_host(stream)
        .map_err(|err| format!("failed to read self-attn block sequence paged cache: {err}"))?;

    Ok(Qwen3SelfAttnBlockSequenceOutput {
        attention_output,
        attention_projection_input,
        projected_output,
        block_output,
        paged_cache,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn qwen3_decoder_layer_sequence_to_host_f32(
    layer_weights: &Qwen3DecoderLayerRuntimeWeights,
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    shape: PagedDecodeShape,
    block_table: &[u32],
    softmax_scale: f32,
    mlp_epsilon: f32,
    q_sequence: &[f32],
    k_sequence: &[f32],
    v_sequence: &[f32],
    output_gate_sequence: Option<&[f32]>,
    residual_sequence: &[f32],
    sequence_len: usize,
) -> Result<Qwen3DecoderLayerSequenceOutput, String> {
    shape.validate()?;
    validate_qwen3_decoder_layer_decode_shape(layer_weights, shape)?;
    if sequence_len == 0 {
        return Err("decoder layer sequence length must be greater than zero".to_string());
    }
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err("decoder layer softmax_scale must be finite and greater than zero".to_string());
    }
    if !mlp_epsilon.is_finite() || mlp_epsilon <= 0.0 {
        return Err("decoder layer mlp epsilon must be finite and greater than zero".to_string());
    }

    let q_token_elements = shape.q_elements()?;
    let k_token_elements = shape.k_token_elements()?;
    let v_token_elements = shape.v_token_elements()?;
    let attention_elements = shape.output_elements()?;
    let hidden = layer_weights.post_attention.hidden;
    if hidden == 0 {
        return Err("decoder layer hidden size must be greater than zero".to_string());
    }

    let expected_q_len = q_token_elements
        .checked_mul(sequence_len)
        .ok_or_else(|| "decoder layer q sequence length overflows".to_string())?;
    let expected_k_len = k_token_elements
        .checked_mul(sequence_len)
        .ok_or_else(|| "decoder layer k sequence length overflows".to_string())?;
    let expected_v_len = v_token_elements
        .checked_mul(sequence_len)
        .ok_or_else(|| "decoder layer v sequence length overflows".to_string())?;
    let expected_residual_len = hidden
        .checked_mul(sequence_len)
        .ok_or_else(|| "decoder layer residual sequence length overflows".to_string())?;
    let expected_attention_len = attention_elements
        .checked_mul(sequence_len)
        .ok_or_else(|| "decoder layer attention sequence length overflows".to_string())?;
    if q_sequence.len() != expected_q_len {
        return Err(format!(
            "decoder layer q sequence length {} does not match sequence_len={sequence_len} q_token_elements={q_token_elements}",
            q_sequence.len()
        ));
    }
    if k_sequence.len() != expected_k_len {
        return Err(format!(
            "decoder layer k sequence length {} does not match sequence_len={sequence_len} k_token_elements={k_token_elements}",
            k_sequence.len()
        ));
    }
    if v_sequence.len() != expected_v_len {
        return Err(format!(
            "decoder layer v sequence length {} does not match sequence_len={sequence_len} v_token_elements={v_token_elements}",
            v_sequence.len()
        ));
    }
    if residual_sequence.len() != expected_residual_len {
        return Err(format!(
            "decoder layer residual sequence length {} does not match sequence_len={sequence_len} hidden={hidden}",
            residual_sequence.len()
        ));
    }
    if let Some(gate_sequence) = output_gate_sequence {
        if gate_sequence.len() != expected_attention_len {
            return Err(format!(
                "decoder layer output gate length {} does not match sequence_len={sequence_len} attention_elements={attention_elements}",
                gate_sequence.len()
            ));
        }
    }

    let mut layer_runtime = Qwen3DecoderLayerRuntime::new(
        context,
        stream,
        layer_weights,
        shape,
        block_table.to_vec(),
        softmax_scale,
        mlp_epsilon,
    )
    .map_err(|err| format!("failed to create decoder layer sequence state: {err}"))?;

    let mut attention_output = Vec::with_capacity(expected_attention_len);
    let mut attention_projection_input = Vec::with_capacity(expected_attention_len);
    let mut projected_output = Vec::with_capacity(expected_residual_len);
    let mut block_output = Vec::with_capacity(expected_residual_len);
    let mut post_normed = Vec::with_capacity(expected_residual_len);
    let mut mlp_output = Vec::with_capacity(expected_residual_len);
    let mut layer_output = Vec::with_capacity(expected_residual_len);

    for timestep in 0..sequence_len {
        let q_start = timestep
            .checked_mul(q_token_elements)
            .ok_or_else(|| "decoder layer q slice start overflows".to_string())?;
        let q_end = q_start
            .checked_add(q_token_elements)
            .ok_or_else(|| "decoder layer q slice end overflows".to_string())?;
        let k_start = timestep
            .checked_mul(k_token_elements)
            .ok_or_else(|| "decoder layer k slice start overflows".to_string())?;
        let k_end = k_start
            .checked_add(k_token_elements)
            .ok_or_else(|| "decoder layer k slice end overflows".to_string())?;
        let v_start = timestep
            .checked_mul(v_token_elements)
            .ok_or_else(|| "decoder layer v slice start overflows".to_string())?;
        let v_end = v_start
            .checked_add(v_token_elements)
            .ok_or_else(|| "decoder layer v slice end overflows".to_string())?;
        let residual_start = timestep
            .checked_mul(hidden)
            .ok_or_else(|| "decoder layer residual slice start overflows".to_string())?;
        let residual_end = residual_start
            .checked_add(hidden)
            .ok_or_else(|| "decoder layer residual slice end overflows".to_string())?;
        let attention_gate = if let Some(gate) = output_gate_sequence {
            let gate_start = timestep
                .checked_mul(attention_elements)
                .ok_or_else(|| "decoder layer gate slice start overflows".to_string())?;
            let gate_end = gate_start
                .checked_add(attention_elements)
                .ok_or_else(|| "decoder layer gate slice end overflows".to_string())?;
            Some(&gate[gate_start..gate_end])
        } else {
            None
        };

        let step = layer_runtime
            .step(
                stream,
                &q_sequence[q_start..q_end],
                &k_sequence[k_start..k_end],
                &v_sequence[v_start..v_end],
                attention_gate,
                &residual_sequence[residual_start..residual_end],
            )
            .map_err(|err| {
                format!("failed to run decoder layer sequence step {timestep}: {err}")
            })?;
        if step.cache_position != timestep {
            return Err(format!(
                "decoder layer sequence step wrote position {}, expected {timestep}",
                step.cache_position
            ));
        }
        if step.cache_len != timestep + 1 {
            return Err(format!(
                "decoder layer sequence step reported cache_len {}, expected {}",
                step.cache_len,
                timestep + 1
            ));
        }

        attention_output.extend_from_slice(&step.attention_output);
        attention_projection_input.extend_from_slice(&step.attention_projection_input);
        projected_output.extend_from_slice(&step.projected_output);
        block_output.extend_from_slice(&step.block_output);
        post_normed.extend_from_slice(&step.post_normed);
        mlp_output.extend_from_slice(&step.mlp_output);
        layer_output.extend_from_slice(&step.layer_output);
    }

    let paged_cache = layer_runtime
        .read_cache_to_host(stream)
        .map_err(|err| format!("failed to read decoder layer sequence paged cache: {err}"))?;

    Ok(Qwen3DecoderLayerSequenceOutput {
        attention_output,
        attention_projection_input,
        projected_output,
        block_output,
        post_normed,
        mlp_output,
        layer_output,
        paged_cache,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn qwen3_self_attn_prepare_sequence_runtime_f32(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    weights: &Qwen3SelfAttnRuntimeWeights,
    projected: Qwen3SelfAttnProjectedSequence,
    sequence_len: usize,
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
    rotary_dim: usize,
    position_offset: usize,
    rope_base: f32,
) -> Result<Qwen3SelfAttnRuntimePreparedSequence, String> {
    let shape = qwen3_self_attn_runtime_shape(weights)?;
    let Qwen3SelfAttnProjectedSequence {
        q_projected,
        k_projected,
        v_projected,
    } = projected;
    if sequence_len == 0 {
        return Err("self-attn prepared sequence length must be greater than zero".to_string());
    }
    if q_norm_weight.is_empty() || k_norm_weight.is_empty() {
        return Err("self-attn q/k norm weights must not be empty".to_string());
    }
    if q_norm_weight.len() != shape.head_dim || k_norm_weight.len() != shape.head_dim {
        return Err(format!(
            "self-attn q/k norm weight length must equal head_dim: q_norm_len={}, k_norm_len={}, head_dim={}",
            q_norm_weight.len(),
            k_norm_weight.len(),
            shape.head_dim
        ));
    }
    if q_norm_weight.len() != k_norm_weight.len() {
        return Err("self-attn q/k norm weights must match in length".to_string());
    }
    if !rope_base.is_finite() || rope_base <= 1.0 {
        return Err("self-attn RoPE base must be finite and greater than one".to_string());
    }

    let q_rows = weights.q_rows;
    let k_rows = weights.k_rows;
    let v_rows = weights.v_rows;
    let expected_q_len = q_rows
        .checked_mul(sequence_len)
        .ok_or_else(|| "self-attn projected q length overflows".to_string())?;
    let expected_k_len = k_rows
        .checked_mul(sequence_len)
        .ok_or_else(|| "self-attn projected k length overflows".to_string())?;
    let expected_v_len = v_rows
        .checked_mul(sequence_len)
        .ok_or_else(|| "self-attn projected v length overflows".to_string())?;
    if q_projected.len() != expected_q_len {
        return Err(format!(
            "self-attn projected q length {} does not match sequence_len={sequence_len} q_rows={q_rows}",
            q_projected.len()
        ));
    }
    if k_projected.len() != expected_k_len {
        return Err(format!(
            "self-attn projected k length {} does not match sequence_len={sequence_len} k_rows={k_rows}",
            k_projected.len(),
        ));
    }
    if v_projected.len() != expected_v_len {
        return Err(format!(
            "self-attn projected v length {} does not match sequence_len={sequence_len} v_rows={v_rows}",
            v_projected.len(),
        ));
    }
    if !rotary_dim.is_multiple_of(2) {
        return Err("self-attn RoPE rotary_dim must be even".to_string());
    }
    if rotary_dim == 0 || rotary_dim > shape.head_dim {
        return Err(format!(
            "self-attn RoPE rotary_dim must be no more than head_dim and greater than zero: rotary_dim={} head_dim={}",
            rotary_dim, shape.head_dim
        ));
    }

    let q_projection_split = split_qwen3_self_attn_q_projection(
        &q_projected,
        sequence_len,
        weights.q_rows,
        shape.hidden,
        shape.head_dim,
    )?;
    if q_projection_split.layout != shape.q_projection_layout {
        return Err(
            "self-attn q projection layout changed between shape detection and split".to_string(),
        );
    }
    if q_projection_split.q_heads != shape.q_heads {
        return Err(
            "self-attn q projection head count changed between shape detection and split"
                .to_string(),
        );
    }
    let Qwen3SelfAttnQProjectionSplit {
        query: q_query,
        gate: q_gate,
        q_heads: split_q_heads,
        layout: q_projection_layout,
    } = q_projection_split;
    let q_gate_elements = q_gate.as_ref().map_or(0, Vec::len);

    let epsilon = 1e-5_f32;
    let q_normed =
        qwen3_headwise_rmsnorm_to_host_f32(context, stream, &q_query, q_norm_weight, epsilon)?;
    let k_normed =
        qwen3_headwise_rmsnorm_to_host_f32(context, stream, &k_projected, k_norm_weight, epsilon)?;
    let q_rope = qwen3_rope_to_host_f32(
        context,
        stream,
        &q_normed,
        sequence_len,
        split_q_heads,
        shape.head_dim,
        rotary_dim,
        position_offset,
        rope_base,
    )?;
    let k_rope = qwen3_rope_to_host_f32(
        context,
        stream,
        &k_normed,
        sequence_len,
        shape.kv_heads,
        shape.head_dim,
        rotary_dim,
        position_offset,
        rope_base,
    )?;

    let softmax_scale = 1.0_f32 / (shape.head_dim as f32).sqrt();
    let attention_output = qwen3_causal_attn_to_host_f32(
        context,
        stream,
        &q_rope,
        &k_rope,
        &v_projected,
        sequence_len,
        shape.q_heads,
        shape.kv_heads,
        shape.head_dim,
        shape.value_dim,
        softmax_scale,
    )?;
    let output_gate_layout = if q_gate.is_some() {
        "runtime-sigmoid"
    } else {
        "none"
    };

    Ok(Qwen3SelfAttnRuntimePreparedSequence {
        q_query,
        k_projected,
        q_normed,
        k_normed,
        q_rope,
        k_rope,
        v_projected,
        q_gate,
        attention_output,
        shape: shape.clone(),
        softmax_scale,
        q_projection_layout,
        q_gate_elements,
        output_gate_layout,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn qwen3_self_attn_prepare_sequence_for_paged_decode_f32(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    weights: &Qwen3SelfAttnRuntimeWeights,
    residual_sequence: Vec<f32>,
    sequence_len: usize,
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
    rotary_dim: usize,
    position_offset: usize,
    rope_base: f32,
    block_table: &[u32],
    block_size: usize,
    cache_blocks: usize,
) -> Result<Qwen3SelfAttnRuntimePreparedSequenceForPagedDecode, String> {
    let projected = qwen3_self_attn_project_sequence_to_host_f32(
        context,
        stream,
        weights,
        &residual_sequence,
        sequence_len,
    )?;
    let prepared = qwen3_self_attn_prepare_sequence_runtime_f32(
        context,
        stream,
        weights,
        projected,
        sequence_len,
        q_norm_weight,
        k_norm_weight,
        rotary_dim,
        position_offset,
        rope_base,
    )?;

    let packed_shape = PagedDecodeShape {
        block_size,
        cache_blocks,
        q_heads: prepared.shape.q_heads,
        kv_heads: prepared.shape.kv_heads,
        head_dim: prepared.shape.head_dim,
        value_dim: prepared.shape.value_dim,
    };

    let PagedKvCacheReadback {
        k: paged_k_cache,
        v: paged_v_cache,
    } = pack_paged_kv_cache_for_block_table(
        &prepared.k_rope,
        &prepared.v_projected,
        block_table,
        sequence_len,
        packed_shape,
    )?;

    Ok(Qwen3SelfAttnRuntimePreparedSequenceForPagedDecode {
        residual_sequence,
        prepared,
        paged_k_cache,
        paged_v_cache,
        paged_block_table: block_table.to_vec(),
        paged_block_size: block_size,
        paged_cache_blocks: cache_blocks,
    })
}

pub fn qwen3_self_attn_runtime_shape(
    weights: &Qwen3SelfAttnRuntimeWeights,
) -> Result<Qwen3SelfAttnRuntimeShape, String> {
    let hidden = weights.q_cols;
    if hidden == 0 || weights.q_rows == 0 || weights.k_rows == 0 || weights.v_rows == 0 {
        return Err("self-attn runtime shape has zero dimension".to_string());
    }
    if weights.head_dim == 0 || weights.kv_heads == 0 || weights.value_dim == 0 {
        return Err(
            "self-attn runtime shape head_dim, kv_heads, and value_dim must be greater than zero"
                .to_string(),
        );
    }

    let two_head_dim = weights
        .head_dim
        .checked_mul(2)
        .ok_or_else(|| "self-attn q projection layout check overflows".to_string())?;
    let two_hidden_rows = weights
        .q_cols
        .checked_mul(2)
        .ok_or_else(|| "self-attn q projection layout check overflows".to_string())?;
    let q_projection_layout = if weights.q_rows == two_hidden_rows
        && weights.q_rows.is_multiple_of(two_head_dim)
    {
        "qwen3.5-gated"
    } else if weights.q_rows.is_multiple_of(weights.head_dim) {
        "plain"
    } else {
        return Err(format!(
            "self-attn q rows must indicate plain or qwen3.5-gated layout: q_rows={}, head_dim={}, hidden={}",
            weights.q_rows, weights.head_dim, hidden
        ));
    };

    let q_heads = match q_projection_layout {
        "qwen3.5-gated" => weights
            .q_rows
            .checked_div(two_head_dim)
            .ok_or_else(|| "self-attn q projection division overflow".to_string())?,
        "plain" => weights
            .q_rows
            .checked_div(weights.head_dim)
            .ok_or_else(|| "self-attn q projection division overflow".to_string())?,
        _ => return Err("self-attn q projection layout is unknown".to_string()),
    };
    if q_heads == 0 {
        return Err("self-attn q projection has zero heads".to_string());
    }

    let k_rows = weights
        .kv_heads
        .checked_mul(weights.head_dim)
        .ok_or_else(|| "self-attn k rows multiplication overflows".to_string())?;
    if weights.k_rows != k_rows {
        return Err(format!(
            "self-attn k rows mismatch: k_rows={}, kv_heads={}, head_dim={}",
            weights.k_rows, weights.kv_heads, weights.head_dim
        ));
    }

    let v_rows = weights
        .kv_heads
        .checked_mul(weights.value_dim)
        .ok_or_else(|| "self-attn v rows multiplication overflows".to_string())?;
    if weights.v_rows != v_rows {
        return Err(format!(
            "self-attn v rows must equal kv_heads * value_dim: v_rows={}, kv_heads={}, value_dim={}",
            weights.v_rows, weights.kv_heads, weights.value_dim
        ));
    }

    let attention_width = q_heads
        .checked_mul(weights.value_dim)
        .ok_or_else(|| "self-attn attention_width multiplication overflows".to_string())?;
    if !q_heads.is_multiple_of(weights.kv_heads) {
        return Err(format!(
            "self-attn q projection heads must be multiple of kv_heads: q_heads={q_heads}, kv_heads={}",
            weights.kv_heads
        ));
    }
    if weights.o_rows != hidden || weights.o_cols != attention_width {
        return Err(format!(
            "self-attn o projection shape mismatch: o_rows={}, o_cols={}, hidden={}, attention_width={attention_width}",
            weights.o_rows, weights.o_cols, hidden
        ));
    }

    Ok(Qwen3SelfAttnRuntimeShape {
        hidden,
        q_heads,
        kv_heads: weights.kv_heads,
        head_dim: weights.head_dim,
        value_dim: weights.value_dim,
        attention_width,
        q_projection_layout,
    })
}

pub fn qwen3_self_attn_project_sequence_to_host_f32(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    weights: &Qwen3SelfAttnRuntimeWeights,
    input_sequence: &[f32],
    sequence_len: usize,
) -> Result<Qwen3SelfAttnProjectedSequence, String> {
    let shape = qwen3_self_attn_runtime_shape(weights)?;
    if sequence_len == 0 {
        return Err(
            "self-attn sequence projection sequence_len must be greater than zero".to_string(),
        );
    }

    let hidden = shape.hidden;
    let expected_input_len = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| "self-attn sequence projection input length overflows".to_string())?;
    if input_sequence.len() != expected_input_len {
        return Err(format!(
            "self-attn sequence projection input length {} does not match sequence_len={sequence_len} hidden={hidden}",
            input_sequence.len()
        ));
    }

    let q_elements = sequence_len
        .checked_mul(weights.q_rows)
        .ok_or_else(|| "self-attn sequence projection q output length overflows".to_string())?;
    let k_rows = weights.k_rows;
    let v_rows = weights.v_rows;
    let k_elements = sequence_len
        .checked_mul(k_rows)
        .ok_or_else(|| "self-attn sequence projection k output length overflows".to_string())?;
    let v_elements = sequence_len
        .checked_mul(v_rows)
        .ok_or_else(|| "self-attn sequence projection v output length overflows".to_string())?;

    let hidden_bytes = hidden
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "self-attn sequence projection input byte size overflows".to_string())?;
    let q_bytes = weights
        .q_rows
        .checked_mul(hidden)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "self-attn q projection matrix byte size overflows".to_string())?;
    if weights.q_matrix.size()? != q_bytes {
        return Err(format!(
            "self-attn q projection matrix byte size mismatch: expected {q_bytes}, got {}",
            weights.q_matrix.size()?
        ));
    }
    let k_bytes = k_rows
        .checked_mul(hidden)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "self-attn k projection matrix byte size overflows".to_string())?;
    if weights.k_matrix.size()? != k_bytes {
        return Err(format!(
            "self-attn k projection matrix byte size mismatch: expected {k_bytes}, got {}",
            weights.k_matrix.size()?
        ));
    }
    let v_bytes = v_rows
        .checked_mul(hidden)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "self-attn v projection matrix byte size overflows".to_string())?;
    if weights.v_matrix.size()? != v_bytes {
        return Err(format!(
            "self-attn v projection matrix byte size mismatch: expected {v_bytes}, got {}",
            weights.v_matrix.size()?
        ));
    }
    let q_step_bytes = weights
        .q_rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "self-attn q projection step output byte size overflows".to_string())?;
    let k_step_bytes = k_rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "self-attn k projection step output byte size overflows".to_string())?;
    let v_step_bytes = v_rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "self-attn v projection step output byte size overflows".to_string())?;

    let mut input_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
        format!("failed to allocate Qwen3 self-attn projection input buffer: {err}")
    })?;
    let mut q_buffer = context.alloc_buffer(q_step_bytes).map_err(|err| {
        format!("failed to allocate Qwen3 self-attn q projection output buffer: {err}")
    })?;
    let mut k_buffer = context.alloc_buffer(k_step_bytes).map_err(|err| {
        format!("failed to allocate Qwen3 self-attn k projection output buffer: {err}")
    })?;
    let mut v_buffer = context.alloc_buffer(v_step_bytes).map_err(|err| {
        format!("failed to allocate Qwen3 self-attn v projection output buffer: {err}")
    })?;

    let mut q_projected = Vec::with_capacity(q_elements);
    let mut k_projected = Vec::with_capacity(k_elements);
    let mut v_projected = Vec::with_capacity(v_elements);

    for timestep in 0..sequence_len {
        let step_input = &input_sequence[timestep * hidden..(timestep + 1) * hidden];
        input_buffer
            .copy_from_host(0, &f32s_to_le_bytes(step_input), Some(stream))
            .map_err(|err| {
                format!(
                    "failed to copy Qwen3 self-attn sequence input at timestep {timestep}: {err}"
                )
            })?;
        stream.synchronize().map_err(|err| {
            format!(
                "failed to synchronize Qwen3 self-attn sequence input copy at timestep {timestep}: {err}"
            )
        })?;

        q_projected.extend(qwen3_self_attn_project_to_host_f32(
            &weights.q_matrix,
            &input_buffer,
            &mut q_buffer,
            weights.q_rows,
            hidden,
            stream,
            "q",
            timestep,
        )?);
        k_projected.extend(qwen3_self_attn_project_to_host_f32(
            &weights.k_matrix,
            &input_buffer,
            &mut k_buffer,
            k_rows,
            hidden,
            stream,
            "k",
            timestep,
        )?);
        v_projected.extend(qwen3_self_attn_project_to_host_f32(
            &weights.v_matrix,
            &input_buffer,
            &mut v_buffer,
            v_rows,
            hidden,
            stream,
            "v",
            timestep,
        )?);
    }

    if q_projected.len() != q_elements
        || k_projected.len() != k_elements
        || v_projected.len() != v_elements
    {
        return Err("self-attn sequence projection produced unexpected output length".to_string());
    }

    let q_output_bytes = q_projected
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "self-attn q projection output byte size overflows".to_string())?;
    let k_output_bytes = k_projected
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "self-attn k projection output byte size overflows".to_string())?;
    let v_output_bytes = v_projected
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "self-attn v projection output byte size overflows".to_string())?;

    let expected_q_output_bytes = q_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "self-attn expected q projection byte size overflows".to_string())?;
    let expected_k_output_bytes = k_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "self-attn expected k projection byte size overflows".to_string())?;
    let expected_v_output_bytes = v_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "self-attn expected v projection byte size overflows".to_string())?;

    if q_output_bytes != expected_q_output_bytes
        || k_output_bytes != expected_k_output_bytes
        || v_output_bytes != expected_v_output_bytes
    {
        return Err("self-attn sequence projection output byte size mismatch".to_string());
    }

    Ok(Qwen3SelfAttnProjectedSequence {
        q_projected,
        k_projected,
        v_projected,
    })
}

fn qwen3_self_attn_project_to_host_f32(
    matrix: &RuntimeBuffer,
    input_buffer: &RuntimeBuffer,
    output_buffer: &mut RuntimeBuffer,
    rows: usize,
    cols: usize,
    stream: &mut RuntimeStream,
    projection_name: &str,
    timestep: usize,
) -> Result<Vec<f32>, String> {
    if rows == 0 {
        return Err(format!(
            "self-attn {projection_name} projection rows must be greater than zero"
        ));
    }
    if cols == 0 {
        return Err(format!(
            "self-attn {projection_name} projection cols must be greater than zero"
        ));
    }
    let output_bytes = rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            format!("self-attn {projection_name} projection output byte size overflows")
        })?;
    if output_buffer.size()? != output_bytes {
        return Err(format!(
            "self-attn {projection_name} projection output buffer has unexpected byte size: got {} expected {}",
            output_buffer.size()?,
            output_bytes
        ));
    }
    ullm_runtime_sys::matvec_f32(
        matrix,
        input_buffer,
        rows,
        cols,
        output_buffer,
        Some(stream),
    )
    .map_err(|err| {
        format!("failed to run Qwen3 self-attn {projection_name} projection at timestep {timestep}: {err}")
    })?;
    stream.synchronize().map_err(|err| {
        format!(
            "failed to synchronize Qwen3 self-attn {projection_name} projection at timestep {timestep}: {err}"
        )
    })?;
    let mut output_bytes_host = vec![0_u8; output_bytes];
    output_buffer
        .copy_to_host(0, &mut output_bytes_host, Some(stream))
        .map_err(|err| {
            format!(
                "failed to copy Qwen3 self-attn {projection_name} projection at timestep {timestep}: {err}"
            )
        })?;
    stream.synchronize().map_err(|err| {
        format!(
            "failed to synchronize Qwen3 self-attn {projection_name} projection output copy at timestep {timestep}: {err}"
        )
    })?;
    Ok(le_bytes_to_f32s(&output_bytes_host))
}

#[derive(Debug)]
pub struct Qwen3SelfAttnQProjectionSplit {
    pub query: Vec<f32>,
    pub gate: Option<Vec<f32>>,
    pub q_heads: usize,
    pub layout: &'static str,
}

pub fn split_qwen3_self_attn_q_projection(
    projected: &[f32],
    sequence_len: usize,
    q_rows: usize,
    hidden: usize,
    head_dim: usize,
) -> Result<Qwen3SelfAttnQProjectionSplit, String> {
    if sequence_len == 0 || q_rows == 0 || hidden == 0 || head_dim == 0 {
        return Err("self-attn q projection split received a zero dimension".to_string());
    }
    let expected_len = sequence_len
        .checked_mul(q_rows)
        .ok_or_else(|| "self-attn q projection split length overflows".to_string())?;
    if projected.len() != expected_len {
        return Err(format!(
            "self-attn q projection length mismatch: got {}, expected {expected_len}",
            projected.len()
        ));
    }

    let qwen35_gated = hidden
        .checked_mul(2)
        .is_some_and(|gated_rows| gated_rows == q_rows)
        && q_rows.is_multiple_of(2 * head_dim);
    if qwen35_gated {
        let q_heads = q_rows / (2 * head_dim);
        if q_heads == 0 {
            return Err("self-attn gated q projection has zero heads".to_string());
        }
        let query_len = sequence_len
            .checked_mul(q_heads)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or_else(|| "self-attn gated q projection output length overflows".to_string())?;
        let mut query = Vec::with_capacity(query_len);
        let mut gate = Vec::with_capacity(query_len);
        for timestep in 0..sequence_len {
            let timestep_start = timestep * q_rows;
            for head in 0..q_heads {
                let head_start = timestep_start + head * 2 * head_dim;
                let query_start = head_start;
                let gate_start = head_start + head_dim;
                query.extend_from_slice(&projected[query_start..query_start + head_dim]);
                gate.extend_from_slice(&projected[gate_start..gate_start + head_dim]);
            }
        }
        return Ok(Qwen3SelfAttnQProjectionSplit {
            query,
            gate: Some(gate),
            q_heads,
            layout: "qwen3.5-gated",
        });
    }

    if !q_rows.is_multiple_of(head_dim) {
        return Err(format!(
            "self-attn q rows must be a multiple of head_dim: q_rows={q_rows}, head_dim={head_dim}"
        ));
    }
    let q_heads = q_rows / head_dim;
    if q_heads == 0 {
        return Err("self-attn q projection has zero heads".to_string());
    }
    Ok(Qwen3SelfAttnQProjectionSplit {
        query: projected.to_vec(),
        gate: None,
        q_heads,
        layout: "plain",
    })
}

pub fn qwen3_headwise_rmsnorm_to_host_f32(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, String> {
    let head_dim = weight.len();
    if head_dim == 0 {
        return Err("Qwen3 headwise RMSNorm weight must not be empty".to_string());
    }
    if !input.len().is_multiple_of(head_dim) {
        return Err(format!(
            "Qwen3 headwise RMSNorm input length {} is not a multiple of head_dim {}",
            input.len(),
            head_dim
        ));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(
            "Qwen3 headwise RMSNorm epsilon must be finite and greater than zero".to_string(),
        );
    }

    let weight_bytes = f32s_to_le_bytes(weight);
    let mut weight_buffer = context
        .alloc_buffer(weight_bytes.len())
        .map_err(|err| format!("failed to allocate Qwen3 headwise RMSNorm weight buffer: {err}"))?;
    weight_buffer
        .copy_from_host(0, &weight_bytes, Some(stream))
        .map_err(|err| format!("failed to copy Qwen3 headwise RMSNorm weight: {err}"))?;
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize Qwen3 headwise RMSNorm weight copy: {err}")
    })?;

    let head_bytes = f32_bytes(head_dim);
    let mut input_buffer = context
        .alloc_buffer(head_bytes)
        .map_err(|err| format!("failed to allocate Qwen3 headwise RMSNorm input buffer: {err}"))?;
    let mut output_buffer = context
        .alloc_buffer(head_bytes)
        .map_err(|err| format!("failed to allocate Qwen3 headwise RMSNorm output buffer: {err}"))?;
    let mut output = Vec::with_capacity(input.len());
    let mut output_head_bytes = vec![0_u8; head_bytes];
    for head_input in input.chunks_exact(head_dim) {
        let head_input_bytes = f32s_to_le_bytes(head_input);
        input_buffer
            .copy_from_host(0, &head_input_bytes, Some(stream))
            .map_err(|err| format!("failed to copy Qwen3 headwise RMSNorm head input: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize Qwen3 headwise RMSNorm head input copy: {err}")
        })?;
        ullm_runtime_sys::rmsnorm_f32(
            &input_buffer,
            &weight_buffer,
            head_dim,
            epsilon,
            &mut output_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run Qwen3 headwise RMSNorm: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize Qwen3 headwise RMSNorm: {err}"))?;
        output_buffer
            .copy_to_host(0, &mut output_head_bytes, Some(stream))
            .map_err(|err| format!("failed to copy Qwen3 headwise RMSNorm head output: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize Qwen3 headwise RMSNorm head output copy: {err}")
        })?;
        output.extend(le_bytes_to_f32s(&output_head_bytes));
    }
    Ok(output)
}

pub fn qwen3_rope_to_host_f32(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    input: &[f32],
    sequence_len: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    position_offset: usize,
    rope_base: f32,
) -> Result<Vec<f32>, String> {
    if sequence_len == 0 {
        return Err("Qwen3 RoPE sequence length must be greater than zero".to_string());
    }
    if heads == 0 {
        return Err("Qwen3 RoPE heads must be greater than zero".to_string());
    }
    if head_dim == 0 {
        return Err("Qwen3 RoPE head_dim must be greater than zero".to_string());
    }
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err("Qwen3 RoPE rotary_dim must be even and no greater than head_dim".to_string());
    }
    if !rope_base.is_finite() || rope_base <= 1.0 {
        return Err("Qwen3 RoPE base must be finite and greater than one".to_string());
    }
    let input_elements = sequence_len
        .checked_mul(heads)
        .ok_or_else(|| "Qwen3 RoPE head-sequence element count overflows".to_string())?
        .checked_mul(head_dim)
        .ok_or_else(|| "Qwen3 RoPE element count overflows".to_string())?;
    if input.len() != input_elements {
        return Err(format!(
            "Qwen3 RoPE input length {} does not match sequence_len={sequence_len} heads={heads} head_dim={head_dim}",
            input.len()
        ));
    }
    let bytes = input_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "Qwen3 RoPE byte size overflows".to_string())?;
    let input_bytes = f32s_to_le_bytes(input);
    let mut input_buffer = context
        .alloc_buffer(bytes)
        .map_err(|err| format!("failed to allocate Qwen3 RoPE input buffer: {err}"))?;
    let mut output_buffer = context
        .alloc_buffer(bytes)
        .map_err(|err| format!("failed to allocate Qwen3 RoPE output buffer: {err}"))?;
    input_buffer
        .copy_from_host(0, &input_bytes, Some(stream))
        .map_err(|err| format!("failed to copy Qwen3 RoPE input: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize Qwen3 RoPE input copy: {err}"))?;
    ullm_runtime_sys::rope_f32(
        &input_buffer,
        sequence_len,
        heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        &mut output_buffer,
        Some(stream),
    )
    .map_err(|err| format!("failed to run Qwen3 RoPE: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize Qwen3 RoPE: {err}"))?;
    let mut output_bytes = vec![0_u8; bytes];
    output_buffer
        .copy_to_host(0, &mut output_bytes, Some(stream))
        .map_err(|err| format!("failed to copy Qwen3 RoPE output: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize Qwen3 RoPE output copy: {err}"))?;
    Ok(le_bytes_to_f32s(&output_bytes))
}

pub fn qwen3_causal_attn_to_host_f32(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sequence_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
) -> Result<Vec<f32>, String> {
    if sequence_len == 0 {
        return Err("Qwen3 causal attention sequence length must be greater than zero".to_string());
    }
    if q_heads == 0 {
        return Err("Qwen3 causal attention q_heads must be greater than zero".to_string());
    }
    if kv_heads == 0 {
        return Err("Qwen3 causal attention kv_heads must be greater than zero".to_string());
    }
    if head_dim == 0 {
        return Err("Qwen3 causal attention head_dim must be greater than zero".to_string());
    }
    if value_dim == 0 {
        return Err("Qwen3 causal attention value_dim must be greater than zero".to_string());
    }
    if !q_heads.is_multiple_of(kv_heads) {
        return Err(format!(
            "Qwen3 causal attention q_heads must be a multiple of kv_heads: q_heads={q_heads} kv_heads={kv_heads}"
        ));
    }
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err(
            "Qwen3 causal attention softmax_scale must be finite and greater than zero".to_string(),
        );
    }

    let q_elements = sequence_len
        .checked_mul(q_heads)
        .ok_or_else(|| "Qwen3 causal attention q element count overflows".to_string())?
        .checked_mul(head_dim)
        .ok_or_else(|| "Qwen3 causal attention q element count overflows".to_string())?;
    if q.len() != q_elements {
        return Err(format!(
            "Qwen3 causal attention q length {} does not match sequence_len={sequence_len} q_heads={q_heads} head_dim={head_dim}",
            q.len()
        ));
    }
    let q_byte_count = q_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "Qwen3 causal attention q byte size overflows".to_string())?;

    let k_elements = sequence_len
        .checked_mul(kv_heads)
        .ok_or_else(|| "Qwen3 causal attention k element count overflows".to_string())?
        .checked_mul(head_dim)
        .ok_or_else(|| "Qwen3 causal attention k element count overflows".to_string())?;
    if k.len() != k_elements {
        return Err(format!(
            "Qwen3 causal attention k length {} does not match sequence_len={sequence_len} kv_heads={kv_heads} head_dim={head_dim}",
            k.len()
        ));
    }
    let k_byte_count = k_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "Qwen3 causal attention k byte size overflows".to_string())?;

    let v_elements = sequence_len
        .checked_mul(kv_heads)
        .ok_or_else(|| "Qwen3 causal attention v element count overflows".to_string())?
        .checked_mul(value_dim)
        .ok_or_else(|| "Qwen3 causal attention v element count overflows".to_string())?;
    if v.len() != v_elements {
        return Err(format!(
            "Qwen3 causal attention v length {} does not match sequence_len={sequence_len} kv_heads={kv_heads} value_dim={value_dim}",
            v.len()
        ));
    }
    let v_byte_count = v_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "Qwen3 causal attention v byte size overflows".to_string())?;

    let q_bytes = f32s_to_le_bytes(q);
    let k_bytes = f32s_to_le_bytes(k);
    let v_bytes = f32s_to_le_bytes(v);
    let output_elements = sequence_len
        .checked_mul(q_heads)
        .ok_or_else(|| "Qwen3 causal attention output element count overflows".to_string())?
        .checked_mul(value_dim)
        .ok_or_else(|| "Qwen3 causal attention output element count overflows".to_string())?;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "Qwen3 causal attention output byte size overflows".to_string())?;

    let mut q_buffer = context
        .alloc_buffer(q_byte_count)
        .map_err(|err| format!("failed to allocate Qwen3 causal attention q buffer: {err}"))?;
    let mut k_buffer = context
        .alloc_buffer(k_byte_count)
        .map_err(|err| format!("failed to allocate Qwen3 causal attention k buffer: {err}"))?;
    let mut v_buffer = context
        .alloc_buffer(v_byte_count)
        .map_err(|err| format!("failed to allocate Qwen3 causal attention v buffer: {err}"))?;
    let mut output_buffer = context
        .alloc_buffer(output_bytes)
        .map_err(|err| format!("failed to allocate Qwen3 causal attention output buffer: {err}"))?;

    q_buffer
        .copy_from_host(0, &q_bytes, Some(stream))
        .map_err(|err| format!("failed to copy Qwen3 causal attention q input: {err}"))?;
    k_buffer
        .copy_from_host(0, &k_bytes, Some(stream))
        .map_err(|err| format!("failed to copy Qwen3 causal attention k input: {err}"))?;
    v_buffer
        .copy_from_host(0, &v_bytes, Some(stream))
        .map_err(|err| format!("failed to copy Qwen3 causal attention v input: {err}"))?;
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize Qwen3 causal attention input copies: {err}")
    })?;

    ullm_runtime_sys::causal_attn_f32(
        &q_buffer,
        &k_buffer,
        &v_buffer,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        &mut output_buffer,
        Some(stream),
    )
    .map_err(|err| format!("failed to run Qwen3 causal attention: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize Qwen3 causal attention: {err}"))?;

    let mut output_bytes_host = vec![0_u8; output_bytes];
    output_buffer
        .copy_to_host(0, &mut output_bytes_host, Some(stream))
        .map_err(|err| format!("failed to copy Qwen3 causal attention output: {err}"))?;
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize Qwen3 causal attention output copy: {err}")
    })?;
    Ok(le_bytes_to_f32s(&output_bytes_host))
}

pub struct Qwen3MlpRuntimeWeights {
    pub gate_rows: usize,
    pub gate_cols: usize,
    pub gate_matrix: RuntimeBuffer,
    pub up_matrix: RuntimeBuffer,
    pub down_matrix: RuntimeBuffer,
}

pub struct Qwen3PostAttentionRuntimeWeights {
    pub hidden: usize,
    pub intermediate: usize,
    pub post_norm_weight: RuntimeBuffer,
    pub mlp: Qwen3MlpRuntimeWeights,
}

pub struct Qwen3DecoderLayerRuntimeWeights {
    pub self_attn: Qwen3SelfAttnRuntimeWeights,
    pub post_attention: Qwen3PostAttentionRuntimeWeights,
}

pub struct Qwen3DecoderLayerRuntime<'weights> {
    weights: &'weights Qwen3DecoderLayerRuntimeWeights,
    step_state: Qwen3DecoderLayerStepState,
}

impl<'weights> Qwen3DecoderLayerRuntime<'weights> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        weights: &'weights Qwen3DecoderLayerRuntimeWeights,
        decode_shape: PagedDecodeShape,
        block_table: Vec<u32>,
        softmax_scale: f32,
        mlp_epsilon: f32,
    ) -> Result<Self, String> {
        validate_qwen3_decoder_layer_decode_shape(weights, decode_shape)?;
        let post_attention = &weights.post_attention;
        if weights.self_attn.q_cols != post_attention.hidden {
            return Err(format!(
                "Qwen3 decoder layer runtime hidden mismatch: self_attn_hidden={} post_attention_hidden={}",
                weights.self_attn.q_cols, post_attention.hidden
            ));
        }
        if post_attention.mlp.gate_rows != post_attention.intermediate
            || post_attention.mlp.gate_cols != post_attention.hidden
        {
            return Err("Qwen3 decoder layer runtime MLP gate shape is inconsistent".to_string());
        }
        let step_state = Qwen3DecoderLayerStepState::new(
            context,
            stream,
            decode_shape,
            block_table,
            post_attention.hidden,
            post_attention.intermediate,
            softmax_scale,
            mlp_epsilon,
        )?;

        Ok(Self {
            weights,
            step_state,
        })
    }

    pub fn step(
        &mut self,
        stream: &mut RuntimeStream,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output_gate: Option<&[f32]>,
        residual: &[f32],
    ) -> Result<Qwen3DecoderLayerStepOutput, String> {
        let post_attention = &self.weights.post_attention;
        self.step_state.step(
            stream,
            &self.weights.self_attn.o_matrix,
            &post_attention.post_norm_weight,
            &post_attention.mlp.gate_matrix,
            &post_attention.mlp.up_matrix,
            &post_attention.mlp.down_matrix,
            q,
            k,
            v,
            output_gate,
            residual,
        )
    }

    pub fn step_output_only(
        &mut self,
        stream: &mut RuntimeStream,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output_gate: Option<&[f32]>,
        residual: &[f32],
    ) -> Result<Qwen3DecoderLayerOutputStep, String> {
        let post_attention = &self.weights.post_attention;
        self.step_state.step_output_only(
            stream,
            &self.weights.self_attn.o_matrix,
            &post_attention.post_norm_weight,
            &post_attention.mlp.gate_matrix,
            &post_attention.mlp.up_matrix,
            &post_attention.mlp.down_matrix,
            q,
            k,
            v,
            output_gate,
            residual,
        )
    }

    pub fn read_cache_to_host(
        &self,
        stream: &mut RuntimeStream,
    ) -> Result<PagedKvCacheReadback, String> {
        self.step_state.read_cache_to_host(stream)
    }

    pub fn written_len(&self) -> usize {
        self.step_state.written_len()
    }

    pub fn block_table(&self) -> &[u32] {
        self.step_state.block_table()
    }
}

fn validate_qwen3_decoder_layer_decode_shape(
    weights: &Qwen3DecoderLayerRuntimeWeights,
    decode_shape: PagedDecodeShape,
) -> Result<Qwen3SelfAttnRuntimeShape, String> {
    let self_attn_shape = qwen3_self_attn_runtime_shape(&weights.self_attn)
        .map_err(|err| format!("Qwen3 decoder layer self-attn shape is invalid: {err}"))?;
    if self_attn_shape.hidden != weights.post_attention.hidden {
        return Err(format!(
            "Qwen3 decoder layer hidden mismatch: self_attn_hidden={} post_attention_hidden={}",
            self_attn_shape.hidden, weights.post_attention.hidden
        ));
    }
    if decode_shape.q_heads != self_attn_shape.q_heads
        || decode_shape.kv_heads != self_attn_shape.kv_heads
        || decode_shape.head_dim != self_attn_shape.head_dim
        || decode_shape.value_dim != self_attn_shape.value_dim
    {
        return Err(format!(
            "Qwen3 decoder layer decode shape mismatch: decode=[q_heads={}, kv_heads={}, head_dim={}, value_dim={}] weights=[q_heads={}, kv_heads={}, head_dim={}, value_dim={}]",
            decode_shape.q_heads,
            decode_shape.kv_heads,
            decode_shape.head_dim,
            decode_shape.value_dim,
            self_attn_shape.q_heads,
            self_attn_shape.kv_heads,
            self_attn_shape.head_dim,
            self_attn_shape.value_dim
        ));
    }
    Ok(self_attn_shape)
}

pub fn pack_paged_kv_cache_for_block_table(
    logical_k_cache: &[f32],
    logical_v_cache: &[f32],
    block_table: &[u32],
    cache_len: usize,
    shape: PagedDecodeShape,
) -> Result<PagedKvCacheReadback, String> {
    shape.validate()?;
    if cache_len == 0 {
        return Err("paged decode cache_len must be greater than zero".to_string());
    }
    if shape.block_size == 0 {
        return Err("paged decode block_size must be greater than zero".to_string());
    }
    if shape.kv_heads == 0 || shape.head_dim == 0 || shape.value_dim == 0 {
        return Err(format!(
            "paged decode dimensions must be nonzero: kv_heads={} head_dim={} value_dim={}",
            shape.kv_heads, shape.head_dim, shape.value_dim
        ));
    }
    if shape.cache_blocks == 0 {
        return Err("paged decode cache_blocks must be greater than zero".to_string());
    }

    let k_token_elements = shape
        .kv_heads
        .checked_mul(shape.head_dim)
        .ok_or_else(|| "paged decode k token element count overflows".to_string())?;
    let v_token_elements = shape
        .kv_heads
        .checked_mul(shape.value_dim)
        .ok_or_else(|| "paged decode v token element count overflows".to_string())?;
    let expected_k_len = cache_len
        .checked_mul(k_token_elements)
        .ok_or_else(|| "paged decode logical k cache length overflows".to_string())?;
    let expected_v_len = cache_len
        .checked_mul(v_token_elements)
        .ok_or_else(|| "paged decode logical v cache length overflows".to_string())?;

    if logical_k_cache.len() != expected_k_len {
        return Err(format!(
            "logical k cache length {} does not match cache_len={cache_len} kv_heads={} head_dim={} (expected={expected_k_len})",
            logical_k_cache.len(),
            shape.kv_heads,
            shape.head_dim
        ));
    }
    if logical_v_cache.len() != expected_v_len {
        return Err(format!(
            "logical v cache length {} does not match cache_len={cache_len} kv_heads={} value_dim={} (expected={expected_v_len})",
            logical_v_cache.len(),
            shape.kv_heads,
            shape.value_dim
        ));
    }

    let block_table_entries = (cache_len - 1) / shape.block_size + 1;
    if block_table.len() < block_table_entries {
        return Err(format!(
            "paged decode block table length {} is shorter than expected entries {}",
            block_table.len(),
            block_table_entries
        ));
    }
    if block_table.is_empty() {
        return Err("paged decode block table must not be empty".to_string());
    }
    for (index, block_id) in block_table.iter().copied().enumerate() {
        if block_id as usize >= shape.cache_blocks {
            return Err(format!(
                "paged decode block_table[{index}]={block_id} exceeds cache_blocks={}",
                shape.cache_blocks
            ));
        }
    }

    let physical_tokens = shape
        .cache_blocks
        .checked_mul(shape.block_size)
        .ok_or_else(|| "paged decode physical token count overflows".to_string())?;
    let physical_k_elements = physical_tokens
        .checked_mul(k_token_elements)
        .ok_or_else(|| "paged decode physical k element count overflows".to_string())?;
    let physical_v_elements = physical_tokens
        .checked_mul(v_token_elements)
        .ok_or_else(|| "paged decode physical v element count overflows".to_string())?;

    let mut physical_k_cache = vec![0.0_f32; physical_k_elements];
    let mut physical_v_cache = vec![0.0_f32; physical_v_elements];

    for timestep in 0..cache_len {
        let logical_block = timestep / shape.block_size;
        let block_offset = timestep - logical_block * shape.block_size;
        let physical_block = block_table[logical_block] as usize;
        let physical_timestep = physical_block
            .checked_mul(shape.block_size)
            .and_then(|base| base.checked_add(block_offset))
            .ok_or_else(|| "paged decode physical timestep index overflows".to_string())?;
        if physical_timestep >= physical_tokens {
            return Err(format!(
                "paged decode physical timestep {physical_timestep} exceeds physical token count {physical_tokens}"
            ));
        }

        let logical_k_start = timestep
            .checked_mul(k_token_elements)
            .ok_or_else(|| "paged decode logical k start index overflows".to_string())?;
        let logical_v_start = timestep
            .checked_mul(v_token_elements)
            .ok_or_else(|| "paged decode logical v start index overflows".to_string())?;
        let physical_k_start = physical_timestep
            .checked_mul(k_token_elements)
            .ok_or_else(|| "paged decode physical k start index overflows".to_string())?;
        let physical_v_start = physical_timestep
            .checked_mul(v_token_elements)
            .ok_or_else(|| "paged decode physical v start index overflows".to_string())?;

        let logical_k_end = logical_k_start
            .checked_add(k_token_elements)
            .ok_or_else(|| "paged decode logical k end index overflows".to_string())?;
        let logical_v_end = logical_v_start
            .checked_add(v_token_elements)
            .ok_or_else(|| "paged decode logical v end index overflows".to_string())?;
        let physical_k_end = physical_k_start
            .checked_add(k_token_elements)
            .ok_or_else(|| "paged decode physical k end index overflows".to_string())?;
        let physical_v_end = physical_v_start
            .checked_add(v_token_elements)
            .ok_or_else(|| "paged decode physical v end index overflows".to_string())?;

        physical_k_cache[physical_k_start..physical_k_end]
            .copy_from_slice(&logical_k_cache[logical_k_start..logical_k_end]);
        physical_v_cache[physical_v_start..physical_v_end]
            .copy_from_slice(&logical_v_cache[logical_v_start..logical_v_end]);
    }

    Ok(PagedKvCacheReadback {
        k: physical_k_cache,
        v: physical_v_cache,
    })
}

impl PagedDecodeShape {
    pub fn validate(&self) -> Result<(), String> {
        if self.block_size == 0 {
            return Err("paged decode shape block_size must be greater than zero".to_string());
        }
        if self.cache_blocks == 0 {
            return Err("paged decode shape cache_blocks must be greater than zero".to_string());
        }
        if self.q_heads == 0 {
            return Err("paged decode shape q_heads must be greater than zero".to_string());
        }
        if self.kv_heads == 0 {
            return Err("paged decode shape kv_heads must be greater than zero".to_string());
        }
        if self.head_dim == 0 {
            return Err("paged decode shape head_dim must be greater than zero".to_string());
        }
        if self.value_dim == 0 {
            return Err("paged decode shape value_dim must be greater than zero".to_string());
        }
        if !self.q_heads.is_multiple_of(self.kv_heads) {
            return Err("paged decode shape q_heads must be a multiple of kv_heads".to_string());
        }
        self.physical_tokens()?;
        self.q_elements()?;
        self.k_token_elements()?;
        self.v_token_elements()?;
        self.k_cache_elements()?;
        self.v_cache_elements()?;
        self.output_elements()?;
        Ok(())
    }

    pub fn physical_tokens(&self) -> Result<usize, String> {
        self.cache_blocks
            .checked_mul(self.block_size)
            .ok_or_else(|| "paged decode shape physical token count overflows".to_string())
    }

    pub fn q_elements(&self) -> Result<usize, String> {
        self.q_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| "paged decode shape q element count overflows".to_string())
    }

    pub fn k_token_elements(&self) -> Result<usize, String> {
        self.kv_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| "paged decode shape k token element count overflows".to_string())
    }

    pub fn v_token_elements(&self) -> Result<usize, String> {
        self.kv_heads
            .checked_mul(self.value_dim)
            .ok_or_else(|| "paged decode shape v token element count overflows".to_string())
    }

    pub fn k_cache_elements(&self) -> Result<usize, String> {
        self.physical_tokens()?
            .checked_mul(self.k_token_elements()?)
            .ok_or_else(|| "paged decode shape k cache element count overflows".to_string())
    }

    pub fn v_cache_elements(&self) -> Result<usize, String> {
        self.physical_tokens()?
            .checked_mul(self.v_token_elements()?)
            .ok_or_else(|| "paged decode shape v cache element count overflows".to_string())
    }

    pub fn output_elements(&self) -> Result<usize, String> {
        self.q_heads
            .checked_mul(self.value_dim)
            .ok_or_else(|| "paged decode shape output element count overflows".to_string())
    }
}

impl PagedDecodeState {
    pub fn new(
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        shape: PagedDecodeShape,
        block_table: Vec<u32>,
    ) -> Result<Self, String> {
        shape.validate()?;
        validate_block_table(&block_table, shape.cache_blocks)?;

        let block_table_bytes = u32s_to_le_bytes(&block_table);
        let mut block_table_buffer = context
            .alloc_buffer(block_table_bytes.len())
            .map_err(|err| format!("failed to allocate paged decoder block table: {err}"))?;
        let mut q_buffer = context
            .alloc_buffer(f32_bytes(shape.q_elements()?))
            .map_err(|err| format!("failed to allocate paged decoder q buffer: {err}"))?;
        let mut k_token_buffer = context
            .alloc_buffer(f32_bytes(shape.k_token_elements()?))
            .map_err(|err| format!("failed to allocate paged decoder k token buffer: {err}"))?;
        let mut v_token_buffer = context
            .alloc_buffer(f32_bytes(shape.v_token_elements()?))
            .map_err(|err| format!("failed to allocate paged decoder v token buffer: {err}"))?;
        let mut k_cache_buffer = context
            .alloc_buffer(f32_bytes(shape.k_cache_elements()?))
            .map_err(|err| format!("failed to allocate paged decoder k cache: {err}"))?;
        let mut v_cache_buffer = context
            .alloc_buffer(f32_bytes(shape.v_cache_elements()?))
            .map_err(|err| format!("failed to allocate paged decoder v cache: {err}"))?;
        let output_buffer = context
            .alloc_buffer(f32_bytes(shape.output_elements()?))
            .map_err(|err| format!("failed to allocate paged decoder output: {err}"))?;

        block_table_buffer
            .copy_from_host(0, &block_table_bytes, Some(stream))
            .map_err(|err| format!("failed to copy paged decoder block table: {err}"))?;
        zero_buffer(&mut q_buffer, Some(stream))?;
        zero_buffer(&mut k_token_buffer, Some(stream))?;
        zero_buffer(&mut v_token_buffer, Some(stream))?;
        zero_buffer(&mut k_cache_buffer, Some(stream))?;
        zero_buffer(&mut v_cache_buffer, Some(stream))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize paged decoder setup: {err}"))?;

        Ok(Self {
            shape,
            block_table,
            written_len: 0,
            block_table_buffer,
            q_buffer,
            k_token_buffer,
            v_token_buffer,
            k_cache_buffer,
            v_cache_buffer,
            output_buffer,
        })
    }

    pub fn shape(&self) -> PagedDecodeShape {
        self.shape
    }

    pub fn block_table(&self) -> &[u32] {
        &self.block_table
    }

    pub fn written_len(&self) -> usize {
        self.written_len
    }

    pub(crate) fn enqueue_serving_reset(
        &mut self,
        stream: &mut RuntimeStream,
    ) -> Result<(), String> {
        for (label, buffer) in [
            ("q", &mut self.q_buffer),
            ("k token", &mut self.k_token_buffer),
            ("v token", &mut self.v_token_buffer),
            ("k cache", &mut self.k_cache_buffer),
            ("v cache", &mut self.v_cache_buffer),
            ("output", &mut self.output_buffer),
        ] {
            let bytes = buffer
                .size()
                .map_err(|err| format!("failed to inspect serving paged {label} buffer: {err}"))?;
            buffer
                .zero(0, bytes, Some(&mut *stream))
                .map_err(|err| format!("failed to enqueue serving paged {label} reset: {err}"))?;
        }
        Ok(())
    }

    pub(crate) fn commit_serving_reset(&mut self) {
        self.written_len = 0;
    }

    pub fn reset(&mut self, stream: &mut RuntimeStream) -> Result<(), String> {
        zero_buffer(&mut self.k_cache_buffer, Some(stream))?;
        zero_buffer(&mut self.v_cache_buffer, Some(stream))?;
        self.written_len = 0;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize paged decoder reset: {err}"))
    }

    pub fn write_token(
        &mut self,
        stream: &mut RuntimeStream,
        k: &[f32],
        v: &[f32],
    ) -> Result<usize, String> {
        let cache_position = self.written_len;
        self.write_token_at(stream, cache_position, k, v)?;
        Ok(cache_position)
    }

    pub fn write_token_at(
        &mut self,
        stream: &mut RuntimeStream,
        cache_position: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), String> {
        self.validate_cache_position(cache_position)?;
        if k.len() != self.shape.k_token_elements()? {
            return Err(format!(
                "paged decoder k token length {} does not match expected {}",
                k.len(),
                self.shape.k_token_elements()?
            ));
        }
        if v.len() != self.shape.v_token_elements()? {
            return Err(format!(
                "paged decoder v token length {} does not match expected {}",
                v.len(),
                self.shape.v_token_elements()?
            ));
        }

        self.k_token_buffer
            .copy_from_host(0, &f32s_to_le_bytes(k), Some(stream))
            .map_err(|err| format!("failed to copy paged decoder k token: {err}"))?;
        self.v_token_buffer
            .copy_from_host(0, &f32s_to_le_bytes(v), Some(stream))
            .map_err(|err| format!("failed to copy paged decoder v token: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize paged decoder token input: {err}"))?;

        ullm_runtime_sys::paged_kv_write_f32(
            &self.k_token_buffer,
            &self.v_token_buffer,
            &self.block_table_buffer,
            cache_position,
            self.shape.block_size,
            self.shape.cache_blocks,
            self.shape.kv_heads,
            self.shape.head_dim,
            self.shape.value_dim,
            &mut self.k_cache_buffer,
            &mut self.v_cache_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run paged decoder KV write: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize paged decoder KV write: {err}"))?;
        self.written_len = self.written_len.max(cache_position + 1);
        Ok(())
    }

    pub(crate) fn write_sequence_from_device(
        &mut self,
        stream: &mut RuntimeStream,
        k_sequence: &RuntimeBuffer,
        v_sequence: &RuntimeBuffer,
        token_count: usize,
    ) -> Result<std::ops::Range<usize>, String> {
        if token_count == 0 {
            return Err("paged decoder device KV sequence must contain at least one token".into());
        }
        let start = self.written_len;
        let end = start
            .checked_add(token_count)
            .ok_or_else(|| "paged decoder device KV sequence position overflows".to_string())?;
        self.validate_cache_position(end - 1)?;

        let k_token_bytes = f32_bytes(self.shape.k_token_elements()?);
        let v_token_bytes = f32_bytes(self.shape.v_token_elements()?);
        let expected_k_bytes = k_token_bytes
            .checked_mul(token_count)
            .ok_or_else(|| "paged decoder device K sequence byte size overflows".to_string())?;
        let expected_v_bytes = v_token_bytes
            .checked_mul(token_count)
            .ok_or_else(|| "paged decoder device V sequence byte size overflows".to_string())?;
        validate_device_buffer_bytes(k_sequence, expected_k_bytes, "K sequence")?;
        validate_device_buffer_bytes(v_sequence, expected_v_bytes, "V sequence")?;

        for token_index in 0..token_count {
            let k_offset = k_token_bytes
                .checked_mul(token_index)
                .ok_or_else(|| "paged decoder device K offset overflows".to_string())?;
            let v_offset = v_token_bytes
                .checked_mul(token_index)
                .ok_or_else(|| "paged decoder device V offset overflows".to_string())?;
            self.k_token_buffer
                .copy_from_buffer(0, k_sequence, k_offset, k_token_bytes, Some(&mut *stream))
                .map_err(|err| format!("failed to copy paged decoder device K token: {err}"))?;
            self.v_token_buffer
                .copy_from_buffer(0, v_sequence, v_offset, v_token_bytes, Some(&mut *stream))
                .map_err(|err| format!("failed to copy paged decoder device V token: {err}"))?;
            ullm_runtime_sys::paged_kv_write_f32(
                &self.k_token_buffer,
                &self.v_token_buffer,
                &self.block_table_buffer,
                start + token_index,
                self.shape.block_size,
                self.shape.cache_blocks,
                self.shape.kv_heads,
                self.shape.head_dim,
                self.shape.value_dim,
                &mut self.k_cache_buffer,
                &mut self.v_cache_buffer,
                Some(&mut *stream),
            )
            .map_err(|err| format!("failed to run paged decoder device KV write: {err}"))?;
            self.written_len = start + token_index + 1;
        }
        Ok(start..end)
    }

    /// Appends one resident K/V chunk and attends it over an identity-mapped cached prefix.
    ///
    /// K/V writes and attention are ordered on `stream`; callers own synchronization and must
    /// poison their request if any later enqueue or synchronization fails.
    pub(crate) fn prefill_chunk_from_device(
        &mut self,
        stream: &mut RuntimeStream,
        q: &RuntimeBuffer,
        k: &RuntimeBuffer,
        v: &RuntimeBuffer,
        new_tokens: usize,
        softmax_scale: f32,
        output: &mut RuntimeBuffer,
    ) -> Result<std::ops::Range<usize>, String> {
        if new_tokens == 0 {
            return Err("paged decoder prefill chunk must contain at least one token".into());
        }
        validate_softmax_scale(softmax_scale)?;
        if let Some((logical_block, physical_block)) = self
            .block_table
            .iter()
            .copied()
            .enumerate()
            .find(|(logical_block, physical_block)| *physical_block as usize != *logical_block)
        {
            return Err(format!(
                "paged decoder cached-prefix attention requires an identity block table: block_table[{logical_block}]={physical_block}"
            ));
        }

        let cached_prefix_len = self.written_len;
        let end = cached_prefix_len
            .checked_add(new_tokens)
            .ok_or_else(|| "paged decoder prefill chunk position overflows".to_string())?;
        self.validate_cache_position(end - 1)?;
        validate_device_buffer_bytes(
            q,
            f32_bytes(
                new_tokens
                    .checked_mul(self.shape.q_elements()?)
                    .ok_or_else(|| "paged decoder prefill chunk Q size overflows".to_string())?,
            ),
            "prefill chunk Q",
        )?;
        validate_device_buffer_bytes(
            k,
            f32_bytes(
                new_tokens
                    .checked_mul(self.shape.k_token_elements()?)
                    .ok_or_else(|| "paged decoder prefill chunk K size overflows".to_string())?,
            ),
            "prefill chunk K",
        )?;
        validate_device_buffer_bytes(
            v,
            f32_bytes(
                new_tokens
                    .checked_mul(self.shape.v_token_elements()?)
                    .ok_or_else(|| "paged decoder prefill chunk V size overflows".to_string())?,
            ),
            "prefill chunk V",
        )?;
        validate_device_buffer_bytes(
            output,
            f32_bytes(
                new_tokens
                    .checked_mul(self.shape.output_elements()?)
                    .ok_or_else(|| {
                        "paged decoder prefill chunk output size overflows".to_string()
                    })?,
            ),
            "prefill chunk output",
        )?;

        let written = self.write_sequence_from_device(stream, k, v, new_tokens)?;
        ullm_runtime_sys::cached_prefix_attn_f32_flash2(
            q,
            &self.k_cache_buffer,
            &self.v_cache_buffer,
            cached_prefix_len,
            new_tokens,
            self.shape.q_heads,
            self.shape.kv_heads,
            self.shape.head_dim,
            self.shape.value_dim,
            softmax_scale,
            output,
            Some(stream),
        )
        .map_err(|err| format!("failed to run paged decoder cached-prefix attention: {err}"))?;
        Ok(written)
    }

    pub fn decode_step(
        &mut self,
        stream: &mut RuntimeStream,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        softmax_scale: f32,
    ) -> Result<PagedDecodeStepOutput, String> {
        let device_step = self.decode_step_to_device(stream, q, k, v, softmax_scale)?;
        let output = read_f32_buffer(self.output_buffer(), stream, self.shape.output_elements()?)?;
        Ok(PagedDecodeStepOutput {
            cache_position: device_step.cache_position,
            cache_len: device_step.cache_len,
            output,
        })
    }

    fn decode_step_to_device(
        &mut self,
        stream: &mut RuntimeStream,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        softmax_scale: f32,
    ) -> Result<PagedDecodeDeviceStepOutput, String> {
        self.validate_decode_input(q, softmax_scale)?;
        let cache_position = self.written_len;
        self.validate_cache_position(cache_position)?;
        if k.len() != self.shape.k_token_elements()? {
            return Err(format!(
                "paged decoder k token length {} does not match expected {}",
                k.len(),
                self.shape.k_token_elements()?
            ));
        }
        if v.len() != self.shape.v_token_elements()? {
            return Err(format!(
                "paged decoder v token length {} does not match expected {}",
                v.len(),
                self.shape.v_token_elements()?
            ));
        }

        let q_bytes = f32s_to_le_bytes(q);
        let k_bytes = f32s_to_le_bytes(k);
        let v_bytes = f32s_to_le_bytes(v);
        self.q_buffer
            .copy_from_host(0, &q_bytes, Some(stream))
            .map_err(|err| format!("failed to copy paged decoder q input: {err}"))?;
        self.k_token_buffer
            .copy_from_host(0, &k_bytes, Some(stream))
            .map_err(|err| format!("failed to copy paged decoder k token: {err}"))?;
        self.v_token_buffer
            .copy_from_host(0, &v_bytes, Some(stream))
            .map_err(|err| format!("failed to copy paged decoder v token: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize paged decoder token inputs: {err}"))?;

        ullm_runtime_sys::paged_kv_write_f32(
            &self.k_token_buffer,
            &self.v_token_buffer,
            &self.block_table_buffer,
            cache_position,
            self.shape.block_size,
            self.shape.cache_blocks,
            self.shape.kv_heads,
            self.shape.head_dim,
            self.shape.value_dim,
            &mut self.k_cache_buffer,
            &mut self.v_cache_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run paged decoder KV write: {err}"))?;
        self.written_len = self.written_len.max(cache_position + 1);
        let cache_len = self.written_len;

        ullm_runtime_sys::paged_decode_attn_f32(
            &self.q_buffer,
            &self.k_cache_buffer,
            &self.v_cache_buffer,
            &self.block_table_buffer,
            cache_len,
            self.shape.block_size,
            self.shape.cache_blocks,
            self.shape.q_heads,
            self.shape.kv_heads,
            self.shape.head_dim,
            self.shape.value_dim,
            softmax_scale,
            &mut self.output_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run paged decoder decode attention: {err}"))?;
        Ok(PagedDecodeDeviceStepOutput {
            cache_position,
            cache_len,
        })
    }

    pub(crate) fn decode_step_from_device(
        &mut self,
        stream: &mut RuntimeStream,
        q: &RuntimeBuffer,
        k: &RuntimeBuffer,
        v: &RuntimeBuffer,
        softmax_scale: f32,
    ) -> Result<PagedDecodeDeviceStepOutput, String> {
        validate_softmax_scale(softmax_scale)?;
        validate_device_buffer_bytes(q, f32_bytes(self.shape.q_elements()?), "Q token")?;
        validate_device_buffer_bytes(k, f32_bytes(self.shape.k_token_elements()?), "K token")?;
        validate_device_buffer_bytes(v, f32_bytes(self.shape.v_token_elements()?), "V token")?;

        let cache_position = self.written_len;
        self.validate_cache_position(cache_position)?;
        self.q_buffer
            .copy_from_buffer(
                0,
                q,
                0,
                f32_bytes(self.shape.q_elements()?),
                Some(&mut *stream),
            )
            .map_err(|err| format!("failed to copy paged decoder device Q token: {err}"))?;
        self.k_token_buffer
            .copy_from_buffer(
                0,
                k,
                0,
                f32_bytes(self.shape.k_token_elements()?),
                Some(&mut *stream),
            )
            .map_err(|err| format!("failed to copy paged decoder device K token: {err}"))?;
        self.v_token_buffer
            .copy_from_buffer(
                0,
                v,
                0,
                f32_bytes(self.shape.v_token_elements()?),
                Some(&mut *stream),
            )
            .map_err(|err| format!("failed to copy paged decoder device V token: {err}"))?;

        ullm_runtime_sys::paged_kv_write_f32(
            &self.k_token_buffer,
            &self.v_token_buffer,
            &self.block_table_buffer,
            cache_position,
            self.shape.block_size,
            self.shape.cache_blocks,
            self.shape.kv_heads,
            self.shape.head_dim,
            self.shape.value_dim,
            &mut self.k_cache_buffer,
            &mut self.v_cache_buffer,
            Some(&mut *stream),
        )
        .map_err(|err| format!("failed to run paged decoder device KV write: {err}"))?;
        self.written_len = cache_position + 1;
        let cache_len = self.written_len;
        ullm_runtime_sys::paged_decode_attn_f32(
            &self.q_buffer,
            &self.k_cache_buffer,
            &self.v_cache_buffer,
            &self.block_table_buffer,
            cache_len,
            self.shape.block_size,
            self.shape.cache_blocks,
            self.shape.q_heads,
            self.shape.kv_heads,
            self.shape.head_dim,
            self.shape.value_dim,
            softmax_scale,
            &mut self.output_buffer,
            Some(&mut *stream),
        )
        .map_err(|err| format!("failed to run paged decoder device attention: {err}"))?;
        Ok(PagedDecodeDeviceStepOutput {
            cache_position,
            cache_len,
        })
    }

    pub fn decode_written(
        &mut self,
        stream: &mut RuntimeStream,
        q: &[f32],
        softmax_scale: f32,
    ) -> Result<Vec<f32>, String> {
        self.decode(stream, q, self.written_len, softmax_scale)
    }

    pub fn decode(
        &mut self,
        stream: &mut RuntimeStream,
        q: &[f32],
        cache_len: usize,
        softmax_scale: f32,
    ) -> Result<Vec<f32>, String> {
        self.validate_cache_len(cache_len)?;
        self.validate_decode_input(q, softmax_scale)?;

        self.q_buffer
            .copy_from_host(0, &f32s_to_le_bytes(q), Some(stream))
            .map_err(|err| format!("failed to copy paged decoder q input: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize paged decoder q input: {err}"))?;
        ullm_runtime_sys::paged_decode_attn_f32(
            &self.q_buffer,
            &self.k_cache_buffer,
            &self.v_cache_buffer,
            &self.block_table_buffer,
            cache_len,
            self.shape.block_size,
            self.shape.cache_blocks,
            self.shape.q_heads,
            self.shape.kv_heads,
            self.shape.head_dim,
            self.shape.value_dim,
            softmax_scale,
            &mut self.output_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run paged decoder decode attention: {err}"))?;

        read_f32_buffer(&self.output_buffer, stream, self.shape.output_elements()?)
    }

    pub(crate) fn output_buffer(&self) -> &RuntimeBuffer {
        &self.output_buffer
    }

    pub fn read_cache_to_host(
        &self,
        stream: &mut RuntimeStream,
    ) -> Result<PagedKvCacheReadback, String> {
        let k = read_f32_buffer(&self.k_cache_buffer, stream, self.shape.k_cache_elements()?)?;
        let v = read_f32_buffer(&self.v_cache_buffer, stream, self.shape.v_cache_elements()?)?;
        Ok(PagedKvCacheReadback { k, v })
    }

    fn validate_decode_input(&self, q: &[f32], softmax_scale: f32) -> Result<(), String> {
        if q.len() != self.shape.q_elements()? {
            return Err(format!(
                "paged decoder q length {} does not match expected {}",
                q.len(),
                self.shape.q_elements()?
            ));
        }
        if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
            return Err(
                "paged decoder softmax_scale must be finite and greater than zero".to_string(),
            );
        }
        Ok(())
    }

    fn validate_cache_position(&self, cache_position: usize) -> Result<(), String> {
        if cache_position >= self.shape.physical_tokens()? {
            return Err("paged decoder cache position exceeds physical cache capacity".to_string());
        }
        let block_index = cache_position / self.shape.block_size;
        if block_index >= self.block_table.len() {
            return Err(format!(
                "paged decoder cache position {cache_position} needs block table index {block_index}, but only {} entries exist",
                self.block_table.len()
            ));
        }
        Ok(())
    }

    fn validate_cache_len(&self, cache_len: usize) -> Result<(), String> {
        if cache_len == 0 {
            return Err("paged decoder cache_len must be greater than zero".to_string());
        }
        if cache_len > self.written_len {
            return Err(format!(
                "paged decoder cache_len {cache_len} exceeds written_len {}",
                self.written_len
            ));
        }
        if cache_len > self.shape.physical_tokens()? {
            return Err("paged decoder cache_len exceeds physical cache capacity".to_string());
        }
        let entries = (cache_len - 1) / self.shape.block_size + 1;
        if entries > self.block_table.len() {
            return Err(format!(
                "paged decoder cache_len {cache_len} needs {entries} block table entries, but only {} entries exist",
                self.block_table.len()
            ));
        }
        Ok(())
    }
}

fn validate_device_buffer_bytes(
    buffer: &RuntimeBuffer,
    expected_bytes: usize,
    label: &str,
) -> Result<(), String> {
    let actual_bytes = buffer.size()?;
    if actual_bytes != expected_bytes {
        return Err(format!(
            "paged decoder device {label} byte size mismatch: expected={expected_bytes} actual={actual_bytes}"
        ));
    }
    Ok(())
}

fn validate_softmax_scale(softmax_scale: f32) -> Result<(), String> {
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err("paged decoder softmax_scale must be finite and greater than zero".into());
    }
    Ok(())
}

impl Qwen3SelfAttnDecodeState {
    pub fn new(
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        shape: PagedDecodeShape,
        block_table: Vec<u32>,
        softmax_scale: f32,
    ) -> Result<Self, String> {
        if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
            return Err(
                "Qwen3 self-attn softmax_scale must be finite and greater than zero".to_string(),
            );
        }
        let state = PagedDecodeState::new(context, stream, shape, block_table)?;
        Ok(Self {
            state,
            softmax_scale,
        })
    }

    pub fn step(
        &mut self,
        stream: &mut RuntimeStream,
        q: &[f32],
        k: &[f32],
        v: &[f32],
    ) -> Result<Qwen3SelfAttnDecodeStepOutput, String> {
        let step = self.step_to_device(stream, q, k, v)?;
        let attention_output = read_f32_buffer(
            self.output_buffer(),
            stream,
            self.state.shape.output_elements()?,
        )?;
        Ok(Qwen3SelfAttnDecodeStepOutput {
            cache_position: step.cache_position,
            cache_len: step.cache_len,
            attention_output,
        })
    }

    fn step_to_device(
        &mut self,
        stream: &mut RuntimeStream,
        q: &[f32],
        k: &[f32],
        v: &[f32],
    ) -> Result<PagedDecodeDeviceStepOutput, String> {
        self.state
            .decode_step_to_device(stream, q, k, v, self.softmax_scale)
    }

    fn output_buffer(&self) -> &RuntimeBuffer {
        self.state.output_buffer()
    }

    pub fn shape(&self) -> PagedDecodeShape {
        self.state.shape()
    }

    pub fn written_len(&self) -> usize {
        self.state.written_len()
    }

    pub fn block_table(&self) -> &[u32] {
        self.state.block_table()
    }

    pub fn softmax_scale(&self) -> f32 {
        self.softmax_scale
    }

    pub fn reset(&mut self, stream: &mut RuntimeStream) -> Result<(), String> {
        self.state.reset(stream)
    }

    pub fn read_cache_to_host(
        &self,
        stream: &mut RuntimeStream,
    ) -> Result<PagedKvCacheReadback, String> {
        self.state.read_cache_to_host(stream)
    }
}

impl Qwen3DecoderLayerStepState {
    pub fn new(
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        shape: PagedDecodeShape,
        block_table: Vec<u32>,
        hidden: usize,
        intermediate: usize,
        softmax_scale: f32,
        mlp_epsilon: f32,
    ) -> Result<Self, String> {
        if hidden == 0 {
            return Err("Qwen3 decoder layer hidden size must be greater than zero".to_string());
        }
        if intermediate == 0 {
            return Err(
                "Qwen3 decoder layer intermediate size must be greater than zero".to_string(),
            );
        }
        if !mlp_epsilon.is_finite() || mlp_epsilon <= 0.0 {
            return Err(
                "Qwen3 decoder layer MLP epsilon must be finite and greater than zero".to_string(),
            );
        }
        let block_state = Qwen3SelfAttnBlockStepState::new(
            context,
            stream,
            shape,
            block_table,
            hidden,
            softmax_scale,
        )?;
        let hidden_bytes = f32_bytes(hidden);
        let intermediate_bytes = f32_bytes(intermediate);
        let mut post_normed_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
            format!("failed to allocate Qwen3 decoder layer post normed buffer: {err}")
        })?;
        let mut gate_buffer = context
            .alloc_buffer(intermediate_bytes)
            .map_err(|err| format!("failed to allocate Qwen3 decoder layer gate buffer: {err}"))?;
        let mut up_buffer = context
            .alloc_buffer(intermediate_bytes)
            .map_err(|err| format!("failed to allocate Qwen3 decoder layer up buffer: {err}"))?;
        let mut activated_buffer = context.alloc_buffer(intermediate_bytes).map_err(|err| {
            format!("failed to allocate Qwen3 decoder layer activated buffer: {err}")
        })?;
        let mut mlp_output_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
            format!("failed to allocate Qwen3 decoder layer MLP output buffer: {err}")
        })?;
        let mut layer_output_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
            format!("failed to allocate Qwen3 decoder layer output buffer: {err}")
        })?;
        zero_buffer(&mut post_normed_buffer, Some(stream))?;
        zero_buffer(&mut gate_buffer, Some(stream))?;
        zero_buffer(&mut up_buffer, Some(stream))?;
        zero_buffer(&mut activated_buffer, Some(stream))?;
        zero_buffer(&mut mlp_output_buffer, Some(stream))?;
        zero_buffer(&mut layer_output_buffer, Some(stream))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize Qwen3 decoder layer setup: {err}"))?;
        Ok(Self {
            block_state,
            intermediate,
            mlp_epsilon,
            post_normed_buffer,
            gate_buffer,
            up_buffer,
            activated_buffer,
            mlp_output_buffer,
            layer_output_buffer,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn step_to_device(
        &mut self,
        stream: &mut RuntimeStream,
        o_projection_matrix: &RuntimeBuffer,
        post_norm_weight: &RuntimeBuffer,
        mlp_gate_matrix: &RuntimeBuffer,
        mlp_up_matrix: &RuntimeBuffer,
        mlp_down_matrix: &RuntimeBuffer,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output_gate: Option<&[f32]>,
        residual: &[f32],
    ) -> Result<Qwen3SelfAttnBlockDeviceStepOutput, String> {
        let block_step = self.block_state.step_to_device(
            stream,
            o_projection_matrix,
            q,
            k,
            v,
            output_gate,
            residual,
        )?;
        let hidden = self.block_state.hidden();
        let hidden_bytes = f32_bytes(hidden);
        if post_norm_weight.size()? != hidden_bytes {
            return Err(format!(
                "Qwen3 decoder layer post RMSNorm weight size must match hidden {hidden}"
            ));
        }

        let gate_elements = self.intermediate.checked_mul(hidden).ok_or_else(|| {
            "Qwen3 decoder layer MLP gate matrix element count overflows".to_string()
        })?;
        let gate_bytes = f32_bytes(gate_elements);
        if mlp_gate_matrix.size()? != gate_bytes {
            return Err(format!(
                "Qwen3 decoder layer MLP gate matrix does not match [{},{}]",
                self.intermediate, hidden
            ));
        }
        if mlp_up_matrix.size()? != gate_bytes {
            return Err(format!(
                "Qwen3 decoder layer MLP up matrix does not match [{},{}]",
                self.intermediate, hidden
            ));
        }
        let down_elements = hidden.checked_mul(self.intermediate).ok_or_else(|| {
            "Qwen3 decoder layer MLP down matrix element count overflows".to_string()
        })?;
        let down_bytes = f32_bytes(down_elements);
        if mlp_down_matrix.size()? != down_bytes {
            return Err(format!(
                "Qwen3 decoder layer MLP down matrix does not match [{},{}]",
                hidden, self.intermediate
            ));
        }

        ullm_runtime_sys::rmsnorm_f32(
            self.block_state.block_buffer(),
            post_norm_weight,
            hidden,
            self.mlp_epsilon,
            &mut self.post_normed_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run Qwen3 decoder layer post RMSNorm: {err}"))?;

        ullm_runtime_sys::matvec_f32(
            mlp_gate_matrix,
            &self.post_normed_buffer,
            self.intermediate,
            hidden,
            &mut self.gate_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run Qwen3 decoder layer MLP gate matvec: {err}"))?;
        ullm_runtime_sys::matvec_f32(
            mlp_up_matrix,
            &self.post_normed_buffer,
            self.intermediate,
            hidden,
            &mut self.up_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run Qwen3 decoder layer MLP up matvec: {err}"))?;
        ullm_runtime_sys::silu_mul_f32(
            &self.gate_buffer,
            &self.up_buffer,
            self.intermediate,
            &mut self.activated_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run Qwen3 decoder layer MLP SiLU-mul: {err}"))?;
        ullm_runtime_sys::matvec_f32(
            mlp_down_matrix,
            &self.activated_buffer,
            hidden,
            self.intermediate,
            &mut self.mlp_output_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run Qwen3 decoder layer MLP down matvec: {err}"))?;

        ullm_runtime_sys::add_f32(
            self.block_state.block_buffer(),
            &self.mlp_output_buffer,
            hidden,
            &mut self.layer_output_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run Qwen3 decoder layer MLP residual add: {err}"))?;

        Ok(block_step)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        stream: &mut RuntimeStream,
        o_projection_matrix: &RuntimeBuffer,
        post_norm_weight: &RuntimeBuffer,
        mlp_gate_matrix: &RuntimeBuffer,
        mlp_up_matrix: &RuntimeBuffer,
        mlp_down_matrix: &RuntimeBuffer,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output_gate: Option<&[f32]>,
        residual: &[f32],
    ) -> Result<Qwen3DecoderLayerStepOutput, String> {
        let block_step = self.step_to_device(
            stream,
            o_projection_matrix,
            post_norm_weight,
            mlp_gate_matrix,
            mlp_up_matrix,
            mlp_down_matrix,
            q,
            k,
            v,
            output_gate,
            residual,
        )?;
        let hidden = self.block_state.hidden();
        let cache_position = block_step.cache_position;
        let cache_len = block_step.cache_len;
        let attention_output = self.block_state.read_attention_output(stream)?;
        let attention_projection_input = self
            .block_state
            .read_attention_projection_input(stream, block_step.gated_projection_input)?;
        let projected_output = self.block_state.read_projected_output(stream)?;
        let block_output = self.block_state.read_block_output(stream)?;
        let post_normed = read_f32_buffer(&self.post_normed_buffer, stream, hidden)?;
        let mlp_output = read_f32_buffer(&self.mlp_output_buffer, stream, hidden)?;
        let layer_output = read_f32_buffer(&self.layer_output_buffer, stream, hidden)?;
        Ok(Qwen3DecoderLayerStepOutput {
            cache_position,
            cache_len,
            attention_output,
            attention_projection_input,
            projected_output,
            block_output,
            post_normed,
            mlp_output,
            layer_output,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_output_only(
        &mut self,
        stream: &mut RuntimeStream,
        o_projection_matrix: &RuntimeBuffer,
        post_norm_weight: &RuntimeBuffer,
        mlp_gate_matrix: &RuntimeBuffer,
        mlp_up_matrix: &RuntimeBuffer,
        mlp_down_matrix: &RuntimeBuffer,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output_gate: Option<&[f32]>,
        residual: &[f32],
    ) -> Result<Qwen3DecoderLayerOutputStep, String> {
        let block_step = self.step_to_device(
            stream,
            o_projection_matrix,
            post_norm_weight,
            mlp_gate_matrix,
            mlp_up_matrix,
            mlp_down_matrix,
            q,
            k,
            v,
            output_gate,
            residual,
        )?;
        let layer_output =
            read_f32_buffer(&self.layer_output_buffer, stream, self.block_state.hidden())?;
        Ok(Qwen3DecoderLayerOutputStep {
            cache_position: block_step.cache_position,
            cache_len: block_step.cache_len,
            layer_output,
        })
    }

    pub fn shape(&self) -> PagedDecodeShape {
        self.block_state.shape()
    }

    pub fn hidden(&self) -> usize {
        self.block_state.hidden()
    }

    pub fn intermediate(&self) -> usize {
        self.intermediate
    }

    pub fn mlp_epsilon(&self) -> f32 {
        self.mlp_epsilon
    }

    pub fn written_len(&self) -> usize {
        self.block_state.written_len()
    }

    pub fn block_table(&self) -> &[u32] {
        self.block_state.block_table()
    }

    pub fn reset(&mut self, stream: &mut RuntimeStream) -> Result<(), String> {
        self.block_state.reset(stream)
    }

    pub fn read_cache_to_host(
        &self,
        stream: &mut RuntimeStream,
    ) -> Result<PagedKvCacheReadback, String> {
        self.block_state.read_cache_to_host(stream)
    }
}

impl Qwen3SelfAttnBlockStepState {
    pub fn new(
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        shape: PagedDecodeShape,
        block_table: Vec<u32>,
        hidden: usize,
        softmax_scale: f32,
    ) -> Result<Self, String> {
        if hidden == 0 {
            return Err("Qwen3 self-attn block hidden size must be greater than zero".to_string());
        }
        shape.validate()?;
        let attention_elements = shape.output_elements()?;
        let attention_bytes = f32_bytes(attention_elements);
        let hidden_bytes = f32_bytes(hidden);
        let decode =
            Qwen3SelfAttnDecodeState::new(context, stream, shape, block_table, softmax_scale)?;
        let mut gate_buffer = context
            .alloc_buffer(attention_bytes)
            .map_err(|err| format!("failed to allocate Qwen3 self-attn gate buffer: {err}"))?;
        let mut projection_input_buffer = context.alloc_buffer(attention_bytes).map_err(|err| {
            format!("failed to allocate Qwen3 self-attn projection input buffer: {err}")
        })?;
        let mut projected_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate Qwen3 self-attn projected buffer: {err}"))?;
        let mut residual_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate Qwen3 self-attn residual buffer: {err}"))?;
        let mut block_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate Qwen3 self-attn block buffer: {err}"))?;
        zero_buffer(&mut gate_buffer, Some(stream))?;
        zero_buffer(&mut projection_input_buffer, Some(stream))?;
        zero_buffer(&mut projected_buffer, Some(stream))?;
        zero_buffer(&mut residual_buffer, Some(stream))?;
        zero_buffer(&mut block_buffer, Some(stream))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize Qwen3 self-attn block setup: {err}"))?;
        Ok(Self {
            decode,
            hidden,
            attention_elements,
            gate_buffer,
            projection_input_buffer,
            projected_buffer,
            residual_buffer,
            block_buffer,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        stream: &mut RuntimeStream,
        o_projection_matrix: &RuntimeBuffer,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output_gate: Option<&[f32]>,
        residual: &[f32],
    ) -> Result<Qwen3SelfAttnBlockStepOutput, String> {
        let step =
            self.step_to_device(stream, o_projection_matrix, q, k, v, output_gate, residual)?;
        let attention_output = self.read_attention_output(stream)?;
        let attention_projection_input =
            self.read_attention_projection_input(stream, step.gated_projection_input)?;
        let projected_output = self.read_projected_output(stream)?;
        let block_output = self.read_block_output(stream)?;
        Ok(Qwen3SelfAttnBlockStepOutput {
            cache_position: step.cache_position,
            cache_len: step.cache_len,
            attention_output,
            attention_projection_input,
            projected_output,
            block_output,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn step_to_device(
        &mut self,
        stream: &mut RuntimeStream,
        o_projection_matrix: &RuntimeBuffer,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output_gate: Option<&[f32]>,
        residual: &[f32],
    ) -> Result<Qwen3SelfAttnBlockDeviceStepOutput, String> {
        if let Some(gate) = output_gate {
            if gate.len() != self.attention_elements {
                return Err(format!(
                    "Qwen3 self-attn output gate length {} does not match expected {}",
                    gate.len(),
                    self.attention_elements
                ));
            }
        }
        if residual.len() != self.hidden {
            return Err(format!(
                "Qwen3 self-attn residual length {} does not match hidden {}",
                residual.len(),
                self.hidden
            ));
        }

        let decode_step = self.decode.step_to_device(stream, q, k, v)?;
        let gated_projection_input = output_gate.is_some();
        if let Some(gate) = output_gate {
            self.gate_buffer
                .copy_from_host(0, &f32s_to_le_bytes(gate), Some(stream))
                .map_err(|err| format!("failed to copy Qwen3 self-attn output gate: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize Qwen3 self-attn output gate input: {err}")
            })?;
            ullm_runtime_sys::sigmoid_mul_f32(
                &self.gate_buffer,
                self.decode.output_buffer(),
                self.attention_elements,
                &mut self.projection_input_buffer,
                Some(stream),
            )
            .map_err(|err| format!("failed to run Qwen3 self-attn output gate: {err}"))?;
        }
        let projection_input_buffer = if gated_projection_input {
            &self.projection_input_buffer
        } else {
            self.decode.output_buffer()
        };

        ullm_runtime_sys::matvec_f32(
            o_projection_matrix,
            projection_input_buffer,
            self.hidden,
            self.attention_elements,
            &mut self.projected_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run Qwen3 self-attn o projection: {err}"))?;

        let residual_bytes = f32s_to_le_bytes(residual);
        self.residual_buffer
            .copy_from_host(0, &residual_bytes, Some(stream))
            .map_err(|err| format!("failed to copy Qwen3 self-attn residual: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize Qwen3 self-attn residual input: {err}")
        })?;
        ullm_runtime_sys::add_f32(
            &self.residual_buffer,
            &self.projected_buffer,
            self.hidden,
            &mut self.block_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run Qwen3 self-attn residual add: {err}"))?;

        Ok(Qwen3SelfAttnBlockDeviceStepOutput {
            cache_position: decode_step.cache_position,
            cache_len: decode_step.cache_len,
            gated_projection_input,
        })
    }

    pub fn shape(&self) -> PagedDecodeShape {
        self.decode.shape()
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn attention_elements(&self) -> usize {
        self.attention_elements
    }

    fn attention_output_buffer(&self) -> &RuntimeBuffer {
        self.decode.output_buffer()
    }

    fn projection_input_buffer(&self, gated_projection_input: bool) -> &RuntimeBuffer {
        if gated_projection_input {
            &self.projection_input_buffer
        } else {
            self.decode.output_buffer()
        }
    }

    fn projected_buffer(&self) -> &RuntimeBuffer {
        &self.projected_buffer
    }

    fn block_buffer(&self) -> &RuntimeBuffer {
        &self.block_buffer
    }

    fn read_attention_output(&self, stream: &mut RuntimeStream) -> Result<Vec<f32>, String> {
        read_f32_buffer(
            self.attention_output_buffer(),
            stream,
            self.attention_elements,
        )
    }

    fn read_attention_projection_input(
        &self,
        stream: &mut RuntimeStream,
        gated_projection_input: bool,
    ) -> Result<Vec<f32>, String> {
        read_f32_buffer(
            self.projection_input_buffer(gated_projection_input),
            stream,
            self.attention_elements,
        )
    }

    fn read_projected_output(&self, stream: &mut RuntimeStream) -> Result<Vec<f32>, String> {
        read_f32_buffer(self.projected_buffer(), stream, self.hidden)
    }

    fn read_block_output(&self, stream: &mut RuntimeStream) -> Result<Vec<f32>, String> {
        read_f32_buffer(self.block_buffer(), stream, self.hidden)
    }

    pub fn written_len(&self) -> usize {
        self.decode.written_len()
    }

    pub fn block_table(&self) -> &[u32] {
        self.decode.block_table()
    }

    pub fn softmax_scale(&self) -> f32 {
        self.decode.softmax_scale()
    }

    pub fn reset(&mut self, stream: &mut RuntimeStream) -> Result<(), String> {
        self.decode.reset(stream)
    }

    pub fn read_cache_to_host(
        &self,
        stream: &mut RuntimeStream,
    ) -> Result<PagedKvCacheReadback, String> {
        self.decode.read_cache_to_host(stream)
    }
}

fn validate_block_table(block_table: &[u32], cache_blocks: usize) -> Result<(), String> {
    if block_table.is_empty() {
        return Err("paged decoder block table must not be empty".to_string());
    }
    for (index, block_id) in block_table.iter().copied().enumerate() {
        if block_id as usize >= cache_blocks {
            return Err(format!(
                "paged decoder block_table[{index}]={block_id} exceeds cache_blocks={cache_blocks}"
            ));
        }
    }
    Ok(())
}

fn zero_buffer(
    buffer: &mut RuntimeBuffer,
    mut stream: Option<&mut RuntimeStream>,
) -> Result<(), String> {
    let bytes = buffer.size()?;
    if bytes == 0 {
        return Ok(());
    }
    const ZERO_CHUNK_BYTES: usize = 1 << 20;
    let zero_chunk = vec![0_u8; bytes.min(ZERO_CHUNK_BYTES)];
    let mut offset = 0_usize;
    while offset < bytes {
        let chunk = (bytes - offset).min(zero_chunk.len());
        buffer
            .copy_from_host(offset, &zero_chunk[..chunk], stream.as_deref_mut())
            .map_err(|err| format!("failed to zero paged decoder buffer: {err}"))?;
        offset += chunk;
    }
    Ok(())
}

fn read_f32_buffer(
    buffer: &RuntimeBuffer,
    stream: &mut RuntimeStream,
    elements: usize,
) -> Result<Vec<f32>, String> {
    let bytes = f32_bytes(elements);
    let mut raw = vec![0_u8; bytes];
    buffer
        .copy_to_host(0, &mut raw, Some(stream))
        .map_err(|err| format!("failed to read paged decoder f32 buffer: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize paged decoder readback: {err}"))?;
    Ok(le_bytes_to_f32s(&raw))
}

fn f32_bytes(elements: usize) -> usize {
    elements
        .checked_mul(std::mem::size_of::<f32>())
        .expect("validated f32 byte count overflow")
}

fn f32s_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn u32s_to_le_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn le_bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk size checked")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paged_decode_state_writes_and_decodes_cpu() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let shape = PagedDecodeShape {
            block_size: 2,
            cache_blocks: 4,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 3,
            value_dim: 2,
        };
        let block_table = vec![3_u32, 0_u32];
        let mut state =
            PagedDecodeState::new(&mut context, &mut stream, shape, block_table.clone()).unwrap();
        let cache_len = 3_usize;
        let logical_k = (0..cache_len * shape.kv_heads * shape.head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let logical_v = (0..cache_len * shape.kv_heads * shape.value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        for timestep in 0..cache_len {
            let k_start = timestep * shape.k_token_elements().unwrap();
            let k_end = k_start + shape.k_token_elements().unwrap();
            let v_start = timestep * shape.v_token_elements().unwrap();
            let v_end = v_start + shape.v_token_elements().unwrap();
            state
                .write_token(
                    &mut stream,
                    &logical_k[k_start..k_end],
                    &logical_v[v_start..v_end],
                )
                .unwrap();
            assert_eq!(state.written_len(), timestep + 1);
        }

        let readback = state.read_cache_to_host(&mut stream).unwrap();
        let (expected_k, expected_v) =
            pack_paged_kv_for_test(&logical_k, &logical_v, &block_table, cache_len, shape);
        assert_f32s_close(&readback.k, &expected_k, 1e-6);
        assert_f32s_close(&readback.v, &expected_v, 1e-6);

        let q = (0..shape.q_elements().unwrap())
            .map(|index| (index as f32 - 8.0) / 11.0)
            .collect::<Vec<_>>();
        let softmax_scale = 1.0_f32 / (shape.head_dim as f32).sqrt();
        let output = state
            .decode_written(&mut stream, &q, softmax_scale)
            .unwrap();
        let expected = expected_paged_decode_attn(
            &q,
            &expected_k,
            &expected_v,
            &block_table,
            cache_len,
            shape,
            softmax_scale,
        );
        assert_f32s_close(&output, &expected, 1e-5);
    }

    #[test]
    fn paged_decode_state_decode_step_matches_prefix_decode_cpu() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let shape = PagedDecodeShape {
            block_size: 2,
            cache_blocks: 4,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 3,
            value_dim: 2,
        };
        let block_table = vec![3_u32, 0_u32];
        let mut state =
            PagedDecodeState::new(&mut context, &mut stream, shape, block_table.clone()).unwrap();
        let cache_len = 3_usize;
        let logical_q = (0..cache_len * shape.q_heads * shape.head_dim)
            .map(|index| ((index * 7) as f32 - 11.0) / 19.0)
            .collect::<Vec<_>>();
        let logical_k = (0..cache_len * shape.kv_heads * shape.head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let logical_v = (0..cache_len * shape.kv_heads * shape.value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let softmax_scale = 1.0_f32 / (shape.head_dim as f32).sqrt();
        for timestep in 0..cache_len {
            let q_start = timestep * shape.q_elements().unwrap();
            let q_end = q_start + shape.q_elements().unwrap();
            let k_start = timestep * shape.k_token_elements().unwrap();
            let k_end = k_start + shape.k_token_elements().unwrap();
            let v_start = timestep * shape.v_token_elements().unwrap();
            let v_end = v_start + shape.v_token_elements().unwrap();
            let step = state
                .decode_step(
                    &mut stream,
                    &logical_q[q_start..q_end],
                    &logical_k[k_start..k_end],
                    &logical_v[v_start..v_end],
                    softmax_scale,
                )
                .unwrap();
            assert_eq!(step.cache_position, timestep);
            assert_eq!(step.cache_len, timestep + 1);
            let (expected_k, expected_v) = pack_paged_kv_for_test(
                &logical_k[..k_end],
                &logical_v[..v_end],
                &block_table,
                timestep + 1,
                shape,
            );
            let expected = expected_paged_decode_attn(
                &logical_q[q_start..q_end],
                &expected_k,
                &expected_v,
                &block_table,
                timestep + 1,
                shape,
                softmax_scale,
            );
            assert_f32s_close(&step.output, &expected, 1e-5);
        }
        assert_eq!(state.written_len(), cache_len);
    }

    #[test]
    fn serving_reset_zeroes_multiple_paged_states_before_metadata_commit() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let shape = PagedDecodeShape {
            block_size: 2,
            cache_blocks: 2,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            value_dim: 2,
        };
        let mut states = (0..2)
            .map(|_| {
                PagedDecodeState::new(&mut context, &mut stream, shape, vec![0_u32, 1_u32]).unwrap()
            })
            .collect::<Vec<_>>();
        for state in &mut states {
            state
                .write_token(&mut stream, &[1.0, -2.0], &[3.0, -4.0])
                .unwrap();
            assert_eq!(state.written_len(), 1);
            state.enqueue_serving_reset(&mut stream).unwrap();
            assert_eq!(state.written_len(), 1);
        }
        stream.synchronize().unwrap();
        for state in &mut states {
            state.commit_serving_reset();
            assert_eq!(state.written_len(), 0);
            let cache = state.read_cache_to_host(&mut stream).unwrap();
            assert!(cache.k.iter().all(|value| *value == 0.0));
            assert!(cache.v.iter().all(|value| *value == 0.0));
        }
    }

    #[test]
    fn paged_decode_state_device_sequence_and_step_match_host_path_cpu() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let shape = PagedDecodeShape {
            block_size: 2,
            cache_blocks: 4,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 3,
            value_dim: 2,
        };
        let block_table = vec![3_u32, 0_u32];
        let prompt_len = 2_usize;
        let logical_k = (0..(prompt_len + 1) * shape.k_token_elements().unwrap())
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let logical_v = (0..(prompt_len + 1) * shape.v_token_elements().unwrap())
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let q = (0..shape.q_elements().unwrap())
            .map(|index| ((index * 7) as f32 - 11.0) / 19.0)
            .collect::<Vec<_>>();

        let prompt_k = upload_test_f32_buffer(
            &mut context,
            &mut stream,
            &logical_k[..prompt_len * shape.k_token_elements().unwrap()],
        );
        let prompt_v = upload_test_f32_buffer(
            &mut context,
            &mut stream,
            &logical_v[..prompt_len * shape.v_token_elements().unwrap()],
        );
        let q_buffer = upload_test_f32_buffer(&mut context, &mut stream, &q);
        let decode_k = upload_test_f32_buffer(
            &mut context,
            &mut stream,
            &logical_k[prompt_len * shape.k_token_elements().unwrap()..],
        );
        let decode_v = upload_test_f32_buffer(
            &mut context,
            &mut stream,
            &logical_v[prompt_len * shape.v_token_elements().unwrap()..],
        );

        let mut state =
            PagedDecodeState::new(&mut context, &mut stream, shape, block_table.clone()).unwrap();
        let written = state
            .write_sequence_from_device(&mut stream, &prompt_k, &prompt_v, prompt_len)
            .unwrap();
        assert_eq!(written, 0..prompt_len);
        let step = state
            .decode_step_from_device(
                &mut stream,
                &q_buffer,
                &decode_k,
                &decode_v,
                1.0 / (shape.head_dim as f32).sqrt(),
            )
            .unwrap();
        assert_eq!(step.cache_position, prompt_len);
        assert_eq!(step.cache_len, prompt_len + 1);

        let cache = state.read_cache_to_host(&mut stream).unwrap();
        let (expected_k, expected_v) =
            pack_paged_kv_for_test(&logical_k, &logical_v, &block_table, prompt_len + 1, shape);
        assert_f32s_close(&cache.k, &expected_k, 1e-6);
        assert_f32s_close(&cache.v, &expected_v, 1e-6);
        let output = read_f32_buffer(
            state.output_buffer(),
            &mut stream,
            shape.output_elements().unwrap(),
        )
        .unwrap();
        let expected = expected_paged_decode_attn(
            &q,
            &expected_k,
            &expected_v,
            &block_table,
            prompt_len + 1,
            shape,
            1.0 / (shape.head_dim as f32).sqrt(),
        );
        assert_f32s_close(&output, &expected, 1e-5);
    }

    #[test]
    fn paged_decode_state_prefill_chunk_attends_prefix_and_chunk_causally_cpu() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let shape = PagedDecodeShape {
            block_size: 2,
            cache_blocks: 4,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 3,
            value_dim: 2,
        };
        let block_table = vec![0_u32, 1, 2, 3];
        let prefix_len = 2_usize;
        let new_tokens = 3_usize;
        let total_tokens = prefix_len + new_tokens;
        let logical_k = (0..total_tokens * shape.k_token_elements().unwrap())
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let logical_v = (0..total_tokens * shape.v_token_elements().unwrap())
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let q = (0..new_tokens * shape.q_elements().unwrap())
            .map(|index| ((index * 7) as f32 - 11.0) / 19.0)
            .collect::<Vec<_>>();
        let prefix_k = upload_test_f32_buffer(
            &mut context,
            &mut stream,
            &logical_k[..prefix_len * shape.k_token_elements().unwrap()],
        );
        let prefix_v = upload_test_f32_buffer(
            &mut context,
            &mut stream,
            &logical_v[..prefix_len * shape.v_token_elements().unwrap()],
        );
        let chunk_k = upload_test_f32_buffer(
            &mut context,
            &mut stream,
            &logical_k[prefix_len * shape.k_token_elements().unwrap()..],
        );
        let chunk_v = upload_test_f32_buffer(
            &mut context,
            &mut stream,
            &logical_v[prefix_len * shape.v_token_elements().unwrap()..],
        );
        let q_buffer = upload_test_f32_buffer(&mut context, &mut stream, &q);
        let mut output = context
            .alloc_buffer(f32_bytes(new_tokens * shape.output_elements().unwrap()))
            .unwrap();

        let mut state =
            PagedDecodeState::new(&mut context, &mut stream, shape, block_table.clone()).unwrap();
        assert_eq!(
            state
                .write_sequence_from_device(&mut stream, &prefix_k, &prefix_v, prefix_len)
                .unwrap(),
            0..prefix_len
        );
        let written = state
            .prefill_chunk_from_device(
                &mut stream,
                &q_buffer,
                &chunk_k,
                &chunk_v,
                new_tokens,
                1.0 / (shape.head_dim as f32).sqrt(),
                &mut output,
            )
            .unwrap();
        assert_eq!(written, prefix_len..total_tokens);
        assert_eq!(state.written_len(), total_tokens);

        let cache = state.read_cache_to_host(&mut stream).unwrap();
        let actual = read_f32_buffer(
            &output,
            &mut stream,
            new_tokens * shape.output_elements().unwrap(),
        )
        .unwrap();
        let mut expected = Vec::with_capacity(actual.len());
        for chunk_index in 0..new_tokens {
            let q_start = chunk_index * shape.q_elements().unwrap();
            let q_end = q_start + shape.q_elements().unwrap();
            expected.extend(expected_paged_decode_attn(
                &q[q_start..q_end],
                &cache.k,
                &cache.v,
                &block_table,
                prefix_len + chunk_index + 1,
                shape,
                1.0 / (shape.head_dim as f32).sqrt(),
            ));
        }
        assert_f32s_close(&actual, &expected, 1e-5);
    }

    #[test]
    fn paged_decode_state_prefill_chunk_rejects_nonidentity_table_before_mutation() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let shape = PagedDecodeShape {
            block_size: 2,
            cache_blocks: 4,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 3,
            value_dim: 2,
        };
        let new_tokens = 2_usize;
        let q = upload_test_f32_buffer(
            &mut context,
            &mut stream,
            &vec![0.0; new_tokens * shape.q_elements().unwrap()],
        );
        let k = upload_test_f32_buffer(
            &mut context,
            &mut stream,
            &vec![0.0; new_tokens * shape.k_token_elements().unwrap()],
        );
        let v = upload_test_f32_buffer(
            &mut context,
            &mut stream,
            &vec![0.0; new_tokens * shape.v_token_elements().unwrap()],
        );
        let mut output = context
            .alloc_buffer(f32_bytes(new_tokens * shape.output_elements().unwrap()))
            .unwrap();
        let mut state =
            PagedDecodeState::new(&mut context, &mut stream, shape, vec![1, 0, 2, 3]).unwrap();

        let err = state
            .prefill_chunk_from_device(&mut stream, &q, &k, &v, new_tokens, 1.0, &mut output)
            .unwrap_err();
        assert!(err.contains("identity block table"), "{err}");
        assert_eq!(state.written_len(), 0);
    }

    #[test]
    fn qwen3_self_attn_decode_step_state_step_is_thin_alias() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let shape = PagedDecodeShape {
            block_size: 2,
            cache_blocks: 4,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 3,
            value_dim: 2,
        };
        let block_table = vec![3_u32, 0_u32];
        let mut state = Qwen3SelfAttnDecodeState::new(
            &mut context,
            &mut stream,
            shape,
            block_table.clone(),
            1.0_f32 / (3.0_f32).sqrt(),
        )
        .unwrap();
        let cache_len = 3_usize;
        let softmax_scale = state.softmax_scale();
        let logical_q = (0..cache_len * shape.q_heads * shape.head_dim)
            .map(|index| ((index * 7) as f32 - 11.0) / 19.0)
            .collect::<Vec<_>>();
        let logical_k = (0..cache_len * shape.kv_heads * shape.head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let logical_v = (0..cache_len * shape.kv_heads * shape.value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        for timestep in 0..cache_len {
            let q_start = timestep * shape.q_elements().unwrap();
            let q_end = q_start + shape.q_elements().unwrap();
            let k_start = timestep * shape.k_token_elements().unwrap();
            let k_end = k_start + shape.k_token_elements().unwrap();
            let v_start = timestep * shape.v_token_elements().unwrap();
            let v_end = v_start + shape.v_token_elements().unwrap();
            let step = state
                .step(
                    &mut stream,
                    &logical_q[q_start..q_end],
                    &logical_k[k_start..k_end],
                    &logical_v[v_start..v_end],
                )
                .unwrap();
            assert_eq!(step.cache_position, timestep);
            assert_eq!(step.cache_len, timestep + 1);

            let (expected_k, expected_v) = pack_paged_kv_for_test(
                &logical_k[..k_end],
                &logical_v[..v_end],
                &block_table,
                timestep + 1,
                shape,
            );
            let expected = expected_paged_decode_attn(
                &logical_q[q_start..q_end],
                &expected_k,
                &expected_v,
                &block_table,
                timestep + 1,
                shape,
                softmax_scale,
            );
            assert_f32s_close(&step.attention_output, &expected, 1e-5);
        }
        assert_eq!(state.written_len(), cache_len);

        let readback = state.read_cache_to_host(&mut stream).unwrap();
        let (expected_k, expected_v) =
            pack_paged_kv_for_test(&logical_k, &logical_v, &block_table, cache_len, shape);
        assert_f32s_close(&readback.k, &expected_k, 1e-6);
        assert_f32s_close(&readback.v, &expected_v, 1e-6);
    }

    #[test]
    fn qwen3_self_attn_block_step_state_runs_gate_projection_and_residual_cpu() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let shape = PagedDecodeShape {
            block_size: 1,
            cache_blocks: 2,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            value_dim: 2,
        };
        let hidden = 3_usize;
        let block_table = vec![1_u32, 0_u32];
        let softmax_scale = 1.0_f32 / (shape.head_dim as f32).sqrt();
        let mut state = Qwen3SelfAttnBlockStepState::new(
            &mut context,
            &mut stream,
            shape,
            block_table.clone(),
            hidden,
            softmax_scale,
        )
        .unwrap();
        let attention_elements = shape.output_elements().unwrap();
        let o_matrix = (0..hidden * attention_elements)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let mut o_matrix_buffer = context.alloc_buffer(f32_bytes(o_matrix.len())).unwrap();
        o_matrix_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&o_matrix), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let cache_len = 2_usize;
        let logical_q = (0..cache_len * shape.q_elements().unwrap())
            .map(|index| ((index * 7) as f32 - 11.0) / 19.0)
            .collect::<Vec<_>>();
        let logical_k = (0..cache_len * shape.k_token_elements().unwrap())
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let logical_v = (0..cache_len * shape.v_token_elements().unwrap())
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let gate_sequence = (0..cache_len * attention_elements)
            .map(|index| ((index * 11) as f32 - 13.0) / 23.0)
            .collect::<Vec<_>>();
        let residual_sequence = (0..cache_len * hidden)
            .map(|index| ((index * 13) as f32 - 17.0) / 29.0)
            .collect::<Vec<_>>();

        for timestep in 0..cache_len {
            let q_start = timestep * shape.q_elements().unwrap();
            let q_end = q_start + shape.q_elements().unwrap();
            let k_start = timestep * shape.k_token_elements().unwrap();
            let k_end = k_start + shape.k_token_elements().unwrap();
            let v_start = timestep * shape.v_token_elements().unwrap();
            let v_end = v_start + shape.v_token_elements().unwrap();
            let gate_start = timestep * attention_elements;
            let gate_end = gate_start + attention_elements;
            let residual_start = timestep * hidden;
            let residual_end = residual_start + hidden;
            let step = state
                .step(
                    &mut stream,
                    &o_matrix_buffer,
                    &logical_q[q_start..q_end],
                    &logical_k[k_start..k_end],
                    &logical_v[v_start..v_end],
                    Some(&gate_sequence[gate_start..gate_end]),
                    &residual_sequence[residual_start..residual_end],
                )
                .unwrap();
            assert_eq!(step.cache_position, timestep);
            assert_eq!(step.cache_len, timestep + 1);

            let (expected_k, expected_v) = pack_paged_kv_for_test(
                &logical_k[..k_end],
                &logical_v[..v_end],
                &block_table,
                timestep + 1,
                shape,
            );
            let expected_attention = expected_paged_decode_attn(
                &logical_q[q_start..q_end],
                &expected_k,
                &expected_v,
                &block_table,
                timestep + 1,
                shape,
                softmax_scale,
            );
            let expected_projection_input =
                sigmoid_mul_for_test(&gate_sequence[gate_start..gate_end], &expected_attention);
            let expected_projected = matvec_for_test(
                &o_matrix,
                &expected_projection_input,
                hidden,
                attention_elements,
            );
            let expected_block = add_for_test(
                &residual_sequence[residual_start..residual_end],
                &expected_projected,
            );
            assert_f32s_close(&step.attention_output, &expected_attention, 1e-5);
            assert_f32s_close(
                &step.attention_projection_input,
                &expected_projection_input,
                1e-5,
            );
            assert_f32s_close(&step.projected_output, &expected_projected, 1e-5);
            assert_f32s_close(&step.block_output, &expected_block, 1e-5);
        }
        assert_eq!(state.written_len(), cache_len);
    }

    #[test]
    fn qwen3_self_attn_block_sequence_to_host_f32_runs_cpu() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let shape = PagedDecodeShape {
            block_size: 2,
            cache_blocks: 4,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 3,
            value_dim: 2,
        };
        let hidden = 4_usize;
        let block_table = vec![1_u32, 0_u32];
        let sequence_len = 3_usize;
        let softmax_scale = 1.0_f32 / (shape.head_dim as f32).sqrt();
        let attention_elements = shape.output_elements().unwrap();
        let q_token_elements = shape.q_elements().unwrap();
        let k_token_elements = shape.k_token_elements().unwrap();
        let v_token_elements = shape.v_token_elements().unwrap();

        let o_matrix = (0..hidden * attention_elements)
            .map(|index| ((index * 7) as f32 - 13.0) / 23.0)
            .collect::<Vec<_>>();
        let mut o_matrix_buffer = context.alloc_buffer(f32_bytes(o_matrix.len())).unwrap();
        o_matrix_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&o_matrix), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let q_sequence = (0..sequence_len * q_token_elements)
            .map(|index| ((index * 5) as f32 - 11.0) / 19.0)
            .collect::<Vec<_>>();
        let k_sequence = (0..sequence_len * k_token_elements)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v_sequence = (0..sequence_len * v_token_elements)
            .map(|index| ((index * 9) as f32 - 17.0) / 29.0)
            .collect::<Vec<_>>();
        let residual_sequence = (0..sequence_len * hidden)
            .map(|index| ((index * 11) as f32 - 5.0) / 31.0)
            .collect::<Vec<_>>();
        let output_gate_sequence = (0..sequence_len * attention_elements)
            .map(|index| ((index * 13) as f32 - 17.0) / 37.0)
            .collect::<Vec<_>>();

        let output = qwen3_self_attn_block_sequence_to_host_f32(
            &mut context,
            &mut stream,
            shape,
            &block_table,
            hidden,
            softmax_scale,
            &o_matrix_buffer,
            &q_sequence,
            &k_sequence,
            &v_sequence,
            Some(&output_gate_sequence),
            &residual_sequence,
            sequence_len,
        )
        .unwrap();

        let mut expected_attention_output = Vec::with_capacity(sequence_len * attention_elements);
        let mut expected_attention_projection_input =
            Vec::with_capacity(sequence_len * attention_elements);
        let mut expected_projected = Vec::with_capacity(sequence_len * hidden);
        let mut expected_block = Vec::with_capacity(sequence_len * hidden);
        for timestep in 0..sequence_len {
            let q_start = timestep * q_token_elements;
            let q_end = q_start + q_token_elements;
            let gate_end = (timestep + 1) * attention_elements;
            let residual_end = (timestep + 1) * hidden;
            let k_end = (timestep + 1) * k_token_elements;
            let v_end = (timestep + 1) * v_token_elements;

            let packed = pack_paged_kv_for_test(
                &k_sequence[..k_end],
                &v_sequence[..v_end],
                &block_table,
                timestep + 1,
                shape,
            );
            let expected_attention = expected_paged_decode_attn(
                &q_sequence[q_start..q_end],
                &packed.0,
                &packed.1,
                &block_table,
                timestep + 1,
                shape,
                softmax_scale,
            );
            let gate_start = timestep * attention_elements;
            let gate_slice = &output_gate_sequence[gate_start..gate_end];
            let expected_projection_input = sigmoid_mul_for_test(gate_slice, &expected_attention);
            let expected_step_projected = matvec_for_test(
                &o_matrix,
                &expected_projection_input,
                hidden,
                attention_elements,
            );
            let residual_start = timestep * hidden;
            let residual_slice = &residual_sequence[residual_start..residual_end];
            let expected_step_block = add_for_test(&residual_slice, &expected_step_projected);

            expected_attention_output.extend_from_slice(&expected_attention);
            expected_attention_projection_input.extend_from_slice(&expected_projection_input);
            expected_projected.extend_from_slice(&expected_step_projected);
            expected_block.extend_from_slice(&expected_step_block);
        }

        let expected_cache = pack_paged_kv_cache_for_block_table(
            &k_sequence,
            &v_sequence,
            &block_table,
            sequence_len,
            shape,
        )
        .unwrap();

        assert_f32s_close(&output.attention_output, &expected_attention_output, 1e-5);
        assert_f32s_close(
            &output.attention_projection_input,
            &expected_attention_projection_input,
            1e-5,
        );
        assert_f32s_close(&output.projected_output, &expected_projected, 1e-5);
        assert_f32s_close(&output.block_output, &expected_block, 1e-5);
        assert_f32s_close(&output.paged_cache.k, &expected_cache.k, 1e-6);
        assert_f32s_close(&output.paged_cache.v, &expected_cache.v, 1e-6);
    }

    #[test]
    fn qwen3_decoder_layer_sequence_to_host_f32_runs_cpu() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let shape = PagedDecodeShape {
            block_size: 1,
            cache_blocks: 4,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            value_dim: 2,
        };
        let hidden = 4_usize;
        let intermediate = 3_usize;
        let block_table = vec![1_u32, 0_u32, 2_u32];
        let sequence_len = 3_usize;
        let softmax_scale = 1.0_f32 / (shape.head_dim as f32).sqrt();
        let mlp_epsilon = 1e-5_f32;
        let attention_elements = shape.output_elements().unwrap();
        let q_token_elements = shape.q_elements().unwrap();
        let k_token_elements = shape.k_token_elements().unwrap();
        let v_token_elements = shape.v_token_elements().unwrap();
        let q_matrix = (0..shape.q_elements().unwrap() * hidden)
            .map(|index| ((index * 3) as f32 - 11.0) / 17.0)
            .collect::<Vec<_>>();
        let k_matrix = (0..shape.k_token_elements().unwrap() * hidden)
            .map(|index| ((index * 5) as f32 - 13.0) / 19.0)
            .collect::<Vec<_>>();
        let v_matrix = (0..shape.v_token_elements().unwrap() * hidden)
            .map(|index| ((index * 7) as f32 - 23.0) / 29.0)
            .collect::<Vec<_>>();
        let o_matrix = (0..hidden * attention_elements)
            .map(|index| ((index * 11) as f32 - 13.0) / 31.0)
            .collect::<Vec<_>>();
        let post_norm_weight = (0..hidden)
            .map(|index| ((index * 2) as f32 + 1.0) / 7.0)
            .collect::<Vec<_>>();
        let mlp_gate_matrix = (0..intermediate * hidden)
            .map(|index| ((index * 13) as f32 - 17.0) / 23.0)
            .collect::<Vec<_>>();
        let mlp_up_matrix = (0..intermediate * hidden)
            .map(|index| ((index * 29) as f32 - 31.0) / 37.0)
            .collect::<Vec<_>>();
        let mlp_down_matrix = (0..hidden * intermediate)
            .map(|index| ((index * 41) as f32 - 43.0) / 47.0)
            .collect::<Vec<_>>();

        let mut q_matrix_buffer = context.alloc_buffer(f32_bytes(q_matrix.len())).unwrap();
        let mut k_matrix_buffer = context.alloc_buffer(f32_bytes(k_matrix.len())).unwrap();
        let mut v_matrix_buffer = context.alloc_buffer(f32_bytes(v_matrix.len())).unwrap();
        let mut o_matrix_buffer = context.alloc_buffer(f32_bytes(o_matrix.len())).unwrap();
        let mut post_norm_weight_buffer = context
            .alloc_buffer(f32_bytes(post_norm_weight.len()))
            .unwrap();
        let mut mlp_gate_matrix_buffer = context
            .alloc_buffer(f32_bytes(mlp_gate_matrix.len()))
            .unwrap();
        let mut mlp_up_matrix_buffer = context
            .alloc_buffer(f32_bytes(mlp_up_matrix.len()))
            .unwrap();
        let mut mlp_down_matrix_buffer = context
            .alloc_buffer(f32_bytes(mlp_down_matrix.len()))
            .unwrap();

        q_matrix_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&q_matrix), Some(&mut stream))
            .unwrap();
        k_matrix_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&k_matrix), Some(&mut stream))
            .unwrap();
        v_matrix_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&v_matrix), Some(&mut stream))
            .unwrap();
        o_matrix_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&o_matrix), Some(&mut stream))
            .unwrap();
        post_norm_weight_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&post_norm_weight), Some(&mut stream))
            .unwrap();
        mlp_gate_matrix_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&mlp_gate_matrix), Some(&mut stream))
            .unwrap();
        mlp_up_matrix_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&mlp_up_matrix), Some(&mut stream))
            .unwrap();
        mlp_down_matrix_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&mlp_down_matrix), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let q_sequence = (0..sequence_len * q_token_elements)
            .map(|index| ((index * 2) as f32 - 7.0) / 11.0)
            .collect::<Vec<_>>();
        let k_sequence = (0..sequence_len * k_token_elements)
            .map(|index| ((index * 5) as f32 - 3.0) / 13.0)
            .collect::<Vec<_>>();
        let v_sequence = (0..sequence_len * v_token_elements)
            .map(|index| ((index * 7) as f32 - 5.0) / 17.0)
            .collect::<Vec<_>>();
        let residual_sequence = (0..sequence_len * hidden)
            .map(|index| ((index * 9) as f32 - 2.0) / 29.0)
            .collect::<Vec<_>>();
        let output_gate_sequence = (0..sequence_len * attention_elements)
            .map(|index| ((index * 11) as f32 - 13.0) / 23.0)
            .collect::<Vec<_>>();
        let weights = Qwen3DecoderLayerRuntimeWeights {
            self_attn: Qwen3SelfAttnRuntimeWeights {
                q_rows: shape.q_elements().unwrap(),
                q_cols: hidden,
                k_rows: shape.k_token_elements().unwrap(),
                v_rows: shape.v_token_elements().unwrap(),
                o_rows: hidden,
                o_cols: attention_elements,
                head_dim: shape.head_dim,
                kv_heads: shape.kv_heads,
                value_dim: shape.value_dim,
                q_matrix: q_matrix_buffer,
                k_matrix: k_matrix_buffer,
                v_matrix: v_matrix_buffer,
                o_matrix: o_matrix_buffer,
            },
            post_attention: Qwen3PostAttentionRuntimeWeights {
                hidden,
                intermediate,
                post_norm_weight: post_norm_weight_buffer,
                mlp: Qwen3MlpRuntimeWeights {
                    gate_rows: intermediate,
                    gate_cols: hidden,
                    gate_matrix: mlp_gate_matrix_buffer,
                    up_matrix: mlp_up_matrix_buffer,
                    down_matrix: mlp_down_matrix_buffer,
                },
            },
        };

        let output = qwen3_decoder_layer_sequence_to_host_f32(
            &weights,
            &mut context,
            &mut stream,
            shape,
            &block_table,
            softmax_scale,
            mlp_epsilon,
            &q_sequence,
            &k_sequence,
            &v_sequence,
            Some(&output_gate_sequence),
            &residual_sequence,
            sequence_len,
        )
        .unwrap();

        let mut expected_attention_output = Vec::with_capacity(sequence_len * attention_elements);
        let mut expected_attention_projection_input =
            Vec::with_capacity(sequence_len * attention_elements);
        let mut expected_projected_output = Vec::with_capacity(sequence_len * hidden);
        let mut expected_block_output = Vec::with_capacity(sequence_len * hidden);
        let mut expected_post_normed = Vec::with_capacity(sequence_len * hidden);
        let mut expected_mlp_output = Vec::with_capacity(sequence_len * hidden);
        let mut expected_layer_output = Vec::with_capacity(sequence_len * hidden);
        for timestep in 0..sequence_len {
            let q_start = timestep * q_token_elements;
            let q_end = q_start + q_token_elements;
            let k_end = (timestep + 1) * k_token_elements;
            let v_end = (timestep + 1) * v_token_elements;
            let attention_start = timestep * attention_elements;
            let attention_end = attention_start + attention_elements;
            let residual_start = timestep * hidden;
            let residual_end = residual_start + hidden;

            let (expected_k, expected_v) = pack_paged_kv_for_test(
                &k_sequence[..k_end],
                &v_sequence[..v_end],
                &block_table,
                timestep + 1,
                shape,
            );
            let expected_attention = expected_paged_decode_attn(
                &q_sequence[q_start..q_end],
                &expected_k,
                &expected_v,
                &block_table,
                timestep + 1,
                shape,
                softmax_scale,
            );
            let expected_projection_input = sigmoid_mul_for_test(
                &output_gate_sequence[attention_start..attention_end],
                &expected_attention,
            );
            let expected_projected = matvec_for_test(
                &o_matrix,
                &expected_projection_input,
                hidden,
                attention_elements,
            );
            let expected_block = {
                let residual_slice = &residual_sequence[residual_start..residual_end];
                add_for_test(residual_slice, &expected_projected)
            };
            let expected_post_norm =
                expected_rmsnorm_for_test(&expected_block, &post_norm_weight, mlp_epsilon);
            let expected_mlp_gate =
                matvec_for_test(&mlp_gate_matrix, &expected_post_norm, intermediate, hidden);
            let expected_mlp_up =
                matvec_for_test(&mlp_up_matrix, &expected_post_norm, intermediate, hidden);
            let expected_mlp_activated = silu_mul_for_test(&expected_mlp_gate, &expected_mlp_up);
            let expected_mlp = matvec_for_test(
                &mlp_down_matrix,
                &expected_mlp_activated,
                hidden,
                intermediate,
            );
            let expected_layer = add_for_test(&expected_block, &expected_mlp);

            expected_attention_output.extend_from_slice(&expected_attention);
            expected_attention_projection_input.extend_from_slice(&expected_projection_input);
            expected_projected_output.extend_from_slice(&expected_projected);
            expected_block_output.extend_from_slice(&expected_block);
            expected_post_normed.extend_from_slice(&expected_post_norm);
            expected_mlp_output.extend_from_slice(&expected_mlp);
            expected_layer_output.extend_from_slice(&expected_layer);
        }
        let expected_cache = pack_paged_kv_cache_for_block_table(
            &k_sequence,
            &v_sequence,
            &block_table,
            sequence_len,
            shape,
        )
        .unwrap();

        assert_f32s_close(&output.attention_output, &expected_attention_output, 1e-5);
        assert_f32s_close(
            &output.attention_projection_input,
            &expected_attention_projection_input,
            1e-5,
        );
        assert_f32s_close(&output.projected_output, &expected_projected_output, 1e-5);
        assert_f32s_close(&output.block_output, &expected_block_output, 1e-5);
        assert_f32s_close(&output.post_normed, &expected_post_normed, 1e-5);
        assert_f32s_close(&output.mlp_output, &expected_mlp_output, 1e-5);
        assert_f32s_close(&output.layer_output, &expected_layer_output, 1e-5);
        assert_f32s_close(&output.paged_cache.k, &expected_cache.k, 1e-6);
        assert_f32s_close(&output.paged_cache.v, &expected_cache.v, 1e-6);

        let bad_shape = PagedDecodeShape {
            q_heads: shape.q_heads * 2,
            ..shape
        };
        let err = qwen3_decoder_layer_sequence_to_host_f32(
            &weights,
            &mut context,
            &mut stream,
            bad_shape,
            &block_table,
            softmax_scale,
            mlp_epsilon,
            &q_sequence,
            &k_sequence,
            &v_sequence,
            Some(&output_gate_sequence),
            &residual_sequence,
            sequence_len,
        )
        .unwrap_err();
        assert!(err.contains("decode shape mismatch"), "{err}");
    }

    #[test]
    fn qwen3_decoder_layer_step_state_runs_post_norm_and_mlp_cpu() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let shape = PagedDecodeShape {
            block_size: 2,
            cache_blocks: 4,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 2,
            value_dim: 2,
        };
        let hidden = 4_usize;
        let intermediate = 3_usize;
        let block_table = vec![3_u32, 0_u32];
        let softmax_scale = 1.0_f32 / (shape.head_dim as f32).sqrt();
        let mlp_epsilon = 1e-5_f32;
        let mut state = Qwen3DecoderLayerStepState::new(
            &mut context,
            &mut stream,
            shape,
            block_table.clone(),
            hidden,
            intermediate,
            softmax_scale,
            mlp_epsilon,
        )
        .unwrap();

        let attention_elements = shape.output_elements().unwrap();
        let o_matrix = (0..hidden * attention_elements)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let mut o_matrix_buffer = context.alloc_buffer(f32_bytes(o_matrix.len())).unwrap();
        o_matrix_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&o_matrix), Some(&mut stream))
            .unwrap();

        let post_norm_weight = (0..hidden)
            .map(|index| ((index * 7) as f32 - 3.0) / 11.0)
            .collect::<Vec<_>>();
        let mut post_norm_weight_buffer = context
            .alloc_buffer(f32_bytes(post_norm_weight.len()))
            .unwrap();
        post_norm_weight_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&post_norm_weight), Some(&mut stream))
            .unwrap();

        let mlp_gate_matrix = (0..intermediate * hidden)
            .map(|index| ((index * 11) as f32 - 13.0) / 19.0)
            .collect::<Vec<_>>();
        let mut mlp_gate_matrix_buffer = context
            .alloc_buffer(f32_bytes(mlp_gate_matrix.len()))
            .unwrap();
        mlp_gate_matrix_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&mlp_gate_matrix), Some(&mut stream))
            .unwrap();

        let mlp_up_matrix = (0..intermediate * hidden)
            .map(|index| ((index * 17) as f32 - 23.0) / 29.0)
            .collect::<Vec<_>>();
        let mut mlp_up_matrix_buffer = context
            .alloc_buffer(f32_bytes(mlp_up_matrix.len()))
            .unwrap();
        mlp_up_matrix_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&mlp_up_matrix), Some(&mut stream))
            .unwrap();

        let mlp_down_matrix = (0..hidden * intermediate)
            .map(|index| ((index * 31) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let mut mlp_down_matrix_buffer = context
            .alloc_buffer(f32_bytes(mlp_down_matrix.len()))
            .unwrap();
        mlp_down_matrix_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&mlp_down_matrix), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let cache_len = 3_usize;
        let logical_q = (0..cache_len * shape.q_elements().unwrap())
            .map(|index| ((index * 7) as f32 - 11.0) / 19.0)
            .collect::<Vec<_>>();
        let logical_k = (0..cache_len * shape.k_token_elements().unwrap())
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let logical_v = (0..cache_len * shape.v_token_elements().unwrap())
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();
        let gate_sequence = (0..cache_len * attention_elements)
            .map(|index| ((index * 13) as f32 - 15.0) / 23.0)
            .collect::<Vec<_>>();
        let residual_sequence = (0..cache_len * hidden)
            .map(|index| ((index * 9) as f32 - 10.0) / 31.0)
            .collect::<Vec<_>>();

        for timestep in 0..cache_len {
            let q_start = timestep * shape.q_elements().unwrap();
            let q_end = q_start + shape.q_elements().unwrap();
            let k_start = timestep * shape.k_token_elements().unwrap();
            let k_end = k_start + shape.k_token_elements().unwrap();
            let v_start = timestep * shape.v_token_elements().unwrap();
            let v_end = v_start + shape.v_token_elements().unwrap();
            let gate_start = timestep * attention_elements;
            let gate_end = gate_start + attention_elements;
            let residual_start = timestep * hidden;
            let residual_end = residual_start + hidden;
            let step = state
                .step(
                    &mut stream,
                    &o_matrix_buffer,
                    &post_norm_weight_buffer,
                    &mlp_gate_matrix_buffer,
                    &mlp_up_matrix_buffer,
                    &mlp_down_matrix_buffer,
                    &logical_q[q_start..q_end],
                    &logical_k[k_start..k_end],
                    &logical_v[v_start..v_end],
                    Some(&gate_sequence[gate_start..gate_end]),
                    &residual_sequence[residual_start..residual_end],
                )
                .unwrap();
            assert_eq!(step.cache_position, timestep);
            assert_eq!(step.cache_len, timestep + 1);

            let (expected_k, expected_v) = pack_paged_kv_for_test(
                &logical_k[..k_end],
                &logical_v[..v_end],
                &block_table,
                timestep + 1,
                shape,
            );
            let expected_attention = expected_paged_decode_attn(
                &logical_q[q_start..q_end],
                &expected_k,
                &expected_v,
                &block_table,
                timestep + 1,
                shape,
                softmax_scale,
            );
            let expected_projection_input =
                sigmoid_mul_for_test(&gate_sequence[gate_start..gate_end], &expected_attention);
            let expected_block = {
                let projected = matvec_for_test(
                    &o_matrix,
                    &expected_projection_input,
                    hidden,
                    attention_elements,
                );
                add_for_test(&residual_sequence[residual_start..residual_end], &projected)
            };
            let expected_post_normed =
                expected_rmsnorm_for_test(&expected_block, &post_norm_weight, mlp_epsilon);
            let expected_mlp_gate = matvec_for_test(
                &mlp_gate_matrix,
                &expected_post_normed,
                intermediate,
                hidden,
            );
            let expected_mlp_up =
                matvec_for_test(&mlp_up_matrix, &expected_post_normed, intermediate, hidden);
            let expected_mlp_activated = silu_mul_for_test(&expected_mlp_gate, &expected_mlp_up);
            let expected_mlp_output = matvec_for_test(
                &mlp_down_matrix,
                &expected_mlp_activated,
                hidden,
                intermediate,
            );
            let expected_layer_output = add_for_test(&expected_block, &expected_mlp_output);

            assert_f32s_close(&step.block_output, &expected_block, 1e-5);
            assert_f32s_close(&step.post_normed, &expected_post_normed, 1e-5);
            assert_f32s_close(&step.mlp_output, &expected_mlp_output, 1e-5);
            assert_f32s_close(&step.layer_output, &expected_layer_output, 1e-5);
        }
        assert_eq!(state.written_len(), cache_len);
    }

    #[test]
    fn split_qwen3_self_attn_q_projection_plain_layout() {
        let sequence_len = 2_usize;
        let head_dim = 4_usize;
        let q_heads = 3_usize;
        let q_rows = q_heads * head_dim;
        let hidden = q_rows;
        let mut projected = vec![0.0_f32; sequence_len * q_rows];
        for (index, value) in projected.iter_mut().enumerate() {
            *value = index as f32;
        }

        let split =
            split_qwen3_self_attn_q_projection(&projected, sequence_len, q_rows, hidden, head_dim)
                .unwrap();

        assert_eq!(split.q_heads, q_heads);
        assert_eq!(split.layout, "plain");
        assert!(split.gate.is_none());
        assert_eq!(split.query, projected);
    }

    #[test]
    fn split_qwen3_self_attn_q_projection_qwen35_gated_layout() {
        let sequence_len = 2_usize;
        let head_dim = 4_usize;
        let q_heads = 3_usize;
        let q_rows = q_heads * 2 * head_dim;
        let hidden = q_rows / 2;
        let mut projected = vec![0.0_f32; sequence_len * q_rows];
        for (index, value) in projected.iter_mut().enumerate() {
            *value = (index + 1) as f32;
        }

        let split =
            split_qwen3_self_attn_q_projection(&projected, sequence_len, q_rows, hidden, head_dim)
                .unwrap();

        let mut expected_query = Vec::with_capacity(sequence_len * q_heads * head_dim);
        let mut expected_gate = Vec::with_capacity(sequence_len * q_heads * head_dim);
        for timestep in 0..sequence_len {
            let timestep_start = timestep * q_rows;
            for head in 0..q_heads {
                let head_start = timestep_start + head * 2 * head_dim;
                let query_start = head_start;
                let gate_start = head_start + head_dim;
                expected_query.extend_from_slice(&projected[query_start..query_start + head_dim]);
                expected_gate.extend_from_slice(&projected[gate_start..gate_start + head_dim]);
            }
        }

        assert_eq!(split.q_heads, q_heads);
        assert_eq!(split.layout, "qwen3.5-gated");
        assert_eq!(split.gate.as_deref(), Some(expected_gate.as_slice()));
        assert_eq!(split.query, expected_query);
    }

    #[test]
    fn qwen3_self_attn_project_sequence_to_host_f32_runs_cpu() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let q_rows = 4_usize;
        let q_cols = 3_usize;
        let head_dim = 2_usize;
        let kv_heads = 2_usize;
        let value_dim = 3_usize;
        let o_rows = q_cols;
        let o_cols = 6_usize;
        let sequence_len = 3_usize;

        let q_matrix_host = vec![
            0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
        ];
        let k_matrix_host = vec![
            1.2_f32, -0.2, 0.7, 0.9, -0.5, 1.4, -0.9, 0.8, 0.3, 1.1, -0.7, -1.0,
        ];
        let v_matrix_host = vec![
            0.7_f32, 0.4, -0.1, 0.2, 0.3, 0.9, 1.5, -0.5, -1.1, 0.6, 0.8, -0.2, 0.4, 0.5, -0.7,
            1.3, 0.8, -0.9,
        ];
        let o_matrix_host = vec![0.2_f32; o_rows * o_cols];

        let mut q_matrix = context
            .alloc_buffer(q_matrix_host.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut k_matrix = context
            .alloc_buffer(k_matrix_host.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut v_matrix = context
            .alloc_buffer(v_matrix_host.len() * std::mem::size_of::<f32>())
            .unwrap();
        let mut o_matrix = context
            .alloc_buffer(o_rows * o_cols * std::mem::size_of::<f32>())
            .unwrap();

        q_matrix
            .copy_from_host(0, &f32s_to_le_bytes(&q_matrix_host), Some(&mut stream))
            .unwrap();
        k_matrix
            .copy_from_host(0, &f32s_to_le_bytes(&k_matrix_host), Some(&mut stream))
            .unwrap();
        v_matrix
            .copy_from_host(0, &f32s_to_le_bytes(&v_matrix_host), Some(&mut stream))
            .unwrap();
        o_matrix
            .copy_from_host(0, &f32s_to_le_bytes(&o_matrix_host), Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();

        let weights = Qwen3SelfAttnRuntimeWeights {
            q_rows,
            q_cols,
            k_rows: kv_heads * head_dim,
            v_rows: kv_heads * value_dim,
            o_rows,
            o_cols,
            head_dim,
            kv_heads,
            value_dim,
            q_matrix,
            k_matrix,
            v_matrix,
            o_matrix,
        };

        let input_sequence = (0..sequence_len * q_cols)
            .map(|index| (index as f32 * 0.25) - 0.3)
            .collect::<Vec<_>>();

        let projected = qwen3_self_attn_project_sequence_to_host_f32(
            &mut context,
            &mut stream,
            &weights,
            &input_sequence,
            sequence_len,
        )
        .unwrap();

        let mut expected_q = Vec::with_capacity(sequence_len * q_rows);
        let mut expected_k = Vec::with_capacity(sequence_len * weights.k_rows);
        let mut expected_v = Vec::with_capacity(sequence_len * weights.v_rows);
        for timestep in 0..sequence_len {
            let step_input = &input_sequence[timestep * q_cols..(timestep + 1) * q_cols];
            expected_q.extend(matvec_for_test(&q_matrix_host, step_input, q_rows, q_cols));
            expected_k.extend(matvec_for_test(
                &k_matrix_host,
                step_input,
                weights.k_rows,
                q_cols,
            ));
            expected_v.extend(matvec_for_test(
                &v_matrix_host,
                step_input,
                weights.v_rows,
                q_cols,
            ));
        }

        assert_f32s_close(&projected.q_projected, &expected_q, 1e-5);
        assert_f32s_close(&projected.k_projected, &expected_k, 1e-5);
        assert_f32s_close(&projected.v_projected, &expected_v, 1e-5);
    }

    fn make_qwen3_self_attn_runtime_weights(
        context: &mut RuntimeContext,
        q_rows: usize,
        q_cols: usize,
        head_dim: usize,
        kv_heads: usize,
        value_dim: usize,
        o_cols: usize,
    ) -> Qwen3SelfAttnRuntimeWeights {
        let k_rows = kv_heads
            .checked_mul(head_dim)
            .expect("test k_rows multiplication overflow");
        let v_rows = kv_heads
            .checked_mul(value_dim)
            .expect("test v_rows multiplication overflow");
        let q_matrix_elements = q_rows
            .checked_mul(q_cols)
            .expect("test q matrix element count overflow");
        let k_matrix_elements = k_rows
            .checked_mul(q_cols)
            .expect("test k matrix element count overflow");
        let v_matrix_elements = v_rows
            .checked_mul(q_cols)
            .expect("test v matrix element count overflow");
        let o_matrix_elements = q_cols
            .checked_mul(o_cols)
            .expect("test o matrix element count overflow");
        let mut q_matrix = context
            .alloc_buffer(
                q_matrix_elements
                    .checked_mul(std::mem::size_of::<f32>())
                    .expect("test q matrix byte size overflow"),
            )
            .unwrap();
        let mut k_matrix = context
            .alloc_buffer(
                k_matrix_elements
                    .checked_mul(std::mem::size_of::<f32>())
                    .expect("test k matrix byte size overflow"),
            )
            .unwrap();
        let mut v_matrix = context
            .alloc_buffer(
                v_matrix_elements
                    .checked_mul(std::mem::size_of::<f32>())
                    .expect("test v matrix byte size overflow"),
            )
            .unwrap();
        let mut o_matrix = context
            .alloc_buffer(
                o_matrix_elements
                    .checked_mul(std::mem::size_of::<f32>())
                    .expect("test o matrix byte size overflow"),
            )
            .unwrap();
        q_matrix
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&test_matrix_values(q_matrix_elements, 3)),
                None,
            )
            .unwrap();
        k_matrix
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&test_matrix_values(k_matrix_elements, 5)),
                None,
            )
            .unwrap();
        v_matrix
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&test_matrix_values(v_matrix_elements, 7)),
                None,
            )
            .unwrap();
        o_matrix
            .copy_from_host(
                0,
                &f32s_to_le_bytes(&test_matrix_values(o_matrix_elements, 11)),
                None,
            )
            .unwrap();

        Qwen3SelfAttnRuntimeWeights {
            q_rows,
            q_cols,
            k_rows,
            v_rows,
            o_rows: q_cols,
            o_cols,
            head_dim,
            kv_heads,
            value_dim,
            q_matrix,
            k_matrix,
            v_matrix,
            o_matrix,
        }
    }

    fn test_matrix_values(elements: usize, stride: usize) -> Vec<f32> {
        (0..elements)
            .map(|index| {
                let residue = (index
                    .checked_mul(stride)
                    .and_then(|value| value.checked_add(3))
                    .expect("test matrix value index overflow")
                    % 23) as f32;
                (residue - 11.0_f32) / 13.0_f32
            })
            .collect()
    }

    #[test]
    fn qwen3_self_attn_runtime_shape_plain() {
        let mut context = RuntimeContext::create(0).unwrap();
        let weights = make_qwen3_self_attn_runtime_weights(&mut context, 24, 16, 4, 2, 3, 18);

        let shape = qwen3_self_attn_runtime_shape(&weights).unwrap();

        assert_eq!(shape.hidden, 16);
        assert_eq!(shape.q_heads, 6);
        assert_eq!(shape.kv_heads, 2);
        assert_eq!(shape.head_dim, 4);
        assert_eq!(shape.value_dim, 3);
        assert_eq!(shape.attention_width, 18);
        assert_eq!(shape.q_projection_layout, "plain");
    }

    #[test]
    fn qwen3_headwise_rmsnorm_to_host_f32_runs_cpu() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let head_dim = 4_usize;
        let q_heads = 3_usize;
        let epsilon = 1e-5_f32;
        let input = (0..q_heads * head_dim)
            .map(|index| (index as f32 * 2.0_f32 - 1.0_f32) / 7.0_f32)
            .collect::<Vec<_>>();
        let weight = vec![0.25_f32, 0.5_f32, 0.75_f32, 1.0_f32];

        let output =
            qwen3_headwise_rmsnorm_to_host_f32(&mut context, &mut stream, &input, &weight, epsilon)
                .unwrap();

        let mut expected = Vec::with_capacity(input.len());
        for head_input in input.chunks_exact(head_dim) {
            expected.extend(expected_rmsnorm_for_test(head_input, &weight, epsilon));
        }
        assert_f32s_close(&output, &expected, 1e-5);
    }

    #[test]
    fn qwen3_rope_to_host_f32_runs_cpu() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let sequence_len = 2_usize;
        let heads = 2_usize;
        let head_dim = 4_usize;
        let rotary_dim = 2_usize;
        let position_offset = 1_usize;
        let rope_base = 10000.0_f32;
        let input = (0..sequence_len * heads * head_dim)
            .map(|index| (index as f32 * 3.0_f32 - 5.0_f32) / 11.0_f32)
            .collect::<Vec<_>>();
        let output = qwen3_rope_to_host_f32(
            &mut context,
            &mut stream,
            &input,
            sequence_len,
            heads,
            head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        )
        .unwrap();

        let mut expected = vec![0.0_f32; input.len()];
        let half = rotary_dim / 2;
        for timestep in 0..sequence_len {
            let position = (position_offset + timestep) as f32;
            for head in 0..heads {
                let base = (timestep * heads + head) * head_dim;
                for pair in 0..half {
                    let exponent = (2.0_f32 * pair as f32) / rotary_dim as f32;
                    let theta = position / rope_base.powf(exponent);
                    let c = theta.cos();
                    let s = theta.sin();
                    let first = input[base + pair];
                    let second = input[base + half + pair];
                    expected[base + pair] = first * c - second * s;
                    expected[base + half + pair] = second * c + first * s;
                }
                expected[base + rotary_dim..base + head_dim]
                    .copy_from_slice(&input[base + rotary_dim..base + head_dim]);
            }
        }

        assert_f32s_close(&output, &expected, 1e-6);
    }

    #[test]
    fn qwen3_causal_attn_to_host_f32_runs_cpu() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let sequence_len = 3_usize;
        let q_heads = 4_usize;
        let kv_heads = 2_usize;
        let head_dim = 2_usize;
        let value_dim = 3_usize;
        let softmax_scale = 0.75_f32;
        let q = (0..sequence_len * q_heads * head_dim)
            .map(|index| ((index * 2) as f32 - 3.0) / 11.0_f32)
            .collect::<Vec<_>>();
        let k = (0..sequence_len * kv_heads * head_dim)
            .map(|index| ((index * 5) as f32 - 7.0) / 13.0_f32)
            .collect::<Vec<_>>();
        let v = (0..sequence_len * kv_heads * value_dim)
            .map(|index| ((index * 7) as f32 - 5.0) / 17.0_f32)
            .collect::<Vec<_>>();

        let output = qwen3_causal_attn_to_host_f32(
            &mut context,
            &mut stream,
            &q,
            &k,
            &v,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        )
        .unwrap();

        let mut expected = vec![0.0_f32; sequence_len * q_heads * value_dim];
        let q_per_kv = q_heads / kv_heads;
        for timestep in 0..sequence_len {
            for q_head in 0..q_heads {
                let kv_head = q_head / q_per_kv;
                let q_base = (timestep * q_heads + q_head) * head_dim;
                let mut scores = Vec::with_capacity(timestep + 1);
                for source_timestep in 0..=timestep {
                    let k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                    let score = (0..head_dim)
                        .map(|dim| q[q_base + dim] * k[k_base + dim])
                        .sum::<f32>()
                        * softmax_scale;
                    scores.push(score);
                }
                let max_score = scores
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, |max, score| max.max(score));
                let weights = scores
                    .iter()
                    .map(|score| (*score - max_score).exp())
                    .collect::<Vec<_>>();
                let denominator = weights.iter().sum::<f32>();
                let output_base = (timestep * q_heads + q_head) * value_dim;
                for value in 0..value_dim {
                    let mut weighted = 0.0_f32;
                    for (source_timestep, weight) in weights.iter().enumerate() {
                        let v_index = (source_timestep * kv_heads + kv_head) * value_dim + value;
                        weighted += *weight * v[v_index];
                    }
                    expected[output_base + value] = weighted / denominator;
                }
            }
        }

        assert_f32s_close(&output, &expected, 1e-5);
    }

    #[test]
    fn qwen3_self_attn_prepare_sequence_runtime_f32_runs_cpu() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let sequence_len = 3_usize;
        let q_rows = 6_usize;
        let q_cols = 6_usize;
        let head_dim = 2_usize;
        let kv_heads = 3_usize;
        let value_dim = 4_usize;
        let shape = Qwen3SelfAttnRuntimeShape {
            hidden: q_cols,
            q_heads: q_rows / head_dim,
            kv_heads,
            head_dim,
            value_dim,
            attention_width: (q_rows / head_dim) * value_dim,
            q_projection_layout: "plain",
        };
        let v_rows = kv_heads
            .checked_mul(value_dim)
            .expect("test projected v rows multiplication overflow");
        let o_rows = q_cols;
        let o_cols = shape.attention_width;

        let weights = Qwen3SelfAttnRuntimeWeights {
            q_rows,
            q_cols,
            k_rows: kv_heads * head_dim,
            v_rows,
            o_rows,
            o_cols,
            head_dim,
            kv_heads,
            value_dim,
            q_matrix: context.alloc_buffer(0).unwrap(),
            k_matrix: context.alloc_buffer(0).unwrap(),
            v_matrix: context.alloc_buffer(0).unwrap(),
            o_matrix: context.alloc_buffer(0).unwrap(),
        };

        let q_projected = (0..sequence_len * q_rows)
            .map(|index| (index as f32 * 0.5) - 1.0)
            .collect::<Vec<_>>();
        let k_projected = (0..sequence_len * kv_heads * head_dim)
            .map(|index| (index as f32 * 0.25) + 0.3)
            .collect::<Vec<_>>();
        let v_projected = (0..sequence_len * kv_heads * value_dim)
            .map(|index| (index as f32 * 0.125) - 0.7)
            .collect::<Vec<_>>();

        let q_norm_weight = vec![1.0_f32, 0.75_f32];
        let k_norm_weight = vec![0.5_f32, 1.25_f32];
        let rotary_dim = 2_usize;
        let position_offset = 5_usize;
        let rope_base = 10_000.0_f32;

        let prepared = qwen3_self_attn_prepare_sequence_runtime_f32(
            &mut context,
            &mut stream,
            &weights,
            Qwen3SelfAttnProjectedSequence {
                q_projected: q_projected.clone(),
                k_projected: k_projected.clone(),
                v_projected: v_projected.clone(),
            },
            sequence_len,
            &q_norm_weight,
            &k_norm_weight,
            rotary_dim,
            position_offset,
            rope_base,
        )
        .unwrap();

        assert_eq!(prepared.shape, shape);
        assert_eq!(prepared.q_projection_layout, "plain");
        assert_eq!(prepared.output_gate_layout, "none");
        assert_eq!(prepared.q_gate_elements, 0);
        assert_eq!(prepared.q_query, q_projected);
        assert_eq!(prepared.k_projected, k_projected);

        let mut expected_q_normed = Vec::with_capacity(q_projected.len());
        for head_input in q_projected.chunks_exact(shape.head_dim) {
            expected_q_normed.extend(expected_rmsnorm_for_test(
                head_input,
                &q_norm_weight,
                1e-5_f32,
            ));
        }
        let mut expected_k_normed = Vec::with_capacity(k_projected.len());
        for head_input in k_projected.chunks_exact(shape.head_dim) {
            expected_k_normed.extend(expected_rmsnorm_for_test(
                head_input,
                &k_norm_weight,
                1e-5_f32,
            ));
        }
        let expected_q_rope = expected_rope_for_test(
            &expected_q_normed,
            sequence_len,
            shape.q_heads,
            shape.head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        let expected_k_rope = expected_rope_for_test(
            &expected_k_normed,
            sequence_len,
            shape.kv_heads,
            shape.head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        let expected_attention = expected_causal_attn_for_test(
            &expected_q_rope,
            &expected_k_rope,
            &v_projected,
            sequence_len,
            shape.q_heads,
            shape.kv_heads,
            shape.head_dim,
            shape.value_dim,
            prepared.softmax_scale,
        );

        assert_f32s_close(&prepared.q_normed, &expected_q_normed, 1e-5);
        assert_f32s_close(&prepared.k_normed, &expected_k_normed, 1e-5);
        assert_f32s_close(&prepared.q_rope, &expected_q_rope, 1e-5);
        assert_f32s_close(&prepared.k_rope, &expected_k_rope, 1e-5);
        assert_f32s_close(&prepared.attention_output, &expected_attention, 1e-5);
        assert_eq!(prepared.v_projected, v_projected);
        assert!(prepared.q_gate.is_none());
        assert_eq!(
            prepared.softmax_scale,
            1.0_f32 / (shape.head_dim as f32).sqrt()
        );
    }

    #[test]
    fn qwen3_self_attn_prepare_sequence_runtime_f32_runs_gated_cpu() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let sequence_len = 2_usize;
        let q_cols = 4_usize;
        let head_dim = 2_usize;
        let q_heads = 2_usize;
        let q_rows = q_cols * 2;
        let kv_heads = 1_usize;
        let value_dim = 3_usize;
        let attention_width = q_heads * value_dim;
        let shape = Qwen3SelfAttnRuntimeShape {
            hidden: q_cols,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            attention_width,
            q_projection_layout: "qwen3.5-gated",
        };
        let weights = Qwen3SelfAttnRuntimeWeights {
            q_rows,
            q_cols,
            k_rows: kv_heads * head_dim,
            v_rows: kv_heads * value_dim,
            o_rows: q_cols,
            o_cols: attention_width,
            head_dim,
            kv_heads,
            value_dim,
            q_matrix: context.alloc_buffer(0).unwrap(),
            k_matrix: context.alloc_buffer(0).unwrap(),
            v_matrix: context.alloc_buffer(0).unwrap(),
            o_matrix: context.alloc_buffer(0).unwrap(),
        };

        let q_projected = vec![
            0.10_f32, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, -0.10, -0.20, -0.30, -0.40, -0.50,
            -0.60, -0.70, -0.80,
        ];
        let k_projected = vec![0.25_f32, -0.50, 0.75, -1.00];
        let v_projected = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let q_norm_weight = vec![1.0_f32, 0.5];
        let k_norm_weight = vec![0.75_f32, 1.25];
        let rotary_dim = 2_usize;
        let position_offset = 1_usize;
        let rope_base = 10_000.0_f32;

        let prepared = qwen3_self_attn_prepare_sequence_runtime_f32(
            &mut context,
            &mut stream,
            &weights,
            Qwen3SelfAttnProjectedSequence {
                q_projected: q_projected.clone(),
                k_projected: k_projected.clone(),
                v_projected: v_projected.clone(),
            },
            sequence_len,
            &q_norm_weight,
            &k_norm_weight,
            rotary_dim,
            position_offset,
            rope_base,
        )
        .unwrap();

        let split = split_qwen3_self_attn_q_projection(
            &q_projected,
            sequence_len,
            q_rows,
            q_cols,
            head_dim,
        )
        .unwrap();
        assert_eq!(prepared.shape, shape);
        assert_eq!(prepared.q_projection_layout, "qwen3.5-gated");
        assert_eq!(prepared.output_gate_layout, "runtime-sigmoid");
        assert_eq!(prepared.q_gate_elements, split.gate.as_ref().unwrap().len());
        assert_eq!(prepared.q_gate, split.gate);
        assert_eq!(prepared.q_query, split.query);
        assert_eq!(prepared.k_projected, k_projected);

        let mut expected_q_normed = Vec::with_capacity(prepared.q_query.len());
        for head_input in prepared.q_query.chunks_exact(shape.head_dim) {
            expected_q_normed.extend(expected_rmsnorm_for_test(
                head_input,
                &q_norm_weight,
                1e-5_f32,
            ));
        }
        let mut expected_k_normed = Vec::with_capacity(k_projected.len());
        for head_input in k_projected.chunks_exact(shape.head_dim) {
            expected_k_normed.extend(expected_rmsnorm_for_test(
                head_input,
                &k_norm_weight,
                1e-5_f32,
            ));
        }
        let expected_q_rope = expected_rope_for_test(
            &expected_q_normed,
            sequence_len,
            shape.q_heads,
            shape.head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        let expected_k_rope = expected_rope_for_test(
            &expected_k_normed,
            sequence_len,
            shape.kv_heads,
            shape.head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        let expected_attention = expected_causal_attn_for_test(
            &expected_q_rope,
            &expected_k_rope,
            &v_projected,
            sequence_len,
            shape.q_heads,
            shape.kv_heads,
            shape.head_dim,
            shape.value_dim,
            prepared.softmax_scale,
        );

        assert_f32s_close(&prepared.q_normed, &expected_q_normed, 1e-5);
        assert_f32s_close(&prepared.k_normed, &expected_k_normed, 1e-5);
        assert_f32s_close(&prepared.q_rope, &expected_q_rope, 1e-5);
        assert_f32s_close(&prepared.k_rope, &expected_k_rope, 1e-5);
        assert_f32s_close(&prepared.attention_output, &expected_attention, 1e-5);
        assert_eq!(prepared.v_projected, v_projected);
    }

    #[test]
    fn qwen3_self_attn_prepare_sequence_for_paged_decode_f32_runs_cpu() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let sequence_len = 3_usize;
        let q_rows = 4_usize;
        let q_cols = 4_usize;
        let head_dim = 2_usize;
        let kv_heads = 2_usize;
        let value_dim = 3_usize;
        let shape = Qwen3SelfAttnRuntimeShape {
            hidden: q_cols,
            q_heads: q_rows / head_dim,
            kv_heads,
            head_dim,
            value_dim,
            attention_width: q_rows / head_dim * value_dim,
            q_projection_layout: "plain",
        };
        let weights = make_qwen3_self_attn_runtime_weights(
            &mut context,
            q_rows,
            q_cols,
            head_dim,
            kv_heads,
            value_dim,
            6,
        );

        let residual_sequence = (0..sequence_len * q_cols)
            .map(|index| index as f32 * 0.25 - 0.3)
            .collect::<Vec<_>>();
        let q_norm_weight = vec![1.0_f32, 0.75_f32];
        let k_norm_weight = vec![0.5_f32, 1.25_f32];
        let rotary_dim = 2_usize;
        let position_offset = 1_usize;
        let rope_base = 10_000.0_f32;
        let block_table = vec![3_u32, 0_u32];
        let block_size = 2_usize;
        let cache_blocks = 4_usize;

        let prepared = qwen3_self_attn_prepare_sequence_for_paged_decode_f32(
            &mut context,
            &mut stream,
            &weights,
            residual_sequence.clone(),
            sequence_len,
            &q_norm_weight,
            &k_norm_weight,
            rotary_dim,
            position_offset,
            rope_base,
            &block_table,
            block_size,
            cache_blocks,
        )
        .unwrap();

        assert_eq!(prepared.residual_sequence, residual_sequence);
        assert_eq!(prepared.paged_block_table, block_table);
        assert_eq!(prepared.paged_block_size, block_size);
        assert_eq!(prepared.paged_cache_blocks, cache_blocks);
        assert_eq!(prepared.prepared.shape, shape);
        assert_eq!(prepared.prepared.q_projection_layout, "plain");
        assert_eq!(prepared.prepared.output_gate_layout, "none");
        assert_eq!(prepared.prepared.q_gate_elements, 0);
        assert!(prepared.prepared.q_gate.is_none());
        assert!(
            prepared
                .prepared
                .q_query
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(
            prepared
                .prepared
                .k_projected
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(
            prepared
                .prepared
                .v_projected
                .iter()
                .all(|value| value.is_finite())
        );

        let mut expected_q_normed = Vec::with_capacity(prepared.prepared.q_query.len());
        for head_input in prepared.prepared.q_query.chunks_exact(shape.head_dim) {
            expected_q_normed.extend(expected_rmsnorm_for_test(
                head_input,
                &q_norm_weight,
                1e-5_f32,
            ));
        }
        let mut expected_k_normed = Vec::with_capacity(prepared.prepared.k_projected.len());
        for head_input in prepared.prepared.k_projected.chunks_exact(shape.head_dim) {
            expected_k_normed.extend(expected_rmsnorm_for_test(
                head_input,
                &k_norm_weight,
                1e-5_f32,
            ));
        }
        let expected_q_rope = expected_rope_for_test(
            &expected_q_normed,
            sequence_len,
            shape.q_heads,
            shape.head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        let expected_k_rope = expected_rope_for_test(
            &expected_k_normed,
            sequence_len,
            shape.kv_heads,
            shape.head_dim,
            rotary_dim,
            position_offset,
            rope_base,
        );
        let expected_attention = expected_causal_attn_for_test(
            &expected_q_rope,
            &expected_k_rope,
            &prepared.prepared.v_projected,
            sequence_len,
            shape.q_heads,
            shape.kv_heads,
            shape.head_dim,
            shape.value_dim,
            prepared.prepared.softmax_scale,
        );

        assert_f32s_close(&prepared.prepared.q_normed, &expected_q_normed, 1e-5);
        assert_f32s_close(&prepared.prepared.k_normed, &expected_k_normed, 1e-5);
        assert_f32s_close(&prepared.prepared.q_rope, &expected_q_rope, 1e-5);
        assert_f32s_close(&prepared.prepared.k_rope, &expected_k_rope, 1e-5);
        assert_f32s_close(
            &prepared.prepared.attention_output,
            &expected_attention,
            1e-5,
        );

        let expected_shape = PagedDecodeShape {
            block_size,
            cache_blocks,
            q_heads: shape.q_heads,
            kv_heads: shape.kv_heads,
            head_dim,
            value_dim,
        };
        let expected_pack = pack_paged_kv_cache_for_block_table(
            &prepared.prepared.k_rope,
            &prepared.prepared.v_projected,
            &block_table,
            sequence_len,
            expected_shape,
        )
        .unwrap();

        assert_eq!(prepared.paged_k_cache, expected_pack.k);
        assert_eq!(prepared.paged_v_cache, expected_pack.v);
    }

    #[test]
    fn qwen3_self_attn_prepare_sequence_for_paged_decode_f32_rejects_bad_block_table() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let sequence_len = 3_usize;
        let weights = make_qwen3_self_attn_runtime_weights(&mut context, 24, 16, 4, 2, 3, 18);
        let residual_sequence = vec![0.0_f32; sequence_len * weights.q_cols];
        let q_norm_weight = vec![1.0_f32; 4];
        let k_norm_weight = vec![1.0_f32; 4];
        let err = qwen3_self_attn_prepare_sequence_for_paged_decode_f32(
            &mut context,
            &mut stream,
            &weights,
            residual_sequence,
            sequence_len,
            &q_norm_weight,
            &k_norm_weight,
            4_usize,
            0_usize,
            10_000.0_f32,
            &[0_u32],
            2_usize,
            4_usize,
        )
        .unwrap_err();

        assert!(
            err.contains("block table length"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn qwen3_self_attn_runtime_shape_qwen35_gated() {
        let mut context = RuntimeContext::create(0).unwrap();
        let weights = make_qwen3_self_attn_runtime_weights(&mut context, 16, 8, 4, 1, 5, 10);

        let shape = qwen3_self_attn_runtime_shape(&weights).unwrap();

        assert_eq!(shape.q_heads, 2);
        assert_eq!(shape.kv_heads, 1);
        assert_eq!(shape.value_dim, 5);
        assert_eq!(shape.q_projection_layout, "qwen3.5-gated");
        assert_eq!(shape.attention_width, 10);
    }

    #[test]
    fn qwen3_self_attn_runtime_shape_rejects_non_multiple_heads() {
        let mut context = RuntimeContext::create(0).unwrap();
        let weights = make_qwen3_self_attn_runtime_weights(&mut context, 24, 16, 4, 4, 3, 18);

        let err = qwen3_self_attn_runtime_shape(&weights).unwrap_err();
        assert!(err.contains("multiple of kv_heads"));
    }

    #[test]
    fn qwen3_self_attn_runtime_shape_rejects_o_projection_shape_mismatch() {
        let mut context = RuntimeContext::create(0).unwrap();
        let weights = make_qwen3_self_attn_runtime_weights(&mut context, 16, 16, 4, 2, 3, 11);

        let err = qwen3_self_attn_runtime_shape(&weights).unwrap_err();
        assert!(err.contains("o projection shape mismatch"));
    }

    #[test]
    fn paged_decode_state_rejects_short_block_table() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let shape = PagedDecodeShape {
            block_size: 2,
            cache_blocks: 4,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 3,
            value_dim: 2,
        };
        let mut state =
            PagedDecodeState::new(&mut context, &mut stream, shape, vec![3_u32]).unwrap();
        let k = vec![0.0_f32; shape.k_token_elements().unwrap()];
        let v = vec![0.0_f32; shape.v_token_elements().unwrap()];
        let err = state.write_token_at(&mut stream, 3, &k, &v).unwrap_err();
        assert!(err.contains("block table index"));
    }

    #[test]
    fn paged_decode_state_rejects_unwritten_decode() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let shape = PagedDecodeShape {
            block_size: 2,
            cache_blocks: 4,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 3,
            value_dim: 2,
        };
        let mut state =
            PagedDecodeState::new(&mut context, &mut stream, shape, vec![3_u32, 0_u32]).unwrap();
        let q = vec![0.0_f32; shape.q_elements().unwrap()];
        let err = state.decode(&mut stream, &q, 1, 1.0).unwrap_err();
        assert!(err.contains("written_len"));
        let err = state.decode(&mut stream, &q, 0, 1.0).unwrap_err();
        assert!(err.contains("greater than zero"));

        let k = vec![0.0_f32; shape.k_token_elements().unwrap()];
        let v = vec![0.0_f32; shape.v_token_elements().unwrap()];
        state.write_token(&mut stream, &k, &v).unwrap();
        assert_eq!(state.written_len(), 1);
        state.reset(&mut stream).unwrap();
        assert_eq!(state.written_len(), 0);
        let err = state.decode_written(&mut stream, &q, 1.0).unwrap_err();
        assert!(err.contains("greater than zero"));
    }

    #[test]
    fn pack_paged_kv_cache_for_block_table_packs_expected_layout() {
        let shape = PagedDecodeShape {
            block_size: 2,
            cache_blocks: 4,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            value_dim: 3,
        };
        let logical_k = vec![1.0_f32, 1.1, 2.0, 2.1, 3.0, 3.1];
        let logical_v = vec![10.0_f32, 10.1, 10.2, 20.0, 20.1, 20.2, 30.0, 30.1, 30.2];

        let readback =
            pack_paged_kv_cache_for_block_table(&logical_k, &logical_v, &[3_u32, 0_u32], 3, shape)
                .unwrap();

        let mut expected_k = vec![0.0_f32; shape.k_cache_elements().unwrap()];
        let mut expected_v = vec![0.0_f32; shape.v_cache_elements().unwrap()];
        expected_k[12..14].copy_from_slice(&[1.0, 1.1]);
        expected_k[14..16].copy_from_slice(&[2.0, 2.1]);
        expected_k[0..2].copy_from_slice(&[3.0, 3.1]);
        expected_v[18..21].copy_from_slice(&[10.0, 10.1, 10.2]);
        expected_v[21..24].copy_from_slice(&[20.0, 20.1, 20.2]);
        expected_v[0..3].copy_from_slice(&[30.0, 30.1, 30.2]);

        assert_eq!(readback.k, expected_k);
        assert_eq!(readback.v, expected_v);
    }

    #[test]
    fn pack_paged_kv_cache_for_block_table_rejects_invalid_block_table() {
        let shape = PagedDecodeShape {
            block_size: 2,
            cache_blocks: 4,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 3,
            value_dim: 2,
        };
        let logical_k = vec![0.0_f32; shape.kv_heads * shape.head_dim * 3];
        let logical_v = vec![0.0_f32; shape.kv_heads * shape.value_dim * 3];
        let err = pack_paged_kv_cache_for_block_table(&logical_k, &logical_v, &[4_u32], 3, shape)
            .unwrap_err();
        assert!(err.contains("block table length"));
        let err =
            pack_paged_kv_cache_for_block_table(&logical_k, &logical_v, &[0_u32, 4_u32], 3, shape)
                .unwrap_err();
        assert!(err.contains("exceeds cache_blocks"));
    }

    fn pack_paged_kv_for_test(
        logical_k: &[f32],
        logical_v: &[f32],
        block_table: &[u32],
        cache_len: usize,
        shape: PagedDecodeShape,
    ) -> (Vec<f32>, Vec<f32>) {
        let readback = pack_paged_kv_cache_for_block_table(
            logical_k,
            logical_v,
            block_table,
            cache_len,
            shape,
        )
        .expect("pack_paged_kv_cache_for_block_table helper");
        (readback.k, readback.v)
    }

    fn upload_test_f32_buffer(
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        values: &[f32],
    ) -> RuntimeBuffer {
        let mut buffer = context.alloc_buffer(f32_bytes(values.len())).unwrap();
        buffer
            .copy_from_host(0, &f32s_to_le_bytes(values), Some(stream))
            .unwrap();
        buffer
    }

    fn expected_paged_decode_attn(
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        block_table: &[u32],
        cache_len: usize,
        shape: PagedDecodeShape,
        softmax_scale: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0_f32; shape.q_heads * shape.value_dim];
        let q_per_kv = shape.q_heads / shape.kv_heads;
        for q_head in 0..shape.q_heads {
            let kv_head = q_head / q_per_kv;
            let q_base = q_head * shape.head_dim;
            let mut scores = Vec::with_capacity(cache_len);
            for source_timestep in 0..cache_len {
                let block_index = source_timestep / shape.block_size;
                let block_offset = source_timestep - block_index * shape.block_size;
                let physical_timestep =
                    block_table[block_index] as usize * shape.block_size + block_offset;
                let k_base = (physical_timestep * shape.kv_heads + kv_head) * shape.head_dim;
                let score = (0..shape.head_dim)
                    .map(|dim| q[q_base + dim] * k_cache[k_base + dim])
                    .sum::<f32>()
                    * softmax_scale;
                scores.push(score);
            }
            let max_score = scores
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, |max, score| max.max(score));
            let weights = scores
                .iter()
                .map(|score| (*score - max_score).exp())
                .collect::<Vec<_>>();
            let denominator = weights.iter().sum::<f32>();
            let output_base = q_head * shape.value_dim;
            for value in 0..shape.value_dim {
                let mut weighted = 0.0_f32;
                for (source_timestep, weight) in weights.iter().enumerate() {
                    let block_index = source_timestep / shape.block_size;
                    let block_offset = source_timestep - block_index * shape.block_size;
                    let physical_timestep =
                        block_table[block_index] as usize * shape.block_size + block_offset;
                    let v_index =
                        (physical_timestep * shape.kv_heads + kv_head) * shape.value_dim + value;
                    weighted += *weight * v_cache[v_index];
                }
                output[output_base + value] = weighted / denominator;
            }
        }
        output
    }

    fn assert_f32s_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: actual={actual} expected={expected}"
            );
        }
    }

    fn sigmoid_mul_for_test(gate: &[f32], input: &[f32]) -> Vec<f32> {
        gate.iter()
            .zip(input)
            .map(|(gate, input)| (1.0_f32 / (1.0_f32 + (-gate).exp())) * input)
            .collect()
    }

    fn matvec_for_test(matrix: &[f32], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut output = vec![0.0_f32; rows];
        for row in 0..rows {
            let base = row * cols;
            output[row] = (0..cols).map(|col| matrix[base + col] * input[col]).sum();
        }
        output
    }

    fn expected_rmsnorm_for_test(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
        assert_eq!(input.len(), weight.len());
        let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
        let inv_sqrt = 1.0_f32 / (mean_square + epsilon).sqrt();
        input
            .iter()
            .zip(weight)
            .map(|(value, weight)| value * inv_sqrt * weight)
            .collect()
    }

    fn silu_mul_for_test(gate: &[f32], up: &[f32]) -> Vec<f32> {
        gate.iter()
            .zip(up)
            .map(|(gate, up)| {
                let sigmoid = 1.0_f32 / (1.0_f32 + (-gate).exp());
                gate * sigmoid * up
            })
            .collect()
    }

    fn expected_rope_for_test(
        input: &[f32],
        sequence_len: usize,
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        position_offset: usize,
        rope_base: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0_f32; input.len()];
        let half = rotary_dim / 2;
        for timestep in 0..sequence_len {
            let position = (position_offset + timestep) as f32;
            for head in 0..heads {
                let base = (timestep * heads + head) * head_dim;
                for pair_dim in 0..half {
                    let exponent = (2.0 * pair_dim as f32) / rotary_dim as f32;
                    let theta = position / rope_base.powf(exponent);
                    let c = theta.cos();
                    let s = theta.sin();
                    let first = input[base + pair_dim];
                    let second = input[base + half + pair_dim];
                    output[base + pair_dim] = first * c - second * s;
                    output[base + half + pair_dim] = second * c + first * s;
                }
                let start = base + rotary_dim;
                let end = base + head_dim;
                output[start..end].copy_from_slice(&input[start..end]);
            }
        }
        output
    }

    fn expected_causal_attn_for_test(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        sequence_len: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0_f32; sequence_len * q_heads * value_dim];
        let q_per_kv = q_heads / kv_heads;
        for timestep in 0..sequence_len {
            for q_head in 0..q_heads {
                let kv_head = q_head / q_per_kv;
                let q_base = (timestep * q_heads + q_head) * head_dim;
                let mut scores = Vec::with_capacity(timestep + 1);
                for source_timestep in 0..=timestep {
                    let k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                    let score = (0..head_dim)
                        .map(|dim| q[q_base + dim] * k[k_base + dim])
                        .sum::<f32>()
                        * softmax_scale;
                    scores.push(score);
                }
                let max_score = scores
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, |max_score, score| max_score.max(score));
                let weights = scores
                    .iter()
                    .map(|score| (*score - max_score).exp())
                    .collect::<Vec<_>>();
                let denominator = weights.iter().sum::<f32>();
                let output_base = (timestep * q_heads + q_head) * value_dim;
                for value in 0..value_dim {
                    let mut weighted = 0.0_f32;
                    for (source_timestep, weight) in weights.iter().enumerate() {
                        let v_index = (source_timestep * kv_heads + kv_head) * value_dim + value;
                        weighted += *weight * v[v_index];
                    }
                    output[output_base + value] = weighted / denominator;
                }
            }
        }
        output
    }

    fn add_for_test(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        lhs.iter().zip(rhs).map(|(lhs, rhs)| lhs + rhs).collect()
    }
}
