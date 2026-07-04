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
    pub block_output: Vec<f32>,
    pub post_normed: Vec<f32>,
    pub mlp_output: Vec<f32>,
    pub layer_output: Vec<f32>,
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
    attention_buffer: RuntimeBuffer,
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
    post_norm_input_buffer: RuntimeBuffer,
    gate_buffer: RuntimeBuffer,
    up_buffer: RuntimeBuffer,
    activated_buffer: RuntimeBuffer,
    mlp_output_buffer: RuntimeBuffer,
    block_output_buffer: RuntimeBuffer,
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

    pub fn read_cache_to_host(
        &mut self,
        stream: &mut RuntimeStream,
    ) -> Result<PagedKvCacheReadback, String> {
        self.step_state.read_cache_to_host(stream)
    }
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

    pub fn decode_step(
        &mut self,
        stream: &mut RuntimeStream,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        softmax_scale: f32,
    ) -> Result<PagedDecodeStepOutput, String> {
        self.validate_decode_input(q, softmax_scale)?;
        let cache_position = self.write_token(stream, k, v)?;
        let cache_len = self.written_len;
        let output = self.decode(stream, q, cache_len, softmax_scale)?;
        Ok(PagedDecodeStepOutput {
            cache_position,
            cache_len,
            output,
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
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize paged decoder decode attention: {err}")
        })?;

        read_f32_buffer(&self.output_buffer, stream, self.shape.output_elements()?)
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
        let step = self
            .state
            .decode_step(stream, q, k, v, self.softmax_scale)?;
        Ok(Qwen3SelfAttnDecodeStepOutput {
            cache_position: step.cache_position,
            cache_len: step.cache_len,
            attention_output: step.output,
        })
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
        let mut post_norm_input_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
            format!("failed to allocate Qwen3 decoder layer post norm input buffer: {err}")
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
        let mut block_output_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
            format!("failed to allocate Qwen3 decoder layer block output buffer: {err}")
        })?;
        let mut layer_output_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
            format!("failed to allocate Qwen3 decoder layer output buffer: {err}")
        })?;
        zero_buffer(&mut post_normed_buffer, Some(stream))?;
        zero_buffer(&mut post_norm_input_buffer, Some(stream))?;
        zero_buffer(&mut gate_buffer, Some(stream))?;
        zero_buffer(&mut up_buffer, Some(stream))?;
        zero_buffer(&mut activated_buffer, Some(stream))?;
        zero_buffer(&mut mlp_output_buffer, Some(stream))?;
        zero_buffer(&mut block_output_buffer, Some(stream))?;
        zero_buffer(&mut layer_output_buffer, Some(stream))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize Qwen3 decoder layer setup: {err}"))?;
        Ok(Self {
            block_state,
            intermediate,
            mlp_epsilon,
            post_normed_buffer,
            post_norm_input_buffer,
            gate_buffer,
            up_buffer,
            activated_buffer,
            mlp_output_buffer,
            block_output_buffer,
            layer_output_buffer,
        })
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
        let block_step =
            self.block_state
                .step(stream, o_projection_matrix, q, k, v, output_gate, residual)?;

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

        self.post_norm_input_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&block_step.block_output), Some(stream))
            .map_err(|err| {
                format!("failed to copy Qwen3 decoder layer attention block output: {err}")
            })?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize Qwen3 decoder layer attention block output copy: {err}")
        })?;
        ullm_runtime_sys::rmsnorm_f32(
            &self.post_norm_input_buffer,
            post_norm_weight,
            hidden,
            self.mlp_epsilon,
            &mut self.post_normed_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run Qwen3 decoder layer post RMSNorm: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize Qwen3 decoder layer post RMSNorm: {err}")
        })?;

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
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize Qwen3 decoder layer MLP gate/up matvec: {err}")
        })?;
        ullm_runtime_sys::silu_mul_f32(
            &self.gate_buffer,
            &self.up_buffer,
            self.intermediate,
            &mut self.activated_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run Qwen3 decoder layer MLP SiLU-mul: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize Qwen3 decoder layer MLP SiLU-mul: {err}")
        })?;
        ullm_runtime_sys::matvec_f32(
            mlp_down_matrix,
            &self.activated_buffer,
            hidden,
            self.intermediate,
            &mut self.mlp_output_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run Qwen3 decoder layer MLP down matvec: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize Qwen3 decoder layer MLP down matvec: {err}")
        })?;

        self.block_output_buffer
            .copy_from_host(0, &f32s_to_le_bytes(&block_step.block_output), Some(stream))
            .map_err(|err| format!("failed to copy Qwen3 decoder layer block output: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize Qwen3 decoder layer block output copy: {err}")
        })?;
        ullm_runtime_sys::add_f32(
            &self.block_output_buffer,
            &self.mlp_output_buffer,
            hidden,
            &mut self.layer_output_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run Qwen3 decoder layer MLP residual add: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize Qwen3 decoder layer MLP residual add: {err}")
        })?;

        let post_normed = read_f32_buffer(&self.post_normed_buffer, stream, hidden)?;
        let mlp_output = read_f32_buffer(&self.mlp_output_buffer, stream, hidden)?;
        let layer_output = read_f32_buffer(&self.layer_output_buffer, stream, hidden)?;
        Ok(Qwen3DecoderLayerStepOutput {
            cache_position: block_step.cache_position,
            cache_len: block_step.cache_len,
            block_output: block_step.block_output,
            post_normed,
            mlp_output,
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
        let mut attention_buffer = context
            .alloc_buffer(attention_bytes)
            .map_err(|err| format!("failed to allocate Qwen3 self-attn attention buffer: {err}"))?;
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
        zero_buffer(&mut attention_buffer, Some(stream))?;
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
            attention_buffer,
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

        let decode_step = self.decode.step(stream, q, k, v)?;
        let attention_bytes = f32s_to_le_bytes(&decode_step.attention_output);
        self.attention_buffer
            .copy_from_host(0, &attention_bytes, Some(stream))
            .map_err(|err| format!("failed to copy Qwen3 self-attn attention output: {err}"))?;
        if let Some(gate) = output_gate {
            self.gate_buffer
                .copy_from_host(0, &f32s_to_le_bytes(gate), Some(stream))
                .map_err(|err| format!("failed to copy Qwen3 self-attn output gate: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize Qwen3 self-attn output gate input: {err}")
            })?;
            ullm_runtime_sys::sigmoid_mul_f32(
                &self.gate_buffer,
                &self.attention_buffer,
                self.attention_elements,
                &mut self.projection_input_buffer,
                Some(stream),
            )
            .map_err(|err| format!("failed to run Qwen3 self-attn output gate: {err}"))?;
        } else {
            self.projection_input_buffer
                .copy_from_host(0, &attention_bytes, Some(stream))
                .map_err(|err| format!("failed to copy Qwen3 self-attn projection input: {err}"))?;
        }
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize Qwen3 self-attn projection input: {err}")
        })?;

        ullm_runtime_sys::matvec_f32(
            o_projection_matrix,
            &self.projection_input_buffer,
            self.hidden,
            self.attention_elements,
            &mut self.projected_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run Qwen3 self-attn o projection: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize Qwen3 self-attn o projection: {err}"))?;

        self.residual_buffer
            .copy_from_host(0, &f32s_to_le_bytes(residual), Some(stream))
            .map_err(|err| format!("failed to copy Qwen3 self-attn residual: {err}"))?;
        ullm_runtime_sys::add_f32(
            &self.residual_buffer,
            &self.projected_buffer,
            self.hidden,
            &mut self.block_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run Qwen3 self-attn residual add: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize Qwen3 self-attn residual add: {err}"))?;

        let attention_projection_input = read_f32_buffer(
            &self.projection_input_buffer,
            stream,
            self.attention_elements,
        )?;
        let projected_output = read_f32_buffer(&self.projected_buffer, stream, self.hidden)?;
        let block_output = read_f32_buffer(&self.block_buffer, stream, self.hidden)?;
        Ok(Qwen3SelfAttnBlockStepOutput {
            cache_position: decode_step.cache_position,
            cache_len: decode_step.cache_len,
            attention_output: decode_step.attention_output,
            attention_projection_input,
            projected_output,
            block_output,
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
        let q_matrix = context.alloc_buffer(0).unwrap();
        let k_matrix = context.alloc_buffer(0).unwrap();
        let v_matrix = context.alloc_buffer(0).unwrap();
        let o_matrix = context.alloc_buffer(0).unwrap();

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

    #[allow(clippy::too_many_arguments)]
    fn pack_paged_kv_for_test(
        logical_k: &[f32],
        logical_v: &[f32],
        block_table: &[u32],
        cache_len: usize,
        shape: PagedDecodeShape,
    ) -> (Vec<f32>, Vec<f32>) {
        let physical_tokens = shape.physical_tokens().unwrap();
        let mut k_cache = vec![0.0_f32; physical_tokens * shape.kv_heads * shape.head_dim];
        let mut v_cache = vec![0.0_f32; physical_tokens * shape.kv_heads * shape.value_dim];
        for timestep in 0..cache_len {
            let logical_block = timestep / shape.block_size;
            let block_offset = timestep - logical_block * shape.block_size;
            let physical_timestep =
                block_table[logical_block] as usize * shape.block_size + block_offset;
            let k_src = timestep * shape.kv_heads * shape.head_dim;
            let k_dst = physical_timestep * shape.kv_heads * shape.head_dim;
            k_cache[k_dst..k_dst + shape.kv_heads * shape.head_dim]
                .copy_from_slice(&logical_k[k_src..k_src + shape.kv_heads * shape.head_dim]);
            let v_src = timestep * shape.kv_heads * shape.value_dim;
            let v_dst = physical_timestep * shape.kv_heads * shape.value_dim;
            v_cache[v_dst..v_dst + shape.kv_heads * shape.value_dim]
                .copy_from_slice(&logical_v[v_src..v_src + shape.kv_heads * shape.value_dim]);
        }
        (k_cache, v_cache)
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

    fn add_for_test(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        lhs.iter().zip(rhs).map(|(lhs, rhs)| lhs + rhs).collect()
    }
}
