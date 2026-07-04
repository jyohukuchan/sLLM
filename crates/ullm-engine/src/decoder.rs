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
}
