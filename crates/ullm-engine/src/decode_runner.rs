// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Scheduler-facing decode runner utilities.
//!
//! This module keeps scheduler request ids and runtime decode state together
//! without making the low-level decoder module depend on scheduler ownership.

use std::collections::{BTreeMap, BTreeSet};

use crate::decoder::{
    PagedDecodeShape, PagedKvCacheReadback, Qwen3DecoderLayerRuntime,
    Qwen3DecoderLayerRuntimeWeights, Qwen3DecoderLayerStepOutput, Qwen3SelfAttnDecodeState,
    Qwen3SelfAttnDecodeStepOutput,
};
use crate::scheduler::{RequestId, SchedulerDecodeRequest, SchedulerState};
use ullm_runtime_sys::{RuntimeContext, RuntimeStream};

#[derive(Debug)]
struct Qwen3SelfAttnRequestDecodeState {
    block_table: Vec<u32>,
    state: Qwen3SelfAttnDecodeState,
}

struct Qwen3DecoderLayerRequestDecodeState<'weights> {
    block_table: Vec<u32>,
    runtime: Qwen3DecoderLayerRuntime<'weights>,
}

#[derive(Debug, Clone, Copy)]
pub struct Qwen3SelfAttnDecodeBatchInput<'a> {
    pub request_id: RequestId,
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3SelfAttnDecodeBatchOutput {
    pub request_id: RequestId,
    pub cache_position: usize,
    pub cache_len: usize,
    pub attention_output: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct Qwen3DecoderLayerDecodeBatchInput<'a> {
    pub request_id: RequestId,
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub output_gate: Option<&'a [f32]>,
    pub residual: &'a [f32],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3DecoderLayerDecodeBatchOutput {
    pub request_id: RequestId,
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

#[derive(Debug, Default)]
pub struct Qwen3SelfAttnRequestDecodeRunner {
    states: BTreeMap<RequestId, Qwen3SelfAttnRequestDecodeState>,
}

#[derive(Default)]
pub struct Qwen3DecoderLayerRequestDecodeRunner<'weights> {
    states: BTreeMap<RequestId, Qwen3DecoderLayerRequestDecodeState<'weights>>,
}

#[derive(Default)]
pub struct Qwen3DecoderLayerStackRequestDecodeRunner<'weights> {
    layers: Vec<Qwen3DecoderLayerRequestDecodeRunner<'weights>>,
}

impl Qwen3SelfAttnRequestDecodeRunner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    pub fn contains_request(&self, request_id: RequestId) -> bool {
        self.states.contains_key(&request_id)
    }

    pub fn insert_request(
        &mut self,
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        request_id: RequestId,
        shape: PagedDecodeShape,
        block_table: Vec<u32>,
        softmax_scale: f32,
    ) -> Result<(), String> {
        if self.states.contains_key(&request_id) {
            return Err(format!(
                "Qwen3 self-attn decode runner already has request {:?}",
                request_id
            ));
        }
        let state = Qwen3SelfAttnDecodeState::new(
            context,
            stream,
            shape,
            block_table.clone(),
            softmax_scale,
        )
        .map_err(|err| {
            format!(
                "failed to create Qwen3 self-attn decode runner state for {:?}: {err}",
                request_id
            )
        })?;
        self.states.insert(
            request_id,
            Qwen3SelfAttnRequestDecodeState { block_table, state },
        );
        Ok(())
    }

    pub fn remove_request(&mut self, request_id: RequestId) -> bool {
        self.states.remove(&request_id).is_some()
    }

    pub fn run_prefill_step(
        &mut self,
        stream: &mut RuntimeStream,
        input: Qwen3SelfAttnDecodeBatchInput<'_>,
    ) -> Result<Qwen3SelfAttnDecodeBatchOutput, String> {
        let slot = self.states.get_mut(&input.request_id).ok_or_else(|| {
            format!(
                "Qwen3 self-attn decode runner has no request {:?}",
                input.request_id
            )
        })?;
        let step = slot
            .state
            .step(stream, input.q, input.k, input.v)
            .map_err(|err| {
                format!(
                    "failed to run Qwen3 self-attn prefill step for {:?}: {err}",
                    input.request_id
                )
            })?;
        Ok(batch_output_from_step(input.request_id, step))
    }

    pub fn run_ready_batch(
        &mut self,
        stream: &mut RuntimeStream,
        scheduler: &mut SchedulerState,
        ready_batch: &[SchedulerDecodeRequest],
        inputs: &[Qwen3SelfAttnDecodeBatchInput<'_>],
    ) -> Result<Vec<Qwen3SelfAttnDecodeBatchOutput>, String> {
        validate_batch_inputs(ready_batch, inputs)?;
        for request in ready_batch {
            let active = scheduler
                .active_request(request.request.id)
                .ok_or_else(|| format!("request {:?} is not active", request.request.id))?;
            if active.cached_tokens != request.cached_tokens
                || active.generated_tokens != request.generated_tokens
            {
                return Err(format!(
                    "ready decode request {:?} is stale: scheduler cached/generated={}/{} batch cached/generated={}/{}",
                    request.request.id,
                    active.cached_tokens,
                    active.generated_tokens,
                    request.cached_tokens,
                    request.generated_tokens
                ));
            }
            let slot = self.states.get(&request.request.id).ok_or_else(|| {
                format!(
                    "Qwen3 self-attn decode runner has no request {:?}",
                    request.request.id
                )
            })?;
            if slot.block_table != request.allocation.blocks {
                return Err(format!(
                    "request {:?} runner block table {:?} does not match scheduler allocation {:?}",
                    request.request.id, slot.block_table, request.allocation.blocks
                ));
            }
            if slot.state.written_len() != request.cache_position {
                return Err(format!(
                    "request {:?} runner written_len {} does not match ready cache_position {}",
                    request.request.id,
                    slot.state.written_len(),
                    request.cache_position
                ));
            }
        }

        let mut outputs = Vec::with_capacity(ready_batch.len());
        for request in ready_batch {
            let input = inputs
                .iter()
                .find(|input| input.request_id == request.request.id)
                .ok_or_else(|| format!("missing decode input for {:?}", request.request.id))?;
            let slot = self.states.get_mut(&request.request.id).ok_or_else(|| {
                format!(
                    "Qwen3 self-attn decode runner has no request {:?}",
                    request.request.id
                )
            })?;
            let step = slot
                .state
                .step(stream, input.q, input.k, input.v)
                .map_err(|err| {
                    format!(
                        "failed to run Qwen3 self-attn decode step for {:?}: {err}",
                        request.request.id
                    )
                })?;
            if step.cache_position != request.cache_position {
                return Err(format!(
                    "request {:?} step cache_position {} does not match ready cache_position {}",
                    request.request.id, step.cache_position, request.cache_position
                ));
            }
            if step.cache_len != request.next_cache_len {
                return Err(format!(
                    "request {:?} step cache_len {} does not match ready next_cache_len {}",
                    request.request.id, step.cache_len, request.next_cache_len
                ));
            }
            outputs.push(batch_output_from_step(request.request.id, step));
        }

        for request in ready_batch {
            scheduler
                .advance_decode(request.request.id)
                .map_err(|err| {
                    format!("failed to advance request {:?}: {err}", request.request.id)
                })?;
        }
        Ok(outputs)
    }

    pub fn read_cache_to_host(
        &self,
        request_id: RequestId,
        stream: &mut RuntimeStream,
    ) -> Result<PagedKvCacheReadback, String> {
        let slot = self.states.get(&request_id).ok_or_else(|| {
            format!(
                "Qwen3 self-attn decode runner has no request {:?}",
                request_id
            )
        })?;
        slot.state.read_cache_to_host(stream)
    }

    pub fn written_len(&self, request_id: RequestId) -> Option<usize> {
        self.states
            .get(&request_id)
            .map(|slot| slot.state.written_len())
    }

    pub fn block_table(&self, request_id: RequestId) -> Option<&[u32]> {
        self.states
            .get(&request_id)
            .map(|slot| slot.block_table.as_slice())
    }
}

impl<'weights> Qwen3DecoderLayerRequestDecodeRunner<'weights> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    pub fn contains_request(&self, request_id: RequestId) -> bool {
        self.states.contains_key(&request_id)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn insert_request(
        &mut self,
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        request_id: RequestId,
        weights: &'weights Qwen3DecoderLayerRuntimeWeights,
        shape: PagedDecodeShape,
        block_table: Vec<u32>,
        softmax_scale: f32,
        mlp_epsilon: f32,
    ) -> Result<(), String> {
        if self.states.contains_key(&request_id) {
            return Err(format!(
                "Qwen3 decoder layer decode runner already has request {:?}",
                request_id
            ));
        }
        let runtime = Qwen3DecoderLayerRuntime::new(
            context,
            stream,
            weights,
            shape,
            block_table.clone(),
            softmax_scale,
            mlp_epsilon,
        )
        .map_err(|err| {
            format!(
                "failed to create Qwen3 decoder layer decode runner state for {:?}: {err}",
                request_id
            )
        })?;
        self.states.insert(
            request_id,
            Qwen3DecoderLayerRequestDecodeState {
                block_table,
                runtime,
            },
        );
        Ok(())
    }

    pub fn remove_request(&mut self, request_id: RequestId) -> bool {
        self.states.remove(&request_id).is_some()
    }

    pub fn run_prefill_step(
        &mut self,
        stream: &mut RuntimeStream,
        input: Qwen3DecoderLayerDecodeBatchInput<'_>,
    ) -> Result<Qwen3DecoderLayerDecodeBatchOutput, String> {
        let slot = self.states.get_mut(&input.request_id).ok_or_else(|| {
            format!(
                "Qwen3 decoder layer decode runner has no request {:?}",
                input.request_id
            )
        })?;
        let step = slot
            .runtime
            .step(
                stream,
                input.q,
                input.k,
                input.v,
                input.output_gate,
                input.residual,
            )
            .map_err(|err| {
                format!(
                    "failed to run Qwen3 decoder layer prefill step for {:?}: {err}",
                    input.request_id
                )
            })?;
        Ok(layer_batch_output_from_step(input.request_id, step))
    }

    pub fn run_ready_batch(
        &mut self,
        stream: &mut RuntimeStream,
        scheduler: &mut SchedulerState,
        ready_batch: &[SchedulerDecodeRequest],
        inputs: &[Qwen3DecoderLayerDecodeBatchInput<'_>],
    ) -> Result<Vec<Qwen3DecoderLayerDecodeBatchOutput>, String> {
        let outputs =
            self.run_ready_batch_without_advance(stream, scheduler, ready_batch, inputs)?;
        for request in ready_batch {
            scheduler
                .advance_decode(request.request.id)
                .map_err(|err| {
                    format!("failed to advance request {:?}: {err}", request.request.id)
                })?;
        }
        Ok(outputs)
    }

    pub fn run_ready_batch_without_advance(
        &mut self,
        stream: &mut RuntimeStream,
        scheduler: &SchedulerState,
        ready_batch: &[SchedulerDecodeRequest],
        inputs: &[Qwen3DecoderLayerDecodeBatchInput<'_>],
    ) -> Result<Vec<Qwen3DecoderLayerDecodeBatchOutput>, String> {
        self.validate_ready_batch_without_advance(scheduler, ready_batch, inputs)?;

        let mut outputs = Vec::with_capacity(ready_batch.len());
        for request in ready_batch {
            let input = inputs
                .iter()
                .find(|input| input.request_id == request.request.id)
                .ok_or_else(|| format!("missing decode input for {:?}", request.request.id))?;
            let slot = self.states.get_mut(&request.request.id).ok_or_else(|| {
                format!(
                    "Qwen3 decoder layer decode runner has no request {:?}",
                    request.request.id
                )
            })?;
            let step = slot
                .runtime
                .step(
                    stream,
                    input.q,
                    input.k,
                    input.v,
                    input.output_gate,
                    input.residual,
                )
                .map_err(|err| {
                    format!(
                        "failed to run Qwen3 decoder layer decode step for {:?}: {err}",
                        request.request.id
                    )
                })?;
            if step.cache_position != request.cache_position {
                return Err(format!(
                    "request {:?} step cache_position {} does not match ready cache_position {}",
                    request.request.id, step.cache_position, request.cache_position
                ));
            }
            if step.cache_len != request.next_cache_len {
                return Err(format!(
                    "request {:?} step cache_len {} does not match ready next_cache_len {}",
                    request.request.id, step.cache_len, request.next_cache_len
                ));
            }
            outputs.push(layer_batch_output_from_step(request.request.id, step));
        }
        Ok(outputs)
    }

    fn validate_ready_batch_without_advance(
        &self,
        scheduler: &SchedulerState,
        ready_batch: &[SchedulerDecodeRequest],
        inputs: &[Qwen3DecoderLayerDecodeBatchInput<'_>],
    ) -> Result<(), String> {
        validate_layer_batch_inputs(ready_batch, inputs)?;
        for request in ready_batch {
            let active = scheduler
                .active_request(request.request.id)
                .ok_or_else(|| format!("request {:?} is not active", request.request.id))?;
            if active.cached_tokens != request.cached_tokens
                || active.generated_tokens != request.generated_tokens
            {
                return Err(format!(
                    "ready decode request {:?} is stale: scheduler cached/generated={}/{} batch cached/generated={}/{}",
                    request.request.id,
                    active.cached_tokens,
                    active.generated_tokens,
                    request.cached_tokens,
                    request.generated_tokens
                ));
            }
            let slot = self.states.get(&request.request.id).ok_or_else(|| {
                format!(
                    "Qwen3 decoder layer decode runner has no request {:?}",
                    request.request.id
                )
            })?;
            if slot.block_table != request.allocation.blocks {
                return Err(format!(
                    "request {:?} runner block table {:?} does not match scheduler allocation {:?}",
                    request.request.id, slot.block_table, request.allocation.blocks
                ));
            }
            if slot.runtime.written_len() != request.cache_position {
                return Err(format!(
                    "request {:?} runner written_len {} does not match ready cache_position {}",
                    request.request.id,
                    slot.runtime.written_len(),
                    request.cache_position
                ));
            }
        }
        Ok(())
    }

    pub fn read_cache_to_host(
        &self,
        request_id: RequestId,
        stream: &mut RuntimeStream,
    ) -> Result<PagedKvCacheReadback, String> {
        let slot = self.states.get(&request_id).ok_or_else(|| {
            format!(
                "Qwen3 decoder layer decode runner has no request {:?}",
                request_id
            )
        })?;
        slot.runtime.read_cache_to_host(stream)
    }

    pub fn written_len(&self, request_id: RequestId) -> Option<usize> {
        self.states
            .get(&request_id)
            .map(|slot| slot.runtime.written_len())
    }

    pub fn block_table(&self, request_id: RequestId) -> Option<&[u32]> {
        self.states
            .get(&request_id)
            .map(|slot| slot.block_table.as_slice())
    }
}

impl<'weights> Qwen3DecoderLayerStackRequestDecodeRunner<'weights> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn push_layer(&mut self) -> usize {
        let layer_index = self.layers.len();
        self.layers
            .push(Qwen3DecoderLayerRequestDecodeRunner::new());
        layer_index
    }

    #[allow(clippy::too_many_arguments)]
    pub fn insert_request(
        &mut self,
        layer_index: usize,
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        request_id: RequestId,
        weights: &'weights Qwen3DecoderLayerRuntimeWeights,
        shape: PagedDecodeShape,
        block_table: Vec<u32>,
        softmax_scale: f32,
        mlp_epsilon: f32,
    ) -> Result<(), String> {
        self.layer_mut(layer_index)?
            .insert_request(
                context,
                stream,
                request_id,
                weights,
                shape,
                block_table,
                softmax_scale,
                mlp_epsilon,
            )
            .map_err(|err| {
                format!(
                    "failed to insert request {:?} into decoder layer {layer_index}: {err}",
                    request_id
                )
            })
    }

    pub fn run_prefill_step(
        &mut self,
        layer_index: usize,
        stream: &mut RuntimeStream,
        input: Qwen3DecoderLayerDecodeBatchInput<'_>,
    ) -> Result<Qwen3DecoderLayerDecodeBatchOutput, String> {
        self.layer_mut(layer_index)?
            .run_prefill_step(stream, input)
            .map_err(|err| format!("failed to run decoder layer {layer_index} prefill: {err}"))
    }

    /// Runs one scheduler ready decode batch through every registered layer.
    ///
    /// `layer_inputs` must have one input slice per stack layer, in layer order.
    /// Each slice must contain exactly one input for every request in
    /// `ready_batch`. The method validates every layer before executing any
    /// decode step, runs each layer without advancing the scheduler, and then
    /// advances the scheduler batch once after all layers have succeeded.
    pub fn run_ready_batch_across_layers(
        &mut self,
        stream: &mut RuntimeStream,
        scheduler: &mut SchedulerState,
        ready_batch: &[SchedulerDecodeRequest],
        layer_inputs: &[&[Qwen3DecoderLayerDecodeBatchInput<'_>]],
    ) -> Result<Vec<Vec<Qwen3DecoderLayerDecodeBatchOutput>>, String> {
        if layer_inputs.len() != self.layers.len() {
            return Err(format!(
                "decoder layer stack has {} layers but {} layer input batches were provided",
                self.layers.len(),
                layer_inputs.len()
            ));
        }
        if self.layers.is_empty() && !ready_batch.is_empty() {
            return Err(
                "decoder layer stack cannot advance a non-empty batch with no layers".to_string(),
            );
        }

        for (layer_index, (runner, inputs)) in
            self.layers.iter().zip(layer_inputs.iter()).enumerate()
        {
            runner
                .validate_ready_batch_without_advance(scheduler, ready_batch, inputs)
                .map_err(|err| {
                    format!(
                        "decoder layer {layer_index} rejected ready batch before stack run: {err}"
                    )
                })?;
        }

        let mut outputs_by_layer = Vec::with_capacity(self.layers.len());
        for (layer_index, (runner, inputs)) in
            self.layers.iter_mut().zip(layer_inputs.iter()).enumerate()
        {
            let outputs = runner
                .run_ready_batch_without_advance(stream, scheduler, ready_batch, inputs)
                .map_err(|err| {
                    format!("failed to run decoder layer {layer_index} ready batch: {err}")
                })?;
            outputs_by_layer.push(outputs);
        }
        scheduler
            .advance_decode_batch(ready_batch)
            .map_err(|err| format!("failed to advance decoder layer stack ready batch: {err}"))?;
        Ok(outputs_by_layer)
    }

    pub fn read_layer_cache_to_host(
        &self,
        layer_index: usize,
        request_id: RequestId,
        stream: &mut RuntimeStream,
    ) -> Result<PagedKvCacheReadback, String> {
        self.layer(layer_index)?
            .read_cache_to_host(request_id, stream)
            .map_err(|err| {
                format!(
                    "failed to read decoder layer {layer_index} cache for {:?}: {err}",
                    request_id
                )
            })
    }

    pub fn written_len(&self, layer_index: usize, request_id: RequestId) -> Result<usize, String> {
        self.layer(layer_index)?
            .written_len(request_id)
            .ok_or_else(|| {
                format!(
                    "decoder layer {layer_index} has no written_len for request {:?}",
                    request_id
                )
            })
    }

    pub fn block_table(&self, layer_index: usize, request_id: RequestId) -> Result<&[u32], String> {
        self.layer(layer_index)?
            .block_table(request_id)
            .ok_or_else(|| {
                format!(
                    "decoder layer {layer_index} has no block table for request {:?}",
                    request_id
                )
            })
    }

    fn layer(
        &self,
        layer_index: usize,
    ) -> Result<&Qwen3DecoderLayerRequestDecodeRunner<'weights>, String> {
        self.layers
            .get(layer_index)
            .ok_or_else(|| format!("decoder layer index {layer_index} is out of bounds"))
    }

    fn layer_mut(
        &mut self,
        layer_index: usize,
    ) -> Result<&mut Qwen3DecoderLayerRequestDecodeRunner<'weights>, String> {
        self.layers
            .get_mut(layer_index)
            .ok_or_else(|| format!("decoder layer index {layer_index} is out of bounds"))
    }
}

fn batch_output_from_step(
    request_id: RequestId,
    step: Qwen3SelfAttnDecodeStepOutput,
) -> Qwen3SelfAttnDecodeBatchOutput {
    Qwen3SelfAttnDecodeBatchOutput {
        request_id,
        cache_position: step.cache_position,
        cache_len: step.cache_len,
        attention_output: step.attention_output,
    }
}

fn layer_batch_output_from_step(
    request_id: RequestId,
    step: Qwen3DecoderLayerStepOutput,
) -> Qwen3DecoderLayerDecodeBatchOutput {
    Qwen3DecoderLayerDecodeBatchOutput {
        request_id,
        cache_position: step.cache_position,
        cache_len: step.cache_len,
        attention_output: step.attention_output,
        attention_projection_input: step.attention_projection_input,
        projected_output: step.projected_output,
        block_output: step.block_output,
        post_normed: step.post_normed,
        mlp_output: step.mlp_output,
        layer_output: step.layer_output,
    }
}

fn validate_batch_inputs(
    ready_batch: &[SchedulerDecodeRequest],
    inputs: &[Qwen3SelfAttnDecodeBatchInput<'_>],
) -> Result<(), String> {
    if ready_batch.len() != inputs.len() {
        return Err(format!(
            "ready decode batch has {} requests but {} inputs were provided",
            ready_batch.len(),
            inputs.len()
        ));
    }
    let mut ready_ids = BTreeSet::new();
    for request in ready_batch {
        if !ready_ids.insert(request.request.id) {
            return Err(format!(
                "ready decode batch contains duplicate request {:?}",
                request.request.id
            ));
        }
    }
    let mut input_ids = BTreeSet::new();
    for input in inputs {
        if !input_ids.insert(input.request_id) {
            return Err(format!(
                "decode batch inputs contain duplicate request {:?}",
                input.request_id
            ));
        }
        if !ready_ids.contains(&input.request_id) {
            return Err(format!(
                "decode batch input {:?} does not correspond to a ready request",
                input.request_id
            ));
        }
    }
    for request_id in ready_ids {
        if !input_ids.contains(&request_id) {
            return Err(format!("missing decode input for {:?}", request_id));
        }
    }
    Ok(())
}

fn validate_layer_batch_inputs(
    ready_batch: &[SchedulerDecodeRequest],
    inputs: &[Qwen3DecoderLayerDecodeBatchInput<'_>],
) -> Result<(), String> {
    if ready_batch.len() != inputs.len() {
        return Err(format!(
            "ready decode batch has {} requests but {} inputs were provided",
            ready_batch.len(),
            inputs.len()
        ));
    }
    let mut ready_ids = BTreeSet::new();
    for request in ready_batch {
        if !ready_ids.insert(request.request.id) {
            return Err(format!(
                "ready decode batch contains duplicate request {:?}",
                request.request.id
            ));
        }
    }
    let mut input_ids = BTreeSet::new();
    for input in inputs {
        if !input_ids.insert(input.request_id) {
            return Err(format!(
                "decode batch inputs contain duplicate request {:?}",
                input.request_id
            ));
        }
        if !ready_ids.contains(&input.request_id) {
            return Err(format!(
                "decode batch input {:?} does not correspond to a ready request",
                input.request_id
            ));
        }
    }
    for request_id in ready_ids {
        if !input_ids.contains(&request_id) {
            return Err(format!("missing decode input for {:?}", request_id));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::{
        Qwen3MlpRuntimeWeights, Qwen3PostAttentionRuntimeWeights, Qwen3SelfAttnRuntimeWeights,
        pack_paged_kv_cache_for_block_table, qwen3_decoder_layer_sequence_to_host_f32,
    };
    use crate::scheduler::Request;
    use ullm_runtime_sys::RuntimeBuffer;

    fn assert_f32s_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff <= tolerance,
                "index {index} actual={actual} expected={expected} diff={diff} tolerance={tolerance}"
            );
        }
    }

    fn f32_bytes(elements: usize) -> usize {
        elements * std::mem::size_of::<f32>()
    }

    fn f32s_to_le_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn f32_buffer(
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

    fn make_layer_weights(
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        shape: PagedDecodeShape,
        hidden: usize,
        intermediate: usize,
    ) -> Qwen3DecoderLayerRuntimeWeights {
        let q_matrix = (0..shape.q_elements().unwrap() * hidden)
            .map(|index| ((index * 3) as f32 - 11.0) / 17.0)
            .collect::<Vec<_>>();
        let k_matrix = (0..shape.k_token_elements().unwrap() * hidden)
            .map(|index| ((index * 5) as f32 - 13.0) / 19.0)
            .collect::<Vec<_>>();
        let v_matrix = (0..shape.v_token_elements().unwrap() * hidden)
            .map(|index| ((index * 7) as f32 - 23.0) / 29.0)
            .collect::<Vec<_>>();
        let o_matrix = (0..hidden * shape.output_elements().unwrap())
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

        let q_matrix = f32_buffer(context, stream, &q_matrix);
        let k_matrix = f32_buffer(context, stream, &k_matrix);
        let v_matrix = f32_buffer(context, stream, &v_matrix);
        let o_matrix = f32_buffer(context, stream, &o_matrix);
        let post_norm_weight = f32_buffer(context, stream, &post_norm_weight);
        let mlp_gate_matrix = f32_buffer(context, stream, &mlp_gate_matrix);
        let mlp_up_matrix = f32_buffer(context, stream, &mlp_up_matrix);
        let mlp_down_matrix = f32_buffer(context, stream, &mlp_down_matrix);
        stream.synchronize().unwrap();

        Qwen3DecoderLayerRuntimeWeights {
            self_attn: Qwen3SelfAttnRuntimeWeights {
                q_rows: shape.q_elements().unwrap(),
                q_cols: hidden,
                k_rows: shape.k_token_elements().unwrap(),
                v_rows: shape.v_token_elements().unwrap(),
                o_rows: hidden,
                o_cols: shape.output_elements().unwrap(),
                head_dim: shape.head_dim,
                kv_heads: shape.kv_heads,
                value_dim: shape.value_dim,
                q_matrix,
                k_matrix,
                v_matrix,
                o_matrix,
            },
            post_attention: Qwen3PostAttentionRuntimeWeights {
                hidden,
                intermediate,
                post_norm_weight,
                mlp: Qwen3MlpRuntimeWeights {
                    gate_rows: intermediate,
                    gate_cols: hidden,
                    gate_matrix: mlp_gate_matrix,
                    up_matrix: mlp_up_matrix,
                    down_matrix: mlp_down_matrix,
                },
            },
        }
    }

    #[test]
    fn qwen3_self_attn_request_decode_runner_runs_ready_batch_cpu() {
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
        let softmax_scale = 1.0_f32 / (shape.head_dim as f32).sqrt();
        let mut scheduler = SchedulerState::with_block_size(4, 2);
        scheduler.enqueue(Request::new(11, 1, 1));
        let mut scheduled = scheduler
            .pop_prefill_batch_with_allocation(1)
            .expect("allocation should succeed");
        let allocation = scheduled.remove(0).allocation;

        let mut runner = Qwen3SelfAttnRequestDecodeRunner::new();
        runner
            .insert_request(
                &mut context,
                &mut stream,
                allocation.request_id,
                shape,
                allocation.blocks.clone(),
                softmax_scale,
            )
            .expect("runner insert should succeed");
        assert_eq!(runner.len(), 1);
        assert!(runner.contains_request(RequestId(11)));

        let q = (0..2 * shape.q_heads * shape.head_dim)
            .map(|index| ((index * 7) as f32 - 11.0) / 19.0)
            .collect::<Vec<_>>();
        let k = (0..2 * shape.kv_heads * shape.head_dim)
            .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v = (0..2 * shape.kv_heads * shape.value_dim)
            .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
            .collect::<Vec<_>>();

        runner
            .run_prefill_step(
                &mut stream,
                Qwen3SelfAttnDecodeBatchInput {
                    request_id: RequestId(11),
                    q: &q[..shape.q_elements().unwrap()],
                    k: &k[..shape.k_token_elements().unwrap()],
                    v: &v[..shape.v_token_elements().unwrap()],
                },
            )
            .expect("prefill step should run");
        scheduler
            .complete_prefill(RequestId(11))
            .expect("prefill completion should succeed");

        let ready = scheduler
            .ready_decode_batch(4)
            .expect("ready batch should be generated");
        let q_step = shape.q_elements().unwrap();
        let k_step = shape.k_token_elements().unwrap();
        let v_step = shape.v_token_elements().unwrap();
        let outputs = runner
            .run_ready_batch(
                &mut stream,
                &mut scheduler,
                &ready,
                &[Qwen3SelfAttnDecodeBatchInput {
                    request_id: RequestId(11),
                    q: &q[q_step..2 * q_step],
                    k: &k[k_step..2 * k_step],
                    v: &v[v_step..2 * v_step],
                }],
            )
            .expect("ready batch should run");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].request_id, RequestId(11));
        assert_eq!(outputs[0].cache_position, 1);
        assert_eq!(outputs[0].cache_len, 2);
        assert_eq!(
            scheduler
                .active_request(RequestId(11))
                .expect("request should still be active")
                .generated_tokens,
            1
        );

        let expected = pack_paged_kv_cache_for_block_table(&k, &v, &allocation.blocks, 2, shape)
            .expect("expected cache packing should succeed");
        let actual = runner
            .read_cache_to_host(RequestId(11), &mut stream)
            .expect("cache readback should succeed");
        assert_f32s_close(&actual.k, &expected.k, 1e-5);
        assert_f32s_close(&actual.v, &expected.v, 1e-5);
    }

    #[test]
    fn qwen3_self_attn_request_decode_runner_rejects_missing_runner_before_advance_cpu() {
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
        let mut scheduler = SchedulerState::with_block_size(4, 2);
        scheduler.enqueue(Request::new(11, 1, 1));
        scheduler
            .pop_prefill_batch_with_allocation(1)
            .expect("allocation should succeed");
        scheduler
            .complete_prefill(RequestId(11))
            .expect("prefill completion should succeed");
        let ready = scheduler
            .ready_decode_batch(4)
            .expect("ready batch should be generated");

        let q = vec![0.0_f32; shape.q_elements().unwrap()];
        let k = vec![0.0_f32; shape.k_token_elements().unwrap()];
        let v = vec![0.0_f32; shape.v_token_elements().unwrap()];
        let mut runner = Qwen3SelfAttnRequestDecodeRunner::new();
        let err = runner
            .run_ready_batch(
                &mut stream,
                &mut scheduler,
                &ready,
                &[Qwen3SelfAttnDecodeBatchInput {
                    request_id: RequestId(11),
                    q: &q,
                    k: &k,
                    v: &v,
                }],
            )
            .expect_err("missing runner state should be rejected");
        assert!(err.contains("has no request"), "{err}");
        let active = scheduler
            .active_request(RequestId(11))
            .expect("request should remain active");
        assert_eq!(active.cached_tokens, 1);
        assert_eq!(active.generated_tokens, 0);
    }

    #[test]
    fn qwen3_decoder_layer_request_decode_runner_runs_ready_batch_cpu() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let shape = PagedDecodeShape {
            block_size: 2,
            cache_blocks: 4,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            value_dim: 2,
        };
        let hidden = 4_usize;
        let intermediate = 3_usize;
        let sequence_len = 2_usize;
        let softmax_scale = 1.0_f32 / (shape.head_dim as f32).sqrt();
        let mlp_epsilon = 1e-5_f32;
        let weights = make_layer_weights(&mut context, &mut stream, shape, hidden, intermediate);
        let mut scheduler = SchedulerState::with_block_size(4, 2);
        scheduler.enqueue(Request::new(21, 1, 1));
        let mut scheduled = scheduler
            .pop_prefill_batch_with_allocation(1)
            .expect("allocation should succeed");
        let allocation = scheduled.remove(0).allocation;

        let mut runner = Qwen3DecoderLayerRequestDecodeRunner::new();
        runner
            .insert_request(
                &mut context,
                &mut stream,
                allocation.request_id,
                &weights,
                shape,
                allocation.blocks.clone(),
                softmax_scale,
                mlp_epsilon,
            )
            .expect("runner insert should succeed");
        assert_eq!(runner.len(), 1);
        assert!(runner.contains_request(RequestId(21)));

        let q_step = shape.q_elements().unwrap();
        let k_step = shape.k_token_elements().unwrap();
        let v_step = shape.v_token_elements().unwrap();
        let attention_step = shape.output_elements().unwrap();
        let q = (0..sequence_len * q_step)
            .map(|index| ((index * 2) as f32 - 7.0) / 11.0)
            .collect::<Vec<_>>();
        let k = (0..sequence_len * k_step)
            .map(|index| ((index * 5) as f32 - 3.0) / 13.0)
            .collect::<Vec<_>>();
        let v = (0..sequence_len * v_step)
            .map(|index| ((index * 7) as f32 - 5.0) / 17.0)
            .collect::<Vec<_>>();
        let gate = (0..sequence_len * attention_step)
            .map(|index| ((index * 11) as f32 - 13.0) / 23.0)
            .collect::<Vec<_>>();
        let residual = (0..sequence_len * hidden)
            .map(|index| ((index * 9) as f32 - 2.0) / 29.0)
            .collect::<Vec<_>>();

        let prefill = runner
            .run_prefill_step(
                &mut stream,
                Qwen3DecoderLayerDecodeBatchInput {
                    request_id: RequestId(21),
                    q: &q[..q_step],
                    k: &k[..k_step],
                    v: &v[..v_step],
                    output_gate: Some(&gate[..attention_step]),
                    residual: &residual[..hidden],
                },
            )
            .expect("prefill step should run");
        assert_eq!(prefill.cache_position, 0);
        assert_eq!(prefill.cache_len, 1);
        scheduler
            .complete_prefill(RequestId(21))
            .expect("prefill completion should succeed");

        let ready = scheduler
            .ready_decode_batch(4)
            .expect("ready batch should be generated");
        let outputs = runner
            .run_ready_batch(
                &mut stream,
                &mut scheduler,
                &ready,
                &[Qwen3DecoderLayerDecodeBatchInput {
                    request_id: RequestId(21),
                    q: &q[q_step..2 * q_step],
                    k: &k[k_step..2 * k_step],
                    v: &v[v_step..2 * v_step],
                    output_gate: Some(&gate[attention_step..2 * attention_step]),
                    residual: &residual[hidden..2 * hidden],
                }],
            )
            .expect("ready batch should run");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].request_id, RequestId(21));
        assert_eq!(outputs[0].cache_position, 1);
        assert_eq!(outputs[0].cache_len, 2);
        let active = scheduler
            .active_request(RequestId(21))
            .expect("request should still be active");
        assert_eq!(active.cached_tokens, 2);
        assert_eq!(active.generated_tokens, 1);
        assert_eq!(runner.written_len(RequestId(21)), Some(2));

        let mut layer_output = prefill.layer_output;
        layer_output.extend_from_slice(&outputs[0].layer_output);
        let sequence_output = qwen3_decoder_layer_sequence_to_host_f32(
            &weights,
            &mut context,
            &mut stream,
            shape,
            &allocation.blocks,
            softmax_scale,
            mlp_epsilon,
            &q,
            &k,
            &v,
            Some(&gate),
            &residual,
            sequence_len,
        )
        .expect("sequence output should run");
        assert_f32s_close(&layer_output, &sequence_output.layer_output, 1e-5);

        let expected = pack_paged_kv_cache_for_block_table(&k, &v, &allocation.blocks, 2, shape)
            .expect("expected cache packing should succeed");
        let actual = runner
            .read_cache_to_host(RequestId(21), &mut stream)
            .expect("cache readback should succeed");
        assert_f32s_close(&actual.k, &expected.k, 1e-6);
        assert_f32s_close(&actual.v, &expected.v, 1e-6);
    }

    #[test]
    fn qwen3_decoder_layer_request_decode_runner_can_run_without_scheduler_advance_cpu() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let shape = PagedDecodeShape {
            block_size: 2,
            cache_blocks: 4,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            value_dim: 2,
        };
        let hidden = 4_usize;
        let intermediate = 3_usize;
        let softmax_scale = 1.0_f32 / (shape.head_dim as f32).sqrt();
        let mlp_epsilon = 1e-5_f32;
        let weights = make_layer_weights(&mut context, &mut stream, shape, hidden, intermediate);
        let mut scheduler = SchedulerState::with_block_size(4, 2);
        scheduler.enqueue(Request::new(31, 1, 1));
        let mut scheduled = scheduler
            .pop_prefill_batch_with_allocation(1)
            .expect("allocation should succeed");
        let allocation = scheduled.remove(0).allocation;

        let mut runner = Qwen3DecoderLayerRequestDecodeRunner::new();
        runner
            .insert_request(
                &mut context,
                &mut stream,
                allocation.request_id,
                &weights,
                shape,
                allocation.blocks,
                softmax_scale,
                mlp_epsilon,
            )
            .expect("runner insert should succeed");

        let q_step = shape.q_elements().unwrap();
        let k_step = shape.k_token_elements().unwrap();
        let v_step = shape.v_token_elements().unwrap();
        let attention_step = shape.output_elements().unwrap();
        let q = (0..2 * q_step)
            .map(|index| ((index * 2) as f32 - 3.0) / 11.0)
            .collect::<Vec<_>>();
        let k = (0..2 * k_step)
            .map(|index| ((index * 5) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v = (0..2 * v_step)
            .map(|index| ((index * 7) as f32 - 11.0) / 17.0)
            .collect::<Vec<_>>();
        let gate = (0..2 * attention_step)
            .map(|index| ((index * 11) as f32 - 13.0) / 19.0)
            .collect::<Vec<_>>();
        let residual = (0..2 * hidden)
            .map(|index| ((index * 13) as f32 - 17.0) / 23.0)
            .collect::<Vec<_>>();

        runner
            .run_prefill_step(
                &mut stream,
                Qwen3DecoderLayerDecodeBatchInput {
                    request_id: RequestId(31),
                    q: &q[..q_step],
                    k: &k[..k_step],
                    v: &v[..v_step],
                    output_gate: Some(&gate[..attention_step]),
                    residual: &residual[..hidden],
                },
            )
            .expect("prefill step should run");
        scheduler
            .complete_prefill(RequestId(31))
            .expect("prefill completion should succeed");
        let ready = scheduler
            .ready_decode_batch(4)
            .expect("ready batch should be generated");

        let outputs = runner
            .run_ready_batch_without_advance(
                &mut stream,
                &scheduler,
                &ready,
                &[Qwen3DecoderLayerDecodeBatchInput {
                    request_id: RequestId(31),
                    q: &q[q_step..2 * q_step],
                    k: &k[k_step..2 * k_step],
                    v: &v[v_step..2 * v_step],
                    output_gate: Some(&gate[attention_step..2 * attention_step]),
                    residual: &residual[hidden..2 * hidden],
                }],
            )
            .expect("ready batch should run without scheduler advance");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].cache_position, 1);
        assert_eq!(outputs[0].cache_len, 2);
        assert_eq!(runner.written_len(RequestId(31)), Some(2));

        let active = scheduler
            .active_request(RequestId(31))
            .expect("request should remain active");
        assert_eq!(active.cached_tokens, 1);
        assert_eq!(active.generated_tokens, 0);

        scheduler
            .advance_decode(RequestId(31))
            .expect("model loop owner should be able to advance after all layers");
        let active = scheduler
            .active_request(RequestId(31))
            .expect("request should remain active after manual advance");
        assert_eq!(active.cached_tokens, 2);
        assert_eq!(active.generated_tokens, 1);
    }

    #[test]
    fn qwen3_decoder_layer_stack_runner_advances_ready_batch_once_cpu() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let shape = PagedDecodeShape {
            block_size: 2,
            cache_blocks: 4,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            value_dim: 2,
        };
        let hidden = 4_usize;
        let intermediate = 3_usize;
        let softmax_scale = 1.0_f32 / (shape.head_dim as f32).sqrt();
        let mlp_epsilon = 1e-5_f32;
        let weights = make_layer_weights(&mut context, &mut stream, shape, hidden, intermediate);
        let mut scheduler = SchedulerState::with_block_size(4, 2);
        scheduler.enqueue(Request::new(41, 1, 1));
        let mut scheduled = scheduler
            .pop_prefill_batch_with_allocation(1)
            .expect("allocation should succeed");
        let allocation = scheduled.remove(0).allocation;

        let mut stack = Qwen3DecoderLayerStackRequestDecodeRunner::new();
        let first_layer = stack.push_layer();
        let second_layer = stack.push_layer();
        assert_eq!(stack.layer_count(), 2);
        for layer_index in [first_layer, second_layer] {
            stack
                .insert_request(
                    layer_index,
                    &mut context,
                    &mut stream,
                    allocation.request_id,
                    &weights,
                    shape,
                    allocation.blocks.clone(),
                    softmax_scale,
                    mlp_epsilon,
                )
                .expect("stack insert should succeed");
        }

        let q_step = shape.q_elements().unwrap();
        let k_step = shape.k_token_elements().unwrap();
        let v_step = shape.v_token_elements().unwrap();
        let attention_step = shape.output_elements().unwrap();
        let q = (0..2 * q_step)
            .map(|index| ((index * 2) as f32 - 3.0) / 11.0)
            .collect::<Vec<_>>();
        let k = (0..2 * k_step)
            .map(|index| ((index * 5) as f32 - 7.0) / 13.0)
            .collect::<Vec<_>>();
        let v = (0..2 * v_step)
            .map(|index| ((index * 7) as f32 - 11.0) / 17.0)
            .collect::<Vec<_>>();
        let gate = (0..2 * attention_step)
            .map(|index| ((index * 11) as f32 - 13.0) / 19.0)
            .collect::<Vec<_>>();
        let first_residual = (0..2 * hidden)
            .map(|index| ((index * 13) as f32 - 17.0) / 23.0)
            .collect::<Vec<_>>();
        let second_residual = (0..2 * hidden)
            .map(|index| ((index * 17) as f32 - 19.0) / 29.0)
            .collect::<Vec<_>>();

        for (layer_index, residual) in [
            (first_layer, first_residual.as_slice()),
            (second_layer, second_residual.as_slice()),
        ] {
            stack
                .run_prefill_step(
                    layer_index,
                    &mut stream,
                    Qwen3DecoderLayerDecodeBatchInput {
                        request_id: RequestId(41),
                        q: &q[..q_step],
                        k: &k[..k_step],
                        v: &v[..v_step],
                        output_gate: Some(&gate[..attention_step]),
                        residual: &residual[..hidden],
                    },
                )
                .expect("stack prefill should run");
        }
        scheduler
            .complete_prefill(RequestId(41))
            .expect("prefill completion should succeed");
        let ready = scheduler
            .ready_decode_batch(4)
            .expect("ready batch should be generated");

        let layer_inputs = vec![
            vec![Qwen3DecoderLayerDecodeBatchInput {
                request_id: RequestId(41),
                q: &q[q_step..2 * q_step],
                k: &k[k_step..2 * k_step],
                v: &v[v_step..2 * v_step],
                output_gate: Some(&gate[attention_step..2 * attention_step]),
                residual: &first_residual[hidden..2 * hidden],
            }],
            vec![Qwen3DecoderLayerDecodeBatchInput {
                request_id: RequestId(41),
                q: &q[q_step..2 * q_step],
                k: &k[k_step..2 * k_step],
                v: &v[v_step..2 * v_step],
                output_gate: Some(&gate[attention_step..2 * attention_step]),
                residual: &second_residual[hidden..2 * hidden],
            }],
        ];
        let layer_input_refs = layer_inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let outputs = stack
            .run_ready_batch_across_layers(&mut stream, &mut scheduler, &ready, &layer_input_refs)
            .expect("stack ready batch should run");
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].len(), 1);
        assert_eq!(outputs[1].len(), 1);
        assert_eq!(outputs[0][0].cache_position, 1);
        assert_eq!(outputs[1][0].cache_position, 1);
        assert_eq!(stack.written_len(first_layer, RequestId(41)).unwrap(), 2);
        assert_eq!(stack.written_len(second_layer, RequestId(41)).unwrap(), 2);
        let active = scheduler
            .active_request(RequestId(41))
            .expect("request should remain active");
        assert_eq!(active.cached_tokens, 2);
        assert_eq!(active.generated_tokens, 1);
    }

    #[test]
    fn qwen3_decoder_layer_stack_runner_rejects_bad_layer_input_before_mutation_cpu() {
        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let shape = PagedDecodeShape {
            block_size: 2,
            cache_blocks: 4,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            value_dim: 2,
        };
        let hidden = 4_usize;
        let intermediate = 3_usize;
        let softmax_scale = 1.0_f32 / (shape.head_dim as f32).sqrt();
        let mlp_epsilon = 1e-5_f32;
        let weights = make_layer_weights(&mut context, &mut stream, shape, hidden, intermediate);
        let mut scheduler = SchedulerState::with_block_size(4, 2);
        scheduler.enqueue(Request::new(51, 1, 1));
        let mut scheduled = scheduler
            .pop_prefill_batch_with_allocation(1)
            .expect("allocation should succeed");
        let allocation = scheduled.remove(0).allocation;

        let mut stack = Qwen3DecoderLayerStackRequestDecodeRunner::new();
        let first_layer = stack.push_layer();
        let second_layer = stack.push_layer();
        for layer_index in [first_layer, second_layer] {
            stack
                .insert_request(
                    layer_index,
                    &mut context,
                    &mut stream,
                    allocation.request_id,
                    &weights,
                    shape,
                    allocation.blocks.clone(),
                    softmax_scale,
                    mlp_epsilon,
                )
                .expect("stack insert should succeed");
        }

        let q_step = shape.q_elements().unwrap();
        let k_step = shape.k_token_elements().unwrap();
        let v_step = shape.v_token_elements().unwrap();
        let attention_step = shape.output_elements().unwrap();
        let q = vec![0.125_f32; 2 * q_step];
        let k = vec![0.25_f32; 2 * k_step];
        let v = vec![0.375_f32; 2 * v_step];
        let gate = vec![0.5_f32; 2 * attention_step];
        let residual = vec![0.625_f32; 2 * hidden];

        for layer_index in [first_layer, second_layer] {
            stack
                .run_prefill_step(
                    layer_index,
                    &mut stream,
                    Qwen3DecoderLayerDecodeBatchInput {
                        request_id: RequestId(51),
                        q: &q[..q_step],
                        k: &k[..k_step],
                        v: &v[..v_step],
                        output_gate: Some(&gate[..attention_step]),
                        residual: &residual[..hidden],
                    },
                )
                .expect("stack prefill should run");
        }
        scheduler
            .complete_prefill(RequestId(51))
            .expect("prefill completion should succeed");
        let ready = scheduler
            .ready_decode_batch(4)
            .expect("ready batch should be generated");
        let layer_inputs = vec![
            vec![Qwen3DecoderLayerDecodeBatchInput {
                request_id: RequestId(51),
                q: &q[q_step..2 * q_step],
                k: &k[k_step..2 * k_step],
                v: &v[v_step..2 * v_step],
                output_gate: Some(&gate[attention_step..2 * attention_step]),
                residual: &residual[hidden..2 * hidden],
            }],
            Vec::new(),
        ];
        let layer_input_refs = layer_inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let err = stack
            .run_ready_batch_across_layers(&mut stream, &mut scheduler, &ready, &layer_input_refs)
            .expect_err("bad second-layer input should be rejected");
        assert!(err.contains("decoder layer 1 rejected"), "{err}");
        let active = scheduler
            .active_request(RequestId(51))
            .expect("request should remain active");
        assert_eq!(active.cached_tokens, 1);
        assert_eq!(active.generated_tokens, 0);
        assert_eq!(stack.written_len(first_layer, RequestId(51)).unwrap(), 1);
        assert_eq!(stack.written_len(second_layer, RequestId(51)).unwrap(), 1);
    }

    #[test]
    fn qwen3_decoder_layer_request_decode_runner_rejects_missing_runner_before_advance_cpu() {
        let mut scheduler = SchedulerState::with_block_size(4, 2);
        scheduler.enqueue(Request::new(21, 1, 1));
        scheduler
            .pop_prefill_batch_with_allocation(1)
            .expect("allocation should succeed");
        scheduler
            .complete_prefill(RequestId(21))
            .expect("prefill completion should succeed");
        let ready = scheduler
            .ready_decode_batch(4)
            .expect("ready batch should be generated");

        let mut context = RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let shape = PagedDecodeShape {
            block_size: 2,
            cache_blocks: 4,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            value_dim: 2,
        };
        let q = vec![0.0_f32; shape.q_elements().unwrap()];
        let k = vec![0.0_f32; shape.k_token_elements().unwrap()];
        let v = vec![0.0_f32; shape.v_token_elements().unwrap()];
        let gate = vec![0.0_f32; shape.output_elements().unwrap()];
        let residual = vec![0.0_f32; 4];
        let mut runner = Qwen3DecoderLayerRequestDecodeRunner::new();
        let err = runner
            .run_ready_batch(
                &mut stream,
                &mut scheduler,
                &ready,
                &[Qwen3DecoderLayerDecodeBatchInput {
                    request_id: RequestId(21),
                    q: &q,
                    k: &k,
                    v: &v,
                    output_gate: Some(&gate),
                    residual: &residual,
                }],
            )
            .expect_err("missing runner state should be rejected");
        assert!(err.contains("has no request"), "{err}");
        let active = scheduler
            .active_request(RequestId(21))
            .expect("request should remain active");
        assert_eq!(active.cached_tokens, 1);
        assert_eq!(active.generated_tokens, 0);
    }
}
