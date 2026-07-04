// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Scheduler-facing decode runner utilities.
//!
//! This module keeps scheduler request ids and runtime decode state together
//! without making the low-level decoder module depend on scheduler ownership.

use std::collections::{BTreeMap, BTreeSet};

use crate::decoder::{
    PagedDecodeShape, PagedKvCacheReadback, Qwen3SelfAttnDecodeState, Qwen3SelfAttnDecodeStepOutput,
};
use crate::scheduler::{RequestId, SchedulerDecodeRequest, SchedulerState};
use ullm_runtime_sys::{RuntimeContext, RuntimeStream};

#[derive(Debug)]
struct Qwen3SelfAttnRequestDecodeState {
    block_table: Vec<u32>,
    state: Qwen3SelfAttnDecodeState,
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

#[derive(Debug, Default)]
pub struct Qwen3SelfAttnRequestDecodeRunner {
    states: BTreeMap<RequestId, Qwen3SelfAttnRequestDecodeState>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::pack_paged_kv_cache_for_block_table;
    use crate::scheduler::Request;

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
}
