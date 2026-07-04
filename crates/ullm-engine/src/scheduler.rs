// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub const DEFAULT_KV_BLOCK_SIZE_TOKENS: u32 = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestId(pub u64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub id: RequestId,
    pub prompt_tokens: usize,
    pub max_new_tokens: usize,
}

impl Request {
    pub fn new(id: u64, prompt_tokens: usize, max_new_tokens: usize) -> Self {
        Self {
            id: RequestId(id),
            prompt_tokens,
            max_new_tokens,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerRequestAllocation {
    pub request: Request,
    pub allocation: BlockAllocation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveRequestState {
    pub request: Request,
    pub allocation: BlockAllocation,
    pub cached_tokens: usize,
    pub generated_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerDecodeRequest {
    pub request: Request,
    pub allocation: BlockAllocation,
    pub cached_tokens: usize,
    pub generated_tokens: usize,
    pub cache_position: usize,
    pub next_cache_len: usize,
    pub remaining_new_tokens: usize,
}

#[derive(Debug)]
pub struct SchedulerState {
    queue: RequestQueue,
    allocator: KvBlockAllocator,
    active: BTreeMap<RequestId, ActiveRequestState>,
}

impl SchedulerState {
    pub fn new(total_kv_blocks: u32) -> Self {
        Self::with_block_size(total_kv_blocks, DEFAULT_KV_BLOCK_SIZE_TOKENS)
    }

    pub fn with_block_size(total_kv_blocks: u32, block_size_tokens: u32) -> Self {
        Self {
            queue: RequestQueue::new(),
            allocator: KvBlockAllocator::with_block_size(total_kv_blocks, block_size_tokens),
            active: BTreeMap::new(),
        }
    }

    pub fn enqueue(&mut self, request: Request) {
        self.queue.push(request);
    }

    pub fn waiting_len(&self) -> usize {
        self.queue.len()
    }

    pub fn waiting_is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn allocator_stats(&self) -> KvBlockAllocatorStats {
        self.allocator.stats()
    }

    pub fn active_len(&self) -> usize {
        self.active.len()
    }

    pub fn active_request(&self, request_id: RequestId) -> Option<&ActiveRequestState> {
        self.active.get(&request_id)
    }

    pub fn active_request_ids(&self) -> Vec<RequestId> {
        self.active.keys().copied().collect()
    }

    pub fn ready_decode_batch(
        &self,
        max_requests: usize,
    ) -> Result<Vec<SchedulerDecodeRequest>, String> {
        if max_requests == 0 {
            return Ok(Vec::new());
        }

        let mut ready = Vec::new();
        for state in self.active.values() {
            if state.cached_tokens < state.request.prompt_tokens {
                continue;
            }
            if state.generated_tokens >= state.request.max_new_tokens {
                continue;
            }
            let next_cache_len = state.cached_tokens.checked_add(1).ok_or_else(|| {
                format!("request {:?} cached token count overflow", state.request.id)
            })?;
            let max_cache_capacity = state
                .allocation
                .blocks
                .len()
                .checked_mul(self.allocator.block_size_tokens() as usize)
                .ok_or_else(|| {
                    format!("request {:?} cache capacity overflows", state.request.id)
                })?;
            if next_cache_len > max_cache_capacity {
                return Err(format!(
                    "request {:?} exceeds allocated cache capacity {}",
                    state.request.id, max_cache_capacity
                ));
            }
            let remaining_new_tokens = state
                .request
                .max_new_tokens
                .checked_sub(state.generated_tokens)
                .ok_or_else(|| {
                    format!(
                        "request {:?} generated token count exceeds max_new_tokens",
                        state.request.id
                    )
                })?;
            ready.push(SchedulerDecodeRequest {
                request: state.request.clone(),
                allocation: state.allocation.clone(),
                cached_tokens: state.cached_tokens,
                generated_tokens: state.generated_tokens,
                cache_position: state.cached_tokens,
                next_cache_len,
                remaining_new_tokens,
            });
            if ready.len() == max_requests {
                break;
            }
        }
        Ok(ready)
    }

    pub fn release_request(&mut self, request_id: RequestId) -> usize {
        self.active.remove(&request_id);
        self.allocator.free_request(request_id)
    }

    pub fn complete_prefill(&mut self, request_id: RequestId) -> Result<(), String> {
        let state = self
            .active
            .get_mut(&request_id)
            .ok_or_else(|| format!("request {:?} is not active", request_id))?;

        if state.cached_tokens == state.request.prompt_tokens {
            return Err(format!(
                "request {:?} prefill already completed",
                request_id
            ));
        }

        if state.cached_tokens > state.request.prompt_tokens {
            return Err(format!(
                "request {:?} has invalid cached token progress",
                request_id
            ));
        }

        let max_cache_capacity = state
            .allocation
            .blocks
            .len()
            .checked_mul(self.allocator.block_size_tokens() as usize)
            .ok_or_else(|| format!("request {:?} cache capacity overflows", request_id))?;
        if state.request.prompt_tokens > max_cache_capacity {
            return Err(format!(
                "request {:?} prompt_tokens {} exceeds allocated cache capacity {}",
                request_id, state.request.prompt_tokens, max_cache_capacity
            ));
        }

        state.cached_tokens = state.request.prompt_tokens;
        Ok(())
    }

    pub fn advance_decode(&mut self, request_id: RequestId) -> Result<(), String> {
        let state = self
            .active
            .get_mut(&request_id)
            .ok_or_else(|| format!("request {:?} is not active", request_id))?;

        if state.cached_tokens < state.request.prompt_tokens {
            return Err(format!("request {:?} prefill not completed", request_id));
        }

        if state.generated_tokens >= state.request.max_new_tokens {
            return Err(format!(
                "request {:?} exceeds max_new_tokens {}",
                request_id, state.request.max_new_tokens
            ));
        }

        let next_cached = state
            .cached_tokens
            .checked_add(1)
            .ok_or_else(|| format!("request {:?} cached token count overflow", request_id))?;
        let max_cache_capacity = state
            .allocation
            .blocks
            .len()
            .checked_mul(self.allocator.block_size_tokens() as usize)
            .ok_or_else(|| format!("request {:?} cache capacity overflows", request_id))?;
        if next_cached > max_cache_capacity {
            return Err(format!(
                "request {:?} exceeds allocated cache capacity {}",
                request_id, max_cache_capacity
            ));
        }

        state.cached_tokens = next_cached;
        state.generated_tokens += 1;
        Ok(())
    }

    pub fn advance_decode_batch(
        &mut self,
        ready_batch: &[SchedulerDecodeRequest],
    ) -> Result<(), String> {
        let mut request_ids = BTreeSet::new();
        for request in ready_batch {
            if !request_ids.insert(request.request.id) {
                return Err(format!(
                    "ready decode batch contains duplicate request {:?}",
                    request.request.id
                ));
            }
            let active = self
                .active
                .get(&request.request.id)
                .ok_or_else(|| format!("request {:?} is not active", request.request.id))?;
            if active.request != request.request {
                return Err(format!(
                    "ready decode request {:?} metadata is stale",
                    request.request.id
                ));
            }
            if active.allocation != request.allocation {
                return Err(format!(
                    "ready decode request {:?} allocation is stale",
                    request.request.id
                ));
            }
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
            if request.cache_position != active.cached_tokens {
                return Err(format!(
                    "ready decode request {:?} cache_position {} does not match cached_tokens {}",
                    request.request.id, request.cache_position, active.cached_tokens
                ));
            }
            let expected_next_cache_len = active.cached_tokens.checked_add(1).ok_or_else(|| {
                format!(
                    "request {:?} cached token count overflow",
                    request.request.id
                )
            })?;
            if request.next_cache_len != expected_next_cache_len {
                return Err(format!(
                    "ready decode request {:?} next_cache_len {} does not match cached_tokens + 1",
                    request.request.id, request.next_cache_len
                ));
            }
            if request.remaining_new_tokens
                != active
                    .request
                    .max_new_tokens
                    .checked_sub(active.generated_tokens)
                    .ok_or_else(|| {
                        format!(
                            "request {:?} generated token count exceeds max_new_tokens",
                            request.request.id
                        )
                    })?
            {
                return Err(format!(
                    "ready decode request {:?} remaining_new_tokens {} is stale",
                    request.request.id, request.remaining_new_tokens
                ));
            }
        }

        for request in ready_batch {
            self.advance_decode(request.request.id)?;
        }
        Ok(())
    }

    pub fn pop_prefill_batch_with_allocation(
        &mut self,
        token_budget: usize,
    ) -> Result<Vec<SchedulerRequestAllocation>, String> {
        let requests = self.queue.pop_prefill_batch(token_budget);
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let mut selected_ids = BTreeSet::new();
        for request in requests.iter() {
            let already_active = self.active.contains_key(&request.id);
            let duplicated_in_batch = !selected_ids.insert(request.id);
            if already_active || duplicated_in_batch {
                for request in requests.iter().rev() {
                    self.queue.push_front(request.clone());
                }
                if already_active {
                    return Err(format!("request {:?} is already active", request.id));
                }
                return Err(format!("request {:?} appears multiple times", request.id));
            }
        }

        let mut allocations: Vec<(Request, BlockAllocation)> = Vec::with_capacity(requests.len());
        for request in &requests {
            let token_count = match request.prompt_tokens.checked_add(request.max_new_tokens) {
                Some(token_count) => token_count,
                None => {
                    for (_, allocated) in allocations.drain(..) {
                        self.allocator.free_request(allocated.request_id);
                    }
                    for request in requests.iter().rev() {
                        self.queue.push_front(request.clone());
                    }
                    return Err(format!(
                        "token count overflow for request {:?}: {} + {}",
                        request.id, request.prompt_tokens, request.max_new_tokens
                    ));
                }
            };

            match self.allocator.allocate_for_tokens(request.id, token_count) {
                Ok(allocation) => allocations.push((request.clone(), allocation)),
                Err(err) => {
                    for (_, allocated) in allocations.drain(..) {
                        self.allocator.free_request(allocated.request_id);
                    }
                    for request in requests.iter().rev() {
                        self.queue.push_front(request.clone());
                    }
                    return Err(format!(
                        "allocation failed for request {:?} with token_count {}: {}",
                        request.id, token_count, err
                    ));
                }
            }
        }

        for (request, allocation) in allocations.iter() {
            self.active.insert(
                request.id,
                ActiveRequestState {
                    request: request.clone(),
                    allocation: allocation.clone(),
                    cached_tokens: 0,
                    generated_tokens: 0,
                },
            );
        }

        Ok(allocations
            .into_iter()
            .map(|(request, allocation)| SchedulerRequestAllocation {
                request,
                allocation,
            })
            .collect())
    }
}

#[derive(Debug, Default)]
pub struct RequestQueue {
    waiting: VecDeque<Request>,
}

impl RequestQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, request: Request) {
        self.waiting.push_back(request);
    }

    pub fn len(&self) -> usize {
        self.waiting.len()
    }

    pub fn is_empty(&self) -> bool {
        self.waiting.is_empty()
    }

    pub fn pop_prefill_batch(&mut self, token_budget: usize) -> Vec<Request> {
        let mut selected = Vec::new();
        let mut used_tokens = 0_usize;
        while let Some(front) = self.waiting.front() {
            let would_use = used_tokens.saturating_add(front.prompt_tokens);
            if !selected.is_empty() && would_use > token_budget {
                break;
            }
            let request = self.waiting.pop_front().expect("front existed");
            used_tokens = used_tokens.saturating_add(request.prompt_tokens);
            selected.push(request);
        }
        selected
    }

    fn push_front(&mut self, request: Request) {
        self.waiting.push_front(request);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockAllocation {
    pub request_id: RequestId,
    pub blocks: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvBlockAllocatorStats {
    pub block_size_tokens: u32,
    pub total_blocks: u32,
    pub free_blocks: usize,
    pub allocated_blocks: usize,
    pub free_runs: usize,
    pub largest_free_run: usize,
}

#[derive(Debug)]
pub struct KvBlockAllocator {
    block_size_tokens: u32,
    total_blocks: u32,
    free: VecDeque<u32>,
    allocations: BTreeMap<RequestId, Vec<u32>>,
}

impl KvBlockAllocator {
    pub fn new(total_blocks: u32) -> Self {
        Self::with_block_size(total_blocks, DEFAULT_KV_BLOCK_SIZE_TOKENS)
    }

    pub fn with_block_size(total_blocks: u32, block_size_tokens: u32) -> Self {
        assert!(block_size_tokens > 0, "block size must be nonzero");
        Self {
            block_size_tokens,
            total_blocks,
            free: (0..total_blocks).collect(),
            allocations: BTreeMap::new(),
        }
    }

    pub fn total_blocks(&self) -> u32 {
        self.total_blocks
    }

    pub fn block_size_tokens(&self) -> u32 {
        self.block_size_tokens
    }

    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    pub fn allocated_blocks(&self) -> usize {
        self.allocations.values().map(Vec::len).sum()
    }

    pub fn block_count_for_tokens(&self, token_count: usize) -> usize {
        if token_count == 0 {
            return 0;
        }
        let block_size = self.block_size_tokens as usize;
        (token_count - 1) / block_size + 1
    }

    pub fn allocate(
        &mut self,
        request_id: RequestId,
        block_count: usize,
    ) -> Result<BlockAllocation, String> {
        if self.allocations.contains_key(&request_id) {
            return Err(format!(
                "request {:?} already has allocated blocks",
                request_id
            ));
        }
        if block_count > self.free.len() {
            return Err(format!(
                "not enough KV blocks: requested {}, free {}",
                block_count,
                self.free.len()
            ));
        }
        let mut blocks = Vec::with_capacity(block_count);
        for _ in 0..block_count {
            blocks.push(self.free.pop_front().expect("free length checked"));
        }
        self.allocations.insert(request_id, blocks.clone());
        Ok(BlockAllocation { request_id, blocks })
    }

    pub fn allocate_for_tokens(
        &mut self,
        request_id: RequestId,
        token_count: usize,
    ) -> Result<BlockAllocation, String> {
        let block_count = self.block_count_for_tokens(token_count);
        self.allocate(request_id, block_count)
    }

    pub fn free_request(&mut self, request_id: RequestId) -> usize {
        let Some(blocks) = self.allocations.remove(&request_id) else {
            return 0;
        };
        let count = blocks.len();
        for block in blocks {
            self.free.push_back(block);
        }
        count
    }

    pub fn allocation(&self, request_id: RequestId) -> Option<&[u32]> {
        self.allocations.get(&request_id).map(Vec::as_slice)
    }

    pub fn stats(&self) -> KvBlockAllocatorStats {
        let (free_runs, largest_free_run) = free_run_stats(&self.free);
        KvBlockAllocatorStats {
            block_size_tokens: self.block_size_tokens,
            total_blocks: self.total_blocks,
            free_blocks: self.free_blocks(),
            allocated_blocks: self.allocated_blocks(),
            free_runs,
            largest_free_run,
        }
    }
}

fn free_run_stats(free: &VecDeque<u32>) -> (usize, usize) {
    if free.is_empty() {
        return (0, 0);
    }
    let mut sorted = free.iter().copied().collect::<Vec<_>>();
    sorted.sort_unstable();
    let mut runs = 1_usize;
    let mut current = 1_usize;
    let mut largest = 1_usize;
    for pair in sorted.windows(2) {
        if pair[1] == pair[0] + 1 {
            current += 1;
        } else {
            largest = largest.max(current);
            runs += 1;
            current = 1;
        }
    }
    largest = largest.max(current);
    (runs, largest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_batch_respects_token_budget_after_first_request() {
        let mut queue = RequestQueue::new();
        queue.push(Request::new(1, 8, 4));
        queue.push(Request::new(2, 16, 4));
        queue.push(Request::new(3, 4, 4));

        let first = queue.pop_prefill_batch(16);
        assert_eq!(
            first.iter().map(|r| r.id).collect::<Vec<_>>(),
            vec![RequestId(1)]
        );
        assert_eq!(queue.len(), 2);

        let second = queue.pop_prefill_batch(32);
        assert_eq!(
            second.iter().map(|r| r.id).collect::<Vec<_>>(),
            vec![RequestId(2), RequestId(3)]
        );
        assert!(queue.is_empty());
    }

    #[test]
    fn oversized_first_request_is_still_selected() {
        let mut queue = RequestQueue::new();
        queue.push(Request::new(1, 128, 4));
        queue.push(Request::new(2, 1, 4));

        let batch = queue.pop_prefill_batch(32);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].id, RequestId(1));
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn kv_allocator_allocates_and_reuses_blocks() {
        let mut allocator = KvBlockAllocator::new(4);
        assert_eq!(allocator.block_size_tokens(), DEFAULT_KV_BLOCK_SIZE_TOKENS);
        let a = allocator.allocate(RequestId(10), 3).unwrap();
        assert_eq!(a.blocks, vec![0, 1, 2]);
        assert_eq!(allocator.free_blocks(), 1);
        assert_eq!(allocator.allocated_blocks(), 3);

        assert!(allocator.allocate(RequestId(11), 2).is_err());
        assert_eq!(allocator.free_request(RequestId(10)), 3);
        assert_eq!(allocator.free_blocks(), 4);

        let b = allocator.allocate(RequestId(11), 2).unwrap();
        assert_eq!(b.blocks, vec![3, 0]);
        assert_eq!(allocator.allocation(RequestId(11)), Some([3, 0].as_slice()));
    }

    #[test]
    fn kv_allocator_allocates_required_blocks_for_tokens() {
        let mut allocator = KvBlockAllocator::with_block_size(4, 4);
        assert_eq!(allocator.block_count_for_tokens(0), 0);
        assert_eq!(allocator.block_count_for_tokens(1), 1);
        assert_eq!(allocator.block_count_for_tokens(4), 1);
        assert_eq!(allocator.block_count_for_tokens(5), 2);
        assert_eq!(allocator.block_count_for_tokens(9), 3);

        let allocation = allocator.allocate_for_tokens(RequestId(20), 9).unwrap();
        assert_eq!(allocation.blocks, vec![0, 1, 2]);
        assert_eq!(allocator.free_blocks(), 1);
        assert!(allocator.allocate_for_tokens(RequestId(21), 5).is_err());
    }

    #[test]
    fn kv_allocator_reports_fragmentation_stats() {
        let mut allocator = KvBlockAllocator::with_block_size(8, 16);
        let _a = allocator.allocate(RequestId(1), 2).unwrap();
        let _b = allocator.allocate(RequestId(2), 2).unwrap();
        let _c = allocator.allocate(RequestId(3), 2).unwrap();
        assert_eq!(allocator.free_request(RequestId(2)), 2);

        let stats = allocator.stats();
        assert_eq!(stats.block_size_tokens, 16);
        assert_eq!(stats.total_blocks, 8);
        assert_eq!(stats.free_blocks, 4);
        assert_eq!(stats.allocated_blocks, 4);
        assert_eq!(stats.free_runs, 2);
        assert_eq!(stats.largest_free_run, 2);
    }

    #[test]
    fn scheduler_state_prefill_with_allocation_returns_block_assignments() {
        let mut scheduler = SchedulerState::with_block_size(4, 8);
        scheduler.enqueue(Request::new(1, 8, 4));
        scheduler.enqueue(Request::new(2, 3, 6));
        scheduler.enqueue(Request::new(3, 20, 2));

        let selected = scheduler
            .pop_prefill_batch_with_allocation(12)
            .expect("allocation should succeed");
        assert_eq!(selected.len(), 2);
        assert_eq!(
            selected[0].request.id,
            RequestId(1),
            "first selected request"
        );
        assert_eq!(selected[0].allocation.blocks, vec![0, 1]);
        assert_eq!(
            selected[1].request.id,
            RequestId(2),
            "second selected request"
        );
        assert_eq!(selected[1].allocation.blocks, vec![2, 3]);
        assert_eq!(scheduler.waiting_len(), 1);
        assert_eq!(scheduler.active_len(), 2);
        assert_eq!(
            scheduler
                .active_request(RequestId(1))
                .expect("request 1 should be active")
                .cached_tokens,
            0
        );
        assert_eq!(
            scheduler
                .active_request(RequestId(1))
                .expect("request 1 should be active")
                .generated_tokens,
            0
        );
        assert_eq!(
            scheduler.active_request_ids(),
            vec![RequestId(1), RequestId(2)]
        );
    }

    #[test]
    fn scheduler_state_pop_prefill_with_allocation_rolls_back_when_insufficient_blocks() {
        let mut scheduler = SchedulerState::with_block_size(3, 8);
        scheduler.enqueue(Request::new(1, 1, 1));
        scheduler.enqueue(Request::new(2, 1, 24));
        scheduler.enqueue(Request::new(3, 1, 1));

        let err = scheduler
            .pop_prefill_batch_with_allocation(2)
            .expect_err("should fail due to block shortage");
        assert!(
            err.contains("not enough KV blocks") && err.contains("RequestId(2)"),
            "{err}"
        );
        assert_eq!(scheduler.waiting_len(), 3);
        assert_eq!(
            scheduler
                .queue
                .waiting
                .iter()
                .map(|r| r.id)
                .collect::<Vec<_>>(),
            vec![RequestId(1), RequestId(2), RequestId(3)]
        );
        assert_eq!(scheduler.allocator_stats().free_blocks, 3);
        assert_eq!(scheduler.active_len(), 0);
        assert!(scheduler.active_request_ids().is_empty());
    }

    #[test]
    fn scheduler_state_pop_prefill_with_allocation_reports_overflow_error() {
        let mut scheduler = SchedulerState::with_block_size(4, 8);
        scheduler.enqueue(Request::new(1, usize::MAX, 1));

        let err = scheduler
            .pop_prefill_batch_with_allocation(1)
            .expect_err("should fail due to overflow");
        assert!(
            err.contains("token count overflow") && err.contains("RequestId(1)"),
            "{err}"
        );
        assert_eq!(scheduler.waiting_len(), 1);
        assert_eq!(scheduler.active_len(), 0);
        assert!(scheduler.active_request_ids().is_empty());
        assert_eq!(scheduler.allocator_stats().free_blocks, 4);
    }

    #[test]
    fn scheduler_state_pop_prefill_with_allocation_rolls_back_after_partial_overflow() {
        let mut scheduler = SchedulerState::with_block_size(4, 8);
        scheduler.enqueue(Request::new(1, 1, 1));
        scheduler.enqueue(Request::new(2, 1, usize::MAX));
        scheduler.enqueue(Request::new(3, 1, 1));

        let err = scheduler
            .pop_prefill_batch_with_allocation(2)
            .expect_err("should fail due to overflow in the second selected request");
        assert!(
            err.contains("token count overflow") && err.contains("RequestId(2)"),
            "{err}"
        );
        assert_eq!(scheduler.waiting_len(), 3);
        assert_eq!(
            scheduler
                .queue
                .waiting
                .iter()
                .map(|r| r.id)
                .collect::<Vec<_>>(),
            vec![RequestId(1), RequestId(2), RequestId(3)]
        );
        assert_eq!(scheduler.allocator_stats().free_blocks, 4);
        assert_eq!(scheduler.active_len(), 0);
        assert!(scheduler.active_request_ids().is_empty());
    }

    #[test]
    fn scheduler_state_pop_prefill_rejects_duplicate_request_ids_before_allocation() {
        let mut scheduler = SchedulerState::with_block_size(4, 8);
        scheduler.enqueue(Request::new(1, 1, 1));
        scheduler.enqueue(Request::new(1, 1, 1));

        let err = scheduler
            .pop_prefill_batch_with_allocation(2)
            .expect_err("duplicate request ids should be rejected");
        assert!(err.contains("appears multiple times"), "{err}");
        assert_eq!(
            scheduler
                .queue
                .waiting
                .iter()
                .map(|r| r.id)
                .collect::<Vec<_>>(),
            vec![RequestId(1), RequestId(1)]
        );
        assert_eq!(scheduler.allocator_stats().free_blocks, 4);
        assert_eq!(scheduler.active_len(), 0);
        assert!(scheduler.active_request_ids().is_empty());
    }

    #[test]
    fn scheduler_state_pop_prefill_rejects_already_active_request_before_allocation() {
        let mut scheduler = SchedulerState::with_block_size(4, 8);
        scheduler.enqueue(Request::new(1, 8, 4));
        let selected = scheduler
            .pop_prefill_batch_with_allocation(8)
            .expect("initial allocation should succeed");
        assert_eq!(selected[0].allocation.blocks, vec![0, 1]);
        let free_blocks_after_initial_allocation = scheduler.allocator_stats().free_blocks;

        scheduler.enqueue(Request::new(1, 1, 1));
        let err = scheduler
            .pop_prefill_batch_with_allocation(1)
            .expect_err("already active request id should be rejected");
        assert!(err.contains("already active"), "{err}");
        assert_eq!(
            scheduler
                .queue
                .waiting
                .iter()
                .map(|r| r.id)
                .collect::<Vec<_>>(),
            vec![RequestId(1)]
        );
        assert_eq!(
            scheduler.allocator_stats().free_blocks,
            free_blocks_after_initial_allocation
        );
        assert_eq!(scheduler.active_len(), 1);
        assert_eq!(
            scheduler
                .active_request(RequestId(1))
                .expect("request 1 should stay active")
                .allocation
                .blocks,
            vec![0, 1]
        );
    }

    #[test]
    fn scheduler_state_release_request_frees_allocated_blocks() {
        let mut scheduler = SchedulerState::with_block_size(4, 8);
        scheduler.enqueue(Request::new(1, 8, 4));
        let selected = scheduler
            .pop_prefill_batch_with_allocation(8)
            .expect("allocation should succeed");
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].allocation.blocks.len(), 2);
        assert_eq!(scheduler.allocator_stats().free_blocks, 2);

        assert_eq!(scheduler.release_request(RequestId(1)), 2);
        assert_eq!(scheduler.allocator_stats().free_blocks, 4);
    }

    #[test]
    fn scheduler_state_prefill_completion_and_decode_step_progress() {
        let mut scheduler = SchedulerState::with_block_size(8, 8);
        scheduler.enqueue(Request::new(10, 4, 3));
        let selected = scheduler
            .pop_prefill_batch_with_allocation(4)
            .expect("allocation should succeed");
        assert_eq!(selected.len(), 1);
        assert_eq!(scheduler.active_len(), 1);

        scheduler
            .complete_prefill(RequestId(10))
            .expect("complete prefill should succeed");
        let err = scheduler
            .complete_prefill(RequestId(10))
            .expect_err("prefill completion should reject when already done");
        assert!(err.contains("prefill already completed"), "{err}");

        let active = scheduler
            .active_request(RequestId(10))
            .expect("request 10 should be active");
        assert_eq!(active.cached_tokens, 4);
        assert_eq!(active.generated_tokens, 0);

        scheduler
            .advance_decode(RequestId(10))
            .expect("first decode step should succeed");
        let active = scheduler
            .active_request(RequestId(10))
            .expect("request 10 should be active");
        assert_eq!(active.cached_tokens, 5);
        assert_eq!(active.generated_tokens, 1);

        scheduler
            .advance_decode(RequestId(10))
            .expect("second decode step should succeed");
        let active = scheduler
            .active_request(RequestId(10))
            .expect("request 10 should be active");
        assert_eq!(active.cached_tokens, 6);
        assert_eq!(active.generated_tokens, 2);
    }

    #[test]
    fn scheduler_state_advance_decode_rejects_without_prefill_completion() {
        let mut scheduler = SchedulerState::with_block_size(4, 8);
        scheduler.enqueue(Request::new(1, 4, 2));
        scheduler
            .pop_prefill_batch_with_allocation(4)
            .expect("allocation should succeed");

        let err = scheduler
            .advance_decode(RequestId(1))
            .expect_err("decode should require prefill completion");
        assert!(err.contains("prefill not completed"), "{err}");
        let active = scheduler
            .active_request(RequestId(1))
            .expect("request 1 should be active");
        assert_eq!(active.cached_tokens, 0);
        assert_eq!(active.generated_tokens, 0);
    }

    #[test]
    fn scheduler_state_advance_decode_rejects_after_max_new_tokens() {
        let mut scheduler = SchedulerState::with_block_size(4, 8);
        scheduler.enqueue(Request::new(1, 1, 1));
        scheduler
            .pop_prefill_batch_with_allocation(1)
            .expect("allocation should succeed");
        scheduler
            .complete_prefill(RequestId(1))
            .expect("prefill completion should succeed");

        scheduler
            .advance_decode(RequestId(1))
            .expect("first decode step should succeed");
        let err = scheduler
            .advance_decode(RequestId(1))
            .expect_err("second decode step should exceed max_new_tokens");
        assert!(err.contains("exceeds max_new_tokens"), "{err}");
    }

    #[test]
    fn scheduler_state_release_request_clears_active_state() {
        let mut scheduler = SchedulerState::with_block_size(4, 8);
        scheduler.enqueue(Request::new(1, 8, 4));
        scheduler
            .pop_prefill_batch_with_allocation(8)
            .expect("allocation should succeed");
        assert_eq!(scheduler.active_len(), 1);
        assert!(scheduler.active_request(RequestId(1)).is_some());

        let freed = scheduler.release_request(RequestId(1));
        assert_eq!(freed, 2);
        assert_eq!(scheduler.active_len(), 0);
        assert!(scheduler.active_request(RequestId(1)).is_none());
        assert!(scheduler.active_request_ids().is_empty());
    }

    #[test]
    fn scheduler_state_ready_decode_batch_selects_only_ready_requests() {
        let mut scheduler = SchedulerState::with_block_size(8, 8);
        scheduler.enqueue(Request::new(1, 3, 2));
        scheduler.enqueue(Request::new(2, 2, 1));
        scheduler.enqueue(Request::new(3, 4, 3));
        let selected = scheduler
            .pop_prefill_batch_with_allocation(16)
            .expect("allocation should succeed");
        assert_eq!(
            selected
                .iter()
                .map(|entry| entry.request.id)
                .collect::<Vec<_>>(),
            vec![RequestId(1), RequestId(2), RequestId(3)]
        );

        scheduler
            .complete_prefill(RequestId(1))
            .expect("request 1 prefill complete");
        scheduler
            .complete_prefill(RequestId(2))
            .expect("request 2 prefill complete");
        scheduler
            .complete_prefill(RequestId(3))
            .expect("request 3 prefill complete");
        scheduler
            .advance_decode(RequestId(2))
            .expect("request 2 should advance one token");

        let ready = scheduler
            .ready_decode_batch(8)
            .expect("ready decode batch should be generated");
        assert_eq!(
            ready
                .iter()
                .map(|entry| entry.request.id)
                .collect::<Vec<_>>(),
            vec![RequestId(1), RequestId(3)]
        );
        assert_eq!(
            ready[0],
            SchedulerDecodeRequest {
                request: Request {
                    id: RequestId(1),
                    prompt_tokens: 3,
                    max_new_tokens: 2
                },
                allocation: scheduler
                    .active_request(RequestId(1))
                    .expect("request 1 should be active")
                    .allocation
                    .clone(),
                cached_tokens: 3,
                generated_tokens: 0,
                cache_position: 3,
                next_cache_len: 4,
                remaining_new_tokens: 2
            }
        );
        assert_eq!(ready[0].allocation.request_id, RequestId(1));
    }

    #[test]
    fn scheduler_state_ready_decode_batch_respects_max_requests_limit() {
        let mut scheduler = SchedulerState::with_block_size(8, 8);
        scheduler.enqueue(Request::new(1, 3, 2));
        scheduler.enqueue(Request::new(2, 2, 4));
        scheduler
            .pop_prefill_batch_with_allocation(8)
            .expect("allocation should succeed");
        scheduler
            .complete_prefill(RequestId(1))
            .expect("request 1 prefill complete");
        scheduler
            .complete_prefill(RequestId(2))
            .expect("request 2 prefill complete");
        let limited = scheduler
            .ready_decode_batch(1)
            .expect("ready decode batch should be generated");
        assert_eq!(
            limited
                .iter()
                .map(|entry| entry.request.id)
                .collect::<Vec<_>>(),
            vec![RequestId(1)]
        );
        assert_eq!(limited[0].cache_position, 3);
        assert_eq!(limited[0].next_cache_len, 4);
        assert_eq!(limited[0].remaining_new_tokens, 2);
    }

    #[test]
    fn scheduler_state_ready_decode_batch_tracks_multi_request_progress() {
        let mut scheduler = SchedulerState::with_block_size(8, 4);
        scheduler.enqueue(Request::new(1, 2, 2));
        scheduler.enqueue(Request::new(2, 3, 1));
        scheduler.enqueue(Request::new(3, 1, 0));
        scheduler
            .pop_prefill_batch_with_allocation(8)
            .expect("allocation should succeed");
        scheduler
            .complete_prefill(RequestId(1))
            .expect("request 1 prefill complete");
        scheduler
            .complete_prefill(RequestId(2))
            .expect("request 2 prefill complete");
        scheduler
            .complete_prefill(RequestId(3))
            .expect("request 3 prefill complete");

        let first = scheduler
            .ready_decode_batch(8)
            .expect("first ready decode batch should be generated");
        assert_eq!(
            first
                .iter()
                .map(|entry| entry.request.id)
                .collect::<Vec<_>>(),
            vec![RequestId(1), RequestId(2)]
        );
        assert_eq!(
            first
                .iter()
                .map(|entry| (
                    entry.cache_position,
                    entry.next_cache_len,
                    entry.remaining_new_tokens
                ))
                .collect::<Vec<_>>(),
            vec![(2, 3, 2), (3, 4, 1)]
        );

        scheduler
            .advance_decode_batch(&first)
            .expect("first ready batch should advance");
        let second = scheduler
            .ready_decode_batch(8)
            .expect("second ready decode batch should be generated");
        assert_eq!(
            second
                .iter()
                .map(|entry| entry.request.id)
                .collect::<Vec<_>>(),
            vec![RequestId(1)]
        );
        assert_eq!(second[0].cache_position, 3);
        assert_eq!(second[0].next_cache_len, 4);
        assert_eq!(second[0].remaining_new_tokens, 1);
    }

    #[test]
    fn scheduler_state_advance_decode_batch_rejects_stale_batch_without_partial_progress() {
        let mut scheduler = SchedulerState::with_block_size(8, 4);
        scheduler.enqueue(Request::new(1, 2, 2));
        scheduler.enqueue(Request::new(2, 3, 1));
        scheduler
            .pop_prefill_batch_with_allocation(8)
            .expect("allocation should succeed");
        scheduler
            .complete_prefill(RequestId(1))
            .expect("request 1 prefill complete");
        scheduler
            .complete_prefill(RequestId(2))
            .expect("request 2 prefill complete");

        let mut ready = scheduler
            .ready_decode_batch(8)
            .expect("ready decode batch should be generated");
        ready[1].generated_tokens = 99;
        let err = scheduler
            .advance_decode_batch(&ready)
            .expect_err("stale batch should be rejected");
        assert!(err.contains("stale"), "{err}");

        let first = scheduler
            .active_request(RequestId(1))
            .expect("request 1 should remain active");
        assert_eq!(first.cached_tokens, 2);
        assert_eq!(first.generated_tokens, 0);
        let second = scheduler
            .active_request(RequestId(2))
            .expect("request 2 should remain active");
        assert_eq!(second.cached_tokens, 3);
        assert_eq!(second.generated_tokens, 0);
    }

    #[test]
    fn scheduler_state_ready_decode_batch_excludes_unready_and_complete_requests() {
        let mut scheduler = SchedulerState::with_block_size(8, 8);
        scheduler.enqueue(Request::new(1, 3, 1));
        scheduler.enqueue(Request::new(2, 2, 1));
        scheduler
            .pop_prefill_batch_with_allocation(6)
            .expect("allocation should succeed");
        scheduler
            .complete_prefill(RequestId(1))
            .expect("request 1 prefill complete");

        let ready_before = scheduler
            .ready_decode_batch(4)
            .expect("ready decode batch should be generated");
        assert_eq!(
            ready_before
                .iter()
                .map(|entry| entry.request.id)
                .collect::<Vec<_>>(),
            vec![RequestId(1)]
        );
        assert_eq!(ready_before[0].cache_position, 3);
        assert_eq!(ready_before[0].next_cache_len, 4);
        assert_eq!(ready_before[0].remaining_new_tokens, 1);

        scheduler
            .advance_decode(RequestId(1))
            .expect("request 1 decode should advance");
        let ready_after = scheduler
            .ready_decode_batch(4)
            .expect("ready decode batch should be generated");
        assert!(ready_after.is_empty());
    }

    #[test]
    fn scheduler_state_ready_decode_batch_returns_empty_for_zero_limit_or_no_ready_requests() {
        let mut scheduler = SchedulerState::with_block_size(4, 8);
        scheduler.enqueue(Request::new(1, 8, 4));
        scheduler
            .pop_prefill_batch_with_allocation(8)
            .expect("allocation should succeed");
        let zero_limit = scheduler
            .ready_decode_batch(0)
            .expect("zero limit should be accepted");
        assert!(zero_limit.is_empty());

        let not_ready = scheduler
            .ready_decode_batch(4)
            .expect("ready decode batch should be generated");
        assert!(not_ready.is_empty());
    }
}
