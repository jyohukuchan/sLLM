// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, VecDeque};

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
}
