#ifndef SLLM_PAGED_KV_POOL_STATE_HPP
#define SLLM_PAGED_KV_POOL_STATE_HPP

// Host ownership bookkeeping for a 128-token paged KV pool. Device plane
// allocation, block copies, and table publication stay with the HIP runtime.

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sllm_paged_kv {

constexpr uint32_t kBlockTokens = 128U;
constexpr uint32_t kInvalidBlock = std::numeric_limits<uint32_t>::max();
// A 1024-token sliding window occupies eight complete blocks.  The ninth
// slot carries the partial block at either edge while the window advances.
// Keeping that slot in the host contract avoids a transient ten-block pool
// requirement at the 1024/1025 and 1152/1153 boundaries.
constexpr uint64_t kSlidingWindowTokens = 1024U;
constexpr uint32_t kSlidingRingSlots = 9U;

class SlidingState;

enum class Status : uint8_t {
  Ok,
  Invalid,
  Busy,
  Capacity,
  PoolExhausted,
  Corrupt,
};

class Pool final {
public:
  explicit Pool(const uint32_t capacity_blocks)
      : capacity_blocks_(validate_capacity(capacity_blocks)),
        references_(capacity_blocks_, 0U), reserved_(capacity_blocks_, false) {
    // Every operation that returns a block to the free list is non-throwing
    // after construction. This lets append rollback restore the pool after a
    // multi-block reservation.
    free_.reserve(capacity_blocks_);
  }

  uint32_t capacity_blocks() const noexcept { return capacity_blocks_; }
  uint32_t reference_count(const uint32_t block) const noexcept {
    return block < capacity_blocks_ ? references_[block] : 0U;
  }
  uint32_t free_count() const noexcept {
    return static_cast<uint32_t>(free_.size()) + capacity_blocks_ -
           next_unused_;
  }

private:
  friend class State;
  friend class SlidingState;

  std::optional<uint32_t> reserve() {
    uint32_t block = kInvalidBlock;
    if (!free_.empty()) {
      block = free_.back();
      if (block >= capacity_blocks_ || references_[block] != 0U ||
          reserved_[block])
        throw std::logic_error("paged KV free list is corrupt");
      free_.pop_back();
    } else if (next_unused_ < capacity_blocks_) {
      block = next_unused_;
      if (references_[block] != 0U || reserved_[block])
        throw std::logic_error("paged KV allocation cursor is corrupt");
      ++next_unused_;
    } else {
      return std::nullopt;
    }
    if (references_[block] != 0U || reserved_[block])
      throw std::logic_error("paged KV free list is corrupt");
    reserved_[block] = true;
    return block;
  }

  bool can_abort_reserved(const uint32_t block) const noexcept {
    return block < capacity_blocks_ && reserved_[block] &&
           references_[block] == 0U;
  }

  bool abort_reserved(const uint32_t block) noexcept {
    if (block >= capacity_blocks_ || !reserved_[block] ||
        references_[block] != 0U)
      return false;
    // free_ was fully reserved at construction, so this cannot allocate.
    free_.push_back(block);
    reserved_[block] = false;
    return true;
  }

  bool retain(const uint32_t block) noexcept {
    if (block >= capacity_blocks_ || reserved_[block] ||
        references_[block] == 0U ||
        references_[block] == std::numeric_limits<uint32_t>::max())
      return false;
    ++references_[block];
    return true;
  }

  bool can_release(const uint32_t block) const noexcept {
    return block < capacity_blocks_ && !reserved_[block] &&
           references_[block] != 0U;
  }

  bool release(const uint32_t block) noexcept {
    if (block >= capacity_blocks_ || reserved_[block] ||
        references_[block] == 0U)
      return false;
    if (--references_[block] == 0U) {
      // free_ was fully reserved at construction, so this cannot allocate.
      free_.push_back(block);
    }
    return true;
  }

  bool can_replace(const uint32_t old_block,
                   const uint32_t new_block) const noexcept {
    return new_block < capacity_blocks_ && reserved_[new_block] &&
           references_[new_block] == 0U &&
           (old_block == kInvalidBlock || can_release(old_block));
  }

  bool replace_reserved(const uint32_t old_block,
                        const uint32_t new_block) noexcept {
    if (!can_replace(old_block, new_block))
      return false;
    references_[new_block] = 1U;
    reserved_[new_block] = false;
    if (old_block != kInvalidBlock && --references_[old_block] == 0U)
      free_.push_back(old_block);
    return true;
  }

  static uint32_t validate_capacity(const uint32_t capacity_blocks) {
    if (capacity_blocks == 0U || capacity_blocks == kInvalidBlock)
      throw std::invalid_argument("invalid paged KV pool capacity");
    return capacity_blocks;
  }

  uint32_t capacity_blocks_;
  uint32_t next_unused_ = 0U;
  std::vector<uint32_t> references_;
  std::vector<bool> reserved_;
  std::vector<uint32_t> free_;
};

class State final {
public:
  struct Change final {
    uint32_t logical_block;
    uint32_t old_physical;
    uint32_t new_physical;
    uint32_t prefix_tokens_to_copy;
  };

  struct Append final {
    State *owner = nullptr;
    uint64_t start = 0U;
    uint64_t end = 0U;
    std::vector<Change> changes;
    bool active = false;
    bool graph_reservation = false;
  };

  struct ImageImport final {
    State *owner = nullptr;
    uint64_t published_length = 0U;
    uint64_t generation = 0U;
    std::vector<uint32_t> physical_table;
    std::vector<uint32_t> reserved_blocks;
    bool active = false;
  };

  State(std::shared_ptr<Pool> pool, const uint64_t capacity_tokens)
      : pool_(std::move(pool)), capacity_tokens_(capacity_tokens),
        table_(block_count_for_capacity(capacity_tokens), kInvalidBlock) {
    if (!pool_)
      throw std::invalid_argument("invalid paged KV state pool");
  }

  uint64_t capacity_tokens() const noexcept { return capacity_tokens_; }
  uint64_t published_length() const noexcept { return published_length_; }
  uint64_t generation() const noexcept { return generation_; }
  bool pending() const noexcept { return pending_; }
  uint32_t physical_block(const uint32_t logical) const noexcept {
    return logical < table_.size() ? table_[logical] : kInvalidBlock;
  }
  size_t table_capacity() const noexcept { return table_.size(); }
  std::shared_ptr<Pool> pool() const noexcept { return pool_; }

  // The caller must finish all K/V and scale-plane writes before commit.
  // A private partial tail can be written in place because its new bytes are
  // beyond published_length; a shared tail receives a reserved COW block.
  Status prepare_append(const uint64_t tokens, Append &append) {
    if (graph_staged_)
      return Status::Busy;
    return prepare_from(published_length_, tokens, append, false);
  }

  // Reserve logical mappings for future graph replays without publishing
  // token length. A graph may run several replays before its final position
  // is known to the host, so staged mappings can grow incrementally.
  Status prepare_graph_through(const uint64_t end, Append &reservation) {
    const uint64_t start =
        graph_staged_ ? graph_staged_length_ : published_length_;
    if (end <= start)
      return Status::Invalid;
    return prepare_from(start, end - start, reservation, true);
  }

  Status commit_append(Append &append) {
    return commit_prepared(append, false);
  }

  Status commit_graph_reservation(Append &reservation) {
    return commit_prepared(reservation, true);
  }

  Status rollback_append(Append &append) {
    return rollback_prepared(append, false);
  }

  Status rollback_graph_reservation(Append &reservation) {
    return rollback_prepared(reservation, true);
  }

  // Called after the graph has drained. Unused speculative blocks are
  // released; only then does the actual final length become visible.
  Status finish_graph(const uint64_t final_length,
                      const uint64_t successful_generations) {
    if (released_ || !graph_staged_ || pending_ ||
        final_length < published_length_ ||
        final_length > graph_staged_length_ ||
        (final_length != published_length_ && successful_generations == 0U) ||
        successful_generations >
            std::numeric_limits<uint64_t>::max() - generation_)
      return Status::Invalid;
    const uint64_t first_released =
        final_length / kBlockTokens + (final_length % kBlockTokens != 0U);
    const uint64_t used = graph_staged_length_ / kBlockTokens +
                          (graph_staged_length_ % kBlockTokens != 0U);
    if (!can_release_suffix(first_released, used))
      return Status::Corrupt;
    release_suffix(first_released, used);
    published_length_ = final_length;
    generation_ += successful_generations;
    graph_staged_ = false;
    graph_staged_length_ = 0U;
    return Status::Ok;
  }

  bool graph_staged() const noexcept { return graph_staged_; }
  uint64_t graph_staged_length() const noexcept {
    return graph_staged_ ? graph_staged_length_ : published_length_;
  }

  Status prepare_image_import(const std::vector<uint32_t> &compact_table,
                              const uint64_t physical_blocks,
                              const uint64_t published_length,
                              const uint64_t generation, ImageImport &import) {
    if (released_ || pending_ || graph_staged_ || import.active)
      return Status::Busy;
    if (published_length_ != 0U || generation_ != 0U ||
        compact_table.size() != table_.size() ||
        physical_blocks > pool_->capacity_blocks() ||
        physical_blocks > std::numeric_limits<uint32_t>::max() ||
        published_length > capacity_tokens_ ||
        generation == std::numeric_limits<uint64_t>::max())
      return Status::Invalid;
    if (published_length != 0U && physical_blocks == 0U)
      return Status::Invalid;
    try {
      import.physical_table.assign(table_.size(), kInvalidBlock);
      import.reserved_blocks.reserve(static_cast<size_t>(physical_blocks));
      std::vector<bool> seen(static_cast<size_t>(physical_blocks), false);
      for (const uint32_t compact : compact_table) {
        if (compact == kInvalidBlock)
          continue;
        if (compact >= physical_blocks || seen[compact]) {
          rollback_image_import(import);
          return Status::Invalid;
        }
        seen[compact] = true;
        const auto reserved = pool_->reserve();
        if (!reserved) {
          rollback_image_import(import);
          return Status::PoolExhausted;
        }
        import.reserved_blocks.push_back(*reserved);
      }
      if (import.reserved_blocks.size() != physical_blocks) {
        rollback_image_import(import);
        return Status::Invalid;
      }
      for (size_t logical = 0U; logical != compact_table.size(); ++logical) {
        const uint32_t compact = compact_table[logical];
        if (compact != kInvalidBlock)
          import.physical_table[logical] = import.reserved_blocks[compact];
      }
      import.owner = this;
      import.published_length = published_length;
      import.generation = generation;
      import.active = true;
      return Status::Ok;
    } catch (...) {
      rollback_image_import(import);
      return Status::Corrupt;
    }
  }

  Status commit_image_import(ImageImport &import) noexcept {
    if (import.owner != this || !import.active || pending_ || graph_staged_)
      return Status::Invalid;
    for (const uint32_t physical : import.reserved_blocks) {
      if (!pool_->replace_reserved(kInvalidBlock, physical))
        return Status::Corrupt;
    }
    table_ = import.physical_table;
    published_length_ = import.published_length;
    generation_ = import.generation;
    import.active = false;
    import.owner = nullptr;
    import.reserved_blocks.clear();
    return Status::Ok;
  }

  Status rollback_image_import(ImageImport &import) noexcept {
    if (import.owner != nullptr && import.owner != this)
      return Status::Invalid;
    bool ok = true;
    for (auto it = import.reserved_blocks.rbegin();
         it != import.reserved_blocks.rend(); ++it) {
      if (!pool_->abort_reserved(*it))
        ok = false;
    }
    import = ImageImport{};
    return ok ? Status::Ok : Status::Corrupt;
  }

private:
  Status prepare_from(const uint64_t start, const uint64_t tokens,
                      Append &append, const bool graph_reservation) {
    if (released_ || tokens == 0U || append.active)
      return Status::Invalid;
    if (pending_)
      return Status::Busy;
    if (start > capacity_tokens_ || tokens > capacity_tokens_ - start)
      return Status::Capacity;
    if (generation_ == std::numeric_limits<uint64_t>::max())
      return Status::Capacity;
    Append next;
    next.owner = this;
    next.start = start;
    next.end = start + tokens;
    next.graph_reservation = graph_reservation;
    const uint64_t first = next.start / kBlockTokens;
    const uint64_t last = (next.end - 1U) / kBlockTokens;
    // Allocate this metadata before touching the pool. A failed allocation
    // therefore leaves both the logical table and physical pool unchanged.
    next.changes.reserve(static_cast<size_t>(last - first + 1U));
    try {
      for (uint64_t logical = first; logical <= last; ++logical) {
        if (logical >= table_.size()) {
          abort_prepared(next);
          return Status::Corrupt;
        }
        const uint32_t old_block = table_[logical];
        if (old_block != kInvalidBlock &&
            pool_->reference_count(old_block) == 0U) {
          abort_prepared(next);
          return Status::Corrupt;
        }
        if (old_block != kInvalidBlock &&
            pool_->reference_count(old_block) == 1U)
          continue;
        const auto reserved = pool_->reserve();
        if (!reserved) {
          abort_prepared(next);
          return Status::PoolExhausted;
        }
        const uint64_t block_begin = logical * kBlockTokens;
        const uint32_t prefix = old_block == kInvalidBlock
                                    ? 0U
                                    : static_cast<uint32_t>(std::min<uint64_t>(
                                          published_length_ > block_begin
                                              ? published_length_ - block_begin
                                              : 0U,
                                          kBlockTokens));
        next.changes.push_back(Change{static_cast<uint32_t>(logical), old_block,
                                      *reserved, prefix});
      }
    } catch (...) {
      // reserve() may throw if an internal free-list invariant is corrupt.
      // Restore every reservation made before the exception before exposing
      // the exception to the caller.
      (void)abort_prepared(next);
      throw;
    }
    try {
      // Append is intentionally a public transport object because the HIP
      // runtime needs to inspect its table changes. Keep a private copy of
      // the reservation plan so that a caller cannot edit, erase, or append
      // entries after prepare and make commit operate on a different plan.
      pending_changes_ = next.changes;
    } catch (...) {
      (void)abort_prepared(next);
      throw;
    }
    pending_start_ = next.start;
    pending_end_ = next.end;
    pending_graph_ = graph_reservation;
    pending_ = true;
    next.active = true;
    append = std::move(next);
    pending_append_ = &append;
    return Status::Ok;
  }

  // Call only after device table entry updates have completed on the append
  // stream. The caller owns device rollback if an update fails before here.
  Status commit_prepared(Append &append, const bool graph_reservation) {
    if (append.owner != this || !append.active || !pending_ ||
        pending_append_ != &append || pending_graph_ != graph_reservation ||
        append.graph_reservation != graph_reservation)
      return Status::Invalid;
    if (!matches_pending(append))
      return Status::Corrupt;
    for (size_t index = 0U; index < pending_changes_.size(); ++index) {
      const Change &change = pending_changes_[index];
      for (size_t prior = 0U; prior < index; ++prior) {
        const Change &previous = pending_changes_[prior];
        if (previous.logical_block == change.logical_block ||
            previous.new_physical == change.new_physical)
          return Status::Corrupt;
      }
      if (change.logical_block >= table_.size() ||
          table_[change.logical_block] != change.old_physical ||
          !pool_->can_replace(change.old_physical, change.new_physical))
        return Status::Corrupt;
      // A corrupt logical table could alias one physical block more than
      // once. Preflight all releases so the commit loop remains atomic even
      // under that condition.
      if (change.old_physical != kInvalidBlock) {
        uint64_t needed = 1U;
        for (size_t prior = 0U; prior < index; ++prior)
          if (pending_changes_[prior].old_physical == change.old_physical)
            ++needed;
        if (pool_->reference_count(change.old_physical) < needed)
          return Status::Corrupt;
      }
    }
    for (const Change &change : pending_changes_) {
      // All failure conditions were checked above. replace_reserved is
      // non-throwing, so commit cannot fail halfway through this transaction.
      if (!pool_->replace_reserved(change.old_physical, change.new_physical))
        return Status::Corrupt;
      table_[change.logical_block] = change.new_physical;
    }
    if (graph_reservation) {
      graph_staged_ = true;
      graph_staged_length_ = append.end;
    } else {
      published_length_ = append.end;
      ++generation_;
    }
    pending_ = false;
    append.active = false;
    append.owner = nullptr;
    pending_append_ = nullptr;
    pending_changes_.clear();
    pending_graph_ = false;
    return Status::Ok;
  }

  Status rollback_prepared(Append &append, const bool graph_reservation) {
    if (append.owner != this || !append.active || !pending_ ||
        pending_append_ != &append || pending_graph_ != graph_reservation ||
        append.graph_reservation != graph_reservation)
      return Status::Invalid;
    // Use the private plan so rollback remains able to release every
    // reservation even when public Append::changes was truncated or edited
    // after prepare.
    if (!abort_prepared(pending_changes_))
      return Status::Corrupt;
    pending_ = false;
    append.active = false;
    append.owner = nullptr;
    pending_append_ = nullptr;
    pending_changes_.clear();
    pending_graph_ = false;
    return Status::Ok;
  }

public:
  Status fork_into(std::unique_ptr<State> &out) {
    return fork_into(out, capacity_tokens_);
  }

  Status fork_into(std::unique_ptr<State> &out,
                   const uint64_t child_capacity_tokens) {
    if (released_ || out)
      return Status::Invalid;
    if (pending_)
      return Status::Busy;
    if (graph_staged_)
      return Status::Busy;
    if (child_capacity_tokens < published_length_ ||
        child_capacity_tokens == 0U)
      return Status::Capacity;
    auto child = std::make_unique<State>(pool_, child_capacity_tokens);
    const uint64_t used = published_length_ / kBlockTokens +
                          (published_length_ % kBlockTokens != 0U);
    for (uint64_t logical = 0U; logical < used; ++logical) {
      const uint32_t block = table_[logical];
      if (block == kInvalidBlock || !pool_->retain(block)) {
        for (uint64_t undo = 0U; undo < logical; ++undo)
          (void)pool_->release(child->table_[undo]);
        return Status::Corrupt;
      }
      child->table_[logical] = block;
    }
    child->published_length_ = published_length_;
    child->generation_ = generation_;
    out = std::move(child);
    return Status::Ok;
  }

  // A quiescent rollback changes only the published suffix. Blocks no longer
  // reachable from the new length are released; a retained partial tail stays
  // shared until a later append performs COW.
  Status rewind_last(const uint64_t expected_length,
                     const uint64_t rewind_length) {
    if (released_ || expected_length != published_length_ ||
        rewind_length > expected_length)
      return Status::Invalid;
    if (pending_ || graph_staged_)
      return Status::Busy;
    if (generation_ == std::numeric_limits<uint64_t>::max())
      return Status::Capacity;
    const uint64_t first_released =
        rewind_length / kBlockTokens + (rewind_length % kBlockTokens != 0U);
    const uint64_t used = published_length_ / kBlockTokens +
                          (published_length_ % kBlockTokens != 0U);
    if (!can_release_suffix(first_released, used))
      return Status::Corrupt;
    release_suffix(first_released, used);
    published_length_ = rewind_length;
    ++generation_;
    return Status::Ok;
  }

  Status release() {
    if (released_)
      return Status::Invalid;
    if (pending_ || graph_staged_)
      return Status::Busy;
    const uint64_t used = published_length_ / kBlockTokens +
                          (published_length_ % kBlockTokens != 0U);
    if (used > table_.size())
      return Status::Corrupt;
    for (uint64_t logical = 0U; logical < used; ++logical) {
      if (table_[logical] == kInvalidBlock ||
          !pool_->can_release(table_[logical]))
        return Status::Corrupt;
      uint64_t needed = 1U;
      for (uint64_t prior = 0U; prior < logical; ++prior)
        if (table_[prior] == table_[logical])
          ++needed;
      if (pool_->reference_count(table_[logical]) < needed)
        return Status::Corrupt;
    }
    for (uint64_t logical = 0U; logical < used; ++logical) {
      (void)pool_->release(table_[logical]);
      table_[logical] = kInvalidBlock;
    }
    published_length_ = 0U;
    released_ = true;
    return Status::Ok;
  }

private:
  bool can_release_suffix(const uint64_t first,
                          const uint64_t used) const noexcept {
    if (first > used || used > table_.size())
      return false;
    for (uint64_t logical = first; logical < used; ++logical) {
      const uint32_t block = table_[logical];
      if (block == kInvalidBlock || !pool_->can_release(block))
        return false;
      uint64_t needed = 1U;
      for (uint64_t prior = first; prior < logical; ++prior)
        if (table_[prior] == block)
          ++needed;
      if (pool_->reference_count(block) < needed)
        return false;
    }
    return true;
  }

  void release_suffix(const uint64_t first, const uint64_t used) noexcept {
    for (uint64_t logical = first; logical < used; ++logical) {
      (void)pool_->release(table_[logical]);
      table_[logical] = kInvalidBlock;
    }
  }

  static size_t block_count_for_capacity(const uint64_t capacity_tokens) {
    if (capacity_tokens == 0U)
      throw std::invalid_argument("invalid paged KV state capacity");
    const uint64_t blocks =
        capacity_tokens / kBlockTokens + (capacity_tokens % kBlockTokens != 0U);
    if (blocks > static_cast<uint64_t>(kInvalidBlock))
      throw std::invalid_argument("paged KV state capacity is too large");
    return static_cast<size_t>(blocks);
  }

  bool abort_prepared(const std::vector<Change> &changes) {
    for (const Change &change : changes) {
      if (!pool_->can_abort_reserved(change.new_physical))
        return false;
    }
    for (auto change = changes.rbegin(); change != changes.rend(); ++change)
      (void)pool_->abort_reserved(change->new_physical);
    return true;
  }

  bool abort_prepared(const Append &append) {
    return abort_prepared(append.changes);
  }

  bool matches_pending(const Append &append) const noexcept {
    if (append.start != pending_start_ || append.end != pending_end_ ||
        append.start != (pending_graph_ && graph_staged_ ? graph_staged_length_
                                                         : published_length_) ||
        append.graph_reservation != pending_graph_ ||
        append.end <= append.start || append.end > capacity_tokens_ ||
        append.changes.size() != pending_changes_.size())
      return false;
    for (size_t index = 0U; index < pending_changes_.size(); ++index) {
      const Change &actual = append.changes[index];
      const Change &expected = pending_changes_[index];
      if (actual.logical_block != expected.logical_block ||
          actual.old_physical != expected.old_physical ||
          actual.new_physical != expected.new_physical ||
          actual.prefix_tokens_to_copy != expected.prefix_tokens_to_copy)
        return false;
    }
    return true;
  }

  std::shared_ptr<Pool> pool_;
  uint64_t capacity_tokens_;
  uint64_t published_length_ = 0U;
  uint64_t generation_ = 0U;
  std::vector<uint32_t> table_;
  bool pending_ = false;
  bool pending_graph_ = false;
  bool graph_staged_ = false;
  uint64_t graph_staged_length_ = 0U;
  bool released_ = false;
  Append *pending_append_ = nullptr;
  uint64_t pending_start_ = 0U;
  uint64_t pending_end_ = 0U;
  std::vector<Change> pending_changes_;
};

// Host ownership state for a fixed 1024-token sliding window.  Unlike State,
// logical_block in this class is an absolute block number and the physical
// table has nine ring slots.  The absolute tag is part of the contract: a
// device table entry is usable only when its slot tag matches the requested
// absolute block.  This keeps wrap at block 9 (token 1152) explicit and lets
// the runtime reject a stale table entry instead of silently reading an old
// window.
class SlidingState final {
public:
  static constexpr uint64_t kWindowTokens = kSlidingWindowTokens;
  static constexpr uint32_t kRingSlots = kSlidingRingSlots;

  struct Change final {
    uint64_t absolute_block = 0U;
    uint32_t ring_slot = 0U;
    uint32_t old_physical = kInvalidBlock;
    uint32_t new_physical = kInvalidBlock;
    uint32_t prefix_tokens_to_copy = 0U;
    // A sole owner of an expired slot can be retagged in place.  Reusing the
    // block is what permits a single-state ring to operate with nine pool
    // blocks; a shared expired block instead takes a normal COW reservation.
    bool recycle_expired = false;
  };

  struct ImageImport final {
    SlidingState *owner = nullptr;
    uint64_t published_length = 0U;
    uint64_t generation = 0U;
    std::array<uint32_t, kRingSlots> table{};
    std::array<uint64_t, kRingSlots> tags{};
    std::vector<uint32_t> reserved_blocks;
    bool active = false;
  };

  struct Append final {
    SlidingState *owner = nullptr;
    uint64_t start = 0U;
    uint64_t end = 0U;
    std::vector<Change> changes;
    bool active = false;
  };

  explicit SlidingState(std::shared_ptr<Pool> pool) : pool_(std::move(pool)) {
    if (!pool_)
      throw std::invalid_argument("invalid sliding paged KV state pool");
    table_.fill(kInvalidBlock);
    tags_.fill(kInvalidTag);
  }

  uint64_t published_length() const noexcept { return published_length_; }
  uint64_t generation() const noexcept { return generation_; }
  bool pending() const noexcept { return pending_; }
  uint64_t retained_begin() const noexcept {
    return published_length_ > kWindowTokens ? published_length_ - kWindowTokens
                                             : 0U;
  }
  uint64_t retained_end() const noexcept { return published_length_; }
  uint32_t ring_slot(const uint64_t absolute_block) const noexcept {
    return static_cast<uint32_t>(absolute_block % kRingSlots);
  }
  uint64_t absolute_block_tag(const uint32_t slot) const noexcept {
    return slot < kRingSlots ? tags_[slot] : kInvalidTag;
  }
  uint32_t physical_block(const uint64_t absolute_block) const noexcept {
    const uint32_t slot = ring_slot(absolute_block);
    return tags_[slot] == absolute_block ? table_[slot] : kInvalidBlock;
  }
  uint32_t physical_block_at_slot(const uint32_t slot) const noexcept {
    return slot < kRingSlots ? table_[slot] : kInvalidBlock;
  }
  uint32_t retained_block_count() const noexcept {
    const uint64_t begin = retained_begin();
    if (begin >= published_length_)
      return 0U;
    const uint64_t first = begin / kBlockTokens;
    const uint64_t last = (published_length_ - 1U) / kBlockTokens;
    return static_cast<uint32_t>(last - first + 1U);
  }
  std::shared_ptr<Pool> pool() const noexcept { return pool_; }

  Status prepare_image_import(const std::array<uint32_t, kRingSlots> &table,
                              const std::array<uint64_t, kRingSlots> &tags,
                              const uint64_t physical_blocks,
                              const uint64_t published_length,
                              const uint64_t generation, ImageImport &import) {
    if (released_ || pending_ || import.active)
      return Status::Busy;
    if (published_length_ != 0U || generation_ != 0U ||
        physical_blocks > pool_->capacity_blocks() ||
        physical_blocks > std::numeric_limits<uint32_t>::max() ||
        published_length < retained_begin() ||
        generation == std::numeric_limits<uint64_t>::max())
      return Status::Invalid;
    std::array<bool, kRingSlots> seen{};
    try {
      import.reserved_blocks.reserve(static_cast<size_t>(physical_blocks));
      for (uint32_t slot = 0U; slot != kRingSlots; ++slot) {
        if (tags[slot] == std::numeric_limits<uint64_t>::max()) {
          if (table[slot] != kInvalidBlock)
            return Status::Invalid;
          continue;
        }
        if (ring_slot(tags[slot]) != slot || table[slot] == kInvalidBlock ||
            table[slot] >= physical_blocks || seen[slot])
          return Status::Invalid;
        seen[slot] = true;
      }
      std::vector<bool> compact_seen(static_cast<size_t>(physical_blocks),
                                     false);
      for (uint32_t slot = 0U; slot != kRingSlots; ++slot) {
        if (table[slot] == kInvalidBlock)
          continue;
        if (compact_seen[table[slot]])
          return Status::Invalid;
        compact_seen[table[slot]] = true;
        const auto reserved = pool_->reserve();
        if (!reserved) {
          rollback_image_import(import);
          return Status::PoolExhausted;
        }
        import.reserved_blocks.push_back(*reserved);
      }
      if (import.reserved_blocks.size() != physical_blocks) {
        rollback_image_import(import);
        return Status::Invalid;
      }
      import.table.fill(kInvalidBlock);
      import.tags = tags;
      for (uint32_t slot = 0U; slot != kRingSlots; ++slot) {
        if (table[slot] != kInvalidBlock)
          import.table[slot] = import.reserved_blocks[table[slot]];
      }
      import.owner = this;
      import.published_length = published_length;
      import.generation = generation;
      import.active = true;
      return Status::Ok;
    } catch (...) {
      rollback_image_import(import);
      return Status::Corrupt;
    }
  }

  Status commit_image_import(ImageImport &import) noexcept {
    if (import.owner != this || !import.active || pending_)
      return Status::Invalid;
    for (const uint32_t physical : import.reserved_blocks) {
      if (!pool_->replace_reserved(kInvalidBlock, physical))
        return Status::Corrupt;
    }
    table_ = import.table;
    tags_ = import.tags;
    published_length_ = import.published_length;
    generation_ = import.generation;
    import.active = false;
    import.owner = nullptr;
    import.reserved_blocks.clear();
    return Status::Ok;
  }

  Status rollback_image_import(ImageImport &import) noexcept {
    if (import.owner != nullptr && import.owner != this)
      return Status::Invalid;
    bool ok = true;
    for (auto it = import.reserved_blocks.rbegin();
         it != import.reserved_blocks.rend(); ++it) {
      if (!pool_->abort_reserved(*it))
        ok = false;
    }
    import = ImageImport{};
    return ok ? Status::Ok : Status::Corrupt;
  }

  // An append may cover at most one complete ring turn.  Larger prefills are
  // intentionally chunked by the caller so each transaction has one unique
  // slot per Change and remains fully rollbackable.
  Status prepare_append(const uint64_t tokens, Append &append) {
    if (released_ || pending_ || append.active || tokens == 0U)
      return pending_ ? Status::Busy : Status::Invalid;
    if (tokens > static_cast<uint64_t>(kRingSlots) * kBlockTokens ||
        published_length_ > std::numeric_limits<uint64_t>::max() - tokens)
      return Status::Capacity;
    const uint64_t end = published_length_ + tokens;
    const uint64_t first = published_length_ / kBlockTokens;
    const uint64_t last = (end - 1U) / kBlockTokens;
    if (last - first + 1U > kRingSlots ||
        generation_ == std::numeric_limits<uint64_t>::max())
      return Status::Capacity;

    Append next;
    next.owner = this;
    next.start = published_length_;
    next.end = end;
    try {
      next.changes.reserve(static_cast<size_t>(last - first + 1U));
      for (uint64_t absolute = first; absolute <= last; ++absolute) {
        const uint32_t slot = ring_slot(absolute);
        const uint64_t old_tag = tags_[slot];
        const uint32_t old_block = table_[slot];
        if (old_tag == absolute) {
          if (old_block == kInvalidBlock ||
              pool_->reference_count(old_block) == 0U) {
            abort_prepared(next.changes);
            return Status::Corrupt;
          }
          if (pool_->reference_count(old_block) == 1U)
            continue;
          const auto reserved = pool_->reserve();
          if (!reserved) {
            abort_prepared(next.changes);
            return Status::PoolExhausted;
          }
          next.changes.push_back(Change{absolute, slot, old_block, *reserved,
                                        prefix_for_write(absolute, next.start),
                                        false});
          continue;
        }

        // A slot can only contain an older block when it is outside the new
        // retained window.  Reuse it in place when this state is its sole
        // owner; a forked sibling requires a fresh physical block.
        const bool has_old = old_tag != kInvalidTag;
        if (has_old && (old_block == kInvalidBlock ||
                        pool_->reference_count(old_block) == 0U)) {
          abort_prepared(next.changes);
          return Status::Corrupt;
        }
        if (has_old && pool_->reference_count(old_block) == 1U) {
          next.changes.push_back(
              Change{absolute, slot, old_block, old_block, 0U, true});
          continue;
        }
        const auto reserved = pool_->reserve();
        if (!reserved) {
          abort_prepared(next.changes);
          return Status::PoolExhausted;
        }
        next.changes.push_back(Change{absolute, slot,
                                      has_old ? old_block : kInvalidBlock,
                                      *reserved, 0U, false});
      }
    } catch (...) {
      (void)abort_prepared(next.changes);
      throw;
    }
    try {
      pending_changes_ = next.changes;
    } catch (...) {
      (void)abort_prepared(next.changes);
      throw;
    }
    pending_start_ = next.start;
    pending_end_ = next.end;
    pending_ = true;
    next.active = true;
    pending_append_ = &append;
    append = std::move(next);
    pending_append_ = &append;
    return Status::Ok;
  }

  Status commit_append(Append &append) {
    if (append.owner != this || !append.active || !pending_ ||
        pending_append_ != &append || !matches_pending(append))
      return Status::Invalid;
    const uint64_t new_retained_begin =
        append.end > kWindowTokens ? append.end - kWindowTokens : 0U;
    const uint64_t first_kept = new_retained_begin / kBlockTokens;
    if (!validate_commit_plan() || !validate_retire_plan(first_kept))
      return Status::Corrupt;

    for (const Change &change : pending_changes_) {
      const uint32_t current = table_[change.ring_slot];
      if (change.recycle_expired) {
        // The physical block remains owned by this state.  Only its absolute
        // tag changes, so no pool counter is touched.
        table_[change.ring_slot] = change.new_physical;
      } else if (!pool_->replace_reserved(current, change.new_physical)) {
        return Status::Corrupt;
      } else {
        table_[change.ring_slot] = change.new_physical;
      }
      tags_[change.ring_slot] = change.absolute_block;
    }

    for (uint32_t slot = 0U; slot < kRingSlots; ++slot) {
      if (tags_[slot] == kInvalidTag || tags_[slot] >= first_kept)
        continue;
      if (!pool_->can_release(table_[slot]) || !pool_->release(table_[slot]))
        return Status::Corrupt;
      tags_[slot] = kInvalidTag;
      table_[slot] = kInvalidBlock;
    }
    published_length_ = append.end;
    ++generation_;
    clear_pending(append);
    return Status::Ok;
  }

  Status rollback_append(Append &append) {
    if (append.owner != this || !append.active || !pending_ ||
        pending_append_ != &append)
      return Status::Invalid;
    if (!matches_pending(append))
      return Status::Corrupt;
    if (!abort_prepared(pending_changes_))
      return Status::Corrupt;
    clear_pending(append);
    return Status::Ok;
  }

  Status fork_into(std::unique_ptr<SlidingState> &out) {
    if (released_ || out)
      return Status::Invalid;
    if (pending_)
      return Status::Busy;
    auto child = std::make_unique<SlidingState>(pool_);
    for (uint32_t slot = 0U; slot < kRingSlots; ++slot) {
      if (tags_[slot] == kInvalidTag)
        continue;
      const uint32_t block = table_[slot];
      if (block == kInvalidBlock || !pool_->retain(block)) {
        for (uint32_t undo = 0U; undo < slot; ++undo) {
          if (child->tags_[undo] != kInvalidTag)
            (void)pool_->release(child->table_[undo]);
        }
        return Status::Corrupt;
      }
      child->tags_[slot] = tags_[slot];
      child->table_[slot] = block;
    }
    child->published_length_ = published_length_;
    child->generation_ = generation_;
    out = std::move(child);
    return Status::Ok;
  }

  // Rewind is valid only within the still-retained absolute window.  Blocks
  // beyond the new end are released; a partial tail remains available for a
  // subsequent append and follows the same COW rules as a normal tail.
  Status rewind_last(const uint64_t expected_length,
                     const uint64_t rewind_length) {
    if (released_ || pending_ || expected_length != published_length_ ||
        rewind_length > expected_length || rewind_length < retained_begin())
      return (pending_ ? Status::Busy : Status::Invalid);
    if (generation_ == std::numeric_limits<uint64_t>::max())
      return Status::Capacity;
    const uint64_t first_kept =
        (rewind_length > kWindowTokens ? rewind_length - kWindowTokens : 0U) /
        kBlockTokens;
    const uint64_t first_removed =
        rewind_length / kBlockTokens + (rewind_length % kBlockTokens != 0U);
    for (uint32_t slot = 0U; slot < kRingSlots; ++slot) {
      const uint64_t tag = tags_[slot];
      if (tag == kInvalidTag || (tag >= first_kept && tag < first_removed))
        continue;
      if (!pool_->can_release(table_[slot]))
        return Status::Corrupt;
    }
    for (uint32_t slot = 0U; slot < kRingSlots; ++slot) {
      const uint64_t tag = tags_[slot];
      if (tag == kInvalidTag || (tag >= first_kept && tag < first_removed))
        continue;
      (void)pool_->release(table_[slot]);
      tags_[slot] = kInvalidTag;
      table_[slot] = kInvalidBlock;
    }
    published_length_ = rewind_length;
    ++generation_;
    return Status::Ok;
  }

  Status release() {
    if (released_)
      return Status::Invalid;
    if (pending_)
      return Status::Busy;
    for (uint32_t slot = 0U; slot < kRingSlots; ++slot) {
      if (tags_[slot] == kInvalidTag)
        continue;
      if (!pool_->can_release(table_[slot]))
        return Status::Corrupt;
    }
    for (uint32_t slot = 0U; slot < kRingSlots; ++slot) {
      if (tags_[slot] == kInvalidTag)
        continue;
      (void)pool_->release(table_[slot]);
      tags_[slot] = kInvalidTag;
      table_[slot] = kInvalidBlock;
    }
    published_length_ = 0U;
    released_ = true;
    return Status::Ok;
  }

private:
  static constexpr uint64_t kInvalidTag = std::numeric_limits<uint64_t>::max();

  uint32_t prefix_for_write(const uint64_t absolute_block,
                            const uint64_t start) const noexcept {
    const uint64_t block_begin = absolute_block * kBlockTokens;
    if (start <= block_begin)
      return 0U;
    return static_cast<uint32_t>(std::min<uint64_t>(
        start - block_begin, static_cast<uint64_t>(kBlockTokens)));
  }

  bool matches_pending(const Append &append) const noexcept {
    if (append.start != pending_start_ || append.end != pending_end_ ||
        append.changes.size() != pending_changes_.size())
      return false;
    for (size_t index = 0U; index < pending_changes_.size(); ++index) {
      const Change &actual = append.changes[index];
      const Change &expected = pending_changes_[index];
      if (actual.absolute_block != expected.absolute_block ||
          actual.ring_slot != expected.ring_slot ||
          actual.old_physical != expected.old_physical ||
          actual.new_physical != expected.new_physical ||
          actual.prefix_tokens_to_copy != expected.prefix_tokens_to_copy ||
          actual.recycle_expired != expected.recycle_expired)
        return false;
    }
    return true;
  }

  bool validate_commit_plan() const noexcept {
    for (size_t index = 0U; index < pending_changes_.size(); ++index) {
      const Change &change = pending_changes_[index];
      if (change.ring_slot >= kRingSlots ||
          ring_slot(change.absolute_block) != change.ring_slot ||
          table_[change.ring_slot] != change.old_physical)
        return false;
      for (size_t prior = 0U; prior < index; ++prior) {
        if (pending_changes_[prior].ring_slot == change.ring_slot)
          return false;
      }
      if (change.recycle_expired) {
        if (change.old_physical == kInvalidBlock ||
            pool_->reference_count(change.old_physical) != 1U ||
            tags_[change.ring_slot] == change.absolute_block)
          return false;
      } else if (change.new_physical == kInvalidBlock ||
                 !pool_->can_replace(change.old_physical,
                                     change.new_physical)) {
        return false;
      }
    }
    return true;
  }

  bool validate_retire_plan(const uint64_t first_kept) const noexcept {
    for (uint32_t slot = 0U; slot < kRingSlots; ++slot) {
      const uint64_t tag = tags_[slot];
      if (tag == kInvalidTag || tag >= first_kept)
        continue;
      bool replaced = false;
      for (const Change &change : pending_changes_) {
        if (change.ring_slot == slot) {
          replaced = true;
          break;
        }
      }
      if (!replaced &&
          (table_[slot] == kInvalidBlock || !pool_->can_release(table_[slot])))
        return false;
    }
    return true;
  }

  void clear_pending(Append &append) noexcept {
    pending_ = false;
    pending_append_ = nullptr;
    pending_changes_.clear();
    append.active = false;
    append.owner = nullptr;
  }

  bool abort_prepared(const std::vector<Change> &changes) noexcept {
    for (const Change &change : changes) {
      if (!change.recycle_expired && change.new_physical != kInvalidBlock &&
          !pool_->can_abort_reserved(change.new_physical))
        return false;
    }
    for (auto change = changes.rbegin(); change != changes.rend(); ++change) {
      if (!change->recycle_expired && change->new_physical != kInvalidBlock)
        (void)pool_->abort_reserved(change->new_physical);
    }
    return true;
  }

  std::shared_ptr<Pool> pool_;
  std::array<uint32_t, kRingSlots> table_{};
  std::array<uint64_t, kRingSlots> tags_{};
  uint64_t published_length_ = 0U;
  uint64_t generation_ = 0U;
  bool pending_ = false;
  bool released_ = false;
  Append *pending_append_ = nullptr;
  uint64_t pending_start_ = 0U;
  uint64_t pending_end_ = 0U;
  std::vector<Change> pending_changes_;
};

} // namespace sllm_paged_kv

#endif // SLLM_PAGED_KV_POOL_STATE_HPP
