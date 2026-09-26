#include "../src/paged_kv_pool_state.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using sllm_paged_kv::kInvalidBlock;
using sllm_paged_kv::Pool;
using sllm_paged_kv::State;
using sllm_paged_kv::Status;

void expect(const bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

void expect_status(const Status actual, const Status expected,
                   const std::string &message) {
  expect(actual == expected, message);
}

void append(State &state, const uint64_t tokens) {
  State::Append transaction;
  expect_status(state.prepare_append(tokens, transaction), Status::Ok,
                "append prepare failed");
  expect_status(state.commit_append(transaction), Status::Ok,
                "append commit failed");
  expect(!state.pending(), "append left state pending");
}

void sliding_append(sllm_paged_kv::SlidingState &state, const uint64_t tokens) {
  sllm_paged_kv::SlidingState::Append transaction;
  expect_status(state.prepare_append(tokens, transaction), Status::Ok,
                "sliding append prepare failed");
  expect_status(state.commit_append(transaction), Status::Ok,
                "sliding append commit failed");
  expect(!state.pending(), "sliding append left state pending");
}

template <typename Function> void expect_invalid_argument(Function &&function) {
  bool threw = false;
  try {
    function();
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  expect(threw, "expected std::invalid_argument");
}

void test_capacity_validation_and_boundaries() {
  expect_invalid_argument([] { Pool pool(0U); });
  expect_invalid_argument([] { Pool pool(kInvalidBlock); });
  expect_invalid_argument([] {
    auto pool = std::make_shared<Pool>(1U);
    State state(pool, 0U);
  });
  expect_invalid_argument([] {
    auto pool = std::make_shared<Pool>(1U);
    State state(pool, UINT64_MAX);
  });

  auto pool = std::make_shared<Pool>(520U);
  State state(pool, 65537U);
  expect(state.table_capacity() == 513U, "boundary table size mismatch");
  append(state, 65535U);
  expect(state.published_length() == 65535U, "127 boundary append mismatch");
  expect(state.physical_block(511U) != kInvalidBlock,
         "partial tail block was not allocated");
  expect(state.physical_block(512U) == kInvalidBlock,
         "next block was allocated too early");

  const uint32_t tail = state.physical_block(511U);
  append(state, 1U);
  expect(state.published_length() == 65536U, "128 boundary append mismatch");
  expect(state.physical_block(511U) == tail,
         "private partial tail was unnecessarily copied");
  expect(state.physical_block(512U) == kInvalidBlock,
         "boundary block changed before it was used");

  append(state, 1U);
  expect(state.published_length() == 65537U, "129 boundary append mismatch");
  expect(state.physical_block(512U) != kInvalidBlock,
         "new block was not allocated at 65536/65537 boundary");
  expect(pool->free_count() == 7U, "boundary allocation free count mismatch");
  expect_status(state.release(), Status::Ok, "boundary release failed");
  expect(pool->free_count() == pool->capacity_blocks(),
         "boundary release did not return all blocks");
  expect_status(state.release(), Status::Invalid,
                "release was not idempotence-protected");
}

void test_partial_tail_cow_and_refcounts() {
  auto pool = std::make_shared<Pool>(8U);
  auto parent = std::make_unique<State>(pool, 512U);
  append(*parent, 129U);
  const uint32_t prefix = parent->physical_block(0U);
  const uint32_t shared_tail = parent->physical_block(1U);

  std::unique_ptr<State> child;
  expect_status(parent->fork_into(child), Status::Ok, "fork failed");
  expect(child != nullptr, "fork did not return child state");
  expect(pool->reference_count(prefix) == 2U,
         "fork did not retain prefix block");
  expect(pool->reference_count(shared_tail) == 2U,
         "fork did not retain partial tail");

  State::Append cow;
  expect_status(parent->prepare_append(1U, cow), Status::Ok,
                "shared tail COW prepare failed");
  expect(cow.changes.size() == 1U, "COW did not prepare one replacement");
  expect(cow.changes.front().prefix_tokens_to_copy == 1U,
         "COW prefix length mismatch");
  const uint32_t copied_tail = cow.changes.front().new_physical;
  expect(copied_tail != shared_tail, "COW reused shared tail block");
  expect_status(parent->commit_append(cow), Status::Ok,
                "shared tail COW commit failed");
  expect(parent->physical_block(1U) == copied_tail,
         "parent did not publish COW block");
  expect(child->physical_block(1U) == shared_tail,
         "child mapping changed during parent COW");
  expect(pool->reference_count(shared_tail) == 1U,
         "COW did not release parent reference");

  // The child now owns the old tail privately and can extend it in place.
  append(*child, 1U);
  expect(child->physical_block(1U) == shared_tail,
         "private tail was unnecessarily copied");
  expect(parent->published_length() == 130U &&
             child->published_length() == 130U,
         "forked lengths diverged unexpectedly");

  expect_status(parent->release(), Status::Ok, "parent release failed");
  expect_status(child->release(), Status::Ok, "child release failed");
  expect(pool->free_count() == pool->capacity_blocks(),
         "forked release leaked a physical block");
}

void test_fork_can_grow_logical_capacity() {
  auto pool = std::make_shared<Pool>(5U);
  State parent(pool, 129U);
  append(parent, 129U);
  const uint32_t shared_tail = parent.physical_block(1U);

  std::unique_ptr<State> rejected;
  expect_status(parent.fork_into(rejected, 128U), Status::Capacity,
                "fork accepted capacity below published length");
  expect(!rejected && pool->reference_count(shared_tail) == 1U,
         "rejected fork changed ownership");

  std::unique_ptr<State> child;
  expect_status(parent.fork_into(child, 300U), Status::Ok,
                "fork with larger logical capacity failed");
  expect(child->capacity_tokens() == 300U && child->table_capacity() == 3U,
         "fork did not use the child capacity");
  expect(pool->reference_count(shared_tail) == 2U,
         "grown fork did not retain shared tail");
  append(*child, 128U);
  expect(child->published_length() == 257U && parent.published_length() == 129U,
         "grown child append changed the parent length");
  expect(child->physical_block(1U) != shared_tail &&
             child->physical_block(2U) != kInvalidBlock,
         "grown child did not COW and allocate the new block");
  expect_status(parent.release(), Status::Ok, "grown parent release failed");
  expect_status(child->release(), Status::Ok, "grown child release failed");
  expect(pool->free_count() == pool->capacity_blocks(),
         "grown fork leaked blocks");
}

void test_rewind_preserves_sibling_and_cow_tail() {
  auto pool = std::make_shared<Pool>(6U);
  State parent(pool, 384U);
  append(parent, 257U);
  const uint32_t shared_tail = parent.physical_block(1U);
  const uint32_t removed_block = parent.physical_block(2U);
  std::unique_ptr<State> child;
  expect_status(parent.fork_into(child), Status::Ok, "rewind fork failed");
  const uint64_t generation = child->generation();
  expect_status(child->rewind_last(256U, 129U), Status::Invalid,
                "rewind accepted stale expected length");
  expect_status(child->rewind_last(257U, 129U), Status::Ok,
                "rewind across block boundary failed");
  expect(child->published_length() == 129U &&
             child->generation() == generation + 1U &&
             child->physical_block(2U) == kInvalidBlock,
         "rewind did not publish length and release suffix");
  expect(parent.published_length() == 257U &&
             parent.physical_block(2U) == removed_block &&
             pool->reference_count(removed_block) == 1U,
         "rewind changed the sibling's suffix");
  append(*child, 1U);
  expect(child->physical_block(1U) != shared_tail &&
             parent.physical_block(1U) == shared_tail,
         "rewound shared tail did not COW on append");
  expect_status(parent.release(), Status::Ok, "rewind parent release failed");
  expect_status(child->release(), Status::Ok, "rewind child release failed");
  expect(pool->free_count() == pool->capacity_blocks(),
         "rewind leaked physical blocks");
}

void test_graph_reservation_stages_without_publishing() {
  auto pool = std::make_shared<Pool>(8U);
  State parent(pool, 512U);
  append(parent, 127U);
  std::unique_ptr<State> child;
  expect_status(parent.fork_into(child), Status::Ok,
                "graph reservation fork failed");
  const uint32_t shared_tail = child->physical_block(0U);
  const uint64_t initial_generation = child->generation();

  State::Append first;
  expect_status(child->prepare_graph_through(145U, first), Status::Ok,
                "first graph reservation prepare failed");
  expect(first.graph_reservation && first.changes.size() == 2U &&
             first.changes[0].prefix_tokens_to_copy == 127U,
         "first graph reservation did not COW shared tail");
  expect_status(child->commit_append(first), Status::Invalid,
                "graph reservation accepted ordinary append commit");
  expect_status(child->commit_graph_reservation(first), Status::Ok,
                "first graph reservation commit failed");
  expect(child->published_length() == 127U &&
             child->graph_staged_length() == 145U &&
             child->generation() == initial_generation &&
             child->physical_block(0U) != shared_tail,
         "graph reservation changed visible length or missed COW");
  expect_status(child->release(), Status::Busy,
                "graph-staged child was released");

  State::Append second;
  expect_status(child->prepare_graph_through(257U, second), Status::Ok,
                "second graph reservation prepare failed");
  expect(second.changes.size() == 1U && second.changes[0].logical_block == 2U,
         "second graph reservation did not add the next block");
  expect_status(child->rollback_graph_reservation(second), Status::Ok,
                "second graph reservation rollback failed");
  expect(child->graph_staged_length() == 145U &&
             child->physical_block(2U) == kInvalidBlock,
         "graph rollback changed staged mapping");
  expect_status(child->prepare_graph_through(257U, second), Status::Ok,
                "second graph reservation retry failed");
  expect_status(child->commit_graph_reservation(second), Status::Ok,
                "second graph reservation commit failed");
  expect_status(child->finish_graph(129U, 0U), Status::Invalid,
                "graph length advanced without a successful generation");
  expect_status(child->finish_graph(129U, 2U), Status::Ok,
                "graph final publication failed");
  expect(!child->graph_staged() && child->published_length() == 129U &&
             child->generation() == initial_generation + 2U &&
             child->physical_block(2U) == kInvalidBlock &&
             parent.published_length() == 127U &&
             parent.physical_block(0U) == shared_tail,
         "graph finalization leaked speculative suffix or changed parent");
  expect_status(parent.release(), Status::Ok, "graph parent cleanup failed");
  expect_status(child->release(), Status::Ok, "graph child cleanup failed");
  expect(pool->free_count() == pool->capacity_blocks(),
         "graph reservation leaked physical blocks");
}

void test_rollback_and_pool_exhaustion() {
  auto pool = std::make_shared<Pool>(8U);
  State state(pool, 512U);
  append(state, 129U);
  const uint32_t old_tail = state.physical_block(1U);
  const uint64_t old_generation = state.generation();
  const uint32_t free_before = pool->free_count();

  State::Append transaction;
  expect_status(state.prepare_append(128U, transaction), Status::Ok,
                "rollback prepare failed");
  expect(state.pending(), "prepare did not mark state pending");
  expect(pool->free_count() + 1U == free_before,
         "rollback prepare did not reserve exactly one block");
  expect_status(state.release(), Status::Busy,
                "release ignored a pending append");
  expect_status(state.rollback_append(transaction), Status::Ok,
                "rollback failed");
  expect(!state.pending(), "rollback left state pending");
  expect(state.published_length() == 129U &&
             state.generation() == old_generation,
         "rollback changed published state");
  expect(state.physical_block(1U) == old_tail,
         "rollback changed logical table");
  expect(pool->free_count() == free_before,
         "rollback leaked or consumed a physical block");

  // Append metadata is public because the runtime fills it with device table
  // work. Reject a tampered transaction before changing any pool ownership.
  auto malformed_pool = std::make_shared<Pool>(4U);
  State malformed(malformed_pool, 256U);
  State::Append malformed_transaction;
  expect_status(malformed.prepare_append(256U, malformed_transaction),
                Status::Ok, "malformed transaction setup failed");
  expect(malformed_transaction.changes.size() == 2U,
         "malformed transaction setup shape mismatch");
  malformed_transaction.end = 257U;
  expect_status(malformed.commit_append(malformed_transaction), Status::Corrupt,
                "transaction beyond capacity was committed");
  malformed_transaction.end = 256U;
  malformed_transaction.changes[1].logical_block =
      malformed_transaction.changes[0].logical_block;
  expect_status(malformed.commit_append(malformed_transaction), Status::Corrupt,
                "tampered transaction was committed");
  expect_status(malformed.rollback_append(malformed_transaction), Status::Ok,
                "tampered transaction could not be rolled back");
  expect(malformed.published_length() == 0U &&
             malformed_pool->free_count() == 4U,
         "tampered transaction changed pool ownership");

  auto exhausted_pool = std::make_shared<Pool>(1U);
  State exhausted(exhausted_pool, 256U);
  State::Append failed;
  expect_status(exhausted.prepare_append(129U, failed), Status::PoolExhausted,
                "pool exhaustion was not reported");
  expect(!exhausted.pending() && !failed.active,
         "failed prepare left transaction active");
  expect(exhausted.published_length() == 0U &&
             exhausted_pool->free_count() == 1U,
         "failed multi-block prepare did not roll back its reservation");

  append(exhausted, 128U);
  State::Append second;
  expect_status(exhausted.prepare_append(1U, second), Status::PoolExhausted,
                "full pool did not reject append");
  expect(exhausted_pool->free_count() == 0U,
         "full pool free count changed after rejection");
  expect_status(exhausted.release(), Status::Ok,
                "exhausted state release failed");
  expect(exhausted_pool->free_count() == 1U,
         "exhausted state did not return its block");
}

void test_append_metadata_is_transactional() {
  // Removing a required change must not hide its reservation from rollback.
  {
    auto pool = std::make_shared<Pool>(4U);
    State state(pool, 256U);
    State::Append transaction;
    expect_status(state.prepare_append(256U, transaction), Status::Ok,
                  "clear-change setup failed");
    expect(transaction.changes.size() == 2U,
           "clear-change setup shape mismatch");
    const uint32_t free_before = pool->free_count();
    transaction.changes.clear();
    expect_status(state.commit_append(transaction), Status::Corrupt,
                  "cleared change was committed");
    expect(state.pending() && state.published_length() == 0U,
           "cleared change altered published state");
    expect(pool->free_count() == free_before,
           "cleared change altered pool ownership");
    expect_status(state.rollback_append(transaction), Status::Ok,
                  "cleared change could not be rolled back");
    expect(pool->free_count() == pool->capacity_blocks(),
           "cleared change leaked hidden reservations");
  }

  // A valid in-range end value is still invalid when it differs from the
  // prepared transaction. This case has no Change entries because the
  // private partial tail is extended in place.
  {
    auto pool = std::make_shared<Pool>(2U);
    State state(pool, 256U);
    append(state, 1U);
    State::Append transaction;
    expect_status(state.prepare_append(1U, transaction), Status::Ok,
                  "in-range end setup failed");
    expect(transaction.changes.empty(),
           "private-tail setup unexpectedly COWed");
    transaction.end = 127U;
    expect_status(state.commit_append(transaction), Status::Corrupt,
                  "in-range end tampering was committed");
    expect(state.published_length() == 1U && state.pending(),
           "in-range end tampering changed state");
    expect_status(state.rollback_append(transaction), Status::Ok,
                  "in-range end tampering could not be rolled back");

    expect_status(state.prepare_append(1U, transaction), Status::Ok,
                  "in-range start setup failed");
    transaction.start = 129U;
    expect_status(state.commit_append(transaction), Status::Corrupt,
                  "in-range start tampering was committed");
    expect(state.published_length() == 1U && state.pending(),
           "in-range start tampering changed state");
    expect_status(state.rollback_append(transaction), Status::Ok,
                  "in-range start tampering could not be rolled back");
    expect_status(state.release(), Status::Ok,
                  "in-range tampering cleanup failed");
  }

  // An inserted unrelated change must be rejected before the valid
  // reservation is consumed.
  {
    auto pool = std::make_shared<Pool>(3U);
    State state(pool, 256U);
    State::Append transaction;
    expect_status(state.prepare_append(128U, transaction), Status::Ok,
                  "inserted-change setup failed");
    const uint32_t free_before = pool->free_count();
    transaction.changes.push_back(
        State::Change{1U, kInvalidBlock, kInvalidBlock, 0U});
    expect_status(state.commit_append(transaction), Status::Corrupt,
                  "inserted unrelated change was committed");
    expect(pool->free_count() == free_before && state.pending(),
           "inserted change altered pool ownership");
    expect_status(state.rollback_append(transaction), Status::Ok,
                  "inserted change could not be rolled back");
  }

  // Replacing a reserved block with an unreserved block must fail before
  // ownership mutation, even when the logical table entry is otherwise valid.
  {
    auto pool = std::make_shared<Pool>(3U);
    State state(pool, 256U);
    State::Append transaction;
    expect_status(state.prepare_append(128U, transaction), Status::Ok,
                  "unreserved-block setup failed");
    expect(transaction.changes.size() == 1U,
           "unreserved-block setup shape mismatch");
    const uint32_t reserved = transaction.changes.front().new_physical;
    const uint32_t unreserved = reserved == 0U ? 2U : 0U;
    const uint32_t free_before = pool->free_count();
    transaction.changes.front().new_physical = unreserved;
    expect_status(state.commit_append(transaction), Status::Corrupt,
                  "unreserved block claim was committed");
    expect(pool->free_count() == free_before && state.pending(),
           "unreserved block claim altered pool ownership");
    expect_status(state.rollback_append(transaction), Status::Ok,
                  "unreserved block claim could not be rolled back");
    expect(pool->free_count() == pool->capacity_blocks(),
           "unreserved block cleanup leaked the reservation");
  }
}

void test_fork_and_pending_contracts() {
  auto pool = std::make_shared<Pool>(4U);
  State state(pool, 256U);
  State::Append pending;
  expect_status(state.prepare_append(128U, pending), Status::Ok,
                "pending setup failed");
  State::Append copied = pending;
  const uint32_t free_during_pending = pool->free_count();
  expect_status(state.commit_append(copied), Status::Invalid,
                "copied append handle committed the reservation");
  expect_status(state.rollback_append(copied), Status::Invalid,
                "copied append handle released the reservation");
  expect(state.pending() && pool->free_count() == free_during_pending,
         "copied append handle changed pending ownership");
  std::unique_ptr<State> child;
  expect_status(state.fork_into(child), Status::Busy,
                "fork ignored pending append");
  expect_status(state.rollback_append(pending), Status::Ok,
                "pending cleanup failed");

  append(state, 128U);
  std::unique_ptr<State> occupied = std::make_unique<State>(pool, 256U);
  expect_status(state.fork_into(occupied), Status::Invalid,
                "fork overwrote an occupied output");
  expect_status(state.fork_into(child), Status::Ok,
                "fork after rollback failed");
  expect_status(state.release(), Status::Ok, "source release failed");
  expect_status(child->release(), Status::Ok, "child cleanup failed");
  expect(occupied->release() == Status::Ok, "occupied output cleanup failed");
}

void test_sliding_ring_boundaries_and_recycling() {
  auto pool = std::make_shared<Pool>(sllm_paged_kv::kSlidingRingSlots);
  sllm_paged_kv::SlidingState state(pool);

  sliding_append(state, 1023U);
  expect(state.published_length() == 1023U && state.retained_begin() == 0U,
         "sliding 1023 boundary length mismatch");
  expect(state.retained_block_count() == 8U,
         "sliding 1023 retained block count mismatch");
  const uint32_t block7 = state.physical_block(7U);
  expect(block7 != kInvalidBlock && state.absolute_block_tag(7U) == 0U + 7U,
         "sliding 1023 absolute tag mismatch");

  sliding_append(state, 1U);
  expect(state.published_length() == 1024U && state.retained_begin() == 0U &&
             state.retained_block_count() == 8U &&
             state.physical_block(7U) == block7,
         "sliding 1024 boundary changed the partial tail");

  sliding_append(state, 1U);
  expect(state.published_length() == 1025U && state.retained_begin() == 1U &&
             state.retained_block_count() == 9U &&
             state.physical_block(0U) != kInvalidBlock &&
             state.physical_block(8U) != kInvalidBlock,
         "sliding 1025 boundary did not expose nine tagged slots");
  expect(state.absolute_block_tag(8U) == 8U,
         "sliding 1025 new block tag mismatch");

  sliding_append(state, 126U);
  expect(state.published_length() == 1151U && state.retained_begin() == 127U &&
             state.retained_block_count() == 9U,
         "sliding 1151 boundary mismatch");
  sliding_append(state, 1U);
  expect(state.published_length() == 1152U && state.retained_begin() == 128U &&
             state.retained_block_count() == 8U &&
             state.absolute_block_tag(0U) ==
                 std::numeric_limits<uint64_t>::max() &&
             pool->free_count() == 1U,
         "sliding 1152 did not retire the expired block");

  const uint32_t free_before_wrap = pool->free_count();
  sllm_paged_kv::SlidingState::Append rolled_back;
  expect_status(state.prepare_append(1U, rolled_back), Status::Ok,
                "sliding 1153 rollback prepare failed");
  expect(pool->free_count() + 1U == free_before_wrap,
         "sliding 1153 rollback did not reserve one block");
  expect_status(state.rollback_append(rolled_back), Status::Ok,
                "sliding 1153 rollback failed");
  expect(state.published_length() == 1152U &&
             pool->free_count() == free_before_wrap,
         "sliding rollback changed the retained ring");

  sliding_append(state, 1U);
  expect(state.published_length() == 1153U && state.retained_begin() == 129U &&
             state.retained_block_count() == 9U &&
             state.absolute_block_tag(0U) == 9U &&
             state.physical_block(9U) != kInvalidBlock,
         "sliding 1153 wrap did not retag slot zero");
  expect_status(state.release(), Status::Ok, "sliding ring release failed");
  expect(pool->free_count() == pool->capacity_blocks(),
         "sliding ring release leaked a physical block");
}

void test_sliding_shared_cow_and_rewind() {
  auto pool = std::make_shared<Pool>(18U);
  sllm_paged_kv::SlidingState parent(pool);
  sliding_append(parent, 1023U);
  const uint32_t shared_tail = parent.physical_block(7U);
  std::unique_ptr<sllm_paged_kv::SlidingState> child;
  expect_status(parent.fork_into(child), Status::Ok,
                "sliding fork at 1023 failed");
  expect(pool->reference_count(shared_tail) == 2U,
         "sliding fork did not share the partial tail");

  sliding_append(*child, 1U);
  expect(child->published_length() == 1024U &&
             child->physical_block(7U) != shared_tail &&
             parent.physical_block(7U) == shared_tail &&
             pool->reference_count(shared_tail) == 1U,
         "sliding partial-tail COW changed the sibling");

  sliding_append(parent, 1U);
  expect(parent.published_length() == 1024U &&
             parent.physical_block(7U) == shared_tail,
         "sliding parent tail was unexpectedly copied");
  sliding_append(parent, 1U);
  expect(parent.published_length() == 1025U &&
             parent.absolute_block_tag(8U) == 8U,
         "sliding parent failed to add block eight");
  expect_status(parent.rewind_last(1025U, 1024U), Status::Ok,
                "sliding rewind across 1024 boundary failed");
  expect(parent.published_length() == 1024U && parent.retained_begin() == 0U &&
             parent.physical_block(8U) == kInvalidBlock,
         "sliding rewind retained a removed block");

  expect_status(parent.release(), Status::Ok, "sliding parent cleanup failed");
  expect_status(child->release(), Status::Ok, "sliding child cleanup failed");
  expect(pool->free_count() == pool->capacity_blocks(),
         "sliding fork cleanup leaked a physical block");
}

} // namespace

int main() {
  try {
    test_capacity_validation_and_boundaries();
    test_partial_tail_cow_and_refcounts();
    test_fork_can_grow_logical_capacity();
    test_rewind_preserves_sibling_and_cow_tail();
    test_graph_reservation_stages_without_publishing();
    test_rollback_and_pool_exhaustion();
    test_append_metadata_is_transactional();
    test_fork_and_pending_contracts();
    test_sliding_ring_boundaries_and_recycling();
    test_sliding_shared_cow_and_rewind();
  } catch (const std::exception &error) {
    std::cerr << "phase87_stage10_paged_pool_host_test: FAIL: " << error.what()
              << '\n';
    return 1;
  }
  std::cout << "phase87_stage10_paged_pool_host_test: PASS\n";
  return 0;
}
