#ifndef SLLM_PAGED_KV_SLAB_PLAN_HPP
#define SLLM_PAGED_KV_SLAB_PLAN_HPP

// Host-only arithmetic for the segmented physical Paged KV pool.  This file
// intentionally does not allocate device memory or publish a descriptor.  It
// computes the byte locations that a later HIP allocator can use to fill a
// BlockDescriptor for one physical block.

#include "paged_kv_device_layout.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace sllm_paged_kv {

constexpr std::uint32_t kSlabBlocks = 8U;

enum class Plane : std::uint8_t {
  Key = 0U,
  Value,
  KeyScale,
  ValueScale,
  KeyOuterScale,
  ValueOuterScale,
};

using PlaneBytes = std::array<std::uint64_t, 6U>;
using PlaneOffsets = std::array<std::size_t, 6U>;

struct SlabPlanInput final {
  // The three counts describe one K or V token.  Key and value have the same
  // shape, while a zero scale count denotes an absent optional plane.
  std::uint64_t value_bytes_per_token;
  std::uint64_t scale_bytes_per_token;
  std::uint64_t outer_scale_bytes_per_token;
  std::uint64_t logical_capacity_tokens;
  std::uint64_t pool_capacity_blocks;
};

struct BlockLocation final {
  std::uint64_t slab;
  std::uint32_t slot;
};

// Offsets are relative to the beginning of a slab.  The corresponding
// descriptor can be constructed later as base + each offset.  A zero offset
// is valid for the first required plane; `present` distinguishes an absent
// optional plane from a plane that happens to start at offset zero.
struct DescriptorOffsetPlan final {
  PlaneOffsets offsets{};
  std::array<bool, 6U> present{};
};

class SlabPlan final {
public:
  static SlabPlan make(const SlabPlanInput &input) {
    validate_input(input);

    SlabPlan plan;
    plan.input_ = input;
    plan.logical_block_count_ =
        ceil_div(input.logical_capacity_tokens, kBlockTokens);
    if (plan.logical_block_count_ >
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
      throw std::invalid_argument("paged KV logical table exceeds ID space");
    plan.slab_count_ = ceil_div(input.pool_capacity_blocks, kSlabBlocks);
    plan.plane_bytes_per_block_ = {
        checked_mul(input.value_bytes_per_token, kBlockTokens),
        checked_mul(input.value_bytes_per_token, kBlockTokens),
        checked_mul(input.scale_bytes_per_token, kBlockTokens),
        checked_mul(input.scale_bytes_per_token, kBlockTokens),
        checked_mul(input.outer_scale_bytes_per_token, kBlockTokens),
        checked_mul(input.outer_scale_bytes_per_token, kBlockTokens),
    };

    // Blocks are laid out as six contiguous planes within each block.  This
    // gives every descriptor a fixed block stride while retaining null
    // pointers for optional planes.
    std::uint64_t block_stride = 0U;
    for (const std::uint64_t bytes : plan.plane_bytes_per_block_)
      block_stride = checked_add(block_stride, bytes);
    plan.block_stride_ = checked_size(block_stride);
    plan.slab_footprint_ = checked_size(
        checked_mul(block_stride, static_cast<std::uint64_t>(kSlabBlocks)));

    std::uint64_t plane_offset = 0U;
    for (std::size_t index = 0U; index < plan.plane_bytes_per_block_.size();
         ++index) {
      plan.present_[index] = plan.plane_bytes_per_block_[index] != 0U;
      plan.plane_offsets_[index] =
          plan.present_[index] ? checked_size(plane_offset) : 0U;
      plane_offset =
          checked_add(plane_offset, plan.plane_bytes_per_block_[index]);
    }
    return plan;
  }

  const SlabPlanInput &input() const noexcept { return input_; }
  std::uint64_t logical_block_count() const noexcept {
    return logical_block_count_;
  }
  std::uint64_t slab_count() const noexcept { return slab_count_; }
  std::uint64_t pool_capacity_blocks() const noexcept {
    return input_.pool_capacity_blocks;
  }
  std::size_t block_stride() const noexcept { return block_stride_; }
  std::size_t slab_footprint() const noexcept { return slab_footprint_; }
  const PlaneBytes &plane_bytes_per_block() const noexcept {
    return plane_bytes_per_block_;
  }
  const PlaneOffsets &plane_offsets() const noexcept { return plane_offsets_; }
  const std::array<bool, 6U> &present() const noexcept { return present_; }

  BlockLocation location(const std::uint64_t physical_id) const {
    if (physical_id >= input_.pool_capacity_blocks)
      throw std::out_of_range("paged KV physical block id");
    return {physical_id / kSlabBlocks,
            static_cast<std::uint32_t>(physical_id % kSlabBlocks)};
  }

  // Return all six pointers as offsets.  The caller must add the returned
  // values to the base of the slab selected by location(physical_id).
  DescriptorOffsetPlan
  descriptor_offsets(const std::uint64_t physical_id) const {
    const BlockLocation block = location(physical_id);
    const std::size_t slot_base =
        checked_size(checked_mul(static_cast<std::uint64_t>(block.slot),
                                 static_cast<std::uint64_t>(block_stride_)));
    DescriptorOffsetPlan result;
    for (std::size_t index = 0U; index < plane_offsets_.size(); ++index) {
      result.present[index] = present_[index];
      if (present_[index])
        result.offsets[index] = checked_size(
            checked_add(static_cast<std::uint64_t>(slot_base),
                        static_cast<std::uint64_t>(plane_offsets_[index])));
    }
    return result;
  }

private:
  SlabPlan() = default;

  static constexpr std::uint64_t kBlockTokens = 128U;

  static std::uint64_t checked_add(const std::uint64_t lhs,
                                   const std::uint64_t rhs) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs)
      throw std::overflow_error("paged KV slab size overflow");
    return lhs + rhs;
  }

  static std::uint64_t checked_mul(const std::uint64_t lhs,
                                   const std::uint64_t rhs) {
    if (lhs != 0U && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
      throw std::overflow_error("paged KV slab size overflow");
    return lhs * rhs;
  }

  static std::size_t checked_size(const std::uint64_t value) {
    if (value > std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("paged KV slab size_t overflow");
    return static_cast<std::size_t>(value);
  }

  static std::uint64_t ceil_div(const std::uint64_t value,
                                const std::uint64_t divisor) noexcept {
    return value / divisor + (value % divisor != 0U ? 1U : 0U);
  }

  static void validate_input(const SlabPlanInput &input) {
    if (input.value_bytes_per_token == 0U)
      throw std::invalid_argument("paged KV value plane is empty");
    if (input.logical_capacity_tokens == 0U)
      throw std::invalid_argument("paged KV logical capacity is empty");
    if (input.pool_capacity_blocks == 0U)
      throw std::invalid_argument("paged KV physical pool is empty");
    // Device logical and physical tables use 32-bit physical IDs.  Rejecting
    // a larger pool here keeps the host plan from producing an unaddressable
    // descriptor entry even on a 64-bit host.
    if (input.pool_capacity_blocks >=
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
      throw std::invalid_argument("paged KV physical pool exceeds ID space");
  }

  SlabPlanInput input_{};
  std::uint64_t logical_block_count_ = 0U;
  std::uint64_t slab_count_ = 0U;
  PlaneBytes plane_bytes_per_block_{};
  PlaneOffsets plane_offsets_{};
  std::array<bool, 6U> present_{};
  std::size_t block_stride_ = 0U;
  std::size_t slab_footprint_ = 0U;
};

static_assert(sizeof(BlockDescriptor) == 6U * sizeof(void *));

} // namespace sllm_paged_kv

#endif // SLLM_PAGED_KV_SLAB_PLAN_HPP
