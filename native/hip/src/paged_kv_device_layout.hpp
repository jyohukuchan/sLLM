#ifndef SLLM_PAGED_KV_DEVICE_LAYOUT_HPP
#define SLLM_PAGED_KV_DEVICE_LAYOUT_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace sllm_paged_kv {

// A logical 128-token block indexes this fixed device table through its
// physical block ID. Every plane of one block uses the same descriptor, so a
// fork or tail COW changes only the logical table entry. Optional planes are
// null for encodings that do not use them.
struct BlockDescriptor final {
  std::uint8_t *key;
  std::uint8_t *value;
  std::uint8_t *key_scale;
  std::uint8_t *value_scale;
  std::uint8_t *key_outer_scale;
  std::uint8_t *value_outer_scale;
};

static_assert(std::is_standard_layout_v<BlockDescriptor>);
static_assert(std::is_trivially_copyable_v<BlockDescriptor>);
static_assert(sizeof(BlockDescriptor) == 6U * sizeof(void *));
static_assert(offsetof(BlockDescriptor, value) == sizeof(void *));
static_assert(offsetof(BlockDescriptor, key_scale) == 2U * sizeof(void *));

} // namespace sllm_paged_kv

#endif // SLLM_PAGED_KV_DEVICE_LAYOUT_HPP
