#include "sllm/hip.h"

#include <cstddef>
#include <iostream>
#include <type_traits>

static_assert(SLLM_HIP_KV_MEMORY_KIND_CAPABILITY_SELECTED == 0U);
static_assert(SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS == 1U);
static_assert(SLLM_HIP_KV_MEMORY_KIND_CONTIGUOUS_RESIDENT == 2U);
static_assert(SLLM_HIP_KV_MEMORY_KIND_PAGED == 3U);
static_assert(SLLM_HIP_KV_PAGED_TOKEN_BLOCK_SIZE == 128U);
static_assert(SLLM_HIP_KV_PAGED_STATE_FORK_VERSION == 1U);
static_assert(SLLM_HIP_KV_PAGED_STATE_FORK_INFO_VERSION == 1U);

static_assert(std::is_standard_layout_v<sllm_kv_state_paged_create_info_t>);
static_assert(sizeof(sllm_kv_state_paged_create_info_t) == 128U);
static_assert(alignof(sllm_kv_state_paged_create_info_t) == 8U);
static_assert(offsetof(sllm_kv_state_paged_create_info_t,
                       quantization_block_size) == 68U);
static_assert(offsetof(sllm_kv_state_paged_create_info_t, token_block_size) ==
              72U);
static_assert(offsetof(sllm_kv_state_paged_create_info_t,
                       physical_layout_version) == 76U);
static_assert(offsetof(sllm_kv_state_paged_create_info_t,
                       logical_table_capacity) == 80U);
static_assert(offsetof(sllm_kv_state_paged_create_info_t,
                       max_physical_blocks) == 88U);
static_assert(offsetof(sllm_kv_state_paged_create_info_t,
                       sliding_window_tokens) == 96U);
static_assert(offsetof(sllm_kv_state_paged_create_info_t,
                       static_key_scale_bits) == 104U);
static_assert(offsetof(sllm_kv_state_paged_create_info_t,
                       static_value_scale_bits) == 108U);

static_assert(std::is_standard_layout_v<sllm_kv_paged_view_info_t>);
static_assert(sizeof(sllm_kv_paged_view_info_t) == 200U);
static_assert(alignof(sllm_kv_paged_view_info_t) == 8U);
static_assert(offsetof(sllm_kv_paged_view_info_t, token_block_size) == 52U);
static_assert(offsetof(sllm_kv_paged_view_info_t, physical_layout_version) ==
              56U);
static_assert(offsetof(sllm_kv_paged_view_info_t, logical_table_capacity) ==
              88U);
static_assert(offsetof(sllm_kv_paged_view_info_t, max_physical_blocks) == 96U);
static_assert(offsetof(sllm_kv_paged_view_info_t, allocated_physical_blocks) ==
              104U);
static_assert(offsetof(sllm_kv_paged_view_info_t, committed_bytes_per_plane) ==
              112U);
static_assert(offsetof(sllm_kv_paged_view_info_t, committed_bytes_total) ==
              160U);

static_assert(std::is_standard_layout_v<sllm_kv_paged_state_fork_info_t>);
static_assert(sizeof(sllm_kv_paged_state_fork_info_t) == 152U);
static_assert(alignof(sllm_kv_paged_state_fork_info_t) == 8U);
static_assert(offsetof(sllm_kv_paged_state_fork_info_t, token_block_size) ==
              72U);
static_assert(offsetof(sllm_kv_paged_state_fork_info_t,
                       physical_layout_version) == 76U);
static_assert(offsetof(sllm_kv_paged_state_fork_info_t,
                       source_logical_table_capacity) == 80U);
static_assert(offsetof(sllm_kv_paged_state_fork_info_t,
                       committed_bytes_total) == 128U);

static_assert(std::is_standard_layout_v<sllm_kv_paged_image_info_t>);
static_assert(sizeof(sllm_kv_paged_image_info_t) == 272U);
static_assert(alignof(sllm_kv_paged_image_info_t) == 8U);
static_assert(offsetof(sllm_kv_paged_image_info_t, flags) == 12U);
static_assert(offsetof(sllm_kv_paged_image_info_t, session_id) == 16U);
static_assert(offsetof(sllm_kv_paged_image_info_t, capacity_tokens) == 56U);
static_assert(offsetof(sllm_kv_paged_image_info_t, logical_table_capacity) ==
              104U);
static_assert(offsetof(sllm_kv_paged_image_info_t, plane_count) == 120U);
static_assert(offsetof(sllm_kv_paged_image_info_t, plane_block_stride) == 136U);
static_assert(offsetof(sllm_kv_paged_image_info_t, plane_bytes) == 184U);
static_assert(offsetof(sllm_kv_paged_image_info_t, static_key_scale_bits) ==
              232U);
static_assert(offsetof(sllm_kv_paged_image_info_t, reserved) == 240U);

static_assert(std::is_standard_layout_v<sllm_kv_paged_image_chunk_t>);
static_assert(sizeof(sllm_kv_paged_image_chunk_t) == 72U);
static_assert(alignof(sllm_kv_paged_image_chunk_t) == 8U);
static_assert(offsetof(sllm_kv_paged_image_chunk_t, section) == 12U);
static_assert(offsetof(sllm_kv_paged_image_chunk_t, plane) == 16U);
static_assert(offsetof(sllm_kv_paged_image_chunk_t, byte_offset) == 24U);
static_assert(offsetof(sllm_kv_paged_image_chunk_t, byte_length) == 32U);
static_assert(offsetof(sllm_kv_paged_image_chunk_t, host_pointer) == 40U);
static_assert(offsetof(sllm_kv_paged_image_chunk_t, host_capacity) == 48U);
static_assert(offsetof(sllm_kv_paged_image_chunk_t, reserved) == 56U);

int main() {
  sllm_kv_state_paged_create_info_t create{};
  create.struct_size = sizeof(create);
  create.abi_version = SLLM_HIP_ABI_VERSION;
  create.create_info_version = SLLM_HIP_KV_PAGED_CREATE_INFO_VERSION;
  create.memory_kind = SLLM_HIP_KV_MEMORY_KIND_PAGED;
  create.token_block_size = SLLM_HIP_KV_PAGED_TOKEN_BLOCK_SIZE;
  create.physical_layout_version = SLLM_HIP_KV_PAGED_LAYOUT_VERSION;
  create.quantization_block_size = 32U;
  create.logical_table_capacity = 1024U;
  create.max_physical_blocks = 2048U;
  create.sliding_window_tokens = 1024U;
  create.static_key_scale_bits = 0x3f800000U;
  create.static_value_scale_bits = create.static_key_scale_bits;

  sllm_kv_paged_view_info_t view{};
  view.struct_size = sizeof(view);
  view.abi_version = SLLM_HIP_ABI_VERSION;
  view.info_version = SLLM_HIP_KV_PAGED_VIEW_INFO_VERSION;
  view.memory_kind = SLLM_HIP_KV_MEMORY_KIND_PAGED;
  view.token_block_size = SLLM_HIP_KV_PAGED_TOKEN_BLOCK_SIZE;
  view.physical_layout_version = SLLM_HIP_KV_PAGED_LAYOUT_VERSION;
  view.logical_table_capacity = create.logical_table_capacity;
  view.max_physical_blocks = create.max_physical_blocks;
  uint64_t committed_total = 0U;
  for (std::size_t plane = 0U; plane != 6U; ++plane) {
    view.committed_bytes_per_plane[plane] =
        static_cast<uint64_t>(plane + 1U) * 8U * 1024U;
    committed_total += view.committed_bytes_per_plane[plane];
  }
  view.committed_bytes_total = committed_total;

  sllm_kv_paged_state_fork_info_t fork{};
  fork.struct_size = sizeof(fork);
  fork.abi_version = SLLM_HIP_ABI_VERSION;
  fork.info_version = SLLM_HIP_KV_PAGED_STATE_FORK_INFO_VERSION;
  fork.token_block_size = SLLM_HIP_KV_PAGED_TOKEN_BLOCK_SIZE;
  fork.physical_layout_version = SLLM_HIP_KV_PAGED_LAYOUT_VERSION;
  fork.committed_bytes_total = view.committed_bytes_total;

  sllm_kv_paged_image_info_t image{};
  image.struct_size = sizeof(image);
  image.abi_version = SLLM_HIP_ABI_VERSION;
  image.image_version = SLLM_HIP_KV_PAGED_IMAGE_VERSION;
  image.flags = SLLM_HIP_KV_PAGED_IMAGE_FLAG_SLIDING |
                SLLM_HIP_KV_PAGED_IMAGE_FLAG_STATIC_SCALES;
  image.token_block_size = SLLM_HIP_KV_PAGED_TOKEN_BLOCK_SIZE;
  image.physical_layout_version = SLLM_HIP_KV_PAGED_LAYOUT_VERSION;
  image.logical_table_capacity = 1024U;
  image.physical_block_count = 9U;
  image.plane_count = 6U;
  image.ring_slot_count = 9U;
  image.table_entry_width = SLLM_HIP_KV_PAGED_IMAGE_TABLE_ENTRY_U32;
  image.retained_start = 1024U;
  image.retained_length = 1024U;
  image.sliding_window_tokens = SLLM_HIP_KV_SLIDING_WINDOW_GEMMA4;
  for (std::size_t plane = 0U; plane != 6U; ++plane) {
    image.plane_block_stride[plane] = static_cast<uint64_t>(plane + 1U) * 128U;
    image.plane_bytes[plane] =
        image.plane_block_stride[plane] * image.physical_block_count;
  }
  sllm_kv_paged_image_chunk_t chunk{};
  chunk.struct_size = sizeof(chunk);
  chunk.abi_version = SLLM_HIP_ABI_VERSION;
  chunk.image_version = SLLM_HIP_KV_PAGED_IMAGE_VERSION;
  chunk.section = SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TAGS;
  chunk.byte_length = 9U * sizeof(uint64_t);
  chunk.host_capacity = chunk.byte_length;

  if (create.struct_size != sizeof(sllm_kv_state_paged_create_info_t) ||
      view.struct_size != sizeof(sllm_kv_paged_view_info_t) ||
      fork.struct_size != sizeof(sllm_kv_paged_state_fork_info_t) ||
      create.memory_kind != SLLM_HIP_KV_MEMORY_KIND_PAGED ||
      view.memory_kind != SLLM_HIP_KV_MEMORY_KIND_PAGED ||
      view.logical_table_capacity != create.logical_table_capacity ||
      view.max_physical_blocks != create.max_physical_blocks ||
      fork.token_block_size != create.token_block_size ||
      fork.committed_bytes_total != view.committed_bytes_total ||
      view.committed_bytes_total != committed_total ||
      image.struct_size != sizeof(image) ||
      image.image_version != SLLM_HIP_KV_PAGED_IMAGE_VERSION ||
      image.table_entry_width != SLLM_HIP_KV_PAGED_IMAGE_TABLE_ENTRY_U32 ||
      image.ring_slot_count != 9U || image.plane_count != 6U ||
      image.plane_bytes[5] != image.plane_block_stride[5] * 9U ||
      chunk.struct_size != sizeof(chunk) ||
      chunk.section != SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TAGS ||
      chunk.byte_length != chunk.host_capacity) {
    return 1;
  }
  std::cout << "phase87_stage10_paged_abi_layout_host_test: PASS\n";
  return 0;
}
