#include "../src/paged_kv_image_api.hpp"

#include <cstdint>
#include <iostream>

namespace {

using sllm_paged_kv_image::section_size;
using sllm_paged_kv_image::validate_chunk_for_info;
using sllm_paged_kv_image::validate_info;
using sllm_paged_kv_image::validate_recipe;

sllm_kv_paged_image_info_t base_image() {
  sllm_kv_paged_image_info_t image{};
  image.struct_size = sizeof(image);
  image.abi_version = SLLM_HIP_ABI_VERSION;
  image.image_version = SLLM_HIP_KV_PAGED_IMAGE_VERSION;
  image.byte_order = SLLM_HIP_KV_PAGED_IMAGE_ENDIAN_LITTLE;
  image.byte_order = SLLM_HIP_KV_PAGED_IMAGE_ENDIAN_LITTLE;
  image.session_id = 1U;
  image.layer_id = 3U;
  image.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
  image.encoding = SLLM_HIP_KV_ENCODING_MXFP8_E4_V1;
  image.head_count = 4U;
  image.head_dim = 256U;
  image.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
  image.token_block_size = SLLM_HIP_KV_PAGED_TOKEN_BLOCK_SIZE;
  image.physical_layout_version = SLLM_HIP_KV_PAGED_LAYOUT_VERSION;
  image.capacity_tokens = 129U;
  image.published_length = 129U;
  image.generation = 1U;
  image.retained_length = image.published_length;
  image.logical_table_capacity = 2U;
  image.physical_block_count = 2U;
  image.plane_count = 4U;
  image.table_entry_width = SLLM_HIP_KV_PAGED_IMAGE_TABLE_ENTRY_U32;
  for (uint32_t plane = 0U; plane != image.plane_count; ++plane) {
    image.plane_block_stride[plane] = 128U * (plane + 1U);
    image.plane_bytes[plane] =
        image.plane_block_stride[plane] * image.physical_block_count;
  }
  return image;
}

sllm_kv_state_paged_create_info_t base_recipe() {
  sllm_kv_state_paged_create_info_t recipe{};
  recipe.struct_size = sizeof(recipe);
  recipe.abi_version = SLLM_HIP_ABI_VERSION;
  recipe.create_info_version = SLLM_HIP_KV_PAGED_CREATE_INFO_VERSION;
  recipe.session_id = 1U;
  recipe.layer_id = 3U;
  recipe.capacity_tokens = 129U;
  recipe.head_count = 4U;
  recipe.head_dim = 256U;
  recipe.memory_kind = SLLM_HIP_KV_MEMORY_KIND_PAGED;
  recipe.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
  recipe.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
  recipe.encoding = SLLM_HIP_KV_ENCODING_MXFP8_E4_V1;
  recipe.scale_dtype = SLLM_TENSOR_DTYPE_U8;
  recipe.quantization_block_size = 32U;
  recipe.token_block_size = SLLM_HIP_KV_PAGED_TOKEN_BLOCK_SIZE;
  recipe.physical_layout_version = SLLM_HIP_KV_PAGED_LAYOUT_VERSION;
  recipe.logical_table_capacity = 2U;
  recipe.max_physical_blocks = 4U;
  return recipe;
}

bool expect(const bool condition, const char *const message) {
  if (!condition)
    std::cerr << message << '\n';
  return condition;
}

bool checks() {
  bool ok = true;
  auto image = base_image();
  ok &= expect(validate_info(&image, nullptr) == SLLM_STATUS_OK,
               "base image rejected");

  uint64_t bytes = 0U;
  ok &=
      expect(section_size(&image, SLLM_HIP_KV_PAGED_IMAGE_SECTION_LOGICAL_TABLE,
                          0U, &bytes, nullptr) == SLLM_STATUS_OK &&
                 bytes == 8U,
             "logical table section size mismatch");
  ok &= expect(section_size(&image, SLLM_HIP_KV_PAGED_IMAGE_SECTION_PLANE, 1U,
                            &bytes, nullptr) == SLLM_STATUS_OK &&
                   bytes == 256U,
               "plane section size mismatch");

  uint32_t table[2] = {0U, 1U};
  sllm_kv_paged_image_chunk_t chunk{};
  chunk.struct_size = sizeof(chunk);
  chunk.abi_version = SLLM_HIP_ABI_VERSION;
  chunk.image_version = SLLM_HIP_KV_PAGED_IMAGE_VERSION;
  chunk.section = SLLM_HIP_KV_PAGED_IMAGE_SECTION_LOGICAL_TABLE;
  chunk.byte_length = sizeof(table);
  chunk.host_pointer = table;
  chunk.host_capacity = sizeof(table);
  ok &=
      expect(validate_chunk_for_info(&image, &chunk, nullptr) == SLLM_STATUS_OK,
             "valid table chunk rejected");
  chunk.byte_offset = 2U;
  ok &=
      expect(validate_chunk_for_info(&image, &chunk, nullptr) != SLLM_STATUS_OK,
             "misaligned table chunk accepted");
  chunk.byte_offset = 0U;
  chunk.byte_length = 12U;
  chunk.host_capacity = 12U;
  ok &=
      expect(validate_chunk_for_info(&image, &chunk, nullptr) != SLLM_STATUS_OK,
             "oversized table chunk accepted");

  auto sliding = base_image();
  sliding.flags = SLLM_HIP_KV_PAGED_IMAGE_FLAG_SLIDING |
                  SLLM_HIP_KV_PAGED_IMAGE_FLAG_STATIC_SCALES;
  sliding.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
  sliding.encoding = SLLM_HIP_KV_ENCODING_FP8_STATIC_V1;
  sliding.capacity_tokens = 1025U;
  sliding.published_length = 1025U;
  sliding.retained_start = 1U;
  sliding.retained_length = 1024U;
  sliding.sliding_window_tokens = SLLM_HIP_KV_SLIDING_WINDOW_GEMMA4;
  sliding.logical_table_capacity = 9U;
  sliding.physical_block_count = 9U;
  sliding.plane_count = 2U;
  sliding.ring_slot_count = 9U;
  sliding.static_key_scale_bits = UINT32_C(0x3f800000);
  sliding.static_value_scale_bits = UINT32_C(0x3f800000);
  for (uint32_t plane = 0U; plane != sliding.plane_count; ++plane) {
    sliding.plane_block_stride[plane] =
        static_cast<uint64_t>(plane + 1U) * 128U;
    sliding.plane_bytes[plane] =
        sliding.plane_block_stride[plane] * sliding.physical_block_count;
  }
  for (uint32_t plane = 2U; plane != 6U; ++plane) {
    sliding.plane_block_stride[plane] = 0U;
    sliding.plane_bytes[plane] = 0U;
  }
  ok &= expect(validate_info(&sliding, nullptr) == SLLM_STATUS_OK,
               "valid sliding image rejected");
  ok &= expect(section_size(&sliding, SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TAGS,
                            0U, &bytes, nullptr) == SLLM_STATUS_OK &&
                   bytes == 72U,
               "ring tag section size mismatch");
  ok &=
      expect(section_size(&sliding, SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TABLE,
                          0U, &bytes, nullptr) == SLLM_STATUS_OK &&
                 bytes == 36U,
             "ring table section size mismatch");

  auto bad = image;
  bad.byte_order = 2U;
  ok &= expect(validate_info(&bad, nullptr) != SLLM_STATUS_OK,
               "unknown byte order accepted");
  bad = image;
  bad.plane_count = 6U;
  ok &= expect(validate_info(&bad, nullptr) != SLLM_STATUS_OK,
               "incompatible plane count accepted");
  bad = sliding;
  bad.retained_start = 0U;
  ok &= expect(validate_info(&bad, nullptr) != SLLM_STATUS_OK,
               "invalid retained start accepted");

  const auto recipe = base_recipe();
  ok &= expect(validate_recipe(&image, &recipe, nullptr) == SLLM_STATUS_OK,
               "compatible recipe rejected");
  auto mismatched_recipe = recipe;
  mismatched_recipe.encoding = SLLM_HIP_KV_ENCODING_NVFP4_V1;
  mismatched_recipe.dtype = SLLM_TENSOR_DTYPE_U8;
  mismatched_recipe.scale_dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
  mismatched_recipe.quantization_block_size = 16U;
  ok &= expect(validate_recipe(&image, &mismatched_recipe, nullptr) !=
                   SLLM_STATUS_OK,
               "incompatible recipe accepted");
  return ok;
}

} // namespace

int main() {
  if (!checks()) {
    std::cerr << "phase87_stage10_paged_image_api_host_test: FAIL\n";
    return 1;
  }
  std::cout << "phase87_stage10_paged_image_api_host_test: PASS\n";
  return 0;
}
