#include "paged_kv_image_api.hpp"

#include "kv_state_api.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace sllm_paged_kv_image {
namespace {

bool all_zero(const uint32_t *const values, const std::size_t count) noexcept {
  for (std::size_t index = 0U; index != count; ++index) {
    if (values[index] != 0U)
      return false;
  }
  return true;
}

bool multiply_overflows(const uint64_t left, const uint64_t right,
                        uint64_t *const result) noexcept {
  if (left != 0U && right > std::numeric_limits<uint64_t>::max() / left)
    return true;
  *result = left * right;
  return false;
}

bool add_overflows(const uint64_t left, const uint64_t right,
                   uint64_t *const result) noexcept {
  if (right > std::numeric_limits<uint64_t>::max() - left)
    return true;
  *result = left + right;
  return false;
}

uint32_t expected_plane_count(const uint32_t encoding) noexcept {
  switch (encoding) {
  case SLLM_HIP_KV_ENCODING_FP16_V1:
  case SLLM_HIP_KV_ENCODING_FP8_STATIC_V1:
    return 2U;
  case SLLM_HIP_KV_ENCODING_FP8_V1:
  case SLLM_HIP_KV_ENCODING_MXFP8_E4_V1:
  case SLLM_HIP_KV_ENCODING_MXFP8_E5_V1:
    return 4U;
  case SLLM_HIP_KV_ENCODING_NVFP4_V1:
    return 6U;
  default:
    return 0U;
  }
}

bool static_scale_bits_valid(const uint32_t key_bits,
                             const uint32_t value_bits) noexcept {
  float key = 0.0F;
  float value = 0.0F;
  std::memcpy(&key, &key_bits, sizeof(key));
  std::memcpy(&value, &value_bits, sizeof(value));
  return std::isfinite(key) && key > 0.0F && std::isfinite(value) &&
         value > 0.0F;
}

sllm_status_t invalid(sllm_error_sink_t *const sink,
                      const char *const message) noexcept {
  return sllm_public_runtime::write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                                          message);
}

sllm_status_t overflow(sllm_error_sink_t *const sink,
                       const char *const message) noexcept {
  return sllm_public_runtime::write_error(sink, SLLM_STATUS_METADATA_OVERFLOW,
                                          message);
}

} // namespace

sllm_status_t validate_info(const sllm_kv_paged_image_info_t *const info,
                            sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr)
    return invalid(sink, "paged KV image metadata is null");
  if (info->struct_size != sizeof(*info))
    return invalid(sink, "paged KV image metadata has an unsupported size");
  if (info->abi_version != SLLM_HIP_ABI_VERSION ||
      info->image_version != SLLM_HIP_KV_PAGED_IMAGE_VERSION ||
      info->byte_order != SLLM_HIP_KV_PAGED_IMAGE_ENDIAN_LITTLE) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "paged KV image ABI, version, or byte order is unsupported");
  }
  constexpr uint32_t kKnownFlags = SLLM_HIP_KV_PAGED_IMAGE_FLAG_SLIDING |
                                   SLLM_HIP_KV_PAGED_IMAGE_FLAG_STATIC_SCALES;
  if ((info->flags & ~kKnownFlags) != 0U || info->reserved0 != 0U ||
      !all_zero(info->reserved, 8U)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "paged KV image flags or reserved fields are nonzero");
  }
  if (info->session_id == 0U || info->capacity_tokens == 0U ||
      info->capacity_tokens > SLLM_HIP_KV_MAX_CAPACITY ||
      info->published_length > info->capacity_tokens ||
      info->token_block_size != SLLM_HIP_KV_PAGED_TOKEN_BLOCK_SIZE ||
      info->physical_layout_version != SLLM_HIP_KV_PAGED_LAYOUT_VERSION ||
      info->layout != SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR ||
      info->head_count == 0U || info->head_count > 8U || info->head_dim == 0U ||
      info->head_dim > SLLM_HIP_KV_MAX_HEAD_DIM) {
    return invalid(sink, "paged KV image recipe or capacity is invalid");
  }
  const uint64_t required_blocks =
      info->capacity_tokens / info->token_block_size +
      (info->capacity_tokens % info->token_block_size != 0U ? 1U : 0U);
  if (info->logical_table_capacity < required_blocks ||
      info->physical_block_count > info->logical_table_capacity ||
      info->table_entry_width != SLLM_HIP_KV_PAGED_IMAGE_TABLE_ENTRY_U32) {
    return invalid(sink, "paged KV image table metadata is invalid");
  }
  const bool sliding =
      (info->flags & SLLM_HIP_KV_PAGED_IMAGE_FLAG_SLIDING) != 0U;
  const uint64_t expected_retained_start =
      sliding ? info->published_length - std::min(info->published_length,
                                                  info->sliding_window_tokens)
              : 0U;
  const uint64_t expected_retained_length =
      info->published_length - expected_retained_start;
  if (!sliding) {
    if (info->sliding_window_tokens != 0U || info->retained_start != 0U ||
        info->retained_length != info->published_length ||
        info->ring_slot_count != 0U)
      return invalid(sink, "non-sliding Paged image has ring metadata");
  } else if (info->sliding_window_tokens != SLLM_HIP_KV_SLIDING_WINDOW_GEMMA4 ||
             info->capacity_tokens < info->sliding_window_tokens ||
             info->capacity_tokens > SLLM_HIP_KV_SLIDING_MAX_CAPACITY ||
             info->ring_slot_count != 9U ||
             info->retained_start != expected_retained_start ||
             info->retained_length != expected_retained_length) {
    return invalid(sink, "sliding Paged image ring metadata is invalid");
  }
  const uint32_t expected_planes = expected_plane_count(info->encoding);
  if (expected_planes == 0U || info->plane_count != expected_planes)
    return invalid(sink, "paged KV image plane count is incompatible");
  const bool static_scales =
      (info->flags & SLLM_HIP_KV_PAGED_IMAGE_FLAG_STATIC_SCALES) != 0U;
  if (static_scales != (info->encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1))
    return invalid(sink, "paged KV image static-scale flag is incompatible");
  if (static_scales && !static_scale_bits_valid(info->static_key_scale_bits,
                                                info->static_value_scale_bits))
    return invalid(sink, "paged KV image static scales are invalid");
  if (!static_scales && (info->static_key_scale_bits != 0U ||
                         info->static_value_scale_bits != 0U))
    return invalid(sink, "paged KV image has unexpected static scales");
  for (uint32_t plane = 0U; plane != 6U; ++plane) {
    const bool present = plane < info->plane_count;
    if (!present) {
      if (info->plane_block_stride[plane] != 0U ||
          info->plane_bytes[plane] != 0U)
        return invalid(sink, "paged KV image has bytes for an absent plane");
      continue;
    }
    uint64_t expected_bytes = 0U;
    if (info->physical_block_count != 0U &&
        (info->plane_block_stride[plane] == 0U ||
         multiply_overflows(info->physical_block_count,
                            info->plane_block_stride[plane], &expected_bytes)))
      return overflow(sink, "paged KV image plane size overflowed u64");
    if (info->plane_bytes[plane] != expected_bytes)
      return invalid(sink, "paged KV image plane size is inconsistent");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t validate_chunk(const sllm_kv_paged_image_chunk_t *const chunk,
                             sllm_error_sink_t *const sink) noexcept {
  if (chunk == nullptr)
    return invalid(sink, "paged KV image chunk is null");
  if (chunk->struct_size != sizeof(*chunk))
    return invalid(sink, "paged KV image chunk has an unsupported size");
  if (chunk->abi_version != SLLM_HIP_ABI_VERSION ||
      chunk->image_version != SLLM_HIP_KV_PAGED_IMAGE_VERSION)
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "paged KV image chunk ABI or version is unsupported");
  if (chunk->reserved0 != 0U || !all_zero(chunk->reserved, 4U))
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "paged KV image chunk reserved fields are nonzero");
  if (chunk->host_pointer == nullptr || chunk->byte_length == 0U ||
      chunk->byte_length > chunk->host_capacity)
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_BUFFER_TOO_SMALL,
        "paged KV image chunk host range is invalid");
  uint64_t end = 0U;
  if (add_overflows(chunk->byte_offset, chunk->byte_length, &end))
    return overflow(sink, "paged KV image chunk range overflowed u64");
  (void)end;
  if (chunk->section < SLLM_HIP_KV_PAGED_IMAGE_SECTION_LOGICAL_TABLE ||
      chunk->section > SLLM_HIP_KV_PAGED_IMAGE_SECTION_PLANE)
    return invalid(sink, "paged KV image chunk section is invalid");
  return SLLM_STATUS_OK;
}

sllm_status_t section_size(const sllm_kv_paged_image_info_t *const info,
                           const uint32_t section, const uint32_t plane,
                           uint64_t *const size_bytes,
                           sllm_error_sink_t *const sink) noexcept {
  if (size_bytes == nullptr)
    return invalid(sink, "paged KV image section size output is null");
  *size_bytes = 0U;
  const sllm_status_t status = validate_info(info, sink);
  if (status != SLLM_STATUS_OK)
    return status;
  uint64_t elements = 0U;
  uint64_t width = 0U;
  switch (section) {
  case SLLM_HIP_KV_PAGED_IMAGE_SECTION_LOGICAL_TABLE:
    elements = info->logical_table_capacity;
    width = sizeof(uint32_t);
    break;
  case SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TAGS:
    if (info->ring_slot_count == 0U)
      return invalid(sink, "non-sliding Paged image has no ring tags");
    elements = info->ring_slot_count;
    width = sizeof(uint64_t);
    break;
  case SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TABLE:
    if (info->ring_slot_count == 0U)
      return invalid(sink, "non-sliding Paged image has no ring table");
    elements = info->ring_slot_count;
    width = sizeof(uint32_t);
    break;
  case SLLM_HIP_KV_PAGED_IMAGE_SECTION_PLANE:
    if (plane == 0U || plane > info->plane_count)
      return invalid(sink, "paged KV image plane section is invalid");
    *size_bytes = info->plane_bytes[plane - 1U];
    return SLLM_STATUS_OK;
  default:
    return invalid(sink, "paged KV image section is invalid");
  }
  if (multiply_overflows(elements, width, size_bytes))
    return overflow(sink, "paged KV image section size overflowed u64");
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_chunk_for_info(const sllm_kv_paged_image_info_t *const info,
                        const sllm_kv_paged_image_chunk_t *const chunk,
                        sllm_error_sink_t *const sink) noexcept {
  const sllm_status_t info_status = validate_info(info, sink);
  if (info_status != SLLM_STATUS_OK)
    return info_status;
  const sllm_status_t chunk_status = validate_chunk(chunk, sink);
  if (chunk_status != SLLM_STATUS_OK)
    return chunk_status;
  uint64_t section_bytes = 0U;
  const sllm_status_t size_status =
      section_size(info, chunk->section, chunk->plane, &section_bytes, sink);
  if (size_status != SLLM_STATUS_OK)
    return size_status;
  uint64_t end = 0U;
  if (add_overflows(chunk->byte_offset, chunk->byte_length, &end) ||
      end > section_bytes)
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
        "paged KV image chunk exceeds its section");
  const uint64_t element_width =
      chunk->section == SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TAGS
          ? sizeof(uint64_t)
      : chunk->section == SLLM_HIP_KV_PAGED_IMAGE_SECTION_LOGICAL_TABLE ||
              chunk->section == SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TABLE
          ? sizeof(uint32_t)
          : 1U;
  if (chunk->byte_offset % element_width != 0U ||
      chunk->byte_length % element_width != 0U)
    return invalid(sink, "paged KV image chunk is element-misaligned");
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_recipe(const sllm_kv_paged_image_info_t *const image,
                const sllm_kv_state_paged_create_info_t *const recipe,
                sllm_error_sink_t *const sink) noexcept {
  const sllm_status_t image_status = validate_info(image, sink);
  if (image_status != SLLM_STATUS_OK)
    return image_status;
  const sllm_status_t recipe_status =
      sllm_kv_state::validate_state_create_info_paged(recipe, sink);
  if (recipe_status != SLLM_STATUS_OK)
    return recipe_status;
  const uint32_t recipe_heads =
      recipe->head_count == 0U ? SLLM_HIP_KV_HEAD_COUNT : recipe->head_count;
  const uint32_t recipe_dim =
      recipe->head_dim == 0U ? SLLM_HIP_KV_HEAD_DIM : recipe->head_dim;
  if (image->session_id != recipe->session_id ||
      image->layer_id != recipe->layer_id || image->dtype != recipe->dtype ||
      image->encoding != recipe->encoding ||
      image->head_count != recipe_heads || image->head_dim != recipe_dim ||
      image->layout != recipe->layout ||
      image->capacity_tokens != recipe->capacity_tokens ||
      image->token_block_size != recipe->token_block_size ||
      image->physical_layout_version != recipe->physical_layout_version ||
      image->logical_table_capacity != recipe->logical_table_capacity ||
      image->sliding_window_tokens != recipe->sliding_window_tokens ||
      image->static_key_scale_bits != recipe->static_key_scale_bits ||
      image->static_value_scale_bits != recipe->static_value_scale_bits) {
    return invalid(sink, "paged KV image recipe differs from destination");
  }
  if (image->physical_block_count > recipe->max_physical_blocks)
    return invalid(sink, "paged KV image uses too many physical blocks");
  return SLLM_STATUS_OK;
}

} // namespace sllm_paged_kv_image
