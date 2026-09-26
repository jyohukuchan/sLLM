#ifndef SLLM_PAGED_KV_IMAGE_API_HPP
#define SLLM_PAGED_KV_IMAGE_API_HPP

#include "public_runtime_internal.hpp"

#include <cstdint>

namespace sllm_paged_kv_image {

/* Validate versioned metadata before any image section is read or written. */
sllm_status_t validate_info(const sllm_kv_paged_image_info_t *info,
                            sllm_error_sink_t *sink) noexcept;

/* Validate one section transfer header and its byte interval. */
sllm_status_t validate_chunk(const sllm_kv_paged_image_chunk_t *chunk,
                             sllm_error_sink_t *sink) noexcept;

/* Validate a section transfer against the queried metadata and checked size. */
sllm_status_t validate_chunk_for_info(const sllm_kv_paged_image_info_t *info,
                                      const sllm_kv_paged_image_chunk_t *chunk,
                                      sllm_error_sink_t *sink) noexcept;

/* Return the exact serialized byte size of one logical-table, ring, or plane
 * section.  No allocation or device access is performed. */
sllm_status_t section_size(const sllm_kv_paged_image_info_t *info,
                           uint32_t section, uint32_t plane,
                           uint64_t *size_bytes,
                           sllm_error_sink_t *sink) noexcept;

/* Check that image metadata is compatible with a Paged create recipe. */
sllm_status_t validate_recipe(const sllm_kv_paged_image_info_t *image,
                              const sllm_kv_state_paged_create_info_t *recipe,
                              sllm_error_sink_t *sink) noexcept;

} // namespace sllm_paged_kv_image

#endif // SLLM_PAGED_KV_IMAGE_API_HPP
