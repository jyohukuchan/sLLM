// Phase 87 Stage 10: public Paged KV image roundtrip.
//
// The image is exported after a 129-token append, imported into a fresh state,
// and then both states append one more token and run one causal attention row.
// The destination must match the source bitwise.  The test also checks the
// independent FP8 E4M3/E8M0 attention oracle for one output element and that a
// missing plane is rejected without publishing the destination.
#include "sllm/hip.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#error SLLM_TEST_EXPECTED_TARGET must name one exact GPU target
#endif

namespace {

constexpr uint32_t kKvHeads = 2U;
constexpr uint32_t kQueryHeads = 8U;
constexpr uint32_t kHeadDim = 128U;
constexpr uint64_t kPrefix = 129U;
constexpr uint64_t kCapacity = 256U;
constexpr uint64_t kLogicalBlocks = 2U;
constexpr uint64_t kMaxPhysicalBlocks = 4U;
constexpr uint32_t kTokenBlock = 128U;
constexpr uint32_t kTimeoutMs = 30'000U;

struct Error final {
  char message[512]{};
  sllm_error_sink_t sink{sizeof(sllm_error_sink_t),
                         SLLM_HIP_ABI_VERSION,
                         message,
                         sizeof(message),
                         0U,
                         {0U, 0U}};
};

bool expect(const sllm_status_t actual, const sllm_status_t wanted,
            const char *const operation, const Error &error) {
  if (actual == wanted)
    return true;
  std::cerr << operation << " returned " << actual << ", expected " << wanted
            << ": " << error.message << '\n';
  return false;
}

bool release_buffer(sllm_buffer_t **const buffer) {
  if (buffer == nullptr || *buffer == nullptr)
    return true;
  Error error;
  return expect(sllm_buffer_release(buffer, &error.sink), SLLM_STATUS_OK,
                "buffer release", error) &&
         *buffer == nullptr;
}

bool release_state(sllm_kv_state_t **const state) {
  if (state == nullptr || *state == nullptr)
    return true;
  Error error;
  return expect(sllm_kv_state_release(state, &error.sink), SLLM_STATUS_OK,
                "state release", error) &&
         *state == nullptr;
}

bool release_queue(sllm_queue_t **const queue) {
  if (queue == nullptr || *queue == nullptr)
    return true;
  Error error;
  return expect(sllm_queue_release(queue, &error.sink), SLLM_STATUS_OK,
                "queue release", error) &&
         *queue == nullptr;
}

bool release_context(sllm_context_t **const context) {
  if (context == nullptr || *context == nullptr)
    return true;
  Error error;
  return expect(sllm_context_release(context, &error.sink), SLLM_STATUS_OK,
                "context release", error) &&
         *context == nullptr;
}

bool wait_release(sllm_completion_t **const completion,
                  const char *const operation) {
  if (completion == nullptr || *completion == nullptr) {
    std::cerr << operation << " returned no completion\n";
    return false;
  }
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  const bool waited = expect(sllm_completion_wait(*completion, kTimeoutMs,
                                                  &result, &error.sink),
                             SLLM_STATUS_OK, operation, error) &&
                      result.state == SLLM_COMPLETION_STATE_SUCCESS;
  const bool released = expect(sllm_completion_release(completion, &error.sink),
                               SLLM_STATUS_OK, "completion release", error) &&
                        *completion == nullptr;
  return waited && released;
}

bool create_buffer(const sllm_context_t *const context, const uint64_t bytes,
                   sllm_buffer_t **const buffer) {
  sllm_buffer_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.size_bytes = bytes;
  Error error;
  return expect(sllm_buffer_create(context, &info, buffer, &error.sink),
                SLLM_STATUS_OK, "buffer create", error) &&
         *buffer != nullptr;
}

bool upload(const sllm_queue_t *const queue, const sllm_buffer_t *const buffer,
            const void *const source, const uint64_t bytes) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = const_cast<void *>(source);
  transfer.size_bytes = bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect(sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                   &error.sink),
              SLLM_STATUS_OK, "buffer upload", error))
    return false;
  return wait_release(&completion, "buffer upload wait");
}

bool download(const sllm_queue_t *const queue,
              const sllm_buffer_t *const buffer, void *const destination,
              const uint64_t bytes) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.size_bytes = bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect(sllm_buffer_copy_d2h(queue, buffer, &transfer, &completion,
                                   &error.sink),
              SLLM_STATUS_OK, "buffer download", error) ||
      completion == nullptr)
    return false;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(
          sllm_completion_wait(completion, kTimeoutMs, &result, &error.sink),
          SLLM_STATUS_OK, "buffer download wait", error) ||
      result.state != SLLM_COMPLETION_STATE_SUCCESS) {
    (void)sllm_completion_release(&completion, &error.sink);
    return false;
  }
  uint64_t written = 0U;
  const bool read = expect(sllm_completion_read(completion, destination, bytes,
                                                &written, &error.sink),
                           SLLM_STATUS_OK, "buffer download read", error) &&
                    written == bytes;
  const bool released =
      expect(sllm_completion_release(&completion, &error.sink), SLLM_STATUS_OK,
             "download completion release", error) &&
      completion == nullptr;
  return read && released;
}

sllm_tensor_binding_t binding(const sllm_buffer_t *const buffer,
                              const uint32_t dtype, const uint32_t rank,
                              const uint64_t *const shape) {
  sllm_tensor_binding_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.buffer = buffer;
  result.dtype = dtype;
  result.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  result.rank = rank;
  uint64_t stride = 1U;
  for (uint32_t reverse = 0U; reverse != rank; ++reverse) {
    const uint32_t index = rank - reverse - 1U;
    result.shape[index] = shape[index];
    result.stride_elements[index] = stride;
    stride *= shape[index];
  }
  return result;
}

uint16_t bf16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  return static_cast<uint16_t>(
      upper + ((lower > UINT32_C(0x8000) ||
                (lower == UINT32_C(0x8000) && (upper & 1U) != 0U))
                   ? 1U
                   : 0U));
}

float bf16_float(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint32_t float_bits(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float e4m3(const uint8_t bits) {
  const float sign = (bits & 0x80U) != 0U ? -1.0F : 1.0F;
  const uint32_t magnitude = bits & 0x7fU;
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & 7U;
  if (exponent == 0U)
    return sign * static_cast<float>(mantissa) * std::ldexp(1.0F, -9);
  if (magnitude == 0x7fU)
    return NAN;
  return sign * std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                           static_cast<int>(exponent) - 7);
}

uint8_t nearest_e4m3(const float value) {
  const uint8_t sign = std::signbit(value) ? UINT8_C(0x80) : 0U;
  const float magnitude = std::fabs(value);
  uint8_t selected = 0U;
  float error = magnitude;
  for (uint32_t code = 1U; code <= 0x7eU; ++code) {
    const float candidate = e4m3(static_cast<uint8_t>(code));
    const float candidate_error = std::fabs(magnitude - candidate);
    if (candidate_error < error ||
        (candidate_error == error && (code & 1U) == 0U &&
         (selected & 1U) != 0U)) {
      selected = static_cast<uint8_t>(code);
      error = candidate_error;
    }
  }
  return static_cast<uint8_t>(sign | selected);
}

uint8_t mxfp8_scale(const float maximum) {
  if (!(maximum > 0.0F) || !std::isfinite(maximum))
    return UINT8_C(127);
  int exponent = std::ilogb(maximum) - 8 + 127;
  exponent = std::clamp(exponent, 0, 254);
  return static_cast<uint8_t>(exponent);
}

float quantized(const std::vector<uint16_t> &values, const uint64_t token,
                const uint32_t head, const uint32_t dimension) {
  const size_t row = (static_cast<size_t>(token) * kKvHeads + head) * kHeadDim;
  const uint32_t begin = (dimension / 32U) * 32U;
  const uint32_t end = std::min(begin + 32U, kHeadDim);
  float maximum = 0.0F;
  for (uint32_t lane = begin; lane != end; ++lane)
    maximum = std::max(maximum, std::fabs(bf16_float(values[row + lane])));
  const uint8_t scale_code = mxfp8_scale(maximum);
  const float scale = std::ldexp(1.0F, static_cast<int>(scale_code) - 127);
  return e4m3(nearest_e4m3(bf16_float(values[row + dimension]) / scale)) *
         scale;
}

std::vector<uint16_t> make_kv(const uint64_t count, const bool key) {
  std::vector<uint16_t> result(static_cast<size_t>(count) * kKvHeads *
                               kHeadDim);
  for (uint64_t token = 0U; token != count; ++token) {
    for (uint32_t head = 0U; head != kKvHeads; ++head) {
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        const float base =
            key ? 0.11F + 0.017F * static_cast<float>(head) +
                      0.0019F * static_cast<float>(token % 23U) +
                      0.0021F * static_cast<float>(dimension % 29U)
                : 0.23F + 0.013F * static_cast<float>(head) +
                      0.0023F * static_cast<float>(token % 19U) +
                      0.0017F * static_cast<float>(dimension % 31U);
        const float signed_value =
            ((token + head + dimension) % (key ? 17U : 19U)) == 0U ? -base
                                                                   : base;
        result[(static_cast<size_t>(token) * kKvHeads + head) * kHeadDim +
               dimension] = bf16(signed_value);
      }
    }
  }
  return result;
}

std::vector<uint16_t> make_query() {
  std::vector<uint16_t> result(kQueryHeads * kHeadDim);
  for (uint32_t head = 0U; head != kQueryHeads; ++head) {
    for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
      float value = 0.019F + 0.0029F * static_cast<float>(head % 5U) +
                    0.00041F * static_cast<float>(dimension % 27U);
      if ((head * 5U + dimension) % 13U == 0U)
        value = -value;
      result[static_cast<size_t>(head) * kHeadDim + dimension] = bf16(value);
    }
  }
  return result;
}

float oracle_head0(const std::vector<uint16_t> &keys,
                   const std::vector<uint16_t> &values,
                   const std::vector<uint16_t> &query) {
  constexpr uint64_t count = kPrefix + 1U;
  std::array<double, count> scores{};
  double maximum = -std::numeric_limits<double>::infinity();
  for (uint64_t token = 0U; token != count; ++token) {
    double dot = 0.0;
    for (uint32_t lane = 0U; lane != kHeadDim; ++lane)
      dot += static_cast<double>(bf16_float(query[lane])) *
             quantized(keys, token, 0U, lane);
    scores[token] = dot / std::sqrt(static_cast<double>(kHeadDim));
    maximum = std::max(maximum, scores[token]);
  }
  double denominator = 0.0;
  double numerator = 0.0;
  for (uint64_t token = 0U; token != count; ++token) {
    const double weight = std::exp(scores[token] - maximum);
    denominator += weight;
    numerator += weight * quantized(values, token, 0U, 0U);
  }
  return static_cast<float>(numerator / denominator);
}

sllm_kv_state_paged_create_info_t create_info(const uint64_t session) {
  sllm_kv_state_paged_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.create_info_version = SLLM_HIP_KV_PAGED_CREATE_INFO_VERSION;
  info.session_id = session;
  info.capacity_tokens = kCapacity;
  info.head_count = kKvHeads;
  info.head_dim = kHeadDim;
  info.memory_kind = SLLM_HIP_KV_MEMORY_KIND_PAGED;
  info.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
  info.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
  info.encoding = SLLM_HIP_KV_ENCODING_MXFP8_E4_V1;
  info.scale_dtype = SLLM_TENSOR_DTYPE_U8;
  info.quantization_block_size = 32U;
  info.token_block_size = kTokenBlock;
  info.physical_layout_version = SLLM_HIP_KV_PAGED_LAYOUT_VERSION;
  info.logical_table_capacity = kLogicalBlocks;
  info.max_physical_blocks = kMaxPhysicalBlocks;
  return info;
}

sllm_kv_state_paged_create_info_t sliding_info(const uint64_t session) {
  sllm_kv_state_paged_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.create_info_version = SLLM_HIP_KV_PAGED_CREATE_INFO_VERSION;
  info.session_id = session;
  info.capacity_tokens = 2048U;
  info.head_count = kKvHeads;
  info.head_dim = kHeadDim;
  info.memory_kind = SLLM_HIP_KV_MEMORY_KIND_PAGED;
  info.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
  info.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
  info.encoding = SLLM_HIP_KV_ENCODING_FP8_STATIC_V1;
  info.scale_dtype = SLLM_TENSOR_DTYPE_F32;
  info.token_block_size = kTokenBlock;
  info.physical_layout_version = SLLM_HIP_KV_PAGED_LAYOUT_VERSION;
  info.logical_table_capacity = 16U;
  info.max_physical_blocks = 16U;
  info.sliding_window_tokens = SLLM_HIP_KV_SLIDING_WINDOW_GEMMA4;
  info.static_key_scale_bits = float_bits(1.0F);
  info.static_value_scale_bits = float_bits(1.0F);
  return info;
}

bool append(const sllm_kv_state_t *const state, const sllm_queue_t *const queue,
            const sllm_buffer_t *const key, const sllm_buffer_t *const value,
            const uint64_t start, const uint64_t count) {
  const uint64_t shape[] = {count, kKvHeads, kHeadDim};
  sllm_kv_append_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.append_version = SLLM_HIP_KV_STATE_VERSION;
  descriptor.expected_length = start;
  descriptor.start_position = start;
  descriptor.key_input = binding(key, SLLM_TENSOR_DTYPE_BF16, 3U, shape);
  descriptor.value_input = binding(value, SLLM_TENSOR_DTYPE_BF16, 3U, shape);
  sllm_kv_append_info_t append_info{};
  append_info.struct_size = sizeof(append_info);
  append_info.abi_version = SLLM_HIP_ABI_VERSION;
  append_info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect(sllm_kv_state_append(state, queue, &descriptor, &completion,
                                   &append_info, &error.sink),
              SLLM_STATUS_OK, "KV append", error))
    return false;
  return wait_release(&completion, "KV append wait");
}

bool attention(const sllm_kv_state_t *const state,
               const sllm_context_t *const context,
               const sllm_queue_t *const queue, const sllm_buffer_t *const q,
               const sllm_buffer_t *const output, const uint64_t position,
               sllm_completion_t **const completion,
               const uint64_t sliding_window = 0U) {
  const uint64_t shape[] = {1U, kQueryHeads, kHeadDim};
  sllm_causal_attention_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = sliding_window == 0U
                              ? SLLM_HIP_CAUSAL_ATTENTION_VERSION
                              : SLLM_HIP_CAUSAL_ATTENTION_SLIDING_VERSION;
  descriptor.start_position = position;
  descriptor.expected_kv_length = position + 1U;
  descriptor.kv_state = state;
  descriptor.query = binding(q, SLLM_TENSOR_DTYPE_BF16, 3U, shape);
  descriptor.output = binding(output, SLLM_TENSOR_DTYPE_BF16, 3U, shape);
  if (sliding_window != 0U) {
    descriptor.reserved[0] = static_cast<uint32_t>(sliding_window);
    descriptor.reserved[1] = static_cast<uint32_t>(sliding_window >> 32U);
  }
  sllm_causal_attention_dispatch_info_t dispatch{};
  dispatch.struct_size = sizeof(dispatch);
  dispatch.abi_version = SLLM_HIP_ABI_VERSION;
  dispatch.info_version = SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION;
  Error error;
  return expect(sllm_causal_attention_execute(context, queue, &descriptor,
                                              completion, &dispatch,
                                              &error.sink),
                SLLM_STATUS_OK, "causal attention", error) &&
         *completion != nullptr;
}

struct ImageBuffers final {
  std::vector<uint8_t> logical;
  std::vector<uint8_t> ring_tags;
  std::vector<uint8_t> ring_table;
  std::array<std::vector<uint8_t>, 6U> planes;
};

sllm_kv_paged_image_chunk_t chunk_header(const uint32_t section,
                                         const uint32_t plane, void *pointer,
                                         const uint64_t bytes) {
  sllm_kv_paged_image_chunk_t chunk{};
  chunk.struct_size = sizeof(chunk);
  chunk.abi_version = SLLM_HIP_ABI_VERSION;
  chunk.image_version = SLLM_HIP_KV_PAGED_IMAGE_VERSION;
  chunk.section = section;
  chunk.plane = plane;
  chunk.byte_length = bytes;
  chunk.host_pointer = pointer;
  chunk.host_capacity = bytes;
  return chunk;
}

bool export_image(const sllm_kv_state_t *const state,
                  const sllm_kv_paged_image_info_t &info,
                  ImageBuffers *const image) {
  Error error;
  uint64_t logical_bytes = 0U;
  if (!expect(sllm_kv_state_paged_image_section_size(
                  state, SLLM_HIP_KV_PAGED_IMAGE_SECTION_LOGICAL_TABLE, 0U,
                  &logical_bytes, &error.sink),
              SLLM_STATUS_OK, "logical section size", error))
    return false;
  image->logical.resize(static_cast<size_t>(logical_bytes));
  sllm_kv_paged_image_chunk_t logical =
      chunk_header(SLLM_HIP_KV_PAGED_IMAGE_SECTION_LOGICAL_TABLE, 0U,
                   image->logical.data(), logical_bytes);
  if (!expect(sllm_kv_state_paged_image_export(state, &logical, &error.sink),
              SLLM_STATUS_OK, "logical export", error))
    return false;
  if (info.ring_slot_count != 0U) {
    uint64_t tags_bytes = 0U;
    uint64_t table_bytes = 0U;
    if (!expect(sllm_kv_state_paged_image_section_size(
                    state, SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TAGS, 0U,
                    &tags_bytes, &error.sink),
                SLLM_STATUS_OK, "ring tag section size", error) ||
        !expect(sllm_kv_state_paged_image_section_size(
                    state, SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TABLE, 0U,
                    &table_bytes, &error.sink),
                SLLM_STATUS_OK, "ring table section size", error))
      return false;
    image->ring_tags.resize(static_cast<size_t>(tags_bytes));
    image->ring_table.resize(static_cast<size_t>(table_bytes));
    sllm_kv_paged_image_chunk_t tags =
        chunk_header(SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TAGS, 0U,
                     image->ring_tags.data(), tags_bytes);
    sllm_kv_paged_image_chunk_t table =
        chunk_header(SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TABLE, 0U,
                     image->ring_table.data(), table_bytes);
    if (!expect(sllm_kv_state_paged_image_export(state, &tags, &error.sink),
                SLLM_STATUS_OK, "ring tag export", error) ||
        !expect(sllm_kv_state_paged_image_export(state, &table, &error.sink),
                SLLM_STATUS_OK, "ring table export", error))
      return false;
  }
  for (uint32_t plane = 0U; plane != 6U; ++plane) {
    if (plane >= info.plane_count) {
      if (info.plane_bytes[plane] != 0U || info.plane_block_stride[plane] != 0U)
        return false;
      continue;
    }
    image->planes[plane].resize(static_cast<size_t>(info.plane_bytes[plane]));
    sllm_kv_paged_image_chunk_t transfer =
        chunk_header(SLLM_HIP_KV_PAGED_IMAGE_SECTION_PLANE, plane + 1U,
                     image->planes[plane].data(), info.plane_bytes[plane]);
    if (!expect(sllm_kv_state_paged_image_export(state, &transfer, &error.sink),
                SLLM_STATUS_OK, "plane export", error))
      return false;
  }
  return true;
}

bool import_image(const sllm_kv_state_t *const state,
                  const sllm_kv_paged_image_info_t &info,
                  const ImageBuffers &image, const bool omit_first_plane,
                  const bool omit_ring_tags = false) {
  Error error;
  if (!omit_first_plane) {
    sllm_kv_paged_image_chunk_t logical = chunk_header(
        SLLM_HIP_KV_PAGED_IMAGE_SECTION_LOGICAL_TABLE, 0U,
        const_cast<uint8_t *>(image.logical.data()), image.logical.size());
    if (!expect(sllm_kv_state_paged_image_import(state, &logical, &error.sink),
                SLLM_STATUS_OK, "logical import", error))
      return false;
  }
  if (info.ring_slot_count != 0U) {
    if (!omit_ring_tags) {
      sllm_kv_paged_image_chunk_t tags =
          chunk_header(SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TAGS, 0U,
                       const_cast<uint8_t *>(image.ring_tags.data()),
                       image.ring_tags.size());
      if (!expect(sllm_kv_state_paged_image_import(state, &tags, &error.sink),
                  SLLM_STATUS_OK, "ring tag import", error))
        return false;
    }
    sllm_kv_paged_image_chunk_t table =
        chunk_header(SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TABLE, 0U,
                     const_cast<uint8_t *>(image.ring_table.data()),
                     image.ring_table.size());
    if (!expect(sllm_kv_state_paged_image_import(state, &table, &error.sink),
                SLLM_STATUS_OK, "ring table import", error))
      return false;
  }
  for (uint32_t plane = 0U; plane != info.plane_count; ++plane) {
    if (omit_first_plane && plane == 0U)
      continue;
    sllm_kv_paged_image_chunk_t transfer =
        chunk_header(SLLM_HIP_KV_PAGED_IMAGE_SECTION_PLANE, plane + 1U,
                     const_cast<uint8_t *>(image.planes[plane].data()),
                     image.planes[plane].size());
    if (!expect(sllm_kv_state_paged_image_import(state, &transfer, &error.sink),
                SLLM_STATUS_OK, "plane import", error))
      return false;
  }
  return true;
}

bool finalize_image(const sllm_kv_state_t *const state,
                    const sllm_kv_paged_image_info_t &info,
                    const sllm_status_t wanted) {
  Error error;
  return expect(
      sllm_kv_state_paged_image_import_finalize(state, &info, &error.sink),
      wanted, "image finalize", error);
}

bool run_sliding_roundtrip(const sllm_context_t *const context,
                           const sllm_queue_t *const queue) {
  sllm_kv_state_t *source = nullptr;
  sllm_kv_state_t *destination = nullptr;
  sllm_kv_state_t *invalid_destination = nullptr;
  std::array<sllm_buffer_t *, 9U> buffers{};
  bool passed = true;
  const auto cleanup = [&]() {
    bool result = true;
    (void)release_state(&invalid_destination);
    (void)release_state(&destination);
    (void)release_state(&source);
    for (sllm_buffer_t *&buffer : buffers)
      result = release_buffer(&buffer) && result;
    return result;
  };

  const std::vector<uint16_t> prefix_keys = make_kv(1152U, true);
  const std::vector<uint16_t> prefix_values = make_kv(1152U, false);
  const std::vector<uint16_t> block8_keys = make_kv(128U, true);
  const std::vector<uint16_t> block8_values = make_kv(128U, false);
  const std::vector<uint16_t> wrap_keys = make_kv(1U, true);
  const std::vector<uint16_t> wrap_values = make_kv(1U, false);
  const std::vector<uint16_t> query = make_query();
  const uint64_t prefix_bytes =
      static_cast<uint64_t>(prefix_keys.size()) * sizeof(uint16_t);
  const uint64_t token_bytes =
      static_cast<uint64_t>(wrap_keys.size()) * sizeof(uint16_t);
  const uint64_t query_bytes =
      static_cast<uint64_t>(query.size()) * sizeof(uint16_t);
  const uint64_t output_bytes =
      static_cast<uint64_t>(kQueryHeads) * kHeadDim * sizeof(uint16_t);
  passed &= create_buffer(context, prefix_bytes, &buffers[0]);
  passed &= create_buffer(context, prefix_bytes, &buffers[1]);
  passed &= create_buffer(context, prefix_bytes, &buffers[2]);
  passed &= create_buffer(context, prefix_bytes, &buffers[3]);
  passed &= create_buffer(context, token_bytes, &buffers[4]);
  passed &= create_buffer(context, token_bytes, &buffers[5]);
  passed &= create_buffer(context, query_bytes, &buffers[6]);
  passed &= create_buffer(context, output_bytes, &buffers[7]);
  passed &= create_buffer(context, output_bytes, &buffers[8]);
  if (!passed || !upload(queue, buffers[0], prefix_keys.data(), prefix_bytes) ||
      !upload(queue, buffers[1], prefix_values.data(), prefix_bytes) ||
      !upload(queue, buffers[2], block8_keys.data(),
              static_cast<uint64_t>(block8_keys.size()) * sizeof(uint16_t)) ||
      !upload(queue, buffers[3], block8_values.data(),
              static_cast<uint64_t>(block8_values.size()) * sizeof(uint16_t)) ||
      !upload(queue, buffers[4], wrap_keys.data(), token_bytes) ||
      !upload(queue, buffers[5], wrap_values.data(), token_bytes) ||
      !upload(queue, buffers[6], query.data(), query_bytes))
    return cleanup() && false;

  const sllm_kv_state_paged_create_info_t info = sliding_info(0x8711U);
  Error error;
  passed &=
      expect(sllm_kv_state_create_paged(context, &info, &source, &error.sink),
             SLLM_STATUS_OK, "sliding source create", error) &&
      source != nullptr;
  passed &= expect(sllm_kv_state_create_paged(context, &info, &destination,
                                              &error.sink),
                   SLLM_STATUS_OK, "sliding destination create", error) &&
            destination != nullptr;
  passed &=
      expect(sllm_kv_state_create_paged(context, &info, &invalid_destination,
                                        &error.sink),
             SLLM_STATUS_OK, "sliding invalid destination create", error) &&
      invalid_destination != nullptr;
  if (!passed)
    return cleanup() && false;
  passed &= append(source, queue, buffers[0], buffers[1], 0U, 1024U);
  passed &=
      passed && append(source, queue, buffers[2], buffers[3], 1024U, 128U);
  passed &= passed && append(source, queue, buffers[4], buffers[5], 1152U, 1U);
  if (!passed)
    return cleanup() && false;

  sllm_kv_paged_image_info_t image_info{};
  image_info.struct_size = sizeof(image_info);
  image_info.abi_version = SLLM_HIP_ABI_VERSION;
  image_info.image_version = SLLM_HIP_KV_PAGED_IMAGE_VERSION;
  passed &=
      expect(sllm_kv_state_paged_image_query(source, &image_info, &error.sink),
             SLLM_STATUS_OK, "sliding image query", error) &&
      (image_info.flags & SLLM_HIP_KV_PAGED_IMAGE_FLAG_SLIDING) != 0U &&
      image_info.encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1 &&
      image_info.published_length == 1153U &&
      image_info.retained_start == 129U &&
      image_info.retained_length == 1024U && image_info.ring_slot_count == 9U &&
      image_info.physical_block_count == 9U;
  ImageBuffers image;
  passed &= export_image(source, image_info, &image);
  passed &= import_image(invalid_destination, image_info, image, false, true);
  passed &= finalize_image(invalid_destination, image_info,
                           SLLM_STATUS_INVALID_ARGUMENT);
  sllm_kv_paged_view_info_t invalid_view{};
  invalid_view.struct_size = sizeof(invalid_view);
  invalid_view.abi_version = SLLM_HIP_ABI_VERSION;
  invalid_view.info_version = SLLM_HIP_KV_PAGED_VIEW_INFO_VERSION;
  passed &=
      expect(sllm_kv_state_query_paged(invalid_destination, &invalid_view,
                                       &error.sink),
             SLLM_STATUS_OK, "sliding invalid destination query", error) &&
      invalid_view.observed_length == 0U && invalid_view.generation == 0U;

  passed &= import_image(destination, image_info, image, false, false) &&
            finalize_image(destination, image_info, SLLM_STATUS_OK);
  sllm_kv_paged_image_info_t destination_info{};
  destination_info.struct_size = sizeof(destination_info);
  destination_info.abi_version = SLLM_HIP_ABI_VERSION;
  destination_info.image_version = SLLM_HIP_KV_PAGED_IMAGE_VERSION;
  passed &= expect(sllm_kv_state_paged_image_query(
                       destination, &destination_info, &error.sink),
                   SLLM_STATUS_OK, "sliding destination image query", error) &&
            destination_info.retained_start == image_info.retained_start &&
            destination_info.retained_length == image_info.retained_length &&
            destination_info.generation == image_info.generation;
  ImageBuffers destination_image;
  passed &= export_image(destination, destination_info, &destination_image) &&
            destination_image.ring_tags == image.ring_tags &&
            destination_image.ring_table == image.ring_table;

  passed &= append(source, queue, buffers[4], buffers[5], 1153U, 1U) &&
            append(destination, queue, buffers[4], buffers[5], 1153U, 1U);
  sllm_completion_t *source_attention = nullptr;
  sllm_completion_t *destination_attention = nullptr;
  passed &= attention(source, context, queue, buffers[6], buffers[7], 1153U,
                      &source_attention, 1024U) &&
            wait_release(&source_attention, "sliding source attention wait") &&
            attention(destination, context, queue, buffers[6], buffers[8],
                      1153U, &destination_attention, 1024U) &&
            wait_release(&destination_attention,
                         "sliding destination attention wait");
  std::vector<uint16_t> source_output(output_bytes / sizeof(uint16_t));
  std::vector<uint16_t> destination_output(output_bytes / sizeof(uint16_t));
  passed &=
      download(queue, buffers[7], source_output.data(), output_bytes) &&
      download(queue, buffers[8], destination_output.data(), output_bytes) &&
      source_output == destination_output;
  if (source_attention != nullptr)
    passed &=
        wait_release(&source_attention, "sliding source attention cleanup");
  if (destination_attention != nullptr)
    passed &= wait_release(&destination_attention,
                           "sliding destination attention cleanup");
  return cleanup() && passed;
}

} // namespace

int main() {
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_kv_state_t *source = nullptr;
  sllm_kv_state_t *destination = nullptr;
  sllm_kv_state_t *invalid_destination = nullptr;
  std::array<sllm_buffer_t *, 8U> buffers{};
  Error error;
  bool passed = true;

  uint32_t device_count = 0U;
  passed &= expect(sllm_device_count(&device_count, &error.sink),
                   SLLM_STATUS_OK, "device count", error) &&
            device_count == 1U;
  if (device_count != 1U)
    std::cerr << "unexpected device count=" << device_count << '\n';
  sllm_device_info_t device{};
  device.struct_size = sizeof(device);
  device.abi_version = SLLM_HIP_ABI_VERSION;
  passed &= expect(sllm_device_query(0U, &device, &error.sink), SLLM_STATUS_OK,
                   "device query", error) &&
            std::strcmp(device.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0;
  if (std::strcmp(device.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) != 0)
    std::cerr << "device target=" << device.gcn_arch_name
              << " expected=" << SLLM_TEST_EXPECTED_TARGET << '\n';
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  context_info.device_index = 0U;
  std::snprintf(context_info.expected_gcn_arch_name,
                sizeof(context_info.expected_gcn_arch_name), "%s",
                SLLM_TEST_EXPECTED_TARGET);
  passed &= expect(sllm_context_create(&context_info, &context, &error.sink),
                   SLLM_STATUS_OK, "context create", error) &&
            context != nullptr;
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  passed &= context != nullptr &&
            expect(sllm_queue_create(context, &queue_info, &queue, &error.sink),
                   SLLM_STATUS_OK, "queue create", error) &&
            queue != nullptr;
  if (!passed) {
    (void)release_queue(&queue);
    (void)release_context(&context);
    return 1;
  }

  sllm_completion_t *source_attention = nullptr;
  sllm_completion_t *destination_attention = nullptr;
  const bool roundtrip = [&]() -> bool {
    bool result = true;
    const std::vector<uint16_t> keys = make_kv(kPrefix, true);
    const std::vector<uint16_t> values = make_kv(kPrefix, false);
    const std::vector<uint16_t> tail_keys = make_kv(1U, true);
    const std::vector<uint16_t> tail_values = make_kv(1U, false);
    const std::vector<uint16_t> query = make_query();
    const uint64_t kv_bytes =
        static_cast<uint64_t>(keys.size()) * sizeof(uint16_t);
    const uint64_t tail_bytes =
        static_cast<uint64_t>(tail_keys.size()) * sizeof(uint16_t);
    const uint64_t query_bytes =
        static_cast<uint64_t>(query.size()) * sizeof(uint16_t);
    const uint64_t output_bytes =
        static_cast<uint64_t>(kQueryHeads) * kHeadDim * sizeof(uint16_t);
    passed &= create_buffer(context, kv_bytes, &buffers[0]);
    passed &= create_buffer(context, kv_bytes, &buffers[1]);
    passed &= create_buffer(context, tail_bytes, &buffers[2]);
    passed &= create_buffer(context, tail_bytes, &buffers[3]);
    passed &= create_buffer(context, query_bytes, &buffers[4]);
    passed &= create_buffer(context, output_bytes, &buffers[5]);
    passed &= create_buffer(context, output_bytes, &buffers[6]);
    if (!passed || !upload(queue, buffers[0], keys.data(), kv_bytes) ||
        !upload(queue, buffers[1], values.data(), kv_bytes) ||
        !upload(queue, buffers[2], tail_keys.data(), tail_bytes) ||
        !upload(queue, buffers[3], tail_values.data(), tail_bytes) ||
        !upload(queue, buffers[4], query.data(), query_bytes))
      return false;

    const sllm_kv_state_paged_create_info_t info = create_info(0x8710U);
    passed &=
        expect(sllm_kv_state_create_paged(context, &info, &source, &error.sink),
               SLLM_STATUS_OK, "source create", error) &&
        source != nullptr;
    const sllm_kv_state_paged_create_info_t destination_info =
        create_info(0x8710U);
    passed &= expect(sllm_kv_state_create_paged(context, &destination_info,
                                                &destination, &error.sink),
                     SLLM_STATUS_OK, "destination create", error) &&
              destination != nullptr;
    passed &=
        expect(sllm_kv_state_create_paged(context, &destination_info,
                                          &invalid_destination, &error.sink),
               SLLM_STATUS_OK, "invalid destination create", error) &&
        invalid_destination != nullptr;
    if (!passed)
      return false;
    passed &= append(source, queue, buffers[0], buffers[1], 0U, kPrefix);
    if (!passed)
      return false;

    sllm_kv_paged_image_info_t image_info{};
    image_info.struct_size = sizeof(image_info);
    image_info.abi_version = SLLM_HIP_ABI_VERSION;
    image_info.image_version = SLLM_HIP_KV_PAGED_IMAGE_VERSION;
    passed &= expect(sllm_kv_state_paged_image_query(source, &image_info,
                                                     &error.sink),
                     SLLM_STATUS_OK, "image query", error) &&
              image_info.encoding == SLLM_HIP_KV_ENCODING_MXFP8_E4_V1 &&
              image_info.published_length == kPrefix &&
              image_info.plane_count == 4U &&
              image_info.physical_block_count == 2U;
    ImageBuffers image;
    passed &= export_image(source, image_info, &image);
    passed &= import_image(destination, image_info, image, false);
    passed &= finalize_image(destination, image_info, SLLM_STATUS_OK);
    passed &= import_image(invalid_destination, image_info, image, true);
    passed &= finalize_image(invalid_destination, image_info,
                             SLLM_STATUS_INVALID_ARGUMENT);

    sllm_kv_paged_view_info_t invalid_view{};
    invalid_view.struct_size = sizeof(invalid_view);
    invalid_view.abi_version = SLLM_HIP_ABI_VERSION;
    invalid_view.info_version = SLLM_HIP_KV_PAGED_VIEW_INFO_VERSION;
    passed &= expect(sllm_kv_state_query_paged(invalid_destination,
                                               &invalid_view, &error.sink),
                     SLLM_STATUS_OK, "invalid destination query", error) &&
              invalid_view.observed_length == 0U &&
              invalid_view.generation == 0U &&
              invalid_view.allocated_physical_blocks == 0U;

    sllm_kv_paged_view_info_t destination_view{};
    destination_view.struct_size = sizeof(destination_view);
    destination_view.abi_version = SLLM_HIP_ABI_VERSION;
    destination_view.info_version = SLLM_HIP_KV_PAGED_VIEW_INFO_VERSION;
    passed &= expect(sllm_kv_state_query_paged(destination, &destination_view,
                                               &error.sink),
                     SLLM_STATUS_OK, "destination query", error) &&
              destination_view.observed_length == image_info.published_length &&
              destination_view.generation == image_info.generation;

    passed &= append(source, queue, buffers[2], buffers[3], kPrefix, 1U) &&
              append(destination, queue, buffers[2], buffers[3], kPrefix, 1U);
    passed &=
        attention(source, context, queue, buffers[4], buffers[5], kPrefix,
                  &source_attention) &&
        wait_release(&source_attention, "source attention wait") &&
        attention(destination, context, queue, buffers[4], buffers[6], kPrefix,
                  &destination_attention) &&
        wait_release(&destination_attention, "destination attention wait");
    std::vector<uint16_t> source_output(output_bytes / sizeof(uint16_t));
    std::vector<uint16_t> destination_output(output_bytes / sizeof(uint16_t));
    passed &=
        download(queue, buffers[5], source_output.data(), output_bytes) &&
        download(queue, buffers[6], destination_output.data(), output_bytes) &&
        source_output == destination_output;

    std::vector<uint16_t> all_keys = keys;
    std::vector<uint16_t> all_values = values;
    all_keys.insert(all_keys.end(), tail_keys.begin(), tail_keys.end());
    all_values.insert(all_values.end(), tail_values.begin(), tail_values.end());
    const float expected = oracle_head0(all_keys, all_values, query);
    const float actual = bf16_float(source_output[0]);
    passed &= std::isfinite(actual) && std::fabs(actual - expected) < 0.2F;
    if (!passed) {
      std::cerr
          << "Paged image roundtrip oracle or bitwise comparison failed\n";
    }
    result = passed;
    return result;
  }();
  passed &= roundtrip;
  if (source_attention != nullptr)
    passed &= wait_release(&source_attention, "source attention cleanup");
  if (destination_attention != nullptr)
    passed &=
        wait_release(&destination_attention, "destination attention cleanup");
  passed &= run_sliding_roundtrip(context, queue);
  for (sllm_buffer_t *&buffer : buffers)
    (void)release_buffer(&buffer);
  (void)release_state(&invalid_destination);
  (void)release_state(&destination);
  (void)release_state(&source);
  (void)release_queue(&queue);
  (void)release_context(&context);
  if (passed) {
    std::cout << "phase87_stage10_paged_image_roundtrip_gpu_test: PASS target="
              << SLLM_TEST_EXPECTED_TARGET
              << " prefix=129 import=1 missing_plane_rejected=1"
                 " append_after_import=1 bitwise=1 oracle=1 cleanup=0\n";
  }
  return passed ? 0 : 1;
}
