#include "sllm/hip.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "gfx1201"
#endif

namespace {

constexpr uint32_t kCompletionTimeoutMs = 30'000U;
constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kKvHeadDim = 256U;
constexpr uint64_t kKvElementsPerToken =
    static_cast<uint64_t>(kKvHeads) * kKvHeadDim;
constexpr uint64_t kFp16BytesPerToken = kKvElementsPerToken * sizeof(uint16_t);

struct Error final {
  char message[512]{};
  sllm_error_sink_t sink{sizeof(sllm_error_sink_t),
                         SLLM_HIP_ABI_VERSION,
                         message,
                         sizeof(message),
                         0U,
                         {0U, 0U}};
};

bool expect(const sllm_status_t actual, const sllm_status_t expected,
            const char *const operation, const Error &error) {
  if (actual == expected) {
    return true;
  }
  std::cerr << operation << " returned " << actual << ", expected " << expected
            << ": " << error.message << '\n';
  return false;
}

bool wait_and_release(sllm_completion_t **const completion,
                      const char *const operation) {
  if (completion == nullptr || *completion == nullptr) {
    std::cerr << operation << " returned no completion\n";
    return false;
  }
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  const bool waited =
      expect(sllm_completion_wait(*completion, kCompletionTimeoutMs, &result,
                                  &error.sink),
             SLLM_STATUS_OK, operation, error) &&
      result.state == SLLM_COMPLETION_STATE_SUCCESS;
  const bool released = expect(sllm_completion_release(completion, &error.sink),
                               SLLM_STATUS_OK, "completion release", error);
  return waited && released && *completion == nullptr;
}

bool release_buffer(sllm_buffer_t **const buffer) {
  if (buffer == nullptr || *buffer == nullptr) {
    return true;
  }
  Error error;
  return expect(sllm_buffer_release(buffer, &error.sink), SLLM_STATUS_OK,
                "buffer release", error) &&
         *buffer == nullptr;
}

bool release_kv_state(sllm_kv_state_t **const state) {
  if (state == nullptr || *state == nullptr) {
    return true;
  }
  Error error;
  return expect(sllm_kv_state_release(state, &error.sink), SLLM_STATUS_OK,
                "KV state release", error) &&
         *state == nullptr;
}

bool release_linear_state(sllm_linear_attention_state_t **const state) {
  if (state == nullptr || *state == nullptr) {
    return true;
  }
  Error error;
  return expect(sllm_linear_attention_state_release(state, &error.sink),
                SLLM_STATUS_OK, "linear state release", error) &&
         *state == nullptr;
}

bool create_buffer(const sllm_context_t *const context, const uint64_t bytes,
                   sllm_buffer_t **const output) {
  sllm_buffer_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.size_bytes = bytes;
  Error error;
  return expect(sllm_buffer_create(context, &info, output, &error.sink),
                SLLM_STATUS_OK, "buffer create", error) &&
         *output != nullptr;
}

bool upload(const sllm_queue_t *const queue, const sllm_buffer_t *const buffer,
            const void *const bytes, const uint64_t size_bytes) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = const_cast<void *>(bytes);
  transfer.size_bytes = size_bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  return expect(sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                     &error.sink),
                SLLM_STATUS_OK, "buffer upload", error) &&
         wait_and_release(&completion, "buffer upload wait");
}

bool download(const sllm_queue_t *const queue,
              const sllm_buffer_t *const buffer, void *const destination,
              const uint64_t size_bytes) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.size_bytes = size_bytes;
  sllm_completion_t *completion = nullptr;
  Error error;
  if (!expect(sllm_buffer_copy_d2h(queue, buffer, &transfer, &completion,
                                   &error.sink),
              SLLM_STATUS_OK, "buffer download", error) ||
      completion == nullptr) {
    return false;
  }
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(sllm_completion_wait(completion, kCompletionTimeoutMs, &result,
                                   &error.sink),
              SLLM_STATUS_OK, "buffer download wait", error) ||
      result.state != SLLM_COMPLETION_STATE_SUCCESS) {
    (void)sllm_completion_release(&completion, &error.sink);
    return false;
  }
  uint64_t written = 0U;
  const bool read =
      expect(sllm_completion_read(completion, destination, size_bytes, &written,
                                  &error.sink),
             SLLM_STATUS_OK, "buffer download read", error) &&
      written == size_bytes;
  const bool released =
      expect(sllm_completion_release(&completion, &error.sink), SLLM_STATUS_OK,
             "buffer download completion release", error);
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
  for (uint32_t backwards = 0U; backwards != rank; ++backwards) {
    const uint32_t index = rank - 1U - backwards;
    result.shape[index] = shape[index];
    result.stride_elements[index] = stride;
    stride *= shape[index];
  }
  return result;
}

std::vector<uint16_t> kv_bf16_words(const uint64_t token_count,
                                    const uint32_t salt) {
  constexpr std::array<uint16_t, 4> patterns{
      UINT16_C(0x0000), UINT16_C(0x3f80), UINT16_C(0xc000), UINT16_C(0x3f00)};
  const std::size_t count =
      static_cast<std::size_t>(token_count * kKvElementsPerToken);
  std::vector<uint16_t> words(count);
  for (std::size_t index = 0U; index != count; ++index) {
    words[index] = patterns[(index + salt) % patterns.size()];
  }
  return words;
}

std::vector<uint16_t>
expected_fp16_words(const std::vector<uint16_t> &bf16_words) {
  std::vector<uint16_t> result(bf16_words.size());
  for (std::size_t index = 0U; index != bf16_words.size(); ++index) {
    switch (bf16_words[index]) {
    case UINT16_C(0x0000):
      result[index] = UINT16_C(0x0000);
      break;
    case UINT16_C(0x3f80):
      result[index] = UINT16_C(0x3c00);
      break;
    case UINT16_C(0xc000):
      result[index] = UINT16_C(0xc000);
      break;
    case UINT16_C(0x3f00):
      result[index] = UINT16_C(0x3800);
      break;
    default:
      std::abort();
    }
  }
  return result;
}

uint16_t f32_to_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t upper = bits >> 16U;
  bits += UINT32_C(0x7fff) + (upper & 1U);
  return static_cast<uint16_t>(bits >> 16U);
}

float bf16_to_f32(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

sllm_kv_append_desc_t append_descriptor(const sllm_buffer_t *const key,
                                        const sllm_buffer_t *const value,
                                        const uint64_t token_count,
                                        const uint64_t position) {
  const uint64_t shape[] = {token_count, kKvHeads, kKvHeadDim};
  sllm_kv_append_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.append_version = SLLM_HIP_KV_STATE_VERSION;
  descriptor.expected_length = position;
  descriptor.start_position = position;
  descriptor.key_input = binding(key, SLLM_TENSOR_DTYPE_BF16, 3U, shape);
  descriptor.value_input = binding(value, SLLM_TENSOR_DTYPE_BF16, 3U, shape);
  return descriptor;
}

bool append_kv(const sllm_kv_state_t *const state,
               const sllm_queue_t *const queue, const sllm_buffer_t *const key,
               const sllm_buffer_t *const value, const uint64_t token_count,
               const uint64_t position) {
  const sllm_kv_append_desc_t descriptor =
      append_descriptor(key, value, token_count, position);
  sllm_kv_append_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  Error error;
  const bool submitted =
      expect(sllm_kv_state_append(state, queue, &descriptor, &completion, &info,
                                  &error.sink),
             SLLM_STATUS_OK, "KV append", error);
  const bool exact =
      submitted && info.backend == SLLM_BACKEND_HIP &&
      info.dispatch_count == 1U && info.token_count == token_count &&
      info.start_position == position &&
      info.end_position == position + token_count &&
      info.commit_allowed == 1U && info.fallback_allowed == 0U &&
      info.fallback_used == 0U &&
      std::strcmp(info.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0;
  if (!exact) {
    std::cerr << "KV append did not report exact HIP/no-fallback target\n";
  }
  return exact && wait_and_release(&completion, "KV append wait");
}

bool query_kv(const sllm_kv_state_t *const state, const uint64_t length,
              const uint64_t generation, const uint32_t view_encoding) {
  sllm_kv_view_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_KV_VIEW_INFO_VERSION;
  Error error;
  return expect(sllm_kv_state_query(state, &info, &error.sink), SLLM_STATUS_OK,
                "KV state query", error) &&
         info.observed_length == length && info.generation == generation &&
         info.encoding == view_encoding && info.head_count == kKvHeads &&
         info.head_dim == kKvHeadDim && info.context_identity != 0U &&
         info.state_identity != 0U;
}

bool export_kv_words(const sllm_kv_state_t *const state, const uint32_t plane,
                     const uint64_t byte_offset,
                     std::vector<uint16_t> *const words) {
  sllm_state_chunk_t chunk{};
  chunk.struct_size = sizeof(chunk);
  chunk.abi_version = SLLM_HIP_ABI_VERSION;
  chunk.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
  chunk.plane = plane;
  chunk.byte_offset = byte_offset;
  chunk.byte_length = static_cast<uint64_t>(words->size()) * sizeof(uint16_t);
  chunk.host_pointer = words->data();
  chunk.host_capacity = chunk.byte_length;
  Error error;
  return expect(sllm_kv_state_export(state, &chunk, &error.sink),
                SLLM_STATUS_OK, "KV plane export", error);
}

bool run_fp16_cow_case(const sllm_context_t *const context,
                       const sllm_queue_t *const queue,
                       const uint64_t prefix_length) {
  constexpr uint64_t capacity = 257U;
  sllm_kv_state_t *source = nullptr;
  sllm_kv_state_t *child = nullptr;
  std::array<sllm_buffer_t *, 4> buffers{};
  Error error;
  const bool case_result = [&]() {
    sllm_kv_state_create_info_t create{};
    create.struct_size = sizeof(create);
    create.abi_version = SLLM_HIP_ABI_VERSION;
    create.session_id = UINT64_C(0x4100) + prefix_length;
    create.layer_id = static_cast<uint32_t>(prefix_length);
    create.capacity_tokens = capacity;
    create.head_count = kKvHeads;
    create.head_dim = kKvHeadDim;
    create.memory_kind = SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS;
    create.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
    if (!expect(sllm_kv_state_create(context, &create, &source, &error.sink),
                SLLM_STATUS_OK, "FP16 VMM state create", error) ||
        source == nullptr) {
      return false;
    }
    const std::vector<uint16_t> key_input = kv_bf16_words(prefix_length, 1U);
    const std::vector<uint16_t> value_input = kv_bf16_words(prefix_length, 2U);
    const std::vector<uint16_t> expected_key = expected_fp16_words(key_input);
    const std::vector<uint16_t> expected_value =
        expected_fp16_words(value_input);
    const uint64_t prefix_bytes = prefix_length * kFp16BytesPerToken;
    if (!create_buffer(context, prefix_bytes, &buffers[0]) ||
        !create_buffer(context, prefix_bytes, &buffers[1]) ||
        !upload(queue, buffers[0], key_input.data(), prefix_bytes) ||
        !upload(queue, buffers[1], value_input.data(), prefix_bytes) ||
        !append_kv(source, queue, buffers[0], buffers[1], prefix_length, 0U) ||
        !query_kv(source, prefix_length, 1U, SLLM_HIP_KV_ENCODING_FP16_V1)) {
      return false;
    }

    sllm_kv_state_create_info_v2_t destination{};
    destination.struct_size = sizeof(destination);
    destination.abi_version = SLLM_HIP_ABI_VERSION;
    destination.create_info_version = SLLM_HIP_KV_STATE_CREATE_INFO_V2_VERSION;
    destination.session_id = create.session_id;
    destination.layer_id = create.layer_id;
    destination.capacity_tokens = capacity;
    destination.head_count = kKvHeads;
    destination.head_dim = kKvHeadDim;
    destination.memory_kind = SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS;
    destination.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
    destination.dtype = SLLM_TENSOR_DTYPE_F16;
    destination.encoding = SLLM_HIP_KV_ENCODING_FP16_V1;
    sllm_state_fork_info_t fork{};
    fork.struct_size = sizeof(fork);
    fork.abi_version = SLLM_HIP_ABI_VERSION;
    fork.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
    if (!expect(sllm_kv_state_fork(source, &destination, &child, &fork,
                                   &error.sink),
                SLLM_STATUS_OK, "FP16 VMM state fork", error) ||
        child == nullptr ||
        fork.mode != SLLM_HIP_STATE_FORK_MODE_SHARED_READ_ONLY_PAGES ||
        fork.published_length != prefix_length || fork.shared_bytes == 0U ||
        fork.copied_bytes != 0U || fork.child_owned_bytes != 0U ||
        fork.page_bytes == 0U) {
      return false;
    }

    std::vector<uint16_t> source_key(expected_key.size());
    std::vector<uint16_t> source_value(expected_value.size());
    std::vector<uint16_t> child_key(expected_key.size());
    std::vector<uint16_t> child_value(expected_value.size());
    if (!export_kv_words(source, SLLM_HIP_KV_STATE_PLANE_KEY, 0U,
                         &source_key) ||
        !export_kv_words(source, SLLM_HIP_KV_STATE_PLANE_VALUE, 0U,
                         &source_value) ||
        !export_kv_words(child, SLLM_HIP_KV_STATE_PLANE_KEY, 0U, &child_key) ||
        !export_kv_words(child, SLLM_HIP_KV_STATE_PLANE_VALUE, 0U,
                         &child_value) ||
        source_key != expected_key || source_value != expected_value ||
        child_key != expected_key || child_value != expected_value) {
      std::cerr << "FP16 fork prefix byte oracle failed at length "
                << prefix_length << '\n';
      return false;
    }

    const std::vector<uint16_t> tail_key_input = kv_bf16_words(1U, 3U);
    const std::vector<uint16_t> tail_value_input = kv_bf16_words(1U, 0U);
    const std::vector<uint16_t> expected_tail_key =
        expected_fp16_words(tail_key_input);
    const std::vector<uint16_t> expected_tail_value =
        expected_fp16_words(tail_value_input);
    if (!create_buffer(context, kFp16BytesPerToken, &buffers[2]) ||
        !create_buffer(context, kFp16BytesPerToken, &buffers[3]) ||
        !upload(queue, buffers[2], tail_key_input.data(), kFp16BytesPerToken) ||
        !upload(queue, buffers[3], tail_value_input.data(),
                kFp16BytesPerToken) ||
        !append_kv(child, queue, buffers[2], buffers[3], 1U, prefix_length) ||
        !query_kv(source, prefix_length, 1U, SLLM_HIP_KV_ENCODING_FP16_V1) ||
        !query_kv(child, prefix_length + 1U, 2U,
                  SLLM_HIP_KV_ENCODING_FP16_V1)) {
      return false;
    }

    sllm_state_fork_info_t cow{};
    cow.struct_size = sizeof(cow);
    cow.abi_version = SLLM_HIP_ABI_VERSION;
    cow.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
    if (!expect(sllm_kv_state_fork_query(child, &cow, &error.sink),
                SLLM_STATUS_OK, "FP16 COW audit query", error) ||
        cow.copied_bytes < cow.page_bytes ||
        cow.shared_bytes >= fork.shared_bytes) {
      return false;
    }

    std::vector<uint16_t> child_key_after = expected_key;
    child_key_after.insert(child_key_after.end(), expected_tail_key.begin(),
                           expected_tail_key.end());
    std::vector<uint16_t> child_value_after = expected_value;
    child_value_after.insert(child_value_after.end(),
                             expected_tail_value.begin(),
                             expected_tail_value.end());
    std::vector<uint16_t> observed_child_key(child_key_after.size());
    std::vector<uint16_t> observed_child_value(child_value_after.size());
    std::fill(source_key.begin(), source_key.end(), UINT16_C(0xffff));
    std::fill(source_value.begin(), source_value.end(), UINT16_C(0xffff));
    if (!export_kv_words(source, SLLM_HIP_KV_STATE_PLANE_KEY, 0U,
                         &source_key) ||
        !export_kv_words(source, SLLM_HIP_KV_STATE_PLANE_VALUE, 0U,
                         &source_value) ||
        !export_kv_words(child, SLLM_HIP_KV_STATE_PLANE_KEY, 0U,
                         &observed_child_key) ||
        !export_kv_words(child, SLLM_HIP_KV_STATE_PLANE_VALUE, 0U,
                         &observed_child_value) ||
        source_key != expected_key || source_value != expected_value ||
        observed_child_key != child_key_after ||
        observed_child_value != child_value_after) {
      std::cerr << "FP16 COW byte oracle failed at length " << prefix_length
                << '\n';
      return false;
    }
    return true;
  }();

  bool cleaned = release_kv_state(&child) && release_kv_state(&source);
  for (auto iterator = buffers.rbegin(); iterator != buffers.rend();
       ++iterator) {
    cleaned = release_buffer(&*iterator) && cleaned;
  }
  return case_result && cleaned;
}

struct LowBitRecipe final {
  const char *name;
  uint32_t dtype;
  uint32_t encoding;
  uint32_t create_version;
  uint32_t block_size;
  uint32_t scale_dtype;
  uint32_t plane_count;
  uint64_t value_bytes_per_token;
  uint64_t scale_bytes_per_token;
  uint64_t outer_bytes_per_token;
};

std::vector<uint8_t> byte_pattern(const std::size_t bytes,
                                  const uint32_t salt) {
  std::vector<uint8_t> result(bytes);
  for (std::size_t index = 0U; index != bytes; ++index) {
    result[index] = static_cast<uint8_t>(
        (static_cast<uint64_t>(index) * UINT64_C(131) + salt * 17U) &
        UINT64_C(0xff));
  }
  return result;
}

bool import_kv_bytes(const sllm_kv_state_t *const state, const uint32_t plane,
                     std::vector<uint8_t> *const bytes) {
  sllm_state_chunk_t chunk{};
  chunk.struct_size = sizeof(chunk);
  chunk.abi_version = SLLM_HIP_ABI_VERSION;
  chunk.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
  chunk.plane = plane;
  chunk.byte_length = static_cast<uint64_t>(bytes->size());
  chunk.host_pointer = bytes->data();
  chunk.host_capacity = chunk.byte_length;
  Error error;
  return expect(sllm_kv_state_import(state, &chunk, &error.sink),
                SLLM_STATUS_OK, "low-bit plane import", error);
}

bool export_kv_bytes(const sllm_kv_state_t *const state, const uint32_t plane,
                     std::vector<uint8_t> *const bytes) {
  sllm_state_chunk_t chunk{};
  chunk.struct_size = sizeof(chunk);
  chunk.abi_version = SLLM_HIP_ABI_VERSION;
  chunk.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
  chunk.plane = plane;
  chunk.byte_length = static_cast<uint64_t>(bytes->size());
  chunk.host_pointer = bytes->data();
  chunk.host_capacity = chunk.byte_length;
  Error error;
  return expect(sllm_kv_state_export(state, &chunk, &error.sink),
                SLLM_STATUS_OK, "low-bit plane export", error);
}

bool run_lowbit_case(const sllm_context_t *const context,
                     const LowBitRecipe &recipe,
                     const uint64_t published_length,
                     const uint32_t case_index) {
  constexpr uint64_t capacity = 129U;
  sllm_kv_state_t *source = nullptr;
  sllm_kv_state_t *child = nullptr;
  Error error;
  const bool case_result = [&]() {
    sllm_kv_state_create_info_v2_t create{};
    create.struct_size = sizeof(create);
    create.abi_version = SLLM_HIP_ABI_VERSION;
    create.create_info_version = recipe.create_version;
    create.session_id = UINT64_C(0x5100) + case_index;
    create.layer_id = 100U + case_index;
    create.capacity_tokens = capacity;
    create.head_count = kKvHeads;
    create.head_dim = kKvHeadDim;
    create.memory_kind = SLLM_HIP_KV_MEMORY_KIND_CONTIGUOUS_RESIDENT;
    create.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
    create.dtype = recipe.dtype;
    create.encoding = recipe.encoding;
    create.block_size = recipe.block_size;
    create.scale_dtype = recipe.scale_dtype;
    if (recipe.encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1) {
      const float key_scale = 0.125F;
      const float value_scale = 0.25F;
      std::memcpy(&create.reserved[0], &key_scale, sizeof(key_scale));
      std::memcpy(&create.reserved[1], &value_scale, sizeof(value_scale));
    }
    if (!expect(sllm_kv_state_create_v2(context, &create, &source, &error.sink),
                SLLM_STATUS_OK, "low-bit state create", error) ||
        source == nullptr) {
      return false;
    }
    sllm_state_image_info_t image{};
    image.struct_size = sizeof(image);
    image.abi_version = SLLM_HIP_ABI_VERSION;
    image.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
    if (!expect(sllm_kv_state_image_query(source, &image, &error.sink),
                SLLM_STATUS_OK, "low-bit image query", error) ||
        image.plane_count != recipe.plane_count ||
        image.capacity_tokens != capacity ||
        image.encoding != recipe.encoding) {
      return false;
    }
    const std::array<uint64_t, 6> bytes_per_token{
        recipe.value_bytes_per_token, recipe.value_bytes_per_token,
        recipe.scale_bytes_per_token, recipe.scale_bytes_per_token,
        recipe.outer_bytes_per_token, recipe.outer_bytes_per_token};
    std::array<std::vector<uint8_t>, 6> patterns;
    for (uint32_t plane = 1U; plane <= recipe.plane_count; ++plane) {
      const uint64_t expected_size =
          capacity * bytes_per_token[static_cast<std::size_t>(plane - 1U)];
      uint64_t actual_size = 0U;
      if (!expect(sllm_kv_state_image_plane_size(source, plane, &actual_size,
                                                 &error.sink),
                  SLLM_STATUS_OK, "low-bit plane size query", error) ||
          actual_size != expected_size) {
        std::cerr << recipe.name << " plane " << plane
                  << " geometry mismatch actual=" << actual_size
                  << " expected=" << expected_size << '\n';
        return false;
      }
      patterns[static_cast<std::size_t>(plane - 1U)] = byte_pattern(
          static_cast<std::size_t>(expected_size), case_index + plane);
      if (!import_kv_bytes(source, plane,
                           &patterns[static_cast<std::size_t>(plane - 1U)])) {
        return false;
      }
    }
    image.published_length = published_length;
    image.generation = 7U;
    if (!expect(sllm_kv_state_import_finalize(source, &image, &error.sink),
                SLLM_STATUS_OK, "low-bit image finalize", error) ||
        !query_kv(source, published_length, 7U,
                  recipe.encoding == SLLM_HIP_KV_ENCODING_NVFP4_V1
                      ? SLLM_TENSOR_ENCODING_NVFP4_BLOCK16_E4M3FN_F32
                      : SLLM_TENSOR_ENCODING_FP8_OUTER_F32)) {
      return false;
    }

    sllm_state_fork_info_t fork{};
    fork.struct_size = sizeof(fork);
    fork.abi_version = SLLM_HIP_ABI_VERSION;
    fork.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
    if (!expect(sllm_kv_state_fork(source, &create, &child, &fork, &error.sink),
                SLLM_STATUS_OK, "low-bit state fork", error) ||
        child == nullptr || fork.mode != SLLM_HIP_STATE_FORK_MODE_DEVICE_COPY ||
        fork.published_length != published_length || fork.copied_bytes == 0U ||
        fork.shared_bytes != 0U ||
        !query_kv(child, published_length, 7U,
                  recipe.encoding == SLLM_HIP_KV_ENCODING_NVFP4_V1
                      ? SLLM_TENSOR_ENCODING_NVFP4_BLOCK16_E4M3FN_F32
                      : SLLM_TENSOR_ENCODING_FP8_OUTER_F32)) {
      return false;
    }
    for (uint32_t plane = 1U; plane <= recipe.plane_count; ++plane) {
      const uint64_t visible_bytes =
          published_length *
          bytes_per_token[static_cast<std::size_t>(plane - 1U)];
      std::vector<uint8_t> expected(
          patterns[static_cast<std::size_t>(plane - 1U)].begin(),
          patterns[static_cast<std::size_t>(plane - 1U)].begin() +
              static_cast<std::ptrdiff_t>(visible_bytes));
      std::vector<uint8_t> source_bytes(expected.size());
      std::vector<uint8_t> child_bytes(expected.size());
      if (!export_kv_bytes(source, plane, &source_bytes) ||
          !export_kv_bytes(child, plane, &child_bytes) ||
          source_bytes != expected || child_bytes != expected) {
        std::cerr << recipe.name << " plane " << plane
                  << " fork byte oracle failed\n";
        return false;
      }
    }
    return true;
  }();
  const bool cleaned = release_kv_state(&child) && release_kv_state(&source);
  return case_result && cleaned;
}

float attention_pattern(const uint64_t value) {
  constexpr std::array<float, 5> values{-2.0F, -1.0F, 0.0F, 1.0F, 2.0F};
  return values[static_cast<std::size_t>(value % values.size())];
}

bool run_explicit_scale_attention_case(const sllm_context_t *const context,
                                       const sllm_queue_t *const queue,
                                       const bool sliding,
                                       const uint32_t head_dim,
                                       const uint32_t q_heads) {
  constexpr uint64_t window = SLLM_HIP_KV_SLIDING_WINDOW_GEMMA4;
  const uint64_t final_length = sliding ? 1026U : 1025U;
  const uint64_t capacity =
      sliding ? SLLM_HIP_KV_SLIDING_MAX_CAPACITY : final_length;
  const uint64_t elements_per_token =
      static_cast<uint64_t>(kKvHeads) * head_dim;
  const uint64_t input_bytes = final_length * elements_per_token * 2U;
  const uint64_t query_elements = UINT64_C(2) * q_heads * head_dim;
  sllm_kv_state_t *state = nullptr;
  sllm_kv_state_t *fresh_recovery = nullptr;
  std::array<sllm_buffer_t *, 4> buffers{};
  Error error;
  sllm_completion_t *completion = nullptr;
  const bool case_result = [&]() {
    sllm_kv_state_create_info_v2_t create{};
    create.struct_size = sizeof(create);
    create.abi_version = SLLM_HIP_ABI_VERSION;
    create.create_info_version =
        sliding ? SLLM_HIP_KV_STATE_CREATE_INFO_SLIDING_STATIC_FP8_VERSION
                : SLLM_HIP_KV_STATE_CREATE_INFO_STATIC_FP8_VERSION;
    create.session_id = sliding ? UINT64_C(0x7155) : UINT64_C(0x7156);
    create.layer_id = sliding ? 55U : 56U;
    create.capacity_tokens = capacity;
    create.head_count = kKvHeads;
    create.head_dim = head_dim;
    create.memory_kind = sliding ? SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS
                                 : SLLM_HIP_KV_MEMORY_KIND_CONTIGUOUS_RESIDENT;
    create.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
    create.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
    create.encoding = SLLM_HIP_KV_ENCODING_FP8_STATIC_V1;
    create.scale_dtype = SLLM_TENSOR_DTYPE_F32;
    const float unit = 1.0F;
    std::memcpy(&create.reserved[0], &unit, sizeof(unit));
    std::memcpy(&create.reserved[1], &unit, sizeof(unit));
    if (sliding) {
      create.reserved[2] = static_cast<uint32_t>(window);
      create.reserved[3] = static_cast<uint32_t>(window >> 32U);
    }
    if (!expect(sllm_kv_state_create_v2(context, &create, &state, &error.sink),
                SLLM_STATUS_OK, "scaled static KV create", error) ||
        state == nullptr) {
      return false;
    }

    std::vector<uint16_t> key_words(
        static_cast<std::size_t>(final_length * elements_per_token));
    std::vector<uint16_t> value_words(key_words.size());
    for (uint64_t token = 0U; token != final_length; ++token) {
      for (uint32_t head = 0U; head != kKvHeads; ++head) {
        for (uint32_t dimension = 0U; dimension != head_dim; ++dimension) {
          const std::size_t index = static_cast<std::size_t>(
              (token * kKvHeads + head) * head_dim + dimension);
          key_words[index] = f32_to_bf16_rne(
              dimension == 0U ? attention_pattern(token + head) : 0.0F);
          value_words[index] =
              f32_to_bf16_rne(attention_pattern(token * 3U + head + dimension));
        }
      }
    }
    std::vector<uint16_t> query_words(static_cast<std::size_t>(query_elements),
                                      UINT16_C(0));
    for (uint64_t row = 0U; row != 2U; ++row) {
      for (uint32_t head = 0U; head != q_heads; ++head) {
        const std::size_t index =
            static_cast<std::size_t>((row * q_heads + head) * head_dim);
        query_words[index] = f32_to_bf16_rne(1.0F);
      }
    }
    if (!create_buffer(context, input_bytes, &buffers[0]) ||
        !create_buffer(context, input_bytes, &buffers[1]) ||
        !create_buffer(context, query_elements * 2U, &buffers[2]) ||
        !create_buffer(context, query_elements * 2U, &buffers[3]) ||
        !upload(queue, buffers[0], key_words.data(), input_bytes) ||
        !upload(queue, buffers[1], value_words.data(), input_bytes) ||
        !upload(queue, buffers[2], query_words.data(), query_elements * 2U)) {
      return false;
    }

    const auto append_range = [&](const uint64_t count, const uint64_t position,
                                  const sllm_status_t expected_status) {
      const uint64_t shape[] = {count, kKvHeads, head_dim};
      sllm_kv_append_desc_t descriptor{};
      descriptor.struct_size = sizeof(descriptor);
      descriptor.abi_version = SLLM_HIP_ABI_VERSION;
      descriptor.append_version = SLLM_HIP_KV_STATE_VERSION;
      descriptor.expected_length = position;
      descriptor.start_position = position;
      descriptor.key_input =
          binding(buffers[0], SLLM_TENSOR_DTYPE_BF16, 3U, shape);
      descriptor.value_input =
          binding(buffers[1], SLLM_TENSOR_DTYPE_BF16, 3U, shape);
      // A rejected saturated M > 1 append still needs otherwise-valid input
      // bindings so the oracle proves the sliding-state restriction itself.
      const uint64_t offset =
          expected_status == SLLM_STATUS_OK
              ? position * elements_per_token * sizeof(uint16_t)
              : 0U;
      descriptor.key_input.byte_offset = offset;
      descriptor.value_input.byte_offset = offset;
      sllm_kv_append_info_t info{};
      info.struct_size = sizeof(info);
      info.abi_version = SLLM_HIP_ABI_VERSION;
      info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
      completion = nullptr;
      if (!expect(sllm_kv_state_append(state, queue, &descriptor, &completion,
                                       &info, &error.sink),
                  expected_status, "scaled static append", error)) {
        return false;
      }
      if (expected_status != SLLM_STATUS_OK) {
        return completion == nullptr;
      }
      return completion != nullptr && info.dispatch_count == 1U &&
             info.fallback_allowed == 0U && info.fallback_used == 0U &&
             std::strcmp(info.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) == 0 &&
             wait_and_release(&completion, "scaled static append wait");
    };
    const auto query_length = [&](const uint64_t length) {
      sllm_kv_view_info_t info{};
      info.struct_size = sizeof(info);
      info.abi_version = SLLM_HIP_ABI_VERSION;
      info.info_version = SLLM_HIP_KV_VIEW_INFO_VERSION;
      if (!expect(sllm_kv_state_query(state, &info, &error.sink),
                  SLLM_STATUS_OK, "scaled static query", error) ||
          info.observed_length != length) {
        return false;
      }
      return !sliding ||
             (info.info_version == SLLM_HIP_KV_VIEW_INFO_SLIDING_VERSION &&
              info.mapped_token_capacity <= window + 1U &&
              info.reserved[0] == window &&
              info.reserved[2] == static_cast<uint32_t>(
                                      length > window ? length - window : 0U));
    };
    const auto execute_attention = [&](const uint64_t start,
                                       const uint32_t query_count,
                                       const uint64_t length) {
      const uint64_t shape[] = {query_count, q_heads, head_dim};
      sllm_causal_attention_desc_t descriptor{};
      descriptor.struct_size = sizeof(descriptor);
      descriptor.abi_version = SLLM_HIP_ABI_VERSION;
      descriptor.op_version = SLLM_HIP_CAUSAL_ATTENTION_EXPLICIT_SCALE_VERSION;
      descriptor.start_position = start;
      descriptor.expected_kv_length = length;
      descriptor.kv_state = state;
      descriptor.query = binding(buffers[2], SLLM_TENSOR_DTYPE_BF16, 3U, shape);
      descriptor.output =
          binding(buffers[3], SLLM_TENSOR_DTYPE_BF16, 3U, shape);
      if (sliding) {
        descriptor.reserved[0] = static_cast<uint32_t>(window);
      }
      std::memcpy(&descriptor.reserved[2], &unit, sizeof(unit));
      sllm_causal_attention_dispatch_info_t dispatch{};
      dispatch.struct_size = sizeof(dispatch);
      dispatch.abi_version = SLLM_HIP_ABI_VERSION;
      dispatch.info_version = SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION;
      completion = nullptr;
      if (!expect(sllm_causal_attention_execute(context, queue, &descriptor,
                                                &completion, &dispatch,
                                                &error.sink),
                  SLLM_STATUS_OK, "explicit-scale causal attention", error) ||
          completion == nullptr || dispatch.dispatch_count != 1U ||
          dispatch.kernel_id !=
              (sliding
                   ? SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_SLIDING_STATIC_FP8_V1
                   : SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_SCALED_STATIC_FP8_V1) ||
          dispatch.fallback_allowed != 0U || dispatch.fallback_used != 0U ||
          dispatch.scale_denominator != 0U ||
          dispatch.reserved[4] != UINT32_C(0x3f800000) ||
          dispatch.reserved[5] != 1U ||
          std::strcmp(dispatch.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) != 0 ||
          !wait_and_release(&completion, "explicit-scale attention wait")) {
        return false;
      }
      const uint64_t output_elements =
          static_cast<uint64_t>(query_count) * q_heads * head_dim;
      std::vector<uint16_t> observed(static_cast<std::size_t>(output_elements));
      if (!download(queue, buffers[3], observed.data(), output_elements * 2U)) {
        return false;
      }
      const auto oracle = [&](const float score_scale) {
        std::vector<uint16_t> expected(
            static_cast<std::size_t>(output_elements));
        for (uint32_t row = 0U; row != query_count; ++row) {
          const uint64_t query_position = start + row;
          const uint64_t key_begin = sliding && query_position + 1U > window
                                         ? query_position + 1U - window
                                         : 0U;
          for (uint32_t query_head = 0U; query_head != q_heads; ++query_head) {
            const uint32_t kv_head = query_head / (q_heads / kKvHeads);
            std::vector<float> accumulation(head_dim, 0.0F);
            float maximum = -std::numeric_limits<float>::infinity();
            float denominator = 0.0F;
            for (uint64_t token = key_begin; token <= query_position; ++token) {
              const float score =
                  attention_pattern(token + kv_head) * score_scale;
              const float next_maximum = std::max(maximum, score);
              const float rescale = std::exp(maximum - next_maximum);
              const float contribution = std::exp(score - next_maximum);
              denominator = denominator * rescale + contribution;
              for (uint32_t dimension = 0U; dimension != head_dim;
                   ++dimension) {
                accumulation[dimension] =
                    accumulation[dimension] * rescale +
                    contribution *
                        attention_pattern(token * 3U + kv_head + dimension);
              }
              maximum = next_maximum;
            }
            for (uint32_t dimension = 0U; dimension != head_dim; ++dimension) {
              const std::size_t index = static_cast<std::size_t>(
                  (static_cast<uint64_t>(row) * q_heads + query_head) *
                      head_dim +
                  dimension);
              expected[index] =
                  f32_to_bf16_rne(accumulation[dimension] / denominator);
            }
          }
        }
        return expected;
      };
      const std::vector<uint16_t> expected = oracle(1.0F);
      const std::vector<uint16_t> legacy =
          oracle(1.0F / std::sqrt(static_cast<float>(head_dim)));
      bool distinguishes_legacy = query_count > 1U || length > 2U;
      bool exact = true;
      bool observed_differs_from_legacy = false;
      for (std::size_t index = 0U; index != observed.size(); ++index) {
        const float actual = bf16_to_f32(observed[index]);
        const float wanted = bf16_to_f32(expected[index]);
        if (std::fabs(actual - wanted) > 0.03125F) {
          exact = false;
          std::cerr << "explicit-scale oracle mismatch index=" << index
                    << " actual=" << actual << " expected=" << wanted
                    << " start=" << start << " length=" << length << '\n';
          break;
        }
        observed_differs_from_legacy =
            observed_differs_from_legacy || observed[index] != legacy[index];
      }
      if (!exact || (distinguishes_legacy && !observed_differs_from_legacy)) {
        std::cerr << "explicit 1.0 scale did not separate from rsqrt(head_dim)"
                  << " start=" << start << " length=" << length << '\n';
        return false;
      }
      return true;
    };

    if (sliding) {
      if (!append_range(2U, 0U, SLLM_STATUS_OK) || !query_length(2U) ||
          !execute_attention(0U, 2U, 2U) ||
          !append_range(1021U, 2U, SLLM_STATUS_OK) || !query_length(1023U) ||
          !execute_attention(1022U, 1U, 1023U) ||
          !append_range(1U, 1023U, SLLM_STATUS_OK) || !query_length(1024U) ||
          !execute_attention(1023U, 1U, 1024U) ||
          !append_range(1U, 1024U, SLLM_STATUS_OK) || !query_length(1025U) ||
          !execute_attention(1024U, 1U, 1025U) ||
          !append_range(2U, 1025U, SLLM_STATUS_INVALID_KV_APPEND_DESCRIPTOR) ||
          !append_range(1U, 1025U, SLLM_STATUS_OK) || !query_length(1026U) ||
          !execute_attention(1025U, 1U, 1026U)) {
        return false;
      }
      sllm_kv_append_desc_t canceled =
          append_descriptor(buffers[0], buffers[1], 1U, 1026U);
      canceled.key_input.byte_offset = 1025U * elements_per_token * 2U;
      canceled.value_input.byte_offset = 1025U * elements_per_token * 2U;
      sllm_kv_append_info_t cancel_info{};
      cancel_info.struct_size = sizeof(cancel_info);
      cancel_info.abi_version = SLLM_HIP_ABI_VERSION;
      cancel_info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
      if (!expect(sllm_kv_state_append(state, queue, &canceled, &completion,
                                       &cancel_info, &error.sink),
                  SLLM_STATUS_OK, "sliding cancel submission", error) ||
          completion == nullptr ||
          !expect(sllm_kv_state_append_cancel(state, completion, &error.sink),
                  SLLM_STATUS_OK, "sliding cancel", error) ||
          !wait_and_release(&completion, "sliding canceled completion") ||
          !query_length(1026U)) {
        return false;
      }
      if (!expect(sllm_kv_state_create_v2(context, &create, &fresh_recovery,
                                          &error.sink),
                  SLLM_STATUS_OK, "fresh sliding recovery state", error) ||
          fresh_recovery == nullptr) {
        return false;
      }
    } else if (!append_range(final_length, 0U, SLLM_STATUS_OK) ||
               !query_length(final_length) ||
               !execute_attention(final_length - 1U, 1U, final_length)) {
      return false;
    }
    return true;
  }();

  if (completion != nullptr) {
    Error completion_error;
    (void)sllm_completion_release(&completion, &completion_error.sink);
  }
  bool cleaned = release_kv_state(&fresh_recovery) && release_kv_state(&state);
  for (auto iterator = buffers.rbegin(); iterator != buffers.rend();
       ++iterator) {
    cleaned = release_buffer(&*iterator) && cleaned;
  }
  return case_result && cleaned;
}

bool import_linear_bytes(const sllm_linear_attention_state_t *const state,
                         const uint32_t plane,
                         std::vector<uint8_t> *const bytes) {
  sllm_state_chunk_t chunk{};
  chunk.struct_size = sizeof(chunk);
  chunk.abi_version = SLLM_HIP_ABI_VERSION;
  chunk.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
  chunk.plane = plane;
  chunk.byte_length = static_cast<uint64_t>(bytes->size());
  chunk.host_pointer = bytes->data();
  chunk.host_capacity = chunk.byte_length;
  Error error;
  return expect(sllm_linear_attention_state_import(state, &chunk, &error.sink),
                SLLM_STATUS_OK, "linear plane import", error);
}

bool export_linear_bytes(const sllm_linear_attention_state_t *const state,
                         const uint32_t plane,
                         std::vector<uint8_t> *const bytes) {
  sllm_state_chunk_t chunk{};
  chunk.struct_size = sizeof(chunk);
  chunk.abi_version = SLLM_HIP_ABI_VERSION;
  chunk.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
  chunk.plane = plane;
  chunk.byte_length = static_cast<uint64_t>(bytes->size());
  chunk.host_pointer = bytes->data();
  chunk.host_capacity = chunk.byte_length;
  Error error;
  return expect(sllm_linear_attention_state_export(state, &chunk, &error.sink),
                SLLM_STATUS_OK, "linear plane export", error);
}

bool run_linear_case(const sllm_context_t *const context,
                     const sllm_queue_t *const queue) {
  constexpr uint32_t qk_heads = 16U;
  constexpr uint32_t value_heads = 16U;
  constexpr uint32_t head_dim = 128U;
  constexpr uint32_t conv_kernel = 4U;
  constexpr uint64_t source_capacity = 129U;
  constexpr uint64_t destination_capacity = 257U;
  constexpr uint64_t qkv_width =
      static_cast<uint64_t>(2U * qk_heads + value_heads) * head_dim;
  constexpr uint64_t output_width =
      static_cast<uint64_t>(value_heads) * head_dim;
  sllm_linear_attention_state_t *source = nullptr;
  sllm_linear_attention_state_t *child = nullptr;
  std::array<sllm_buffer_t *, 9> buffers{};
  Error error;
  const bool case_result = [&]() {
    sllm_linear_attention_state_create_info_t create{};
    create.struct_size = sizeof(create);
    create.abi_version = SLLM_HIP_ABI_VERSION;
    create.session_id = UINT64_C(0x6100);
    create.layer_id = 17U;
    create.capacity_tokens = source_capacity;
    create.qk_heads = qk_heads;
    create.value_heads = value_heads;
    create.head_dim = head_dim;
    create.conv_kernel_size = conv_kernel;
    if (!expect(sllm_linear_attention_state_create(context, &create, &source,
                                                   &error.sink),
                SLLM_STATUS_OK, "linear state create", error) ||
        source == nullptr) {
      return false;
    }

    const std::array<uint64_t, 9> sizes{
        qkv_width * sizeof(uint16_t),
        output_width * sizeof(uint16_t),
        static_cast<uint64_t>(value_heads) * sizeof(uint16_t),
        static_cast<uint64_t>(value_heads) * sizeof(uint16_t),
        qkv_width * conv_kernel * sizeof(uint16_t),
        static_cast<uint64_t>(value_heads) * sizeof(float),
        static_cast<uint64_t>(value_heads) * sizeof(uint16_t),
        static_cast<uint64_t>(head_dim) * sizeof(float),
        output_width * sizeof(uint16_t)};
    for (std::size_t index = 0U; index != buffers.size(); ++index) {
      if (!create_buffer(context, sizes[index], &buffers[index])) {
        return false;
      }
      const std::vector<uint8_t> zeros(static_cast<std::size_t>(sizes[index]),
                                       UINT8_C(0));
      if (!upload(queue, buffers[index], zeros.data(), sizes[index])) {
        return false;
      }
    }

    constexpr uint64_t qkv_shape[] = {1U, qkv_width};
    constexpr uint64_t output_shape[] = {1U, output_width};
    constexpr uint64_t scalar_shape[] = {1U, value_heads};
    constexpr uint64_t conv_shape[] = {qkv_width, 1U, conv_kernel};
    constexpr uint64_t head_shape[] = {value_heads};
    constexpr uint64_t norm_shape[] = {head_dim};
    sllm_linear_attention_desc_t descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = SLLM_HIP_ABI_VERSION;
    descriptor.op_version = SLLM_HIP_LINEAR_ATTENTION_VERSION;
    descriptor.expected_length = 1U;
    descriptor.state = source;
    descriptor.qkv = binding(buffers[0], SLLM_TENSOR_DTYPE_BF16, 2U, qkv_shape);
    descriptor.z =
        binding(buffers[1], SLLM_TENSOR_DTYPE_BF16, 2U, output_shape);
    descriptor.b_input =
        binding(buffers[2], SLLM_TENSOR_DTYPE_BF16, 2U, scalar_shape);
    descriptor.a_input =
        binding(buffers[3], SLLM_TENSOR_DTYPE_BF16, 2U, scalar_shape);
    descriptor.conv_weight =
        binding(buffers[4], SLLM_TENSOR_DTYPE_BF16, 3U, conv_shape);
    descriptor.a_log =
        binding(buffers[5], SLLM_TENSOR_DTYPE_F32, 1U, head_shape);
    descriptor.dt_bias =
        binding(buffers[6], SLLM_TENSOR_DTYPE_BF16, 1U, head_shape);
    descriptor.norm_weight =
        binding(buffers[7], SLLM_TENSOR_DTYPE_F32, 1U, norm_shape);
    descriptor.output =
        binding(buffers[8], SLLM_TENSOR_DTYPE_BF16, 2U, output_shape);
    sllm_linear_attention_dispatch_info_t dispatch{};
    dispatch.struct_size = sizeof(dispatch);
    dispatch.abi_version = SLLM_HIP_ABI_VERSION;
    dispatch.info_version = SLLM_HIP_LINEAR_ATTENTION_DISPATCH_INFO_VERSION;
    sllm_completion_t *completion = nullptr;
    if (!expect(sllm_linear_attention_execute(context, queue, &descriptor,
                                              &completion, &dispatch,
                                              &error.sink),
                SLLM_STATUS_OK, "linear execute", error) ||
        dispatch.backend != SLLM_BACKEND_HIP || dispatch.dispatch_count != 2U ||
        dispatch.fallback_allowed != 0U || dispatch.fallback_used != 0U ||
        std::strcmp(dispatch.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) != 0 ||
        !wait_and_release(&completion, "linear execute wait")) {
      return false;
    }
    std::vector<uint16_t> output(static_cast<std::size_t>(output_width));
    if (!download(queue, buffers[8], output.data(), sizes[8]) ||
        !std::all_of(output.begin(), output.end(),
                     [](const uint16_t value) { return value == 0U; })) {
      std::cerr << "linear zero numerical oracle failed\n";
      return false;
    }

    sllm_state_image_info_t image{};
    image.struct_size = sizeof(image);
    image.abi_version = SLLM_HIP_ABI_VERSION;
    image.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
    if (!expect(sllm_linear_attention_state_image_query(source, &image,
                                                        &error.sink),
                SLLM_STATUS_OK, "linear image query", error) ||
        image.active_slot != 1U || image.published_length != 1U ||
        image.generation != 1U || image.plane_count != 5U) {
      return false;
    }
    std::array<std::vector<uint8_t>, 5> patterns;
    for (uint32_t plane = 1U; plane <= image.plane_count; ++plane) {
      uint64_t bytes = 0U;
      if (!expect(sllm_linear_attention_state_image_plane_size(
                      source, plane, &bytes, &error.sink),
                  SLLM_STATUS_OK, "linear plane size query", error) ||
          bytes == 0U) {
        return false;
      }
      patterns[static_cast<std::size_t>(plane - 1U)] =
          byte_pattern(static_cast<std::size_t>(bytes), 80U + plane);
      if (!import_linear_bytes(
              source, plane, &patterns[static_cast<std::size_t>(plane - 1U)])) {
        return false;
      }
    }
    image.published_length = source_capacity;
    image.generation = 7U;
    if (!expect(sllm_linear_attention_state_import_finalize(source, &image,
                                                            &error.sink),
                SLLM_STATUS_OK, "linear image finalize", error)) {
      return false;
    }

    sllm_linear_attention_state_create_info_t destination = create;
    destination.capacity_tokens = destination_capacity;
    sllm_state_fork_info_t fork{};
    fork.struct_size = sizeof(fork);
    fork.abi_version = SLLM_HIP_ABI_VERSION;
    fork.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
    if (!expect(sllm_linear_attention_state_fork(source, &destination, &child,
                                                 &fork, &error.sink),
                SLLM_STATUS_OK, "linear state fork", error) ||
        child == nullptr || fork.mode != SLLM_HIP_STATE_FORK_MODE_DEVICE_COPY ||
        fork.published_length != source_capacity || fork.copied_bytes == 0U ||
        fork.shared_bytes != 0U) {
      return false;
    }
    sllm_linear_attention_view_info_t source_view{};
    source_view.struct_size = sizeof(source_view);
    source_view.abi_version = SLLM_HIP_ABI_VERSION;
    source_view.info_version = SLLM_HIP_LINEAR_ATTENTION_VIEW_INFO_VERSION;
    sllm_linear_attention_view_info_t child_view = source_view;
    if (!expect(sllm_linear_attention_state_query(source, &source_view,
                                                  &error.sink),
                SLLM_STATUS_OK, "linear source query", error) ||
        !expect(
            sllm_linear_attention_state_query(child, &child_view, &error.sink),
            SLLM_STATUS_OK, "linear child query", error) ||
        source_view.active_slot != 1U || child_view.active_slot != 1U ||
        source_view.observed_length != source_capacity ||
        child_view.observed_length != source_capacity ||
        source_view.generation != 7U || child_view.generation != 7U ||
        child_view.capacity_tokens != destination_capacity) {
      return false;
    }
    for (uint32_t plane = 1U; plane <= image.plane_count; ++plane) {
      const std::vector<uint8_t> &expected =
          patterns[static_cast<std::size_t>(plane - 1U)];
      std::vector<uint8_t> source_bytes(expected.size());
      std::vector<uint8_t> child_bytes(expected.size());
      if (!export_linear_bytes(source, plane, &source_bytes) ||
          !export_linear_bytes(child, plane, &child_bytes) ||
          source_bytes != expected || child_bytes != expected) {
        std::cerr << "linear plane " << plane << " fork byte oracle failed\n";
        return false;
      }
    }
    return true;
  }();

  bool cleaned = release_linear_state(&child) && release_linear_state(&source);
  for (auto iterator = buffers.rbegin(); iterator != buffers.rend();
       ++iterator) {
    cleaned = release_buffer(&*iterator) && cleaned;
  }
  return case_result && cleaned;
}

bool validate_visible_device(sllm_context_t **const context,
                             sllm_queue_t **const queue) {
  Error error;
  uint32_t count = 0U;
  if (!expect(sllm_device_count(&count, &error.sink), SLLM_STATUS_OK,
              "device count", error) ||
      count != 1U) {
    std::cerr << "runner requires exactly one ROCR-visible GPU, got " << count
              << '\n';
    return false;
  }
  sllm_device_info_t device{};
  device.struct_size = sizeof(device);
  device.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(sllm_device_query(0U, &device, &error.sink), SLLM_STATUS_OK,
              "device query", error) ||
      device.visible_device_count != 1U || device.device_index != 0U ||
      device.wavefront_size != 32U ||
      std::strcmp(device.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) != 0) {
    std::cerr << "visible device is not exact target "
              << SLLM_TEST_EXPECTED_TARGET << '\n';
    return false;
  }
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  context_info.device_index = 0U;
  std::snprintf(context_info.expected_gcn_arch_name,
                sizeof(context_info.expected_gcn_arch_name), "%s",
                SLLM_TEST_EXPECTED_TARGET);
  if (!expect(sllm_context_create(&context_info, context, &error.sink),
              SLLM_STATUS_OK, "context create", error) ||
      *context == nullptr) {
    return false;
  }
  sllm_context_probe_result_t probe{};
  probe.struct_size = sizeof(probe);
  probe.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(sllm_context_probe(*context, &probe, &error.sink), SLLM_STATUS_OK,
              "context probe", error) ||
      probe.context_present != 1U || probe.hip_available != 1U) {
    return false;
  }
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  return expect(sllm_queue_create(*context, &queue_info, queue, &error.sink),
                SLLM_STATUS_OK, "queue create", error) &&
         *queue != nullptr;
}

} // namespace

int main() {
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  bool success = validate_visible_device(&context, &queue);

  constexpr std::array<uint64_t, 6> boundaries{63U, 64U, 65U, 127U, 128U, 129U};
  for (const uint64_t boundary : boundaries) {
    if (success && !run_fp16_cow_case(context, queue, boundary)) {
      std::cerr << "FP16 state/COW case failed at boundary " << boundary
                << '\n';
      success = false;
    }
  }

  constexpr std::array<LowBitRecipe, 3> recipes{{
      {"fp8-dynamic", SLLM_TENSOR_DTYPE_F8_E4M3_FN, SLLM_HIP_KV_ENCODING_FP8_V1,
       SLLM_HIP_KV_STATE_CREATE_INFO_V2_VERSION, 0U, SLLM_TENSOR_DTYPE_F32, 4U,
       kKvElementsPerToken, static_cast<uint64_t>(kKvHeads) * sizeof(float),
       0U},
      {"fp8-static", SLLM_TENSOR_DTYPE_F8_E4M3_FN,
       SLLM_HIP_KV_ENCODING_FP8_STATIC_V1,
       SLLM_HIP_KV_STATE_CREATE_INFO_STATIC_FP8_VERSION, 0U,
       SLLM_TENSOR_DTYPE_F32, 2U, kKvElementsPerToken, 0U, 0U},
      {"nvfp4", SLLM_TENSOR_DTYPE_U8, SLLM_HIP_KV_ENCODING_NVFP4_V1,
       SLLM_HIP_KV_STATE_CREATE_INFO_V2_VERSION, 16U,
       SLLM_TENSOR_DTYPE_F8_E4M3_FN, 6U, kKvElementsPerToken / 2U,
       static_cast<uint64_t>(kKvHeads) * (kKvHeadDim / 16U),
       static_cast<uint64_t>(kKvHeads) * sizeof(float)},
  }};
  constexpr std::array<uint64_t, 3> lowbit_lengths{65U, 128U, 129U};
  for (std::size_t index = 0U; index != recipes.size(); ++index) {
    if (success &&
        !run_lowbit_case(context, recipes[index], lowbit_lengths[index],
                         static_cast<uint32_t>(index))) {
      std::cerr << "low-bit state case failed: " << recipes[index].name << '\n';
      success = false;
    }
  }
  if (success &&
      !run_explicit_scale_attention_case(context, queue, false, 512U, 8U)) {
    std::cerr << "full static FP8 explicit-scale attention case failed\n";
    success = false;
  }
  if (success &&
      !run_explicit_scale_attention_case(context, queue, true, 256U, 16U)) {
    std::cerr << "sliding static FP8 explicit-scale attention case failed\n";
    success = false;
  }
  if (success && !run_linear_case(context, queue)) {
    std::cerr << "linear state image/fork case failed\n";
    success = false;
  }

  Error error;
  if (queue != nullptr) {
    success = expect(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
                     "queue release", error) &&
              queue == nullptr && success;
  }
  if (context != nullptr) {
    success = expect(sllm_context_release(&context, &error.sink),
                     SLLM_STATUS_OK, "context release", error) &&
              context == nullptr && success;
  }
  if (success) {
    std::cout << "phase41 state GPU PASS target=" << SLLM_TEST_EXPECTED_TARGET
              << " fp16_boundaries=6 lowbit_encodings=3 linear_planes=5"
                 " explicit_scale_oracles=full512+sliding256"
                 " sliding_boundaries=1023,1024,1025 wrap=1026"
                 " numerical_oracles=13 fallback=false cleanup_failures=0\n";
  }
  return success ? 0 : 1;
}
