// Phase 87 Stage 10: bounded production-source Paged prefill AB/BA probe.
//
// This probe measures only the reviewed M=128 prefill shapes at the two hot
// KV lengths.  The public ABI does not expose logical-table permutation, so
// the reverse arm is the same-process BA order; that limitation is reported.
#include "sllm/hip.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#error SLLM_TEST_EXPECTED_TARGET must name one exact GPU target
#endif
#ifndef SLLM_TEST_EXPECTED_UUID
#error SLLM_TEST_EXPECTED_UUID must name one exact GPU UUID
#endif

namespace {

constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kQueryHeads = 24U;
constexpr uint32_t kHeadDim = 256U;
constexpr uint32_t kM = 128U;
constexpr uint32_t kQuantizationBlock = 32U;
constexpr uint32_t kTokenBlock = 128U;
constexpr uint64_t kChunk = 2048U;
constexpr uint64_t kLogicalBlocksExtra = 8U;
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

struct Pair final {
  sllm_kv_state_t *paged = nullptr;
  sllm_kv_state_t *contiguous = nullptr;
  sllm_buffer_t *key = nullptr;
  sllm_buffer_t *value = nullptr;
  sllm_buffer_t *query = nullptr;
  sllm_buffer_t *paged_output = nullptr;
  sllm_buffer_t *contiguous_output = nullptr;
};

bool expect(const sllm_status_t actual, const sllm_status_t expected,
            const char *const operation, const Error &error) {
  if (actual == expected)
    return true;
  std::cerr << operation << " status=" << actual << " expected=" << expected
            << " message=" << error.message << '\n';
  return false;
}

std::string uuid_text(const hipUUID &uuid) {
  std::string value(uuid.bytes, sizeof(uuid.bytes));
  for (char &character : value) {
    const auto byte = static_cast<unsigned char>(character);
    if (!std::isxdigit(byte))
      return {};
    character = static_cast<char>(std::tolower(byte));
  }
  return "GPU-" + value;
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

bool cleanup(Pair *const pair) {
  bool result = true;
  result = release_state(&pair->contiguous) && result;
  result = release_state(&pair->paged) && result;
  result = release_buffer(&pair->contiguous_output) && result;
  result = release_buffer(&pair->paged_output) && result;
  result = release_buffer(&pair->query) && result;
  result = release_buffer(&pair->value) && result;
  result = release_buffer(&pair->key) && result;
  return result;
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

bool wait_release(sllm_completion_t **const completion,
                  const char *const operation, uint64_t *const elapsed_ns) {
  if (completion == nullptr || *completion == nullptr)
    return false;
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  const bool waited = expect(sllm_completion_wait(*completion, kTimeoutMs,
                                                  &result, &error.sink),
                             SLLM_STATUS_OK, operation, error) &&
                      result.state == SLLM_COMPLETION_STATE_SUCCESS;
  bool timed = true;
  if (elapsed_ns != nullptr) {
    sllm_completion_timing_t timing{};
    timing.struct_size = sizeof(timing);
    timing.abi_version = SLLM_HIP_ABI_VERSION;
    timed = waited &&
            expect(sllm_completion_timing(*completion, &timing, &error.sink),
                   SLLM_STATUS_OK, "completion timing", error) &&
            timing.valid != 0U && timing.elapsed_ns != 0U;
    *elapsed_ns = timing.elapsed_ns;
  }
  const bool released = expect(sllm_completion_release(completion, &error.sink),
                               SLLM_STATUS_OK, "completion release", error) &&
                        *completion == nullptr;
  return timed && released;
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
  return expect(sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                     &error.sink),
                SLLM_STATUS_OK, "upload", error) &&
         wait_release(&completion, "upload wait", nullptr);
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
              SLLM_STATUS_OK, "download", error) ||
      completion == nullptr)
    return false;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(
          sllm_completion_wait(completion, kTimeoutMs, &result, &error.sink),
          SLLM_STATUS_OK, "download wait", error) ||
      result.state != SLLM_COMPLETION_STATE_SUCCESS) {
    (void)sllm_completion_release(&completion, &error.sink);
    return false;
  }
  uint64_t written = 0U;
  const bool read = expect(sllm_completion_read(completion, destination, bytes,
                                                &written, &error.sink),
                           SLLM_STATUS_OK, "download read", error) &&
                    written == bytes;
  const bool released =
      expect(sllm_completion_release(&completion, &error.sink), SLLM_STATUS_OK,
             "download release", error) &&
      completion == nullptr;
  return read && released;
}

sllm_tensor_binding_t binding(const sllm_buffer_t *const buffer,
                              const uint32_t dtype, const uint64_t *const shape,
                              const uint32_t rank) {
  sllm_tensor_binding_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.buffer = buffer;
  result.dtype = dtype;
  result.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  result.rank = rank;
  uint64_t stride = 1U;
  for (uint32_t index = rank; index != 0U; --index) {
    const uint32_t current = index - 1U;
    result.shape[current] = shape[current];
    result.stride_elements[current] = stride;
    stride *= shape[current];
  }
  return result;
}

sllm_kv_state_paged_create_info_t paged_info(const uint64_t length,
                                             const uint64_t session) {
  sllm_kv_state_paged_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.create_info_version = SLLM_HIP_KV_PAGED_CREATE_INFO_VERSION;
  info.session_id = session;
  info.layer_id = static_cast<uint32_t>(session);
  info.capacity_tokens = length;
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
  info.logical_table_capacity = (length + kTokenBlock - 1U) / kTokenBlock;
  info.max_physical_blocks = info.logical_table_capacity + kLogicalBlocksExtra;
  return info;
}

sllm_kv_state_create_info_v2_t contiguous_info(const uint64_t length,
                                               const uint64_t session) {
  sllm_kv_state_create_info_v2_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.create_info_version = SLLM_HIP_KV_STATE_CREATE_INFO_V2_VERSION;
  info.session_id = session;
  info.layer_id = static_cast<uint32_t>(session);
  info.capacity_tokens = length;
  info.head_count = kKvHeads;
  info.head_dim = kHeadDim;
  info.memory_kind = SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS;
  info.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
  info.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
  info.encoding = SLLM_HIP_KV_ENCODING_MXFP8_E4_V1;
  info.block_size = kQuantizationBlock;
  info.scale_dtype = SLLM_TENSOR_DTYPE_U8;
  return info;
}

void fill_chunk(const uint64_t start, const uint32_t count,
                std::vector<uint16_t> *const key,
                std::vector<uint16_t> *const value) {
  key->resize(static_cast<std::size_t>(count) * kKvHeads * kHeadDim);
  value->resize(key->size());
  for (uint32_t token = 0U; token != count; ++token) {
    for (uint32_t head = 0U; head != kKvHeads; ++head) {
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        const uint64_t absolute = start + token;
        const float base = 0.07F +
                           0.00003F * static_cast<float>(absolute % 997U) +
                           0.0007F * static_cast<float>(dimension % 31U) +
                           0.002F * static_cast<float>(head);
        const std::size_t index =
            (static_cast<std::size_t>(token) * kKvHeads + head) * kHeadDim +
            dimension;
        (*key)[index] =
            bf16((absolute + head + dimension) % 19U == 0U ? -base : base);
        (*value)[index] =
            bf16((absolute + head + dimension) % 23U == 0U ? -(base * 1.13F)
                                                           : base * 1.13F);
      }
    }
  }
}

bool append_chunk(const sllm_kv_state_t *const state,
                  const sllm_queue_t *const queue,
                  const sllm_buffer_t *const key,
                  const sllm_buffer_t *const value, const uint64_t start,
                  const uint32_t count, const bool paged) {
  const uint64_t shape[] = {count, kKvHeads, kHeadDim};
  sllm_kv_append_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.append_version = SLLM_HIP_KV_STATE_VERSION;
  descriptor.expected_length = start;
  descriptor.start_position = start;
  descriptor.key_input = binding(key, SLLM_TENSOR_DTYPE_BF16, shape, 3U);
  descriptor.value_input = binding(value, SLLM_TENSOR_DTYPE_BF16, shape, 3U);
  sllm_completion_t *completion = nullptr;
  sllm_kv_append_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
  Error error;
  const bool submitted =
      expect(sllm_kv_state_append(state, queue, &descriptor, &completion, &info,
                                  &error.sink),
             SLLM_STATUS_OK, paged ? "paged append" : "contiguous append",
             error) &&
      completion != nullptr && info.backend == SLLM_BACKEND_HIP &&
      info.fallback_allowed == 0U && info.fallback_used == 0U;
  return submitted && wait_release(&completion, "append completion", nullptr);
}

bool build_pair(const sllm_context_t *const context,
                const sllm_queue_t *const queue, const uint64_t length,
                const uint64_t session, Pair *const pair) {
  Error error;
  const auto paged_create = paged_info(length, session);
  const auto contiguous_create = contiguous_info(length, session);
  bool passed = expect(sllm_kv_state_create_paged(context, &paged_create,
                                                  &pair->paged, &error.sink),
                       SLLM_STATUS_OK, "paged pair create", error) &&
                expect(sllm_kv_state_create_v2(context, &contiguous_create,
                                               &pair->contiguous, &error.sink),
                       SLLM_STATUS_OK, "contiguous pair create", error);
  const uint64_t chunk_bytes = kChunk * kKvHeads * kHeadDim * sizeof(uint16_t);
  const uint64_t query_bytes = kM * kQueryHeads * kHeadDim * sizeof(uint16_t);
  const uint64_t output_bytes = query_bytes;
  passed &= create_buffer(context, chunk_bytes, &pair->key);
  passed &= create_buffer(context, chunk_bytes, &pair->value);
  passed &= create_buffer(context, query_bytes, &pair->query);
  passed &= create_buffer(context, output_bytes, &pair->paged_output);
  passed &= create_buffer(context, output_bytes, &pair->contiguous_output);
  std::vector<uint16_t> key;
  std::vector<uint16_t> value;
  for (uint64_t start = 0U; passed && start < length;) {
    const uint32_t count =
        static_cast<uint32_t>(std::min<uint64_t>(kChunk, length - start));
    fill_chunk(start, count, &key, &value);
    passed &= upload(queue, pair->key, key.data(),
                     static_cast<uint64_t>(key.size()) * sizeof(uint16_t));
    passed &= upload(queue, pair->value, value.data(),
                     static_cast<uint64_t>(value.size()) * sizeof(uint16_t));
    passed &= append_chunk(pair->paged, queue, pair->key, pair->value, start,
                           count, true);
    passed &= append_chunk(pair->contiguous, queue, pair->key, pair->value,
                           start, count, false);
    start += count;
  }
  std::vector<uint16_t> query(kM * kQueryHeads * kHeadDim);
  for (std::size_t index = 0U; index != query.size(); ++index)
    query[index] = bf16(0.01F + 0.00001F * static_cast<float>(index % 997U));
  passed &= upload(queue, pair->query, query.data(), query_bytes);
  return passed;
}

bool run_attention(const sllm_context_t *const context,
                   const sllm_queue_t *const queue, const Pair *const pair,
                   const sllm_kv_state_t *const state,
                   const sllm_buffer_t *const output, const uint64_t length,
                   uint64_t *const elapsed_ns, const char *const operation) {
  const uint64_t shape[] = {kM, kQueryHeads, kHeadDim};
  sllm_causal_attention_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_CAUSAL_ATTENTION_VERSION;
  descriptor.start_position = length - kM;
  descriptor.expected_kv_length = length;
  descriptor.kv_state = state;
  descriptor.query = binding(pair->query, SLLM_TENSOR_DTYPE_BF16, shape, 3U);
  descriptor.output = binding(output, SLLM_TENSOR_DTYPE_BF16, shape, 3U);
  sllm_causal_attention_dispatch_info_t dispatch{};
  dispatch.struct_size = sizeof(dispatch);
  dispatch.abi_version = SLLM_HIP_ABI_VERSION;
  dispatch.info_version = SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  Error error;
  const bool submitted =
      expect(sllm_causal_attention_execute(context, queue, &descriptor,
                                           &completion, &dispatch, &error.sink),
             SLLM_STATUS_OK, operation, error) &&
      completion != nullptr && dispatch.backend == SLLM_BACKEND_HIP &&
      dispatch.fallback_allowed == 0U && dispatch.fallback_used == 0U;
  return submitted && wait_release(&completion, operation, elapsed_ns);
}

double median(std::vector<uint64_t> values) {
  std::sort(values.begin(), values.end());
  return static_cast<double>(values[values.size() / 2U]);
}

bool run_length(const sllm_context_t *const context,
                const sllm_queue_t *const queue, const uint64_t length,
                const uint64_t session) {
  Pair pair;
  if (!build_pair(context, queue, length, session, &pair)) {
    (void)cleanup(&pair);
    return false;
  }
  bool passed = true;
  uint64_t elapsed = 0U;
  passed &= run_attention(context, queue, &pair, pair.paged, pair.paged_output,
                          length, &elapsed, "paged warmup");
  passed &= run_attention(context, queue, &pair, pair.contiguous,
                          pair.contiguous_output, length, &elapsed,
                          "contiguous warmup");
  std::array<std::vector<uint64_t>, 2U> paged_samples;
  std::array<std::vector<uint64_t>, 2U> contiguous_samples;
  for (uint32_t order = 0U; order != 2U && passed; ++order) {
    const bool reverse = order == 1U;
    for (uint32_t repetition = 0U; repetition != 2U && passed; ++repetition) {
      if (!reverse) {
        passed &=
            run_attention(context, queue, &pair, pair.paged, pair.paged_output,
                          length, &elapsed, "paged ABBA timing");
        paged_samples[order].push_back(elapsed);
        passed &= run_attention(context, queue, &pair, pair.contiguous,
                                pair.contiguous_output, length, &elapsed,
                                "contiguous ABBA timing");
        contiguous_samples[order].push_back(elapsed);
      } else {
        passed &= run_attention(context, queue, &pair, pair.contiguous,
                                pair.contiguous_output, length, &elapsed,
                                "contiguous BA timing");
        contiguous_samples[order].push_back(elapsed);
        passed &=
            run_attention(context, queue, &pair, pair.paged, pair.paged_output,
                          length, &elapsed, "paged BA timing");
        paged_samples[order].push_back(elapsed);
      }
    }
  }
  const uint64_t output_bytes =
      static_cast<uint64_t>(kM) * kQueryHeads * kHeadDim * sizeof(uint16_t);
  std::vector<uint16_t> paged_output(output_bytes / sizeof(uint16_t));
  std::vector<uint16_t> contiguous_output(paged_output.size());
  passed &=
      download(queue, pair.paged_output, paged_output.data(), output_bytes) &&
      download(queue, pair.contiguous_output, contiguous_output.data(),
               output_bytes) &&
      paged_output == contiguous_output;
  if (!passed)
    std::cerr << "M128 KV=" << length << " failed bitwise/provider checks\n";
  for (uint32_t order = 0U; order != 2U; ++order) {
    const double paged_ns = median(paged_samples[order]);
    const double contiguous_ns = median(contiguous_samples[order]);
    const double ratio = paged_ns / contiguous_ns;
    std::cout << "abba length=" << length
              << " order=" << (order == 0U ? "AB" : "BA")
              << " paged_ns=" << paged_ns << " contiguous_ns=" << contiguous_ns
              << " ratio=" << ratio << '\n';
    passed &= std::isfinite(ratio) && ratio <= 1.10;
  }
  const bool cleaned = cleanup(&pair);
  std::cout << "phase87 final ABBA KV=" << length
            << " status=" << (passed && cleaned ? "PASS" : "FAIL") << '\n';
  return passed && cleaned;
}

} // namespace

int main() {
  hipUUID uuid{};
  if (hipDeviceGetUuid(&uuid, 0) != hipSuccess ||
      uuid_text(uuid) != SLLM_TEST_EXPECTED_UUID) {
    std::cerr << "unexpected GPU UUID: " << uuid_text(uuid) << '\n';
    return 2;
  }
  Error error;
  sllm_device_info_t device{};
  device.struct_size = sizeof(device);
  device.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(sllm_device_query(0U, &device, &error.sink), SLLM_STATUS_OK,
              "device query", error) ||
      std::string(device.gcn_arch_name) != SLLM_TEST_EXPECTED_TARGET)
    return 2;
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  std::strncpy(context_info.expected_gcn_arch_name, SLLM_TEST_EXPECTED_TARGET,
               sizeof(context_info.expected_gcn_arch_name) - 1U);
  bool passed =
      expect(sllm_context_create(&context_info, &context, &error.sink),
             SLLM_STATUS_OK, "context create", error);
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  if (passed)
    passed &=
        expect(sllm_queue_create(context, &queue_info, &queue, &error.sink),
               SLLM_STATUS_OK, "queue create", error);
  if (passed)
    passed &=
        expect(sllm_queue_set_completion_mode(
                   queue, SLLM_QUEUE_COMPLETION_MODE_PROFILED, &error.sink),
               SLLM_STATUS_OK, "profiled completion mode", error);
  std::cout << "reverse_logical_table=unsupported_public_abi\n";
  if (passed)
    passed = run_length(context, queue, UINT64_C(65536), UINT64_C(0x8765536));
  if (passed)
    passed = run_length(context, queue, UINT64_C(131072), UINT64_C(0x87131072));
  passed = release_queue(&queue) && passed;
  passed = release_context(&context) && passed;
  if (passed)
    std::cout << "phase87_stage10_paged_final_abba_probe: PASS target="
              << SLLM_TEST_EXPECTED_TARGET
              << " M=128 KV=65536,131072 warmup=1 measured=2 "
                 "orders=AB,BA fallback=0 cleanup=0\n";
  return passed ? 0 : 1;
}
