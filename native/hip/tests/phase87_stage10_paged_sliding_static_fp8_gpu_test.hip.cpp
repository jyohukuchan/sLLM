#include "causal_attention_kernel_internal.hpp"
#include "kv_state_kernel_internal.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#error SLLM_TEST_EXPECTED_TARGET must name one exact GPU target
#endif

namespace {

constexpr uint32_t kHeadCount = 1U;
constexpr uint32_t kHeadDim = 4U;
constexpr uint32_t kDescriptorCount = 9U;
constexpr uint32_t kRingSlots = 9U;
constexpr uint64_t kInvalidTag = std::numeric_limits<uint64_t>::max();

void check(const hipError_t status, const char *const operation) {
  if (status != hipSuccess) {
    std::cerr << operation << ": " << hipGetErrorString(status) << '\n';
    std::exit(1);
  }
}

uint16_t f32_to_bf16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & 1U) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

float bf16_to_f32(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

float source_value(const uint64_t token) {
  // Powers of two are exactly representable in E4M3FN with a unit static
  // scale.  That leaves the oracle sensitive to stale ring slots while
  // avoiding a second implementation of the FP8 encoder.
  return static_cast<float>(1U << (token & 3U));
}

float oracle_average(const uint64_t retained_start, const uint64_t position) {
  double sum = 0.0;
  uint64_t count = 0U;
  for (uint64_t token = retained_start; token <= position; ++token) {
    sum += source_value(token);
    ++count;
  }
  return static_cast<float>(sum / static_cast<double>(count));
}

void copy_h2d(void *const destination, const void *const source,
              const size_t bytes) {
  check(hipMemcpy(destination, source, bytes, hipMemcpyHostToDevice),
        "host to device copy");
}

void copy_d2h(void *const destination, const void *const source,
              const size_t bytes) {
  check(hipMemcpy(destination, source, bytes, hipMemcpyDeviceToHost),
        "device to host copy");
}

struct DeviceState final {
  std::array<sllm_paged_kv::BlockDescriptor, kDescriptorCount>
      host_descriptors{};
  std::array<uint8_t *, kDescriptorCount> keys{};
  std::array<uint8_t *, kDescriptorCount> values{};
  uint32_t *ring_table = nullptr;
  uint64_t *ring_tags = nullptr;
  sllm_paged_kv::BlockDescriptor *descriptors = nullptr;
  uint32_t *status = nullptr;
  uint16_t *query = nullptr;
  uint16_t *output = nullptr;

  void allocate() {
    constexpr size_t bytes = 128U * kHeadCount * kHeadDim;
    for (uint32_t block = 0U; block < kDescriptorCount; ++block) {
      check(hipMalloc(reinterpret_cast<void **>(&keys[block]), bytes),
            "key block allocation");
      check(hipMalloc(reinterpret_cast<void **>(&values[block]), bytes),
            "value block allocation");
      host_descriptors[block] = {keys[block], values[block], nullptr,
                                 nullptr,     nullptr,       nullptr};
    }
    check(hipMalloc(reinterpret_cast<void **>(&ring_table),
                    kRingSlots * sizeof(uint32_t)),
          "ring table allocation");
    check(hipMalloc(reinterpret_cast<void **>(&ring_tags),
                    kRingSlots * sizeof(uint64_t)),
          "ring tags allocation");
    check(hipMalloc(reinterpret_cast<void **>(&descriptors),
                    sizeof(host_descriptors)),
          "descriptor allocation");
    check(hipMalloc(reinterpret_cast<void **>(&status), sizeof(uint32_t)),
          "status allocation");
    check(hipMalloc(reinterpret_cast<void **>(&query),
                    kHeadCount * kHeadDim * sizeof(uint16_t)),
          "query allocation");
    check(hipMalloc(reinterpret_cast<void **>(&output),
                    kHeadCount * kHeadDim * sizeof(uint16_t)),
          "output allocation");
    copy_h2d(descriptors, host_descriptors.data(), sizeof(host_descriptors));
  }

  ~DeviceState() {
    for (uint32_t block = 0U; block < kDescriptorCount; ++block) {
      if (keys[block] != nullptr)
        (void)hipFree(keys[block]);
      if (values[block] != nullptr)
        (void)hipFree(values[block]);
    }
    if (ring_table != nullptr)
      (void)hipFree(ring_table);
    if (ring_tags != nullptr)
      (void)hipFree(ring_tags);
    if (descriptors != nullptr)
      (void)hipFree(descriptors);
    if (status != nullptr)
      (void)hipFree(status);
    if (query != nullptr)
      (void)hipFree(query);
    if (output != nullptr)
      (void)hipFree(output);
  }
};

void upload_ring(const DeviceState &device,
                 const std::array<uint32_t, kRingSlots> &table,
                 const std::array<uint64_t, kRingSlots> &tags) {
  copy_h2d(device.ring_table, table.data(), sizeof(table));
  copy_h2d(device.ring_tags, tags.data(), sizeof(tags));
}

void append_tokens(DeviceState &device, const uint64_t start,
                   const uint32_t count, const uint64_t retained_start,
                   const std::array<uint32_t, kRingSlots> &table,
                   const std::array<uint64_t, kRingSlots> &tags,
                   hipStream_t stream) {
  std::vector<uint16_t> key_input(static_cast<size_t>(count) * kHeadDim);
  std::vector<uint16_t> value_input(static_cast<size_t>(count) * kHeadDim);
  for (uint32_t row = 0U; row < count; ++row) {
    const float value = source_value(start + row);
    for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
      // Key zero gives every key the same score; value remains token specific.
      key_input[static_cast<size_t>(row) * kHeadDim + dimension] =
          f32_to_bf16(0.0F);
      value_input[static_cast<size_t>(row) * kHeadDim + dimension] =
          f32_to_bf16(value);
    }
  }
  uint16_t *key_device = nullptr;
  uint16_t *value_device = nullptr;
  check(hipMalloc(reinterpret_cast<void **>(&key_device),
                  key_input.size() * sizeof(uint16_t)),
        "append key input allocation");
  check(hipMalloc(reinterpret_cast<void **>(&value_device),
                  value_input.size() * sizeof(uint16_t)),
        "append value input allocation");
  copy_h2d(key_device, key_input.data(), key_input.size() * sizeof(uint16_t));
  copy_h2d(value_device, value_input.data(),
           value_input.size() * sizeof(uint16_t));
  upload_ring(device, table, tags);
  check(sllm_kv_state_kernel::launch_paged_sliding_static_fp8(
            key_device, value_device, device.descriptors, device.ring_table,
            device.ring_tags, kRingSlots, kDescriptorCount, count,
            retained_start, start, kHeadCount, kHeadDim, 1.0F, 1.0F,
            device.status, stream),
        "sliding static FP8 append launch");
  check(hipStreamSynchronize(stream), "sliding append synchronize");
  uint32_t status = 0U;
  copy_d2h(&status, device.status, sizeof(status));
  if (status != 0U) {
    std::cerr << "sliding append device status=" << status << '\n';
    std::exit(1);
  }
  check(hipFree(key_device), "append key input release");
  check(hipFree(value_device), "append value input release");
}

void check_attention(DeviceState &device, const uint64_t position,
                     const uint64_t retained_start,
                     const std::array<uint32_t, kRingSlots> &table,
                     const std::array<uint64_t, kRingSlots> &tags,
                     hipStream_t stream) {
  const std::array<uint16_t, kHeadDim> query_host{};
  upload_ring(device, table, tags);
  copy_h2d(device.query, query_host.data(), sizeof(query_host));
  check(sllm_causal_attention_kernel::launch_paged_sliding_static_fp8_attention(
            device.query, device.ring_table, device.ring_tags,
            device.descriptors, kRingSlots, kDescriptorCount, device.status,
            device.output, 1U, position, position + 1U, retained_start,
            kHeadCount, kHeadCount, kHeadDim, 1.0F, 1.0F,
            1.0F / std::sqrt(static_cast<float>(kHeadDim)), stream),
        "sliding static FP8 attention launch");
  check(hipStreamSynchronize(stream), "sliding attention synchronize");
  uint32_t status = 0U;
  copy_d2h(&status, device.status, sizeof(status));
  if (status != 0U) {
    std::cerr << "sliding attention device status=" << status << '\n';
    std::exit(1);
  }
  std::array<uint16_t, kHeadDim> output_host{};
  copy_d2h(output_host.data(), device.output, sizeof(output_host));
  const float expected = oracle_average(retained_start, position);
  for (const uint16_t value : output_host) {
    if (std::fabs(bf16_to_f32(value) - expected) > 0.08F) {
      std::cerr << "sliding attention oracle mismatch at position " << position
                << ": got=" << bf16_to_f32(value) << " expected=" << expected
                << '\n';
      std::exit(1);
    }
  }
}

} // namespace

int main() {
  hipDeviceProp_t properties{};
  check(hipGetDeviceProperties(&properties, 0), "device query");
  if (std::string(properties.gcnArchName).find(SLLM_TEST_EXPECTED_TARGET) ==
      std::string::npos) {
    std::cerr << "unexpected target " << properties.gcnArchName << '\n';
    return 1;
  }
  check(hipSetDevice(0), "device select");
  hipStream_t stream = nullptr;
  check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
        "stream create");

  DeviceState device;
  device.allocate();
  std::array<uint32_t, kRingSlots> table{};
  std::array<uint64_t, kRingSlots> tags{};
  tags.fill(kInvalidTag);
  for (uint32_t slot = 0U; slot < kRingSlots; ++slot)
    table[slot] = slot;
  for (uint32_t slot = 0U; slot < 8U; ++slot)
    tags[slot] = slot;

  append_tokens(device, 0U, 1023U, 0U, table, tags, stream);
  check_attention(device, 1022U, 0U, table, tags, stream);
  append_tokens(device, 1023U, 1U, 0U, table, tags, stream);
  check_attention(device, 1023U, 0U, table, tags, stream);

  tags[8] = 8U;
  append_tokens(device, 1024U, 1U, 1U, table, tags, stream);
  check_attention(device, 1024U, 1U, table, tags, stream);
  append_tokens(device, 1025U, 126U, 127U, table, tags, stream);
  check_attention(device, 1150U, 127U, table, tags, stream);
  append_tokens(device, 1151U, 1U, 128U, table, tags, stream);
  tags[0] = kInvalidTag;
  check_attention(device, 1151U, 128U, table, tags, stream);

  // Recycle physical block zero into absolute block nine and verify that the
  // tag, rather than slot zero alone, controls the final attention lookup.
  tags[0] = 9U;
  append_tokens(device, 1152U, 1U, 129U, table, tags, stream);
  check_attention(device, 1152U, 129U, table, tags, stream);

  tags[0] = 0U;
  upload_ring(device, table, tags);
  std::array<uint16_t, kHeadDim> query_host{};
  copy_h2d(device.query, query_host.data(), sizeof(query_host));
  check(sllm_causal_attention_kernel::launch_paged_sliding_static_fp8_attention(
            device.query, device.ring_table, device.ring_tags,
            device.descriptors, kRingSlots, kDescriptorCount, device.status,
            device.output, 1U, 1152U, 1153U, 129U, kHeadCount, kHeadCount,
            kHeadDim, 1.0F, 1.0F, 1.0F, stream),
        "stale-tag attention launch");
  check(hipStreamSynchronize(stream), "stale-tag synchronize");
  uint32_t status = 0U;
  copy_d2h(&status, device.status, sizeof(status));
  if (status == 0U) {
    std::cerr << "stale ring tag was accepted\n";
    return 1;
  }

  check(hipStreamDestroy(stream), "stream destroy");
  std::cout << "phase87_stage10_paged_sliding_static_fp8_gpu_test: PASS target="
            << properties.gcnArchName
            << " boundaries=1023,1024,1025,1151,1152,1153"
            << " ring_slots=9 window=1024 stale_tag=1 fallback=0\n";
  return 0;
}
