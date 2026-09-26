// Phase 87 Stage 10: production Paged prefill table-order probe.
//
// This is an internal, model-free probe.  It uses the production launchers
// and DevicePool descriptors directly so that identity and reverse logical to
// physical page tables exercise the same M=128, KV=131072 path.
#include "sllm/hip.h"

#include "causal_attention_kernel_internal.hpp"
#include "paged_kv_device_pool.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#error SLLM_TEST_EXPECTED_TARGET must name one exact GPU target
#endif
#ifndef SLLM_TEST_EXPECTED_UUID
#error SLLM_TEST_EXPECTED_UUID must name one exact GPU UUID
#endif

namespace {

using sllm_paged_kv::BlockDescriptor;
using sllm_paged_kv::DevicePool;
using sllm_paged_kv::DeviceStatus;
using sllm_paged_kv::LogicalEntryUpdate;
using sllm_paged_kv::SlabPlan;
using sllm_paged_kv::SlabPlanInput;

constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kQueryHeads = 24U;
constexpr uint32_t kHeadDim = 256U;
constexpr uint32_t kScaleBlock = 32U;
constexpr uint32_t kScalesPerRow = kHeadDim / kScaleBlock;
constexpr uint32_t kPageTokens = 128U;
constexpr uint32_t kQueryCount = 128U;
constexpr uint64_t kKvLength = 131072U;
constexpr uint64_t kStartPosition = kKvLength - kQueryCount;
constexpr uint64_t kPageCount = kKvLength / kPageTokens;
constexpr size_t kOutputElements =
    static_cast<size_t>(kQueryCount) * kQueryHeads * kHeadDim;
constexpr uint32_t kEncoding = SLLM_HIP_KV_ENCODING_MXFP8_E4_V1;

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

void check(const hipError_t status, const char *const operation) {
  if (status != hipSuccess)
    fail(std::string(operation) + ": " + hipGetErrorString(status));
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

uint16_t f32_to_bf16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & 1U) != 0U))
    ++upper;
  return static_cast<uint16_t>(upper);
}

float bf16_to_f32(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint8_t e4m3fn_encode(const float value) {
  const uint8_t sign = std::signbit(value) ? UINT8_C(0x80) : 0U;
  const float magnitude = std::fabs(value);
  if (magnitude == 0.0F)
    return sign;
  if (!std::isfinite(magnitude) || magnitude >= 448.0F)
    return static_cast<uint8_t>(sign | UINT8_C(0x7e));
  uint32_t bits = 0U;
  std::memcpy(&bits, &magnitude, sizeof(bits));
  const uint32_t rounded =
      bits + UINT32_C(0x0007ffff) + ((bits >> 20U) & UINT32_C(1));
  const uint32_t exponent = ((rounded >> 23U) & UINT32_C(0xff)) - 120U;
  const uint32_t code = (exponent << 3U) | ((rounded >> 20U) & UINT32_C(7));
  return static_cast<uint8_t>(
      sign | static_cast<uint8_t>(std::min(code, UINT32_C(0x7e))));
}

double e4m3fn_decode(const uint8_t value) {
  const double sign = (value & UINT8_C(0x80)) != 0U ? -1.0 : 1.0;
  const uint32_t magnitude = value & UINT8_C(0x7f);
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & UINT32_C(7);
  if (exponent == 0U)
    return sign * static_cast<double>(mantissa) * std::ldexp(1.0, -9);
  return sign * std::ldexp(1.0 + static_cast<double>(mantissa) / 8.0,
                           static_cast<int>(exponent) - 7);
}

struct Fixture final {
  std::vector<uint16_t> query;
  std::vector<uint8_t> key;
  std::vector<uint8_t> value;
  std::vector<uint8_t> key_scales;
  std::vector<uint8_t> value_scales;
};

size_t kv_index(const uint64_t token, const uint32_t head,
                const uint32_t dimension) {
  return (static_cast<size_t>(token) * kKvHeads + head) * kHeadDim + dimension;
}

size_t scale_index(const uint64_t token, const uint32_t head,
                   const uint32_t block) {
  return (static_cast<size_t>(token) * kKvHeads + head) * kScalesPerRow + block;
}

Fixture make_fixture() {
  Fixture fixture;
  fixture.query.resize(kOutputElements);
  const size_t kv_elements =
      static_cast<size_t>(kKvLength) * kKvHeads * kHeadDim;
  const size_t scale_elements =
      static_cast<size_t>(kKvLength) * kKvHeads * kScalesPerRow;
  fixture.key.resize(kv_elements);
  fixture.value.resize(kv_elements);
  fixture.key_scales.resize(scale_elements);
  fixture.value_scales.resize(scale_elements);

  for (uint32_t row = 0U; row < kQueryCount; ++row) {
    for (uint32_t head = 0U; head < kQueryHeads; ++head) {
      for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
        float source = 0.026F + 0.0021F * static_cast<float>(row) +
                       0.0017F * static_cast<float>(head % 9U) +
                       0.00031F * static_cast<float>(dimension % 23U);
        if (((row + head * 3U + dimension) % 11U) == 0U)
          source = -source;
        fixture
            .query[(static_cast<size_t>(row) * kQueryHeads + head) * kHeadDim +
                   dimension] = f32_to_bf16(source);
      }
    }
  }

  for (uint64_t token = 0U; token < kKvLength; ++token) {
    for (uint32_t head = 0U; head < kKvHeads; ++head) {
      for (uint32_t block = 0U; block < kScalesPerRow; ++block) {
        const uint8_t key_scale =
            static_cast<uint8_t>(125U + ((token + head + block) & UINT64_C(3)));
        const uint8_t value_scale = static_cast<uint8_t>(
            126U + ((2U * token + head + block) & UINT64_C(3)));
        fixture.key_scales[scale_index(token, head, block)] = key_scale;
        fixture.value_scales[scale_index(token, head, block)] = value_scale;
        const float key_scale_value =
            std::ldexp(1.0F, static_cast<int>(key_scale) - 127);
        const float value_scale_value =
            std::ldexp(1.0F, static_cast<int>(value_scale) - 127);
        for (uint32_t lane = 0U; lane < kScaleBlock; ++lane) {
          const uint32_t dimension = block * kScaleBlock + lane;
          float key_source = 0.19F + 0.013F * static_cast<float>(block) +
                             0.007F * static_cast<float>(head) +
                             0.0007F * static_cast<float>(token % 17U) +
                             0.0011F * static_cast<float>(lane % 19U);
          float value_source = 0.31F + 0.021F * static_cast<float>(block) +
                               0.009F * static_cast<float>(head) +
                               0.0009F * static_cast<float>(token % 13U) +
                               0.0013F * static_cast<float>(lane % 17U);
          if (((token + head + dimension) % 13U) == 0U)
            key_source = -key_source;
          if (((2U * token + head + dimension) % 17U) == 0U)
            value_source = -value_source;
          fixture.key[kv_index(token, head, dimension)] =
              e4m3fn_encode(key_source / key_scale_value);
          fixture.value[kv_index(token, head, dimension)] =
              e4m3fn_encode(value_source / value_scale_value);
        }
      }
    }
  }
  return fixture;
}

bool cleanup_ok = true;

template <class T> struct DeviceBuffer final {
  T *data = nullptr;
  size_t count = 0U;

  explicit DeviceBuffer(const size_t elements) : count(elements) {
    check(hipMalloc(reinterpret_cast<void **>(&data), count * sizeof(T)),
          "hipMalloc");
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer() {
    if (data != nullptr && hipFree(data) != hipSuccess)
      cleanup_ok = false;
  }

  void upload(const void *const source, const size_t bytes,
              const hipStream_t stream) {
    if (bytes != count * sizeof(T))
      fail("device upload size mismatch");
    check(hipMemcpyAsync(data, source, bytes, hipMemcpyHostToDevice, stream),
          "hipMemcpyAsync upload");
  }
};

struct Stream final {
  hipStream_t value = nullptr;
  Stream() { check(hipStreamCreate(&value), "hipStreamCreate"); }
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;
  ~Stream() {
    if (value != nullptr && hipStreamDestroy(value) != hipSuccess)
      cleanup_ok = false;
  }
};

struct Variant final {
  std::unique_ptr<DevicePool> pool;
  std::unique_ptr<DevicePool::LogicalTable> table;
  DeviceBuffer<uint16_t> output;
  DeviceBuffer<uint32_t> status;
  std::vector<uint32_t> mapping;

  explicit Variant(const size_t output_elements)
      : output(output_elements), status(1U) {}
};

std::vector<uint32_t> make_mapping(const bool reverse) {
  std::vector<uint32_t> result(static_cast<size_t>(kPageCount));
  for (uint64_t logical = 0U; logical < kPageCount; ++logical)
    result[static_cast<size_t>(logical)] =
        static_cast<uint32_t>(reverse ? kPageCount - logical - 1U : logical);
  return result;
}

SlabPlan make_plan() {
  return SlabPlan::make(
      SlabPlanInput{static_cast<uint64_t>(kKvHeads) * kHeadDim,
                    static_cast<uint64_t>(kKvHeads) * kScalesPerRow, 0U,
                    kKvLength, kPageCount});
}

void upload_variant(Variant &variant, const Fixture &fixture,
                    const hipStream_t stream) {
  variant.pool = std::make_unique<DevicePool>(make_plan());
  variant.table = variant.pool->make_logical_table(kPageCount, stream);
  if (variant.mapping.size() != static_cast<size_t>(kPageCount))
    fail("invalid physical mapping");

  const std::vector<uint32_t> physical_ids = variant.mapping;
  if (variant.pool->ensure_blocks(physical_ids, stream) != DeviceStatus::Ok)
    fail("DevicePool ensure_blocks failed");

  constexpr size_t page_kv_bytes =
      static_cast<size_t>(kPageTokens) * kKvHeads * kHeadDim;
  constexpr size_t page_scale_bytes =
      static_cast<size_t>(kPageTokens) * kKvHeads * kScalesPerRow;
  for (uint64_t logical = 0U; logical < kPageCount; ++logical) {
    const uint32_t physical = variant.mapping[static_cast<size_t>(logical)];
    const BlockDescriptor &descriptor = variant.pool->host_descriptor(physical);
    const size_t token_offset = static_cast<size_t>(logical) * kPageTokens;
    check(
        hipMemcpyAsync(descriptor.key,
                       fixture.key.data() + token_offset * kKvHeads * kHeadDim,
                       page_kv_bytes, hipMemcpyHostToDevice, stream),
        "upload page key");
    check(hipMemcpyAsync(descriptor.value,
                         fixture.value.data() +
                             token_offset * kKvHeads * kHeadDim,
                         page_kv_bytes, hipMemcpyHostToDevice, stream),
          "upload page value");
    check(hipMemcpyAsync(descriptor.key_scale,
                         fixture.key_scales.data() +
                             token_offset * kKvHeads * kScalesPerRow,
                         page_scale_bytes, hipMemcpyHostToDevice, stream),
          "upload page key scale");
    check(hipMemcpyAsync(descriptor.value_scale,
                         fixture.value_scales.data() +
                             token_offset * kKvHeads * kScalesPerRow,
                         page_scale_bytes, hipMemcpyHostToDevice, stream),
          "upload page value scale");
  }

  std::vector<LogicalEntryUpdate> updates;
  updates.reserve(static_cast<size_t>(kPageCount));
  for (uint64_t logical = 0U; logical < kPageCount; ++logical)
    updates.push_back(LogicalEntryUpdate{
        static_cast<uint32_t>(logical), sllm_paged_kv::kInvalidBlock,
        variant.mapping[static_cast<size_t>(logical)]});
  if (variant.table->update_entries(updates, stream) != DeviceStatus::Ok)
    fail("LogicalTable update_entries failed");
  check(hipStreamSynchronize(stream), "paged fixture upload synchronize");
  if (variant.table->poll_pending() != DeviceStatus::Ok)
    fail("LogicalTable update did not complete");
}

double decoded(const Fixture &fixture, const uint64_t token,
               const uint32_t head, const uint32_t dimension,
               const bool value) {
  const uint32_t block = dimension / kScaleBlock;
  const uint8_t scale =
      value ? fixture.value_scales[scale_index(token, head, block)]
            : fixture.key_scales[scale_index(token, head, block)];
  const uint8_t encoded = value
                              ? fixture.value[kv_index(token, head, dimension)]
                              : fixture.key[kv_index(token, head, dimension)];
  return e4m3fn_decode(encoded) *
         std::ldexp(1.0, static_cast<int>(scale) - 127);
}

double oracle_max_abs(const Fixture &fixture,
                      const std::vector<uint16_t> &output) {
  const std::array<std::pair<uint32_t, uint32_t>, 3> samples = {
      std::pair<uint32_t, uint32_t>{0U, 0U},
      {kQueryCount / 2U, 11U},
      {kQueryCount - 1U, kQueryHeads - 1U}};
  constexpr std::array<uint32_t, 6> dimensions = {0U,   31U,  32U,
                                                  127U, 128U, 255U};
  double maximum_error = 0.0;
  for (const auto &[row, query_head] : samples) {
    const uint32_t kv_head = query_head / (kQueryHeads / kKvHeads);
    const uint64_t visible = kStartPosition + row + 1U;
    std::vector<double> scores(static_cast<size_t>(visible));
    double maximum = -std::numeric_limits<double>::infinity();
    for (uint64_t token = 0U; token < visible; ++token) {
      double score = 0.0;
      for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
        const size_t query_index =
            (static_cast<size_t>(row) * kQueryHeads + query_head) * kHeadDim +
            dimension;
        score += static_cast<double>(bf16_to_f32(fixture.query[query_index])) *
                 decoded(fixture, token, kv_head, dimension, false);
      }
      score /= 16.0;
      scores[static_cast<size_t>(token)] = score;
      maximum = std::max(maximum, score);
    }
    double denominator = 0.0;
    for (double &score : scores) {
      score = std::exp(score - maximum);
      denominator += score;
    }
    for (const uint32_t dimension : dimensions) {
      double numerator = 0.0;
      for (uint64_t token = 0U; token < visible; ++token) {
        numerator += scores[static_cast<size_t>(token)] *
                     decoded(fixture, token, kv_head, dimension, true);
      }
      const size_t output_index =
          (static_cast<size_t>(row) * kQueryHeads + query_head) * kHeadDim +
          dimension;
      const double actual =
          static_cast<double>(bf16_to_f32(output[output_index]));
      maximum_error =
          std::max(maximum_error, std::abs(actual - numerator / denominator));
    }
  }
  return maximum_error;
}

void upload_contiguous(DeviceBuffer<uint16_t> &query,
                       DeviceBuffer<uint8_t> &key, DeviceBuffer<uint8_t> &value,
                       DeviceBuffer<uint8_t> &key_scales,
                       DeviceBuffer<uint8_t> &value_scales,
                       const Fixture &fixture, const hipStream_t stream) {
  query.upload(fixture.query.data(), fixture.query.size() * sizeof(uint16_t),
               stream);
  key.upload(fixture.key.data(), fixture.key.size(), stream);
  value.upload(fixture.value.data(), fixture.value.size(), stream);
  key_scales.upload(fixture.key_scales.data(), fixture.key_scales.size(),
                    stream);
  value_scales.upload(fixture.value_scales.data(), fixture.value_scales.size(),
                      stream);
  check(hipStreamSynchronize(stream), "contiguous fixture upload synchronize");
}

hipError_t launch_contiguous(const DeviceBuffer<uint16_t> &query,
                             const DeviceBuffer<uint8_t> &key,
                             const DeviceBuffer<uint8_t> &value,
                             const DeviceBuffer<uint8_t> &key_scales,
                             const DeviceBuffer<uint8_t> &value_scales,
                             DeviceBuffer<uint16_t> &output,
                             const hipStream_t stream) {
  return sllm_causal_attention_kernel::launch_gqa6_qtile8_w16(
      query.data, key.data, value.data, key_scales.data, value_scales.data,
      nullptr, nullptr, output.data, kQueryCount, kStartPosition, kQueryHeads,
      kKvHeads, kHeadDim, kEncoding, 1.0F, 1.0F, false, stream);
}

hipError_t launch_paged(const DeviceBuffer<uint16_t> &query, Variant &variant,
                        const hipStream_t stream) {
  return sllm_causal_attention_kernel::launch_paged_prefill_gqa6(
      query.data, variant.table->device_table(),
      variant.pool->device_descriptor_table(),
      static_cast<uint32_t>(kPageCount), static_cast<uint32_t>(kPageCount),
      variant.status.data, variant.output.data, kQueryCount, kStartPosition,
      kKvLength, kQueryHeads, kKvHeads, kHeadDim, kEncoding, 1.0F, 1.0F, false,
      false, false, stream);
}

void launch_checked(const hipError_t status, const char *const operation) {
  check(status, operation);
}

double event_timed(const bool paged, Variant *const variant,
                   const DeviceBuffer<uint16_t> &query,
                   const DeviceBuffer<uint8_t> &key,
                   const DeviceBuffer<uint8_t> &value,
                   const DeviceBuffer<uint8_t> &key_scales,
                   const DeviceBuffer<uint8_t> &value_scales,
                   DeviceBuffer<uint16_t> &output, const hipStream_t stream) {
  hipEvent_t begin = nullptr;
  hipEvent_t end = nullptr;
  check(hipEventCreate(&begin), "hipEventCreate begin");
  check(hipEventCreate(&end), "hipEventCreate end");
  check(hipEventRecord(begin, stream), "hipEventRecord begin");
  if (paged)
    launch_checked(launch_paged(query, *variant, stream), "paged launch");
  else
    launch_checked(launch_contiguous(query, key, value, key_scales,
                                     value_scales, output, stream),
                   "contiguous launch");
  check(hipEventRecord(end, stream), "hipEventRecord end");
  check(hipEventSynchronize(end), "hipEventSynchronize end");
  float milliseconds = 0.0F;
  check(hipEventElapsedTime(&milliseconds, begin, end), "hipEventElapsedTime");
  check(hipEventDestroy(begin), "hipEventDestroy begin");
  check(hipEventDestroy(end), "hipEventDestroy end");
  return static_cast<double>(milliseconds) * 1.0e6;
}

double median(std::vector<double> values) {
  if (values.empty())
    fail("empty timing series");
  std::sort(values.begin(), values.end());
  return (values.front() + values.back()) * 0.5;
}

struct OrderTiming final {
  std::vector<double> candidate;
  std::vector<double> control;
};

struct Timing final {
  OrderTiming ab;
  OrderTiming ba;
};

void warmup(const DeviceBuffer<uint16_t> &query,
            const DeviceBuffer<uint8_t> &key,
            const DeviceBuffer<uint8_t> &value,
            const DeviceBuffer<uint8_t> &key_scales,
            const DeviceBuffer<uint8_t> &value_scales,
            DeviceBuffer<uint16_t> &output, Variant &sequential,
            Variant &reverse, const hipStream_t stream) {
  launch_checked(launch_contiguous(query, key, value, key_scales, value_scales,
                                   output, stream),
                 "contiguous warmup");
  launch_checked(launch_paged(query, sequential, stream), "sequential warmup");
  launch_checked(launch_paged(query, reverse, stream), "reverse warmup");
  check(hipStreamSynchronize(stream), "warmup synchronize");
}

void measure_order(const bool sequential_first, const uint32_t samples,
                   const DeviceBuffer<uint16_t> &query,
                   const DeviceBuffer<uint8_t> &key,
                   const DeviceBuffer<uint8_t> &value,
                   const DeviceBuffer<uint8_t> &key_scales,
                   const DeviceBuffer<uint8_t> &value_scales,
                   DeviceBuffer<uint16_t> &output, Variant &variant,
                   OrderTiming &timing, const hipStream_t stream) {
  for (uint32_t sample = 0U; sample < samples; ++sample) {
    if (sequential_first) {
      timing.candidate.push_back(event_timed(true, &variant, query, key, value,
                                             key_scales, value_scales, output,
                                             stream));
      timing.control.push_back(event_timed(false, nullptr, query, key, value,
                                           key_scales, value_scales, output,
                                           stream));
    } else {
      timing.control.push_back(event_timed(false, nullptr, query, key, value,
                                           key_scales, value_scales, output,
                                           stream));
      timing.candidate.push_back(event_timed(true, &variant, query, key, value,
                                             key_scales, value_scales, output,
                                             stream));
    }
  }
}

uint32_t read_status(const Variant &variant) {
  uint32_t status = UINT32_MAX;
  check(hipMemcpy(&status, variant.status.data, sizeof(status),
                  hipMemcpyDeviceToHost),
        "read paged status");
  return status;
}

void clear_status(const Variant &variant) {
  check(hipMemset(variant.status.data, 0, sizeof(uint32_t)),
        "clear paged status");
}

void print_timing(const char *const mapping, const char *const order,
                  const std::vector<double> &candidate,
                  const std::vector<double> &control) {
  const double candidate_median = median(candidate);
  const double control_median = median(control);
  std::cout << "reverse_table mapping=" << mapping << " order=" << order
            << " paged_ns=" << candidate_median
            << " contiguous_ns=" << control_median
            << " ratio=" << candidate_median / control_median << '\n';
}

bool ratio_ok(const std::vector<double> &candidate,
              const std::vector<double> &control) {
  const double ratio = median(candidate) / median(control);
  return std::isfinite(ratio) && ratio <= 1.10;
}

int run() {
  int count = 0;
  check(hipGetDeviceCount(&count), "hipGetDeviceCount");
  if (count != 1)
    fail("probe requires exactly one visible GPU");
  hipUUID uuid{};
  check(hipDeviceGetUuid(&uuid, 0), "hipDeviceGetUuid");
  if (uuid_text(uuid) != SLLM_TEST_EXPECTED_UUID)
    fail("unexpected GPU UUID: " + uuid_text(uuid));
  hipDeviceProp_t properties{};
  check(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties");
  const std::string target(properties.gcnArchName);
  if (target != SLLM_TEST_EXPECTED_TARGET &&
      target.rfind(std::string(SLLM_TEST_EXPECTED_TARGET) + ":", 0U) != 0U)
    fail("unexpected GPU target: " + target);

  Stream stream;
  Fixture fixture = make_fixture();
  DeviceBuffer<uint16_t> query(kOutputElements);
  DeviceBuffer<uint8_t> key(fixture.key.size());
  DeviceBuffer<uint8_t> value(fixture.value.size());
  DeviceBuffer<uint8_t> key_scales(fixture.key_scales.size());
  DeviceBuffer<uint8_t> value_scales(fixture.value_scales.size());
  DeviceBuffer<uint16_t> output(kOutputElements);
  upload_contiguous(query, key, value, key_scales, value_scales, fixture,
                    stream.value);

  Variant sequential(kOutputElements);
  sequential.mapping = make_mapping(false);
  upload_variant(sequential, fixture, stream.value);
  Variant reverse(kOutputElements);
  reverse.mapping = make_mapping(true);
  upload_variant(reverse, fixture, stream.value);

  warmup(query, key, value, key_scales, value_scales, output, sequential,
         reverse, stream.value);
  Timing sequential_timing;
  Timing reverse_timing;
  measure_order(true, 2U, query, key, value, key_scales, value_scales, output,
                sequential, sequential_timing.ab, stream.value);
  measure_order(false, 2U, query, key, value, key_scales, value_scales, output,
                sequential, sequential_timing.ba, stream.value);
  // The reverse table is measured separately with the same AB/BA event order.
  Timing reverse_mapping_timing;
  measure_order(true, 2U, query, key, value, key_scales, value_scales, output,
                reverse, reverse_mapping_timing.ab, stream.value);
  measure_order(false, 2U, query, key, value, key_scales, value_scales, output,
                reverse, reverse_mapping_timing.ba, stream.value);

  clear_status(sequential);
  clear_status(reverse);
  launch_checked(launch_contiguous(query, key, value, key_scales, value_scales,
                                   output, stream.value),
                 "contiguous numerical launch");
  launch_checked(launch_paged(query, sequential, stream.value),
                 "sequential numerical launch");
  launch_checked(launch_paged(query, reverse, stream.value),
                 "reverse numerical launch");
  check(hipStreamSynchronize(stream.value), "numerical synchronize");
  const std::vector<uint16_t> contiguous = [&]() {
    std::vector<uint16_t> values(kOutputElements);
    check(hipMemcpy(values.data(), output.data,
                    values.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
          "download contiguous output");
    return values;
  }();
  const std::vector<uint16_t> sequential_output = [&]() {
    std::vector<uint16_t> values(kOutputElements);
    check(hipMemcpy(values.data(), sequential.output.data,
                    values.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
          "download sequential output");
    return values;
  }();
  const std::vector<uint16_t> reverse_output = [&]() {
    std::vector<uint16_t> values(kOutputElements);
    check(hipMemcpy(values.data(), reverse.output.data,
                    values.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
          "download reverse output");
    return values;
  }();
  const bool sequential_bitwise = contiguous == sequential_output;
  const bool reverse_bitwise = contiguous == reverse_output;
  const bool mappings_bitwise = sequential_output == reverse_output;
  const double oracle = oracle_max_abs(fixture, contiguous);
  const bool status_ok =
      read_status(sequential) == 0U && read_status(reverse) == 0U;

  print_timing("sequential", "AB", sequential_timing.ab.candidate,
               sequential_timing.ab.control);
  print_timing("sequential", "BA", sequential_timing.ba.candidate,
               sequential_timing.ba.control);
  print_timing("reverse", "AB", reverse_mapping_timing.ab.candidate,
               reverse_mapping_timing.ab.control);
  print_timing("reverse", "BA", reverse_mapping_timing.ba.candidate,
               reverse_mapping_timing.ba.control);
  const bool performance_ok =
      ratio_ok(sequential_timing.ab.candidate, sequential_timing.ab.control) &&
      ratio_ok(sequential_timing.ba.candidate, sequential_timing.ba.control) &&
      ratio_ok(reverse_mapping_timing.ab.candidate,
               reverse_mapping_timing.ab.control) &&
      ratio_ok(reverse_mapping_timing.ba.candidate,
               reverse_mapping_timing.ba.control);
  const bool numeric_ok = sequential_bitwise && reverse_bitwise &&
                          mappings_bitwise && status_ok &&
                          std::isfinite(oracle) && oracle <= 0.03125;
  const bool passed = numeric_ok && performance_ok && cleanup_ok;
  std::cout << "reverse_table target=" << SLLM_TEST_EXPECTED_TARGET
            << " uuid=" << SLLM_TEST_EXPECTED_UUID << " kv_length=" << kKvLength
            << " m=" << kQueryCount << " pages=" << kPageCount
            << " sequential_identity=1 reverse_table=pages-1-logical"
            << " sequential_bitwise=" << (sequential_bitwise ? 1 : 0)
            << " reverse_bitwise=" << (reverse_bitwise ? 1 : 0)
            << " mappings_bitwise=" << (mappings_bitwise ? 1 : 0)
            << " oracle_max_abs=" << oracle
            << " status=" << (status_ok ? 0U : 1U)
            << " performance_below_10pct=" << (performance_ok ? 1 : 0)
            << " fallback=0 cleanup=" << (cleanup_ok ? 0 : 1)
            << " state=" << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}

} // namespace

int main() {
  try {
    return run();
  } catch (const std::exception &error) {
    std::cerr << "phase87_stage10_paged_reverse_table_probe: " << error.what()
              << '\n';
    return 2;
  }
}
