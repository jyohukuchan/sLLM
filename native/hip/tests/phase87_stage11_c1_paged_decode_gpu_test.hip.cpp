// Phase 87 Stage11 C1: query-row-sharing candidate for Paged MXFP8-E4 GQA6.
//
// This is an opt-in, exact-target test for the candidate entry point.  The
// production selector is intentionally not changed here.  The baseline is
// the current Paged decode provider; the independent check decodes the host
// fixture and uses FP64 for the reference softmax/value calculation.
#include "../src/causal_attention_kernel_internal.hpp"
#include "../src/paged_kv_device_layout.hpp"
#include "sllm/hip.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#error SLLM_TEST_EXPECTED_TARGET must name one exact GPU target
#endif
#ifndef SLLM_TEST_EXPECTED_UUID
#error SLLM_TEST_EXPECTED_UUID must name one exact GPU UUID
#endif

namespace {
constexpr uint32_t kQHeads = 24U;
constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kHeadDim = 256U;
constexpr uint32_t kPageTokens = 128U;
constexpr uint32_t kScaleBytes = 8U;
constexpr uint32_t kEncoding = SLLM_HIP_KV_ENCODING_MXFP8_E4_V1;
constexpr uint64_t kMaxTokens = 65537U;
constexpr uint32_t kMaxRows = 3U;

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

void check(const hipError_t status, const char *const operation) {
  if (status != hipSuccess) {
    fail(std::string(operation) + ": " + hipGetErrorString(status));
  }
}

template <typename T> struct DeviceBuffer final {
  T *ptr = nullptr;
  size_t count = 0U;

  DeviceBuffer() = default;
  explicit DeviceBuffer(const size_t elements) : count(elements) {
    check(hipMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T)),
          "hipMalloc");
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer() {
    if (ptr != nullptr) {
      (void)hipFree(ptr);
    }
  }
};

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

double e4m3fn_decode(const uint8_t value) {
  const double sign = (value & 0x80U) != 0U ? -1.0 : 1.0;
  const uint32_t magnitude = value & 0x7fU;
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & 7U;
  if (exponent == 0U) {
    return sign * static_cast<double>(mantissa) * std::ldexp(1.0, -9);
  }
  return sign * std::ldexp(1.0 + static_cast<double>(mantissa) / 8.0,
                           static_cast<int>(exponent) - 7);
}

struct Fixture final {
  const uint32_t pages =
      static_cast<uint32_t>((kMaxTokens + kPageTokens - 1U) / kPageTokens);
  const size_t page_value_bytes =
      static_cast<size_t>(kPageTokens) * kKvHeads * kHeadDim;
  const size_t page_scale_bytes =
      static_cast<size_t>(kPageTokens) * kKvHeads * kScaleBytes;
  std::vector<uint16_t> query;
  std::vector<uint8_t> key;
  std::vector<uint8_t> value;
  std::vector<uint8_t> physical_key;
  std::vector<uint8_t> physical_value;
  std::vector<uint8_t> physical_key_scales;
  std::vector<uint8_t> physical_value_scales;

  Fixture()
      : query(static_cast<size_t>(kMaxRows) * kQHeads * kHeadDim),
        key(static_cast<size_t>(kMaxTokens) * kKvHeads * kHeadDim),
        value(key.size()),
        physical_key(static_cast<size_t>(pages) * page_value_bytes),
        physical_value(physical_key.size()),
        physical_key_scales(static_cast<size_t>(pages) * page_scale_bytes,
                            0x7fU),
        physical_value_scales(physical_key_scales) {
    for (size_t index = 0U; index < query.size(); ++index) {
      const float sample =
          0.03125F * static_cast<float>((index * 17U) % 31U) - 0.45F;
      query[index] = f32_to_bf16(sample);
    }
    for (uint64_t token = 0U; token < kMaxTokens; ++token) {
      const uint32_t logical_page = static_cast<uint32_t>(token / 128U);
      const uint32_t local_token = static_cast<uint32_t>(token % 128U);
      const uint32_t physical_page = pages - 1U - logical_page;
      for (uint32_t head = 0U; head < kKvHeads; ++head) {
        const size_t logical_row =
            (static_cast<size_t>(token) * kKvHeads + head) * kHeadDim;
        const size_t physical_row =
            (static_cast<size_t>(physical_page) * kPageTokens + local_token) *
                kKvHeads * kHeadDim +
            static_cast<size_t>(head) * kHeadDim;
        for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
          const uint8_t key_code = static_cast<uint8_t>(
              0x08U + ((token * 19U + head * 7U + dimension * 3U) % 0x70U));
          const uint8_t value_code = static_cast<uint8_t>(
              0x10U + ((token * 11U + head * 5U + dimension * 13U) % 0x60U));
          key[logical_row + dimension] = key_code;
          value[logical_row + dimension] = value_code;
          physical_key[physical_row + dimension] = key_code;
          physical_value[physical_row + dimension] = value_code;
        }
      }
    }
  }
};

double host_oracle(const Fixture &fixture, const uint64_t position,
                   const uint32_t row, const uint32_t query_head,
                   const uint32_t dimension) {
  const uint32_t kv_head = query_head / 6U;
  const size_t query_row =
      (static_cast<size_t>(row) * kQHeads + query_head) * kHeadDim;
  std::vector<double> scores(static_cast<size_t>(position) + 1U);
  double maximum = -std::numeric_limits<double>::infinity();
  for (uint64_t token = 0U; token <= position; ++token) {
    const size_t key_row =
        (static_cast<size_t>(token) * kKvHeads + kv_head) * kHeadDim;
    double dot = 0.0;
    for (uint32_t index = 0U; index < kHeadDim; ++index) {
      dot +=
          static_cast<double>(bf16_to_f32(fixture.query[query_row + index])) *
          e4m3fn_decode(fixture.key[key_row + index]);
    }
    scores[token] = dot / 16.0;
    maximum = std::max(maximum, scores[token]);
  }
  double denominator = 0.0;
  double numerator = 0.0;
  for (uint64_t token = 0U; token <= position; ++token) {
    const double weight = std::exp(scores[token] - maximum);
    denominator += weight;
    const size_t value_row =
        (static_cast<size_t>(token) * kKvHeads + kv_head) * kHeadDim;
    numerator += weight * e4m3fn_decode(fixture.value[value_row + dimension]);
  }
  return numerator / denominator;
}

void upload_fixture(const Fixture &fixture, DeviceBuffer<uint16_t> &query,
                    DeviceBuffer<uint8_t> &physical_key,
                    DeviceBuffer<uint8_t> &physical_value,
                    DeviceBuffer<uint8_t> &physical_key_scales,
                    DeviceBuffer<uint8_t> &physical_value_scales,
                    DeviceBuffer<uint32_t> &logical_table,
                    DeviceBuffer<sllm_paged_kv::BlockDescriptor> &descriptors) {
  check(hipMemcpy(query.ptr, fixture.query.data(),
                  fixture.query.size() * sizeof(uint16_t),
                  hipMemcpyHostToDevice),
        "upload query");
  check(hipMemcpy(physical_key.ptr, fixture.physical_key.data(),
                  fixture.physical_key.size(), hipMemcpyHostToDevice),
        "upload physical key");
  check(hipMemcpy(physical_value.ptr, fixture.physical_value.data(),
                  fixture.physical_value.size(), hipMemcpyHostToDevice),
        "upload physical value");
  check(hipMemcpy(physical_key_scales.ptr, fixture.physical_key_scales.data(),
                  fixture.physical_key_scales.size(), hipMemcpyHostToDevice),
        "upload key scales");
  check(hipMemcpy(physical_value_scales.ptr,
                  fixture.physical_value_scales.data(),
                  fixture.physical_value_scales.size(), hipMemcpyHostToDevice),
        "upload value scales");

  std::vector<uint32_t> logical_pages(fixture.pages);
  std::vector<sllm_paged_kv::BlockDescriptor> host_descriptors(fixture.pages);
  for (uint32_t logical_page = 0U; logical_page < fixture.pages;
       ++logical_page) {
    const uint32_t physical_page = fixture.pages - 1U - logical_page;
    logical_pages[logical_page] = physical_page;
    host_descriptors[physical_page] = {
        physical_key.ptr +
            static_cast<size_t>(physical_page) * fixture.page_value_bytes,
        physical_value.ptr +
            static_cast<size_t>(physical_page) * fixture.page_value_bytes,
        physical_key_scales.ptr +
            static_cast<size_t>(physical_page) * fixture.page_scale_bytes,
        physical_value_scales.ptr +
            static_cast<size_t>(physical_page) * fixture.page_scale_bytes,
        nullptr,
        nullptr};
  }
  check(hipMemcpy(logical_table.ptr, logical_pages.data(),
                  logical_pages.size() * sizeof(uint32_t),
                  hipMemcpyHostToDevice),
        "upload logical table");
  check(hipMemcpy(descriptors.ptr, host_descriptors.data(),
                  host_descriptors.size() * sizeof(host_descriptors[0]),
                  hipMemcpyHostToDevice),
        "upload descriptors");
}

void set_control(DeviceBuffer<sllm_decode_control::ControlV1> &control,
                 const uint64_t position, const uint32_t rows) {
  sllm_decode_control::ControlV1 host{};
  host.version = sllm_decode_control::kVersion;
  host.status = static_cast<uint32_t>(sllm_decode_control::Status::Ok);
  host.mode = sllm_decode_control::kModeMtp;
  host.phase_position = position;
  host.phase_rows = rows;
  host.phase_kind = sllm_decode_control::kPhaseTarget;
  host.phase_active = 1U;
  check(hipMemcpy(control.ptr, &host, sizeof(host), hipMemcpyHostToDevice),
        "upload control");
}

void set_overflow_control(DeviceBuffer<sllm_decode_control::ControlV1> &control,
                          const uint32_t rows) {
  sllm_decode_control::ControlV1 host{};
  host.version = sllm_decode_control::kVersion;
  host.status = static_cast<uint32_t>(sllm_decode_control::Status::Ok);
  host.mode = sllm_decode_control::kModeMtp;
  host.phase_position = UINT64_MAX;
  host.phase_rows = rows;
  host.phase_kind = sllm_decode_control::kPhaseTarget;
  host.phase_active = 1U;
  check(hipMemcpy(control.ptr, &host, sizeof(host), hipMemcpyHostToDevice),
        "upload overflow control");
}

hipError_t
launch_path(const bool candidate, const bool device_control,
            const uint64_t position, const uint32_t rows,
            const Fixture &fixture, DeviceBuffer<uint16_t> &query,
            DeviceBuffer<uint32_t> &logical_table,
            DeviceBuffer<sllm_paged_kv::BlockDescriptor> &descriptors,
            DeviceBuffer<uint32_t> &status, DeviceBuffer<float> &workspace,
            DeviceBuffer<uint16_t> &output,
            DeviceBuffer<sllm_decode_control::ControlV1> &control,
            const bool gqa_shared, const hipStream_t stream) {
  const uint64_t committed = position + rows;
  if (device_control) {
    set_control(control, position, rows);
  }
  if (!candidate) {
    return sllm_causal_attention_kernel::launch_paged_decode_gqa6(
        query.ptr, logical_table.ptr, descriptors.ptr, fixture.pages,
        fixture.pages, status.ptr, workspace.ptr,
        workspace.count * sizeof(float), output.ptr, rows, position, committed,
        kQHeads, kKvHeads, kHeadDim, kEncoding, gqa_shared, stream);
  }
  if (device_control) {
    return sllm_causal_attention_kernel::launch_paged_decode_gqa6_c1_device(
        query.ptr, logical_table.ptr, descriptors.ptr, fixture.pages,
        fixture.pages, status.ptr, workspace.ptr,
        workspace.count * sizeof(float), output.ptr, rows, position, committed,
        kQHeads, kKvHeads, kHeadDim, kEncoding, control.ptr, stream);
  }
  return sllm_causal_attention_kernel::launch_paged_decode_gqa6_c1(
      query.ptr, logical_table.ptr, descriptors.ptr, fixture.pages,
      fixture.pages, status.ptr, workspace.ptr, workspace.count * sizeof(float),
      output.ptr, rows, position, committed, kQHeads, kKvHeads, kHeadDim,
      kEncoding, stream);
}

std::vector<uint16_t>
run_and_read(const bool candidate, const bool device_control,
             const uint64_t position, const uint32_t rows,
             const Fixture &fixture, DeviceBuffer<uint16_t> &query,
             DeviceBuffer<uint32_t> &logical_table,
             DeviceBuffer<sllm_paged_kv::BlockDescriptor> &descriptors,
             DeviceBuffer<uint32_t> &status, DeviceBuffer<float> &workspace,
             DeviceBuffer<uint16_t> &output,
             DeviceBuffer<sllm_decode_control::ControlV1> &control,
             const bool gqa_shared, const hipStream_t stream) {
  check(hipMemset(output.ptr, 0xcd, output.count * sizeof(uint16_t)),
        "clear output");
  const hipError_t launch = launch_path(
      candidate, device_control, position, rows, fixture, query, logical_table,
      descriptors, status, workspace, output, control, gqa_shared, stream);
  check(launch, candidate ? "launch C1" : "launch baseline");
  check(hipStreamSynchronize(stream), "synchronize attention");
  uint32_t status_value = 0U;
  check(hipMemcpy(&status_value, status.ptr, sizeof(status_value),
                  hipMemcpyDeviceToHost),
        "download status");
  if (status_value != 0U) {
    fail("paged attention status=" + std::to_string(status_value));
  }
  std::vector<uint16_t> host(output.count);
  check(hipMemcpy(host.data(), output.ptr, host.size() * sizeof(uint16_t),
                  hipMemcpyDeviceToHost),
        "download output");
  const size_t elements = static_cast<size_t>(rows) * kQHeads * kHeadDim;
  for (size_t index = 0U; index < elements; ++index) {
    if (!std::isfinite(bf16_to_f32(host[index]))) {
      fail("non-finite BF16 output at index=" + std::to_string(index));
    }
  }
  return host;
}

void verify_oracle(const Fixture &fixture, const std::vector<uint16_t> &output,
                   const uint64_t position, const uint32_t rows) {
  const uint32_t last_row = rows - 1U;
  const uint32_t checks[][3] = {{0U, 0U, 0U}, {last_row, 23U, 128U}};
  for (const auto &entry : checks) {
    const uint32_t row = entry[0];
    const uint32_t head = entry[1];
    const uint32_t dimension = entry[2];
    const size_t index =
        (static_cast<size_t>(row) * kQHeads + head) * kHeadDim + dimension;
    const double expected =
        host_oracle(fixture, position + row, row, head, dimension);
    const double actual = static_cast<double>(bf16_to_f32(output[index]));
    if (!std::isfinite(actual) || std::fabs(actual - expected) > 0.20) {
      fail("independent oracle mismatch row=" + std::to_string(row) + " head=" +
           std::to_string(head) + " dim=" + std::to_string(dimension) +
           " expected=" + std::to_string(expected) +
           " actual=" + std::to_string(actual));
    }
  }
}

float measure_one(const bool candidate, const uint64_t position,
                  const uint32_t rows, const Fixture &fixture,
                  DeviceBuffer<uint16_t> &query,
                  DeviceBuffer<uint32_t> &logical_table,
                  DeviceBuffer<sllm_paged_kv::BlockDescriptor> &descriptors,
                  DeviceBuffer<uint32_t> &status,
                  DeviceBuffer<float> &workspace,
                  DeviceBuffer<uint16_t> &output,
                  DeviceBuffer<sllm_decode_control::ControlV1> &control,
                  const bool gqa_shared, const hipStream_t stream,
                  hipEvent_t start, hipEvent_t stop) {
  check(hipEventRecord(start, stream), "record timing start");
  check(launch_path(candidate, false, position, rows, fixture, query,
                    logical_table, descriptors, status, workspace, output,
                    control, gqa_shared, stream),
        candidate ? "timed C1 launch" : "timed baseline launch");
  check(hipEventRecord(stop, stream), "record timing stop");
  check(hipEventSynchronize(stop), "synchronize timing");
  float milliseconds = 0.0F;
  check(hipEventElapsedTime(&milliseconds, start, stop), "read timing");
  return milliseconds;
}

void compare_case(const uint64_t position, const uint32_t rows,
                  const Fixture &fixture, DeviceBuffer<uint16_t> &query,
                  DeviceBuffer<uint32_t> &logical_table,
                  DeviceBuffer<sllm_paged_kv::BlockDescriptor> &descriptors,
                  DeviceBuffer<uint32_t> &status,
                  DeviceBuffer<float> &workspace,
                  DeviceBuffer<uint16_t> &output,
                  DeviceBuffer<sllm_decode_control::ControlV1> &control,
                  const bool gqa_shared, const hipStream_t stream) {
  const std::vector<uint16_t> baseline = run_and_read(
      false, false, position, rows, fixture, query, logical_table, descriptors,
      status, workspace, output, control, gqa_shared, stream);
  const std::vector<uint16_t> candidate = run_and_read(
      true, false, position, rows, fixture, query, logical_table, descriptors,
      status, workspace, output, control, gqa_shared, stream);
  const size_t elements = static_cast<size_t>(rows) * kQHeads * kHeadDim;
  const auto mismatch = std::mismatch(
      candidate.begin(), candidate.begin() + static_cast<ptrdiff_t>(elements),
      baseline.begin());
  if (mismatch.first != candidate.begin() + static_cast<ptrdiff_t>(elements)) {
    const size_t index =
        static_cast<size_t>(std::distance(candidate.begin(), mismatch.first));
    fail("C1 bitwise mismatch committed=" + std::to_string(position + rows) +
         " rows=" + std::to_string(rows) + " index=" + std::to_string(index) +
         " c1=" + std::to_string(candidate[index]) +
         " baseline=" + std::to_string(baseline[index]));
  }
  verify_oracle(fixture, candidate, position, rows);

  if (position + rows == 8193U && rows == 3U) {
    const std::vector<uint16_t> device_candidate = run_and_read(
        true, true, position, rows, fixture, query, logical_table, descriptors,
        status, workspace, output, control, gqa_shared, stream);
    if (!std::equal(device_candidate.begin(),
                    device_candidate.begin() + static_cast<ptrdiff_t>(elements),
                    candidate.begin())) {
      fail("C1 ControlV1 output differs from eager output");
    }
  }
}

void verify_control_overflow(
    const Fixture &fixture, DeviceBuffer<uint16_t> &query,
    DeviceBuffer<uint32_t> &logical_table,
    DeviceBuffer<sllm_paged_kv::BlockDescriptor> &descriptors,
    DeviceBuffer<uint32_t> &status, DeviceBuffer<float> &workspace,
    DeviceBuffer<uint16_t> &output,
    DeviceBuffer<sllm_decode_control::ControlV1> &control,
    const hipStream_t stream) {
  constexpr uint32_t kRows = 3U;
  constexpr uint64_t kOrdinaryPosition = 8188U;
  check(hipMemset(output.ptr, 0xcd, output.count * sizeof(uint16_t)),
        "clear overflow output");
  set_overflow_control(control, kRows);
  check(sllm_causal_attention_kernel::launch_paged_decode_gqa6_c1_device(
            query.ptr, logical_table.ptr, descriptors.ptr, fixture.pages,
            fixture.pages, status.ptr, workspace.ptr,
            workspace.count * sizeof(float), output.ptr, kRows,
            kOrdinaryPosition, kOrdinaryPosition + kRows, kQHeads, kKvHeads,
            kHeadDim, kEncoding, control.ptr, stream),
        "launch overflow control");
  check(hipStreamSynchronize(stream), "synchronize overflow control");
  uint32_t status_value = 0U;
  sllm_decode_control::ControlV1 observed{};
  check(hipMemcpy(&status_value, status.ptr, sizeof(status_value),
                  hipMemcpyDeviceToHost),
        "download overflow status");
  check(hipMemcpy(&observed, control.ptr, sizeof(observed),
                  hipMemcpyDeviceToHost),
        "download overflow control");
  if (status_value == 0U || observed.halted == 0U ||
      observed.phase_active != 0U ||
      observed.status !=
          static_cast<uint32_t>(sllm_decode_control::Status::InvalidPosition)) {
    fail("C1 ControlV1 position overflow was not rejected");
  }
}

void verify_partial_control(
    const Fixture &fixture, DeviceBuffer<uint16_t> &query,
    DeviceBuffer<uint32_t> &logical_table,
    DeviceBuffer<sllm_paged_kv::BlockDescriptor> &descriptors,
    DeviceBuffer<uint32_t> &status, DeviceBuffer<float> &workspace,
    DeviceBuffer<uint16_t> &output,
    DeviceBuffer<sllm_decode_control::ControlV1> &control,
    const bool gqa_shared, const hipStream_t stream) {
  constexpr uint64_t kCapturedPosition = 8190U;
  constexpr uint64_t kCapturedCommitted = 8193U;
  constexpr uint64_t kReplayPosition = 8191U;
  constexpr uint32_t kCapturedRows = 3U;
  const std::vector<uint16_t> eager_rows = run_and_read(
      true, false, kReplayPosition, 2U, fixture, query, logical_table,
      descriptors, status, workspace, output, control, gqa_shared, stream);
  const size_t row_stride = static_cast<size_t>(kQHeads) * kHeadDim;
  for (const uint32_t phase_rows : {1U, 2U}) {
    check(hipMemset(output.ptr, 0xcd, output.count * sizeof(uint16_t)),
          "clear partial control output");
    set_control(control, kReplayPosition, phase_rows);
    check(sllm_causal_attention_kernel::launch_paged_decode_gqa6_c1_device(
              query.ptr, logical_table.ptr, descriptors.ptr, fixture.pages,
              fixture.pages, status.ptr, workspace.ptr,
              workspace.count * sizeof(float), output.ptr, kCapturedRows,
              kCapturedPosition, kCapturedCommitted, kQHeads, kKvHeads,
              kHeadDim, kEncoding, control.ptr, stream),
          "launch partial ControlV1 replay");
    check(hipStreamSynchronize(stream), "synchronize partial ControlV1 replay");
    uint32_t status_value = 0U;
    sllm_decode_control::ControlV1 observed{};
    check(hipMemcpy(&status_value, status.ptr, sizeof(status_value),
                    hipMemcpyDeviceToHost),
          "download partial ControlV1 status");
    check(hipMemcpy(&observed, control.ptr, sizeof(observed),
                    hipMemcpyDeviceToHost),
          "download partial ControlV1 control");
    if (status_value != 0U || observed.halted != 0U ||
        observed.phase_active == 0U) {
      fail("partial ControlV1 replay did not complete");
    }
    std::vector<uint16_t> replay(output.count);
    check(hipMemcpy(replay.data(), output.ptr, replay.size() * sizeof(uint16_t),
                    hipMemcpyDeviceToHost),
          "download partial ControlV1 output");
    const size_t elements = static_cast<size_t>(phase_rows) * row_stride;
    if (!std::equal(replay.begin(),
                    replay.begin() + static_cast<ptrdiff_t>(elements),
                    eager_rows.begin())) {
      fail("partial ControlV1 replay differs from eager C1 output");
    }
    verify_oracle(fixture, replay, kReplayPosition, phase_rows);
  }
}

void run_ab_ba(const Fixture &fixture, DeviceBuffer<uint16_t> &query,
               DeviceBuffer<uint32_t> &logical_table,
               DeviceBuffer<sllm_paged_kv::BlockDescriptor> &descriptors,
               DeviceBuffer<uint32_t> &status, DeviceBuffer<float> &workspace,
               DeviceBuffer<uint16_t> &output,
               DeviceBuffer<sllm_decode_control::ControlV1> &control,
               const bool gqa_shared, const hipStream_t stream) {
  auto warmup = [&](const bool candidate, const uint64_t position,
                    const uint32_t rows, const bool baseline_gqa_shared) {
    check(launch_path(candidate, false, position, rows, fixture, query,
                      logical_table, descriptors, status, workspace, output,
                      control, baseline_gqa_shared, stream),
          candidate ? "warmup C1 launch" : "warmup baseline launch");
    check(hipStreamSynchronize(stream), "synchronize warmup");
  };
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  check(hipEventCreate(&start), "create timing start");
  check(hipEventCreate(&stop), "create timing stop");
  const bool timing_baseline_gqa_shared = true;
  for (const uint64_t committed : {8191U, 8192U, 8193U}) {
    for (const uint32_t rows : {2U, 3U}) {
      const uint64_t position = committed - rows;
      for (uint32_t iteration = 0U; iteration < 3U; ++iteration) {
        warmup(false, position, rows, timing_baseline_gqa_shared);
        warmup(true, position, rows, gqa_shared);
      }
      std::vector<float> baseline_ab;
      std::vector<float> candidate_ab;
      std::vector<float> baseline_ba;
      std::vector<float> candidate_ba;
      for (uint32_t iteration = 0U; iteration < 3U; ++iteration) {
        baseline_ab.push_back(
            measure_one(false, position, rows, fixture, query, logical_table,
                        descriptors, status, workspace, output, control,
                        timing_baseline_gqa_shared, stream, start, stop));
        candidate_ab.push_back(measure_one(true, position, rows, fixture, query,
                                           logical_table, descriptors, status,
                                           workspace, output, control,
                                           gqa_shared, stream, start, stop));
      }
      for (uint32_t iteration = 0U; iteration < 3U; ++iteration) {
        candidate_ba.push_back(measure_one(true, position, rows, fixture, query,
                                           logical_table, descriptors, status,
                                           workspace, output, control,
                                           gqa_shared, stream, start, stop));
        baseline_ba.push_back(
            measure_one(false, position, rows, fixture, query, logical_table,
                        descriptors, status, workspace, output, control,
                        timing_baseline_gqa_shared, stream, start, stop));
      }
      std::printf(
          "C1 short-ABBA target=%s committed=%llu rows=%u "
          "baseline_gqa_shared=1 "
          "AB=[%.4f/%.4f(%.3f),%.4f/%.4f(%.3f),%.4f/%.4f(%.3f)] "
          "BA=[%.4f/%.4f(%.3f),%.4f/%.4f(%.3f),%.4f/%.4f(%.3f)]\n",
          SLLM_TEST_EXPECTED_TARGET, static_cast<unsigned long long>(committed),
          rows, baseline_ab[0], candidate_ab[0],
          candidate_ab[0] / baseline_ab[0], baseline_ab[1], candidate_ab[1],
          candidate_ab[1] / baseline_ab[1], baseline_ab[2], candidate_ab[2],
          candidate_ab[2] / baseline_ab[2], baseline_ba[0], candidate_ba[0],
          candidate_ba[0] / baseline_ba[0], baseline_ba[1], candidate_ba[1],
          candidate_ba[1] / baseline_ba[1], baseline_ba[2], candidate_ba[2],
          candidate_ba[2] / baseline_ba[2]);
    }
  }
  check(hipEventDestroy(start), "destroy timing start");
  check(hipEventDestroy(stop), "destroy timing stop");
}

std::string uuid_text(const hipUUID &uuid) {
  std::string value(uuid.bytes, sizeof(uuid.bytes));
  for (char &character : value) {
    const auto byte = static_cast<unsigned char>(character);
    if (!std::isxdigit(byte)) {
      return {};
    }
    character = static_cast<char>(std::tolower(byte));
  }
  return "GPU-" + value;
}
} // namespace

int main() {
  try {
    check(hipSetDevice(0), "set device");
    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, 0), "device properties");
    const std::string target(properties.gcnArchName);
    if (target != SLLM_TEST_EXPECTED_TARGET ||
        (target != "gfx1030" && target != "gfx1201")) {
      fail("exact target mismatch runtime=" + target +
           " expected=" + SLLM_TEST_EXPECTED_TARGET);
    }
    hipUUID uuid{};
    check(hipDeviceGetUuid(&uuid, 0), "device UUID");
    if (uuid_text(uuid) != SLLM_TEST_EXPECTED_UUID) {
      fail("exact UUID mismatch runtime=" + uuid_text(uuid) +
           " expected=" + SLLM_TEST_EXPECTED_UUID);
    }

    const Fixture fixture;
    DeviceBuffer<uint16_t> query(fixture.query.size());
    DeviceBuffer<uint8_t> physical_key(fixture.physical_key.size());
    DeviceBuffer<uint8_t> physical_value(fixture.physical_value.size());
    DeviceBuffer<uint8_t> physical_key_scales(
        fixture.physical_key_scales.size());
    DeviceBuffer<uint8_t> physical_value_scales(
        fixture.physical_value_scales.size());
    DeviceBuffer<uint32_t> logical_table(fixture.pages);
    DeviceBuffer<sllm_paged_kv::BlockDescriptor> descriptors(fixture.pages);
    DeviceBuffer<uint32_t> status(1U);
    DeviceBuffer<float> workspace(static_cast<size_t>(kMaxRows) * kQHeads *
                                  128U * (kHeadDim + 2U));
    DeviceBuffer<uint16_t> output(static_cast<size_t>(kMaxRows) * kQHeads *
                                  kHeadDim);
    DeviceBuffer<sllm_decode_control::ControlV1> control(1U);
    upload_fixture(fixture, query, physical_key, physical_value,
                   physical_key_scales, physical_value_scales, logical_table,
                   descriptors);

    hipStream_t stream = nullptr;
    check(hipStreamCreate(&stream), "create stream");
    const bool gqa_shared = target == "gfx1030";
    const uint64_t committed_lengths[] = {127U,  128U,   129U,   8191U, 8192U,
                                          8193U, 65535U, 65536U, 65537U};
    uint32_t cases = 0U;
    for (const uint64_t committed : committed_lengths) {
      for (const uint32_t rows : {2U, 3U}) {
        if (committed < rows) {
          continue;
        }
        compare_case(committed - rows, rows, fixture, query, logical_table,
                     descriptors, status, workspace, output, control,
                     gqa_shared, stream);
        ++cases;
      }
    }
    verify_control_overflow(fixture, query, logical_table, descriptors, status,
                            workspace, output, control, stream);
    verify_partial_control(fixture, query, logical_table, descriptors, status,
                           workspace, output, control, gqa_shared, stream);
    run_ab_ba(fixture, query, logical_table, descriptors, status, workspace,
              output, control, gqa_shared, stream);
    check(hipStreamDestroy(stream), "destroy stream");
    check(hipDeviceSynchronize(), "final synchronize");
    std::printf("phase87-stage11-c1-paged-decode target=%s cases=%u "
                "bitwise=PASS oracle=PASS control_v1=PASS partial=PASS "
                "overflow=PASS "
                "fallback=0 "
                "cleanup=PASS\n",
                target.c_str(), cases);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "phase87-stage11-c1-paged-decode FAIL: %s\n",
                 error.what());
    return 1;
  }
}
