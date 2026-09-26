// Phase 87 Stage10: paged FP16 causal-attention control path.
//
// This focused test includes the production HIP translation unit so it can be
// built without changing the native test graph.  It covers both reviewed
// GQA geometries, M=1..5 decode, a short prefill, nonidentity 128-token page
// tables, and the 65535/65536/65537 logical-token boundaries.  The paged
// result is compared with the same generic FP16 contiguous kernel and with an
// independent host attention calculation.  The FP16 paged path is opt-in and
// is not a production-runtime default in this test.
#define SLLM_PUBLIC_RUNTIME_HOST_TEST 1
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#include "../src/causal_attention_kernel.hip.cpp"
#pragma clang diagnostic pop

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kHeadDim = 256U;
constexpr uint32_t kPageTokens = 128U;
constexpr uint32_t kMaxQueryCount = 265U;
constexpr uint64_t kMaxTokens = 65538U;
constexpr uint32_t kEncoding = SLLM_HIP_KV_ENCODING_FP16_V1;
constexpr float kScoreScale = 0.0625F;

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

uint16_t f32_to_f16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16U) & 0x8000U;
  const uint32_t exponent = (bits >> 23U) & 0xffU;
  const uint32_t fraction = bits & 0x7fffffU;
  if (exponent >= 143U) {
    return static_cast<uint16_t>(sign | 0x7c00U);
  }
  if (exponent <= 112U) {
    if (exponent < 103U) {
      return static_cast<uint16_t>(sign);
    }
    const uint32_t mantissa = fraction | 0x800000U;
    const uint32_t shift = 126U - exponent;
    uint32_t half = mantissa >> shift;
    const uint32_t remainder = mantissa & ((1U << shift) - 1U);
    const uint32_t halfway = 1U << (shift - 1U);
    if (remainder > halfway || (remainder == halfway && (half & 1U) != 0U)) {
      ++half;
    }
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half_exponent = exponent - 112U;
  uint32_t half_fraction = fraction >> 13U;
  const uint32_t remainder = fraction & 0x1fffU;
  if (remainder > 0x1000U ||
      (remainder == 0x1000U && (half_fraction & 1U) != 0U)) {
    ++half_fraction;
    if (half_fraction == 0x400U) {
      half_fraction = 0U;
      ++half_exponent;
    }
  }
  if (half_exponent >= 31U) {
    return static_cast<uint16_t>(sign | 0x7c00U);
  }
  return static_cast<uint16_t>(sign | (half_exponent << 10U) | half_fraction);
}

float bf16_to_f32(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

float f16_to_f32(const uint16_t raw) {
  const uint32_t sign = (static_cast<uint32_t>(raw) & 0x8000U) << 16U;
  const uint32_t exponent = (static_cast<uint32_t>(raw) >> 10U) & 0x1fU;
  const uint32_t fraction = static_cast<uint32_t>(raw) & 0x03ffU;
  uint32_t bits = 0U;
  if (exponent == 0U) {
    if (fraction == 0U) {
      bits = sign;
    } else {
      uint32_t normalized = fraction;
      uint32_t shift = 0U;
      while ((normalized & 0x0400U) == 0U) {
        normalized <<= 1U;
        ++shift;
      }
      normalized &= 0x03ffU;
      bits = sign | ((127U - 14U - shift) << 23U) | (normalized << 13U);
    }
  } else if (exponent == 0x1fU) {
    bits = sign | 0x7f800000U | (fraction << 13U);
  } else {
    bits = sign | ((exponent + 112U) << 23U) | (fraction << 13U);
  }
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

struct Fixture final {
  const uint32_t q_heads;
  const uint32_t pages = static_cast<uint32_t>((kMaxTokens + 127U) / 128U);
  const size_t page_elements =
      static_cast<size_t>(kPageTokens) * kKvHeads * kHeadDim;
  const size_t elements = static_cast<size_t>(kMaxTokens) * kKvHeads *
                          static_cast<size_t>(kHeadDim);
  std::vector<uint16_t> query;
  std::vector<uint16_t> key;
  std::vector<uint16_t> value;
  std::vector<uint16_t> physical_key;
  std::vector<uint16_t> physical_value;

  explicit Fixture(const uint32_t heads)
      : q_heads(heads),
        query(static_cast<size_t>(kMaxQueryCount) * q_heads * kHeadDim),
        key(elements), value(elements),
        physical_key(static_cast<size_t>(pages) * page_elements),
        physical_value(static_cast<size_t>(pages) * page_elements) {
    for (size_t index = 0U; index < query.size(); ++index) {
      query[index] = f32_to_bf16(
          0.003F * static_cast<float>((index * 17U) % 127U) - 0.18F);
    }
    constexpr uint16_t kHalfValues[] = {
        0x3c00U, // 1.0
        0xbc00U, // -1.0
        0x3800U, // 0.5
        0xb800U, // -0.5
        0x4000U, // 2.0
        0x3400U, // 0.25
        0xc000U, // -2.0
        0x3a00U, // 0.75
    };
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
          const size_t index = static_cast<size_t>(
              (token * 13U + head * 7U + dimension * 3U) %
              (sizeof(kHalfValues) / sizeof(kHalfValues[0])));
          key[logical_row + dimension] = kHalfValues[index];
          value[logical_row + dimension] =
              kHalfValues[(index + 3U) %
                          (sizeof(kHalfValues) / sizeof(kHalfValues[0]))];
          physical_key[physical_row + dimension] = key[logical_row + dimension];
          physical_value[physical_row + dimension] =
              value[logical_row + dimension];
        }
      }
    }
  }
};

void fill_high_entropy_gqa4_fixture(Fixture &fixture) {
  if (fixture.q_heads != 16U) {
    return;
  }
  const auto value = [](uint64_t index) {
    uint32_t state = static_cast<uint32_t>(index) * 747796405U + 2891336453U;
    state ^= state >> 16U;
    state *= 2246822519U;
    state ^= state >> 13U;
    return static_cast<float>(state % 2001U) / 200.0F - 5.0F;
  };
  for (uint32_t row = 0U; row < kMaxQueryCount; ++row) {
    for (uint32_t head = 0U; head < fixture.q_heads; ++head) {
      for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
        const size_t index =
            (static_cast<size_t>(row) * fixture.q_heads + head) * kHeadDim +
            dimension;
        fixture.query[index] = f32_to_bf16(value(2'000'000U + index));
      }
    }
  }
  for (uint64_t token = 0U; token < 265U; ++token) {
    for (uint32_t head = 0U; head < kKvHeads; ++head) {
      const size_t row =
          (static_cast<size_t>(token) * kKvHeads + head) * kHeadDim;
      const uint32_t physical_page =
          static_cast<uint32_t>(fixture.pages - 1U - token / kPageTokens);
      const size_t physical_row =
          (static_cast<size_t>(physical_page) * kPageTokens +
           static_cast<size_t>(token % kPageTokens)) *
              kKvHeads * kHeadDim +
          static_cast<size_t>(head) * kHeadDim;
      for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
        const size_t index = row + dimension;
        const uint16_t key = f32_to_f16(value(index));
        const uint16_t val = f32_to_f16(value(1'000'000U + index));
        fixture.key[index] = key;
        fixture.value[index] = val;
        fixture.physical_key[physical_row + dimension] = key;
        fixture.physical_value[physical_row + dimension] = val;
      }
    }
  }
}

double host_oracle_dim0(const Fixture &fixture, const uint64_t start_position,
                        const uint32_t row, const uint32_t query_head) {
  const uint64_t query_position = start_position + row;
  const uint32_t kv_head = query_head / (fixture.q_heads / kKvHeads);
  const uint16_t *const query_row =
      fixture.query.data() +
      (static_cast<size_t>(row) * fixture.q_heads + query_head) * kHeadDim;
  double maximum = -std::numeric_limits<double>::infinity();
  std::vector<double> scores(static_cast<size_t>(query_position) + 1U);
  for (uint64_t token = 0U; token <= query_position; ++token) {
    const size_t key_row =
        (static_cast<size_t>(token) * kKvHeads + kv_head) * kHeadDim;
    double dot = 0.0;
    for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
      dot += static_cast<double>(bf16_to_f32(query_row[dimension])) *
             static_cast<double>(f16_to_f32(fixture.key[key_row + dimension]));
    }
    scores[token] = dot * static_cast<double>(kScoreScale);
    maximum = std::max(maximum, scores[token]);
  }
  double denominator = 0.0;
  double numerator = 0.0;
  for (uint64_t token = 0U; token <= query_position; ++token) {
    const double weight = std::exp(scores[token] - maximum);
    denominator += weight;
    const size_t value_row =
        (static_cast<size_t>(token) * kKvHeads + kv_head) * kHeadDim;
    numerator +=
        weight * static_cast<double>(f16_to_f32(fixture.value[value_row]));
  }
  return numerator / denominator;
}

void launch_contiguous(const Fixture &fixture, DeviceBuffer<uint16_t> &query,
                       DeviceBuffer<uint16_t> &key,
                       DeviceBuffer<uint16_t> &value,
                       DeviceBuffer<uint16_t> &output, const uint32_t rows,
                       const uint64_t start_position, hipStream_t stream) {
  const uint64_t committed = start_position + rows;
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(sllm_causal_attention_kernel::causal_attention_kernel<
                      false, SLLM_HIP_KV_ENCODING_FP16_V1>),
      dim3(rows * fixture.q_heads), dim3(256U), 0U, stream, query.ptr, key.ptr,
      value.ptr, nullptr, nullptr, nullptr, nullptr, output.ptr, rows,
      kMaxTokens, start_position, committed, fixture.q_heads, kKvHeads,
      kHeadDim, 1.0F, 1.0F, 0U, kScoreScale);
  check(hipGetLastError(), "launch contiguous FP16 attention");
}

void launch_contiguous_gqa4(const Fixture &fixture,
                            DeviceBuffer<uint16_t> &query,
                            DeviceBuffer<uint16_t> &key,
                            DeviceBuffer<uint16_t> &value,
                            DeviceBuffer<uint16_t> &output, const uint32_t rows,
                            const uint64_t start_position, hipStream_t stream) {
  const uint64_t blocks = static_cast<uint64_t>(rows) * kKvHeads;
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(sllm_causal_attention_kernel::
                          causal_attention_prefill_gqa4_shared_kernel<
                              SLLM_HIP_KV_ENCODING_FP16_V1>),
      dim3(static_cast<uint32_t>(blocks)), dim3(256U), 0U, stream, query.ptr,
      key.ptr, value.ptr, nullptr, nullptr, nullptr, nullptr, output.ptr, rows,
      start_position, fixture.q_heads, kKvHeads, kHeadDim, 1.0F, 1.0F);
  check(hipGetLastError(), "launch contiguous FP16 GQA4 attention");
}

void compare_outputs(const Fixture &fixture, const std::vector<uint16_t> &paged,
                     const std::vector<uint16_t> &control, const uint32_t rows,
                     const uint64_t start_position, const char *const mode) {
  const size_t elements =
      static_cast<size_t>(rows) * fixture.q_heads * kHeadDim;
  for (size_t index = 0U; index < elements; ++index) {
    if (paged[index] != control[index]) {
      fail(std::string(mode) + " differs from contiguous control index=" +
           std::to_string(index) + " paged=" + std::to_string(paged[index]) +
           " control=" + std::to_string(control[index]));
    }
  }
  const double expected = host_oracle_dim0(fixture, start_position, 0U, 0U);
  const double actual = static_cast<double>(bf16_to_f32(paged[0]));
  if (!std::isfinite(actual) || std::fabs(actual - expected) > 0.02) {
    fail(std::string(mode) + " host oracle mismatch expected=" +
         std::to_string(expected) + " actual=" + std::to_string(actual));
  }
}

void run_case(const Fixture &fixture, const bool prefill, const uint32_t rows,
              const uint64_t start_position, DeviceBuffer<uint16_t> &query,
              DeviceBuffer<uint16_t> &key, DeviceBuffer<uint16_t> &value,
              DeviceBuffer<uint32_t> &logical_table,
              DeviceBuffer<sllm_paged_kv::BlockDescriptor> &descriptors,
              DeviceBuffer<uint32_t> &status, DeviceBuffer<uint16_t> &paged,
              DeviceBuffer<uint16_t> &control, hipStream_t stream) {
  check(hipMemset(paged.ptr, 0xcd, paged.count * sizeof(uint16_t)),
        "clear paged output");
  check(hipMemset(control.ptr, 0xcd, control.count * sizeof(uint16_t)),
        "clear control output");
  if (prefill && fixture.q_heads == 16U && rows >= 64U) {
    check(sllm_causal_attention_kernel::launch_paged_prefill_gqa4(
              query.ptr, logical_table.ptr, descriptors.ptr, fixture.pages,
              fixture.pages, status.ptr, paged.ptr, rows, start_position,
              start_position + rows, fixture.q_heads, kKvHeads, kHeadDim,
              stream),
          "launch paged FP16 GQA4 prefill");
  } else if (prefill) {
    check(sllm_causal_attention_kernel::launch_paged_prefill_fp16(
              query.ptr, logical_table.ptr, descriptors.ptr, fixture.pages,
              fixture.pages, status.ptr, paged.ptr, rows, start_position,
              start_position + rows, fixture.q_heads, kKvHeads, kHeadDim,
              stream),
          "launch paged FP16 prefill");
  } else {
    check(sllm_causal_attention_kernel::launch_paged_decode_fp16(
              query.ptr, logical_table.ptr, descriptors.ptr, fixture.pages,
              fixture.pages, status.ptr, paged.ptr, rows, start_position,
              start_position + rows, fixture.q_heads, kKvHeads, kHeadDim,
              stream),
          "launch paged FP16 decode");
  }
  const bool gqa4_prefill = prefill && fixture.q_heads == 16U && rows >= 64U;
  if (gqa4_prefill) {
    launch_contiguous_gqa4(fixture, query, key, value, control, rows,
                           start_position, stream);
  } else {
    launch_contiguous(fixture, query, key, value, control, rows, start_position,
                      stream);
  }
  check(hipStreamSynchronize(stream), "synchronize attention case");
  uint32_t status_value = 0U;
  check(hipMemcpy(&status_value, status.ptr, sizeof(status_value),
                  hipMemcpyDeviceToHost),
        "download paged status");
  if (status_value != 0U) {
    fail("paged FP16 descriptor validation status=" +
         std::to_string(status_value));
  }
  std::vector<uint16_t> paged_host(paged.count);
  std::vector<uint16_t> control_host(control.count);
  check(hipMemcpy(paged_host.data(), paged.ptr,
                  paged_host.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
        "download paged output");
  check(hipMemcpy(control_host.data(), control.ptr,
                  control_host.size() * sizeof(uint16_t),
                  hipMemcpyDeviceToHost),
        "download contiguous output");
  compare_outputs(fixture, paged_host, control_host, rows, start_position,
                  prefill ? "paged FP16 prefill" : "paged FP16 decode");
}

void run_geometry(const uint32_t q_heads, hipStream_t stream) {
  const bool gqa4_only = std::getenv("SLLM_TEST_GQA4_ONLY") != nullptr;
  if (gqa4_only && q_heads != 16U) {
    return;
  }
  Fixture fixture(q_heads);
  fill_high_entropy_gqa4_fixture(fixture);
  DeviceBuffer<uint16_t> query(fixture.query.size());
  DeviceBuffer<uint16_t> key(fixture.key.size());
  DeviceBuffer<uint16_t> value(fixture.value.size());
  DeviceBuffer<uint16_t> physical_key(fixture.physical_key.size());
  DeviceBuffer<uint16_t> physical_value(fixture.physical_value.size());
  DeviceBuffer<uint32_t> logical_table(fixture.pages);
  DeviceBuffer<sllm_paged_kv::BlockDescriptor> descriptors(fixture.pages);
  DeviceBuffer<uint32_t> status(1U);
  DeviceBuffer<uint16_t> paged(static_cast<size_t>(kMaxQueryCount) * q_heads *
                               kHeadDim);
  DeviceBuffer<uint16_t> control(paged.count);
  check(hipMemcpy(query.ptr, fixture.query.data(),
                  fixture.query.size() * sizeof(uint16_t),
                  hipMemcpyHostToDevice),
        "upload query");
  check(hipMemcpy(key.ptr, fixture.key.data(),
                  fixture.key.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
        "upload contiguous key");
  check(hipMemcpy(value.ptr, fixture.value.data(),
                  fixture.value.size() * sizeof(uint16_t),
                  hipMemcpyHostToDevice),
        "upload contiguous value");
  check(hipMemcpy(physical_key.ptr, fixture.physical_key.data(),
                  fixture.physical_key.size() * sizeof(uint16_t),
                  hipMemcpyHostToDevice),
        "upload paged key");
  check(hipMemcpy(physical_value.ptr, fixture.physical_value.data(),
                  fixture.physical_value.size() * sizeof(uint16_t),
                  hipMemcpyHostToDevice),
        "upload paged value");

  std::vector<uint32_t> logical_pages(fixture.pages);
  std::vector<sllm_paged_kv::BlockDescriptor> descriptor_host(fixture.pages);
  const size_t page_bytes = fixture.page_elements * sizeof(uint16_t);
  for (uint32_t logical_page = 0U; logical_page < fixture.pages;
       ++logical_page) {
    const uint32_t physical_page = fixture.pages - 1U - logical_page;
    logical_pages[logical_page] = physical_page;
    descriptor_host[physical_page] = {
        reinterpret_cast<uint8_t *>(physical_key.ptr) +
            static_cast<size_t>(physical_page) * page_bytes,
        reinterpret_cast<uint8_t *>(physical_value.ptr) +
            static_cast<size_t>(physical_page) * page_bytes,
        nullptr,
        nullptr,
        nullptr,
        nullptr};
  }
  check(hipMemcpy(logical_table.ptr, logical_pages.data(),
                  logical_pages.size() * sizeof(uint32_t),
                  hipMemcpyHostToDevice),
        "upload logical table");
  check(hipMemcpy(descriptors.ptr, descriptor_host.data(),
                  descriptor_host.size() * sizeof(descriptor_host[0]),
                  hipMemcpyHostToDevice),
        "upload FP16 descriptors");

  if (!gqa4_only) {
    // One launch with M=5 exercises every decode row in the requested M1..M5
    // range.  The remaining launches exercise both sides of all physical page
    // boundaries without adding a long benchmark loop.
    run_case(fixture, false, 5U, 127U, query, key, value, logical_table,
             descriptors, status, paged, control, stream);
    for (const uint64_t start : {128U, 129U, 65535U, 65536U, 65537U}) {
      run_case(fixture, false, 1U, start, query, key, value, logical_table,
               descriptors, status, paged, control, stream);
    }
    run_case(fixture, true, 5U, 127U, query, key, value, logical_table,
             descriptors, status, paged, control, stream);
  }
  if (q_heads == 16U) {
    for (const uint32_t rows : {65U, 85U, 265U}) {
      run_case(fixture, true, rows, 0U, query, key, value, logical_table,
               descriptors, status, paged, control, stream);
    }
  }

  // An invalid logical ID must report through the device status and leave the
  // sentinel output untouched; no contiguous or CPU fallback is permitted.
  const uint32_t invalid = UINT32_MAX;
  check(hipMemcpy(logical_table.ptr, &invalid, sizeof(invalid),
                  hipMemcpyHostToDevice),
        "upload invalid logical ID");
  check(hipMemset(paged.ptr, 0xcd, paged.count * sizeof(uint16_t)),
        "clear invalid output");
  check(sllm_causal_attention_kernel::launch_paged_decode_fp16(
            query.ptr, logical_table.ptr, descriptors.ptr, fixture.pages,
            fixture.pages, status.ptr, paged.ptr, 1U, 0U, 1U, fixture.q_heads,
            kKvHeads, kHeadDim, stream),
        "launch invalid FP16 paged descriptor");
  check(hipStreamSynchronize(stream), "synchronize invalid descriptor");
  uint32_t invalid_status = 0U;
  check(hipMemcpy(&invalid_status, status.ptr, sizeof(invalid_status),
                  hipMemcpyDeviceToHost),
        "download invalid status");
  if (invalid_status == 0U) {
    fail("invalid FP16 logical ID did not set device status");
  }
  std::vector<uint16_t> invalid_output(paged.count);
  check(hipMemcpy(invalid_output.data(), paged.ptr,
                  invalid_output.size() * sizeof(uint16_t),
                  hipMemcpyDeviceToHost),
        "download invalid output");
  const size_t elements = static_cast<size_t>(fixture.q_heads) * kHeadDim;
  if (!std::all_of(invalid_output.begin(), invalid_output.begin() + elements,
                   [](const uint16_t value) { return value == 0xcdcdU; })) {
    fail("invalid FP16 descriptor wrote output or used a fallback");
  }
}
} // namespace

int main() {
  try {
    check(hipSetDevice(0), "set device");
    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, 0), "device properties");
    const std::string actual_target(properties.gcnArchName);
    if (actual_target.rfind("gfx1030", 0U) != 0U &&
        actual_target.rfind("gfx1201", 0U) != 0U) {
      std::fprintf(stderr,
                   "phase87-stage10-paged-fp16-attention SKIP target=%s\n",
                   actual_target.c_str());
      return 0;
    }
    hipStream_t stream = nullptr;
    check(hipStreamCreate(&stream), "create stream");
    run_geometry(16U, stream);
    run_geometry(24U, stream);
    check(hipStreamDestroy(stream), "destroy stream");
    std::printf("phase87-stage10-paged-fp16-attention target=%s geometries=2 "
                "boundaries=6 state=PASS\n",
                actual_target.c_str());
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "phase87-stage10-paged-fp16-attention FAIL: %s\n",
                 error.what());
    return 1;
  }
}
