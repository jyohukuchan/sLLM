// Phase 87 Stage10: production paged MXFP8-E4 attention smoke.
//
// This focused test includes the production HIP translation unit so it can be
// built without changing the native test graph. It exercises a reversed,
// nonidentity descriptor table at both sides of the 128-token and 65536-token
// boundaries, compares the paged result byte-for-byte with the contiguous
// provider, and checks one output element against an independent host oracle.
#define SLLM_PUBLIC_RUNTIME_HOST_TEST 1
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#include "../src/causal_attention_kernel.hip.cpp"
#pragma clang diagnostic pop

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint32_t kQHeads = 24U;
constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kHeadDim = 256U;
constexpr uint32_t kScaleBytes = 8U;
constexpr uint32_t kPageTokens = 128U;
constexpr uint64_t kPrefillStart = 65535U;
constexpr uint32_t kPrefillRows = 128U;
constexpr uint64_t kMaxTokens = kPrefillStart + kPrefillRows;
constexpr uint32_t kEncoding = SLLM_HIP_KV_ENCODING_MXFP8_E4_V1;

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

double e8m0_decode(const uint8_t value) {
  return std::ldexp(1.0, static_cast<int>(value) - 127);
}

struct Fixture final {
  uint32_t pages = static_cast<uint32_t>((kMaxTokens + 127U) / 128U);
  size_t page_value_bytes =
      static_cast<size_t>(kPageTokens) * kKvHeads * kHeadDim;
  size_t page_scale_bytes =
      static_cast<size_t>(kPageTokens) * kKvHeads * kScaleBytes;
  std::vector<uint16_t> query;
  std::vector<uint8_t> key;
  std::vector<uint8_t> value;
  std::vector<uint8_t> key_scales;
  std::vector<uint8_t> value_scales;
  std::vector<uint8_t> physical_key;
  std::vector<uint8_t> physical_value;
  std::vector<uint8_t> physical_key_scales;
  std::vector<uint8_t> physical_value_scales;

  Fixture()
      : query(static_cast<size_t>(kPrefillRows) * kQHeads * kHeadDim),
        key(static_cast<size_t>(kMaxTokens) * kKvHeads * kHeadDim),
        value(key.size()),
        key_scales(static_cast<size_t>(kMaxTokens) * kKvHeads * kScaleBytes,
                   0x7fU),
        value_scales(key_scales),
        physical_key(static_cast<size_t>(pages) * page_value_bytes),
        physical_value(physical_key.size()),
        physical_key_scales(static_cast<size_t>(pages) * page_scale_bytes),
        physical_value_scales(physical_key_scales) {
    for (size_t index = 0U; index < query.size(); ++index) {
      query[index] =
          f32_to_bf16(0.07F * static_cast<float>((index * 13U) % 17U) - 0.4F);
    }
    for (uint64_t token = 0U; token < kMaxTokens; ++token) {
      const uint32_t logical_page = static_cast<uint32_t>(token / 128U);
      const uint32_t local_token = static_cast<uint32_t>(token % 128U);
      const uint32_t physical_page = pages - 1U - logical_page;
      for (uint32_t head = 0U; head < kKvHeads; ++head) {
        const size_t row =
            (static_cast<size_t>(token) * kKvHeads + head) * kHeadDim;
        const size_t physical_row =
            (static_cast<size_t>(physical_page) * kPageTokens + local_token) *
                kKvHeads * kHeadDim +
            static_cast<size_t>(head) * kHeadDim;
        for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
          const uint8_t key_code = static_cast<uint8_t>(
              0x08U + ((token * 17U + head * 5U + dimension) % 0x70U));
          const uint8_t value_code = static_cast<uint8_t>(
              0x10U + ((token * 7U + head * 11U + dimension * 3U) % 0x60U));
          key[row + dimension] = key_code;
          value[row + dimension] = value_code;
          physical_key[physical_row + dimension] = key_code;
          physical_value[physical_row + dimension] = value_code;
        }
        const size_t scale_row =
            (static_cast<size_t>(token) * kKvHeads + head) * kScaleBytes;
        const size_t physical_scale_row =
            (static_cast<size_t>(physical_page) * kPageTokens + local_token) *
                kKvHeads * kScaleBytes +
            static_cast<size_t>(head) * kScaleBytes;
        std::memcpy(physical_key_scales.data() + physical_scale_row,
                    key_scales.data() + scale_row, kScaleBytes);
        std::memcpy(physical_value_scales.data() + physical_scale_row,
                    value_scales.data() + scale_row, kScaleBytes);
      }
    }
  }
};

double host_oracle_dim0(const Fixture &fixture, const uint64_t start_position) {
  double maximum = -std::numeric_limits<double>::infinity();
  std::vector<double> scores(static_cast<size_t>(start_position) + 1U);
  for (uint64_t token = 0U; token <= start_position; ++token) {
    const size_t key_row = static_cast<size_t>(token) * kKvHeads * kHeadDim;
    double dot = 0.0;
    for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
      dot += static_cast<double>(bf16_to_f32(fixture.query[dimension])) *
             e4m3fn_decode(fixture.key[key_row + dimension]);
    }
    scores[token] = dot / 16.0;
    maximum = std::max(maximum, scores[token]);
  }
  double denominator = 0.0;
  double numerator = 0.0;
  for (uint64_t token = 0U; token <= start_position; ++token) {
    const double weight = std::exp(scores[token] - maximum);
    denominator += weight;
    const size_t value_row = static_cast<size_t>(token) * kKvHeads * kHeadDim;
    numerator += weight * e4m3fn_decode(fixture.value[value_row]);
  }
  return numerator / denominator;
}

void upload(const Fixture &fixture, DeviceBuffer<uint16_t> &query,
            DeviceBuffer<uint8_t> &key, DeviceBuffer<uint8_t> &value,
            DeviceBuffer<uint8_t> &key_scales,
            DeviceBuffer<uint8_t> &value_scales,
            DeviceBuffer<uint8_t> &physical_key,
            DeviceBuffer<uint8_t> &physical_value,
            DeviceBuffer<uint8_t> &physical_key_scales,
            DeviceBuffer<uint8_t> &physical_value_scales) {
  check(hipMemcpy(query.ptr, fixture.query.data(),
                  fixture.query.size() * sizeof(uint16_t),
                  hipMemcpyHostToDevice),
        "upload query");
  check(hipMemcpy(key.ptr, fixture.key.data(), fixture.key.size(),
                  hipMemcpyHostToDevice),
        "upload key");
  check(hipMemcpy(value.ptr, fixture.value.data(), fixture.value.size(),
                  hipMemcpyHostToDevice),
        "upload value");
  check(hipMemcpy(key_scales.ptr, fixture.key_scales.data(),
                  fixture.key_scales.size(), hipMemcpyHostToDevice),
        "upload key scales");
  check(hipMemcpy(value_scales.ptr, fixture.value_scales.data(),
                  fixture.value_scales.size(), hipMemcpyHostToDevice),
        "upload value scales");
  check(hipMemcpy(physical_key.ptr, fixture.physical_key.data(),
                  fixture.physical_key.size(), hipMemcpyHostToDevice),
        "upload physical key");
  check(hipMemcpy(physical_value.ptr, fixture.physical_value.data(),
                  fixture.physical_value.size(), hipMemcpyHostToDevice),
        "upload physical value");
  check(hipMemcpy(physical_key_scales.ptr, fixture.physical_key_scales.data(),
                  fixture.physical_key_scales.size(), hipMemcpyHostToDevice),
        "upload physical key scales");
  check(hipMemcpy(physical_value_scales.ptr,
                  fixture.physical_value_scales.data(),
                  fixture.physical_value_scales.size(), hipMemcpyHostToDevice),
        "upload physical value scales");
}

void run_case(const Fixture &fixture, const uint64_t start_position,
              const uint32_t query_count, const bool prefill,
              const bool gqa_shared, DeviceBuffer<uint16_t> &query,
              DeviceBuffer<uint8_t> &key, DeviceBuffer<uint8_t> &value,
              DeviceBuffer<uint8_t> &key_scales,
              DeviceBuffer<uint8_t> &value_scales,
              DeviceBuffer<uint32_t> &logical_table,
              DeviceBuffer<uint8_t> &descriptor_table,
              DeviceBuffer<uint16_t> &paged_output,
              DeviceBuffer<uint16_t> &control_output,
              DeviceBuffer<float> &workspace, DeviceBuffer<uint32_t> &status,
              hipStream_t stream) {
  const uint64_t committed = start_position + query_count;
  const bool wave_local_kv = !gqa_shared;
  check(
      hipMemset(paged_output.ptr, 0xcd, paged_output.count * sizeof(uint16_t)),
      "clear paged output");
  check(hipMemset(control_output.ptr, 0xcd,
                  control_output.count * sizeof(uint16_t)),
        "clear control output");
  if (prefill) {
    check(sllm_causal_attention_kernel::launch_paged_prefill_gqa6(
              query.ptr, logical_table.ptr,
              reinterpret_cast<const sllm_paged_kv::BlockDescriptor *>(
                  descriptor_table.ptr),
              fixture.pages, fixture.pages, status.ptr, paged_output.ptr,
              query_count, start_position, committed, kQHeads, kKvHeads,
              kHeadDim, kEncoding, 1.0F, 1.0F, wave_local_kv, stream),
          "paged prefill");
    check(sllm_causal_attention_kernel::launch_gqa6_qtile8_w16(
              query.ptr, key.ptr, value.ptr, key_scales.ptr, value_scales.ptr,
              nullptr, nullptr, control_output.ptr, query_count, start_position,
              kQHeads, kKvHeads, kHeadDim, kEncoding, 1.0F, 1.0F, wave_local_kv,
              stream),
          "contiguous prefill");
  } else {
    check(sllm_causal_attention_kernel::launch_paged_decode_gqa6(
              query.ptr, logical_table.ptr,
              reinterpret_cast<const sllm_paged_kv::BlockDescriptor *>(
                  descriptor_table.ptr),
              fixture.pages, fixture.pages, status.ptr, workspace.ptr,
              workspace.count * sizeof(float), paged_output.ptr, query_count,
              start_position, committed, kQHeads, kKvHeads, kHeadDim, kEncoding,
              gqa_shared, stream),
          "paged decode");
    const uint32_t splits = committed >= 8192U ? 128U : 32U;
    if (splits == 128U) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
              sllm_causal_attention_kernel::
                  causal_attention_decode_gqa6_staged32_split_stage1_kernel<
                      128U>),
          dim3(query_count * kKvHeads * 128U), dim3(192U), 0U, stream,
          query.ptr, key.ptr, value.ptr, key_scales.ptr, value_scales.ptr,
          workspace.ptr, query_count, start_position, nullptr);
      check(hipGetLastError(), "contiguous decode stage1");
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
              sllm_causal_attention_kernel::
                  causal_attention_decode_gqa6_staged32_split_merge_kernel<
                      128U>),
          dim3(query_count * kQHeads), dim3(256U), 0U, stream, workspace.ptr,
          control_output.ptr, query_count, nullptr);
    } else {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
              sllm_causal_attention_kernel::
                  causal_attention_decode_gqa6_staged32_stage1_kernel),
          dim3(query_count * kKvHeads * 32U), dim3(192U), 0U, stream, query.ptr,
          key.ptr, value.ptr, key_scales.ptr, value_scales.ptr, workspace.ptr,
          query_count, start_position);
      check(hipGetLastError(), "contiguous decode stage1");
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
              sllm_causal_attention_kernel::
                  causal_attention_decode_wave_split_staged_stage2_kernel<32U>),
          dim3(query_count * kQHeads), dim3(256U), 0U, stream, workspace.ptr,
          control_output.ptr, query_count, kQHeads, kKvHeads, kHeadDim,
          nullptr);
    }
    check(hipGetLastError(), "contiguous decode merge");
  }
  check(hipStreamSynchronize(stream), "synchronize case");
  uint32_t status_value = 0U;
  check(hipMemcpy(&status_value, status.ptr, sizeof(status_value),
                  hipMemcpyDeviceToHost),
        "download paged status");
  if (status_value != 0U) {
    fail("paged descriptor validation status=" + std::to_string(status_value));
  }
  std::vector<uint16_t> paged(paged_output.count);
  std::vector<uint16_t> control(control_output.count);
  check(hipMemcpy(paged.data(), paged_output.ptr,
                  paged.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
        "download paged output");
  check(hipMemcpy(control.data(), control_output.ptr,
                  control.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
        "download control output");
  const size_t elements = static_cast<size_t>(query_count) * kQHeads * kHeadDim;
  if (prefill && start_position == kPrefillStart) {
    if (!std::equal(paged.begin(), paged.begin() + elements, control.begin())) {
      for (size_t index = 0U; index < elements; ++index) {
        if (paged[index] != control[index]) {
          fail("paged prefill differs from contiguous control index=" +
               std::to_string(index) +
               " paged=" + std::to_string(paged[index]) +
               " control=" + std::to_string(control[index]));
        }
      }
    }
  } else if (!prefill) {
    if (!std::equal(paged.begin(), paged.begin() + elements, control.begin())) {
      for (size_t index = 0U; index < elements; ++index) {
        if (paged[index] != control[index]) {
          fail("paged decode differs from contiguous control index=" +
               std::to_string(index) +
               " paged=" + std::to_string(paged[index]) +
               " control=" + std::to_string(control[index]));
        }
      }
    }
    if (start_position == 127U) {
      const double expected = host_oracle_dim0(fixture, start_position);
      const double actual = bf16_to_f32(paged[0]);
      if (!std::isfinite(actual) || std::fabs(actual - expected) > 0.08) {
        fail("paged decode host oracle mismatch expected=" +
             std::to_string(expected) + " actual=" + std::to_string(actual));
      }
    }
  }
}
} // namespace

int main() {
  try {
    check(hipSetDevice(0), "set device");
    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, 0), "device properties");
    const std::string actual_target(properties.gcnArchName);
    const bool gqa_shared = actual_target.rfind("gfx1030", 0U) == 0U;
    Fixture fixture;
    const size_t rows = static_cast<size_t>(kMaxTokens) * kKvHeads;
    DeviceBuffer<uint16_t> query(fixture.query.size());
    DeviceBuffer<uint8_t> key(fixture.key.size());
    DeviceBuffer<uint8_t> value(fixture.value.size());
    DeviceBuffer<uint8_t> key_scales(fixture.key_scales.size());
    DeviceBuffer<uint8_t> value_scales(fixture.value_scales.size());
    DeviceBuffer<uint8_t> physical_key(fixture.physical_key.size());
    DeviceBuffer<uint8_t> physical_value(fixture.physical_value.size());
    DeviceBuffer<uint8_t> physical_key_scales(
        fixture.physical_key_scales.size());
    DeviceBuffer<uint8_t> physical_value_scales(
        fixture.physical_value_scales.size());
    DeviceBuffer<uint8_t> descriptor_table(
        static_cast<size_t>(fixture.pages) *
        sizeof(sllm_paged_kv::BlockDescriptor));
    DeviceBuffer<uint32_t> logical_table(fixture.pages);
    DeviceBuffer<uint32_t> paged_status(1U);
    DeviceBuffer<uint16_t> paged_output(static_cast<size_t>(kPrefillRows) *
                                        kQHeads * kHeadDim);
    DeviceBuffer<uint16_t> control_output(paged_output.count);
    DeviceBuffer<float> workspace(static_cast<size_t>(5U) * kQHeads * 128U *
                                  (kHeadDim + 2U));
    upload(fixture, query, key, value, key_scales, value_scales, physical_key,
           physical_value, physical_key_scales, physical_value_scales);

    std::vector<uint32_t> logical_pages(fixture.pages);
    std::vector<sllm_paged_kv::BlockDescriptor> descriptors(fixture.pages);
    for (uint32_t logical_page = 0U; logical_page < fixture.pages;
         ++logical_page) {
      const uint32_t physical_page = fixture.pages - 1U - logical_page;
      logical_pages[logical_page] = physical_page;
      descriptors[physical_page] = {
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
                    logical_pages.size() * sizeof(logical_pages[0]),
                    hipMemcpyHostToDevice),
          "upload logical table");
    check(hipMemcpy(descriptor_table.ptr, descriptors.data(),
                    descriptors.size() * sizeof(descriptors[0]),
                    hipMemcpyHostToDevice),
          "upload descriptor table");

    hipStream_t stream = nullptr;
    check(hipStreamCreate(&stream), "create stream");
    run_case(fixture, 127U, 3U, false, gqa_shared, query, key, value,
             key_scales, value_scales, logical_table, descriptor_table,
             paged_output, control_output, workspace, paged_status, stream);
    run_case(fixture, kPrefillStart, 5U, false, gqa_shared, query, key, value,
             key_scales, value_scales, logical_table, descriptor_table,
             paged_output, control_output, workspace, paged_status, stream);
    run_case(fixture, kPrefillStart, kPrefillRows, true, gqa_shared, query, key,
             value, key_scales, value_scales, logical_table, descriptor_table,
             paged_output, control_output, workspace, paged_status, stream);
    const uint32_t invalid_page = UINT32_MAX;
    check(hipMemcpy(logical_table.ptr, &invalid_page, sizeof(invalid_page),
                    hipMemcpyHostToDevice),
          "upload invalid logical page");
    check(sllm_causal_attention_kernel::launch_paged_decode_gqa6(
              query.ptr, logical_table.ptr,
              reinterpret_cast<const sllm_paged_kv::BlockDescriptor *>(
                  descriptor_table.ptr),
              fixture.pages, fixture.pages, paged_status.ptr, workspace.ptr,
              workspace.count * sizeof(float), paged_output.ptr, 1U, 0U, 1U,
              kQHeads, kKvHeads, kHeadDim, kEncoding, gqa_shared, stream),
          "paged invalid-id launch");
    check(hipStreamSynchronize(stream), "invalid-id synchronize");
    uint32_t invalid_status = 0U;
    check(hipMemcpy(&invalid_status, paged_status.ptr, sizeof(invalid_status),
                    hipMemcpyDeviceToHost),
          "download invalid-id status");
    if (invalid_status != 1U) {
      fail("invalid physical ID was not rejected");
    }
    check(hipStreamDestroy(stream), "destroy stream");
    check(hipDeviceSynchronize(), "final synchronize");
    std::printf(
        "phase87-stage10-paged-attention target=%s cases=4 state=PASS\n",
#if defined(__gfx1030__)
        "gfx1030"
#elif defined(__gfx1201__)
        "gfx1201"
#else
        "unknown"
#endif
    );
    (void)rows;
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "phase87-stage10-paged-attention FAIL: %s\n",
                 error.what());
    return 1;
  }
}
