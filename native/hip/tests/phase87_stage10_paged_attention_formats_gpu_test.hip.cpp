// Phase 87 Stage 10: paged attention format coverage.
//
// This focused probe exercises the format-generic paged control path for the
// currently usable non-FP16 formats.  It deliberately uses a reverse logical
// table, a non-aligned GQA geometry (6 query heads, 2 KV heads, head_dim=129),
// and the 127/128/129 page boundaries.  The paged result must be bitwise equal
// to the generic contiguous kernel; selected dimensions are also checked with
// an independent host online-softmax calculation.
#define SLLM_PUBLIC_RUNTIME_HOST_TEST 1
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#include "../src/causal_attention_kernel.hip.cpp"
#pragma clang diagnostic pop

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint32_t kQHeads = 6U;
constexpr uint32_t kKvHeads = 2U;
constexpr uint32_t kHeadDim = 129U;
constexpr uint32_t kPageTokens = 128U;
constexpr uint32_t kPages = 3U;
constexpr uint32_t kMaxRows = 3U;
constexpr uint32_t kMaxTokens = kPages * kPageTokens;
constexpr float kDynamicKeyScale = 0.75F;
constexpr float kDynamicValueScale = 1.25F;
constexpr float kStaticKeyScale = 1.25F;
constexpr float kStaticValueScale = 0.75F;
constexpr float kScoreScale = 1.0F / 11.357816691600547F;

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

  explicit DeviceBuffer(const size_t elements) : count(elements) {
    if (elements == 0U) {
      fail("zero-sized device allocation");
    }
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

template <typename T> struct OptionalDeviceBuffer final {
  std::unique_ptr<DeviceBuffer<T>> allocation;

  OptionalDeviceBuffer() = default;
  explicit OptionalDeviceBuffer(const size_t elements) {
    if (elements != 0U) {
      allocation = std::make_unique<DeviceBuffer<T>>(elements);
    }
  }
  T *get() const noexcept {
    return allocation == nullptr ? nullptr : allocation->ptr;
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

float e4m3_decode(const uint8_t bits) {
  const float sign = (bits & 0x80U) != 0U ? -1.0F : 1.0F;
  const uint32_t magnitude = bits & 0x7fU;
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & 7U;
  if (exponent == 0U) {
    return sign * static_cast<float>(mantissa) * std::ldexp(1.0F, -9);
  }
  if (magnitude == 0x7fU) {
    return NAN;
  }
  return sign * std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                           static_cast<int>(exponent) - 7);
}

float e5m2_decode(const uint8_t bits) {
  uint16_t half_bits = static_cast<uint16_t>(bits) << 8U;
  return f16_to_f32(half_bits);
}

float e2m1_decode(const uint8_t bits) {
  constexpr float magnitudes[] = {0.0F, 0.5F, 1.0F, 1.5F,
                                  2.0F, 3.0F, 4.0F, 6.0F};
  const float magnitude = magnitudes[bits & 7U];
  return (bits & 8U) == 0U ? magnitude : -magnitude;
}

struct FormatSpec final {
  uint32_t encoding = 0U;
  uint32_t value_stride = 0U;
  uint32_t scale_stride = 0U;
  bool dynamic_scale = false;
  bool static_scale = false;
  bool nvfp4 = false;
  bool mxfp8 = false;
  bool e5 = false;
};

FormatSpec format_spec(const uint32_t encoding) {
  if (encoding == SLLM_HIP_KV_ENCODING_FP8_V1) {
    return {encoding, kHeadDim, sizeof(float), true,
            false,    false,    false,         false};
  }
  if (encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1) {
    return {encoding, kHeadDim, 0U, false, true, false, false, false};
  }
  if (encoding == SLLM_HIP_KV_ENCODING_NVFP4_V1) {
    return {encoding,
            (kHeadDim + 1U) / 2U,
            (kHeadDim + 15U) / 16U,
            false,
            false,
            true,
            false,
            false};
  }
  if (encoding == SLLM_HIP_KV_ENCODING_MXFP8_E4_V1 ||
      encoding == SLLM_HIP_KV_ENCODING_MXFP8_E5_V1) {
    return {encoding,
            ((kHeadDim + 31U) / 32U) * 32U,
            (kHeadDim + 31U) / 32U,
            false,
            false,
            false,
            true,
            encoding == SLLM_HIP_KV_ENCODING_MXFP8_E5_V1};
  }
  fail("unsupported format spec");
}

struct Fixture final {
  const FormatSpec spec;
  const size_t rows = static_cast<size_t>(kMaxTokens) * kKvHeads;
  const size_t page_rows = static_cast<size_t>(kPageTokens) * kKvHeads;
  std::vector<uint16_t> query;
  std::vector<uint8_t> key;
  std::vector<uint8_t> value;
  std::vector<uint8_t> key_scale;
  std::vector<uint8_t> value_scale;
  std::vector<float> key_outer;
  std::vector<float> value_outer;
  std::vector<uint8_t> physical_key;
  std::vector<uint8_t> physical_value;
  std::vector<uint8_t> physical_key_scale;
  std::vector<uint8_t> physical_value_scale;
  std::vector<float> physical_key_outer;
  std::vector<float> physical_value_outer;

  explicit Fixture(const FormatSpec format)
      : spec(format), query(static_cast<size_t>(kMaxRows) * kQHeads * kHeadDim),
        key(rows * spec.value_stride), value(rows * spec.value_stride),
        key_scale(rows * spec.scale_stride),
        value_scale(rows * spec.scale_stride),
        key_outer(spec.nvfp4 ? rows : 0U), value_outer(spec.nvfp4 ? rows : 0U),
        physical_key(static_cast<size_t>(kPages) * page_rows *
                     spec.value_stride),
        physical_value(physical_key.size()),
        physical_key_scale(static_cast<size_t>(kPages) * page_rows *
                           spec.scale_stride),
        physical_value_scale(physical_key_scale.size()),
        physical_key_outer(spec.nvfp4 ? static_cast<size_t>(kPages) * page_rows
                                      : 0U),
        physical_value_outer(physical_key_outer.size()) {
    for (size_t index = 0U; index < query.size(); ++index) {
      query[index] =
          f32_to_bf16(0.011F * static_cast<float>((index * 19U) % 31U) - 0.16F);
    }
    for (uint32_t token = 0U; token < kMaxTokens; ++token) {
      const uint32_t logical_page = token / kPageTokens;
      const uint32_t local_token = token % kPageTokens;
      const uint32_t physical_page = kPages - 1U - logical_page;
      for (uint32_t head = 0U; head < kKvHeads; ++head) {
        const size_t row =
            (static_cast<size_t>(token) * kKvHeads + head) * spec.value_stride;
        const size_t physical_row =
            (static_cast<size_t>(physical_page) * kPageTokens + local_token) *
                kKvHeads * spec.value_stride +
            static_cast<size_t>(head) * spec.value_stride;
        fill_row(key, row, token, head, false);
        fill_row(value, row, token, head, true);
        std::memcpy(physical_key.data() + physical_row, key.data() + row,
                    spec.value_stride);
        std::memcpy(physical_value.data() + physical_row, value.data() + row,
                    spec.value_stride);
        if (spec.dynamic_scale) {
          const float key_scale_value =
              kDynamicKeyScale + 0.125F * static_cast<float>(head);
          const float value_scale_value =
              kDynamicValueScale - 0.125F * static_cast<float>(head);
          std::memcpy(key_scale.data() +
                          (static_cast<size_t>(token) * kKvHeads + head) *
                              spec.scale_stride,
                      &key_scale_value, sizeof(key_scale_value));
          std::memcpy(value_scale.data() +
                          (static_cast<size_t>(token) * kKvHeads + head) *
                              spec.scale_stride,
                      &value_scale_value, sizeof(value_scale_value));
        } else if (spec.nvfp4) {
          const size_t scale_row =
              (static_cast<size_t>(token) * kKvHeads + head) *
              spec.scale_stride;
          std::fill(key_scale.begin() + static_cast<ptrdiff_t>(scale_row),
                    key_scale.begin() +
                        static_cast<ptrdiff_t>(scale_row + spec.scale_stride),
                    static_cast<uint8_t>(0x38U));
          std::fill(value_scale.begin() + static_cast<ptrdiff_t>(scale_row),
                    value_scale.begin() +
                        static_cast<ptrdiff_t>(scale_row + spec.scale_stride),
                    static_cast<uint8_t>(0x38U));
          key_outer[static_cast<size_t>(token) * kKvHeads + head] = 1.0F;
          value_outer[static_cast<size_t>(token) * kKvHeads + head] = 1.0F;
        } else if (spec.mxfp8) {
          const size_t scale_row =
              (static_cast<size_t>(token) * kKvHeads + head) *
              spec.scale_stride;
          std::fill(key_scale.begin() + static_cast<ptrdiff_t>(scale_row),
                    key_scale.begin() +
                        static_cast<ptrdiff_t>(scale_row + spec.scale_stride),
                    static_cast<uint8_t>(127U));
          std::fill(value_scale.begin() + static_cast<ptrdiff_t>(scale_row),
                    value_scale.begin() +
                        static_cast<ptrdiff_t>(scale_row + spec.scale_stride),
                    static_cast<uint8_t>(127U));
        }
        const size_t physical_scale_row =
            (static_cast<size_t>(physical_page) * kPageTokens + local_token) *
                kKvHeads * spec.scale_stride +
            static_cast<size_t>(head) * spec.scale_stride;
        if (spec.scale_stride != 0U) {
          std::memcpy(physical_key_scale.data() + physical_scale_row,
                      key_scale.data() +
                          (static_cast<size_t>(token) * kKvHeads + head) *
                              spec.scale_stride,
                      spec.scale_stride);
          std::memcpy(physical_value_scale.data() + physical_scale_row,
                      value_scale.data() +
                          (static_cast<size_t>(token) * kKvHeads + head) *
                              spec.scale_stride,
                      spec.scale_stride);
        }
        if (spec.nvfp4) {
          const size_t logical_outer =
              static_cast<size_t>(token) * kKvHeads + head;
          const size_t physical_outer =
              static_cast<size_t>(physical_page) * kPageTokens * kKvHeads +
              static_cast<size_t>(local_token) * kKvHeads + head;
          physical_key_outer[physical_outer] = key_outer[logical_outer];
          physical_value_outer[physical_outer] = value_outer[logical_outer];
        }
      }
    }
  }

  void fill_row(std::vector<uint8_t> &destination, const size_t row,
                const uint32_t token, const uint32_t head,
                const bool value_row) const {
    if (spec.nvfp4) {
      std::fill(destination.begin() + static_cast<ptrdiff_t>(row),
                destination.begin() +
                    static_cast<ptrdiff_t>(row + spec.value_stride),
                0U);
      for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
        const uint8_t code = static_cast<uint8_t>(
            ((token * 3U + head * 5U + dimension * (value_row ? 3U : 1U)) %
             7U) +
            1U);
        uint8_t &packed = destination[row + dimension / 2U];
        packed = static_cast<uint8_t>(
            packed | static_cast<uint8_t>(code << ((dimension & 1U) * 4U)));
      }
      return;
    }
    for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
      const uint8_t base = static_cast<uint8_t>(
          0x20U +
          ((token * 7U + head * 11U + dimension * (value_row ? 5U : 3U)) %
           0x30U));
      destination[row + dimension] =
          spec.e5 ? static_cast<uint8_t>(0x30U + (base & 0x0fU))
                  : static_cast<uint8_t>(base);
    }
    for (uint32_t dimension = kHeadDim; dimension < spec.value_stride;
         ++dimension) {
      destination[row + dimension] = 0U;
    }
  }

  float decode(const std::vector<uint8_t> &source,
               const std::vector<uint8_t> &scales,
               const std::vector<float> &outer, const uint32_t token,
               const uint32_t head, const uint32_t dimension,
               const bool key_row) const {
    const size_t row =
        (static_cast<size_t>(token) * kKvHeads + head) * spec.value_stride;
    if (spec.nvfp4) {
      const uint8_t code = static_cast<uint8_t>(
          (source[row + dimension / 2U] >> ((dimension & 1U) * 4U)) & 0x0fU);
      const size_t scale_row =
          (static_cast<size_t>(token) * kKvHeads + head) * spec.scale_stride;
      return e2m1_decode(code) *
             e4m3_decode(scales[scale_row + dimension / 16U]) *
             outer[static_cast<size_t>(token) * kKvHeads + head];
    }
    float scale = 1.0F;
    if (spec.dynamic_scale) {
      const size_t scale_row =
          (static_cast<size_t>(token) * kKvHeads + head) * spec.scale_stride;
      std::memcpy(&scale, scales.data() + scale_row, sizeof(scale));
    } else if (spec.static_scale) {
      scale = key_row ? kStaticKeyScale : kStaticValueScale;
    } else if (spec.mxfp8) {
      const size_t scale_row =
          (static_cast<size_t>(token) * kKvHeads + head) * spec.scale_stride;
      scale = std::ldexp(
          1.0F, static_cast<int>(scales[scale_row + dimension / 32U]) - 127);
    }
    const uint8_t code = source[row + dimension];
    return (spec.e5 ? e5m2_decode(code) : e4m3_decode(code)) * scale;
  }
};

float oracle_dim(const Fixture &fixture, const uint16_t *const query,
                 const uint64_t start_position, const uint32_t row,
                 const uint32_t query_head, const uint32_t dimension) {
  const uint64_t query_position = start_position + row;
  const uint32_t kv_head = query_head / (kQHeads / kKvHeads);
  const uint16_t *const query_row =
      query + (static_cast<size_t>(row) * kQHeads + query_head) * kHeadDim;
  double maximum = -std::numeric_limits<double>::infinity();
  std::vector<double> scores(static_cast<size_t>(query_position) + 1U);
  for (uint64_t token = 0U; token <= query_position; ++token) {
    double dot = 0.0;
    for (uint32_t current = 0U; current < kHeadDim; ++current) {
      dot += static_cast<double>(bf16_to_f32(query_row[current])) *
             static_cast<double>(fixture.decode(
                 fixture.key, fixture.key_scale, fixture.key_outer,
                 static_cast<uint32_t>(token), kv_head, current, true));
    }
    scores[token] = dot * static_cast<double>(kScoreScale);
    maximum = std::max(maximum, scores[token]);
  }
  double denominator = 0.0;
  double numerator = 0.0;
  for (uint64_t token = 0U; token <= query_position; ++token) {
    const double weight = std::exp(scores[token] - maximum);
    denominator += weight;
    numerator +=
        weight * static_cast<double>(fixture.decode(
                     fixture.value, fixture.value_scale, fixture.value_outer,
                     static_cast<uint32_t>(token), kv_head, dimension, false));
  }
  return static_cast<float>(numerator / denominator);
}

void upload(const Fixture &fixture, DeviceBuffer<uint16_t> &query,
            DeviceBuffer<uint8_t> &key, DeviceBuffer<uint8_t> &value,
            OptionalDeviceBuffer<uint8_t> &key_scale,
            OptionalDeviceBuffer<uint8_t> &value_scale,
            OptionalDeviceBuffer<float> &key_outer,
            OptionalDeviceBuffer<float> &value_outer,
            DeviceBuffer<uint8_t> &physical_key,
            DeviceBuffer<uint8_t> &physical_value,
            OptionalDeviceBuffer<uint8_t> &physical_key_scale,
            OptionalDeviceBuffer<uint8_t> &physical_value_scale,
            OptionalDeviceBuffer<float> &physical_key_outer,
            OptionalDeviceBuffer<float> &physical_value_outer) {
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
  auto copy_optional = [](auto &destination, const auto &source,
                          const char *const operation) {
    if (destination.get() != nullptr) {
      check(hipMemcpy(destination.get(), source.data(),
                      source.size() * sizeof(source[0]), hipMemcpyHostToDevice),
            operation);
    }
  };
  copy_optional(key_scale, fixture.key_scale, "upload key scale");
  copy_optional(value_scale, fixture.value_scale, "upload value scale");
  copy_optional(key_outer, fixture.key_outer, "upload key outer scale");
  copy_optional(value_outer, fixture.value_outer, "upload value outer scale");
  check(hipMemcpy(physical_key.ptr, fixture.physical_key.data(),
                  fixture.physical_key.size(), hipMemcpyHostToDevice),
        "upload physical key");
  check(hipMemcpy(physical_value.ptr, fixture.physical_value.data(),
                  fixture.physical_value.size(), hipMemcpyHostToDevice),
        "upload physical value");
  copy_optional(physical_key_scale, fixture.physical_key_scale,
                "upload physical key scale");
  copy_optional(physical_value_scale, fixture.physical_value_scale,
                "upload physical value scale");
  copy_optional(physical_key_outer, fixture.physical_key_outer,
                "upload physical key outer");
  copy_optional(physical_value_outer, fixture.physical_value_outer,
                "upload physical value outer");
}

template <uint32_t Encoding>
void run_case_encoding(
    const Fixture &fixture, DeviceBuffer<uint16_t> &query,
    DeviceBuffer<uint8_t> &key, DeviceBuffer<uint8_t> &value,
    OptionalDeviceBuffer<uint8_t> &key_scale,
    OptionalDeviceBuffer<uint8_t> &value_scale,
    OptionalDeviceBuffer<float> &key_outer,
    OptionalDeviceBuffer<float> &value_outer,
    DeviceBuffer<uint32_t> &logical_table,
    DeviceBuffer<sllm_paged_kv::BlockDescriptor> &descriptors,
    DeviceBuffer<uint32_t> &status, DeviceBuffer<uint16_t> &paged,
    DeviceBuffer<uint16_t> &control, const uint32_t rows,
    const uint64_t start_position, hipStream_t stream) {
  const uint64_t committed = start_position + rows;
  check(sllm_causal_attention_kernel::launch_paged_attention(
            query.ptr, logical_table.ptr, descriptors.ptr, kPages, kPages,
            status.ptr, paged.ptr, rows, start_position, committed, kQHeads,
            kKvHeads, kHeadDim, fixture.spec.encoding,
            fixture.spec.static_scale ? kStaticKeyScale : 1.0F,
            fixture.spec.static_scale ? kStaticValueScale : 1.0F, kScoreScale,
            stream),
        "launch paged attention format");
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(
          sllm_causal_attention_kernel::causal_attention_kernel<false,
                                                                Encoding>),
      dim3(rows * kQHeads), dim3(256U), 0U, stream, query.ptr, key.ptr,
      value.ptr, key_scale.get(), value_scale.get(), key_outer.get(),
      value_outer.get(), control.ptr, rows, kMaxTokens, start_position,
      committed, kQHeads, kKvHeads, kHeadDim,
      fixture.spec.static_scale ? kStaticKeyScale : 1.0F,
      fixture.spec.static_scale ? kStaticValueScale : 1.0F, 0U, kScoreScale);
  check(hipGetLastError(), "launch contiguous attention format");
  check(hipStreamSynchronize(stream), "synchronize attention format");
  uint32_t status_value = 0U;
  check(hipMemcpy(&status_value, status.ptr, sizeof(status_value),
                  hipMemcpyDeviceToHost),
        "download paged status");
  if (status_value != 0U) {
    fail("paged format descriptor validation status=" +
         std::to_string(status_value));
  }
  std::vector<uint16_t> paged_host(paged.count);
  std::vector<uint16_t> control_host(control.count);
  check(hipMemcpy(paged_host.data(), paged.ptr,
                  paged_host.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
        "download paged format output");
  check(hipMemcpy(control_host.data(), control.ptr,
                  control_host.size() * sizeof(uint16_t),
                  hipMemcpyDeviceToHost),
        "download contiguous format output");
  const size_t elements = static_cast<size_t>(rows) * kQHeads * kHeadDim;
  if (!std::equal(paged_host.begin(), paged_host.begin() + elements,
                  control_host.begin())) {
    for (size_t index = 0U; index < elements; ++index) {
      if (paged_host[index] != control_host[index]) {
        fail("paged format differs from contiguous index=" +
             std::to_string(index));
      }
    }
  }
  const uint16_t *const query_host = fixture.query.data();
  for (uint32_t row = 0U; row < rows; ++row) {
    for (const uint32_t head : {0U, kQHeads - 1U}) {
      for (const uint32_t dimension : {0U, 128U}) {
        const float expected = oracle_dim(fixture, query_host, start_position,
                                          row, head, dimension);
        const float actual = bf16_to_f32(
            paged_host[(static_cast<size_t>(row) * kQHeads + head) * kHeadDim +
                       dimension]);
        if (!std::isfinite(actual) || std::fabs(actual - expected) > 0.2F) {
          fail("paged format oracle mismatch expected=" +
               std::to_string(expected) + " actual=" + std::to_string(actual) +
               " encoding=" + std::to_string(fixture.spec.encoding) +
               " start=" + std::to_string(start_position) +
               " rows=" + std::to_string(rows) + " row=" + std::to_string(row) +
               " head=" + std::to_string(head) +
               " dim=" + std::to_string(dimension) + " control=" +
               std::to_string(bf16_to_f32(
                   control_host[(static_cast<size_t>(row) * kQHeads + head) *
                                    kHeadDim +
                                dimension])));
        }
      }
    }
  }
}

template <uint32_t Encoding>
void run_format(const uint32_t encoding, hipStream_t stream) {
  const Fixture fixture(format_spec(encoding));
  DeviceBuffer<uint16_t> query(fixture.query.size());
  DeviceBuffer<uint8_t> key(fixture.key.size());
  DeviceBuffer<uint8_t> value(fixture.value.size());
  OptionalDeviceBuffer<uint8_t> key_scale(fixture.key_scale.size());
  OptionalDeviceBuffer<uint8_t> value_scale(fixture.value_scale.size());
  OptionalDeviceBuffer<float> key_outer(fixture.key_outer.size());
  OptionalDeviceBuffer<float> value_outer(fixture.value_outer.size());
  DeviceBuffer<uint8_t> physical_key(fixture.physical_key.size());
  DeviceBuffer<uint8_t> physical_value(fixture.physical_value.size());
  OptionalDeviceBuffer<uint8_t> physical_key_scale(
      fixture.physical_key_scale.size());
  OptionalDeviceBuffer<uint8_t> physical_value_scale(
      fixture.physical_value_scale.size());
  OptionalDeviceBuffer<float> physical_key_outer(
      fixture.physical_key_outer.size());
  OptionalDeviceBuffer<float> physical_value_outer(
      fixture.physical_value_outer.size());
  DeviceBuffer<uint32_t> logical_table(kPages);
  DeviceBuffer<sllm_paged_kv::BlockDescriptor> descriptors(kPages);
  DeviceBuffer<uint32_t> status(1U);
  DeviceBuffer<uint16_t> paged(static_cast<size_t>(kMaxRows) * kQHeads *
                               kHeadDim);
  DeviceBuffer<uint16_t> control(paged.count);
  upload(fixture, query, key, value, key_scale, value_scale, key_outer,
         value_outer, physical_key, physical_value, physical_key_scale,
         physical_value_scale, physical_key_outer, physical_value_outer);

  std::vector<uint32_t> logical_pages(kPages);
  std::vector<sllm_paged_kv::BlockDescriptor> descriptor_host(kPages);
  for (uint32_t logical_page = 0U; logical_page < kPages; ++logical_page) {
    const uint32_t physical_page = kPages - 1U - logical_page;
    logical_pages[logical_page] = physical_page;
    descriptor_host[physical_page] = {
        physical_key.ptr + static_cast<size_t>(physical_page) *
                               fixture.page_rows * fixture.spec.value_stride,
        physical_value.ptr + static_cast<size_t>(physical_page) *
                                 fixture.page_rows * fixture.spec.value_stride,
        physical_key_scale.get() == nullptr
            ? nullptr
            : physical_key_scale.get() + static_cast<size_t>(physical_page) *
                                             fixture.page_rows *
                                             fixture.spec.scale_stride,
        physical_value_scale.get() == nullptr
            ? nullptr
            : physical_value_scale.get() + static_cast<size_t>(physical_page) *
                                               fixture.page_rows *
                                               fixture.spec.scale_stride,
        physical_key_outer.get() == nullptr
            ? nullptr
            : reinterpret_cast<uint8_t *>(physical_key_outer.get() +
                                          static_cast<size_t>(physical_page) *
                                              fixture.page_rows),
        physical_value_outer.get() == nullptr
            ? nullptr
            : reinterpret_cast<uint8_t *>(physical_value_outer.get() +
                                          static_cast<size_t>(physical_page) *
                                              fixture.page_rows)};
  }
  check(hipMemcpy(logical_table.ptr, logical_pages.data(),
                  logical_pages.size() * sizeof(uint32_t),
                  hipMemcpyHostToDevice),
        "upload format logical table");
  check(hipMemcpy(descriptors.ptr, descriptor_host.data(),
                  descriptor_host.size() * sizeof(descriptor_host[0]),
                  hipMemcpyHostToDevice),
        "upload format descriptors");

  for (const uint64_t start : {127U, 128U, 129U}) {
    for (const uint32_t row_count : {1U, 3U}) {
      check(hipMemset(paged.ptr, 0xcd, paged.count * sizeof(uint16_t)),
            "clear paged format output");
      check(hipMemset(control.ptr, 0xcd, control.count * sizeof(uint16_t)),
            "clear contiguous format output");
      run_case_encoding<Encoding>(fixture, query, key, value, key_scale,
                                  value_scale, key_outer, value_outer,
                                  logical_table, descriptors, status, paged,
                                  control, row_count, start, stream);
    }
  }

  const uint32_t invalid_page = UINT32_MAX;
  check(hipMemcpy(logical_table.ptr, &invalid_page, sizeof(invalid_page),
                  hipMemcpyHostToDevice),
        "upload invalid format logical ID");
  check(hipMemset(paged.ptr, 0xcd, paged.count * sizeof(uint16_t)),
        "clear invalid format output");
  check(sllm_causal_attention_kernel::launch_paged_attention(
            query.ptr, logical_table.ptr, descriptors.ptr, kPages, kPages,
            status.ptr, paged.ptr, 1U, 0U, 1U, kQHeads, kKvHeads, kHeadDim,
            fixture.spec.encoding,
            fixture.spec.static_scale ? kStaticKeyScale : 1.0F,
            fixture.spec.static_scale ? kStaticValueScale : 1.0F, kScoreScale,
            stream),
        "launch invalid format descriptor");
  check(hipStreamSynchronize(stream), "synchronize invalid format descriptor");
  uint32_t invalid_status = 0U;
  check(hipMemcpy(&invalid_status, status.ptr, sizeof(invalid_status),
                  hipMemcpyDeviceToHost),
        "download invalid format status");
  if (invalid_status == 0U) {
    fail("invalid format logical ID did not set device status");
  }
  std::vector<uint16_t> invalid_output(paged.count);
  check(hipMemcpy(invalid_output.data(), paged.ptr,
                  invalid_output.size() * sizeof(uint16_t),
                  hipMemcpyDeviceToHost),
        "download invalid format output");
  const size_t invalid_elements = static_cast<size_t>(kQHeads) * kHeadDim;
  if (!std::all_of(invalid_output.begin(),
                   invalid_output.begin() + invalid_elements,
                   [](const uint16_t value) { return value == 0xcdcdU; })) {
    fail("invalid format descriptor wrote output or used a fallback");
  }
}
} // namespace

int main() {
  try {
    check(hipSetDevice(0), "set device");
    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, 0), "device properties");
    const std::string target(properties.gcnArchName);
    if (target.rfind("gfx1030", 0U) != 0U &&
        target.rfind("gfx1201", 0U) != 0U) {
      std::fprintf(stderr,
                   "phase87-stage10-paged-attention-formats SKIP target=%s\n",
                   target.c_str());
      return 0;
    }
    hipStream_t stream = nullptr;
    check(hipStreamCreate(&stream), "create stream");
    run_format<SLLM_HIP_KV_ENCODING_FP8_STATIC_V1>(
        SLLM_HIP_KV_ENCODING_FP8_STATIC_V1, stream);
    run_format<SLLM_HIP_KV_ENCODING_FP8_V1>(SLLM_HIP_KV_ENCODING_FP8_V1,
                                            stream);
    run_format<SLLM_HIP_KV_ENCODING_NVFP4_V1>(SLLM_HIP_KV_ENCODING_NVFP4_V1,
                                              stream);
    run_format<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1>(
        SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, stream);
    if (target.rfind("gfx1030", 0U) == 0U) {
      run_format<SLLM_HIP_KV_ENCODING_MXFP8_E5_V1>(
          SLLM_HIP_KV_ENCODING_MXFP8_E5_V1, stream);
    }
    check(hipStreamDestroy(stream), "destroy stream");
    std::printf(
        "phase87-stage10-paged-attention-formats target=%s state=PASS\n",
        target.c_str());
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "phase87-stage10-paged-attention-formats FAIL: %s\n",
                 error.what());
    return 1;
  }
}
