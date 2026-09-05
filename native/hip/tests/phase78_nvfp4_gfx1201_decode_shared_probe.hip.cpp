// Portions derived from llama.cpp.
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-phase78-nvfp4-byte-permute-001
// Upstream: https://github.com/ggml-org/llama.cpp @
// 3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70, ggml/src/ggml-cuda/vecdotq.cuh
// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT

// Phase 78 standalone gfx1201 NVFP4 W4A4 decode activation-sharing probe.
//
// ID67's wave4col32 geometry is kept as the control.  The two candidates
// decode the activation row and its block16 E4M3 scales once into workgroup
// LDS, then reuse those values across all eight output waves.  The arithmetic
// contract is unchanged: four signed byte dot4 operations per block, /4 for
// the value*2 packing, block scale multiplication in FP32, tensor scales, and
// BF16 RNE.  This file is evidence-only and does not enter production.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

namespace {

constexpr uint32_t kThreads = 256U;
constexpr uint32_t kWave = 32U;
constexpr uint32_t kWaves = 8U;
constexpr uint32_t kColumnsPerWave = 4U;
constexpr uint32_t kColumnsPerBlock = 32U;
constexpr uint32_t kWarmups = 3U;
constexpr uint32_t kMeasured = 10U;

enum class DotKind : uint32_t { Sdot4, Sudot4 };

struct PackedI8Pair final {
  uint32_t even;
  uint32_t odd;
};

__device__ __forceinline__ PackedI8Pair scaled_pack(const uint32_t packed) {
  constexpr uint32_t table_0_3 = UINT32_C(0x03020100);
  constexpr uint32_t table_4_7 = UINT32_C(0x0c080604);
  constexpr uint32_t table_8_11 = UINT32_C(0xfdfeff00);
  constexpr uint32_t table_12_15 = UINT32_C(0xf4f8fafc);
  constexpr uint32_t low_mask = UINT32_C(0x07070707);
  constexpr uint32_t identity = UINT32_C(0x03020100);
  constexpr uint32_t sign_mask = UINT32_C(0x08080808);
  const uint32_t even_indices = packed;
  const uint32_t odd_indices = packed >> 4U;
  const uint32_t even_low =
      __builtin_amdgcn_perm(table_4_7, table_0_3, even_indices & low_mask);
  const uint32_t odd_low =
      __builtin_amdgcn_perm(table_4_7, table_0_3, odd_indices & low_mask);
  const uint32_t even_high =
      __builtin_amdgcn_perm(table_12_15, table_8_11, even_indices & low_mask);
  const uint32_t odd_high =
      __builtin_amdgcn_perm(table_12_15, table_8_11, odd_indices & low_mask);
  const uint32_t even_select = identity | ((even_indices & sign_mask) >> 1U);
  const uint32_t odd_select = identity | ((odd_indices & sign_mask) >> 1U);
  return {__builtin_amdgcn_perm(even_high, even_low, even_select),
          __builtin_amdgcn_perm(odd_high, odd_low, odd_select)};
}

template <DotKind Kind>
__device__ __forceinline__ int32_t dot4(const uint32_t lhs, const uint32_t rhs,
                                        const int32_t accumulator) {
  if constexpr (Kind == DotKind::Sudot4) {
#if __has_builtin(__builtin_amdgcn_sudot4)
    return __builtin_amdgcn_sudot4(true, lhs, true, rhs, accumulator, false);
#else
    return accumulator;
#endif
  } else {
#if __has_builtin(__builtin_amdgcn_sdot4)
    return __builtin_amdgcn_sdot4(lhs, rhs, accumulator, false);
#else
    int32_t result = accumulator;
#pragma unroll
    for (uint32_t i = 0U; i < 4U; ++i)
      result += static_cast<int8_t>(lhs >> (i * 8U)) *
                static_cast<int8_t>(rhs >> (i * 8U));
    return result;
#endif
  }
}

__device__ __forceinline__ float e4m3(const uint8_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x80U) << 24U;
  const uint32_t magnitude = bits & 0x7fU;
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & 7U;
  if (exponent == 0U) {
    if (mantissa == 0U)
      return __uint_as_float(sign);
    const float value = static_cast<float>(mantissa) * 0x1p-9F;
    return __uint_as_float(__float_as_uint(value) | sign);
  }
  if (magnitude == 0x7fU)
    return __uint_as_float(sign | UINT32_C(0x7fc00000));
  return __uint_as_float(sign | ((exponent + 120U) << 23U) | (mantissa << 20U));
}

__device__ __forceinline__ uint16_t bf16_rne(const float value) {
  const uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7f800000U) == 0x7f800000U) {
    if ((bits & 0x007fffffU) != 0U)
      return static_cast<uint16_t>(((bits >> 16U) & 0x8000U) | 0x7fc0U |
                                   ((bits >> 16U) & 0x003fU));
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & 0xffffU;
  if (lower > 0x8000U || (lower == 0x8000U && (upper & 1U) != 0U))
    ++upper;
  return static_cast<uint16_t>(upper);
}

template <DotKind Kind, bool ActivationLds>
__global__ __launch_bounds__(kThreads, 1) void decode_kernel(
    const uint8_t *const activation, const uint8_t *const activation_scales,
    const uint8_t *const weight, const uint8_t *const weight_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint32_t blocks_per_row, const uint32_t n) {
  extern __shared__ uint8_t shared[];
  uint32_t *const activation_words_lds = reinterpret_cast<uint32_t *>(shared);
  float *const activation_scales_lds = reinterpret_cast<float *>(
      shared + static_cast<uint64_t>(blocks_per_row) * 16U);
  if constexpr (ActivationLds) {
    for (uint32_t block = threadIdx.x; block < blocks_per_row;
         block += kThreads) {
      const uint8_t *const source =
          activation + static_cast<uint64_t>(block) * 8U;
      const uint32_t word0 = __builtin_nontemporal_load(
          reinterpret_cast<const uint32_t *>(source));
      const uint32_t word1 = __builtin_nontemporal_load(
          reinterpret_cast<const uint32_t *>(source + 4U));
      const PackedI8Pair first = scaled_pack(word0);
      const PackedI8Pair second = scaled_pack(word1);
      activation_words_lds[static_cast<uint64_t>(block) * 4U + 0U] = first.even;
      activation_words_lds[static_cast<uint64_t>(block) * 4U + 1U] = first.odd;
      activation_words_lds[static_cast<uint64_t>(block) * 4U + 2U] =
          second.even;
      activation_words_lds[static_cast<uint64_t>(block) * 4U + 3U] = second.odd;
      activation_scales_lds[block] = e4m3(activation_scales[block]);
    }
    __syncthreads();
  }

  const uint32_t lane = threadIdx.x & (kWave - 1U);
  const uint32_t wave = threadIdx.x / kWave;
  const uint32_t column_base =
      blockIdx.x * kColumnsPerBlock + wave * kColumnsPerWave;
  const uint64_t packed_row_bytes = static_cast<uint64_t>(blocks_per_row) * 8U;
  float accumulators[kColumnsPerWave] = {};

  for (uint32_t block = lane; block < blocks_per_row; block += kWave) {
    uint32_t activation_words[4];
    float activation_scale = 0.0F;
    if constexpr (ActivationLds) {
      const uint32_t *const source = activation_words_lds + block * 4U;
#pragma unroll
      for (uint32_t group = 0U; group < 4U; ++group)
        activation_words[group] = source[group];
      activation_scale = activation_scales_lds[block];
    } else {
      const uint8_t *const source =
          activation + static_cast<uint64_t>(block) * 8U;
      const PackedI8Pair first = scaled_pack(__builtin_nontemporal_load(
          reinterpret_cast<const uint32_t *>(source)));
      const PackedI8Pair second = scaled_pack(__builtin_nontemporal_load(
          reinterpret_cast<const uint32_t *>(source + 4U)));
      activation_words[0] = first.even;
      activation_words[1] = first.odd;
      activation_words[2] = second.even;
      activation_words[3] = second.odd;
      activation_scale =
          e4m3(__builtin_nontemporal_load(activation_scales + block));
    }

#pragma unroll
    for (uint32_t column_offset = 0U; column_offset < kColumnsPerWave;
         ++column_offset) {
      const uint32_t column = column_base + column_offset;
      if (column >= n)
        continue;
      const uint8_t *const weight_block =
          weight + static_cast<uint64_t>(column) * packed_row_bytes +
          static_cast<uint64_t>(block) * 8U;
      const uint8_t *const weight_scale_row =
          weight_scales + static_cast<uint64_t>(column) * blocks_per_row;
      const uint32_t word0 = __builtin_nontemporal_load(
          reinterpret_cast<const uint32_t *>(weight_block));
      const uint32_t word1 = __builtin_nontemporal_load(
          reinterpret_cast<const uint32_t *>(weight_block + 4U));
      const PackedI8Pair first = scaled_pack(word0);
      const PackedI8Pair second = scaled_pack(word1);
      int32_t integer_sum = 0;
      integer_sum = dot4<Kind>(activation_words[0], first.even, integer_sum);
      integer_sum = dot4<Kind>(activation_words[1], first.odd, integer_sum);
      integer_sum = dot4<Kind>(activation_words[2], second.even, integer_sum);
      integer_sum = dot4<Kind>(activation_words[3], second.odd, integer_sum);
      const float weight_scale = e4m3(weight_scale_row[block]);
      accumulators[column_offset] += static_cast<float>(integer_sum) * 0.25F *
                                     activation_scale * weight_scale;
    }
  }

#pragma unroll
  for (uint32_t column_offset = 0U; column_offset < kColumnsPerWave;
       ++column_offset) {
#pragma unroll
    for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U)
      accumulators[column_offset] +=
          __shfl_down(accumulators[column_offset], offset, kWave);
    const uint32_t column = column_base + column_offset;
    if (lane == 0U && column < n) {
      output[column] = bf16_rne(accumulators[column_offset] *
                                weight_tensor_scale[0] * input_tensor_scale[0]);
    }
  }
}

struct Buffers final {
  uint8_t *activation = nullptr;
  uint8_t *activation_scales = nullptr;
  uint8_t *weight = nullptr;
  uint8_t *weight_scales = nullptr;
  uint16_t *output = nullptr;
  float *weight_tensor_scale = nullptr;
  float *input_tensor_scale = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
};

void cleanup(Buffers *const b) {
  if (b == nullptr)
    return;
  if (b->stop)
    (void)hipEventDestroy(b->stop);
  if (b->start)
    (void)hipEventDestroy(b->start);
  if (b->stream)
    (void)hipStreamDestroy(b->stream);
  if (b->input_tensor_scale)
    (void)hipFree(b->input_tensor_scale);
  if (b->weight_tensor_scale)
    (void)hipFree(b->weight_tensor_scale);
  if (b->output)
    (void)hipFree(b->output);
  if (b->weight_scales)
    (void)hipFree(b->weight_scales);
  if (b->weight)
    (void)hipFree(b->weight);
  if (b->activation_scales)
    (void)hipFree(b->activation_scales);
  if (b->activation)
    (void)hipFree(b->activation);
  *b = {};
}

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess)
    return true;
  std::fprintf(stderr, "hip error operation=%s status=%s\n", operation,
               hipGetErrorString(status));
  return false;
}

float host_e2m1(const uint8_t code) {
  constexpr float positive[8] = {0.0F, 0.5F, 1.0F, 1.5F,
                                 2.0F, 3.0F, 4.0F, 6.0F};
  const float value = positive[code & 7U];
  return (code & 8U) == 0U ? value : -value;
}

float host_e4m3(const uint8_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x80U) << 24U;
  const uint32_t magnitude = bits & 0x7fU;
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & 7U;
  if (exponent == 0U) {
    if (mantissa == 0U)
      return 0.0F;
    float value = static_cast<float>(mantissa) * 0x1p-9F;
    uint32_t value_bits = 0U;
    std::memcpy(&value_bits, &value, sizeof(value_bits));
    value_bits |= sign;
    std::memcpy(&value, &value_bits, sizeof(value));
    return value;
  }
  if (magnitude == 0x7fU)
    return std::numeric_limits<float>::quiet_NaN();
  const uint32_t result = sign | ((exponent + 120U) << 23U) | (mantissa << 20U);
  float value = 0.0F;
  std::memcpy(&value, &result, sizeof(value));
  return value;
}

uint16_t host_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000U) == 0x7f800000U)
    return static_cast<uint16_t>(bits >> 16U);
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & 0xffffU;
  if (lower > 0x8000U || (lower == 0x8000U && (upper & 1U) != 0U))
    ++upper;
  return static_cast<uint16_t>(upper);
}

uint32_t ulp_distance(const uint16_t a, const uint16_t b) {
  if ((a & 0x7fffU) == 0U && (b & 0x7fffU) == 0U)
    return 0U;
  const int32_t ai = (a & 0x8000U) ? 0x8000 - (a & 0x7fffU) : 0x8000 + a;
  const int32_t bi = (b & 0x8000U) ? 0x8000 - (b & 0x7fffU) : 0x8000 + b;
  return static_cast<uint32_t>(std::abs(ai - bi));
}

struct HostInputs final {
  uint32_t blocks = 0U;
  uint32_t n = 0U;
  std::vector<uint8_t> activation, activation_scales, weight, weight_scales;
};

HostInputs make_inputs(const uint32_t blocks, const uint32_t n) {
  HostInputs h{blocks, n};
  h.activation.resize(static_cast<size_t>(blocks) * 8U);
  h.activation_scales.resize(blocks);
  h.weight.resize(static_cast<size_t>(n) * blocks * 8U);
  h.weight_scales.resize(static_cast<size_t>(n) * blocks);
  constexpr std::array<uint8_t, 8> scale_codes = {0x30U, 0x38U, 0x3cU, 0x40U,
                                                  0x44U, 0x48U, 0x4cU, 0x34U};
  for (uint32_t block = 0U; block < blocks; ++block) {
    h.activation_scales[block] = scale_codes[block % scale_codes.size()];
    for (uint32_t byte = 0U; byte < 8U; ++byte) {
      const uint8_t low =
          static_cast<uint8_t>((block * 5U + byte * 3U) & 0x0fU);
      const uint8_t high =
          static_cast<uint8_t>((block * 11U + byte * 7U + 1U) & 0x0fU);
      h.activation[static_cast<size_t>(block) * 8U + byte] =
          static_cast<uint8_t>(low | static_cast<uint8_t>(high << 4U));
    }
  }
  for (uint32_t column = 0U; column < n; ++column) {
    for (uint32_t block = 0U; block < blocks; ++block) {
      const size_t offset = (static_cast<size_t>(column) * blocks + block) * 8U;
      h.weight_scales[static_cast<size_t>(column) * blocks + block] =
          scale_codes[(column * 3U + block * 5U + 2U) % scale_codes.size()];
      for (uint32_t byte = 0U; byte < 8U; ++byte) {
        const uint8_t low = static_cast<uint8_t>(
            (column * 13U + block * 5U + byte * 9U + 2U) & 0x0fU);
        const uint8_t high = static_cast<uint8_t>(
            (column * 7U + block * 11U + byte * 3U + 4U) & 0x0fU);
        h.weight[offset + byte] =
            static_cast<uint8_t>(low | static_cast<uint8_t>(high << 4U));
      }
    }
  }
  return h;
}

std::vector<uint16_t> host_oracle(const HostInputs &h) {
  std::vector<uint16_t> result(h.n, 0U);
  for (uint32_t column = 0U; column < h.n; ++column) {
    float total = 0.0F;
    for (uint32_t block = 0U; block < h.blocks; ++block) {
      float subtotal = 0.0F;
      const size_t ao = static_cast<size_t>(block) * 8U;
      const size_t wo = (static_cast<size_t>(column) * h.blocks + block) * 8U;
      for (uint32_t index = 0U; index < 16U; ++index) {
        const uint8_t ap = h.activation[ao + index / 2U];
        const uint8_t wp = h.weight[wo + index / 2U];
        const uint8_t ac = (index & 1U) == 0U ? ap & 0x0fU : ap >> 4U;
        const uint8_t wc = (index & 1U) == 0U ? wp & 0x0fU : wp >> 4U;
        subtotal += host_e2m1(ac) * host_e2m1(wc);
      }
      total +=
          subtotal * host_e4m3(h.activation_scales[block]) *
          host_e4m3(
              h.weight_scales[static_cast<size_t>(column) * h.blocks + block]);
    }
    result[column] = host_bf16_rne(total * 0.75F * 1.125F);
  }
  return result;
}

bool make_buffers(const HostInputs &h, Buffers *const b) {
  return hip_ok(hipMalloc(reinterpret_cast<void **>(&b->activation),
                          h.activation.size()),
                "malloc activation") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->activation_scales),
                          h.activation_scales.size()),
                "malloc activation scales") &&
         hip_ok(
             hipMalloc(reinterpret_cast<void **>(&b->weight), h.weight.size()),
             "malloc weight") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->weight_scales),
                          h.weight_scales.size()),
                "malloc weight scales") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->output),
                          h.n * sizeof(uint16_t)),
                "malloc output") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->weight_tensor_scale),
                          sizeof(float)),
                "malloc weight tensor scale") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&b->input_tensor_scale),
                          sizeof(float)),
                "malloc input tensor scale") &&
         hip_ok(hipStreamCreate(&b->stream), "stream") &&
         hip_ok(hipEventCreate(&b->start), "event start") &&
         hip_ok(hipEventCreate(&b->stop), "event stop");
}

bool upload(const HostInputs &h, Buffers *const b) {
  const float wt = 0.75F, at = 1.125F;
  return hip_ok(hipMemcpy(b->activation, h.activation.data(),
                          h.activation.size(), hipMemcpyHostToDevice),
                "upload activation") &&
         hip_ok(hipMemcpy(b->activation_scales, h.activation_scales.data(),
                          h.activation_scales.size(), hipMemcpyHostToDevice),
                "upload activation scales") &&
         hip_ok(hipMemcpy(b->weight, h.weight.data(), h.weight.size(),
                          hipMemcpyHostToDevice),
                "upload weight") &&
         hip_ok(hipMemcpy(b->weight_scales, h.weight_scales.data(),
                          h.weight_scales.size(), hipMemcpyHostToDevice),
                "upload weight scales") &&
         hip_ok(hipMemcpy(b->weight_tensor_scale, &wt, sizeof(wt),
                          hipMemcpyHostToDevice),
                "upload weight tensor scale") &&
         hip_ok(hipMemcpy(b->input_tensor_scale, &at, sizeof(at),
                          hipMemcpyHostToDevice),
                "upload input tensor scale") &&
         hip_ok(hipMemset(b->output, 0, h.n * sizeof(uint16_t)),
                "clear output");
}

template <DotKind Kind, bool ActivationLds>
bool launch(const HostInputs &h, Buffers *const b) {
  const uint32_t grid = (h.n + kColumnsPerBlock - 1U) / kColumnsPerBlock;
  const size_t dynamic_lds =
      ActivationLds ? static_cast<size_t>(h.blocks) * 20U : 0U;
  hipLaunchKernelGGL((decode_kernel<Kind, ActivationLds>), dim3(grid),
                     dim3(kThreads), dynamic_lds, b->stream, b->activation,
                     b->activation_scales, b->weight, b->weight_scales,
                     b->weight_tensor_scale, b->input_tensor_scale, b->output,
                     h.blocks, h.n);
  return hip_ok(hipGetLastError(), "kernel launch");
}

template <DotKind Kind, bool ActivationLds>
bool measure(const HostInputs &h, Buffers *const b, const uint32_t calls,
             float *const median_us, const char *const name) {
  for (uint32_t warmup = 0U; warmup < kWarmups; ++warmup) {
    for (uint32_t call = 0U; call < calls; ++call)
      if (!launch<Kind, ActivationLds>(h, b))
        return false;
    if (!hip_ok(hipStreamSynchronize(b->stream), "warmup sync"))
      return false;
  }
  std::array<float, kMeasured> samples{};
  for (uint32_t iteration = 0U; iteration < kMeasured; ++iteration) {
    if (!hip_ok(hipEventRecord(b->start, b->stream), "event start"))
      return false;
    for (uint32_t call = 0U; call < calls; ++call)
      if (!launch<Kind, ActivationLds>(h, b))
        return false;
    if (!hip_ok(hipEventRecord(b->stop, b->stream), "event stop") ||
        !hip_ok(hipEventSynchronize(b->stop), "event sync") ||
        !hip_ok(hipEventElapsedTime(&samples[iteration], b->start, b->stop),
                "elapsed"))
      return false;
    samples[iteration] *= 1000.0F;
  }
  std::sort(samples.begin(), samples.end());
  *median_us = samples[kMeasured / 2U];
  std::array<float, kMeasured> deviations{};
  for (uint32_t i = 0U; i < kMeasured; ++i)
    deviations[i] = std::fabs(samples[i] - *median_us);
  std::sort(deviations.begin(), deviations.end());
  std::printf("timing candidate=%s calls=%u aggregate_us_median=%.3f mad=%.3f "
              "min=%.3f max=%.3f per_call_us_median=%.3f\n",
              name, calls, *median_us, deviations[kMeasured / 2U],
              samples.front(), samples.back(),
              *median_us / static_cast<float>(calls));
  return true;
}

void compare(const char *const name, const std::vector<uint16_t> &ref,
             const std::vector<uint16_t> &actual) {
  uint32_t max_ulp = 0U;
  uint64_t over_one = 0U;
  uint64_t bitwise_diff = 0U;
  for (size_t i = 0U; i < ref.size(); ++i) {
    if (ref[i] != actual[i])
      ++bitwise_diff;
    const uint32_t ulp = ulp_distance(ref[i], actual[i]);
    max_ulp = std::max(max_ulp, ulp);
    if (ulp > 1U)
      ++over_one;
  }
  std::printf("compare candidate=%s values=%zu bitwise_diff=%llu "
              "max_bf16_ulp=%u over1=%llu status=%s\n",
              name, ref.size(), static_cast<unsigned long long>(bitwise_diff),
              max_ulp, static_cast<unsigned long long>(over_one),
              over_one == 0U ? "PASS" : "INFO");
}

bool copy_output(const HostInputs &h, const Buffers *const b,
                 std::vector<uint16_t> *const output,
                 const char *const operation) {
  output->resize(h.n);
  return hip_ok(hipMemcpy(output->data(), b->output,
                          output->size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                operation);
}

template <DotKind Kind, bool ActivationLds>
bool check_determinism(const char *const name, const HostInputs &h,
                       Buffers *const b, const std::vector<uint16_t> &first) {
  if (!launch<Kind, ActivationLds>(h, b) ||
      !hip_ok(hipStreamSynchronize(b->stream), "determinism sync"))
    return false;
  std::vector<uint16_t> second;
  if (!copy_output(h, b, &second, "determinism copy"))
    return false;
  uint64_t bitwise_diff = 0U;
  for (size_t i = 0U; i < first.size(); ++i)
    if (first[i] != second[i])
      ++bitwise_diff;
  std::printf(
      "determinism candidate=%s values=%zu bitwise_diff=%llu status=%s\n", name,
      first.size(), static_cast<unsigned long long>(bitwise_diff),
      bitwise_diff == 0U ? "PASS" : "FAIL");
  return bitwise_diff == 0U;
}

template <DotKind Kind, bool ActivationLds>
void print_resources(const char *const name, const uint32_t blocks) {
  const void *const fn =
      reinterpret_cast<const void *>(decode_kernel<Kind, ActivationLds>);
  hipFuncAttributes attr{};
  const hipError_t attrs = hipFuncGetAttributes(&attr, fn);
  const size_t dynamic_lds =
      ActivationLds ? static_cast<size_t>(blocks) * 20U : 0U;
  int active = 0;
  const hipError_t occ = hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, fn, kThreads, dynamic_lds);
  std::printf("resources candidate=%s vgpr=%d lds_static=%zu lds_dynamic=%zu "
              "scratch=%zu active_blocks=%d attrs=%s occupancy=%s\n",
              name, attr.numRegs, attr.sharedSizeBytes, dynamic_lds,
              attr.localSizeBytes, active, hipGetErrorString(attrs),
              hipGetErrorString(occ));
}

template <DotKind Kind, bool ActivationLds>
bool run_candidate(const char *const name, const HostInputs &h,
                   Buffers *const b, const uint32_t calls,
                   const std::vector<uint16_t> &control,
                   const float control_us) {
  float median_us = 0.0F;
  if (!measure<Kind, ActivationLds>(h, b, calls, &median_us, name))
    return false;
  std::vector<uint16_t> actual;
  if (!copy_output(h, b, &actual, "copy output"))
    return false;
  if (!check_determinism<Kind, ActivationLds>(name, h, b, actual))
    return false;
  compare(name, control, actual);
  std::printf("result candidate=%s k=%u n=%u calls=%u aggregate_ms=%.6f "
              "ms_per_call=%.6f speedup_vs_control=%.3f\n",
              name, h.blocks * 16U, h.n, calls, median_us / 1000.0F,
              median_us / (1000.0F * calls), control_us / median_us);
  return true;
}

bool run_shape(const uint32_t k, const uint32_t n, const uint32_t calls) {
  const HostInputs h = make_inputs(k / 16U, n);
  Buffers b;
  if (!make_buffers(h, &b) || !upload(h, &b)) {
    cleanup(&b);
    return false;
  }
  float control_us = 0.0F;
  if (!measure<DotKind::Sdot4, false>(h, &b, calls, &control_us,
                                      "id67-control-sdot4")) {
    cleanup(&b);
    return false;
  }
  std::vector<uint16_t> control;
  if (!copy_output(h, &b, &control, "copy control") ||
      !check_determinism<DotKind::Sdot4, false>("id67-control-sdot4", h, &b,
                                                control)) {
    cleanup(&b);
    return false;
  }
  std::printf("control k=%u n=%u calls=%u aggregate_ms=%.6f ms_per_call=%.6f\n",
              k, n, calls, control_us / 1000.0F,
              control_us / (1000.0F * calls));
  bool ok = run_candidate<DotKind::Sdot4, true>("activation-shared-sdot4", h,
                                                &b, calls, control, control_us);
  ok = run_candidate<DotKind::Sudot4, true>("activation-shared-sudot4", h, &b,
                                            calls, control, control_us) &&
       ok;
  cleanup(&b);
  return ok;
}

} // namespace

int main() {
  if (!hip_ok(hipSetDevice(0), "hipSetDevice"))
    return EXIT_FAILURE;
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, 0), "device properties"))
    return EXIT_FAILURE;
  std::printf("target=%s pci=%04x:%02x:%02x name=%s\n", properties.gcnArchName,
              properties.pciDomainID, properties.pciBusID,
              properties.pciDeviceID, properties.name);
  if (std::string_view(properties.gcnArchName).compare(0U, 7U, "gfx1201") !=
      0U) {
    std::fprintf(stderr, "gfx1201 required\n");
    return EXIT_FAILURE;
  }
  // Report both exact production block counts: K=5120 uses 320 blocks and
  // K=17408 uses 1088 blocks. Dynamic LDS is the sharing cost.
  print_resources<DotKind::Sdot4, false>("id67-control-sdot4-k5120", 320U);
  print_resources<DotKind::Sdot4, true>("activation-shared-sdot4-k5120", 320U);
  print_resources<DotKind::Sudot4, true>("activation-shared-sudot4-k5120",
                                         320U);
  print_resources<DotKind::Sdot4, false>("id67-control-sdot4-k17408", 1088U);
  print_resources<DotKind::Sdot4, true>("activation-shared-sdot4-k17408",
                                        1088U);
  print_resources<DotKind::Sudot4, true>("activation-shared-sudot4-k17408",
                                         1088U);

  // Complete non-aligned oracle: K=160 (10 block16s), N=65 (tail block).
  {
    const HostInputs h = make_inputs(10U, 65U);
    const std::vector<uint16_t> expected = host_oracle(h);
    Buffers b;
    if (!make_buffers(h, &b) || !upload(h, &b) ||
        !launch<DotKind::Sdot4, false>(h, &b) ||
        !hip_ok(hipStreamSynchronize(b.stream), "small control sync")) {
      cleanup(&b);
      return EXIT_FAILURE;
    }
    std::vector<uint16_t> actual(h.n);
    if (!hip_ok(hipMemcpy(actual.data(), b.output,
                          actual.size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "small control copy")) {
      cleanup(&b);
      return EXIT_FAILURE;
    }
    compare("control-small-host-oracle", expected, actual);
    if (!launch<DotKind::Sdot4, true>(h, &b) ||
        !hip_ok(hipStreamSynchronize(b.stream), "small sdot sync") ||
        !hip_ok(hipMemcpy(actual.data(), b.output,
                          actual.size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "small sdot copy")) {
      cleanup(&b);
      return EXIT_FAILURE;
    }
    compare("activation-shared-sdot4-small-host-oracle", expected, actual);
    if (!launch<DotKind::Sudot4, true>(h, &b) ||
        !hip_ok(hipStreamSynchronize(b.stream), "small sudot sync") ||
        !hip_ok(hipMemcpy(actual.data(), b.output,
                          actual.size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "small sudot copy")) {
      cleanup(&b);
      return EXIT_FAILURE;
    }
    compare("activation-shared-sudot4-small-host-oracle", expected, actual);
    cleanup(&b);
  }

  // Qwen3.8 exact projection shapes.  The call counts model the decode graph
  // multiplicity requested by the Phase 78 comparison: 112 wide projections
  // and 56 narrow projections.
  const bool ok =
      run_shape(5120U, 17408U, 112U) && run_shape(17408U, 5120U, 56U);
  std::printf("summary status=%s warmups=%u measured=%u\n",
              ok ? "PASS" : "FAIL", kWarmups, kMeasured);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
