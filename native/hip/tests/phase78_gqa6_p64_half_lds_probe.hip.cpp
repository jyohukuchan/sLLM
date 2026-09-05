// Phase 78 GQA6 P64 raw-FP16 LDS numerical probe.
//
// The input layout and the 24-query-head/4-KV-head/256-dimension contract are
// the same as phase78_gqa6_decode_partition_sweep_probe.hip.cpp.  This probe
// calls the production P64 and P128 launch wrappers from
// causal_attention_kernel_internal.hpp.  Its oracle is deliberately separate
// from that probe: score, stable softmax, and value accumulation are all done
// in double precision before the final BF16 rounding.
//
// The control and prototype have the same real attention equation
//   O[d] = sum_t exp(q.k_t/sqrt(256)-m) V_t[d] / sum_t exp(...).
// The prototype changes only stage-1 K/V LDS storage from FP32 to raw FP16;
// stage-1 partition boundaries, score/reduction order, online softmax, and
// stage-2 merge order remain unchanged.
//
// The measured rows are an N0 prototype comparison only.  They do not
// establish a standard worst-case error bound or authorize production routing.
// GPU execution is intentionally bounded to the scheduled V620 probe.

#include "causal_attention_kernel_internal.hpp"

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

#include "sllm/hip.h"

namespace {

constexpr uint32_t kQHeads = 24U;
constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kGqa = 6U;
constexpr uint32_t kHeadDim = SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM;
constexpr uint32_t kWarmups = 3U;
constexpr uint32_t kMeasured = 10U;
constexpr uint32_t kSeedCount = 2U;
constexpr std::array<uint64_t, 4> kContexts = {UINT64_C(8191), UINT64_C(8192),
                                               UINT64_C(8193), UINT64_C(9435)};
constexpr std::array<uint32_t, kSeedCount> kSeeds = {0U, 7919U};

constexpr uint32_t kP64Partitions = 64U;
constexpr uint32_t kP64TileTokens = 16U;
constexpr uint32_t kP64WaveSize = 32U;
constexpr uint32_t kP64GqaRatio = 6U;
constexpr uint32_t kP64WorkgroupSize = 192U;
constexpr uint32_t kP64WorkspaceStride = kHeadDim + 2U;
constexpr uint64_t kP64WorkspaceBytes = static_cast<uint64_t>(kQHeads) *
                                        kP64Partitions * kP64WorkspaceStride *
                                        sizeof(float);
constexpr size_t kControlStage1LdsBytes =
    static_cast<size_t>(2U) * kP64TileTokens * kHeadDim * sizeof(float);
constexpr size_t kCandidateStage1LdsBytes =
    static_cast<size_t>(2U) * kP64TileTokens * kHeadDim * sizeof(uint16_t);

__device__ float candidate_f16_to_f32(const uint16_t raw) noexcept {
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
  return __uint_as_float(bits);
}

__device__ float candidate_bf16_to_f32(const uint16_t raw) noexcept {
  return __uint_as_float(static_cast<uint32_t>(raw) << 16U);
}

__device__ uint16_t candidate_f32_to_bf16_rne(const float value) noexcept {
  const uint32_t bits = __float_as_uint(value);
  constexpr uint32_t exponent_mask = 0x7f800000U;
  constexpr uint32_t fraction_mask = 0x007fffffU;
  if ((bits & exponent_mask) == exponent_mask) {
    if ((bits & fraction_mask) != 0U) {
      const uint16_t sign = static_cast<uint16_t>((bits >> 16U) & 0x8000U);
      const uint16_t payload = static_cast<uint16_t>((bits >> 16U) & 0x003fU);
      return static_cast<uint16_t>(sign | 0x7fc0U | payload);
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & 0xffffU;
  if (lower > 0x8000U || (lower == 0x8000U && (upper & 1U) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

__global__
__launch_bounds__(kP64WorkgroupSize, 1) void phase78_p64_half_lds_stage1(
    const uint16_t *const query, const uint16_t *const key,
    const uint16_t *const value, uint16_t *const output,
    const uint64_t committed_kv_length, float *const workspace) {
  const uint32_t block = blockIdx.x;
  const uint32_t kv_head = block / kP64Partitions;
  const uint32_t partition = block % kP64Partitions;
  if (kv_head >= kKvHeads)
    return;
  const uint64_t split_begin = committed_kv_length * partition / kP64Partitions;
  const uint64_t split_end =
      committed_kv_length * (partition + 1U) / kP64Partitions;
  const uint32_t thread = threadIdx.x;
  const uint32_t lane = thread & (kP64WaveSize - 1U);
  const uint32_t wave = thread / kP64WaveSize;
  const uint32_t query_head = kv_head * kP64GqaRatio + wave;
  const uint16_t *const query_row =
      query + static_cast<uint64_t>(query_head) * kHeadDim;
  constexpr uint32_t kDimensionsPerLane = kHeadDim / kP64WaveSize;
  float query_values[kDimensionsPerLane];
  float accumulations[kDimensionsPerLane] = {};
#pragma unroll
  for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
    query_values[index] =
        candidate_bf16_to_f32(query_row[lane + index * kP64WaveSize]);
  }
  const uint64_t workspace_index =
      (static_cast<uint64_t>(query_head) * kP64Partitions + partition) *
      kP64WorkspaceStride;
  float *const partial = workspace + workspace_index;
  if (split_begin >= split_end) {
    for (uint32_t index = lane; index < kHeadDim; index += kP64WaveSize) {
      partial[index] = 0.0F;
    }
    if (lane == 0U) {
      partial[kHeadDim] = -INFINITY;
      partial[kHeadDim + 1U] = 0.0F;
    }
    return;
  }

  // Only this storage type differs from production P64.  Conversion happens
  // at each use below, preserving the production FP32 arithmetic/order.
  __shared__ uint16_t key_tile[kP64TileTokens][kHeadDim];
  __shared__ uint16_t value_tile[kP64TileTokens][kHeadDim];
  float local_maximum = -INFINITY;
  float local_denominator = 0.0F;
  for (uint64_t tile_begin = split_begin; tile_begin < split_end;
       tile_begin += kP64TileTokens) {
    const uint64_t remaining = split_end - tile_begin;
    const uint32_t tile_count = remaining < kP64TileTokens
                                    ? static_cast<uint32_t>(remaining)
                                    : kP64TileTokens;
    for (uint32_t element = thread; element < tile_count * kHeadDim;
         element += kP64WorkgroupSize) {
      const uint32_t token = element / kHeadDim;
      const uint32_t dimension = element % kHeadDim;
      const uint64_t kv_row = (tile_begin + token) * kKvHeads + kv_head;
      key_tile[token][dimension] = key[kv_row * kHeadDim + dimension];
      value_tile[token][dimension] = value[kv_row * kHeadDim + dimension];
    }
    __syncthreads();
    for (uint32_t token = 0U; token < tile_count; ++token) {
      float score_partial = 0.0F;
#pragma unroll
      for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
        score_partial +=
            query_values[index] *
            candidate_f16_to_f32(key_tile[token][lane + index * kP64WaveSize]);
      }
      for (uint32_t offset = kP64WaveSize / 2U; offset != 0U; offset >>= 1U) {
        score_partial += __shfl_down(score_partial, offset, kP64WaveSize);
      }
      float rescale = 0.0F;
      float contribution = 0.0F;
      if (lane == 0U) {
        const float current_score =
            score_partial * rsqrtf(static_cast<float>(kHeadDim));
        const float next_maximum = fmaxf(local_maximum, current_score);
        rescale = expf(local_maximum - next_maximum);
        contribution = expf(current_score - next_maximum);
        local_denominator = local_denominator * rescale + contribution;
        local_maximum = next_maximum;
      }
      rescale = __shfl(rescale, 0U, kP64WaveSize);
      contribution = __shfl(contribution, 0U, kP64WaveSize);
#pragma unroll
      for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
        accumulations[index] =
            accumulations[index] * rescale +
            contribution * candidate_f16_to_f32(
                               value_tile[token][lane + index * kP64WaveSize]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
    partial[lane + index * kP64WaveSize] = accumulations[index];
  }
  if (lane == 0U) {
    partial[kHeadDim] = local_maximum;
    partial[kHeadDim + 1U] = local_denominator;
  }
  (void)output;
}

__global__
__launch_bounds__(kP64WorkgroupSize, 1) void phase78_p64_half_lds_stage2(
    const uint16_t *const query, uint16_t *const output, const uint32_t q_heads,
    const float *const workspace) {
  const uint32_t query_head = blockIdx.x;
  if (query_head >= q_heads || q_heads != kQHeads)
    return;
  const uint32_t dimension = threadIdx.x;
  const uint64_t base =
      static_cast<uint64_t>(query_head) * kP64Partitions * kP64WorkspaceStride;
  __shared__ float maxima[kP64Partitions];
  __shared__ float denominators[kP64Partitions];
  __shared__ float merge_scales[kP64Partitions];
  __shared__ float global_maximum;
  __shared__ float global_denominator;
  if (dimension < kP64Partitions) {
    maxima[dimension] =
        workspace[base + dimension * kP64WorkspaceStride + kHeadDim];
    denominators[dimension] =
        workspace[base + dimension * kP64WorkspaceStride + kHeadDim + 1U];
  }
  __syncthreads();
  if (dimension == 0U) {
    float maximum = maxima[0];
#pragma unroll
    for (uint32_t partition = 1U; partition < kP64Partitions; ++partition) {
      maximum = fmaxf(maximum, maxima[partition]);
    }
    global_maximum = maximum;
  }
  __syncthreads();
  if (dimension < kP64Partitions) {
    merge_scales[dimension] = expf(maxima[dimension] - global_maximum);
  }
  __syncthreads();
  if (dimension == 0U) {
    float denominator = 0.0F;
#pragma unroll
    for (uint32_t partition = 0U; partition < kP64Partitions; ++partition) {
      denominator += denominators[partition] * merge_scales[partition];
    }
    global_denominator = denominator;
  }
  __syncthreads();
  uint16_t *const output_row =
      output + static_cast<uint64_t>(query_head) * kHeadDim;
  for (uint32_t current = dimension; current < kHeadDim;
       current += kP64WorkgroupSize) {
    float merged = 0.0F;
#pragma unroll
    for (uint32_t partition = 0U; partition < kP64Partitions; ++partition) {
      merged += workspace[base + partition * kP64WorkspaceStride + current] *
                merge_scales[partition];
    }
    output_row[current] =
        candidate_f32_to_bf16_rne(merged / global_denominator);
  }
  (void)query;
}

hipError_t launch_candidate_p64(
    const uint16_t *const query, const void *const key, const void *const value,
    const void *const key_scales, const void *const value_scales,
    const float *const key_outer_scales, const float *const value_outer_scales,
    uint16_t *const output, const uint32_t query_count,
    const uint64_t start_position, const uint64_t committed_kv_length,
    const uint32_t q_heads, const uint32_t kv_heads, const uint32_t head_dim,
    const uint32_t encoding, const float static_key_scale,
    const float static_value_scale, void *const workspace,
    const uint64_t workspace_bytes, const hipStream_t stream) noexcept {
  (void)key_scales;
  (void)value_scales;
  (void)key_outer_scales;
  (void)value_outer_scales;
  (void)static_key_scale;
  (void)static_value_scale;
  if (query == nullptr || key == nullptr || value == nullptr ||
      output == nullptr || workspace == nullptr || query_count != 1U ||
      start_position + 1U != committed_kv_length || q_heads != kQHeads ||
      kv_heads != kKvHeads || head_dim != kHeadDim ||
      encoding != SLLM_HIP_KV_ENCODING_FP16_V1 ||
      workspace_bytes < kP64WorkspaceBytes) {
    return hipErrorInvalidValue;
  }
  const auto *const key_fp16 = static_cast<const uint16_t *>(key);
  const auto *const value_fp16 = static_cast<const uint16_t *>(value);
  hipLaunchKernelGGL(phase78_p64_half_lds_stage1,
                     dim3(kv_heads * kP64Partitions), dim3(kP64WorkgroupSize),
                     0U, stream, query, key_fp16, value_fp16, output,
                     committed_kv_length, static_cast<float *>(workspace));
  hipError_t status = hipGetLastError();
  if (status != hipSuccess)
    return status;
  hipLaunchKernelGGL(phase78_p64_half_lds_stage2, dim3(q_heads),
                     dim3(kP64WorkgroupSize), 0U, stream, query, output,
                     q_heads, static_cast<const float *>(workspace));
  return hipGetLastError();
}

using LaunchFn = hipError_t (*)(const uint16_t *, const void *, const void *,
                                const void *, const void *, const float *,
                                const float *, uint16_t *, uint32_t, uint64_t,
                                uint64_t, uint32_t, uint32_t, uint32_t,
                                uint32_t, float, float, void *, uint64_t,
                                hipStream_t) noexcept;

struct Buffers final {
  uint16_t *query = nullptr;
  uint16_t *key = nullptr;
  uint16_t *value = nullptr;
  uint16_t *output = nullptr;
  float *workspace = nullptr;
  size_t workspace_bytes = 0U;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
};

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess)
    return true;
  std::fprintf(stderr, "hip_error operation=%s status=%s\n", operation,
               hipGetErrorString(status));
  return false;
}

uint16_t host_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000U) == 0x7f800000U) {
    if ((bits & 0x007fffffU) != 0U) {
      return static_cast<uint16_t>(((bits >> 16U) & 0x8000U) | 0x7fc0U |
                                   ((bits >> 16U) & 0x003fU));
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & 0xffffU;
  if (lower > 0x8000U || (lower == 0x8000U && (upper & 1U) != 0U))
    ++upper;
  return static_cast<uint16_t>(upper);
}

float host_bf16_to_float(const uint16_t bits) {
  const uint32_t value = static_cast<uint32_t>(bits) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

uint16_t host_fp16_rne(const float value) {
  // The generated inputs are ordinary finite values in [-0.65, 0.65].
  const float absolute = std::fabs(value);
  if (absolute == 0.0F)
    return static_cast<uint16_t>(std::signbit(value) ? 0x8000U : 0U);
  const uint32_t sign = std::signbit(value) ? 0x8000U : 0U;
  int exponent = 0;
  const float normalized = std::frexp(absolute, &exponent);
  int half_exponent = exponent + 14;
  if (half_exponent <= 0) {
    const int mantissa = static_cast<int>(std::lrint(absolute * 16777216.0F));
    return static_cast<uint16_t>(
        sign | static_cast<uint32_t>(std::clamp(mantissa, 0, 1023)));
  }
  if (half_exponent >= 31)
    return static_cast<uint16_t>(sign | 0x7c00U);
  const float scaled = std::ldexp(normalized, 11);
  int mantissa = static_cast<int>(std::lrint(scaled)) - 1024;
  if (mantissa >= 1024) {
    mantissa = 0;
    ++half_exponent;
  }
  return static_cast<uint16_t>(sign |
                               static_cast<uint32_t>(half_exponent << 10) |
                               static_cast<uint32_t>(mantissa));
}

float host_fp16_to_float(const uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16U;
  const uint32_t exponent = (bits >> 10U) & 31U;
  const uint32_t mantissa = bits & 1023U;
  uint32_t result = sign;
  if (exponent == 0U) {
    if (mantissa != 0U) {
      const float value = static_cast<float>(mantissa) * 0x1p-24F;
      uint32_t value_bits = 0U;
      std::memcpy(&value_bits, &value, sizeof(value_bits));
      result = (value_bits & 0x7fffffffU) | sign;
    }
  } else if (exponent == 31U) {
    result |= 0x7f800000U | (mantissa << 13U);
  } else {
    result |= ((exponent + 112U) << 23U) | (mantissa << 13U);
  }
  float output = 0.0F;
  std::memcpy(&output, &result, sizeof(output));
  return output;
}

double host_bf16_to_double(const uint16_t bits) {
  return static_cast<double>(host_bf16_to_float(bits));
}

double host_fp16_to_double(const uint16_t bits) {
  return static_cast<double>(host_fp16_to_float(bits));
}

int32_t bf16_order(const uint16_t bits) {
  return (bits & 0x8000U) != 0U ? 0x8000 - static_cast<int32_t>(bits & 0x7fffU)
                                : 0x8000 + static_cast<int32_t>(bits);
}

uint32_t bf16_ulp(const uint16_t left, const uint16_t right) {
  if ((left & 0x7fffU) == 0U && (right & 0x7fffU) == 0U)
    return 0U;
  return static_cast<uint32_t>(std::abs(bf16_order(left) - bf16_order(right)));
}

void fill_inputs(const uint64_t context, const uint32_t seed,
                 std::vector<uint16_t> *const query,
                 std::vector<uint16_t> *const key,
                 std::vector<uint16_t> *const value) {
  query->resize(static_cast<size_t>(kQHeads) * kHeadDim);
  const size_t kv_elements = static_cast<size_t>(context) * kKvHeads * kHeadDim;
  key->resize(kv_elements);
  value->resize(kv_elements);

  // seed=0 is byte-for-byte compatible with the existing partition probe's
  // deterministic fixture.  The second seed shifts the same bounded fixture
  // so the oracle is exercised on more than one finite input distribution.
  const uint32_t q_shift = seed % 97U;
  const uint32_t k_shift = seed % 131U;
  const uint32_t v_shift = seed % 127U;
  for (uint32_t head = 0U; head < kQHeads; ++head) {
    for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
      const uint32_t phase = (head * 13U + dimension * 3U + q_shift) % 97U;
      const float source = 0.45F * std::sin(static_cast<float>(phase) * 0.071F);
      (*query)[static_cast<size_t>(head) * kHeadDim + dimension] =
          host_bf16_rne(source);
    }
  }
  for (uint64_t token = 0U; token < context; ++token) {
    for (uint32_t head = 0U; head < kKvHeads; ++head) {
      for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
        const uint32_t key_phase = static_cast<uint32_t>(
            (token * 11U + head * 5U + dimension * 17U + k_shift) % 131U);
        const uint32_t value_phase = static_cast<uint32_t>(
            (token * 3U + head * 19U + dimension * 7U + v_shift) % 127U);
        const float key_source =
            0.65F * std::cos(static_cast<float>(key_phase) * 0.053F);
        const float value_source =
            0.55F * std::sin(static_cast<float>(value_phase) * 0.067F);
        const size_t index =
            (static_cast<size_t>(token) * kKvHeads + head) * kHeadDim +
            dimension;
        (*key)[index] = host_fp16_rne(key_source);
        (*value)[index] = host_fp16_rne(value_source);
      }
    }
  }
}

// Independent double precision stable-softmax oracle.  It intentionally does
// not call or share the FP32 oracle in the partition sweep probe.
void fp64_oracle(const uint64_t context, const std::vector<uint16_t> &query,
                 const std::vector<uint16_t> &key,
                 const std::vector<uint16_t> &value,
                 std::vector<double> *const values,
                 std::vector<uint16_t> *const rounded) {
  values->assign(static_cast<size_t>(kQHeads) * kHeadDim, 0.0);
  rounded->assign(static_cast<size_t>(kQHeads) * kHeadDim, 0U);
  const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
  for (uint32_t head = 0U; head < kQHeads; ++head) {
    const uint32_t kv_head = head / kGqa;
    std::vector<double> scores(static_cast<size_t>(context));
    double maximum = -std::numeric_limits<double>::infinity();
    for (uint64_t token = 0U; token < context; ++token) {
      double dot = 0.0;
      const size_t query_base = static_cast<size_t>(head) * kHeadDim;
      const size_t key_base =
          (static_cast<size_t>(token) * kKvHeads + kv_head) * kHeadDim;
      for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
        dot += host_bf16_to_double(query[query_base + dimension]) *
               host_fp16_to_double(key[key_base + dimension]);
      }
      scores[static_cast<size_t>(token)] = dot * scale;
      maximum = std::max(maximum, scores[static_cast<size_t>(token)]);
    }

    double denominator = 0.0;
    for (double &score : scores) {
      score = std::exp(score - maximum);
      denominator += score;
    }
    for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
      double numerator = 0.0;
      for (uint64_t token = 0U; token < context; ++token) {
        const size_t value_index =
            (static_cast<size_t>(token) * kKvHeads + kv_head) * kHeadDim +
            dimension;
        numerator += scores[static_cast<size_t>(token)] *
                     host_fp16_to_double(value[value_index]);
      }
      const double result = numerator / denominator;
      const size_t output_index =
          static_cast<size_t>(head) * kHeadDim + dimension;
      (*values)[output_index] = result;
      // The final BF16 conversion is the only float conversion in the oracle.
      // Every tested result is in the ordinary finite range, so the float
      // narrowing is exact enough to select the same BF16 RNE bin, including
      // the representable midpoint between adjacent BF16 values.
      (*rounded)[output_index] = host_bf16_rne(static_cast<float>(result));
    }
  }
}

struct ErrorStats final {
  double l1 = 0.0;
  double l2 = 0.0;
  double max_abs = 0.0;
  uint32_t max_ulp = 0U;
  uint64_t actual_nonfinite = 0U;
  uint64_t oracle_nonfinite = 0U;
  uint64_t bit_mismatch = 0U;
};

ErrorStats compare_output(const std::vector<double> &oracle,
                          const std::vector<uint16_t> &expected,
                          const std::vector<uint16_t> &actual) {
  ErrorStats result;
  if (oracle.size() != actual.size() || expected.size() != actual.size()) {
    result.actual_nonfinite = 1U;
    result.oracle_nonfinite = 1U;
    return result;
  }
  double l2_squared = 0.0;
  for (size_t index = 0U; index < actual.size(); ++index) {
    const double expected_value = oracle[index];
    const double actual_value = host_bf16_to_double(actual[index]);
    if (!std::isfinite(expected_value))
      ++result.oracle_nonfinite;
    if (!std::isfinite(actual_value))
      ++result.actual_nonfinite;
    const double error = std::fabs(actual_value - expected_value);
    result.l1 += error;
    l2_squared += error * error;
    result.max_abs = std::max(result.max_abs, error);
    result.max_ulp =
        std::max(result.max_ulp, bf16_ulp(expected[index], actual[index]));
    if (expected[index] != actual[index])
      ++result.bit_mismatch;
  }
  result.l2 = std::sqrt(l2_squared);
  return result;
}

void free_buffers(Buffers *const buffers) {
  if (buffers == nullptr)
    return;
  if (buffers->stop != nullptr)
    (void)hipEventDestroy(buffers->stop);
  if (buffers->start != nullptr)
    (void)hipEventDestroy(buffers->start);
  if (buffers->stream != nullptr)
    (void)hipStreamDestroy(buffers->stream);
  if (buffers->workspace != nullptr)
    (void)hipFree(buffers->workspace);
  if (buffers->output != nullptr)
    (void)hipFree(buffers->output);
  if (buffers->value != nullptr)
    (void)hipFree(buffers->value);
  if (buffers->key != nullptr)
    (void)hipFree(buffers->key);
  if (buffers->query != nullptr)
    (void)hipFree(buffers->query);
  *buffers = {};
}

bool make_buffers(const uint64_t context, Buffers *const buffers) {
  const size_t query_bytes =
      static_cast<size_t>(kQHeads) * kHeadDim * sizeof(uint16_t);
  const size_t kv_bytes =
      static_cast<size_t>(context) * kKvHeads * kHeadDim * sizeof(uint16_t);
  // Allocate enough for P128 and pass the same workspace to both providers.
  buffers->workspace_bytes =
      static_cast<size_t>(kQHeads) * 128U * (kHeadDim + 2U) * sizeof(float);
  if (!hip_ok(
          hipMalloc(reinterpret_cast<void **>(&buffers->query), query_bytes),
          "malloc query") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->key), kv_bytes),
              "malloc key") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->value), kv_bytes),
              "malloc value") ||
      !hip_ok(
          hipMalloc(reinterpret_cast<void **>(&buffers->output), query_bytes),
          "malloc output") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->workspace),
                        buffers->workspace_bytes),
              "malloc workspace") ||
      !hip_ok(hipStreamCreate(&buffers->stream), "create stream") ||
      !hip_ok(hipEventCreate(&buffers->start), "create start event") ||
      !hip_ok(hipEventCreate(&buffers->stop), "create stop event")) {
    free_buffers(buffers);
    return false;
  }
  return true;
}

bool upload_inputs(const std::vector<uint16_t> &query,
                   const std::vector<uint16_t> &key,
                   const std::vector<uint16_t> &value, Buffers *const buffers) {
  return hip_ok(hipMemcpy(buffers->query, query.data(),
                          query.size() * sizeof(uint16_t),
                          hipMemcpyHostToDevice),
                "copy query") &&
         hip_ok(hipMemcpy(buffers->key, key.data(),
                          key.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
                "copy key") &&
         hip_ok(hipMemcpy(buffers->value, value.data(),
                          value.size() * sizeof(uint16_t),
                          hipMemcpyHostToDevice),
                "copy value");
}

bool launch_provider(const LaunchFn provider, const uint64_t context,
                     Buffers *const buffers) {
  const hipError_t status = provider(
      buffers->query, buffers->key, buffers->value, nullptr, nullptr, nullptr,
      nullptr, buffers->output, 1U, context - 1U, context, kQHeads, kKvHeads,
      kHeadDim, SLLM_HIP_KV_ENCODING_FP16_V1, 1.0F, 1.0F, buffers->workspace,
      buffers->workspace_bytes, buffers->stream);
  return hip_ok(status, "production GQA6 launch");
}

bool copy_output(Buffers *const buffers, std::vector<uint16_t> *const output) {
  output->resize(static_cast<size_t>(kQHeads) * kHeadDim);
  return hip_ok(hipMemcpy(output->data(), buffers->output,
                          output->size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "copy output");
}

bool compare_bitwise(const std::vector<uint16_t> &left,
                     const std::vector<uint16_t> &right) {
  return left.size() == right.size() &&
         std::equal(left.begin(), left.end(), right.begin());
}

bool measure_provider(const LaunchFn provider, const uint64_t context,
                      Buffers *const buffers, float *const median_us,
                      std::vector<uint16_t> *const output) {
  for (uint32_t iteration = 0U; iteration < kWarmups; ++iteration) {
    if (!launch_provider(provider, context, buffers) ||
        !hip_ok(hipStreamSynchronize(buffers->stream), "warmup synchronize"))
      return false;
  }
  std::array<float, kMeasured> samples{};
  for (float &sample : samples) {
    if (!hip_ok(hipEventRecord(buffers->start, buffers->stream),
                "event start") ||
        !launch_provider(provider, context, buffers) ||
        !hip_ok(hipEventRecord(buffers->stop, buffers->stream), "event stop") ||
        !hip_ok(hipEventSynchronize(buffers->stop), "event synchronize") ||
        !hip_ok(hipEventElapsedTime(&sample, buffers->start, buffers->stop),
                "event elapsed"))
      return false;
    sample *= 1000.0F;
  }
  std::sort(samples.begin(), samples.end());
  *median_us = samples[kMeasured / 2U];
  return copy_output(buffers, output);
}

bool common_prewarm(const LaunchFn first, const LaunchFn second,
                    const uint64_t context, Buffers *const buffers) {
  for (uint32_t iteration = 0U; iteration < kWarmups; ++iteration) {
    const LaunchFn provider = (iteration & 1U) == 0U ? first : second;
    if (!launch_provider(provider, context, buffers) ||
        !hip_ok(hipStreamSynchronize(buffers->stream),
                "common prewarm synchronize")) {
      return false;
    }
  }
  return true;
}

bool repeat_provider(const LaunchFn provider, const uint64_t context,
                     Buffers *const buffers,
                     std::vector<uint16_t> *const output,
                     std::vector<uint16_t> *const repeat) {
  if (!launch_provider(provider, context, buffers) ||
      !hip_ok(hipStreamSynchronize(buffers->stream), "repeat synchronize 1") ||
      !copy_output(buffers, output) ||
      !launch_provider(provider, context, buffers) ||
      !hip_ok(hipStreamSynchronize(buffers->stream), "repeat synchronize 2") ||
      !copy_output(buffers, repeat))
    return false;
  return true;
}

void print_stats(const char *const candidate, const uint64_t context,
                 const uint32_t seed, const ErrorStats &stats,
                 const float median_us, const bool repeat_bitwise) {
  std::printf("oracle candidate=%s context=%llu seed=%u median_us=%.3f "
              "l1=%.9e l2=%.9e max_abs=%.9e max_bf16_ulp=%u "
              "bit_mismatch=%llu oracle_nonfinite=%llu actual_nonfinite=%llu "
              "repeat_bitwise=%s classification=N0_prototype\n",
              candidate, static_cast<unsigned long long>(context), seed,
              median_us, stats.l1, stats.l2, stats.max_abs, stats.max_ulp,
              static_cast<unsigned long long>(stats.bit_mismatch),
              static_cast<unsigned long long>(stats.oracle_nonfinite),
              static_cast<unsigned long long>(stats.actual_nonfinite),
              repeat_bitwise ? "PASS" : "FAIL");
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 2) {
    std::fprintf(stderr, "usage: phase78_gqa6_p64_half_lds_probe [device]\n");
    return EXIT_FAILURE;
  }
  uint32_t device = 0U;
  if (argc == 2) {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(argv[1], &end, 10);
    if (end == argv[1] || *end != '\0' || parsed > UINT32_MAX) {
      std::fprintf(stderr, "invalid device\n");
      return EXIT_FAILURE;
    }
    device = static_cast<uint32_t>(parsed);
  }
  if (!hip_ok(hipSetDevice(static_cast<int>(device)), "set device"))
    return EXIT_FAILURE;
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, device), "device properties"))
    return EXIT_FAILURE;
  std::printf("target=%s device=%u pci=%04x:%02x:%02x name=%s "
              "contexts=8191,8192,8193,9435 q_heads=%u kv_heads=%u head_dim=%u "
              "seeds=0,7919 warmups=%u measured=%u "
              "oracle=fp64_stable_softmax_v1 classification=N0_prototype "
              "worst_case_bound=UNPROVEN\n",
              properties.gcnArchName, device, properties.pciDomainID,
              properties.pciBusID, properties.pciDeviceID, properties.name,
              kQHeads, kKvHeads, kHeadDim, kWarmups, kMeasured);
  if (std::string_view(properties.gcnArchName).find("gfx1030") ==
      std::string_view::npos) {
    std::fprintf(stderr, "unsupported target: %s (expected gfx1030)\n",
                 properties.gcnArchName);
    return EXIT_FAILURE;
  }

  const LaunchFn providers[] = {
      ::sllm_causal_attention_kernel::launch_decode_gqa6_split_p64,
      launch_candidate_p64};
  const char *const names[] = {"control_p64", "candidate_half_lds_p64"};
  bool all_ok = true;
  double control_weighted_us = 0.0;
  double candidate_weighted_us = 0.0;
  std::printf(
      "lds_resources control_stage1_bytes=%zu candidate_stage1_bytes=%zu "
      "control_workspace_bytes=%llu candidate_workspace_bytes=%llu "
      "same_partition_count=%u same_tile_tokens=%u\n",
      kControlStage1LdsBytes, kCandidateStage1LdsBytes,
      static_cast<unsigned long long>(kP64WorkspaceBytes),
      static_cast<unsigned long long>(kP64WorkspaceBytes), kP64Partitions,
      kP64TileTokens);
  for (const uint32_t seed : kSeeds) {
    for (const uint64_t context : kContexts) {
      std::vector<uint16_t> query, key, value, expected;
      std::vector<double> oracle;
      fill_inputs(context, seed, &query, &key, &value);
      fp64_oracle(context, query, key, value, &oracle, &expected);

      Buffers buffers;
      if (!make_buffers(context, &buffers) ||
          !upload_inputs(query, key, value, &buffers)) {
        free_buffers(&buffers);
        return EXIT_FAILURE;
      }
      std::vector<uint16_t> outputs[2];
      std::array<float, 2> medians{};
      // A shared warmup before each measurement pair reduces clock/order bias;
      // alternate the first measured provider across rows as a second guard.
      if (!common_prewarm(providers[0], providers[1], context, &buffers)) {
        free_buffers(&buffers);
        return EXIT_FAILURE;
      }
      const size_t first =
          ((seed + static_cast<uint32_t>(context)) & 1U) == 0U ? 0U : 1U;
      const size_t order[] = {first, 1U - first};
      for (const size_t provider_index : order) {
        if (!measure_provider(providers[provider_index], context, &buffers,
                              &medians[provider_index],
                              &outputs[provider_index])) {
          all_ok = false;
          break;
        }
        std::vector<uint16_t> repeat;
        if (!repeat_provider(providers[provider_index], context, &buffers,
                             &outputs[provider_index], &repeat)) {
          all_ok = false;
          break;
        }
        const bool repeat_bitwise =
            compare_bitwise(outputs[provider_index], repeat);
        const ErrorStats stats =
            compare_output(oracle, expected, outputs[provider_index]);
        print_stats(names[provider_index], context, seed, stats,
                    medians[provider_index], repeat_bitwise);
        if (!repeat_bitwise || stats.actual_nonfinite != 0U ||
            stats.oracle_nonfinite != 0U)
          all_ok = false;
        if (provider_index == 0U)
          control_weighted_us += static_cast<double>(medians[provider_index]);
        else
          candidate_weighted_us += static_cast<double>(medians[provider_index]);
      }
      if (outputs[0].size() == outputs[1].size() && !outputs[0].empty()) {
        uint32_t max_ulp = 0U;
        uint64_t mismatches = 0U;
        for (size_t index = 0U; index < outputs[0].size(); ++index) {
          max_ulp =
              std::max(max_ulp, bf16_ulp(outputs[0][index], outputs[1][index]));
          if (outputs[0][index] != outputs[1][index])
            ++mismatches;
        }
        std::printf("control_candidate_compare context=%llu seed=%u "
                    "max_bf16_ulp=%u bit_mismatch=%llu\n",
                    static_cast<unsigned long long>(context), seed, max_ulp,
                    static_cast<unsigned long long>(mismatches));
      }
      free_buffers(&buffers);
      if (!all_ok)
        break;
    }
    if (!all_ok)
      break;
  }
  std::printf(
      "weighted_summary rows=%zu control_p64_median_sum_us=%.3f "
      "candidate_half_lds_p64_median_sum_us=%.3f candidate_over_control=%.6f "
      "status=%s\n",
      kSeeds.size() * kContexts.size(), control_weighted_us,
      candidate_weighted_us,
      control_weighted_us == 0.0 ? 0.0
                                 : candidate_weighted_us / control_weighted_us,
      all_ok ? "PASS" : "FAIL");
  std::printf(
      "classification=N0_prototype reason=bounded measured FP64 errors; "
      "raw-FP16 LDS only; no production routing or worst_case_proof; "
      "measured samples are not a worst_case bound\n");
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
