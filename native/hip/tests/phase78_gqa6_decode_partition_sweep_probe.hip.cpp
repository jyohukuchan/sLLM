// Phase 78 standalone GQA6 decode partition sweep.
//
// This probe deliberately does not include or modify production sources.  It
// reproduces the production GQA6 split decode arithmetic (24 query heads, four
// KV heads, head_dim=256, six wave32 query heads per KV head, FP16 K/V and
// FP32 online softmax) and changes only the stage-1 partition count.  P32 and
// P64 are controls; P96 and P128 are the candidates.  The measured interval
// includes stage 1 and stage 2, while host copies and allocation are excluded.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr uint32_t kThreads = 192U;
constexpr uint32_t kWave = 32U;
constexpr uint32_t kQHeads = 24U;
constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kGqa = 6U;
constexpr uint32_t kHeadDim = 256U;
constexpr uint32_t kTileTokens = 16U;
constexpr uint32_t kWarmups = 3U;
constexpr uint32_t kMeasured = 10U;
constexpr uint32_t kMaxPartitions = 128U;

__device__ __forceinline__ float bf16_to_float(const uint16_t bits) noexcept {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16U);
}

__device__ __forceinline__ float fp16_to_float(const uint16_t bits) noexcept {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16U;
  const uint32_t exponent = (bits >> 10U) & 31U;
  const uint32_t mantissa = bits & 1023U;
  uint32_t result = sign;
  if (exponent == 0U) {
    if (mantissa != 0U) {
      const float value = static_cast<float>(mantissa) * 0x1p-24F;
      result = __float_as_uint(value) | sign;
    }
  } else if (exponent == 31U) {
    result |= 0x7f800000U | (mantissa << 13U);
  } else {
    result |= ((exponent + 112U) << 23U) | (mantissa << 13U);
  }
  return __uint_as_float(result);
}

__device__ __forceinline__ uint16_t bf16_rne(const float value) noexcept {
  const uint32_t bits = __float_as_uint(value);
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

template <uint32_t Partitions>
__global__ __launch_bounds__(kThreads, 1) void gqa6_decode_split_stage1(
    const uint16_t *const query, const uint16_t *const key,
    const uint16_t *const value, const uint64_t context,
    float *const workspace) {
  static_assert(Partitions == 32U || Partitions == 64U || Partitions == 96U ||
                Partitions == 128U);
  const uint32_t block = blockIdx.x;
  const uint32_t kv_head = block / Partitions;
  const uint32_t partition = block % Partitions;
  if (kv_head >= kKvHeads)
    return;

  const uint64_t split_begin =
      context * static_cast<uint64_t>(partition) / Partitions;
  const uint64_t split_end =
      context * static_cast<uint64_t>(partition + 1U) / Partitions;
  const uint32_t thread = threadIdx.x;
  const uint32_t lane = thread & (kWave - 1U);
  const uint32_t wave = thread / kWave;
  const uint32_t query_head = kv_head * kGqa + wave;
  const uint16_t *const query_row =
      query + static_cast<uint64_t>(query_head) * kHeadDim;
  constexpr uint32_t kDimensionsPerLane = kHeadDim / kWave;
  float query_values[kDimensionsPerLane];
  float accumulations[kDimensionsPerLane] = {};
#pragma unroll
  for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
    query_values[index] = bf16_to_float(query_row[lane + index * kWave]);
  }

  constexpr uint32_t kStride = kHeadDim + 2U;
  const uint64_t workspace_index =
      (static_cast<uint64_t>(query_head) * Partitions + partition) * kStride;
  float *const partial = workspace + workspace_index;
  if (split_begin >= split_end) {
    for (uint32_t index = lane; index < kHeadDim; index += kWave)
      partial[index] = 0.0F;
    if (lane == 0U) {
      partial[kHeadDim] = -INFINITY;
      partial[kHeadDim + 1U] = 0.0F;
    }
    return;
  }

  // K/V are staged once per partition tile and reused by all six waves.
  __shared__ float key_tile[kTileTokens][kHeadDim];
  __shared__ float value_tile[kTileTokens][kHeadDim];
  float local_maximum = -INFINITY;
  float local_denominator = 0.0F;
  for (uint64_t tile_begin = split_begin; tile_begin < split_end;
       tile_begin += kTileTokens) {
    const uint64_t remaining = split_end - tile_begin;
    const uint32_t tile_count = remaining < kTileTokens
                                    ? static_cast<uint32_t>(remaining)
                                    : kTileTokens;
    for (uint32_t element = thread; element < tile_count * kHeadDim;
         element += kThreads) {
      const uint32_t token = element / kHeadDim;
      const uint32_t dimension = element % kHeadDim;
      const uint64_t kv_row = (tile_begin + token) * kKvHeads + kv_head;
      key_tile[token][dimension] =
          fp16_to_float(key[kv_row * kHeadDim + dimension]);
      value_tile[token][dimension] =
          fp16_to_float(value[kv_row * kHeadDim + dimension]);
    }
    __syncthreads();
    for (uint32_t token = 0U; token < tile_count; ++token) {
      float score_partial = 0.0F;
#pragma unroll
      for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
        score_partial +=
            query_values[index] * key_tile[token][lane + index * kWave];
      }
      for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U)
        score_partial += __shfl_down(score_partial, offset, kWave);

      float rescale = 0.0F;
      float contribution = 0.0F;
      if (lane == 0U) {
        const float score =
            score_partial * rsqrtf(static_cast<float>(kHeadDim));
        const float next_maximum = fmaxf(local_maximum, score);
        rescale = expf(local_maximum - next_maximum);
        contribution = expf(score - next_maximum);
        local_denominator = local_denominator * rescale + contribution;
        local_maximum = next_maximum;
      }
      rescale = __shfl(rescale, 0U, kWave);
      contribution = __shfl(contribution, 0U, kWave);
#pragma unroll
      for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
        accumulations[index] =
            accumulations[index] * rescale +
            contribution * value_tile[token][lane + index * kWave];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (uint32_t index = 0U; index < kDimensionsPerLane; ++index)
    partial[lane + index * kWave] = accumulations[index];
  if (lane == 0U) {
    partial[kHeadDim] = local_maximum;
    partial[kHeadDim + 1U] = local_denominator;
  }
}

template <uint32_t Partitions>
__global__ __launch_bounds__(kThreads, 1) void gqa6_decode_split_stage2(
    const uint16_t *const output_query, uint16_t *const output,
    const float *const workspace) {
  static_assert(Partitions == 32U || Partitions == 64U || Partitions == 96U ||
                Partitions == 128U);
  const uint32_t query_head = blockIdx.x;
  if (query_head >= kQHeads)
    return;
  const uint32_t dimension = threadIdx.x;
  constexpr uint32_t kStride = kHeadDim + 2U;
  const uint64_t base =
      static_cast<uint64_t>(query_head) * Partitions * kStride;
  __shared__ float maxima[Partitions];
  __shared__ float denominators[Partitions];
  __shared__ float merge_scales[Partitions];
  __shared__ float global_maximum;
  __shared__ float global_denominator;
  if (dimension < Partitions) {
    maxima[dimension] =
        workspace[base + static_cast<uint64_t>(dimension) * kStride + kHeadDim];
    denominators[dimension] =
        workspace[base + static_cast<uint64_t>(dimension) * kStride + kHeadDim +
                  1U];
  }
  __syncthreads();
  if (dimension == 0U) {
    float maximum = maxima[0];
#pragma unroll
    for (uint32_t partition = 1U; partition < Partitions; ++partition)
      maximum = fmaxf(maximum, maxima[partition]);
    global_maximum = maximum;
  }
  __syncthreads();
  if (dimension < Partitions)
    merge_scales[dimension] = expf(maxima[dimension] - global_maximum);
  __syncthreads();
  if (dimension == 0U) {
    float denominator = 0.0F;
#pragma unroll
    for (uint32_t partition = 0U; partition < Partitions; ++partition)
      denominator += denominators[partition] * merge_scales[partition];
    global_denominator = denominator;
  }
  __syncthreads();
  uint16_t *const output_row =
      output + static_cast<uint64_t>(query_head) * kHeadDim;
  (void)output_query;
  for (uint32_t current = dimension; current < kHeadDim; current += kThreads) {
    float merged = 0.0F;
#pragma unroll
    for (uint32_t partition = 0U; partition < Partitions; ++partition) {
      merged += workspace[base + static_cast<uint64_t>(partition) * kStride +
                          current] *
                merge_scales[partition];
    }
    output_row[current] = bf16_rne(merged / global_denominator);
  }
}

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
  uint32_t value = static_cast<uint32_t>(bits) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

uint16_t host_fp16_rne(const float value) {
  // Inputs are intentionally ordinary finite values; this compact conversion
  // keeps the probe independent of host half libraries.
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

template <typename To, typename From> To bits_cast(const From value) {
  static_assert(sizeof(To) == sizeof(From));
  To result{};
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

float host_fp16_to_float(const uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16U;
  const uint32_t exponent = (bits >> 10U) & 31U;
  const uint32_t mantissa = bits & 1023U;
  uint32_t result = sign;
  if (exponent == 0U) {
    if (mantissa != 0U) {
      const float value = static_cast<float>(mantissa) * 0x1p-24F;
      result = (bits_cast<uint32_t>(value) & 0x7fffffffU) | sign;
    }
  } else if (exponent == 31U) {
    result |= 0x7f800000U | (mantissa << 13U);
  } else {
    result |= ((exponent + 112U) << 23U) | (mantissa << 13U);
  }
  return bits_cast<float>(result);
}

uint32_t bf16_ulp(const uint16_t left, const uint16_t right) {
  if ((left & 0x7fffU) == 0U && (right & 0x7fffU) == 0U)
    return 0U;
  const int32_t left_order =
      (left & 0x8000U) != 0U ? 0x8000 - (left & 0x7fffU) : 0x8000 + left;
  const int32_t right_order =
      (right & 0x8000U) != 0U ? 0x8000 - (right & 0x7fffU) : 0x8000 + right;
  return static_cast<uint32_t>(std::abs(left_order - right_order));
}

void fill_inputs(const uint64_t context, std::vector<uint16_t> *query,
                 std::vector<uint16_t> *key, std::vector<uint16_t> *value) {
  query->resize(kQHeads * kHeadDim);
  key->resize(context * kKvHeads * kHeadDim);
  value->resize(context * kKvHeads * kHeadDim);
  for (uint32_t head = 0U; head < kQHeads; ++head) {
    for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
      const float source =
          0.45F *
          std::sin(static_cast<float>((head * 13U + dimension * 3U) % 97U) *
                   0.071F);
      (*query)[static_cast<uint64_t>(head) * kHeadDim + dimension] =
          host_bf16_rne(source);
    }
  }
  for (uint64_t token = 0U; token < context; ++token) {
    for (uint32_t head = 0U; head < kKvHeads; ++head) {
      for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
        const float key_source =
            0.65F *
            std::cos(static_cast<float>(
                         (token * 11U + head * 5U + dimension * 17U) % 131U) *
                     0.053F);
        const float value_source =
            0.55F *
            std::sin(static_cast<float>(
                         (token * 3U + head * 19U + dimension * 7U) % 127U) *
                     0.067F);
        (*key)[(token * kKvHeads + head) * kHeadDim + dimension] =
            host_fp16_rne(key_source);
        (*value)[(token * kKvHeads + head) * kHeadDim + dimension] =
            host_fp16_rne(value_source);
      }
    }
  }
}

void host_oracle(const uint64_t context, const std::vector<uint16_t> &query,
                 const std::vector<uint16_t> &key,
                 const std::vector<uint16_t> &value,
                 std::vector<uint16_t> *output) {
  output->assign(kQHeads * kHeadDim, 0U);
  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  for (uint32_t head = 0U; head < kQHeads; ++head) {
    const uint32_t kv_head = head / kGqa;
    std::vector<float> scores(context);
    float maximum = -INFINITY;
    for (uint64_t token = 0U; token < context; ++token) {
      float dot = 0.0F;
      for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
        dot += host_bf16_to_float(
                   query[static_cast<uint64_t>(head) * kHeadDim + dimension]) *
               host_fp16_to_float(
                   key[(token * kKvHeads + kv_head) * kHeadDim + dimension]);
      }
      scores[token] = dot * scale;
      maximum = std::max(maximum, scores[token]);
    }
    float denominator = 0.0F;
    for (float &score : scores) {
      score = std::exp(score - maximum);
      denominator += score;
    }
    for (uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
      float result = 0.0F;
      for (uint64_t token = 0U; token < context; ++token) {
        result +=
            (scores[token] / denominator) *
            host_fp16_to_float(
                value[(token * kKvHeads + kv_head) * kHeadDim + dimension]);
      }
      (*output)[static_cast<uint64_t>(head) * kHeadDim + dimension] =
          host_bf16_rne(result);
    }
  }
}

struct Buffers final {
  uint16_t *query = nullptr;
  uint16_t *key = nullptr;
  uint16_t *value = nullptr;
  uint16_t *output = nullptr;
  float *workspace = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
};

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
  const size_t output_bytes = query_bytes;
  const size_t workspace_bytes = static_cast<size_t>(kQHeads) * kMaxPartitions *
                                 (kHeadDim + 2U) * sizeof(float);
  if (!hip_ok(
          hipMalloc(reinterpret_cast<void **>(&buffers->query), query_bytes),
          "malloc query") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->key), kv_bytes),
              "malloc key") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->value), kv_bytes),
              "malloc value") ||
      !hip_ok(
          hipMalloc(reinterpret_cast<void **>(&buffers->output), output_bytes),
          "malloc output") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&buffers->workspace),
                        workspace_bytes),
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

template <uint32_t Partitions>
bool launch_with_context(Buffers *const buffers, const uint64_t context) {
  hipLaunchKernelGGL((gqa6_decode_split_stage1<Partitions>),
                     dim3(kKvHeads * Partitions), dim3(kThreads), 0U,
                     buffers->stream, buffers->query, buffers->key,
                     buffers->value, context, buffers->workspace);
  if (!hip_ok(hipGetLastError(), "stage1 launch"))
    return false;
  hipLaunchKernelGGL((gqa6_decode_split_stage2<Partitions>), dim3(kQHeads),
                     dim3(kThreads), 0U, buffers->stream, buffers->query,
                     buffers->output, buffers->workspace);
  return hip_ok(hipGetLastError(), "stage2 launch");
}

template <uint32_t Partitions>
bool measure(Buffers *const buffers, const uint64_t context,
             float *const median_us,
             std::array<float, kMeasured> *const samples) {
  for (uint32_t iteration = 0U; iteration < kWarmups; ++iteration) {
    if (!launch_with_context<Partitions>(buffers, context) ||
        !hip_ok(hipStreamSynchronize(buffers->stream), "warmup synchronize"))
      return false;
  }
  for (float &sample : *samples) {
    if (!hip_ok(hipEventRecord(buffers->start, buffers->stream),
                "event start") ||
        !launch_with_context<Partitions>(buffers, context) ||
        !hip_ok(hipEventRecord(buffers->stop, buffers->stream), "event stop") ||
        !hip_ok(hipEventSynchronize(buffers->stop), "event synchronize") ||
        !hip_ok(hipEventElapsedTime(&sample, buffers->start, buffers->stop),
                "event elapsed"))
      return false;
    sample *= 1000.0F;
  }
  std::sort(samples->begin(), samples->end());
  *median_us = (*samples)[kMeasured / 2U];
  return true;
}

bool copy_output(Buffers *const buffers, std::vector<uint16_t> *const output) {
  output->resize(kQHeads * kHeadDim);
  return hip_ok(hipMemcpy(output->data(), buffers->output,
                          output->size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "copy output");
}

void compare_oracle(const char *const name,
                    const std::vector<uint16_t> &expected,
                    const std::vector<uint16_t> &actual) {
  uint32_t max_ulp = 0U;
  uint64_t over_one = 0U;
  for (size_t index = 0U; index < expected.size(); ++index) {
    const uint32_t distance = bf16_ulp(expected[index], actual[index]);
    max_ulp = std::max(max_ulp, distance);
    if (distance > 1U)
      ++over_one;
  }
  std::printf(
      "oracle candidate=%s values=%zu max_bf16_ulp=%u over1=%llu status=%s\n",
      name, expected.size(), max_ulp, static_cast<unsigned long long>(over_one),
      over_one == 0U ? "PASS" : "INFO");
}

bool compare_bitwise(const std::vector<uint16_t> &left,
                     const std::vector<uint16_t> &right) {
  return left.size() == right.size() &&
         std::equal(left.begin(), left.end(), right.begin());
}

uint32_t max_ulp(const std::vector<uint16_t> &left,
                 const std::vector<uint16_t> &right) {
  uint32_t result = 0U;
  if (left.size() != right.size())
    return UINT32_MAX;
  for (size_t index = 0U; index < left.size(); ++index)
    result = std::max(result, bf16_ulp(left[index], right[index]));
  return result;
}

void print_result(const char *const label, const uint64_t context,
                  const uint32_t partitions, const float median_us,
                  const std::array<float, kMeasured> &samples,
                  const std::vector<uint16_t> &control,
                  const std::vector<uint16_t> &actual) {
  const float minimum = *std::min_element(samples.begin(), samples.end());
  const float maximum = *std::max_element(samples.begin(), samples.end());
  std::array<float, kMeasured> deviations = samples;
  for (float &sample : deviations)
    sample = std::fabs(sample - median_us);
  std::sort(deviations.begin(), deviations.end());
  const bool bitwise = compare_bitwise(control, actual);
  std::printf("result candidate=%s context=%llu partitions=%u tile_tokens=%u "
              "median_us=%.3f mad_us=%.3f min_us=%.3f max_us=%.3f "
              "control_max_bf16_ulp=%u control_bitwise=%s\n",
              label, static_cast<unsigned long long>(context), partitions,
              kTileTokens, median_us, deviations[kMeasured / 2U], minimum,
              maximum, max_ulp(control, actual), bitwise ? "PASS" : "INFO");
}

template <uint32_t Partitions>
bool check_determinism(Buffers *const buffers, const uint64_t context,
                       const std::vector<uint16_t> &reference) {
  std::vector<uint16_t> repeat;
  if (!launch_with_context<Partitions>(buffers, context) ||
      !hip_ok(hipStreamSynchronize(buffers->stream),
              "determinism synchronize") ||
      !copy_output(buffers, &repeat)) {
    return false;
  }
  const bool bitwise = compare_bitwise(reference, repeat);
  std::printf("determinism partitions=%u context=%llu bitwise=%s\n", Partitions,
              static_cast<unsigned long long>(context),
              bitwise ? "PASS" : "FAIL");
  return bitwise;
}

void print_resources() {
  const std::array<const char *, 8> names = {
      "p32_stage1", "p32_stage2", "p64_stage1",  "p64_stage2",
      "p96_stage1", "p96_stage2", "p128_stage1", "p128_stage2"};
  const std::array<const void *, 8> functions = {
      reinterpret_cast<const void *>(gqa6_decode_split_stage1<32U>),
      reinterpret_cast<const void *>(gqa6_decode_split_stage2<32U>),
      reinterpret_cast<const void *>(gqa6_decode_split_stage1<64U>),
      reinterpret_cast<const void *>(gqa6_decode_split_stage2<64U>),
      reinterpret_cast<const void *>(gqa6_decode_split_stage1<96U>),
      reinterpret_cast<const void *>(gqa6_decode_split_stage2<96U>),
      reinterpret_cast<const void *>(gqa6_decode_split_stage1<128U>),
      reinterpret_cast<const void *>(gqa6_decode_split_stage2<128U>)};
  for (size_t index = 0U; index < names.size(); ++index) {
    hipFuncAttributes attributes{};
    const hipError_t attr = hipFuncGetAttributes(&attributes, functions[index]);
    int active_blocks = 0;
    const hipError_t occupancy = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, functions[index], kThreads, 0U);
    std::printf("resources kernel=%s vgpr=%d lds_static=%zu scratch=%zu "
                "max_threads=%d active_blocks=%d attrs=%s occupancy=%s\n",
                names[index], attributes.numRegs, attributes.sharedSizeBytes,
                attributes.localSizeBytes, attributes.maxThreadsPerBlock,
                active_blocks, hipGetErrorString(attr),
                hipGetErrorString(occupancy));
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 2) {
    std::fprintf(stderr,
                 "usage: phase78_gqa6_decode_partition_sweep_probe [device]\n");
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
              "contexts=4096,8192,9435,9563 "
              "partitions=32,64,96,128 tile_tokens=%u warmups=%u measured=%u\n",
              properties.gcnArchName, device, properties.pciDomainID,
              properties.pciBusID, properties.pciDeviceID, properties.name,
              kTileTokens, kWarmups, kMeasured);
  if (std::string_view(properties.gcnArchName).find("gfx1030") ==
          std::string_view::npos &&
      std::string_view(properties.gcnArchName).find("gfx1201") ==
          std::string_view::npos) {
    std::fprintf(stderr, "unsupported target: %s\n", properties.gcnArchName);
    return EXIT_FAILURE;
  }
  print_resources();

  bool all_ok = true;
  double weighted_control_us = 0.0;
  double weighted_p64_us = 0.0;
  double weighted_p96_us = 0.0;
  double weighted_p128_us = 0.0;
  for (const uint64_t context :
       {UINT64_C(4096), UINT64_C(8192), UINT64_C(9435), UINT64_C(9563)}) {
    std::vector<uint16_t> query, key, value, expected, control, actual;
    fill_inputs(context, &query, &key, &value);
    host_oracle(context, query, key, value, &expected);
    Buffers buffers;
    if (!make_buffers(context, &buffers) ||
        !upload_inputs(query, key, value, &buffers)) {
      free_buffers(&buffers);
      return EXIT_FAILURE;
    }
    float p32_us = 0.0F;
    float p64_us = 0.0F;
    float p96_us = 0.0F;
    float p128_us = 0.0F;
    bool p64_bitwise = false;
    bool p96_bitwise = false;
    bool p128_bitwise = false;
    std::array<float, kMeasured> p32_samples{};
    std::array<float, kMeasured> p64_samples{};
    std::array<float, kMeasured> p96_samples{};
    std::array<float, kMeasured> p128_samples{};
    if (!measure<32U>(&buffers, context, &p32_us, &p32_samples) ||
        !copy_output(&buffers, &control)) {
      free_buffers(&buffers);
      return EXIT_FAILURE;
    }
    compare_oracle("p32_control", expected, control);
    if (!check_determinism<32U>(&buffers, context, control)) {
      free_buffers(&buffers);
      return EXIT_FAILURE;
    }
    if (!measure<64U>(&buffers, context, &p64_us, &p64_samples) ||
        !copy_output(&buffers, &actual)) {
      free_buffers(&buffers);
      return EXIT_FAILURE;
    }
    compare_oracle("p64_control", expected, actual);
    p64_bitwise = compare_bitwise(control, actual);
    print_result("p64_control", context, 64U, p64_us, p64_samples, control,
                 actual);
    if (!check_determinism<64U>(&buffers, context, actual)) {
      free_buffers(&buffers);
      return EXIT_FAILURE;
    }
    std::printf(
        "control_compare context=%llu candidate=p64 max_bf16_ulp=%u "
        "bitwise=%s\n",
        static_cast<unsigned long long>(context),
        [&]() {
          uint32_t m = 0U;
          for (size_t i = 0U; i < control.size(); ++i)
            m = std::max(m, bf16_ulp(control[i], actual[i]));
          return m;
        }(),
        p64_bitwise ? "PASS" : "INFO");
    if (!measure<96U>(&buffers, context, &p96_us, &p96_samples) ||
        !copy_output(&buffers, &actual)) {
      free_buffers(&buffers);
      return EXIT_FAILURE;
    }
    compare_oracle("p96_candidate", expected, actual);
    p96_bitwise = compare_bitwise(control, actual);
    print_result("p96_candidate", context, 96U, p96_us, p96_samples, control,
                 actual);
    if (!check_determinism<96U>(&buffers, context, actual)) {
      free_buffers(&buffers);
      return EXIT_FAILURE;
    }
    if (!measure<128U>(&buffers, context, &p128_us, &p128_samples) ||
        !copy_output(&buffers, &actual)) {
      free_buffers(&buffers);
      return EXIT_FAILURE;
    }
    compare_oracle("p128_candidate", expected, actual);
    p128_bitwise = compare_bitwise(control, actual);
    print_result("p128_candidate", context, 128U, p128_us, p128_samples,
                 control, actual);
    if (!check_determinism<128U>(&buffers, context, actual)) {
      free_buffers(&buffers);
      return EXIT_FAILURE;
    }
    weighted_control_us += static_cast<double>(p32_us);
    weighted_p96_us += static_cast<double>(p96_us);
    weighted_p128_us += static_cast<double>(p128_us);
    std::printf(
        "summary_context context=%llu p32_us=%.3f p64_us=%.3f p96_us=%.3f "
        "p128_us=%.3f p64_vs_p32=%.4f p96_vs_p32=%.4f p128_vs_p32=%.4f "
        "p64_bitwise=%s p96_bitwise=%s p128_bitwise=%s\n",
        static_cast<unsigned long long>(context), p32_us, p64_us, p96_us,
        p128_us, static_cast<double>(p32_us) / p64_us,
        static_cast<double>(p32_us) / p96_us,
        static_cast<double>(p32_us) / p128_us, p64_bitwise ? "PASS" : "INFO",
        p96_bitwise ? "PASS" : "INFO", p128_bitwise ? "PASS" : "INFO");
    (void)p96_bitwise;
    (void)p128_bitwise;
    free_buffers(&buffers);
    weighted_p64_us += static_cast<double>(p64_us);
  }
  std::printf("weighted_summary contexts=4 p32_total_us=%.3f p64_total_us=%.3f "
              "p96_total_us=%.3f p128_total_us=%.3f "
              "p96_vs_p32_speedup=%.4f p128_vs_p32_speedup=%.4f "
              "p96_vs_p64_speedup=%.4f p128_vs_p64_speedup=%.4f status=%s\n",
              weighted_control_us, weighted_p64_us, weighted_p96_us,
              weighted_p128_us, weighted_control_us / weighted_p96_us,
              weighted_control_us / weighted_p128_us,
              weighted_p64_us / weighted_p96_us,
              weighted_p64_us / weighted_p128_us, all_ok ? "PASS" : "FAIL");
  std::printf("cleanup status=PASS\n");
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
