#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#if !defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
#include <hipblas/hipblas.h>
#include <mutex>
#include <rocblas/rocblas.h>
#endif

#include "causal_attention_kernel_internal.hpp"
#include "paged_kv_device_layout.hpp"
#include "sllm/hip.h"
#include <lowp/detail/low_precision_block_codec.hpp>

namespace sllm_causal_attention_kernel {
namespace {

__device__ __forceinline__ bool
decode_control_phase(sllm_decode_control::ControlV1 *const control,
                     const uint32_t query_index, const uint32_t query_count,
                     uint64_t *const start_position,
                     uint32_t *const phase_rows) noexcept {
  if (control == nullptr) {
    return true;
  }
  if (control->phase_active == 0U || control->halted != 0U ||
      control->phase_rows == 0U || query_index >= control->phase_rows ||
      query_count == 0U) {
    return false;
  }
  if (control->phase_position > UINT64_MAX - query_index) {
    atomicExch(
        &control->status,
        static_cast<uint32_t>(sllm_decode_control::Status::InvalidPosition));
    atomicExch(&control->halted, 1U);
    atomicExch(&control->phase_active, 0U);
    return false;
  }
  *start_position = control->phase_position;
  *phase_rows = control->phase_rows;
  return true;
}

__device__ __forceinline__ uint32_t
decode_control_split_count(sllm_decode_control::ControlV1 *const control,
                           const uint32_t fallback) noexcept {
  if (control == nullptr) {
    return fallback;
  }
  // The normal selector uses P128 when the committed tail reaches 8192.
  // Keep this threshold in the device path so graph replay can cross it
  // without changing a kernel node or reinstantiating the graph.
  return sllm_causal_attention_kernel::decode_dynamic_split_count(
      control->phase_position, control->phase_rows);
}

__device__ __forceinline__ void mark_paged_attention_invalid(
    uint32_t *const device_status,
    sllm_decode_control::ControlV1 *const control) noexcept {
  if (device_status != nullptr) {
    atomicExch(device_status, 1U);
  }
  if (control != nullptr) {
    atomicExch(
        &control->status,
        static_cast<uint32_t>(sllm_decode_control::Status::InvalidPosition));
    atomicExch(&control->halted, 1U);
    atomicExch(&control->phase_active, 0U);
  }
}

// Keep the QK reduction, online softmax update, and value accumulation in one
// device helper.  The staged decode control and the C1 row-sharing candidate
// both call this helper so a candidate topology cannot silently change the
// arithmetic order of the reviewed provider.
template <uint32_t kElementsPerLane>
__device__ __forceinline__ void accumulate_mxfp8_e4_decode_key(
    const float *const query_values, const float (*const key_tile)[256U],
    const float (*const value_tile)[256U], const uint32_t key_index,
    const uint32_t lane, float *const accumulations,
    float *const running_maximum, float *const running_denominator) noexcept {
  constexpr uint32_t kWaveSize = 32U;
  constexpr uint32_t kHeadDim = 256U;
  float partial_score = 0.0F;
#pragma unroll
  for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
    const uint32_t current = lane + index * kWaveSize;
    partial_score += query_values[index] * key_tile[key_index][current];
  }
  for (uint32_t offset = kWaveSize / 2U; offset != 0U; offset >>= 1U) {
    partial_score += __shfl_down(partial_score, offset, kWaveSize);
  }
  float rescale = 0.0F;
  float contribution = 0.0F;
  if (lane == 0U) {
    const float current_score =
        partial_score * rsqrtf(static_cast<float>(kHeadDim));
    const float next_maximum = fmaxf(*running_maximum, current_score);
    rescale = expf(*running_maximum - next_maximum);
    contribution = expf(current_score - next_maximum);
    *running_denominator = *running_denominator * rescale + contribution;
    *running_maximum = next_maximum;
  }
  rescale = __shfl(rescale, 0U, kWaveSize);
  contribution = __shfl(contribution, 0U, kWaveSize);
#pragma unroll
  for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
    const uint32_t current = lane + index * kWaveSize;
    accumulations[index] = accumulations[index] * rescale +
                           contribution * value_tile[key_index][current];
  }
}

__device__ float f16_to_f32(const uint16_t raw) noexcept {
#if defined(__gfx1030__) || defined(__gfx1201__)
  // Every finite FP16 value, including subnormals and signed zero, converts
  // exactly to FP32. Preserve the storage decoder's NaN payload/sign bits.
  if ((raw & 0x7c00U) == 0x7c00U) {
    return __uint_as_float((static_cast<uint32_t>(raw & 0x8000U) << 16U) |
                           0x7f800000U |
                           (static_cast<uint32_t>(raw & 0x03ffU) << 13U));
  }
  return __half2float(__ushort_as_half(raw));
#else
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
#endif
}

__device__ float bf16_to_f32(const uint16_t raw) noexcept {
  return __uint_as_float(static_cast<uint32_t>(raw) << 16U);
}

__device__ float e4m3fn_to_f32(const uint8_t bits) noexcept {
  return sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode(bits);
}

template <uint32_t Encoding>
__device__ __forceinline__ float
load_kv_specialized(const void *const values, const void *const scales,
                    const float *const outer_scales, const uint64_t row,
                    const uint32_t dimension, const uint32_t head_dim,
                    const float static_scale) noexcept {
  if constexpr (Encoding == SLLM_HIP_KV_ENCODING_FP16_V1) {
    return f16_to_f32(
        static_cast<const uint16_t *>(values)[row * head_dim + dimension]);
  } else if constexpr (Encoding == SLLM_HIP_KV_ENCODING_FP8_V1 ||
                       Encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1) {
    const float scale = Encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1
                            ? static_scale
                            : static_cast<const float *>(scales)[row];
    return e4m3fn_to_f32(static_cast<const uint8_t *>(
               values)[row * head_dim + dimension]) *
           scale;
  } else if constexpr (Encoding == SLLM_HIP_KV_ENCODING_MXFP8_E4_V1) {
    const auto view =
        sllm_lowp::make_block_scaled_view<sllm_lowp::Mxfp8E4Block32>(
            values, scales, nullptr, head_dim);
    return sllm_lowp::BlockCodec<sllm_lowp::Mxfp8E4Block32>::load(view, row,
                                                                  dimension);
  } else if constexpr (Encoding == SLLM_HIP_KV_ENCODING_MXFP8_E5_V1) {
    const auto view =
        sllm_lowp::make_block_scaled_view<sllm_lowp::Mxfp8E5Block32>(
            values, scales, nullptr, head_dim);
    return sllm_lowp::BlockCodec<sllm_lowp::Mxfp8E5Block32>::load(view, row,
                                                                  dimension);
  } else if constexpr (Encoding == SLLM_HIP_KV_ENCODING_NVFP4_V1) {
    const auto view =
        sllm_lowp::make_block_scaled_view<sllm_lowp::Nvfp4Block16>(
            values, scales, outer_scales, head_dim);
    return sllm_lowp::BlockCodec<sllm_lowp::Nvfp4Block16>::load(view, row,
                                                                dimension);
  } else {
    return NAN;
  }
}

// The qtile4 provider assigns one 32-lane wave to each 32-value MX block.
// Load and decode the shared E8M0 scale once per wave instead of repeating the
// same work in every lane.  Other providers keep the format-neutral loader
// because their lane-to-dimension mapping is not guaranteed to match a block.
template <typename BlockFormat, bool UseFusedE4Decode = false>
__device__ float load_kv_qtile4_mx(const void *const values,
                                   const void *const scales, const uint64_t row,
                                   const uint32_t dimension,
                                   const uint32_t head_dim) noexcept {
  const auto view = sllm_lowp::make_block_scaled_view<BlockFormat>(
      values, scales, nullptr, head_dim);
  const uint8_t value = view.values[row * view.value_stride + dimension];
  if constexpr (UseFusedE4Decode && BlockFormat::kElementPower == 8) {
    uint32_t scale_code = 0U;
    if ((dimension & (BlockFormat::kBlockSize - 1U)) == 0U) {
      scale_code = static_cast<uint32_t>(
          view.block_scales[row * view.scale_stride +
                            dimension / BlockFormat::kBlockSize]);
    }
    scale_code = __shfl(scale_code, 0U, BlockFormat::kBlockSize);
    return sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode_scaled(
        value, static_cast<uint8_t>(scale_code));
  } else {
    const float decoded =
        sllm_lowp::ScalarCodec<typename BlockFormat::Element>::decode(value);
    float scale = 0.0F;
    if ((dimension & (BlockFormat::kBlockSize - 1U)) == 0U) {
      const uint8_t scale_bits =
          view.block_scales[row * view.scale_stride +
                            dimension / BlockFormat::kBlockSize];
      scale = sllm_lowp::ScalarCodec<sllm_lowp::E8M0>::decode(scale_bits);
    }
    scale = __shfl(scale, 0U, BlockFormat::kBlockSize);
    return decoded * scale;
  }
}

template <uint32_t Encoding, bool UseFusedE4Decode = false>
__device__ __forceinline__ float
load_kv_qtile4(const void *const values, const void *const scales,
               const float *const outer_scales, const uint64_t row,
               const uint32_t dimension, const uint32_t head_dim,
               const float static_scale) noexcept {
  if constexpr (Encoding == SLLM_HIP_KV_ENCODING_MXFP8_E4_V1) {
    return load_kv_qtile4_mx<sllm_lowp::Mxfp8E4Block32, UseFusedE4Decode>(
        values, scales, row, dimension, head_dim);
  } else if constexpr (Encoding == SLLM_HIP_KV_ENCODING_MXFP8_E5_V1) {
    return load_kv_qtile4_mx<sllm_lowp::Mxfp8E5Block32, UseFusedE4Decode>(
        values, scales, row, dimension, head_dim);
  } else {
    return load_kv_specialized<Encoding>(values, scales, outer_scales, row,
                                         dimension, head_dim, static_scale);
  }
}

__device__ uint16_t f32_to_bf16_rne(const float value) noexcept {
  const uint32_t bits = __float_as_uint(value);
  constexpr uint32_t exponent_mask = UINT32_C(0x7f800000);
  constexpr uint32_t fraction_mask = UINT32_C(0x007fffff);
  if ((bits & exponent_mask) == exponent_mask) {
    if ((bits & fraction_mask) != 0U) {
      const uint16_t sign =
          static_cast<uint16_t>((bits >> 16U) & UINT32_C(0x8000));
      const uint16_t payload =
          static_cast<uint16_t>((bits >> 16U) & UINT32_C(0x003f));
      return static_cast<uint16_t>(sign | UINT16_C(0x7fc0) | payload);
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & UINT32_C(1)) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

template <uint32_t kWaveCount, bool DeviceControl = false>
__global__
__launch_bounds__(256, 1) void causal_attention_decode_wave_split_staged_stage2_kernel(
    const float *const workspace, uint16_t *const output,
    const uint32_t query_count, const uint32_t q_heads, const uint32_t kv_heads,
    const uint32_t head_dim, sllm_decode_control::ControlV1 *const control) {
  constexpr uint32_t kHeadDim = SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM;
  const uint64_t flat = static_cast<uint64_t>(blockIdx.x);
  const uint32_t query_index = static_cast<uint32_t>(flat / q_heads);
  const uint32_t query_head = static_cast<uint32_t>(flat % q_heads);
  if (query_index >= query_count || query_head >= q_heads || kv_heads == 0U ||
      head_dim != kHeadDim) {
    return;
  }
  if constexpr (DeviceControl) {
    uint64_t control_start_position = 0U;
    uint32_t control_rows = query_count;
    if (!decode_control_phase(control, query_index, query_count,
                              &control_start_position, &control_rows)) {
      return;
    }
  }
  const uint32_t dimension = threadIdx.x;
  constexpr uint64_t kStride = static_cast<uint64_t>(kHeadDim) + 2U;
  const uint64_t head_base =
      (static_cast<uint64_t>(query_index) * q_heads + query_head) * kWaveCount *
      kStride;
  __shared__ float partial_values[kWaveCount * kHeadDim];
  __shared__ float partial_maxima[kWaveCount];
  __shared__ float partial_denominators[kWaveCount];
  for (uint32_t index = dimension; index < kWaveCount; index += 256U) {
    partial_maxima[index] = workspace[head_base + index * kStride];
    partial_denominators[index] = workspace[head_base + index * kStride + 1U];
  }
  for (uint32_t index = dimension; index < kWaveCount * kHeadDim;
       index += 256U) {
    const uint32_t split = index / kHeadDim;
    const uint32_t current = index % kHeadDim;
    partial_values[index] =
        workspace[head_base + split * kStride + 2U + current];
  }
  __syncthreads();

  float global_maximum = partial_maxima[0];
#pragma unroll
  for (uint32_t split = 1U; split < kWaveCount; ++split) {
    global_maximum = fmaxf(global_maximum, partial_maxima[split]);
  }
  float global_denominator = 0.0F;
  float merged = 0.0F;
#pragma unroll
  for (uint32_t split = 0U; split < kWaveCount; ++split) {
    const float scale = expf(partial_maxima[split] - global_maximum);
    global_denominator += partial_denominators[split] * scale;
    if (dimension < head_dim) {
      merged += partial_values[split * kHeadDim + dimension] * scale;
    }
  }
  uint16_t *const output_row =
      output +
      (static_cast<uint64_t>(query_index) * q_heads + query_head) * head_dim;
  if (dimension < head_dim) {
    output_row[dimension] = f32_to_bf16_rne(merged / global_denominator);
  }
  const uint32_t second = dimension + 256U;
  if (second < head_dim) {
    float merged_second = 0.0F;
#pragma unroll
    for (uint32_t split = 0U; split < kWaveCount; ++split) {
      const float scale = expf(partial_maxima[split] - global_maximum);
      merged_second += partial_values[split * kHeadDim + second] * scale;
    }
    output_row[second] = f32_to_bf16_rne(merged_second / global_denominator);
  }
}

// Global-workspace merge for WU1.1 split counts.  The per-dimension loop is
// intentionally sequential and matches the scratch candidate; no LDS array of
// Splits*head_dim is allocated, so Split=128 remains valid on both targets.
template <uint32_t kGridSplits, uint32_t kActiveSplits>
__device__ __forceinline__ void
causal_attention_decode_gqa6_staged32_split_merge_body(
    const float *const workspace, uint16_t *const output, const uint32_t head,
    const uint32_t dimension) {
  constexpr uint32_t kHeadDim = 256U;
  constexpr uint32_t kWorkspaceStride = kHeadDim + 2U;
  const uint64_t base =
      static_cast<uint64_t>(head) * kGridSplits * kWorkspaceStride;
  float maximum = -std::numeric_limits<float>::infinity();
  for (uint32_t split = 0U; split < kActiveSplits; ++split) {
    maximum = fmaxf(maximum, workspace[base + split * kWorkspaceStride]);
  }
  float denominator = 0.0F;
  float accumulated = 0.0F;
  for (uint32_t split = 0U; split < kActiveSplits; ++split) {
    const uint64_t partial = base + split * kWorkspaceStride;
    const float scale = expf(workspace[partial] - maximum);
    denominator += workspace[partial + 1U] * scale;
    accumulated += workspace[partial + 2U + dimension] * scale;
  }
  output[static_cast<uint64_t>(head) * kHeadDim + dimension] =
      f32_to_bf16_rne(accumulated / denominator);
}

template <uint32_t kGridSplits, bool DeviceControl = false>
__global__
__launch_bounds__(256, 1) void causal_attention_decode_gqa6_staged32_split_merge_kernel(
    const float *const workspace, uint16_t *const output,
    const uint32_t query_count, sllm_decode_control::ControlV1 *const control) {
  constexpr uint32_t kQHeads = 24U;
  constexpr uint32_t kHeadDim = 256U;
  const uint32_t head = blockIdx.x;
  const uint32_t dimension = threadIdx.x;
  if (head >= query_count * kQHeads || dimension >= kHeadDim) {
    return;
  }
  const uint32_t query_index = head / kQHeads;
  if constexpr (DeviceControl) {
    uint64_t control_start_position = 0U;
    uint32_t control_rows = query_count;
    if (!decode_control_phase(control, query_index, query_count,
                              &control_start_position, &control_rows)) {
      return;
    }
    const uint32_t active_splits =
        decode_control_split_count(control, kGridSplits);
    if (active_splits == 32U) {
      causal_attention_decode_gqa6_staged32_split_merge_body<kGridSplits, 32U>(
          workspace, output, head, dimension);
    } else {
      causal_attention_decode_gqa6_staged32_split_merge_body<kGridSplits, 128U>(
          workspace, output, head, dimension);
    }
  } else {
    causal_attention_decode_gqa6_staged32_split_merge_body<kGridSplits,
                                                           kGridSplits>(
        workspace, output, head, dimension);
  }
}

template <uint32_t Encoding>
__device__ __forceinline__ bool
paged_descriptor_valid(const sllm_paged_kv::BlockDescriptor &page) noexcept;

// Paged counterpart of causal_attention_prefill_gqa4_shared_kernel.  The
// reduction tree, online-softmax update, and BF16 output conversion are kept
// identical; only logical page resolution replaces the contiguous row base.
__global__
__launch_bounds__(256, 1) void causal_attention_paged_prefill_gqa4_shared_kernel(
    const uint16_t *const query, const uint32_t *const logical_table,
    const sllm_paged_kv::BlockDescriptor *const descriptor_table,
    const uint32_t logical_table_count, const uint32_t descriptor_count,
    uint32_t *const device_status, uint16_t *const output,
    const uint32_t query_count, const uint64_t start_position,
    const uint64_t committed_kv_length, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim) {
  constexpr uint32_t kWaveSize = 32U;
  constexpr uint32_t kWaveCount = 8U;
  constexpr uint32_t kGqaRatio = 4U;
  const uint64_t flat = blockIdx.x;
  const uint64_t row = flat / kv_heads;
  const uint32_t kv_head = static_cast<uint32_t>(flat % kv_heads);
  if (row >= query_count || q_heads != 16U || kv_heads != 4U ||
      head_dim != 256U) {
    return;
  }
  const uint64_t query_position = start_position + row;
  if (query_position >= committed_kv_length) {
    return;
  }
  const uint32_t dimension = threadIdx.x;
  const uint32_t lane = dimension & (kWaveSize - 1U);
  const uint32_t wave = dimension / kWaveSize;
  const uint32_t first_query_head = kv_head * kGqaRatio;
  __shared__ float reductions[kGqaRatio][kWaveCount];
  __shared__ float rescale[kGqaRatio];
  __shared__ float contribution[kGqaRatio];
  __shared__ float running_maximum[kGqaRatio];
  __shared__ float running_denominator[kGqaRatio];
  __shared__ sllm_paged_kv::BlockDescriptor page;
  __shared__ uint32_t page_valid;
  if (dimension < kGqaRatio) {
    running_maximum[dimension] = -std::numeric_limits<float>::infinity();
    running_denominator[dimension] = 0.0F;
  }
  __syncthreads();

  float accumulations[kGqaRatio] = {0.0F, 0.0F, 0.0F, 0.0F};
  for (uint64_t key_position = 0U; key_position <= query_position;
       ++key_position) {
    if (dimension == 0U) {
      const uint64_t logical_page = key_position / 128U;
      const uint32_t local_token = static_cast<uint32_t>(key_position % 128U);
      (void)local_token;
      const uint32_t physical_page = logical_page < logical_table_count
                                         ? logical_table[logical_page]
                                         : UINT32_MAX;
      page_valid =
          physical_page != UINT32_MAX && physical_page < descriptor_count ? 1U
                                                                          : 0U;
      if (page_valid != 0U) {
        page = descriptor_table[physical_page];
        page_valid = paged_descriptor_valid<SLLM_HIP_KV_ENCODING_FP16_V1>(page)
                         ? 1U
                         : 0U;
      }
      if (page_valid == 0U) {
        atomicExch(device_status, 1U);
      }
    }
    __syncthreads();
    if (page_valid == 0U) {
      return;
    }
    const uint64_t kv_row = (key_position % 128U) * kv_heads + kv_head;
    const uint16_t *const query_row =
        query +
        (row * q_heads + first_query_head) * static_cast<uint64_t>(head_dim);
    const float key_value =
        dimension < head_dim
            ? load_kv_specialized<SLLM_HIP_KV_ENCODING_FP16_V1>(
                  page.key, page.key_scale,
                  reinterpret_cast<const float *>(page.key_outer_scale), kv_row,
                  dimension, head_dim, 1.0F)
            : 0.0F;
#pragma unroll
    for (uint32_t head = 0U; head < kGqaRatio; ++head) {
      const uint16_t *const head_query =
          query_row + static_cast<uint64_t>(head) * head_dim;
      float partial = dimension < head_dim
                          ? bf16_to_f32(head_query[dimension]) * key_value
                          : 0.0F;
      for (uint32_t offset = kWaveSize / 2U; offset != 0U; offset >>= 1U) {
        partial += __shfl_down(partial, offset, kWaveSize);
      }
      if (lane == 0U) {
        reductions[head][wave] = partial;
      }
    }
    __syncthreads();
    if (wave < kGqaRatio) {
      float block_sum = lane < kWaveCount ? reductions[wave][lane] : 0.0F;
      for (uint32_t offset = kWaveSize / 2U; offset != 0U; offset >>= 1U) {
        block_sum += __shfl_down(block_sum, offset, kWaveSize);
      }
      if (lane == 0U) {
        const float current_score =
            block_sum * rsqrtf(static_cast<float>(head_dim));
        const float next_maximum = fmaxf(running_maximum[wave], current_score);
        rescale[wave] = expf(running_maximum[wave] - next_maximum);
        contribution[wave] = expf(current_score - next_maximum);
        running_denominator[wave] =
            running_denominator[wave] * rescale[wave] + contribution[wave];
        running_maximum[wave] = next_maximum;
      }
    }
    __syncthreads();
    const float value_element =
        dimension < head_dim
            ? load_kv_specialized<SLLM_HIP_KV_ENCODING_FP16_V1>(
                  page.value, page.value_scale,
                  reinterpret_cast<const float *>(page.value_outer_scale),
                  kv_row, dimension, head_dim, 1.0F)
            : 0.0F;
    if (dimension < head_dim) {
#pragma unroll
      for (uint32_t head = 0U; head < kGqaRatio; ++head) {
        accumulations[head] = accumulations[head] * rescale[head] +
                              contribution[head] * value_element;
      }
    }
    __syncthreads();
  }
  if (dimension < head_dim) {
#pragma unroll
    for (uint32_t head = 0U; head < kGqaRatio; ++head) {
      uint16_t *const output_row =
          output + (row * q_heads + first_query_head + head) *
                       static_cast<uint64_t>(head_dim);
      output_row[dimension] =
          f32_to_bf16_rne(accumulations[head] / running_denominator[head]);
    }
  }
}

// Phase 87 Stage10 generic paged control path.  The body deliberately mirrors
// causal_attention_kernel<false, Encoding> in its query reduction,
// online-softmax update, accumulation, and BF16 RNE output.  The only changed
// operation is resolving each logical 128-token page through the descriptor
// table before loading the K/V row.
template <uint32_t Encoding>
__device__ __forceinline__ bool
paged_descriptor_valid(const sllm_paged_kv::BlockDescriptor &page) noexcept {
  if (page.key == nullptr || page.value == nullptr) {
    return false;
  }
  if constexpr (Encoding == SLLM_HIP_KV_ENCODING_FP16_V1 ||
                Encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1) {
    return true;
  } else if constexpr (Encoding == SLLM_HIP_KV_ENCODING_FP8_V1 ||
                       Encoding == SLLM_HIP_KV_ENCODING_MXFP8_E4_V1 ||
                       Encoding == SLLM_HIP_KV_ENCODING_MXFP8_E5_V1) {
    return page.key_scale != nullptr && page.value_scale != nullptr;
  } else if constexpr (Encoding == SLLM_HIP_KV_ENCODING_NVFP4_V1) {
    return page.key_scale != nullptr && page.value_scale != nullptr &&
           page.key_outer_scale != nullptr && page.value_outer_scale != nullptr;
  } else {
    return false;
  }
}

template <uint32_t Encoding, bool UseWaveProvider = false>
__global__ __launch_bounds__(256, 1) void causal_attention_paged_kernel(
    const uint16_t *const query, const uint32_t *const logical_table,
    const sllm_paged_kv::BlockDescriptor *const descriptor_table,
    const uint32_t logical_table_count, const uint32_t descriptor_count,
    uint32_t *const device_status, uint16_t *const output,
    const uint32_t query_count, const uint64_t start_position,
    const uint64_t committed_kv_length, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim,
    const float static_key_scale, const float static_value_scale,
    const float score_scale) {
  const uint64_t flat = static_cast<uint64_t>(blockIdx.x);
  if (flat >= static_cast<uint64_t>(query_count) * q_heads) {
    return;
  }
  const uint64_t row = flat / q_heads;
  const uint32_t query_head = static_cast<uint32_t>(flat % q_heads);
  const uint64_t query_position = start_position + row;
  if (query_position >= committed_kv_length) {
    return;
  }
  const uint16_t *const query_row =
      query + row * q_heads * head_dim +
      static_cast<uint64_t>(query_head) * head_dim;
  const uint32_t kv_head = query_head / (q_heads / kv_heads);
  uint16_t *const output_row = output + row * q_heads * head_dim +
                               static_cast<uint64_t>(query_head) * head_dim;

  const uint32_t dimension = threadIdx.x;
  __shared__ float reductions[SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE];
  __shared__ float rescale;
  __shared__ float contribution;
  __shared__ float running_maximum;
  __shared__ float running_denominator;
  __shared__ sllm_paged_kv::BlockDescriptor page;
  __shared__ uint32_t page_valid;
  if (dimension == 0U) {
    running_maximum = -std::numeric_limits<float>::infinity();
    running_denominator = 0.0F;
  }
  __syncthreads();

  float accumulation0 = 0.0F;
  float accumulation1 = 0.0F;
  const uint64_t key_begin = 0U;
  for (uint64_t key_position = key_begin; key_position <= query_position;
       ++key_position) {
    const uint64_t logical_page = key_position / 128U;
    const uint32_t local_token = static_cast<uint32_t>(key_position % 128U);
    if (dimension == 0U) {
      const uint32_t physical_page = logical_page < logical_table_count
                                         ? logical_table[logical_page]
                                         : UINT32_MAX;
      page_valid =
          physical_page != UINT32_MAX && physical_page < descriptor_count ? 1U
                                                                          : 0U;
      if (page_valid != 0U) {
        page = descriptor_table[physical_page];
        page_valid = paged_descriptor_valid<Encoding>(page) ? 1U : 0U;
      }
      if (page_valid == 0U) {
        atomicExch(device_status, 1U);
      }
    }
    __syncthreads();
    if (page_valid == 0U) {
      return;
    }
    const uint64_t kv_row =
        static_cast<uint64_t>(local_token) * kv_heads + kv_head;
#if defined(__gfx1201__)
    if constexpr (UseWaveProvider && Encoding == SLLM_HIP_KV_ENCODING_FP16_V1) {
      float partial = 0.0F;
      for (uint32_t current = dimension; current < head_dim;
           current += blockDim.x) {
        partial += bf16_to_f32(query_row[current]) *
                   load_kv_specialized<Encoding>(
                       page.key, page.key_scale,
                       reinterpret_cast<const float *>(page.key_outer_scale),
                       kv_row, current, head_dim, static_key_scale);
      }
      for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
        partial += __shfl_down(partial, offset, 32U);
      }
      const uint32_t lane = dimension & 31U;
      const uint32_t wave = dimension >> 5U;
      if (lane == 0U) {
        reductions[wave] = partial;
      }
      __syncthreads();
      if (wave == 0U) {
        float block_sum = lane < 8U ? reductions[lane] : 0.0F;
        for (uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
          block_sum += __shfl_down(block_sum, offset, 32U);
        }
        if (lane == 0U) {
          const float current_score = block_sum * score_scale;
          const float next_maximum = fmaxf(running_maximum, current_score);
          rescale = expf(running_maximum - next_maximum);
          contribution = expf(current_score - next_maximum);
          running_denominator = running_denominator * rescale + contribution;
          running_maximum = next_maximum;
        }
      }
      __syncthreads();
    } else
#endif
    {
      reductions[dimension] = 0.0F;
      for (uint32_t current = dimension; current < head_dim;
           current += blockDim.x) {
        reductions[dimension] +=
            bf16_to_f32(query_row[current]) *
            load_kv_specialized<Encoding>(
                page.key, page.key_scale,
                reinterpret_cast<const float *>(page.key_outer_scale), kv_row,
                current, head_dim, static_key_scale);
      }
      __syncthreads();
      for (uint32_t stride = SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE / 2U;
           stride != 0U; stride >>= 1U) {
        if (dimension < stride) {
          reductions[dimension] += reductions[dimension + stride];
        }
        __syncthreads();
      }
      if (dimension == 0U) {
        const float current_score = reductions[0] * score_scale;
        const float next_maximum = fmaxf(running_maximum, current_score);
        rescale = expf(running_maximum - next_maximum);
        contribution = expf(current_score - next_maximum);
        running_denominator = running_denominator * rescale + contribution;
        running_maximum = next_maximum;
      }
      __syncthreads();
    }
    if (dimension < head_dim) {
      accumulation0 =
          accumulation0 * rescale +
          contribution *
              load_kv_specialized<Encoding>(
                  page.value, page.value_scale,
                  reinterpret_cast<const float *>(page.value_outer_scale),
                  kv_row, dimension, head_dim, static_value_scale);
    }
    const uint32_t second = dimension + blockDim.x;
    if (second < head_dim) {
      accumulation1 =
          accumulation1 * rescale +
          contribution *
              load_kv_specialized<Encoding>(
                  page.value, page.value_scale,
                  reinterpret_cast<const float *>(page.value_outer_scale),
                  kv_row, second, head_dim, static_value_scale);
    }
    __syncthreads();
  }
  if (dimension < head_dim) {
    output_row[dimension] =
        f32_to_bf16_rne(accumulation0 / running_denominator);
  }
  const uint32_t second = dimension + blockDim.x;
  if (second < head_dim) {
    output_row[second] = f32_to_bf16_rne(accumulation1 / running_denominator);
  }
}

// Capture-stable FP16 paged decode. ControlV1 supplies the phase position and
// active row count at replay time; all table and descriptor pointers remain
// stable across graph execution.
__global__
__launch_bounds__(256, 1) void causal_attention_paged_fp16_device_kernel(
    const uint16_t *const query, const uint32_t *const logical_table,
    const sllm_paged_kv::BlockDescriptor *const descriptor_table,
    const uint32_t logical_table_count, const uint32_t descriptor_count,
    uint32_t *const device_status, uint16_t *const output,
    const uint32_t query_count, const uint32_t q_heads, const uint32_t kv_heads,
    const uint32_t head_dim, sllm_decode_control::ControlV1 *const control) {
  const uint64_t flat = static_cast<uint64_t>(blockIdx.x);
  if (flat >= static_cast<uint64_t>(query_count) * q_heads || q_heads == 0U ||
      kv_heads == 0U || q_heads % kv_heads != 0U) {
    return;
  }
  const uint64_t row = flat / q_heads;
  const uint32_t query_head = static_cast<uint32_t>(flat % q_heads);
  uint64_t effective_start = 0U;
  uint32_t phase_rows = query_count;
  if (!decode_control_phase(control, static_cast<uint32_t>(row), query_count,
                            &effective_start, &phase_rows)) {
    return;
  }
  if (effective_start > UINT64_MAX - static_cast<uint64_t>(row)) {
    mark_paged_attention_invalid(device_status, control);
    return;
  }
  const uint64_t query_position = effective_start + row;
  const uint64_t committed_length = effective_start + phase_rows;
  if (query_position >= committed_length) {
    return;
  }
  const uint16_t *const query_row =
      query + row * q_heads * head_dim +
      static_cast<uint64_t>(query_head) * head_dim;
  const uint32_t kv_head = query_head / (q_heads / kv_heads);
  uint16_t *const output_row = output + row * q_heads * head_dim +
                               static_cast<uint64_t>(query_head) * head_dim;
  const uint32_t dimension = threadIdx.x;
  __shared__ float reductions[SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE];
  __shared__ float rescale;
  __shared__ float contribution;
  __shared__ float running_maximum;
  __shared__ float running_denominator;
  __shared__ sllm_paged_kv::BlockDescriptor page;
  __shared__ uint32_t page_valid;
  if (dimension == 0U) {
    running_maximum = -std::numeric_limits<float>::infinity();
    running_denominator = 0.0F;
  }
  __syncthreads();
  float accumulation0 = 0.0F;
  float accumulation1 = 0.0F;
  for (uint64_t key_position = 0U;; ++key_position) {
    if (key_position > query_position)
      break;
    const uint64_t logical_page = key_position / 128U;
    const uint32_t local_token = static_cast<uint32_t>(key_position % 128U);
    if (dimension == 0U) {
      const uint32_t physical_page = logical_page < logical_table_count
                                         ? logical_table[logical_page]
                                         : UINT32_MAX;
      page_valid =
          physical_page != UINT32_MAX && physical_page < descriptor_count ? 1U
                                                                          : 0U;
      if (page_valid != 0U) {
        page = descriptor_table[physical_page];
        page_valid = page.key != nullptr && page.value != nullptr ? 1U : 0U;
      }
      if (page_valid == 0U)
        mark_paged_attention_invalid(device_status, control);
    }
    __syncthreads();
    if (page_valid == 0U)
      return;
    const uint64_t kv_row = static_cast<uint64_t>(local_token) * kv_heads +
                            static_cast<uint64_t>(kv_head);
    reductions[dimension] = 0.0F;
    for (uint32_t current = dimension; current < head_dim;
         current += blockDim.x) {
      reductions[dimension] +=
          bf16_to_f32(query_row[current]) *
          load_kv_specialized<SLLM_HIP_KV_ENCODING_FP16_V1>(
              page.key, nullptr, nullptr, kv_row, current, head_dim, 1.0F);
    }
    __syncthreads();
    for (uint32_t stride = SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE / 2U;
         stride != 0U; stride >>= 1U) {
      if (dimension < stride)
        reductions[dimension] += reductions[dimension + stride];
      __syncthreads();
    }
    if (dimension == 0U) {
      const float current_score =
          reductions[0] * rsqrtf(static_cast<float>(head_dim));
      const float next_maximum = fmaxf(running_maximum, current_score);
      rescale = expf(running_maximum - next_maximum);
      contribution = expf(current_score - next_maximum);
      running_denominator = running_denominator * rescale + contribution;
      running_maximum = next_maximum;
    }
    __syncthreads();
    if (dimension < head_dim) {
      accumulation0 =
          accumulation0 * rescale +
          contribution * load_kv_specialized<SLLM_HIP_KV_ENCODING_FP16_V1>(
                             page.value, nullptr, nullptr, kv_row, dimension,
                             head_dim, 1.0F);
    }
    const uint32_t second = dimension + blockDim.x;
    if (second < head_dim) {
      accumulation1 =
          accumulation1 * rescale +
          contribution *
              load_kv_specialized<SLLM_HIP_KV_ENCODING_FP16_V1>(
                  page.value, nullptr, nullptr, kv_row, second, head_dim, 1.0F);
    }
    __syncthreads();
    if (key_position == query_position)
      break;
  }
  if (dimension < head_dim)
    output_row[dimension] =
        f32_to_bf16_rne(accumulation0 / running_denominator);
  const uint32_t second = dimension + blockDim.x;
  if (second < head_dim)
    output_row[second] = f32_to_bf16_rne(accumulation1 / running_denominator);
}

// Static FP8 attention for the nine-slot absolute ring.  The ordinary
// paged kernel indexes a monotonically growing logical table; this variant
// resolves every key through the absolute block tag before touching the
// descriptor.  `retained_start` is supplied by the host ring state and is
// applied together with the fixed 1024-token causal window.
__global__
__launch_bounds__(256, 1) void causal_attention_paged_sliding_static_fp8_ring_kernel(
    const uint16_t *const query, const uint32_t *const ring_table,
    const uint64_t *const ring_tags,
    const sllm_paged_kv::BlockDescriptor *const descriptor_table,
    const uint32_t ring_slot_count, const uint32_t descriptor_count,
    uint32_t *const device_status, uint16_t *const output,
    const uint32_t query_count, const uint64_t start_position,
    const uint64_t committed_kv_length, const uint64_t retained_start,
    const uint32_t q_heads, const uint32_t kv_heads, const uint32_t head_dim,
    const float static_key_scale, const float static_value_scale,
    const float score_scale) {
  const uint64_t flat = static_cast<uint64_t>(blockIdx.x);
  if (flat >= static_cast<uint64_t>(query_count) * q_heads) {
    return;
  }
  const uint64_t row = flat / q_heads;
  const uint32_t query_head = static_cast<uint32_t>(flat % q_heads);
  const uint64_t query_position = start_position + row;
  if (query_position >= committed_kv_length) {
    return;
  }
  const uint16_t *const query_row =
      query + row * q_heads * head_dim +
      static_cast<uint64_t>(query_head) * head_dim;
  const uint32_t kv_head = query_head / (q_heads / kv_heads);
  uint16_t *const output_row = output + row * q_heads * head_dim +
                               static_cast<uint64_t>(query_head) * head_dim;

  const uint32_t dimension = threadIdx.x;
  __shared__ float reductions[SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE];
  __shared__ float rescale;
  __shared__ float contribution;
  __shared__ float running_maximum;
  __shared__ float running_denominator;
  __shared__ sllm_paged_kv::BlockDescriptor page;
  __shared__ uint32_t page_valid;
  if (dimension == 0U) {
    running_maximum = -std::numeric_limits<float>::infinity();
    running_denominator = 0.0F;
  }
  __syncthreads();

  float accumulation0 = 0.0F;
  float accumulation1 = 0.0F;
  const uint64_t lower_by_window =
      query_position + 1U > kPagedSlidingWindowTokens
          ? query_position + 1U - kPagedSlidingWindowTokens
          : 0U;
  const uint64_t key_begin =
      retained_start > lower_by_window ? retained_start : lower_by_window;
  if (key_begin > query_position || ring_slot_count != kPagedSlidingRingSlots) {
    if (dimension == 0U) {
      atomicExch(device_status, 1U);
    }
    return;
  }
  for (uint64_t key_position = key_begin;; ++key_position) {
    const uint64_t absolute_block = key_position / 128U;
    const uint32_t ring_slot = static_cast<uint32_t>(
        absolute_block % static_cast<uint64_t>(ring_slot_count));
    if (dimension == 0U) {
      page_valid = ring_tags[ring_slot] == absolute_block ? 1U : 0U;
      if (page_valid != 0U) {
        const uint32_t physical_block = ring_table[ring_slot];
        page_valid = physical_block < descriptor_count ? 1U : 0U;
        if (page_valid != 0U) {
          page = descriptor_table[physical_block];
          page_valid = page.key != nullptr && page.value != nullptr ? 1U : 0U;
        }
      }
      if (page_valid == 0U) {
        atomicExch(device_status, 1U);
      }
    }
    __syncthreads();
    if (page_valid == 0U) {
      return;
    }
    const uint32_t local_token = static_cast<uint32_t>(key_position % 128U);
    const uint64_t kv_row = static_cast<uint64_t>(local_token) * kv_heads +
                            static_cast<uint64_t>(kv_head);
    reductions[dimension] = 0.0F;
    for (uint32_t current = dimension; current < head_dim;
         current += blockDim.x) {
      reductions[dimension] +=
          bf16_to_f32(query_row[current]) *
          load_kv_specialized<SLLM_HIP_KV_ENCODING_FP8_STATIC_V1>(
              page.key, nullptr, nullptr, kv_row, current, head_dim,
              static_key_scale);
    }
    __syncthreads();
    for (uint32_t stride = SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE / 2U;
         stride != 0U; stride >>= 1U) {
      if (dimension < stride) {
        reductions[dimension] += reductions[dimension + stride];
      }
      __syncthreads();
    }
    if (dimension == 0U) {
      const float current_score = reductions[0] * score_scale;
      const float next_maximum = fmaxf(running_maximum, current_score);
      rescale = expf(running_maximum - next_maximum);
      contribution = expf(current_score - next_maximum);
      running_denominator = running_denominator * rescale + contribution;
      running_maximum = next_maximum;
    }
    __syncthreads();
    if (dimension < head_dim) {
      accumulation0 =
          accumulation0 * rescale +
          contribution *
              load_kv_specialized<SLLM_HIP_KV_ENCODING_FP8_STATIC_V1>(
                  page.value, nullptr, nullptr, kv_row, dimension, head_dim,
                  static_value_scale);
    }
    const uint32_t second = dimension + blockDim.x;
    if (second < head_dim) {
      accumulation1 =
          accumulation1 * rescale +
          contribution *
              load_kv_specialized<SLLM_HIP_KV_ENCODING_FP8_STATIC_V1>(
                  page.value, nullptr, nullptr, kv_row, second, head_dim,
                  static_value_scale);
    }
    __syncthreads();
    if (key_position == query_position) {
      break;
    }
  }
  if (dimension < head_dim) {
    output_row[dimension] =
        f32_to_bf16_rne(accumulation0 / running_denominator);
  }
  const uint32_t second = dimension + blockDim.x;
  if (second < head_dim) {
    output_row[second] = f32_to_bf16_rne(accumulation1 / running_denominator);
  }
}

// Phase 87 Stage10 production paged decode.  This is the WU-P1 GQA-shared
// stage-1 arithmetic with only the physical-row lookup changed to a fixed
// descriptor table.  The existing split merge kernel is deliberately reused
// so the workspace layout and reduction order stay identical to the contiguous
// control path.
template <uint32_t kSplits, uint32_t kWorkspaceSplits = kSplits,
          bool DeviceControl = false>
__global__
__launch_bounds__(192, 1) void causal_attention_paged_decode_gqa6_stage1_kernel(
    const uint16_t *const query, const uint32_t *const logical_table,
    const sllm_paged_kv::BlockDescriptor *const descriptor_table,
    const uint32_t logical_table_count, const uint32_t descriptor_count,
    uint32_t *const device_status, float *const workspace,
    const uint32_t query_count, const uint64_t start_position,
    sllm_decode_control::ControlV1 *const control) {
  constexpr uint32_t kWaveSize = 32U;
  constexpr uint32_t kWaveCount = 6U;
  constexpr uint32_t kKvHeads = 4U;
  constexpr uint32_t kQHeads = 24U;
  constexpr uint32_t kGqaRatio = 6U;
  constexpr uint32_t kHeadDim = 256U;
  constexpr uint32_t kKeyTile = 8U;
  constexpr uint32_t kWorkspaceStride = kHeadDim + 2U;
  constexpr uint32_t kElementsPerLane = kHeadDim / kWaveSize;
  static_assert(kSplits == 32U || kSplits == 128U);
#if defined(__gfx1030__)
  constexpr bool kUseFusedE4Decode = true;
#else
  constexpr bool kUseFusedE4Decode = false;
#endif

  const uint32_t block = static_cast<uint32_t>(blockIdx.x);
  const uint32_t split = block % kSplits;
  const uint32_t kv_head = (block / kSplits) % kKvHeads;
  const uint32_t query_index = block / (kKvHeads * kSplits);
  const uint32_t wave = threadIdx.x / kWaveSize;
  const uint32_t lane = threadIdx.x & (kWaveSize - 1U);
  const uint32_t query_head = kv_head * kGqaRatio + wave;
  if (query_index >= query_count || query_head >= kQHeads ||
      wave >= kWaveCount) {
    return;
  }

  uint64_t effective_start_position = start_position;
  if constexpr (DeviceControl) {
    uint32_t control_rows = query_count;
    if (!decode_control_phase(control, query_index, query_count,
                              &effective_start_position, &control_rows) ||
        decode_control_split_count(control, kSplits) != kSplits) {
      return;
    }
    if (effective_start_position >
        UINT64_MAX - static_cast<uint64_t>(query_index) - 1U) {
      mark_paged_attention_invalid(device_status, control);
      return;
    }
  }

  const uint64_t committed_kv_length =
      effective_start_position + static_cast<uint64_t>(query_index) + 1U;
  const uint64_t split_begin =
      committed_kv_length * static_cast<uint64_t>(split) / kSplits;
  const uint64_t split_end =
      committed_kv_length * static_cast<uint64_t>(split + 1U) / kSplits;
  const uint16_t *const query_row =
      query +
      (static_cast<uint64_t>(query_index) * kQHeads + query_head) * kHeadDim;
  const uint64_t workspace_base =
      ((static_cast<uint64_t>(query_index) * kQHeads + query_head) *
           kWorkspaceSplits +
       split) *
      kWorkspaceStride;
  float *const partial = workspace + workspace_base;

  float query_values[kElementsPerLane];
  float accumulations[kElementsPerLane] = {};
#pragma unroll
  for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
    query_values[index] = bf16_to_f32(query_row[lane + index * kWaveSize]);
  }
  if (split_begin >= split_end) {
    for (uint32_t index = lane; index < kHeadDim; index += kWaveSize) {
      partial[2U + index] = 0.0F;
    }
    if (lane == 0U) {
      partial[0] = -std::numeric_limits<float>::infinity();
      partial[1] = 0.0F;
    }
    return;
  }

  float local_maximum = -std::numeric_limits<float>::infinity();
  float local_denominator = 0.0F;
  __shared__ float key_tile[kKeyTile][kHeadDim];
  __shared__ float value_tile[kKeyTile][kHeadDim];
  __shared__ sllm_paged_kv::BlockDescriptor page;
  __shared__ uint32_t page_valid;
  const uint64_t first_page = split_begin / 128U;
  const uint64_t page_count = (split_end - 1U) / 128U + 1U;
  for (uint64_t logical_page = first_page; logical_page < page_count;
       ++logical_page) {
    if (threadIdx.x == 0U) {
      const uint32_t physical_page = logical_page < logical_table_count
                                         ? logical_table[logical_page]
                                         : UINT32_MAX;
      page_valid =
          physical_page != UINT32_MAX && physical_page < descriptor_count ? 1U
                                                                          : 0U;
      if (page_valid != 0U) {
        page = descriptor_table[physical_page];
        page_valid = page.key != nullptr && page.value != nullptr &&
                             page.key_scale != nullptr &&
                             page.value_scale != nullptr
                         ? 1U
                         : 0U;
      }
      if (page_valid == 0U) {
        if constexpr (DeviceControl) {
          mark_paged_attention_invalid(device_status, control);
        } else {
          atomicExch(device_status, 1U);
        }
      }
    }
    __syncthreads();
    if (page_valid == 0U) {
      return;
    }
    const uint64_t page_begin = logical_page * 128U;
    const uint64_t segment_begin =
        split_begin > page_begin ? split_begin : page_begin;
    const uint64_t page_end =
        page_begin > UINT64_MAX - 128U ? UINT64_MAX : page_begin + 128U;
    const uint64_t segment_end = split_end < page_end ? split_end : page_end;
    for (uint64_t tile_begin = segment_begin; tile_begin < segment_end;
         tile_begin += kKeyTile) {
      const uint64_t remaining = segment_end - tile_begin;
      const uint32_t tile_count =
          remaining < kKeyTile ? static_cast<uint32_t>(remaining) : kKeyTile;
      for (uint32_t element = threadIdx.x; element < kKeyTile * 2U * kHeadDim;
           element += blockDim.x) {
        const uint32_t key_index = element / (2U * kHeadDim);
        const uint32_t plane_dimension = element % (2U * kHeadDim);
        if (key_index < tile_count) {
          const uint64_t local_token =
              tile_begin + static_cast<uint64_t>(key_index) - page_begin;
          const uint64_t kv_row = local_token * kKvHeads + kv_head;
          if (plane_dimension < kHeadDim) {
            key_tile[key_index][plane_dimension] =
                load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1,
                               kUseFusedE4Decode>(
                    page.key, page.key_scale, nullptr, kv_row, plane_dimension,
                    kHeadDim, 1.0F);
          } else {
            const uint32_t dimension = plane_dimension - kHeadDim;
            value_tile[key_index][dimension] =
                load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1,
                               kUseFusedE4Decode>(page.value, page.value_scale,
                                                  nullptr, kv_row, dimension,
                                                  kHeadDim, 1.0F);
          }
        }
      }
      __syncthreads();
      for (uint32_t key_index = 0U; key_index < tile_count; ++key_index) {
        accumulate_mxfp8_e4_decode_key<kElementsPerLane>(
            query_values, key_tile, value_tile, key_index, lane, accumulations,
            &local_maximum, &local_denominator);
      }
      __syncthreads();
    }
  }

  if (lane == 0U) {
    partial[0] = local_maximum;
    partial[1] = local_denominator;
  }
#pragma unroll
  for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
    const uint32_t current = lane + index * kWaveSize;
    partial[2U + current] = accumulations[index];
  }
}

// Stage 11 C1 candidate: group two or three adjacent query rows under one
// block. The K/V tile is loaded once, then each row keeps its own QK,
// online-softmax, and value accumulation state. Workspace layout and the
// existing stage-2 merge are unchanged so the control remains bitwise.
template <uint32_t kRowsPerGroup, uint32_t kSplits,
          uint32_t kWorkspaceSplits = kSplits, bool DeviceControl = false>
__global__
__launch_bounds__(192, 1) void causal_attention_paged_decode_gqa6_c1_stage1_kernel(
    const uint16_t *const query, const uint32_t *const logical_table,
    const sllm_paged_kv::BlockDescriptor *const descriptor_table,
    const uint32_t logical_table_count, const uint32_t descriptor_count,
    uint32_t *const device_status, float *const workspace,
    const uint32_t query_count, const uint64_t start_position,
    sllm_decode_control::ControlV1 *const control) {
  static_assert(kRowsPerGroup == 2U || kRowsPerGroup == 3U);
  static_assert(kSplits == 32U || kSplits == 128U);
  static_assert(kWorkspaceSplits >= kSplits);
  constexpr uint32_t kWaveSize = 32U;
  constexpr uint32_t kWaveCount = 6U;
  constexpr uint32_t kKvHeads = 4U;
  constexpr uint32_t kQHeads = 24U;
  constexpr uint32_t kGqaRatio = 6U;
  constexpr uint32_t kHeadDim = 256U;
  constexpr uint32_t kKeyTile = 8U;
  constexpr uint32_t kElementsPerLane = kHeadDim / kWaveSize;
  constexpr uint32_t kWorkspaceStride = kHeadDim + 2U;
#if defined(__gfx1030__)
  constexpr bool kUseFusedE4Decode = true;
#else
  constexpr bool kUseFusedE4Decode = false;
#endif

  const uint32_t block = static_cast<uint32_t>(blockIdx.x);
  const uint32_t groups = (query_count + kRowsPerGroup - 1U) / kRowsPerGroup;
  const uint32_t group = block / (kKvHeads * kSplits);
  const uint32_t remainder = block % (kKvHeads * kSplits);
  const uint32_t split = remainder % kSplits;
  const uint32_t kv_head = remainder / kSplits;
  const uint32_t wave = threadIdx.x / kWaveSize;
  const uint32_t lane = threadIdx.x & (kWaveSize - 1U);
  const uint32_t query_head = kv_head * kGqaRatio + wave;
  if (group >= groups || kv_head >= kKvHeads || wave >= kWaveCount) {
    return;
  }

  uint64_t effective_start_position = start_position;
  uint32_t phase_rows = query_count;
  if constexpr (DeviceControl) {
    if (!decode_control_phase(control, group * kRowsPerGroup, query_count,
                              &effective_start_position, &phase_rows) ||
        decode_control_split_count(control, kSplits) != kSplits) {
      return;
    }
  }
  const uint32_t group_begin = group * kRowsPerGroup;
  if (group_begin >= query_count || group_begin >= phase_rows) {
    return;
  }
  const uint32_t active_rows = min(
      kRowsPerGroup, min(query_count - group_begin, phase_rows - group_begin));

  // ControlV1 can change the effective position after the host launcher has
  // validated its ordinary arguments.  Check the extra +1 used for the
  // committed length before any row arithmetic so a replay cannot wrap to a
  // small KV range.  All threads take the same fail-closed path.
  bool valid_position = true;
#pragma unroll
  for (uint32_t row = 0U; row < kRowsPerGroup; ++row) {
    const uint32_t query_index = group_begin + row;
    if (row < active_rows &&
        effective_start_position >
            UINT64_MAX - static_cast<uint64_t>(query_index) - 1U) {
      valid_position = false;
    }
  }
  if (!valid_position) {
    if (threadIdx.x == 0U) {
      mark_paged_attention_invalid(device_status,
                                   DeviceControl ? control : nullptr);
    }
    return;
  }

  uint64_t row_begin[kRowsPerGroup];
  uint64_t row_end[kRowsPerGroup];
  float query_values[kRowsPerGroup][kElementsPerLane];
  float accumulations[kRowsPerGroup][kElementsPerLane] = {};
  float running_maximum[kRowsPerGroup];
  float running_denominator[kRowsPerGroup];
#pragma unroll
  for (uint32_t row = 0U; row < kRowsPerGroup; ++row) {
    running_maximum[row] = -std::numeric_limits<float>::infinity();
    running_denominator[row] = 0.0F;
    const uint32_t query_index = group_begin + row;
    const uint32_t safe_query_index = row < active_rows ? query_index : 0U;
    const uint64_t query_position =
        effective_start_position + static_cast<uint64_t>(query_index);
    const uint64_t committed = query_position + 1U;
    row_begin[row] = committed * static_cast<uint64_t>(split) / kSplits;
    row_end[row] = committed * static_cast<uint64_t>(split + 1U) / kSplits;
    const uint16_t *const query_row =
        query +
        (static_cast<uint64_t>(safe_query_index) * kQHeads + query_head) *
            kHeadDim;
#pragma unroll
    for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
      const uint32_t current = lane + index * kWaveSize;
      query_values[row][index] =
          row < active_rows ? bf16_to_f32(query_row[current]) : 0.0F;
    }
  }

  const uint64_t union_begin = row_begin[0];
  const uint64_t union_end = row_end[active_rows - 1U];
  __shared__ float key_tile[kKeyTile][kHeadDim];
  __shared__ float value_tile[kKeyTile][kHeadDim];
  __shared__ sllm_paged_kv::BlockDescriptor page;
  __shared__ uint32_t page_valid;
  for (uint64_t page_begin = (union_begin / 128U) * 128U;
       page_begin < union_end; page_begin += 128U) {
    if (threadIdx.x == 0U) {
      const uint64_t logical_page = page_begin / 128U;
      const uint32_t physical = logical_page < logical_table_count
                                    ? logical_table[logical_page]
                                    : UINT32_MAX;
      page_valid = physical != UINT32_MAX && physical < descriptor_count;
      if (page_valid != 0U) {
        page = descriptor_table[physical];
        page_valid = page.key != nullptr && page.value != nullptr &&
                             page.key_scale != nullptr &&
                             page.value_scale != nullptr
                         ? 1U
                         : 0U;
      }
      if (page_valid == 0U) {
        mark_paged_attention_invalid(device_status,
                                     DeviceControl ? control : nullptr);
      }
    }
    __syncthreads();
    if (page_valid == 0U) {
      return;
    }
    const uint64_t page_end =
        min(union_end, page_begin + static_cast<uint64_t>(128U));
    for (uint64_t tile_begin = max(union_begin, page_begin);
         tile_begin < page_end; tile_begin += kKeyTile) {
      const uint32_t tile_count = static_cast<uint32_t>(
          min(page_end - tile_begin, static_cast<uint64_t>(kKeyTile)));
      for (uint32_t element = threadIdx.x; element < kKeyTile * 2U * kHeadDim;
           element += blockDim.x) {
        const uint32_t key_index = element / (2U * kHeadDim);
        const uint32_t plane_dimension = element % (2U * kHeadDim);
        if (key_index < tile_count) {
          const uint64_t local_token = tile_begin + key_index - page_begin;
          const uint64_t kv_row = local_token * kKvHeads + kv_head;
          if (plane_dimension < kHeadDim) {
            key_tile[key_index][plane_dimension] =
                load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1,
                               kUseFusedE4Decode>(
                    page.key, page.key_scale, nullptr, kv_row, plane_dimension,
                    kHeadDim, 1.0F);
          } else {
            const uint32_t dimension = plane_dimension - kHeadDim;
            value_tile[key_index][dimension] =
                load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1,
                               kUseFusedE4Decode>(page.value, page.value_scale,
                                                  nullptr, kv_row, dimension,
                                                  kHeadDim, 1.0F);
          }
        }
      }
      __syncthreads();

#pragma unroll
      for (uint32_t row = 0U; row < kRowsPerGroup; ++row) {
        if (row >= active_rows) {
          continue;
        }
        for (uint32_t key_index = 0U; key_index < tile_count; ++key_index) {
          const uint64_t key_position = tile_begin + key_index;
          if (key_position < row_begin[row] || key_position >= row_end[row]) {
            continue;
          }
          accumulate_mxfp8_e4_decode_key<kElementsPerLane>(
              query_values[row], key_tile, value_tile, key_index, lane,
              accumulations[row], &running_maximum[row],
              &running_denominator[row]);
        }
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (uint32_t row = 0U; row < kRowsPerGroup; ++row) {
    if (row >= active_rows) {
      continue;
    }
    const uint32_t query_index = group_begin + row;
    const uint64_t base =
        (static_cast<uint64_t>(query_index) * kQHeads + query_head) *
            kWorkspaceSplits * kWorkspaceStride +
        static_cast<uint64_t>(split) * kWorkspaceStride;
    if (lane == 0U) {
      workspace[base] = running_maximum[row];
      workspace[base + 1U] = running_denominator[row];
    }
#pragma unroll
    for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
      const uint32_t current = lane + index * kWaveSize;
      workspace[base + 2U + current] = accumulations[row][index];
    }
  }
}

// gfx1201 uses one wave per (query head, split), matching the reviewed WU-P1
// wave-split provider. Keeping this as a separate kernel avoids changing the
// register/LDS contract of the gfx1030 GQA-shared path.
template <uint32_t kSplits, uint32_t kWorkspaceSplits = kSplits,
          bool DeviceControl = false>
__global__
__launch_bounds__(32, 1) void causal_attention_paged_decode_wave_split_stage1_kernel(
    const uint16_t *const query, const uint32_t *const logical_table,
    const sllm_paged_kv::BlockDescriptor *const descriptor_table,
    const uint32_t logical_table_count, const uint32_t descriptor_count,
    uint32_t *const device_status, float *const workspace,
    const uint32_t query_count, const uint64_t start_position,
    const uint32_t q_heads, const uint32_t kv_heads, const uint32_t head_dim,
    sllm_decode_control::ControlV1 *const control) {
  constexpr uint32_t kWaveSize = 32U;
  constexpr uint32_t kHeadDim = 256U;
  constexpr uint32_t kElementsPerLane = kHeadDim / kWaveSize;
  const uint64_t flat = static_cast<uint64_t>(blockIdx.x);
  const uint64_t blocks_per_query = static_cast<uint64_t>(q_heads) * kSplits;
  const uint32_t query_index = static_cast<uint32_t>(flat / blocks_per_query);
  const uint32_t query_head =
      static_cast<uint32_t>((flat % blocks_per_query) / kSplits);
  const uint32_t split = static_cast<uint32_t>(flat % kSplits);
  if (query_index >= query_count || query_head >= q_heads || q_heads != 24U ||
      kv_heads != 4U || head_dim != kHeadDim) {
    return;
  }
  uint64_t effective_start_position = start_position;
  if constexpr (DeviceControl) {
    uint32_t control_rows = query_count;
    if (!decode_control_phase(control, query_index, query_count,
                              &effective_start_position, &control_rows) ||
        decode_control_split_count(control, kSplits) != kSplits) {
      return;
    }
    if (effective_start_position >
        UINT64_MAX - static_cast<uint64_t>(query_index) - 1U) {
      mark_paged_attention_invalid(device_status, control);
      return;
    }
  }
  const uint32_t lane = threadIdx.x & (kWaveSize - 1U);
  const uint32_t kv_head = query_head / (q_heads / kv_heads);
  const uint16_t *const query_row =
      query +
      (static_cast<uint64_t>(query_index) * q_heads + query_head) * head_dim;
  const uint64_t committed_kv_length =
      effective_start_position + static_cast<uint64_t>(query_index) + 1U;
  const uint64_t split_begin =
      committed_kv_length * static_cast<uint64_t>(split) / kSplits;
  const uint64_t split_end =
      committed_kv_length * static_cast<uint64_t>(split + 1U) / kSplits;
  const uint64_t base =
      ((static_cast<uint64_t>(query_index) * q_heads + query_head) *
           kWorkspaceSplits +
       split) *
      (kHeadDim + 2U);

  float accumulations[kElementsPerLane] = {};
  float query_values[kElementsPerLane];
#pragma unroll
  for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
    const uint32_t current = lane + index * kWaveSize;
    query_values[index] = bf16_to_f32(query_row[current]);
  }
  if (split_begin >= split_end) {
    for (uint32_t index = lane; index < head_dim; index += kWaveSize) {
      workspace[base + 2U + index] = 0.0F;
    }
    if (lane == 0U) {
      workspace[base] = -std::numeric_limits<float>::infinity();
      workspace[base + 1U] = 0.0F;
    }
    return;
  }

  float local_maximum = -std::numeric_limits<float>::infinity();
  float local_denominator = 0.0F;
  __shared__ sllm_paged_kv::BlockDescriptor page;
  __shared__ uint32_t page_valid;
  const uint64_t first_page = split_begin / 128U;
  const uint64_t page_count = (split_end - 1U) / 128U + 1U;
  for (uint64_t logical_page = first_page; logical_page < page_count;
       ++logical_page) {
    if (lane == 0U) {
      const uint32_t physical_page = logical_page < logical_table_count
                                         ? logical_table[logical_page]
                                         : UINT32_MAX;
      page_valid =
          physical_page != UINT32_MAX && physical_page < descriptor_count ? 1U
                                                                          : 0U;
      if (page_valid != 0U) {
        page = descriptor_table[physical_page];
        page_valid = page.key != nullptr && page.value != nullptr &&
                             page.key_scale != nullptr &&
                             page.value_scale != nullptr
                         ? 1U
                         : 0U;
      }
      if (page_valid == 0U) {
        if constexpr (DeviceControl) {
          mark_paged_attention_invalid(device_status, control);
        } else {
          atomicExch(device_status, 1U);
        }
      }
    }
    __syncthreads();
    if (page_valid == 0U) {
      return;
    }
    const uint64_t page_begin = logical_page * 128U;
    const uint64_t segment_begin =
        split_begin > page_begin ? split_begin : page_begin;
    const uint64_t page_end =
        page_begin > UINT64_MAX - 128U ? UINT64_MAX : page_begin + 128U;
    const uint64_t segment_end = split_end < page_end ? split_end : page_end;
    for (uint64_t key_position = segment_begin; key_position < segment_end;
         ++key_position) {
      const uint64_t local_token = key_position - page_begin;
      const uint64_t kv_row = local_token * kv_heads + kv_head;
      float partial = 0.0F;
#pragma unroll
      for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
        const uint32_t current = lane + index * kWaveSize;
        partial += query_values[index] *
                   load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, false>(
                       page.key, page.key_scale, nullptr, kv_row, current,
                       head_dim, 1.0F);
      }
      for (uint32_t offset = kWaveSize / 2U; offset != 0U; offset >>= 1U) {
        partial += __shfl_down(partial, offset, kWaveSize);
      }
      float rescale = 0.0F;
      float contribution = 0.0F;
      if (lane == 0U) {
        const float current_score =
            partial * rsqrtf(static_cast<float>(head_dim));
        const float next_maximum = fmaxf(local_maximum, current_score);
        rescale = expf(local_maximum - next_maximum);
        contribution = expf(current_score - next_maximum);
        local_denominator = local_denominator * rescale + contribution;
        local_maximum = next_maximum;
      }
      rescale = __shfl(rescale, 0U, kWaveSize);
      contribution = __shfl(contribution, 0U, kWaveSize);
#pragma unroll
      for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
        const uint32_t current = lane + index * kWaveSize;
        accumulations[index] =
            accumulations[index] * rescale +
            contribution *
                load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, false>(
                    page.value, page.value_scale, nullptr, kv_row, current,
                    head_dim, 1.0F);
      }
    }
  }
  if (lane == 0U) {
    workspace[base] = local_maximum;
    workspace[base + 1U] = local_denominator;
  }
#pragma unroll
  for (uint32_t index = 0U; index < kElementsPerLane; ++index) {
    const uint32_t current = lane + index * kWaveSize;
    workspace[base + 2U + current] = accumulations[index];
  }
}

// The long-prefill counterpart keeps the qtile8/w16 query, reduction, and
// online-softmax order of the contiguous provider. A descriptor is loaded
// once for each logical 128-token page; the six plane pointers are then used
// for all rows in that page. This is deliberately an exact MXFP8-E4/GQA6
// provider, rather than a new generic attention selector.
// gfx1201's contiguous MXFP8 provider uses one 256-thread block per
// (query-row, query-head) and the wave-provider reduction in
// causal_attention_kernel<true>. Keep that arithmetic for Paged prefill and
// change only the K/V row lookup through the logical page table. This avoids
// changing BF16 rounding at a Qwen prompt boundary while gfx1030 retains the
// measured qtile8 path below.
__global__
__launch_bounds__(256, 1) void causal_attention_paged_prefill_gfx1201_wave_kernel(
    const uint16_t *const query, const uint32_t *const logical_table,
    const sllm_paged_kv::BlockDescriptor *const descriptor_table,
    const uint32_t logical_table_count, const uint32_t descriptor_count,
    uint32_t *const device_status, uint16_t *const output,
    const uint32_t query_count, const uint64_t start_position,
    const uint32_t q_heads, const uint32_t kv_heads, const uint32_t head_dim,
    const float score_scale) {
  const uint64_t flat = static_cast<uint64_t>(blockIdx.x);
  if (flat >= static_cast<uint64_t>(query_count) * q_heads || q_heads != 24U ||
      kv_heads != 4U || head_dim != 256U) {
    return;
  }
  const uint64_t row = flat / q_heads;
  const uint32_t query_head = static_cast<uint32_t>(flat % q_heads);
  const uint64_t query_position = start_position + row;
  const uint16_t *const query_row =
      query + row * static_cast<uint64_t>(q_heads) * head_dim +
      static_cast<uint64_t>(query_head) * head_dim;
  uint16_t *const output_row = output +
                               row * static_cast<uint64_t>(q_heads) * head_dim +
                               static_cast<uint64_t>(query_head) * head_dim;
  const uint32_t dimension = threadIdx.x;
  __shared__ float reductions[SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE];
  __shared__ float rescale;
  __shared__ float contribution;
  __shared__ float running_maximum;
  __shared__ float running_denominator;
  __shared__ sllm_paged_kv::BlockDescriptor page;
  __shared__ uint32_t page_valid;
  if (dimension == 0U) {
    running_maximum = -std::numeric_limits<float>::infinity();
    running_denominator = 0.0F;
  }
  __syncthreads();

  float accumulation0 = 0.0F;
  float accumulation1 = 0.0F;
  const uint32_t kv_head = query_head / (q_heads / kv_heads);
  uint64_t previous_page = UINT64_MAX;
  for (uint64_t key_position = 0U; key_position <= query_position;
       ++key_position) {
    const uint64_t logical_page = key_position / 128U;
    if (logical_page != previous_page) {
      previous_page = logical_page;
      if (dimension == 0U) {
        const uint32_t physical_page = logical_page < logical_table_count
                                           ? logical_table[logical_page]
                                           : UINT32_MAX;
        page_valid =
            physical_page != UINT32_MAX && physical_page < descriptor_count
                ? 1U
                : 0U;
        if (page_valid != 0U) {
          page = descriptor_table[physical_page];
          page_valid = page.key != nullptr && page.value != nullptr &&
                               page.key_scale != nullptr &&
                               page.value_scale != nullptr
                           ? 1U
                           : 0U;
        }
        if (page_valid == 0U)
          atomicExch(device_status, 1U);
      }
    }
    __syncthreads();
    if (page_valid == 0U)
      return;
    const uint64_t kv_row = (key_position % 128U) * kv_heads + kv_head;
#if defined(__gfx1201__)
    float partial = 0.0F;
    for (uint32_t current = dimension; current < head_dim;
         current += blockDim.x) {
      partial += bf16_to_f32(query_row[current]) *
                 load_kv_specialized<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1>(
                     page.key, page.key_scale, nullptr, kv_row, current,
                     head_dim, 1.0F);
    }
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U)
      partial += __shfl_down(partial, offset, 32U);
    const uint32_t lane = dimension & 31U;
    const uint32_t wave = dimension >> 5U;
    if (lane == 0U)
      reductions[wave] = partial;
    __syncthreads();
    if (wave == 0U) {
      float block_sum = lane < 8U ? reductions[lane] : 0.0F;
      for (uint32_t offset = 16U; offset != 0U; offset >>= 1U)
        block_sum += __shfl_down(block_sum, offset, 32U);
      if (lane == 0U) {
        const float current_score = block_sum * score_scale;
        const float next_maximum = fmaxf(running_maximum, current_score);
        rescale = expf(running_maximum - next_maximum);
        contribution = expf(current_score - next_maximum);
        running_denominator = running_denominator * rescale + contribution;
        running_maximum = next_maximum;
      }
    }
    __syncthreads();
#else
    reductions[dimension] = 0.0F;
    for (uint32_t current = dimension; current < head_dim;
         current += blockDim.x) {
      reductions[dimension] +=
          bf16_to_f32(query_row[current]) *
          load_kv_specialized<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1>(
              page.key, page.key_scale, nullptr, kv_row, current, head_dim,
              1.0F);
    }
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2U; stride != 0U; stride >>= 1U) {
      if (dimension < stride)
        reductions[dimension] += reductions[dimension + stride];
      __syncthreads();
    }
    if (dimension == 0U) {
      const float current_score = reductions[0] * score_scale;
      const float next_maximum = fmaxf(running_maximum, current_score);
      rescale = expf(running_maximum - next_maximum);
      contribution = expf(current_score - next_maximum);
      running_denominator = running_denominator * rescale + contribution;
      running_maximum = next_maximum;
    }
    __syncthreads();
#endif
    if (dimension < head_dim) {
      accumulation0 =
          accumulation0 * rescale +
          contribution * load_kv_specialized<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1>(
                             page.value, page.value_scale, nullptr, kv_row,
                             dimension, head_dim, 1.0F);
    }
    const uint32_t second = dimension + blockDim.x;
    if (second < head_dim) {
      accumulation1 =
          accumulation1 * rescale +
          contribution * load_kv_specialized<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1>(
                             page.value, page.value_scale, nullptr, kv_row,
                             second, head_dim, 1.0F);
    }
    __syncthreads();
  }
  if (dimension < head_dim)
    output_row[dimension] =
        f32_to_bf16_rne(accumulation0 / running_denominator);
  const uint32_t second = dimension + blockDim.x;
  if (second < head_dim)
    output_row[second] = f32_to_bf16_rne(accumulation1 / running_denominator);
}

// Paged counterpart of the contiguous qtile4 GQA6 provider.  It preserves
// the qtile4 reduction tree and changes only page resolution.
__global__
__launch_bounds__(256, 1) void causal_attention_paged_prefill_gqa6_qtile4_kernel(
    const uint16_t *query, const uint32_t *logical_table,
    const sllm_paged_kv::BlockDescriptor *descriptor_table,
    uint32_t logical_table_count, uint32_t descriptor_count,
    uint32_t *device_status, uint16_t *output, uint32_t query_count,
    uint64_t start_position, uint32_t q_heads, uint32_t kv_heads,
    uint32_t head_dim) {
  constexpr uint32_t kWave = 32U;
  constexpr uint32_t kRatio = 6U;
  constexpr uint32_t kTile = 4U;
  constexpr uint32_t kItems = 3U;
  const uint64_t flat = static_cast<uint64_t>(blockIdx.x);
  const uint64_t tile = flat / kv_heads;
  const uint32_t kv_head = static_cast<uint32_t>(flat % kv_heads);
  const uint64_t first_row = tile * kTile;
  if (first_row >= query_count || q_heads != 24U || kv_heads != 4U ||
      head_dim != 256U)
    return;
  const uint32_t thread = threadIdx.x;
  const uint32_t lane = thread & (kWave - 1U);
  const uint32_t wave = thread / kWave;
  const uint32_t first_head = kv_head * kRatio;
  __shared__ float key_tile[256U];
  __shared__ float value_tile[256U];
  __shared__ sllm_paged_kv::BlockDescriptor page;
  __shared__ uint32_t page_valid;
  float query_values[kItems][8U];
  float accumulations[kItems][8U] = {};
  float running_maximum[kItems];
  float running_denominator[kItems];
#pragma unroll
  for (uint32_t item = 0U; item < kItems; ++item) {
    running_maximum[item] = -std::numeric_limits<float>::infinity();
    running_denominator[item] = 0.0F;
    const uint32_t logical = wave * kItems + item;
    const uint64_t row = first_row + logical / kRatio;
    const uint64_t safe = row < query_count ? row : query_count - 1U;
    const uint32_t head = first_head + logical % kRatio;
    const uint16_t *row_ptr = query + (safe * q_heads + head) * head_dim;
#pragma unroll
    for (uint32_t index = 0U; index < 8U; ++index) {
      const uint32_t current = lane + index * kWave;
      query_values[item][index] =
          row < query_count ? bf16_to_f32(row_ptr[current]) : 0.0F;
    }
  }
  const uint64_t last_row =
      (first_row + kTile < query_count ? first_row + kTile : query_count) - 1U;
  const uint64_t last_position = start_position + last_row;
  uint64_t previous_page = UINT64_MAX;
  for (uint64_t key_position = 0U; key_position <= last_position;
       ++key_position) {
    const uint64_t logical_page = key_position / 128U;
    if (logical_page != previous_page) {
      previous_page = logical_page;
      if (thread == 0U) {
        const uint32_t physical = logical_page < logical_table_count
                                      ? logical_table[logical_page]
                                      : UINT32_MAX;
        page_valid = physical != UINT32_MAX && physical < descriptor_count;
        if (page_valid != 0U) {
          page = descriptor_table[physical];
          page_valid = page.key != nullptr && page.value != nullptr &&
                       page.key_scale != nullptr && page.value_scale != nullptr;
        }
        if (page_valid == 0U)
          atomicExch(device_status, 1U);
      }
    }
    __syncthreads();
    if (page_valid == 0U)
      return;
    const uint64_t row = (key_position % 128U) * kv_heads + kv_head;
    if (thread < head_dim) {
      key_tile[thread] = load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1>(
          page.key, page.key_scale, nullptr, row, thread, head_dim, 1.0F);
      value_tile[thread] = load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1>(
          page.value, page.value_scale, nullptr, row, thread, head_dim, 1.0F);
    }
    __syncthreads();
#pragma unroll
    for (uint32_t item = 0U; item < kItems; ++item) {
      const uint32_t logical = wave * kItems + item;
      const uint64_t query_row = first_row + logical / kRatio;
      const bool active =
          query_row < query_count && key_position <= start_position + query_row;
      float products[8U];
#pragma unroll
      for (uint32_t index = 0U; index < 8U; ++index) {
        const uint32_t current = lane + index * kWave;
        products[index] =
            active ? query_values[item][index] * key_tile[current] : 0.0F;
      }
      float partial =
          ((products[0] + products[1]) + (products[2] + products[3])) +
          ((products[4] + products[5]) + (products[6] + products[7]));
      for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U)
        partial += __shfl_down(partial, offset, kWave);
      float rescale = 1.0F;
      float contribution = 0.0F;
      float next = running_maximum[item];
      if (lane == 0U && active) {
        const float score = partial * rsqrtf(static_cast<float>(head_dim));
        next = fmaxf(running_maximum[item], score);
        rescale = expf(running_maximum[item] - next);
        contribution = expf(score - next);
      }
      rescale = __shfl(rescale, 0U, kWave);
      contribution = __shfl(contribution, 0U, kWave);
      next = __shfl(next, 0U, kWave);
      running_denominator[item] =
          running_denominator[item] * rescale + contribution;
      running_maximum[item] = next;
#pragma unroll
      for (uint32_t index = 0U; index < 8U; ++index) {
        const uint32_t current = lane + index * kWave;
        if (active)
          accumulations[item][index] = accumulations[item][index] * rescale +
                                       contribution * value_tile[current];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (uint32_t item = 0U; item < kItems; ++item) {
    const uint32_t logical = wave * kItems + item;
    const uint64_t row = first_row + logical / kRatio;
    const uint32_t head = first_head + logical % kRatio;
    if (row < query_count) {
      uint16_t *out = output + (row * q_heads + head) * head_dim;
#pragma unroll
      for (uint32_t index = 0U; index < 8U; ++index) {
        const uint32_t current = lane + index * kWave;
        out[current] = f32_to_bf16_rne(accumulations[item][index] /
                                       running_denominator[item]);
      }
    }
  }
}

template <bool WaveLocalKv = false>
__global__
__launch_bounds__(512, 1) void causal_attention_paged_prefill_gqa6_qtile8_kernel(
    const uint16_t *const query, const uint32_t *const logical_table,
    const sllm_paged_kv::BlockDescriptor *const descriptor_table,
    const uint32_t logical_table_count, const uint32_t descriptor_count,
    uint32_t *const device_status, uint16_t *const output,
    const uint32_t query_count, const uint64_t start_position,
    const uint32_t q_heads, const uint32_t kv_heads, const uint32_t head_dim,
    const float static_key_scale, const float static_value_scale) {
  constexpr uint32_t kWaveSize = 32U;
  constexpr uint32_t kWaveCount = 16U;
  constexpr uint32_t kGqaRatio = 6U;
  constexpr uint32_t kQueryTile = 8U;
  constexpr uint32_t kLogicalQueries = kGqaRatio * kQueryTile;
  constexpr uint32_t kQueriesPerWave = kLogicalQueries / kWaveCount;
  constexpr uint32_t kHeadDim = 256U;
  constexpr uint32_t kDimensionsPerLane = kHeadDim / kWaveSize;
  static_assert(kQueriesPerWave == 3U);
#if defined(__gfx1030__)
  constexpr uint32_t kKeyTile = 4U;
  __shared__ float key_tile[kKeyTile][kHeadDim];
  __shared__ float value_tile[kKeyTile][kHeadDim];
#endif
  __shared__ sllm_paged_kv::BlockDescriptor page;
  __shared__ uint32_t page_valid;

  const uint64_t flat = blockIdx.x;
  const uint64_t tile = flat / kv_heads;
  const uint32_t kv_head = static_cast<uint32_t>(flat % kv_heads);
  const uint64_t first_row = tile * kQueryTile;
  if (first_row >= query_count || q_heads != 24U || kv_heads != 4U ||
      head_dim != kHeadDim) {
    return;
  }
  const uint32_t dimension = threadIdx.x;
  const uint32_t lane = dimension & (kWaveSize - 1U);
  const uint32_t wave = dimension / kWaveSize;
  const uint32_t first_query_head = kv_head * kGqaRatio;
  const uint64_t wave_row =
      first_row + (static_cast<uint64_t>(wave) * kQueriesPerWave) / kGqaRatio;
  const bool wave_row_valid = wave_row < query_count;
  const uint64_t wave_causal_limit =
      wave_row_valid ? start_position + wave_row : 0U;

  float query_values[kQueriesPerWave][kDimensionsPerLane];
  float accumulations[kQueriesPerWave][kDimensionsPerLane] = {};
  float own_running_maximum = -std::numeric_limits<float>::infinity();
  float own_running_denominator = 0.0F;
#pragma unroll
  for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
    const uint32_t logical_query = wave * kQueriesPerWave + item;
    const uint64_t row = first_row + logical_query / kGqaRatio;
    const uint64_t safe_row =
        row < query_count ? row : static_cast<uint64_t>(query_count - 1U);
    const uint32_t query_head = first_query_head + logical_query % kGqaRatio;
    const uint16_t *const query_row =
        query +
        (safe_row * q_heads + query_head) * static_cast<uint64_t>(head_dim);
#pragma unroll
    for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
      const uint32_t current = lane + index * kWaveSize;
      query_values[item][index] = row < query_count && current < head_dim
                                      ? bf16_to_f32(query_row[current])
                                      : 0.0F;
    }
  }

  const uint64_t tile_end = first_row + kQueryTile;
  const uint64_t last_row =
      (tile_end < query_count ? tile_end : query_count) - 1U;
  const uint64_t last_query_position = start_position + last_row;
  for (uint64_t page_begin = 0U; page_begin <= last_query_position;
       page_begin += 128U) {
    if (threadIdx.x == 0U) {
      const uint64_t logical_page = page_begin / 128U;
      const uint32_t physical_page = logical_page < logical_table_count
                                         ? logical_table[logical_page]
                                         : UINT32_MAX;
      page_valid =
          physical_page != UINT32_MAX && physical_page < descriptor_count ? 1U
                                                                          : 0U;
      if (page_valid != 0U) {
        page = descriptor_table[physical_page];
        page_valid = page.key != nullptr && page.value != nullptr &&
                             page.key_scale != nullptr &&
                             page.value_scale != nullptr
                         ? 1U
                         : 0U;
      }
      if (page_valid == 0U) {
        atomicExch(device_status, 1U);
      }
    }
    __syncthreads();
    if (page_valid == 0U) {
      return;
    }
    const uint64_t remaining = last_query_position - page_begin + 1U;
    const uint64_t page_end =
        remaining > 128U ? page_begin + 128U : last_query_position + 1U;
#if defined(__gfx1030__)
    for (uint64_t key_begin = page_begin; key_begin < page_end;
         key_begin += kKeyTile) {
      const uint64_t tile_remaining = page_end - key_begin;
      const uint32_t key_count = tile_remaining < kKeyTile
                                     ? static_cast<uint32_t>(tile_remaining)
                                     : kKeyTile;
      for (uint32_t element = dimension; element < kKeyTile * 2U * kHeadDim;
           element += 512U) {
        const uint32_t key_index = element / (2U * kHeadDim);
        const uint32_t plane_dimension = element % (2U * kHeadDim);
        if (key_index < key_count) {
          const uint64_t local_token =
              key_begin + static_cast<uint64_t>(key_index) - page_begin;
          const uint64_t kv_row = local_token * kv_heads + kv_head;
          if (plane_dimension < head_dim) {
            key_tile[key_index][plane_dimension] =
                load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, false>(
                    page.key, page.key_scale, nullptr, kv_row, plane_dimension,
                    head_dim, static_key_scale);
          } else {
            const uint32_t value_dimension = plane_dimension - head_dim;
            value_tile[key_index][value_dimension] =
                load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, false>(
                    page.value, page.value_scale, nullptr, kv_row,
                    value_dimension, head_dim, static_value_scale);
          }
        }
      }
      __syncthreads();
      for (uint32_t key_index = 0U; key_index < key_count; ++key_index) {
        const uint64_t key_position = key_begin + key_index;
        float own_score = 0.0F;
        uint32_t own_active = 0U;
#pragma unroll
        for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
          const bool item_active =
              wave_row_valid && key_position <= wave_causal_limit;
          float products[kDimensionsPerLane];
#pragma unroll
          for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
            const uint32_t current = lane + index * kWaveSize;
            products[index] =
                item_active && current < head_dim
                    ? query_values[item][index] * key_tile[key_index][current]
                    : 0.0F;
          }
          const float pair0 = products[0] + products[1];
          const float pair1 = products[2] + products[3];
          const float pair2 = products[4] + products[5];
          const float pair3 = products[6] + products[7];
          float partial = (pair0 + pair1) + (pair2 + pair3);
          for (uint32_t offset = kWaveSize / 2U; offset != 0U; offset >>= 1U) {
            partial += __shfl_down(partial, offset, kWaveSize);
          }
          const float reduced_score = __shfl(partial, 0U, kWaveSize);
          if (lane == item) {
            own_score = reduced_score;
            own_active = item_active ? 1U : 0U;
          }
        }
        if (lane < kQueriesPerWave && own_active != 0U) {
          own_score *= rsqrtf(static_cast<float>(head_dim));
        }
        float own_rescale = 1.0F;
        float own_contribution = 0.0F;
        if (lane < kQueriesPerWave && own_active != 0U) {
          const float own_next_maximum = fmaxf(own_running_maximum, own_score);
          own_rescale = expf(own_running_maximum - own_next_maximum);
          own_contribution = expf(own_score - own_next_maximum);
          own_running_denominator =
              own_running_denominator * own_rescale + own_contribution;
          own_running_maximum = own_next_maximum;
        }
#pragma unroll
        for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
          const float rescale =
              __shfl(own_rescale, static_cast<int>(item), kWaveSize);
          const float contribution =
              __shfl(own_contribution, static_cast<int>(item), kWaveSize);
          const bool item_active =
              wave_row_valid && key_position <= wave_causal_limit;
#pragma unroll
          for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
            const uint32_t current = lane + index * kWaveSize;
            if (item_active && current < head_dim) {
              accumulations[item][index] =
                  accumulations[item][index] * rescale +
                  contribution * value_tile[key_index][current];
            }
          }
        }
      }
      __syncthreads();
    }
#else
    if constexpr (WaveLocalKv) {
      for (uint64_t key_position = page_begin; key_position < page_end;
           ++key_position) {
        const uint64_t local_token = key_position - page_begin;
        const uint64_t kv_row = local_token * kv_heads + kv_head;
        float key_register[kDimensionsPerLane];
#pragma unroll
        for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
          const uint32_t current = lane + index * kWaveSize;
          key_register[index] =
              load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, false>(
                  page.key, page.key_scale, nullptr, kv_row, current, head_dim,
                  static_key_scale);
        }
        float own_score = 0.0F;
        uint32_t own_active = 0U;
#pragma unroll
        for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
          const bool item_active =
              wave_row_valid && key_position <= wave_causal_limit;
          float products[kDimensionsPerLane];
#pragma unroll
          for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
            const uint32_t current = lane + index * kWaveSize;
            products[index] =
                item_active && current < head_dim
                    ? query_values[item][index] * key_register[index]
                    : 0.0F;
          }
          const float pair0 = products[0] + products[1];
          const float pair1 = products[2] + products[3];
          const float pair2 = products[4] + products[5];
          const float pair3 = products[6] + products[7];
          float partial = (pair0 + pair1) + (pair2 + pair3);
          for (uint32_t offset = kWaveSize / 2U; offset != 0U; offset >>= 1U) {
            partial += __shfl_down(partial, offset, kWaveSize);
          }
          const float reduced_score = __shfl(partial, 0U, kWaveSize);
          if (lane == item) {
            own_score = reduced_score;
            own_active = item_active ? 1U : 0U;
          }
        }
        if (lane < kQueriesPerWave && own_active != 0U) {
          own_score *= rsqrtf(static_cast<float>(head_dim));
        }
        float own_rescale = 1.0F;
        float own_contribution = 0.0F;
        if (lane < kQueriesPerWave && own_active != 0U) {
          const float own_next_maximum = fmaxf(own_running_maximum, own_score);
          own_rescale = expf(own_running_maximum - own_next_maximum);
          own_contribution = expf(own_score - own_next_maximum);
          own_running_denominator =
              own_running_denominator * own_rescale + own_contribution;
          own_running_maximum = own_next_maximum;
        }
        float value_register[kDimensionsPerLane];
#pragma unroll
        for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
          const uint32_t current = lane + index * kWaveSize;
          value_register[index] =
              load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, false>(
                  page.value, page.value_scale, nullptr, kv_row, current,
                  head_dim, static_value_scale);
        }
#pragma unroll
        for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
          const float rescale =
              __shfl(own_rescale, static_cast<int>(item), kWaveSize);
          const float contribution =
              __shfl(own_contribution, static_cast<int>(item), kWaveSize);
          const bool item_active =
              wave_row_valid && key_position <= wave_causal_limit;
#pragma unroll
          for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
            const uint32_t current = lane + index * kWaveSize;
            if (item_active && current < head_dim) {
              accumulations[item][index] =
                  accumulations[item][index] * rescale +
                  contribution * value_register[index];
            }
          }
        }
      }
    } else {
      __shared__ float key_tile[kHeadDim];
      __shared__ float value_tile[kHeadDim];
      for (uint64_t key_position = page_begin; key_position < page_end;
           ++key_position) {
        const uint64_t local_token = key_position - page_begin;
        const uint64_t kv_row = local_token * kv_heads + kv_head;
        if (dimension < head_dim) {
          key_tile[dimension] =
              load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, false>(
                  page.key, page.key_scale, nullptr, kv_row, dimension,
                  head_dim, static_key_scale);
        } else if (dimension < 2U * head_dim) {
          const uint32_t value_dimension = dimension - head_dim;
          value_tile[value_dimension] =
              load_kv_qtile4<SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, false>(
                  page.value, page.value_scale, nullptr, kv_row,
                  value_dimension, head_dim, static_value_scale);
        }
        __syncthreads();
        float own_score = 0.0F;
        uint32_t own_active = 0U;
#pragma unroll
        for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
          const bool item_active =
              wave_row_valid && key_position <= wave_causal_limit;
          float products[kDimensionsPerLane];
#pragma unroll
          for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
            const uint32_t current = lane + index * kWaveSize;
            products[index] =
                item_active && current < head_dim
                    ? query_values[item][index] * key_tile[current]
                    : 0.0F;
          }
          const float pair0 = products[0] + products[1];
          const float pair1 = products[2] + products[3];
          const float pair2 = products[4] + products[5];
          const float pair3 = products[6] + products[7];
          float partial = (pair0 + pair1) + (pair2 + pair3);
          for (uint32_t offset = kWaveSize / 2U; offset != 0U; offset >>= 1U) {
            partial += __shfl_down(partial, offset, kWaveSize);
          }
          const float reduced_score = __shfl(partial, 0U, kWaveSize);
          if (lane == item) {
            own_score = reduced_score;
            own_active = item_active ? 1U : 0U;
          }
        }
        if (lane < kQueriesPerWave && own_active != 0U) {
          own_score *= rsqrtf(static_cast<float>(head_dim));
        }
        float own_rescale = 1.0F;
        float own_contribution = 0.0F;
        if (lane < kQueriesPerWave && own_active != 0U) {
          const float own_next_maximum = fmaxf(own_running_maximum, own_score);
          own_rescale = expf(own_running_maximum - own_next_maximum);
          own_contribution = expf(own_score - own_next_maximum);
          own_running_denominator =
              own_running_denominator * own_rescale + own_contribution;
          own_running_maximum = own_next_maximum;
        }
#pragma unroll
        for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
          const float rescale =
              __shfl(own_rescale, static_cast<int>(item), kWaveSize);
          const float contribution =
              __shfl(own_contribution, static_cast<int>(item), kWaveSize);
          const bool item_active =
              wave_row_valid && key_position <= wave_causal_limit;
#pragma unroll
          for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
            const uint32_t current = lane + index * kWaveSize;
            if (item_active && current < head_dim) {
              accumulations[item][index] =
                  accumulations[item][index] * rescale +
                  contribution * value_tile[current];
            }
          }
        }
        __syncthreads();
      }
    }
#endif
  }

#pragma unroll
  for (uint32_t item = 0U; item < kQueriesPerWave; ++item) {
    const uint32_t logical_query = wave * kQueriesPerWave + item;
    const uint64_t row = first_row + logical_query / kGqaRatio;
    const uint32_t query_head = first_query_head + logical_query % kGqaRatio;
    if (row < query_count) {
      uint16_t *const output_row = output + (row * q_heads + query_head) *
                                                static_cast<uint64_t>(head_dim);
      const float denominator =
          __shfl(own_running_denominator, static_cast<int>(item), kWaveSize);
#pragma unroll
      for (uint32_t index = 0U; index < kDimensionsPerLane; ++index) {
        const uint32_t current = lane + index * kWaveSize;
        if (current < head_dim) {
          output_row[current] =
              f32_to_bf16_rne(accumulations[item][index] / denominator);
        }
      }
    }
  }
}

} // namespace

namespace {

template <uint32_t Encoding>
hipError_t launch_paged_attention_impl(
    const uint16_t *const query, const uint32_t *const logical_table,
    const sllm_paged_kv::BlockDescriptor *const descriptor_table,
    const uint32_t logical_table_count, const uint32_t descriptor_count,
    uint32_t *const device_status, uint16_t *const output,
    const uint32_t query_count, const uint64_t start_position,
    const uint64_t committed_kv_length, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim,
    const float static_key_scale, const float static_value_scale,
    const float score_scale, const hipStream_t stream) noexcept {
  if (query == nullptr || logical_table == nullptr ||
      descriptor_table == nullptr || device_status == nullptr ||
      output == nullptr || query_count == 0U || q_heads == 0U ||
      kv_heads == 0U || q_heads % kv_heads != 0U || head_dim == 0U ||
      head_dim > SLLM_HIP_KV_MAX_HEAD_DIM ||
      start_position > UINT64_MAX - static_cast<uint64_t>(query_count) ||
      start_position + query_count != committed_kv_length ||
      committed_kv_length == 0U || logical_table_count == 0U ||
      descriptor_count == 0U ||
      logical_table_count < committed_kv_length / 128U +
                                (committed_kv_length % 128U != 0U ? 1U : 0U) ||
      !std::isfinite(score_scale) || score_scale <= 0.0F ||
      (Encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1 &&
       (!std::isfinite(static_key_scale) || static_key_scale <= 0.0F ||
        !std::isfinite(static_value_scale) || static_value_scale <= 0.0F))) {
    return hipErrorInvalidValue;
  }
  const uint64_t block_count =
      static_cast<uint64_t>(query_count) * static_cast<uint64_t>(q_heads);
  if (block_count > std::numeric_limits<uint32_t>::max()) {
    return hipErrorInvalidValue;
  }
  hipError_t status =
      hipMemsetAsync(device_status, 0, sizeof(uint32_t), stream);
  if (status != hipSuccess) {
    return status;
  }
  if (Encoding == SLLM_HIP_KV_ENCODING_FP16_V1 &&
      (query_count == 1U || query_count >= 32U)) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(causal_attention_paged_kernel<Encoding, true>),
        dim3(static_cast<uint32_t>(block_count)),
        dim3(SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE), 0U, stream, query,
        logical_table, descriptor_table, logical_table_count, descriptor_count,
        device_status, output, query_count, start_position, committed_kv_length,
        q_heads, kv_heads, head_dim, static_key_scale, static_value_scale,
        score_scale);
  } else {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(causal_attention_paged_kernel<Encoding, false>),
        dim3(static_cast<uint32_t>(block_count)),
        dim3(SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE), 0U, stream, query,
        logical_table, descriptor_table, logical_table_count, descriptor_count,
        device_status, output, query_count, start_position, committed_kv_length,
        q_heads, kv_heads, head_dim, static_key_scale, static_value_scale,
        score_scale);
  }
  return hipGetLastError();
}

} // namespace

hipError_t launch_paged_attention(
    const uint16_t *const query, const uint32_t *const logical_table,
    const sllm_paged_kv::BlockDescriptor *const descriptor_table,
    const uint32_t logical_table_count, const uint32_t descriptor_count,
    uint32_t *const device_status, uint16_t *const output,
    const uint32_t query_count, const uint64_t start_position,
    const uint64_t committed_kv_length, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim, const uint32_t encoding,
    const float static_key_scale, const float static_value_scale,
    const float score_scale, const hipStream_t stream) noexcept {
#define SLLM_LAUNCH_PAGED_FORMAT(EncodingValue)                                \
  return launch_paged_attention_impl<EncodingValue>(                           \
      query, logical_table, descriptor_table, logical_table_count,             \
      descriptor_count, device_status, output, query_count, start_position,    \
      committed_kv_length, q_heads, kv_heads, head_dim, static_key_scale,      \
      static_value_scale, score_scale, stream)
  switch (encoding) {
  case SLLM_HIP_KV_ENCODING_FP16_V1:
    SLLM_LAUNCH_PAGED_FORMAT(SLLM_HIP_KV_ENCODING_FP16_V1);
  case SLLM_HIP_KV_ENCODING_FP8_V1:
    SLLM_LAUNCH_PAGED_FORMAT(SLLM_HIP_KV_ENCODING_FP8_V1);
  case SLLM_HIP_KV_ENCODING_FP8_STATIC_V1:
    SLLM_LAUNCH_PAGED_FORMAT(SLLM_HIP_KV_ENCODING_FP8_STATIC_V1);
  case SLLM_HIP_KV_ENCODING_NVFP4_V1:
    SLLM_LAUNCH_PAGED_FORMAT(SLLM_HIP_KV_ENCODING_NVFP4_V1);
  case SLLM_HIP_KV_ENCODING_MXFP8_E4_V1:
    SLLM_LAUNCH_PAGED_FORMAT(SLLM_HIP_KV_ENCODING_MXFP8_E4_V1);
  case SLLM_HIP_KV_ENCODING_MXFP8_E5_V1:
    SLLM_LAUNCH_PAGED_FORMAT(SLLM_HIP_KV_ENCODING_MXFP8_E5_V1);
  default:
    return hipErrorInvalidValue;
  }
#undef SLLM_LAUNCH_PAGED_FORMAT
}

hipError_t launch_paged_sliding_static_fp8_attention(
    const uint16_t *const query, const uint32_t *const ring_table,
    const uint64_t *const ring_tags,
    const sllm_paged_kv::BlockDescriptor *const descriptor_table,
    const uint32_t ring_slot_count, const uint32_t descriptor_count,
    uint32_t *const device_status, uint16_t *const output,
    const uint32_t query_count, const uint64_t start_position,
    const uint64_t committed_kv_length, const uint64_t retained_start,
    const uint32_t q_heads, const uint32_t kv_heads, const uint32_t head_dim,
    const float static_key_scale, const float static_value_scale,
    const float score_scale, const hipStream_t stream) noexcept {
  if (query == nullptr || ring_table == nullptr || ring_tags == nullptr ||
      descriptor_table == nullptr || device_status == nullptr ||
      output == nullptr || ring_slot_count != kPagedSlidingRingSlots ||
      descriptor_count == 0U || query_count == 0U || q_heads == 0U ||
      kv_heads == 0U || q_heads % kv_heads != 0U || head_dim == 0U ||
      head_dim > SLLM_HIP_KV_MAX_HEAD_DIM ||
      start_position > UINT64_MAX - static_cast<uint64_t>(query_count) ||
      start_position + query_count != committed_kv_length ||
      retained_start > committed_kv_length || committed_kv_length == 0U ||
      !std::isfinite(static_key_scale) || static_key_scale <= 0.0F ||
      !std::isfinite(static_value_scale) || static_value_scale <= 0.0F ||
      !std::isfinite(score_scale) || score_scale <= 0.0F) {
    return hipErrorInvalidValue;
  }
  const uint64_t block_count =
      static_cast<uint64_t>(query_count) * static_cast<uint64_t>(q_heads);
  if (block_count == 0U || block_count > UINT32_MAX) {
    return hipErrorInvalidValue;
  }
  hipError_t status =
      hipMemsetAsync(device_status, 0, sizeof(uint32_t), stream);
  if (status != hipSuccess) {
    return status;
  }
  hipLaunchKernelGGL(causal_attention_paged_sliding_static_fp8_ring_kernel,
                     dim3(static_cast<uint32_t>(block_count)),
                     dim3(SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE), 0U, stream,
                     query, ring_table, ring_tags, descriptor_table,
                     ring_slot_count, descriptor_count, device_status, output,
                     query_count, start_position, committed_kv_length,
                     retained_start, q_heads, kv_heads, head_dim,
                     static_key_scale, static_value_scale, score_scale);
  return hipGetLastError();
}

hipError_t launch_paged_decode_fp16(
    const uint16_t *const query, const uint32_t *const logical_table,
    const sllm_paged_kv::BlockDescriptor *const descriptor_table,
    const uint32_t logical_table_count, const uint32_t descriptor_count,
    uint32_t *const device_status, uint16_t *const output,
    const uint32_t query_count, const uint64_t start_position,
    const uint64_t committed_kv_length, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim,
    const hipStream_t stream) noexcept {
  if (query_count > 5U) {
    return hipErrorInvalidValue;
  }
  return launch_paged_attention(
      query, logical_table, descriptor_table, logical_table_count,
      descriptor_count, device_status, output, query_count, start_position,
      committed_kv_length, q_heads, kv_heads, head_dim,
      SLLM_HIP_KV_ENCODING_FP16_V1, 1.0F, 1.0F,
      1.0F / std::sqrt(static_cast<float>(head_dim)), stream);
}

hipError_t launch_paged_decode_fp16_device(
    const uint16_t *const query, const uint32_t *const logical_table,
    const sllm_paged_kv::BlockDescriptor *const descriptor_table,
    const uint32_t logical_table_count, const uint32_t descriptor_count,
    uint32_t *const device_status, uint16_t *const output,
    const uint32_t query_count, const uint32_t q_heads, const uint32_t kv_heads,
    const uint32_t head_dim, sllm_decode_control::ControlV1 *const control,
    const hipStream_t stream) noexcept {
  if (query == nullptr || logical_table == nullptr ||
      descriptor_table == nullptr || device_status == nullptr ||
      output == nullptr || control == nullptr || query_count == 0U ||
      query_count > 5U || q_heads == 0U || kv_heads == 0U ||
      q_heads % kv_heads != 0U || head_dim == 0U ||
      head_dim > SLLM_HIP_KV_MAX_HEAD_DIM || logical_table_count == 0U ||
      descriptor_count == 0U) {
    return hipErrorInvalidValue;
  }
  const uint64_t blocks =
      static_cast<uint64_t>(query_count) * static_cast<uint64_t>(q_heads);
  if (blocks > UINT32_MAX)
    return hipErrorInvalidValue;
  hipError_t status =
      hipMemsetAsync(device_status, 0, sizeof(uint32_t), stream);
  if (status != hipSuccess)
    return status;
  hipLaunchKernelGGL(causal_attention_paged_fp16_device_kernel,
                     dim3(static_cast<uint32_t>(blocks)),
                     dim3(SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE), 0U, stream,
                     query, logical_table, descriptor_table,
                     logical_table_count, descriptor_count, device_status,
                     output, query_count, q_heads, kv_heads, head_dim, control);
  return hipGetLastError();
}

hipError_t launch_paged_prefill_fp16(
    const uint16_t *const query, const uint32_t *const logical_table,
    const sllm_paged_kv::BlockDescriptor *const descriptor_table,
    const uint32_t logical_table_count, const uint32_t descriptor_count,
    uint32_t *const device_status, uint16_t *const output,
    const uint32_t query_count, const uint64_t start_position,
    const uint64_t committed_kv_length, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim,
    const hipStream_t stream) noexcept {
  return launch_paged_attention(
      query, logical_table, descriptor_table, logical_table_count,
      descriptor_count, device_status, output, query_count, start_position,
      committed_kv_length, q_heads, kv_heads, head_dim,
      SLLM_HIP_KV_ENCODING_FP16_V1, 1.0F, 1.0F,
      1.0F / std::sqrt(static_cast<float>(head_dim)), stream);
}

hipError_t launch_paged_prefill_gqa4(
    const uint16_t *const query, const uint32_t *const logical_table,
    const sllm_paged_kv::BlockDescriptor *const descriptor_table,
    const uint32_t logical_table_count, const uint32_t descriptor_count,
    uint32_t *const device_status, uint16_t *const output,
    const uint32_t query_count, const uint64_t start_position,
    const uint64_t committed_kv_length, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim,
    const hipStream_t stream) noexcept {
  if (query == nullptr || logical_table == nullptr ||
      descriptor_table == nullptr || device_status == nullptr ||
      output == nullptr || query_count < 64U ||
      query_count > SLLM_HIP_CAUSAL_ATTENTION_MAX_M || q_heads != 16U ||
      kv_heads != 4U || head_dim != 256U || logical_table_count == 0U ||
      descriptor_count == 0U || committed_kv_length == 0U ||
      start_position > UINT64_MAX - query_count ||
      start_position + query_count > committed_kv_length) {
    return hipErrorInvalidValue;
  }
  const uint64_t blocks = static_cast<uint64_t>(query_count) * kv_heads;
  if (blocks > UINT32_MAX) {
    return hipErrorInvalidValue;
  }
  hipError_t status =
      hipMemsetAsync(device_status, 0, sizeof(uint32_t), stream);
  if (status != hipSuccess) {
    return status;
  }
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(causal_attention_paged_prefill_gqa4_shared_kernel),
      dim3(static_cast<uint32_t>(blocks)),
      dim3(SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE), 0U, stream, query,
      logical_table, descriptor_table, logical_table_count, descriptor_count,
      device_status, output, query_count, start_position, committed_kv_length,
      q_heads, kv_heads, head_dim);
  return hipGetLastError();
}

hipError_t launch_paged_decode_gqa6(
    const uint16_t *const query, const uint32_t *const logical_table,
    const sllm_paged_kv::BlockDescriptor *const descriptor_table,
    const uint32_t logical_table_count, const uint32_t descriptor_count,
    uint32_t *const device_status, float *const workspace,
    const uint64_t workspace_bytes, uint16_t *const output,
    const uint32_t query_count, const uint64_t start_position,
    const uint64_t committed_kv_length, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim, const uint32_t encoding,
    const bool use_gqa_shared, const hipStream_t stream) noexcept {
  constexpr uint32_t kMaxQueryCount = 5U;
  constexpr uint32_t kQHeads = 24U;
  constexpr uint32_t kKvHeads = 4U;
  constexpr uint32_t kHeadDim = 256U;
  constexpr uint32_t kStride = kHeadDim + 2U;
  if (query == nullptr || logical_table == nullptr ||
      descriptor_table == nullptr || device_status == nullptr ||
      workspace == nullptr || output == nullptr || query_count == 0U ||
      query_count > kMaxQueryCount || logical_table_count == 0U ||
      descriptor_count == 0U || q_heads != kQHeads || kv_heads != kKvHeads ||
      head_dim != kHeadDim || encoding != SLLM_HIP_KV_ENCODING_MXFP8_E4_V1 ||
      start_position > UINT64_MAX - query_count ||
      start_position + query_count != committed_kv_length ||
      committed_kv_length == 0U ||
      logical_table_count < committed_kv_length / 128U +
                                (committed_kv_length % 128U != 0U ? 1U : 0U)) {
    return hipErrorInvalidValue;
  }
  const uint32_t splits = committed_kv_length >= 8192U ? 128U : 32U;
  const uint64_t required_workspace = static_cast<uint64_t>(query_count) *
                                      kQHeads * splits * kStride *
                                      sizeof(float);
  if (workspace_bytes < required_workspace) {
    return hipErrorInvalidValue;
  }
  hipError_t status =
      hipMemsetAsync(device_status, 0, sizeof(uint32_t), stream);
  if (status != hipSuccess) {
    return status;
  }
  if (use_gqa_shared) {
    if (splits == 128U) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
              causal_attention_paged_decode_gqa6_stage1_kernel<128U>),
          dim3(query_count * kKvHeads * 128U), dim3(192U), 0U, stream, query,
          logical_table, descriptor_table, logical_table_count,
          descriptor_count, device_status, workspace, query_count,
          start_position, nullptr);
    } else {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
              causal_attention_paged_decode_gqa6_stage1_kernel<32U>),
          dim3(query_count * kKvHeads * 32U), dim3(192U), 0U, stream, query,
          logical_table, descriptor_table, logical_table_count,
          descriptor_count, device_status, workspace, query_count,
          start_position, nullptr);
    }
  } else if (splits == 128U) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            causal_attention_paged_decode_wave_split_stage1_kernel<128U>),
        dim3(query_count * kQHeads * 128U), dim3(32U), 0U, stream, query,
        logical_table, descriptor_table, logical_table_count, descriptor_count,
        device_status, workspace, query_count, start_position, q_heads,
        kv_heads, head_dim, nullptr);
  } else {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            causal_attention_paged_decode_wave_split_stage1_kernel<32U>),
        dim3(query_count * kQHeads * 32U), dim3(32U), 0U, stream, query,
        logical_table, descriptor_table, logical_table_count, descriptor_count,
        device_status, workspace, query_count, start_position, q_heads,
        kv_heads, head_dim, nullptr);
  }
  status = hipGetLastError();
  if (status != hipSuccess) {
    return status;
  }
  if (splits == 128U) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            causal_attention_decode_gqa6_staged32_split_merge_kernel<128U>),
        dim3(query_count * kQHeads), dim3(256U), 0U, stream, workspace, output,
        query_count, nullptr);
  } else {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            causal_attention_decode_wave_split_staged_stage2_kernel<32U>),
        dim3(query_count * kQHeads), dim3(256U), 0U, stream, workspace, output,
        query_count, kQHeads, kKvHeads, kHeadDim, nullptr);
  }
  return hipGetLastError();
}

hipError_t launch_paged_decode_gqa6_device(
    const uint16_t *const query, const uint32_t *const logical_table,
    const sllm_paged_kv::BlockDescriptor *const descriptor_table,
    const uint32_t logical_table_count, const uint32_t descriptor_count,
    uint32_t *const device_status, float *const workspace,
    const uint64_t workspace_bytes, uint16_t *const output,
    const uint32_t query_count, const uint64_t start_position,
    const uint64_t committed_kv_length, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim, const uint32_t encoding,
    const bool use_gqa_shared, sllm_decode_control::ControlV1 *const control,
    const hipStream_t stream) noexcept {
  constexpr uint32_t kMaxQueryCount = 5U;
  constexpr uint32_t kQHeads = 24U;
  constexpr uint32_t kKvHeads = 4U;
  constexpr uint32_t kHeadDim = 256U;
  constexpr uint32_t kWorkspaceSplits = 128U;
  constexpr uint32_t kWorkspaceStride = kHeadDim + 2U;
  constexpr uint64_t kBytesPerQuery = static_cast<uint64_t>(kQHeads) *
                                      kWorkspaceSplits * kWorkspaceStride *
                                      sizeof(float);
  if (query == nullptr || logical_table == nullptr ||
      descriptor_table == nullptr || device_status == nullptr ||
      workspace == nullptr || output == nullptr || control == nullptr ||
      query_count == 0U || query_count > kMaxQueryCount ||
      logical_table_count == 0U || descriptor_count == 0U ||
      q_heads != kQHeads || kv_heads != kKvHeads || head_dim != kHeadDim ||
      encoding != SLLM_HIP_KV_ENCODING_MXFP8_E4_V1 ||
      start_position > UINT64_MAX - static_cast<uint64_t>(query_count) ||
      start_position + query_count != committed_kv_length ||
      committed_kv_length == 0U ||
      logical_table_count < committed_kv_length / 128U +
                                (committed_kv_length % 128U != 0U ? 1U : 0U) ||
      workspace_bytes < kBytesPerQuery * query_count) {
    return hipErrorInvalidValue;
  }

  hipError_t status =
      hipMemsetAsync(device_status, 0, sizeof(uint32_t), stream);
  if (status != hipSuccess) {
    return status;
  }

  // Both stage-1 nodes are captured with fixed pointers.  Each node checks
  // ControlV1.phase_position/phase_rows and exits unless its compile-time
  // split count is active.  The merge node below always uses the P128 layout
  // and consumes only the active P32 prefix when the control selects P32.
  if (use_gqa_shared) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            causal_attention_paged_decode_gqa6_stage1_kernel<32U, 128U, true>),
        dim3(query_count * kKvHeads * 32U), dim3(192U), 0U, stream, query,
        logical_table, descriptor_table, logical_table_count, descriptor_count,
        device_status, workspace, query_count, start_position, control);
    status = hipGetLastError();
    if (status != hipSuccess) {
      return status;
    }
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            causal_attention_paged_decode_gqa6_stage1_kernel<128U, 128U, true>),
        dim3(query_count * kKvHeads * 128U), dim3(192U), 0U, stream, query,
        logical_table, descriptor_table, logical_table_count, descriptor_count,
        device_status, workspace, query_count, start_position, control);
  } else {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            causal_attention_paged_decode_wave_split_stage1_kernel<32U, 128U,
                                                                   true>),
        dim3(query_count * kQHeads * 32U), dim3(32U), 0U, stream, query,
        logical_table, descriptor_table, logical_table_count, descriptor_count,
        device_status, workspace, query_count, start_position, q_heads,
        kv_heads, head_dim, control);
    status = hipGetLastError();
    if (status != hipSuccess) {
      return status;
    }
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            causal_attention_paged_decode_wave_split_stage1_kernel<128U, 128U,
                                                                   true>),
        dim3(query_count * kQHeads * 128U), dim3(32U), 0U, stream, query,
        logical_table, descriptor_table, logical_table_count, descriptor_count,
        device_status, workspace, query_count, start_position, q_heads,
        kv_heads, head_dim, control);
  }
  status = hipGetLastError();
  if (status != hipSuccess) {
    return status;
  }
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(
          causal_attention_decode_gqa6_staged32_split_merge_kernel<128U, true>),
      dim3(query_count * kQHeads), dim3(256U), 0U, stream, workspace, output,
      query_count, control);
  return hipGetLastError();
}

template <uint32_t kRowsPerGroup, uint32_t kSplits, uint32_t kWorkspaceSplits,
          bool DeviceControl>
hipError_t launch_paged_decode_gqa6_c1_impl(
    const uint16_t *const query, const uint32_t *const logical_table,
    const sllm_paged_kv::BlockDescriptor *const descriptor_table,
    const uint32_t logical_table_count, const uint32_t descriptor_count,
    uint32_t *const device_status, float *const workspace,
    const uint64_t workspace_bytes, uint16_t *const output,
    const uint32_t query_count, const uint64_t start_position,
    const uint64_t committed_kv_length, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim,
    sllm_decode_control::ControlV1 *const control,
    const hipStream_t stream) noexcept {
  constexpr uint32_t kQHeads = 24U;
  constexpr uint32_t kKvHeads = 4U;
  constexpr uint32_t kHeadDim = 256U;
  constexpr uint32_t kWorkspaceStride = kHeadDim + 2U;
  if (query == nullptr || logical_table == nullptr ||
      descriptor_table == nullptr || device_status == nullptr ||
      workspace == nullptr || output == nullptr || query_count < 2U ||
      query_count > 3U || logical_table_count == 0U || descriptor_count == 0U ||
      q_heads != kQHeads || kv_heads != kKvHeads || head_dim != kHeadDim ||
      committed_kv_length == 0U ||
      start_position > UINT64_MAX - static_cast<uint64_t>(query_count) ||
      start_position + query_count != committed_kv_length ||
      logical_table_count < committed_kv_length / 128U +
                                (committed_kv_length % 128U != 0U ? 1U : 0U) ||
      workspace_bytes < static_cast<uint64_t>(query_count) * kQHeads *
                            kWorkspaceSplits * kWorkspaceStride *
                            sizeof(float) ||
      (DeviceControl && control == nullptr)) {
    return hipErrorInvalidValue;
  }
  const uint32_t groups = (query_count + kRowsPerGroup - 1U) / kRowsPerGroup;
  const uint64_t grid = static_cast<uint64_t>(groups) * kKvHeads * kSplits;
  if (grid > UINT32_MAX) {
    return hipErrorInvalidValue;
  }
  hipError_t status =
      hipMemsetAsync(device_status, 0, sizeof(uint32_t), stream);
  if (status != hipSuccess) {
    return status;
  }
  hipLaunchKernelGGL(
      (causal_attention_paged_decode_gqa6_c1_stage1_kernel<
          kRowsPerGroup, kSplits, kWorkspaceSplits, DeviceControl>),
      dim3(static_cast<uint32_t>(grid)), dim3(192U), 0U, stream, query,
      logical_table, descriptor_table, logical_table_count, descriptor_count,
      device_status, workspace, query_count, start_position, control);
  status = hipGetLastError();
  if (status != hipSuccess) {
    return status;
  }
  if constexpr (kSplits == 128U) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            causal_attention_decode_gqa6_staged32_split_merge_kernel<
                128U, DeviceControl>),
        dim3(query_count * kQHeads), dim3(256U), 0U, stream, workspace, output,
        query_count, control);
  } else {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(causal_attention_decode_wave_split_staged_stage2_kernel<
                        32U, DeviceControl>),
        dim3(query_count * kQHeads), dim3(256U), 0U, stream, workspace, output,
        query_count, kQHeads, kKvHeads, kHeadDim, control);
  }
  return hipGetLastError();
}

hipError_t launch_paged_decode_gqa6_c1(
    const uint16_t *const query, const uint32_t *const logical_table,
    const sllm_paged_kv::BlockDescriptor *const descriptor_table,
    const uint32_t logical_table_count, const uint32_t descriptor_count,
    uint32_t *const device_status, float *const workspace,
    const uint64_t workspace_bytes, uint16_t *const output,
    const uint32_t query_count, const uint64_t start_position,
    const uint64_t committed_kv_length, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim, const uint32_t encoding,
    const hipStream_t stream) noexcept {
  if (encoding != SLLM_HIP_KV_ENCODING_MXFP8_E4_V1 || query_count < 2U ||
      query_count > 3U) {
    return hipErrorInvalidValue;
  }
  if (committed_kv_length >= 8192U) {
    return query_count == 2U
               ? launch_paged_decode_gqa6_c1_impl<2U, 128U, 128U, false>(
                     query, logical_table, descriptor_table,
                     logical_table_count, descriptor_count, device_status,
                     workspace, workspace_bytes, output, query_count,
                     start_position, committed_kv_length, q_heads, kv_heads,
                     head_dim, nullptr, stream)
               : launch_paged_decode_gqa6_c1_impl<3U, 128U, 128U, false>(
                     query, logical_table, descriptor_table,
                     logical_table_count, descriptor_count, device_status,
                     workspace, workspace_bytes, output, query_count,
                     start_position, committed_kv_length, q_heads, kv_heads,
                     head_dim, nullptr, stream);
  }
  return query_count == 2U
             ? launch_paged_decode_gqa6_c1_impl<2U, 32U, 32U, false>(
                   query, logical_table, descriptor_table, logical_table_count,
                   descriptor_count, device_status, workspace, workspace_bytes,
                   output, query_count, start_position, committed_kv_length,
                   q_heads, kv_heads, head_dim, nullptr, stream)
             : launch_paged_decode_gqa6_c1_impl<3U, 32U, 32U, false>(
                   query, logical_table, descriptor_table, logical_table_count,
                   descriptor_count, device_status, workspace, workspace_bytes,
                   output, query_count, start_position, committed_kv_length,
                   q_heads, kv_heads, head_dim, nullptr, stream);
}

hipError_t launch_paged_decode_gqa6_c1_device(
    const uint16_t *const query, const uint32_t *const logical_table,
    const sllm_paged_kv::BlockDescriptor *const descriptor_table,
    const uint32_t logical_table_count, const uint32_t descriptor_count,
    uint32_t *const device_status, float *const workspace,
    const uint64_t workspace_bytes, uint16_t *const output,
    const uint32_t query_count, const uint64_t start_position,
    const uint64_t committed_kv_length, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim, const uint32_t encoding,
    sllm_decode_control::ControlV1 *const control,
    const hipStream_t stream) noexcept {
  if (encoding != SLLM_HIP_KV_ENCODING_MXFP8_E4_V1 || control == nullptr ||
      query_count < 2U || query_count > 3U) {
    return hipErrorInvalidValue;
  }
  if (committed_kv_length >= 8192U) {
    return query_count == 2U
               ? launch_paged_decode_gqa6_c1_impl<2U, 128U, 128U, true>(
                     query, logical_table, descriptor_table,
                     logical_table_count, descriptor_count, device_status,
                     workspace, workspace_bytes, output, query_count,
                     start_position, committed_kv_length, q_heads, kv_heads,
                     head_dim, control, stream)
               : launch_paged_decode_gqa6_c1_impl<3U, 128U, 128U, true>(
                     query, logical_table, descriptor_table,
                     logical_table_count, descriptor_count, device_status,
                     workspace, workspace_bytes, output, query_count,
                     start_position, committed_kv_length, q_heads, kv_heads,
                     head_dim, control, stream);
  }
  return query_count == 2U
             ? launch_paged_decode_gqa6_c1_impl<2U, 32U, 128U, true>(
                   query, logical_table, descriptor_table, logical_table_count,
                   descriptor_count, device_status, workspace, workspace_bytes,
                   output, query_count, start_position, committed_kv_length,
                   q_heads, kv_heads, head_dim, control, stream)
             : launch_paged_decode_gqa6_c1_impl<3U, 32U, 128U, true>(
                   query, logical_table, descriptor_table, logical_table_count,
                   descriptor_count, device_status, workspace, workspace_bytes,
                   output, query_count, start_position, committed_kv_length,
                   q_heads, kv_heads, head_dim, control, stream);
}

hipError_t launch_paged_prefill_gqa6(
    const uint16_t *const query, const uint32_t *const logical_table,
    const sllm_paged_kv::BlockDescriptor *const descriptor_table,
    const uint32_t logical_table_count, const uint32_t descriptor_count,
    uint32_t *const device_status, uint16_t *const output,
    const uint32_t query_count, const uint64_t start_position,
    const uint64_t committed_kv_length, const uint32_t q_heads,
    const uint32_t kv_heads, const uint32_t head_dim, const uint32_t encoding,
    const float static_key_scale, const float static_value_scale,
    const bool wave_local_kv, const bool use_gfx1201_wave_provider,
    const bool use_gfx1201_qtile4_provider, const hipStream_t stream) noexcept {
  constexpr uint32_t kQHeads = 24U;
  constexpr uint32_t kKvHeads = 4U;
  constexpr uint32_t kHeadDim = 256U;
  if (query == nullptr || logical_table == nullptr ||
      descriptor_table == nullptr || device_status == nullptr ||
      output == nullptr || query_count == 0U || logical_table_count == 0U ||
      descriptor_count == 0U || q_heads != kQHeads || kv_heads != kKvHeads ||
      head_dim != kHeadDim || encoding != SLLM_HIP_KV_ENCODING_MXFP8_E4_V1 ||
      start_position > UINT64_MAX - query_count ||
      start_position + query_count != committed_kv_length ||
      committed_kv_length == 0U ||
      logical_table_count < committed_kv_length / 128U +
                                (committed_kv_length % 128U != 0U ? 1U : 0U)) {
    return hipErrorInvalidValue;
  }
  const uint64_t tiles = static_cast<uint64_t>(query_count) / 8U +
                         (query_count % 8U != 0U ? 1U : 0U);
  if (tiles > std::numeric_limits<uint32_t>::max() / kKvHeads) {
    return hipErrorInvalidValue;
  }
  const uint32_t grid_count = static_cast<uint32_t>(tiles * kKvHeads);
  const uint64_t qtile4_grid64 =
      (static_cast<uint64_t>(query_count) + 3U) / 4U * kKvHeads;
  if (qtile4_grid64 > std::numeric_limits<uint32_t>::max())
    return hipErrorInvalidValue;
  hipError_t status =
      hipMemsetAsync(device_status, 0, sizeof(uint32_t), stream);
  if (status != hipSuccess) {
    return status;
  }
  if (use_gfx1201_wave_provider) {
    const uint64_t wave_grid = static_cast<uint64_t>(query_count) * q_heads;
    if (wave_grid > std::numeric_limits<uint32_t>::max())
      return hipErrorInvalidValue;
    hipLaunchKernelGGL(
        causal_attention_paged_prefill_gfx1201_wave_kernel,
        dim3(static_cast<uint32_t>(wave_grid)), dim3(256U), 0U, stream, query,
        logical_table, descriptor_table, logical_table_count, descriptor_count,
        device_status, output, query_count, start_position, q_heads, kv_heads,
        head_dim, 1.0F / std::sqrt(static_cast<float>(head_dim)));
  } else if (use_gfx1201_qtile4_provider) {
    hipLaunchKernelGGL(causal_attention_paged_prefill_gqa6_qtile4_kernel,
                       dim3(static_cast<uint32_t>(qtile4_grid64)), dim3(256U),
                       0U, stream, query, logical_table, descriptor_table,
                       logical_table_count, descriptor_count, device_status,
                       output, query_count, start_position, q_heads, kv_heads,
                       head_dim);
  } else if (wave_local_kv) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            causal_attention_paged_prefill_gqa6_qtile8_kernel<true>),
        dim3(grid_count), dim3(512U), 0U, stream, query, logical_table,
        descriptor_table, logical_table_count, descriptor_count, device_status,
        output, query_count, start_position, q_heads, kv_heads, head_dim,
        static_key_scale, static_value_scale);
  } else {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            causal_attention_paged_prefill_gqa6_qtile8_kernel<false>),
        dim3(grid_count), dim3(512U), 0U, stream, query, logical_table,
        descriptor_table, logical_table_count, descriptor_count, device_status,
        output, query_count, start_position, q_heads, kv_heads, head_dim,
        static_key_scale, static_value_scale);
  }
  return hipGetLastError();
}

} // namespace sllm_causal_attention_kernel
