#include "kv_state_kernel_internal.hpp"
#include <lowp/detail/low_precision_block_codec.hpp>

#include <cmath>
#include <cstdint>

namespace {

using Bf16Input = const uint16_t *;

__device__ __forceinline__ uint16_t float_bits_to_f16(const uint32_t bits) {
  const uint32_t sign = (bits >> 16U) & UINT32_C(0x8000);
  const uint32_t exponent = (bits >> 23U) & UINT32_C(0xff);
  const uint32_t fraction = bits & UINT32_C(0x7fffff);
  if (exponent == UINT32_C(0xff)) {
    if (fraction == 0U) {
      return static_cast<uint16_t>(sign | UINT32_C(0x7c00));
    }
    return static_cast<uint16_t>(sign | UINT32_C(0x7e00));
  }
  const int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<uint16_t>(sign | UINT32_C(0x7c00));
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    const uint32_t mantissa = fraction | UINT32_C(0x800000);
    const uint32_t shift = static_cast<uint32_t>(14 - half_exponent);
    uint32_t rounded = mantissa >> shift;
    const uint32_t remainder = mantissa & ((UINT32_C(1) << shift) - 1U);
    const uint32_t halfway = UINT32_C(1) << (shift - 1U);
    if (remainder > halfway ||
        (remainder == halfway && (rounded & UINT32_C(1)) != 0U)) {
      ++rounded;
    }
    return static_cast<uint16_t>(sign | rounded);
  }
  uint32_t rounded_fraction = fraction >> 13U;
  const uint32_t remainder = fraction & UINT32_C(0x1fff);
  if (remainder > UINT32_C(0x1000) ||
      (remainder == UINT32_C(0x1000) &&
       (rounded_fraction & UINT32_C(1)) != 0U)) {
    ++rounded_fraction;
    if (rounded_fraction == UINT32_C(0x400)) {
      rounded_fraction = 0U;
      if (half_exponent + 1 >= 31) {
        return static_cast<uint16_t>(sign | UINT32_C(0x7c00));
      }
      return static_cast<uint16_t>(
          sign | (static_cast<uint32_t>(half_exponent + 1) << 10U));
    }
  }
  return static_cast<uint16_t>(
      sign | (static_cast<uint32_t>(half_exponent) << 10U) | rounded_fraction);
}

__device__ __forceinline__ uint16_t bf16_to_f16(const uint16_t value) {
  return float_bits_to_f16(static_cast<uint32_t>(value) << 16U);
}

__device__ __forceinline__ float bf16_to_float(const uint16_t value) {
  return __uint_as_float(static_cast<uint32_t>(value) << 16U);
}

__device__ __forceinline__ float e4m3fn_to_float(const uint8_t bits) {
  return sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode(bits);
}

__device__ __forceinline__ uint8_t float_to_e4m3fn(float value) {
  return sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::encode(value);
}

__device__ __forceinline__ uint8_t
float_to_e4m3fn_fp8_append(const float value) {
  return sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::encode(value);
}

__device__ __forceinline__ float e8m0_to_float(const uint8_t bits) {
  return sllm_lowp::ScalarCodec<sllm_lowp::E8M0>::decode(bits);
}

__device__ __forceinline__ uint8_t float_to_e2m1(float value) {
  return sllm_lowp::ScalarCodec<sllm_lowp::E2M1>::encode(value);
}

__device__ __forceinline__ void
mark_graph_phase_invalid(sllm_decode_control::ControlV1 *const control,
                         const sllm_decode_control::Status status);

__device__ __forceinline__ void
mark_graph_phase_invalid(sllm_decode_control::ControlV1 *const control,
                         const sllm_decode_control::Status status) {
  if (control != nullptr) {
    atomicExch(&control->status, static_cast<uint32_t>(status));
    atomicExch(&control->halted, 1U);
    atomicExch(&control->phase_active, 0U);
  }
}

__device__ __forceinline__ void mark_paged_append_status(uint32_t *const status,
                                                         const uint32_t value) {
  if (status != nullptr) {
    atomicCAS(status, sllm_kv_state_kernel::kPagedAppendStatusOk, value);
  }
}

__device__ __forceinline__ bool resolve_paged_phase(
    sllm_decode_control::ControlV1 *const control,
    const uint32_t requested_tokens, const uint64_t capacity_tokens,
    const uint64_t fixed_start_position, uint32_t *const active_tokens,
    uint64_t *const start_position, uint32_t *const status) {
  if (control == nullptr) {
    if (fixed_start_position > capacity_tokens ||
        requested_tokens > capacity_tokens - fixed_start_position) {
      mark_paged_append_status(
          status, sllm_kv_state_kernel::kPagedAppendStatusInvalidPosition);
      return false;
    }
    *active_tokens = requested_tokens;
    *start_position = fixed_start_position;
    return true;
  }
  if (control->phase_active == 0U || control->halted != 0U) {
    return false;
  }
  if (control->phase_rows == 0U) {
    *active_tokens = 0U;
    *start_position = control->phase_position;
    return false;
  }
  if (control->phase_rows > requested_tokens ||
      control->phase_position > UINT64_MAX - control->phase_rows ||
      control->phase_position > capacity_tokens ||
      control->phase_rows > capacity_tokens - control->phase_position) {
    mark_graph_phase_invalid(
        control, control->phase_rows > requested_tokens
                     ? sllm_decode_control::Status::InvalidWidth
                     : sllm_decode_control::Status::InvalidCapacity);
    mark_paged_append_status(
        status, sllm_kv_state_kernel::kPagedAppendStatusInvalidPosition);
    return false;
  }
  *active_tokens = control->phase_rows;
  *start_position = control->phase_position;
  return true;
}

__device__ __forceinline__ bool resolve_paged_row(
    const sllm_paged_kv::BlockDescriptor *const block_descriptors,
    const uint32_t *const logical_table, const uint32_t table_capacity,
    const uint32_t descriptor_capacity, const uint32_t encoding,
    const uint64_t absolute_token, const uint32_t head_count,
    const uint32_t head_dim, const uint64_t row,
    sllm_paged_kv::BlockDescriptor *const descriptor,
    uint64_t *const output_row, sllm_decode_control::ControlV1 *const control,
    uint32_t *const status) {
  constexpr uint64_t kBlockTokens = 128U;
  const uint64_t logical_block = absolute_token / kBlockTokens;
  if (logical_block >= table_capacity) {
    mark_graph_phase_invalid(control,
                             sllm_decode_control::Status::InvalidPosition);
    mark_paged_append_status(
        status, sllm_kv_state_kernel::kPagedAppendStatusInvalidTable);
    return false;
  }
  const uint32_t physical_block = logical_table[logical_block];
  if (physical_block >= descriptor_capacity) {
    mark_graph_phase_invalid(control,
                             sllm_decode_control::Status::InvalidPosition);
    mark_paged_append_status(
        status, sllm_kv_state_kernel::kPagedAppendStatusInvalidTable);
    return false;
  }
  const sllm_paged_kv::BlockDescriptor selected =
      block_descriptors[physical_block];
  const bool needs_token_scales = encoding == SLLM_HIP_KV_ENCODING_FP8_V1;
  const bool needs_block_scales =
      encoding == SLLM_HIP_KV_ENCODING_NVFP4_V1 ||
      encoding == SLLM_HIP_KV_ENCODING_MXFP8_E4_V1 ||
      encoding == SLLM_HIP_KV_ENCODING_MXFP8_E5_V1;
  const bool needs_outer_scales = encoding == SLLM_HIP_KV_ENCODING_NVFP4_V1;
  if (selected.key == nullptr || selected.value == nullptr ||
      (needs_token_scales &&
       (selected.key_scale == nullptr || selected.value_scale == nullptr)) ||
      (needs_block_scales &&
       (selected.key_scale == nullptr || selected.value_scale == nullptr)) ||
      (needs_outer_scales && (selected.key_outer_scale == nullptr ||
                              selected.value_outer_scale == nullptr))) {
    mark_graph_phase_invalid(control,
                             sllm_decode_control::Status::InvalidPosition);
    mark_paged_append_status(
        status, sllm_kv_state_kernel::kPagedAppendStatusInvalidDescriptor);
    return false;
  }
  const uint64_t row_in_block = absolute_token % kBlockTokens;
  const uint64_t head = row % head_count;
  const uint64_t block_row = row_in_block * head_count + head;
  const uint64_t padded_head_dim =
      (encoding == SLLM_HIP_KV_ENCODING_MXFP8_E4_V1 ||
       encoding == SLLM_HIP_KV_ENCODING_MXFP8_E5_V1)
          ? ((static_cast<uint64_t>(head_dim) + 31U) / 32U) * 32U
          : head_dim;
  if (block_row > (UINT64_MAX - (padded_head_dim - 1U)) / padded_head_dim) {
    mark_graph_phase_invalid(control,
                             sllm_decode_control::Status::InvalidPosition);
    mark_paged_append_status(
        status, sllm_kv_state_kernel::kPagedAppendStatusInvalidPosition);
    return false;
  }
  *descriptor = selected;
  *output_row = block_row;
  return true;
}

// Sliding states keep the physical table in ring-slot order while the tag
// table preserves the absolute block identity.  Do not infer the identity
// from the slot alone: after block 9, slot zero may still contain block zero
// until the host has completed its retention update.
__device__ __forceinline__ bool resolve_paged_sliding_row(
    const sllm_paged_kv::BlockDescriptor *const block_descriptors,
    const uint32_t *const ring_table, const uint64_t *const ring_tags,
    const uint32_t ring_slot_count, const uint32_t descriptor_capacity,
    const uint32_t encoding, const uint64_t absolute_token,
    const uint32_t head_count, const uint32_t head_dim, const uint64_t row,
    sllm_paged_kv::BlockDescriptor *const descriptor,
    uint64_t *const output_row, sllm_decode_control::ControlV1 *const control,
    uint32_t *const status) {
  constexpr uint64_t kBlockTokens = 128U;
  if (ring_slot_count != sllm_kv_state_kernel::kSlidingRingSlots ||
      ring_table == nullptr || ring_tags == nullptr) {
    mark_graph_phase_invalid(control,
                             sllm_decode_control::Status::InvalidPosition);
    mark_paged_append_status(
        status, sllm_kv_state_kernel::kPagedAppendStatusInvalidTable);
    return false;
  }
  const uint64_t absolute_block = absolute_token / kBlockTokens;
  const uint32_t slot = static_cast<uint32_t>(
      absolute_block % static_cast<uint64_t>(ring_slot_count));
  if (ring_tags[slot] != absolute_block) {
    mark_graph_phase_invalid(control,
                             sllm_decode_control::Status::InvalidPosition);
    mark_paged_append_status(
        status, sllm_kv_state_kernel::kPagedAppendStatusInvalidTable);
    return false;
  }
  const uint32_t physical_block = ring_table[slot];
  if (physical_block >= descriptor_capacity) {
    mark_graph_phase_invalid(control,
                             sllm_decode_control::Status::InvalidPosition);
    mark_paged_append_status(
        status, sllm_kv_state_kernel::kPagedAppendStatusInvalidTable);
    return false;
  }
  const sllm_paged_kv::BlockDescriptor selected =
      block_descriptors[physical_block];
  if (selected.key == nullptr || selected.value == nullptr ||
      encoding != SLLM_HIP_KV_ENCODING_FP8_STATIC_V1) {
    mark_graph_phase_invalid(control,
                             sllm_decode_control::Status::InvalidPosition);
    mark_paged_append_status(
        status, sllm_kv_state_kernel::kPagedAppendStatusInvalidDescriptor);
    return false;
  }
  const uint64_t row_in_block = absolute_token % kBlockTokens;
  const uint64_t head = row % head_count;
  const uint64_t block_row = row_in_block * head_count + head;
  if (block_row > (UINT64_MAX - (static_cast<uint64_t>(head_dim) - 1U)) /
                      static_cast<uint64_t>(head_dim)) {
    mark_graph_phase_invalid(control,
                             sllm_decode_control::Status::InvalidPosition);
    mark_paged_append_status(
        status, sllm_kv_state_kernel::kPagedAppendStatusInvalidPosition);
    return false;
  }
  *descriptor = selected;
  *output_row = block_row;
  return true;
}

} // namespace

template <typename BlockFormat>
__device__ void
quantize_mxfp8_pair(const uint16_t *const key_input,
                    const uint16_t *const value_input,
                    uint8_t *const key_output, uint8_t *const value_output,
                    uint8_t *const key_scales, uint8_t *const value_scales,
                    const uint64_t input_base, const uint64_t output_row,
                    const uint32_t head_dim) {
  constexpr uint32_t kBlockSize = BlockFormat::kBlockSize;
  constexpr uint32_t kWaveSize = 32U;
  const uint64_t blocks_per_row =
      (static_cast<uint64_t>(head_dim) + kBlockSize - 1U) / kBlockSize;
  const uint64_t padded_head_dim = blocks_per_row * kBlockSize;
  const uint32_t lane = threadIdx.x & (kWaveSize - 1U);
  const uint32_t wave = threadIdx.x / kWaveSize;
  const uint32_t wave_count = blockDim.x / kWaveSize;
  sllm_lowp::MutableBlockScaledView<BlockFormat> key_view{
      key_output, key_scales,      nullptr,
      head_dim,   padded_head_dim, blocks_per_row};
  sllm_lowp::MutableBlockScaledView<BlockFormat> value_view{
      value_output, value_scales,    nullptr,
      head_dim,     padded_head_dim, blocks_per_row};
  for (uint64_t block = wave; block < blocks_per_row; block += wave_count) {
    const uint32_t dimension = static_cast<uint32_t>(block * kBlockSize) + lane;
    const bool active = dimension < head_dim;
    const float key_value =
        active ? bf16_to_float(key_input[input_base + dimension]) : 0.0F;
    const float value_value =
        active ? bf16_to_float(value_input[input_base + dimension]) : 0.0F;
    float key_maximum = isfinite(key_value) ? fabsf(key_value) : 0.0F;
    float value_maximum = isfinite(value_value) ? fabsf(value_value) : 0.0F;
    uint32_t key_all_zero = !active || key_value == 0.0F ? 1U : 0U;
    uint32_t value_all_zero = !active || value_value == 0.0F ? 1U : 0U;
    key_maximum = sllm_lowp::wave_amax(key_maximum);
    value_maximum = sllm_lowp::wave_amax(value_maximum);
    key_all_zero = sllm_lowp::wave_and(key_all_zero);
    value_all_zero = sllm_lowp::wave_and(value_all_zero);
    uint32_t key_scale_bits = 0U;
    uint32_t value_scale_bits = 0U;
    float key_scale = 1.0F;
    float value_scale = 1.0F;
    if (lane == 0U) {
      key_scale_bits =
          sllm_lowp::BlockCodec<BlockFormat>::scale_code(key_maximum);
      value_scale_bits =
          sllm_lowp::BlockCodec<BlockFormat>::scale_code(value_maximum);
      key_view.block_scales[output_row * key_view.scale_stride + block] =
          static_cast<uint8_t>(key_scale_bits);
      value_view.block_scales[output_row * value_view.scale_stride + block] =
          static_cast<uint8_t>(value_scale_bits);
      key_scale = e8m0_to_float(static_cast<uint8_t>(key_scale_bits));
      value_scale = e8m0_to_float(static_cast<uint8_t>(value_scale_bits));
    }
    key_all_zero = __shfl(key_all_zero, 0U, kWaveSize);
    value_all_zero = __shfl(value_all_zero, 0U, kWaveSize);
    key_scale = __shfl(key_scale, 0U, kWaveSize);
    value_scale = __shfl(value_scale, 0U, kWaveSize);
    if (dimension < padded_head_dim) {
      key_view.values[output_row * key_view.value_stride + dimension] =
          !active || key_all_zero != 0U
              ? 0U
              : sllm_lowp::ScalarCodec<typename BlockFormat::Element>::encode(
                    key_value / key_scale);
      value_view.values[output_row * value_view.value_stride + dimension] =
          !active || value_all_zero != 0U
              ? 0U
              : sllm_lowp::ScalarCodec<typename BlockFormat::Element>::encode(
                    value_value / value_scale);
    }
  }
}

template <bool StaticScale>
__device__ void quantize_fp8_paged_pair(
    const uint16_t *const key_input, const uint16_t *const value_input,
    sllm_paged_kv::BlockDescriptor descriptor, const uint64_t input_base,
    const uint64_t output_row, const uint32_t head_dim,
    const float static_key_scale, const float static_value_scale,
    float *const key_maxima, float *const value_maxima) {
  const uint32_t dimension = threadIdx.x;
  float key_maximum = 0.0F;
  float value_maximum = 0.0F;
  for (uint32_t current = dimension; current < head_dim;
       current += blockDim.x) {
    const float key_value = bf16_to_float(key_input[input_base + current]);
    const float value_value = bf16_to_float(value_input[input_base + current]);
    key_maximum =
        fmaxf(key_maximum, isfinite(key_value) ? fabsf(key_value) : 0.0F);
    value_maximum =
        fmaxf(value_maximum, isfinite(value_value) ? fabsf(value_value) : 0.0F);
  }
  key_maxima[dimension] = key_maximum;
  value_maxima[dimension] = value_maximum;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2U; stride != 0U; stride >>= 1U) {
    if (dimension < stride) {
      key_maxima[dimension] =
          fmaxf(key_maxima[dimension], key_maxima[dimension + stride]);
      value_maxima[dimension] =
          fmaxf(value_maxima[dimension], value_maxima[dimension + stride]);
    }
    __syncthreads();
  }
  const float key_scale =
      StaticScale ? static_key_scale
                  : (key_maxima[0] == 0.0F ? 1.0F : key_maxima[0] / 448.0F);
  const float value_scale =
      StaticScale ? static_value_scale
                  : (value_maxima[0] == 0.0F ? 1.0F : value_maxima[0] / 448.0F);
  if (dimension == 0U && !StaticScale) {
    static_cast<float *>(
        static_cast<void *>(descriptor.key_scale))[output_row] = key_scale;
    static_cast<float *>(
        static_cast<void *>(descriptor.value_scale))[output_row] = value_scale;
  }
  for (uint32_t current = dimension; current < head_dim;
       current += blockDim.x) {
    const uint64_t output_offset = output_row * head_dim + current;
    descriptor.key[output_offset] = float_to_e4m3fn_fp8_append(
        bf16_to_float(key_input[input_base + current]) / key_scale);
    descriptor.value[output_offset] = float_to_e4m3fn_fp8_append(
        bf16_to_float(value_input[input_base + current]) / value_scale);
  }
}

template <bool Key>
__device__ void
quantize_nvfp4_row(const uint16_t *const input, uint8_t *const packed,
                   uint8_t *const block_scales, float *const outer_scales,
                   const uint64_t input_base, const uint64_t output_row,
                   const uint32_t head_dim) {
  if (threadIdx.x != 0U) {
    return;
  }
  float maximum = 0.0F;
  for (uint32_t dimension = 0U; dimension != head_dim; ++dimension) {
    const float value = bf16_to_float(input[input_base + dimension]);
    maximum = fmaxf(maximum, isfinite(value) ? fabsf(value) : 0.0F);
  }
  const float outer = maximum == 0.0F ? 1.0F : maximum / (448.0F * 6.0F);
  const uint64_t packed_per_row = (static_cast<uint64_t>(head_dim) + 1U) / 2U;
  const uint64_t blocks_per_row = (static_cast<uint64_t>(head_dim) + 15U) / 16U;
  sllm_lowp::MutableBlockScaledView<sllm_lowp::Nvfp4Block16> view{
      packed,   block_scales,   outer_scales,
      head_dim, packed_per_row, blocks_per_row};
  view.outer_scales[output_row] = outer;
  for (uint64_t block = 0U; block != blocks_per_row; ++block) {
    const uint32_t begin = static_cast<uint32_t>(block * 16U);
    const uint32_t end = min(begin + 16U, head_dim);
    float block_maximum = 0.0F;
    bool block_has_infinity = false;
    for (uint32_t dimension = begin; dimension != end; ++dimension) {
      const float value = bf16_to_float(input[input_base + dimension]);
      block_maximum =
          fmaxf(block_maximum, isfinite(value) ? fabsf(value) : 0.0F);
      block_has_infinity = block_has_infinity || isinf(value);
    }
    // E2M1 has no non-finite encodings. Preserve NaN as canonical zero, and
    // saturate either infinity to the largest value representable by the
    // row's finite-derived outer scale. For an all-infinite row outer is one.
    if (block_has_infinity) {
      block_maximum = 448.0F * 6.0F * outer;
    }
    const uint8_t scale_bits = float_to_e4m3fn((block_maximum / 6.0F) / outer);
    const float decoded_scale = e4m3fn_to_float(scale_bits);
    view.block_scales[output_row * view.scale_stride + block] = scale_bits;
    for (uint32_t dimension = begin; dimension < end; dimension += 2U) {
      const float first = bf16_to_float(input[input_base + dimension]);
      const uint8_t low = decoded_scale == 0.0F
                              ? 0U
                              : float_to_e2m1(first / (decoded_scale * outer));
      uint8_t high = 0U;
      if (dimension + 1U < end) {
        const float second = bf16_to_float(input[input_base + dimension + 1U]);
        high = decoded_scale == 0.0F
                   ? 0U
                   : float_to_e2m1(second / (decoded_scale * outer));
      }
      view.values[output_row * view.value_stride + dimension / 2U] =
          static_cast<uint8_t>(low | (high << 4U));
    }
  }
  (void)Key;
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_KV_WORKGROUP_SIZE,
    1) void sllm_kv_state_bf16_to_paged_f16_v1(Bf16Input key_input,
                                               Bf16Input value_input,
                                               const sllm_paged_kv::
                                                   BlockDescriptor
                                                       *const block_descriptors,
                                               const uint32_t
                                                   *const logical_table,
                                               const uint32_t table_capacity,
                                               const uint32_t
                                                   descriptor_capacity,
                                               const uint32_t token_count,
                                               const uint64_t capacity_tokens,
                                               const uint64_t
                                                   fixed_start_position,
                                               const uint32_t head_count,
                                               const uint32_t head_dim,
                                               sllm_decode_control::ControlV1
                                                   *const control,
                                               uint32_t *const device_status) {
  uint32_t active_tokens = token_count;
  uint64_t start_position = fixed_start_position;
  if (!resolve_paged_phase(control, token_count, capacity_tokens,
                           fixed_start_position, &active_tokens,
                           &start_position, device_status)) {
    return;
  }
  const uint64_t row_width = static_cast<uint64_t>(head_count) * head_dim;
  const uint64_t element = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                           static_cast<uint64_t>(threadIdx.x);
  const uint64_t total = static_cast<uint64_t>(active_tokens) * row_width;
  if (element >= total || *device_status != 0U) {
    return;
  }
  const uint64_t row = element / head_dim;
  const uint32_t dimension = static_cast<uint32_t>(element % head_dim);
  sllm_paged_kv::BlockDescriptor descriptor{};
  uint64_t output_row = 0U;
  if (!resolve_paged_row(block_descriptors, logical_table, table_capacity,
                         descriptor_capacity, SLLM_HIP_KV_ENCODING_FP16_V1,
                         start_position + row / head_count, head_count,
                         head_dim, row, &descriptor, &output_row, control,
                         device_status) ||
      *device_status != 0U) {
    return;
  }
  const uint64_t output_offset = output_row * head_dim + dimension;
  static_cast<uint16_t *>(static_cast<void *>(descriptor.key))[output_offset] =
      bf16_to_f16(key_input[element]);
  static_cast<uint16_t *>(static_cast<void *>(
      descriptor.value))[output_offset] = bf16_to_f16(value_input[element]);
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_KV_WORKGROUP_SIZE,
    1) void sllm_kv_state_bf16_to_paged_mxfp8_e4_v1(Bf16Input key_input,
                                                    Bf16Input value_input,
                                                    const sllm_paged_kv::
                                                        BlockDescriptor *const
                                                            block_descriptors,
                                                    const uint32_t
                                                        *const logical_table,
                                                    const uint32_t
                                                        table_capacity,
                                                    const uint32_t
                                                        descriptor_capacity,
                                                    const uint32_t token_count,
                                                    const uint64_t
                                                        capacity_tokens,
                                                    const uint64_t
                                                        fixed_start_position,
                                                    const uint32_t head_count,
                                                    const uint32_t head_dim,
                                                    sllm_decode_control::
                                                        ControlV1
                                                            *const control,
                                                    uint32_t
                                                        *const device_status) {
  uint32_t active_tokens = token_count;
  uint64_t start_position = fixed_start_position;
  if (!resolve_paged_phase(control, token_count, capacity_tokens,
                           fixed_start_position, &active_tokens,
                           &start_position, device_status)) {
    return;
  }
  const uint64_t row = blockIdx.x;
  const uint64_t total_rows = static_cast<uint64_t>(active_tokens) * head_count;
  if (row >= total_rows || *device_status != 0U) {
    return;
  }
  sllm_paged_kv::BlockDescriptor descriptor{};
  uint64_t output_row = 0U;
  if (!resolve_paged_row(block_descriptors, logical_table, table_capacity,
                         descriptor_capacity, SLLM_HIP_KV_ENCODING_MXFP8_E4_V1,
                         start_position + row / head_count, head_count,
                         head_dim, row, &descriptor, &output_row, control,
                         device_status) ||
      *device_status != 0U) {
    return;
  }
  quantize_mxfp8_pair<sllm_lowp::Mxfp8E4Block32>(
      key_input, value_input, descriptor.key, descriptor.value,
      descriptor.key_scale, descriptor.value_scale,
      row * static_cast<uint64_t>(head_dim), output_row, head_dim);
}

template <bool StaticScale>
__device__ void quantize_paged_fp8_kernel_row(
    Bf16Input key_input, Bf16Input value_input,
    const sllm_paged_kv::BlockDescriptor *const block_descriptors,
    const uint32_t *const logical_table, const uint32_t table_capacity,
    const uint32_t descriptor_capacity, const uint32_t token_count,
    const uint64_t capacity_tokens, const uint64_t fixed_start_position,
    const uint32_t head_count, const uint32_t head_dim,
    const float static_key_scale, const float static_value_scale,
    sllm_decode_control::ControlV1 *const control,
    uint32_t *const device_status) {
  uint32_t active_tokens = token_count;
  uint64_t start_position = fixed_start_position;
  if (!resolve_paged_phase(control, token_count, capacity_tokens,
                           fixed_start_position, &active_tokens,
                           &start_position, device_status)) {
    return;
  }
  const uint64_t row = blockIdx.x;
  const uint64_t total_rows = static_cast<uint64_t>(active_tokens) * head_count;
  if (row >= total_rows || *device_status != 0U) {
    return;
  }
  sllm_paged_kv::BlockDescriptor descriptor{};
  uint64_t output_row = 0U;
  const uint32_t encoding = StaticScale ? SLLM_HIP_KV_ENCODING_FP8_STATIC_V1
                                        : SLLM_HIP_KV_ENCODING_FP8_V1;
  if (!resolve_paged_row(
          block_descriptors, logical_table, table_capacity, descriptor_capacity,
          encoding, start_position + row / head_count, head_count, head_dim,
          row, &descriptor, &output_row, control, device_status) ||
      *device_status != 0U) {
    return;
  }
  __shared__ float key_maxima[SLLM_HIP_KV_WORKGROUP_SIZE];
  __shared__ float value_maxima[SLLM_HIP_KV_WORKGROUP_SIZE];
  quantize_fp8_paged_pair<StaticScale>(
      key_input, value_input, descriptor, row * static_cast<uint64_t>(head_dim),
      output_row, head_dim, static_key_scale, static_value_scale, key_maxima,
      value_maxima);
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_KV_WORKGROUP_SIZE,
    1) void sllm_kv_state_bf16_to_paged_fp8_v1(Bf16Input key_input,
                                               Bf16Input value_input,
                                               const sllm_paged_kv::
                                                   BlockDescriptor
                                                       *const block_descriptors,
                                               const uint32_t
                                                   *const logical_table,
                                               const uint32_t table_capacity,
                                               const uint32_t
                                                   descriptor_capacity,
                                               const uint32_t token_count,
                                               const uint64_t capacity_tokens,
                                               const uint64_t
                                                   fixed_start_position,
                                               const uint32_t head_count,
                                               const uint32_t head_dim,
                                               sllm_decode_control::ControlV1
                                                   *const control,
                                               uint32_t *const device_status) {
  quantize_paged_fp8_kernel_row<false>(
      key_input, value_input, block_descriptors, logical_table, table_capacity,
      descriptor_capacity, token_count, capacity_tokens, fixed_start_position,
      head_count, head_dim, 1.0F, 1.0F, control, device_status);
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_KV_WORKGROUP_SIZE,
    1) void sllm_kv_state_bf16_to_paged_fp8_static_v1(Bf16Input key_input,
                                                      Bf16Input value_input,
                                                      const sllm_paged_kv::
                                                          BlockDescriptor *const
                                                              block_descriptors,
                                                      const uint32_t
                                                          *const logical_table,
                                                      const uint32_t
                                                          table_capacity,
                                                      const uint32_t
                                                          descriptor_capacity,
                                                      const uint32_t
                                                          token_count,
                                                      const uint64_t
                                                          capacity_tokens,
                                                      const uint64_t
                                                          fixed_start_position,
                                                      const uint32_t head_count,
                                                      const uint32_t head_dim,
                                                      const float
                                                          static_key_scale,
                                                      const float
                                                          static_value_scale,
                                                      sllm_decode_control::
                                                          ControlV1
                                                              *const control,
                                                      uint32_t *const
                                                          device_status) {
  quantize_paged_fp8_kernel_row<true>(
      key_input, value_input, block_descriptors, logical_table, table_capacity,
      descriptor_capacity, token_count, capacity_tokens, fixed_start_position,
      head_count, head_dim, static_key_scale, static_value_scale, control,
      device_status);
}

template <bool StaticScale>
__device__ void quantize_paged_sliding_fp8_kernel_row(
    Bf16Input key_input, Bf16Input value_input,
    const sllm_paged_kv::BlockDescriptor *const block_descriptors,
    const uint32_t *const ring_table, const uint64_t *const ring_tags,
    const uint32_t ring_slot_count, const uint32_t descriptor_capacity,
    const uint32_t token_count, const uint64_t retained_start,
    const uint64_t start_position, const uint32_t head_count,
    const uint32_t head_dim, const float static_key_scale,
    const float static_value_scale, uint32_t *const device_status) {
  static_assert(StaticScale);
  (void)retained_start;
  const uint64_t row = blockIdx.x;
  const uint64_t total_rows = static_cast<uint64_t>(token_count) * head_count;
  if (row >= total_rows || *device_status != 0U) {
    return;
  }
  const uint64_t token = start_position + row / head_count;
  sllm_paged_kv::BlockDescriptor descriptor{};
  uint64_t output_row = 0U;
  if (!resolve_paged_sliding_row(block_descriptors, ring_table, ring_tags,
                                 ring_slot_count, descriptor_capacity,
                                 SLLM_HIP_KV_ENCODING_FP8_STATIC_V1, token,
                                 head_count, head_dim, row, &descriptor,
                                 &output_row, nullptr, device_status) ||
      *device_status != 0U) {
    return;
  }
  __shared__ float key_maxima[SLLM_HIP_KV_WORKGROUP_SIZE];
  __shared__ float value_maxima[SLLM_HIP_KV_WORKGROUP_SIZE];
  quantize_fp8_paged_pair<true>(key_input, value_input, descriptor,
                                row * static_cast<uint64_t>(head_dim),
                                output_row, head_dim, static_key_scale,
                                static_value_scale, key_maxima, value_maxima);
}

extern "C" __global__
__launch_bounds__(SLLM_HIP_KV_WORKGROUP_SIZE, 1) void sllm_kv_state_bf16_to_paged_fp8_static_sliding_ring_v1(
    Bf16Input key_input, Bf16Input value_input,
    const sllm_paged_kv::BlockDescriptor *const block_descriptors,
    const uint32_t *const ring_table, const uint64_t *const ring_tags,
    const uint32_t ring_slot_count, const uint32_t descriptor_capacity,
    const uint32_t token_count, const uint64_t retained_start,
    const uint64_t start_position, const uint32_t head_count,
    const uint32_t head_dim, const float static_key_scale,
    const float static_value_scale, uint32_t *const device_status) {
  quantize_paged_sliding_fp8_kernel_row<true>(
      key_input, value_input, block_descriptors, ring_table, ring_tags,
      ring_slot_count, descriptor_capacity, token_count, retained_start,
      start_position, head_count, head_dim, static_key_scale,
      static_value_scale, device_status);
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_KV_WORKGROUP_SIZE,
    1) void sllm_kv_state_bf16_to_paged_nvfp4_v1(Bf16Input key_input,
                                                 Bf16Input value_input,
                                                 const sllm_paged_kv::
                                                     BlockDescriptor *const
                                                         block_descriptors,
                                                 const uint32_t
                                                     *const logical_table,
                                                 const uint32_t table_capacity,
                                                 const uint32_t
                                                     descriptor_capacity,
                                                 const uint32_t token_count,
                                                 const uint64_t capacity_tokens,
                                                 const uint64_t
                                                     fixed_start_position,
                                                 const uint32_t head_count,
                                                 const uint32_t head_dim,
                                                 sllm_decode_control::ControlV1
                                                     *const control,
                                                 uint32_t
                                                     *const device_status) {
  uint32_t active_tokens = token_count;
  uint64_t start_position = fixed_start_position;
  if (!resolve_paged_phase(control, token_count, capacity_tokens,
                           fixed_start_position, &active_tokens,
                           &start_position, device_status)) {
    return;
  }
  const uint64_t row = blockIdx.x;
  const uint64_t total_rows = static_cast<uint64_t>(active_tokens) * head_count;
  if (row >= total_rows || *device_status != 0U) {
    return;
  }
  sllm_paged_kv::BlockDescriptor descriptor{};
  uint64_t output_row = 0U;
  if (!resolve_paged_row(block_descriptors, logical_table, table_capacity,
                         descriptor_capacity, SLLM_HIP_KV_ENCODING_NVFP4_V1,
                         start_position + row / head_count, head_count,
                         head_dim, row, &descriptor, &output_row, control,
                         device_status) ||
      *device_status != 0U) {
    return;
  }
  const uint64_t input_base = row * static_cast<uint64_t>(head_dim);
  quantize_nvfp4_row<true>(
      key_input, descriptor.key, descriptor.key_scale,
      reinterpret_cast<float *>(descriptor.key_outer_scale), input_base,
      output_row, head_dim);
  quantize_nvfp4_row<false>(
      value_input, descriptor.value, descriptor.value_scale,
      reinterpret_cast<float *>(descriptor.value_outer_scale), input_base,
      output_row, head_dim);
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_KV_WORKGROUP_SIZE,
    1) void sllm_kv_state_bf16_to_paged_mxfp8_e5_v1(Bf16Input key_input,
                                                    Bf16Input value_input,
                                                    const sllm_paged_kv::
                                                        BlockDescriptor *const
                                                            block_descriptors,
                                                    const uint32_t
                                                        *const logical_table,
                                                    const uint32_t
                                                        table_capacity,
                                                    const uint32_t
                                                        descriptor_capacity,
                                                    const uint32_t token_count,
                                                    const uint64_t
                                                        capacity_tokens,
                                                    const uint64_t
                                                        fixed_start_position,
                                                    const uint32_t head_count,
                                                    const uint32_t head_dim,
                                                    sllm_decode_control::
                                                        ControlV1
                                                            *const control,
                                                    uint32_t
                                                        *const device_status) {
  uint32_t active_tokens = token_count;
  uint64_t start_position = fixed_start_position;
  if (!resolve_paged_phase(control, token_count, capacity_tokens,
                           fixed_start_position, &active_tokens,
                           &start_position, device_status)) {
    return;
  }
  const uint64_t row = blockIdx.x;
  const uint64_t total_rows = static_cast<uint64_t>(active_tokens) * head_count;
  if (row >= total_rows || *device_status != 0U) {
    return;
  }
  sllm_paged_kv::BlockDescriptor descriptor{};
  uint64_t output_row = 0U;
  if (!resolve_paged_row(block_descriptors, logical_table, table_capacity,
                         descriptor_capacity, SLLM_HIP_KV_ENCODING_MXFP8_E5_V1,
                         start_position + row / head_count, head_count,
                         head_dim, row, &descriptor, &output_row, control,
                         device_status) ||
      *device_status != 0U) {
    return;
  }
  quantize_mxfp8_pair<sllm_lowp::Mxfp8E5Block32>(
      key_input, value_input, descriptor.key, descriptor.value,
      descriptor.key_scale, descriptor.value_scale,
      row * static_cast<uint64_t>(head_dim), output_row, head_dim);
}

namespace sllm_kv_state_kernel {

namespace {

hipError_t launch_paged_impl(
    const uint16_t *const key_input, const uint16_t *const value_input,
    const sllm_paged_kv::BlockDescriptor *const block_descriptors,
    const uint32_t *const logical_table, const uint32_t table_capacity,
    const uint32_t descriptor_capacity, const uint32_t token_count,
    const uint64_t capacity_tokens, const uint64_t start_position,
    const uint32_t head_count, const uint32_t head_dim, const uint32_t encoding,
    sllm_decode_control::ControlV1 *const control,
    uint32_t *const device_status, const hipStream_t stream,
    const float static_key_scale, const float static_value_scale) noexcept {
  if (key_input == nullptr || value_input == nullptr ||
      block_descriptors == nullptr || logical_table == nullptr ||
      device_status == nullptr || table_capacity == 0U ||
      descriptor_capacity == 0U || token_count == 0U || capacity_tokens == 0U ||
      head_count == 0U || head_dim == 0U) {
    return hipErrorInvalidValue;
  }
  if (encoding != SLLM_HIP_KV_ENCODING_FP16_V1 &&
      encoding != SLLM_HIP_KV_ENCODING_FP8_V1 &&
      encoding != SLLM_HIP_KV_ENCODING_FP8_STATIC_V1 &&
      encoding != SLLM_HIP_KV_ENCODING_NVFP4_V1 &&
      encoding != SLLM_HIP_KV_ENCODING_MXFP8_E4_V1 &&
      encoding != SLLM_HIP_KV_ENCODING_MXFP8_E5_V1) {
    return hipErrorNotSupported;
  }
  if (encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1 &&
      (!std::isfinite(static_key_scale) || static_key_scale <= 0.0F ||
       !std::isfinite(static_value_scale) || static_value_scale <= 0.0F)) {
    return hipErrorInvalidValue;
  }
  if (control == nullptr && (start_position > capacity_tokens ||
                             token_count > capacity_tokens - start_position)) {
    return hipErrorInvalidValue;
  }
  const uint64_t rows = static_cast<uint64_t>(token_count) * head_count;
  if (rows == 0U || rows > UINT32_MAX) {
    return hipErrorInvalidValue;
  }
  if (control == nullptr) {
    const uint64_t last_position = start_position + token_count - 1U;
    const uint64_t required_table_entries = last_position / 128U + 1U;
    if (required_table_entries > table_capacity) {
      return hipErrorInvalidValue;
    }
  }
  uint64_t grid_count = rows;
  if (encoding == SLLM_HIP_KV_ENCODING_FP16_V1) {
    if (rows > UINT64_MAX / head_dim) {
      return hipErrorInvalidValue;
    }
    const uint64_t elements = rows * head_dim;
    if (elements > UINT64_MAX - (SLLM_HIP_KV_WORKGROUP_SIZE - 1U)) {
      return hipErrorInvalidValue;
    }
    grid_count = (elements + SLLM_HIP_KV_WORKGROUP_SIZE - 1U) /
                 SLLM_HIP_KV_WORKGROUP_SIZE;
  }
  if (grid_count == 0U || grid_count > UINT32_MAX) {
    return hipErrorInvalidValue;
  }
  hipError_t result =
      hipMemsetAsync(device_status, 0, sizeof(uint32_t), stream);
  if (result != hipSuccess) {
    return result;
  }
  const dim3 grid(static_cast<uint32_t>(grid_count), 1U, 1U);
  const dim3 block(SLLM_HIP_KV_WORKGROUP_SIZE, 1U, 1U);
  if (encoding == SLLM_HIP_KV_ENCODING_FP16_V1) {
    hipLaunchKernelGGL(sllm_kv_state_bf16_to_paged_f16_v1, grid, block, 0U,
                       stream, key_input, value_input, block_descriptors,
                       logical_table, table_capacity, descriptor_capacity,
                       token_count, capacity_tokens, start_position, head_count,
                       head_dim, control, device_status);
  } else if (encoding == SLLM_HIP_KV_ENCODING_FP8_V1) {
    hipLaunchKernelGGL(sllm_kv_state_bf16_to_paged_fp8_v1, grid, block, 0U,
                       stream, key_input, value_input, block_descriptors,
                       logical_table, table_capacity, descriptor_capacity,
                       token_count, capacity_tokens, start_position, head_count,
                       head_dim, control, device_status);
  } else if (encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1) {
    hipLaunchKernelGGL(sllm_kv_state_bf16_to_paged_fp8_static_v1, grid, block,
                       0U, stream, key_input, value_input, block_descriptors,
                       logical_table, table_capacity, descriptor_capacity,
                       token_count, capacity_tokens, start_position, head_count,
                       head_dim, static_key_scale, static_value_scale, control,
                       device_status);
  } else if (encoding == SLLM_HIP_KV_ENCODING_NVFP4_V1) {
    hipLaunchKernelGGL(sllm_kv_state_bf16_to_paged_nvfp4_v1, grid, block, 0U,
                       stream, key_input, value_input, block_descriptors,
                       logical_table, table_capacity, descriptor_capacity,
                       token_count, capacity_tokens, start_position, head_count,
                       head_dim, control, device_status);
  } else if (encoding == SLLM_HIP_KV_ENCODING_MXFP8_E4_V1) {
    hipLaunchKernelGGL(sllm_kv_state_bf16_to_paged_mxfp8_e4_v1, grid, block, 0U,
                       stream, key_input, value_input, block_descriptors,
                       logical_table, table_capacity, descriptor_capacity,
                       token_count, capacity_tokens, start_position, head_count,
                       head_dim, control, device_status);
  } else {
    hipLaunchKernelGGL(sllm_kv_state_bf16_to_paged_mxfp8_e5_v1, grid, block, 0U,
                       stream, key_input, value_input, block_descriptors,
                       logical_table, table_capacity, descriptor_capacity,
                       token_count, capacity_tokens, start_position, head_count,
                       head_dim, control, device_status);
  }
  return hipGetLastError();
}

} // namespace

hipError_t
launch_paged(const uint16_t *const key_input, const uint16_t *const value_input,
             const sllm_paged_kv::BlockDescriptor *const block_descriptors,
             const uint32_t *const logical_table, const uint32_t table_capacity,
             const uint32_t descriptor_capacity, const uint32_t token_count,
             const uint64_t capacity_tokens, const uint64_t start_position,
             const uint32_t head_count, const uint32_t head_dim,
             const uint32_t encoding, uint32_t *const device_status,
             const hipStream_t stream, const float static_key_scale,
             const float static_value_scale) noexcept {
  return launch_paged_impl(
      key_input, value_input, block_descriptors, logical_table, table_capacity,
      descriptor_capacity, token_count, capacity_tokens, start_position,
      head_count, head_dim, encoding, nullptr, device_status, stream,
      static_key_scale, static_value_scale);
}

hipError_t launch_paged_device(
    const uint16_t *const key_input, const uint16_t *const value_input,
    const sllm_paged_kv::BlockDescriptor *const block_descriptors,
    const uint32_t *const logical_table, const uint32_t table_capacity,
    const uint32_t descriptor_capacity, const uint32_t token_count,
    const uint64_t capacity_tokens, const uint32_t head_count,
    const uint32_t head_dim, const uint32_t encoding,
    sllm_decode_control::ControlV1 *const control,
    uint32_t *const device_status, const hipStream_t stream,
    const float static_key_scale, const float static_value_scale) noexcept {
  if (control == nullptr) {
    return hipErrorInvalidValue;
  }
  return launch_paged_impl(
      key_input, value_input, block_descriptors, logical_table, table_capacity,
      descriptor_capacity, token_count, capacity_tokens, UINT64_C(0),
      head_count, head_dim, encoding, control, device_status, stream,
      static_key_scale, static_value_scale);
}

hipError_t launch_paged_sliding_static_fp8(
    const uint16_t *const key_input, const uint16_t *const value_input,
    const sllm_paged_kv::BlockDescriptor *const block_descriptors,
    const uint32_t *const ring_table, const uint64_t *const ring_tags,
    const uint32_t ring_slot_count, const uint32_t descriptor_capacity,
    const uint32_t token_count, const uint64_t retained_start,
    const uint64_t start_position, const uint32_t head_count,
    const uint32_t head_dim, const float static_key_scale,
    const float static_value_scale, uint32_t *const device_status,
    const hipStream_t stream) noexcept {
  if (key_input == nullptr || value_input == nullptr ||
      block_descriptors == nullptr || ring_table == nullptr ||
      ring_tags == nullptr || device_status == nullptr ||
      ring_slot_count != kSlidingRingSlots || descriptor_capacity == 0U ||
      token_count == 0U || head_count == 0U || head_dim == 0U ||
      head_dim > SLLM_HIP_KV_MAX_HEAD_DIM ||
      start_position > UINT64_MAX - static_cast<uint64_t>(token_count) ||
      retained_start > start_position + token_count ||
      !std::isfinite(static_key_scale) || static_key_scale <= 0.0F ||
      !std::isfinite(static_value_scale) || static_value_scale <= 0.0F) {
    return hipErrorInvalidValue;
  }
  const uint64_t rows = static_cast<uint64_t>(token_count) * head_count;
  if (rows == 0U || rows > UINT32_MAX) {
    return hipErrorInvalidValue;
  }
  hipError_t status =
      hipMemsetAsync(device_status, 0, sizeof(uint32_t), stream);
  if (status != hipSuccess) {
    return status;
  }
  hipLaunchKernelGGL(sllm_kv_state_bf16_to_paged_fp8_static_sliding_ring_v1,
                     dim3(static_cast<uint32_t>(rows)),
                     dim3(SLLM_HIP_KV_WORKGROUP_SIZE), 0U, stream, key_input,
                     value_input, block_descriptors, ring_table, ring_tags,
                     ring_slot_count, descriptor_capacity, token_count,
                     retained_start, start_position, head_count, head_dim,
                     static_key_scale, static_value_scale, device_status);
  return hipGetLastError();
}

} // namespace sllm_kv_state_kernel
