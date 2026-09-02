#include "low_precision_block_codec.hpp"
#include "low_precision_matmul_provider.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "unknown"
#endif

namespace {

static_assert(!std::is_copy_assignable_v<sllm_lowp::PreparedProviderPlan>);

constexpr auto kMxfp8Provider =
    sllm_lowp::prepare_provider_plan(sllm_lowp::make_provider_request(
        sllm_lowp::MatmulFormat::Mxfp8E4M3W8A8, sllm_lowp::ExactTarget::Gfx1201,
        256U, 4096U, 4096U));
static_assert(kMxfp8Provider.supported());
static_assert(kMxfp8Provider.provider ==
              sllm_lowp::ProviderKind::Mxfp8Gfx1201Wmma);
static_assert(kMxfp8Provider.tile == sllm_lowp::TilePolicy::Wmma128x128x32);
static_assert(kMxfp8Provider.activation_pack ==
              sllm_lowp::ActivationPack::Mxfp8E4M3Block32);
static_assert(kMxfp8Provider.block_contract.weight_block ==
              sllm_lowp::BlockKind::Mxfp8E4M3Block32);
static_assert(kMxfp8Provider.inner_product ==
              sllm_lowp::InnerProduct::E4M3WmmaFp32);

constexpr auto kMxfp8M129Provider =
    sllm_lowp::prepare_provider_plan(sllm_lowp::make_provider_request(
        sllm_lowp::MatmulFormat::Mxfp8E4M3W8A8, sllm_lowp::ExactTarget::Gfx1201,
        129U, 9216U, 2560U));
static_assert(kMxfp8M129Provider.supported());
static_assert(kMxfp8M129Provider.provider ==
              sllm_lowp::ProviderKind::Mxfp8Gfx1201Wmma);
static_assert(kMxfp8M129Provider.tile == sllm_lowp::TilePolicy::Wmma128x64x32);
static_assert(kMxfp8M129Provider.inner_product ==
              sllm_lowp::InnerProduct::E4M3WmmaFp32);

constexpr auto kMxfp8WideNUpperProvider =
    sllm_lowp::prepare_provider_plan(sllm_lowp::make_provider_request(
        sllm_lowp::MatmulFormat::Mxfp8E4M3W8A8, sllm_lowp::ExactTarget::Gfx1201,
        128U, 32768U, 4096U));
static_assert(kMxfp8WideNUpperProvider.supported());
static_assert(kMxfp8WideNUpperProvider.provider ==
              sllm_lowp::ProviderKind::Mxfp8Gfx1201Wmma);
static_assert(kMxfp8WideNUpperProvider.tile ==
              sllm_lowp::TilePolicy::Wmma128x128x32);
constexpr auto kMxfp8WideNAboveProvider =
    sllm_lowp::prepare_provider_plan(sllm_lowp::make_provider_request(
        sllm_lowp::MatmulFormat::Mxfp8E4M3W8A8, sllm_lowp::ExactTarget::Gfx1201,
        128U, 32832U, 4096U));
static_assert(kMxfp8WideNAboveProvider.supported());
static_assert(kMxfp8WideNAboveProvider.provider ==
              sllm_lowp::ProviderKind::Mxfp8Block32);

constexpr auto kMxfp6WideNUpperProvider =
    sllm_lowp::prepare_provider_plan(sllm_lowp::make_provider_request(
        sllm_lowp::MatmulFormat::Mxfp6E3M2W6A6, sllm_lowp::ExactTarget::Gfx1201,
        17U, 32768U, 2048U));
static_assert(kMxfp6WideNUpperProvider.supported());
static_assert(kMxfp6WideNUpperProvider.provider ==
              sllm_lowp::ProviderKind::Mxfp6Gfx1201WmmaViaE4M3);
static_assert(kMxfp6WideNUpperProvider.tile ==
              sllm_lowp::TilePolicy::Wmma128x64x32);
constexpr auto kMxfp6WideNAboveProvider =
    sllm_lowp::prepare_provider_plan(sllm_lowp::make_provider_request(
        sllm_lowp::MatmulFormat::Mxfp6E3M2W6A6, sllm_lowp::ExactTarget::Gfx1201,
        17U, 32769U, 2048U));
static_assert(kMxfp6WideNAboveProvider.supported());
static_assert(kMxfp6WideNAboveProvider.provider ==
              sllm_lowp::ProviderKind::Mxfp6Block32);

constexpr auto kUnalignedProvider =
    sllm_lowp::prepare_provider_plan(sllm_lowp::make_provider_request(
        sllm_lowp::MatmulFormat::Mxfp4W4A4, sllm_lowp::ExactTarget::Gfx1201,
        17U, 33U, 33U));
static_assert(kUnalignedProvider.supported());
static_assert(kUnalignedProvider.provider ==
              sllm_lowp::ProviderKind::Mxfp4W4A4Block32);
constexpr auto kUnalignedMxfp8Provider =
    sllm_lowp::prepare_provider_plan(sllm_lowp::make_provider_request(
        sllm_lowp::MatmulFormat::Mxfp8E4M3W8A8, sllm_lowp::ExactTarget::Gfx1201,
        17U, 33U, 33U));
static_assert(!kUnalignedMxfp8Provider.supported());
static_assert(kUnalignedMxfp8Provider.rejection ==
              sllm_lowp::ProviderRejection::KNotBlockAligned);
static_assert(sllm_lowp::exact_target_from_name("gfx1030") ==
              sllm_lowp::ExactTarget::Gfx1030);
static_assert(sllm_lowp::exact_target_from_name("gfx1201") ==
              sllm_lowp::ExactTarget::Gfx1201);
static_assert(sllm_lowp::exact_target_from_name("gfx1030:xnack-") ==
              sllm_lowp::ExactTarget::Unknown);
static_assert(sllm_lowp::exact_target_from_name("gfx1201:sramecc-:xnack-") ==
              sllm_lowp::ExactTarget::Unknown);
static_assert(sllm_lowp::exact_target_from_name("gfx942:sramecc+:xnack-") ==
              sllm_lowp::ExactTarget::Gfx942SrameccOnXnackOff);

enum class ScalarFormat : uint32_t {
  E4M3Fn,
  E4M3FnuZ,
  E5M2,
  E3M2,
  E2M1,
  E8M0,
};

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::cerr << operation << ": " << hipGetErrorName(status) << " ("
            << hipGetErrorString(status) << ")\n";
  return false;
}

float signed_value(const uint32_t sign, const float magnitude) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &magnitude, sizeof(bits));
  bits |= sign;
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

float host_decode(const ScalarFormat format, const uint8_t raw) {
  if (format == ScalarFormat::E4M3Fn) {
    const uint32_t sign = static_cast<uint32_t>(raw & 0x80U) << 24U;
    const uint32_t magnitude = static_cast<uint32_t>(raw & 0x7fU);
    const uint32_t exponent = magnitude >> 3U;
    const uint32_t mantissa = magnitude & 7U;
    if (exponent == 0U) {
      return mantissa == 0U
                 ? signed_value(sign, 0.0F)
                 : signed_value(sign, static_cast<float>(mantissa) * 0x1p-9F);
    }
    if (magnitude == 0x7fU) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    const uint32_t bits = sign | ((exponent + 120U) << 23U) | (mantissa << 20U);
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  if (format == ScalarFormat::E4M3FnuZ) {
    if (raw == 0x80U) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    const uint32_t sign = static_cast<uint32_t>(raw & 0x80U) << 24U;
    const uint32_t magnitude = static_cast<uint32_t>(raw & 0x7fU);
    const uint32_t exponent = magnitude >> 3U;
    const uint32_t mantissa = magnitude & 7U;
    if (exponent == 0U) {
      return signed_value(sign, static_cast<float>(mantissa) * 0x1p-10F);
    }
    const uint32_t bits = sign | ((exponent + 119U) << 23U) | (mantissa << 20U);
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  if (format == ScalarFormat::E5M2) {
    const uint32_t sign = static_cast<uint32_t>(raw & 0x80U) << 24U;
    const uint32_t exponent = (static_cast<uint32_t>(raw) >> 2U) & 0x1fU;
    const uint32_t mantissa = static_cast<uint32_t>(raw) & 3U;
    if (exponent == 0U) {
      return mantissa == 0U
                 ? signed_value(sign, 0.0F)
                 : signed_value(sign, static_cast<float>(mantissa) * 0x1p-16F);
    }
    if (exponent == 0x1fU) {
      return mantissa == 0U
                 ? signed_value(sign, std::numeric_limits<float>::infinity())
                 : std::numeric_limits<float>::quiet_NaN();
    }
    const uint32_t bits = sign | ((exponent + 112U) << 23U) | (mantissa << 21U);
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  if (format == ScalarFormat::E3M2) {
    const uint8_t bits = raw & 0x3fU;
    const uint32_t sign = static_cast<uint32_t>(bits & 0x20U) << 26U;
    const uint32_t exponent = (static_cast<uint32_t>(bits) >> 2U) & 7U;
    const uint32_t mantissa = static_cast<uint32_t>(bits) & 3U;
    if (exponent == 0U) {
      return mantissa == 0U
                 ? signed_value(sign, 0.0F)
                 : signed_value(sign, static_cast<float>(mantissa) * 0.0625F);
    }
    const uint32_t output =
        sign | ((exponent + 124U) << 23U) | (mantissa << 21U);
    float value = 0.0F;
    std::memcpy(&value, &output, sizeof(value));
    return value;
  }
  if (format == ScalarFormat::E2M1) {
    constexpr float values[8] = {0.0F, 0.5F, 1.0F, 1.5F,
                                 2.0F, 3.0F, 4.0F, 6.0F};
    return (raw & 8U) == 0U ? values[raw & 7U] : -values[raw & 7U];
  }
  if (raw == 0xffU) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const uint32_t bits =
      raw == 0U ? UINT32_C(0x00400000) : static_cast<uint32_t>(raw) << 23U;
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint8_t host_encode(const ScalarFormat format, const float value) {
  if (format == ScalarFormat::E4M3FnuZ && std::isnan(value)) {
    return 0x80U;
  }
  if (format == ScalarFormat::E4M3Fn && std::isnan(value)) {
    return 0x7fU;
  }
  if (format == ScalarFormat::E5M2 && std::isnan(value)) {
    return 0x7fU;
  }
  const uint8_t sign = value < 0.0F || std::signbit(value)
                           ? (format == ScalarFormat::E3M2
                                  ? 0x20U
                                  : (format == ScalarFormat::E2M1 ? 8U : 0x80U))
                           : 0U;
  if (format == ScalarFormat::E2M1 && std::isnan(value)) {
    return sign;
  }
  const float magnitude = std::fabs(value);
  if (magnitude == 0.0F) {
    return format == ScalarFormat::E4M3FnuZ ? 0U : sign;
  }
  uint8_t maximum_code = 0U;
  float maximum = 0.0F;
  switch (format) {
  case ScalarFormat::E4M3Fn:
    maximum_code = 0x7eU;
    maximum = 448.0F;
    break;
  case ScalarFormat::E4M3FnuZ:
    maximum_code = 0x7fU;
    maximum = 240.0F;
    break;
  case ScalarFormat::E5M2:
    maximum_code = 0x7bU;
    maximum = 57344.0F;
    break;
  case ScalarFormat::E3M2:
    maximum_code = 0x1fU;
    maximum = 28.0F;
    break;
  case ScalarFormat::E2M1:
    maximum_code = 7U;
    maximum = 6.0F;
    break;
  case ScalarFormat::E8M0:
    return 0U;
  }
  if (!std::isfinite(magnitude) || magnitude >= maximum) {
    return static_cast<uint8_t>(sign | maximum_code);
  }
  uint8_t selected = 0U;
  float selected_error = std::numeric_limits<float>::infinity();
  for (uint32_t candidate = 0U; candidate <= maximum_code; ++candidate) {
    const float error = std::fabs(
        host_decode(format, static_cast<uint8_t>(candidate)) - magnitude);
    if (error < selected_error ||
        (error == selected_error && (candidate & 1U) == 0U &&
         (selected & 1U) != 0U)) {
      selected = static_cast<uint8_t>(candidate);
      selected_error = error;
    }
  }
  if (format == ScalarFormat::E4M3FnuZ && selected == 0U) {
    return 0U;
  }
  return static_cast<uint8_t>(sign | selected);
}

__global__ void decode_kernel(const ScalarFormat format,
                              const uint8_t *const input, float *const output,
                              const uint32_t count) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  switch (format) {
  case ScalarFormat::E4M3Fn:
    output[index] =
        sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode(input[index]);
    break;
  case ScalarFormat::E4M3FnuZ:
    output[index] =
        sllm_lowp::ScalarCodec<sllm_lowp::E4M3FnuZ>::decode(input[index]);
    break;
  case ScalarFormat::E5M2:
    output[index] =
        sllm_lowp::ScalarCodec<sllm_lowp::E5M2>::decode(input[index]);
    break;
  case ScalarFormat::E3M2:
    output[index] =
        sllm_lowp::ScalarCodec<sllm_lowp::E3M2>::decode(input[index]);
    break;
  case ScalarFormat::E2M1:
    output[index] =
        sllm_lowp::ScalarCodec<sllm_lowp::E2M1>::decode(input[index]);
    break;
  case ScalarFormat::E8M0:
    output[index] =
        sllm_lowp::ScalarCodec<sllm_lowp::E8M0>::decode(input[index]);
    break;
  }
}

__global__ void encode_kernel(const ScalarFormat format,
                              const float *const input, uint8_t *const output,
                              const uint32_t count) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  switch (format) {
  case ScalarFormat::E4M3Fn:
    output[index] =
        sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::encode(input[index]);
    break;
  case ScalarFormat::E4M3FnuZ:
    output[index] =
        sllm_lowp::ScalarCodec<sllm_lowp::E4M3FnuZ>::encode(input[index]);
    break;
  case ScalarFormat::E5M2:
    output[index] =
        sllm_lowp::ScalarCodec<sllm_lowp::E5M2>::encode(input[index]);
    break;
  case ScalarFormat::E3M2:
    output[index] =
        sllm_lowp::ScalarCodec<sllm_lowp::E3M2>::encode(input[index]);
    break;
  case ScalarFormat::E2M1:
    output[index] =
        sllm_lowp::ScalarCodec<sllm_lowp::E2M1>::encode(input[index]);
    break;
  case ScalarFormat::E8M0:
    output[index] = 0U;
    break;
  }
}

__global__ void e3m2_to_e4m3_exact_kernel(const uint8_t *const packed_input,
                                          uint8_t *const converted,
                                          float *const decoded) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= 64U * 4U) {
    return;
  }
  const uint8_t e3m2 = sllm_lowp::packed_e3m2_at(packed_input, index);
  const uint8_t e4m3 = sllm_lowp::e3m2_to_e4m3fn_exact_bits(e3m2);
  converted[index] = e4m3;
  decoded[index] =
      sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode_mx_value_plane(e4m3);
}

__global__ void e3m2x4_to_e4m3_exact_kernel(const uint8_t *const packed_input,
                                            uint32_t *const converted_groups,
                                            const uint32_t group_count) {
  const uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= group_count) {
    return;
  }
  converted_groups[group] = sllm_lowp::e3m2x4_to_e4m3fn_exact_bits(
      sllm_lowp::packed_e3m2x4_at(packed_input, group * 4U));
}

__global__ void
e3m2x4_to_e4m3_swar_exact_kernel(const uint8_t *const packed_input,
                                 uint32_t *const converted_groups,
                                 const uint32_t group_count) {
  const uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= group_count) {
    return;
  }
  converted_groups[group] = sllm_lowp::e3m2x4_to_e4m3fn_exact_bits_swar(
      sllm_lowp::packed_e3m2x4_at(packed_input, group * 4U));
}

__global__ void e3m2_to_fp16_bits_kernel(const uint8_t *const packed_input,
                                         uint16_t *const converted) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= 64U * 4U) {
    return;
  }
  converted[index] = sllm_lowp::e3m2_to_fp16_bits(
      sllm_lowp::packed_e3m2_at(packed_input, index));
}

template <typename BlockFormat>
__global__ void block_load_kernel(const uint8_t *const values,
                                  const uint8_t *const scales,
                                  const float *const outer_scales,
                                  float *const output, const uint32_t columns) {
  const uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= columns) {
    return;
  }
  const auto view = sllm_lowp::make_block_scaled_view<BlockFormat>(
      values, scales, outer_scales, columns);
  output[column] = sllm_lowp::BlockCodec<BlockFormat>::load(view, 0U, column);
}

bool compare_float(const float actual, const float expected) {
  if (std::isnan(expected)) {
    return std::isnan(actual);
  }
  uint32_t actual_bits = 0U;
  uint32_t expected_bits = 0U;
  std::memcpy(&actual_bits, &actual, sizeof(actual_bits));
  std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
  return actual_bits == expected_bits;
}

bool run_e3m2_to_e4m3_exact_conversion() {
  constexpr uint32_t code_count = 64U;
  constexpr uint32_t packed_lanes = 4U;
  constexpr uint32_t value_count = code_count * packed_lanes;
  std::vector<uint8_t> packed(value_count * 3U / 4U, 0U);
  for (uint32_t index = 0U; index < value_count; ++index) {
    const uint32_t group = index / packed_lanes;
    const uint32_t lane = index % packed_lanes;
    const uint32_t code = (group + lane * 17U) & (code_count - 1U);
    const uint32_t bit = index * 6U;
    const uint32_t byte = bit / 8U;
    const uint32_t shift = bit % 8U;
    const uint32_t shifted_code = code << shift;
    packed[byte] = static_cast<uint8_t>(packed[byte] | shifted_code);
    if (byte + 1U < packed.size()) {
      packed[byte + 1U] =
          static_cast<uint8_t>(packed[byte + 1U] | (shifted_code >> 8U));
    }
  }

  std::vector<uint8_t> converted(value_count, 0U);
  std::vector<uint32_t> converted_groups(value_count / packed_lanes, 0U);
  std::vector<uint32_t> converted_swar_groups(value_count / packed_lanes, 0U);
  std::vector<uint16_t> converted_fp16(value_count, 0U);
  std::vector<float> decoded(value_count, 0.0F);
  uint8_t *device_packed = nullptr;
  uint8_t *device_converted = nullptr;
  uint32_t *device_converted_groups = nullptr;
  uint32_t *device_converted_swar_groups = nullptr;
  uint16_t *device_converted_fp16 = nullptr;
  float *device_decoded = nullptr;
  bool ok =
      hip_ok(
          hipMalloc(reinterpret_cast<void **>(&device_packed), packed.size()),
          "hipMalloc exact packed input") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_converted),
                       converted.size()),
             "hipMalloc exact converted output") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_converted_groups),
                       converted_groups.size() * sizeof(uint32_t)),
             "hipMalloc exact packed-group converted output") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_converted_swar_groups),
                       converted_swar_groups.size() * sizeof(uint32_t)),
             "hipMalloc exact SWAR packed-group converted output") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_converted_fp16),
                       converted_fp16.size() * sizeof(uint16_t)),
             "hipMalloc exact FP16 converted output") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_decoded),
                       decoded.size() * sizeof(float)),
             "hipMalloc exact decoded output") &&
      hip_ok(hipMemcpy(device_packed, packed.data(), packed.size(),
                       hipMemcpyHostToDevice),
             "hipMemcpy exact packed input");
  if (ok) {
    hipLaunchKernelGGL(e3m2_to_e4m3_exact_kernel, dim3(1U), dim3(value_count),
                       0U, nullptr, device_packed, device_converted,
                       device_decoded);
    hipLaunchKernelGGL(e3m2x4_to_e4m3_exact_kernel, dim3(1U),
                       dim3(value_count / packed_lanes), 0U, nullptr,
                       device_packed, device_converted_groups,
                       value_count / packed_lanes);
    hipLaunchKernelGGL(e3m2x4_to_e4m3_swar_exact_kernel, dim3(1U),
                       dim3(value_count / packed_lanes), 0U, nullptr,
                       device_packed, device_converted_swar_groups,
                       value_count / packed_lanes);
    hipLaunchKernelGGL(e3m2_to_fp16_bits_kernel, dim3(1U), dim3(value_count),
                       0U, nullptr, device_packed, device_converted_fp16);
    ok =
        hip_ok(hipGetLastError(), "exact E3M2 to E4M3 kernel") &&
        hip_ok(hipMemcpy(converted.data(), device_converted, converted.size(),
                         hipMemcpyDeviceToHost),
               "hipMemcpy exact converted output") &&
        hip_ok(hipMemcpy(converted_groups.data(), device_converted_groups,
                         converted_groups.size() * sizeof(uint32_t),
                         hipMemcpyDeviceToHost),
               "hipMemcpy exact packed-group converted output") &&
        hip_ok(hipMemcpy(converted_swar_groups.data(),
                         device_converted_swar_groups,
                         converted_swar_groups.size() * sizeof(uint32_t),
                         hipMemcpyDeviceToHost),
               "hipMemcpy exact SWAR packed-group converted output") &&
        hip_ok(hipMemcpy(converted_fp16.data(), device_converted_fp16,
                         converted_fp16.size() * sizeof(uint16_t),
                         hipMemcpyDeviceToHost),
               "hipMemcpy exact FP16 converted output") &&
        hip_ok(hipMemcpy(decoded.data(), device_decoded,
                         decoded.size() * sizeof(float), hipMemcpyDeviceToHost),
               "hipMemcpy exact decoded output");
  }
  if (ok) {
    for (uint32_t index = 0U; index < value_count; ++index) {
      const uint32_t group = index / packed_lanes;
      const uint32_t lane = index % packed_lanes;
      const uint32_t code = (group + lane * 17U) & (code_count - 1U);
      const uint8_t expected_bits =
          sllm_lowp::e3m2_to_e4m3fn_exact_bits(static_cast<uint8_t>(code));
      const float expected =
          host_decode(ScalarFormat::E3M2, static_cast<uint8_t>(code));
      if (converted[index] != expected_bits ||
          converted_fp16[index] !=
              sllm_lowp::e3m2_to_fp16_bits(static_cast<uint8_t>(code)) ||
          !compare_float(decoded[index], expected)) {
        std::cerr << "exact E3M2 conversion mismatch code=" << code
                  << " lane=" << (index % packed_lanes) << '\n';
        ok = false;
        break;
      }
    }
  }
  if (ok) {
    for (uint32_t group = 0U; group < converted_groups.size(); ++group) {
      uint32_t expected = 0U;
      for (uint32_t lane = 0U; lane < packed_lanes; ++lane) {
        expected |=
            static_cast<uint32_t>(converted[group * packed_lanes + lane])
            << (lane * 8U);
      }
      if (converted_groups[group] != expected) {
        std::cerr << "packed-group E3M2 to E4M3 mismatch group=" << group
                  << '\n';
        ok = false;
        break;
      }
      if (converted_swar_groups[group] != converted_groups[group]) {
        std::cerr << "SWAR packed-group E3M2 to E4M3 mismatch group=" << group
                  << '\n';
        ok = false;
        break;
      }
    }
  }
  for (void *allocation : {static_cast<void *>(device_decoded),
                           static_cast<void *>(device_converted_fp16),
                           static_cast<void *>(device_converted_swar_groups),
                           static_cast<void *>(device_converted_groups),
                           static_cast<void *>(device_converted),
                           static_cast<void *>(device_packed)}) {
    if (allocation != nullptr) {
      ok = hip_ok(hipFree(allocation), "hipFree exact conversion allocation") &&
           ok;
    }
  }
  return ok;
}

bool run_scalar_decode(const ScalarFormat format, const uint32_t count) {
  std::vector<uint8_t> input(count);
  for (uint32_t index = 0U; index != count; ++index) {
    input[index] = static_cast<uint8_t>(index);
  }
  std::vector<float> output(count);
  uint8_t *device_input = nullptr;
  float *device_output = nullptr;
  bool ok =
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_input), input.size()),
             "hipMalloc decode input") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_output),
                       output.size() * sizeof(float)),
             "hipMalloc decode output") &&
      hip_ok(hipMemcpy(device_input, input.data(), input.size(),
                       hipMemcpyHostToDevice),
             "hipMemcpy decode input");
  if (ok) {
    hipLaunchKernelGGL(decode_kernel, dim3((count + 255U) / 256U), dim3(256U),
                       0U, nullptr, format, device_input, device_output, count);
    ok = hip_ok(hipGetLastError(), "decode kernel") &&
         hip_ok(hipMemcpy(output.data(), device_output,
                          output.size() * sizeof(float), hipMemcpyDeviceToHost),
                "hipMemcpy decode output");
  }
  if (ok) {
    for (uint32_t index = 0U; index != count; ++index) {
      if (!compare_float(output[index], host_decode(format, input[index]))) {
        std::cerr << "decode mismatch format=" << static_cast<uint32_t>(format)
                  << " code=" << index << '\n';
        ok = false;
        break;
      }
    }
  }
  if (device_output != nullptr) {
    ok = hip_ok(hipFree(device_output), "hipFree decode output") && ok;
  }
  if (device_input != nullptr) {
    ok = hip_ok(hipFree(device_input), "hipFree decode input") && ok;
  }
  return ok;
}

std::vector<float> encode_inputs(const ScalarFormat format) {
  uint8_t maximum_code = 0U;
  switch (format) {
  case ScalarFormat::E4M3Fn:
    maximum_code = 0x7eU;
    break;
  case ScalarFormat::E4M3FnuZ:
    maximum_code = 0x7fU;
    break;
  case ScalarFormat::E5M2:
    maximum_code = 0x7bU;
    break;
  case ScalarFormat::E3M2:
    maximum_code = 0x1fU;
    break;
  case ScalarFormat::E2M1:
    maximum_code = 7U;
    break;
  case ScalarFormat::E8M0:
    return {};
  }
  std::vector<float> values = {0.0F, -0.0F,
                               std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::quiet_NaN()};
  for (uint32_t code = 0U; code < maximum_code; ++code) {
    const float left = host_decode(format, static_cast<uint8_t>(code));
    const float right = host_decode(format, static_cast<uint8_t>(code + 1U));
    const float middle = left + (right - left) * 0.5F;
    values.push_back(
        std::nextafter(middle, -std::numeric_limits<float>::infinity()));
    values.push_back(middle);
    values.push_back(
        std::nextafter(middle, std::numeric_limits<float>::infinity()));
    values.push_back(-middle);
  }
  const float maximum = host_decode(format, maximum_code);
  values.push_back(maximum);
  values.push_back(
      std::nextafter(maximum, std::numeric_limits<float>::infinity()));
  values.push_back(-maximum);
  return values;
}

bool run_scalar_encode(const ScalarFormat format) {
  const std::vector<float> input = encode_inputs(format);
  std::vector<uint8_t> output(input.size());
  float *device_input = nullptr;
  uint8_t *device_output = nullptr;
  bool ok =
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_input),
                       input.size() * sizeof(float)),
             "hipMalloc encode input") &&
      hip_ok(
          hipMalloc(reinterpret_cast<void **>(&device_output), output.size()),
          "hipMalloc encode output") &&
      hip_ok(hipMemcpy(device_input, input.data(), input.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "hipMemcpy encode input");
  if (ok) {
    const uint32_t count = static_cast<uint32_t>(input.size());
    hipLaunchKernelGGL(encode_kernel, dim3((count + 255U) / 256U), dim3(256U),
                       0U, nullptr, format, device_input, device_output, count);
    ok = hip_ok(hipGetLastError(), "encode kernel") &&
         hip_ok(hipMemcpy(output.data(), device_output, output.size(),
                          hipMemcpyDeviceToHost),
                "hipMemcpy encode output");
  }
  if (ok) {
    for (std::size_t index = 0U; index != input.size(); ++index) {
      const uint8_t expected = host_encode(format, input[index]);
      if (output[index] != expected) {
        std::cerr << "encode mismatch format=" << static_cast<uint32_t>(format)
                  << " index=" << index
                  << " actual=" << static_cast<uint32_t>(output[index])
                  << " expected=" << static_cast<uint32_t>(expected) << '\n';
        ok = false;
        break;
      }
    }
  }
  if (device_output != nullptr) {
    ok = hip_ok(hipFree(device_output), "hipFree encode output") && ok;
  }
  if (device_input != nullptr) {
    ok = hip_ok(hipFree(device_input), "hipFree encode input") && ok;
  }
  return ok;
}

template <typename BlockFormat> bool run_block_load(const uint32_t columns) {
  const uint32_t blocks =
      (columns + BlockFormat::kBlockSize - 1U) / BlockFormat::kBlockSize;
  const uint32_t value_bytes =
      BlockFormat::kPacked
          ? (BlockFormat::kBitsPerElement == 4U ? (columns + 1U) / 2U
                                                : ((columns + 3U) / 4U) * 3U)
          : blocks * BlockFormat::kBlockSize;
  std::vector<uint8_t> values(value_bytes, 0U);
  std::vector<uint8_t> scales(blocks);
  std::vector<float> expected(columns);
  constexpr bool nv = BlockFormat::kHasOuterScale;
  for (uint32_t block = 0U; block != blocks; ++block) {
    if constexpr (std::is_same_v<BlockFormat, sllm_lowp::Mxfp8E4Block32>) {
      constexpr uint8_t scale_codes[] = {255U, 0U,   1U,   118U,
                                         127U, 134U, 254U, 113U};
      scales[block] = scale_codes[block % 8U];
    } else {
      scales[block] = nv ? static_cast<uint8_t>(0x38U + block)
                         : static_cast<uint8_t>(127U + block);
    }
  }
  for (uint32_t column = 0U; column != columns; ++column) {
    uint8_t code = BlockFormat::kBitsPerElement == 4U
                       ? static_cast<uint8_t>((column % 7U) + 1U)
                       : static_cast<uint8_t>((column % 30U) + 1U);
    if constexpr (std::is_same_v<BlockFormat, sllm_lowp::Mxfp8E4Block32>) {
      constexpr uint8_t value_codes[] = {0U, 128U, 1U,   7U,   129U, 135U,
                                         8U, 126U, 127U, 255U, 64U,  192U};
      code = value_codes[column % 12U];
    }
    if constexpr (!BlockFormat::kPacked) {
      values[column] = code;
    } else if constexpr (BlockFormat::kBitsPerElement == 4U) {
      values[column / 2U] |= static_cast<uint8_t>(code << ((column & 1U) * 4U));
    } else {
      const uint32_t byte = (column / 4U) * 3U;
      const uint32_t shift = (column & 3U) * 6U;
      uint32_t packed = static_cast<uint32_t>(values[byte]) |
                        (static_cast<uint32_t>(values[byte + 1U]) << 8U) |
                        (static_cast<uint32_t>(values[byte + 2U]) << 16U);
      packed |= static_cast<uint32_t>(code) << shift;
      values[byte] = static_cast<uint8_t>(packed);
      values[byte + 1U] = static_cast<uint8_t>(packed >> 8U);
      values[byte + 2U] = static_cast<uint8_t>(packed >> 16U);
    }
    ScalarFormat element = ScalarFormat::E4M3Fn;
    if constexpr (BlockFormat::kBitsPerElement == 4U) {
      element = ScalarFormat::E2M1;
    } else if constexpr (BlockFormat::kBitsPerElement == 6U) {
      element = ScalarFormat::E3M2;
    } else if constexpr (BlockFormat::kElementPower == 15) {
      element = ScalarFormat::E5M2;
    }
    const float outer = nv ? 0.75F : 1.0F;
    expected[column] =
        host_decode(element, code) *
        host_decode(nv ? ScalarFormat::E4M3Fn : ScalarFormat::E8M0,
                    scales[column / BlockFormat::kBlockSize]) *
        outer;
  }
  const float outer = 0.75F;
  uint8_t *device_values = nullptr;
  uint8_t *device_scales = nullptr;
  float *device_outer = nullptr;
  float *device_output = nullptr;
  std::vector<float> output(columns);
  bool ok =
      hip_ok(
          hipMalloc(reinterpret_cast<void **>(&device_values), values.size()),
          "hipMalloc block values") &&
      hip_ok(
          hipMalloc(reinterpret_cast<void **>(&device_scales), scales.size()),
          "hipMalloc block scales") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_outer), sizeof(float)),
             "hipMalloc block outer") &&
      hip_ok(hipMalloc(reinterpret_cast<void **>(&device_output),
                       output.size() * sizeof(float)),
             "hipMalloc block output") &&
      hip_ok(hipMemcpy(device_values, values.data(), values.size(),
                       hipMemcpyHostToDevice),
             "hipMemcpy block values") &&
      hip_ok(hipMemcpy(device_scales, scales.data(), scales.size(),
                       hipMemcpyHostToDevice),
             "hipMemcpy block scales") &&
      hip_ok(
          hipMemcpy(device_outer, &outer, sizeof(float), hipMemcpyHostToDevice),
          "hipMemcpy block outer");
  if (ok) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(block_load_kernel<BlockFormat>),
                       dim3((columns + 255U) / 256U), dim3(256U), 0U, nullptr,
                       device_values, device_scales, device_outer,
                       device_output, columns);
    ok = hip_ok(hipGetLastError(), "block load kernel") &&
         hip_ok(hipMemcpy(output.data(), device_output,
                          output.size() * sizeof(float), hipMemcpyDeviceToHost),
                "hipMemcpy block output");
  }
  if (ok) {
    for (uint32_t column = 0U; column != columns; ++column) {
      if (!compare_float(output[column], expected[column])) {
        std::cerr << "block mismatch columns=" << columns
                  << " column=" << column << '\n';
        ok = false;
        break;
      }
    }
  }
  for (void *allocation :
       {static_cast<void *>(device_output), static_cast<void *>(device_outer),
        static_cast<void *>(device_scales),
        static_cast<void *>(device_values)}) {
    if (allocation != nullptr) {
      ok = hip_ok(hipFree(allocation), "hipFree block allocation") && ok;
    }
  }
  return ok;
}

bool run_provider_contract() {
  using namespace sllm_lowp;
  const auto mxfp8 = prepare_provider_plan(make_provider_request(
      MatmulFormat::Mxfp8E4M3W8A8, ExactTarget::Gfx1030, 17U, 9216U, 2560U));
  const auto mxfp8_n128 = prepare_provider_plan(make_provider_request(
      MatmulFormat::Mxfp8E4M3W8A8, ExactTarget::Gfx1201, 128U, 9216U, 2560U));
  const auto mxfp8_m129 = prepare_provider_plan(make_provider_request(
      MatmulFormat::Mxfp8E4M3W8A8, ExactTarget::Gfx1201, 129U, 9216U, 2560U));
  const auto mxfp6_gfx1201 = prepare_provider_plan(make_provider_request(
      MatmulFormat::Mxfp6E3M2W6A6, ExactTarget::Gfx1201, 17U, 9216U, 2560U));
  const auto mxfp6_gfx1030 = prepare_provider_plan(make_provider_request(
      MatmulFormat::Mxfp6E3M2W6A6, ExactTarget::Gfx1030, 17U, 9216U, 2560U));
  const auto mxfp6_below_scope = prepare_provider_plan(make_provider_request(
      MatmulFormat::Mxfp6E3M2W6A6, ExactTarget::Gfx1201, 16U, 1024U, 2048U));
  const auto nvfp4_w4a16 = prepare_provider_plan(make_provider_request(
      MatmulFormat::Nvfp4W4A16, ExactTarget::Gfx1201, 17U, 9216U, 2560U));
  const auto nvfp4_w4a4 = prepare_provider_plan(make_provider_request(
      MatmulFormat::Nvfp4W4A4, ExactTarget::Gfx1201, 17U, 9216U, 2560U));
  ProviderRequest mxfp4_request = make_provider_request(
      MatmulFormat::Mxfp4W4A4, ExactTarget::Gfx1201, 17U, 9216U, 2560U);
  mxfp4_request.weight_layout = BlockLayout::ConsumerTiledBlockScaled;
  const auto mxfp4 = prepare_provider_plan(mxfp4_request);
  const auto cdna = prepare_provider_plan(make_provider_request(
      MatmulFormat::Mxfp8E4M3W8A8, ExactTarget::Gfx942SrameccOnXnackOff, 17U,
      9216U, 2560U));

  const FormatContract nvfp4_w4a16_contract =
      format_contract(MatmulFormat::Nvfp4W4A16);
  const FormatContract mxfp4_contract =
      format_contract(MatmulFormat::Mxfp4W4A4);
  const bool ok =
      mxfp8.supported() && mxfp8.provider == ProviderKind::Mxfp8Block32 &&
      mxfp8_n128.supported() &&
      mxfp8_n128.provider == ProviderKind::Mxfp8Gfx1201Wmma &&
      mxfp8_n128.tile == TilePolicy::Wmma128x128x32 &&
      mxfp8_n128.inner_product == InnerProduct::E4M3WmmaFp32 &&
      mxfp8_m129.supported() &&
      mxfp8_m129.provider == ProviderKind::Mxfp8Gfx1201Wmma &&
      mxfp8_m129.tile == TilePolicy::Wmma128x64x32 &&
      mxfp8_m129.inner_product == InnerProduct::E4M3WmmaFp32 &&
      mxfp6_gfx1201.supported() &&
      mxfp6_gfx1201.provider == ProviderKind::Mxfp6Gfx1201WmmaViaE4M3 &&
      mxfp6_gfx1201.tile == TilePolicy::Wmma128x64x32 &&
      mxfp6_gfx1201.inner_product == InnerProduct::E3M2ViaE4M3WmmaFp32 &&
      mxfp6_gfx1030.supported() &&
      mxfp6_gfx1030.provider == ProviderKind::Mxfp6Block32 &&
      mxfp6_below_scope.supported() &&
      mxfp6_below_scope.provider == ProviderKind::Mxfp6Block32 &&
      nvfp4_w4a16.supported() &&
      nvfp4_w4a16.provider == ProviderKind::Nvfp4W4A16Block16 &&
      nvfp4_w4a16.activation_pack == ActivationPack::NoneBf16 &&
      nvfp4_w4a16_contract.activation_element == ScalarType::Bf16 &&
      nvfp4_w4a4.supported() &&
      nvfp4_w4a4.provider == ProviderKind::Nvfp4W4A4Block16 &&
      nvfp4_w4a4.activation_pack == ActivationPack::Nvfp4E2M1Block16 &&
      mxfp4.supported() && mxfp4.provider == ProviderKind::Mxfp4W4A4Block32 &&
      mxfp4.weight_layout == BlockLayout::ConsumerTiledBlockScaled &&
      mxfp4_contract.weight_block_size == 32U &&
      mxfp4_contract.weight_scale == BlockScaleType::E8M0 &&
      !cdna.supported() &&
      cdna.rejection == ProviderRejection::UnsupportedTarget;
  if (!ok) {
    std::cerr << "low-precision provider contract mismatch\n";
  }
  return ok;
}

} // namespace

int main() {
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, 0),
              "hipGetDeviceProperties") ||
      std::string(properties.gcnArchName) != SLLM_TEST_EXPECTED_TARGET) {
    std::cerr << "exact target mismatch: observed=" << properties.gcnArchName
              << " expected=" << SLLM_TEST_EXPECTED_TARGET << '\n';
    return 1;
  }
  bool ok = true;
  ok = run_provider_contract() && ok;
  ok = run_e3m2_to_e4m3_exact_conversion() && ok;
  ok = run_scalar_decode(ScalarFormat::E4M3Fn, 256U) && ok;
  ok = run_scalar_decode(ScalarFormat::E4M3FnuZ, 256U) && ok;
  ok = run_scalar_decode(ScalarFormat::E5M2, 256U) && ok;
  ok = run_scalar_decode(ScalarFormat::E3M2, 64U) && ok;
  ok = run_scalar_decode(ScalarFormat::E2M1, 16U) && ok;
  ok = run_scalar_decode(ScalarFormat::E8M0, 256U) && ok;
  for (const ScalarFormat format :
       {ScalarFormat::E4M3Fn, ScalarFormat::E4M3FnuZ, ScalarFormat::E5M2,
        ScalarFormat::E3M2, ScalarFormat::E2M1}) {
    ok = run_scalar_encode(format) && ok;
  }
  for (const uint32_t columns : {31U, 32U, 33U, 256U}) {
    ok = run_block_load<sllm_lowp::Mxfp8E4Block32>(columns) && ok;
    ok = run_block_load<sllm_lowp::Mxfp8E5Block32>(columns) && ok;
    ok = run_block_load<sllm_lowp::Mxfp6E3Block32>(columns) && ok;
    ok = run_block_load<sllm_lowp::Mxfp4E2Block32>(columns) && ok;
  }
  for (const uint32_t columns : {15U, 16U, 17U, 256U}) {
    ok = run_block_load<sllm_lowp::Nvfp4Block16>(columns) && ok;
  }
  std::cout << "{\"schema_version\":\"sllm-phase62-lowp-codec-gpu-v1\","
               "\"state\":\""
            << (ok ? "PASS" : "FAIL") << "\",\"target\":\""
            << properties.gcnArchName
            << "\",\"decode_codes\":1104,"
               "\"exact_e3m2_to_e4m3_codes\":64,"
               "\"exact_e3m2_to_fp16_bits_codes\":64,"
               "\"exact_e3m2_packed_lanes\":4,"
               "\"encode_boundary_sets\":5,"
               "\"mx_boundaries\":[31,32,33,256],"
               "\"mxfp4_boundaries\":[31,32,33,256],"
               "\"nv_boundaries\":[15,16,17,256],"
               "\"provider_contract\":true,"
               "\"fallback_used\":false}\n";
  return ok ? 0 : 1;
}
