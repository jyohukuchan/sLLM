// Phase 78 Qwen3.8 M1 GDN decode probe.
//
// This is an evidence-only probe.  It intentionally does not alter the
// production selector or the linear-attention ABI.  The model has 48 GDN
// layers, qk_heads=16, value_heads=48, head_dim=128 and conv_kernel_size=4.
// Two FP32 recurrent-state copies therefore occupy exactly 288 MiB:
// 48 * 2 * 48 * 128 * 128 * sizeof(float).

#include "linear_attention_kernel_internal.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

extern "C" __global__ void sllm_linear_attention_recurrent_gated_norm_v1(
    const uint16_t *, const uint16_t *, const uint16_t *, const uint16_t *,
    const float *, const uint16_t *, const float *, const float *, float *,
    uint16_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" __global__ void sllm_linear_attention_column_preprocess_v2(
    uint16_t *, const uint16_t *, const uint16_t *, const float *,
    const uint16_t *, float *, float *, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t);
extern "C" __global__ void sllm_linear_attention_recurrent_column_state_v2(
    const uint16_t *, const float *, const float *, const float *, float *,
    uint16_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" __global__ void
sllm_linear_attention_column_postprocess_v2(const uint16_t *, const float *,
                                            uint16_t *, uint32_t, uint32_t,
                                            uint32_t, uint32_t);

namespace {

constexpr uint32_t kLayers = 48U;
constexpr uint32_t kQkHeads = 16U;
constexpr uint32_t kValueHeads = 48U;
constexpr uint32_t kHeadDim = 128U;
constexpr uint32_t kQkvWidth = (2U * kQkHeads + kValueHeads) * kHeadDim;
constexpr uint32_t kOutputWidth = kValueHeads * kHeadDim;
constexpr uint32_t kConvHistory = 3U;
constexpr uint32_t kConvKernel = 4U;
constexpr uint32_t kWarmups = 16U;
constexpr uint32_t kMeasured = 128U;
constexpr uint32_t kThreads = 128U;
constexpr uint64_t kStateElements =
    static_cast<uint64_t>(kValueHeads) * kHeadDim * kHeadDim;
constexpr uint64_t kStateBytesPerCopy = kStateElements * sizeof(float);
constexpr uint64_t kWorkingStateBytes =
    static_cast<uint64_t>(kLayers) * 2U * kStateBytesPerCopy;
static_assert(kQkvWidth == 10240U);
static_assert(kOutputWidth == 6144U);
static_assert(kWorkingStateBytes == UINT64_C(288) * 1024U * 1024U);

float bf16_to_float(const uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint16_t float_to_bf16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t exponent = bits & UINT32_C(0x7f800000);
  const uint32_t fraction = bits & UINT32_C(0x007fffff);
  if (exponent == UINT32_C(0x7f800000)) {
    if (fraction != 0U) {
      return static_cast<uint16_t>(((bits >> 16U) & UINT32_C(0x8000)) |
                                   UINT16_C(0x7fc0) |
                                   ((bits >> 16U) & UINT32_C(0x003f)));
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & 1U) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

uint32_t ordered_bf16(const uint16_t bits) {
  return (bits & UINT16_C(0x8000)) != 0U
             ? ~static_cast<uint32_t>(bits)
             : static_cast<uint32_t>(bits) | UINT32_C(0x8000);
}

bool hip_ok(const hipError_t status, const char *const where) {
  if (status != hipSuccess) {
    std::cerr << where << ": " << hipGetErrorString(status) << "\n";
    return false;
  }
  return true;
}

bool exact_target(const char *const observed, const std::string &target) {
  return std::string(observed) == target ||
         (target == "gfx1030" && std::string(observed).find("gfx1030") == 0U) ||
         (target == "gfx1201" && std::string(observed).find("gfx1201") == 0U);
}

struct HostModel final {
  std::vector<uint16_t> qkv;
  std::vector<uint16_t> z;
  std::vector<uint16_t> conv_weight;
  std::vector<uint16_t> previous_conv_state;
  std::vector<uint16_t> b_input;
  std::vector<uint16_t> a_input;
  std::vector<float> a_log;
  std::vector<uint16_t> dt_bias;
  std::vector<float> norm_weight;
  std::vector<float> previous_state;
};

std::size_t qkv_offset(const uint32_t layer) {
  return static_cast<std::size_t>(layer) * kQkvWidth;
}
std::size_t output_offset(const uint32_t layer) {
  return static_cast<std::size_t>(layer) * kOutputWidth;
}
std::size_t scalar_offset(const uint32_t layer) {
  return static_cast<std::size_t>(layer) * kValueHeads;
}
std::size_t state_offset(const uint32_t layer) {
  return static_cast<std::size_t>(layer) * kStateElements;
}
std::size_t conv_weight_offset(const uint32_t layer) {
  return static_cast<std::size_t>(layer) * kQkvWidth * kConvKernel;
}
std::size_t conv_state_offset(const uint32_t layer) {
  return static_cast<std::size_t>(layer) * kConvHistory * kQkvWidth;
}

HostModel make_host_model() {
  HostModel model;
  model.qkv.resize(static_cast<std::size_t>(kLayers) * kQkvWidth);
  model.z.resize(static_cast<std::size_t>(kLayers) * kOutputWidth);
  model.conv_weight.resize(static_cast<std::size_t>(kLayers) * kQkvWidth *
                           kConvKernel);
  model.previous_conv_state.resize(static_cast<std::size_t>(kLayers) *
                                   kConvHistory * kQkvWidth);
  model.b_input.resize(static_cast<std::size_t>(kLayers) * kValueHeads);
  model.a_input.resize(static_cast<std::size_t>(kLayers) * kValueHeads);
  model.a_log.resize(static_cast<std::size_t>(kLayers) * kValueHeads);
  model.dt_bias.resize(static_cast<std::size_t>(kLayers) * kValueHeads);
  model.norm_weight.resize(kHeadDim);
  model.previous_state.resize(static_cast<std::size_t>(kLayers) *
                              kStateElements);

  for (uint32_t layer = 0U; layer != kLayers; ++layer) {
    for (uint32_t index = 0U; index != kQkvWidth; ++index) {
      const float value =
          std::sin(static_cast<float>(index + 11U * layer) * 0.0097F) +
          0.125F * std::cos(static_cast<float>(index + layer) * 0.071F);
      model.qkv[qkv_offset(layer) + index] = float_to_bf16(value);
    }
    for (uint32_t index = 0U; index != kOutputWidth; ++index) {
      const float value =
          0.75F * std::cos(static_cast<float>(index + layer) * 0.017F);
      model.z[output_offset(layer) + index] = float_to_bf16(value);
    }
    for (uint32_t channel = 0U; channel != kQkvWidth; ++channel) {
      for (uint32_t tap = 0U; tap != kConvKernel; ++tap) {
        const float value =
            0.08F * std::sin(static_cast<float>(channel + tap + layer) * 0.13F);
        model.conv_weight[conv_weight_offset(layer) +
                          static_cast<std::size_t>(channel) * kConvKernel +
                          tap] = float_to_bf16(value);
      }
    }
    for (uint32_t index = 0U; index != kConvHistory * kQkvWidth; ++index) {
      model.previous_conv_state[conv_state_offset(layer) + index] =
          float_to_bf16(0.15F *
                        std::cos(static_cast<float>(index + layer) * 0.021F));
    }
    for (uint32_t value_head = 0U; value_head != kValueHeads; ++value_head) {
      // Alternating extremes exercise sigmoid and softplus saturation while
      // remaining finite and deterministic.
      const float b = (value_head & 1U) == 0U ? 18.0F : -18.0F;
      const float a = (value_head % 3U) == 0U   ? 12.0F
                      : (value_head % 3U) == 1U ? -12.0F
                                                : 0.25F;
      model.b_input[scalar_offset(layer) + value_head] = float_to_bf16(b);
      model.a_input[scalar_offset(layer) + value_head] = float_to_bf16(a);
      model.a_log[scalar_offset(layer) + value_head] =
          (value_head & 1U) == 0U ? -0.5F : -1.25F;
      model.dt_bias[scalar_offset(layer) + value_head] =
          float_to_bf16((value_head & 1U) == 0U ? 0.125F : -0.125F);
    }
    for (uint64_t index = 0U; index != kStateElements; ++index) {
      model.previous_state[state_offset(layer) + index] =
          0.03125F *
          std::sin(static_cast<float>((index + layer * 7U) % 4096U) * 0.037F);
    }
  }
  for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
    model.norm_weight[dimension] =
        0.75F + 0.25F * std::cos(static_cast<float>(dimension) * 0.031F);
  }
  return model;
}

struct DeviceArrays final {
  uint16_t *qkv = nullptr;
  uint16_t *z = nullptr;
  uint16_t *conv_weight = nullptr;
  uint16_t *previous_conv_state = nullptr;
  uint16_t *b_input = nullptr;
  uint16_t *a_input = nullptr;
  float *a_log = nullptr;
  uint16_t *dt_bias = nullptr;
  float *norm_weight = nullptr;
  uint16_t *convolved = nullptr;
  uint16_t *next_conv_state = nullptr;
  uint16_t *output = nullptr;
  float *previous_state = nullptr;
  float *next_state = nullptr;
  float *beta = nullptr;
  float *decay = nullptr;
  std::vector<uint16_t *> conv_in;
  std::vector<uint16_t *> conv_out;
  std::vector<float *> state_in;
  std::vector<float *> state_out;
};

template <typename T>
bool alloc_copy(T **const destination, const std::vector<T> &source,
                const char *const label) {
  if (!hip_ok(hipMalloc(destination, source.size() * sizeof(T)), label))
    return false;
  return hip_ok(hipMemcpy(*destination, source.data(),
                          source.size() * sizeof(T), hipMemcpyHostToDevice),
                label);
}

template <typename T>
bool alloc_zero(T **const destination, const std::size_t count,
                const char *const label) {
  if (!hip_ok(hipMalloc(destination, count * sizeof(T)), label))
    return false;
  return hip_ok(hipMemset(*destination, 0, count * sizeof(T)), label);
}

bool allocate_device(const HostModel &model, DeviceArrays *const device) {
  if (!alloc_copy(&device->qkv, model.qkv, "copy qkv") ||
      !alloc_copy(&device->z, model.z, "copy z") ||
      !alloc_copy(&device->conv_weight, model.conv_weight,
                  "copy conv weight") ||
      !alloc_copy(&device->previous_conv_state, model.previous_conv_state,
                  "copy previous conv state") ||
      !alloc_copy(&device->b_input, model.b_input, "copy b") ||
      !alloc_copy(&device->a_input, model.a_input, "copy a") ||
      !alloc_copy(&device->a_log, model.a_log, "copy a_log") ||
      !alloc_copy(&device->dt_bias, model.dt_bias, "copy dt_bias") ||
      !alloc_copy(&device->norm_weight, model.norm_weight, "copy norm") ||
      !alloc_zero(&device->convolved,
                  static_cast<std::size_t>(kLayers) * kQkvWidth,
                  "alloc convolved") ||
      !alloc_zero(&device->next_conv_state,
                  static_cast<std::size_t>(kLayers) * kConvHistory * kQkvWidth,
                  "alloc next conv state") ||
      !alloc_zero(&device->output,
                  static_cast<std::size_t>(kLayers) * kOutputWidth,
                  "alloc output") ||
      !alloc_copy(&device->previous_state, model.previous_state,
                  "copy previous state") ||
      !alloc_zero(&device->next_state,
                  static_cast<std::size_t>(kLayers) * kStateElements,
                  "alloc next state") ||
      !alloc_zero(&device->beta,
                  static_cast<std::size_t>(kLayers) * kValueHeads,
                  "alloc beta") ||
      !alloc_zero(&device->decay,
                  static_cast<std::size_t>(kLayers) * kValueHeads,
                  "alloc decay")) {
    return false;
  }
  device->conv_in.resize(kLayers);
  device->conv_out.resize(kLayers);
  device->state_in.resize(kLayers);
  device->state_out.resize(kLayers);
  for (uint32_t layer = 0U; layer != kLayers; ++layer) {
    device->conv_in[layer] =
        device->previous_conv_state +
        static_cast<std::size_t>(layer) * kConvHistory * kQkvWidth;
    device->conv_out[layer] =
        device->next_conv_state +
        static_cast<std::size_t>(layer) * kConvHistory * kQkvWidth;
    device->state_in[layer] = device->previous_state +
                              static_cast<std::size_t>(layer) * kStateElements;
    device->state_out[layer] =
        device->next_state + static_cast<std::size_t>(layer) * kStateElements;
  }
  return true;
}

template <typename T> bool free_one(T *const pointer, const char *const label) {
  return pointer == nullptr || hip_ok(hipFree(pointer), label);
}

bool free_device(DeviceArrays *const device) {
  bool ok = true;
  ok = free_one(device->qkv, "free qkv") && ok;
  ok = free_one(device->z, "free z") && ok;
  ok = free_one(device->conv_weight, "free conv weight") && ok;
  ok = free_one(device->previous_conv_state, "free previous conv state") && ok;
  ok = free_one(device->b_input, "free b") && ok;
  ok = free_one(device->a_input, "free a") && ok;
  ok = free_one(device->a_log, "free a_log") && ok;
  ok = free_one(device->dt_bias, "free dt_bias") && ok;
  ok = free_one(device->norm_weight, "free norm") && ok;
  ok = free_one(device->convolved, "free convolved") && ok;
  ok = free_one(device->next_conv_state, "free next conv state") && ok;
  ok = free_one(device->output, "free output") && ok;
  ok = free_one(device->previous_state, "free previous state") && ok;
  ok = free_one(device->next_state, "free next state") && ok;
  ok = free_one(device->beta, "free beta") && ok;
  ok = free_one(device->decay, "free decay") && ok;
  *device = DeviceArrays{};
  return ok;
}

struct LayerPointers final {
  std::vector<uint16_t *> conv_in;
  std::vector<uint16_t *> conv_out;
  std::vector<float *> state_in;
  std::vector<float *> state_out;
};

LayerPointers initial_pointers(DeviceArrays *const device) {
  return {device->conv_in, device->conv_out, device->state_in,
          device->state_out};
}

void swap_state_and_conv(LayerPointers *const pointers) {
  for (uint32_t layer = 0U; layer != kLayers; ++layer) {
    std::swap(pointers->conv_in[layer], pointers->conv_out[layer]);
    std::swap(pointers->state_in[layer], pointers->state_out[layer]);
  }
}

// Probe-local gfx1030 candidate: a 32-row state tile in LDS.  The two-pass
// state update retains the generic arithmetic order.  It is deliberately
// conservative: a tile is reloaded for the update pass because the four
// key-tiles cannot all reside in LDS.  This makes the cost/benefit measurable
// and prevents a correctness shortcut from being mistaken for an optimization.
__device__ __forceinline__ float probe_bf16_to_float(const uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16U);
}

__device__ __forceinline__ uint16_t probe_float_to_bf16(const float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t exponent = bits & UINT32_C(0x7f800000);
  const uint32_t fraction = bits & UINT32_C(0x007fffff);
  if (exponent == UINT32_C(0x7f800000)) {
    if (fraction != 0U)
      return static_cast<uint16_t>(((bits >> 16U) & UINT32_C(0x8000)) |
                                   UINT16_C(0x7fc0) |
                                   ((bits >> 16U) & UINT32_C(0x003f)));
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & 1U) != 0U))
    ++upper;
  return static_cast<uint16_t>(upper);
}

template <uint32_t Width>
__device__ __forceinline__ float probe_wave_sum(float value) {
  for (uint32_t offset = Width / 2U; offset != 0U; offset >>= 1U)
    value += __shfl_down(value, offset, Width);
  return value;
}

__device__ __forceinline__ float probe_softplus(const float value) {
  return fmaxf(value, 0.0F) + log1pf(expf(-fabsf(value)));
}

__global__ __launch_bounds__(128, 1) void phase78_gdn_row32_lds_candidate(
    const uint16_t *const convolved_qkv, const uint16_t *const z,
    const uint16_t *const b_input, const uint16_t *const a_input,
    const float *const a_log, const uint16_t *const dt_bias,
    const float *const norm_weight, const float *const previous_state,
    float *const next_state, uint16_t *const output, const uint32_t token_count,
    const uint32_t qk_heads, const uint32_t value_heads,
    const uint32_t head_dim, const uint32_t qkv_width,
    const uint32_t output_width) {
  constexpr uint32_t kTileRows = 32U;
  const uint32_t value_head = blockIdx.x;
  const uint32_t dimension = threadIdx.x;
  if (value_head >= value_heads || dimension >= head_dim || head_dim != 128U)
    return;
  const uint32_t qk_head = value_head / (value_heads / qk_heads);
  const uint64_t state_base =
      static_cast<uint64_t>(value_head) * head_dim * head_dim;
  const uint32_t lane = dimension & 31U;
  const uint32_t wave = dimension >> 5U;
  __shared__ float q_values[128];
  __shared__ float k_values[128];
  __shared__ float q_wave_sums[4];
  __shared__ float k_wave_sums[4];
  __shared__ float q_inverse_norm;
  __shared__ float k_inverse_norm;
  __shared__ float output_values[128];
  __shared__ float output_wave_sums[4];
  __shared__ float output_inverse_rms;
  __shared__ float state_tile[kTileRows][128];

  for (uint32_t token = 0U; token != token_count; ++token) {
    const uint64_t qkv_row = static_cast<uint64_t>(token) * qkv_width;
    q_values[dimension] = probe_bf16_to_float(
        convolved_qkv[qkv_row + static_cast<uint64_t>(qk_head) * head_dim +
                      dimension]);
    k_values[dimension] = probe_bf16_to_float(
        convolved_qkv[qkv_row + static_cast<uint64_t>(qk_heads) * head_dim +
                      static_cast<uint64_t>(qk_head) * head_dim + dimension]);
    __syncthreads();
    const float q_wave =
        probe_wave_sum<32U>(q_values[dimension] * q_values[dimension]);
    const float k_wave =
        probe_wave_sum<32U>(k_values[dimension] * k_values[dimension]);
    if (lane == 0U) {
      q_wave_sums[wave] = q_wave;
      k_wave_sums[wave] = k_wave;
    }
    __syncthreads();
    if (dimension == 0U) {
      float q_sum = 0.0F;
      float k_sum = 0.0F;
      for (uint32_t index = 0U; index != 4U; ++index) {
        q_sum += q_wave_sums[index];
        k_sum += k_wave_sums[index];
      }
      q_inverse_norm = 1.0F / sqrtf(q_sum + 1.0e-6F);
      k_inverse_norm = 1.0F / sqrtf(k_sum + 1.0e-6F);
    }
    __syncthreads();
    q_values[dimension] = probe_bf16_to_float(
        probe_float_to_bf16(q_values[dimension] * q_inverse_norm));
    q_values[dimension] *= 1.0F / sqrtf(static_cast<float>(head_dim));
    k_values[dimension] = probe_bf16_to_float(
        probe_float_to_bf16(k_values[dimension] * k_inverse_norm));
    __syncthreads();

    const uint64_t scalar_index =
        static_cast<uint64_t>(token) * value_heads + value_head;
    const float b = probe_bf16_to_float(b_input[scalar_index]);
    const float beta =
        probe_bf16_to_float(probe_float_to_bf16(1.0F / (1.0F + expf(-b))));
    const float a = probe_bf16_to_float(a_input[scalar_index]) +
                    probe_bf16_to_float(dt_bias[value_head]);
    const float decay = expf(-expf(a_log[value_head]) * probe_softplus(a));
    float previous_projection = 0.0F;

    // First pass: load and decay the four 32-row tiles, accumulating the
    // previous projection in the same key-dimension order as the generic
    // kernel. gfx1030's state layout is [key][dimension].
    for (uint32_t tile = 0U; tile != head_dim; tile += kTileRows) {
      for (uint32_t row = 0U; row != kTileRows; ++row) {
        const uint32_t key = tile + row;
        const uint64_t index =
            state_base + static_cast<uint64_t>(key) * head_dim + dimension;
        state_tile[row][dimension] =
            (token == 0U ? previous_state[index] : next_state[index]) * decay;
      }
      __syncthreads();
      for (uint32_t row = 0U; row != kTileRows; ++row)
        previous_projection +=
            state_tile[row][dimension] * k_values[tile + row];
      for (uint32_t row = 0U; row != kTileRows; ++row) {
        const uint32_t key = tile + row;
        next_state[state_base + static_cast<uint64_t>(key) * head_dim +
                   dimension] = state_tile[row][dimension];
      }
      __syncthreads();
    }

    const float value = probe_bf16_to_float(
        convolved_qkv[qkv_row +
                      static_cast<uint64_t>(2U * qk_heads + value_head) *
                          head_dim +
                      dimension]);
    const float delta = beta * (value - previous_projection);
    float current_projection = 0.0F;
    for (uint32_t tile = 0U; tile != head_dim; tile += kTileRows) {
      for (uint32_t row = 0U; row != kTileRows; ++row) {
        const uint32_t key = tile + row;
        const uint64_t index =
            state_base + static_cast<uint64_t>(key) * head_dim + dimension;
        state_tile[row][dimension] = next_state[index] + delta * k_values[key];
      }
      __syncthreads();
      for (uint32_t row = 0U; row != kTileRows; ++row) {
        const uint32_t key = tile + row;
        const float updated = state_tile[row][dimension];
        next_state[state_base + static_cast<uint64_t>(key) * head_dim +
                   dimension] = updated;
        current_projection += updated * q_values[key];
      }
      __syncthreads();
    }
    output_values[dimension] =
        probe_bf16_to_float(probe_float_to_bf16(current_projection));
    const float output_wave = probe_wave_sum<32U>(output_values[dimension] *
                                                  output_values[dimension]);
    if (lane == 0U)
      output_wave_sums[wave] = output_wave;
    __syncthreads();
    if (dimension == 0U) {
      float sum = 0.0F;
      for (uint32_t index = 0U; index != 4U; ++index)
        sum += output_wave_sums[index];
      output_inverse_rms =
          1.0F / sqrtf(sum / static_cast<float>(head_dim) + 1.0e-6F);
    }
    __syncthreads();
    const uint64_t output_index = static_cast<uint64_t>(token) * output_width +
                                  static_cast<uint64_t>(value_head) * head_dim +
                                  dimension;
    const float z_value = probe_bf16_to_float(z[output_index]);
    const float z_silu = z_value / (1.0F + expf(-z_value));
    const float normalized = probe_bf16_to_float(
        probe_float_to_bf16(output_values[dimension] * output_inverse_rms));
    output[output_index] =
        probe_float_to_bf16(normalized * norm_weight[dimension] * z_silu);
    __syncthreads();
  }
}

enum class Variant : uint32_t { Baseline, Candidate };

struct StageTimes final {
  double conv_ms = 0.0;
  double preprocess_ms = 0.0;
  double recurrent_ms = 0.0;
  double postprocess_ms = 0.0;
  double total_ms = 0.0;
};

struct EventPair final {
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
};

bool create_event_pair(EventPair *const pair) {
  return hip_ok(hipEventCreate(&pair->start), "event create start") &&
         hip_ok(hipEventCreate(&pair->stop), "event create stop");
}

void destroy_event_pair(EventPair *const pair) {
  if (pair->start != nullptr)
    static_cast<void>(hipEventDestroy(pair->start));
  if (pair->stop != nullptr)
    static_cast<void>(hipEventDestroy(pair->stop));
  *pair = EventPair{};
}

bool launch_conv_stage(const HostModel &, DeviceArrays *const device,
                       LayerPointers *const pointers) {
  for (uint32_t layer = 0U; layer != kLayers; ++layer) {
    const hipError_t status = sllm_linear_attention_kernel::launch_convolution(
        device->qkv + qkv_offset(layer),
        device->conv_weight + conv_weight_offset(layer),
        pointers->conv_in[layer], device->convolved + qkv_offset(layer),
        pointers->conv_out[layer], 1U, kQkvWidth, kConvKernel, nullptr);
    if (!hip_ok(status, "launch convolution"))
      return false;
  }
  return true;
}

bool launch_recurrent_stage(const std::string &target, const Variant variant,
                            DeviceArrays *const device,
                            LayerPointers *const pointers) {
  for (uint32_t layer = 0U; layer != kLayers; ++layer) {
    const std::size_t qkv = qkv_offset(layer);
    const std::size_t output = output_offset(layer);
    const std::size_t scalar = scalar_offset(layer);
    hipError_t status = hipSuccess;
    if (variant == Variant::Baseline) {
      status = sllm_linear_attention_kernel::launch_recurrent(
          device->convolved + qkv, device->z + output, device->b_input + scalar,
          device->a_input + scalar, device->a_log + scalar,
          device->dt_bias + scalar, device->norm_weight,
          pointers->state_in[layer], pointers->state_out[layer],
          device->output + output, 1U, kQkHeads, kValueHeads, kHeadDim,
          kQkvWidth, kOutputWidth, nullptr);
    } else if (target == "gfx1201") {
      status = sllm_linear_attention_kernel::launch_column_recurrent(
          device->convolved + qkv, device->beta + scalar,
          device->decay + scalar, pointers->state_in[layer],
          pointers->state_out[layer], device->output + output, 1U, kQkHeads,
          kValueHeads, kHeadDim, kQkvWidth, kOutputWidth, nullptr);
    } else {
      hipLaunchKernelGGL(phase78_gdn_row32_lds_candidate, dim3(kValueHeads),
                         dim3(kThreads), 0U, nullptr, device->convolved + qkv,
                         device->z + output, device->b_input + scalar,
                         device->a_input + scalar, device->a_log + scalar,
                         device->dt_bias + scalar, device->norm_weight,
                         pointers->state_in[layer], pointers->state_out[layer],
                         device->output + output, 1U, kQkHeads, kValueHeads,
                         kHeadDim, kQkvWidth, kOutputWidth);
      status = hipGetLastError();
    }
    if (!hip_ok(status, "launch recurrent"))
      return false;
  }
  return true;
}

bool launch_preprocess_stage(DeviceArrays *const device) {
  for (uint32_t layer = 0U; layer != kLayers; ++layer) {
    const std::size_t qkv = qkv_offset(layer);
    const std::size_t scalar = scalar_offset(layer);
    if (!hip_ok(sllm_linear_attention_kernel::launch_column_preprocess(
                    device->convolved + qkv, device->b_input + scalar,
                    device->a_input + scalar, device->a_log + scalar,
                    device->dt_bias + scalar, device->beta + scalar,
                    device->decay + scalar, 1U, kQkHeads, kValueHeads, kHeadDim,
                    kQkvWidth, nullptr),
                "launch column preprocess"))
      return false;
  }
  return true;
}

bool launch_postprocess_stage(DeviceArrays *const device) {
  for (uint32_t layer = 0U; layer != kLayers; ++layer) {
    const std::size_t output = output_offset(layer);
    if (!hip_ok(sllm_linear_attention_kernel::launch_column_postprocess(
                    device->z + output, device->norm_weight,
                    device->output + output, 1U, kValueHeads, kHeadDim,
                    kOutputWidth, nullptr),
                "launch column postprocess"))
      return false;
  }
  return true;
}

bool record_start(const EventPair &pair) {
  return hip_ok(hipEventRecord(pair.start, nullptr), "event record start");
}
bool record_stop(const EventPair &pair) {
  return hip_ok(hipEventRecord(pair.stop, nullptr), "event record stop");
}
double elapsed_ms(const EventPair &pair) {
  float value = 0.0F;
  if (!hip_ok(hipEventElapsedTime(&value, pair.start, pair.stop),
              "event elapsed"))
    return std::numeric_limits<double>::quiet_NaN();
  return static_cast<double>(value);
}

bool run_one_sweep(const std::string &target, const Variant variant,
                   DeviceArrays *const device, LayerPointers *const pointers,
                   StageTimes *const stage_times, const bool timed) {
  EventPair conv{}, preprocess{}, recurrent{}, postprocess{};
  if (timed &&
      (!create_event_pair(&conv) || !create_event_pair(&preprocess) ||
       !create_event_pair(&recurrent) || !create_event_pair(&postprocess))) {
    destroy_event_pair(&conv);
    destroy_event_pair(&preprocess);
    destroy_event_pair(&recurrent);
    destroy_event_pair(&postprocess);
    return false;
  }
  bool ok = true;
  if (timed)
    ok = record_start(conv);
  ok = launch_conv_stage(HostModel{}, device, pointers) && ok;
  if (timed)
    ok = record_stop(conv) && ok;

  if (variant == Variant::Candidate && target == "gfx1201") {
    if (timed)
      ok = record_start(preprocess) && ok;
    ok = launch_preprocess_stage(device) && ok;
    if (timed)
      ok = record_stop(preprocess) && ok;
  }
  if (timed)
    ok = record_start(recurrent) && ok;
  ok = launch_recurrent_stage(target, variant, device, pointers) && ok;
  if (timed)
    ok = record_stop(recurrent) && ok;
  if (variant == Variant::Candidate && target == "gfx1201") {
    if (timed)
      ok = record_start(postprocess) && ok;
    ok = launch_postprocess_stage(device) && ok;
    if (timed)
      ok = record_stop(postprocess) && ok;
  }
  if (timed) {
    // gfx1030 can report HIP_ERROR_NOT_READY for an earlier event even after
    // the final event has been recorded on the same default stream.  A device
    // sync is equivalent for this standalone timing lane and makes the event
    // evidence fail-closed on both targets.
    ok = hip_ok(hipDeviceSynchronize(), "sweep device synchronize") && ok;
    stage_times->conv_ms += elapsed_ms(conv);
    stage_times->recurrent_ms += elapsed_ms(recurrent);
    if (variant == Variant::Candidate && target == "gfx1201") {
      stage_times->preprocess_ms += elapsed_ms(preprocess);
      stage_times->postprocess_ms += elapsed_ms(postprocess);
    }
    stage_times->total_ms += stage_times->conv_ms + stage_times->preprocess_ms +
                             stage_times->recurrent_ms +
                             stage_times->postprocess_ms;
    destroy_event_pair(&conv);
    destroy_event_pair(&preprocess);
    destroy_event_pair(&recurrent);
    destroy_event_pair(&postprocess);
  }
  swap_state_and_conv(pointers);
  return ok;
}

struct HostSnapshot final {
  std::vector<uint16_t> conv_state;
  std::vector<uint16_t> output;
  std::vector<float> state;
};

bool copy_snapshot(const DeviceArrays &device, HostSnapshot *const snapshot) {
  snapshot->conv_state.resize(static_cast<std::size_t>(kLayers) * kConvHistory *
                              kQkvWidth);
  snapshot->output.resize(static_cast<std::size_t>(kLayers) * kOutputWidth);
  snapshot->state.resize(static_cast<std::size_t>(kLayers) * kStateElements);
  return hip_ok(hipMemcpy(snapshot->conv_state.data(), device.next_conv_state,
                          snapshot->conv_state.size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "copy conv snapshot") &&
         hip_ok(hipMemcpy(snapshot->output.data(), device.output,
                          snapshot->output.size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "copy output snapshot") &&
         hip_ok(hipMemcpy(snapshot->state.data(), device.next_state,
                          snapshot->state.size() * sizeof(float),
                          hipMemcpyDeviceToHost),
                "copy state snapshot");
}

struct OracleSnapshot final {
  std::vector<uint16_t> conv_state;
  std::vector<uint16_t> output;
  std::vector<float> state;
};

float host_softplus(const float value) {
  return std::max(value, 0.0F) + std::log1p(std::exp(-std::abs(value)));
}

uint64_t state_index(const std::string &target, const uint64_t base,
                     const uint32_t dimension, const uint32_t key) {
  return target == "gfx1030"
             ? base + static_cast<uint64_t>(key) * kHeadDim + dimension
             : base + static_cast<uint64_t>(dimension) * kHeadDim + key;
}

OracleSnapshot host_oracle(const HostModel &model, const std::string &target) {
  OracleSnapshot oracle;
  oracle.conv_state.resize(static_cast<std::size_t>(kLayers) * kConvHistory *
                           kQkvWidth);
  oracle.output.resize(static_cast<std::size_t>(kLayers) * kOutputWidth);
  oracle.state = model.previous_state;
  for (uint32_t layer = 0U; layer != kLayers; ++layer) {
    const std::size_t qi = qkv_offset(layer);
    const std::size_t oi = output_offset(layer);
    const std::size_t si = scalar_offset(layer);
    const std::size_t ci = conv_weight_offset(layer);
    const std::size_t pi = conv_state_offset(layer);
    std::vector<uint16_t> convolved(kQkvWidth);
    for (uint32_t channel = 0U; channel != kQkvWidth; ++channel) {
      float sum = 0.0F;
      for (uint32_t tap = 0U; tap != kConvKernel; ++tap) {
        const int32_t source = static_cast<int32_t>(tap) - 3;
        const uint16_t value =
            source < 0
                ? model.previous_conv_state
                      [pi + static_cast<std::size_t>(source + 3) * kQkvWidth +
                       channel]
                : model.qkv[qi + static_cast<std::size_t>(source) * kQkvWidth +
                            channel];
        sum +=
            bf16_to_float(value) *
            bf16_to_float(model.conv_weight[ci +
                                            static_cast<std::size_t>(channel) *
                                                kConvKernel +
                                            tap]);
      }
      convolved[channel] = float_to_bf16(sum / (1.0F + std::exp(-sum)));
    }
    for (uint32_t row = 0U; row != kConvHistory; ++row) {
      const int32_t source = static_cast<int32_t>(row) - 2;
      for (uint32_t channel = 0U; channel != kQkvWidth; ++channel) {
        oracle.conv_state[pi + static_cast<std::size_t>(row) * kQkvWidth +
                          channel] =
            source < 0
                ? model.previous_conv_state
                      [pi + static_cast<std::size_t>(source + 3) * kQkvWidth +
                       channel]
                : model.qkv[qi + static_cast<std::size_t>(source) * kQkvWidth +
                            channel];
      }
    }
    std::vector<float> q(kHeadDim), k(kHeadDim);
    std::vector<float> qnorm(kHeadDim), knorm(kHeadDim);
    for (uint32_t value_head = 0U; value_head != kValueHeads; ++value_head) {
      const uint32_t qk_head = value_head / 3U;
      float qsum = 0.0F;
      float ksum = 0.0F;
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        q[dimension] = bf16_to_float(convolved[qk_head * kHeadDim + dimension]);
        k[dimension] = bf16_to_float(
            convolved[(kQkHeads + qk_head) * kHeadDim + dimension]);
        qsum += q[dimension] * q[dimension];
        ksum += k[dimension] * k[dimension];
      }
      const float qi_norm = 1.0F / std::sqrt(qsum + 1.0e-6F);
      const float ki_norm = 1.0F / std::sqrt(ksum + 1.0e-6F);
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        qnorm[dimension] =
            bf16_to_float(float_to_bf16(q[dimension] * qi_norm)) /
            std::sqrt(static_cast<float>(kHeadDim));
        knorm[dimension] = bf16_to_float(float_to_bf16(k[dimension] * ki_norm));
      }
      const float b = bf16_to_float(model.b_input[si + value_head]);
      const float beta =
          bf16_to_float(float_to_bf16(1.0F / (1.0F + std::exp(-b))));
      const float av = bf16_to_float(model.a_input[si + value_head]) +
                       bf16_to_float(model.dt_bias[si + value_head]);
      const float decay =
          std::exp(-std::exp(model.a_log[si + value_head]) * host_softplus(av));
      const uint64_t state_base =
          state_offset(layer) +
          static_cast<uint64_t>(value_head) * kHeadDim * kHeadDim;
      std::vector<float> projected(kHeadDim);
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        float previous_projection = 0.0F;
        for (uint32_t key = 0U; key != kHeadDim; ++key) {
          const uint64_t index =
              state_index(target, state_base, dimension, key);
          oracle.state[index] *= decay;
          previous_projection += oracle.state[index] * knorm[key];
        }
        const float value = bf16_to_float(
            convolved[(2U * kQkHeads + value_head) * kHeadDim + dimension]);
        const float delta = beta * (value - previous_projection);
        float current_projection = 0.0F;
        for (uint32_t key = 0U; key != kHeadDim; ++key) {
          const uint64_t index =
              state_index(target, state_base, dimension, key);
          oracle.state[index] += delta * knorm[key];
          current_projection += oracle.state[index] * qnorm[key];
        }
        projected[dimension] = bf16_to_float(float_to_bf16(current_projection));
      }
      float rms_sum = 0.0F;
      for (const float value : projected)
        rms_sum += value * value;
      const float inverse_rms =
          1.0F / std::sqrt(rms_sum / static_cast<float>(kHeadDim) + 1.0e-6F);
      for (uint32_t dimension = 0U; dimension != kHeadDim; ++dimension) {
        const std::size_t output_index = oi + value_head * kHeadDim + dimension;
        const float z = bf16_to_float(model.z[output_index]);
        const float z_silu = z / (1.0F + std::exp(-z));
        const float normalized =
            bf16_to_float(float_to_bf16(projected[dimension] * inverse_rms));
        oracle.output[output_index] =
            float_to_bf16(normalized * model.norm_weight[dimension] * z_silu);
      }
    }
  }
  return oracle;
}

struct CompareResult final {
  std::size_t bf16_mismatches = 0U;
  uint32_t max_ulp = 0U;
  float max_state_abs = 0.0F;
  float max_state_rel = 0.0F;
  std::size_t nonfinite = 0U;
};

CompareResult compare_snapshot(const HostSnapshot &actual,
                               const OracleSnapshot &expected) {
  CompareResult result;
  for (std::size_t index = 0U; index != actual.output.size(); ++index) {
    result.bf16_mismatches += actual.output[index] != expected.output[index];
    const uint32_t lhs = ordered_bf16(actual.output[index]);
    const uint32_t rhs = ordered_bf16(expected.output[index]);
    result.max_ulp =
        std::max(result.max_ulp, lhs > rhs ? lhs - rhs : rhs - lhs);
  }
  for (std::size_t index = 0U; index != actual.state.size(); ++index) {
    const float observed = actual.state[index];
    const float expected_value = expected.state[index];
    result.max_state_abs =
        std::max(result.max_state_abs, std::abs(observed - expected_value));
    result.max_state_rel = std::max(
        result.max_state_rel, std::abs(observed - expected_value) /
                                  std::max(1.0e-6F, std::abs(expected_value)));
    result.nonfinite += static_cast<std::size_t>(
        !std::isfinite(observed) || !std::isfinite(expected_value));
  }
  for (const uint16_t value : actual.conv_state)
    result.nonfinite += static_cast<std::size_t>((value & UINT16_C(0x7f80)) ==
                                                 UINT16_C(0x7f80));
  for (const uint16_t value : actual.output)
    result.nonfinite += static_cast<std::size_t>((value & UINT16_C(0x7f80)) ==
                                                 UINT16_C(0x7f80));
  return result;
}

bool compare_variants(const HostSnapshot &lhs, const HostSnapshot &rhs,
                      const char *const label) {
  std::size_t output_mismatches = 0U;
  uint32_t output_max_ulp = 0U;
  float state_max_abs = 0.0F;
  for (std::size_t index = 0U; index != lhs.output.size(); ++index) {
    output_mismatches += lhs.output[index] != rhs.output[index];
    const uint32_t left = ordered_bf16(lhs.output[index]);
    const uint32_t right = ordered_bf16(rhs.output[index]);
    output_max_ulp =
        std::max(output_max_ulp, left > right ? left - right : right - left);
  }
  for (std::size_t index = 0U; index != lhs.state.size(); ++index)
    state_max_abs =
        std::max(state_max_abs, std::abs(lhs.state[index] - rhs.state[index]));
  const bool pass = output_max_ulp <= 16U &&
                    output_mismatches <= lhs.output.size() / 100U &&
                    state_max_abs <= 1.0e-4F;
  std::cout << "variant_compare label=" << label
            << " output_bf16_mismatches=" << output_mismatches
            << " output_max_bf16_ulp=" << output_max_ulp
            << " state_max_abs=" << std::setprecision(9) << state_max_abs
            << " status=" << (pass ? "PASS" : "FAIL") << "\n";
  return pass;
}

bool reset_device(const HostModel &model, DeviceArrays *const device) {
  return hip_ok(hipMemcpy(device->previous_conv_state,
                          model.previous_conv_state.data(),
                          model.previous_conv_state.size() * sizeof(uint16_t),
                          hipMemcpyHostToDevice),
                "reset conv state") &&
         hip_ok(hipMemcpy(device->previous_state, model.previous_state.data(),
                          model.previous_state.size() * sizeof(float),
                          hipMemcpyHostToDevice),
                "reset recurrent state") &&
         hip_ok(hipMemset(device->next_conv_state, 0,
                          static_cast<std::size_t>(kLayers) * kConvHistory *
                              kQkvWidth * sizeof(uint16_t)),
                "clear next conv state") &&
         hip_ok(hipMemset(device->next_state, 0,
                          static_cast<std::size_t>(kLayers) * kStateElements *
                              sizeof(float)),
                "clear next recurrent state") &&
         hip_ok(hipMemset(device->output, 0,
                          static_cast<std::size_t>(kLayers) * kOutputWidth *
                              sizeof(uint16_t)),
                "clear output");
}

bool run_probe(const std::string &target, const Variant variant,
               const HostModel &model, DeviceArrays *const device,
               StageTimes *const stage_times, HostSnapshot *const snapshot,
               bool *const deterministic) {
  if (!reset_device(model, device))
    return false;
  LayerPointers pointers = initial_pointers(device);
  // The first sweep is used for numerical evidence, then state pointers are
  // reset before the timed warmup/measured sweep sequence.
  if (!run_one_sweep(target, variant, device, &pointers, stage_times, false) ||
      !hip_ok(hipDeviceSynchronize(), "oracle sweep synchronize") ||
      !copy_snapshot(*device, snapshot))
    return false;
  HostSnapshot repeat_snapshot;
  if (!reset_device(model, device))
    return false;
  pointers = initial_pointers(device);
  if (!run_one_sweep(target, variant, device, &pointers, stage_times, false) ||
      !hip_ok(hipDeviceSynchronize(), "repeat sweep synchronize") ||
      !copy_snapshot(*device, &repeat_snapshot))
    return false;
  *deterministic = snapshot->output == repeat_snapshot.output &&
                   snapshot->conv_state == repeat_snapshot.conv_state &&
                   snapshot->state == repeat_snapshot.state;
  std::cout << "determinism variant="
            << (variant == Variant::Baseline ? "baseline" : "candidate")
            << " status=" << (*deterministic ? "PASS" : "FAIL") << "\n";
  if (!reset_device(model, device))
    return false;
  pointers = initial_pointers(device);
  for (uint32_t warmup = 0U; warmup != kWarmups; ++warmup) {
    if (!run_one_sweep(target, variant, device, &pointers, stage_times, false))
      return false;
  }
  for (uint32_t measured = 0U; measured != kMeasured; ++measured) {
    if (!run_one_sweep(target, variant, device, &pointers, stage_times, true))
      return false;
  }
  return hip_ok(hipDeviceSynchronize(), "measured synchronize");
}

struct Resource final {
  std::string name;
  int registers = -1;
  std::size_t lds = 0U;
  std::size_t scratch = 0U;
  int active_blocks = 0;
  bool available = false;
};

Resource resource(const char *const name, const void *const function) {
  Resource result;
  result.name = name;
  hipFuncAttributes attributes{};
  const hipError_t attr = hipFuncGetAttributes(&attributes, function);
  int active = 0;
  const hipError_t occupancy = hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, function, kThreads, 0U);
  result.available = attr == hipSuccess && occupancy == hipSuccess;
  if (result.available) {
    result.registers = attributes.numRegs;
    result.lds = attributes.sharedSizeBytes;
    result.scratch = attributes.localSizeBytes;
    result.active_blocks = active;
  }
  std::cout << "resources kernel=" << result.name
            << " registers=" << result.registers << " lds=" << result.lds
            << " scratch=" << result.scratch
            << " active_blocks=" << result.active_blocks
            << " status=" << (result.available ? "PASS" : "FAIL") << "\n";
  return result;
}

void print_times(const char *const name, const StageTimes &times) {
  const double denominator = static_cast<double>(kMeasured);
  const double total = (times.conv_ms + times.preprocess_ms +
                        times.recurrent_ms + times.postprocess_ms) /
                       denominator;
  std::cout << "timing variant=" << name << " sweeps=" << kMeasured
            << " layers=" << kLayers << " warmups=" << kWarmups
            << " conv_ms=" << times.conv_ms / denominator
            << " preprocess_ms=" << times.preprocess_ms / denominator
            << " recurrent_ms=" << times.recurrent_ms / denominator
            << " postprocess_ms=" << times.postprocess_ms / denominator
            << " total_ms=" << total
            << " decode_tokens_per_s=" << (total > 0.0 ? 1000.0 / total : 0.0)
            << "\n";
}

} // namespace

int main(int argc, char **argv) {
  int device_index = 0;
  std::string requested_target;
  if (argc > 3 || argc < 2) {
    std::cerr << "usage: phase78_gdn_qwen38_decode_probe <gfx1030|gfx1201> "
                 "[device]\n";
    return EXIT_FAILURE;
  }
  requested_target = argv[1];
  if (argc == 3)
    device_index = std::atoi(argv[2]);
  if (requested_target != "gfx1030" && requested_target != "gfx1201") {
    std::cerr << "unsupported target\n";
    return EXIT_FAILURE;
  }
  if (!hip_ok(hipSetDevice(device_index), "set device"))
    return EXIT_FAILURE;
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, device_index),
              "get device properties"))
    return EXIT_FAILURE;
  std::cout << "identity device=" << device_index << " name=" << properties.name
            << " arch=" << properties.gcnArchName << " qk_heads=" << kQkHeads
            << " value_heads=" << kValueHeads << " head_dim=" << kHeadDim
            << " conv_kernel=" << kConvKernel << " layers=" << kLayers
            << " state_working_set_bytes=" << kWorkingStateBytes
            << " state_working_set_mib="
            << static_cast<double>(kWorkingStateBytes) / (1024.0 * 1024.0)
            << " b_a_extreme=1 nonzero_state=1\n";
  if (!exact_target(properties.gcnArchName, requested_target)) {
    std::cerr << "wrong target: expected " << requested_target << " observed "
              << properties.gcnArchName << "\n";
    return EXIT_FAILURE;
  }

  const HostModel model = make_host_model();
  const OracleSnapshot oracle = host_oracle(model, requested_target);
  DeviceArrays baseline{}, candidate{};
  bool ok =
      allocate_device(model, &baseline) && allocate_device(model, &candidate);
  if (!ok) {
    static_cast<void>(free_device(&baseline));
    static_cast<void>(free_device(&candidate));
    return EXIT_FAILURE;
  }

  Resource baseline_resource =
      resource("sllm_linear_attention_recurrent_gated_norm_v1",
               reinterpret_cast<const void *>(
                   sllm_linear_attention_recurrent_gated_norm_v1));
  Resource candidate_resource{};
  std::vector<Resource> candidate_resources;
  if (requested_target == "gfx1201") {
    candidate_resources.push_back(
        resource("sllm_linear_attention_column_preprocess_v2",
                 reinterpret_cast<const void *>(
                     sllm_linear_attention_column_preprocess_v2)));
    candidate_resources.push_back(
        resource("sllm_linear_attention_recurrent_column_state_v2",
                 reinterpret_cast<const void *>(
                     sllm_linear_attention_recurrent_column_state_v2)));
    candidate_resources.push_back(
        resource("sllm_linear_attention_column_postprocess_v2",
                 reinterpret_cast<const void *>(
                     sllm_linear_attention_column_postprocess_v2)));
  } else {
    candidate_resource = resource(
        "phase78_gdn_row32_lds_candidate",
        reinterpret_cast<const void *>(phase78_gdn_row32_lds_candidate));
    candidate_resources.push_back(candidate_resource);
  }
  const bool resources_ok =
      baseline_resource.available && baseline_resource.scratch == 0U &&
      baseline_resource.active_blocks > 0 &&
      std::all_of(candidate_resources.begin(), candidate_resources.end(),
                  [](const Resource &entry) {
                    return entry.available && entry.scratch == 0U &&
                           entry.active_blocks > 0 && entry.lds <= 64U * 1024U;
                  });
  std::cout << "resource_gate state_working_set_mib=288 resources_ok="
            << (resources_ok ? "PASS" : "FAIL") << "\n";

  StageTimes baseline_times{}, candidate_times{};
  HostSnapshot baseline_snapshot{}, candidate_snapshot{};
  bool baseline_deterministic = false;
  bool candidate_deterministic = false;
  ok =
      resources_ok &&
      run_probe(requested_target, Variant::Baseline, model, &baseline,
                &baseline_times, &baseline_snapshot, &baseline_deterministic) &&
      run_probe(requested_target, Variant::Candidate, model, &candidate,
                &candidate_times, &candidate_snapshot,
                &candidate_deterministic);
  if (ok) {
    const CompareResult baseline_compare =
        compare_snapshot(baseline_snapshot, oracle);
    const CompareResult candidate_compare =
        compare_snapshot(candidate_snapshot, oracle);
    std::cout << "oracle variant=baseline output_bf16_mismatches="
              << baseline_compare.bf16_mismatches
              << " output_max_bf16_ulp=" << baseline_compare.max_ulp
              << " state_max_abs=" << baseline_compare.max_state_abs
              << " state_max_rel=" << baseline_compare.max_state_rel
              << " nonfinite=" << baseline_compare.nonfinite << " status="
              << (baseline_compare.max_ulp <= 16U &&
                          baseline_compare.bf16_mismatches <=
                              baseline_snapshot.output.size() / 100U &&
                          baseline_compare.max_state_abs <= 1.0e-4F &&
                          baseline_compare.nonfinite == 0U
                      ? "PASS"
                      : "FAIL")
              << "\n";
    std::cout << "oracle variant=candidate output_bf16_mismatches="
              << candidate_compare.bf16_mismatches
              << " output_max_bf16_ulp=" << candidate_compare.max_ulp
              << " state_max_abs=" << candidate_compare.max_state_abs
              << " state_max_rel=" << candidate_compare.max_state_rel
              << " nonfinite=" << candidate_compare.nonfinite << " status="
              << (candidate_compare.max_ulp <= 16U &&
                          candidate_compare.bf16_mismatches <=
                              candidate_snapshot.output.size() / 100U &&
                          candidate_compare.max_state_abs <= 1.0e-4F &&
                          candidate_compare.nonfinite == 0U
                      ? "PASS"
                      : "FAIL")
              << "\n";
    ok = baseline_deterministic && candidate_deterministic &&
         baseline_compare.max_ulp <= 16U &&
         baseline_compare.bf16_mismatches <=
             baseline_snapshot.output.size() / 100U &&
         baseline_compare.max_state_abs <= 1.0e-4F &&
         candidate_compare.max_ulp <= 16U &&
         candidate_compare.bf16_mismatches <=
             candidate_snapshot.output.size() / 100U &&
         candidate_compare.max_state_abs <= 1.0e-4F &&
         baseline_compare.nonfinite == 0U &&
         candidate_compare.nonfinite == 0U &&
         compare_variants(baseline_snapshot, candidate_snapshot,
                          requested_target == "gfx1201"
                              ? "baseline-vs-column"
                              : "baseline-vs-row32-lds") &&
         baseline_snapshot.conv_state == candidate_snapshot.conv_state;
    std::cout << "conv_state_compare status="
              << (baseline_snapshot.conv_state == candidate_snapshot.conv_state
                      ? "PASS"
                      : "FAIL")
              << "\n";
  }
  print_times("baseline", baseline_times);
  print_times("candidate", candidate_times);
  const double baseline_total =
      (baseline_times.conv_ms + baseline_times.recurrent_ms) /
      static_cast<double>(kMeasured);
  const double candidate_total =
      (candidate_times.conv_ms + candidate_times.preprocess_ms +
       candidate_times.recurrent_ms + candidate_times.postprocess_ms) /
      static_cast<double>(kMeasured);
  std::cout << "weighted_speedup candidate_vs_baseline="
            << (candidate_total > 0.0 ? baseline_total / candidate_total : 0.0)
            << "\n";
  std::cout << "auxiliary_m2_value32=SKIP reason=exact-qwen38-m1-working-set\n";
  std::cout << "PHASE78_GDN_QWEN38_DECODE_EVIDENCE=" << (ok ? "PASS" : "FAIL")
            << "\n";
  std::cout << "PHASE78_GDN_QWEN38_DECODE_DECISION="
            << (ok && candidate_total < baseline_total ? "GO" : "N0") << "\n";
  const bool cleanup_ok = free_device(&baseline) && free_device(&candidate) &&
                          hip_ok(hipDeviceSynchronize(), "cleanup synchronize");
  std::cout << "cleanup status=" << (cleanup_ok ? "PASS" : "FAIL") << "\n";
  return ok && cleanup_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
