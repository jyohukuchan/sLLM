#include "token_selector_kernel_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)

#include <cstring>

namespace {

float bf16_to_float(const uint16_t value) noexcept {
  uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint64_t splitmix64(uint64_t value) noexcept {
  value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31U);
}

uint32_t ordered_key(const float value) noexcept {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT32_C(0x80000000)) != 0U ? ~bits
                                             : (bits ^ UINT32_C(0x80000000));
}

hipError_t select_host(const uint16_t *const bf16_logits,
                       const float *const additive_logits,
                       const uint8_t *const valid_mask,
                       const uint64_t vocab_size, const float temperature,
                       const uint64_t seed, const uint64_t counter,
                       sllm_token_selector_record_t *const output) noexcept {
  output->token_id = -1;
  output->status = SLLM_STATUS_OK;
  output->logprob = -INFINITY;
  output->reserved0 = 0U;
  float maximum = -INFINITY;
  bool has_candidate = false;
  for (uint64_t index = 0U; index != vocab_size; ++index) {
    if (valid_mask[index] == 0U) {
      continue;
    }
    const float value =
        bf16_to_float(bf16_logits[index]) + additive_logits[index];
    if (!std::isfinite(value)) {
      output->status = SLLM_STATUS_TOKEN_SELECTOR_NONFINITE;
      return hipSuccess;
    }
    if (!has_candidate || value > maximum) {
      maximum = value;
      has_candidate = true;
    }
  }
  if (!has_candidate) {
    output->status = SLLM_STATUS_TOKEN_SELECTOR_ALL_MASKED;
    return hipSuccess;
  }
  double sum = 0.0;
  for (uint64_t index = 0U; index != vocab_size; ++index) {
    if (valid_mask[index] != 0U) {
      sum += std::exp(static_cast<double>(bf16_to_float(bf16_logits[index]) +
                                          additive_logits[index] - maximum) /
                      static_cast<double>(temperature));
    }
  }
  if (!(sum > 0.0F) || !std::isfinite(sum)) {
    output->status = SLLM_STATUS_TOKEN_SELECTOR_NONFINITE;
    return hipSuccess;
  }
  // Keep the device draw stream bit-identical to OsSamplingRandom: counter
  // zero is the first post-seed SplitMix increment.
  const uint64_t gamma = UINT64_C(0x9e3779b97f4a7c15);
  const uint64_t draw_state = seed + (counter + UINT64_C(1)) * gamma;
  const uint64_t random_bits = splitmix64(draw_state);
  const double unit =
      static_cast<double>(random_bits >> 11U) * (1.0 / 9007199254740992.0);
  const double target = unit * sum;
  uint32_t minimum_key = UINT32_MAX;
  uint32_t maximum_key = 0U;
  for (uint64_t index = 0U; index != vocab_size; ++index) {
    if (valid_mask[index] == 0U) {
      continue;
    }
    const float value =
        bf16_to_float(bf16_logits[index]) + additive_logits[index];
    const uint32_t key = ordered_key(value);
    minimum_key = std::min(minimum_key, key);
    maximum_key = std::max(maximum_key, key);
  }
  // Find the greatest ordered key whose inclusive mass still exceeds the
  // draw. This is the effective-logit cutoff in the legacy descending-logit
  // categorical order. The search is over the finite f32 key space and needs
  // no sorted index/workspace buffer.
  uint32_t low = minimum_key;
  uint32_t high = maximum_key;
  uint32_t best = minimum_key;
  while (low <= high) {
    const uint32_t mid = low + ((high - low) >> 1U);
    double mass = 0.0;
    for (uint64_t index = 0U; index != vocab_size; ++index) {
      if (valid_mask[index] == 0U) {
        continue;
      }
      const float value =
          bf16_to_float(bf16_logits[index]) + additive_logits[index];
      if (ordered_key(value) >= mid) {
        mass += std::exp(static_cast<double>(value - maximum) /
                         static_cast<double>(temperature));
      }
    }
    if (mass > target) {
      best = mid;
      if (mid == UINT32_MAX) {
        break;
      }
      low = mid + 1U;
    } else {
      if (mid == 0U) {
        break;
      }
      high = mid - 1U;
    }
  }
  uint32_t cutoff_key = minimum_key;
  for (uint64_t index = 0U; index != vocab_size; ++index) {
    if (valid_mask[index] == 0U) {
      continue;
    }
    const float value =
        bf16_to_float(bf16_logits[index]) + additive_logits[index];
    const uint32_t key = ordered_key(value);
    if (key <= best && key > cutoff_key) {
      cutoff_key = key;
    }
  }
  double cumulative = 0.0;
  for (uint64_t index = 0U; index != vocab_size; ++index) {
    if (valid_mask[index] == 0U) {
      continue;
    }
    const float value =
        bf16_to_float(bf16_logits[index]) + additive_logits[index];
    if (ordered_key(value) > cutoff_key) {
      cumulative += std::exp(static_cast<double>(value - maximum) /
                             static_cast<double>(temperature));
    }
  }
  uint64_t selected = vocab_size - 1U;
  double selected_probability = 0.0;
  for (uint64_t index = 0U; index != vocab_size; ++index) {
    if (valid_mask[index] == 0U) {
      continue;
    }
    const float value =
        bf16_to_float(bf16_logits[index]) + additive_logits[index];
    if (ordered_key(value) != cutoff_key) {
      continue;
    }
    const double probability = std::exp(static_cast<double>(value - maximum) /
                                        static_cast<double>(temperature));
    cumulative += probability;
    if (target < cumulative) {
      selected = index;
      selected_probability = probability / sum;
      break;
    }
  }
  if (selected_probability == 0.0) {
    // This only handles a roundoff-boundary draw. Choose the last token in
    // the canonical cutoff tie group, matching the host sampler's fallback.
    for (uint64_t index = 0U; index != vocab_size; ++index) {
      if (valid_mask[index] == 0U) {
        continue;
      }
      const float value =
          bf16_to_float(bf16_logits[index]) + additive_logits[index];
      if (ordered_key(value) == cutoff_key) {
        selected = index;
        selected_probability = std::exp(static_cast<double>(value - maximum) /
                                        static_cast<double>(temperature)) /
                               sum;
      }
    }
  }
  output->token_id = static_cast<int32_t>(selected);
  output->logprob = static_cast<float>(std::log(selected_probability));
  return hipSuccess;
}

} // namespace

namespace sllm_token_selector_kernel {

hipError_t launch(const uint16_t *const bf16_logits,
                  const float *const additive_logits,
                  const uint8_t *const valid_mask, const uint64_t vocab_size,
                  const float temperature, const uint64_t seed,
                  const uint64_t counter,
                  sllm_token_selector_record_t *const output,
                  const hipStream_t /*stream*/) noexcept {
  return select_host(bf16_logits, additive_logits, valid_mask, vocab_size,
                     temperature, seed, counter, output);
}

} // namespace sllm_token_selector_kernel

#else

namespace {

__device__ float bf16_to_float(const uint16_t value) noexcept {
  union {
    uint32_t bits;
    float value;
  } converted = {static_cast<uint32_t>(value) << 16U};
  return converted.value;
}

__device__ uint64_t splitmix64(uint64_t value) noexcept {
  value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31U);
}

__device__ uint32_t ordered_key(const float value) noexcept {
  union {
    uint32_t bits;
    float value;
  } converted = {0U};
  converted.value = value;
  return (converted.bits & UINT32_C(0x80000000)) != 0U
             ? ~converted.bits
             : (converted.bits ^ UINT32_C(0x80000000));
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE,
    1) void sllm_token_selector_bf16_f32_mask_v1(const uint16_t
                                                     *const bf16_logits,
                                                 const float
                                                     *const additive_logits,
                                                 const uint8_t
                                                     *const valid_mask,
                                                 const uint64_t vocab_size,
                                                 const float temperature,
                                                 const uint64_t seed,
                                                 const uint64_t counter,
                                                 sllm_token_selector_record_t
                                                     *const output) {
  if (threadIdx.x != 0U) {
    return;
  }
  output->token_id = -1;
  output->status = SLLM_STATUS_OK;
  output->logprob = -INFINITY;
  output->reserved0 = 0U;
  if (!isfinite(temperature) || temperature <= 0.0F) {
    output->status = SLLM_STATUS_TOKEN_SELECTOR_INVALID_TEMPERATURE;
    return;
  }
  float maximum = -INFINITY;
  bool has_candidate = false;
  for (uint64_t index = 0U; index != vocab_size; ++index) {
    if (valid_mask[index] == 0U) {
      continue;
    }
    const float value =
        bf16_to_float(bf16_logits[index]) + additive_logits[index];
    if (!isfinite(value)) {
      output->status = SLLM_STATUS_TOKEN_SELECTOR_NONFINITE;
      return;
    }
    if (!has_candidate || value > maximum) {
      maximum = value;
      has_candidate = true;
    }
  }
  if (!has_candidate) {
    output->status = SLLM_STATUS_TOKEN_SELECTOR_ALL_MASKED;
    return;
  }
  double sum = 0.0;
  for (uint64_t index = 0U; index != vocab_size; ++index) {
    if (valid_mask[index] != 0U) {
      sum += exp(static_cast<double>(bf16_to_float(bf16_logits[index]) +
                                     additive_logits[index] - maximum) /
                 static_cast<double>(temperature));
    }
  }
  if (!(sum > 0.0F) || !isfinite(sum)) {
    output->status = SLLM_STATUS_TOKEN_SELECTOR_NONFINITE;
    return;
  }
  // Keep the device draw stream bit-identical to OsSamplingRandom: counter
  // zero is the first post-seed SplitMix increment.
  const uint64_t gamma = UINT64_C(0x9e3779b97f4a7c15);
  const uint64_t draw_state = seed + (counter + UINT64_C(1)) * gamma;
  const uint64_t random_bits = splitmix64(draw_state);
  const double unit =
      static_cast<double>(random_bits >> 11U) * (1.0 / 9007199254740992.0);
  const double target = unit * sum;
  uint32_t minimum_key = UINT32_MAX;
  uint32_t maximum_key = 0U;
  for (uint64_t index = 0U; index != vocab_size; ++index) {
    if (valid_mask[index] == 0U) {
      continue;
    }
    const float value =
        bf16_to_float(bf16_logits[index]) + additive_logits[index];
    const uint32_t key = ordered_key(value);
    minimum_key = key < minimum_key ? key : minimum_key;
    maximum_key = key > maximum_key ? key : maximum_key;
  }
  // Search the finite f32 ordered-key space for the effective-logit cutoff.
  // The categorical order is descending effective logit, then ascending token
  // ID for ties; no sorted index buffer or vocabulary-sized D2H is needed.
  uint32_t low = minimum_key;
  uint32_t high = maximum_key;
  uint32_t best = minimum_key;
  while (low <= high) {
    const uint32_t mid = low + ((high - low) >> 1U);
    double mass = 0.0;
    for (uint64_t index = 0U; index != vocab_size; ++index) {
      if (valid_mask[index] == 0U) {
        continue;
      }
      const float value =
          bf16_to_float(bf16_logits[index]) + additive_logits[index];
      if (ordered_key(value) >= mid) {
        mass += exp(static_cast<double>(value - maximum) /
                    static_cast<double>(temperature));
      }
    }
    if (mass > target) {
      best = mid;
      if (mid == UINT32_MAX) {
        break;
      }
      low = mid + 1U;
    } else {
      if (mid == 0U) {
        break;
      }
      high = mid - 1U;
    }
  }
  uint32_t cutoff_key = minimum_key;
  for (uint64_t index = 0U; index != vocab_size; ++index) {
    if (valid_mask[index] == 0U) {
      continue;
    }
    const float value =
        bf16_to_float(bf16_logits[index]) + additive_logits[index];
    const uint32_t key = ordered_key(value);
    if (key <= best && key > cutoff_key) {
      cutoff_key = key;
    }
  }
  double cumulative = 0.0;
  for (uint64_t index = 0U; index != vocab_size; ++index) {
    if (valid_mask[index] == 0U) {
      continue;
    }
    const float value =
        bf16_to_float(bf16_logits[index]) + additive_logits[index];
    if (ordered_key(value) > cutoff_key) {
      cumulative += exp(static_cast<double>(value - maximum) /
                        static_cast<double>(temperature));
    }
  }
  uint64_t selected = vocab_size - 1U;
  double selected_probability = 0.0;
  for (uint64_t index = 0U; index != vocab_size; ++index) {
    if (valid_mask[index] == 0U) {
      continue;
    }
    const float value =
        bf16_to_float(bf16_logits[index]) + additive_logits[index];
    if (ordered_key(value) != cutoff_key) {
      continue;
    }
    const double probability = exp(static_cast<double>(value - maximum) /
                                   static_cast<double>(temperature));
    cumulative += probability;
    if (target < cumulative) {
      selected = index;
      selected_probability = probability / sum;
      break;
    }
  }
  if (selected_probability == 0.0) {
    for (uint64_t index = 0U; index != vocab_size; ++index) {
      if (valid_mask[index] == 0U) {
        continue;
      }
      const float value =
          bf16_to_float(bf16_logits[index]) + additive_logits[index];
      if (ordered_key(value) == cutoff_key) {
        selected = index;
        selected_probability = exp(static_cast<double>(value - maximum) /
                                   static_cast<double>(temperature)) /
                               sum;
      }
    }
  }
  output->token_id = static_cast<int32_t>(selected);
  output->logprob = static_cast<float>(log(selected_probability));
}

} // namespace

namespace sllm_token_selector_kernel {

hipError_t launch(const uint16_t *const bf16_logits,
                  const float *const additive_logits,
                  const uint8_t *const valid_mask, const uint64_t vocab_size,
                  const float temperature, const uint64_t seed,
                  const uint64_t counter,
                  sllm_token_selector_record_t *const output,
                  const hipStream_t stream) noexcept {
  const dim3 grid(1U, 1U, 1U);
  const dim3 block(SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE, 1U, 1U);
  hipLaunchKernelGGL(sllm_token_selector_bf16_f32_mask_v1, grid, block, 0U,
                     stream, bf16_logits, additive_logits, valid_mask,
                     vocab_size, temperature, seed, counter, output);
  return hipGetLastError();
}

} // namespace sllm_token_selector_kernel

#endif
