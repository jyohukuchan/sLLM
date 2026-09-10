#include "token_selector_kernel_internal.hpp"
#include "token_selector_support_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

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
                  const uint8_t *const valid_mask, uint8_t *const /*workspace*/,
                  const uint64_t vocab_size, const float temperature,
                  const uint32_t top_k, const float top_p, const uint32_t flags,
                  const uint64_t seed, const uint64_t counter,
                  sllm_token_selector_record_t *const output,
                  const hipStream_t /*stream*/) noexcept {
  if (top_k != 0U || top_p != 1.0F) {
    output->token_id = -1;
    output->status = SLLM_STATUS_OK;
    output->logprob = -INFINITY;
    output->reserved0 = 0U;
    std::vector<std::pair<float, uint32_t>> candidates;
    candidates.reserve(static_cast<std::size_t>(vocab_size));
    for (uint64_t index = 0U; index != vocab_size; ++index) {
      if ((flags & SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT) != 0U &&
          valid_mask[index] == 0U) {
        continue;
      }
      const float value =
          bf16_to_float(bf16_logits[index]) +
          ((flags & SLLM_HIP_TOKEN_SELECTOR_FLAG_ADDITIVE_PRESENT) != 0U
               ? additive_logits[index]
               : 0.0F);
      if (!std::isfinite(value)) {
        output->status = SLLM_STATUS_TOKEN_SELECTOR_NONFINITE;
        return hipSuccess;
      }
      candidates.emplace_back(value, static_cast<uint32_t>(index));
    }
    if (candidates.empty()) {
      output->status = SLLM_STATUS_TOKEN_SELECTOR_ALL_MASKED;
      return hipSuccess;
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto &left, const auto &right) {
                return left.first != right.first ? left.first > right.first
                                                 : left.second < right.second;
              });
    const std::size_t requested = top_k == 0U ? candidates.size() : top_k;
    const std::size_t count =
        std::min<std::size_t>(requested, candidates.size());
    double sum = 0.0;
    for (std::size_t index = 0U; index != count; ++index) {
      sum += std::exp(static_cast<double>(candidates[index].first -
                                          candidates.front().first));
    }
    const double cutoff = static_cast<double>(top_p) * sum;
    std::size_t included = 0U;
    double cumulative = 0.0;
    for (; included != count; ++included) {
      cumulative += std::exp(static_cast<double>(candidates[included].first -
                                                 candidates.front().first));
      if (cumulative >= cutoff) {
        ++included;
        break;
      }
    }
    included = std::max<std::size_t>(1U, included);
    const uint64_t gamma = UINT64_C(0x9e3779b97f4a7c15);
    const uint64_t draw_state = seed + (counter + UINT64_C(1)) * gamma;
    const uint64_t random_bits = splitmix64(draw_state);
    const double unit =
        static_cast<double>(random_bits >> 11U) * (1.0 / 9007199254740992.0);
    double target = unit * cumulative;
    std::size_t selected = included - 1U;
    double selected_weight = 0.0;
    double running = 0.0;
    for (std::size_t index = 0U; index != included; ++index) {
      const double weight = std::exp(static_cast<double>(
          candidates[index].first - candidates.front().first));
      running += weight;
      if (target < running) {
        selected = index;
        selected_weight = weight;
        break;
      }
    }
    if (selected_weight == 0.0) {
      selected_weight = std::exp(static_cast<double>(
          candidates[selected].first - candidates.front().first));
    }
    output->token_id = static_cast<int32_t>(candidates[selected].second);
    output->logprob =
        static_cast<float>(std::log(selected_weight / cumulative));
    return hipSuccess;
  }
  return select_host(bf16_logits, additive_logits, valid_mask, vocab_size,
                     temperature, seed, counter, output);
}

hipError_t launch(const uint16_t *const bf16_logits,
                  const float *const additive_logits,
                  const uint8_t *const valid_mask, const uint64_t vocab_size,
                  const float temperature, const uint64_t seed,
                  const uint64_t counter,
                  sllm_token_selector_record_t *const output,
                  const hipStream_t stream) noexcept {
  return launch(bf16_logits, additive_logits, valid_mask, nullptr, vocab_size,
                temperature, 0U, 1.0F,
                SLLM_HIP_TOKEN_SELECTOR_FLAG_ADDITIVE_PRESENT |
                    SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT,
                seed, counter, output, stream);
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

constexpr uint32_t kK0Bins = 65536U;
constexpr uint64_t kK0CountsOffset = 0U;
constexpr uint64_t kK0PrefixOffset = 262144U;
constexpr uint64_t kK0HeaderOffset = 524288U;
// Header words: selected key, selected rank, sample mass, boundary key,
// maximum key, immutable total mass, and boundary count. Word 6 aliases the
// first block-count slot; the boundary stage writes it before the token-count
// stage consumes that slot for block counts.
constexpr uint64_t kK0BlockCountsOffset = 524312U;
constexpr uint64_t kK0BlockOffsetsOffset = 528408U;

__device__ uint16_t bf16_ordered_key(const uint16_t bits) {
  const uint16_t canonical =
      (bits & UINT16_C(0x7fff)) == 0U ? UINT16_C(0) : bits;
  return (canonical & UINT16_C(0x8000)) != 0U
             ? static_cast<uint16_t>(~canonical)
             : static_cast<uint16_t>(canonical ^ UINT16_C(0x8000));
}

__device__ uint16_t bf16_from_ordered_key(const uint16_t key) {
  return (key & UINT16_C(0x8000)) != 0U
             ? static_cast<uint16_t>(key ^ UINT16_C(0x8000))
             : static_cast<uint16_t>(~key);
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE,
    1) void sllm_token_selector_fixed_topp_histogram_v1(const uint16_t
                                                            *const bf16_logits,
                                                        const uint8_t
                                                            *const valid_mask,
                                                        uint8_t
                                                            *const workspace,
                                                        const uint64_t
                                                            vocab_size,
                                                        const uint32_t flags,
                                                        sllm_token_selector_record_t
                                                            *const output) {
  uint32_t *const counts =
      reinterpret_cast<uint32_t *>(workspace + kK0CountsOffset);
  const uint32_t tid = threadIdx.x;
  const uint64_t tile_start = static_cast<uint64_t>(blockIdx.x) * 1024U;
  for (uint32_t offset = 0U; offset != 1024U;
       offset += SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE) {
    const uint64_t index = tile_start + tid + offset;
    if (index >= vocab_size ||
        ((flags & SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT) != 0U &&
         valid_mask[index] == 0U)) {
      continue;
    }
    const uint16_t bits = bf16_logits[index];
    const float value = bf16_to_float(bits);
    if (!isfinite(value)) {
      atomicExch(&output->status, SLLM_STATUS_TOKEN_SELECTOR_NONFINITE);
      continue;
    }
    const uint16_t key = bf16_ordered_key(bits);
    atomicAdd(&counts[key], 1U);
    atomicMax(reinterpret_cast<uint32_t *>(workspace + kK0HeaderOffset +
                                           4U * sizeof(uint32_t)),
              static_cast<uint32_t>(key));
  }
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE,
    1) void sllm_token_selector_fixed_topp_prefix_v1(uint8_t *const workspace,
                                                     const uint64_t vocab_size,
                                                     sllm_token_selector_record_t
                                                         *const output) {
  const uint32_t tid = threadIdx.x;
  const uint32_t key =
      blockIdx.x * SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE + tid;
  uint32_t *const counts =
      reinterpret_cast<uint32_t *>(workspace + kK0CountsOffset);
  float *const weights = reinterpret_cast<float *>(workspace + kK0PrefixOffset);
  uint32_t *const header =
      reinterpret_cast<uint32_t *>(workspace + kK0HeaderOffset);
  __shared__ float partial[SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE];
  float value = 0.0F;
  if (key < kK0Bins && counts[key] != 0U) {
    const float maximum =
        bf16_to_float(bf16_from_ordered_key(static_cast<uint16_t>(header[4])));
    const float score =
        bf16_to_float(bf16_from_ordered_key(static_cast<uint16_t>(key)));
    value = static_cast<float>(counts[key]) * expf(score - maximum);
  }
  if (key < kK0Bins) {
    weights[key] = value / (key < kK0Bins && counts[key] != 0U
                                ? static_cast<float>(counts[key])
                                : 1.0F);
  }
  partial[tid] = value;
  __syncthreads();
  for (uint32_t stride = SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE / 2U;
       stride != 0U; stride >>= 1U) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0U) {
    reinterpret_cast<float *>(workspace + kK0BlockCountsOffset)[blockIdx.x] =
        partial[0];
  }
  (void)vocab_size;
  (void)output;
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE,
    1) void sllm_token_selector_fixed_topp_weight_prefix_v1(uint8_t *const
                                                                workspace,
                                                            sllm_token_selector_record_t
                                                                *const output) {
  if (threadIdx.x != 0U) {
    return;
  }
  uint32_t *const header =
      reinterpret_cast<uint32_t *>(workspace + kK0HeaderOffset);
  const float *const block_weights =
      reinterpret_cast<const float *>(workspace + kK0BlockCountsOffset);
  float *const block_offsets =
      reinterpret_cast<float *>(workspace + kK0BlockOffsetsOffset);
  float cumulative = 0.0F;
  for (int32_t block = static_cast<int32_t>(kK0Bins / 256U) - 1; block >= 0;
       --block) {
    block_offsets[block] = cumulative;
    cumulative += block_weights[block];
  }
  union {
    float value;
    uint32_t bits;
  } total = {cumulative};
  header[5] = total.bits;
  if (output->status != SLLM_STATUS_OK || !(cumulative > 0.0F) ||
      !isfinite(cumulative)) {
    if (output->status == SLLM_STATUS_OK) {
      output->status = SLLM_STATUS_TOKEN_SELECTOR_ALL_MASKED;
    }
    output->token_id = -1;
    output->logprob = -INFINITY;
    output->reserved0 = 0U;
  }
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE,
    1) void sllm_token_selector_fixed_topp_boundary_v1(uint8_t *const workspace,
                                                       const float top_p,
                                                       sllm_token_selector_record_t
                                                           *const output) {
  if (output->status != SLLM_STATUS_OK || threadIdx.x != 0U) {
    return;
  }
  uint32_t *const header =
      reinterpret_cast<uint32_t *>(workspace + kK0HeaderOffset);
  const uint32_t *const counts =
      reinterpret_cast<const uint32_t *>(workspace + kK0CountsOffset);
  const float *const weights =
      reinterpret_cast<const float *>(workspace + kK0PrefixOffset);
  const float *const block_offsets =
      reinterpret_cast<const float *>(workspace + kK0BlockOffsetsOffset);
  const float total = reinterpret_cast<const float *>(&header[5])[0];
  const double target = static_cast<double>(top_p) * total;
  double cumulative = block_offsets[blockIdx.x];
  // Exactly one block owns the mass interval containing the target. Blocks
  // below it must leave the boundary header untouched; at an exact block
  // boundary the preceding (higher-score) block owns the inclusive endpoint.
  if (target <= cumulative) {
    return;
  }
  const int32_t first = static_cast<int32_t>(blockIdx.x * 256U + 255U);
  const int32_t last = static_cast<int32_t>(blockIdx.x * 256U);
  for (int32_t key = first; key >= last; --key) {
    const uint32_t count = counts[key];
    if (count == 0U) {
      continue;
    }
    const double weight = weights[key];
    if (!(weight > 0.0) || !isfinite(weight)) {
      continue;
    }
    const double bucket = static_cast<double>(count) * weight;
    if (cumulative + bucket >= target) {
      const double remaining = target - cumulative;
      const uint32_t needed = static_cast<uint32_t>(fmin(
          fmax(ceil(remaining / weight), 1.0), static_cast<double>(count)));
      const float sample_total =
          static_cast<float>(cumulative + static_cast<double>(needed) * weight);
      header[3] = static_cast<uint32_t>(key);
      header[6] = needed;
      union {
        float value;
        uint32_t bits;
      } sample = {sample_total};
      header[2] = sample.bits;
      return;
    }
    cumulative += bucket;
  }
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE,
    1) void sllm_token_selector_fixed_topp_select_key_v1(uint8_t
                                                             *const workspace,
                                                         const uint64_t seed,
                                                         const uint64_t counter,
                                                         sllm_token_selector_record_t
                                                             *const output) {
  if (output->status != SLLM_STATUS_OK || threadIdx.x != 0U) {
    return;
  }
  uint32_t *const header =
      reinterpret_cast<uint32_t *>(workspace + kK0HeaderOffset);
  const uint32_t *const counts =
      reinterpret_cast<const uint32_t *>(workspace + kK0CountsOffset);
  const float *const weights =
      reinterpret_cast<const float *>(workspace + kK0PrefixOffset);
  const float *const block_offsets =
      reinterpret_cast<const float *>(workspace + kK0BlockOffsetsOffset);
  const uint16_t limit_key = static_cast<uint16_t>(header[3]);
  const uint32_t limit_count = header[6];
  const float sample_total = reinterpret_cast<const float *>(&header[2])[0];
  const float maximum =
      bf16_to_float(bf16_from_ordered_key(static_cast<uint16_t>(header[4])));
  const uint64_t gamma = UINT64_C(0x9e3779b97f4a7c15);
  const uint64_t random_bits =
      splitmix64(seed + (counter + UINT64_C(1)) * gamma);
  const double unit =
      static_cast<double>(random_bits >> 11U) / 9007199254740992.0;
  const double target = unit * static_cast<double>(sample_total);
  double cumulative = block_offsets[blockIdx.x];
  // Blocks above the draw's mass range must not claim the draw merely because
  // their first bucket would make `next` larger than target.
  if (target < cumulative) {
    return;
  }
  const int32_t first = static_cast<int32_t>(blockIdx.x * 256U + 255U);
  const int32_t last = static_cast<int32_t>(blockIdx.x * 256U);
  for (int32_t key = first; key >= last; --key) {
    if (static_cast<uint32_t>(key) < limit_key) {
      continue;
    }
    const uint32_t count = counts[key];
    if (count == 0U) {
      continue;
    }
    const double weight = weights[key];
    const uint32_t usable =
        static_cast<uint32_t>(key) == limit_key ? limit_count : count;
    const double next = cumulative + static_cast<double>(usable) * weight;
    if (target < next) {
      uint32_t rank = static_cast<uint32_t>((target - cumulative) / weight);
      rank = rank >= usable ? usable - 1U : rank;
      header[0] = static_cast<uint32_t>(key);
      header[1] = rank;
      return;
    }
    cumulative = next;
  }
  (void)maximum;
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE,
    1) void sllm_token_selector_fixed_topp_count_v1(const uint16_t
                                                        *const bf16_logits,
                                                    const uint8_t
                                                        *const valid_mask,
                                                    const uint8_t
                                                        *const workspace,
                                                    const uint64_t vocab_size,
                                                    const uint32_t flags) {
  const uint32_t *const header =
      reinterpret_cast<const uint32_t *>(workspace + kK0HeaderOffset);
  const uint16_t selected_key = static_cast<uint16_t>(header[0]);
  uint32_t *const block_counts = reinterpret_cast<uint32_t *>(
      const_cast<uint8_t *>(workspace) + kK0BlockCountsOffset);
  const uint32_t tid = threadIdx.x;
  const uint64_t tile_start = static_cast<uint64_t>(blockIdx.x) * 1024U;
  uint32_t local_count = 0U;
  for (uint32_t offset = 0U; offset != 1024U;
       offset += SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE) {
    const uint64_t index = tile_start + tid + offset;
    if (index < vocab_size &&
        ((flags & SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT) == 0U ||
         valid_mask[index] != 0U) &&
        bf16_ordered_key(bf16_logits[index]) == selected_key) {
      ++local_count;
    }
  }
  __shared__ uint32_t counts[256];
  counts[tid] = local_count;
  __syncthreads();
  for (uint32_t stride = 128U; stride != 0U; stride >>= 1U) {
    if (tid < stride) {
      counts[tid] += counts[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0U) {
    block_counts[blockIdx.x] = counts[0];
  }
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE,
    1) void sllm_token_selector_fixed_topp_token_block_prefix_v1(uint8_t *const
                                                                     workspace,
                                                                 const uint32_t
                                                                     block_count) {
  if (threadIdx.x != 0U) {
    return;
  }
  uint32_t *const block_counts =
      reinterpret_cast<uint32_t *>(workspace + kK0BlockCountsOffset);
  uint32_t *const block_offsets =
      reinterpret_cast<uint32_t *>(workspace + kK0BlockOffsetsOffset);
  uint32_t cumulative = 0U;
  for (uint32_t block = 0U; block != block_count; ++block) {
    block_offsets[block] = cumulative;
    cumulative += block_counts[block];
  }
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE,
    1) void sllm_token_selector_fixed_topp_select_v1(const uint16_t
                                                         *const bf16_logits,
                                                     const uint8_t
                                                         *const valid_mask,
                                                     const uint8_t
                                                         *const workspace,
                                                     const uint64_t vocab_size,
                                                     const uint32_t flags,
                                                     sllm_token_selector_record_t
                                                         *const output) {
  if (output->status != SLLM_STATUS_OK || threadIdx.x != 0U) {
    return;
  }
  const uint32_t *const header =
      reinterpret_cast<const uint32_t *>(workspace + kK0HeaderOffset);
  const uint32_t *const block_offsets =
      reinterpret_cast<const uint32_t *>(workspace + kK0BlockOffsetsOffset);
  const uint16_t selected_key = static_cast<uint16_t>(header[0]);
  const uint32_t selected_rank = header[1];
  const float sample_total = reinterpret_cast<const float *>(&header[2])[0];
  const uint64_t tile_start = static_cast<uint64_t>(blockIdx.x) * 1024U;
  uint32_t local_rank = block_offsets[blockIdx.x];
  for (uint32_t offset = 0U; offset != 1024U; ++offset) {
    const uint64_t index = tile_start + offset;
    if (index >= vocab_size ||
        ((flags & SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT) != 0U &&
         valid_mask[index] == 0U) ||
        bf16_ordered_key(bf16_logits[index]) != selected_key) {
      continue;
    }
    if (local_rank == selected_rank) {
      const float maximum = bf16_to_float(
          bf16_from_ordered_key(static_cast<uint16_t>(header[4])));
      const float selected = bf16_to_float(bf16_logits[index]);
      output->token_id = static_cast<int32_t>(index);
      output->logprob = logf(expf(selected - maximum) / sample_total);
      return;
    }
    ++local_rank;
  }
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

#if 0 // superseded by the tiled initial/reduction/final fixed path below
extern "C" __global__ __launch_bounds__(
    SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE,
    1) void sllm_token_selector_fixed_topk_topp_v1(
    const uint16_t *const bf16_logits, const float *const additive_logits,
    const uint8_t *const valid_mask, uint8_t *const workspace,
    const uint64_t vocab_size, const uint32_t top_k, const float top_p,
    const uint32_t flags, const uint64_t seed, const uint64_t counter,
    sllm_token_selector_record_t *const output) {
  if (threadIdx.x != 0U) {
    return;
  }
  output->token_id = -1;
  output->status = SLLM_STATUS_OK;
  output->logprob = -INFINITY;
  output->reserved0 = 0U;
  float scores[64];
  uint32_t ids[64];
  uint32_t count = 0U;
  for (uint64_t index = 0U; index != vocab_size; ++index) {
    if ((flags & SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT) != 0U &&
        valid_mask[index] == 0U) {
      continue;
    }
    const float value = bf16_to_float(bf16_logits[index]) +
                        ((flags & SLLM_HIP_TOKEN_SELECTOR_FLAG_ADDITIVE_PRESENT) != 0U
                             ? additive_logits[index]
                             : 0.0F);
    if (!isfinite(value)) {
      output->status = SLLM_STATUS_TOKEN_SELECTOR_NONFINITE;
      return;
    }
    uint32_t position = count < top_k ? count : top_k;
    if (count < top_k) {
      ++count;
    } else if (value < scores[top_k - 1U] ||
               (value == scores[top_k - 1U] && index > ids[top_k - 1U])) {
      continue;
    }
    while (position != 0U &&
           (value > scores[position - 1U] ||
            (value == scores[position - 1U] && index < ids[position - 1U]))) {
      if (position < top_k) {
        scores[position] = scores[position - 1U];
        ids[position] = ids[position - 1U];
      }
      --position;
    }
    scores[position] = value;
    ids[position] = static_cast<uint32_t>(index);
  }
  if (count == 0U) {
    output->status = SLLM_STATUS_TOKEN_SELECTOR_ALL_MASKED;
    return;
  }
  if (workspace != nullptr) {
    // The persistent workspace is also the ABI-visible ownership boundary.
    // Store the final candidates for future block-reduction implementations;
    // selection itself remains deterministic for this launch.
    for (uint32_t index = 0U; index != count; ++index) {
      reinterpret_cast<uint32_t *>(workspace)[index * 2U] = ids[index];
      reinterpret_cast<float *>(workspace)[index * 2U + 1U] = scores[index];
    }
  }
  double sum = 0.0;
  for (uint32_t index = 0U; index != count; ++index) {
    sum += exp(static_cast<double>(scores[index] - scores[0]));
  }
  const double cutoff = static_cast<double>(top_p) * sum;
  uint32_t included = 0U;
  double cumulative = 0.0;
  for (; included != count; ++included) {
    cumulative += exp(static_cast<double>(scores[included] - scores[0]));
    if (cumulative >= cutoff) {
      ++included;
      break;
    }
  }
  included = included == 0U ? 1U : included;
  const uint64_t gamma = UINT64_C(0x9e3779b97f4a7c15);
  const uint64_t draw_state = seed + (counter + UINT64_C(1)) * gamma;
  const uint64_t random_bits = splitmix64(draw_state);
  const double target = static_cast<double>(random_bits >> 11U) *
                        (1.0 / 9007199254740992.0) * cumulative;
  double running = 0.0;
  uint32_t selected = included - 1U;
  for (uint32_t index = 0U; index != included; ++index) {
    running += exp(static_cast<double>(scores[index] - scores[0]));
    if (target < running) {
      selected = index;
      break;
    }
  }
  const double selected_weight =
      exp(static_cast<double>(scores[selected] - scores[0]));
  output->token_id = static_cast<int32_t>(ids[selected]);
  output->logprob = static_cast<float>(log(selected_weight / cumulative));
}

#endif

__device__ bool token_selector_before(const float left_score,
                                      const uint32_t left_id,
                                      const float right_score,
                                      const uint32_t right_id) {
  return left_score > right_score ||
         (left_score == right_score && left_id < right_id);
}

__device__ void token_selector_bitonic_sort(float *const scores,
                                            uint32_t *const ids) {
  const uint32_t tid = threadIdx.x;
  for (uint32_t k = 2U; k <= SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE; k <<= 1U) {
    for (uint32_t j = k >> 1U; j != 0U; j >>= 1U) {
      for (uint32_t index = tid; index < SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE;
           index += SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE) {
        const uint32_t partner = index ^ j;
        if (partner > index) {
          const bool ascending = (index & k) == 0U;
          const bool swap =
              ascending ? token_selector_before(scores[partner], ids[partner],
                                                scores[index], ids[index])
                        : token_selector_before(scores[index], ids[index],
                                                scores[partner], ids[partner]);
          if (swap) {
            const float score = scores[index];
            scores[index] = scores[partner];
            scores[partner] = score;
            const uint32_t id = ids[index];
            ids[index] = ids[partner];
            ids[partner] = id;
          }
        }
      }
      __syncthreads();
    }
  }
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE,
    1) void sllm_token_selector_fixed_topk_initial_v1(const uint16_t
                                                          *const bf16_logits,
                                                      const float *const
                                                          additive_logits,
                                                      const uint8_t
                                                          *const valid_mask,
                                                      uint32_t *const workspace,
                                                      const uint64_t vocab_size,
                                                      const uint32_t top_k,
                                                      const uint32_t flags,
                                                      sllm_token_selector_record_t
                                                          *const output) {
  __shared__ float scores[SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE];
  __shared__ uint32_t ids[SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE];
  const uint32_t tid = threadIdx.x;
  const uint64_t tile_start =
      static_cast<uint64_t>(blockIdx.x) * SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE;
  for (uint32_t offset = 0U; offset != SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE;
       offset += SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE) {
    const uint64_t index = tile_start + tid + offset;
    float score = -INFINITY;
    uint32_t id = UINT32_MAX;
    if (index < vocab_size &&
        ((flags & SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT) == 0U ||
         valid_mask[index] != 0U)) {
      score = bf16_to_float(bf16_logits[index]) +
              ((flags & SLLM_HIP_TOKEN_SELECTOR_FLAG_ADDITIVE_PRESENT) != 0U
                   ? additive_logits[index]
                   : 0.0F);
      id = static_cast<uint32_t>(index);
      if (!isfinite(score)) {
        atomicExch(&output->status, SLLM_STATUS_TOKEN_SELECTOR_NONFINITE);
        score = -INFINITY;
        id = UINT32_MAX;
      }
    }
    scores[tid + offset] = score;
    ids[tid + offset] = id;
  }
  __syncthreads();
  token_selector_bitonic_sort(scores, ids);
  if (tid < top_k) {
    const uint64_t record =
        (static_cast<uint64_t>(blockIdx.x) * top_k + tid) * 2U;
    workspace[record] = ids[tid];
    union {
      float value;
      uint32_t bits;
    } encoded = {scores[tid]};
    workspace[record + 1U] = encoded.bits;
  }
  if (blockIdx.x == 0U && tid == 0U) {
    output->token_id = -1;
    output->logprob = -INFINITY;
    output->reserved0 = 0U;
  }
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE,
    1) void sllm_token_selector_fixed_topk_reduce_v1(const uint32_t
                                                         *const input,
                                                     uint32_t *const
                                                         output_workspace,
                                                     const uint64_t
                                                         input_records,
                                                     const uint32_t top_k) {
  __shared__ float scores[SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE];
  __shared__ uint32_t ids[SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE];
  const uint32_t tid = threadIdx.x;
  const uint64_t base =
      static_cast<uint64_t>(blockIdx.x) * SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE;
  for (uint32_t offset = 0U; offset != SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE;
       offset += SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE) {
    const uint64_t index = base + tid + offset;
    float score = -INFINITY;
    uint32_t id = UINT32_MAX;
    if (index < input_records) {
      const uint64_t record = index * 2U;
      id = input[record];
      union {
        float value;
        uint32_t bits;
      } encoded = {0.0F};
      encoded.bits = input[record + 1U];
      score = encoded.value;
    }
    scores[tid + offset] = score;
    ids[tid + offset] = id;
  }
  __syncthreads();
  token_selector_bitonic_sort(scores, ids);
  if (tid < top_k) {
    const uint64_t record =
        (static_cast<uint64_t>(blockIdx.x) * top_k + tid) * 2U;
    output_workspace[record] = ids[tid];
    union {
      float value;
      uint32_t bits;
    } encoded = {scores[tid]};
    output_workspace[record + 1U] = encoded.bits;
  }
}

__device__ void token_selector_support_store_u32(uint8_t *const base,
                                                 const uint32_t offset,
                                                 const uint32_t value) {
  base[offset + 0U] = static_cast<uint8_t>(value);
  base[offset + 1U] = static_cast<uint8_t>(value >> 8U);
  base[offset + 2U] = static_cast<uint8_t>(value >> 16U);
  base[offset + 3U] = static_cast<uint8_t>(value >> 24U);
}

__device__ void token_selector_support_store_f64(uint8_t *const base,
                                                 const uint32_t offset,
                                                 const double value) {
  uint64_t bits = 0U;
  __builtin_memcpy(&bits, &value, sizeof(bits));
  for (uint32_t byte = 0U; byte != 8U; ++byte) {
    base[offset + byte] = static_cast<uint8_t>(bits >> (byte * 8U));
  }
}

template <bool CaptureSupport>
__device__ __forceinline__ void token_selector_fixed_topk_final_impl(
    const uint32_t *const input, const uint64_t input_records,
    const uint32_t top_k, const float top_p, const uint64_t seed,
    const uint64_t counter, sllm_token_selector_record_t *const output,
    uint8_t *const support) {
  __shared__ float scores[SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE];
  __shared__ uint32_t ids[SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE];
  const uint32_t tid = threadIdx.x;
  for (uint32_t offset = 0U; offset != SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE;
       offset += SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE) {
    const uint64_t index = static_cast<uint64_t>(tid) + offset;
    float score = -INFINITY;
    uint32_t id = UINT32_MAX;
    if (index < input_records) {
      const uint64_t record = index * 2U;
      id = input[record];
      union {
        float value;
        uint32_t bits;
      } encoded = {0.0F};
      encoded.bits = input[record + 1U];
      score = encoded.value;
    }
    scores[tid + offset] = score;
    ids[tid + offset] = id;
  }
  __syncthreads();
  token_selector_bitonic_sort(scores, ids);
  if (tid != 0U) {
    return;
  }
  if constexpr (CaptureSupport) {
    token_selector_support_store_u32(support, 0U,
                                     sllm_token_selector_support::kVersionV1);
    token_selector_support_store_u32(support, 4U, output->status);
    token_selector_support_store_u32(support, 8U, 0U);
    token_selector_support_store_u32(support, 12U, 0U);
    for (uint32_t index = 0U; index != sllm_token_selector_support::kMaxCountV1;
         ++index) {
      token_selector_support_store_u32(support, 16U + index * 4U, 0U);
      token_selector_support_store_f64(support, 96U + index * 8U, 0.0);
    }
  }
  if (output->status != SLLM_STATUS_OK) {
    output->token_id = -1;
    output->logprob = -INFINITY;
    output->reserved0 = 0U;
    return;
  }
  if (ids[0] == UINT32_MAX) {
    output->status = SLLM_STATUS_TOKEN_SELECTOR_ALL_MASKED;
    if constexpr (CaptureSupport) {
      token_selector_support_store_u32(support, 4U, output->status);
    }
    return;
  }
  const uint32_t count =
      static_cast<uint32_t>(input_records < top_k ? input_records : top_k);
  double sum = 0.0;
  for (uint32_t index = 0U; index != count; ++index) {
    sum += exp(static_cast<double>(scores[index] - scores[0]));
  }
  const double cutoff = static_cast<double>(top_p) * sum;
  uint32_t included = 0U;
  double cumulative = 0.0;
  for (; included != count; ++included) {
    cumulative += exp(static_cast<double>(scores[included] - scores[0]));
    if (cumulative >= cutoff) {
      ++included;
      break;
    }
  }
  included = included == 0U ? 1U : included;
  if constexpr (CaptureSupport) {
    token_selector_support_store_u32(support, 8U, included);
    for (uint32_t index = 0U; index != included; ++index) {
      token_selector_support_store_u32(support, 16U + index * 4U, ids[index]);
      token_selector_support_store_f64(
          support, 96U + index * 8U,
          exp(static_cast<double>(scores[index] - scores[0])) / cumulative);
    }
  }
  const uint64_t gamma = UINT64_C(0x9e3779b97f4a7c15);
  const uint64_t draw_state = seed + (counter + UINT64_C(1)) * gamma;
  const uint64_t random_bits = splitmix64(draw_state);
  const double target = static_cast<double>(random_bits >> 11U) *
                        (1.0 / 9007199254740992.0) * cumulative;
  double running = 0.0;
  uint32_t selected = included - 1U;
  for (uint32_t index = 0U; index != included; ++index) {
    running += exp(static_cast<double>(scores[index] - scores[0]));
    if (target < running) {
      selected = index;
      break;
    }
  }
  const double selected_weight =
      exp(static_cast<double>(scores[selected] - scores[0]));
  output->token_id = static_cast<int32_t>(ids[selected]);
  output->logprob = static_cast<float>(log(selected_weight / cumulative));
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE,
    1) void sllm_token_selector_fixed_topk_final_v1(const uint32_t *const input,
                                                    const uint64_t
                                                        input_records,
                                                    const uint32_t top_k,
                                                    const float top_p,
                                                    const uint64_t seed,
                                                    const uint64_t counter,
                                                    sllm_token_selector_record_t
                                                        *const output) {
  token_selector_fixed_topk_final_impl<false>(
      input, input_records, top_k, top_p, seed, counter, output, nullptr);
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE,
    1) void sllm_token_selector_fixed_topk_final_support_v1(const uint32_t
                                                                *const input,
                                                            const uint64_t
                                                                input_records,
                                                            const uint32_t
                                                                top_k,
                                                            const float top_p,
                                                            const uint64_t seed,
                                                            const uint64_t
                                                                counter,
                                                            sllm_token_selector_record_t
                                                                *const output,
                                                            uint8_t *const
                                                                support) {
  token_selector_fixed_topk_final_impl<true>(input, input_records, top_k, top_p,
                                             seed, counter, output, support);
}

} // namespace

namespace sllm_token_selector_kernel {

hipError_t launch(const uint16_t *const bf16_logits,
                  const float *const additive_logits,
                  const uint8_t *const valid_mask, uint8_t *const workspace,
                  const uint64_t vocab_size, const float temperature,
                  const uint32_t top_k, const float top_p, const uint32_t flags,
                  const uint64_t seed, const uint64_t counter,
                  sllm_token_selector_record_t *const output,
                  const hipStream_t stream) noexcept {
  if (top_k != 0U) {
    const dim3 block(SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE, 1U, 1U);
    const uint64_t block_count =
        (vocab_size + SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE - 1U) /
        SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE;
    const uint64_t region_bytes =
        block_count * static_cast<uint64_t>(top_k) * UINT64_C(8);
    const dim3 initial_grid(static_cast<unsigned int>(block_count), 1U, 1U);
    hipError_t launch_status =
        hipMemsetAsync(output, 0, sizeof(*output), stream);
    if (launch_status != hipSuccess) {
      return launch_status;
    }
    hipLaunchKernelGGL(sllm_token_selector_fixed_topk_initial_v1, initial_grid,
                       block, 0U, stream, bf16_logits, additive_logits,
                       valid_mask, reinterpret_cast<uint32_t *>(workspace),
                       vocab_size, top_k, flags, output);
    launch_status = hipGetLastError();
    if (launch_status != hipSuccess) {
      return launch_status;
    }
    uint64_t input_records = block_count * static_cast<uint64_t>(top_k);
    uint64_t input_offset = 0U;
    uint64_t output_offset = region_bytes;
    while (input_records > SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE) {
      const uint64_t reduce_blocks =
          (input_records + SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE - 1U) /
          SLLM_HIP_TOKEN_SELECTOR_TILE_SIZE;
      const dim3 reduce_grid(static_cast<unsigned int>(reduce_blocks), 1U, 1U);
      hipLaunchKernelGGL(
          sllm_token_selector_fixed_topk_reduce_v1, reduce_grid, block, 0U,
          stream, reinterpret_cast<const uint32_t *>(workspace + input_offset),
          reinterpret_cast<uint32_t *>(workspace + output_offset),
          input_records, top_k);
      launch_status = hipGetLastError();
      if (launch_status != hipSuccess) {
        return launch_status;
      }
      input_records = reduce_blocks * static_cast<uint64_t>(top_k);
      const uint64_t old_input_offset = input_offset;
      input_offset = output_offset;
      output_offset = old_input_offset;
    }
    const dim3 final_grid(1U, 1U, 1U);
    if (top_k == sllm_token_selector_support::kMaxCountV1) {
      hipLaunchKernelGGL(
          sllm_token_selector_fixed_topk_final_support_v1, final_grid, block,
          0U, stream,
          reinterpret_cast<const uint32_t *>(workspace + input_offset),
          input_records, top_k, top_p, seed, counter, output, workspace);
    } else {
      hipLaunchKernelGGL(
          sllm_token_selector_fixed_topk_final_v1, final_grid, block, 0U,
          stream, reinterpret_cast<const uint32_t *>(workspace + input_offset),
          input_records, top_k, top_p, seed, counter, output);
    }
    return hipGetLastError();
  }
  if (top_p != 1.0F) {
    const dim3 block(SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE, 1U, 1U);
    const uint64_t block_count = (vocab_size + 1023U) / 1024U;
    const dim3 grid(static_cast<unsigned int>(block_count), 1U, 1U);
    hipError_t launch_status =
        hipMemsetAsync(output, 0, sizeof(*output), stream);
    if (launch_status != hipSuccess) {
      return launch_status;
    }
    launch_status = hipMemsetAsync(
        workspace, 0, SLLM_HIP_TOKEN_SELECTOR_K0_WORKSPACE_BYTES, stream);
    if (launch_status != hipSuccess) {
      return launch_status;
    }
    hipLaunchKernelGGL(sllm_token_selector_fixed_topp_histogram_v1, grid, block,
                       0U, stream, bf16_logits, valid_mask, workspace,
                       vocab_size, flags, output);
    launch_status = hipGetLastError();
    if (launch_status != hipSuccess) {
      return launch_status;
    }
    const dim3 weight_grid(kK0Bins / 256U, 1U, 1U);
    hipLaunchKernelGGL(sllm_token_selector_fixed_topp_prefix_v1, weight_grid,
                       block, 0U, stream, workspace, vocab_size, output);
    launch_status = hipGetLastError();
    if (launch_status != hipSuccess) {
      return launch_status;
    }
    hipLaunchKernelGGL(sllm_token_selector_fixed_topp_weight_prefix_v1,
                       dim3(1U, 1U, 1U), block, 0U, stream, workspace, output);
    launch_status = hipGetLastError();
    if (launch_status != hipSuccess) {
      return launch_status;
    }
    hipLaunchKernelGGL(sllm_token_selector_fixed_topp_boundary_v1, weight_grid,
                       block, 0U, stream, workspace, top_p, output);
    launch_status = hipGetLastError();
    if (launch_status != hipSuccess) {
      return launch_status;
    }
    hipLaunchKernelGGL(sllm_token_selector_fixed_topp_select_key_v1,
                       weight_grid, block, 0U, stream, workspace, seed, counter,
                       output);
    launch_status = hipGetLastError();
    if (launch_status != hipSuccess) {
      return launch_status;
    }
    hipLaunchKernelGGL(sllm_token_selector_fixed_topp_count_v1, grid, block, 0U,
                       stream, bf16_logits, valid_mask, workspace, vocab_size,
                       flags);
    launch_status = hipGetLastError();
    if (launch_status != hipSuccess) {
      return launch_status;
    }
    hipLaunchKernelGGL(sllm_token_selector_fixed_topp_token_block_prefix_v1,
                       dim3(1U, 1U, 1U), block, 0U, stream, workspace,
                       static_cast<uint32_t>(block_count));
    launch_status = hipGetLastError();
    if (launch_status != hipSuccess) {
      return launch_status;
    }
    hipLaunchKernelGGL(sllm_token_selector_fixed_topp_select_v1, grid, block,
                       0U, stream, bf16_logits, valid_mask, workspace,
                       vocab_size, flags, output);
    return hipGetLastError();
  }
  const dim3 grid(1U, 1U, 1U);
  const dim3 block(SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE, 1U, 1U);
  hipLaunchKernelGGL(sllm_token_selector_bf16_f32_mask_v1, grid, block, 0U,
                     stream, bf16_logits, additive_logits, valid_mask,
                     vocab_size, temperature, seed, counter, output);
  return hipGetLastError();
}

hipError_t launch(const uint16_t *const bf16_logits,
                  const float *const additive_logits,
                  const uint8_t *const valid_mask, const uint64_t vocab_size,
                  const float temperature, const uint64_t seed,
                  const uint64_t counter,
                  sllm_token_selector_record_t *const output,
                  const hipStream_t stream) noexcept {
  return launch(bf16_logits, additive_logits, valid_mask, nullptr, vocab_size,
                temperature, 0U, 1.0F,
                SLLM_HIP_TOKEN_SELECTOR_FLAG_ADDITIVE_PRESENT |
                    SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT,
                seed, counter, output, stream);
}

} // namespace sllm_token_selector_kernel

#endif
