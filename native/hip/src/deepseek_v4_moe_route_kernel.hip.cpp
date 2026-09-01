#include "deepseek_v4_moe_route_kernel_internal.hpp"

#include <cmath>

namespace {

__device__ __forceinline__ float bf16_to_float(const uint16_t bits) noexcept {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16U);
}

__device__ __forceinline__ float softplus(const float value) noexcept {
  return value > 0.0F ? value + log1pf(expf(-value)) : log1pf(expf(value));
}

__device__ __forceinline__ float unbiased_score(const float logit) noexcept {
  return sqrtf(softplus(logit));
}

__device__ __forceinline__ void
fail_row(int32_t *const expert_ids, float *const expert_weights,
         int32_t *const status, const uint64_t pair_base,
         const int32_t failure_status) noexcept {
  (void)atomicCAS(status, SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_OK, failure_status);
  for (uint32_t slot = 0U;
       slot < SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_SELECTED_EXPERT_COUNT; ++slot) {
    expert_ids[pair_base + slot] = -1;
    expert_weights[pair_base + slot] = NAN;
  }
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_WORKGROUP_SIZE,
    1) void sllm_deepseek_v4_moe_route_score_hash_v1(const uint16_t
                                                         *const logits,
                                                     const float
                                                         *const selection_bias,
                                                     const int32_t
                                                         *const hash_expert_ids,
                                                     int32_t *const expert_ids,
                                                     float
                                                         *const expert_weights,
                                                     int32_t *const status,
                                                     const uint32_t mode,
                                                     const uint32_t renormalize,
                                                     const float routed_scale) {
  const uint64_t token = blockIdx.x;
  if (threadIdx.x != 0U) {
    return;
  }
  constexpr uint64_t expert_count = SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_EXPERT_COUNT;
  constexpr uint32_t selected_count =
      SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_SELECTED_EXPERT_COUNT;
  const uint64_t row = token * expert_count;
  const uint64_t pair_base = token * selected_count;
  for (uint64_t expert = 0U; expert < expert_count; ++expert) {
    const float logit = bf16_to_float(logits[row + expert]);
    if (!isfinite(logit)) {
      fail_row(expert_ids, expert_weights, status, pair_base,
               SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_NONFINITE);
      return;
    }
    if (mode == SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_SCORE &&
        !isfinite(selection_bias[expert])) {
      fail_row(expert_ids, expert_weights, status, pair_base,
               SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_NONFINITE);
      return;
    }
  }

  float selected_scores[selected_count];
  if (mode == SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_SCORE) {
    for (uint32_t slot = 0U; slot < selected_count; ++slot) {
      int32_t best_id = -1;
      float best_selection_score = -INFINITY;
      float best_unbiased_score = 0.0F;
      for (uint32_t expert = 0U; expert < expert_count; ++expert) {
        bool used = false;
        for (uint32_t prior = 0U; prior < slot; ++prior) {
          used = used ||
                 expert_ids[pair_base + prior] == static_cast<int32_t>(expert);
        }
        const float score = unbiased_score(bf16_to_float(logits[row + expert]));
        const float selection_score = score + selection_bias[expert];
        if (!isfinite(score) || !isfinite(selection_score)) {
          fail_row(expert_ids, expert_weights, status, pair_base,
                   SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_NONFINITE);
          return;
        }
        if (!used && (best_id < 0 || selection_score > best_selection_score ||
                      (selection_score == best_selection_score &&
                       expert < static_cast<uint32_t>(best_id)))) {
          best_id = static_cast<int32_t>(expert);
          best_selection_score = selection_score;
          best_unbiased_score = score;
        }
      }
      expert_ids[pair_base + slot] = best_id;
      selected_scores[slot] = best_unbiased_score;
    }
  } else {
    for (uint32_t slot = 0U; slot < selected_count; ++slot) {
      const int32_t expert_id = hash_expert_ids[pair_base + slot];
      if (expert_id < 0 || expert_id >= static_cast<int32_t>(expert_count)) {
        fail_row(expert_ids, expert_weights, status, pair_base,
                 SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_EXPERT_OUT_OF_RANGE);
        return;
      }
      for (uint32_t prior = 0U; prior < slot; ++prior) {
        if (hash_expert_ids[pair_base + prior] == expert_id) {
          fail_row(expert_ids, expert_weights, status, pair_base,
                   SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_DUPLICATE_EXPERT);
          return;
        }
      }
      expert_ids[pair_base + slot] = expert_id;
      selected_scores[slot] = unbiased_score(
          bf16_to_float(logits[row + static_cast<uint32_t>(expert_id)]));
      if (!isfinite(selected_scores[slot])) {
        fail_row(expert_ids, expert_weights, status, pair_base,
                 SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_NONFINITE);
        return;
      }
    }
  }

  float normalizer = 1.0F;
  if (renormalize != 0U) {
    normalizer = 0.0F;
    for (uint32_t slot = 0U; slot < selected_count; ++slot) {
      normalizer += selected_scores[slot];
    }
    if (!isfinite(normalizer) || normalizer <= 0.0F) {
      fail_row(expert_ids, expert_weights, status, pair_base,
               SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_ZERO_NORMALIZER);
      return;
    }
  }
  for (uint32_t slot = 0U; slot < selected_count; ++slot) {
    const float weight = (selected_scores[slot] / normalizer) * routed_scale;
    if (!isfinite(weight)) {
      fail_row(expert_ids, expert_weights, status, pair_base,
               SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_NONFINITE);
      return;
    }
    expert_weights[pair_base + slot] = weight;
  }
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_WORKGROUP_SIZE,
    1) void sllm_deepseek_v4_moe_route_stable_group_v1(int32_t
                                                           *const expert_ids,
                                                       float *const
                                                           expert_weights,
                                                       int32_t
                                                           *const expert_counts,
                                                       int32_t *const
                                                           expert_offsets,
                                                       int32_t *const
                                                           grouped_token_ids,
                                                       int32_t *const
                                                           grouped_topk_slots,
                                                       const int32_t
                                                           *const status,
                                                       const uint64_t
                                                           token_count) {
  constexpr uint32_t expert_count =
      static_cast<uint32_t>(SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_EXPERT_COUNT);
  constexpr uint32_t selected_count =
      SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_SELECTED_EXPERT_COUNT;
  const uint32_t expert = threadIdx.x;
  const uint64_t pair_count = token_count * selected_count;
  if (expert < expert_count) {
    int32_t count = 0;
    if (*status == SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_OK) {
      for (uint64_t pair = 0U; pair < pair_count; ++pair) {
        count += expert_ids[pair] == static_cast<int32_t>(expert) ? 1 : 0;
      }
    }
    expert_counts[expert] = count;
  }
  __syncthreads();
  if (expert == 0U) {
    int32_t cursor = 0;
    expert_offsets[0] = 0;
    for (uint32_t index = 0U; index < expert_count; ++index) {
      cursor += expert_counts[index];
      expert_offsets[index + 1U] = cursor;
    }
  }
  __syncthreads();
  if (expert < expert_count &&
      *status == SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_OK) {
    int32_t cursor = expert_offsets[expert];
    for (uint64_t token = 0U; token < token_count; ++token) {
      for (uint32_t slot = 0U; slot < selected_count; ++slot) {
        const uint64_t pair = token * selected_count + slot;
        if (expert_ids[pair] == static_cast<int32_t>(expert)) {
          grouped_token_ids[cursor] = static_cast<int32_t>(token);
          grouped_topk_slots[cursor] = static_cast<int32_t>(slot);
          ++cursor;
        }
      }
    }
  } else if (expert == 0U && *status != SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_OK) {
    for (uint64_t pair = 0U; pair < pair_count; ++pair) {
      expert_ids[pair] = -1;
      expert_weights[pair] = NAN;
      grouped_token_ids[pair] = -1;
      grouped_topk_slots[pair] = -1;
    }
  }
}

} // namespace

namespace sllm_deepseek_v4_moe_route_kernel {

hipError_t launch(const uint16_t *const logits,
                  const float *const selection_bias,
                  const int32_t *const hash_expert_ids,
                  int32_t *const expert_ids, float *const expert_weights,
                  int32_t *const expert_counts, int32_t *const expert_offsets,
                  int32_t *const grouped_token_ids,
                  int32_t *const grouped_topk_slots, int32_t *const status,
                  const uint64_t token_count, const uint32_t mode,
                  const uint32_t renormalize, const float routed_scale,
                  const hipStream_t stream) noexcept {
  hipError_t result = hipMemsetAsync(status, 0, sizeof(*status), stream);
  if (result != hipSuccess) {
    return result;
  }
  hipLaunchKernelGGL(sllm_deepseek_v4_moe_route_score_hash_v1,
                     dim3(static_cast<uint32_t>(token_count)),
                     dim3(SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_WORKGROUP_SIZE), 0U,
                     stream, logits, selection_bias, hash_expert_ids,
                     expert_ids, expert_weights, status, mode, renormalize,
                     routed_scale);
  result = hipGetLastError();
  if (result != hipSuccess) {
    return result;
  }
  hipLaunchKernelGGL(sllm_deepseek_v4_moe_route_stable_group_v1, dim3(1U),
                     dim3(SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_WORKGROUP_SIZE), 0U,
                     stream, expert_ids, expert_weights, expert_counts,
                     expert_offsets, grouped_token_ids, grouped_topk_slots,
                     status, token_count);
  return hipGetLastError();
}

} // namespace sllm_deepseek_v4_moe_route_kernel
