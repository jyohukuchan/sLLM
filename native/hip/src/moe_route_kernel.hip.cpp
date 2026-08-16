#include "moe_route_kernel_internal.hpp"

#include <cmath>

namespace {

__device__ __forceinline__ float bf16_to_float(const uint16_t bits) noexcept {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16U);
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_MOE_ROUTE_WORKGROUP_SIZE,
    1) void sllm_moe_route_bf16_stable_topk_v1(const uint16_t *const logits,
                                               int32_t *const expert_ids,
                                               float *const expert_weights,
                                               int32_t *const status,
                                               const uint64_t expert_count,
                                               const uint32_t
                                                   selected_expert_count) {
  const uint64_t token = blockIdx.x;
  if (threadIdx.x != 0U) {
    return;
  }
  const uint64_t row = token * expert_count;
  bool finite = true;
  for (uint64_t expert = 0U; expert < expert_count; ++expert) {
    finite = finite && isfinite(bf16_to_float(logits[row + expert]));
  }
  const uint64_t pair_base = token * selected_expert_count;
  if (!finite) {
    atomicExch(status, 1);
    for (uint32_t slot = 0U; slot < selected_expert_count; ++slot) {
      expert_ids[pair_base + slot] = -1;
      expert_weights[pair_base + slot] = NAN;
    }
    return;
  }
  float selected_logits[SLLM_HIP_MOE_ROUTE_MAX_SELECTED];
  float selected_exponentials[SLLM_HIP_MOE_ROUTE_MAX_SELECTED];
  for (uint32_t slot = 0U; slot < selected_expert_count; ++slot) {
    int32_t best_id = -1;
    float best_value = -INFINITY;
    for (uint32_t expert = 0U; expert < expert_count; ++expert) {
      bool used = false;
      for (uint32_t prior = 0U; prior < slot; ++prior) {
        used = used ||
               expert_ids[pair_base + prior] == static_cast<int32_t>(expert);
      }
      const float value = bf16_to_float(logits[row + expert]);
      if (!used &&
          (best_id < 0 || value > best_value ||
           (value == best_value && expert < static_cast<uint32_t>(best_id)))) {
        best_id = static_cast<int32_t>(expert);
        best_value = value;
      }
    }
    expert_ids[pair_base + slot] = best_id;
    selected_logits[slot] = best_value;
  }
  float maximum = -INFINITY;
  for (uint32_t expert = 0U; expert < expert_count; ++expert) {
    maximum = fmaxf(maximum, bf16_to_float(logits[row + expert]));
  }
  float denominator = 0.0F;
  for (uint32_t expert = 0U; expert < expert_count; ++expert) {
    denominator += expf(bf16_to_float(logits[row + expert]) - maximum);
  }
  float selected_sum = 0.0F;
  for (uint32_t slot = 0U; slot < selected_expert_count; ++slot) {
    const float value = expf(selected_logits[slot] - maximum) / denominator;
    selected_exponentials[slot] = value;
    selected_sum += value;
  }
  for (uint32_t slot = 0U; slot < selected_expert_count; ++slot) {
    expert_weights[pair_base + slot] =
        selected_exponentials[slot] / selected_sum;
  }
}

extern "C" __global__ __launch_bounds__(
    SLLM_HIP_MOE_ROUTE_WORKGROUP_SIZE,
    1) void sllm_moe_route_stable_group_v1(const int32_t *const expert_ids,
                                           int32_t *const expert_counts,
                                           int32_t *const expert_offsets,
                                           int32_t *const grouped_token_ids,
                                           int32_t *const grouped_topk_slots,
                                           const int32_t *const status,
                                           const uint64_t token_count,
                                           const uint64_t expert_count,
                                           const uint32_t
                                               selected_expert_count) {
  const uint32_t expert = threadIdx.x;
  const uint64_t pair_count = token_count * selected_expert_count;
  if (expert < expert_count) {
    int32_t count = 0;
    if (*status == 0) {
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
  if (expert < expert_count && *status == 0) {
    int32_t cursor = expert_offsets[expert];
    for (uint64_t token = 0U; token < token_count; ++token) {
      for (uint32_t slot = 0U; slot < selected_expert_count; ++slot) {
        const uint64_t pair = token * selected_expert_count + slot;
        if (expert_ids[pair] == static_cast<int32_t>(expert)) {
          grouped_token_ids[cursor] = static_cast<int32_t>(token);
          grouped_topk_slots[cursor] = static_cast<int32_t>(slot);
          ++cursor;
        }
      }
    }
  } else if (expert == 0U && *status != 0) {
    for (uint64_t pair = 0U; pair < pair_count; ++pair) {
      grouped_token_ids[pair] = -1;
      grouped_topk_slots[pair] = -1;
    }
  }
}

} // namespace

namespace sllm_moe_route_kernel {

hipError_t launch(const uint16_t *const logits, int32_t *const expert_ids,
                  float *const expert_weights, int32_t *const expert_counts,
                  int32_t *const expert_offsets,
                  int32_t *const grouped_token_ids,
                  int32_t *const grouped_topk_slots, int32_t *const status,
                  const uint64_t token_count, const uint64_t expert_count,
                  const uint32_t selected_expert_count,
                  const hipStream_t stream) noexcept {
  hipError_t result = hipMemsetAsync(status, 0, sizeof(*status), stream);
  if (result != hipSuccess) {
    return result;
  }
  hipLaunchKernelGGL(sllm_moe_route_bf16_stable_topk_v1,
                     dim3(static_cast<uint32_t>(token_count)),
                     dim3(SLLM_HIP_MOE_ROUTE_WORKGROUP_SIZE), 0U, stream,
                     logits, expert_ids, expert_weights, status, expert_count,
                     selected_expert_count);
  result = hipGetLastError();
  if (result != hipSuccess) {
    return result;
  }
  hipLaunchKernelGGL(sllm_moe_route_stable_group_v1, dim3(1U),
                     dim3(SLLM_HIP_MOE_ROUTE_WORKGROUP_SIZE), 0U, stream,
                     expert_ids, expert_counts, expert_offsets,
                     grouped_token_ids, grouped_topk_slots, status, token_count,
                     expert_count, selected_expert_count);
  return hipGetLastError();
}

} // namespace sllm_moe_route_kernel
