#include "argmax_kernel_internal.hpp"

#include <cmath>

namespace {

struct Candidate final {
  float value;
  uint32_t index;
  uint32_t valid;
  uint32_t has_nonfinite;
};

__device__ float bf16_to_float(const uint16_t bits) noexcept {
  union {
    uint32_t bits;
    float value;
  } converted = {static_cast<uint32_t>(bits) << 16U};
  return converted.value;
}

__device__ void merge_candidate(Candidate *const left,
                                const Candidate &right) noexcept {
  const uint32_t has_nonfinite = left->has_nonfinite | right.has_nonfinite;
  if (!right.valid || (left->valid && (left->value > right.value ||
                                       (left->value == right.value &&
                                        left->index <= right.index)))) {
    left->has_nonfinite = has_nonfinite;
    return;
  }
  *left = right;
  left->has_nonfinite = has_nonfinite;
}

extern "C" __global__
__launch_bounds__(SLLM_HIP_ARGMAX_WORKGROUP_SIZE,
                  1) void sllm_argmax_bf16_f32_v1(const uint16_t *const logits,
                                                  int32_t *const output,
                                                  const uint64_t vocab_size) {
  __shared__ Candidate candidates[SLLM_HIP_ARGMAX_WORKGROUP_SIZE];
  const uint32_t lane = static_cast<uint32_t>(threadIdx.x);
  const uint32_t row = static_cast<uint32_t>(blockIdx.x);
  Candidate local = {0.0F, 0U, 0U, 0U};
  const uint64_t row_offset = static_cast<uint64_t>(row) * vocab_size;
  for (uint64_t column = lane; column < vocab_size;
       column += SLLM_HIP_ARGMAX_WORKGROUP_SIZE) {
    const float value = bf16_to_float(logits[row_offset + column]);
    if (!isfinite(value)) {
      local.has_nonfinite = 1U;
      continue;
    }
    const uint32_t index = static_cast<uint32_t>(column);
    if (!local.valid || value > local.value ||
        (value == local.value && index < local.index)) {
      local = {value, index, 1U, local.has_nonfinite};
    }
  }
  candidates[lane] = local;
  __syncthreads();
  for (uint32_t stride = SLLM_HIP_ARGMAX_WORKGROUP_SIZE / 2U; stride != 0U;
       stride >>= 1U) {
    if (lane < stride) {
      merge_candidate(&candidates[lane], candidates[lane + stride]);
    }
    __syncthreads();
  }
  if (lane == 0U) {
    output[row] = candidates[0].has_nonfinite
                      ? -1
                      : static_cast<int32_t>(candidates[0].index);
  }
}

} // namespace

namespace sllm_argmax_kernel {

hipError_t launch(const uint16_t *const logits, int32_t *const output,
                  const uint64_t m, const uint64_t v,
                  const hipStream_t stream) noexcept {
  const dim3 grid(static_cast<uint32_t>(m), 1U, 1U);
  const dim3 block(SLLM_HIP_ARGMAX_WORKGROUP_SIZE, 1U, 1U);
  hipLaunchKernelGGL(sllm_argmax_bf16_f32_v1, grid, block, 0U, stream, logits,
                     output, v);
  return hipGetLastError();
}

} // namespace sllm_argmax_kernel
