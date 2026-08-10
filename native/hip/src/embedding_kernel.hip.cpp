#include "embedding_kernel_internal.hpp"

extern "C" __global__
__launch_bounds__(256, 1) void sllm_embedding_gather_bf16_i32_v1(
    const uint16_t *const weight, const int32_t *const token_ids,
    uint16_t *const output, const uint64_t output_elements,
    const uint64_t hidden_size) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < output_elements) {
    const uint64_t token = index / hidden_size;
    const uint64_t column = index - token * hidden_size;
    const uint64_t row = static_cast<uint64_t>(token_ids[token]);
    output[index] = weight[row * hidden_size + column];
  }
}

namespace sllm_embedding_kernel {

hipError_t launch_gather(const uint16_t *const weight,
                         const int32_t *const token_ids, uint16_t *const output,
                         const uint64_t token_count, const uint64_t hidden_size,
                         const hipStream_t stream) noexcept {
  const uint64_t output_elements = token_count * hidden_size;
  const uint32_t grid_size = static_cast<uint32_t>(
      (output_elements + kWorkgroupSize - 1U) / kWorkgroupSize);
  hipLaunchKernelGGL(sllm_embedding_gather_bf16_i32_v1, dim3(grid_size),
                     dim3(kWorkgroupSize), 0U, stream, weight, token_ids,
                     output, output_elements, hidden_size);
  return hipGetLastError();
}

} // namespace sllm_embedding_kernel
