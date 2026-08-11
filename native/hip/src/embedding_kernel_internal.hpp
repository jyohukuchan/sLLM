#ifndef SLLM_EMBEDDING_KERNEL_INTERNAL_HPP
#define SLLM_EMBEDDING_KERNEL_INTERNAL_HPP

#include <hip/hip_runtime.h>

#include <cstdint>

namespace sllm_embedding_kernel {

constexpr uint32_t kWorkgroupSize = 256U;
constexpr const char *kLogicalKernelId = "embedding.gather.bf16_i32.v1";
constexpr const char *kDeviceSymbol = "sllm_embedding_gather_bf16_i32_v1";

hipError_t launch_gather(const uint16_t *weight, const int32_t *token_ids,
                         uint16_t *output, uint64_t token_count,
                         uint64_t hidden_size, hipStream_t stream) noexcept;

} // namespace sllm_embedding_kernel

#endif // SLLM_EMBEDDING_KERNEL_INTERNAL_HPP
