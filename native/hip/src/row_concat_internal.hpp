#ifndef SLLM_ROW_CONCAT_INTERNAL_HPP
#define SLLM_ROW_CONCAT_INTERNAL_HPP

#include <hip/hip_runtime.h>

#include <cstdint>

namespace sllm_row_concat_kernel {

inline constexpr const char *kLogicalKernelId = "row_concat.bf16.v1";
inline constexpr const char *kDeviceSymbol = "sllm_concat_rows_bf16_v1";
inline constexpr uint32_t kWorkgroupSize = 256U;

hipError_t launch(const uint16_t *left, const uint16_t *right, uint16_t *output,
                  uint64_t rows, uint64_t left_columns, uint64_t right_columns,
                  hipStream_t stream) noexcept;

} // namespace sllm_row_concat_kernel

#endif // SLLM_ROW_CONCAT_INTERNAL_HPP
