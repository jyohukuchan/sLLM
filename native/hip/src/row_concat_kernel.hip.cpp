#include "row_concat_internal.hpp"

#include <limits>

#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)

namespace sllm_row_concat_kernel {

hipError_t launch(const uint16_t *const, const uint16_t *const, uint16_t *const,
                  const uint64_t, const uint64_t, const uint64_t,
                  const hipStream_t) noexcept {
  return hipErrorNotSupported;
}

} // namespace sllm_row_concat_kernel

#else

extern "C" __global__ __launch_bounds__(256, 1) void sllm_concat_rows_bf16_v1(
    const uint16_t *const left, const uint16_t *const right,
    uint16_t *const output, const uint64_t rows, const uint64_t left_columns,
    const uint64_t right_columns) {
  const uint64_t output_columns = left_columns + right_columns;
  const uint64_t column = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                          static_cast<uint64_t>(threadIdx.x);
  if (column >= output_columns) {
    return;
  }
  for (uint64_t row = static_cast<uint64_t>(blockIdx.y); row < rows;
       row += static_cast<uint64_t>(gridDim.y)) {
    const uint64_t output_offset = row * output_columns + column;
    if (column < left_columns) {
      output[output_offset] = left[row * left_columns + column];
    } else {
      output[output_offset] =
          right[row * right_columns + (column - left_columns)];
    }
  }
}

namespace sllm_row_concat_kernel {

hipError_t launch(const uint16_t *const left, const uint16_t *const right,
                  uint16_t *const output, const uint64_t rows,
                  const uint64_t left_columns, const uint64_t right_columns,
                  const hipStream_t stream) noexcept {
  if (left_columns > std::numeric_limits<uint64_t>::max() - right_columns) {
    return hipErrorInvalidValue;
  }
  const uint64_t output_columns = left_columns + right_columns;
  const uint64_t blocks =
      output_columns / kWorkgroupSize +
      static_cast<uint64_t>(output_columns % kWorkgroupSize != 0U);
  constexpr uint32_t kMaxGridY = 65535U;
  const uint32_t grid_y = static_cast<uint32_t>(
      rows > static_cast<uint64_t>(kMaxGridY) ? kMaxGridY : rows);
  if (blocks == 0U || blocks > std::numeric_limits<uint32_t>::max() ||
      grid_y == 0U) {
    return hipErrorInvalidValue;
  }
  hipLaunchKernelGGL(sllm_concat_rows_bf16_v1,
                     dim3(static_cast<uint32_t>(blocks), grid_y, 1U),
                     dim3(kWorkgroupSize, 1U, 1U), 0U, stream, left, right,
                     output, rows, left_columns, right_columns);
  return hipGetLastError();
}

} // namespace sllm_row_concat_kernel

#endif // SLLM_PUBLIC_RUNTIME_HOST_TEST
