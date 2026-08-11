#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

extern "C" __global__ void sllm_hip_compile_probe(std::uint32_t *values,
                                                  std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    values[index] ^= 0x9e3779b9u;
  }
}

int main() { return 0; }
