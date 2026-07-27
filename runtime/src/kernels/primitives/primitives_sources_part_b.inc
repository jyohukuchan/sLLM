// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
    static const char *add_kernel_source() {
        return R"(
extern "C" __global__ void ullm_add_f32_kernel(
    const float *lhs,
    const float *rhs,
    unsigned long long elements,
    float *output) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    output[index] = lhs[index] + rhs[index];
}
)";
    }

