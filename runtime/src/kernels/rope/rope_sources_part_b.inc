// Ordered fragment included into the anonymous namespace of ullm_runtime.cpp; do not include elsewhere.
    static const char *rope_kernel_source() {
        return R"(
extern "C" __global__ void ullm_rope_f32_kernel(
    const float *input,
    unsigned long long sequence_len,
    unsigned long long heads,
    unsigned long long head_dim,
    unsigned long long rotary_dim,
    unsigned long long position_offset,
    float rope_base,
    float *output) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long half = rotary_dim >> 1;
    const unsigned long long work_dim = half + (head_dim - rotary_dim);
    const unsigned long long work_items = sequence_len * heads * work_dim;
    if (index >= work_items) {
        return;
    }
    const unsigned long long local_dim = index % work_dim;
    const unsigned long long head_index = index / work_dim;
    const unsigned long long base = head_index * head_dim;
    if (local_dim >= half) {
        const unsigned long long dim = rotary_dim + (local_dim - half);
        output[base + dim] = input[base + dim];
        return;
    }
    const unsigned long long pair_dim = local_dim;
    const unsigned long long timestep = head_index / heads;
    const float position = static_cast<float>(position_offset + timestep);
    const float exponent =
        (2.0f * static_cast<float>(pair_dim)) / static_cast<float>(rotary_dim);
    const float theta = position / powf(rope_base, exponent);
    const float c = cosf(theta);
    const float s = sinf(theta);
    const float first = input[base + pair_dim];
    const float second = input[base + half + pair_dim];
    output[base + pair_dim] = first * c - second * s;
    output[base + half + pair_dim] = second * c + first * s;
}
)";
    }

