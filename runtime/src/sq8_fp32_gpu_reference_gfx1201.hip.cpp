// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

// This is intentionally a correctness control, not a candidate kernel.  It
// does not include, call, or dispatch through any production SQ8 CK/WMMA or
// HIPRTC implementation.  Canonical OCP E4M3FN bytes are decoded to F32 by a
// direct scalar kernel, then every projection uses ordinary hipBLAS SGEMM.

#include "sq8_fp32_gpu_reference_gfx1201.h"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace {

constexpr size_t kLayers = 40u;
constexpr size_t kHidden = 5120u;
constexpr size_t kIntermediate = 17408u;
constexpr size_t kQHeads = 40u;
constexpr size_t kKvHeads = 8u;
constexpr size_t kHeadDim = 128u;
constexpr size_t kQWidth = kQHeads * kHeadDim;
constexpr size_t kKvWidth = kKvHeads * kHeadDim;
constexpr size_t kVocab = 151936u;
constexpr size_t kFp8Block = 128u;
constexpr float kRmsNormEpsilon = 1.0e-6f;
constexpr float kRopeTheta = 1000000.0f;

void write_error(char* error, size_t capacity, std::string_view message) noexcept {
    if (error == nullptr || capacity == 0u) {
        return;
    }
    const size_t count = std::min(message.size(), capacity - 1u);
    std::memcpy(error, message.data(), count);
    error[count] = '\0';
}

void hip_check(hipError_t status, std::string_view operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + hipGetErrorString(status));
    }
}

void hipblas_check(hipblasStatus_t status, std::string_view operation) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with hipBLAS status " +
            std::to_string(static_cast<int>(status)));
    }
}

bool exact_gfx1201(const char* gcn_arch_name) {
    if (gcn_arch_name == nullptr || std::strncmp(gcn_arch_name, "gfx1201", 7u) != 0) {
        return false;
    }
    return gcn_arch_name[7] == '\0' || gcn_arch_name[7] == ':';
}

void validate_visible_r9700(struct ullm_sq8_fp32_gpu_reference_gfx1201_device_info* info) {
    const char* hip_visible = std::getenv("HIP_VISIBLE_DEVICES");
    const char* ullm_hip_visible = std::getenv("ULLM_HIP_VISIBLE_DEVICES");
    if (hip_visible == nullptr || std::strcmp(hip_visible, "1") != 0 ||
        ullm_hip_visible == nullptr || std::strcmp(ullm_hip_visible, "1") != 0) {
        throw std::runtime_error(
            "GPU F32 reference requires HIP_VISIBLE_DEVICES=1 and "
            "ULLM_HIP_VISIBLE_DEVICES=1 (the pinned R9700 token)");
    }
    int device_count = 0;
    hip_check(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
    if (device_count != 1) {
        throw std::runtime_error(
            "GPU F32 reference requires exactly one filtered HIP device");
    }
    hip_check(hipSetDevice(0), "hipSetDevice(0)");
    hipDeviceProp_t properties{};
    hip_check(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties(0)");
    properties.gcnArchName[sizeof(properties.gcnArchName) - 1u] = '\0';
    if (!exact_gfx1201(properties.gcnArchName)) {
        throw std::runtime_error(
            std::string("GPU F32 reference requires exact gfx1201; selected ") +
            properties.gcnArchName);
    }
    char pci_bdf[sizeof(info->pci_bdf)] = {};
    hip_check(hipDeviceGetPCIBusId(pci_bdf, sizeof(pci_bdf), 0), "hipDeviceGetPCIBusId(0)");
    std::memset(info, 0, sizeof(*info));
    info->total_global_mem_bytes = static_cast<uint64_t>(properties.totalGlobalMem);
    std::strncpy(info->name, properties.name, sizeof(info->name) - 1u);
    std::strncpy(info->gcn_arch_name, properties.gcnArchName, sizeof(info->gcn_arch_name) - 1u);
    std::strncpy(info->pci_bdf, pci_bdf, sizeof(info->pci_bdf) - 1u);
}

size_t checked_product(size_t left, size_t right, std::string_view label) {
    if (left == 0u || right == 0u || left > std::numeric_limits<size_t>::max() / right) {
        throw std::runtime_error(std::string(label) + " size overflows");
    }
    return left * right;
}

size_t checked_bytes(size_t elements, size_t element_bytes, std::string_view label) {
    return checked_product(elements, element_bytes, label);
}

unsigned int grid_for(size_t elements, std::string_view label) {
    constexpr size_t kThreads = 256u;
    const size_t blocks = (elements + kThreads - 1u) / kThreads;
    if (blocks == 0u || blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error(std::string(label) + " grid is outside supported range");
    }
    return static_cast<unsigned int>(blocks);
}

class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : pointer_(std::exchange(other.pointer_, nullptr)), bytes_(std::exchange(other.bytes_, 0u)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release_noexcept();
            pointer_ = std::exchange(other.pointer_, nullptr);
            bytes_ = std::exchange(other.bytes_, 0u);
        }
        return *this;
    }

    ~DeviceBuffer() { release_noexcept(); }

    void allocate(size_t bytes, std::string_view label) {
        if (bytes == 0u) {
            throw std::runtime_error(std::string(label) + " allocation requested zero bytes");
        }
        if (pointer_ != nullptr) {
            throw std::runtime_error(std::string(label) + " allocation was requested twice");
        }
        void* pointer = nullptr;
        hip_check(hipMalloc(&pointer, bytes), std::string(label) + " hipMalloc");
        pointer_ = pointer;
        bytes_ = bytes;
    }

    void release_checked(std::string_view label) {
        if (pointer_ != nullptr) {
            hip_check(hipFree(pointer_), std::string(label) + " hipFree");
            pointer_ = nullptr;
            bytes_ = 0u;
        }
    }

    [[nodiscard]] void* data() const { return pointer_; }
    [[nodiscard]] size_t bytes() const { return bytes_; }
    [[nodiscard]] bool allocated() const { return pointer_ != nullptr; }

  private:
    void release_noexcept() noexcept {
        if (pointer_ != nullptr) {
            (void)hipFree(pointer_);
            pointer_ = nullptr;
            bytes_ = 0u;
        }
    }

    void* pointer_ = nullptr;
    size_t bytes_ = 0u;
};

struct QuantizedWeight {
    DeviceBuffer payload;
    // Keep canonical scale bytes until the direct dequantization kernel reads
    // them.  In particular, do not borrow CPU-decoded F32 scales: doing so
    // would make the GPU control depend on the CPU reference's decoder.
    DeviceBuffer scales_bf16;
    size_t rows = 0u;
    size_t cols = 0u;
    size_t uploaded_bytes = 0u;
};

struct Bf16Upload {
    DeviceBuffer payload;
    size_t expected_bytes = 0u;
    size_t uploaded_bytes = 0u;
};

struct DeviceLayerNorms {
    DeviceBuffer input;
    DeviceBuffer post_attention;
    DeviceBuffer q;
    DeviceBuffer k;
    bool uploaded = false;
};

__device__ float ocp_e4m3fn_to_f32(unsigned char raw) {
    const unsigned int magnitude = raw & 0x7fu;
    if (magnitude == 0x7fu) {
        return __int_as_float(static_cast<int>(0x7fc00000u));
    }
    const float sign = (raw & 0x80u) == 0u ? 1.0f : -1.0f;
    const unsigned int exponent = (raw >> 3u) & 0x0fu;
    const unsigned int mantissa = raw & 0x07u;
    if (exponent == 0u) {
        return sign * static_cast<float>(mantissa) * 0.001953125f;
    }
    return sign * ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f,
                         static_cast<int>(exponent) - 7);
}

__device__ float bf16_to_f32(uint16_t raw) {
    return __int_as_float(static_cast<int>(static_cast<uint32_t>(raw) << 16u));
}

__global__ void dequant_sq8_ocp_block128_to_f32(
    const unsigned char* payload,
    const uint16_t* scales_bf16,
    float* output,
    unsigned long long rows,
    unsigned long long cols) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long elements = rows * cols;
    if (index >= elements) {
        return;
    }
    const unsigned long long row = index / cols;
    const unsigned long long col = index % cols;
    const unsigned long long scale_cols = (cols + 127ull) / 128ull;
    const unsigned long long scale_index = (row / 128ull) * scale_cols + col / 128ull;
    output[index] = ocp_e4m3fn_to_f32(payload[index]) * bf16_to_f32(scales_bf16[scale_index]);
}

__global__ void bf16_vector_to_f32(const uint16_t* input, float* output, unsigned long long elements) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = bf16_to_f32(input[index]);
    }
}

__global__ void embedding_row_bf16_to_f32(
    const uint16_t* embedding,
    unsigned int token_id,
    float* output) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < kHidden) {
        output[index] = bf16_to_f32(embedding[static_cast<size_t>(token_id) * kHidden + index]);
    }
}

// One GPU thread owns one normalization row.  This is intentionally serial
// within a row: no warp reduction, no fused projection, and no candidate
// reduction tree is shared with the optimized path.
__global__ void rmsnorm_serial_f32(
    const float* input,
    const uint16_t* weight_bf16,
    float* output,
    unsigned int rows,
    unsigned int cols) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const size_t base = static_cast<size_t>(row) * cols;
    float sum_squares = 0.0f;
    for (unsigned int column = 0u; column < cols; ++column) {
        const float value = input[base + column];
        sum_squares = fmaf(value, value, sum_squares);
    }
    const float inverse_rms = 1.0f / sqrtf(sum_squares / static_cast<float>(cols) + kRmsNormEpsilon);
    for (unsigned int column = 0u; column < cols; ++column) {
        output[base + column] =
            (input[base + column] * inverse_rms) * bf16_to_f32(weight_bf16[column]);
    }
}

__global__ void rope_split_half_f32(
    const float* input,
    float* output,
    unsigned int heads,
    unsigned int position) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int half = kHeadDim / 2u;
    const unsigned int pairs = heads * half;
    if (index >= pairs) {
        return;
    }
    const unsigned int head = index / half;
    const unsigned int pair_dim = index % half;
    const float exponent = static_cast<float>(2u * pair_dim) / static_cast<float>(kHeadDim);
    const float angle = static_cast<float>(position) / powf(kRopeTheta, exponent);
    float sine = 0.0f;
    float cosine = 0.0f;
    sincosf(angle, &sine, &cosine);
    const size_t base = static_cast<size_t>(head) * kHeadDim;
    const float first = input[base + pair_dim];
    const float second = input[base + half + pair_dim];
    output[base + pair_dim] = first * cosine - second * sine;
    output[base + half + pair_dim] = second * cosine + first * sine;
}

__global__ void copy_kv_f32(
    const float* key,
    const float* value,
    float* keys,
    float* values,
    unsigned long long layer_base,
    unsigned long long position) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < kKvWidth) {
        const unsigned long long destination = layer_base + position * kKvWidth + index;
        keys[destination] = key[index];
        values[destination] = value[index];
    }
}

// A literal three-pass causal softmax: scores/max, exp/sum, then weighted V.
// Each query head is one serial thread so this shares no tiled/Flash/WMMA
// machinery with the measured kernels.
__global__ void attention_scores_and_max_f32(
    const float* q,
    const float* keys,
    float* scores,
    float* max_scores,
    unsigned int tokens,
    unsigned int max_context) {
    const unsigned int q_head = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_head >= kQHeads) {
        return;
    }
    const unsigned int q_per_kv = kQHeads / kKvHeads;
    const unsigned int kv_head = q_head / q_per_kv;
    const size_t q_base = static_cast<size_t>(q_head) * kHeadDim;
    float max_score = -INFINITY;
    const float scale = 1.0f / sqrtf(static_cast<float>(kHeadDim));
    for (unsigned int source = 0u; source < tokens; ++source) {
        const size_t key_base =
            (static_cast<size_t>(source) * kKvHeads + kv_head) * kHeadDim;
        float dot = 0.0f;
        for (unsigned int dim = 0u; dim < kHeadDim; ++dim) {
            dot = fmaf(q[q_base + dim], keys[key_base + dim], dot);
        }
        const float score = dot * scale;
        scores[static_cast<size_t>(q_head) * max_context + source] = score;
        max_score = fmaxf(max_score, score);
    }
    max_scores[q_head] = max_score;
}

__global__ void attention_exp_and_sum_f32(
    float* scores,
    const float* max_scores,
    float* denominators,
    unsigned int tokens,
    unsigned int max_context) {
    const unsigned int q_head = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_head >= kQHeads) {
        return;
    }
    float denominator = 0.0f;
    const float max_score = max_scores[q_head];
    for (unsigned int source = 0u; source < tokens; ++source) {
        float& score = scores[static_cast<size_t>(q_head) * max_context + source];
        score = expf(score - max_score);
        denominator += score;
    }
    denominators[q_head] = denominator;
}

__global__ void attention_weighted_values_f32(
    const float* scores,
    const float* denominators,
    const float* values,
    float* output,
    unsigned int tokens,
    unsigned int max_context) {
    const unsigned int q_head = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_head >= kQHeads) {
        return;
    }
    const unsigned int q_per_kv = kQHeads / kKvHeads;
    const unsigned int kv_head = q_head / q_per_kv;
    const size_t output_base = static_cast<size_t>(q_head) * kHeadDim;
    for (unsigned int dim = 0u; dim < kHeadDim; ++dim) {
        float weighted = 0.0f;
        for (unsigned int source = 0u; source < tokens; ++source) {
            const size_t value_index =
                (static_cast<size_t>(source) * kKvHeads + kv_head) * kHeadDim + dim;
            const float probability = scores[static_cast<size_t>(q_head) * max_context + source];
            weighted = fmaf(probability, values[value_index], weighted);
        }
        output[output_base + dim] = weighted / denominators[q_head];
    }
}

__global__ void add_in_place_f32(float* lhs, const float* rhs, unsigned long long elements) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        lhs[index] = lhs[index] + rhs[index];
    }
}

__global__ void silu_mul_in_place_f32(float* gate, const float* up, unsigned long long elements) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const float x = gate[index];
    const float sigmoid = x >= 0.0f ? 1.0f / (1.0f + expf(-x))
                                    : expf(x) / (1.0f + expf(x));
    gate[index] = (x * sigmoid) * up[index];
}

__global__ void mark_nonfinite_f32(const float* values, unsigned long long elements, unsigned int* flag) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements && !isfinite(values[index])) {
        atomicExch(flag, 1u);
    }
}

struct Session {
    struct ullm_sq8_fp32_gpu_reference_gfx1201_device_info device_info{};
    hipStream_t stream = nullptr;
    hipblasHandle_t blas = nullptr;
    size_t max_context = 0u;
    size_t position = 0u;
    bool finalized = false;
    bool poisoned = false;

    std::unordered_map<std::string, QuantizedWeight> weights;
    Bf16Upload embedding;
    Bf16Upload lm_head_bf16;
    DeviceBuffer lm_head_f32;
    std::array<DeviceLayerNorms, kLayers> norms;
    DeviceBuffer final_norm;

    DeviceBuffer workspace_f32;
    DeviceBuffer hidden;
    DeviceBuffer input_norm;
    DeviceBuffer q;
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer q_norm;
    DeviceBuffer k_norm;
    DeviceBuffer q_rope;
    DeviceBuffer k_rope;
    DeviceBuffer attention;
    DeviceBuffer attention_output;
    DeviceBuffer post_attention_norm;
    DeviceBuffer gate;
    DeviceBuffer up;
    DeviceBuffer down;
    DeviceBuffer final_hidden;
    DeviceBuffer logits;
    DeviceBuffer keys;
    DeviceBuffer values;
    DeviceBuffer scores;
    DeviceBuffer max_scores;
    DeviceBuffer denominators;
    DeviceBuffer finite_flag;

    ~Session() {
        // DeviceBuffer members release after this destructor body.  Best
        // effort selection keeps teardown on the admitted HIP device even if
        // the caller destroys a session from a different host thread.
        (void)hipSetDevice(0);
        if (blas != nullptr) {
            (void)hipblasDestroy(blas);
        }
        if (stream != nullptr) {
            (void)hipStreamDestroy(stream);
        }
    }

    void create(size_t requested_max_context) {
        if (requested_max_context == 0u || requested_max_context > 4096u) {
            throw std::runtime_error("GPU F32 reference max_context must be in 1..=4096");
        }
        max_context = requested_max_context;
        validate_visible_r9700(&device_info);
        hip_check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");
        try {
            hipblas_check(hipblasCreate(&blas), "hipblasCreate");
            hipblas_check(hipblasSetStream(blas, stream), "hipblasSetStream");
            hipblas_check(hipblasSetPointerMode(blas, HIPBLAS_POINTER_MODE_HOST),
                          "hipblasSetPointerMode host");
            hipblas_check(hipblasSetMathMode(blas, HIPBLAS_DEFAULT_MATH),
                          "hipblasSetMathMode default");
            hipblas_check(hipblasSetAtomicsMode(blas, HIPBLAS_ATOMICS_NOT_ALLOWED),
                          "hipblasSetAtomicsMode deterministic");
        } catch (...) {
            if (blas != nullptr) {
                (void)hipblasDestroy(blas);
                blas = nullptr;
            }
            (void)hipStreamDestroy(stream);
            stream = nullptr;
            throw;
        }
    }

    // HIP's current-device selection is thread-local.  Every public method
    // re-selects the only admitted filtered device before touching a stream,
    // allocation, or hipBLAS handle.
    void select_r9700() const {
        hip_check(hipSetDevice(0), "hipSetDevice(0)");
    }

    void refresh_free_memory() {
        select_r9700();
        size_t free_bytes = 0u;
        size_t total_bytes = 0u;
        hip_check(hipMemGetInfo(&free_bytes, &total_bytes), "hipMemGetInfo");
        device_info.free_global_mem_bytes = static_cast<uint64_t>(free_bytes);
        // hipMemGetInfo is the allocator's current view and can differ in
        // representation from hipDeviceProp_t::totalGlobalMem.  Record the
        // allocator value rather than treating that harmless distinction as a
        // device-admission failure.
        device_info.total_global_mem_bytes = static_cast<uint64_t>(total_bytes);
    }

    void reserve_weight(
        std::string_view name,
        size_t rows,
        size_t cols,
        const void* scales_bf16,
        size_t scale_bytes) {
        select_r9700();
        if (finalized) {
            throw std::runtime_error("GPU F32 reference rejects weight upload after finalize");
        }
        if (name.empty() || name.size() > 256u || rows == 0u || cols == 0u ||
            rows % kFp8Block != 0u || cols % kFp8Block != 0u || scales_bf16 == nullptr) {
            throw std::runtime_error("GPU F32 reference received an invalid SQ8 weight descriptor");
        }
        const size_t expected_scales = checked_product(rows / kFp8Block, cols / kFp8Block,
                                                       "SQ8 scale grid");
        const size_t expected_scale_bytes = checked_bytes(expected_scales, sizeof(uint16_t), "SQ8 scales");
        if (scale_bytes != expected_scale_bytes) {
            throw std::runtime_error("GPU F32 reference SQ8 weight BF16 scale byte count mismatch");
        }
        const std::string owned_name(name);
        if (weights.contains(owned_name)) {
            throw std::runtime_error("GPU F32 reference received duplicate SQ8 weight " + owned_name);
        }
        QuantizedWeight weight{};
        const size_t elements = checked_product(rows, cols, "SQ8 payload");
        weight.payload.allocate(elements, owned_name + " payload");
        weight.scales_bf16.allocate(expected_scale_bytes, owned_name + " BF16 scales");
        hip_check(hipMemcpy(weight.scales_bf16.data(),
                            scales_bf16,
                            weight.scales_bf16.bytes(),
                            hipMemcpyHostToDevice),
                  owned_name + " BF16 scale copy");
        weight.rows = rows;
        weight.cols = cols;
        weights.emplace(owned_name, std::move(weight));
    }

    void upload_weight_chunk(
        std::string_view name,
        size_t offset_bytes,
        const void* source,
        size_t bytes) {
        select_r9700();
        if (source == nullptr || bytes == 0u) {
            throw std::runtime_error("GPU F32 reference SQ8 weight upload has no source bytes");
        }
        auto it = weights.find(std::string(name));
        if (it == weights.end()) {
            throw std::runtime_error("GPU F32 reference upload names an unreserved SQ8 weight");
        }
        QuantizedWeight& weight = it->second;
        if (offset_bytes != weight.uploaded_bytes || bytes > weight.payload.bytes() - offset_bytes) {
            throw std::runtime_error("GPU F32 reference SQ8 upload is not contiguous or exceeds payload");
        }
        auto* destination = static_cast<unsigned char*>(weight.payload.data()) + offset_bytes;
        hip_check(hipMemcpy(destination, source, bytes, hipMemcpyHostToDevice), "SQ8 payload chunk copy");
        weight.uploaded_bytes += bytes;
    }

    Bf16Upload& bf16_slot(std::string_view slot) {
        if (slot == "embedding") {
            return embedding;
        }
        if (slot == "lm_head") {
            return lm_head_bf16;
        }
        throw std::runtime_error("GPU F32 reference has no such BF16 upload slot");
    }

    void reserve_bf16(std::string_view slot, size_t elements) {
        select_r9700();
        if (finalized || elements != checked_product(kVocab, kHidden, "BF16 tensor elements")) {
            throw std::runtime_error("GPU F32 reference BF16 tensor shape is invalid");
        }
        Bf16Upload& upload = bf16_slot(slot);
        if (upload.payload.allocated()) {
            throw std::runtime_error("GPU F32 reference BF16 tensor was reserved twice");
        }
        upload.expected_bytes = checked_bytes(elements, sizeof(uint16_t), "BF16 tensor");
        upload.payload.allocate(upload.expected_bytes, std::string(slot));
    }

    void upload_bf16_chunk(
        std::string_view slot,
        size_t offset_bytes,
        const void* source,
        size_t bytes) {
        select_r9700();
        if (source == nullptr || bytes == 0u) {
            throw std::runtime_error("GPU F32 reference BF16 upload has no source bytes");
        }
        Bf16Upload& upload = bf16_slot(slot);
        if (!upload.payload.allocated() || offset_bytes != upload.uploaded_bytes ||
            bytes > upload.expected_bytes - offset_bytes) {
            throw std::runtime_error("GPU F32 reference BF16 upload is invalid or non-contiguous");
        }
        auto* destination = static_cast<unsigned char*>(upload.payload.data()) + offset_bytes;
        hip_check(hipMemcpy(destination, source, bytes, hipMemcpyHostToDevice), "BF16 payload chunk copy");
        upload.uploaded_bytes += bytes;
    }

    static void copy_norm(DeviceBuffer& destination, const void* source, size_t elements,
                          std::string_view label) {
        if (source == nullptr) {
            throw std::runtime_error(std::string(label) + " source is null");
        }
        destination.allocate(checked_bytes(elements, sizeof(uint16_t), label), label);
        hip_check(hipMemcpy(destination.data(), source, destination.bytes(), hipMemcpyHostToDevice),
                  label);
    }

    void upload_norms(
        size_t layer_index,
        const void* input,
        const void* post,
        const void* q_weight,
        const void* k_weight) {
        select_r9700();
        if (finalized || layer_index >= kLayers) {
            throw std::runtime_error("GPU F32 reference norm layer index is invalid");
        }
        DeviceLayerNorms& layer = norms[layer_index];
        if (layer.uploaded) {
            throw std::runtime_error("GPU F32 reference norms were uploaded twice");
        }
        copy_norm(layer.input, input, kHidden, "input RMSNorm copy");
        copy_norm(layer.post_attention, post, kHidden, "post-attention RMSNorm copy");
        copy_norm(layer.q, q_weight, kHeadDim, "Q RMSNorm copy");
        copy_norm(layer.k, k_weight, kHeadDim, "K RMSNorm copy");
        layer.uploaded = true;
    }

    void upload_final_norm(const void* source) {
        select_r9700();
        if (finalized || final_norm.allocated()) {
            throw std::runtime_error("GPU F32 reference final norm upload is invalid");
        }
        copy_norm(final_norm, source, kHidden, "final RMSNorm copy");
    }

    const QuantizedWeight& weight(std::string_view name) const {
        const auto it = weights.find(std::string(name));
        if (it == weights.end()) {
            throw std::runtime_error("GPU F32 reference misses required SQ8 weight " +
                                     std::string(name));
        }
        if (it->second.uploaded_bytes != it->second.payload.bytes()) {
            throw std::runtime_error("GPU F32 reference SQ8 weight upload is incomplete for " +
                                     std::string(name));
        }
        return it->second;
    }

    void validate_model_contract() const {
        if (weights.size() != kLayers * 7u) {
            throw std::runtime_error("GPU F32 reference requires exactly 280 canonical SQ8 weights");
        }
        if (!embedding.payload.allocated() || embedding.uploaded_bytes != embedding.expected_bytes ||
            !lm_head_bf16.payload.allocated() || lm_head_bf16.uploaded_bytes != lm_head_bf16.expected_bytes ||
            !final_norm.allocated()) {
            throw std::runtime_error("GPU F32 reference has incomplete BF16 model uploads");
        }
        for (size_t layer_index = 0u; layer_index < kLayers; ++layer_index) {
            if (!norms[layer_index].uploaded) {
                throw std::runtime_error("GPU F32 reference misses layer norms");
            }
            const std::string prefix = "model.layers." + std::to_string(layer_index) + ".";
            const std::array<std::tuple<std::string, size_t, size_t>, 7> expected = {{
                {prefix + "self_attn.q_proj.weight", kQWidth, kHidden},
                {prefix + "self_attn.k_proj.weight", kKvWidth, kHidden},
                {prefix + "self_attn.v_proj.weight", kKvWidth, kHidden},
                {prefix + "self_attn.o_proj.weight", kHidden, kHidden},
                {prefix + "mlp.gate_proj.weight", kIntermediate, kHidden},
                {prefix + "mlp.up_proj.weight", kIntermediate, kHidden},
                {prefix + "mlp.down_proj.weight", kHidden, kIntermediate},
            }};
            for (const auto& [name, rows, cols] : expected) {
                const QuantizedWeight& candidate = weight(name);
                if (candidate.rows != rows || candidate.cols != cols) {
                    throw std::runtime_error("GPU F32 reference SQ8 weight shape mismatch for " + name);
                }
            }
        }
    }

    void allocate_execution_buffers() {
        size_t largest_weight_elements = 0u;
        for (const auto& [_, candidate] : weights) {
            largest_weight_elements = std::max(
                largest_weight_elements,
                checked_product(candidate.rows, candidate.cols, "SQ8 F32 workspace"));
        }
        workspace_f32.allocate(checked_bytes(largest_weight_elements, sizeof(float), "F32 workspace"),
                               "F32 workspace");
        hidden.allocate(checked_bytes(kHidden, sizeof(float), "hidden"), "hidden");
        input_norm.allocate(checked_bytes(kHidden, sizeof(float), "input norm"), "input norm");
        q.allocate(checked_bytes(kQWidth, sizeof(float), "Q"), "Q");
        k.allocate(checked_bytes(kKvWidth, sizeof(float), "K"), "K");
        v.allocate(checked_bytes(kKvWidth, sizeof(float), "V"), "V");
        q_norm.allocate(checked_bytes(kQWidth, sizeof(float), "Q norm"), "Q norm");
        k_norm.allocate(checked_bytes(kKvWidth, sizeof(float), "K norm"), "K norm");
        q_rope.allocate(checked_bytes(kQWidth, sizeof(float), "Q RoPE"), "Q RoPE");
        k_rope.allocate(checked_bytes(kKvWidth, sizeof(float), "K RoPE"), "K RoPE");
        attention.allocate(checked_bytes(kQWidth, sizeof(float), "attention"), "attention");
        attention_output.allocate(checked_bytes(kHidden, sizeof(float), "attention output"),
                                  "attention output");
        post_attention_norm.allocate(checked_bytes(kHidden, sizeof(float), "post attention norm"),
                                     "post attention norm");
        gate.allocate(checked_bytes(kIntermediate, sizeof(float), "gate"), "gate");
        up.allocate(checked_bytes(kIntermediate, sizeof(float), "up"), "up");
        down.allocate(checked_bytes(kHidden, sizeof(float), "down"), "down");
        final_hidden.allocate(checked_bytes(kHidden, sizeof(float), "final hidden"), "final hidden");
        logits.allocate(checked_bytes(kVocab, sizeof(float), "logits"), "logits");
        const size_t cache_elements = checked_product(
            checked_product(kLayers, max_context, "KV cache layers"), kKvWidth, "KV cache");
        keys.allocate(checked_bytes(cache_elements, sizeof(float), "F32 K cache"), "F32 K cache");
        values.allocate(checked_bytes(cache_elements, sizeof(float), "F32 V cache"), "F32 V cache");
        scores.allocate(checked_bytes(checked_product(kQHeads, max_context, "attention scores"),
                                      sizeof(float), "attention scores"),
                        "attention scores");
        max_scores.allocate(checked_bytes(kQHeads, sizeof(float), "attention maxima"),
                            "attention maxima");
        denominators.allocate(checked_bytes(kQHeads, sizeof(float), "attention denominators"),
                              "attention denominators");
        finite_flag.allocate(sizeof(unsigned int), "finite flag");
    }

    void finalize() {
        select_r9700();
        if (finalized) {
            throw std::runtime_error("GPU F32 reference was finalized twice");
        }
        validate_model_contract();
        lm_head_f32.allocate(checked_bytes(checked_product(kVocab, kHidden, "LM head"),
                                           sizeof(float), "F32 LM head"),
                             "F32 LM head");
        const size_t lm_head_elements = checked_product(kVocab, kHidden, "LM head elements");
        hipLaunchKernelGGL(bf16_vector_to_f32,
                           dim3(grid_for(lm_head_elements, "LM head F32 conversion")),
                           dim3(256u),
                           0u,
                           stream,
                           static_cast<const uint16_t*>(lm_head_bf16.payload.data()),
                           static_cast<float*>(lm_head_f32.data()),
                           static_cast<unsigned long long>(lm_head_elements));
        hip_check(hipGetLastError(), "LM head BF16-to-F32 launch");
        hip_check(hipStreamSynchronize(stream), "LM head BF16-to-F32 synchronization");
        lm_head_bf16.payload.release_checked("LM head BF16 release");
        allocate_execution_buffers();
        refresh_free_memory();
        finalized = true;
    }

    void launch_rmsnorm(
        const float* input_values,
        const uint16_t* weight_values,
        float* output_values,
        unsigned int rows,
        unsigned int columns,
        std::string_view label) {
        hipLaunchKernelGGL(rmsnorm_serial_f32,
                           dim3((rows + 63u) / 64u),
                           dim3(64u),
                           0u,
                           stream,
                           input_values,
                           weight_values,
                           output_values,
                           rows,
                           columns);
        hip_check(hipGetLastError(), std::string(label) + " RMSNorm launch");
    }

    void project(std::string_view tensor_name, const float* input_values, float* output_values) {
        const QuantizedWeight& candidate = weight(tensor_name);
        const size_t elements = checked_product(candidate.rows, candidate.cols, "projection dequant");
        if (checked_bytes(elements, sizeof(float), "projection dequant") > workspace_f32.bytes()) {
            throw std::runtime_error("GPU F32 reference projection workspace is too small");
        }
        hipLaunchKernelGGL(dequant_sq8_ocp_block128_to_f32,
                           dim3(grid_for(elements, "SQ8 F32 dequant")),
                           dim3(256u),
                           0u,
                           stream,
                           static_cast<const unsigned char*>(candidate.payload.data()),
                           static_cast<const uint16_t*>(candidate.scales_bf16.data()),
                           static_cast<float*>(workspace_f32.data()),
                           static_cast<unsigned long long>(candidate.rows),
                           static_cast<unsigned long long>(candidate.cols));
        hip_check(hipGetLastError(), std::string(tensor_name) + " OCP-to-F32 launch");

        if (candidate.rows > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            candidate.cols > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("GPU F32 reference projection exceeds hipBLAS int dimensions");
        }
        // The canonical payload is row-major W[N,K].  The same bytes are a
        // column-major KxN matrix, so transposing that operand produces
        // W[N,K] * x[K,1] in the ordinary SGEMM interface.
        const float alpha = 1.0f;
        const float beta = 0.0f;
        hipblas_check(hipblasSgemm(blas,
                                   HIPBLAS_OP_T,
                                   HIPBLAS_OP_N,
                                   static_cast<int>(candidate.rows),
                                   1,
                                   static_cast<int>(candidate.cols),
                                   &alpha,
                                   static_cast<const float*>(workspace_f32.data()),
                                   static_cast<int>(candidate.cols),
                                   input_values,
                                   static_cast<int>(candidate.cols),
                                   &beta,
                                   output_values,
                                   static_cast<int>(candidate.rows)),
                      std::string(tensor_name) + " hipblasSgemm");
    }

    void lm_head_project(const float* input_values) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        hipblas_check(hipblasSgemm(blas,
                                   HIPBLAS_OP_T,
                                   HIPBLAS_OP_N,
                                   static_cast<int>(kVocab),
                                   1,
                                   static_cast<int>(kHidden),
                                   &alpha,
                                   static_cast<const float*>(lm_head_f32.data()),
                                   static_cast<int>(kHidden),
                                   input_values,
                                   static_cast<int>(kHidden),
                                   &beta,
                                   static_cast<float*>(logits.data()),
                                   static_cast<int>(kVocab)),
                      "LM head hipblasSgemm");
    }

    void ensure_finite(const float* source, size_t elements, std::string_view label) {
        hip_check(hipMemsetAsync(finite_flag.data(), 0, sizeof(unsigned int), stream),
                  std::string(label) + " finite flag zero");
        hipLaunchKernelGGL(mark_nonfinite_f32,
                           dim3(grid_for(elements, "finite check")),
                           dim3(256u),
                           0u,
                           stream,
                           source,
                           static_cast<unsigned long long>(elements),
                           static_cast<unsigned int*>(finite_flag.data()));
        hip_check(hipGetLastError(), std::string(label) + " finite check launch");
        unsigned int host_flag = 0u;
        hip_check(hipMemcpyAsync(&host_flag,
                                 finite_flag.data(),
                                 sizeof(host_flag),
                                 hipMemcpyDeviceToHost,
                                 stream),
                  std::string(label) + " finite flag copy");
        hip_check(hipStreamSynchronize(stream), std::string(label) + " finite check synchronization");
        if (host_flag != 0u) {
            throw std::runtime_error(std::string(label) + " contains a non-finite F32 value");
        }
    }

    void forward(
        uint32_t token_id,
        float* logits_host,
        size_t logits_elements,
        float* final_hidden_host,
        size_t final_hidden_elements,
        float* layer_hidden_host,
        size_t layer_hidden_elements) {
        select_r9700();
        if (!finalized) {
            throw std::runtime_error("GPU F32 reference cannot forward before finalize");
        }
        if (poisoned) {
            throw std::runtime_error("GPU F32 reference session is poisoned after a failed forward");
        }
        if (token_id >= kVocab || position >= max_context || logits_host == nullptr ||
            final_hidden_host == nullptr || layer_hidden_host == nullptr || logits_elements != kVocab ||
            final_hidden_elements != kHidden || layer_hidden_elements != kLayers * kHidden) {
            throw std::runtime_error("GPU F32 reference forward arguments are invalid");
        }
        struct PoisonOnFailure {
            bool& target;
            bool complete = false;
            ~PoisonOnFailure() {
                if (!complete) {
                    target = true;
                }
            }
        } poison_on_failure{poisoned};
        hipLaunchKernelGGL(embedding_row_bf16_to_f32,
                           dim3((kHidden + 255u) / 256u),
                           dim3(256u),
                           0u,
                           stream,
                           static_cast<const uint16_t*>(embedding.payload.data()),
                           token_id,
                           static_cast<float*>(hidden.data()));
        hip_check(hipGetLastError(), "embedding BF16-to-F32 launch");

        const unsigned int tokens = static_cast<unsigned int>(position + 1u);
        for (size_t layer_index = 0u; layer_index < kLayers; ++layer_index) {
            const DeviceLayerNorms& layer_norms = norms[layer_index];
            const std::string prefix = "model.layers." + std::to_string(layer_index) + ".";
            launch_rmsnorm(static_cast<const float*>(hidden.data()),
                           static_cast<const uint16_t*>(layer_norms.input.data()),
                           static_cast<float*>(input_norm.data()),
                           1u,
                           static_cast<unsigned int>(kHidden),
                           "input");
            project(prefix + "self_attn.q_proj.weight",
                    static_cast<const float*>(input_norm.data()), static_cast<float*>(q.data()));
            project(prefix + "self_attn.k_proj.weight",
                    static_cast<const float*>(input_norm.data()), static_cast<float*>(k.data()));
            project(prefix + "self_attn.v_proj.weight",
                    static_cast<const float*>(input_norm.data()), static_cast<float*>(v.data()));
            launch_rmsnorm(static_cast<const float*>(q.data()),
                           static_cast<const uint16_t*>(layer_norms.q.data()),
                           static_cast<float*>(q_norm.data()),
                           static_cast<unsigned int>(kQHeads),
                           static_cast<unsigned int>(kHeadDim),
                           "Q");
            launch_rmsnorm(static_cast<const float*>(k.data()),
                           static_cast<const uint16_t*>(layer_norms.k.data()),
                           static_cast<float*>(k_norm.data()),
                           static_cast<unsigned int>(kKvHeads),
                           static_cast<unsigned int>(kHeadDim),
                           "K");
            hipLaunchKernelGGL(rope_split_half_f32,
                               dim3((kQHeads * (kHeadDim / 2u) + 255u) / 256u),
                               dim3(256u),
                               0u,
                               stream,
                               static_cast<const float*>(q_norm.data()),
                               static_cast<float*>(q_rope.data()),
                               static_cast<unsigned int>(kQHeads),
                               static_cast<unsigned int>(position));
            hip_check(hipGetLastError(), "Q RoPE launch");
            hipLaunchKernelGGL(rope_split_half_f32,
                               dim3((kKvHeads * (kHeadDim / 2u) + 255u) / 256u),
                               dim3(256u),
                               0u,
                               stream,
                               static_cast<const float*>(k_norm.data()),
                               static_cast<float*>(k_rope.data()),
                               static_cast<unsigned int>(kKvHeads),
                               static_cast<unsigned int>(position));
            hip_check(hipGetLastError(), "K RoPE launch");
            const size_t layer_base = checked_product(layer_index, max_context, "KV layer base") * kKvWidth;
            hipLaunchKernelGGL(copy_kv_f32,
                               dim3((kKvWidth + 255u) / 256u),
                               dim3(256u),
                               0u,
                               stream,
                               static_cast<const float*>(k_rope.data()),
                               static_cast<const float*>(v.data()),
                               static_cast<float*>(keys.data()),
                               static_cast<float*>(values.data()),
                               static_cast<unsigned long long>(layer_base),
                               static_cast<unsigned long long>(position));
            hip_check(hipGetLastError(), "F32 KV write launch");
            const auto* layer_keys = static_cast<const float*>(keys.data()) + layer_base;
            const auto* layer_values = static_cast<const float*>(values.data()) + layer_base;
            hipLaunchKernelGGL(attention_scores_and_max_f32,
                               dim3(1u),
                               dim3(64u),
                               0u,
                               stream,
                               static_cast<const float*>(q_rope.data()),
                               layer_keys,
                               static_cast<float*>(scores.data()),
                               static_cast<float*>(max_scores.data()),
                               tokens,
                               static_cast<unsigned int>(max_context));
            hip_check(hipGetLastError(), "causal attention score launch");
            hipLaunchKernelGGL(attention_exp_and_sum_f32,
                               dim3(1u),
                               dim3(64u),
                               0u,
                               stream,
                               static_cast<float*>(scores.data()),
                               static_cast<const float*>(max_scores.data()),
                               static_cast<float*>(denominators.data()),
                               tokens,
                               static_cast<unsigned int>(max_context));
            hip_check(hipGetLastError(), "causal attention softmax launch");
            hipLaunchKernelGGL(attention_weighted_values_f32,
                               dim3(1u),
                               dim3(64u),
                               0u,
                               stream,
                               static_cast<const float*>(scores.data()),
                               static_cast<const float*>(denominators.data()),
                               layer_values,
                               static_cast<float*>(attention.data()),
                               tokens,
                               static_cast<unsigned int>(max_context));
            hip_check(hipGetLastError(), "causal attention value launch");
            project(prefix + "self_attn.o_proj.weight",
                    static_cast<const float*>(attention.data()),
                    static_cast<float*>(attention_output.data()));
            hipLaunchKernelGGL(add_in_place_f32,
                               dim3(grid_for(kHidden, "attention residual")),
                               dim3(256u),
                               0u,
                               stream,
                               static_cast<float*>(hidden.data()),
                               static_cast<const float*>(attention_output.data()),
                               static_cast<unsigned long long>(kHidden));
            hip_check(hipGetLastError(), "attention residual launch");
            launch_rmsnorm(static_cast<const float*>(hidden.data()),
                           static_cast<const uint16_t*>(layer_norms.post_attention.data()),
                           static_cast<float*>(post_attention_norm.data()),
                           1u,
                           static_cast<unsigned int>(kHidden),
                           "post attention");
            project(prefix + "mlp.gate_proj.weight",
                    static_cast<const float*>(post_attention_norm.data()), static_cast<float*>(gate.data()));
            project(prefix + "mlp.up_proj.weight",
                    static_cast<const float*>(post_attention_norm.data()), static_cast<float*>(up.data()));
            hipLaunchKernelGGL(silu_mul_in_place_f32,
                               dim3(grid_for(kIntermediate, "SiLU")),
                               dim3(256u),
                               0u,
                               stream,
                               static_cast<float*>(gate.data()),
                               static_cast<const float*>(up.data()),
                               static_cast<unsigned long long>(kIntermediate));
            hip_check(hipGetLastError(), "SiLU multiply launch");
            project(prefix + "mlp.down_proj.weight",
                    static_cast<const float*>(gate.data()), static_cast<float*>(down.data()));
            hipLaunchKernelGGL(add_in_place_f32,
                               dim3(grid_for(kHidden, "MLP residual")),
                               dim3(256u),
                               0u,
                               stream,
                               static_cast<float*>(hidden.data()),
                               static_cast<const float*>(down.data()),
                               static_cast<unsigned long long>(kHidden));
            hip_check(hipGetLastError(), "MLP residual launch");
            ensure_finite(static_cast<const float*>(hidden.data()), kHidden, "post-layer hidden");
            hip_check(hipMemcpyAsync(layer_hidden_host + layer_index * kHidden,
                                     hidden.data(),
                                     checked_bytes(kHidden, sizeof(float), "layer hidden copy"),
                                     hipMemcpyDeviceToHost,
                                     stream),
                      "layer hidden copy");
        }
        launch_rmsnorm(static_cast<const float*>(hidden.data()),
                       static_cast<const uint16_t*>(final_norm.data()),
                       static_cast<float*>(final_hidden.data()),
                       1u,
                       static_cast<unsigned int>(kHidden),
                       "final");
        ensure_finite(static_cast<const float*>(final_hidden.data()), kHidden, "final hidden");
        lm_head_project(static_cast<const float*>(final_hidden.data()));
        ensure_finite(static_cast<const float*>(logits.data()), kVocab, "logits");
        hip_check(hipMemcpyAsync(final_hidden_host,
                                 final_hidden.data(),
                                 checked_bytes(kHidden, sizeof(float), "final hidden copy"),
                                 hipMemcpyDeviceToHost,
                                 stream),
                  "final hidden copy");
        hip_check(hipMemcpyAsync(logits_host,
                                 logits.data(),
                                 checked_bytes(kVocab, sizeof(float), "logits copy"),
                                 hipMemcpyDeviceToHost,
                                 stream),
                  "logits copy");
        hip_check(hipStreamSynchronize(stream), "GPU F32 reference forward synchronization");
        ++position;
        poison_on_failure.complete = true;
    }

    void reset() {
        select_r9700();
        if (!finalized) {
            throw std::runtime_error("GPU F32 reference cannot reset before finalize");
        }
        hip_check(hipStreamSynchronize(stream), "GPU F32 reference reset synchronization");
        position = 0u;
        poisoned = false;
    }
};

Session& require_session(struct ullm_sq8_fp32_gpu_reference_gfx1201_session* opaque);
const Session& require_session(const struct ullm_sq8_fp32_gpu_reference_gfx1201_session* opaque);

} // namespace

struct ullm_sq8_fp32_gpu_reference_gfx1201_session {
    Session value;
};

namespace {

Session& require_session(struct ullm_sq8_fp32_gpu_reference_gfx1201_session* opaque) {
    if (opaque == nullptr) {
        throw std::runtime_error("GPU F32 reference session is null");
    }
    return opaque->value;
}

const Session& require_session(const struct ullm_sq8_fp32_gpu_reference_gfx1201_session* opaque) {
    if (opaque == nullptr) {
        throw std::runtime_error("GPU F32 reference session is null");
    }
    return opaque->value;
}

template <typename Function>
int invoke(Function&& function, char* error, size_t error_capacity) {
    try {
        function();
        write_error(error, error_capacity, "");
        return 1;
    } catch (const std::exception& exception) {
        write_error(error, error_capacity, exception.what());
        return 0;
    } catch (...) {
        write_error(error, error_capacity, "GPU F32 reference failed with an unknown exception");
        return 0;
    }
}

} // namespace

extern "C" int ullm_sq8_fp32_gpu_reference_gfx1201_create(
    size_t max_context,
    struct ullm_sq8_fp32_gpu_reference_gfx1201_session** session,
    char* error,
    size_t error_capacity) {
    return invoke(
        [&] {
            if (session == nullptr || *session != nullptr) {
                throw std::runtime_error("GPU F32 reference create output pointer is invalid");
            }
            auto created = std::make_unique<ullm_sq8_fp32_gpu_reference_gfx1201_session>();
            created->value.create(max_context);
            *session = created.release();
        },
        error,
        error_capacity);
}

extern "C" void ullm_sq8_fp32_gpu_reference_gfx1201_destroy(
    struct ullm_sq8_fp32_gpu_reference_gfx1201_session* session) {
    delete session;
}

extern "C" int ullm_sq8_fp32_gpu_reference_gfx1201_device_info(
    const struct ullm_sq8_fp32_gpu_reference_gfx1201_session* session,
    struct ullm_sq8_fp32_gpu_reference_gfx1201_device_info* info,
    char* error,
    size_t error_capacity) {
    return invoke(
        [&] {
            if (info == nullptr) {
                throw std::runtime_error("GPU F32 reference device info output is null");
            }
            const Session& value = require_session(session);
            *info = value.device_info;
            Session& mutable_value = const_cast<Session&>(value);
            mutable_value.refresh_free_memory();
            *info = mutable_value.device_info;
        },
        error,
        error_capacity);
}

extern "C" int ullm_sq8_fp32_gpu_reference_gfx1201_reserve_sq8_weight(
    struct ullm_sq8_fp32_gpu_reference_gfx1201_session* session,
    const char* tensor_name,
    size_t rows,
    size_t cols,
    const void* scales_bf16,
    size_t scale_bytes,
    char* error,
    size_t error_capacity) {
    return invoke(
        [&] {
            if (tensor_name == nullptr) {
                throw std::runtime_error("GPU F32 reference SQ8 tensor name is null");
            }
            require_session(session).reserve_weight(
                tensor_name, rows, cols, scales_bf16, scale_bytes);
        },
        error,
        error_capacity);
}

extern "C" int ullm_sq8_fp32_gpu_reference_gfx1201_upload_sq8_weight_chunk(
    struct ullm_sq8_fp32_gpu_reference_gfx1201_session* session,
    const char* tensor_name,
    size_t offset_bytes,
    const void* source,
    size_t bytes,
    char* error,
    size_t error_capacity) {
    return invoke(
        [&] {
            if (tensor_name == nullptr) {
                throw std::runtime_error("GPU F32 reference SQ8 tensor name is null");
            }
            require_session(session).upload_weight_chunk(tensor_name, offset_bytes, source, bytes);
        },
        error,
        error_capacity);
}

extern "C" int ullm_sq8_fp32_gpu_reference_gfx1201_reserve_bf16_tensor(
    struct ullm_sq8_fp32_gpu_reference_gfx1201_session* session,
    const char* slot,
    size_t elements,
    char* error,
    size_t error_capacity) {
    return invoke(
        [&] {
            if (slot == nullptr) {
                throw std::runtime_error("GPU F32 reference BF16 slot is null");
            }
            require_session(session).reserve_bf16(slot, elements);
        },
        error,
        error_capacity);
}

extern "C" int ullm_sq8_fp32_gpu_reference_gfx1201_upload_bf16_tensor_chunk(
    struct ullm_sq8_fp32_gpu_reference_gfx1201_session* session,
    const char* slot,
    size_t offset_bytes,
    const void* source,
    size_t bytes,
    char* error,
    size_t error_capacity) {
    return invoke(
        [&] {
            if (slot == nullptr) {
                throw std::runtime_error("GPU F32 reference BF16 slot is null");
            }
            require_session(session).upload_bf16_chunk(slot, offset_bytes, source, bytes);
        },
        error,
        error_capacity);
}

extern "C" int ullm_sq8_fp32_gpu_reference_gfx1201_upload_layer_norms(
    struct ullm_sq8_fp32_gpu_reference_gfx1201_session* session,
    size_t layer_index,
    const void* input_norm,
    const void* post_attention_norm,
    const void* q_norm,
    const void* k_norm,
    char* error,
    size_t error_capacity) {
    return invoke(
        [&] {
            require_session(session).upload_norms(
                layer_index, input_norm, post_attention_norm, q_norm, k_norm);
        },
        error,
        error_capacity);
}

extern "C" int ullm_sq8_fp32_gpu_reference_gfx1201_upload_final_norm(
    struct ullm_sq8_fp32_gpu_reference_gfx1201_session* session,
    const void* final_norm,
    char* error,
    size_t error_capacity) {
    return invoke(
        [&] { require_session(session).upload_final_norm(final_norm); }, error, error_capacity);
}

extern "C" int ullm_sq8_fp32_gpu_reference_gfx1201_finalize_model(
    struct ullm_sq8_fp32_gpu_reference_gfx1201_session* session,
    char* error,
    size_t error_capacity) {
    return invoke(
        [&] { require_session(session).finalize(); }, error, error_capacity);
}

extern "C" int ullm_sq8_fp32_gpu_reference_gfx1201_forward(
    struct ullm_sq8_fp32_gpu_reference_gfx1201_session* session,
    uint32_t token_id,
    float* logits_f32,
    size_t logits_elements,
    float* final_hidden_f32,
    size_t final_hidden_elements,
    float* layer_hidden_f32,
    size_t layer_hidden_elements,
    char* error,
    size_t error_capacity) {
    return invoke(
        [&] {
            require_session(session).forward(token_id,
                                             logits_f32,
                                             logits_elements,
                                             final_hidden_f32,
                                             final_hidden_elements,
                                             layer_hidden_f32,
                                             layer_hidden_elements);
        },
        error,
        error_capacity);
}

extern "C" int ullm_sq8_fp32_gpu_reference_gfx1201_reset(
    struct ullm_sq8_fp32_gpu_reference_gfx1201_session* session,
    char* error,
    size_t error_capacity) {
    return invoke(
        [&] { require_session(session).reset(); }, error, error_capacity);
}
