// Phase 78 standalone gfx1201 NVFP4 -> FP8 staging candidate.
//
// The production path is intentionally untouched by this experiment.  It
// compares the existing ID64 packed-E2M1 plus block-E4M3 scale descriptor with
// a two-stage candidate:
//
//   packed E2M1 + E4M3 block scale --(one vectorized staging pass)--> E4M3FN
//   E4M3FN A/B --(native hipBLASLt FP8 GEMM, BF16 output)--> D
//
// Tensor-global scales are represented by alpha in both paths.  The candidate
// does not install matrix-scale descriptors: its block scale has already been
// consumed by the staging kernel.  A stage thread reads one packed 16-value
// block and writes four dwords (16 FP8 values), so there is no per-tile
// re-encoding.  The explicit +/-448 clamp is deliberate; HIP's FP8x2 helper
// has an overflow edge case at that boundary.
//
// This is a measurement/evidence probe, not a production implementation.

#include "low_precision_block_codec.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_version.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr uint64_t kBlockK = 16U;
constexpr uint32_t kStageThreads = 256U;
constexpr int kRequestedAlgorithms = 32;
constexpr int kWarmups = 3;
constexpr int kMeasured = 10;
constexpr uint64_t kWorkspaceLimit = UINT64_C(1) << 30U;
constexpr uint32_t kSeed = UINT32_C(0x243f6a88);
constexpr float kWeightGlobal = 0.75F;
constexpr float kInputGlobal = 1.125F;
constexpr float kAlpha = kWeightGlobal * kInputGlobal;

struct Shape final {
  uint64_t m;
  uint64_t k;
  uint64_t n;
  const char *name;
  uint32_t occurrence;
};

// The six exact Qwen3.8-27B NVFP4 MLP shapes.  The M multiplicity is the
// useful decode/prefill weighting for this bounded experiment; it is not a
// claim about a scheduler's future batch policy.
constexpr std::array<Shape, 6> kQwenShapes = {{
    {128U, 5120U, 17408U, "qwen-wide-m128", 4U},
    {512U, 5120U, 17408U, "qwen-wide-m512", 2U},
    {1024U, 5120U, 17408U, "qwen-wide-m1024", 1U},
    {128U, 17408U, 5120U, "qwen-down-m128", 4U},
    {512U, 17408U, 5120U, "qwen-down-m512", 2U},
    {1024U, 17408U, 5120U, "qwen-down-m1024", 1U},
}};

enum class Outcome { Pass, N0, Fail };

struct Sizes final {
  std::size_t weight_packed = 0U;
  std::size_t activation_packed = 0U;
  std::size_t weight_scales = 0U;
  std::size_t activation_scales = 0U;
  std::size_t weight_fp8 = 0U;
  std::size_t activation_fp8 = 0U;
  std::size_t output = 0U;
};

struct HostInputs final {
  std::vector<uint8_t> weight_packed;
  std::vector<uint8_t> activation_packed;
  std::vector<uint8_t> weight_scales;
  std::vector<uint8_t> activation_scales;
};

struct Heuristic final {
  int rank = -1;
  int algorithm_index = -1;
  std::size_t workspace_bytes = 0U;
  float waves = 0.0F;
  hipblasStatus_t state = HIPBLAS_STATUS_NOT_INITIALIZED;
  std::string solution;
  std::string kernel;
  hipblasLtMatmulAlgo_t algorithm{};
};

struct Resources final {
  bool candidate = false;
  hipblasLtHandle_t handle = nullptr;
  hipblasLtMatmulDesc_t operation = nullptr;
  hipblasLtMatrixLayout_t a = nullptr;
  hipblasLtMatrixLayout_t b = nullptr;
  hipblasLtMatrixLayout_t c = nullptr;
  hipblasLtMatrixLayout_t d = nullptr;
  hipblasLtMatmulPreference_t preference = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t total_start = nullptr;
  hipEvent_t total_stop = nullptr;
  hipEvent_t stage_start = nullptr;
  hipEvent_t stage_stop = nullptr;
  hipEvent_t gemm_start = nullptr;
  hipEvent_t gemm_stop = nullptr;
  uint8_t *weight_packed = nullptr;
  uint8_t *activation_packed = nullptr;
  uint8_t *weight_scales = nullptr;
  uint8_t *activation_scales = nullptr;
  uint8_t *weight_fp8 = nullptr;
  uint8_t *activation_fp8 = nullptr;
  uint16_t *output = nullptr;
  void *workspace = nullptr;
};

struct Timing final {
  std::array<float, kMeasured> total_ms{};
  std::array<float, kMeasured> stage_ms{};
  std::array<float, kMeasured> gemm_ms{};
  float total_median = 0.0F;
  float stage_median = 0.0F;
  float gemm_median = 0.0F;
  float minimum = 0.0F;
  float maximum = 0.0F;
  std::size_t output_mismatches = 0U;
  std::vector<uint16_t> output;
};

bool hip_ok(const hipError_t status, const char *const what) {
  if (status == hipSuccess) {
    return true;
  }
  std::cerr << what << " failed: " << hipGetErrorName(status) << " ("
            << hipGetErrorString(status) << ")\n";
  return false;
}

bool lt_ok(const hipblasStatus_t status, const char *const what) {
  if (status == HIPBLAS_STATUS_SUCCESS) {
    return true;
  }
  std::cerr << what << " failed: hipBLAS status " << static_cast<int>(status)
            << "\n";
  return false;
}

bool n0_status(const hipblasStatus_t status) {
  return status == HIPBLAS_STATUS_INVALID_VALUE ||
         status == HIPBLAS_STATUS_NOT_SUPPORTED;
}

bool exact_gfx1201(const char *const arch) {
  if (arch == nullptr) {
    return false;
  }
  const std::string_view value(arch);
  constexpr std::string_view target = "gfx1201";
  return value == target ||
         (value.size() > target.size() && value.starts_with(target) &&
          value[target.size()] == ':');
}

bool parse_device(const char *const text, int *const device) {
  if (text == nullptr || device == nullptr) {
    return false;
  }
  const std::string_view value(text);
  int parsed = -1;
  const auto result =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
      parsed < 0) {
    return false;
  }
  *device = parsed;
  return true;
}

bool checked_product(const uint64_t lhs, const uint64_t rhs,
                     std::size_t *const result) {
  if (result == nullptr || (rhs != 0U && lhs > SIZE_MAX / rhs)) {
    return false;
  }
  *result = static_cast<std::size_t>(lhs * rhs);
  return true;
}

bool get_sizes(const Shape &shape, Sizes *const sizes) {
  if (sizes == nullptr || shape.m == 0U || shape.k == 0U || shape.n == 0U ||
      (shape.k % kBlockK) != 0U || (shape.k % 2U) != 0U) {
    return false;
  }
  std::size_t weight_values = 0U;
  std::size_t activation_values = 0U;
  std::size_t output_values = 0U;
  if (!checked_product(shape.n, shape.k, &weight_values) ||
      !checked_product(shape.m, shape.k, &activation_values) ||
      !checked_product(shape.m, shape.n, &output_values) ||
      output_values > SIZE_MAX / sizeof(uint16_t)) {
    return false;
  }
  sizes->weight_packed = weight_values / 2U;
  sizes->activation_packed = activation_values / 2U;
  sizes->weight_scales = weight_values / kBlockK;
  sizes->activation_scales = activation_values / kBlockK;
  sizes->weight_fp8 = weight_values;
  sizes->activation_fp8 = activation_values;
  sizes->output = output_values * sizeof(uint16_t);
  return true;
}

uint32_t mix32(uint32_t value) {
  value ^= value >> 16U;
  value *= UINT32_C(0x7feb352d);
  value ^= value >> 15U;
  value *= UINT32_C(0x846ca68b);
  return value ^ (value >> 16U);
}

uint8_t e2m1_code(const uint64_t ordinal, const uint32_t salt) {
  return static_cast<uint8_t>(mix32(static_cast<uint32_t>(ordinal) ^ salt) &
                              UINT32_C(0x0f));
}

// Model-like block scales: positive normal E4M3FN values in approximately
// [0.125, 1.875].  Keeping these realistic avoids making the candidate's
// comparison a saturation benchmark while still varying every K block.
uint8_t model_scale(const uint64_t ordinal, const uint32_t salt) {
  constexpr std::array<uint8_t, 16> codes = {
      UINT8_C(0x20), UINT8_C(0x21), UINT8_C(0x22), UINT8_C(0x23),
      UINT8_C(0x28), UINT8_C(0x2a), UINT8_C(0x2c), UINT8_C(0x2e),
      UINT8_C(0x30), UINT8_C(0x32), UINT8_C(0x34), UINT8_C(0x36),
      UINT8_C(0x38), UINT8_C(0x3a), UINT8_C(0x3c), UINT8_C(0x3e)};
  return codes[mix32(static_cast<uint32_t>(ordinal) ^ salt) & 15U];
}

HostInputs make_inputs(const Shape &shape, const Sizes &sizes) {
  HostInputs inputs;
  inputs.weight_packed.assign(sizes.weight_packed, 0U);
  inputs.activation_packed.assign(sizes.activation_packed, 0U);
  inputs.weight_scales.resize(sizes.weight_scales);
  inputs.activation_scales.resize(sizes.activation_scales);
  const uint64_t blocks = shape.k / kBlockK;
  for (uint64_t row = 0U; row < shape.m; ++row) {
    for (uint64_t inner = 0U; inner < shape.k; ++inner) {
      const uint8_t code = e2m1_code(row * shape.k + inner, kSeed);
      const std::size_t index =
          static_cast<std::size_t>(row * shape.k / 2U + inner / 2U);
      if ((inner & 1U) == 0U) {
        inputs.activation_packed[index] = code;
      } else {
        inputs.activation_packed[index] |= static_cast<uint8_t>(code << 4U);
      }
    }
    for (uint64_t block = 0U; block < blocks; ++block) {
      inputs.activation_scales[static_cast<std::size_t>(row * blocks + block)] =
          model_scale(row * blocks + block, kSeed ^ UINT32_C(0x116aa));
    }
  }
  for (uint64_t column = 0U; column < shape.n; ++column) {
    for (uint64_t inner = 0U; inner < shape.k; ++inner) {
      const uint8_t code =
          e2m1_code(column * shape.k + inner, kSeed ^ UINT32_C(0x9e3779b9));
      const std::size_t index =
          static_cast<std::size_t>(column * shape.k / 2U + inner / 2U);
      if ((inner & 1U) == 0U) {
        inputs.weight_packed[index] = code;
      } else {
        inputs.weight_packed[index] |= static_cast<uint8_t>(code << 4U);
      }
    }
    for (uint64_t block = 0U; block < blocks; ++block) {
      inputs.weight_scales[static_cast<std::size_t>(column * blocks + block)] =
          model_scale(column * blocks + block, kSeed ^ UINT32_C(0xa5a5a5a5));
    }
  }
  return inputs;
}

float host_e2m1(const uint8_t code) {
  constexpr std::array<float, 8> values = {0.0F, 0.5F, 1.0F, 1.5F,
                                           2.0F, 3.0F, 4.0F, 6.0F};
  const float value = values[code & UINT8_C(7)];
  return (code & UINT8_C(8)) == 0U ? value : -value;
}

float host_e4m3fn(const uint8_t bits) {
  const uint32_t magnitude = bits & UINT8_C(0x7f);
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & UINT8_C(7);
  if (exponent == 0U) {
    return static_cast<float>(mantissa) * 0x1p-9F *
           ((bits & UINT8_C(0x80)) == 0U ? 1.0F : -1.0F);
  }
  if (magnitude == UINT32_C(0x7f)) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const float value = std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                                 static_cast<int>(exponent) - 7);
  return (bits & UINT8_C(0x80)) == 0U ? value : -value;
}

// Scalar OCP E4M3FN RNE encoder, kept independent from HIP's packed FP8
// helper.  It is used by the host oracle and by the stage byte oracle.
uint8_t host_encode_e4m3fn(float value) {
  const bool negative = std::signbit(value);
  const float magnitude = std::fabs(value);
  const uint8_t sign = negative ? UINT8_C(0x80) : UINT8_C(0);
  if (std::isnan(magnitude)) {
    return static_cast<uint8_t>(sign | UINT8_C(0x7f));
  }
  if (magnitude == 0.0F) {
    return sign;
  }
  if (!std::isfinite(magnitude) || magnitude >= 448.0F) {
    return static_cast<uint8_t>(sign | UINT8_C(0x7e));
  }
  if (magnitude < 0.015625F) {
    const float scaled = magnitude * 512.0F;
    const uint32_t floor_value = static_cast<uint32_t>(std::floor(scaled));
    const float fraction = scaled - static_cast<float>(floor_value);
    const uint32_t rounded =
        floor_value +
        static_cast<uint32_t>(fraction > 0.5F ||
                              (fraction == 0.5F && (floor_value & 1U) != 0U));
    return static_cast<uint8_t>(sign | static_cast<uint8_t>(rounded));
  }
  int exponent = static_cast<int>(std::floor(std::log2(magnitude)));
  const float normalized = std::ldexp(magnitude, -exponent) - 1.0F;
  const float scaled = normalized * 8.0F;
  const uint32_t floor_value = static_cast<uint32_t>(std::floor(scaled));
  const float fraction = scaled - static_cast<float>(floor_value);
  uint32_t mantissa =
      floor_value +
      static_cast<uint32_t>(fraction > 0.5F ||
                            (fraction == 0.5F && (floor_value & 1U) != 0U));
  if (mantissa == 8U) {
    mantissa = 0U;
    ++exponent;
  }
  const int biased = exponent + 7;
  if (biased >= 15) {
    return static_cast<uint8_t>(sign | UINT8_C(0x7e));
  }
  return static_cast<uint8_t>(sign | static_cast<uint8_t>(biased << 3) |
                              static_cast<uint8_t>(mantissa));
}

uint8_t unpack_nibble(const std::vector<uint8_t> &packed, const uint64_t row,
                      const uint64_t k, const uint64_t inner) {
  const uint8_t pair =
      packed[static_cast<std::size_t>(row * k / 2U + inner / 2U)];
  return (inner & 1U) == 0U ? pair & UINT8_C(0x0f) : pair >> 4U;
}

std::vector<uint8_t> make_host_stage(const Shape &shape, const Sizes &sizes,
                                     const HostInputs &inputs,
                                     const bool weight) {
  const uint64_t rows = weight ? shape.n : shape.m;
  const std::vector<uint8_t> &packed =
      weight ? inputs.weight_packed : inputs.activation_packed;
  const std::vector<uint8_t> &scales =
      weight ? inputs.weight_scales : inputs.activation_scales;
  std::vector<uint8_t> result(weight ? sizes.weight_fp8 : sizes.activation_fp8);
  const uint64_t blocks = shape.k / kBlockK;
  std::size_t clamped = 0U;
  for (uint64_t row = 0U; row < rows; ++row) {
    for (uint64_t block = 0U; block < blocks; ++block) {
      const float scale =
          host_e4m3fn(scales[static_cast<std::size_t>(row * blocks + block)]);
      for (uint64_t lane = 0U; lane < kBlockK; ++lane) {
        const uint64_t inner = block * kBlockK + lane;
        const float value =
            host_e2m1(unpack_nibble(packed, row, shape.k, inner)) * scale;
        if (std::fabs(value) > 448.0F) {
          ++clamped;
        }
        const float bounded = std::max(-448.0F, std::min(448.0F, value));
        result[static_cast<std::size_t>(row * shape.k + inner)] =
            host_encode_e4m3fn(bounded);
      }
    }
  }
  std::cout << "host_stage variant=" << (weight ? "weight" : "activation")
            << " shape=" << shape.name << " elements=" << result.size()
            << " clamp_count=" << clamped << "\n";
  return result;
}

void release(Resources *const resources) {
  if (resources == nullptr) {
    return;
  }
  if (resources->workspace != nullptr)
    (void)hipFree(resources->workspace);
  if (resources->output != nullptr)
    (void)hipFree(resources->output);
  if (resources->activation_fp8 != nullptr)
    (void)hipFree(resources->activation_fp8);
  if (resources->weight_fp8 != nullptr)
    (void)hipFree(resources->weight_fp8);
  if (resources->activation_scales != nullptr)
    (void)hipFree(resources->activation_scales);
  if (resources->weight_scales != nullptr)
    (void)hipFree(resources->weight_scales);
  if (resources->activation_packed != nullptr)
    (void)hipFree(resources->activation_packed);
  if (resources->weight_packed != nullptr)
    (void)hipFree(resources->weight_packed);
  if (resources->gemm_stop != nullptr)
    (void)hipEventDestroy(resources->gemm_stop);
  if (resources->gemm_start != nullptr)
    (void)hipEventDestroy(resources->gemm_start);
  if (resources->stage_stop != nullptr)
    (void)hipEventDestroy(resources->stage_stop);
  if (resources->stage_start != nullptr)
    (void)hipEventDestroy(resources->stage_start);
  if (resources->total_stop != nullptr)
    (void)hipEventDestroy(resources->total_stop);
  if (resources->total_start != nullptr)
    (void)hipEventDestroy(resources->total_start);
  if (resources->stream != nullptr)
    (void)hipStreamDestroy(resources->stream);
  if (resources->preference != nullptr)
    (void)hipblasLtMatmulPreferenceDestroy(resources->preference);
  if (resources->d != nullptr)
    (void)hipblasLtMatrixLayoutDestroy(resources->d);
  if (resources->c != nullptr)
    (void)hipblasLtMatrixLayoutDestroy(resources->c);
  if (resources->b != nullptr)
    (void)hipblasLtMatrixLayoutDestroy(resources->b);
  if (resources->a != nullptr)
    (void)hipblasLtMatrixLayoutDestroy(resources->a);
  if (resources->operation != nullptr)
    (void)hipblasLtMatmulDescDestroy(resources->operation);
  if (resources->handle != nullptr)
    (void)hipblasLtDestroy(resources->handle);
  *resources = {};
}

bool allocate(Resources *const resources, const Sizes &sizes) {
  const auto malloc_bytes = [](void **ptr, const std::size_t bytes,
                               const char *what) {
    return hip_ok(hipMalloc(ptr, bytes), what);
  };
  if (!malloc_bytes(reinterpret_cast<void **>(&resources->weight_packed),
                    sizes.weight_packed, "hipMalloc weight packed") ||
      !malloc_bytes(reinterpret_cast<void **>(&resources->activation_packed),
                    sizes.activation_packed, "hipMalloc activation packed") ||
      !malloc_bytes(reinterpret_cast<void **>(&resources->weight_scales),
                    sizes.weight_scales, "hipMalloc weight scales") ||
      !malloc_bytes(reinterpret_cast<void **>(&resources->activation_scales),
                    sizes.activation_scales, "hipMalloc activation scales") ||
      (resources->candidate &&
       (!malloc_bytes(reinterpret_cast<void **>(&resources->weight_fp8),
                      sizes.weight_fp8, "hipMalloc staged weight") ||
        !malloc_bytes(reinterpret_cast<void **>(&resources->activation_fp8),
                      sizes.activation_fp8, "hipMalloc staged activation"))) ||
      !malloc_bytes(reinterpret_cast<void **>(&resources->output), sizes.output,
                    "hipMalloc BF16 output") ||
      !hip_ok(hipStreamCreate(&resources->stream), "hipStreamCreate") ||
      !hip_ok(hipEventCreate(&resources->total_start),
              "hipEventCreate total start") ||
      !hip_ok(hipEventCreate(&resources->total_stop),
              "hipEventCreate total stop") ||
      !hip_ok(hipEventCreate(&resources->stage_start),
              "hipEventCreate stage start") ||
      !hip_ok(hipEventCreate(&resources->stage_stop),
              "hipEventCreate stage stop") ||
      !hip_ok(hipEventCreate(&resources->gemm_start),
              "hipEventCreate gemm start") ||
      !hip_ok(hipEventCreate(&resources->gemm_stop),
              "hipEventCreate gemm stop")) {
    return false;
  }
  return true;
}

bool upload(const Resources &resources, const Sizes &sizes,
            const HostInputs &inputs) {
  return hip_ok(hipMemcpy(resources.weight_packed, inputs.weight_packed.data(),
                          sizes.weight_packed, hipMemcpyHostToDevice),
                "copy weight packed") &&
         hip_ok(hipMemcpy(resources.activation_packed,
                          inputs.activation_packed.data(),
                          sizes.activation_packed, hipMemcpyHostToDevice),
                "copy activation packed") &&
         hip_ok(hipMemcpy(resources.weight_scales, inputs.weight_scales.data(),
                          sizes.weight_scales, hipMemcpyHostToDevice),
                "copy weight scales") &&
         hip_ok(hipMemcpy(resources.activation_scales,
                          inputs.activation_scales.data(),
                          sizes.activation_scales, hipMemcpyHostToDevice),
                "copy activation scales") &&
         hip_ok(hipMemset(resources.output, 0, sizes.output),
                "clear BF16 output");
}

Outcome descriptor_step(const hipblasStatus_t status, const char *const what) {
  if (status == HIPBLAS_STATUS_SUCCESS)
    return Outcome::Pass;
  std::cout << "descriptor_step operation=" << what
            << " status=" << static_cast<int>(status) << "\n";
  return n0_status(status) ? Outcome::N0 : Outcome::Fail;
}

Outcome create_descriptors(Resources *const resources, const Shape &shape) {
  hipblasStatus_t status = hipblasLtCreate(&resources->handle);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    lt_ok(status, "hipblasLtCreate");
    return Outcome::Fail;
  }
  status = hipblasLtMatmulDescCreate(&resources->operation, HIPBLAS_COMPUTE_32F,
                                     HIP_R_32F);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    lt_ok(status, "hipblasLtMatmulDescCreate");
    return Outcome::Fail;
  }
  const hipblasOperation_t trans_a = HIPBLAS_OP_T;
  const hipblasOperation_t trans_b = HIPBLAS_OP_N;
  status = hipblasLtMatmulDescSetAttribute(resources->operation,
                                           HIPBLASLT_MATMUL_DESC_TRANSA,
                                           &trans_a, sizeof(trans_a));
  Outcome outcome = descriptor_step(status, "TRANSA");
  if (outcome != Outcome::Pass)
    return outcome;
  status = hipblasLtMatmulDescSetAttribute(resources->operation,
                                           HIPBLASLT_MATMUL_DESC_TRANSB,
                                           &trans_b, sizeof(trans_b));
  outcome = descriptor_step(status, "TRANSB");
  if (outcome != Outcome::Pass)
    return outcome;

  if (!resources->candidate) {
    void *weight_scale_pointer = resources->weight_scales;
    void *activation_scale_pointer = resources->activation_scales;
    constexpr hipblasLtMatmulMatrixScale_t scale_mode =
        HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    status = hipblasLtMatmulDescSetAttribute(
        resources->operation, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        &weight_scale_pointer, sizeof(weight_scale_pointer));
    outcome = descriptor_step(status, "ID64_A_SCALE_POINTER");
    if (outcome != Outcome::Pass)
      return outcome;
    status = hipblasLtMatmulDescSetAttribute(
        resources->operation, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        &activation_scale_pointer, sizeof(activation_scale_pointer));
    outcome = descriptor_step(status, "ID64_B_SCALE_POINTER");
    if (outcome != Outcome::Pass)
      return outcome;
    status = hipblasLtMatmulDescSetAttribute(resources->operation,
                                             HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                             &scale_mode, sizeof(scale_mode));
    outcome = descriptor_step(status, "ID64_A_SCALE_MODE_VEC16_UE4M3");
    if (outcome != Outcome::Pass)
      return outcome;
    status = hipblasLtMatmulDescSetAttribute(resources->operation,
                                             HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                             &scale_mode, sizeof(scale_mode));
    outcome = descriptor_step(status, "ID64_B_SCALE_MODE_VEC16_UE4M3");
    if (outcome != Outcome::Pass)
      return outcome;
  }

  const hipDataType input_type = static_cast<hipDataType>(
      resources->candidate ? HIP_R_8F_E4M3 : HIP_R_4F_E2M1_EXT);
  status = hipblasLtMatrixLayoutCreate(&resources->a, input_type, shape.k,
                                       shape.n, static_cast<int64_t>(shape.k));
  outcome = descriptor_step(status, resources->candidate ? "A_LAYOUT_FP8_KxN"
                                                         : "A_LAYOUT_FP4_KxN");
  if (outcome != Outcome::Pass)
    return outcome;
  status = hipblasLtMatrixLayoutCreate(&resources->b, input_type, shape.k,
                                       shape.m, static_cast<int64_t>(shape.k));
  outcome = descriptor_step(status, resources->candidate ? "B_LAYOUT_FP8_KxM"
                                                         : "B_LAYOUT_FP4_KxM");
  if (outcome != Outcome::Pass)
    return outcome;
  status = hipblasLtMatrixLayoutCreate(&resources->c, HIP_R_16BF, shape.n,
                                       shape.m, static_cast<int64_t>(shape.n));
  outcome = descriptor_step(status, "C_LAYOUT_BF16_NxM");
  if (outcome != Outcome::Pass)
    return outcome;
  status = hipblasLtMatrixLayoutCreate(&resources->d, HIP_R_16BF, shape.n,
                                       shape.m, static_cast<int64_t>(shape.n));
  outcome = descriptor_step(status, "D_LAYOUT_BF16_NxM");
  if (outcome != Outcome::Pass)
    return outcome;
  status = hipblasLtMatmulPreferenceCreate(&resources->preference);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    lt_ok(status, "hipblasLtMatmulPreferenceCreate");
    return Outcome::Fail;
  }
  status = hipblasLtMatmulPreferenceSetAttribute(
      resources->preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &kWorkspaceLimit, sizeof(kWorkspaceLimit));
  if (status != HIPBLAS_STATUS_SUCCESS) {
    lt_ok(status, "set workspace limit");
    return Outcome::Fail;
  }
  std::cout << "descriptor variant="
            << (resources->candidate ? "staging-fp8" : "id64-control")
            << " shape=" << shape.name << " m=" << shape.m << " k=" << shape.k
            << " n=" << shape.n << " A=KxN,ld=K,op=T B=KxM,ld=K,op=N"
            << " C=D=NxM,ld=N,BF16 compute=FP32" << " alpha=" << kAlpha
            << (resources->candidate ? " scale=block-consumed-stage,E4M3FN"
                                     : " scale=VEC16_UE4M3")
            << " workspace_limit=" << kWorkspaceLimit << "\n";
  return Outcome::Pass;
}

Outcome query_rank_zero(Resources *const resources, const Shape &shape,
                        Heuristic *const selected) {
  std::array<hipblasLtMatmulHeuristicResult_t, kRequestedAlgorithms> results{};
  int returned = 0;
  const hipblasStatus_t status = hipblasLtMatmulAlgoGetHeuristic(
      resources->handle, resources->operation, resources->a, resources->b,
      resources->c, resources->d, resources->preference, kRequestedAlgorithms,
      results.data(), &returned);
  std::cout << "heuristic_query variant="
            << (resources->candidate ? "staging-fp8" : "id64-control")
            << " shape=" << shape.name << " status=" << static_cast<int>(status)
            << " requested=" << kRequestedAlgorithms << " returned=" << returned
            << "\n";
  if (n0_status(status) ||
      (status == HIPBLAS_STATUS_SUCCESS && returned == 0)) {
    return Outcome::N0;
  }
  if (status != HIPBLAS_STATUS_SUCCESS || returned <= 0 ||
      returned > kRequestedAlgorithms) {
    lt_ok(status, "hipblasLtMatmulAlgoGetHeuristic");
    return Outcome::Fail;
  }
  for (int rank = 0; rank < returned; ++rank) {
    const auto &entry = results[static_cast<std::size_t>(rank)];
    if (entry.state != HIPBLAS_STATUS_SUCCESS)
      continue;
    // hipBLASLt's extension helpers predate const-correct heuristic results.
    // Keep the result immutable and pass only this local descriptor by
    // non-const reference.
    hipblasLtMatmulAlgo_t algorithm = entry.algo;
    const int index = hipblaslt_ext::getIndexFromAlgo(algorithm);
    std::cout << "heuristic variant="
              << (resources->candidate ? "staging-fp8" : "id64-control")
              << " shape=" << shape.name << " rank=" << rank
              << " algorithm_index=" << index
              << " workspace=" << entry.workspaceSize
              << " waves=" << entry.wavesCount << " solution="
              << std::quoted(hipblaslt_ext::getSolutionNameFromAlgo(
                     resources->handle, algorithm))
              << " kernel="
              << std::quoted(hipblaslt_ext::getKernelNameFromAlgo(
                     resources->handle, algorithm))
              << "\n";
  }
  const auto &rank_zero = results[0];
  if (rank_zero.state != HIPBLAS_STATUS_SUCCESS ||
      rank_zero.workspaceSize > kWorkspaceLimit) {
    std::cout << "rank0_unusable variant="
              << (resources->candidate ? "staging-fp8" : "id64-control")
              << " shape=" << shape.name
              << " state=" << static_cast<int>(rank_zero.state)
              << " workspace=" << rank_zero.workspaceSize << "\n";
    return Outcome::N0;
  }
  selected->rank = 0;
  selected->state = rank_zero.state;
  selected->workspace_bytes = rank_zero.workspaceSize;
  selected->waves = rank_zero.wavesCount;
  selected->algorithm = rank_zero.algo;
  selected->algorithm_index =
      hipblaslt_ext::getIndexFromAlgo(selected->algorithm);
  selected->solution = hipblaslt_ext::getSolutionNameFromAlgo(
      resources->handle, selected->algorithm);
  selected->kernel = hipblaslt_ext::getKernelNameFromAlgo(resources->handle,
                                                          selected->algorithm);
  if (selected->workspace_bytes != 0U &&
      !hip_ok(hipMalloc(&resources->workspace, selected->workspace_bytes),
              "hipMalloc selected workspace")) {
    return Outcome::Fail;
  }
  std::cout << "selected variant="
            << (resources->candidate ? "staging-fp8" : "id64-control")
            << " shape=" << shape.name
            << " rank=0 algorithm_index=" << selected->algorithm_index
            << " workspace=" << selected->workspace_bytes
            << " waves=" << selected->waves
            << " solution=" << std::quoted(selected->solution)
            << " kernel=" << std::quoted(selected->kernel) << "\n";
  return Outcome::Pass;
}

__global__ __launch_bounds__(kStageThreads) void stage_fp4_to_fp8_kernel(
    const uint8_t *const packed, const uint8_t *const scales,
    uint8_t *const output, const uint64_t rows, const uint64_t k) {
  const uint64_t blocks_per_row = k / kBlockK;
  const uint64_t block_index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total_blocks = rows * blocks_per_row;
  if (block_index >= total_blocks)
    return;
  const uint64_t row = block_index / blocks_per_row;
  const uint64_t block = block_index - row * blocks_per_row;
  const uint64_t packed_offset = row * (k / 2U) + block * 8U;
  const uint64_t packed_lo =
      *reinterpret_cast<const uint32_t *>(packed + packed_offset);
  const uint64_t packed_hi =
      *reinterpret_cast<const uint32_t *>(packed + packed_offset + 4U);
  const uint64_t packed16 = packed_lo | (packed_hi << 32U);
  const float scale =
      sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::decode(scales[block_index]);
  const uint64_t output_offset = row * k + block * kBlockK;
  uint32_t bytes[4] = {0U, 0U, 0U, 0U};
  for (uint32_t lane = 0U; lane < kBlockK; ++lane) {
    const uint8_t code = static_cast<uint8_t>((packed16 >> (lane * 4U)) & 15U);
    const float value =
        sllm_lowp::ScalarCodec<sllm_lowp::E2M1>::decode(code) * scale;
    // Clamp before the HIP FP8 conversion.  In particular, never pass a
    // value just above max finite to the fp8x2 helper on gfx1201.
    const float bounded = fmaxf(-448.0F, fminf(448.0F, value));
    const uint8_t encoded =
        sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::encode(bounded);
    bytes[lane / 4U] |= static_cast<uint32_t>(encoded) << ((lane % 4U) * 8U);
  }
  reinterpret_cast<uint32_t *>(output + output_offset)[0] = bytes[0];
  reinterpret_cast<uint32_t *>(output + output_offset)[1] = bytes[1];
  reinterpret_cast<uint32_t *>(output + output_offset)[2] = bytes[2];
  reinterpret_cast<uint32_t *>(output + output_offset)[3] = bytes[3];
}

bool launch_stage(const Resources &resources, const Shape &shape) {
  const uint64_t weight_blocks = shape.n * (shape.k / kBlockK);
  const uint64_t activation_blocks = shape.m * (shape.k / kBlockK);
  const uint32_t weight_grid = static_cast<uint32_t>(
      (weight_blocks + kStageThreads - 1U) / kStageThreads);
  const uint32_t activation_grid = static_cast<uint32_t>(
      (activation_blocks + kStageThreads - 1U) / kStageThreads);
  hipLaunchKernelGGL(stage_fp4_to_fp8_kernel, dim3(weight_grid),
                     dim3(kStageThreads), 0U, resources.stream,
                     resources.weight_packed, resources.weight_scales,
                     resources.weight_fp8, shape.n, shape.k);
  if (!hip_ok(hipGetLastError(), "launch weight staging"))
    return false;
  hipLaunchKernelGGL(stage_fp4_to_fp8_kernel, dim3(activation_grid),
                     dim3(kStageThreads), 0U, resources.stream,
                     resources.activation_packed, resources.activation_scales,
                     resources.activation_fp8, shape.m, shape.k);
  return hip_ok(hipGetLastError(), "launch activation staging");
}

hipblasStatus_t launch_gemm(const Resources &resources,
                            const Heuristic &heuristic) {
  constexpr float beta = 0.0F;
  const uint8_t *weight =
      resources.candidate ? resources.weight_fp8 : resources.weight_packed;
  const uint8_t *activation = resources.candidate ? resources.activation_fp8
                                                  : resources.activation_packed;
  return hipblasLtMatmul(resources.handle, resources.operation, &kAlpha, weight,
                         resources.a, activation, resources.b, &beta,
                         resources.output, resources.c, resources.output,
                         resources.d, &heuristic.algorithm, resources.workspace,
                         heuristic.workspace_bytes, resources.stream);
}

bool check_staged_bytes(const Resources &resources, const Shape &shape,
                        const Sizes &sizes, const HostInputs &inputs) {
  const std::vector<uint8_t> expected_weight =
      make_host_stage(shape, sizes, inputs, true);
  const std::vector<uint8_t> expected_activation =
      make_host_stage(shape, sizes, inputs, false);
  std::vector<uint8_t> actual_weight(sizes.weight_fp8);
  std::vector<uint8_t> actual_activation(sizes.activation_fp8);
  if (!hip_ok(hipMemcpy(actual_weight.data(), resources.weight_fp8,
                        actual_weight.size(), hipMemcpyDeviceToHost),
              "copy staged weight") ||
      !hip_ok(hipMemcpy(actual_activation.data(), resources.activation_fp8,
                        actual_activation.size(), hipMemcpyDeviceToHost),
              "copy staged activation")) {
    return false;
  }
  const std::size_t weight_mismatches = static_cast<std::size_t>(std::count_if(
      actual_weight.begin(), actual_weight.end(),
      [index = std::size_t{0U}, &expected_weight](uint8_t value) mutable {
        return value != expected_weight[index++];
      }));
  const std::size_t activation_mismatches = static_cast<std::size_t>(
      std::count_if(actual_activation.begin(), actual_activation.end(),
                    [index = std::size_t{0U},
                     &expected_activation](uint8_t value) mutable {
                      return value != expected_activation[index++];
                    }));
  const bool finite =
      std::all_of(actual_weight.begin(), actual_weight.end(),
                  [](uint8_t value) { return (value & 0x7fU) != 0x7fU; }) &&
      std::all_of(actual_activation.begin(), actual_activation.end(),
                  [](uint8_t value) { return (value & 0x7fU) != 0x7fU; });
  std::cout << "stage_oracle shape=" << shape.name
            << " weight_mismatches=" << weight_mismatches
            << " activation_mismatches=" << activation_mismatches
            << " finite=" << (finite ? "true" : "false")
            << " deterministic=true\n";
  return weight_mismatches == 0U && activation_mismatches == 0U && finite;
}

float host_bf16_to_float(const uint16_t bits) {
  const uint32_t expanded = static_cast<uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &expanded, sizeof(value));
  return value;
}

uint64_t hash_bf16(const std::vector<uint16_t> &values) {
  uint64_t hash = UINT64_C(1469598103934665603);
  for (const uint16_t value : values) {
    hash ^= static_cast<uint8_t>(value & 0xffU);
    hash *= UINT64_C(1099511628211);
    hash ^= static_cast<uint8_t>(value >> 8U);
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

double oracle_error(const Shape &shape, const HostInputs &inputs,
                    const std::vector<uint8_t> *const staged_weight,
                    const std::vector<uint8_t> *const staged_activation,
                    const std::vector<uint16_t> &output) {
  const uint64_t blocks = shape.k / kBlockK;
  constexpr std::size_t kSamples = 64U;
  double max_normalized = 0.0;
  for (std::size_t sample = 0U; sample < kSamples; ++sample) {
    const uint64_t row =
        (static_cast<uint64_t>(sample) * (shape.m - 1U)) / (kSamples - 1U);
    const uint64_t column = (static_cast<uint64_t>(sample * 7919U) % shape.n);
    double sum = 0.0;
    double absolute_sum = 0.0;
    for (uint64_t inner = 0U; inner < shape.k; ++inner) {
      const float activation =
          staged_activation == nullptr
              ? host_e2m1(unpack_nibble(inputs.activation_packed, row, shape.k,
                                        inner)) *
                    host_e4m3fn(
                        inputs.activation_scales[static_cast<std::size_t>(
                            row * blocks + inner / kBlockK)])
              : host_e4m3fn((*staged_activation)[static_cast<std::size_t>(
                    row * shape.k + inner)]);
      const float weight =
          staged_weight == nullptr
              ? host_e2m1(unpack_nibble(inputs.weight_packed, column, shape.k,
                                        inner)) *
                    host_e4m3fn(inputs.weight_scales[static_cast<std::size_t>(
                        column * blocks + inner / kBlockK)])
              : host_e4m3fn((*staged_weight)[static_cast<std::size_t>(
                    column * shape.k + inner)]);
      const double term =
          static_cast<double>(activation) * static_cast<double>(weight);
      sum += term;
      absolute_sum += std::abs(term);
    }
    sum *= static_cast<double>(kAlpha);
    absolute_sum *= static_cast<double>(kAlpha);
    const std::size_t index = static_cast<std::size_t>(row * shape.n + column);
    const double observed = host_bf16_to_float(output[index]);
    max_normalized = std::max(
        max_normalized,
        std::abs(observed - sum) /
            std::max(absolute_sum, std::numeric_limits<double>::min()));
  }
  return max_normalized;
}

float median(std::array<float, kMeasured> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2U];
}

Outcome measure_control(const Resources &resources, const Heuristic &heuristic,
                        const Sizes &sizes, Timing *const timing) {
  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    const hipblasStatus_t status = launch_gemm(resources, heuristic);
    if (status != HIPBLAS_STATUS_SUCCESS) {
      return n0_status(status) ? Outcome::N0 : Outcome::Fail;
    }
  }
  if (!hip_ok(hipStreamSynchronize(resources.stream), "control warmup sync"))
    return Outcome::Fail;
  timing->output.resize(sizes.output / sizeof(uint16_t));
  std::vector<uint16_t> current(timing->output.size());
  for (int iteration = 0; iteration < kMeasured; ++iteration) {
    if (!hip_ok(hipEventRecord(resources.total_start, resources.stream),
                "control start"))
      return Outcome::Fail;
    const hipblasStatus_t status = launch_gemm(resources, heuristic);
    if (status != HIPBLAS_STATUS_SUCCESS)
      return n0_status(status) ? Outcome::N0 : Outcome::Fail;
    if (!hip_ok(hipEventRecord(resources.total_stop, resources.stream),
                "control stop") ||
        !hip_ok(hipEventSynchronize(resources.total_stop), "control sync") ||
        !hip_ok(hipEventElapsedTime(&timing->total_ms[iteration],
                                    resources.total_start,
                                    resources.total_stop),
                "control elapsed") ||
        !hip_ok(hipMemcpy(current.data(), resources.output, sizes.output,
                          hipMemcpyDeviceToHost),
                "copy control output"))
      return Outcome::Fail;
    timing->stage_ms[iteration] = 0.0F;
    timing->gemm_ms[iteration] = timing->total_ms[iteration];
    if (iteration == 0)
      timing->output = current;
    else {
      timing->output_mismatches += static_cast<std::size_t>(std::count_if(
          current.begin(), current.end(),
          [index = std::size_t{0U}, &reference = timing->output](
              uint16_t value) mutable { return value != reference[index++]; }));
    }
  }
  timing->total_median = median(timing->total_ms);
  timing->gemm_median = timing->total_median;
  timing->minimum =
      *std::min_element(timing->total_ms.begin(), timing->total_ms.end());
  timing->maximum =
      *std::max_element(timing->total_ms.begin(), timing->total_ms.end());
  return timing->output_mismatches == 0U ? Outcome::Pass : Outcome::Fail;
}

Outcome measure_candidate(const Resources &resources,
                          const Heuristic &heuristic, const Shape &shape,
                          const Sizes &sizes, Timing *const timing) {
  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    if (!launch_stage(resources, shape))
      return Outcome::Fail;
    const hipblasStatus_t status = launch_gemm(resources, heuristic);
    if (status != HIPBLAS_STATUS_SUCCESS)
      return n0_status(status) ? Outcome::N0 : Outcome::Fail;
  }
  if (!hip_ok(hipStreamSynchronize(resources.stream), "candidate warmup sync"))
    return Outcome::Fail;
  timing->output.resize(sizes.output / sizeof(uint16_t));
  std::vector<uint16_t> current(timing->output.size());
  for (int iteration = 0; iteration < kMeasured; ++iteration) {
    if (!hip_ok(hipEventRecord(resources.total_start, resources.stream),
                "total start") ||
        !hip_ok(hipEventRecord(resources.stage_start, resources.stream),
                "stage start"))
      return Outcome::Fail;
    if (!launch_stage(resources, shape))
      return Outcome::Fail;
    if (!hip_ok(hipEventRecord(resources.stage_stop, resources.stream),
                "stage stop") ||
        !hip_ok(hipEventRecord(resources.gemm_start, resources.stream),
                "gemm start"))
      return Outcome::Fail;
    const hipblasStatus_t status = launch_gemm(resources, heuristic);
    if (status != HIPBLAS_STATUS_SUCCESS)
      return n0_status(status) ? Outcome::N0 : Outcome::Fail;
    if (!hip_ok(hipEventRecord(resources.gemm_stop, resources.stream),
                "gemm stop") ||
        !hip_ok(hipEventRecord(resources.total_stop, resources.stream),
                "total stop") ||
        !hip_ok(hipEventSynchronize(resources.total_stop), "candidate sync") ||
        !hip_ok(hipEventElapsedTime(&timing->stage_ms[iteration],
                                    resources.stage_start,
                                    resources.stage_stop),
                "stage elapsed") ||
        !hip_ok(hipEventElapsedTime(&timing->gemm_ms[iteration],
                                    resources.gemm_start, resources.gemm_stop),
                "gemm elapsed") ||
        !hip_ok(hipEventElapsedTime(&timing->total_ms[iteration],
                                    resources.total_start,
                                    resources.total_stop),
                "total elapsed") ||
        !hip_ok(hipMemcpy(current.data(), resources.output, sizes.output,
                          hipMemcpyDeviceToHost),
                "copy candidate output"))
      return Outcome::Fail;
    if (iteration == 0)
      timing->output = current;
    else {
      timing->output_mismatches += static_cast<std::size_t>(std::count_if(
          current.begin(), current.end(),
          [index = std::size_t{0U}, &reference = timing->output](
              uint16_t value) mutable { return value != reference[index++]; }));
    }
  }
  timing->total_median = median(timing->total_ms);
  timing->stage_median = median(timing->stage_ms);
  timing->gemm_median = median(timing->gemm_ms);
  timing->minimum =
      *std::min_element(timing->total_ms.begin(), timing->total_ms.end());
  timing->maximum =
      *std::max_element(timing->total_ms.begin(), timing->total_ms.end());
  return timing->output_mismatches == 0U ? Outcome::Pass : Outcome::Fail;
}

Outcome run_variant(const Shape &shape, const Sizes &sizes,
                    const HostInputs &inputs, const bool candidate,
                    Timing *const timing, Heuristic *const heuristic,
                    double *const oracle) {
  Resources resources;
  resources.candidate = candidate;
  uint64_t free_before = 0U;
  uint64_t total_memory = 0U;
  (void)hipMemGetInfo(&free_before, &total_memory);
  if (!allocate(&resources, sizes) || !upload(resources, sizes, inputs)) {
    release(&resources);
    return Outcome::Fail;
  }
  Outcome outcome = create_descriptors(&resources, shape);
  if (outcome != Outcome::Pass) {
    release(&resources);
    return outcome;
  }
  outcome = query_rank_zero(&resources, shape, heuristic);
  if (outcome != Outcome::Pass) {
    release(&resources);
    return outcome;
  }
  std::size_t device_bytes = sizes.weight_packed + sizes.activation_packed +
                             sizes.weight_scales + sizes.activation_scales +
                             sizes.output + heuristic->workspace_bytes;
  if (candidate)
    device_bytes += sizes.weight_fp8 + sizes.activation_fp8;
  std::cout << "memory variant=" << (candidate ? "staging-fp8" : "id64-control")
            << " shape=" << shape.name << " device_bytes=" << device_bytes
            << " workspace_bytes=" << heuristic->workspace_bytes
            << " total_vram_bytes=" << total_memory
            << " free_before=" << free_before << "\n";
  if (candidate) {
    if (!launch_stage(resources, shape) ||
        !hip_ok(hipStreamSynchronize(resources.stream), "stage oracle sync") ||
        !check_staged_bytes(resources, shape, sizes, inputs)) {
      release(&resources);
      return Outcome::Fail;
    }
  }
  outcome = candidate
                ? measure_candidate(resources, *heuristic, shape, sizes, timing)
                : measure_control(resources, *heuristic, sizes, timing);
  if (outcome != Outcome::Pass) {
    release(&resources);
    return outcome;
  }
  const std::vector<uint8_t> *staged_weight = nullptr;
  const std::vector<uint8_t> *staged_activation = nullptr;
  std::vector<uint8_t> expected_weight;
  std::vector<uint8_t> expected_activation;
  if (candidate) {
    expected_weight = make_host_stage(shape, sizes, inputs, true);
    expected_activation = make_host_stage(shape, sizes, inputs, false);
    staged_weight = &expected_weight;
    staged_activation = &expected_activation;
  }
  *oracle = oracle_error(shape, inputs, staged_weight, staged_activation,
                         timing->output);
  std::cout << "timing variant=" << (candidate ? "staging-fp8" : "id64-control")
            << " shape=" << shape.name << " warmups=" << kWarmups
            << " measured=" << kMeasured
            << " total_median_ms=" << timing->total_median
            << " stage_median_ms=" << timing->stage_median
            << " gemm_median_ms=" << timing->gemm_median
            << " min_ms=" << timing->minimum << " max_ms=" << timing->maximum
            << " repeat_bf16_mismatches=" << timing->output_mismatches
            << " output_fnv64=0x" << std::hex << hash_bf16(timing->output)
            << std::dec << " oracle_max_normalized=" << *oracle
            << " oracle_status=" << (*oracle <= 0.01 ? "PASS" : "FAIL") << "\n";
  const bool oracle_ok = *oracle <= 0.01;
  release(&resources);
  return oracle_ok ? Outcome::Pass : Outcome::Fail;
}

constexpr bool candidate_aligned(const Shape &shape) {
  return (shape.m % 128U) == 0U && (shape.n % 128U) == 0U &&
         (shape.k % 16U) == 0U;
}

static_assert(!candidate_aligned(Shape{127U, 5120U, 17408U, "boundary-m127",
                                       0U}),
              "the non-aligned boundary must take the ID64 fallback");

void print_boundary_guard() {
  const Shape boundary = {127U, 5120U, 17408U, "boundary-m127", 0U};
  std::cout << "candidate_guard shape=" << boundary.name << " m=" << boundary.m
            << " k=" << boundary.k << " n=" << boundary.n
            << " aligned=" << (candidate_aligned(boundary) ? "true" : "false")
            << " candidate=SKIP fallback=id64-control status=PASS\n";
}

} // namespace

int main(int argc, char **argv) {
  int device = 0;
  if (argc > 2 || (argc == 2 && !parse_device(argv[1], &device))) {
    std::cerr << "usage: phase78_nvfp4_gfx1201_fp8_staging_probe [DEVICE]\n";
    return EXIT_FAILURE;
  }
  if (!hip_ok(hipSetDevice(device), "hipSetDevice"))
    return EXIT_FAILURE;
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, device),
              "hipGetDeviceProperties"))
    return EXIT_FAILURE;
  int runtime_version = 0;
  if (!hip_ok(hipRuntimeGetVersion(&runtime_version), "hipRuntimeGetVersion"))
    return EXIT_FAILURE;
  std::cout << "identity device=" << device
            << " arch=" << properties.gcnArchName
            << " hip_header=" << HIP_VERSION_MAJOR << '.' << HIP_VERSION_MINOR
            << '.' << HIP_VERSION_PATCH << " hip_runtime=" << runtime_version
            << " pci=" << std::hex << std::setw(4) << std::setfill('0')
            << properties.pciDomainID << ':' << std::setw(2)
            << properties.pciBusID << ':' << std::setw(2)
            << properties.pciDeviceID << std::dec << std::setfill(' ') << "\n";
  if (!exact_gfx1201(properties.gcnArchName)) {
    std::cerr << "exact gfx1201 is required\n";
    std::cout << "PHASE78_NVFP4_GFX1201_FP8_STAGING_RESULT=N0\n";
    return EXIT_SUCCESS;
  }

  print_boundary_guard();
  double weighted_control = 0.0;
  double weighted_candidate = 0.0;
  uint32_t weighted_occurrences = 0U;
  bool control_available = true;
  for (const Shape &shape : kQwenShapes) {
    Sizes sizes;
    if (!get_sizes(shape, &sizes)) {
      std::cerr << "invalid shape " << shape.name << "\n";
      return EXIT_FAILURE;
    }
    const HostInputs inputs = make_inputs(shape, sizes);
    Timing control_timing;
    Timing candidate_timing;
    Heuristic control_heuristic;
    Heuristic candidate_heuristic;
    double control_oracle = 0.0;
    double candidate_oracle = 0.0;
    const Outcome control =
        run_variant(shape, sizes, inputs, false, &control_timing,
                    &control_heuristic, &control_oracle);
    const Outcome candidate =
        run_variant(shape, sizes, inputs, true, &candidate_timing,
                    &candidate_heuristic, &candidate_oracle);
    if (candidate == Outcome::N0) {
      std::cout << "shape_result shape=" << shape.name
                << " control=" << (control == Outcome::Pass ? "PASS" : "N0")
                << " candidate=N0\n";
      std::cout << "PHASE78_NVFP4_GFX1201_FP8_STAGING_RESULT=N0\n";
      return EXIT_SUCCESS;
    }
    if (control == Outcome::Fail || candidate != Outcome::Pass) {
      std::cout << "PHASE78_NVFP4_GFX1201_FP8_STAGING_RESULT=FAIL\n";
      return EXIT_FAILURE;
    }
    if (control == Outcome::N0) {
      // Keep measuring the candidate across all six exact shapes even when
      // the local ROCm build cannot expose the ID64 VEC16 control.  This is a
      // useful candidate-only result, but it is never promoted to a speedup
      // or GO claim without a real baseline.
      control_available = false;
      std::cout << "shape_result shape=" << shape.name
                << " control=N0 candidate=PASS candidate_total_median_ms="
                << candidate_timing.total_median
                << " candidate_stage_median_ms="
                << candidate_timing.stage_median
                << " candidate_gemm_median_ms=" << candidate_timing.gemm_median
                << "\n";
      continue;
    }
    const double speedup = static_cast<double>(control_timing.total_median) /
                           static_cast<double>(candidate_timing.total_median);
    std::cout << "shape_result shape=" << shape.name
              << " control_median_ms=" << control_timing.total_median
              << " candidate_total_median_ms=" << candidate_timing.total_median
              << " candidate_stage_median_ms=" << candidate_timing.stage_median
              << " candidate_gemm_median_ms=" << candidate_timing.gemm_median
              << " speedup_control_over_candidate=" << speedup
              << " candidate_algo_rank=" << candidate_heuristic.rank
              << " candidate_algo_index=" << candidate_heuristic.algorithm_index
              << " candidate_workspace_bytes="
              << candidate_heuristic.workspace_bytes << "\n";
    weighted_control +=
        static_cast<double>(shape.occurrence) * control_timing.total_median;
    weighted_candidate +=
        static_cast<double>(shape.occurrence) * candidate_timing.total_median;
    weighted_occurrences += shape.occurrence;
  }
  if (!control_available) {
    std::cout
        << "weighted_summary baseline=ID64_UNAVAILABLE candidate_only=true"
        << " GO=false reason=control_heuristic_N0\n";
    std::cout << "PHASE78_NVFP4_GFX1201_FP8_STAGING_RESULT=N0\n";
    return EXIT_SUCCESS;
  }
  const double weighted_speedup = weighted_control / weighted_candidate;
  std::cout << "weighted_summary occurrences=" << weighted_occurrences
            << " control_ms=" << weighted_control
            << " candidate_total_ms=" << weighted_candidate
            << " speedup=" << weighted_speedup << " GO_threshold=2.0 GO="
            << (weighted_speedup >= 2.0 ? "true" : "false") << "\n";
  std::cout << "PHASE78_NVFP4_GFX1201_FP8_STAGING_RESULT="
            << (weighted_speedup >= 2.0 ? "GO" : "NO-GO") << "\n";
  return EXIT_SUCCESS;
}
