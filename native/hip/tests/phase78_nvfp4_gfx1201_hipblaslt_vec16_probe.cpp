// Phase 78 standalone gfx1201 hipBLASLt NVFP4 block16 capability probe.
//
// This is deliberately not part of the production build.  ROCm 7.14 exposes
// HIP_R_4F_E2M1_EXT and HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, while the
// public header still labels the latter "Not supported yet".  The probe makes
// that ambiguity observable without changing the runtime selector:
//
//   D[M,N] = (input_global * weight_global) * B[M,K] * A[N,K]^T
//
// A and B contain packed OCP E2M1 values.  Their positive-finite OCP E4M3FN
// scales are row-major block planes with one byte for every innermost-K block
// of 16 values.  hipBLASLt accumulates in FP32 and stores BF16.  The input
// generator and tensor scales intentionally match the Phase 78 ID64 probe, so
// a successful output signature can be compared directly with that control.
//
// An INVALID_VALUE/NOT_SUPPORTED descriptor or heuristic response, a zero
// heuristic count, or an unusable rank zero is an expected negative result
// (N0).  The probe stops immediately in that case rather than searching nearby
// layouts and accidentally answering a different question.

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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr uint64_t kBlockK = 16U;
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
  bool host_oracle;
};

constexpr std::array<Shape, 7> kShapes = {{
    {17U, 32U, 17U, "small-m17-k32-n17", true},
    {128U, 5120U, 17408U, "qwen-wide-m128", false},
    {512U, 5120U, 17408U, "qwen-wide-m512", false},
    {1024U, 5120U, 17408U, "qwen-wide-m1024", false},
    {128U, 17408U, 5120U, "qwen-down-m128", false},
    {512U, 17408U, 5120U, "qwen-down-m512", false},
    {1024U, 17408U, 5120U, "qwen-down-m1024", false},
}};

enum class Outcome { Pass, N0, Fail };

struct Heuristic final {
  int rank = -1;
  int algorithm_index = -1;
  hipblasStatus_t state = HIPBLAS_STATUS_NOT_INITIALIZED;
  std::size_t workspace_bytes = 0U;
  float waves = 0.0F;
  std::string solution_name;
  std::string kernel_name;
  hipblasLtMatmulAlgo_t algorithm{};
};

struct Resources final {
  hipblasLtHandle_t handle = nullptr;
  hipblasLtMatmulDesc_t operation = nullptr;
  hipblasLtMatrixLayout_t a = nullptr;
  hipblasLtMatrixLayout_t b = nullptr;
  hipblasLtMatrixLayout_t c = nullptr;
  hipblasLtMatrixLayout_t d = nullptr;
  hipblasLtMatmulPreference_t preference = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  uint8_t *packed_weight = nullptr;
  uint8_t *packed_activation = nullptr;
  uint8_t *weight_scales = nullptr;
  uint8_t *activation_scales = nullptr;
  uint16_t *output = nullptr;
  void *workspace = nullptr;
};

struct HostInputs final {
  std::vector<uint8_t> packed_weight;
  std::vector<uint8_t> packed_activation;
  std::vector<uint8_t> weight_scales;
  std::vector<uint8_t> activation_scales;
};

struct ByteSizes final {
  std::size_t packed_weight = 0U;
  std::size_t packed_activation = 0U;
  std::size_t weight_scales = 0U;
  std::size_t activation_scales = 0U;
  std::size_t output = 0U;
};

struct Timing final {
  std::array<float, kMeasured> milliseconds{};
  float median_ms = 0.0F;
  float minimum_ms = 0.0F;
  float maximum_ms = 0.0F;
  std::size_t deterministic_mismatches = 0U;
  std::vector<uint16_t> output;
};

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::cerr << operation << " failed: " << hipGetErrorName(status) << " ("
            << hipGetErrorString(status) << ")\n";
  return false;
}

bool lt_ok(const hipblasStatus_t status, const char *const operation) {
  if (status == HIPBLAS_STATUS_SUCCESS) {
    return true;
  }
  std::cerr << operation << " failed: hipBLAS status "
            << static_cast<int>(status) << "\n";
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
  const std::string_view input(text);
  int parsed = -1;
  const auto result =
      std::from_chars(input.data(), input.data() + input.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != input.data() + input.size() ||
      parsed < 0) {
    return false;
  }
  *device = parsed;
  return true;
}

bool checked_multiply(const uint64_t lhs, const uint64_t rhs,
                      std::size_t *const result) {
  if (result == nullptr || (rhs != 0U && lhs > SIZE_MAX / rhs)) {
    return false;
  }
  *result = static_cast<std::size_t>(lhs * rhs);
  return true;
}

bool byte_sizes(const Shape &shape, ByteSizes *const sizes) {
  if (sizes == nullptr || shape.m == 0U || shape.k == 0U || shape.n == 0U ||
      (shape.k % kBlockK) != 0U || (shape.k % 2U) != 0U) {
    return false;
  }
  std::size_t weight_values = 0U;
  std::size_t activation_values = 0U;
  std::size_t output_elements = 0U;
  if (!checked_multiply(shape.n, shape.k, &weight_values) ||
      !checked_multiply(shape.m, shape.k, &activation_values) ||
      !checked_multiply(shape.m, shape.n, &output_elements) ||
      output_elements > SIZE_MAX / sizeof(uint16_t)) {
    return false;
  }
  sizes->packed_weight = weight_values / 2U;
  sizes->packed_activation = activation_values / 2U;
  sizes->weight_scales = weight_values / kBlockK;
  sizes->activation_scales = activation_values / kBlockK;
  sizes->output = output_elements * sizeof(uint16_t);
  return true;
}

void release(Resources *const resources) {
  if (resources == nullptr) {
    return;
  }
  if (resources->workspace != nullptr) {
    (void)hipFree(resources->workspace);
  }
  if (resources->output != nullptr) {
    (void)hipFree(resources->output);
  }
  if (resources->activation_scales != nullptr) {
    (void)hipFree(resources->activation_scales);
  }
  if (resources->weight_scales != nullptr) {
    (void)hipFree(resources->weight_scales);
  }
  if (resources->packed_activation != nullptr) {
    (void)hipFree(resources->packed_activation);
  }
  if (resources->packed_weight != nullptr) {
    (void)hipFree(resources->packed_weight);
  }
  if (resources->stop != nullptr) {
    (void)hipEventDestroy(resources->stop);
  }
  if (resources->start != nullptr) {
    (void)hipEventDestroy(resources->start);
  }
  if (resources->stream != nullptr) {
    (void)hipStreamDestroy(resources->stream);
  }
  if (resources->preference != nullptr) {
    (void)hipblasLtMatmulPreferenceDestroy(resources->preference);
  }
  if (resources->d != nullptr) {
    (void)hipblasLtMatrixLayoutDestroy(resources->d);
  }
  if (resources->c != nullptr) {
    (void)hipblasLtMatrixLayoutDestroy(resources->c);
  }
  if (resources->b != nullptr) {
    (void)hipblasLtMatrixLayoutDestroy(resources->b);
  }
  if (resources->a != nullptr) {
    (void)hipblasLtMatrixLayoutDestroy(resources->a);
  }
  if (resources->operation != nullptr) {
    (void)hipblasLtMatmulDescDestroy(resources->operation);
  }
  if (resources->handle != nullptr) {
    (void)hipblasLtDestroy(resources->handle);
  }
  *resources = {};
}

bool allocate(Resources *const resources, const ByteSizes &sizes) {
  return hip_ok(hipMalloc(reinterpret_cast<void **>(&resources->packed_weight),
                          sizes.packed_weight),
                "hipMalloc packed weight") &&
         hip_ok(
             hipMalloc(reinterpret_cast<void **>(&resources->packed_activation),
                       sizes.packed_activation),
             "hipMalloc packed activation") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&resources->weight_scales),
                          sizes.weight_scales),
                "hipMalloc weight scales") &&
         hip_ok(
             hipMalloc(reinterpret_cast<void **>(&resources->activation_scales),
                       sizes.activation_scales),
             "hipMalloc activation scales") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&resources->output),
                          sizes.output),
                "hipMalloc BF16 output") &&
         hip_ok(hipStreamCreate(&resources->stream), "hipStreamCreate") &&
         hip_ok(hipEventCreate(&resources->start), "hipEventCreate start") &&
         hip_ok(hipEventCreate(&resources->stop), "hipEventCreate stop");
}

uint32_t mix32(uint32_t value) {
  value ^= value >> 16U;
  value *= UINT32_C(0x7feb352d);
  value ^= value >> 15U;
  value *= UINT32_C(0x846ca68b);
  return value ^ (value >> 16U);
}

uint8_t positive_finite_e4m3(const uint64_t index, const uint32_t seed) {
  // 73 is coprime with 127: every consecutive 127 entries visit the full
  // non-negative finite E4M3FN code corpus 0x00..0x7e exactly once.
  return static_cast<uint8_t>((index * UINT64_C(73) + seed) % UINT64_C(127));
}

HostInputs make_inputs(const Shape &shape, const ByteSizes &sizes) {
  HostInputs inputs;
  inputs.packed_weight.assign(sizes.packed_weight, 0U);
  inputs.packed_activation.assign(sizes.packed_activation, 0U);
  inputs.weight_scales.resize(sizes.weight_scales);
  inputs.activation_scales.resize(sizes.activation_scales);
  const uint64_t blocks = shape.k / kBlockK;
  for (uint64_t row = 0U; row < shape.m; ++row) {
    for (uint64_t inner = 0U; inner < shape.k; ++inner) {
      const uint32_t ordinal = static_cast<uint32_t>(row * shape.k + inner);
      const uint8_t code =
          static_cast<uint8_t>(mix32(ordinal ^ kSeed) & UINT32_C(0x0f));
      const std::size_t index =
          static_cast<std::size_t>(row * shape.k / 2U + inner / 2U);
      if ((inner & 1U) == 0U) {
        inputs.packed_activation[index] = code;
      } else {
        inputs.packed_activation[index] |= static_cast<uint8_t>(code << 4U);
      }
    }
    for (uint64_t block = 0U; block < blocks; ++block) {
      inputs.activation_scales[static_cast<std::size_t>(row * blocks + block)] =
          positive_finite_e4m3(row * blocks + block, kSeed);
    }
  }
  for (uint64_t column = 0U; column < shape.n; ++column) {
    for (uint64_t inner = 0U; inner < shape.k; ++inner) {
      const uint32_t ordinal = static_cast<uint32_t>(column * shape.k + inner);
      const uint8_t code = static_cast<uint8_t>(
          mix32(ordinal ^ kSeed ^ UINT32_C(0x9e3779b9)) & UINT32_C(0x0f));
      const std::size_t index =
          static_cast<std::size_t>(column * shape.k / 2U + inner / 2U);
      if ((inner & 1U) == 0U) {
        inputs.packed_weight[index] = code;
      } else {
        inputs.packed_weight[index] |= static_cast<uint8_t>(code << 4U);
      }
    }
    for (uint64_t block = 0U; block < blocks; ++block) {
      inputs.weight_scales[static_cast<std::size_t>(column * blocks + block)] =
          positive_finite_e4m3(column * blocks + block,
                               kSeed ^ UINT32_C(0xa5a5a5a5));
    }
  }
  return inputs;
}

bool upload(const Resources &resources, const ByteSizes &sizes,
            const HostInputs &inputs) {
  return hip_ok(hipMemcpy(resources.packed_weight, inputs.packed_weight.data(),
                          sizes.packed_weight, hipMemcpyHostToDevice),
                "copy packed weight") &&
         hip_ok(hipMemcpy(resources.packed_activation,
                          inputs.packed_activation.data(),
                          sizes.packed_activation, hipMemcpyHostToDevice),
                "copy packed activation") &&
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

Outcome descriptor_status(const hipblasStatus_t status,
                          const char *const operation) {
  if (status == HIPBLAS_STATUS_SUCCESS) {
    return Outcome::Pass;
  }
  std::cout << "descriptor_step operation=" << operation
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
  if (status != HIPBLAS_STATUS_SUCCESS) {
    lt_ok(status, "set TRANSA");
    return Outcome::Fail;
  }
  status = hipblasLtMatmulDescSetAttribute(resources->operation,
                                           HIPBLASLT_MATMUL_DESC_TRANSB,
                                           &trans_b, sizeof(trans_b));
  if (status != HIPBLAS_STATUS_SUCCESS) {
    lt_ok(status, "set TRANSB");
    return Outcome::Fail;
  }

  void *const weight_scale_pointer = resources->weight_scales;
  void *const activation_scale_pointer = resources->activation_scales;
  constexpr hipblasLtMatmulMatrixScale_t scale_mode =
      HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  status = hipblasLtMatmulDescSetAttribute(
      resources->operation, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
      &weight_scale_pointer, sizeof(weight_scale_pointer));
  Outcome outcome = descriptor_status(status, "A_SCALE_POINTER");
  if (outcome != Outcome::Pass) {
    return outcome;
  }
  status = hipblasLtMatmulDescSetAttribute(
      resources->operation, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
      &activation_scale_pointer, sizeof(activation_scale_pointer));
  outcome = descriptor_status(status, "B_SCALE_POINTER");
  if (outcome != Outcome::Pass) {
    return outcome;
  }
  status = hipblasLtMatmulDescSetAttribute(resources->operation,
                                           HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                           &scale_mode, sizeof(scale_mode));
  outcome = descriptor_status(status, "A_SCALE_MODE_VEC16_UE4M3");
  if (outcome != Outcome::Pass) {
    return outcome;
  }
  status = hipblasLtMatmulDescSetAttribute(resources->operation,
                                           HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                           &scale_mode, sizeof(scale_mode));
  outcome = descriptor_status(status, "B_SCALE_MODE_VEC16_UE4M3");
  if (outcome != Outcome::Pass) {
    return outcome;
  }

  const auto fp4 = static_cast<hipDataType>(HIP_R_4F_E2M1_EXT);
  status = hipblasLtMatrixLayoutCreate(&resources->a, fp4, shape.k, shape.n,
                                       static_cast<int64_t>(shape.k));
  outcome = descriptor_status(status, "A_LAYOUT_FP4_KxN");
  if (outcome != Outcome::Pass) {
    return outcome;
  }
  status = hipblasLtMatrixLayoutCreate(&resources->b, fp4, shape.k, shape.m,
                                       static_cast<int64_t>(shape.k));
  outcome = descriptor_status(status, "B_LAYOUT_FP4_KxM");
  if (outcome != Outcome::Pass) {
    return outcome;
  }
  status = hipblasLtMatrixLayoutCreate(&resources->c, HIP_R_16BF, shape.n,
                                       shape.m, static_cast<int64_t>(shape.n));
  outcome = descriptor_status(status, "C_LAYOUT_BF16_NxM");
  if (outcome != Outcome::Pass) {
    return outcome;
  }
  status = hipblasLtMatrixLayoutCreate(&resources->d, HIP_R_16BF, shape.n,
                                       shape.m, static_cast<int64_t>(shape.n));
  outcome = descriptor_status(status, "D_LAYOUT_BF16_NxM");
  if (outcome != Outcome::Pass) {
    return outcome;
  }
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

  std::cout << "descriptor shape=" << shape.name << " m=" << shape.m
            << " k=" << shape.k << " n=" << shape.n
            << " A=weight[col-major,KxN,ld=K,packed-E2M1,op=T]"
            << " B=activation[col-major,KxM,ld=K,packed-E2M1,op=N]"
            << " C=D[col-major,NxM,ld=N,BF16] compute=FP32"
            << " scaleA=E4M3[K-block16-per-N]"
            << " scaleB=E4M3[K-block16-per-M]"
            << " scale_mode=VEC16_UE4M3 alpha=" << kInputGlobal << '*'
            << kWeightGlobal << '=' << kAlpha
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
  std::cout << "heuristic_query shape=" << shape.name
            << " status=" << static_cast<int>(status)
            << " requested=" << kRequestedAlgorithms << " returned=" << returned
            << "\n";
  if (n0_status(status) ||
      (status == HIPBLAS_STATUS_SUCCESS && returned == 0)) {
    return Outcome::N0;
  }
  if (status != HIPBLAS_STATUS_SUCCESS || returned < 0 ||
      returned > kRequestedAlgorithms) {
    lt_ok(status, "hipblasLtMatmulAlgoGetHeuristic");
    return Outcome::Fail;
  }

  const auto &rank_zero = results[0];
  if (rank_zero.state != HIPBLAS_STATUS_SUCCESS ||
      rank_zero.workspaceSize > kWorkspaceLimit) {
    std::cout << "rank0_unusable shape=" << shape.name
              << " state=" << static_cast<int>(rank_zero.state)
              << " workspace=" << rank_zero.workspaceSize << "\n";
    return Outcome::N0;
  }

  for (int rank = 0; rank < returned; ++rank) {
    const auto &entry = results[static_cast<std::size_t>(rank)];
    if (entry.state != HIPBLAS_STATUS_SUCCESS) {
      std::cout << "heuristic shape=" << shape.name << " rank=" << rank
                << " algorithm_index=unavailable state="
                << static_cast<int>(entry.state)
                << " workspace=" << entry.workspaceSize
                << " waves=" << entry.wavesCount << "\n";
      continue;
    }
    hipblasLtMatmulAlgo_t algorithm = entry.algo;
    const int algorithm_index = hipblaslt_ext::getIndexFromAlgo(algorithm);
    const std::string solution =
        hipblaslt_ext::getSolutionNameFromAlgo(resources->handle, algorithm);
    const std::string kernel =
        hipblaslt_ext::getKernelNameFromAlgo(resources->handle, algorithm);
    std::cout << "heuristic shape=" << shape.name << " rank=" << rank
              << " algorithm_index=" << algorithm_index
              << " state=" << static_cast<int>(entry.state)
              << " workspace=" << entry.workspaceSize
              << " waves=" << entry.wavesCount
              << " solution=" << std::quoted(solution)
              << " kernel=" << std::quoted(kernel) << "\n";
  }
  selected->rank = 0;
  selected->state = rank_zero.state;
  selected->workspace_bytes = rank_zero.workspaceSize;
  selected->waves = rank_zero.wavesCount;
  selected->algorithm = rank_zero.algo;
  selected->algorithm_index =
      hipblaslt_ext::getIndexFromAlgo(selected->algorithm);
  selected->solution_name = hipblaslt_ext::getSolutionNameFromAlgo(
      resources->handle, selected->algorithm);
  selected->kernel_name = hipblaslt_ext::getKernelNameFromAlgo(
      resources->handle, selected->algorithm);
  if (selected->workspace_bytes != 0U &&
      !hip_ok(hipMalloc(&resources->workspace, selected->workspace_bytes),
              "hipMalloc selected workspace")) {
    return Outcome::Fail;
  }
  std::cout << "selected shape=" << shape.name
            << " rank=0 algorithm_index=" << selected->algorithm_index
            << " workspace=" << selected->workspace_bytes
            << " solution=" << std::quoted(selected->solution_name)
            << " kernel=" << std::quoted(selected->kernel_name) << "\n";
  return Outcome::Pass;
}

hipblasStatus_t launch(const Resources &resources, const Heuristic &heuristic) {
  constexpr float beta = 0.0F;
  return hipblasLtMatmul(resources.handle, resources.operation, &kAlpha,
                         resources.packed_weight, resources.a,
                         resources.packed_activation, resources.b, &beta,
                         resources.output, resources.c, resources.output,
                         resources.d, &heuristic.algorithm, resources.workspace,
                         heuristic.workspace_bytes, resources.stream);
}

Outcome measure(const Resources &resources, const Heuristic &heuristic,
                const ByteSizes &sizes, Timing *const timing) {
  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    const hipblasStatus_t status = launch(resources, heuristic);
    if (status != HIPBLAS_STATUS_SUCCESS) {
      std::cout << "matmul_warmup status=" << static_cast<int>(status)
                << " iteration=" << warmup << "\n";
      return n0_status(status) ? Outcome::N0 : Outcome::Fail;
    }
  }
  if (!hip_ok(hipStreamSynchronize(resources.stream), "warmup synchronize")) {
    return Outcome::Fail;
  }

  const std::size_t output_elements = sizes.output / sizeof(uint16_t);
  std::vector<uint16_t> current(output_elements);
  timing->output.resize(output_elements);
  for (int iteration = 0; iteration < kMeasured; ++iteration) {
    if (!hip_ok(hipEventRecord(resources.start, resources.stream),
                "record timing start")) {
      return Outcome::Fail;
    }
    const hipblasStatus_t status = launch(resources, heuristic);
    if (status != HIPBLAS_STATUS_SUCCESS) {
      std::cout << "matmul_timed status=" << static_cast<int>(status)
                << " iteration=" << iteration << "\n";
      return n0_status(status) ? Outcome::N0 : Outcome::Fail;
    }
    if (!hip_ok(hipEventRecord(resources.stop, resources.stream),
                "record timing stop") ||
        !hip_ok(hipEventSynchronize(resources.stop), "timing synchronize") ||
        !hip_ok(hipEventElapsedTime(
                    &timing->milliseconds[static_cast<std::size_t>(iteration)],
                    resources.start, resources.stop),
                "timing elapsed") ||
        !hip_ok(hipMemcpy(current.data(), resources.output, sizes.output,
                          hipMemcpyDeviceToHost),
                "copy timed output")) {
      return Outcome::Fail;
    }
    if (iteration == 0) {
      timing->output = current;
    } else {
      timing->deterministic_mismatches += static_cast<std::size_t>(
          std::count_if(current.begin(), current.end(),
                        [index = std::size_t{0U}, &reference = timing->output](
                            const uint16_t value) mutable {
                          return value != reference[index++];
                        }));
    }
  }
  std::array<float, kMeasured> sorted = timing->milliseconds;
  std::sort(sorted.begin(), sorted.end());
  timing->minimum_ms = sorted.front();
  timing->median_ms = sorted[sorted.size() / 2U];
  timing->maximum_ms = sorted.back();
  return timing->deterministic_mismatches == 0U ? Outcome::Pass : Outcome::Fail;
}

float host_e2m1(const uint8_t code) {
  constexpr std::array<float, 8> positive = {0.0F, 0.5F, 1.0F, 1.5F,
                                             2.0F, 3.0F, 4.0F, 6.0F};
  const float magnitude = positive[code & UINT8_C(7)];
  return (code & UINT8_C(8)) == 0U ? magnitude : -magnitude;
}

float host_e4m3(const uint8_t bits) {
  const uint32_t magnitude = bits & UINT8_C(0x7f);
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & 7U;
  if (exponent == 0U) {
    return static_cast<float>(mantissa) * 0x1p-9F;
  }
  if (magnitude == 0x7fU) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                    static_cast<int>(exponent) - 7);
}

uint16_t host_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
    if ((bits & UINT32_C(0x007fffff)) != 0U) {
      return static_cast<uint16_t>(((bits >> 16U) & UINT32_C(0x8000)) |
                                   UINT32_C(0x7fc0) |
                                   ((bits >> 16U) & UINT32_C(0x003f)));
    }
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & 1U) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

float host_bf16_to_float(const uint16_t bits) {
  const uint32_t expanded = static_cast<uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &expanded, sizeof(value));
  return value;
}

uint32_t ordered_bf16(const uint16_t bits) {
  if ((bits & UINT16_C(0x7fff)) == 0U) {
    return UINT32_C(0x8000);
  }
  return (bits & UINT16_C(0x8000)) != 0U
             ? static_cast<uint16_t>(~bits)
             : static_cast<uint16_t>(bits | UINT16_C(0x8000));
}

uint64_t hash_bf16(const std::span<const uint16_t> values) {
  uint64_t hash = UINT64_C(1469598103934665603);
  for (const uint16_t value : values) {
    hash ^= static_cast<uint8_t>(value & UINT16_C(0xff));
    hash *= UINT64_C(1099511628211);
    hash ^= static_cast<uint8_t>(value >> 8U);
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

void print_id64_signature(const Shape &shape,
                          const std::vector<uint16_t> &output) {
  std::cout << "id64_compare_output shape=" << shape.name
            << " generator=phase78-id64 seed=0x" << std::hex << kSeed
            << " bf16_fnv64=0x" << hash_bf16(output) << std::dec
            << " sample64=";
  constexpr std::size_t sample_count = 64U;
  for (std::size_t sample = 0U; sample < sample_count; ++sample) {
    const std::size_t index =
        output.size() == 1U
            ? 0U
            : sample * (output.size() - 1U) / (sample_count - 1U);
    if (sample != 0U) {
      std::cout << ',';
    }
    std::cout << index << ":0x" << std::hex << std::setw(4) << std::setfill('0')
              << output[index] << std::dec;
  }
  std::cout << std::setfill(' ') << "\n";
}

bool check_host_double_oracle(const Shape &shape, const HostInputs &inputs,
                              const std::vector<uint16_t> &output) {
  const uint64_t blocks = shape.k / kBlockK;
  std::vector<uint16_t> expected(output.size());
  std::size_t bf16_mismatches = 0U;
  uint32_t max_bf16_ulp = 0U;
  double max_abs = 0.0;
  double max_normalized = 0.0;
  for (uint64_t row = 0U; row < shape.m; ++row) {
    for (uint64_t column = 0U; column < shape.n; ++column) {
      double sum = 0.0;
      double absolute_sum = 0.0;
      for (uint64_t inner = 0U; inner < shape.k; ++inner) {
        const uint8_t activation_pair =
            inputs.packed_activation[static_cast<std::size_t>(
                row * shape.k / 2U + inner / 2U)];
        const uint8_t weight_pair =
            inputs.packed_weight[static_cast<std::size_t>(
                column * shape.k / 2U + inner / 2U)];
        const uint8_t activation_code = (inner & 1U) == 0U
                                            ? activation_pair & UINT8_C(0x0f)
                                            : activation_pair >> 4U;
        const uint8_t weight_code = (inner & 1U) == 0U
                                        ? weight_pair & UINT8_C(0x0f)
                                        : weight_pair >> 4U;
        const double term =
            static_cast<double>(host_e2m1(activation_code)) *
            host_e4m3(inputs.activation_scales[static_cast<std::size_t>(
                row * blocks + inner / kBlockK)]) *
            static_cast<double>(host_e2m1(weight_code)) *
            host_e4m3(inputs.weight_scales[static_cast<std::size_t>(
                column * blocks + inner / kBlockK)]);
        sum += term;
        absolute_sum += std::abs(term);
      }
      sum *= static_cast<double>(kInputGlobal) *
             static_cast<double>(kWeightGlobal);
      absolute_sum *= static_cast<double>(kInputGlobal) *
                      static_cast<double>(kWeightGlobal);
      const std::size_t index =
          static_cast<std::size_t>(row * shape.n + column);
      expected[index] = host_bf16_rne(static_cast<float>(sum));
      const double observed = host_bf16_to_float(output[index]);
      const double absolute_error = std::abs(observed - sum);
      max_abs = std::max(max_abs, absolute_error);
      max_normalized = std::max(
          max_normalized,
          absolute_error /
              std::max(absolute_sum, std::numeric_limits<double>::min()));
      if (expected[index] != output[index]) {
        ++bf16_mismatches;
      }
      const uint32_t lhs = ordered_bf16(expected[index]);
      const uint32_t rhs = ordered_bf16(output[index]);
      max_bf16_ulp = std::max(max_bf16_ulp, lhs > rhs ? lhs - rhs : rhs - lhs);
    }
  }
  std::cout << "host_double_oracle shape=" << shape.name
            << " expected_bf16_fnv64=0x" << std::hex << hash_bf16(expected)
            << " actual_bf16_fnv64=0x" << hash_bf16(output) << std::dec
            << " bf16_mismatches=" << bf16_mismatches
            << " max_bf16_ulp=" << max_bf16_ulp
            << " max_abs=" << std::setprecision(10) << max_abs
            << " max_normalized_error=" << max_normalized
            << " tolerance=0.01 status="
            << (max_normalized <= 0.01 ? "PASS" : "FAIL") << "\n";
  return max_normalized <= 0.01;
}

Outcome run_shape(const Shape &shape) {
  ByteSizes sizes;
  if (!byte_sizes(shape, &sizes)) {
    std::cerr << "invalid or overflowing shape " << shape.name << "\n";
    return Outcome::Fail;
  }
  Resources resources;
  if (!allocate(&resources, sizes)) {
    release(&resources);
    return Outcome::Fail;
  }
  Outcome outcome = create_descriptors(&resources, shape);
  if (outcome != Outcome::Pass) {
    release(&resources);
    return outcome;
  }
  Heuristic heuristic;
  outcome = query_rank_zero(&resources, shape, &heuristic);
  if (outcome != Outcome::Pass) {
    release(&resources);
    return outcome;
  }

  const HostInputs inputs = make_inputs(shape, sizes);
  if (!upload(resources, sizes, inputs)) {
    release(&resources);
    return Outcome::Fail;
  }
  Timing timing;
  outcome = measure(resources, heuristic, sizes, &timing);
  if (outcome != Outcome::Pass) {
    release(&resources);
    return outcome;
  }

  const double operations = 2.0 * static_cast<double>(shape.m) *
                            static_cast<double>(shape.k) *
                            static_cast<double>(shape.n);
  const double tflops =
      operations / (static_cast<double>(timing.median_ms) * 1.0e9);
  const double traffic_bytes = static_cast<double>(
      sizes.packed_weight + sizes.packed_activation + sizes.weight_scales +
      sizes.activation_scales + sizes.output);
  const double gigabytes_per_second =
      traffic_bytes / (static_cast<double>(timing.median_ms) * 1.0e6);
  std::cout << "timing shape=" << shape.name << " warmups=" << kWarmups
            << " measured=" << kMeasured << " samples_ms=";
  for (std::size_t index = 0U; index < timing.milliseconds.size(); ++index) {
    if (index != 0U) {
      std::cout << ',';
    }
    std::cout << std::fixed << std::setprecision(6)
              << timing.milliseconds[index];
  }
  std::cout << " median_ms=" << timing.median_ms
            << " min_ms=" << timing.minimum_ms
            << " max_ms=" << timing.maximum_ms << " tflops=" << tflops
            << " logical_traffic_gbps=" << gigabytes_per_second
            << " repeat_bf16_mismatches=" << timing.deterministic_mismatches
            << "\n";
  print_id64_signature(shape, timing.output);
  if (shape.host_oracle &&
      !check_host_double_oracle(shape, inputs, timing.output)) {
    release(&resources);
    return Outcome::Fail;
  }
  release(&resources);
  return Outcome::Pass;
}

} // namespace

int main(int argc, char **argv) {
  int device = 0;
  if (argc > 2 || (argc == 2 && !parse_device(argv[1], &device))) {
    std::cerr << "usage: phase78_nvfp4_gfx1201_hipblaslt_vec16_probe "
                 "[DEVICE]\n";
    return EXIT_FAILURE;
  }
  if (!hip_ok(hipSetDevice(device), "hipSetDevice")) {
    return EXIT_FAILURE;
  }
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, device),
              "hipGetDeviceProperties")) {
    return EXIT_FAILURE;
  }
  int runtime_version = 0;
  if (!hip_ok(hipRuntimeGetVersion(&runtime_version), "hipRuntimeGetVersion")) {
    return EXIT_FAILURE;
  }
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
    return EXIT_FAILURE;
  }

  for (const Shape &shape : kShapes) {
    const Outcome outcome = run_shape(shape);
    if (outcome == Outcome::N0) {
      std::cout << "n0 shape=" << shape.name
                << " reason=vec16-ue4m3-fp4-descriptor-or-rank0-unavailable\n"
                << "PHASE78_NVFP4_HIPBLASLT_VEC16_RESULT=N0\n";
      return EXIT_SUCCESS;
    }
    if (outcome == Outcome::Fail) {
      std::cout << "PHASE78_NVFP4_HIPBLASLT_VEC16_RESULT=FAIL\n";
      return EXIT_FAILURE;
    }
  }
  std::cout << "PHASE78_NVFP4_HIPBLASLT_VEC16_RESULT=PASS\n";
  return EXIT_SUCCESS;
}
