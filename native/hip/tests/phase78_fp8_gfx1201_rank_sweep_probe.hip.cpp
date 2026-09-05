// Phase 78 standalone gfx1201 FP8 outer-vector M=1 hipBLASLt rank sweep.
//
// This probe is intentionally outside the production build.  It reproduces
// the exact Qwen3.8-27B FP8 W8A8 decode contract:
//   * weight values are column-major-by-logical-row byte planes, followed by
//     N FP32 outer (channel) scales;
//   * activation values are BF16 at ingress and are quantized into a
//     request-local workspace, with one aligned FP32 outer scale after the
//     M*K FP8 value bytes;
//   * hipBLASLt receives the same transposed A / non-transposed B layouts and
//     OUTER_VEC_32F scale pointers used by the production Fp8Native route.
//
// The shape/occurrence manifest below is derived from
// crates/sllm-core/src/quantized_model.rs::build_qwen38_inventory:
// 16 full-attention layers (every fourth layer), 48 linear-attention layers,
// FP8 MLP on layers 56..63, and one FP8 lm_head.  It is deliberately kept in
// this standalone file so a probe run cannot silently use a synthetic shape
// set.  The run still accepts SLLM_FP8_RANK_SWEEP_SHAPE=KxN for a bounded
// single-shape execution during development.

#include "low_precision_block_codec.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_version.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "gfx1201"
#endif

namespace {

constexpr uint32_t kThreads = 256U;
constexpr int kRequestedAlgorithms = 32;
constexpr int kWarmups = 3;
constexpr int kMeasured = 10;
constexpr uint64_t kScaleAlignment = 4U;

struct ShapeCase final {
  uint64_t k;
  uint64_t n;
  uint32_t occurrences;
  const char *roles;
};

// Exact FP8 tensor shape multiplicity in the locked Qwen3.8-27B mixed plan.
// Logical tensor shape is [N,K], while the probe reports the matmul K,N pair.
constexpr std::array<ShapeCase, 8> kQwen38Fp8Shapes = {{
    {5120U, 17408U, 16U, "layers56-63.mlp.gate+up"},
    {17408U, 5120U, 8U, "layers56-63.mlp.down"},
    {5120U, 12288U, 16U, "16.full-attn.q"},
    {5120U, 1024U, 32U, "16.full-attn.k+v"},
    {6144U, 5120U, 64U, "full-attn.o+linear-attn.out"},
    {5120U, 10240U, 48U, "48.linear-attn.qkv"},
    {5120U, 6144U, 48U, "48.linear-attn.z"},
    {5120U, 248320U, 1U, "lm_head"},
}};

constexpr uint32_t shape_occurrences() {
  uint32_t result = 0U;
  for (const ShapeCase &shape : kQwen38Fp8Shapes) {
    result += shape.occurrences;
  }
  return result;
}

static_assert(shape_occurrences() == 233U,
              "Qwen3.8 exact FP8 inventory must contain 233 tensors");

__device__ __forceinline__ float bf16_to_float(const uint16_t bits) noexcept {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16U);
}

// This is the production v2 quantizer's operation order, including one
// block-wide max reduction and E4M3FN scale denominator 448.  It emits one
// byte per activation value because the Lt B descriptor consumes byte strides.
__global__ __launch_bounds__(kThreads, 1) void fp8_outer_quantize_kernel(
    const uint16_t *const activation, uint8_t *const quantized,
    float *const scales, const uint64_t m, const uint64_t k) {
  const uint64_t row = blockIdx.x;
  if (row >= m) {
    return;
  }
  float maximum = 0.0F;
  for (uint64_t column = threadIdx.x; column < k; column += blockDim.x) {
    maximum =
        fmaxf(maximum, fabsf(bf16_to_float(activation[row * k + column])));
  }
  __shared__ float reductions[kThreads];
  reductions[threadIdx.x] = maximum;
  __syncthreads();
  for (uint32_t offset = kThreads / 2U; offset != 0U; offset >>= 1U) {
    if (threadIdx.x < offset) {
      reductions[threadIdx.x] =
          fmaxf(reductions[threadIdx.x], reductions[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  const float scale = reductions[0] == 0.0F ? 1.0F : reductions[0] / 448.0F;
  if (threadIdx.x == 0U) {
    scales[row] = scale;
  }
  __syncthreads();
  for (uint64_t column = threadIdx.x; column < k; column += blockDim.x) {
    const float value = bf16_to_float(activation[row * k + column]) / scale;
    quantized[row * k + column] =
        sllm_lowp::ScalarCodec<sllm_lowp::E4M3Fn>::encode(value);
  }
}

struct Heuristic final {
  int rank = -1;
  int algorithm_index = -1;
  std::string solution_name;
  std::string kernel_name;
  std::size_t workspace_bytes = 0U;
  float waves = 0.0F;
  hipblasStatus_t state = HIPBLAS_STATUS_NOT_INITIALIZED;
  hipblasLtMatmulAlgo_t algorithm{};
};

struct Timing final {
  std::vector<double> milliseconds;
  double median = std::numeric_limits<double>::quiet_NaN();
  double mad = std::numeric_limits<double>::quiet_NaN();
  double minimum = std::numeric_limits<double>::quiet_NaN();
  double maximum = std::numeric_limits<double>::quiet_NaN();
};

struct ShapeResources final {
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
  uint16_t *activation_bf16 = nullptr;
  uint8_t *activation_workspace = nullptr;
  uint8_t *weight_storage = nullptr;
  uint16_t *output = nullptr;
  std::size_t activation_value_bytes = 0U;
  std::size_t activation_scale_offset = 0U;
  std::size_t weight_value_bytes = 0U;
  std::size_t weight_scale_bytes = 0U;
  std::size_t workspace_bytes = 0U;
  std::vector<Heuristic> heuristics;
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

bool exact_gfx1201(const char *const arch) {
  if (arch == nullptr) {
    return false;
  }
  const std::string_view value(arch);
  constexpr std::string_view prefix = "gfx1201";
  return value == prefix || (value.size() > prefix.size() &&
                             value.compare(0U, prefix.size(), prefix) == 0 &&
                             value[prefix.size()] == ':');
}

uint16_t host_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t exponent = bits & UINT32_C(0x7f800000);
  const uint32_t fraction = bits & UINT32_C(0x007fffff);
  if (exponent == UINT32_C(0x7f800000)) {
    if (fraction != 0U) {
      return static_cast<uint16_t>((bits >> 16U) | UINT32_C(0x0040));
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

float host_fp8_e4m3fn_to_float(const uint8_t code) {
  const uint8_t magnitude = code & UINT8_C(0x7f);
  const uint8_t exponent = magnitude >> 3U;
  const uint8_t mantissa = magnitude & UINT8_C(0x07);
  float value = 0.0F;
  if (exponent == 0U) {
    value = static_cast<float>(mantissa) * 0x1p-9F;
  } else if (magnitude == UINT8_C(0x7f)) {
    value = std::numeric_limits<float>::quiet_NaN();
  } else {
    value = std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                       static_cast<int>(exponent) - 7);
  }
  return (code & UINT8_C(0x80)) != 0U ? -value : value;
}

uint64_t hash_u16(const std::vector<uint16_t> &values) {
  uint64_t hash = UINT64_C(1469598103934665603);
  for (const uint16_t value : values) {
    hash ^= static_cast<uint64_t>(value & UINT16_C(0xff));
    hash *= UINT64_C(1099511628211);
    hash ^= static_cast<uint64_t>(value >> 8U);
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

void release_shape(ShapeResources *const resources) {
  if (resources == nullptr) {
    return;
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
  if (resources->output != nullptr) {
    (void)hipFree(resources->output);
  }
  if (resources->weight_storage != nullptr) {
    (void)hipFree(resources->weight_storage);
  }
  if (resources->activation_workspace != nullptr) {
    (void)hipFree(resources->activation_workspace);
  }
  if (resources->activation_bf16 != nullptr) {
    (void)hipFree(resources->activation_bf16);
  }
  *resources = {};
}

bool checked_product(const uint64_t left, const uint64_t right,
                     std::size_t *const result) {
  if (result == nullptr || left == 0U || right == 0U ||
      left > static_cast<uint64_t>(SIZE_MAX) / right) {
    return false;
  }
  *result = static_cast<std::size_t>(left * right);
  return true;
}

bool fill_host_inputs(const ShapeCase &shape,
                      std::vector<uint16_t> *const activation,
                      std::vector<uint8_t> *const weight,
                      std::vector<float> *const weight_scales,
                      std::vector<uint8_t> *const expected_activation) {
  std::size_t weight_elements = 0U;
  if (activation == nullptr || weight == nullptr || weight_scales == nullptr ||
      expected_activation == nullptr ||
      !checked_product(1U, shape.k, &weight_elements) ||
      !checked_product(shape.k, shape.n, &weight_elements)) {
    return false;
  }
  activation->resize(static_cast<std::size_t>(shape.k));
  expected_activation->resize(static_cast<std::size_t>(shape.k));
  weight->resize(weight_elements);
  weight_scales->resize(static_cast<std::size_t>(shape.n));

  // One maximum code makes the production max/448 scale exactly 1/8.  Every
  // generated BF16 value is therefore an exact BF16 representation of the
  // selected E4M3FN code times the expected dynamic scale.
  constexpr float activation_scale = 0.125F;
  constexpr std::array<uint8_t, 12> activation_codes = {
      UINT8_C(0x7e), UINT8_C(0x30), UINT8_C(0xb0), UINT8_C(0x38),
      UINT8_C(0xb8), UINT8_C(0x40), UINT8_C(0xc0), UINT8_C(0x28),
      UINT8_C(0xa8), UINT8_C(0x20), UINT8_C(0xa0), UINT8_C(0x00)};
  for (uint64_t inner = 0U; inner < shape.k; ++inner) {
    const uint8_t code = activation_codes[inner % activation_codes.size()];
    const float value = host_fp8_e4m3fn_to_float(code) * activation_scale;
    (*activation)[static_cast<std::size_t>(inner)] = host_bf16_rne(value);
    (*expected_activation)[static_cast<std::size_t>(inner)] = code;
  }
  constexpr std::array<uint8_t, 12> weight_codes = {
      UINT8_C(0x20), UINT8_C(0xa0), UINT8_C(0x28), UINT8_C(0xa8),
      UINT8_C(0x30), UINT8_C(0xb0), UINT8_C(0x38), UINT8_C(0xb8),
      UINT8_C(0x40), UINT8_C(0xc0), UINT8_C(0x18), UINT8_C(0x98)};
  for (uint64_t column = 0U; column < shape.n; ++column) {
    (*weight_scales)[static_cast<std::size_t>(column)] =
        0x1p-5F * (1.0F + static_cast<float>(column % 17U) * 0.015625F);
    const std::size_t base = static_cast<std::size_t>(column * shape.k);
    for (uint64_t inner = 0U; inner < shape.k; ++inner) {
      (*weight)[base + static_cast<std::size_t>(inner)] =
          weight_codes[(inner + column * 5U) % weight_codes.size()];
    }
  }
  return true;
}

std::vector<uint16_t> cpu_oracle(const ShapeCase &shape,
                                 const std::vector<uint8_t> &activation,
                                 const float activation_scale,
                                 const std::vector<uint8_t> &weight,
                                 const std::vector<float> &weight_scales) {
  std::vector<uint16_t> result(static_cast<std::size_t>(shape.n));
  for (uint64_t column = 0U; column < shape.n; ++column) {
    float accumulator = 0.0F;
    const std::size_t base = static_cast<std::size_t>(column * shape.k);
    for (uint64_t inner = 0U; inner < shape.k; ++inner) {
      accumulator = std::fmaf(
          host_fp8_e4m3fn_to_float(activation[static_cast<std::size_t>(inner)]),
          host_fp8_e4m3fn_to_float(
              weight[base + static_cast<std::size_t>(inner)]),
          accumulator);
    }
    result[static_cast<std::size_t>(column)] =
        host_bf16_rne(accumulator * activation_scale *
                      weight_scales[static_cast<std::size_t>(column)]);
  }
  return result;
}

std::size_t bf16_ulp_distance(const uint16_t left, const uint16_t right) {
  // The oracle values are finite.  Mapping sign-magnitude BF16 to monotonic
  // unsigned order gives a useful distance for a diagnostics-only report.
  const auto ordered = [](const uint16_t bits) -> uint32_t {
    return (bits & UINT16_C(0x8000)) != 0U
               ? UINT32_C(0x8000) -
                     static_cast<uint32_t>(bits & UINT16_C(0x7fff))
               : UINT32_C(0x8000) + static_cast<uint32_t>(bits);
  };
  const uint32_t a = ordered(left);
  const uint32_t b = ordered(right);
  return a > b ? a - b : b - a;
}

Timing summarize(std::vector<double> samples) {
  Timing result;
  result.milliseconds = std::move(samples);
  if (result.milliseconds.empty()) {
    return result;
  }
  std::sort(result.milliseconds.begin(), result.milliseconds.end());
  result.minimum = result.milliseconds.front();
  result.maximum = result.milliseconds.back();
  result.median = result.milliseconds[result.milliseconds.size() / 2U];
  std::vector<double> deviations;
  deviations.reserve(result.milliseconds.size());
  for (const double sample : result.milliseconds) {
    deviations.push_back(std::fabs(sample - result.median));
  }
  std::sort(deviations.begin(), deviations.end());
  result.mad = deviations[deviations.size() / 2U];
  return result;
}

bool parse_shape_filter(std::vector<ShapeCase> *const selected) {
  if (selected == nullptr) {
    return false;
  }
  const char *const filter = std::getenv("SLLM_FP8_RANK_SWEEP_SHAPE");
  if (filter == nullptr || *filter == '\0') {
    selected->assign(kQwen38Fp8Shapes.begin(), kQwen38Fp8Shapes.end());
    return true;
  }
  const std::string value(filter);
  const std::size_t separator = value.find('x');
  if (separator == std::string::npos || separator == 0U ||
      separator + 1U >= value.size()) {
    std::cerr << "SLLM_FP8_RANK_SWEEP_SHAPE must be KxN\n";
    return false;
  }
  char *end = nullptr;
  const uint64_t k =
      std::strtoull(value.substr(0U, separator).c_str(), &end, 10);
  if (end == nullptr || *end != '\0') {
    return false;
  }
  end = nullptr;
  const uint64_t n =
      std::strtoull(value.substr(separator + 1U).c_str(), &end, 10);
  if (end == nullptr || *end != '\0') {
    return false;
  }
  for (const ShapeCase &shape : kQwen38Fp8Shapes) {
    if (shape.k == k && shape.n == n) {
      selected->push_back(shape);
      return true;
    }
  }
  std::cerr << "shape is not in the exact Qwen3.8 FP8 manifest: " << value
            << "\n";
  return false;
}

bool check_identity() {
  int count = 0;
  if (!hip_ok(hipGetDeviceCount(&count), "hipGetDeviceCount") || count != 1) {
    std::cerr << "expected one visible GPU, got " << count << "\n";
    return false;
  }
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, 0),
              "hipGetDeviceProperties")) {
    return false;
  }
  int runtime = 0;
  if (!hip_ok(hipRuntimeGetVersion(&runtime), "hipRuntimeGetVersion")) {
    return false;
  }
  std::cout << "identity compile_target=" << SLLM_TEST_EXPECTED_TARGET
            << " runtime_target=" << properties.gcnArchName
            << " hip_header=" << HIP_VERSION_MAJOR << '.' << HIP_VERSION_MINOR
            << '.' << HIP_VERSION_PATCH << " hip_runtime=" << runtime
            << " visible_devices=" << count << "\n";
  if (std::strcmp(SLLM_TEST_EXPECTED_TARGET, "gfx1201") != 0 ||
      !exact_gfx1201(properties.gcnArchName)) {
    std::cerr << "the rank sweep requires exact gfx1201\n";
    return false;
  }
  return hip_ok(hipSetDevice(0), "hipSetDevice(0)");
}

bool allocate_shape(ShapeResources *const resources, const ShapeCase &shape) {
  if (resources == nullptr ||
      !checked_product(1U, shape.k, &resources->activation_value_bytes) ||
      !checked_product(shape.k, shape.n, &resources->weight_value_bytes)) {
    return false;
  }
  resources->activation_scale_offset =
      (resources->activation_value_bytes + kScaleAlignment - 1U) &
      ~(kScaleAlignment - 1U);
  resources->weight_scale_bytes =
      static_cast<std::size_t>(shape.n) * sizeof(float);
  resources->workspace_bytes =
      resources->activation_scale_offset + sizeof(float);
  const std::size_t output_bytes =
      static_cast<std::size_t>(shape.n) * sizeof(uint16_t);
  return hip_ok(
             hipMalloc(reinterpret_cast<void **>(&resources->activation_bf16),
                       resources->activation_value_bytes * sizeof(uint16_t)),
             "hipMalloc activation BF16") &&
         hip_ok(hipMalloc(
                    reinterpret_cast<void **>(&resources->activation_workspace),
                    resources->workspace_bytes),
                "hipMalloc activation workspace") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&resources->weight_storage),
                          resources->weight_value_bytes +
                              resources->weight_scale_bytes),
                "hipMalloc weight values+scales") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&resources->output),
                          output_bytes),
                "hipMalloc output") &&
         hip_ok(hipStreamCreate(&resources->stream), "hipStreamCreate") &&
         hip_ok(hipEventCreate(&resources->start), "hipEventCreate start") &&
         hip_ok(hipEventCreate(&resources->stop), "hipEventCreate stop");
}

bool copy_inputs(ShapeResources *const resources, const ShapeCase &shape,
                 const std::vector<uint16_t> &activation,
                 const std::vector<uint8_t> &weight,
                 const std::vector<float> &weight_scales) {
  return hip_ok(hipMemcpy(resources->activation_bf16, activation.data(),
                          activation.size() * sizeof(uint16_t),
                          hipMemcpyHostToDevice),
                "copy BF16 activation") &&
         hip_ok(hipMemcpy(resources->weight_storage, weight.data(),
                          weight.size(), hipMemcpyHostToDevice),
                "copy FP8 weight values") &&
         hip_ok(hipMemcpy(
                    resources->weight_storage + resources->weight_value_bytes,
                    weight_scales.data(), weight_scales.size() * sizeof(float),
                    hipMemcpyHostToDevice),
                "copy FP8 weight scales") &&
         hip_ok(hipMemset(resources->output, 0,
                          static_cast<std::size_t>(shape.n) * sizeof(uint16_t)),
                "clear output");
}

bool create_descriptors(ShapeResources *const resources,
                        const ShapeCase &shape) {
  if (!lt_ok(hipblasLtCreate(&resources->handle), "hipblasLtCreate") ||
      !lt_ok(hipblasLtMatmulDescCreate(&resources->operation,
                                       HIPBLAS_COMPUTE_32F, HIP_R_32F),
             "hipblasLtMatmulDescCreate")) {
    return false;
  }
  const hipblasOperation_t trans_a = HIPBLAS_OP_T;
  const hipblasOperation_t trans_b = HIPBLAS_OP_N;
  if (!lt_ok(hipblasLtMatmulDescSetAttribute(resources->operation,
                                             HIPBLASLT_MATMUL_DESC_TRANSA,
                                             &trans_a, sizeof(trans_a)),
             "set TRANSA") ||
      !lt_ok(hipblasLtMatmulDescSetAttribute(resources->operation,
                                             HIPBLASLT_MATMUL_DESC_TRANSB,
                                             &trans_b, sizeof(trans_b)),
             "set TRANSB")) {
    return false;
  }
  void *const weight_scale_pointer =
      resources->weight_storage + resources->weight_value_bytes;
  void *const activation_scale_pointer =
      resources->activation_workspace + resources->activation_scale_offset;
  const hipblasLtMatmulMatrixScale_t scale_mode =
      HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
  if (!lt_ok(hipblasLtMatmulDescSetAttribute(
                 resources->operation, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                 &weight_scale_pointer, sizeof(weight_scale_pointer)),
             "set A scale pointer") ||
      !lt_ok(hipblasLtMatmulDescSetAttribute(
                 resources->operation, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                 &activation_scale_pointer, sizeof(activation_scale_pointer)),
             "set B scale pointer") ||
      !lt_ok(hipblasLtMatmulDescSetAttribute(resources->operation,
                                             HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                             &scale_mode, sizeof(scale_mode)),
             "set A outer scale mode") ||
      !lt_ok(hipblasLtMatmulDescSetAttribute(resources->operation,
                                             HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                             &scale_mode, sizeof(scale_mode)),
             "set B outer scale mode") ||
      !lt_ok(hipblasLtMatrixLayoutCreate(&resources->a, HIP_R_8F_E4M3, shape.k,
                                         shape.n,
                                         static_cast<int64_t>(shape.k)),
             "create A layout") ||
      !lt_ok(hipblasLtMatrixLayoutCreate(&resources->b, HIP_R_8F_E4M3, shape.k,
                                         1U, static_cast<int64_t>(shape.k)),
             "create B layout") ||
      !lt_ok(hipblasLtMatrixLayoutCreate(&resources->c, HIP_R_16BF, shape.n, 1U,
                                         static_cast<int64_t>(shape.n)),
             "create C layout") ||
      !lt_ok(hipblasLtMatrixLayoutCreate(&resources->d, HIP_R_16BF, shape.n, 1U,
                                         static_cast<int64_t>(shape.n)),
             "create D layout") ||
      !lt_ok(hipblasLtMatmulPreferenceCreate(&resources->preference),
             "hipblasLtMatmulPreferenceCreate")) {
    return false;
  }
  const uint64_t workspace_limit = 0U;
  if (!lt_ok(hipblasLtMatmulPreferenceSetAttribute(
                 resources->preference,
                 HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit,
                 sizeof(workspace_limit)),
             "set zero workspace limit")) {
    return false;
  }
  std::array<hipblasLtMatmulHeuristicResult_t, kRequestedAlgorithms> results{};
  int returned = 0;
  if (!lt_ok(hipblasLtMatmulAlgoGetHeuristic(
                 resources->handle, resources->operation, resources->a,
                 resources->b, resources->c, resources->d,
                 resources->preference, kRequestedAlgorithms, results.data(),
                 &returned),
             "hipblasLtMatmulAlgoGetHeuristic") ||
      returned <= 0) {
    return false;
  }
  for (int rank = 0; rank < returned; ++rank) {
    const auto &entry = results[static_cast<std::size_t>(rank)];
    Heuristic heuristic;
    heuristic.rank = rank;
    heuristic.state = entry.state;
    heuristic.workspace_bytes = entry.workspaceSize;
    heuristic.waves = entry.wavesCount;
    heuristic.algorithm = entry.algo;
    heuristic.algorithm_index =
        hipblaslt_ext::getIndexFromAlgo(heuristic.algorithm);
    heuristic.solution_name = hipblaslt_ext::getSolutionNameFromAlgo(
        resources->handle, heuristic.algorithm);
    heuristic.kernel_name = hipblaslt_ext::getKernelNameFromAlgo(
        resources->handle, heuristic.algorithm);
    resources->heuristics.push_back(std::move(heuristic));
  }
  std::cout << "heuristic returned=" << returned << " usable_zero_workspace="
            << std::count_if(resources->heuristics.begin(),
                             resources->heuristics.end(),
                             [](const Heuristic &value) {
                               return value.state == HIPBLAS_STATUS_SUCCESS &&
                                      value.workspace_bytes == 0U;
                             })
            << "\n";
  return true;
}

hipblasStatus_t launch_matmul(const ShapeResources &resources,
                              const Heuristic &heuristic,
                              const ShapeCase &shape) {
  const float alpha = 1.0F;
  const float beta = 0.0F;
  hipLaunchKernelGGL(
      fp8_outer_quantize_kernel, dim3(1U), dim3(kThreads), 0U, resources.stream,
      resources.activation_bf16, resources.activation_workspace,
      reinterpret_cast<float *>(resources.activation_workspace +
                                resources.activation_scale_offset),
      1U, shape.k);
  const hipError_t launch = hipGetLastError();
  if (launch != hipSuccess) {
    return HIPBLAS_STATUS_INTERNAL_ERROR;
  }
  return hipblasLtMatmul(
      resources.handle, resources.operation, &alpha, resources.weight_storage,
      resources.a, resources.activation_workspace, resources.b, &beta,
      resources.output, resources.c, resources.output, resources.d,
      &heuristic.algorithm, nullptr, 0U, resources.stream);
}

bool launch_quantize(const ShapeResources &resources, const ShapeCase &shape) {
  hipLaunchKernelGGL(
      fp8_outer_quantize_kernel, dim3(1U), dim3(kThreads), 0U, resources.stream,
      resources.activation_bf16, resources.activation_workspace,
      reinterpret_cast<float *>(resources.activation_workspace +
                                resources.activation_scale_offset),
      1U, shape.k);
  return hipGetLastError() == hipSuccess;
}

bool quantized_activation_check(const ShapeResources &resources,
                                const ShapeCase &shape,
                                const std::vector<uint8_t> &expected_values) {
  std::vector<uint8_t> values(expected_values.size());
  float scale = 0.0F;
  if (!hip_ok(hipMemcpy(values.data(), resources.activation_workspace,
                        values.size(), hipMemcpyDeviceToHost),
              "copy quantized activation") ||
      !hip_ok(hipMemcpy(&scale,
                        resources.activation_workspace +
                            resources.activation_scale_offset,
                        sizeof(scale), hipMemcpyDeviceToHost),
              "copy activation scale")) {
    return false;
  }
  const std::size_t mismatches = std::count_if(
      values.begin(), values.end(),
      [index = std::size_t{0U}, &expected_values](const uint8_t value) mutable {
        return value != expected_values[index++];
      });
  uint32_t scale_bits = 0U;
  std::memcpy(&scale_bits, &scale, sizeof(scale_bits));
  std::cout << "quantized_oracle K=" << shape.k << " N=" << shape.n
            << " value_mismatches=" << mismatches
            << " scale=" << std::setprecision(9) << scale
            << " expected_scale=0.125 scale_bits=0x" << std::hex << scale_bits
            << std::dec << "\n";
  return mismatches == 0U && scale == 0.125F;
}

bool run_heuristic(const ShapeResources &resources, const ShapeCase &shape,
                   const Heuristic &heuristic,
                   const std::vector<uint16_t> &cpu_expected,
                   const std::vector<uint16_t> &rank0_output,
                   Timing *const timing, std::vector<uint16_t> *const output,
                   std::size_t *const cpu_mismatches,
                   std::size_t *const rank0_mismatches,
                   std::size_t *const max_ulp) {
  if (timing == nullptr || output == nullptr || cpu_mismatches == nullptr ||
      rank0_mismatches == nullptr || max_ulp == nullptr ||
      heuristic.state != HIPBLAS_STATUS_SUCCESS ||
      heuristic.workspace_bytes != 0U) {
    return false;
  }
  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    if (!lt_ok(launch_matmul(resources, heuristic, shape),
               "warmup FP8 quantize+Lt")) {
      return false;
    }
  }
  if (!hip_ok(hipStreamSynchronize(resources.stream), "warmup synchronize")) {
    return false;
  }
  std::vector<double> samples;
  samples.reserve(kMeasured);
  for (int iteration = 0; iteration < kMeasured; ++iteration) {
    if (!hip_ok(hipEventRecord(resources.start, resources.stream),
                "record timing start") ||
        !lt_ok(launch_matmul(resources, heuristic, shape),
               "measured FP8 quantize+Lt") ||
        !hip_ok(hipEventRecord(resources.stop, resources.stream),
                "record timing stop") ||
        !hip_ok(hipEventSynchronize(resources.stop),
                "synchronize timing stop")) {
      return false;
    }
    float milliseconds = 0.0F;
    if (!hip_ok(
            hipEventElapsedTime(&milliseconds, resources.start, resources.stop),
            "hipEventElapsedTime")) {
      return false;
    }
    samples.push_back(static_cast<double>(milliseconds));
  }
  *timing = summarize(std::move(samples));
  output->resize(static_cast<std::size_t>(shape.n));
  if (!hip_ok(hipMemcpy(output->data(), resources.output,
                        output->size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost),
              "copy output for oracle") ||
      !hip_ok(hipStreamSynchronize(resources.stream),
              "synchronize output copy")) {
    return false;
  }
  *cpu_mismatches = 0U;
  *rank0_mismatches = 0U;
  *max_ulp = 0U;
  for (std::size_t index = 0U; index < output->size(); ++index) {
    if ((*output)[index] != cpu_expected[index]) {
      ++(*cpu_mismatches);
    }
    if (!rank0_output.empty() && (*output)[index] != rank0_output[index]) {
      ++(*rank0_mismatches);
    }
    *max_ulp = std::max(
        *max_ulp, bf16_ulp_distance((*output)[index], cpu_expected[index]));
  }
  return true;
}

bool run_shape(const ShapeCase &shape, double *const weighted_rank7,
               double *const weighted_best) {
  std::vector<uint16_t> host_activation;
  std::vector<uint8_t> host_weight;
  std::vector<float> host_weight_scales;
  std::vector<uint8_t> expected_activation;
  if (!fill_host_inputs(shape, &host_activation, &host_weight,
                        &host_weight_scales, &expected_activation)) {
    return false;
  }
  const std::vector<uint8_t> activation_codes = expected_activation;
  const std::vector<uint16_t> cpu_expected = cpu_oracle(
      shape, activation_codes, 0.125F, host_weight, host_weight_scales);

  ShapeResources resources;
  if (!allocate_shape(&resources, shape) ||
      !copy_inputs(&resources, shape, host_activation, host_weight,
                   host_weight_scales) ||
      !create_descriptors(&resources, shape)) {
    release_shape(&resources);
    return false;
  }
  if (!launch_quantize(resources, shape) ||
      !hip_ok(hipStreamSynchronize(resources.stream), "initial synchronize") ||
      !quantized_activation_check(resources, shape, expected_activation)) {
    release_shape(&resources);
    return false;
  }

  hipFuncAttributes attributes{};
  const hipError_t attribute_status = hipFuncGetAttributes(
      &attributes, reinterpret_cast<const void *>(fp8_outer_quantize_kernel));
  if (attribute_status == hipSuccess) {
    std::cout << "resources quantizer vgpr=" << attributes.numRegs
              << " lds_static=" << attributes.sharedSizeBytes
              << " lds_dynamic=" << attributes.maxDynamicSharedSizeBytes
              << " spill_local=" << attributes.localSizeBytes
              << " max_threads=" << attributes.maxThreadsPerBlock << "\n";
  } else {
    std::cout << "resources quantizer unavailable status="
              << hipGetErrorName(attribute_status)
              << " (hipBLASLt algorithm resource metadata is opaque)\n";
  }

  std::vector<uint16_t> rank0_output;
  double rank7_median = std::numeric_limits<double>::quiet_NaN();
  double best_median = std::numeric_limits<double>::infinity();
  int best_rank = -1;
  std::size_t usable = 0U;
  const Heuristic *determinism_heuristic = nullptr;
  std::vector<uint16_t> determinism_reference;
  for (const Heuristic &heuristic : resources.heuristics) {
    std::cout << "heuristic K=" << shape.k << " N=" << shape.n
              << " rank=" << heuristic.rank
              << " algorithm_index=" << heuristic.algorithm_index
              << " state=" << static_cast<int>(heuristic.state)
              << " workspace=" << heuristic.workspace_bytes
              << " waves=" << heuristic.waves
              << " solution=" << std::quoted(heuristic.solution_name)
              << " kernel=" << std::quoted(heuristic.kernel_name) << "\n";
    if (heuristic.state != HIPBLAS_STATUS_SUCCESS ||
        heuristic.workspace_bytes != 0U) {
      continue;
    }
    ++usable;
    Timing timing;
    std::vector<uint16_t> output;
    std::size_t cpu_mismatches = 0U;
    std::size_t rank0_mismatches = 0U;
    std::size_t max_ulp = 0U;
    if (!run_heuristic(resources, shape, heuristic, cpu_expected, rank0_output,
                       &timing, &output, &cpu_mismatches, &rank0_mismatches,
                       &max_ulp)) {
      release_shape(&resources);
      return false;
    }
    if (heuristic.rank == 0) {
      rank0_output = output;
      // Re-check rank zero against itself is intentionally not counted as a
      // cross-rank comparison; CPU oracle comparison remains independent.
      rank0_mismatches = 0U;
    }
    if (heuristic.rank == 7) {
      determinism_heuristic = &heuristic;
      determinism_reference = output;
    } else if (determinism_heuristic == nullptr) {
      // Prefer rank 7 for the repeat check because it is the production
      // control; before seeing rank 7, retain the first usable rank as a
      // fallback for a short heuristic list.
      determinism_heuristic = &heuristic;
      determinism_reference = output;
    }
    if (heuristic.rank == 7) {
      rank7_median = timing.median;
    }
    if (timing.median < best_median) {
      best_median = timing.median;
      best_rank = heuristic.rank;
    }
    std::cout << std::fixed << std::setprecision(6) << "timing K=" << shape.k
              << " N=" << shape.n << " rank=" << heuristic.rank
              << " median_ms=" << timing.median << " mad_ms=" << timing.mad
              << " min_ms=" << timing.minimum << " max_ms=" << timing.maximum
              << " cpu_bf16_mismatches=" << cpu_mismatches
              << " rank0_bf16_mismatches=" << rank0_mismatches
              << " max_cpu_ulp=" << max_ulp << " output_hash=0x" << std::hex
              << hash_u16(output) << std::dec << "\n";
  }
  if (determinism_heuristic != nullptr) {
    if (!lt_ok(launch_matmul(resources, *determinism_heuristic, shape),
               "determinism FP8 quantize+Lt") ||
        !hip_ok(hipStreamSynchronize(resources.stream),
                "determinism synchronize")) {
      release_shape(&resources);
      return false;
    }
    std::vector<uint16_t> repeat_output(static_cast<std::size_t>(shape.n));
    if (!hip_ok(hipMemcpy(repeat_output.data(), resources.output,
                          repeat_output.size() * sizeof(uint16_t),
                          hipMemcpyDeviceToHost),
                "copy determinism output") ||
        !hip_ok(hipStreamSynchronize(resources.stream),
                "synchronize determinism output")) {
      release_shape(&resources);
      return false;
    }
    const std::size_t repeat_mismatches =
        std::count_if(repeat_output.begin(), repeat_output.end(),
                      [index = std::size_t{0U},
                       &determinism_reference](const uint16_t value) mutable {
                        return value != determinism_reference[index++];
                      });
    std::cout << "determinism K=" << shape.k << " N=" << shape.n
              << " rank=" << determinism_heuristic->rank
              << " bit_mismatches=" << repeat_mismatches << " reference_hash=0x"
              << std::hex << hash_u16(determinism_reference)
              << " repeat_hash=0x" << hash_u16(repeat_output) << std::dec
              << "\n";
    if (repeat_mismatches != 0U) {
      release_shape(&resources);
      return false;
    }
  }
  const bool rank7_available = std::isfinite(rank7_median);
  if (rank7_available && weighted_rank7 != nullptr) {
    *weighted_rank7 += rank7_median * static_cast<double>(shape.occurrences);
  }
  if (std::isfinite(best_median) && weighted_best != nullptr) {
    *weighted_best += best_median * static_cast<double>(shape.occurrences);
  }
  std::cout << "shape_summary K=" << shape.k << " N=" << shape.n
            << " occurrences=" << shape.occurrences << " roles=" << shape.roles
            << " usable_ranks=" << usable << " rank7_ms=";
  if (rank7_available) {
    std::cout << rank7_median;
  } else {
    std::cout << "unavailable";
  }
  std::cout << " best_rank=" << best_rank << " best_ms=";
  if (std::isfinite(best_median)) {
    std::cout << best_median;
  } else {
    std::cout << "unavailable";
  }
  std::cout << "\n";
  release_shape(&resources);
  std::cout << "cleanup K=" << shape.k << " N=" << shape.n
            << " status=complete\n";
  return usable != 0U;
}

} // namespace

int main() {
  if (!check_identity()) {
    return 2;
  }
  std::vector<ShapeCase> selected;
  if (!parse_shape_filter(&selected)) {
    return 2;
  }
  std::cout << "manifest exact_qwen38_fp8_tensors=" << shape_occurrences()
            << " selected_shapes=" << selected.size() << " warmups=" << kWarmups
            << " measured=" << kMeasured
            << " heuristic_request=" << kRequestedAlgorithms
            << " production_rank7_control=enabled\n";
  double weighted_rank7 = 0.0;
  double weighted_best = 0.0;
  uint32_t selected_occurrences = 0U;
  for (const ShapeCase &shape : selected) {
    selected_occurrences += shape.occurrences;
    if (!run_shape(shape, &weighted_rank7, &weighted_best)) {
      std::cerr << "rank sweep failed for K=" << shape.k << " N=" << shape.n
                << "\n";
      return 3;
    }
  }
  std::cout << std::fixed << std::setprecision(6)
            << "weighted_total selected_occurrences=" << selected_occurrences
            << " rank7_ms=" << weighted_rank7
            << " best_cache_ms=" << weighted_best;
  if (weighted_rank7 > 0.0 && std::isfinite(weighted_best) &&
      weighted_best > 0.0) {
    std::cout << " expected_speedup=" << weighted_rank7 / weighted_best;
  } else {
    std::cout << " expected_speedup=unavailable(rank7_missing)";
  }
  std::cout << "\n";
  return 0;
}
