// Phase 87 Stage 1 N0 probe. The candidate kernel is exported by the fresh
// gfx1030 lowp archive from nvfp4_decode_scale_lut.inc; this probe contains
// only host setup, the production ID84 launcher, and the external candidate.

#include "../src/matmul_kernel_internal.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <lowp/detail/lowp_kernel_internal.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C" __global__ __launch_bounds__(
    256,
    1) void sllm_nvfp4_w4a4_decode_scale_lut_gfx1030_sgpr_v1(const uint8_t *,
                                                             const uint8_t *,
                                                             const uint8_t *,
                                                             const uint8_t *,
                                                             const float *,
                                                             const float *,
                                                             uint16_t *,
                                                             uint64_t, uint64_t,
                                                             uint64_t);
extern "C" __global__
    __launch_bounds__(256, 1) void sllm_matmul_nvfp4_w4a4_decode_scale_lut_v1(
        const uint8_t *, const uint8_t *, const uint8_t *, const uint8_t *,
        const float *, const float *, uint16_t *, uint64_t, uint64_t, uint64_t);

namespace {

constexpr uint64_t kContextTokens = 8256U;
constexpr uint32_t kKvHeads = 4U;
constexpr uint64_t kHeadDim = 256U;
constexpr uint64_t kScaleBytesPerHead = kHeadDim / 32U;
constexpr uint64_t kKvValuesBytes = kContextTokens * kKvHeads * kHeadDim;
constexpr uint64_t kKvScalesBytes =
    kContextTokens * kKvHeads * kScaleBytesPerHead;
constexpr uint64_t kKvBytes = 2U * (kKvValuesBytes + kKvScalesBytes);
constexpr uint32_t kWarmupMs = 300U;
constexpr uint32_t kRounds = 3U;
constexpr uint32_t kSamples = 9U;
constexpr std::array<std::pair<uint64_t, uint64_t>, 2> kShapes = {{
    {5120U, 17408U},
    {17408U, 5120U},
}};

void need(const bool ok, const char *const message) {
  if (!ok)
    throw std::runtime_error(message);
}

void hipcheck(const hipError_t status, const char *const operation) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
}

std::size_t live_allocations = 0U;
bool frees_ok = true;

template <typename T> struct DeviceBuffer final {
  T *pointer = nullptr;
  std::size_t count = 0U;
  explicit DeviceBuffer(const std::size_t elements) : count(elements) {
    hipcheck(hipMalloc(reinterpret_cast<void **>(&pointer),
                       std::max<std::size_t>(1U, elements * sizeof(T))),
             "hipMalloc");
    ++live_allocations;
  }
  ~DeviceBuffer() {
    if (pointer != nullptr) {
      if (hipFree(pointer) == hipSuccess)
        --live_allocations;
      else
        frees_ok = false;
    }
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  void put(const std::vector<T> &host) const {
    need(host.size() == count, "host/device size mismatch");
    hipcheck(hipMemcpy(pointer, host.data(), count * sizeof(T),
                       hipMemcpyHostToDevice),
             "hipMemcpy H2D");
  }
  std::vector<T> get() const {
    std::vector<T> host(count);
    hipcheck(hipMemcpy(host.data(), pointer, count * sizeof(T),
                       hipMemcpyDeviceToHost),
             "hipMemcpy D2H");
    return host;
  }
};

float e4m3fn_decode(const uint8_t code) {
  const uint32_t exponent = (code >> 3U) & 15U;
  const uint32_t mantissa = code & 7U;
  if (exponent == 15U && mantissa == 7U)
    return std::numeric_limits<float>::quiet_NaN();
  const float magnitude =
      exponent == 0U ? std::ldexp(static_cast<float>(mantissa), -9)
                     : std::ldexp(1.0F + static_cast<float>(mantissa) * 0.125F,
                                  static_cast<int>(exponent) - 7);
  return (code & 0x80U) == 0U ? magnitude : -magnitude;
}

float e2m1_decode(const uint8_t code) {
  static constexpr float values[16] = {0.0F,  0.5F,  1.0F,  1.5F,  2.0F,  3.0F,
                                       4.0F,  6.0F,  -0.0F, -0.5F, -1.0F, -1.5F,
                                       -2.0F, -3.0F, -4.0F, -6.0F};
  return values[code & 15U];
}

uint16_t f32_to_bf16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & 1U) != 0U))
    ++upper;
  return static_cast<uint16_t>(upper);
}

float bf16_to_f32(const uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint16_t nvfp4_bf16_rne(const float value) { return f32_to_bf16(value); }

__host__ __device__ uint8_t packed_code(const uint64_t column,
                                        const uint64_t index,
                                        const uint32_t salt) {
  uint32_t state = static_cast<uint32_t>(column) * 747796405U +
                   static_cast<uint32_t>(index) * 2891336453U + salt;
  state ^= state >> 16U;
  const uint8_t low = static_cast<uint8_t>((state >> 1U) & 0x0fU);
  const uint8_t high = static_cast<uint8_t>((state >> 7U) & 0x0fU);
  return static_cast<uint8_t>(low | (high << 4U));
}

struct Timing final {
  float kv_ms = 0.0F;
  float quant_ms = 0.0F;
  float dot_ms = 0.0F;
  float total_ms = 0.0F;
};

template <uint32_t Splits>
__global__ __launch_bounds__(192, 1) void read_kv_gqa(
    const uint8_t *const key, const uint8_t *const value,
    const uint8_t *const key_scales, const uint8_t *const value_scales,
    const uint64_t token_count, volatile uint32_t *const block_sums) {
  constexpr uint32_t kGqaWaves = 6U;
  constexpr uint32_t kTile = 8U;
  const uint32_t block = blockIdx.x;
  const uint32_t split = block % Splits;
  const uint32_t kv_head = (block / Splits) % kKvHeads;
  const uint64_t begin = token_count * split / Splits;
  const uint64_t end = token_count * (split + 1U) / Splits;
  uint32_t sum = 0U;
  for (uint64_t tile_begin = begin; tile_begin < end; tile_begin += kTile) {
    const uint32_t tile_count =
        static_cast<uint32_t>(std::min<uint64_t>(kTile, end - tile_begin));
    for (uint32_t element = threadIdx.x; element < kTile * 2U * kHeadDim;
         element += blockDim.x) {
      const uint32_t key_index = element / (2U * kHeadDim);
      if (key_index >= tile_count)
        continue;
      const uint32_t plane_dimension = element % (2U * kHeadDim);
      const uint32_t dimension = plane_dimension < kHeadDim
                                     ? plane_dimension
                                     : plane_dimension - kHeadDim;
      const uint64_t row = (tile_begin + key_index) * kKvHeads + kv_head;
      const uint64_t value_base = row * kHeadDim;
      const uint64_t scale_base = row * kScaleBytesPerHead;
      if (plane_dimension < kHeadDim) {
        sum += key[value_base + dimension];
        sum += key_scales[scale_base + dimension / 32U];
      } else {
        sum += value[value_base + dimension];
        sum += value_scales[scale_base + dimension / 32U];
      }
    }
  }
  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t wave = threadIdx.x / 32U;
  for (uint32_t offset = 16U; offset != 0U; offset >>= 1U)
    sum += __shfl_down(sum, offset, 32U);
  __shared__ uint32_t wave_sums[kGqaWaves];
  if (lane == 0U)
    wave_sums[wave] = sum;
  __syncthreads();
  if (wave == 0U) {
    sum = lane < kGqaWaves ? wave_sums[lane] : 0U;
    for (uint32_t offset = 16U; offset != 0U; offset >>= 1U)
      sum += __shfl_down(sum, offset, 32U);
    if (lane == 0U)
      block_sums[block] = sum;
  }
}

struct Matrix final {
  uint64_t k;
  uint64_t n;
  uint64_t blocks;
  uint64_t packed_row_bytes;
  uint64_t copies;
  DeviceBuffer<uint16_t> activation;
  DeviceBuffer<uint8_t> activation_packed;
  DeviceBuffer<uint8_t> activation_scales;
  DeviceBuffer<uint8_t> weights;
  DeviceBuffer<uint8_t> weight_scales;
  DeviceBuffer<float> weight_tensor_scale{1U};
  DeviceBuffer<float> input_tensor_scale{1U};
  DeviceBuffer<uint16_t> control_output;
  DeviceBuffer<uint16_t> candidate_output;
  DeviceBuffer<uint8_t> kv{kKvBytes};
  DeviceBuffer<uint32_t> predecessor_output{kKvHeads * 32U};

  Matrix(const uint64_t K, const uint64_t N)
      : k(K), n(N), blocks(K / 16U), packed_row_bytes(K / 2U),
        copies(std::max<uint64_t>(
            1U, ((UINT64_C(512) << 20) + N * K / 2U - 1U) / (N * K / 2U))),
        activation(K), activation_packed(K / 2U), activation_scales(K / 16U),
        weights(copies * N * K / 2U), weight_scales(copies * N * K / 16U),
        control_output(N + 32U), candidate_output(N + 32U) {}
};

__global__ void fill_nvfp4(uint8_t *const weights, uint8_t *const scales,
                           const uint64_t weight_bytes,
                           const uint64_t scale_bytes, const uint64_t k,
                           const uint64_t n) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < weight_bytes) {
    const uint64_t plane = index % (n * k / 2U);
    weights[index] = packed_code(plane / (k / 2U), plane % (k / 2U),
                                 static_cast<uint32_t>(index / (n * k / 2U)));
  }
  if (index < scale_bytes)
    scales[index] = static_cast<uint8_t>(0x38U + ((index / 17U) & 7U) * 8U);
}

void fill_kv_host(std::vector<uint8_t> *const host) {
  host->resize(kKvBytes);
  for (uint64_t index = 0U; index < kKvBytes; ++index)
    (*host)[index] = static_cast<uint8_t>((index * 13U + index / 97U) & 0xffU);
}

void setup(Matrix &matrix) {
  std::vector<uint16_t> activation(matrix.k);
  for (uint64_t index = 0U; index < matrix.k; ++index) {
    const float value = (index % 13U == 0U)  ? -1.5F
                        : (index % 7U == 0U) ? 0.5F
                                             : 1.0F;
    activation[index] = f32_to_bf16(value);
  }
  matrix.activation.put(activation);
  const float one = 1.0F;
  matrix.weight_tensor_scale.put(std::vector<float>{0.75F});
  matrix.input_tensor_scale.put(std::vector<float>{1.125F});
  hipcheck(hipMemset(matrix.weights.pointer, 0, matrix.weights.count),
           "clear weights");
  hipLaunchKernelGGL(
      fill_nvfp4, dim3((matrix.weights.count + 255U) / 256U), dim3(256U), 0U,
      nullptr, matrix.weights.pointer, matrix.weight_scales.pointer,
      static_cast<uint64_t>(matrix.weights.count),
      static_cast<uint64_t>(matrix.weight_scales.count), matrix.k, matrix.n);
  hipcheck(hipGetLastError(), "fill NVFP4 weights");
  std::vector<uint8_t> host_kv;
  fill_kv_host(&host_kv);
  matrix.kv.put(host_kv);
  hipcheck(hipDeviceSynchronize(), "setup synchronize");
  need(sllm_matmul_kernel::launch_nvfp4_quantize(
           matrix.activation.pointer, matrix.activation_packed.pointer,
           matrix.activation_scales.pointer, matrix.input_tensor_scale.pointer,
           1U, matrix.k, nullptr) == hipSuccess,
       "production NVFP4 quantizer");
  hipcheck(hipDeviceSynchronize(), "quantizer synchronize");
  (void)one;
}

void launch_control(Matrix &matrix, const uint64_t k, const uint64_t n,
                    const uint32_t copy) {
  // The integration candidate makes ID84's public launcher select C1 on
  // gfx1030. Call the original kernel symbol directly so the numeric and
  // same-process timing control remains the decomposed production ID84 body.
  if (std::strcmp(SLLM_HIP_COMPILE_TARGET, "gfx1030") == 0) {
    hipLaunchKernelGGL(
        sllm_matmul_nvfp4_w4a4_decode_scale_lut_v1,
        dim3(static_cast<uint32_t>((n + 31U) / 32U)), dim3(256U),
        static_cast<size_t>((k / 16U) * 20U), nullptr,
        matrix.activation_packed.pointer, matrix.activation_scales.pointer,
        matrix.weights.pointer + copy * n * k / 2U,
        matrix.weight_scales.pointer + copy * n * k / 16U,
        matrix.weight_tensor_scale.pointer, matrix.input_tensor_scale.pointer,
        matrix.control_output.pointer, 1U, k, n);
    hipcheck(hipGetLastError(), "original ID84 control launch");
    return;
  }
  hipcheck(
      sllm_matmul_kernel::launch_nvfp4_w4a4(
          matrix.activation_packed.pointer, matrix.activation_scales.pointer,
          matrix.weights.pointer + copy * n * k / 2U,
          matrix.weight_scales.pointer + copy * n * k / 16U,
          matrix.weight_tensor_scale.pointer, matrix.input_tensor_scale.pointer,
          matrix.control_output.pointer, 1U, k, n,
          sllm_matmul_kernel::KernelVariant::Nvfp4W4A4DecodeScaleLut, nullptr),
      "production ID84 control");
}

void launch_candidate(Matrix &matrix, const uint64_t k, const uint64_t n,
                      const uint32_t copy) {
  const size_t dynamic_shared = static_cast<size_t>((k / 16U) * 20U);
  hipLaunchKernelGGL(sllm_nvfp4_w4a4_decode_scale_lut_gfx1030_sgpr_v1,
                     dim3(static_cast<uint32_t>((n + 31U) / 32U)), dim3(256U),
                     dynamic_shared, nullptr, matrix.activation_packed.pointer,
                     matrix.activation_scales.pointer,
                     matrix.weights.pointer + copy * n * k / 2U,
                     matrix.weight_scales.pointer + copy * n * k / 16U,
                     matrix.weight_tensor_scale.pointer,
                     matrix.input_tensor_scale.pointer,
                     matrix.candidate_output.pointer, 1U, k, n);
  hipcheck(hipGetLastError(), "Stage1 candidate launch");
}

void launch_gqa(Matrix &matrix) {
  const uint8_t *key = matrix.kv.pointer;
  const uint8_t *value = key + kKvValuesBytes;
  const uint8_t *key_scales = value + kKvValuesBytes;
  const uint8_t *value_scales = key_scales + kKvScalesBytes;
  hipLaunchKernelGGL((read_kv_gqa<32U>), dim3(kKvHeads * 32U), dim3(192U), 0U,
                     nullptr, key, value, key_scales, value_scales,
                     kContextTokens, matrix.predecessor_output.pointer);
  hipcheck(hipGetLastError(), "GQA KV predecessor");
}

Timing timed(Matrix &matrix, const uint64_t k, const uint64_t n,
             const uint32_t copy, const bool candidate, const bool gqa) {
  hipEvent_t start = nullptr, kv_end = nullptr, quant_end = nullptr,
             end = nullptr;
  hipcheck(hipEventCreate(&start), "event start");
  hipcheck(hipEventCreate(&kv_end), "event kv");
  hipcheck(hipEventCreate(&quant_end), "event quant");
  hipcheck(hipEventCreate(&end), "event end");
  hipcheck(hipEventRecord(start), "record start");
  if (gqa)
    launch_gqa(matrix);
  hipcheck(hipEventRecord(kv_end), "record kv");
  hipcheck(sllm_matmul_kernel::launch_nvfp4_quantize(
               matrix.activation.pointer, matrix.activation_packed.pointer,
               matrix.activation_scales.pointer,
               matrix.input_tensor_scale.pointer, 1U, k, nullptr),
           "quantize timed");
  hipcheck(hipEventRecord(quant_end), "record quant");
  if (candidate)
    launch_candidate(matrix, k, n, copy);
  else
    launch_control(matrix, k, n, copy);
  hipcheck(hipEventRecord(end), "record end");
  hipcheck(hipEventSynchronize(end), "sync timed");
  Timing result{};
  hipcheck(hipEventElapsedTime(&result.kv_ms, start, kv_end), "elapsed kv");
  hipcheck(hipEventElapsedTime(&result.quant_ms, kv_end, quant_end),
           "elapsed quant");
  hipcheck(hipEventElapsedTime(&result.total_ms, start, end), "elapsed total");
  result.dot_ms = result.total_ms - result.kv_ms - result.quant_ms;
  hipcheck(hipEventDestroy(start), "destroy start");
  hipcheck(hipEventDestroy(kv_end), "destroy kv");
  hipcheck(hipEventDestroy(quant_end), "destroy quant");
  hipcheck(hipEventDestroy(end), "destroy end");
  return result;
}

void warm(Matrix &matrix, const uint64_t k, const uint64_t n,
          const bool candidate, const bool gqa) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(kWarmupMs);
  uint32_t copy = 0U;
  do {
    for (uint32_t index = 0U; index < 32U; ++index) {
      if (gqa)
        launch_gqa(matrix);
      (void)sllm_matmul_kernel::launch_nvfp4_quantize(
          matrix.activation.pointer, matrix.activation_packed.pointer,
          matrix.activation_scales.pointer, matrix.input_tensor_scale.pointer,
          1U, k, nullptr);
      if (candidate)
        launch_candidate(matrix, k, n, copy++ % matrix.copies);
      else
        launch_control(matrix, k, n, copy++ % matrix.copies);
    }
    hipcheck(hipDeviceSynchronize(), "warm synchronize");
  } while (std::chrono::steady_clock::now() < deadline);
}

float median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2U];
}

void print_samples(const char *const name, const std::vector<float> &values) {
  std::cout << ",\"" << name << "\":[";
  for (std::size_t index = 0U; index < values.size(); ++index) {
    if (index != 0U)
      std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']';
}

uint32_t ulp(const uint16_t left, const uint16_t right) {
  return left >= right ? left - right : right - left;
}

std::vector<uint16_t> oracle(const uint64_t k, const uint64_t n,
                             const std::vector<uint8_t> &activation,
                             const std::vector<uint8_t> &activation_scales,
                             const std::vector<uint8_t> &weights,
                             const std::vector<uint8_t> &weight_scales) {
  const uint64_t blocks = k / 16U;
  const uint64_t row_bytes = k / 2U;
  std::vector<uint16_t> expected(n);
  for (uint64_t column = 0U; column < n; ++column) {
    float accumulator = 0.0F;
    for (uint64_t block = 0U; block < blocks; ++block) {
      int32_t block_sum = 0;
      for (uint32_t index = 0U; index < 16U; ++index) {
        const uint8_t a = activation[block * 8U + index / 2U];
        const uint8_t w = weights[column * row_bytes + block * 8U + index / 2U];
        const uint8_t ac = (index & 1U) == 0U ? a & 0x0fU : a >> 4U;
        const uint8_t wc = (index & 1U) == 0U ? w & 0x0fU : w >> 4U;
        block_sum += static_cast<int32_t>(e2m1_decode(ac) * 2.0F) *
                     static_cast<int32_t>(e2m1_decode(wc) * 2.0F);
      }
      const float as = e4m3fn_decode(activation_scales[block]);
      const float ws = e4m3fn_decode(weight_scales[column * blocks + block]);
      accumulator = std::fmaf(static_cast<float>(block_sum) * 0.25F * as, ws,
                              accumulator);
    }
    expected[column] = nvfp4_bf16_rne(accumulator * 0.75F * 1.125F);
  }
  return expected;
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::string target;
    int shape_index = 0;
    bool bench = true;
    for (int index = 1; index < argc; ++index) {
      need(index + 1 < argc, "argument value");
      const std::string key = argv[index];
      const std::string value = argv[++index];
      if (key == "--target")
        target = value;
      else if (key == "--shape")
        shape_index = std::stoi(value);
      else if (key == "--bench")
        bench = std::stoi(value) != 0;
      else
        throw std::runtime_error("unknown argument");
    }
    need(target == "gfx1030" || target == "gfx1201", "target");
    need(shape_index >= 0 && shape_index < 2, "shape index");
    int device_index = 0;
    hipcheck(hipGetDevice(&device_index), "get device");
    hipDeviceProp_t properties{};
    hipcheck(hipGetDeviceProperties(&properties, device_index),
             "device properties");
    need(target == properties.gcnArchName, "visible target mismatch");
    const uint64_t k = kShapes[shape_index].first;
    const uint64_t n = kShapes[shape_index].second;
    auto owner = std::make_unique<Matrix>(k, n);
    Matrix &matrix = *owner;
    setup(matrix);
    std::vector<uint8_t> host_activation(matrix.activation_packed.count);
    std::vector<uint8_t> host_activation_scales(matrix.activation_scales.count);
    std::vector<uint8_t> host_weights(n * k / 2U);
    std::vector<uint8_t> host_weight_scales(n * k / 16U);
    host_activation = matrix.activation_packed.get();
    host_activation_scales = matrix.activation_scales.get();
    hipcheck(hipMemcpy(host_weights.data(), matrix.weights.pointer,
                       host_weights.size(), hipMemcpyDeviceToHost),
             "download weight oracle plane");
    hipcheck(hipMemcpy(host_weight_scales.data(), matrix.weight_scales.pointer,
                       host_weight_scales.size(), hipMemcpyDeviceToHost),
             "download scale oracle plane");
    const auto expected = oracle(k, n, host_activation, host_activation_scales,
                                 host_weights, host_weight_scales);
    constexpr uint16_t canary = UINT16_C(0x5a5a);
    hipcheck(hipMemset(matrix.control_output.pointer, 0x5a,
                       matrix.control_output.count * sizeof(uint16_t)),
             "control canary");
    launch_control(matrix, k, n, 0U);
    hipcheck(hipDeviceSynchronize(), "control synchronize");
    const auto control = matrix.control_output.get();
    hipcheck(hipMemset(matrix.candidate_output.pointer, 0x5a,
                       matrix.candidate_output.count * sizeof(uint16_t)),
             "candidate canary");
    launch_candidate(matrix, k, n, 0U);
    hipcheck(hipDeviceSynchronize(), "candidate synchronize");
    const auto candidate = matrix.candidate_output.get();
    launch_control(matrix, k, n, 0U);
    hipcheck(hipDeviceSynchronize(), "control repeat synchronize");
    const auto control_repeat_output = matrix.control_output.get();
    launch_candidate(matrix, k, n, 0U);
    hipcheck(hipDeviceSynchronize(), "candidate repeat synchronize");
    const auto candidate_repeat_output = matrix.candidate_output.get();
    bool bitwise = control == candidate;
    uint32_t max_ulp = 0U;
    bool finite = true;
    bool guard = true;
    for (uint64_t index = 0U; index < n; ++index) {
      max_ulp = std::max(max_ulp, ulp(candidate[index], expected[index]));
      finite &= std::isfinite(bf16_to_f32(candidate[index]));
    }
    for (uint64_t index = n; index < candidate.size(); ++index)
      guard &= candidate[index] == canary;
    const bool repeat = control == control_repeat_output &&
                        candidate == candidate_repeat_output;
    size_t first_diff = control.size();
    for (size_t index = 0U; index < control.size(); ++index) {
      if (control[index] != candidate[index]) {
        first_diff = index;
        break;
      }
    }
    if (first_diff != control.size()) {
      std::cerr << "first_diff=" << first_diff << " control=0x" << std::hex
                << control[first_diff] << " candidate=0x"
                << candidate[first_diff] << std::dec << '\n';
    }
    const bool numeric = bitwise && repeat && finite && guard && max_ulp <= 4U;
    std::cout << "{\"kind\":\"identity\",\"state\":\""
              << (numeric ? "PASS" : "FAIL") << "\",\"target\":\"" << target
              << "\",\"m\":1,\"k\":" << k << ",\"n\":" << n
              << ",\"candidate\":\"id84_sgpr_scalarized\","
              << "\"control_launcher\":\"production_id84\","
              << "\"control_variant\":84,\"gpu_execution\":true,"
              << "\"fallback_used\":false,\"weight_pool_bytes\":"
              << matrix.weights.count << ",\"copies\":" << matrix.copies
              << "}\n";
    std::cout << "{\"kind\":\"oracle\",\"state\":\""
              << (numeric ? "PASS" : "FAIL")
              << "\",\"finite\":" << (finite ? "true" : "false")
              << ",\"guard\":" << (guard ? "true" : "false")
              << ",\"repeat\":" << (repeat ? "true" : "false") << ","
              << "\"candidate_control_bitwise\":"
              << (bitwise ? "true" : "false") << ",\"max_ulp\":" << max_ulp
              << "}\n";
    if (numeric && bench) {
      for (const bool gqa : {false, true}) {
        std::vector<float> cq, cd, ct, aq, ad, at, ck, ak;
        for (uint32_t round = 0U; round < kRounds; ++round) {
          for (uint32_t position = 0U; position < 2U; ++position) {
            const bool candidate_side =
                ((round % 2U == 0U) == (position == 1U));
            warm(matrix, k, n, candidate_side, gqa);
            for (uint32_t sample = 0U; sample < kSamples; ++sample) {
              const uint32_t copy = static_cast<uint32_t>(
                  ((round * kSamples + sample) *
                   std::max<uint64_t>(1U, matrix.copies / kSamples)) %
                  matrix.copies);
              const Timing timing =
                  timed(matrix, k, n, copy, candidate_side, gqa);
              auto &kv = candidate_side ? ak : ck;
              auto &quant = candidate_side ? aq : cq;
              auto &dot = candidate_side ? ad : cd;
              auto &total = candidate_side ? at : ct;
              kv.push_back(timing.kv_ms);
              quant.push_back(timing.quant_ms);
              dot.push_back(timing.dot_ms);
              total.push_back(timing.total_ms);
            }
          }
        }
        std::cout << "{\"kind\":\"performance\",\"state\":\"PASS\","
                  << "\"mode\":\"" << (gqa ? "gqa32" : "isolated")
                  << "\",\"warmup_ms\":300,\"samples\":27,\"rounds\":3,"
                  << "\"order\":\"AB-BA-AB\",\"control_kv_ms\":" << median(ck)
                  << ",\"candidate_kv_ms\":" << median(ak)
                  << ",\"control_quant_ms\":" << median(cq)
                  << ",\"candidate_quant_ms\":" << median(aq)
                  << ",\"control_dot_ms\":" << median(cd)
                  << ",\"candidate_dot_ms\":" << median(ad)
                  << ",\"control_total_ms\":" << median(ct)
                  << ",\"candidate_total_ms\":" << median(at);
        print_samples("control_kv_samples_ms", ck);
        print_samples("candidate_kv_samples_ms", ak);
        print_samples("control_quant_samples_ms", cq);
        print_samples("candidate_quant_samples_ms", aq);
        print_samples("control_dot_samples_ms", cd);
        print_samples("candidate_dot_samples_ms", ad);
        print_samples("control_total_samples_ms", ct);
        print_samples("candidate_total_samples_ms", at);
        std::cout << '}';
        std::cout << '\n';
      }
    }
    owner.reset();
    need(live_allocations == 0U && frees_ok, "cleanup");
    std::cout << "{\"kind\":\"cleanup\",\"state\":\"PASS\","
              << "\"live_allocations\":0}\n";
    return numeric ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "phase87_stage1_nvfp4_sgpr: " << error.what() << '\n';
    return 1;
  }
}
