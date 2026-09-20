// Phase 87 WU-D1: isolate the Qwen3.8 NVFP4 M=1 decode from its predecessor.
//
// This probe includes the production NVFP4 scale-LUT wrapper, but does not
// modify a selector or a production source file.  The predecessor is a
// deliberately synthetic, read-only KV payload reader.  It models the
// address multiplicity of the two attention arrangements (six query heads
// reading one KV head versus one shared KV-head read) and the 32/128 split
// counts.  It is not a reimplementation of the full attention kernel.
//
// Suggested direct build (one exact target per binary):
//   amdclang++ -D__HIP_ROCclr__=1 -O3 -ffp-contract=off -DNDEBUG \
//     -std=gnu++17 --offload-arch=gfx1030 -mcode-object-version=6 \
//     -mno-wavefrontsize64 -x hip -I native/lowp/include -c this-file \
//     -o phase87-wud1-gfx1030.o
//   amdclang++ -O3 -ffp-contract=off -DNDEBUG --offload-arch=gfx1030 \
//     -mcode-object-version=6 -mno-wavefrontsize64 --hip-link \
//     phase87-wud1-gfx1030.o /path/to/libsllm_lowp.a \
//     -L/opt/rocm/lib -lrocprofiler-sdk-roctx -lamdhip64 \
//     -o phase87-wud1-gfx1030
//
// The default run uses both exact model shapes, four weight-pool copies, a
// 300 ms continuous warmup for each condition, and same-process AB/BA order.
// Use --shape wide --mode gqa32 --rounds 1 --samples 1 for a bounded profiler
// run; the NVFP4 symbol can then be selected with --kernel-include-regex.

#include "../../lowp/include/lowp/detail/lowp_kernel_internal.hpp"

#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kWave = 32U;
constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kQueryHeadsPerKvHead = 6U;
constexpr uint64_t kHeadDim = 256U;
constexpr uint64_t kScaleBytesPerHead = kHeadDim / 32U;
constexpr uint64_t kContextTokens = 8256U;
// The production MXFP8 view uses four separate token-major planes: K values,
// V values, K block-32 scales, and V block-32 scales.  Keeping those planes
// separate is material here because the predecessor hypothesis concerns the
// read order left in the caches, not just the total byte count.
constexpr uint64_t kKvValuesBytes = kContextTokens * kKvHeads * kHeadDim;
constexpr uint64_t kKvScalesBytes =
    kContextTokens * kKvHeads * kScaleBytesPerHead;
constexpr uint64_t kKvBytes = 2U * (kKvValuesBytes + kKvScalesBytes);
constexpr uint32_t kPoolCopies = 4U;
constexpr uint32_t kOracleColumns = 37U;
constexpr uint32_t kDefaultWarmupMs = 300U;
constexpr uint32_t kDefaultRounds = 3U;
constexpr uint32_t kDefaultSamples = 9U;

struct Shape final {
  uint64_t k;
  uint64_t n;
  const char *name;
};

constexpr std::array<Shape, 2> kShapes = {
    Shape{5120U, 17408U, "wide-k5120-n17408"},
    Shape{17408U, 5120U, "down-k17408-n5120"},
};

enum class Mode : uint32_t {
  Isolated,
  Staged32,
  Gqa32,
  Staged128,
  Gqa128,
};

struct Options final {
  std::string target;
  std::string shape = "both";
  std::string mode = "all";
  uint32_t warmup_ms = kDefaultWarmupMs;
  uint32_t rounds = kDefaultRounds;
  uint32_t samples = kDefaultSamples;
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

void check(const hipError_t status, const char *const operation) {
  if (status != hipSuccess)
    fail(std::string(operation) + ": " + hipGetErrorString(status));
}

struct DeviceBytes final {
  uint8_t *ptr = nullptr;
  size_t bytes = 0U;

  DeviceBytes() = default;
  explicit DeviceBytes(const size_t size) : bytes(size) {
    check(hipMalloc(reinterpret_cast<void **>(&ptr), bytes), "hipMalloc");
  }
  DeviceBytes(const DeviceBytes &) = delete;
  DeviceBytes &operator=(const DeviceBytes &) = delete;
  DeviceBytes(DeviceBytes &&other) noexcept
      : ptr(std::exchange(other.ptr, nullptr)),
        bytes(std::exchange(other.bytes, 0U)) {}
  DeviceBytes &operator=(DeviceBytes &&other) noexcept {
    if (this != &other) {
      if (ptr != nullptr)
        (void)hipFree(ptr);
      ptr = std::exchange(other.ptr, nullptr);
      bytes = std::exchange(other.bytes, 0U);
    }
    return *this;
  }
  ~DeviceBytes() {
    if (ptr != nullptr)
      (void)hipFree(ptr);
  }

  void upload(const void *const source) {
    check(hipMemcpy(ptr, source, bytes, hipMemcpyHostToDevice),
          "hipMemcpy H2D");
  }
};

struct DeviceMatrix final {
  uint64_t k;
  uint64_t n;
  DeviceBytes activation;
  DeviceBytes activation_scales;
  std::vector<DeviceBytes> weights;
  std::vector<DeviceBytes> weight_scales;
  DeviceBytes weight_tensor_scale;
  DeviceBytes input_tensor_scale;
  DeviceBytes output;
  DeviceBytes kv;
  DeviceBytes predecessor_output;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;

  DeviceMatrix(const Shape shape)
      : k(shape.k), n(shape.n), activation(static_cast<size_t>(k / 2U)),
        activation_scales(static_cast<size_t>(k / 16U)),
        weight_tensor_scale(sizeof(float)), input_tensor_scale(sizeof(float)),
        output(static_cast<size_t>(n * sizeof(uint16_t))),
        kv(static_cast<size_t>(kKvBytes)),
        predecessor_output(
            static_cast<size_t>(kKvHeads * 6U * 128U * sizeof(uint32_t))) {
    const size_t weight_bytes = static_cast<size_t>(n * (k / 2U));
    const size_t weight_scale_bytes = static_cast<size_t>(n * (k / 16U));
    weights.reserve(kPoolCopies);
    weight_scales.reserve(kPoolCopies);
    for (uint32_t copy = 0U; copy < kPoolCopies; ++copy) {
      weights.emplace_back(weight_bytes);
      weight_scales.emplace_back(weight_scale_bytes);
    }
    check(hipStreamCreate(&stream), "hipStreamCreate");
    check(hipEventCreate(&start), "hipEventCreate start");
    check(hipEventCreate(&stop), "hipEventCreate stop");
  }

  DeviceMatrix(const DeviceMatrix &) = delete;
  DeviceMatrix &operator=(const DeviceMatrix &) = delete;
  ~DeviceMatrix() {
    if (stop != nullptr)
      (void)hipEventDestroy(stop);
    if (start != nullptr)
      (void)hipEventDestroy(start);
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }
};

float e2m1(const uint8_t code) {
  constexpr std::array<float, 8> positive = {0.0F, 0.5F, 1.0F, 1.5F,
                                             2.0F, 3.0F, 4.0F, 6.0F};
  const float value = positive[code & 7U];
  return (code & 8U) == 0U ? value : -value;
}

float e4m3fn(const uint8_t bits) {
  const auto from_bits = [](const uint32_t value) {
    float result = 0.0F;
    std::memcpy(&result, &value, sizeof(result));
    return result;
  };
  const auto to_bits = [](const float value) {
    uint32_t result = 0U;
    std::memcpy(&result, &value, sizeof(result));
    return result;
  };
  const uint32_t sign = static_cast<uint32_t>(bits & 0x80U) << 24U;
  const uint32_t magnitude = static_cast<uint32_t>(bits & 0x7fU);
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & 7U;
  if (exponent == 0U) {
    if (mantissa == 0U)
      return from_bits(sign);
    const float value = static_cast<float>(mantissa) * 0x1p-9F;
    return from_bits(to_bits(value) | sign);
  }
  if (magnitude == 0x7fU)
    return from_bits(sign | UINT32_C(0x7fc00000));
  return from_bits(sign | ((exponent + 120U) << 23U) | (mantissa << 20U));
}

uint8_t nibble(const uint64_t index, const uint64_t salt) {
  return static_cast<uint8_t>((index * 13U + salt * 7U + 3U) & 0x0fU);
}

uint16_t bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & 1U) != 0U))
    ++upper;
  return static_cast<uint16_t>(upper);
}

float bf16_to_float(const uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint32_t bf16_ulp_distance(const uint16_t lhs, const uint16_t rhs) {
  const auto ordered = [](const uint16_t value) {
    const uint32_t magnitude = value & UINT16_C(0x7fff);
    return (value & UINT16_C(0x8000)) != 0U ? UINT32_C(0x8000) - magnitude
                                            : UINT32_C(0x8000) + value;
  };
  const uint32_t a = ordered(lhs);
  const uint32_t b = ordered(rhs);
  return a >= b ? a - b : b - a;
}

void fill_host_matrix(const Shape shape, std::vector<uint8_t> *const activation,
                      std::vector<uint8_t> *const activation_scales,
                      std::vector<uint8_t> *const weights,
                      std::vector<uint8_t> *const weight_scales) {
  const uint64_t blocks = shape.k / 16U;
  activation->assign(static_cast<size_t>(shape.k / 2U), 0U);
  activation_scales->resize(static_cast<size_t>(blocks));
  weights->assign(static_cast<size_t>(shape.n * (shape.k / 2U)), 0U);
  weight_scales->resize(static_cast<size_t>(shape.n * blocks));
  for (uint64_t inner = 0U; inner < shape.k; ++inner) {
    const uint8_t code = nibble(inner, 1U);
    auto &packed = (*activation)[static_cast<size_t>(inner / 2U)];
    if ((inner & 1U) == 0U)
      packed = code;
    else
      packed |= static_cast<uint8_t>(code << 4U);
  }
  for (uint64_t block = 0U; block < blocks; ++block)
    (*activation_scales)[static_cast<size_t>(block)] =
        static_cast<uint8_t>((block & 1U) == 0U ? 0x38U : 0x40U);

  const uint64_t row_bytes = shape.k / 2U;
  for (uint64_t column = 0U; column < shape.n; ++column) {
    for (uint64_t inner = 0U; inner < shape.k; ++inner) {
      const uint8_t code = nibble(inner, column + 11U);
      auto &packed =
          (*weights)[static_cast<size_t>(column * row_bytes + inner / 2U)];
      if ((inner & 1U) == 0U)
        packed = code;
      else
        packed |= static_cast<uint8_t>(code << 4U);
    }
    for (uint64_t block = 0U; block < blocks; ++block)
      (*weight_scales)[static_cast<size_t>(column * blocks + block)] =
          static_cast<uint8_t>((block + column) & 1U ? 0x40U : 0x38U);
  }
}

void fill_host_kv(std::vector<uint8_t> *const kv) {
  kv->resize(static_cast<size_t>(kKvBytes));
  for (size_t index = 0U; index < kv->size(); ++index)
    (*kv)[index] = static_cast<uint8_t>((index * 29U + 17U) & 0xffU);
}

void upload_matrix(const Shape shape, DeviceMatrix *const device) {
  std::vector<uint8_t> activation;
  std::vector<uint8_t> activation_scales;
  std::vector<uint8_t> weights;
  std::vector<uint8_t> weight_scales;
  fill_host_matrix(shape, &activation, &activation_scales, &weights,
                   &weight_scales);
  device->activation.upload(activation.data());
  device->activation_scales.upload(activation_scales.data());
  for (uint32_t copy = 0U; copy < kPoolCopies; ++copy) {
    device->weights[copy].upload(weights.data());
    device->weight_scales[copy].upload(weight_scales.data());
  }
  const float weight_tensor_scale = 0.75F;
  const float input_tensor_scale = 1.125F;
  device->weight_tensor_scale.upload(&weight_tensor_scale);
  device->input_tensor_scale.upload(&input_tensor_scale);
  std::vector<uint8_t> kv;
  fill_host_kv(&kv);
  device->kv.upload(kv.data());
  check(hipMemsetAsync(device->output.ptr, 0, device->output.bytes,
                       device->stream),
        "clear output");
  check(hipMemsetAsync(device->predecessor_output.ptr, 0,
                       device->predecessor_output.bytes, device->stream),
        "clear predecessor output");
  check(hipStreamSynchronize(device->stream), "upload synchronize");
}

void launch_matmul(const Shape shape, DeviceMatrix *const device,
                   const uint32_t weight_slot) {
  const hipError_t status = sllm_matmul_kernel::launch_nvfp4_w4a4(
      device->activation.ptr, device->activation_scales.ptr,
      device->weights[weight_slot].ptr, device->weight_scales[weight_slot].ptr,
      reinterpret_cast<const float *>(device->weight_tensor_scale.ptr),
      reinterpret_cast<const float *>(device->input_tensor_scale.ptr),
      reinterpret_cast<uint16_t *>(device->output.ptr), 1U, shape.k, shape.n,
      sllm_matmul_kernel::KernelVariant::Nvfp4W4A4DecodeScaleLut,
      device->stream);
  check(status, "launch production NVFP4 W4A4 scale-LUT");
}

// Old staged32: one 32-lane query-head block per split.  Each of the six
// query heads mapped to a KV head therefore traverses the same K/V planes.
template <uint32_t Splits>
__global__ __launch_bounds__(kWave, 1) void read_kv_staged(
    const uint8_t *const key, const uint8_t *const value,
    const uint8_t *const key_scales, const uint8_t *const value_scales,
    const uint64_t token_count, volatile uint32_t *const block_sums) {
  const uint32_t block = blockIdx.x;
  const uint32_t split = block % Splits;
  const uint32_t query_head = block / Splits;
  const uint32_t kv_head = query_head / kQueryHeadsPerKvHead;
  if (query_head >= kKvHeads * kQueryHeadsPerKvHead || kv_head >= kKvHeads)
    return;
  const uint64_t begin = token_count * split / Splits;
  const uint64_t end = token_count * (split + 1U) / Splits;
  const uint32_t lane = threadIdx.x & (kWave - 1U);
  uint32_t sum = 0U;
  for (uint64_t token = begin; token < end; ++token) {
    const uint64_t row = token * kKvHeads + kv_head;
    const uint64_t value_base = row * kHeadDim;
    const uint64_t scale_base = row * kScaleBytesPerHead;
#pragma unroll
    for (uint32_t index = 0U; index < kHeadDim / kWave; ++index) {
      const uint32_t dimension = lane + index * kWave;
      // The production qtile4 loader traverses dimensions lane-wise.  It
      // reads a block-32 scale for each dimension, so retain those repeated
      // scale references in this synthetic predecessor.
      sum += key[value_base + dimension] + value[value_base + dimension];
      sum += key_scales[scale_base + dimension / 32U];
      sum += value_scales[scale_base + dimension / 32U];
    }
  }
  for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U)
    sum += __shfl_down(sum, offset, kWave);
  if (lane == 0U)
    block_sums[block] = sum;
}

// GQA-shared stage1: one block per KV head and split, with six waves sharing
// the decoded eight-row tile.  The tile fill follows the production
// element=(key/value, dimension) traversal and reads separate K/V/scale planes.
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
  if (kv_head >= kKvHeads)
    return;
  const uint64_t begin = token_count * split / Splits;
  const uint64_t end = token_count * (split + 1U) / Splits;
  uint32_t sum = 0U;
  for (uint64_t tile_begin = begin; tile_begin < end; tile_begin += kTile) {
    const uint32_t tile_count =
        static_cast<uint32_t>(std::min<uint64_t>(kTile, end - tile_begin));
    for (uint32_t element = threadIdx.x; element < kTile * 2U * kHeadDim;
         element += blockDim.x) {
      const uint32_t key_index = element / (2U * kHeadDim);
      const uint32_t plane_dimension = element % (2U * kHeadDim);
      if (key_index >= tile_count)
        continue;
      const uint64_t row = (tile_begin + key_index) * kKvHeads + kv_head;
      const uint32_t dimension = plane_dimension < kHeadDim
                                     ? plane_dimension
                                     : plane_dimension - kHeadDim;
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
  const uint32_t lane = threadIdx.x & (kWave - 1U);
  const uint32_t wave = threadIdx.x / kWave;
  for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U)
    sum += __shfl_down(sum, offset, kWave);
  __shared__ uint32_t wave_sums[kGqaWaves];
  if (lane == 0U)
    wave_sums[wave] = sum;
  __syncthreads();
  if (wave == 0U) {
    sum = lane < kGqaWaves ? wave_sums[lane] : 0U;
    for (uint32_t offset = kWave / 2U; offset != 0U; offset >>= 1U)
      sum += __shfl_down(sum, offset, kWave);
    if (lane == 0U)
      block_sums[block] = sum;
  }
}

void launch_predecessor(const Mode mode, DeviceMatrix *const device) {
  uint32_t splits = 0U;
  const uint8_t *const key = device->kv.ptr;
  const uint8_t *const value = device->kv.ptr + kKvValuesBytes;
  const uint8_t *const key_scales = value + kKvValuesBytes;
  const uint8_t *const value_scales = key_scales + kKvScalesBytes;
  switch (mode) {
  case Mode::Staged32:
    splits = 32U;
    break;
  case Mode::Gqa32:
    splits = 32U;
    break;
  case Mode::Staged128:
    splits = 128U;
    break;
  case Mode::Gqa128:
    splits = 128U;
    break;
  case Mode::Isolated:
    return;
  }
  const uint32_t blocks = (mode == Mode::Staged32 || mode == Mode::Staged128)
                              ? kKvHeads * kQueryHeadsPerKvHead * splits
                              : kKvHeads * splits;
  if (mode == Mode::Staged32) {
    hipLaunchKernelGGL(
        (read_kv_staged<32U>), dim3(blocks), dim3(kWave), 0U, device->stream,
        key, value, key_scales, value_scales, kContextTokens,
        reinterpret_cast<volatile uint32_t *>(device->predecessor_output.ptr));
  } else if (mode == Mode::Gqa32) {
    hipLaunchKernelGGL(
        (read_kv_gqa<32U>), dim3(blocks), dim3(192U), 0U, device->stream, key,
        value, key_scales, value_scales, kContextTokens,
        reinterpret_cast<volatile uint32_t *>(device->predecessor_output.ptr));
  } else if (mode == Mode::Staged128) {
    hipLaunchKernelGGL(
        (read_kv_staged<128U>), dim3(blocks), dim3(kWave), 0U, device->stream,
        key, value, key_scales, value_scales, kContextTokens,
        reinterpret_cast<volatile uint32_t *>(device->predecessor_output.ptr));
  } else {
    hipLaunchKernelGGL(
        (read_kv_gqa<128U>), dim3(blocks), dim3(192U), 0U, device->stream, key,
        value, key_scales, value_scales, kContextTokens,
        reinterpret_cast<volatile uint32_t *>(device->predecessor_output.ptr));
  }
  check(hipGetLastError(), "launch synthetic KV predecessor");
}

std::string mode_name(const Mode mode) {
  switch (mode) {
  case Mode::Isolated:
    return "isolated";
  case Mode::Staged32:
    return "staged32-qhead6-split32";
  case Mode::Gqa32:
    return "gqa-kvhead1-split32";
  case Mode::Staged128:
    return "staged128-qhead6-split128";
  case Mode::Gqa128:
    return "gqa-kvhead1-split128";
  }
  return "unknown";
}

std::vector<Mode> selected_modes(const Options &options) {
  if (options.mode == "all")
    return {Mode::Isolated, Mode::Staged32, Mode::Gqa32, Mode::Staged128,
            Mode::Gqa128};
  const std::array<std::pair<std::string_view, Mode>, 5> names = {{
      {"isolated", Mode::Isolated},
      {"staged32", Mode::Staged32},
      {"gqa32", Mode::Gqa32},
      {"staged128", Mode::Staged128},
      {"gqa128", Mode::Gqa128},
  }};
  for (const auto &[name, mode] : names)
    if (options.mode == name)
      return {mode};
  fail("unknown --mode: " + options.mode);
}

std::vector<size_t> selected_shapes(const Options &options) {
  if (options.shape == "both")
    return {0U, 1U};
  if (options.shape == "wide")
    return {0U};
  if (options.shape == "down")
    return {1U};
  fail("unknown --shape: " + options.shape);
}

float measure_call(const Shape shape, const Mode mode,
                   DeviceMatrix *const device, const uint32_t weight_slot) {
  if (mode != Mode::Isolated)
    launch_predecessor(mode, device);
  check(hipEventRecord(device->start, device->stream), "event start");
  launch_matmul(shape, device, weight_slot);
  check(hipEventRecord(device->stop, device->stream), "event stop");
  check(hipEventSynchronize(device->stop), "event synchronize");
  float elapsed_ms = 0.0F;
  check(hipEventElapsedTime(&elapsed_ms, device->start, device->stop),
        "event elapsed");
  return elapsed_ms * 1000.0F;
}

void continuous_warmup(const Shape shape, const Mode mode,
                       DeviceMatrix *const device, const uint32_t warmup_ms) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(warmup_ms);
  uint32_t slot = 0U;
  do {
    // Enqueue a small batch before synchronizing.  This keeps the 300 ms
    // warmup representative of a continuous decoder queue while bounding
    // host-side polling and ensuring every predecessor remains immediately
    // adjacent to its NVFP4 call in the stream.
    for (uint32_t batch = 0U; batch < 32U; ++batch) {
      if (mode != Mode::Isolated)
        launch_predecessor(mode, device);
      launch_matmul(shape, device, slot % kPoolCopies);
      ++slot;
    }
    check(hipStreamSynchronize(device->stream), "warmup synchronize");
  } while (std::chrono::steady_clock::now() < deadline);
  std::printf("warmup shape=%s mode=%s ms=%u launches=%u weight_pool=%u\n",
              shape.name, mode_name(mode).c_str(), warmup_ms, slot,
              kPoolCopies);
}

std::vector<uint16_t> download_output(const Shape shape,
                                      DeviceMatrix *const device) {
  std::vector<uint16_t> output(static_cast<size_t>(shape.n));
  check(hipMemcpy(output.data(), device->output.ptr,
                  output.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
        "download output");
  return output;
}

std::vector<uint16_t>
host_oracle_prefix(const Shape shape, const std::vector<uint8_t> &activation,
                   const std::vector<uint8_t> &activation_scales,
                   const std::vector<uint8_t> &weights,
                   const std::vector<uint8_t> &weight_scales) {
  const uint64_t blocks = shape.k / 16U;
  const uint64_t row_bytes = shape.k / 2U;
  const uint32_t count =
      static_cast<uint32_t>(std::min<uint64_t>(shape.n, kOracleColumns));
  std::vector<uint16_t> expected(count);
  for (uint32_t column = 0U; column < count; ++column) {
    float accumulator = 0.0F;
    for (uint64_t inner = 0U; inner < shape.k; ++inner) {
      const uint8_t a_pair = activation[static_cast<size_t>(inner / 2U)];
      const uint8_t w_pair = weights[static_cast<size_t>(
          static_cast<uint64_t>(column) * row_bytes + inner / 2U)];
      const uint8_t a_code = (inner & 1U) == 0U ? a_pair & 0x0fU : a_pair >> 4U;
      const uint8_t w_code = (inner & 1U) == 0U ? w_pair & 0x0fU : w_pair >> 4U;
      accumulator +=
          e2m1(a_code) * e4m3fn(activation_scales[inner / 16U]) * e2m1(w_code) *
          e4m3fn(weight_scales[static_cast<uint64_t>(column) * blocks +
                               inner / 16U]);
    }
    expected[column] = bf16_rne(accumulator * 0.75F * 1.125F);
  }
  return expected;
}

bool check_outputs(const Shape shape, const std::vector<uint16_t> &control,
                   const std::vector<uint16_t> &observed,
                   const char *const label) {
  uint32_t max_ulp = 0U;
  double max_abs = 0.0;
  size_t mismatches = 0U;
  for (size_t index = 0U; index < observed.size(); ++index) {
    max_ulp =
        std::max(max_ulp, bf16_ulp_distance(control[index], observed[index]));
    max_abs = std::max(
        max_abs, std::abs(static_cast<double>(bf16_to_float(control[index])) -
                          static_cast<double>(bf16_to_float(observed[index]))));
    if (control[index] != observed[index])
      ++mismatches;
  }
  std::printf("output_compare shape=%s label=%s mismatches=%zu max_ulp=%u "
              "max_abs=%.9g status=%s\n",
              shape.name, label, mismatches, max_ulp, max_abs,
              mismatches == 0U ? "PASS" : "FAIL");
  return mismatches == 0U;
}

bool check_neighbor_checksum(const Mode mode, DeviceMatrix *const device) {
  if (mode == Mode::Isolated)
    return true;
  const uint32_t copies = mode == Mode::Staged32 || mode == Mode::Staged128
                              ? kQueryHeadsPerKvHead
                              : 1U;
  const uint32_t splits =
      mode == Mode::Staged32 || mode == Mode::Gqa32 ? 32U : 128U;
  const uint32_t block_count =
      (mode == Mode::Staged32 || mode == Mode::Staged128)
          ? kKvHeads * copies * splits
          : kKvHeads * splits;
  std::vector<uint32_t> observed(block_count);
  check(hipMemcpy(observed.data(), device->predecessor_output.ptr,
                  observed.size() * sizeof(uint32_t), hipMemcpyDeviceToHost),
        "download predecessor checksum");
  std::vector<uint8_t> host_kv;
  fill_host_kv(&host_kv);
  bool ok = true;
  for (uint32_t block = 0U; block < block_count; ++block) {
    const uint32_t split = block % splits;
    const uint32_t group = block / splits;
    const uint32_t kv_head = mode == Mode::Staged32 || mode == Mode::Staged128
                                 ? group / copies
                                 : group;
    const uint64_t begin = kContextTokens * split / splits;
    const uint64_t end = kContextTokens * (split + 1U) / splits;
    // The synthetic reader does not encode query-head identity in its sum;
    // copy is retained only to make the staged mapping explicit above.
    uint32_t expected = 0U;
    for (uint64_t token = begin; token < end; ++token) {
      const uint64_t row = token * kKvHeads + kv_head;
      const size_t value_row = static_cast<size_t>(row * kHeadDim);
      const size_t scale_row = static_cast<size_t>(row * kScaleBytesPerHead);
      for (uint64_t dimension = 0U; dimension < kHeadDim; ++dimension) {
        expected += host_kv[value_row + static_cast<size_t>(dimension)];
        expected += host_kv[static_cast<size_t>(kKvValuesBytes) + value_row +
                            static_cast<size_t>(dimension)];
        expected += host_kv[static_cast<size_t>(2U * kKvValuesBytes) +
                            scale_row + static_cast<size_t>(dimension / 32U)];
        expected +=
            host_kv[static_cast<size_t>(2U * kKvValuesBytes + kKvScalesBytes) +
                    scale_row + static_cast<size_t>(dimension / 32U)];
      }
    }
    if (observed[block] != expected)
      ok = false;
  }
  std::printf("predecessor_oracle mode=%s blocks=%u status=%s\n",
              mode_name(mode).c_str(), block_count, ok ? "PASS" : "FAIL");
  return ok;
}

void parse_options(const int argc, char **const argv, Options *const options) {
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    const auto value = [&](const std::string_view prefix) -> std::string {
      if (argument.substr(0U, prefix.size()) != prefix)
        return {};
      return std::string(argument.substr(prefix.size()));
    };
    if (argument == "--help") {
      std::printf("options: --target=<gfx1030|gfx1201> "
                  "--shape=<wide|down|both> "
                  "--mode=<isolated|staged32|gqa32|staged128|gqa128|all> "
                  "--warmup-ms=<N> --rounds=<N> --samples=<N>\n");
      std::exit(EXIT_SUCCESS);
    }
    if (const std::string target = value("--target="); !target.empty()) {
      options->target = target;
      continue;
    }
    if (const std::string shape = value("--shape="); !shape.empty()) {
      options->shape = shape;
      continue;
    }
    if (const std::string mode = value("--mode="); !mode.empty()) {
      options->mode = mode;
      continue;
    }
    if (const std::string warmup = value("--warmup-ms="); !warmup.empty()) {
      options->warmup_ms = static_cast<uint32_t>(std::stoul(warmup));
      continue;
    }
    if (const std::string rounds = value("--rounds="); !rounds.empty()) {
      options->rounds = static_cast<uint32_t>(std::stoul(rounds));
      continue;
    }
    if (const std::string samples = value("--samples="); !samples.empty()) {
      options->samples = static_cast<uint32_t>(std::stoul(samples));
      continue;
    }
    fail("unknown argument: " + std::string(argument));
  }
  if (options->rounds == 0U || options->samples == 0U)
    fail("--rounds and --samples must be nonzero");
}

int run(const Options &options) {
  std::printf("probe=phase87_wud1_nvfp4_neighbor target=%s context=%llu "
              "kv_heads=%u qheads_per_kv=%u kv_values_bytes=%llu "
              "kv_scales_bytes=%llu kv_total_bytes=%llu "
              "synthetic_predecessor=true warmup_ms=%u rounds=%u samples=%u "
              "weight_pool=%u\n",
              options.target.empty() ? "unspecified" : options.target.c_str(),
              static_cast<unsigned long long>(kContextTokens), kKvHeads,
              kQueryHeadsPerKvHead,
              static_cast<unsigned long long>(kKvValuesBytes),
              static_cast<unsigned long long>(kKvScalesBytes),
              static_cast<unsigned long long>(kKvBytes), options.warmup_ms,
              options.rounds, options.samples, kPoolCopies);
  const std::vector<Mode> modes = selected_modes(options);
  bool all_ok = true;
  for (const size_t shape_index : selected_shapes(options)) {
    const Shape shape = kShapes[shape_index];
    DeviceMatrix device(shape);
    upload_matrix(shape, &device);
    std::vector<uint8_t> activation;
    std::vector<uint8_t> activation_scales;
    std::vector<uint8_t> weights;
    std::vector<uint8_t> weight_scales;
    fill_host_matrix(shape, &activation, &activation_scales, &weights,
                     &weight_scales);
    const bool nonzero_weight =
        std::any_of(weights.begin(), weights.end(),
                    [](const uint8_t value) { return value != 0U; });
    const bool varied_weight =
        std::any_of(weights.begin(), weights.end(), [](const uint8_t value) {
          return value != UINT8_C(0x00) && value != UINT8_C(0xff);
        });
    std::printf("fixture shape=%s synthetic=true activation_bytes=%zu "
                "weight_bytes=%zu weight_scales_bytes=%zu "
                "weight_nonzero=%s weight_varied=%s\n",
                shape.name, activation.size(), weights.size(),
                weight_scales.size(), nonzero_weight ? "true" : "false",
                varied_weight ? "true" : "false");
    const std::vector<uint16_t> host_expected = host_oracle_prefix(
        shape, activation, activation_scales, weights, weight_scales);

    // Capture isolated output once.  Every predecessor condition must leave
    // the NVFP4 output bitwise unchanged when the same weight-pool slot is
    // used for this correctness pass.
    launch_matmul(shape, &device, 0U);
    check(hipStreamSynchronize(device.stream), "isolated correctness sync");
    const std::vector<uint16_t> control = download_output(shape, &device);
    uint32_t oracle_max_ulp = 0U;
    for (size_t index = 0U; index < host_expected.size(); ++index)
      oracle_max_ulp =
          std::max(oracle_max_ulp,
                   bf16_ulp_distance(control[index], host_expected[index]));
    std::printf("numerical_oracle shape=%s columns=%u max_ulp=%u status=%s\n",
                shape.name, static_cast<unsigned>(host_expected.size()),
                oracle_max_ulp, oracle_max_ulp <= 4U ? "PASS" : "FAIL");
    all_ok = oracle_max_ulp <= 4U && all_ok;
    for (const Mode mode : modes) {
      if (mode != Mode::Isolated)
        launch_predecessor(mode, &device);
      launch_matmul(shape, &device, 0U);
      check(hipStreamSynchronize(device.stream), "condition correctness sync");
      all_ok = check_neighbor_checksum(mode, &device) && all_ok;
      all_ok = check_outputs(shape, control, download_output(shape, &device),
                             mode_name(mode).c_str()) &&
               all_ok;
    }

    for (uint32_t round = 0U; round < options.rounds; ++round) {
      for (const bool reverse : {false, true}) {
        for (size_t position = 0U; position < modes.size(); ++position) {
          const size_t mode_index =
              reverse ? modes.size() - 1U - position : position;
          const Mode mode = modes[mode_index];
          continuous_warmup(shape, mode, &device, options.warmup_ms);
          std::vector<float> samples;
          samples.reserve(options.samples);
          for (uint32_t sample = 0U; sample < options.samples; ++sample) {
            const std::string region = "phase87_wud1_nvfp4_" +
                                       std::string(shape.name) + "_" +
                                       mode_name(mode);
            (void)roctxRangePushA(region.c_str());
            // rocprofv3 --selected-regions true records only this bounded
            // measured dispatch.  The 300 ms continuous warmup above is
            // deliberately outside the selected region.
            (void)roctxProfilerResume(0);
            const float elapsed_us =
                measure_call(shape, mode, &device, sample % kPoolCopies);
            (void)roctxProfilerPause(0);
            (void)roctxRangePop();
            samples.push_back(elapsed_us);
            std::printf("sample shape=%s mode=%s order=%s round=%u sample=%u "
                        "nvfp4_us=%.3f weight_slot=%u\n",
                        shape.name, mode_name(mode).c_str(),
                        reverse ? "BA" : "AB", round, sample, elapsed_us,
                        sample % kPoolCopies);
          }
          std::sort(samples.begin(), samples.end());
          std::printf("median shape=%s mode=%s order=%s round=%u nvfp4_us=%.3f "
                      "samples=%u\n",
                      shape.name, mode_name(mode).c_str(),
                      reverse ? "BA" : "AB", round,
                      samples[samples.size() / 2U], options.samples);
        }
      }
    }
  }
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace

int main(const int argc, char **const argv) {
  try {
    Options options;
    parse_options(argc, argv, &options);
    return run(options);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "phase87_wud1_nvfp4_neighbor_probe: %s\n",
                 error.what());
    return EXIT_FAILURE;
  }
}
