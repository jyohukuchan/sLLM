// Phase 87 Stage 0/WU0: shape-sized copy and read bandwidth probes.
//
// These probes are intentionally independent of the lowp ABI.  Copy mode
// measures a simple device-to-device copy for the logical payload size of
// production matrix shapes.  Read mode measures a read-only ceiling with
// payload bytes as its bandwidth denominator.  Neither result claims to be a
// matmul kernel's physical DRAM traffic.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Shape {
  uint64_t m;
  uint64_t k;
  uint64_t n;
  std::string encoding;
};

struct Sample {
  Shape shape;
  uint64_t payload_bytes;
  uint64_t traffic_bytes;
  double median_ms;
  double gib_per_second;
  uint64_t buffer_count;
  uint64_t working_set_bytes;
  uint64_t scalar_output_bytes;
  uint64_t logical_read_write_bytes;
  uint64_t read_grid_blocks;
  std::vector<float> samples_ms;
  uint32_t checksum;
};

enum class ProbeMode { kCopy, kRead };

// kTile is the original WU0 16 KiB-tile read kernel.  kInterleaved lets every
// thread of the whole grid read one uint4 per grid stride, so consecutive
// threads across all blocks touch consecutive 16-byte vectors.
enum class ReadKernel { kTile, kInterleaved };

struct RunOptions {
  ReadKernel read_kernel = ReadKernel::kTile;
  // Additional continuous, unsynchronized launches after the counted warmups
  // until this many milliseconds of GPU time elapsed, so automatic clocks
  // reach the state of a continuously running decoder.
  uint32_t warmup_ms = 0U;
};

__device__ uint32_t block_reduce_sum(uint32_t value) {
  const uint32_t lane = threadIdx.x % warpSize;
  const uint32_t wave = threadIdx.x / warpSize;
  for (uint32_t offset = warpSize / 2U; offset; offset /= 2U)
    value += __shfl_down(value, offset, warpSize);
  __shared__ uint32_t wave_sums[8];
  if (lane == 0U)
    wave_sums[wave] = value;
  __syncthreads();
  if (wave == 0U) {
    value = lane < blockDim.x / warpSize ? wave_sums[lane] : 0U;
    for (uint32_t offset = warpSize / 2U; offset; offset /= 2U)
      value += __shfl_down(value, offset, warpSize);
  }
  return value;
}

// Grid-interleaved read-only probe.  Vector v is read by global thread
// v % (gridDim.x * blockDim.x); four grid strides are loaded before any
// dependent addition.  Block 0 thread 0 also reads the sub-16-byte tail.  The
// host derives each block's checksum in closed form.
__global__ void read_bytes_interleaved(const uint8_t *source,
                                       uint32_t *block_results,
                                       uint64_t bytes) {
  const uint64_t vector_count = bytes / sizeof(uint4);
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                   static_cast<uint64_t>(threadIdx.x);
  const uint4 *vectors = reinterpret_cast<const uint4 *>(source);
  uint32_t accumulator0 = 0U, accumulator1 = 0U;
  uint32_t accumulator2 = 0U, accumulator3 = 0U;
  for (; index + 3U * stride < vector_count; index += 4U * stride) {
    const uint4 v0 = vectors[index];
    const uint4 v1 = vectors[index + stride];
    const uint4 v2 = vectors[index + 2U * stride];
    const uint4 v3 = vectors[index + 3U * stride];
    accumulator0 += v0.x + v0.y + v0.z + v0.w;
    accumulator1 += v1.x + v1.y + v1.z + v1.w;
    accumulator2 += v2.x + v2.y + v2.z + v2.w;
    accumulator3 += v3.x + v3.y + v3.z + v3.w;
  }
  for (; index < vector_count; index += stride) {
    const uint4 v = vectors[index];
    accumulator0 += v.x + v.y + v.z + v.w;
  }
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    uint64_t tail = vector_count * sizeof(uint4);
    for (; tail + sizeof(uint32_t) <= bytes; tail += sizeof(uint32_t))
      accumulator0 += *reinterpret_cast<const uint32_t *>(source + tail);
    for (; tail < bytes; ++tail)
      accumulator0 += source[tail];
  }
  const uint32_t value = block_reduce_sum(accumulator0 + accumulator1 +
                                          accumulator2 + accumulator3);
  if (threadIdx.x == 0U)
    block_results[blockIdx.x] = value;
}

__global__ void copy_bytes(const uint8_t *source, uint8_t *destination,
                           uint64_t bytes) {
  constexpr uint64_t kVectorBytes = sizeof(uint4);
  uint64_t index = (static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                    static_cast<uint64_t>(threadIdx.x)) *
                   kVectorBytes;
  const uint64_t stride =
      static_cast<uint64_t>(gridDim.x) * blockDim.x * kVectorBytes;
  for (; index + kVectorBytes <= bytes; index += stride) {
    reinterpret_cast<uint4 *>(destination)[index / kVectorBytes] =
        reinterpret_cast<const uint4 *>(source)[index / kVectorBytes];
  }
  if (index < bytes) {
    for (uint64_t tail = index; tail < bytes; ++tail) {
      destination[tail] = source[tail];
    }
  }
}

// Read-only bandwidth probe.  A block walks contiguous 16384-byte tiles in a
// grid-stride loop, so the host can independently derive every block's
// checksum from the source fill pattern.  Four independent uint32
// accumulators keep the vector loads live until the block reduction.
__global__ void read_bytes(const uint8_t *source, uint32_t *block_results,
                           uint64_t bytes) {
  constexpr uint64_t kTileBytes = 16384U;
  const uint64_t vector_count = bytes / sizeof(uint4);
  const uint64_t tile_stride = static_cast<uint64_t>(gridDim.x) * 1024U;
  uint64_t tile = static_cast<uint64_t>(blockIdx.x) * 1024U;
  uint32_t accumulator0 = 0U, accumulator1 = 0U;
  uint32_t accumulator2 = 0U, accumulator3 = 0U;
  const uint4 *vectors = reinterpret_cast<const uint4 *>(source);
  for (; tile + 1024U <= vector_count; tile += tile_stride) {
    const uint64_t index = tile + threadIdx.x;
    // Independent loads are issued together before any dependent reduction.
    const uint4 v0 = vectors[index];
    const uint4 v1 = vectors[index + 256U];
    const uint4 v2 = vectors[index + 512U];
    const uint4 v3 = vectors[index + 768U];
    accumulator0 += v0.x + v1.x + v2.x + v3.x;
    accumulator1 += v0.y + v1.y + v2.y + v3.y;
    accumulator2 += v0.z + v1.z + v2.z + v3.z;
    accumulator3 += v0.w + v1.w + v2.w + v3.w;
  }
  if (tile * sizeof(uint4) < bytes) {
#pragma unroll
    for (uint32_t offset = 0U; offset < 4U; ++offset) {
      const uint64_t index = tile + threadIdx.x + offset * 256U;
      if (index < vector_count) {
        const uint4 v = vectors[index];
        accumulator0 += v.x;
        accumulator1 += v.y;
        accumulator2 += v.z;
        accumulator3 += v.w;
      }
    }
    if (threadIdx.x == 0U) {
      uint64_t tail = vector_count * sizeof(uint4);
      for (; tail + sizeof(uint32_t) <= bytes; tail += sizeof(uint32_t))
        accumulator0 += *reinterpret_cast<const uint32_t *>(source + tail);
      for (; tail < bytes; ++tail)
        accumulator0 += source[tail];
    }
  }
  (void)kTileBytes;

  uint32_t value = accumulator0 + accumulator1 + accumulator2 + accumulator3;
  const uint32_t lane = threadIdx.x % warpSize;
  const uint32_t wave = threadIdx.x / warpSize;
  for (uint32_t offset = warpSize / 2U; offset; offset /= 2U)
    value += __shfl_down(value, offset, warpSize);
  __shared__ uint32_t wave_sums[8];
  if (lane == 0U)
    wave_sums[wave] = value;
  __syncthreads();
  if (wave == 0U) {
    value = lane < blockDim.x / warpSize ? wave_sums[lane] : 0U;
    for (uint32_t offset = warpSize / 2U; offset; offset /= 2U)
      value += __shfl_down(value, offset, warpSize);
    if (lane == 0U)
      block_results[blockIdx.x] = value;
  }
}

// Give every read probe buffer an address-varying payload.  This catches a
// kernel that accidentally rereads one cache line while retaining a closed
// form host oracle for each block's grid-stride tile sequence.
__global__ void fill_read_pattern(uint8_t *destination, uint64_t bytes,
                                  uint32_t base) {
  uint64_t word_index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                        static_cast<uint64_t>(threadIdx.x);
  const uint64_t word_stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  const uint64_t word_count = bytes / sizeof(uint32_t);
  while (word_index < word_count) {
    constexpr uint32_t kIncrement = 0x9E3779B9U;
    reinterpret_cast<uint32_t *>(destination)[word_index] =
        base + kIncrement * static_cast<uint32_t>(word_index);
    word_index += word_stride;
  }
  if (blockIdx.x == 0U && threadIdx.x == 0U &&
      (bytes % sizeof(uint32_t)) != 0U) {
    constexpr uint32_t kIncrement = 0x9E3779B9U;
    const uint32_t value =
        base + kIncrement * static_cast<uint32_t>(word_count);
    const uint64_t tail_start = word_count * sizeof(uint32_t);
    for (uint64_t offset = tail_start; offset < bytes; ++offset) {
      reinterpret_cast<uint8_t *>(destination)[offset] =
          static_cast<uint8_t>(value >> (8U * (offset - tail_start)));
    }
  }
}

__global__ void verify_copy(const uint8_t *source, const uint8_t *destination,
                            uint32_t *mismatch, uint64_t bytes) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t offset = index; offset < bytes; offset += stride) {
    if (source[offset] != destination[offset]) {
      atomicExch(mismatch, 1U);
      return;
    }
  }
}

bool hip_ok(hipError_t status, const char *operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::cerr << operation << " failed: " << hipGetErrorString(status) << '\n';
  return false;
}

uint64_t ceil_div(uint64_t numerator, uint64_t denominator) {
  return numerator / denominator + (numerator % denominator != 0U ? 1U : 0U);
}

bool checked_mul(uint64_t left, uint64_t right, uint64_t *result) {
  if (left != 0U && right > std::numeric_limits<uint64_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

bool checked_add(uint64_t left, uint64_t right, uint64_t *result) {
  if (right > std::numeric_limits<uint64_t>::max() - left) {
    return false;
  }
  *result = left + right;
  return true;
}

// Bytes are the same logical bytes used by the Stage 0 profile tool.  They
// include value and scale planes, but exclude cache reuse and kernel-specific
// staging.  The copy moves the payload once in each direction.
bool logical_payload_bytes(const Shape &shape, uint64_t *bytes) {
  if (shape.encoding == "bytes") {
    if (shape.m != 1U || shape.n != 1U) {
      return false;
    }
    *bytes = shape.k;
    return true;
  }
  uint64_t weight_elements = 0U;
  uint64_t activation_elements = 0U;
  if (!checked_mul(shape.k, shape.n, &weight_elements) ||
      !checked_mul(shape.m, shape.k, &activation_elements)) {
    return false;
  }
  uint64_t weight_bytes = 0U;
  uint64_t activation_bytes = 0U;
  if (shape.encoding == "nvfp4") {
    if (!checked_mul(shape.n, ceil_div(shape.k, 2U), &weight_bytes)) {
      return false;
    }
    uint64_t scales = 0U;
    if (!checked_mul(ceil_div(shape.k, 16U), shape.n, &scales) ||
        !checked_add(weight_bytes, scales, &weight_bytes)) {
      return false;
    }
    if (!checked_mul(shape.m, ceil_div(shape.k, 2U), &activation_bytes) ||
        !checked_mul(shape.m, ceil_div(shape.k, 16U), &scales) ||
        !checked_add(activation_bytes, scales, &activation_bytes) ||
        !checked_add(activation_bytes, 8U, &activation_bytes)) {
      return false;
    }
  } else if (shape.encoding == "fp8") {
    weight_bytes = weight_elements;
    if (!checked_add(weight_bytes, 4U * shape.n, &weight_bytes)) {
      return false;
    }
    activation_bytes = activation_elements;
    if (!checked_add(activation_bytes, 4U * shape.m, &activation_bytes)) {
      return false;
    }
  } else if (shape.encoding == "bf16") {
    if (!checked_mul(weight_elements, 2U, &weight_bytes) ||
        !checked_mul(activation_elements, 2U, &activation_bytes)) {
      return false;
    }
  } else {
    return false;
  }
  return checked_add(weight_bytes, activation_bytes, bytes);
}

bool parse_shape(const std::string &text, Shape *shape) {
  // MxKxN:encoding, e.g. 1x5120x17408:nvfp4.
  const std::size_t colon = text.find(':');
  if (colon == std::string::npos || colon == 0U || colon + 1U >= text.size()) {
    return false;
  }
  std::string geometry = text.substr(0U, colon);
  std::string encoding = text.substr(colon + 1U);
  std::size_t first = geometry.find('x');
  std::size_t second = first == std::string::npos
                           ? std::string::npos
                           : geometry.find('x', first + 1U);
  if (first == std::string::npos || second == std::string::npos ||
      geometry.find('x', second + 1U) != std::string::npos) {
    return false;
  }
  try {
    shape->m = std::stoull(geometry.substr(0U, first));
    shape->k = std::stoull(geometry.substr(first + 1U, second - first - 1U));
    shape->n = std::stoull(geometry.substr(second + 1U));
  } catch (const std::exception &) {
    return false;
  }
  return shape->m != 0U && shape->k != 0U && shape->n != 0U &&
         (encoding == "nvfp4" || encoding == "fp8" || encoding == "bf16" ||
          encoding == "bytes") &&
         (shape->encoding = std::move(encoding), true);
}

std::vector<Shape> default_shapes() {
  return {
      {1U, 5120U, 17408U, "nvfp4"}, {1U, 17408U, 5120U, "nvfp4"},
      {1U, 6144U, 5120U, "fp8"},    {1U, 5120U, 10240U, "fp8"},
      {1U, 5120U, 6144U, "fp8"},    {1U, 5120U, 16384U, "fp8"},
      {1U, 5120U, 248320U, "fp8"},  {1U, 5120U, 34816U, "nvfp4"},
      {1U, 5120U, 17408U, "fp8"},   {1U, 17408U, 5120U, "fp8"},
  };
}

bool parse_u32(const char *text, uint32_t *value) {
  if (text == nullptr || *text == '\0') {
    return false;
  }
  char *end = nullptr;
  unsigned long parsed = std::strtoul(text, &end, 10);
  if (*end != '\0' || parsed > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  *value = static_cast<uint32_t>(parsed);
  return true;
}

double median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2U;
  if ((values.size() & 1U) != 0U) {
    return static_cast<double>(values[middle]);
  }
  return (static_cast<double>(values[middle - 1U]) +
          static_cast<double>(values[middle])) /
         2.0;
}

// Launches `launch(iteration)` back to back without host synchronization in
// batches of 16 until `warmup_ms` of GPU time has elapsed.  Returns the number
// of launches so the caller can continue the buffer rotation.
template <typename Launch>
bool timed_warmup(uint32_t warmup_ms, hipEvent_t start, hipEvent_t stop,
                  uint64_t first_iteration, Launch launch, uint64_t *launched) {
  *launched = 0U;
  if (warmup_ms == 0U) {
    return true;
  }
  if (!hip_ok(hipEventRecord(start, nullptr), "hipEventRecord warmup")) {
    return false;
  }
  float elapsed = 0.0F;
  while (elapsed < static_cast<float>(warmup_ms)) {
    for (uint32_t batch = 0U; batch < 16U; ++batch) {
      launch(first_iteration + *launched);
      ++*launched;
    }
    if (!hip_ok(hipGetLastError(), "timed warmup launch") ||
        !hip_ok(hipEventRecord(stop, nullptr), "hipEventRecord warmup stop") ||
        !hip_ok(hipEventSynchronize(stop), "hipEventSynchronize warmup") ||
        !hip_ok(hipEventElapsedTime(&elapsed, start, stop),
                "hipEventElapsedTime warmup")) {
      return false;
    }
  }
  return true;
}

bool run_shape(const Shape &shape, uint32_t warmups, uint32_t measured,
               const RunOptions &options, Sample *sample) {
  uint64_t payload_bytes = 0U;
  if (!logical_payload_bytes(shape, &payload_bytes) || payload_bytes == 0U ||
      payload_bytes >
          static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      payload_bytes > std::numeric_limits<uint64_t>::max() / 2U) {
    std::cerr << "invalid or overflowing payload for shape " << shape.m << 'x'
              << shape.k << 'x' << shape.n << ':' << shape.encoding << '\n';
    return false;
  }
  // Matrices and KV payloads rotate over at least 512 MiB so that the
  // V620's large cache cannot turn this into an accidentally hot-cache copy.
  // Sub-MiB probes intentionally characterize launch/cache latency instead.
  const std::size_t kBufferCount = static_cast<std::size_t>(
      payload_bytes >= 1024U * 1024U
          ? std::max<uint64_t>(
                3U, ceil_div(512ULL * 1024U * 1024U, 2U * payload_bytes))
          : 3U);
  if (payload_bytes >
      std::numeric_limits<uint64_t>::max() / (2U * kBufferCount)) {
    return false;
  }
  std::vector<uint8_t *> sources(kBufferCount, nullptr);
  std::vector<uint8_t *> destinations(kBufferCount, nullptr);
  std::vector<bool> touched(kBufferCount, false);
  uint32_t *mismatch = nullptr;
  bool valid = true;
  for (std::size_t slot = 0U; slot < kBufferCount && valid; ++slot) {
    valid = hip_ok(hipMalloc(reinterpret_cast<void **>(&sources[slot]),
                             payload_bytes),
                   "hipMalloc source") &&
            hip_ok(hipMalloc(reinterpret_cast<void **>(&destinations[slot]),
                             payload_bytes),
                   "hipMalloc destination") &&
            hip_ok(hipMemset(sources[slot], 0xA5, payload_bytes),
                   "hipMemset source") &&
            hip_ok(hipMemset(destinations[slot], 0, payload_bytes),
                   "hipMemset destination");
  }
  if (valid) {
    valid = hip_ok(
        hipMalloc(reinterpret_cast<void **>(&mismatch), sizeof(*mismatch)),
        "hipMalloc mismatch");
  }
  const uint64_t block = 256U;
  const uint64_t requested_grid = ceil_div(
      ceil_div(payload_bytes, static_cast<uint64_t>(sizeof(uint4))), block);
  const uint32_t grid = static_cast<uint32_t>(
      std::min<uint64_t>(std::max<uint64_t>(requested_grid, 1U), 65535U));
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  if (valid) {
    valid = hip_ok(hipEventCreate(&start), "hipEventCreate start") &&
            hip_ok(hipEventCreate(&stop), "hipEventCreate stop");
  }
  std::vector<float> milliseconds;
  milliseconds.reserve(measured);
  for (uint32_t iteration = 0U; valid && iteration < warmups; ++iteration) {
    const std::size_t slot = iteration % kBufferCount;
    touched[slot] = true;
    hipLaunchKernelGGL(copy_bytes, dim3(grid), dim3(block), 0U, nullptr,
                       sources[slot], destinations[slot], payload_bytes);
    valid = hip_ok(hipGetLastError(), "copy launch warmup") &&
            hip_ok(hipDeviceSynchronize(), "copy synchronize warmup");
  }
  uint64_t timed_launches = 0U;
  if (valid) {
    valid = timed_warmup(
        options.warmup_ms, start, stop, warmups,
        [&](uint64_t iteration) {
          const std::size_t slot = iteration % kBufferCount;
          touched[slot] = true;
          hipLaunchKernelGGL(copy_bytes, dim3(grid), dim3(block), 0U, nullptr,
                             sources[slot], destinations[slot], payload_bytes);
        },
        &timed_launches);
  }
  for (uint32_t iteration = 0U; valid && iteration < measured; ++iteration) {
    const std::size_t slot =
        (static_cast<uint64_t>(warmups) + timed_launches + iteration) %
        kBufferCount;
    touched[slot] = true;
    valid = hip_ok(hipEventRecord(start, nullptr), "hipEventRecord start");
    if (valid) {
      hipLaunchKernelGGL(copy_bytes, dim3(grid), dim3(block), 0U, nullptr,
                         sources[slot], destinations[slot], payload_bytes);
      valid = hip_ok(hipGetLastError(), "copy launch measured");
    }
    if (valid) {
      valid = hip_ok(hipEventRecord(stop, nullptr), "hipEventRecord stop") &&
              hip_ok(hipEventSynchronize(stop), "hipEventSynchronize stop");
    }
    float elapsed = 0.0F;
    if (valid) {
      valid = hip_ok(hipEventElapsedTime(&elapsed, start, stop),
                     "hipEventElapsedTime") &&
              elapsed > 0.0F;
    }
    if (valid) {
      milliseconds.push_back(elapsed);
    }
  }
  if (valid) {
    for (std::size_t slot = 0U; slot < kBufferCount && valid; ++slot) {
      if (!touched[slot]) {
        continue;
      }
      valid = hip_ok(hipMemset(mismatch, 0, sizeof(*mismatch)),
                     "hipMemset mismatch");
      if (valid) {
        hipLaunchKernelGGL(verify_copy, dim3(grid), dim3(block), 0U, nullptr,
                           sources[slot], destinations[slot], mismatch,
                           payload_bytes);
        valid = hip_ok(hipGetLastError(), "verify launch") &&
                hip_ok(hipDeviceSynchronize(), "verify synchronize");
      }
      uint32_t mismatch_host = 0U;
      if (valid) {
        valid = hip_ok(hipMemcpy(&mismatch_host, mismatch,
                                 sizeof(mismatch_host), hipMemcpyDeviceToHost),
                       "hipMemcpy mismatch") &&
                mismatch_host == 0U;
      }
      if (!valid && mismatch_host != 0U) {
        std::cerr << "copy verification mismatch in buffer " << slot << '\n';
      }
    }
  }
  if (start != nullptr) {
    valid = hip_ok(hipEventDestroy(start), "hipEventDestroy start") && valid;
  }
  if (stop != nullptr) {
    valid = hip_ok(hipEventDestroy(stop), "hipEventDestroy stop") && valid;
  }
  if (mismatch != nullptr) {
    valid = hip_ok(hipFree(mismatch), "hipFree mismatch") && valid;
  }
  for (std::size_t slot = 0U; slot < kBufferCount; ++slot) {
    if (destinations[slot] != nullptr) {
      valid =
          hip_ok(hipFree(destinations[slot]), "hipFree destination") && valid;
    }
    if (sources[slot] != nullptr) {
      valid = hip_ok(hipFree(sources[slot]), "hipFree source") && valid;
    }
  }
  if (!valid || milliseconds.size() != measured) {
    return false;
  }
  const double median_ms = median(std::move(milliseconds));
  const double traffic_bytes = static_cast<double>(payload_bytes) * 2.0;
  sample->buffer_count = kBufferCount;
  sample->working_set_bytes = payload_bytes * 2U * kBufferCount;
  sample->shape = shape;
  sample->payload_bytes = payload_bytes;
  sample->traffic_bytes = payload_bytes * 2U;
  sample->median_ms = median_ms;
  sample->gib_per_second =
      traffic_bytes / (median_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
  return true;
}

uint32_t interleaved_read_grid(uint64_t payload_bytes, uint32_t block) {
  const uint64_t vector_count = payload_bytes / sizeof(uint4);
  return static_cast<uint32_t>(std::min<uint64_t>(
      std::max<uint64_t>(ceil_div(vector_count, block), 1U), 65535U));
}

// Closed-form checksum of the vectors read by one block of
// read_bytes_interleaved.  Vector v holds words 4v..4v+3 whose modular sum is
// 4*base + increment*(16v+6).
uint32_t expected_interleaved_checksum(uint64_t payload_bytes, uint32_t grid,
                                       uint32_t block, uint32_t block_index,
                                       uint32_t base) {
  constexpr uint32_t kIncrement = 0x9E3779B9U;
  const uint64_t vector_count = payload_bytes / sizeof(uint4);
  const uint64_t stride = static_cast<uint64_t>(grid) * block;
  uint32_t expected = 0U;
  for (uint64_t first = static_cast<uint64_t>(block_index) * block;
       first < vector_count; first += stride) {
    const uint64_t last = std::min<uint64_t>(first + block, vector_count);
    const uint64_t count = last - first;
    const uint64_t index_sum = (first + last - 1U) * count / 2U;
    expected += static_cast<uint32_t>(count) * (4U * base + 6U * kIncrement);
    expected += 16U * kIncrement * static_cast<uint32_t>(index_sum);
  }
  if (block_index == 0U) {
    uint64_t tail = vector_count * sizeof(uint4);
    for (; tail + sizeof(uint32_t) <= payload_bytes; tail += sizeof(uint32_t))
      expected += base + kIncrement * static_cast<uint32_t>(tail / 4U);
    const uint32_t tail_value =
        base + kIncrement * static_cast<uint32_t>(tail / 4U);
    for (uint64_t offset = tail; offset < payload_bytes; ++offset)
      expected += static_cast<uint8_t>(
          tail_value >> (8U * static_cast<uint32_t>(offset - tail)));
  }
  return expected;
}

uint32_t read_grid_for_payload(uint64_t payload_bytes, uint32_t block,
                               int compute_units) {
  constexpr uint64_t kTileBytes = 16384U;
  const uint64_t tile_count = ceil_div(payload_bytes, kTileBytes);
  const uint64_t grid =
      std::min<uint64_t>(std::max<uint64_t>(tile_count, 1U), 65535U);
  (void)block;
  (void)compute_units;
  return static_cast<uint32_t>(grid);
}

uint32_t expected_read_checksum(uint64_t payload_bytes, uint32_t grid,
                                uint32_t block_index, uint32_t base) {
  constexpr uint64_t kTileBytes = 16384U;
  constexpr uint32_t kIncrement = 0x9E3779B9U;
  constexpr uint32_t kBytesPerWord = sizeof(uint32_t);
  uint32_t expected = 0U;
  const uint64_t tile_stride = static_cast<uint64_t>(grid) * kTileBytes;
  for (uint64_t tile = static_cast<uint64_t>(block_index) * kTileBytes;
       tile < payload_bytes; tile += tile_stride) {
    const uint64_t tile_bytes =
        std::min<uint64_t>(kTileBytes, payload_bytes - tile);
    const uint32_t word_count =
        static_cast<uint32_t>(tile_bytes / kBytesPerWord);
    const uint32_t first_word = static_cast<uint32_t>(tile / kBytesPerWord);
    const uint32_t first_value = base + kIncrement * first_word;
    // All arithmetic is intentionally uint32_t modular arithmetic, matching
    // each GPU accumulator and the final block reduction.
    expected += word_count * first_value;
    expected += kIncrement * (word_count * (word_count - 1U) / 2U);
    const uint64_t tail_start =
        tile + static_cast<uint64_t>(word_count) * kBytesPerWord;
    const uint32_t tail_value =
        base + kIncrement * static_cast<uint32_t>(tail_start / kBytesPerWord);
    for (uint64_t offset = tail_start; offset < tile + tile_bytes; ++offset) {
      expected += static_cast<uint8_t>(
          tail_value >> (8U * static_cast<uint32_t>(offset - tail_start)));
    }
  }
  return expected;
}

bool verify_read_results(const std::vector<uint32_t> &observed,
                         uint64_t payload_bytes, uint32_t grid, uint32_t base,
                         ReadKernel kernel, uint32_t *aggregate) {
  if (observed.size() != grid) {
    std::cerr << "read result count mismatch: expected " << grid << ", got "
              << observed.size() << '\n';
    return false;
  }
  uint32_t checksum = 0U;
  for (uint32_t block = 0U; block < grid; ++block) {
    const uint32_t expected =
        kernel == ReadKernel::kInterleaved
            ? expected_interleaved_checksum(payload_bytes, grid, 256U, block,
                                            base)
            : expected_read_checksum(payload_bytes, grid, block, base);
    if (observed[block] != expected) {
      std::cerr << "read checksum mismatch in block " << block << ": expected "
                << expected << ", got " << observed[block] << '\n';
      return false;
    }
    checksum += observed[block];
  }
  *aggregate = checksum;
  return true;
}

bool run_shape_read(const Shape &shape, uint32_t warmups, uint32_t measured,
                    int compute_units, const RunOptions &options,
                    Sample *sample) {
  uint64_t payload_bytes = 0U;
  if (!logical_payload_bytes(shape, &payload_bytes) || payload_bytes == 0U ||
      payload_bytes >
          static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    std::cerr << "invalid or overflowing payload for shape " << shape.m << 'x'
              << shape.k << 'x' << shape.n << ':' << shape.encoding << '\n';
    return false;
  }
  // Read mode rotates source allocations whose aggregate size is at least the
  // same 512 MiB streaming pool used by copy mode.  The output is only one
  // uint32 per block and is accounted separately from this source footprint.
  const std::size_t kBufferCount = static_cast<std::size_t>(
      payload_bytes >= 1024U * 1024U
          ? std::max<uint64_t>(3U,
                               ceil_div(512ULL * 1024U * 1024U, payload_bytes))
          : 3U);
  if (payload_bytes > std::numeric_limits<uint64_t>::max() / kBufferCount) {
    return false;
  }
  constexpr uint32_t kBlock = 256U;
  const uint32_t grid =
      options.read_kernel == ReadKernel::kInterleaved
          ? interleaved_read_grid(payload_bytes, kBlock)
          : read_grid_for_payload(payload_bytes, kBlock, compute_units);
  const auto read_kernel = options.read_kernel == ReadKernel::kInterleaved
                               ? read_bytes_interleaved
                               : read_bytes;
  std::vector<uint8_t *> sources(kBufferCount, nullptr);
  uint32_t *block_results = nullptr;
  bool valid = true;
  const uint64_t word_count = ceil_div(payload_bytes, sizeof(uint32_t));
  const uint32_t fill_grid = static_cast<uint32_t>(std::min<uint64_t>(
      std::max<uint64_t>(ceil_div(word_count, kBlock), 1U), 65535U));
  std::vector<uint32_t> bases(kBufferCount, 0U);
  for (std::size_t slot = 0U; slot < kBufferCount && valid; ++slot) {
    bases[slot] = 0x10203040U + static_cast<uint32_t>(slot) * 0x01010101U;
    valid = hip_ok(
        hipMalloc(reinterpret_cast<void **>(&sources[slot]), payload_bytes),
        "hipMalloc read source");
    if (valid) {
      hipLaunchKernelGGL(fill_read_pattern, dim3(fill_grid), dim3(kBlock), 0U,
                         nullptr, sources[slot], payload_bytes, bases[slot]);
      valid = hip_ok(hipGetLastError(), "read pattern launch") &&
              hip_ok(hipDeviceSynchronize(), "read pattern synchronize");
    }
  }
  if (valid) {
    valid = hip_ok(
        hipMalloc(reinterpret_cast<void **>(&block_results),
                  static_cast<std::size_t>(grid) * measured * sizeof(uint32_t)),
        "hipMalloc read results");
  }
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  if (valid) {
    valid = hip_ok(hipEventCreate(&start), "hipEventCreate start") &&
            hip_ok(hipEventCreate(&stop), "hipEventCreate stop");
  }
  std::vector<uint32_t> observed(grid, 0U);
  std::vector<float> milliseconds;
  milliseconds.reserve(measured);
  uint32_t aggregate_checksum = 0U;
  for (uint32_t iteration = 0U; valid && iteration < warmups; ++iteration) {
    const std::size_t slot = iteration % kBufferCount;
    hipLaunchKernelGGL(read_kernel, dim3(grid), dim3(kBlock), 0U, nullptr,
                       sources[slot], block_results, payload_bytes);
    valid = hip_ok(hipGetLastError(), "read launch warmup") &&
            hip_ok(hipDeviceSynchronize(), "read synchronize warmup");
  }
  uint64_t timed_launches = 0U;
  if (valid) {
    valid = timed_warmup(
        options.warmup_ms, start, stop, warmups,
        [&](uint64_t iteration) {
          hipLaunchKernelGGL(read_kernel, dim3(grid), dim3(kBlock), 0U, nullptr,
                             sources[iteration % kBufferCount], block_results,
                             payload_bytes);
        },
        &timed_launches);
  }
  const uint64_t first_measured = warmups + timed_launches;
  for (uint32_t iteration = 0U; valid && iteration < measured; ++iteration) {
    const std::size_t slot = (first_measured + iteration) % kBufferCount;
    valid = hip_ok(hipEventRecord(start, nullptr), "hipEventRecord start");
    if (valid) {
      hipLaunchKernelGGL(
          read_kernel, dim3(grid), dim3(kBlock), 0U, nullptr, sources[slot],
          block_results + static_cast<std::size_t>(iteration) * grid,
          payload_bytes);
      valid = hip_ok(hipGetLastError(), "read launch measured");
    }
    if (valid) {
      valid = hip_ok(hipEventRecord(stop, nullptr), "hipEventRecord stop") &&
              hip_ok(hipEventSynchronize(stop), "hipEventSynchronize stop");
    }
    float elapsed = 0.0F;
    if (valid) {
      valid = hip_ok(hipEventElapsedTime(&elapsed, start, stop),
                     "hipEventElapsedTime") &&
              elapsed > 0.0F;
    }
    if (valid) {
      milliseconds.push_back(elapsed);
    }
  }
  // Like copy mode, keep readback and correctness checking outside the entire
  // timed sequence, not between samples where host work would cool the GPU.
  for (uint32_t iteration = 0U; valid && iteration < measured; ++iteration) {
    const std::size_t slot = (first_measured + iteration) % kBufferCount;
    valid = hip_ok(hipMemcpy(observed.data(),
                             block_results +
                                 static_cast<std::size_t>(iteration) * grid,
                             static_cast<std::size_t>(grid) * sizeof(uint32_t),
                             hipMemcpyDeviceToHost),
                   "hipMemcpy read results") &&
            verify_read_results(observed, payload_bytes, grid, bases[slot],
                                options.read_kernel, &aggregate_checksum);
  }
  if (start != nullptr) {
    valid = hip_ok(hipEventDestroy(start), "hipEventDestroy start") && valid;
  }
  if (stop != nullptr) {
    valid = hip_ok(hipEventDestroy(stop), "hipEventDestroy stop") && valid;
  }
  if (block_results != nullptr) {
    valid = hip_ok(hipFree(block_results), "hipFree read results") && valid;
  }
  for (uint8_t *source : sources) {
    if (source != nullptr) {
      valid = hip_ok(hipFree(source), "hipFree read source") && valid;
    }
  }
  if (!valid || milliseconds.size() != measured) {
    return false;
  }
  sample->samples_ms = milliseconds;
  const double median_ms = median(std::move(milliseconds));
  const uint64_t scalar_output_bytes =
      static_cast<uint64_t>(grid) * sizeof(uint32_t);
  sample->buffer_count = kBufferCount;
  sample->working_set_bytes = payload_bytes * kBufferCount;
  sample->shape = shape;
  sample->payload_bytes = payload_bytes;
  sample->traffic_bytes = payload_bytes;
  sample->scalar_output_bytes = scalar_output_bytes;
  sample->logical_read_write_bytes = payload_bytes + scalar_output_bytes;
  sample->read_grid_blocks = grid;
  sample->checksum = aggregate_checksum;
  sample->median_ms = median_ms;
  sample->gib_per_second = static_cast<double>(payload_bytes) /
                           (median_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
  return true;
}

void print_sample(const Sample &sample, const char *target, uint32_t warmups,
                  uint32_t measured, const RunOptions &options) {
  std::cout << std::fixed << std::setprecision(6)
            << "{\"schema_version\":\"phase87-copy-bandwidth-v1\","
            << "\"status\":\"PASS\",\"warmup_ms\":" << options.warmup_ms
            << ",\"target\":\"" << target << "\",\"m\":" << sample.shape.m
            << ",\"k\":" << sample.shape.k << ",\"n\":" << sample.shape.n
            << ",\"encoding\":\"" << sample.shape.encoding
            << "\",\"warmups\":" << warmups << ",\"measured\":" << measured
            << ",\"payload_bytes\":" << sample.payload_bytes
            << ",\"copy_traffic_bytes\":" << sample.traffic_bytes
            << ",\"buffer_count\":" << sample.buffer_count
            << ",\"working_set_bytes\":" << sample.working_set_bytes
            << ",\"median_ms\":" << sample.median_ms
            << ",\"effective_gib_per_s\":" << sample.gib_per_second << "}\n";
}

void print_read_sample(const Sample &sample, const char *target,
                       uint32_t warmups, uint32_t measured,
                       const RunOptions &options) {
  std::cout << std::fixed << std::setprecision(6)
            << "{\"schema_version\":\"phase87-read-bandwidth-v1\","
            << "\"status\":\"PASS\",\"mode\":\"read\",\"read_kernel\":\""
            << (options.read_kernel == ReadKernel::kInterleaved ? "interleaved"
                                                                : "tile")
            << "\",\"warmup_ms\":" << options.warmup_ms << ",\"target\":\""
            << target << "\",\"m\":" << sample.shape.m
            << ",\"k\":" << sample.shape.k << ",\"n\":" << sample.shape.n
            << ",\"encoding\":\"" << sample.shape.encoding
            << "\",\"warmups\":" << warmups << ",\"measured\":" << measured
            << ",\"payload_bytes\":" << sample.payload_bytes
            << ",\"scalar_output_bytes\":" << sample.scalar_output_bytes
            << ",\"logical_read_write_bytes\":"
            << sample.logical_read_write_bytes
            << ",\"read_grid_blocks\":" << sample.read_grid_blocks
            << ",\"buffer_count\":" << sample.buffer_count
            << ",\"source_working_set_bytes\":" << sample.working_set_bytes
            << ",\"checksum\":" << sample.checksum
            << ",\"median_ms\":" << sample.median_ms
            << ",\"effective_gib_per_s\":" << sample.gib_per_second
            << ",\"samples_ms\":[";
  for (std::size_t i = 0; i < sample.samples_ms.size(); ++i) {
    if (i)
      std::cout << ",";
    std::cout << sample.samples_ms[i];
  }
  std::cout << "]}\n";
}

} // namespace

int main(int argc, char **argv) {
  uint32_t warmups = 1U;
  uint32_t measured = 5U;
  ProbeMode mode = ProbeMode::kCopy;
  bool mode_seen = false;
  RunOptions options{};
  std::vector<Shape> shapes;
  for (int index = 1; index < argc; ++index) {
    const char *argument = argv[index];
    if (std::strcmp(argument, "--mode") == 0 && index + 1 < argc) {
      if (mode_seen) {
        std::cerr << "duplicate --mode\n";
        return 2;
      }
      mode_seen = true;
      const char *value = argv[++index];
      if (std::strcmp(value, "copy") == 0) {
        mode = ProbeMode::kCopy;
      } else if (std::strcmp(value, "read") == 0) {
        mode = ProbeMode::kRead;
      } else {
        std::cerr << "invalid --mode; expected copy or read\n";
        return 2;
      }
    } else if (std::strcmp(argument, "--read-kernel") == 0 &&
               index + 1 < argc) {
      const char *value = argv[++index];
      if (std::strcmp(value, "tile") == 0) {
        options.read_kernel = ReadKernel::kTile;
      } else if (std::strcmp(value, "interleaved") == 0) {
        options.read_kernel = ReadKernel::kInterleaved;
      } else {
        std::cerr << "invalid --read-kernel; expected tile or interleaved\n";
        return 2;
      }
    } else if (std::strcmp(argument, "--warmup-ms") == 0 && index + 1 < argc) {
      if (!parse_u32(argv[++index], &options.warmup_ms)) {
        std::cerr << "invalid --warmup-ms\n";
        return 2;
      }
    } else if (std::strcmp(argument, "--warmups") == 0 && index + 1 < argc) {
      if (!parse_u32(argv[++index], &warmups)) {
        std::cerr << "invalid --warmups\n";
        return 2;
      }
    } else if (std::strcmp(argument, "--measured") == 0 && index + 1 < argc) {
      if (!parse_u32(argv[++index], &measured) || measured == 0U) {
        std::cerr << "invalid --measured\n";
        return 2;
      }
    } else if (std::strcmp(argument, "--shape") == 0 && index + 1 < argc) {
      Shape shape{};
      if (!parse_shape(argv[++index], &shape)) {
        std::cerr << "invalid --shape; expected MxKxN:encoding\n";
        return 2;
      }
      shapes.push_back(std::move(shape));
    } else {
      std::cerr << "usage: phase87_copy_bandwidth [--warmups N] [--measured N]"
                   " [--mode copy|read] [--read-kernel tile|interleaved]"
                   " [--warmup-ms N] [--shape MxKxN:encoding]\n";
      return 2;
    }
  }
  if (shapes.empty()) {
    shapes = default_shapes();
  }
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, 0),
              "hipGetDeviceProperties")) {
    return 1;
  }
  const std::string target(properties.gcnArchName);
  // Match an already-running decoder rather than measuring the first copy
  // while automatic clocks are still rising. Keep the same HIP context and
  // use the largest production payload; this is not a power-policy change.
  Sample preheat{};
  const Shape preheat_shape{1U, 5120U, 248320U, "fp8"};
  const bool preheat_ok =
      mode == ProbeMode::kRead
          ? run_shape_read(preheat_shape, 64U, 3U,
                           properties.multiProcessorCount, options, &preheat)
          : run_shape(preheat_shape, 64U, 3U, options, &preheat);
  if (!preheat_ok) {
    std::cerr << (mode == ProbeMode::kRead ? "read" : "copy")
              << " preheat failed\n";
    return 1;
  }
  std::vector<Sample> samples;
  samples.reserve(shapes.size());
  for (const Shape &shape : shapes) {
    Sample sample{};
    const bool shape_ok =
        mode == ProbeMode::kRead
            ? run_shape_read(shape, warmups, measured,
                             properties.multiProcessorCount, options, &sample)
            : run_shape(shape, warmups, measured, options, &sample);
    if (!shape_ok) {
      std::cerr << (mode == ProbeMode::kRead ? "read" : "copy")
                << " bandwidth probe failed for " << shape.m << 'x' << shape.k
                << 'x' << shape.n << ':' << shape.encoding << '\n';
      return 1;
    }
    if (mode == ProbeMode::kRead) {
      print_read_sample(sample, target.c_str(), warmups, measured, options);
    } else {
      print_sample(sample, target.c_str(), warmups, measured, options);
    }
    samples.push_back(sample);
  }
  if (!hip_ok(hipDeviceSynchronize(), "final hipDeviceSynchronize")) {
    return 1;
  }
  return 0;
}
