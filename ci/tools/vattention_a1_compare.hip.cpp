// SPDX-License-Identifier: MIT
// Model-free Phase 6 A1 comparison: contiguous, vAttention VMM, and paged KV.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kQHeads = 16;
constexpr int kKvHeads = 4;
constexpr int kHeadDim = 256;
constexpr int kBlockTokens = 256;
constexpr int kLogicalCapacity = 4096;
constexpr int kWarmups = 3;
constexpr int kMeasurements = 9;
constexpr int kQueryLengths[] = {1, 37};
constexpr int kKvLengths[] = {255, 256, 257, 1023, 1024, 1025};
constexpr const char *kKernelSymbol = "sllm_a1_online_attention_proxy_v1";

void check(hipError_t status, const char *operation) {
  if (status != hipSuccess) {
    std::ostringstream message;
    message << operation << " failed: " << hipGetErrorName(status) << ": "
            << hipGetErrorString(status);
    throw std::runtime_error(message.str());
  }
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
  if (alignment == 0 || value > SIZE_MAX - (alignment - 1)) {
    throw std::runtime_error("invalid alignment or size overflow");
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

std::string format_bdf(const hipDeviceProp_t &properties) {
  std::ostringstream value;
  value << std::hex << std::setfill('0') << std::setw(4)
        << properties.pciDomainID << ":" << std::setw(2) << properties.pciBusID
        << ":" << std::setw(2) << properties.pciDeviceID << ".0";
  return value.str();
}

std::string json_escape(const std::string &value) {
  std::ostringstream escaped;
  for (unsigned char character : value) {
    if (character == '\\' || character == '"') {
      escaped << '\\' << character;
    } else if (character == '\n') {
      escaped << "\\n";
    } else if (character < 0x20) {
      escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(character) << std::dec;
    } else {
      escaped << character;
    }
  }
  return escaped.str();
}

double percentile(std::vector<float> values, double fraction) {
  if (values.empty()) {
    throw std::runtime_error("empty timing sample");
  }
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(
      fraction * static_cast<double>(values.size() - 1));
  return static_cast<double>(values[index]) * 1000.0;
}

__host__ __device__ std::uint16_t float_to_bf16(float value) {
#if defined(__HIP_DEVICE_COMPILE__)
  const std::uint32_t bits = __float_as_uint(value);
#else
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
#endif
  const std::uint32_t rounding = 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

__host__ __device__ float bf16_to_float(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
#if defined(__HIP_DEVICE_COMPILE__)
  return __uint_as_float(bits);
#else
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
#endif
}

__global__ void fill_query(std::uint16_t *query, std::size_t elements) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    const float value =
        static_cast<float>(static_cast<int>((index * 17U + 13U) % 29U) - 14) /
        32.0F;
    query[index] = float_to_bf16(value);
  }
}

__device__ std::size_t kv_offset(int mode, int token, int head, int dim,
                                 const int *block_table) {
  if (mode == 2) {
    const int logical_block = token / kBlockTokens;
    const int token_in_block = token % kBlockTokens;
    const int physical_block = block_table[logical_block];
    return ((static_cast<std::size_t>(physical_block) * kBlockTokens +
             token_in_block) *
                kKvHeads +
            head) *
               kHeadDim +
           dim;
  }
  return (static_cast<std::size_t>(token) * kKvHeads + head) * kHeadDim + dim;
}

__global__ void fill_kv(__half *key, __half *value, int tokens, int mode,
                        const int *block_table) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t elements =
      static_cast<std::size_t>(tokens) * kKvHeads * kHeadDim;
  if (index >= elements) {
    return;
  }
  const int dim = static_cast<int>(index % kHeadDim);
  const std::size_t head_token = index / kHeadDim;
  const int head = static_cast<int>(head_token % kKvHeads);
  const int token = static_cast<int>(head_token / kKvHeads);
  const std::size_t target = kv_offset(mode, token, head, dim, block_table);
  const float key_value =
      static_cast<float>(static_cast<int>((index * 19U + 7U) % 31U) - 15) /
      32.0F;
  const float value_value =
      static_cast<float>(static_cast<int>((index * 23U + 3U) % 37U) - 18) /
      32.0F;
  key[target] = __float2half(key_value);
  value[target] = __float2half(value_value);
}

// One block computes one (query row, query head). Scores are tiled so the
// softmax state and output accumulator remain fused and online; no score matrix
// is materialized. The mode changes only the KV address accessor above.
extern "C" __global__ void
sllm_a1_online_attention_proxy_v1(const std::uint16_t *query,
                                  const __half *key, const __half *value,
                                  std::uint16_t *output, int query_length,
                                  int kv_length, int mode,
                                  const int *block_table) {
  __shared__ float scores[kBlockTokens];
  __shared__ float reduction[kBlockTokens];
  __shared__ float state_m;
  __shared__ float state_l;
  __shared__ float state_alpha;

  const int dim = threadIdx.x;
  const int query_head = blockIdx.x % kQHeads;
  const int query_row = blockIdx.x / kQHeads;
  const int kv_head = query_head / (kQHeads / kKvHeads);
  const int query_position = kv_length - query_length + query_row;
  const std::size_t query_base =
      (static_cast<std::size_t>(query_row) * kQHeads + query_head) * kHeadDim;
  float accumulator = 0.0F;
  if (dim == 0) {
    state_m = -std::numeric_limits<float>::infinity();
    state_l = 0.0F;
  }
  __syncthreads();

  constexpr float scale = 1.0F / 16.0F; // 1/sqrt(256)
  for (int tile = 0; tile <= query_position; tile += kBlockTokens) {
    const int key_position = tile + dim;
    float score = -std::numeric_limits<float>::infinity();
    if (key_position <= query_position) {
      score = 0.0F;
      for (int inner = 0; inner < kHeadDim; ++inner) {
        const float q = bf16_to_float(query[query_base + inner]);
        const std::size_t offset =
            kv_offset(mode, key_position, kv_head, inner, block_table);
        score += q * __half2float(key[offset]);
      }
      score *= scale;
    }
    scores[dim] = score;
    reduction[dim] = score;
    __syncthreads();
    for (int stride = kBlockTokens / 2; stride > 0; stride /= 2) {
      if (dim < stride) {
        reduction[dim] = fmaxf(reduction[dim], reduction[dim + stride]);
      }
      __syncthreads();
    }
    const float tile_max = reduction[0];
    const float weight = key_position <= query_position
                             ? expf(scores[dim] - tile_max)
                             : 0.0F;
    reduction[dim] = weight;
    __syncthreads();
    for (int stride = kBlockTokens / 2; stride > 0; stride /= 2) {
      if (dim < stride) {
        reduction[dim] += reduction[dim + stride];
      }
      __syncthreads();
    }
    if (dim == 0) {
      const float new_m = fmaxf(state_m, tile_max);
      state_alpha = expf(state_m - new_m);
      state_l = state_l * state_alpha + reduction[0] * expf(tile_max - new_m);
      state_m = new_m;
    }
    __syncthreads();
    accumulator *= state_alpha;
    float tile_output = 0.0F;
    const int tile_end = min(query_position + 1, tile + kBlockTokens);
    for (int position = tile; position < tile_end; ++position) {
      const float normalized_weight = expf(scores[position - tile] - state_m);
      const std::size_t offset =
          kv_offset(mode, position, kv_head, dim, block_table);
      tile_output += normalized_weight * __half2float(value[offset]);
    }
    accumulator += tile_output;
    __syncthreads();
  }
  const std::size_t output_offset = query_base + dim;
  output[output_offset] = float_to_bf16(accumulator / state_l);
}

struct VmmPlane {
  void *address = nullptr;
  std::size_t reserve_bytes = 0;
  std::size_t mapped_bytes = 0;
  std::vector<hipMemGenericAllocationHandle_t> handles;

  VmmPlane() = default;
  VmmPlane(const VmmPlane &) = delete;
  VmmPlane &operator=(const VmmPlane &) = delete;
  ~VmmPlane() { release_noexcept(); }

  void reserve(std::size_t bytes, std::size_t alignment) {
    check(hipMemAddressReserve(&address, bytes, alignment, nullptr, 0),
          "hipMemAddressReserve");
    reserve_bytes = bytes;
  }

  void grow(std::size_t bytes, std::size_t page_bytes,
            const hipMemAllocationProp &properties,
            const hipMemAccessDesc &access) {
    while (mapped_bytes < bytes) {
      hipMemGenericAllocationHandle_t handle = nullptr;
      check(hipMemCreate(&handle, page_bytes, &properties, 0), "hipMemCreate");
      try {
        check(hipMemMap(static_cast<char *>(address) + mapped_bytes, page_bytes,
                        0, handle, 0),
              "hipMemMap");
        check(hipMemSetAccess(static_cast<char *>(address) + mapped_bytes,
                              page_bytes, &access, 1),
              "hipMemSetAccess");
      } catch (...) {
        (void)hipMemRelease(handle);
        throw;
      }
      handles.push_back(handle);
      mapped_bytes += page_bytes;
    }
  }

  void release_checked(std::size_t page_bytes) {
    for (std::size_t index = handles.size(); index > 0; --index) {
      const std::size_t offset = (index - 1) * page_bytes;
      check(hipMemUnmap(static_cast<char *>(address) + offset, page_bytes),
            "hipMemUnmap");
      check(hipMemRelease(handles[index - 1]), "hipMemRelease");
    }
    handles.clear();
    mapped_bytes = 0;
    if (address != nullptr) {
      check(hipMemAddressFree(address, reserve_bytes), "hipMemAddressFree");
      address = nullptr;
      reserve_bytes = 0;
    }
  }

  void release_noexcept() noexcept {
    if (address != nullptr && !handles.empty()) {
      const std::size_t page_bytes = mapped_bytes / handles.size();
      for (std::size_t index = handles.size(); index > 0; --index) {
        (void)hipMemUnmap(static_cast<char *>(address) +
                              (index - 1) * page_bytes,
                          page_bytes);
        (void)hipMemRelease(handles[index - 1]);
      }
    }
    handles.clear();
    if (address != nullptr) {
      (void)hipMemAddressFree(address, reserve_bytes);
    }
    address = nullptr;
    reserve_bytes = 0;
    mapped_bytes = 0;
  }
};

struct CaseResult {
  std::string mode;
  int mode_id = 0;
  int query_length = 0;
  int kv_length = 0;
  double setup_us = 0.0;
  double grow_us = 0.0;
  double kernel_p50_us = 0.0;
  double kernel_p95_us = 0.0;
  std::size_t logical_bytes = 0;
  std::size_t committed_bytes = 0;
  std::size_t metadata_bytes = 0;
  std::size_t observed_vram_delta_bytes = 0;
  std::string output_file;
  bool nonidentity_block_table = false;
};

CaseResult run_case(const std::string &artifact_dir, const std::string &mode,
                    int mode_id, int query_length, int kv_length,
                    std::size_t page_bytes,
                    const hipMemAllocationProp &properties,
                    const hipMemAccessDesc &access) {
  using Clock = std::chrono::steady_clock;
  const std::size_t plane_capacity_bytes =
      static_cast<std::size_t>(kLogicalCapacity) * kKvHeads * kHeadDim *
      sizeof(__half);
  const std::size_t used_plane_bytes =
      static_cast<std::size_t>(kv_length) * kKvHeads * kHeadDim * sizeof(__half);
  const int blocks = (kv_length + kBlockTokens - 1) / kBlockTokens;
  const std::size_t paged_plane_bytes =
      static_cast<std::size_t>(blocks) * kBlockTokens * kKvHeads * kHeadDim *
      sizeof(__half);
  __half *key = nullptr;
  __half *value = nullptr;
  int *block_table = nullptr;
  VmmPlane key_vmm;
  VmmPlane value_vmm;
  std::size_t free_before = 0;
  std::size_t total_bytes = 0;
  check(hipMemGetInfo(&free_before, &total_bytes), "hipMemGetInfo(before setup)");
  auto setup_start = Clock::now();
  double grow_us = 0.0;
  if (mode_id == 0) {
    check(hipMalloc(&key, plane_capacity_bytes), "hipMalloc(contiguous key)");
    check(hipMalloc(&value, plane_capacity_bytes), "hipMalloc(contiguous value)");
  } else if (mode_id == 1) {
    key_vmm.reserve(plane_capacity_bytes, page_bytes);
    value_vmm.reserve(plane_capacity_bytes, page_bytes);
    const auto grow_start = Clock::now();
    const std::size_t commit = align_up(used_plane_bytes, page_bytes);
    key_vmm.grow(commit, page_bytes, properties, access);
    value_vmm.grow(commit, page_bytes, properties, access);
    check(hipDeviceSynchronize(), "hipDeviceSynchronize(VMM grow)");
    grow_us = std::chrono::duration<double, std::micro>(Clock::now() - grow_start)
                  .count();
    key = static_cast<__half *>(key_vmm.address);
    value = static_cast<__half *>(value_vmm.address);
  } else {
    check(hipMalloc(&key, paged_plane_bytes), "hipMalloc(paged key)");
    check(hipMalloc(&value, paged_plane_bytes), "hipMalloc(paged value)");
    std::vector<int> host_table(static_cast<std::size_t>(blocks));
    for (int index = 0; index < blocks; ++index) {
      host_table[static_cast<std::size_t>(index)] = blocks - 1 - index;
    }
    check(hipMalloc(&block_table, host_table.size() * sizeof(int)),
          "hipMalloc(block table)");
    check(hipMemcpy(block_table, host_table.data(),
                    host_table.size() * sizeof(int), hipMemcpyHostToDevice),
          "hipMemcpy(block table)");
  }
  check(hipDeviceSynchronize(), "hipDeviceSynchronize(setup)");
  std::size_t free_after = 0;
  check(hipMemGetInfo(&free_after, &total_bytes), "hipMemGetInfo(after setup)");
  const double setup_us =
      std::chrono::duration<double, std::micro>(Clock::now() - setup_start)
          .count();

  const std::size_t query_elements =
      static_cast<std::size_t>(query_length) * kQHeads * kHeadDim;
  const std::size_t kv_elements =
      static_cast<std::size_t>(kv_length) * kKvHeads * kHeadDim;
  std::uint16_t *query = nullptr;
  std::uint16_t *output = nullptr;
  check(hipMalloc(&query, query_elements * sizeof(std::uint16_t)),
        "hipMalloc(query)");
  check(hipMalloc(&output, query_elements * sizeof(std::uint16_t)),
        "hipMalloc(output)");
  fill_query<<<(query_elements + 255) / 256, 256>>>(query, query_elements);
  fill_kv<<<(kv_elements + 255) / 256, 256>>>(key, value, kv_length, mode_id,
                                              block_table);
  check(hipGetLastError(), "input initialization launch");
  check(hipDeviceSynchronize(), "input initialization synchronize");

  const dim3 grid(static_cast<unsigned int>(query_length * kQHeads));
  const dim3 threads(kHeadDim);
  for (int iteration = 0; iteration < kWarmups; ++iteration) {
    sllm_a1_online_attention_proxy_v1<<<grid, threads>>>(
        query, key, value, output, query_length, kv_length, mode_id, block_table);
  }
  check(hipGetLastError(), "attention warmup launch");
  check(hipDeviceSynchronize(), "attention warmup synchronize");

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  check(hipEventCreate(&start), "hipEventCreate(start)");
  check(hipEventCreate(&stop), "hipEventCreate(stop)");
  std::vector<float> samples;
  samples.reserve(kMeasurements);
  for (int iteration = 0; iteration < kMeasurements; ++iteration) {
    check(hipEventRecord(start), "hipEventRecord(start)");
    sllm_a1_online_attention_proxy_v1<<<grid, threads>>>(
        query, key, value, output, query_length, kv_length, mode_id, block_table);
    check(hipEventRecord(stop), "hipEventRecord(stop)");
    check(hipEventSynchronize(stop), "hipEventSynchronize(stop)");
    float elapsed_ms = 0.0F;
    check(hipEventElapsedTime(&elapsed_ms, start, stop),
          "hipEventElapsedTime");
    samples.push_back(elapsed_ms);
  }
  check(hipEventDestroy(start), "hipEventDestroy(start)");
  check(hipEventDestroy(stop), "hipEventDestroy(stop)");

  std::vector<std::uint16_t> host_output(query_elements);
  check(hipMemcpy(host_output.data(), output,
                  host_output.size() * sizeof(std::uint16_t),
                  hipMemcpyDeviceToHost),
        "hipMemcpy(output)");
  const std::string filename = mode + "-q" + std::to_string(query_length) +
                               "-k" + std::to_string(kv_length) + ".bf16";
  std::ofstream stream(artifact_dir + "/" + filename,
                       std::ios::out | std::ios::binary | std::ios::trunc);
  if (!stream) {
    throw std::runtime_error("cannot create output artifact");
  }
  stream.write(reinterpret_cast<const char *>(host_output.data()),
               static_cast<std::streamsize>(host_output.size() *
                                            sizeof(std::uint16_t)));
  if (!stream) {
    throw std::runtime_error("cannot write output artifact");
  }
  stream.close();

  check(hipFree(query), "hipFree(query)");
  check(hipFree(output), "hipFree(output)");
  if (mode_id == 1) {
    key_vmm.release_checked(page_bytes);
    value_vmm.release_checked(page_bytes);
  } else {
    check(hipFree(key), "hipFree(key)");
    check(hipFree(value), "hipFree(value)");
  }
  if (block_table != nullptr) {
    check(hipFree(block_table), "hipFree(block table)");
  }

  CaseResult result;
  result.mode = mode;
  result.mode_id = mode_id;
  result.query_length = query_length;
  result.kv_length = kv_length;
  result.setup_us = setup_us;
  result.grow_us = grow_us;
  result.kernel_p50_us = percentile(samples, 0.50);
  result.kernel_p95_us = percentile(samples, 0.95);
  result.logical_bytes = plane_capacity_bytes * 2;
  result.committed_bytes =
      mode_id == 0 ? plane_capacity_bytes * 2
                   : (mode_id == 1 ? align_up(used_plane_bytes, page_bytes) * 2
                                   : paged_plane_bytes * 2);
  result.metadata_bytes = mode_id == 2 ? static_cast<std::size_t>(blocks) * 4 : 0;
  result.observed_vram_delta_bytes =
      free_before > free_after ? free_before - free_after : 0;
  result.output_file = filename;
  result.nonidentity_block_table = mode_id == 2 && blocks > 1;
  return result;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3) {
      throw std::runtime_error(
          "usage: vattention-a1-compare <expected-gfx-target> <artifact-dir>");
    }
    const std::string expected_target(argv[1]);
    const std::string artifact_dir(argv[2]);
    if (expected_target != "gfx1030" && expected_target != "gfx1201") {
      throw std::runtime_error("expected target must be gfx1030 or gfx1201");
    }
    int device_count = 0;
    check(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
    if (device_count != 1) {
      throw std::runtime_error("comparison requires exactly one visible HIP device");
    }
    check(hipSetDevice(0), "hipSetDevice");
    check(hipFree(nullptr), "hipFree(nullptr)");
    hipDeviceProp_t device_properties{};
    check(hipGetDeviceProperties(&device_properties, 0),
          "hipGetDeviceProperties");
    if (std::string(device_properties.gcnArchName) != expected_target) {
      throw std::runtime_error("visible device target does not match expected target");
    }
    int vmm_supported = 0;
    check(hipDeviceGetAttribute(
              &vmm_supported,
              hipDeviceAttributeVirtualMemoryManagementSupported, 0),
          "hipDeviceGetAttribute(VMM)");
    if (vmm_supported != 1) {
      throw std::runtime_error("VMM is not supported by the visible device");
    }
    hipMemAllocationProp properties{};
    properties.type = hipMemAllocationTypePinned;
    properties.location.type = hipMemLocationTypeDevice;
    properties.location.id = 0;
    std::size_t minimum = 0;
    std::size_t recommended = 0;
    check(hipMemGetAllocationGranularity(
              &minimum, &properties, hipMemAllocationGranularityMinimum),
          "hipMemGetAllocationGranularity(minimum)");
    check(hipMemGetAllocationGranularity(
              &recommended, &properties, hipMemAllocationGranularityRecommended),
          "hipMemGetAllocationGranularity(recommended)");
    hipMemAccessDesc access{};
    access.location = properties.location;
    access.flags = hipMemAccessFlagsProtReadWrite;

    std::vector<CaseResult> results;
    for (const auto &mode : std::vector<std::pair<std::string, int>>{
             {"contiguous", 0}, {"vattention", 1}, {"paged", 2}}) {
      for (int query_length : kQueryLengths) {
        for (int kv_length : kKvLengths) {
          results.push_back(run_case(artifact_dir, mode.first, mode.second,
                                     query_length, kv_length, recommended,
                                     properties, access));
        }
      }
    }
    check(hipDeviceSynchronize(), "final synchronize");

    std::ostringstream json;
    json << std::fixed << std::setprecision(3);
    json << "{\"protocol\":\"sllm-vattention-a1-compare-v1\","
         << "\"state\":\"PASS\",\"device\":{\"logical_index\":0,"
         << "\"product\":\"" << json_escape(device_properties.name) << "\","
         << "\"target\":\"" << json_escape(device_properties.gcnArchName)
         << "\",\"bdf\":\"" << format_bdf(device_properties)
         << "\",\"vmm_supported\":true},"
         << "\"shape\":{\"q_heads\":16,\"kv_heads\":4,\"head_dim\":256,"
         << "\"logical_capacity\":4096,\"paged_block_tokens\":256,"
         << "\"query_lengths\":[1,37],"
         << "\"kv_lengths\":[255,256,257,1023,1024,1025]},"
         << "\"algorithm\":{\"class\":\"FA2-style tiled online-softmax proxy\","
         << "\"kernel_symbol\":\"" << kKernelSymbol << "\","
         << "\"contiguous_and_vattention_same_kernel\":true,"
         << "\"kv_layout\":\"token-major\",\"causal_alignment\":\"bottom-right\"},"
         << "\"vmm\":{\"minimum_page_bytes\":" << minimum
         << ",\"recommended_page_bytes\":" << recommended
         << ",\"selected_page_bytes\":" << recommended << "},"
         << "\"warmup_iterations\":" << kWarmups
         << ",\"measured_iterations\":" << kMeasurements << ",\"results\":[";
    for (std::size_t index = 0; index < results.size(); ++index) {
      const auto &result = results[index];
      if (index != 0) {
        json << ',';
      }
      json << "{\"mode\":\"" << result.mode << "\",\"mode_id\":"
           << result.mode_id << ",\"query_length\":" << result.query_length
           << ",\"kv_length\":" << result.kv_length
           << ",\"setup_us\":" << result.setup_us
           << ",\"grow_us\":" << result.grow_us
           << ",\"kernel_p50_us\":" << result.kernel_p50_us
           << ",\"kernel_p95_us\":" << result.kernel_p95_us
           << ",\"logical_bytes\":" << result.logical_bytes
           << ",\"committed_bytes\":" << result.committed_bytes
           << ",\"metadata_bytes\":" << result.metadata_bytes
           << ",\"observed_vram_delta_bytes\":"
           << result.observed_vram_delta_bytes
           << ",\"output_file\":\"" << result.output_file
           << "\",\"nonidentity_block_table\":"
           << (result.nonidentity_block_table ? "true" : "false") << '}';
    }
    json << "],\"fallback_used\":false,\"cleanup_complete\":true}";
    std::cout << json.str() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "vAttention A1 comparison failed: " << error.what() << '\n';
    return 1;
  }
}
