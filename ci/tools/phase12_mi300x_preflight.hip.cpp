// Standalone Phase 12 MI300X identity, capability, and tiny-runtime probe.

#include <hip/hip_runtime.h>

#include <array>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void check(const hipError_t status, const char *operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

std::string uuid_text(const hipUUID &uuid) {
  std::string output(uuid.bytes, sizeof(uuid.bytes));
  for (char &character : output) {
    const auto value = static_cast<unsigned char>(character);
    if (std::isxdigit(value) == 0) {
      throw std::runtime_error("HIP UUID is not 16 ASCII hexadecimal digits");
    }
    character = static_cast<char>(std::tolower(value));
  }
  return output;
}

std::string json_escape(const char *value) {
  std::string output;
  for (const unsigned char character : std::string(value)) {
    if (character == '"' || character == '\\') {
      output.push_back('\\');
      output.push_back(static_cast<char>(character));
    } else if (character >= 0x20U) {
      output.push_back(static_cast<char>(character));
    }
  }
  return output;
}

__global__ void increment(std::uint32_t *value) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    *value += 1U;
  }
}

} // namespace

int main() {
  std::uint32_t *device_value = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  try {
    int runtime_version = 0;
    check(hipRuntimeGetVersion(&runtime_version), "hipRuntimeGetVersion");
    int device_count = 0;
    check(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
    if (device_count != 1) {
      throw std::runtime_error(
          "Phase 12 probe requires exactly one visible GPU");
    }

    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties");
    std::array<char, 32U> bdf{};
    check(hipDeviceGetPCIBusId(bdf.data(), static_cast<int>(bdf.size()), 0),
          "hipDeviceGetPCIBusId");
    hipUUID uuid{};
    check(hipDeviceGetUuid(&uuid, static_cast<hipDevice_t>(0)),
          "hipDeviceGetUuid");
    int vmm_supported = 0;
    check(hipDeviceGetAttribute(
              &vmm_supported,
              hipDeviceAttributeVirtualMemoryManagementSupported, 0),
          "hipDeviceGetAttribute(VMM)");
    std::size_t free_bytes = 0U;
    std::size_t total_bytes = 0U;
    check(hipMemGetInfo(&free_bytes, &total_bytes), "hipMemGetInfo");

    const std::uint32_t input = 41U;
    std::uint32_t output = 0U;
    check(hipMalloc(reinterpret_cast<void **>(&device_value), sizeof(input)),
          "hipMalloc");
    check(hipMemcpy(device_value, &input, sizeof(input), hipMemcpyHostToDevice),
          "hipMemcpy(H2D)");
    check(hipEventCreate(&start), "hipEventCreate(start)");
    check(hipEventCreate(&stop), "hipEventCreate(stop)");
    check(hipEventRecord(start, nullptr), "hipEventRecord(start)");
    increment<<<1, 1>>>(device_value);
    check(hipGetLastError(), "increment launch");
    check(hipEventRecord(stop, nullptr), "hipEventRecord(stop)");
    check(hipEventSynchronize(stop), "hipEventSynchronize(stop)");
    float elapsed_ms = 0.0F;
    check(hipEventElapsedTime(&elapsed_ms, start, stop), "hipEventElapsedTime");
    check(
        hipMemcpy(&output, device_value, sizeof(output), hipMemcpyDeviceToHost),
        "hipMemcpy(D2H)");
    if (output != 42U) {
      throw std::runtime_error("tiny kernel numerical result mismatch");
    }

    check(hipEventDestroy(stop), "hipEventDestroy(stop)");
    stop = nullptr;
    check(hipEventDestroy(start), "hipEventDestroy(start)");
    start = nullptr;
    check(hipFree(device_value), "hipFree");
    device_value = nullptr;

    std::cout << "{\"schema_version\":\"phase12-mi300x-preflight-probe-v1\",";
    std::cout << "\"state\":\"PASS\",\"runtime_version\":" << runtime_version
              << ",\"compiled_target\":\"gfx942\",";
    std::cout << "\"device_count\":1,\"device\":{";
    std::cout << "\"name\":\"" << json_escape(properties.name) << "\",";
    std::cout << "\"gcn_arch_name\":\"" << json_escape(properties.gcnArchName)
              << "\",\"bdf\":\"" << bdf.data() << "\",";
    std::cout << "\"uuid\":\"GPU-" << uuid_text(uuid) << "\",";
    std::cout << "\"wave_size\":" << properties.warpSize
              << ",\"compute_units\":" << properties.multiProcessorCount
              << ",\"total_global_memory_bytes\":" << properties.totalGlobalMem
              << "},";
    std::cout << "\"capabilities\":{\"vmm_supported\":"
              << (vmm_supported != 0 ? "true" : "false") << "},";
    std::cout << "\"memory\":{\"free_bytes\":" << free_bytes
              << ",\"total_bytes\":" << total_bytes << "},";
    std::cout << "\"tiny_runtime\":{\"input\":41,\"output\":42,"
                 "\"event_elapsed_ms\":"
              << elapsed_ms
              << ",\"allocation_count\":1,\"copy_count\":2,"
                 "\"kernel_dispatch_count\":1}}\n";
    return 0;
  } catch (const std::exception &error) {
    if (stop != nullptr) {
      (void)hipEventDestroy(stop);
    }
    if (start != nullptr) {
      (void)hipEventDestroy(start);
    }
    if (device_value != nullptr) {
      (void)hipFree(device_value);
    }
    std::cerr << "Phase 12 MI300X preflight probe: " << error.what() << '\n';
    return 2;
  }
}
