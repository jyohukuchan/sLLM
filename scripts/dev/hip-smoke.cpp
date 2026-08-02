#include <hip/hip_runtime.h>

#include <cassert>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#ifndef ULLM_HIP_ARCHITECTURES
#error "ULLM_HIP_ARCHITECTURES must describe the explicitly compiled targets"
#endif

namespace {

__global__ void add_one(int *value) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *value += 1;
  }
}

bool check_hip(hipError_t status, const char *expression, const char *file,
               int line) {
  if (status == hipSuccess) {
    return true;
  }
  std::cerr << "HIP error: expression=" << expression << " file=" << file << ':'
            << line << " status=" << static_cast<int>(status)
            << " message=" << hipGetErrorString(status) << '\n';
  return false;
}

#define ULLM_HIP_CHECK(expression)                                             \
  check_hip((expression), #expression, __FILE__, __LINE__)

std::string base_architecture(const std::string &architecture) {
  const auto feature_separator = architecture.find(':');
  return architecture.substr(0, feature_separator);
}

std::unordered_set<std::string> compiled_architectures() {
  std::unordered_set<std::string> architectures;
  std::string encoded = ULLM_HIP_ARCHITECTURES;
  std::size_t start = 0;
  while (start <= encoded.size()) {
    const auto comma = encoded.find(',', start);
    const auto target = encoded.substr(start, comma - start);
    if (!target.empty()) {
      architectures.insert(base_architecture(target));
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return architectures;
}

struct DeviceRecord {
  int index;
  hipDeviceProp_t properties;
};

} // namespace

int main() {
  int runtime_version = 0;
  if (!ULLM_HIP_CHECK(hipRuntimeGetVersion(&runtime_version))) {
    return 1;
  }
  std::cout << "hip_runtime_version=" << runtime_version << '\n';
  std::cout << "compiled_targets=" << ULLM_HIP_ARCHITECTURES << '\n';

  int device_count = 0;
  if (!ULLM_HIP_CHECK(hipGetDeviceCount(&device_count))) {
    return 1;
  }
  if (device_count <= 0) {
    std::cerr << "HIP smoke requires at least one visible GPU; found "
              << device_count << '\n';
    return 1;
  }

  const auto targets = compiled_architectures();
  std::vector<DeviceRecord> devices;
  devices.reserve(static_cast<std::size_t>(device_count));
  for (int device = 0; device < device_count; ++device) {
    hipDeviceProp_t properties{};
    if (!ULLM_HIP_CHECK(hipGetDeviceProperties(&properties, device))) {
      return 1;
    }
    const std::string exact_arch = properties.gcnArchName;
    if (exact_arch.empty()) {
      std::cerr << "device=" << device
                << " returned an empty exact architecture\n";
      return 1;
    }
    if (targets.find(base_architecture(exact_arch)) == targets.end()) {
      std::cerr << "device=" << device << " name=\"" << properties.name
                << "\" exact_arch=" << exact_arch
                << " has no matching explicitly compiled target\n";
      return 1;
    }
    devices.push_back(DeviceRecord{device, properties});
  }

  for (const auto &device : devices) {
    if (!ULLM_HIP_CHECK(hipSetDevice(device.index))) {
      return 1;
    }

    int input = 41;
    int result = 0;
    int *device_value = nullptr;
    if (!ULLM_HIP_CHECK(
            hipMalloc(reinterpret_cast<void **>(&device_value), sizeof(int)))) {
      return 1;
    }
    if (!ULLM_HIP_CHECK(hipMemcpy(device_value, &input, sizeof(int),
                                  hipMemcpyHostToDevice))) {
      if (!ULLM_HIP_CHECK(hipFree(device_value))) {
        return 1;
      }
      return 1;
    }

    add_one<<<1, 1>>>(device_value);
    if (!ULLM_HIP_CHECK(hipGetLastError()) ||
        !ULLM_HIP_CHECK(hipDeviceSynchronize()) ||
        !ULLM_HIP_CHECK(hipMemcpy(&result, device_value, sizeof(int),
                                  hipMemcpyDeviceToHost))) {
      if (!ULLM_HIP_CHECK(hipFree(device_value))) {
        return 1;
      }
      return 1;
    }
    if (!ULLM_HIP_CHECK(hipFree(device_value))) {
      return 1;
    }

    if (result != 42) {
      std::cerr << "device=" << device.index << " name=\""
                << device.properties.name
                << "\" exact_arch=" << device.properties.gcnArchName
                << " result=" << result << " expected=42\n";
      return 1;
    }
    assert(result == 42);
    std::cout << "device=" << device.index << " name=\""
              << device.properties.name
              << "\" exact_arch=" << device.properties.gcnArchName
              << " result=" << result << '\n';
  }

  return 0;
}
