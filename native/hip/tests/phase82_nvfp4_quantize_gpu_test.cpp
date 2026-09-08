// Phase82 bounded NVFP4 activation-quantizer correctness runner.
//
// This binary links against the real native HIP archive and calls only the
// existing launch_nvfp4_quantize entry point.  It intentionally keeps the
// numerical oracle in scripts/dev/phase82_nvfp4_quantize_oracle.py, so this
// runner does not duplicate the codec implementation under test.
//
// Build shape (the archive and HIP flags are supplied by the caller):
//   hipcc -std=c++17 --offload-arch=gfx1030 -D__HIP_PLATFORM_AMD__=1 \
//     -I native/hip/src -I /opt/rocm/include \
//     native/hip/tests/phase82_nvfp4_quantize_gpu_test.cpp \
//     /path/to/libsllm_hip_stub.a \
//     /opt/rocm/lib/libamdhip64.so /opt/rocm/lib/libhipblas.so \
//     /opt/rocm/lib/libhipblaslt.so /opt/rocm/lib/librocblas.so \
//     -o phase82_nvfp4_quantize_gpu_test
//
// Compile note: matmul_kernel_internal.hpp declares launch_nvfp4_quantize
// unconditionally; this runner does not need
// -DSLLM_ENABLE_HIP_RUNTIME=1 or -DSLLM_ENABLE_PUBLIC_HIP_RUNTIME=1.
// Those macros belong to the archive's CMake build. Build the archive with
// its exact target/runtime settings and carry its recorded ROCm link
// libraries to the final link. The public archive's CMake target records
// amdhip64, hipblas, hipblaslt, and rocblas; retain that complete set even
// though this runner calls only the quantizer entry point.
//
// Usage:
//   phase82_nvfp4_quantize_gpu_test <gfx1030|gfx1201> \
//     <flag0|wave8|default|invalid|forcebaseline> <input.bin> <output.bin>

#include "matmul_kernel_internal.hpp"

#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::array<char, 8> kInputMagic = {'P', '8', '2', 'N',
                                             'Q', 'Z', '1', '\0'};
constexpr std::array<char, 8> kOutputMagic = {'P', '8', '2', 'N',
                                              'Q', 'R', '1', '\0'};
constexpr std::size_t kCanaryBytes = 32U;
constexpr uint8_t kCanary = UINT8_C(0xa5);

struct FixtureCase final {
  uint64_t m = 0U;
  uint64_t k = 0U;
  float input_tensor_scale = 0.0F;
  std::vector<uint16_t> activation;
};

bool read_bytes(std::istream &input, void *const destination,
                const std::size_t bytes) {
  input.read(static_cast<char *>(destination),
             static_cast<std::streamsize>(bytes));
  return input.good() || input.gcount() == static_cast<std::streamsize>(bytes);
}

template <typename T> bool read_scalar(std::istream &input, T *const value) {
  return read_bytes(input, value, sizeof(T));
}

bool write_bytes(std::ostream &output, const void *const source,
                 const std::size_t bytes) {
  output.write(static_cast<const char *>(source),
               static_cast<std::streamsize>(bytes));
  return output.good();
}

template <typename T> bool write_scalar(std::ostream &output, const T value) {
  return write_bytes(output, &value, sizeof(T));
}

bool exact_architecture(const char *const actual,
                        const std::string_view target) {
  if (actual == nullptr) {
    return false;
  }
  const std::string_view value(actual);
  return value == target || (value.size() > target.size() &&
                             value.compare(0U, target.size(), target) == 0 &&
                             value[target.size()] == ':');
}

bool checked_product(const uint64_t left, const uint64_t right,
                     uint64_t *const result) {
  if (left != 0U && right > std::numeric_limits<uint64_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

bool set_mode_environment(const std::string_view mode) {
  unsetenv("SLLM_NVFP4_ACTIVATION_QUANTIZE_WAVE8");
  unsetenv("SLLM_NVFP4_FORCE_BASELINE");
  unsetenv("SLLM_NVFP4_W4A4_FORCE_BASELINE");
  if (mode == "default") {
    return true;
  }
  if (mode == "flag0") {
    return setenv("SLLM_NVFP4_ACTIVATION_QUANTIZE_WAVE8", "0", 1) == 0;
  }
  if (mode == "wave8") {
    return setenv("SLLM_NVFP4_ACTIVATION_QUANTIZE_WAVE8", "1", 1) == 0;
  }
  if (mode == "invalid") {
    return setenv("SLLM_NVFP4_ACTIVATION_QUANTIZE_WAVE8", "invalid", 1) == 0;
  }
  if (mode == "forcebaseline") {
    return setenv("SLLM_NVFP4_ACTIVATION_QUANTIZE_WAVE8", "1", 1) == 0 &&
           setenv("SLLM_NVFP4_FORCE_BASELINE", "1", 1) == 0;
  }
  return false;
}

bool read_fixture(std::istream &input, std::vector<FixtureCase> *const cases) {
  std::array<char, 8> magic{};
  uint32_t version = 0U;
  uint32_t count = 0U;
  if (!read_bytes(input, magic.data(), magic.size()) ||
      !read_scalar(input, &version) || !read_scalar(input, &count) ||
      magic != kInputMagic || version != 1U || count == 0U || count > 1024U) {
    return false;
  }
  cases->clear();
  cases->reserve(count);
  for (uint32_t index = 0U; index != count; ++index) {
    FixtureCase item;
    uint64_t activation_count = 0U;
    if (!read_scalar(input, &item.m) || !read_scalar(input, &item.k) ||
        !read_scalar(input, &item.input_tensor_scale) || item.m == 0U ||
        item.k == 0U || !std::isfinite(item.input_tensor_scale) ||
        !checked_product(item.m, item.k, &activation_count) ||
        activation_count > static_cast<uint64_t>(SIZE_MAX / sizeof(uint16_t))) {
      return false;
    }
    if (!read_scalar(input, &activation_count) ||
        activation_count != item.m * item.k) {
      return false;
    }
    item.activation.resize(static_cast<std::size_t>(activation_count));
    if (!read_bytes(input, item.activation.data(),
                    item.activation.size() * sizeof(uint16_t))) {
      return false;
    }
    cases->push_back(std::move(item));
  }
  return true;
}

bool guard_is_intact(const std::vector<uint8_t> &bytes,
                     const std::size_t payload_offset,
                     const std::size_t payload_bytes) {
  if (payload_offset < kCanaryBytes ||
      payload_offset + payload_bytes + kCanaryBytes != bytes.size()) {
    return false;
  }
  for (std::size_t index = 0U; index != payload_offset; ++index) {
    if (bytes[index] != kCanary) {
      return false;
    }
  }
  for (std::size_t index = payload_offset + payload_bytes;
       index != bytes.size(); ++index) {
    if (bytes[index] != kCanary) {
      return false;
    }
  }
  return true;
}

int run_case(const FixtureCase &item, hipStream_t stream,
             std::ofstream *const output) {
  uint64_t packed_bytes_u64 = 0U;
  uint64_t blocks_per_row = (item.k + UINT64_C(15)) / UINT64_C(16);
  uint64_t scale_bytes_u64 = 0U;
  uint64_t activation_bytes_u64 = 0U;
  if (!checked_product(item.m, (item.k + 1U) / 2U, &packed_bytes_u64) ||
      !checked_product(item.m, blocks_per_row, &scale_bytes_u64) ||
      !checked_product(item.m, item.k, &activation_bytes_u64) ||
      packed_bytes_u64 > SIZE_MAX - 2U * kCanaryBytes ||
      scale_bytes_u64 > SIZE_MAX - 2U * kCanaryBytes ||
      activation_bytes_u64 > SIZE_MAX / sizeof(uint16_t)) {
    std::cerr << "fixture size overflow\n";
    return 2;
  }
  const std::size_t packed_bytes = static_cast<std::size_t>(packed_bytes_u64);
  const std::size_t scale_bytes = static_cast<std::size_t>(scale_bytes_u64);
  const std::size_t packed_total = packed_bytes + 2U * kCanaryBytes;
  const std::size_t scale_total = scale_bytes + 2U * kCanaryBytes;

  uint16_t *device_activation = nullptr;
  float *device_global = nullptr;
  uint8_t *device_packed_raw = nullptr;
  uint8_t *device_scales_raw = nullptr;
  auto cleanup = [&]() {
    bool success = true;
    if (device_scales_raw != nullptr) {
      success = hipFree(device_scales_raw) == hipSuccess && success;
    }
    if (device_packed_raw != nullptr) {
      success = hipFree(device_packed_raw) == hipSuccess && success;
    }
    if (device_global != nullptr) {
      success = hipFree(device_global) == hipSuccess && success;
    }
    if (device_activation != nullptr) {
      success = hipFree(device_activation) == hipSuccess && success;
    }
    device_scales_raw = nullptr;
    device_packed_raw = nullptr;
    device_global = nullptr;
    device_activation = nullptr;
    return success;
  };
  const auto fail = [&]() {
    if (!cleanup()) {
      std::cerr << "device cleanup failed\n";
    }
    return 2;
  };
  hipError_t status = hipMalloc(reinterpret_cast<void **>(&device_activation),
                                item.activation.size() * sizeof(uint16_t));
  if (status != hipSuccess) {
    std::cerr << "hipMalloc activation failed: " << hipGetErrorName(status)
              << " (" << hipGetErrorString(status) << ")\n";
    return fail();
  }
  status = hipMalloc(reinterpret_cast<void **>(&device_global), sizeof(float));
  if (status != hipSuccess) {
    std::cerr << "hipMalloc global failed: " << hipGetErrorName(status) << "\n";
    return fail();
  }
  status =
      hipMalloc(reinterpret_cast<void **>(&device_packed_raw), packed_total);
  if (status != hipSuccess) {
    std::cerr << "hipMalloc packed failed: " << hipGetErrorName(status) << "\n";
    return fail();
  }
  status =
      hipMalloc(reinterpret_cast<void **>(&device_scales_raw), scale_total);
  if (status != hipSuccess) {
    std::cerr << "hipMalloc scales failed: " << hipGetErrorName(status) << "\n";
    return fail();
  }
  const auto check = [&](const hipError_t value, const char *const what) {
    if (value == hipSuccess) {
      return true;
    }
    std::cerr << what << " failed: " << hipGetErrorName(value) << " ("
              << hipGetErrorString(value) << ")\n";
    return false;
  };
  if (!check(hipMemcpyAsync(device_activation, item.activation.data(),
                            item.activation.size() * sizeof(uint16_t),
                            hipMemcpyHostToDevice, stream),
             "activation upload") ||
      !check(hipMemcpyAsync(device_global, &item.input_tensor_scale,
                            sizeof(float), hipMemcpyHostToDevice, stream),
             "global upload") ||
      !check(hipMemsetAsync(device_packed_raw, kCanary, packed_total, stream),
             "packed canary") ||
      !check(hipMemsetAsync(device_scales_raw, kCanary, scale_total, stream),
             "scale canary")) {
    return fail();
  }

  status = sllm_matmul_kernel::launch_nvfp4_quantize(
      device_activation, device_packed_raw + kCanaryBytes,
      device_scales_raw + kCanaryBytes, device_global, item.m, item.k, stream);
  if (!check(status, "launch_nvfp4_quantize") ||
      !check(hipStreamSynchronize(stream), "quantizer synchronize")) {
    return fail();
  }
  std::vector<uint8_t> packed_host(packed_total, kCanary);
  std::vector<uint8_t> scales_host(scale_total, kCanary);
  if (!check(hipMemcpy(packed_host.data(), device_packed_raw, packed_total,
                       hipMemcpyDeviceToHost),
             "packed download") ||
      !check(hipMemcpy(scales_host.data(), device_scales_raw, scale_total,
                       hipMemcpyDeviceToHost),
             "scale download") ||
      !guard_is_intact(packed_host, kCanaryBytes, packed_bytes) ||
      !guard_is_intact(scales_host, kCanaryBytes, scale_bytes)) {
    std::cerr << "output canary was modified\n";
    return fail();
  }
  if (!write_scalar(*output, item.m) || !write_scalar(*output, item.k) ||
      !write_scalar(*output, packed_bytes_u64) ||
      !write_scalar(*output, scale_bytes_u64) ||
      !write_bytes(*output, packed_host.data() + kCanaryBytes, packed_bytes) ||
      !write_bytes(*output, scales_host.data() + kCanaryBytes, scale_bytes)) {
    std::cerr << "result write failed\n";
    return fail();
  }
  if (!cleanup()) {
    std::cerr << "device cleanup failed\n";
    return 2;
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 5 ||
      (std::string_view(argv[1]) != "gfx1030" &&
       std::string_view(argv[1]) != "gfx1201") ||
      !set_mode_environment(argv[2])) {
    std::cerr << "usage: phase82_nvfp4_quantize_gpu_test "
                 "<gfx1030|gfx1201> "
                 "<flag0|wave8|default|invalid|forcebaseline> "
                 "<input.bin> <output.bin>\n";
    return 64;
  }
  const std::string_view target(argv[1]);
  const std::string_view mode(argv[2]);
  std::ifstream input(argv[3], std::ios::binary);
  std::ofstream output(argv[4], std::ios::binary | std::ios::trunc);
  if (!input || !output) {
    std::cerr << "input/output file open failed\n";
    return 2;
  }
  std::vector<FixtureCase> cases;
  if (!read_fixture(input, &cases)) {
    std::cerr << "invalid phase82 quantizer fixture\n";
    return 2;
  }

  int device = 0;
  int device_count = 0;
  hipDeviceProp_t properties{};
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0 ||
      hipSetDevice(device) != hipSuccess ||
      hipGetDeviceProperties(&properties, device) != hipSuccess ||
      !exact_architecture(properties.gcnArchName, target)) {
    std::cerr << "exact target " << target << " is required; got "
              << properties.gcnArchName << "\n";
    return 2;
  }
  hipStream_t stream = nullptr;
  if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess) {
    std::cerr << "hipStreamCreateWithFlags failed\n";
    return 2;
  }
  if (!write_bytes(output, kOutputMagic.data(), kOutputMagic.size()) ||
      !write_scalar(output, uint32_t{1}) ||
      !write_scalar(output, static_cast<uint32_t>(cases.size()))) {
    std::cerr << "result header write failed\n";
    (void)hipStreamDestroy(stream);
    return 2;
  }
  std::array<char, 8> target_field{};
  std::memcpy(target_field.data(), target.data(), target.size());
  std::array<char, 16> mode_field{};
  std::memcpy(mode_field.data(), mode.data(), mode.size());
  if (!write_bytes(output, target_field.data(), target_field.size()) ||
      !write_bytes(output, mode_field.data(), mode_field.size())) {
    std::cerr << "result identity write failed\n";
    (void)hipStreamDestroy(stream);
    return 2;
  }
  for (const FixtureCase &item : cases) {
    if (run_case(item, stream, &output) != 0) {
      (void)hipStreamDestroy(stream);
      return 2;
    }
  }
  if (!output.good() || hipStreamDestroy(stream) != hipSuccess) {
    std::cerr << "result close failed\n";
    return 2;
  }
  std::cout << "phase82_nvfp4_quantize target=" << target << " mode=" << mode
            << " cases=" << cases.size() << " status=ok\n";
  return 0;
}
