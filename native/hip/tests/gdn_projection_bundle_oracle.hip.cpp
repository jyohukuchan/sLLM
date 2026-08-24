#include "gdn_projection_bundle_kernel_internal.hpp"
#include "matmul_kernel_internal.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
float to_float(uint16_t bits) {
  uint32_t value = static_cast<uint32_t>(bits) << 16U;
  float out = 0.0F;
  std::memcpy(&out, &value, sizeof(out));
  return out;
}
uint16_t to_bf16(float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += UINT32_C(0x7fff) + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>(bits >> 16U);
}
bool check(hipError_t status, const char *where) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", where, hipGetErrorString(status));
    return false;
  }
  return true;
}
} // namespace

int main() {
  hipDeviceProp_t properties{};
  if (!check(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties"))
    return 2;
  if (std::strstr(properties.gcnArchName, "gfx1030") == nullptr) {
    std::fprintf(stderr, "wrong target: %s\n", properties.gcnArchName);
    return 3;
  }
  constexpr uint64_t k = 2560U;
  constexpr uint64_t widths[4] = {8192U, 4096U, 32U, 32U};
  std::vector<uint16_t> activation(k);
  for (uint64_t i = 0U; i < k; ++i)
    activation[i] = to_bf16(std::sin(static_cast<float>(i) * 0.013F));
  std::vector<std::vector<uint16_t>> weights(4);
  std::vector<std::vector<uint16_t>> expected(4);
  for (uint32_t role = 0U; role != 4U; ++role) {
    weights[role].resize(widths[role] * k);
    expected[role].resize(widths[role]);
    for (uint64_t i = 0U; i != weights[role].size(); ++i)
      weights[role][i] =
          to_bf16(std::cos(static_cast<float>(i + role * 17U) * 0.001F));
    for (uint64_t column = 0U; column != widths[role]; ++column) {
      float sum = 0.0F;
      for (uint64_t inner = 0U; inner != k; ++inner)
        sum += to_float(activation[inner]) *
               to_float(weights[role][column * k + inner]);
      expected[role][column] = to_bf16(sum);
    }
  }
  uint16_t *activation_device = nullptr;
  uint16_t *weights_device[4] = {};
  uint16_t *outputs_device[4] = {};
  uint16_t *baseline_outputs_device[4] = {};
  std::vector<uint16_t> actual[4];
  if (!check(
          hipMalloc(&activation_device, activation.size() * sizeof(uint16_t)),
          "hipMalloc activation"))
    return 2;
  if (!check(hipMemcpy(activation_device, activation.data(),
                       activation.size() * sizeof(uint16_t),
                       hipMemcpyHostToDevice),
             "copy activation"))
    return 2;
  for (uint32_t role = 0U; role != 4U; ++role) {
    if (!check(hipMalloc(&weights_device[role],
                         weights[role].size() * sizeof(uint16_t)),
               "hipMalloc weight"))
      return 2;
    if (!check(
            hipMalloc(&outputs_device[role], widths[role] * sizeof(uint16_t)),
            "hipMalloc output"))
      return 2;
    if (!check(hipMalloc(&baseline_outputs_device[role],
                         widths[role] * sizeof(uint16_t)),
               "hipMalloc baseline output"))
      return 2;
    if (!check(hipMemcpy(weights_device[role], weights[role].data(),
                         weights[role].size() * sizeof(uint16_t),
                         hipMemcpyHostToDevice),
               "copy weight"))
      return 2;
    actual[role].resize(widths[role]);
  }
  hipError_t launch_status = sllm_gdn_projection_bundle_kernel::launch(
      activation_device, weights_device[0], weights_device[1],
      weights_device[2], weights_device[3], outputs_device[0],
      outputs_device[1], outputs_device[2], outputs_device[3], k, nullptr);
  if (!check(launch_status, "bundle launch") ||
      !check(hipDeviceSynchronize(), "bundle synchronize"))
    return 2;
  for (uint32_t role = 0U; role != 4U; ++role)
    if (!check(hipMemcpy(actual[role].data(), outputs_device[role],
                         widths[role] * sizeof(uint16_t),
                         hipMemcpyDeviceToHost),
               "copy output"))
      return 2;
  for (uint32_t role = 0U; role != 4U; ++role) {
    launch_status = sllm_matmul_kernel::launch(
        activation_device, weights_device[role], baseline_outputs_device[role],
        1U, k, widths[role], sllm_matmul_kernel::KernelVariant::DecodeReduction,
        nullptr);
    if (!check(launch_status, "baseline launch") ||
        !check(hipDeviceSynchronize(), "baseline synchronize"))
      return 2;
    if (!check(hipMemcpy(expected[role].data(), baseline_outputs_device[role],
                         widths[role] * sizeof(uint16_t),
                         hipMemcpyDeviceToHost),
               "copy baseline"))
      return 2;
  }
  uint64_t mismatches = 0U;
  for (uint32_t role = 0U; role != 4U; ++role)
    for (uint64_t i = 0U; i != widths[role]; ++i) {
      if (actual[role][i] != expected[role][i] && mismatches < 8U)
        std::printf("mismatch role=%u col=%llu got=%04x expected=%04x\n", role,
                    static_cast<unsigned long long>(i), actual[role][i],
                    expected[role][i]);
      mismatches += actual[role][i] != expected[role][i];
    }
  constexpr uint32_t repetitions = 20U;
  hipEvent_t bundle_start = nullptr, bundle_stop = nullptr;
  hipEvent_t baseline_start = nullptr, baseline_stop = nullptr;
  if (!check(hipEventCreate(&bundle_start), "bundle start event") ||
      !check(hipEventCreate(&bundle_stop), "bundle stop event") ||
      !check(hipEventCreate(&baseline_start), "baseline start event") ||
      !check(hipEventCreate(&baseline_stop), "baseline stop event"))
    return 2;
  if (!check(hipEventRecord(bundle_start, nullptr), "bundle start record"))
    return 2;
  for (uint32_t iteration = 0U; iteration != repetitions; ++iteration) {
    launch_status = sllm_gdn_projection_bundle_kernel::launch(
        activation_device, weights_device[0], weights_device[1],
        weights_device[2], weights_device[3], outputs_device[0],
        outputs_device[1], outputs_device[2], outputs_device[3], k, nullptr);
    if (!check(launch_status, "bundle timing launch"))
      return 2;
  }
  if (!check(hipEventRecord(bundle_stop, nullptr), "bundle stop record") ||
      !check(hipEventSynchronize(bundle_stop), "bundle timing synchronize"))
    return 2;
  if (!check(hipEventRecord(baseline_start, nullptr), "baseline start record"))
    return 2;
  for (uint32_t role = 0U; role != 4U; ++role) {
    for (uint32_t iteration = 0U; iteration != repetitions; ++iteration) {
      launch_status = sllm_matmul_kernel::launch(
          activation_device, weights_device[role],
          baseline_outputs_device[role], 1U, k, widths[role],
          sllm_matmul_kernel::KernelVariant::DecodeReduction, nullptr);
      if (!check(launch_status, "baseline timing launch"))
        return 2;
    }
  }
  if (!check(hipEventRecord(baseline_stop, nullptr), "baseline stop record") ||
      !check(hipEventSynchronize(baseline_stop), "baseline timing synchronize"))
    return 2;
  float bundle_ms = 0.0F;
  float baseline_ms = 0.0F;
  if (!check(hipEventElapsedTime(&bundle_ms, bundle_start, bundle_stop),
             "bundle elapsed") ||
      !check(hipEventElapsedTime(&baseline_ms, baseline_start, baseline_stop),
             "baseline elapsed"))
    return 2;
  bool cleanup_ok = true;
  for (uint32_t role = 0U; role != 4U; ++role) {
    cleanup_ok =
        cleanup_ok && check(hipFree(weights_device[role]), "free weight");
    cleanup_ok =
        cleanup_ok && check(hipFree(outputs_device[role]), "free output");
    cleanup_ok = cleanup_ok && check(hipFree(baseline_outputs_device[role]),
                                     "free baseline output");
  }
  cleanup_ok =
      cleanup_ok && check(hipFree(activation_device), "free activation");
  cleanup_ok = cleanup_ok &&
               check(hipEventDestroy(bundle_start), "destroy bundle start");
  cleanup_ok =
      cleanup_ok && check(hipEventDestroy(bundle_stop), "destroy bundle stop");
  cleanup_ok = cleanup_ok &&
               check(hipEventDestroy(baseline_start), "destroy baseline start");
  cleanup_ok = cleanup_ok &&
               check(hipEventDestroy(baseline_stop), "destroy baseline stop");
  std::printf("target=%s mismatches=%llu dispatches=1 grid=12352 fallback=0 "
              "cleanup=%u reps=%u bundle_ms=%.3f baseline_ms=%.3f\n",
              properties.gcnArchName,
              static_cast<unsigned long long>(mismatches), cleanup_ok ? 0U : 1U,
              repetitions, bundle_ms, baseline_ms);
  return mismatches == 0U && cleanup_ok ? 0 : 1;
}
