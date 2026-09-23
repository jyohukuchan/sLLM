// Phase 87 stage 7 bounded probe: NVFP4 W4A4 consumer prequant contract.
//
// For M=1 and M=17, compare the same prepared lowp plan through the
// decomposed BF16 -> NVFP4 quantizer path and through
// LOWP_ACTIVATION_PREQUANTIZED. The weight plane and tensor scales are shared
// by both calls, so a mismatch identifies the activation payload/consumer
// contract rather than the producer arithmetic. M=1 is also replayed through
// a native HIP graph to exercise the capture path used by projection packs.

#include <hip/hip_runtime.h>
#include <lowp/lowp.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#define S7_MM_CHECK(call)                                                      \
  do {                                                                         \
    const hipError_t s7_status_ = (call);                                      \
    if (s7_status_ != hipSuccess) {                                            \
      std::cerr << "FAIL: " << #call << " -> "                                 \
                << hipGetErrorString(s7_status_) << '\n';                      \
      return false;                                                            \
    }                                                                          \
  } while (0)

namespace {

struct DeviceBuffer final {
  void *data = nullptr;
  std::size_t bytes = 0U;

  DeviceBuffer() = default;

  bool allocate(const std::size_t size) {
    bytes = size;
    return hipMalloc(&data, size == 0U ? 1U : size) == hipSuccess;
  }
  ~DeviceBuffer() {
    if (data != nullptr) {
      (void)hipFree(data);
    }
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

struct Shape final {
  uint64_t m;
  const char *name;
};

constexpr Shape kShapes[] = {{1U, "m1"}, {17U, "m17"}};
// Qwen3.8 NVFP4 MLP gate/up projection width. Keeping the real N exposes
// shape-specific provider variants that a small synthetic N can miss.
constexpr uint64_t kN = UINT64_C(17408);
constexpr uint64_t kK = UINT64_C(5120);

uint16_t f32_to_bf16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  return static_cast<uint16_t>(
      upper + (lower > UINT32_C(0x8000) ||
               (lower == UINT32_C(0x8000) && (upper & 1U) != 0U)));
}

bool make_plan(const uint64_t m, const lowp_target_t target,
               lowp_matmul_plan_t *const plan) {
  lowp_matmul_request_t request{};
  request.struct_size = sizeof(request);
  request.version = LOWP_ABI_VERSION;
  request.format = LOWP_NVFP4_W4A4;
  request.target = target;
  request.weight_layout = LOWP_ROW_MAJOR_BLOCK_SCALED;
  request.activation_layout = LOWP_ROW_MAJOR_BLOCK_SCALED;
  request.m = m;
  request.n = kN;
  request.k = kK;
  const lowp_status_t status = lowp_matmul_plan(&request, plan);
  if (status != LOWP_SUCCESS || plan->supported == 0U) {
    std::cerr << "FAIL: lowp_matmul_plan M=" << m << " status=" << status
              << " reason=" << (plan->reason == nullptr ? "none" : plan->reason)
              << " variant=" << plan->variant << '\n';
    return false;
  }
  return true;
}

bool run_shape(const Shape &shape, const lowp_target_t target,
               const bool graph_replay) {
  lowp_matmul_plan_t plan{};
  if (!make_plan(shape.m, target, &plan)) {
    return false;
  }
  const uint64_t activation_elements = shape.m * kK;
  std::vector<uint16_t> host_activation(activation_elements);
  for (uint64_t row = 0U; row < shape.m; ++row) {
    for (uint64_t column = 0U; column < kK; ++column) {
      const float value = ((column + row * 3U) % 11U == 0U)
                              ? -2.0F
                              : ((column + row) % 5U == 0U ? 0.5F : 1.0F);
      host_activation[row * kK + column] = f32_to_bf16(value);
    }
  }

  const uint64_t weight_tensor_bytes =
      plan.weight_value_bytes + plan.weight_scale_bytes;
  DeviceBuffer activation;
  DeviceBuffer activation_encoded;
  DeviceBuffer weight;
  DeviceBuffer weight_tensor_scale;
  DeviceBuffer activation_tensor_scale;
  DeviceBuffer output_auto;
  DeviceBuffer output_prequant;
  DeviceBuffer workspace;
  if (!activation.allocate(activation_elements * sizeof(uint16_t)) ||
      !activation_encoded.allocate(plan.activation_value_bytes +
                                   plan.activation_scale_bytes) ||
      !weight.allocate(weight_tensor_bytes) ||
      !weight_tensor_scale.allocate(sizeof(float)) ||
      !activation_tensor_scale.allocate(sizeof(float)) ||
      !output_auto.allocate(plan.output_bytes) ||
      !output_prequant.allocate(plan.output_bytes) ||
      !workspace.allocate(plan.workspace_bytes)) {
    std::cerr << "FAIL: allocation shape=" << shape.name << '\n';
    return false;
  }

  S7_MM_CHECK(hipMemcpy(activation.data, host_activation.data(),
                        host_activation.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice));
  std::vector<uint8_t> host_weight(
      static_cast<std::size_t>(weight_tensor_bytes), UINT8_C(0x22));
  // E4M3FN code 0x38 is 1.0. The low nibble of 0x22 is E2M1 code 2 (1.0),
  // and the high nibble is the same value.
  std::fill(host_weight.begin() +
                static_cast<std::ptrdiff_t>(plan.weight_value_bytes),
            host_weight.end(), UINT8_C(0x38));
  S7_MM_CHECK(hipMemcpy(weight.data, host_weight.data(), host_weight.size(),
                        hipMemcpyHostToDevice));
  const float one = 1.0F;
  S7_MM_CHECK(hipMemcpy(weight_tensor_scale.data, &one, sizeof(one),
                        hipMemcpyHostToDevice));
  S7_MM_CHECK(hipMemcpy(activation_tensor_scale.data, &one, sizeof(one),
                        hipMemcpyHostToDevice));

  if (lowp_quantize_activation(
          &plan, static_cast<const uint16_t *>(activation.data),
          activation_encoded.data,
          static_cast<uint8_t *>(activation_encoded.data) +
              plan.activation_scale_offset,
          static_cast<const float *>(activation_tensor_scale.data),
          nullptr) != LOWP_SUCCESS) {
    std::cerr << "FAIL: activation quantize shape=" << shape.name << '\n';
    return false;
  }

  lowp_matmul_buffers_t buffers{};
  buffers.struct_size = sizeof(buffers);
  buffers.activation = activation.data;
  buffers.weight = weight.data;
  buffers.weight_scales =
      static_cast<const uint8_t *>(weight.data) + plan.weight_value_bytes;
  buffers.weight_tensor_scale =
      static_cast<const float *>(weight_tensor_scale.data);
  buffers.activation_tensor_scale =
      static_cast<const float *>(activation_tensor_scale.data);
  buffers.output = static_cast<uint16_t *>(output_auto.data);
  buffers.workspace = workspace.data;
  buffers.workspace_bytes = plan.workspace_bytes;
  if (lowp_matmul_launch(&plan, &buffers, nullptr) != LOWP_SUCCESS) {
    std::cerr << "FAIL: automatic launch shape=" << shape.name
              << " variant=" << plan.variant << '\n';
    return false;
  }

  buffers.flags = LOWP_ACTIVATION_PREQUANTIZED;
  buffers.activation = activation_encoded.data;
  buffers.activation_scales = static_cast<uint8_t *>(activation_encoded.data) +
                              plan.activation_scale_offset;
  buffers.output = static_cast<uint16_t *>(output_prequant.data);
  if (lowp_matmul_launch(&plan, &buffers, nullptr) != LOWP_SUCCESS) {
    std::cerr << "FAIL: prequant launch shape=" << shape.name
              << " variant=" << plan.variant << '\n';
    return false;
  }
  S7_MM_CHECK(hipDeviceSynchronize());

  std::vector<uint16_t> auto_host(static_cast<std::size_t>(shape.m * kN));
  std::vector<uint16_t> prequant_host(auto_host.size());
  S7_MM_CHECK(hipMemcpy(auto_host.data(), output_auto.data,
                        auto_host.size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost));
  S7_MM_CHECK(hipMemcpy(prequant_host.data(), output_prequant.data,
                        prequant_host.size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost));
  std::size_t first_diff = auto_host.size();
  for (std::size_t index = 0U; index < auto_host.size(); ++index) {
    if (auto_host[index] != prequant_host[index]) {
      first_diff = index;
      break;
    }
  }
  if (first_diff != auto_host.size()) {
    std::cerr << "FAIL: output mismatch shape=" << shape.name
              << " variant=" << plan.variant << " first_index=" << first_diff
              << " auto=0x" << std::hex << auto_host[first_diff]
              << " prequant=0x" << prequant_host[first_diff] << std::dec
              << '\n';
    return false;
  }

  if (graph_replay) {
    hipStream_t stream = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    S7_MM_CHECK(hipStreamCreate(&stream));
    buffers.output = static_cast<uint16_t *>(output_prequant.data);
    buffers.activation = activation_encoded.data;
    buffers.activation_scales =
        static_cast<uint8_t *>(activation_encoded.data) +
        plan.activation_scale_offset;
    S7_MM_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    if (lowp_matmul_launch(&plan, &buffers, stream) != LOWP_SUCCESS) {
      std::cerr << "FAIL: M1 capture launch\n";
      return false;
    }
    S7_MM_CHECK(hipStreamEndCapture(stream, &graph));
    S7_MM_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    S7_MM_CHECK(hipGraphDestroy(graph));
    S7_MM_CHECK(hipGraphLaunch(exec, stream));
    S7_MM_CHECK(hipStreamSynchronize(stream));
    S7_MM_CHECK(hipGraphExecDestroy(exec));
    S7_MM_CHECK(hipStreamDestroy(stream));
  }

  std::cout << "{\"shape\":\"" << shape.name
            << "\",\"variant\":" << plan.variant
            << ",\"workspace_bytes\":" << plan.workspace_bytes
            << ",\"prequant_bitwise\":true,\"graph_replay\":"
            << (graph_replay ? "true" : "false") << "}\n";
  return true;
}

} // namespace

int main(int argc, char **argv) {
  const std::string target = argc > 1 ? argv[1] : "";
  if (target != "gfx1030" && target != "gfx1201") {
    std::cerr << "usage: probe gfx1030|gfx1201\n";
    return 2;
  }
  hipDeviceProp_t properties{};
  if (hipGetDeviceProperties(&properties, 0) != hipSuccess ||
      std::string(properties.gcnArchName).find(target) == std::string::npos) {
    std::cerr << "FAIL: target mismatch observed=" << properties.gcnArchName
              << " expected=" << target << '\n';
    return 1;
  }
  const lowp_target_t lowp_target =
      target == "gfx1030" ? LOWP_TARGET_GFX1030 : LOWP_TARGET_GFX1201;
  bool ok = true;
  for (const Shape &shape : kShapes) {
    ok = run_shape(shape, lowp_target, shape.m == 1U) && ok;
  }
  return ok ? 0 : 1;
}
