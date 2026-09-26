#include "causal_attention_kernel_internal.hpp"
#include "decode_control_kernel_internal.hpp"
#include "paged_kv_device_layout.hpp"

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#error SLLM_TEST_EXPECTED_TARGET must name one exact GPU target
#endif

namespace {
constexpr uint32_t kKvHeads = 4U;
constexpr std::array<uint32_t, 2U> kReviewedQHeads = {16U, 24U};
constexpr uint32_t kMaxQHeads = 24U;
constexpr uint32_t kHeadDim = 256U;
constexpr uint32_t kBlocks = 1U;
constexpr uint32_t kMaxRows = 5U;

void check(const hipError_t status, const char *const operation) {
  if (status != hipSuccess) {
    std::cerr << operation << ": " << hipGetErrorString(status) << '\n';
    std::exit(1);
  }
}

uint16_t bf16_one() { return 0x3f80U; }
uint16_t fp16_one() { return 0x3c00U; }

struct DeviceState final {
  std::uint8_t *key = nullptr;
  std::uint8_t *value = nullptr;
  sllm_paged_kv::BlockDescriptor *descriptors = nullptr;
  uint32_t *table = nullptr;
  uint32_t *status = nullptr;
  uint16_t *query = nullptr;
  uint16_t *output = nullptr;
  sllm_decode_control::ControlV1 *control = nullptr;

  void allocate() {
    const size_t bytes =
        static_cast<size_t>(128U) * kKvHeads * kHeadDim * sizeof(uint16_t);
    check(hipMalloc(reinterpret_cast<void **>(&key), bytes), "key allocation");
    check(hipMalloc(reinterpret_cast<void **>(&value), bytes),
          "value allocation");
    check(hipMalloc(reinterpret_cast<void **>(&descriptors),
                    sizeof(sllm_paged_kv::BlockDescriptor)),
          "descriptor allocation");
    check(hipMalloc(reinterpret_cast<void **>(&table),
                    sizeof(uint32_t) * kBlocks),
          "table allocation");
    check(hipMalloc(reinterpret_cast<void **>(&status), sizeof(uint32_t)),
          "status allocation");
    check(hipMalloc(reinterpret_cast<void **>(&query),
                    static_cast<size_t>(kMaxRows) * kMaxQHeads * kHeadDim *
                        sizeof(uint16_t)),
          "query allocation");
    check(hipMalloc(reinterpret_cast<void **>(&output),
                    static_cast<size_t>(kMaxRows) * kMaxQHeads * kHeadDim *
                        sizeof(uint16_t)),
          "output allocation");
    check(hipMalloc(reinterpret_cast<void **>(&control),
                    sizeof(sllm_decode_control::ControlV1)),
          "control allocation");
    sllm_paged_kv::BlockDescriptor descriptor{key,     value,   nullptr,
                                              nullptr, nullptr, nullptr};
    const uint32_t block = 0U;
    check(hipMemcpy(descriptors, &descriptor, sizeof(descriptor),
                    hipMemcpyHostToDevice),
          "descriptor upload");
    check(hipMemcpy(table, &block, sizeof(block), hipMemcpyHostToDevice),
          "table upload");
    std::vector<uint16_t> query_host(
        static_cast<size_t>(kMaxRows) * kMaxQHeads * kHeadDim, 0U);
    std::vector<uint16_t> kv_host(
        static_cast<size_t>(128U) * kKvHeads * kHeadDim, fp16_one());
    check(hipMemcpy(key, kv_host.data(), bytes, hipMemcpyHostToDevice),
          "key upload");
    check(hipMemcpy(value, kv_host.data(), bytes, hipMemcpyHostToDevice),
          "value upload");
    check(hipMemcpy(query, query_host.data(),
                    query_host.size() * sizeof(uint16_t),
                    hipMemcpyHostToDevice),
          "query upload");
  }

  ~DeviceState() {
    (void)hipFree(key);
    (void)hipFree(value);
    (void)hipFree(descriptors);
    (void)hipFree(table);
    (void)hipFree(status);
    (void)hipFree(query);
    (void)hipFree(output);
    (void)hipFree(control);
  }
};

} // namespace

int main() {
  hipDeviceProp_t properties{};
  check(hipGetDeviceProperties(&properties, 0), "device query");
  if (std::string(properties.gcnArchName).find(SLLM_TEST_EXPECTED_TARGET) ==
      std::string::npos) {
    std::cerr << "unexpected target " << properties.gcnArchName << '\n';
    return 1;
  }
  DeviceState device;
  device.allocate();
  hipStream_t stream = nullptr;
  check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
        "stream create");
  sllm_decode_control::ControlV1 control{};
  control.version = sllm_decode_control::kVersion;
  control.status = static_cast<uint32_t>(sllm_decode_control::Status::Ok);
  control.mode = sllm_decode_control::kModeTargetOnly;
  control.phase_position = 0U;
  control.phase_rows = 1U;
  control.phase_kind = sllm_decode_control::kPhaseTarget;
  control.phase_active = 1U;
  for (const uint32_t q_heads : kReviewedQHeads) {
    check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
          "begin capture");
    check(sllm_causal_attention_kernel::launch_paged_decode_fp16_device(
              device.query, device.table, device.descriptors, kBlocks, kBlocks,
              device.status, device.output, kMaxRows, q_heads, kKvHeads,
              kHeadDim, device.control, stream),
          "capture FP16 paged decode");
    hipGraph_t graph = nullptr;
    check(hipStreamEndCapture(stream, &graph), "end capture");
    hipGraphExec_t executable = nullptr;
    check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
          "instantiate graph");
    for (uint32_t rows = 1U; rows <= kMaxRows; ++rows) {
      control.phase_rows = rows;
      check(hipMemcpyAsync(device.control, &control, sizeof(control),
                           hipMemcpyHostToDevice, stream),
            "control upload");
      check(hipGraphLaunch(executable, stream), "graph replay");
      check(hipStreamSynchronize(stream), "graph synchronize");
      uint32_t status = 0U;
      check(hipMemcpy(&status, device.status, sizeof(status),
                      hipMemcpyDeviceToHost),
            "status download");
      if (status != 0U) {
        std::cerr << "device status=" << status << " q_heads=" << q_heads
                  << " rows=" << rows << '\n';
        return 1;
      }
      std::vector<uint16_t> output(static_cast<size_t>(rows) * q_heads *
                                   kHeadDim);
      check(hipMemcpy(output.data(), device.output,
                      output.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
            "output download");
      for (const uint16_t value : output) {
        if (value != bf16_one()) {
          std::cerr << "FP16 graph oracle mismatch q_heads=" << q_heads
                    << " rows=" << rows << " value=0x" << std::hex << value
                    << std::dec << '\n';
          return 1;
        }
      }
    }
    check(hipGraphExecDestroy(executable), "graph executable destroy");
    check(hipGraphDestroy(graph), "graph destroy");
  }
  check(hipStreamDestroy(stream), "stream destroy");
  std::cout << "phase87_stage10_paged_fp16_graph_gpu_test: PASS target="
            << properties.gcnArchName
            << " q_heads=16,24 rows=1..5"
               " oracle=bitwise fallback=0\n";
  return 0;
}
