#include "sllm/hip.h"

#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <iostream>

#ifndef SLLM_TEST_EXPECTED_TARGET
#error SLLM_TEST_EXPECTED_TARGET must name one exact GPU target
#endif

namespace {

struct Error final {
  char message[512]{};
  sllm_error_sink_t sink{sizeof(sllm_error_sink_t),
                         SLLM_HIP_ABI_VERSION,
                         message,
                         sizeof(message),
                         0U,
                         {0U, 0U}};
};

bool expect(const sllm_status_t actual, const sllm_status_t expected,
            const char *const operation, const Error &error) {
  if (actual == expected)
    return true;
  std::cerr << operation << " returned " << actual << ", expected " << expected
            << ": " << error.message << '\n';
  return false;
}

bool run() {
  Error error;
  uint32_t device_count = 0U;
  if (!expect(sllm_device_count(&device_count, &error.sink), SLLM_STATUS_OK,
              "device count", error) ||
      device_count != 1U)
    return false;
  sllm_device_info_t device{};
  device.struct_size = sizeof(device);
  device.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect(sllm_device_query(0U, &device, &error.sink), SLLM_STATUS_OK,
              "device query", error) ||
      std::strcmp(device.gcn_arch_name, SLLM_TEST_EXPECTED_TARGET) != 0)
    return false;
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  context_info.device_index = 0U;
  std::snprintf(context_info.expected_gcn_arch_name,
                sizeof(context_info.expected_gcn_arch_name), "%s",
                SLLM_TEST_EXPECTED_TARGET);
  sllm_context_t *context = nullptr;
  if (!expect(sllm_context_create(&context_info, &context, &error.sink),
              SLLM_STATUS_OK, "context create", error) ||
      context == nullptr)
    return false;

  bool passed = true;
  for (const uint64_t capacity :
       {UINT64_C(127), UINT64_C(128), UINT64_C(129)}) {
    for (const bool fp16 : {true, false}) {
      const uint64_t blocks = capacity / 128U + (capacity % 128U != 0U);
      sllm_kv_state_paged_create_info_t create{};
      create.struct_size = sizeof(create);
      create.abi_version = SLLM_HIP_ABI_VERSION;
      create.create_info_version = SLLM_HIP_KV_PAGED_CREATE_INFO_VERSION;
      create.session_id = capacity + (fp16 ? 1U : 2U);
      create.capacity_tokens = capacity;
      create.head_count = 4U;
      create.head_dim = 256U;
      create.memory_kind = SLLM_HIP_KV_MEMORY_KIND_PAGED;
      create.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
      create.dtype =
          fp16 ? SLLM_TENSOR_DTYPE_F16 : SLLM_TENSOR_DTYPE_F8_E4M3_FN;
      create.encoding = fp16 ? SLLM_HIP_KV_ENCODING_FP16_V1
                             : SLLM_HIP_KV_ENCODING_MXFP8_E4_V1;
      create.scale_dtype = fp16 ? 0U : SLLM_TENSOR_DTYPE_U8;
      create.quantization_block_size = fp16 ? 0U : 32U;
      create.token_block_size = SLLM_HIP_KV_PAGED_TOKEN_BLOCK_SIZE;
      create.physical_layout_version = SLLM_HIP_KV_PAGED_LAYOUT_VERSION;
      create.logical_table_capacity = blocks;
      create.max_physical_blocks = blocks * 2U;
      sllm_kv_state_t *state = nullptr;
      passed &= expect(
          sllm_kv_state_create_paged(context, &create, &state, &error.sink),
          SLLM_STATUS_OK, "paged create", error);
      if (!passed || state == nullptr)
        break;
      sllm_kv_paged_view_info_t view{};
      view.struct_size = sizeof(view);
      view.abi_version = SLLM_HIP_ABI_VERSION;
      view.info_version = SLLM_HIP_KV_PAGED_VIEW_INFO_VERSION;
      passed &= expect(sllm_kv_state_query_paged(state, &view, &error.sink),
                       SLLM_STATUS_OK, "paged query", error);
      passed &= view.memory_kind == SLLM_HIP_KV_MEMORY_KIND_PAGED &&
                view.capacity_tokens == capacity &&
                view.logical_table_capacity == blocks &&
                view.max_physical_blocks == blocks * 2U &&
                view.observed_length == 0U &&
                view.allocated_physical_blocks == 0U &&
                view.committed_bytes_total == 0U;
      sllm_kv_view_info_t legacy{};
      legacy.struct_size = sizeof(legacy);
      legacy.abi_version = SLLM_HIP_ABI_VERSION;
      legacy.info_version = SLLM_HIP_KV_VIEW_INFO_VERSION;
      passed &=
          expect(sllm_kv_state_query(state, &legacy, &error.sink),
                 SLLM_STATUS_UNSUPPORTED, "legacy query rejection", error);
      sllm_kv_view_t *snapshot = nullptr;
      passed &= expect(sllm_kv_state_snapshot(state, &snapshot, &error.sink),
                       SLLM_STATUS_OK, "paged snapshot", error);
      if (snapshot != nullptr) {
        sllm_kv_paged_view_info_t snapshot_view{};
        snapshot_view.struct_size = sizeof(snapshot_view);
        snapshot_view.abi_version = SLLM_HIP_ABI_VERSION;
        snapshot_view.info_version = SLLM_HIP_KV_PAGED_VIEW_INFO_VERSION;
        passed &= expect(
            sllm_kv_view_query_paged(snapshot, &snapshot_view, &error.sink),
            SLLM_STATUS_OK, "paged snapshot query", error);
        passed &= snapshot_view.state_identity == view.state_identity &&
                  snapshot_view.observed_length == 0U;
        passed &= expect(sllm_kv_view_query(snapshot, &legacy, &error.sink),
                         SLLM_STATUS_UNSUPPORTED,
                         "legacy snapshot query rejection", error);
        passed &= expect(sllm_kv_view_release(&snapshot, &error.sink),
                         SLLM_STATUS_OK, "paged snapshot release", error);
      } else {
        passed = false;
      }
      passed &= expect(sllm_kv_state_release(&state, &error.sink),
                       SLLM_STATUS_OK, "paged release", error);
      passed &= state == nullptr;
      if (!passed)
        break;
    }
    if (!passed)
      break;
  }
  passed &= expect(sllm_context_release(&context, &error.sink), SLLM_STATUS_OK,
                   "context release", error);
  return passed && context == nullptr;
}

} // namespace

int main() {
  if (!run()) {
    std::cerr << "phase87_stage10_paged_state_public_gpu_test: FAIL\n";
    return 1;
  }
  std::cout << "phase87_stage10_paged_state_public_gpu_test: PASS\n";
  return 0;
}
