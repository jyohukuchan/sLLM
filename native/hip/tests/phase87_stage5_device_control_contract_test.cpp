#include "causal_attention_kernel_internal.hpp"
#include "decode_control_kernel_internal.hpp"
#include "kv_state_kernel_internal.hpp"

#include <cstdint>
#include <cstdio>

int main() {
  using sllm_causal_attention_kernel::decode_dynamic_split_count;
  if (decode_dynamic_split_count(8190U, 1U) != 32U ||
      decode_dynamic_split_count(8191U, 1U) != 128U ||
      decode_dynamic_split_count(8190U, 2U) != 128U ||
      decode_dynamic_split_count(UINT64_MAX, 1U) != 32U) {
    std::fprintf(stderr, "stage5 device split boundary contract failed\n");
    return 1;
  }
  sllm_decode_control::ControlV1 control{};
  control.version = sllm_decode_control::kVersion;
  control.phase_position = 8191U;
  control.phase_rows = 1U;
  control.phase_active = 1U;
  if (sizeof(control) != 144U || alignof(decltype(control)) != 16U ||
      control.phase_position + control.phase_rows != 8192U) {
    std::fprintf(stderr, "stage5 ControlV1 ABI contract failed\n");
    return 1;
  }
  std::puts("phase87_stage5_device_control_contract_test: PASS");
  return 0;
}
