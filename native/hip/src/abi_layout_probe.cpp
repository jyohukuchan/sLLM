#include "sllm/hip.h"

#include <cstddef>
#include <iostream>

int main() {
#define SLLM_PRINT_CONSTANT(name)                                              \
  std::cout << "const " #name "=" << name << '\n'
  SLLM_PRINT_CONSTANT(SLLM_HIP_ABI_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_HIP_LIBRARY_VERSION_MAJOR);
  SLLM_PRINT_CONSTANT(SLLM_HIP_LIBRARY_VERSION_MINOR);
  SLLM_PRINT_CONSTANT(SLLM_HIP_LIBRARY_VERSION_PATCH);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_OK);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_INVALID_ARGUMENT);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_BUFFER_TOO_SMALL);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_UNSUPPORTED);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_HIP_UNAVAILABLE);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_INVALID_ABI_VERSION);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_RESERVED_NONZERO);
  SLLM_PRINT_CONSTANT(SLLM_STATUS_INTERNAL_ERROR);
  SLLM_PRINT_CONSTANT(SLLM_BACKEND_HIP);
  SLLM_PRINT_CONSTANT(SLLM_ACCESS_READ);
  SLLM_PRINT_CONSTANT(SLLM_ACCESS_WRITE);
  SLLM_PRINT_CONSTANT(SLLM_ACCESS_READ_WRITE);
#undef SLLM_PRINT_CONSTANT

  std::cout << "layout sllm_error_sink_t size=" << sizeof(sllm_error_sink_t)
            << " align=" << alignof(sllm_error_sink_t)
            << " struct_size=" << offsetof(sllm_error_sink_t, struct_size)
            << " abi_version=" << offsetof(sllm_error_sink_t, abi_version)
            << " message=" << offsetof(sllm_error_sink_t, message)
            << " message_capacity="
            << offsetof(sllm_error_sink_t, message_capacity)
            << " message_length=" << offsetof(sllm_error_sink_t, message_length)
            << " reserved=" << offsetof(sllm_error_sink_t, reserved) << '\n';
  std::cout << "layout sllm_version_info_t size=" << sizeof(sllm_version_info_t)
            << " align=" << alignof(sllm_version_info_t)
            << " struct_size=" << offsetof(sllm_version_info_t, struct_size)
            << " abi_version=" << offsetof(sllm_version_info_t, abi_version)
            << " major=" << offsetof(sllm_version_info_t, major)
            << " minor=" << offsetof(sllm_version_info_t, minor)
            << " patch=" << offsetof(sllm_version_info_t, patch)
            << " reserved=" << offsetof(sllm_version_info_t, reserved) << '\n';
  std::cout << "layout sllm_backend_probe_result_t size="
            << sizeof(sllm_backend_probe_result_t)
            << " align=" << alignof(sllm_backend_probe_result_t)
            << " struct_size="
            << offsetof(sllm_backend_probe_result_t, struct_size)
            << " abi_version="
            << offsetof(sllm_backend_probe_result_t, abi_version)
            << " backend=" << offsetof(sllm_backend_probe_result_t, backend)
            << " available=" << offsetof(sllm_backend_probe_result_t, available)
            << " hip_runtime_present="
            << offsetof(sllm_backend_probe_result_t, hip_runtime_present)
            << " reserved=" << offsetof(sllm_backend_probe_result_t, reserved)
            << '\n';
  std::cout << "layout sllm_context_probe_result_t size="
            << sizeof(sllm_context_probe_result_t)
            << " align=" << alignof(sllm_context_probe_result_t)
            << " struct_size="
            << offsetof(sllm_context_probe_result_t, struct_size)
            << " abi_version="
            << offsetof(sllm_context_probe_result_t, abi_version)
            << " context_present="
            << offsetof(sllm_context_probe_result_t, context_present)
            << " hip_available="
            << offsetof(sllm_context_probe_result_t, hip_available)
            << " reserved=" << offsetof(sllm_context_probe_result_t, reserved)
            << '\n';
  return 0;
}
