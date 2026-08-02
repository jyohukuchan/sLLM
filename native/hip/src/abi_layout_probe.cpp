#include "ullm/hip.h"

#include <cstddef>
#include <iostream>

int main() {
#define ULLM_PRINT_CONSTANT(name)                                              \
  std::cout << "const " #name "=" << name << '\n'
  ULLM_PRINT_CONSTANT(ULLM_HIP_ABI_VERSION);
  ULLM_PRINT_CONSTANT(ULLM_HIP_LIBRARY_VERSION_MAJOR);
  ULLM_PRINT_CONSTANT(ULLM_HIP_LIBRARY_VERSION_MINOR);
  ULLM_PRINT_CONSTANT(ULLM_HIP_LIBRARY_VERSION_PATCH);
  ULLM_PRINT_CONSTANT(ULLM_STATUS_OK);
  ULLM_PRINT_CONSTANT(ULLM_STATUS_INVALID_ARGUMENT);
  ULLM_PRINT_CONSTANT(ULLM_STATUS_BUFFER_TOO_SMALL);
  ULLM_PRINT_CONSTANT(ULLM_STATUS_UNSUPPORTED);
  ULLM_PRINT_CONSTANT(ULLM_STATUS_HIP_UNAVAILABLE);
  ULLM_PRINT_CONSTANT(ULLM_STATUS_INVALID_ABI_VERSION);
  ULLM_PRINT_CONSTANT(ULLM_STATUS_RESERVED_NONZERO);
  ULLM_PRINT_CONSTANT(ULLM_STATUS_INTERNAL_ERROR);
  ULLM_PRINT_CONSTANT(ULLM_BACKEND_HIP);
  ULLM_PRINT_CONSTANT(ULLM_ACCESS_READ);
  ULLM_PRINT_CONSTANT(ULLM_ACCESS_WRITE);
  ULLM_PRINT_CONSTANT(ULLM_ACCESS_READ_WRITE);
#undef ULLM_PRINT_CONSTANT

  std::cout << "layout ullm_error_sink_t size=" << sizeof(ullm_error_sink_t)
            << " align=" << alignof(ullm_error_sink_t)
            << " struct_size=" << offsetof(ullm_error_sink_t, struct_size)
            << " abi_version=" << offsetof(ullm_error_sink_t, abi_version)
            << " message=" << offsetof(ullm_error_sink_t, message)
            << " message_capacity="
            << offsetof(ullm_error_sink_t, message_capacity)
            << " message_length=" << offsetof(ullm_error_sink_t, message_length)
            << " reserved=" << offsetof(ullm_error_sink_t, reserved) << '\n';
  std::cout << "layout ullm_version_info_t size=" << sizeof(ullm_version_info_t)
            << " align=" << alignof(ullm_version_info_t)
            << " struct_size=" << offsetof(ullm_version_info_t, struct_size)
            << " abi_version=" << offsetof(ullm_version_info_t, abi_version)
            << " major=" << offsetof(ullm_version_info_t, major)
            << " minor=" << offsetof(ullm_version_info_t, minor)
            << " patch=" << offsetof(ullm_version_info_t, patch)
            << " reserved=" << offsetof(ullm_version_info_t, reserved) << '\n';
  std::cout << "layout ullm_backend_probe_result_t size="
            << sizeof(ullm_backend_probe_result_t)
            << " align=" << alignof(ullm_backend_probe_result_t)
            << " struct_size="
            << offsetof(ullm_backend_probe_result_t, struct_size)
            << " abi_version="
            << offsetof(ullm_backend_probe_result_t, abi_version)
            << " backend=" << offsetof(ullm_backend_probe_result_t, backend)
            << " available=" << offsetof(ullm_backend_probe_result_t, available)
            << " hip_runtime_present="
            << offsetof(ullm_backend_probe_result_t, hip_runtime_present)
            << " reserved=" << offsetof(ullm_backend_probe_result_t, reserved)
            << '\n';
  std::cout << "layout ullm_context_probe_result_t size="
            << sizeof(ullm_context_probe_result_t)
            << " align=" << alignof(ullm_context_probe_result_t)
            << " struct_size="
            << offsetof(ullm_context_probe_result_t, struct_size)
            << " abi_version="
            << offsetof(ullm_context_probe_result_t, abi_version)
            << " context_present="
            << offsetof(ullm_context_probe_result_t, context_present)
            << " hip_available="
            << offsetof(ullm_context_probe_result_t, hip_available)
            << " reserved=" << offsetof(ullm_context_probe_result_t, reserved)
            << '\n';
  return 0;
}
