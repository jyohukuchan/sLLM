#include "evidence_abi.h"

#include <cstdint>
#include <cstring>
#include <iostream>

namespace {

struct Error final {
  char message[256]{};
  sllm_error_sink_t sink{sizeof(sllm_error_sink_t),
                         SLLM_HIP_ABI_VERSION,
                         message,
                         sizeof(message),
                         0U,
                         {0U, 0U}};
};

sllm_hip_kv_readback_request_t valid_request(uint8_t *const output) {
  sllm_hip_kv_readback_request_t request{};
  request.struct_size = sizeof(request);
  request.abi_version = SLLM_HIP_KV_EVIDENCE_ABI_VERSION;
  request.view = reinterpret_cast<const sllm_kv_view_t *>(1U);
  request.plane = SLLM_HIP_KV_EVIDENCE_PLANE_K;
  request.byte_length = 2U;
  request.host_capacity = 2U;
  request.host_output = output;
  return request;
}

bool expects(const sllm_status_t actual, const sllm_status_t expected,
             const char *const operation, const Error &error) {
  if (actual == expected) {
    return true;
  }
  std::cerr << operation << " returned " << actual << ", expected " << expected
            << ": " << error.message << '\n';
  return false;
}

} // namespace

int main() {
  uint8_t output[2] = {0xa5U, 0xa5U};
  Error error;
  auto request = valid_request(output);
  bool valid =
      expects(sllm_hip_kv_view_readback(&request, &error.sink),
              SLLM_STATUS_HIP_UNAVAILABLE, "valid stub readback", error) &&
      output[0] == 0xa5U && output[1] == 0xa5U;

  auto undersized = request;
  undersized.host_capacity = 1U;
  valid = valid && expects(sllm_hip_kv_view_readback(&undersized, &error.sink),
                           SLLM_STATUS_BUFFER_TOO_SMALL,
                           "stub undersized output", error);
  auto null_output = request;
  null_output.host_output = nullptr;
  valid =
      valid && expects(sllm_hip_kv_view_readback(&null_output, &error.sink),
                       SLLM_STATUS_INVALID_ARGUMENT, "stub null output", error);
  auto reserved = request;
  reserved.reserved[0] = 1U;
  valid = valid &&
          expects(sllm_hip_kv_view_readback(&reserved, &error.sink),
                  SLLM_STATUS_RESERVED_NONZERO, "stub reserved field", error);
  auto wrong_size = request;
  wrong_size.struct_size -= 1U;
  valid =
      valid && expects(sllm_hip_kv_view_readback(&wrong_size, &error.sink),
                       SLLM_STATUS_INVALID_ARGUMENT, "stub wrong size", error);
  auto invalid_plane = request;
  invalid_plane.plane = 2U;
  valid = valid &&
          expects(sllm_hip_kv_view_readback(&invalid_plane, &error.sink),
                  SLLM_STATUS_INVALID_ARGUMENT, "stub invalid plane", error);
  valid = valid &&
          expects(sllm_hip_kv_view_readback(nullptr, &error.sink),
                  SLLM_STATUS_INVALID_ARGUMENT, "stub null request", error);
  if (!valid) {
    return 1;
  }
  std::cout << "private KV evidence stub fail-closed test: PASS\n";
  return 0;
}
