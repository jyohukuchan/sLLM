#ifndef SLLM_LOWP_HOST_STUB_HIP_RUNTIME_H
#define SLLM_LOWP_HOST_STUB_HIP_RUNTIME_H

// Minimal declarations needed by lowp's host selector contract.  Host plan
// tests must not pull the native/hip tree or a real ROCm runtime.
#include <cstdint>

enum hipError_t : int {
  hipSuccess = 0,
  hipErrorInvalidValue = 1,
  hipErrorOutOfMemory = 2,
  hipErrorNotSupported = 801,
};

struct lowp_host_hip_stream;
using hipStream_t = lowp_host_hip_stream *;

#endif
