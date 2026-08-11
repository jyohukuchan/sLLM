#include "evidence_abi.h"

#include "sllm/hip.h"

#include <cstddef>
#include <cstring>
#include <limits>

namespace {

static_assert(sizeof(std::size_t) <= sizeof(uint64_t),
              "HIP evidence ABI sizes require size_t to fit in uint64_t");

void clear_error(sllm_error_sink_t *sink) noexcept {
  if (sink != nullptr && sink->message != nullptr &&
      sink->message_capacity != 0U) {
    sink->message[0] = '\0';
  }
}

uint32_t write_error(sllm_error_sink_t *sink, uint32_t status,
                     const char *message) noexcept {
  if (sink == nullptr) {
    return status;
  }
  const std::size_t length = message == nullptr ? 0U : std::strlen(message);
  sink->message_length = static_cast<uint64_t>(length);
  if (sink->message_capacity == 0U) {
    return SLLM_STATUS_BUFFER_TOO_SMALL;
  }
  if (sink->message == nullptr) {
    return SLLM_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t capacity = static_cast<std::size_t>(sink->message_capacity);
  const std::size_t copied = length < capacity - 1U ? length : capacity - 1U;
  if (copied != 0U) {
    std::memcpy(sink->message, message, copied);
  }
  sink->message[copied] = '\0';
  return length <= capacity - 1U ? status : SLLM_STATUS_BUFFER_TOO_SMALL;
}

uint32_t validate_sink(sllm_error_sink_t *sink) noexcept {
  if (sink == nullptr) {
    return SLLM_STATUS_OK;
  }
  if (sink->struct_size < sizeof(*sink)) {
    return SLLM_STATUS_INVALID_ARGUMENT;
  }
  if (sink->abi_version != SLLM_HIP_ABI_VERSION) {
    return SLLM_STATUS_INVALID_ABI_VERSION;
  }
  if (sink->reserved[0] != 0U || sink->reserved[1] != 0U) {
    return SLLM_STATUS_RESERVED_NONZERO;
  }
  if (sink->message_capacity >
      static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return SLLM_STATUS_INVALID_ARGUMENT;
  }
  if (sink->message_capacity != 0U && sink->message == nullptr) {
    return SLLM_STATUS_INVALID_ARGUMENT;
  }
  sink->message_length = 0U;
  clear_error(sink);
  return SLLM_STATUS_OK;
}

uint32_t validate_request(const sllm_hip_evidence_request_t *request,
                          sllm_error_sink_t *sink) noexcept {
  if (request == nullptr) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "evidence request is null");
  }
  if (request->struct_size < sizeof(*request)) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "evidence request struct_size is too small");
  }
  if (request->abi_version != SLLM_HIP_EVIDENCE_ABI_VERSION) {
    return write_error(sink, SLLM_STATUS_INVALID_ABI_VERSION,
                       "unsupported evidence ABI version");
  }
  for (uint32_t value : request->reserved) {
    if (value != 0U) {
      return write_error(sink, SLLM_STATUS_RESERVED_NONZERO,
                         "evidence request reserved field is non-zero");
    }
  }
  if (request->input == nullptr || request->input_size == 0U) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "evidence input must be non-null and non-empty");
  }
  if (request->input_size >
      static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "evidence input is too large");
  }
  return SLLM_STATUS_OK;
}

} // namespace

extern "C" uint32_t
sllm_hip_evidence_submit(const sllm_hip_evidence_request_t *request,
                         sllm_hip_evidence_completion_t **completion,
                         sllm_error_sink_t *error_sink) {
  try {
    const uint32_t sink_status = validate_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const uint32_t request_status = validate_request(request, error_sink);
    if (request_status != SLLM_STATUS_OK) {
      return request_status;
    }
    if (completion == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "completion output is null");
    }
    *completion = nullptr;
    return write_error(
        error_sink, SLLM_STATUS_HIP_UNAVAILABLE,
        "HIP evidence runtime is unavailable; CPU fallback is forbidden");
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in evidence submit");
  }
}

extern "C" uint32_t sllm_hip_evidence_wait(
    sllm_hip_evidence_completion_t *completion, uint32_t /*timeout_ms*/,
    uint8_t * /*output*/, uint64_t /*output_capacity*/,
    sllm_hip_evidence_result_t *result, sllm_error_sink_t *error_sink) {
  try {
    const uint32_t sink_status = validate_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    (void)result;
    return write_error(error_sink, SLLM_STATUS_HIP_INVALID_HANDLE,
                       completion == nullptr
                           ? "evidence completion handle is null"
                           : "evidence completion handle is stale");
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in evidence wait");
  }
}

extern "C" uint32_t
sllm_hip_evidence_destroy(sllm_hip_evidence_completion_t **completion,
                          sllm_error_sink_t *error_sink) {
  try {
    const uint32_t sink_status = validate_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (completion == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "completion pointer is null");
    }
    if (*completion == nullptr) {
      return write_error(error_sink, SLLM_STATUS_HIP_INVALID_HANDLE,
                         "evidence completion handle is stale");
    }
    return write_error(error_sink, SLLM_STATUS_HIP_INVALID_HANDLE,
                       "evidence completion handle is not live");
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in evidence destroy");
  }
}
