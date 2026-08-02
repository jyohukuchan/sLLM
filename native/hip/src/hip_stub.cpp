#include "ullm/hip.h"

#include <cstddef>
#include <cstring>
#include <limits>

namespace {

void clear_error(ullm_error_sink_t *const sink) noexcept {
  if (sink == nullptr || sink->message == nullptr ||
      sink->message_capacity == 0U) {
    return;
  }
  sink->message[0] = '\0';
}

ullm_status_t write_error(ullm_error_sink_t *const sink,
                          const ullm_status_t primary_status,
                          const char *const message) noexcept {
  if (sink == nullptr) {
    return primary_status;
  }
  const std::size_t message_length =
      message == nullptr ? 0U : std::strlen(message);
  sink->message_length = static_cast<uint64_t>(message_length);
  if (sink->message_capacity == 0U) {
    return ULLM_STATUS_BUFFER_TOO_SMALL;
  }
  if (sink->message == nullptr) {
    return ULLM_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t capacity = static_cast<std::size_t>(sink->message_capacity);
  const std::size_t copied =
      message_length < (capacity - 1U) ? message_length : (capacity - 1U);
  if (copied != 0U) {
    std::memcpy(sink->message, message, copied);
  }
  sink->message[copied] = '\0';
  return message_length <= (capacity - 1U) ? primary_status
                                           : ULLM_STATUS_BUFFER_TOO_SMALL;
}

ullm_status_t validate_error_sink(ullm_error_sink_t *const sink) noexcept {
  if (sink == nullptr) {
    return ULLM_STATUS_OK;
  }
  if (sink->struct_size < sizeof(ullm_error_sink_t)) {
    return ULLM_STATUS_INVALID_ARGUMENT;
  }
  if (sink->abi_version != ULLM_HIP_ABI_VERSION) {
    return ULLM_STATUS_INVALID_ABI_VERSION;
  }
  if (sink->reserved[0] != 0U || sink->reserved[1] != 0U) {
    return ULLM_STATUS_RESERVED_NONZERO;
  }
  if (sink->message_capacity >
      static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return ULLM_STATUS_INVALID_ARGUMENT;
  }
  if (sink->message_capacity != 0U && sink->message == nullptr) {
    return ULLM_STATUS_INVALID_ARGUMENT;
  }
  sink->message_length = 0U;
  clear_error(sink);
  return ULLM_STATUS_OK;
}

template <typename Struct>
ullm_status_t validate_struct(const Struct *const value,
                              ullm_error_sink_t *const sink,
                              const char *const name) noexcept {
  if (value == nullptr) {
    return write_error(sink, ULLM_STATUS_INVALID_ARGUMENT, name);
  }
  if (value->struct_size < sizeof(Struct)) {
    return write_error(sink, ULLM_STATUS_INVALID_ARGUMENT,
                       "struct_size is smaller than the Phase 1 ABI struct");
  }
  if (value->abi_version != ULLM_HIP_ABI_VERSION) {
    return write_error(sink, ULLM_STATUS_INVALID_ABI_VERSION,
                       "unsupported ABI version");
  }
  return ULLM_STATUS_OK;
}

ullm_status_t
validate_backend_probe_result(const ullm_backend_probe_result_t *const result,
                              ullm_error_sink_t *const sink) noexcept {
  const ullm_status_t status =
      validate_struct(result, sink, "probe result is null");
  if (status != ULLM_STATUS_OK) {
    return status;
  }
  if (result->reserved[0] != 0U || result->reserved[1] != 0U ||
      result->reserved[2] != 0U) {
    return write_error(sink, ULLM_STATUS_RESERVED_NONZERO,
                       "probe result reserved fields must be zero");
  }
  return ULLM_STATUS_OK;
}

ullm_status_t
validate_context_probe_result(const ullm_context_probe_result_t *const result,
                              ullm_error_sink_t *const sink) noexcept {
  const ullm_status_t status =
      validate_struct(result, sink, "context probe result is null");
  if (status != ULLM_STATUS_OK) {
    return status;
  }
  if (result->reserved[0] != 0U || result->reserved[1] != 0U ||
      result->reserved[2] != 0U || result->reserved[3] != 0U) {
    return write_error(sink, ULLM_STATUS_RESERVED_NONZERO,
                       "context probe reserved fields must be zero");
  }
  return ULLM_STATUS_OK;
}

} // namespace

extern "C" ullm_status_t
ullm_get_abi_version(uint32_t *const abi_version,
                     ullm_error_sink_t *const error_sink) {
  try {
    const ullm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != ULLM_STATUS_OK) {
      return sink_status;
    }
    if (abi_version == nullptr) {
      return write_error(error_sink, ULLM_STATUS_INVALID_ARGUMENT,
                         "abi_version output is null");
    }
    *abi_version = ULLM_HIP_ABI_VERSION;
    return ULLM_STATUS_OK;
  } catch (...) {
    return write_error(error_sink, ULLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in ABI version query");
  }
}

extern "C" ullm_status_t
ullm_query_version(ullm_version_info_t *const version,
                   ullm_error_sink_t *const error_sink) {
  try {
    const ullm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != ULLM_STATUS_OK) {
      return sink_status;
    }
    const ullm_status_t struct_status =
        validate_struct(version, error_sink, "version output is null");
    if (struct_status != ULLM_STATUS_OK) {
      return struct_status;
    }
    if (version->reserved[0] != 0U || version->reserved[1] != 0U ||
        version->reserved[2] != 0U) {
      return write_error(error_sink, ULLM_STATUS_RESERVED_NONZERO,
                         "version reserved fields must be zero");
    }
    version->major = ULLM_HIP_LIBRARY_VERSION_MAJOR;
    version->minor = ULLM_HIP_LIBRARY_VERSION_MINOR;
    version->patch = ULLM_HIP_LIBRARY_VERSION_PATCH;
    return ULLM_STATUS_OK;
  } catch (...) {
    return write_error(error_sink, ULLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in version query");
  }
}

extern "C" ullm_status_t
ullm_backend_probe(const uint32_t backend,
                   ullm_backend_probe_result_t *const result,
                   ullm_error_sink_t *const error_sink) {
  try {
    const ullm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != ULLM_STATUS_OK) {
      return sink_status;
    }
    const ullm_status_t result_status =
        validate_backend_probe_result(result, error_sink);
    if (result_status != ULLM_STATUS_OK) {
      return result_status;
    }
    if (backend != ULLM_BACKEND_HIP) {
      return write_error(error_sink, ULLM_STATUS_UNSUPPORTED,
                         "unknown backend identifier");
    }
    result->backend = backend;
    result->available = 0U;
    result->hip_runtime_present = 0U;
    return write_error(error_sink, ULLM_STATUS_HIP_UNAVAILABLE,
                       "HIP backend is unavailable in Phase 1 host stub");
  } catch (...) {
    return write_error(error_sink, ULLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in backend probe");
  }
}

extern "C" ullm_status_t
ullm_context_probe(const ullm_context_t *const context,
                   ullm_context_probe_result_t *const result,
                   ullm_error_sink_t *const error_sink) {
  try {
    const ullm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != ULLM_STATUS_OK) {
      return sink_status;
    }
    const ullm_status_t result_status =
        validate_context_probe_result(result, error_sink);
    if (result_status != ULLM_STATUS_OK) {
      return result_status;
    }
    result->context_present = context == nullptr ? 0U : 1U;
    result->hip_available = 0U;
    return write_error(error_sink, ULLM_STATUS_HIP_UNAVAILABLE,
                       "HIP context is unavailable in Phase 1 host stub");
  } catch (...) {
    return write_error(error_sink, ULLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in context probe");
  }
}
