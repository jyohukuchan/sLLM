#include "public_runtime_internal.hpp"
#include "rmsnorm_api.hpp"

#include <cstring>

namespace {

using sllm_public_runtime::validate_buffer_create_info;
using sllm_public_runtime::validate_completion_result;
using sllm_public_runtime::validate_context_create_info;
using sllm_public_runtime::validate_error_sink;
using sllm_public_runtime::validate_queue_create_info;
using sllm_public_runtime::validate_struct;
using sllm_public_runtime::validate_transfer_desc;
using sllm_public_runtime::write_error;

sllm_status_t unavailable(sllm_error_sink_t *const sink) noexcept {
  return write_error(
      sink, SLLM_STATUS_HIP_UNAVAILABLE,
      "public HIP runtime is unavailable; CPU fallback is disabled");
}

sllm_status_t validate_device_info(const sllm_device_info_t *const info,
                                   sllm_error_sink_t *const sink) noexcept {
  const sllm_status_t status =
      validate_struct(info, sink, "device info output is null");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  if (info->reserved0 != 0U || info->reserved[0] != 0U ||
      info->reserved[1] != 0U || info->reserved[2] != 0U ||
      info->reserved[3] != 0U) {
    return write_error(sink, SLLM_STATUS_RESERVED_NONZERO,
                       "device info reserved fields must be zero");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_dispatch_info(const sllm_rmsnorm_dispatch_info_t *const info,
                       sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "RMSNorm dispatch info output is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, info, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_rmsnorm_dispatch_info_t)) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "RMSNorm dispatch info struct size is unsupported");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return write_error(sink, SLLM_STATUS_INVALID_ABI_VERSION,
                       "RMSNorm dispatch info ABI is unsupported");
  }
  if (info->info_version != SLLM_HIP_RMSNORM_DISPATCH_INFO_VERSION ||
      info->reserved[0] != 0U || info->reserved[1] != 0U ||
      info->reserved[2] != 0U || info->reserved[3] != 0U ||
      info->reserved[4] != 0U || info->reserved[5] != 0U ||
      info->reserved[6] != 0U || info->reserved[7] != 0U) {
    return write_error(sink, SLLM_STATUS_RESERVED_NONZERO,
                       "RMSNorm dispatch info version or reserved is invalid");
  }
  return SLLM_STATUS_OK;
}

} // namespace

extern "C" sllm_status_t
sllm_device_count(uint32_t *const count,
                  sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (count == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "device count output is null");
    }
    *count = 0U;
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in public device count");
  }
}

extern "C" sllm_status_t
sllm_device_query(const uint32_t /*device_index*/,
                  sllm_device_info_t *const info,
                  sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t info_status = validate_device_info(info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in public device query");
  }
}

extern "C" sllm_status_t
sllm_context_create(const sllm_context_create_info_t *const info,
                    sllm_context_t **const context,
                    sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (context != nullptr) {
      *context = nullptr;
    }
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t info_status =
        validate_context_create_info(info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    if (context == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "context output is null");
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in public context create");
  }
}

extern "C" sllm_status_t
sllm_context_release(sllm_context_t **const context,
                     sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (context == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "context release pointer is null");
    }
    if (*context == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "context handle is null");
    }
    return write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                       "context handle is not owned by the public runtime");
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in public context release");
  }
}

extern "C" sllm_status_t
sllm_queue_create(const sllm_context_t *const context,
                  const sllm_queue_create_info_t *const info,
                  sllm_queue_t **const queue,
                  sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (queue != nullptr) {
      *queue = nullptr;
    }
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t info_status =
        validate_queue_create_info(info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    if (context == nullptr || queue == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "queue context or output is null");
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in public queue create");
  }
}

extern "C" sllm_status_t
sllm_queue_release(sllm_queue_t **const queue,
                   sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (queue == nullptr || *queue == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "queue handle is null");
    }
    return write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                       "queue handle is not owned by the public runtime");
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in public queue release");
  }
}

extern "C" sllm_status_t
sllm_buffer_create(const sllm_context_t *const context,
                   const sllm_buffer_create_info_t *const info,
                   sllm_buffer_t **const buffer,
                   sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (buffer != nullptr) {
      *buffer = nullptr;
    }
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t info_status =
        validate_buffer_create_info(info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    if (context == nullptr || buffer == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "buffer context or output is null");
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in public buffer create");
  }
}

extern "C" sllm_status_t
sllm_buffer_release(sllm_buffer_t **const buffer,
                    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (buffer == nullptr || *buffer == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "buffer handle is null");
    }
    return write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                       "buffer handle is not owned by the public runtime");
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in public buffer release");
  }
}

extern "C" sllm_status_t
sllm_buffer_size(const sllm_buffer_t *const buffer, uint64_t *const size_bytes,
                 sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (buffer == nullptr || size_bytes == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "buffer or size output is null");
    }
    *size_bytes = 0U;
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in public buffer size");
  }
}

extern "C" sllm_status_t
sllm_event_create(const sllm_context_t *const context,
                  sllm_event_t **const event,
                  sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (event != nullptr) {
      *event = nullptr;
    }
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (context == nullptr || event == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "event context or output is null");
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in public event create");
  }
}

extern "C" sllm_status_t
sllm_event_release(sllm_event_t **const event,
                   sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (event == nullptr || *event == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "event handle is null");
    }
    return write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                       "event handle is not owned by the public runtime");
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in public event release");
  }
}

extern "C" sllm_status_t
sllm_buffer_copy_h2d(const sllm_queue_t *const queue,
                     const sllm_buffer_t *const buffer,
                     const sllm_transfer_desc_t *const transfer,
                     sllm_completion_t **const completion,
                     sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (completion != nullptr) {
      *completion = nullptr;
    }
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t transfer_status =
        validate_transfer_desc(transfer, error_sink);
    if (transfer_status != SLLM_STATUS_OK) {
      return transfer_status;
    }
    if (transfer->host_pointer == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "H2D transfer host pointer is null");
    }
    if (queue == nullptr || buffer == nullptr || completion == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "copy queue, buffer, or completion output is null");
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in public H2D copy");
  }
}

extern "C" sllm_status_t
sllm_buffer_copy_d2h(const sllm_queue_t *const queue,
                     const sllm_buffer_t *const buffer,
                     const sllm_transfer_desc_t *const transfer,
                     sllm_completion_t **const completion,
                     sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (completion != nullptr) {
      *completion = nullptr;
    }
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t transfer_status =
        validate_transfer_desc(transfer, error_sink);
    if (transfer_status != SLLM_STATUS_OK) {
      return transfer_status;
    }
    if (queue == nullptr || buffer == nullptr || completion == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "copy queue, buffer, or completion output is null");
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in public D2H copy");
  }
}

extern "C" sllm_status_t
sllm_completion_query(sllm_completion_t *const completion,
                      sllm_completion_result_t *const result,
                      sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t result_status =
        validate_completion_result(result, error_sink);
    if (result_status != SLLM_STATUS_OK) {
      return result_status;
    }
    if (completion == nullptr) {
      return write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                         "completion handle is null");
    }
    result->state = SLLM_COMPLETION_STATE_FAILURE;
    return write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                       "completion handle is not owned by the public runtime");
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in public completion query");
  }
}

extern "C" sllm_status_t
sllm_completion_wait(sllm_completion_t *const completion,
                     const uint32_t /*timeout_ms*/,
                     sllm_completion_result_t *const result,
                     sllm_error_sink_t *const error_sink) noexcept {
  return sllm_completion_query(completion, result, error_sink);
}

extern "C" sllm_status_t sllm_completion_read(
    sllm_completion_t *const completion, void *const destination,
    const uint64_t /*destination_capacity*/, uint64_t *const bytes_written,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (completion == nullptr || bytes_written == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "completion, destination, or output is null");
    }
    *bytes_written = 0U;
    (void)destination;
    return write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                       "completion handle is not owned by the public runtime");
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in public completion read");
  }
}

extern "C" sllm_status_t
sllm_completion_timing(sllm_completion_t *const /*completion*/,
                       sllm_completion_timing_t *const timing,
                       sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t timing_status =
        validate_struct(timing, error_sink, "completion timing output is null");
    if (timing_status != SLLM_STATUS_OK) {
      return timing_status;
    }
    if (timing->reserved0 != 0U || timing->reserved[0] != 0U ||
        timing->reserved[1] != 0U || timing->reserved[2] != 0U ||
        timing->reserved[3] != 0U) {
      return write_error(error_sink, SLLM_STATUS_RESERVED_NONZERO,
                         "completion timing reserved fields must be zero");
    }
    timing->valid = 0U;
    timing->elapsed_ns = 0U;
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in public completion timing");
  }
}

extern "C" sllm_status_t
sllm_completion_release(sllm_completion_t **const completion,
                        sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (completion == nullptr || *completion == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "completion handle is null");
    }
    return write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                       "completion handle is not owned by the public runtime");
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in public completion release");
  }
}

extern "C" sllm_status_t
sllm_rmsnorm_prepare(const sllm_context_t *const context,
                     const sllm_rmsnorm_desc_t *const descriptor,
                     sllm_rmsnorm_plan_t **const plan,
                     sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (plan != nullptr) {
      *plan = nullptr;
    }
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (plan == nullptr || context == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "RMSNorm context or plan output is null");
    }
    sllm_rmsnorm::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_rmsnorm::validate_and_copy_descriptor(descriptor, &metadata,
                                                   error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in RMSNorm prepare stub");
  }
}

extern "C" sllm_status_t
sllm_rmsnorm_plan_release(sllm_rmsnorm_plan_t **const plan,
                          sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (plan == nullptr || *plan == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "RMSNorm plan handle is null");
    }
    return write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                       "RMSNorm plan is not owned by the unavailable stub");
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in RMSNorm plan release stub");
  }
}

extern "C" sllm_status_t
sllm_rmsnorm_execute(const sllm_rmsnorm_plan_t *const plan,
                     const sllm_queue_t *const queue,
                     sllm_completion_t **const completion,
                     sllm_rmsnorm_dispatch_info_t *const dispatch_info,
                     sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (plan == nullptr || queue == nullptr || completion == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "RMSNorm execute input or completion output is null");
    }
    const sllm_status_t info_status =
        validate_dispatch_info(dispatch_info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in RMSNorm execute stub");
  }
}
