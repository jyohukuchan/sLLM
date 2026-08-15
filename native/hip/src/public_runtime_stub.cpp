#include "argmax_api.hpp"
#include "attention_preprocess_api.hpp"
#include "causal_attention_api.hpp"
#include "elementwise_api.hpp"
#include "embedding_api.hpp"
#include "evidence_abi.h"
#include "kv_state_api.hpp"
#include "linear_attention_api.hpp"
#include "matmul_api.hpp"
#include "public_runtime_internal.hpp"
#include "rmsnorm_api.hpp"
#include "rotary_api.hpp"
#include "windowed_attention_api.hpp"

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
  if (info->reserved0 != 0U || info->available_memory_bytes != 0U ||
      info->reserved[0] != 0U || info->reserved[1] != 0U) {
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

sllm_status_t
validate_dispatch_info(const sllm_elementwise_dispatch_info_t *const info,
                       sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "elementwise dispatch info output is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, info, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_elementwise_dispatch_info_t)) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "elementwise dispatch info struct size is unsupported");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return write_error(sink, SLLM_STATUS_INVALID_ABI_VERSION,
                       "elementwise dispatch info ABI is unsupported");
  }
  if (info->info_version != SLLM_HIP_ELEMENTWISE_DISPATCH_INFO_VERSION) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "elementwise dispatch info version is unsupported");
  }
  for (const uint32_t value : info->reserved) {
    if (value != 0U) {
      return write_error(
          sink, SLLM_STATUS_RESERVED_NONZERO,
          "elementwise dispatch info reserved fields must be zero");
    }
  }
  return SLLM_STATUS_OK;
}

sllm_status_t validate_kv_view_info(const sllm_kv_view_info_t *const info,
                                    sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "KV view info output is null");
  }
  if (info->struct_size != sizeof(*info)) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "KV view info struct size is unsupported");
  }
  if (info->abi_version != SLLM_HIP_ABI_VERSION) {
    return write_error(sink, SLLM_STATUS_INVALID_ABI_VERSION,
                       "KV view info ABI is unsupported");
  }
  if (info->info_version != SLLM_HIP_KV_VIEW_INFO_VERSION ||
      info->reserved0 != 0U || info->reserved1 != 0U) {
    return write_error(sink, SLLM_STATUS_RESERVED_NONZERO,
                       "KV view info version or reserved is invalid");
  }
  for (const uint32_t value : info->reserved) {
    if (value != 0U) {
      return write_error(sink, SLLM_STATUS_RESERVED_NONZERO,
                         "KV view info reserved fields must be zero");
    }
  }
  return SLLM_STATUS_OK;
}

sllm_status_t validate_kv_append_info(const sllm_kv_append_info_t *const info,
                                      sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "KV append info output is null");
  }
  if (info->struct_size != sizeof(*info)) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "KV append info struct size is unsupported");
  }
  if (info->abi_version != SLLM_HIP_ABI_VERSION) {
    return write_error(sink, SLLM_STATUS_INVALID_ABI_VERSION,
                       "KV append info ABI is unsupported");
  }
  if (info->info_version != SLLM_HIP_KV_APPEND_INFO_VERSION ||
      info->reserved0 != 0U) {
    return write_error(sink, SLLM_STATUS_RESERVED_NONZERO,
                       "KV append info version or reserved is invalid");
  }
  for (const uint32_t value : info->reserved) {
    if (value != 0U) {
      return write_error(sink, SLLM_STATUS_RESERVED_NONZERO,
                         "KV append info reserved fields must be zero");
    }
  }
  return SLLM_STATUS_OK;
}

sllm_status_t validate_dispatch_info(
    const sllm_attention_preprocess_dispatch_info_t *const info,
    sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "attention preprocess dispatch info output is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, info, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_attention_preprocess_dispatch_info_t)) {
    return write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "attention preprocess dispatch info struct size is unsupported");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return write_error(sink, SLLM_STATUS_INVALID_ABI_VERSION,
                       "attention preprocess dispatch info ABI is unsupported");
  }
  if (info->info_version !=
      SLLM_HIP_ATTENTION_PREPROCESS_DISPATCH_INFO_VERSION) {
    return write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "attention preprocess dispatch info version is unsupported");
  }
  for (const uint32_t value : info->reserved) {
    if (value != 0U) {
      return write_error(
          sink, SLLM_STATUS_RESERVED_NONZERO,
          "attention preprocess dispatch info reserved is invalid");
    }
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_dispatch_info(const sllm_embedding_dispatch_info_t *const info,
                       sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "embedding dispatch info output is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, info, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_embedding_dispatch_info_t)) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "embedding dispatch info struct size is unsupported");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return write_error(sink, SLLM_STATUS_INVALID_ABI_VERSION,
                       "embedding dispatch info ABI is unsupported");
  }
  if (info->info_version != SLLM_HIP_EMBEDDING_DISPATCH_INFO_VERSION) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "embedding dispatch info version is unsupported");
  }
  for (const uint32_t value : info->reserved) {
    if (value != 0U) {
      return write_error(
          sink, SLLM_STATUS_RESERVED_NONZERO,
          "embedding dispatch info reserved fields must be zero");
    }
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_dispatch_info(const sllm_rotary_dispatch_info_t *const info,
                       sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "rotary dispatch info output is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, info, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_rotary_dispatch_info_t)) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "rotary dispatch info struct size is unsupported");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return write_error(sink, SLLM_STATUS_INVALID_ABI_VERSION,
                       "rotary dispatch info ABI is unsupported");
  }
  if (info->info_version != SLLM_HIP_ROTARY_DISPATCH_INFO_VERSION) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "rotary dispatch info version is unsupported");
  }
  for (const uint32_t value : info->reserved) {
    if (value != 0U) {
      return write_error(sink, SLLM_STATUS_RESERVED_NONZERO,
                         "rotary dispatch info reserved fields must be zero");
    }
  }
  return SLLM_STATUS_OK;
}

sllm_status_t validate_dispatch_info(
    const sllm_windowed_attention_dispatch_info_t *const info,
    sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "windowed attention dispatch info output is null");
  }
  uint32_t prefix[2]{};
  std::memcpy(prefix, info, sizeof(prefix));
  if (prefix[0] != sizeof(*info)) {
    return write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "windowed attention dispatch info struct size is unsupported");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return write_error(sink, SLLM_STATUS_INVALID_ABI_VERSION,
                       "windowed attention dispatch info ABI is unsupported");
  }
  if (info->info_version != SLLM_HIP_WINDOWED_ATTENTION_DISPATCH_INFO_VERSION) {
    return write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "windowed attention dispatch info version is unsupported");
  }
  for (const uint32_t value : info->reserved) {
    if (value != 0U) {
      return write_error(
          sink, SLLM_STATUS_RESERVED_NONZERO,
          "windowed attention dispatch info reserved fields must be zero");
    }
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_dispatch_info(const sllm_matmul_dispatch_info_t *const info,
                       sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "matmul dispatch info output is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, info, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_matmul_dispatch_info_t)) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "matmul dispatch info struct size is unsupported");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return write_error(sink, SLLM_STATUS_INVALID_ABI_VERSION,
                       "matmul dispatch info ABI is unsupported");
  }
  if (info->info_version != SLLM_HIP_MATMUL_DISPATCH_INFO_VERSION) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "matmul dispatch info version is unsupported");
  }
  for (const uint32_t value : info->reserved) {
    if (value != 0U) {
      return write_error(sink, SLLM_STATUS_RESERVED_NONZERO,
                         "matmul dispatch info reserved is invalid");
    }
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

extern "C" sllm_status_t
sllm_elementwise_prepare(const sllm_context_t *const context,
                         const sllm_elementwise_desc_t *const descriptor,
                         sllm_elementwise_plan_t **const plan,
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
                         "elementwise context or plan output is null");
    }
    sllm_elementwise::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_elementwise::validate_and_copy_descriptor(descriptor, &metadata,
                                                       error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in elementwise prepare stub");
  }
}

extern "C" sllm_status_t
sllm_elementwise_plan_release(sllm_elementwise_plan_t **const plan,
                              sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (plan == nullptr || *plan == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "elementwise plan handle is null");
    }
    return write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                       "elementwise plan is not owned by the unavailable stub");
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in elementwise plan release stub");
  }
}

extern "C" sllm_status_t
sllm_elementwise_execute(const sllm_elementwise_plan_t *const plan,
                         const sllm_queue_t *const queue,
                         sllm_completion_t **const completion,
                         sllm_elementwise_dispatch_info_t *const dispatch_info,
                         sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (plan == nullptr || queue == nullptr || completion == nullptr) {
      return write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "elementwise execute input or completion output is null");
    }
    const sllm_status_t info_status =
        validate_dispatch_info(dispatch_info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in elementwise execute stub");
  }
}

extern "C" sllm_status_t
sllm_embedding_prepare(const sllm_context_t *const context,
                       const sllm_embedding_desc_t *const descriptor,
                       sllm_embedding_plan_t **const plan,
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
                         "embedding context or plan output is null");
    }
    sllm_embedding::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_embedding::validate_and_copy_descriptor(descriptor, &metadata,
                                                     error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in embedding prepare stub");
  }
}

extern "C" sllm_status_t
sllm_embedding_plan_release(sllm_embedding_plan_t **const plan,
                            sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (plan == nullptr || *plan == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "embedding plan handle is null");
    }
    return write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                       "embedding plan is not owned by the unavailable stub");
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in embedding plan release stub");
  }
}

extern "C" sllm_status_t
sllm_embedding_execute(const sllm_embedding_plan_t *const plan,
                       const sllm_queue_t *const queue,
                       sllm_completion_t **const completion,
                       sllm_embedding_dispatch_info_t *const dispatch_info,
                       sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (plan == nullptr || queue == nullptr || completion == nullptr) {
      return write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "embedding execute input or completion output is null");
    }
    const sllm_status_t info_status =
        validate_dispatch_info(dispatch_info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in embedding execute stub");
  }
}

extern "C" sllm_status_t
sllm_matmul_prepare(const sllm_context_t *const context,
                    const sllm_matmul_desc_t *const descriptor,
                    sllm_matmul_plan_t **const plan,
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
                         "matmul context or plan output is null");
    }
    sllm_matmul::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_matmul::validate_and_copy_descriptor(descriptor, &metadata,
                                                  error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in matmul prepare stub");
  }
}

extern "C" sllm_status_t
sllm_matmul_plan_release(sllm_matmul_plan_t **const plan,
                         sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (plan == nullptr || *plan == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "matmul plan handle is null");
    }
    return write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                       "matmul plan is not owned by the unavailable stub");
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in matmul plan release stub");
  }
}

extern "C" sllm_status_t
sllm_matmul_execute(const sllm_matmul_plan_t *const plan,
                    const sllm_queue_t *const queue,
                    sllm_completion_t **const completion,
                    sllm_matmul_dispatch_info_t *const dispatch_info,
                    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (plan == nullptr || queue == nullptr || completion == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "matmul execute input or completion output is null");
    }
    const sllm_status_t info_status =
        validate_dispatch_info(dispatch_info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in matmul execute stub");
  }
}

extern "C" sllm_status_t
sllm_argmax_prepare(const sllm_context_t *const context,
                    const sllm_argmax_desc_t *const descriptor,
                    sllm_argmax_plan_t **const plan,
                    sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (plan != nullptr) {
      *plan = nullptr;
    }
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (context == nullptr || plan == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "argmax context or plan output is null");
    }
    sllm_argmax::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_argmax::validate_and_copy_descriptor(descriptor, &metadata,
                                                  error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in argmax prepare stub");
  }
}

extern "C" sllm_status_t
sllm_argmax_plan_release(sllm_argmax_plan_t **const plan,
                         sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (plan == nullptr || *plan == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "argmax plan handle is null");
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in argmax release stub");
  }
}

extern "C" sllm_status_t
sllm_argmax_execute(const sllm_argmax_plan_t *const plan,
                    const sllm_queue_t *const queue,
                    sllm_completion_t **const completion,
                    sllm_argmax_dispatch_info_t *const dispatch_info,
                    sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (completion != nullptr) {
      *completion = nullptr;
    }
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (plan == nullptr || queue == nullptr || completion == nullptr ||
        dispatch_info == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "argmax execute input or output is null");
    }
    if (dispatch_info->struct_size != sizeof(*dispatch_info) ||
        dispatch_info->abi_version != SLLM_HIP_ABI_VERSION ||
        dispatch_info->info_version != SLLM_HIP_ARGMAX_DISPATCH_INFO_VERSION ||
        !std::all_of(std::begin(dispatch_info->reserved),
                     std::end(dispatch_info->reserved),
                     [](const uint32_t value) { return value == 0U; })) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "argmax dispatch info is unsupported");
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in argmax execute stub");
  }
}

extern "C" sllm_status_t sllm_attention_preprocess_prepare(
    const sllm_context_t *const context,
    const sllm_attention_preprocess_desc_t *const descriptor,
    sllm_attention_preprocess_plan_t **const plan,
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
                         "attention preprocess context or plan output is null");
    }
    sllm_attention_preprocess::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_attention_preprocess::validate_and_copy_descriptor(
            descriptor, &metadata, error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in attention preprocess prepare stub");
  }
}

extern "C" sllm_status_t sllm_attention_preprocess_plan_release(
    sllm_attention_preprocess_plan_t **const plan,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (plan == nullptr || *plan == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "attention preprocess plan handle is null");
    }
    return write_error(
        error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
        "attention preprocess plan is not owned by the unavailable stub");
  } catch (...) {
    return write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in attention preprocess plan release stub");
  }
}

extern "C" sllm_status_t sllm_attention_preprocess_execute(
    const sllm_attention_preprocess_plan_t *const plan,
    const sllm_queue_t *const queue, sllm_completion_t **const completion,
    sllm_attention_preprocess_dispatch_info_t *const dispatch_info,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (plan == nullptr || queue == nullptr || completion == nullptr) {
      return write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "attention preprocess execute input or completion output is null");
    }
    const sllm_status_t info_status =
        validate_dispatch_info(dispatch_info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in attention preprocess execute stub");
  }
}

extern "C" sllm_status_t
sllm_rotary_prepare(const sllm_context_t *const context,
                    const sllm_rotary_desc_t *const descriptor,
                    sllm_rotary_plan_t **const plan,
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
                         "rotary context or plan output is null");
    }
    sllm_rotary::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_rotary::validate_and_copy_descriptor(descriptor, &metadata,
                                                  error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in rotary prepare stub");
  }
}

extern "C" sllm_status_t
sllm_rotary_plan_release(sllm_rotary_plan_t **const plan,
                         sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (plan == nullptr || *plan == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "rotary plan handle is null");
    }
    return write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                       "rotary plan is not owned by the unavailable stub");
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in rotary plan release stub");
  }
}

extern "C" sllm_status_t
sllm_rotary_execute(const sllm_rotary_plan_t *const plan,
                    const sllm_queue_t *const queue,
                    sllm_completion_t **const completion,
                    sllm_rotary_dispatch_info_t *const dispatch_info,
                    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (plan == nullptr || queue == nullptr || completion == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "rotary execute input or completion output is null");
    }
    const sllm_status_t info_status =
        validate_dispatch_info(dispatch_info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in rotary execute stub");
  }
}

extern "C" sllm_status_t sllm_windowed_attention_prepare(
    const sllm_context_t *const context,
    const sllm_windowed_attention_desc_t *const descriptor,
    sllm_windowed_attention_plan_t **const plan,
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
                         "windowed attention context or plan output is null");
    }
    sllm_windowed_attention::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_windowed_attention::validate_and_copy_descriptor(
            descriptor, &metadata, error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in windowed attention prepare stub");
  }
}

extern "C" sllm_status_t sllm_windowed_attention_plan_release(
    sllm_windowed_attention_plan_t **const plan,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (plan == nullptr || *plan == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "windowed attention plan handle is null");
    }
    return write_error(
        error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
        "windowed attention plan is not owned by the unavailable stub");
  } catch (...) {
    return write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in windowed attention plan release stub");
  }
}

extern "C" sllm_status_t sllm_windowed_attention_execute(
    const sllm_windowed_attention_plan_t *const plan,
    const sllm_queue_t *const queue, sllm_completion_t **const completion,
    sllm_windowed_attention_dispatch_info_t *const dispatch_info,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (plan == nullptr || queue == nullptr || completion == nullptr) {
      return write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "windowed attention execute input or completion output is null");
    }
    const sllm_status_t info_status =
        validate_dispatch_info(dispatch_info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in windowed attention execute stub");
  }
}

extern "C" sllm_status_t
sllm_kv_state_create(const sllm_context_t *const context,
                     const sllm_kv_state_create_info_t *const info,
                     sllm_kv_state_t **const state,
                     sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (state != nullptr) {
      *state = nullptr;
    }
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t info_status =
        sllm_kv_state::validate_state_create_info(info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    if (context == nullptr || state == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "KV state context or output is null");
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in KV state create stub");
  }
}

extern "C" sllm_status_t
sllm_kv_state_release(sllm_kv_state_t **const state,
                      sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (state == nullptr || *state == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "KV state handle is null");
    }
    return write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                       "KV state is not owned by the unavailable stub");
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in KV state release stub");
  }
}

extern "C" sllm_status_t
sllm_kv_state_query(const sllm_kv_state_t *const state,
                    sllm_kv_view_info_t *const info,
                    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t info_status = validate_kv_view_info(info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    sllm_kv_state::initialize_view_info(info);
    if (state == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "KV state handle is null");
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in KV state query stub");
  }
}

extern "C" sllm_status_t
sllm_kv_state_snapshot(const sllm_kv_state_t *const state,
                       sllm_kv_view_t **const view,
                       sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (view != nullptr) {
      *view = nullptr;
    }
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (state == nullptr || view == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "KV state or snapshot output is null");
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in KV state snapshot stub");
  }
}

extern "C" sllm_status_t
sllm_kv_view_query(const sllm_kv_view_t *const view,
                   sllm_kv_view_info_t *const info,
                   sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t info_status = validate_kv_view_info(info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    sllm_kv_state::initialize_view_info(info);
    if (view == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "KV snapshot handle is null");
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in KV snapshot query stub");
  }
}

extern "C" sllm_status_t
sllm_kv_view_release(sllm_kv_view_t **const view,
                     sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (view == nullptr || *view == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "KV snapshot handle is null");
    }
    return write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                       "KV snapshot is not owned by the unavailable stub");
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in KV snapshot release stub");
  }
}

extern "C" sllm_status_t
sllm_kv_state_append(const sllm_kv_state_t *const state,
                     const sllm_queue_t *const queue,
                     const sllm_kv_append_desc_t *const descriptor,
                     sllm_completion_t **const completion,
                     sllm_kv_append_info_t *const append_info,
                     sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (completion != nullptr) {
      *completion = nullptr;
    }
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t info_status =
        validate_kv_append_info(append_info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    sllm_kv_state::AppendMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_kv_state::validate_and_copy_append(descriptor, &metadata,
                                                error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }
    sllm_kv_state::initialize_append_info(append_info);
    if (state == nullptr || queue == nullptr || completion == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "KV append input or completion output is null");
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in KV state append stub");
  }
}

extern "C" sllm_status_t
sllm_kv_state_append_cancel(const sllm_kv_state_t *const state,
                            sllm_completion_t *const completion,
                            sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (state == nullptr || completion == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "KV append cancel input is null");
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in KV append cancel stub");
  }
}

extern "C" sllm_status_t sllm_causal_attention_execute(
    const sllm_context_t *const context, const sllm_queue_t *const queue,
    const sllm_causal_attention_desc_t *const descriptor,
    sllm_completion_t **const completion,
    sllm_causal_attention_dispatch_info_t *const dispatch_info,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (completion != nullptr) {
      *completion = nullptr;
    }
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (dispatch_info == nullptr ||
        dispatch_info->struct_size != sizeof(*dispatch_info)) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "causal attention dispatch info is invalid");
    }
    if (dispatch_info->abi_version != SLLM_HIP_ABI_VERSION) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ABI_VERSION,
                         "causal attention dispatch info ABI is unsupported");
    }
    if (dispatch_info->info_version !=
            SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION ||
        !std::all_of(std::begin(dispatch_info->reserved),
                     std::end(dispatch_info->reserved),
                     [](const uint32_t value) { return value == 0U; })) {
      return write_error(error_sink, SLLM_STATUS_RESERVED_NONZERO,
                         "causal attention dispatch info is unsupported");
    }
    const sllm_status_t descriptor_status =
        sllm_causal_attention::validate_descriptor_prefix(descriptor,
                                                          error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }
    if (context == nullptr || queue == nullptr || completion == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "causal attention input or completion output is null");
    }
    sllm_causal_attention::initialize_dispatch_info(dispatch_info);
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in causal attention stub");
  }
}

extern "C" sllm_status_t sllm_linear_attention_state_create(
    const sllm_context_t *const context,
    const sllm_linear_attention_state_create_info_t *const info,
    sllm_linear_attention_state_t **const state,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (state != nullptr) {
      *state = nullptr;
    }
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t info_status =
        sllm_linear_attention::validate_state_create_info(info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    if (context == nullptr || state == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "linear attention state context or output is null");
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in linear attention state create stub");
  }
}

extern "C" sllm_status_t sllm_linear_attention_state_release(
    sllm_linear_attention_state_t **const state,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (state == nullptr || *state == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "linear attention state handle is null");
    }
    return write_error(
        error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
        "linear attention state is not owned by the unavailable stub");
  } catch (...) {
    return write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in linear attention state release stub");
  }
}

extern "C" sllm_status_t sllm_linear_attention_state_query(
    const sllm_linear_attention_state_t *const state,
    sllm_linear_attention_view_info_t *const info,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (info == nullptr || info->struct_size != sizeof(*info)) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "linear attention view info is invalid");
    }
    if (info->abi_version != SLLM_HIP_ABI_VERSION) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ABI_VERSION,
                         "linear attention view info ABI is unsupported");
    }
    if (info->info_version != SLLM_HIP_LINEAR_ATTENTION_VIEW_INFO_VERSION ||
        info->reserved0 != 0U ||
        !std::all_of(std::begin(info->reserved), std::end(info->reserved),
                     [](const uint32_t value) { return value == 0U; })) {
      return write_error(error_sink, SLLM_STATUS_RESERVED_NONZERO,
                         "linear attention view info is unsupported");
    }
    sllm_linear_attention::initialize_view_info(info);
    if (state == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "linear attention state handle is null");
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in linear attention query stub");
  }
}

extern "C" sllm_status_t sllm_linear_attention_execute(
    const sllm_context_t *const context, const sllm_queue_t *const queue,
    const sllm_linear_attention_desc_t *const descriptor,
    sllm_completion_t **const completion,
    sllm_linear_attention_dispatch_info_t *const dispatch_info,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (completion != nullptr) {
      *completion = nullptr;
    }
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (dispatch_info == nullptr ||
        dispatch_info->struct_size != sizeof(*dispatch_info)) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "linear attention dispatch info is invalid");
    }
    if (dispatch_info->abi_version != SLLM_HIP_ABI_VERSION) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ABI_VERSION,
                         "linear attention dispatch info ABI is unsupported");
    }
    if (dispatch_info->info_version !=
            SLLM_HIP_LINEAR_ATTENTION_DISPATCH_INFO_VERSION ||
        !std::all_of(std::begin(dispatch_info->reserved),
                     std::end(dispatch_info->reserved),
                     [](const uint32_t value) { return value == 0U; })) {
      return write_error(error_sink, SLLM_STATUS_RESERVED_NONZERO,
                         "linear attention dispatch info is unsupported");
    }
    sllm_linear_attention::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_linear_attention::validate_and_copy_descriptor(
            descriptor, &metadata, error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }
    if (context == nullptr || queue == nullptr || completion == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "linear attention input or completion output is null");
    }
    sllm_linear_attention::initialize_dispatch_info(dispatch_info);
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in linear attention execute stub");
  }
}

extern "C" sllm_status_t
sllm_linear_attention_cancel(const sllm_linear_attention_state_t *const state,
                             sllm_completion_t *const completion,
                             sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (state == nullptr || completion == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "linear attention cancel input is null");
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in linear attention cancel stub");
  }
}

extern "C" sllm_status_t
sllm_hip_kv_view_readback(const sllm_hip_kv_readback_request_t *const request,
                          sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status = validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (request == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "KV evidence readback request is null");
    }
    if (request->struct_size != sizeof(*request)) {
      return write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "KV evidence readback request struct size is unsupported");
    }
    if (request->abi_version != SLLM_HIP_KV_EVIDENCE_ABI_VERSION) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ABI_VERSION,
                         "KV evidence readback ABI is unsupported");
    }
    if (request->reserved0 != 0U || request->reserved[0] != 0U ||
        request->reserved[1] != 0U || request->reserved[2] != 0U ||
        request->reserved[3] != 0U) {
      return write_error(error_sink, SLLM_STATUS_RESERVED_NONZERO,
                         "KV evidence readback reserved fields must be zero");
    }
    if (request->view == nullptr ||
        (request->plane != SLLM_HIP_KV_EVIDENCE_PLANE_K &&
         request->plane != SLLM_HIP_KV_EVIDENCE_PLANE_V) ||
        request->byte_length == 0U || request->host_output == nullptr) {
      return write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                         "KV evidence readback request is invalid");
    }
    if (request->byte_offset % UINT64_C(2) != 0U ||
        request->byte_length % UINT64_C(2) != 0U) {
      return write_error(error_sink, SLLM_STATUS_MISALIGNED_OFFSET,
                         "KV evidence readback range must be FP16-aligned");
    }
    if (request->byte_length > SLLM_HIP_KV_EVIDENCE_MAX_READBACK_BYTES ||
        request->byte_offset > UINT64_MAX - request->byte_length) {
      return write_error(error_sink, SLLM_STATUS_METADATA_OVERFLOW,
                         "KV evidence readback range overflows its bound");
    }
    if (request->host_capacity < request->byte_length) {
      return write_error(error_sink, SLLM_STATUS_BUFFER_TOO_SMALL,
                         "KV evidence readback host output is undersized");
    }
    return unavailable(error_sink);
  } catch (...) {
    return write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                       "unexpected exception in KV evidence readback stub");
  }
}
