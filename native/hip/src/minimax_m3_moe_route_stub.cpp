#include "minimax_m3_moe_route_api.hpp"

namespace {

sllm_status_t unavailable(sllm_error_sink_t *const sink) noexcept {
  return sllm_public_runtime::write_error(
      sink, SLLM_STATUS_HIP_UNAVAILABLE,
      "public HIP runtime is unavailable; CPU fallback is disabled");
}

} // namespace

extern "C" sllm_status_t sllm_minimax_m3_moe_route_prepare(
    const sllm_context_t *const context,
    const sllm_minimax_m3_moe_route_desc_t *const descriptor,
    sllm_minimax_m3_moe_route_plan_t **const plan,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (plan != nullptr) {
      *plan = nullptr;
    }
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (context == nullptr || plan == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "MiniMax M3 route context or plan output is null");
    }
    sllm_minimax_m3_moe_route::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_minimax_m3_moe_route::validate_and_copy_descriptor(
            descriptor, &metadata, error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }
    return unavailable(error_sink);
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in MiniMax M3 route prepare stub");
  }
}

extern "C" sllm_status_t sllm_minimax_m3_moe_route_plan_release(
    sllm_minimax_m3_moe_route_plan_t **const plan,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (plan == nullptr || *plan == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "MiniMax M3 route plan handle is null");
    }
    return unavailable(error_sink);
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in MiniMax M3 route release stub");
  }
}

extern "C" sllm_status_t sllm_minimax_m3_moe_route_execute(
    const sllm_minimax_m3_moe_route_plan_t *const plan,
    const sllm_queue_t *const queue, sllm_completion_t **const completion,
    sllm_minimax_m3_moe_route_dispatch_info_t *const dispatch_info,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (completion != nullptr) {
      *completion = nullptr;
    }
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (plan == nullptr || queue == nullptr || completion == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "MiniMax M3 route execute input or output is null");
    }
    const sllm_status_t dispatch_status =
        sllm_minimax_m3_moe_route::validate_dispatch_info(dispatch_info,
                                                          error_sink);
    if (dispatch_status != SLLM_STATUS_OK) {
      return dispatch_status;
    }
    return unavailable(error_sink);
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in MiniMax M3 route execute stub");
  }
}
