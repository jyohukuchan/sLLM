#include "sllm/sllm.h"

int sllm_header_cpp_compile() {
  char message[8] = {};
  sllm_error_sink_t sink = SLLM_ERROR_SINK_INIT(message);
  sllm_access_mode_t access = SLLM_ACCESS_READ;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *buffer = nullptr;
  sllm_event_t *event = nullptr;
  sllm_completion_t *completion = nullptr;
  sllm_rmsnorm_plan_t *rmsnorm_plan = nullptr;
  sllm_device_info_t device{};
  sllm_context_create_info_t context_info{};
  sllm_queue_create_info_t queue_info{};
  sllm_buffer_create_info_t buffer_info{};
  sllm_transfer_desc_t transfer{};
  sllm_completion_result_t completion_result{};
  sllm_tensor_binding_t binding{};
  sllm_rmsnorm_desc_t rmsnorm{};
  using ContextCreateFn =
      sllm_status_t (*)(const sllm_context_create_info_t *, sllm_context_t **,
                        sllm_error_sink_t *) noexcept;
  const ContextCreateFn context_create = &sllm_context_create;
  using RmsNormPrepareFn =
      sllm_status_t (*)(const sllm_context_t *, const sllm_rmsnorm_desc_t *,
                        sllm_rmsnorm_plan_t **, sllm_error_sink_t *) noexcept;
  const RmsNormPrepareFn rmsnorm_prepare = &sllm_rmsnorm_prepare;
  using RmsNormReleaseFn =
      sllm_status_t (*)(sllm_rmsnorm_plan_t **, sllm_error_sink_t *) noexcept;
  const RmsNormReleaseFn rmsnorm_release = &sllm_rmsnorm_plan_release;
  (void)context_create;
  (void)rmsnorm_prepare;
  (void)rmsnorm_release;

  return sink.abi_version == SLLM_HIP_ABI_VERSION && access != 0U &&
                 queue == nullptr && buffer == nullptr && event == nullptr &&
                 completion == nullptr && device.struct_size == 0U &&
                 context_info.flags == 0U && queue_info.flags == 0U &&
                 buffer_info.size_bytes == 0U && transfer.size_bytes == 0U &&
                 completion_result.state == 0U && rmsnorm_plan == nullptr &&
                 binding.rank == 0U && rmsnorm.op_version == 0U
             ? 0
             : 1;
}
