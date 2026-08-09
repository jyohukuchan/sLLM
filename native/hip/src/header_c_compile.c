#include "sllm/sllm.h"

int sllm_header_c_compile(void) {
  char message[8] = {0};
  sllm_error_sink_t sink = SLLM_ERROR_SINK_INIT(message);
  sllm_access_mode_t access = SLLM_ACCESS_READ_WRITE;
  sllm_queue_t *queue = 0;
  sllm_buffer_t *buffer = 0;
  sllm_event_t *event = 0;
  sllm_completion_t *completion = 0;
  sllm_rmsnorm_plan_t *rmsnorm_plan = 0;
  sllm_device_info_t device = {0};
  sllm_context_create_info_t context_info = {0};
  sllm_queue_create_info_t queue_info = {0};
  sllm_buffer_create_info_t buffer_info = {0};
  sllm_transfer_desc_t transfer = {0};
  sllm_completion_result_t completion_result = {0};
  sllm_tensor_binding_t binding = {0};
  sllm_rmsnorm_desc_t rmsnorm = {0};
  sllm_status_t (*rmsnorm_release)(
      sllm_rmsnorm_plan_t **, sllm_error_sink_t *) = &sllm_rmsnorm_plan_release;

  return sink.abi_version == SLLM_HIP_ABI_VERSION && access != 0U &&
                 queue == 0 && buffer == 0 && event == 0 && completion == 0 &&
                 device.struct_size == 0U && context_info.flags == 0U &&
                 queue_info.flags == 0U && buffer_info.size_bytes == 0U &&
                 transfer.size_bytes == 0U && completion_result.state == 0U &&
                 rmsnorm_plan == 0 && binding.rank == 0U &&
                 rmsnorm.op_version == 0U && rmsnorm_release != 0
             ? 0
             : 1;
}
