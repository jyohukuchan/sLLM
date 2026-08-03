#include "sllm/sllm.h"

int sllm_header_c_compile(void) {
  char message[8] = {0};
  sllm_error_sink_t sink = SLLM_ERROR_SINK_INIT(message);
  sllm_access_mode_t access = SLLM_ACCESS_READ_WRITE;
  sllm_queue_t *queue = 0;
  sllm_buffer_t *buffer = 0;
  sllm_event_t *event = 0;
  sllm_completion_t *completion = 0;

  return sink.abi_version == SLLM_HIP_ABI_VERSION && access != 0U &&
                 queue == 0 && buffer == 0 && event == 0 && completion == 0
             ? 0
             : 1;
}
