#include "sllm/sllm.h"

int sllm_header_cpp_compile() {
  char message[8] = {};
  sllm_error_sink_t sink = SLLM_ERROR_SINK_INIT(message);
  sllm_access_mode_t access = SLLM_ACCESS_READ;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *buffer = nullptr;
  sllm_event_t *event = nullptr;
  sllm_completion_t *completion = nullptr;

  return sink.abi_version == SLLM_HIP_ABI_VERSION && access != 0U &&
                 queue == nullptr && buffer == nullptr && event == nullptr &&
                 completion == nullptr
             ? 0
             : 1;
}
