#include "ullm/ullm.h"

int ullm_header_cpp_compile() {
  char message[8] = {};
  ullm_error_sink_t sink = ULLM_ERROR_SINK_INIT(message);
  ullm_access_mode_t access = ULLM_ACCESS_READ;
  ullm_queue_t *queue = nullptr;
  ullm_buffer_t *buffer = nullptr;
  ullm_event_t *event = nullptr;
  ullm_completion_t *completion = nullptr;

  return sink.abi_version == ULLM_HIP_ABI_VERSION && access != 0U &&
                 queue == nullptr && buffer == nullptr && event == nullptr &&
                 completion == nullptr
             ? 0
             : 1;
}
