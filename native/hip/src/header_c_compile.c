#include "ullm/ullm.h"

int ullm_header_c_compile(void) {
  char message[8] = {0};
  ullm_error_sink_t sink = ULLM_ERROR_SINK_INIT(message);
  ullm_access_mode_t access = ULLM_ACCESS_READ_WRITE;
  ullm_queue_t *queue = 0;
  ullm_buffer_t *buffer = 0;
  ullm_event_t *event = 0;
  ullm_completion_t *completion = 0;

  return sink.abi_version == ULLM_HIP_ABI_VERSION && access != 0U &&
                 queue == 0 && buffer == 0 && event == 0 && completion == 0
             ? 0
             : 1;
}
