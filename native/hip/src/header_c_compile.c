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
  sllm_elementwise_plan_t *elementwise_plan = 0;
  sllm_embedding_plan_t *embedding_plan = 0;
  sllm_matmul_plan_t *matmul_plan = 0;
  sllm_attention_preprocess_plan_t *attention_preprocess_plan = 0;
  sllm_kv_state_t *kv_state = 0;
  sllm_kv_view_t *kv_view = 0;
  sllm_device_info_t device = {0};
  sllm_context_create_info_t context_info = {0};
  sllm_queue_create_info_t queue_info = {0};
  sllm_buffer_create_info_t buffer_info = {0};
  sllm_transfer_desc_t transfer = {0};
  sllm_completion_result_t completion_result = {0};
  sllm_tensor_binding_t binding = {0};
  sllm_rmsnorm_desc_t rmsnorm = {0};
  sllm_elementwise_desc_t elementwise = {0};
  sllm_embedding_desc_t embedding = {0};
  sllm_matmul_desc_t matmul = {0};
  sllm_matmul_dispatch_info_t matmul_dispatch = {0};
  sllm_attention_preprocess_desc_t attention_preprocess = {0};
  sllm_attention_preprocess_dispatch_info_t attention_preprocess_dispatch = {0};
  sllm_kv_state_create_info_t kv_create = {0};
  sllm_kv_view_info_t kv_view_info = {0};
  sllm_kv_append_desc_t kv_append = {0};
  sllm_kv_append_info_t kv_append_info = {0};
  sllm_causal_attention_desc_t causal_attention = {0};
  sllm_causal_attention_dispatch_info_t causal_attention_dispatch = {0};
  sllm_status_t (*rmsnorm_release)(
      sllm_rmsnorm_plan_t **, sllm_error_sink_t *) = &sllm_rmsnorm_plan_release;
  sllm_status_t (*elementwise_release)(sllm_elementwise_plan_t **,
                                       sllm_error_sink_t *) =
      &sllm_elementwise_plan_release;
  sllm_status_t (*embedding_release)(sllm_embedding_plan_t **,
                                     sllm_error_sink_t *) =
      &sllm_embedding_plan_release;
  sllm_status_t (*matmul_prepare)(
      const sllm_context_t *, const sllm_matmul_desc_t *, sllm_matmul_plan_t **,
      sllm_error_sink_t *) = &sllm_matmul_prepare;
  sllm_status_t (*matmul_release)(sllm_matmul_plan_t **, sllm_error_sink_t *) =
      &sllm_matmul_plan_release;
  sllm_status_t (*matmul_execute)(const sllm_matmul_plan_t *,
                                  const sllm_queue_t *, sllm_completion_t **,
                                  sllm_matmul_dispatch_info_t *,
                                  sllm_error_sink_t *) = &sllm_matmul_execute;
  sllm_status_t (*attention_preprocess_prepare)(
      const sllm_context_t *, const sllm_attention_preprocess_desc_t *,
      sllm_attention_preprocess_plan_t **, sllm_error_sink_t *) =
      &sllm_attention_preprocess_prepare;
  sllm_status_t (*attention_preprocess_release)(
      sllm_attention_preprocess_plan_t **, sllm_error_sink_t *) =
      &sllm_attention_preprocess_plan_release;
  sllm_status_t (*attention_preprocess_execute)(
      const sllm_attention_preprocess_plan_t *, const sllm_queue_t *,
      sllm_completion_t **, sllm_attention_preprocess_dispatch_info_t *,
      sllm_error_sink_t *) = &sllm_attention_preprocess_execute;
  sllm_status_t (*kv_state_create)(
      const sllm_context_t *, const sllm_kv_state_create_info_t *,
      sllm_kv_state_t **, sllm_error_sink_t *) = &sllm_kv_state_create;
  sllm_status_t (*kv_state_append)(
      const sllm_kv_state_t *, const sllm_queue_t *,
      const sllm_kv_append_desc_t *, sllm_completion_t **,
      sllm_kv_append_info_t *, sllm_error_sink_t *) = &sllm_kv_state_append;
  sllm_status_t (*kv_state_cancel)(const sllm_kv_state_t *, sllm_completion_t *,
                                   sllm_error_sink_t *) =
      &sllm_kv_state_append_cancel;
  sllm_status_t (*causal_attention_execute)(
      const sllm_context_t *, const sllm_queue_t *,
      const sllm_causal_attention_desc_t *, sllm_completion_t **,
      sllm_causal_attention_dispatch_info_t *, sllm_error_sink_t *) =
      &sllm_causal_attention_execute;

  return sink.abi_version == SLLM_HIP_ABI_VERSION && access != 0U &&
                 queue == 0 && buffer == 0 && event == 0 && completion == 0 &&
                 device.struct_size == 0U && context_info.flags == 0U &&
                 queue_info.flags == 0U && buffer_info.size_bytes == 0U &&
                 transfer.size_bytes == 0U && completion_result.state == 0U &&
                 rmsnorm_plan == 0 && elementwise_plan == 0 &&
                 embedding_plan == 0 && matmul_plan == 0 &&
                 attention_preprocess_plan == 0 && kv_state == 0 &&
                 kv_view == 0 && binding.rank == 0U &&
                 rmsnorm.op_version == 0U && elementwise.op_version == 0U &&
                 embedding.op_version == 0U && matmul.op_version == 0U &&
                 matmul_dispatch.m == 0U &&
                 attention_preprocess.op_version == 0U &&
                 attention_preprocess_dispatch.m == 0U &&
                 kv_create.capacity_tokens == 0U &&
                 kv_view_info.observed_length == 0U &&
                 kv_append.expected_length == 0U &&
                 kv_append_info.token_count == 0U &&
                 causal_attention.expected_kv_length == 0U &&
                 causal_attention_dispatch.query_count == 0U &&
                 rmsnorm_release != 0 && elementwise_release != 0 &&
                 embedding_release != 0 && matmul_prepare != 0 &&
                 matmul_release != 0 && matmul_execute != 0 &&
                 attention_preprocess_prepare != 0 &&
                 attention_preprocess_release != 0 &&
                 attention_preprocess_execute != 0 && kv_state_create != 0 &&
                 kv_state_append != 0 && kv_state_cancel != 0 &&
                 causal_attention_execute != 0
             ? 0
             : 1;
}
