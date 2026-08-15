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
  sllm_elementwise_plan_t *elementwise_plan = nullptr;
  sllm_embedding_plan_t *embedding_plan = nullptr;
  sllm_matmul_plan_t *matmul_plan = nullptr;
  sllm_attention_preprocess_plan_t *attention_preprocess_plan = nullptr;
  sllm_rotary_plan_t *rotary_plan = nullptr;
  sllm_windowed_attention_plan_t *windowed_attention_plan = nullptr;
  sllm_kv_state_t *kv_state = nullptr;
  sllm_kv_view_t *kv_view = nullptr;
  sllm_device_info_t device{};
  sllm_context_create_info_t context_info{};
  sllm_queue_create_info_t queue_info{};
  sllm_buffer_create_info_t buffer_info{};
  sllm_transfer_desc_t transfer{};
  sllm_completion_result_t completion_result{};
  sllm_tensor_binding_t binding{};
  sllm_rmsnorm_desc_t rmsnorm{};
  sllm_elementwise_desc_t elementwise{};
  sllm_embedding_desc_t embedding{};
  sllm_matmul_desc_t matmul{};
  sllm_matmul_dispatch_info_t matmul_dispatch{};
  sllm_attention_preprocess_desc_t attention_preprocess{};
  sllm_attention_preprocess_dispatch_info_t attention_preprocess_dispatch{};
  sllm_rotary_desc_t rotary{};
  sllm_rotary_dispatch_info_t rotary_dispatch{};
  sllm_windowed_attention_desc_t windowed_attention{};
  sllm_windowed_attention_dispatch_info_t windowed_attention_dispatch{};
  sllm_kv_state_create_info_t kv_create{};
  sllm_kv_view_info_t kv_view_info{};
  sllm_kv_append_desc_t kv_append{};
  sllm_kv_append_info_t kv_append_info{};
  sllm_causal_attention_desc_t causal_attention{};
  sllm_causal_attention_dispatch_info_t causal_attention_dispatch{};
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
  using ElementwisePrepareFn = sllm_status_t (*)(
      const sllm_context_t *, const sllm_elementwise_desc_t *,
      sllm_elementwise_plan_t **, sllm_error_sink_t *) noexcept;
  const ElementwisePrepareFn elementwise_prepare = &sllm_elementwise_prepare;
  using ElementwiseReleaseFn = sllm_status_t (*)(sllm_elementwise_plan_t **,
                                                 sllm_error_sink_t *) noexcept;
  const ElementwiseReleaseFn elementwise_release =
      &sllm_elementwise_plan_release;
  using EmbeddingPrepareFn =
      sllm_status_t (*)(const sllm_context_t *, const sllm_embedding_desc_t *,
                        sllm_embedding_plan_t **, sllm_error_sink_t *) noexcept;
  const EmbeddingPrepareFn embedding_prepare = &sllm_embedding_prepare;
  using EmbeddingReleaseFn =
      sllm_status_t (*)(sllm_embedding_plan_t **, sllm_error_sink_t *) noexcept;
  const EmbeddingReleaseFn embedding_release = &sllm_embedding_plan_release;
  using MatmulPrepareFn =
      sllm_status_t (*)(const sllm_context_t *, const sllm_matmul_desc_t *,
                        sllm_matmul_plan_t **, sllm_error_sink_t *) noexcept;
  const MatmulPrepareFn matmul_prepare = &sllm_matmul_prepare;
  using MatmulReleaseFn =
      sllm_status_t (*)(sllm_matmul_plan_t **, sllm_error_sink_t *) noexcept;
  const MatmulReleaseFn matmul_release = &sllm_matmul_plan_release;
  using MatmulExecuteFn = sllm_status_t (*)(
      const sllm_matmul_plan_t *, const sllm_queue_t *, sllm_completion_t **,
      sllm_matmul_dispatch_info_t *, sllm_error_sink_t *) noexcept;
  const MatmulExecuteFn matmul_execute = &sllm_matmul_execute;
  using AttentionPreprocessPrepareFn = sllm_status_t (*)(
      const sllm_context_t *, const sllm_attention_preprocess_desc_t *,
      sllm_attention_preprocess_plan_t **, sllm_error_sink_t *) noexcept;
  const AttentionPreprocessPrepareFn attention_preprocess_prepare =
      &sllm_attention_preprocess_prepare;
  using AttentionPreprocessReleaseFn = sllm_status_t (*)(
      sllm_attention_preprocess_plan_t **, sllm_error_sink_t *) noexcept;
  const AttentionPreprocessReleaseFn attention_preprocess_release =
      &sllm_attention_preprocess_plan_release;
  using AttentionPreprocessExecuteFn = sllm_status_t (*)(
      const sllm_attention_preprocess_plan_t *, const sllm_queue_t *,
      sllm_completion_t **, sllm_attention_preprocess_dispatch_info_t *,
      sllm_error_sink_t *) noexcept;
  const AttentionPreprocessExecuteFn attention_preprocess_execute =
      &sllm_attention_preprocess_execute;
  using RotaryPrepareFn =
      sllm_status_t (*)(const sllm_context_t *, const sllm_rotary_desc_t *,
                        sllm_rotary_plan_t **, sllm_error_sink_t *) noexcept;
  const RotaryPrepareFn rotary_prepare = &sllm_rotary_prepare;
  using RotaryReleaseFn =
      sllm_status_t (*)(sllm_rotary_plan_t **, sllm_error_sink_t *) noexcept;
  const RotaryReleaseFn rotary_release = &sllm_rotary_plan_release;
  using RotaryExecuteFn = sllm_status_t (*)(
      const sllm_rotary_plan_t *, const sllm_queue_t *, sllm_completion_t **,
      sllm_rotary_dispatch_info_t *, sllm_error_sink_t *) noexcept;
  const RotaryExecuteFn rotary_execute = &sllm_rotary_execute;
  using WindowedAttentionPrepareFn = sllm_status_t (*)(
      const sllm_context_t *, const sllm_windowed_attention_desc_t *,
      sllm_windowed_attention_plan_t **, sllm_error_sink_t *) noexcept;
  const WindowedAttentionPrepareFn windowed_attention_prepare =
      &sllm_windowed_attention_prepare;
  using WindowedAttentionReleaseFn = sllm_status_t (*)(
      sllm_windowed_attention_plan_t **, sllm_error_sink_t *) noexcept;
  const WindowedAttentionReleaseFn windowed_attention_release =
      &sllm_windowed_attention_plan_release;
  using WindowedAttentionExecuteFn = sllm_status_t (*)(
      const sllm_windowed_attention_plan_t *, const sllm_queue_t *,
      sllm_completion_t **, sllm_windowed_attention_dispatch_info_t *,
      sllm_error_sink_t *) noexcept;
  const WindowedAttentionExecuteFn windowed_attention_execute =
      &sllm_windowed_attention_execute;
  using KvStateCreateFn = sllm_status_t (*)(
      const sllm_context_t *, const sllm_kv_state_create_info_t *,
      sllm_kv_state_t **, sllm_error_sink_t *) noexcept;
  const KvStateCreateFn kv_state_create = &sllm_kv_state_create;
  using KvStateAppendFn =
      sllm_status_t (*)(const sllm_kv_state_t *, const sllm_queue_t *,
                        const sllm_kv_append_desc_t *, sllm_completion_t **,
                        sllm_kv_append_info_t *, sllm_error_sink_t *) noexcept;
  const KvStateAppendFn kv_state_append = &sllm_kv_state_append;
  using KvStateCancelFn =
      sllm_status_t (*)(const sllm_kv_state_t *, sllm_completion_t *,
                        sllm_error_sink_t *) noexcept;
  const KvStateCancelFn kv_state_cancel = &sllm_kv_state_append_cancel;
  using CausalAttentionExecuteFn = sllm_status_t (*)(
      const sllm_context_t *, const sllm_queue_t *,
      const sllm_causal_attention_desc_t *, sllm_completion_t **,
      sllm_causal_attention_dispatch_info_t *, sllm_error_sink_t *) noexcept;
  const CausalAttentionExecuteFn causal_attention_execute =
      &sllm_causal_attention_execute;
  (void)context_create;
  (void)rmsnorm_prepare;
  (void)rmsnorm_release;
  (void)elementwise_prepare;
  (void)elementwise_release;
  (void)embedding_prepare;
  (void)embedding_release;
  (void)matmul_prepare;
  (void)matmul_release;
  (void)matmul_execute;
  (void)attention_preprocess_prepare;
  (void)attention_preprocess_release;
  (void)attention_preprocess_execute;
  (void)rotary_prepare;
  (void)rotary_release;
  (void)rotary_execute;
  (void)windowed_attention_prepare;
  (void)windowed_attention_release;
  (void)windowed_attention_execute;
  (void)kv_state_create;
  (void)kv_state_append;
  (void)kv_state_cancel;
  (void)causal_attention_execute;

  return sink.abi_version == SLLM_HIP_ABI_VERSION && access != 0U &&
                 queue == nullptr && buffer == nullptr && event == nullptr &&
                 completion == nullptr && device.struct_size == 0U &&
                 context_info.flags == 0U && queue_info.flags == 0U &&
                 buffer_info.size_bytes == 0U && transfer.size_bytes == 0U &&
                 completion_result.state == 0U && rmsnorm_plan == nullptr &&
                 elementwise_plan == nullptr && embedding_plan == nullptr &&
                 matmul_plan == nullptr &&
                 attention_preprocess_plan == nullptr &&
                 rotary_plan == nullptr && kv_state == nullptr &&
                 windowed_attention_plan == nullptr && kv_view == nullptr &&
                 binding.rank == 0U && rmsnorm.op_version == 0U &&
                 elementwise.op_version == 0U && embedding.op_version == 0U &&
                 matmul.op_version == 0U &&
                 attention_preprocess.op_version == 0U &&
                 matmul_dispatch.m == 0U &&
                 attention_preprocess_dispatch.m == 0U &&
                 rotary.op_version == 0U && rotary_dispatch.token_count == 0U &&
                 windowed_attention.op_version == 0U &&
                 windowed_attention_dispatch.query_count == 0U &&
                 kv_create.capacity_tokens == 0U &&
                 kv_view_info.observed_length == 0U &&
                 kv_append.expected_length == 0U &&
                 kv_append_info.token_count == 0U &&
                 causal_attention.expected_kv_length == 0U &&
                 causal_attention_dispatch.query_count == 0U
             ? 0
             : 1;
}
