#ifndef SLLM_DECODE_GRAPH_CAPTURE_INTERNAL_HPP
#define SLLM_DECODE_GRAPH_CAPTURE_INTERNAL_HPP

#include "sllm/hip.h"

#include <cstdint>

/* Private metadata returned by the incremental whole-decode capture ABI.
 * This is intentionally outside the public header: the public graph-span
 * contract remains the immutable stateless-plan API. */
struct sllm_graph_span_capture_info_t final {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t reserved0;
  uint64_t actual_node_count;
  uint64_t kernel_node_count;
  uint64_t memcpy_node_count;
  uint64_t logical_plan_count;
};

constexpr uint32_t SLLM_HIP_GRAPH_SPAN_CAPTURE_INFO_VERSION = 1U;

constexpr uint32_t SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_VERSION = 1U;
constexpr uint32_t SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_BEGIN_PHASE = 1U;
constexpr uint32_t SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_GATHER_SELECTOR = 2U;
constexpr uint32_t SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_GATHER_DECISION = 3U;
constexpr uint32_t SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_GATHER_RESULT = 4U;
constexpr uint32_t SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_GATHER_HIDDEN = 5U;
constexpr uint32_t SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_COMMIT_STEP = 6U;
constexpr uint32_t SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_VERIFY_FIXED_K20_PQ = 7U;
constexpr uint32_t SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_GATHER_ACTIVE_TOKEN = 8U;
constexpr uint32_t SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_GATHER_ACTIVE_HIDDEN = 9U;

/* Generic private command descriptor. Bindings are interpreted by opcode and
 * are never copied through host memory during capture. */
struct sllm_graph_span_decode_command_desc_t final {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t opcode;
  uint32_t phase_kind;
  uint32_t phase_index;
  uint32_t rows;
  uint32_t input_kind;
  uint32_t row_count;
  uint32_t hidden_width;
  uint32_t token_capacity;
  uint32_t vocabulary_size;
  uint32_t stop_count;
  uint32_t reserved0;
  sllm_tensor_binding_t input0;
  sllm_tensor_binding_t input1;
  sllm_tensor_binding_t output;
  sllm_tensor_binding_t stop_ids;
};

static_assert(sizeof(sllm_graph_span_decode_command_desc_t) == 792U,
              "whole graph decode command ABI layout changed");

#ifdef __cplusplus
extern "C" {

sllm_status_t sllm_graph_span_begin_capture(
    const sllm_context_t *context, const sllm_queue_t *queue,
    const sllm_tensor_binding_t *control_binding, sllm_graph_span_t **span,
    sllm_error_sink_t *error_sink) noexcept;

sllm_status_t
sllm_graph_span_end_capture(sllm_graph_span_t *span,
                            sllm_graph_span_capture_info_t *info,
                            sllm_error_sink_t *error_sink) noexcept;

sllm_status_t
sllm_graph_span_abort_capture(sllm_graph_span_t **span,
                              sllm_error_sink_t *error_sink) noexcept;

sllm_status_t
sllm_graph_span_capture_marker(sllm_graph_span_t *span,
                               sllm_completion_t **completion,
                               sllm_error_sink_t *error_sink) noexcept;

sllm_status_t sllm_graph_span_decode_command(
    const sllm_graph_span_t *span,
    const sllm_graph_span_decode_command_desc_t *command,
    sllm_error_sink_t *error_sink) noexcept;

/* Bind graph-owned checkpoint planes for one linear-attention state.  The
 * buffers remain private graph resources; public M3 checkpoint allocations
 * are not repurposed by this ABI. */
sllm_status_t sllm_graph_span_bind_linear_state(
    sllm_graph_span_t *span, const sllm_linear_attention_state_t *state,
    const sllm_tensor_binding_t *checkpoint_conv,
    const sllm_tensor_binding_t *checkpoint_recurrent, uint32_t token_count,
    uint32_t checkpoint_rows, sllm_error_sink_t *error_sink) noexcept;

sllm_status_t sllm_graph_span_select_linear_state(
    sllm_graph_span_t *span, const sllm_linear_attention_state_t *state,
    uint32_t token_count, sllm_error_sink_t *error_sink) noexcept;

/* Adopt host metadata only after every graph replay has drained.  This call
 * performs no GPU work; all device-side state selection has already been
 * performed by the captured control kernels. */
sllm_status_t sllm_graph_span_publish_state_metadata(
    sllm_graph_span_t *span, uint64_t expected_initial_position,
    uint64_t final_position, uint64_t successful_generations,
    sllm_error_sink_t *error_sink) noexcept;

} // extern "C"
#endif

#endif // SLLM_DECODE_GRAPH_CAPTURE_INTERNAL_HPP
