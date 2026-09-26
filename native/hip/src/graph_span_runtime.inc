/* Bounded request-owned HIP graph spans.  This file is included by the
 * production runtime after the public completion state machine has been
 * defined.  The public API intentionally accepts opaque prepared-plan
 * handles: the native runtime resolves and pins the exact plan objects while
 * the request queue is idle. */

#include "decode_control_kernel_internal.hpp"
#include "decode_graph_capture_internal.hpp"

#include <initializer_list>

#if !defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)

namespace {

constexpr uint64_t kGraphSpanMaxPlans = UINT64_C(256);

enum class GraphSpanPlanKind : uint8_t {
  RmsNorm,
  ResidualRmsNorm,
  Elementwise,
  Matmul,
  Qwen38ProjectionPack2,
};

struct GraphSpanPlanRef final {
  GraphSpanPlanKind kind;
  const void *raw;
  QuarantineNode *node;
};

struct GraphKvStateBinding final {
  KvState *state;
  uint64_t initial_published_length;
  uint64_t initial_generation;
  std::vector<sllm_paged_kv::LogicalEntryUpdate> paged_table_updates;
};

struct GraphLinearStateBinding final {
  LinearAttentionState *state;
  void *checkpoint_conv;
  void *checkpoint_recurrent;
  uint32_t token_count;
  uint32_t checkpoint_rows;
  uint32_t anchor_slot;
  uint32_t reserved0;
  uint64_t conv_elements;
  uint64_t recurrent_elements;
  uint64_t initial_published_length;
  uint64_t initial_generation;
};

struct GraphSpan final : QuarantineNode {
  Context *context;
  Queue *queue;
  const sllm_queue_t *raw_queue;
  hipGraph_t graph;
  hipGraphExec_t graph_exec;
  uint64_t actual_node_count;
  uint64_t logical_plan_count;
  bool release_active;
  bool in_flight;
  bool capture_owner;
  bool whole_capture;
  bool capture_failed;
  uint64_t whole_replay_in_flight;
  Buffer *control_buffer;
  uint64_t control_byte_offset;
  bool control_buffer_pinned;
  std::vector<Buffer *> capture_buffers;
  std::vector<QuarantineNode *> capture_graph_nodes;
  uint64_t matmul_workspace_pins;
  uint64_t causal_workspace_pins;
  std::vector<GraphKvStateBinding> kv_states;
  std::vector<GraphLinearStateBinding> linear_states;
  std::vector<GraphSpanPlanRef> plans;
  std::vector<sllm_completion_t *> capture_markers;

  GraphSpan(Context *const context_value, Queue *const queue_value,
            const sllm_queue_t *const raw_queue_value)
      : QuarantineNode(HandleKind::GraphSpan), context(context_value),
        queue(queue_value), raw_queue(raw_queue_value), graph(nullptr),
        graph_exec(nullptr), actual_node_count(0U), logical_plan_count(0U),
        release_active(false), in_flight(false), capture_owner(false),
        whole_capture(false), capture_failed(false), whole_replay_in_flight(0U),
        control_buffer(nullptr), control_byte_offset(0U),
        control_buffer_pinned(false), capture_buffers(), capture_graph_nodes(),
        matmul_workspace_pins(0U), causal_workspace_pins(0U), kv_states(),
        linear_states(), plans(), capture_markers() {}
};

/* The incremental whole-capture API is intentionally thread-local. Native
 * execute entry points are called synchronously by the capture thread and use
 * these accessors to discover the optional device control plane without
 * adding it to the existing public operation ABIs. */
thread_local GraphSpan *graph_capture_span = nullptr;

bool graph_span_tensor_is_m1(
    const sllm_rmsnorm::TensorMetadata &tensor) noexcept {
  if (tensor.rank == 0U || tensor.rank > SLLM_HIP_TENSOR_MAX_RANK) {
    return false;
  }
  uint64_t row_count = 1U;
  for (uint32_t index = 0U; index + 1U < tensor.rank; ++index) {
    if (tensor.shape[index] == 0U ||
        row_count >
            std::numeric_limits<uint64_t>::max() / tensor.shape[index]) {
      return false;
    }
    row_count *= tensor.shape[index];
  }
  return tensor.shape[tensor.rank - 1U] != 0U && row_count == 1U;
}

bool graph_span_tensor_is_m1(
    const sllm_elementwise::TensorMetadata &tensor) noexcept {
  if (tensor.rank == 0U || tensor.rank > SLLM_HIP_TENSOR_MAX_RANK) {
    return false;
  }
  for (uint32_t index = 0U; index != tensor.rank; ++index) {
    if (tensor.shape[index] == 0U) {
      return false;
    }
  }
  /* Elementwise Qwen38 sigmoid-mul tensors are [1, heads, width].  The
   * leading extent is the token batch; head/width extents are part of one
   * stateless token operation and must not be multiplied into an M count. */
  return tensor.rank == 1U || tensor.shape[0] == 1U;
}

bool graph_span_direct_matmul(const ElementwisePlan *const plan,
                              const char *const arch_name) noexcept {
  if (plan == nullptr || !plan->matmul || plan->qwen38_projection_pack2 ||
      arch_name == nullptr || plan->matmul_metadata.m == 0U ||
      plan->matmul_metadata.m != 1U || plan->matmul_metadata.k == 0U ||
      plan->matmul_metadata.n == 0U ||
      plan->matmul_context_workspace_bytes != 0U) {
    return false;
  }
  const auto activation_mode = static_cast<sllm_public_runtime::PrequantMode>(
      plan->matmul_metadata.activation_prequant);
  if (activation_mode == sllm_public_runtime::PrequantMode::Fp8Outer &&
      !plan->matmul_metadata.fp8_outer) {
    return false;
  }
  if (activation_mode == sllm_public_runtime::PrequantMode::Nvfp4Block16 &&
      !plan->matmul_metadata.nvfp4_w4a4) {
    return false;
  }

  const bool exact_gfx1030 = matches_runtime_gcn_arch(arch_name, "gfx1030");
  const bool exact_gfx1201 = matches_runtime_gcn_arch(arch_name, "gfx1201");
  if (!exact_gfx1030 && !exact_gfx1201) {
    return false;
  }

  /* gfx1201's native FP8 route is capture-safe only after prepare has frozen
   * the hipBLASLt plan and its per-plan quantization workspace.  Keep the
   * existing gfx1030 software routes and all shared-staging/library routes
   * unchanged; the graph owns the prepared plan pin for the lifetime of the
   * capture/replay object. */
  if (plan->matmul_kernel_variant ==
      ::sllm_matmul_kernel::HostKernelVariant::Fp8OuterGfx1201Dot4) {
    return exact_gfx1201 && plan->matmul_metadata.fp8_outer &&
           plan->matmul_metadata.fp8_dtype == SLLM_TENSOR_DTYPE_F8_E4M3_FN &&
           plan->fp8_lt_plan == nullptr && plan->matmul_workspace != nullptr &&
           plan->matmul_workspace_bytes != 0U;
  }
  if (plan->matmul_kernel_variant ==
          ::sllm_matmul_kernel::HostKernelVariant::Fp8Native ||
      plan->fp8_lt_plan != nullptr) {
    return exact_gfx1201 && plan->matmul_metadata.fp8_outer &&
           plan->matmul_metadata.fp8_dtype == SLLM_TENSOR_DTYPE_F8_E4M3_FN &&
           plan->matmul_kernel_variant ==
               ::sllm_matmul_kernel::HostKernelVariant::Fp8Native &&
           plan->fp8_lt_plan != nullptr && plan->matmul_workspace != nullptr &&
           plan->matmul_workspace_bytes != 0U;
  }
  if (plan->matmul_kernel_variant ==
      ::sllm_matmul_kernel::HostKernelVariant::HipBlas) {
    return false;
  }
  return true;
}

bool graph_span_qwen38_nvfp4_member_compatible(
    const sllm_matmul::DescriptorMetadata &metadata) noexcept {
  return metadata.m == 1U && metadata.k != 0U &&
         (metadata.k % UINT64_C(16)) == 0U && metadata.n != 0U &&
         metadata.nvfp4_w4a4 && metadata.weight_value_bytes != 0U &&
         metadata.weight_scale_offset >= metadata.weight.byte_offset &&
         metadata.weight_tensor_scale_offset >= metadata.weight_scale_offset &&
         metadata.weight_tensor_scale_offset <= UINT64_MAX - UINT64_C(4) &&
         metadata.input_tensor_scale_offset ==
             metadata.weight_tensor_scale_offset + UINT64_C(4);
}

bool graph_span_qwen38_nvfp4_workspace_bytes(
    const uint64_t k, uint64_t *const workspace_bytes) noexcept {
  if (workspace_bytes == nullptr || k == 0U || (k % UINT64_C(16)) != 0U ||
      k > UINT64_MAX - UINT64_C(1)) {
    return false;
  }
  const uint64_t packed_bytes = (k + UINT64_C(1)) / UINT64_C(2);
  const uint64_t scale_bytes = k / UINT64_C(16);
  if (packed_bytes > UINT64_MAX - scale_bytes) {
    return false;
  }
  *workspace_bytes = packed_bytes + scale_bytes;
  return *workspace_bytes != 0U;
}

bool graph_span_qwen38_nvfp4_variant_safe(
    const sllm_matmul_kernel::KernelVariant variant,
    const sllm_matmul::DescriptorMetadata &first,
    const sllm_matmul::DescriptorMetadata &second,
    const char *const arch_name) noexcept {
  using sllm_matmul_kernel::KernelVariant;
  if (variant == KernelVariant::Nvfp4W4A4Packed ||
      variant == KernelVariant::Nvfp4W4A4Decode) {
    return true;
  }
  const bool exact_gfx1030 = matches_runtime_gcn_arch(arch_name, "gfx1030");
  const bool exact_gfx1201 = matches_runtime_gcn_arch(arch_name, "gfx1201");
  switch (variant) {
  case KernelVariant::Nvfp4W4A4DecodeWave4Column32:
    return (exact_gfx1030 || exact_gfx1201) &&
           sllm_matmul_kernel::phase78_nvfp4_w4a4_decode_wave4col32_shape(
               first.m, first.k, first.n) &&
           sllm_matmul_kernel::phase78_nvfp4_w4a4_decode_wave4col32_shape(
               second.m, second.k, second.n);
  case KernelVariant::Nvfp4W4A4DecodeActivationShared:
    return exact_gfx1030 &&
           sllm_matmul_kernel::
               phase78_nvfp4_w4a4_decode_activation_shared_shape(
                   first.m, first.k, first.n) &&
           sllm_matmul_kernel::
               phase78_nvfp4_w4a4_decode_activation_shared_shape(
                   second.m, second.k, second.n);
  case KernelVariant::Nvfp4W4A4DecodeScaleLut:
    return (exact_gfx1030 &&
            sllm_matmul_kernel::
                phase78_nvfp4_w4a4_decode_activation_shared_shape(
                    first.m, first.k, first.n) &&
            sllm_matmul_kernel::
                phase78_nvfp4_w4a4_decode_activation_shared_shape(
                    second.m, second.k, second.n)) ||
           (exact_gfx1201 &&
            sllm_matmul_kernel::phase78_nvfp4_w4a4_decode_wave4col32_shape(
                first.m, first.k, first.n) &&
            sllm_matmul_kernel::phase78_nvfp4_w4a4_decode_wave4col32_shape(
                second.m, second.k, second.n));
  default:
    return false;
  }
}

bool graph_span_qwen38_pack(const ElementwisePlan *const plan,
                            const char *const arch_name) noexcept {
  if (plan == nullptr || !plan->qwen38_projection_pack2 ||
      arch_name == nullptr) {
    return false;
  }

  const uint32_t role = plan->qwen38_projection_pack2_role;
  if (role == SLLM_HIP_QWEN38_PROJECTION_PACK2_ROLE_NVFP4_MLP_GATE_UP) {
    const auto &first = plan->qwen38_projection_pack2_metadata[0];
    const auto &second = plan->qwen38_projection_pack2_metadata[1];
    uint64_t expected_workspace_bytes = 0U;
    return graph_span_qwen38_nvfp4_member_compatible(first) &&
           graph_span_qwen38_nvfp4_member_compatible(second) &&
           first.m == second.m && first.k == second.k &&
           first.activation.shape[0] == second.activation.shape[0] &&
           first.activation.shape[1] == second.activation.shape[1] &&
           plan->matmul_workspace != nullptr &&
           graph_span_qwen38_nvfp4_workspace_bytes(first.k,
                                                   &expected_workspace_bytes) &&
           plan->matmul_workspace_bytes == expected_workspace_bytes &&
           plan->qwen38_projection_pack2_kernel_variants[0] ==
               plan->qwen38_projection_pack2_kernel_variants[1] &&
           graph_span_qwen38_nvfp4_variant_safe(
               plan->qwen38_projection_pack2_kernel_variants[0], first, second,
               arch_name);
  }
  if (role != SLLM_HIP_QWEN38_PROJECTION_PACK2_ROLE_FP8_GDN_QKV_Z ||
      plan->qwen38_projection_pack2_metadata[0].m != 1U ||
      plan->qwen38_projection_pack2_metadata[0].k != UINT64_C(5120) ||
      plan->qwen38_projection_pack2_metadata[0].n != UINT64_C(10240) ||
      plan->qwen38_projection_pack2_metadata[1].m != 1U ||
      plan->qwen38_projection_pack2_metadata[1].k != UINT64_C(5120) ||
      plan->qwen38_projection_pack2_metadata[1].n != UINT64_C(6144) ||
      plan->matmul_workspace == nullptr ||
      plan->matmul_workspace_bytes !=
          SLLM_HIP_QWEN38_PROJECTION_PACK2_FP8_GDN_WORKSPACE_BYTES) {
    return false;
  }

  const auto &first = plan->qwen38_projection_pack2_metadata[0];
  const auto &second = plan->qwen38_projection_pack2_metadata[1];
  if (!first.fp8_outer || !second.fp8_outer ||
      first.fp8_dtype != SLLM_TENSOR_DTYPE_F8_E4M3_FN ||
      second.fp8_dtype != SLLM_TENSOR_DTYPE_F8_E4M3_FN) {
    return false;
  }

  const bool exact_gfx1201 = matches_runtime_gcn_arch(arch_name, "gfx1201");
  const bool exact_gfx1030 = matches_runtime_gcn_arch(arch_name, "gfx1030");
  if (exact_gfx1201) {
    const auto first_variant = plan->qwen38_projection_pack2_kernel_variants[0];
    const auto second_variant =
        plan->qwen38_projection_pack2_kernel_variants[1];
    const bool lt_pair =
        first_variant == ::sllm_matmul_kernel::HostKernelVariant::Fp8Native &&
        second_variant == ::sllm_matmul_kernel::HostKernelVariant::Fp8Native &&
        plan->qwen38_projection_pack2_fp8_lt_plans[0] != nullptr &&
        plan->qwen38_projection_pack2_fp8_lt_plans[1] != nullptr;
    const bool dot4_pair =
        first_variant ==
            ::sllm_matmul_kernel::HostKernelVariant::Fp8OuterGfx1201Dot4 &&
        second_variant ==
            ::sllm_matmul_kernel::HostKernelVariant::Fp8OuterGfx1201Dot4 &&
        plan->qwen38_projection_pack2_fp8_lt_plans[0] == nullptr &&
        plan->qwen38_projection_pack2_fp8_lt_plans[1] == nullptr;
    return lt_pair || dot4_pair;
  }
  if (exact_gfx1030) {
    const auto variant = plan->qwen38_projection_pack2_kernel_variants[0];
    return variant == plan->qwen38_projection_pack2_kernel_variants[1] &&
           variant != ::sllm_matmul_kernel::HostKernelVariant::Fp8Native &&
           (variant == ::sllm_matmul_kernel::KernelVariant::
                           Fp8OuterDecodeGfx1030Half2Wave4Col32 ||
            variant == ::sllm_matmul_kernel::KernelVariant::
                           Fp8OuterDecodeGfx1030Dword8Wave4Col32 ||
            variant == ::sllm_matmul_kernel::KernelVariant::
                           Fp8OuterDecodeGfx1030LdsLutWave4Col32 ||
            variant == ::sllm_matmul_kernel::KernelVariant::
                           Fp8OuterDecodeGfx1030ActivationSharedWave4Col32 ||
            variant == ::sllm_matmul_kernel::KernelVariant::
                           Fp8OuterDecodeGfx1030ActivationSharedWave8Col64 ||
            variant == ::sllm_matmul_kernel::KernelVariant::Fp8Emulation);
  }
  return false;
}

bool graph_span_elementwise(const ElementwisePlan *const plan) noexcept {
  if (plan == nullptr || plan->embedding || plan->matmul || plan->gdn_bundle ||
      plan->mlp_bundle || plan->qwen38_projection_pack2 || plan->argmax ||
      plan->metadata.operation == SLLM_ELEMENTWISE_OPERATION_COPY ||
      plan->metadata.element_count == 0U ||
      !graph_span_tensor_is_m1(plan->metadata.input0) ||
      !graph_span_tensor_is_m1(plan->metadata.input1) ||
      !graph_span_tensor_is_m1(plan->metadata.output)) {
    return false;
  }
  const auto mode = static_cast<sllm_public_runtime::PrequantMode>(
      plan->metadata.output_prequant);
  if (mode == sllm_public_runtime::PrequantMode::Nvfp4Block16 &&
      plan->input_tensor_scale_device == nullptr) {
    return false;
  }
  if (mode != sllm_public_runtime::PrequantMode::None &&
      plan->metadata.operation != SLLM_ELEMENTWISE_OPERATION_SILU_MUL &&
      plan->metadata.operation != SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL) {
    return false;
  }
  if (mode == sllm_public_runtime::PrequantMode::Nvfp4Block16 &&
      plan->metadata.operation != SLLM_ELEMENTWISE_OPERATION_SILU_MUL) {
    return false;
  }
  return true;
}

bool graph_span_plan_m1(const RmsNormPlan *const plan) noexcept {
  if (plan == nullptr || plan->fused) {
    if (plan == nullptr || !plan->fused ||
        !graph_span_tensor_is_m1(plan->residual_metadata.residual)) {
      return false;
    }
    const auto mode = static_cast<sllm_public_runtime::PrequantMode>(
        plan->residual_metadata.output_prequant);
    return mode != sllm_public_runtime::PrequantMode::Nvfp4Block16 ||
           plan->input_tensor_scale_device != nullptr;
  }
  return plan->metadata.output_prequant == 0U &&
         graph_span_tensor_is_m1(plan->metadata.activation);
}

void graph_span_unpin(GraphSpan *const span) noexcept {
  if (span == nullptr || span->context == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> accounting_lock(span->context->accounting_mutex);
  for (const GraphSpanPlanRef &ref : span->plans) {
    if (ref.node != nullptr && ref.node->graph_pins != 0U) {
      --ref.node->graph_pins;
    }
  }
  if (span->queue != nullptr && span->queue->graph_pins != 0U) {
    --span->queue->graph_pins;
    if (span->whole_capture) {
      span->queue->whole_graph_result_readback_required.store(
          false, std::memory_order_release);
    }
  }
  if (span->control_buffer_pinned && span->control_buffer != nullptr) {
    if (!sllm_public_runtime::AccountingState::release_two_buffer_prepared_plan(
            span->context->accounting, span->control_buffer->accounting,
            span->control_buffer->accounting)) {
      span->context->poisoned.store(true);
    }
    span->control_buffer_pinned = false;
  }
  for (Buffer *const buffer : span->capture_buffers) {
    if (buffer == nullptr || buffer == span->control_buffer) {
      continue;
    }
    if (!sllm_public_runtime::AccountingState::release_two_buffer_prepared_plan(
            span->context->accounting, buffer->accounting,
            buffer->accounting)) {
      span->context->poisoned.store(true);
    }
  }
  span->capture_buffers.clear();
  for (QuarantineNode *const node : span->capture_graph_nodes) {
    if (node != nullptr && node->graph_pins != 0U) {
      --node->graph_pins;
    }
  }
  span->capture_graph_nodes.clear();
  span->kv_states.clear();
  span->linear_states.clear();
  if (span->matmul_workspace_pins != 0U) {
    if (span->context->matmul_f16_staging_owner_queue != span->queue ||
        span->context->matmul_f16_staging_in_flight <
            span->matmul_workspace_pins) {
      span->context->poisoned.store(true);
    } else {
      span->context->matmul_f16_staging_in_flight -=
          span->matmul_workspace_pins;
      if (span->context->matmul_f16_staging_in_flight == 0U) {
        span->context->matmul_f16_staging_owner_queue = nullptr;
      }
    }
    span->matmul_workspace_pins = 0U;
  }
  if (span->causal_workspace_pins != 0U) {
    if (span->context->causal_scaled_prefill_owner_queue != span->queue ||
        span->context->causal_scaled_prefill_in_flight <
            span->causal_workspace_pins) {
      span->context->poisoned.store(true);
    } else {
      span->context->causal_scaled_prefill_in_flight -=
          span->causal_workspace_pins;
      if (span->context->causal_scaled_prefill_in_flight == 0U) {
        span->context->causal_scaled_prefill_owner_queue = nullptr;
      }
    }
    span->causal_workspace_pins = 0U;
  }
  const bool owned_capture = span->capture_owner;
  if (span->queue != nullptr && owned_capture) {
    span->capture_owner = false;
    span->queue->graph_capture_active.store(false, std::memory_order_release);
  }
  if (owned_capture && graph_capture_queue == span->queue) {
    graph_capture_queue = nullptr;
  }
  if (owned_capture && graph_capture_span == span) {
    graph_capture_span = nullptr;
  }
}

void graph_span_finish_capture(GraphSpan *const span) noexcept {
  if (span == nullptr || span->queue == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> accounting_lock(span->context->accounting_mutex);
  const bool owned_capture = span->capture_owner;
  if (owned_capture) {
    span->capture_owner = false;
    span->queue->graph_capture_active.store(false, std::memory_order_release);
  }
  if (owned_capture && graph_capture_queue == span->queue) {
    graph_capture_queue = nullptr;
  }
  if (owned_capture && graph_capture_span == span) {
    graph_capture_span = nullptr;
  }
}

[[maybe_unused]] bool whole_graph_capture_active() noexcept {
  return graph_capture_span != nullptr && graph_capture_span->whole_capture &&
         graph_capture_span->capture_owner && graph_capture_queue != nullptr &&
         graph_capture_span->queue == graph_capture_queue;
}

void invalidate_whole_graph_capture() noexcept {
  if (graph_capture_span != nullptr)
    graph_capture_span->capture_failed = true;
}

[[maybe_unused]] void *whole_graph_capture_device_control() noexcept {
  if (!whole_graph_capture_active() ||
      graph_capture_span->control_buffer == nullptr ||
      graph_capture_span->control_buffer->device_pointer == nullptr) {
    return nullptr;
  }
  return static_cast<char *>(
             graph_capture_span->control_buffer->device_pointer) +
         static_cast<std::size_t>(graph_capture_span->control_byte_offset);
}

void graph_span_quarantine_markers(GraphSpan *const span) noexcept {
  if (span == nullptr) {
    return;
  }
  for (sllm_completion_t *const raw_marker : span->capture_markers) {
    if (raw_marker == nullptr) {
      continue;
    }
    Completion *completion = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      completion = lookup<Completion>(raw_marker, HandleKind::Completion);
    }
    if (completion != nullptr) {
      quarantine_completion(raw_marker, completion);
    }
  }
  span->capture_markers.clear();
}

bool graph_span_finalize_capture_markers(
    GraphSpan *const span, sllm_error_sink_t *const error_sink) noexcept {
  if (span == nullptr || span->capture_markers.empty()) {
    return true;
  }
  sllm_completion_t *fence = nullptr;
  const sllm_status_t fence_status =
      sllm_queue_fence(span->raw_queue, &fence, error_sink);
  if (fence_status != SLLM_STATUS_OK || fence == nullptr) {
    graph_span_quarantine_markers(span);
    return false;
  }
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  const sllm_status_t wait_status =
      sllm_completion_wait(fence, UINT32_MAX, &result, error_sink);
  if (wait_status != SLLM_STATUS_OK) {
    Completion *fence_state = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      fence_state = lookup<Completion>(fence, HandleKind::Completion);
    }
    if (fence_state != nullptr) {
      quarantine_completion(fence, fence_state);
    }
    graph_span_quarantine_markers(span);
    return false;
  }
  if (span->whole_capture) {
    /* Whole-capture markers were rolled back when they were transferred into
     * the span.  The graph owns the captured commands and the final fence
     * above proves that every replay has drained; release the inert marker
     * handles without publishing the host-side transition captured before the
     * first replay. */
    for (sllm_completion_t *&raw_marker : span->capture_markers) {
      if (raw_marker == nullptr) {
        continue;
      }
      const sllm_status_t release_status =
          sllm_completion_release(&raw_marker, error_sink);
      if (release_status != SLLM_STATUS_OK) {
        graph_span_quarantine_markers(span);
        (void)sllm_completion_release(&fence, error_sink);
        return false;
      }
    }
    span->capture_markers.clear();
    const sllm_status_t fence_release =
        sllm_completion_release(&fence, error_sink);
    return fence_release == SLLM_STATUS_OK;
  }
  for (sllm_completion_t *&raw_marker : span->capture_markers) {
    if (raw_marker == nullptr) {
      continue;
    }
    const sllm_status_t finalize_status =
        sllm_completion_finalize_after(raw_marker, fence, &result, error_sink);
    if (finalize_status != SLLM_STATUS_OK) {
      graph_span_quarantine_markers(span);
      (void)sllm_completion_release(&fence, error_sink);
      return false;
    }
    const sllm_status_t release_status =
        sllm_completion_release(&raw_marker, error_sink);
    if (release_status != SLLM_STATUS_OK) {
      graph_span_quarantine_markers(span);
      (void)sllm_completion_release(&fence, error_sink);
      return false;
    }
  }
  span->capture_markers.clear();
  const sllm_status_t fence_release =
      sllm_completion_release(&fence, error_sink);
  if (fence_release != SLLM_STATUS_OK) {
    return false;
  }
  return true;
}

sllm_status_t
graph_span_resolve_plan(const void *const raw, Queue *const queue,
                        GraphSpanPlanRef *const out,
                        sllm_error_sink_t *const error_sink) noexcept {
  if (raw == nullptr || queue == nullptr || out == nullptr) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INVALID_ARGUMENT,
        "graph span plan handle or resolution output is null");
  }
  const RegistryEntry *entry = nullptr;
  const auto registry_it = registry.find(handle_key(raw));
  if (registry_it != registry.end()) {
    entry = &registry_it->second;
  }
  if (entry == nullptr || entry->state == nullptr) {
    (void)sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
        "graph span plan handle is stale or has the wrong kind");
    return SLLM_STATUS_PUBLIC_INVALID_HANDLE;
  }
  out->raw = raw;
  switch (entry->kind) {
  case HandleKind::RmsNormPlan: {
    auto *const plan = static_cast<RmsNormPlan *>(entry->state);
    if (plan->context != queue->context || !graph_span_plan_m1(plan)) {
      break;
    }
    out->kind = plan->fused ? GraphSpanPlanKind::ResidualRmsNorm
                            : GraphSpanPlanKind::RmsNorm;
    out->node = plan;
    return SLLM_STATUS_OK;
  }
  case HandleKind::ElementwisePlan: {
    auto *const plan = static_cast<ElementwisePlan *>(entry->state);
    if (plan->context == queue->context && graph_span_elementwise(plan)) {
      out->kind = GraphSpanPlanKind::Elementwise;
      out->node = plan;
      return SLLM_STATUS_OK;
    }
    break;
  }
  case HandleKind::MatmulPlan: {
    auto *const plan = static_cast<ElementwisePlan *>(entry->state);
    if (plan->context == queue->context) {
      if (graph_span_qwen38_pack(
              plan, queue->context->device_properties.gcnArchName)) {
        out->kind = GraphSpanPlanKind::Qwen38ProjectionPack2;
        out->node = plan;
        return SLLM_STATUS_OK;
      }
      if (graph_span_direct_matmul(
              plan, queue->context->device_properties.gcnArchName)) {
        out->kind = GraphSpanPlanKind::Matmul;
        out->node = plan;
        return SLLM_STATUS_OK;
      }
    }
    break;
  }
  default:
    break;
  }
  (void)sllm_public_runtime::write_error(
      error_sink, SLLM_STATUS_UNSUPPORTED,
      "graph span plan is outside the exact gfx1030/gfx1201 M1 stateless "
      "contract");
  return SLLM_STATUS_UNSUPPORTED;
}

void graph_span_write_dispatch_prefix(
    sllm_rmsnorm_dispatch_info_t *const info) {
  info->struct_size = sizeof(*info);
  info->abi_version = SLLM_HIP_ABI_VERSION;
  info->info_version = SLLM_HIP_RMSNORM_DISPATCH_INFO_VERSION;
}

void graph_span_write_dispatch_prefix(
    sllm_residual_rmsnorm_dispatch_info_t *const info) {
  info->struct_size = sizeof(*info);
  info->abi_version = SLLM_HIP_ABI_VERSION;
  info->info_version = SLLM_HIP_RESIDUAL_RMSNORM_DISPATCH_INFO_VERSION;
}

void graph_span_write_dispatch_prefix(
    sllm_elementwise_dispatch_info_t *const info) {
  info->struct_size = sizeof(*info);
  info->abi_version = SLLM_HIP_ABI_VERSION;
  info->info_version = SLLM_HIP_ELEMENTWISE_DISPATCH_INFO_VERSION;
}

void graph_span_write_dispatch_prefix(sllm_matmul_dispatch_info_t *const info) {
  info->struct_size = sizeof(*info);
  info->abi_version = SLLM_HIP_ABI_VERSION;
  info->info_version = SLLM_HIP_MATMUL_DISPATCH_INFO_VERSION;
}

void graph_span_write_dispatch_prefix(
    sllm_qwen38_projection_pack2_dispatch_info_t *const info) {
  info->struct_size = sizeof(*info);
  info->abi_version = SLLM_HIP_ABI_VERSION;
  info->info_version = SLLM_HIP_QWEN38_PROJECTION_PACK2_DISPATCH_INFO_VERSION;
}

sllm_status_t
graph_span_capture_plan(GraphSpan *const span, const GraphSpanPlanRef &ref,
                        sllm_error_sink_t *const error_sink) noexcept {
  sllm_completion_t *marker = nullptr;
  sllm_status_t status = SLLM_STATUS_INTERNAL_ERROR;
  switch (ref.kind) {
  case GraphSpanPlanKind::RmsNorm: {
    sllm_rmsnorm_dispatch_info_t info{};
    graph_span_write_dispatch_prefix(&info);
    status = sllm_rmsnorm_execute(
        reinterpret_cast<const sllm_rmsnorm_plan_t *>(ref.raw), span->raw_queue,
        &marker, &info, error_sink);
    break;
  }
  case GraphSpanPlanKind::ResidualRmsNorm: {
    sllm_residual_rmsnorm_dispatch_info_t info{};
    graph_span_write_dispatch_prefix(&info);
    status = sllm_residual_rmsnorm_execute(
        reinterpret_cast<const sllm_residual_rmsnorm_plan_t *>(ref.raw),
        span->raw_queue, &marker, &info, error_sink);
    break;
  }
  case GraphSpanPlanKind::Elementwise: {
    sllm_elementwise_dispatch_info_t info{};
    graph_span_write_dispatch_prefix(&info);
    status = sllm_elementwise_execute(
        reinterpret_cast<const sllm_elementwise_plan_t *>(ref.raw),
        span->raw_queue, &marker, &info, error_sink);
    break;
  }
  case GraphSpanPlanKind::Matmul: {
    sllm_matmul_dispatch_info_t info{};
    graph_span_write_dispatch_prefix(&info);
    status = sllm_matmul_execute(
        reinterpret_cast<const sllm_matmul_plan_t *>(ref.raw), span->raw_queue,
        &marker, &info, error_sink);
    break;
  }
  case GraphSpanPlanKind::Qwen38ProjectionPack2: {
    sllm_qwen38_projection_pack2_dispatch_info_t info{};
    graph_span_write_dispatch_prefix(&info);
    status = sllm_qwen38_projection_pack2_execute(
        reinterpret_cast<const sllm_qwen38_projection_pack2_plan_t *>(ref.raw),
        span->raw_queue, &marker, &info, error_sink);
    break;
  }
  }
  if (status != SLLM_STATUS_OK || marker == nullptr) {
    return status == SLLM_STATUS_OK
               ? sllm_public_runtime::write_error(
                     error_sink, SLLM_STATUS_INTERNAL_ERROR,
                     "graph span execute did not return a completion marker")
               : status;
  }
  span->capture_markers.push_back(marker);
  return SLLM_STATUS_OK;
}

bool graph_span_destroy_graph_checked(GraphSpan *const span,
                                      hipError_t *const failure) noexcept {
  if (failure != nullptr) {
    *failure = hipSuccess;
  }
  if (span == nullptr) {
    return true;
  }
  if (span->graph_exec != nullptr) {
    const hipError_t status = hipGraphExecDestroy(span->graph_exec);
    if (status != hipSuccess) {
      if (failure != nullptr) {
        *failure = status;
      }
      return false;
    }
    span->graph_exec = nullptr;
  }
  if (span->graph != nullptr) {
    const hipError_t status = hipGraphDestroy(span->graph);
    if (status != hipSuccess) {
      if (failure != nullptr) {
        *failure = status;
      }
      return false;
    }
    span->graph = nullptr;
  }
  return true;
}

bool graph_span_capture_info_valid(
    const sllm_graph_span_capture_info_t *const info) noexcept {
  return info != nullptr &&
         info->struct_size == sizeof(sllm_graph_span_capture_info_t) &&
         info->abi_version == SLLM_HIP_ABI_VERSION &&
         info->info_version == SLLM_HIP_GRAPH_SPAN_CAPTURE_INFO_VERSION &&
         info->reserved0 == 0U;
}

bool graph_span_capture_node_is_memcpy(const hipGraphNodeType type) noexcept {
  return type == hipGraphNodeTypeMemcpy ||
         type == hipGraphNodeTypeMemcpyFromSymbol ||
         type == hipGraphNodeTypeMemcpyToSymbol;
}

/* Whole decode capture accepts the native nodes needed to transfer the device
 * control plane and model inputs. Event nodes can be emitted by a capture-aware
 * completion implementation; host, alloc/free, nested graph, and semaphore
 * nodes remain rejected until they have an explicit lifetime contract. */
bool graph_span_collect_whole_capture_nodes(
    GraphSpan *const span, sllm_graph_span_capture_info_t *const info,
    sllm_error_sink_t *const error_sink) noexcept {
  if (span == nullptr || span->graph == nullptr || info == nullptr) {
    return false;
  }
  std::size_t total_node_count = 0U;
  hipError_t graph_status =
      hipGraphGetNodes(span->graph, nullptr, &total_node_count);
  if (graph_status != hipSuccess || total_node_count == 0U) {
    if (graph_status != hipSuccess) {
      (void)hip_failure(error_sink, graph_status,
                        "hipGraphGetNodes whole graph capture");
    } else {
      (void)sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_UNSUPPORTED,
          "whole graph capture produced no native nodes");
    }
    return false;
  }
  std::unique_ptr<hipGraphNode_t[]> graph_nodes;
  try {
    graph_nodes = std::make_unique<hipGraphNode_t[]>(total_node_count);
  } catch (...) {
    (void)sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "whole graph capture native node-list allocation failed");
    return false;
  }
  std::size_t listed_node_count = total_node_count;
  graph_status =
      hipGraphGetNodes(span->graph, graph_nodes.get(), &listed_node_count);
  if (graph_status != hipSuccess || listed_node_count != total_node_count) {
    if (graph_status != hipSuccess) {
      (void)hip_failure(error_sink, graph_status,
                        "hipGraphGetNodes whole graph node list");
    } else {
      (void)sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_UNSUPPORTED,
          "whole graph capture native node-list size changed");
    }
    return false;
  }
  uint64_t kernel_node_count = 0U;
  uint64_t memcpy_node_count = 0U;
  for (std::size_t index = 0U; index != listed_node_count; ++index) {
    hipGraphNodeType node_type = hipGraphNodeTypeCount;
    graph_status = hipGraphNodeGetType(graph_nodes[index], &node_type);
    if (graph_status != hipSuccess) {
      (void)hip_failure(error_sink, graph_status,
                        "hipGraphNodeGetType whole graph capture");
      return false;
    }
    if (node_type == hipGraphNodeTypeKernel) {
      ++kernel_node_count;
      continue;
    }
    if (graph_span_capture_node_is_memcpy(node_type)) {
      ++memcpy_node_count;
      continue;
    }
    // Fixed sampling clears its status record with a captured device memset.
    // It is a device node, not a host callback, and remains in
    // total_node_count.
    if (node_type == hipGraphNodeTypeMemset ||
        node_type == hipGraphNodeTypeEventRecord ||
        node_type == hipGraphNodeTypeWaitEvent) {
      continue;
    }
    (void)sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_UNSUPPORTED,
        "whole graph capture contains an unsupported native node type");
    return false;
  }
  if (kernel_node_count == 0U && memcpy_node_count == 0U) {
    (void)sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_UNSUPPORTED,
        "whole graph capture produced no kernel or memcpy nodes");
    return false;
  }
  span->actual_node_count = static_cast<uint64_t>(listed_node_count);
  span->logical_plan_count = 0U;
  info->struct_size = sizeof(sllm_graph_span_capture_info_t);
  info->abi_version = SLLM_HIP_ABI_VERSION;
  info->info_version = SLLM_HIP_GRAPH_SPAN_CAPTURE_INFO_VERSION;
  info->reserved0 = 0U;
  info->actual_node_count = span->actual_node_count;
  info->kernel_node_count = kernel_node_count;
  info->memcpy_node_count = memcpy_node_count;
  info->logical_plan_count = span->logical_plan_count;
  return true;
}

bool graph_span_validate_control_binding(
    const sllm_tensor_binding_t *const binding, Context *const context,
    Buffer **const buffer_output, uint64_t *const byte_offset_output,
    sllm_error_sink_t *const error_sink) noexcept {
  if (buffer_output == nullptr || byte_offset_output == nullptr) {
    return sllm_public_runtime::write_error(
               error_sink, SLLM_STATUS_INVALID_ARGUMENT,
               "whole graph control binding output is null") == SLLM_STATUS_OK;
  }
  *buffer_output = nullptr;
  *byte_offset_output = 0U;
  if (binding == nullptr) {
    return true;
  }
  if (binding->struct_size != sizeof(sllm_tensor_binding_t) ||
      binding->abi_version != SLLM_HIP_ABI_VERSION ||
      binding->buffer == nullptr ||
      (binding->byte_offset & UINT64_C(15)) != 0U) {
    (void)sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INVALID_ARGUMENT,
        "whole graph control binding ABI or buffer is invalid");
    return false;
  }
  Buffer *const buffer = lookup<Buffer>(binding->buffer, HandleKind::Buffer);
  if (buffer == nullptr || buffer->context != context ||
      buffer->release_active || binding->byte_offset > buffer->size_bytes ||
      buffer->size_bytes - binding->byte_offset < UINT64_C(144)) {
    (void)sllm_public_runtime::write_error(
        error_sink,
        buffer == nullptr ? SLLM_STATUS_PUBLIC_INVALID_HANDLE
                          : SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
        "whole graph control binding buffer is stale, foreign, or out of "
        "bounds");
    return false;
  }
  *buffer_output = buffer;
  *byte_offset_output = binding->byte_offset;
  return true;
}

bool graph_span_binding_element_bytes(const uint32_t dtype,
                                      uint64_t *const bytes) noexcept {
  if (bytes == nullptr) {
    return false;
  }
  switch (dtype) {
  case SLLM_TENSOR_DTYPE_BF16:
  case SLLM_TENSOR_DTYPE_F16:
    *bytes = 2U;
    return true;
  case SLLM_TENSOR_DTYPE_F32:
  case SLLM_TENSOR_DTYPE_I32:
    *bytes = 4U;
    return true;
  case SLLM_TENSOR_DTYPE_F8_E4M3_FN:
  case SLLM_TENSOR_DTYPE_F8_E4M3_FNUZ:
  case SLLM_TENSOR_DTYPE_F8_E5M2:
  case SLLM_TENSOR_DTYPE_U8:
    *bytes = 1U;
    return true;
  default:
    return false;
  }
}

bool graph_span_binding_end(const sllm_tensor_binding_t &binding,
                            uint64_t *const end_offset) noexcept {
  if (end_offset == nullptr || binding.struct_size != sizeof(binding) ||
      binding.abi_version != SLLM_HIP_ABI_VERSION ||
      binding.buffer == nullptr || binding.rank == 0U ||
      binding.rank > SLLM_HIP_TENSOR_MAX_RANK ||
      binding.encoding != SLLM_TENSOR_ENCODING_UNQUANTIZED ||
      binding.reserved0 != 0U || binding.reserved[0] != 0U ||
      binding.reserved[1] != 0U) {
    return false;
  }
  uint64_t element_bytes = 0U;
  if (!graph_span_binding_element_bytes(binding.dtype, &element_bytes)) {
    return false;
  }
  uint64_t max_element = 0U;
  for (uint32_t index = 0U; index != binding.rank; ++index) {
    if (binding.shape[index] == 0U) {
      return false;
    }
    const uint64_t extent = binding.shape[index] - 1U;
    if (extent != 0U &&
        binding.stride_elements[index] > (UINT64_MAX - max_element) / extent) {
      return false;
    }
    max_element += extent * binding.stride_elements[index];
  }
  if (max_element == UINT64_MAX ||
      max_element + 1U > UINT64_MAX / element_bytes) {
    return false;
  }
  const uint64_t payload = (max_element + 1U) * element_bytes;
  if (binding.byte_offset > UINT64_MAX - payload) {
    return false;
  }
  *end_offset = binding.byte_offset + payload;
  return true;
}

bool graph_span_binding_is_null(const sllm_tensor_binding_t &binding) noexcept {
  return binding.buffer == nullptr && binding.byte_offset == 0U &&
         binding.dtype == 0U && binding.encoding == 0U && binding.rank == 0U &&
         binding.reserved0 == 0U && binding.reserved[0] == 0U &&
         binding.reserved[1] == 0U &&
         std::all_of(std::begin(binding.shape), std::end(binding.shape),
                     [](const uint64_t value) { return value == 0U; }) &&
         std::all_of(std::begin(binding.stride_elements),
                     std::end(binding.stride_elements),
                     [](const uint64_t value) { return value == 0U; });
}

bool graph_span_pin_command_buffer(GraphSpan *const span,
                                   Buffer *const buffer) {
  if (span == nullptr || buffer == nullptr) {
    return false;
  }
  if (buffer == span->control_buffer) {
    return true;
  }
  if (std::find(span->capture_buffers.begin(), span->capture_buffers.end(),
                buffer) != span->capture_buffers.end()) {
    return true;
  }
  std::lock_guard<std::mutex> accounting_lock(span->context->accounting_mutex);
  if (!sllm_public_runtime::AccountingState::reserve_two_buffer_prepared_plan(
          span->context->accounting, buffer->accounting, buffer->accounting)) {
    return false;
  }
  try {
    span->capture_buffers.push_back(buffer);
  } catch (...) {
    (void)
        sllm_public_runtime::AccountingState::release_two_buffer_prepared_plan(
            span->context->accounting, buffer->accounting, buffer->accounting);
    return false;
  }
  return true;
}

/* Capture-aware native consumers that do not return a Completion (for
 * example, the PQ selector's device-only route) use this hook to retain each
 * backing allocation until the graph span is released.  It is deliberately
 * private to this translation unit: callers must already be on the capture
 * thread and the helper refuses every non-whole-capture call. */
[[maybe_unused]] bool
whole_graph_pin_buffer(Buffer *const buffer,
                       sllm_error_sink_t *const error_sink) noexcept {
  if (!whole_graph_capture_active() || graph_capture_span == nullptr ||
      buffer == nullptr || buffer->context != graph_capture_span->context ||
      buffer->release_active || buffer->device_pointer == nullptr) {
    (void)sllm_public_runtime::write_error(
        error_sink,
        buffer == nullptr ? SLLM_STATUS_INVALID_ARGUMENT
                          : SLLM_STATUS_PUBLIC_BUSY,
        "whole graph buffer pin requires an active matching capture");
    return false;
  }
  if (!graph_span_pin_command_buffer(graph_capture_span, buffer)) {
    (void)sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "whole graph buffer pin accounting exhausted");
    return false;
  }
  return true;
}

bool graph_span_pin_completion_node(GraphSpan *const span,
                                    QuarantineNode *const node) noexcept {
  if (span == nullptr || node == nullptr) {
    return false;
  }
  if (std::find(span->capture_graph_nodes.begin(),
                span->capture_graph_nodes.end(),
                node) != span->capture_graph_nodes.end()) {
    return true;
  }
  std::lock_guard<std::mutex> accounting_lock(span->context->accounting_mutex);
  if (node->graph_pins == UINT64_MAX) {
    return false;
  }
  ++node->graph_pins;
  try {
    span->capture_graph_nodes.push_back(node);
  } catch (...) {
    --node->graph_pins;
    return false;
  }
  return true;
}

GraphKvStateBinding *graph_span_find_kv_state(GraphSpan *const span,
                                              KvState *const state) noexcept {
  if (span == nullptr || state == nullptr) {
    return nullptr;
  }
  for (GraphKvStateBinding &binding : span->kv_states) {
    if (binding.state == state) {
      return &binding;
    }
  }
  return nullptr;
}

/* The device logical table keeps one immutable host snapshot alive while its
 * update is in flight.  A graph preflight must never wait for that event on
 * its successful path: the queue ordering already places the table update
 * before the graph launch.  Cleanup is different; it is allowed to drain
 * the queue so a speculative table entry cannot outlive its host reservation
 * or be observed after the physical block is returned to the pool. */
bool graph_span_poll_paged_table(KvState *const state, Queue *const queue,
                                 const bool allow_sync,
                                 sllm_error_sink_t *const error_sink) noexcept {
  if (state == nullptr || queue == nullptr ||
      state->paged_logical_table == nullptr) {
    return false;
  }
  sllm_paged_kv::DeviceStatus status =
      state->paged_logical_table->poll_pending();
  if (status == sllm_paged_kv::DeviceStatus::Busy && allow_sync) {
    const hipError_t sync = hipStreamSynchronize(queue->stream);
    if (sync != hipSuccess) {
      state->context->poisoned.store(true);
      (void)hip_failure(error_sink, sync,
                        "synchronize paged graph table cleanup");
      return false;
    }
    status = state->paged_logical_table->poll_pending();
  }
  if (status == sllm_paged_kv::DeviceStatus::Ok) {
    return true;
  }
  if (status == sllm_paged_kv::DeviceStatus::Busy) {
    (void)sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_PUBLIC_BUSY,
        "paged graph logical table update is still in flight");
  } else {
    state->context->poisoned.store(true);
    (void)sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "paged graph logical table update failed or was poisoned");
  }
  return false;
}

bool graph_span_restore_paged_table(
    KvState *const state, Queue *const queue,
    const std::vector<sllm_paged_kv::LogicalEntryUpdate> &updates,
    sllm_error_sink_t *const error_sink) noexcept {
  if (updates.empty()) {
    return true;
  }
  if (state == nullptr || queue == nullptr ||
      state->paged_logical_table == nullptr ||
      !graph_span_poll_paged_table(state, queue, true, error_sink)) {
    return false;
  }
  const sllm_paged_kv::DeviceStatus restore =
      state->paged_logical_table->restore_entries(updates, queue->stream);
  if (restore != sllm_paged_kv::DeviceStatus::Ok) {
    state->context->poisoned.store(true);
    (void)sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "paged graph logical table restore failed");
    return false;
  }
  if (!graph_span_poll_paged_table(state, queue, true, error_sink)) {
    return false;
  }
  return true;
}

bool graph_span_rollback_paged_preflight(
    KvState *const state, Queue *const queue,
    sllm_paged_kv::State::Append &reservation,
    const std::vector<sllm_paged_kv::LogicalEntryUpdate> &updates,
    const bool table_update_queued,
    sllm_error_sink_t *const error_sink) noexcept {
  if (state == nullptr || queue == nullptr ||
      state->paged_host_state == nullptr) {
    return false;
  }
  if (table_update_queued &&
      !graph_span_restore_paged_table(state, queue, updates, error_sink)) {
    return false;
  }
  const sllm_paged_kv::Status rollback =
      state->paged_host_state->rollback_graph_reservation(reservation);
  if (rollback != sllm_paged_kv::Status::Ok) {
    state->context->poisoned.store(true);
    (void)sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "paged graph host reservation rollback failed");
    return false;
  }
  return true;
}

/* Finish one staged paged state after the graph has drained.  `final_length`
 * is allowed to be below the conservative bound, so logical entries in the
 * unused suffix are restored on the same stream before the released blocks
 * can be reused.  A normal publication does not synchronize; release and
 * abort cleanup pass allow_sync=true through the table poll helper. */
bool graph_span_finish_paged_binding(
    GraphKvStateBinding &binding, GraphSpan *const span,
    const uint64_t final_length, const uint64_t successful_generations,
    const bool allow_sync, sllm_error_sink_t *const error_sink) noexcept {
  KvState *const state = binding.state;
  if (span == nullptr || state == nullptr || !state->paged ||
      state->paged_host_state == nullptr ||
      state->paged_logical_table == nullptr || span->queue == nullptr) {
    return false;
  }
  if (!state->paged_host_state->graph_staged()) {
    return binding.paged_table_updates.empty();
  }
  if (!graph_span_poll_paged_table(state, span->queue, allow_sync,
                                   error_sink)) {
    return false;
  }

  std::vector<sllm_paged_kv::LogicalEntryUpdate> trim;
  try {
    const uint64_t used_blocks =
        final_length / sllm_paged_kv::kBlockTokens +
        (final_length % sllm_paged_kv::kBlockTokens != 0U);
    trim.reserve(binding.paged_table_updates.size());
    for (const sllm_paged_kv::LogicalEntryUpdate &update :
         binding.paged_table_updates) {
      if (static_cast<uint64_t>(update.logical_block) >= used_blocks) {
        trim.push_back(sllm_paged_kv::LogicalEntryUpdate{
            update.logical_block, update.new_physical,
            update.expected_physical});
      }
    }
  } catch (...) {
    return false;
  }

  const sllm_paged_kv::Status finish = state->paged_host_state->finish_graph(
      final_length, successful_generations);
  if (finish != sllm_paged_kv::Status::Ok) {
    state->context->poisoned.store(true);
    (void)sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "paged graph host state finalization failed");
    return false;
  }
  if (!trim.empty()) {
    const sllm_paged_kv::DeviceStatus update =
        state->paged_logical_table->update_entries(trim, span->queue->stream);
    if (update != sllm_paged_kv::DeviceStatus::Ok) {
      state->context->poisoned.store(true);
      (void)sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "paged graph logical table suffix trim failed");
      return false;
    }
    if (allow_sync &&
        !graph_span_poll_paged_table(state, span->queue, true, error_sink)) {
      return false;
    }
  }
  binding.paged_table_updates.clear();
  return true;
}

bool graph_span_finish_paged_states(
    GraphSpan *const span, const uint64_t final_length,
    const uint64_t successful_generations, const bool allow_sync,
    sllm_error_sink_t *const error_sink) noexcept {
  if (span == nullptr) {
    return false;
  }
  for (GraphKvStateBinding &binding : span->kv_states) {
    if (binding.state == nullptr || !binding.state->paged) {
      continue;
    }
    if (!graph_span_finish_paged_binding(binding, span, final_length,
                                         successful_generations, allow_sync,
                                         error_sink)) {
      return false;
    }
  }
  return true;
}

bool graph_span_abort_paged_states(
    GraphSpan *const span, sllm_error_sink_t *const error_sink) noexcept {
  if (span == nullptr) {
    return false;
  }
  for (GraphKvStateBinding &binding : span->kv_states) {
    if (binding.state == nullptr || !binding.state->paged) {
      continue;
    }
    if (!graph_span_finish_paged_binding(binding, span,
                                         binding.initial_published_length, 0U,
                                         true, error_sink)) {
      return false;
    }
  }
  return true;
}

GraphLinearStateBinding *
graph_span_find_linear_state(GraphSpan *const span,
                             LinearAttentionState *const state) noexcept {
  if (span == nullptr || state == nullptr) {
    return nullptr;
  }
  for (GraphLinearStateBinding &binding : span->linear_states) {
    if (binding.state == state) {
      return &binding;
    }
  }
  return nullptr;
}

bool graph_span_register_kv_state(GraphSpan *const span,
                                  KvState *const state) noexcept {
  if (span == nullptr || state == nullptr || state->context != span->context) {
    return false;
  }
  if (graph_span_find_kv_state(span, state) != nullptr) {
    return true;
  }
  if (state->graph_pins != 0U) {
    return false;
  }
  try {
    span->kv_states.push_back(GraphKvStateBinding{
        state, state->published_length, state->generation, {}});
  } catch (...) {
    return false;
  }
  return true;
}

bool graph_span_register_linear_state(
    GraphSpan *const span, LinearAttentionState *const state) noexcept {
  if (span == nullptr || state == nullptr || state->context != span->context) {
    return false;
  }
  if (graph_span_find_linear_state(span, state) != nullptr) {
    return true;
  }
  if (state->graph_pins != 0U) {
    return false;
  }
  GraphLinearStateBinding binding{};
  binding.state = state;
  binding.anchor_slot = state->active_slot;
  binding.initial_published_length = state->published_length;
  binding.initial_generation = state->generation;
  try {
    span->linear_states.push_back(binding);
  } catch (...) {
    return false;
  }
  return true;
}

bool graph_span_state_is_capture_owner(const GraphSpan *const span) noexcept {
  return span != nullptr && span->whole_capture && span->capture_owner &&
         span->queue != nullptr && graph_capture_queue == span->queue &&
         graph_capture_span == span;
}

[[maybe_unused]] GraphLinearStateBinding *
whole_graph_linear_binding(LinearAttentionState *const state) noexcept {
  if (!graph_span_state_is_capture_owner(graph_capture_span)) {
    return nullptr;
  }
  return graph_span_find_linear_state(graph_capture_span, state);
}

bool graph_span_pin_shared_workspace(GraphSpan *const span, const bool matmul,
                                     const bool causal) noexcept {
  if (span == nullptr || span->context == nullptr || (matmul && causal) ||
      (!matmul && !causal)) {
    return false;
  }
  std::lock_guard<std::mutex> accounting_lock(span->context->accounting_mutex);
  uint64_t *const count =
      matmul ? &span->matmul_workspace_pins : &span->causal_workspace_pins;
  uint64_t *const context_count =
      matmul ? &span->context->matmul_f16_staging_in_flight
             : &span->context->causal_scaled_prefill_in_flight;
  Queue **const owner_queue =
      matmul ? &span->context->matmul_f16_staging_owner_queue
             : &span->context->causal_scaled_prefill_owner_queue;
  if (*owner_queue != nullptr && *owner_queue != span->queue) {
    return false;
  }
  if (*count == UINT64_MAX || *context_count == UINT64_MAX) {
    return false;
  }
  *owner_queue = span->queue;
  ++*count;
  ++*context_count;
  return true;
}

void graph_span_rollback_shared_workspace_pins(
    GraphSpan *const span, const uint64_t matmul_count,
    const uint64_t causal_count) noexcept {
  if (span == nullptr || span->context == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> accounting_lock(span->context->accounting_mutex);
  if (matmul_count != 0U) {
    if (span->matmul_workspace_pins < matmul_count ||
        span->context->matmul_f16_staging_in_flight < matmul_count) {
      span->context->poisoned.store(true);
    } else {
      span->matmul_workspace_pins -= matmul_count;
      span->context->matmul_f16_staging_in_flight -= matmul_count;
      if (span->context->matmul_f16_staging_in_flight == 0U) {
        span->context->matmul_f16_staging_owner_queue = nullptr;
      }
    }
  }
  if (causal_count != 0U) {
    if (span->causal_workspace_pins < causal_count ||
        span->context->causal_scaled_prefill_in_flight < causal_count) {
      span->context->poisoned.store(true);
    } else {
      span->causal_workspace_pins -= causal_count;
      span->context->causal_scaled_prefill_in_flight -= causal_count;
      if (span->context->causal_scaled_prefill_in_flight == 0U) {
        span->context->causal_scaled_prefill_owner_queue = nullptr;
      }
    }
  }
}

void graph_span_rollback_completion_pins(
    GraphSpan *const span, const std::size_t buffer_count,
    const std::size_t node_count, const uint64_t matmul_count,
    const uint64_t causal_count, const std::size_t kv_state_count,
    const std::size_t linear_state_count) noexcept {
  if (span == nullptr || span->context == nullptr) {
    return;
  }
  {
    std::lock_guard<std::mutex> accounting_lock(
        span->context->accounting_mutex);
    while (span->capture_buffers.size() > buffer_count) {
      Buffer *const buffer = span->capture_buffers.back();
      span->capture_buffers.pop_back();
      if (buffer == nullptr || buffer == span->control_buffer) {
        continue;
      }
      if (!sllm_public_runtime::AccountingState::
              release_two_buffer_prepared_plan(span->context->accounting,
                                               buffer->accounting,
                                               buffer->accounting)) {
        span->context->poisoned.store(true);
      }
    }
    while (span->capture_graph_nodes.size() > node_count) {
      QuarantineNode *const node = span->capture_graph_nodes.back();
      span->capture_graph_nodes.pop_back();
      if (node != nullptr && node->graph_pins != 0U) {
        --node->graph_pins;
      }
    }
  }
  if (span->kv_states.size() > kv_state_count) {
    span->kv_states.resize(kv_state_count);
  }
  if (span->linear_states.size() > linear_state_count) {
    span->linear_states.resize(linear_state_count);
  }
  graph_span_rollback_shared_workspace_pins(span, matmul_count, causal_count);
}

bool graph_span_pin_completion_resources(
    GraphSpan *const span, Completion *const completion) noexcept {
  if (span == nullptr || completion == nullptr) {
    return false;
  }
  const std::size_t buffer_count = span->capture_buffers.size();
  const std::size_t node_count = span->capture_graph_nodes.size();
  const uint64_t matmul_count = span->matmul_workspace_pins;
  const uint64_t causal_count = span->causal_workspace_pins;
  const std::size_t kv_state_count = span->kv_states.size();
  const std::size_t linear_state_count = span->linear_states.size();
  const auto pin_node = [span](QuarantineNode *const node) {
    return graph_span_pin_completion_node(span, node);
  };
  const auto pin_buffer = [span](Buffer *const buffer) {
    return buffer != nullptr && graph_span_pin_command_buffer(span, buffer);
  };
  const auto pin_plan_and_buffers =
      [&](QuarantineNode *const plan,
          const std::initializer_list<Buffer *> buffers) {
        if (plan == nullptr || !pin_node(plan)) {
          return false;
        }
        for (Buffer *const buffer : buffers) {
          if (buffer == nullptr || !pin_buffer(buffer)) {
            return false;
          }
        }
        return true;
      };
  bool ok = true;
  if (completion->rmsnorm) {
    ok = pin_plan_and_buffers(completion->rmsnorm_plan,
                              {completion->rmsnorm_activation,
                               completion->rmsnorm_raw_scale,
                               completion->rmsnorm_output});
    if (ok && completion->rmsnorm_fused) {
      ok = pin_buffer(completion->rmsnorm_addend) &&
           pin_buffer(completion->rmsnorm_residual_output);
    }
  } else if (completion->elementwise) {
    ok = pin_plan_and_buffers(completion->elementwise_plan,
                              {completion->elementwise_input0,
                               completion->elementwise_input1,
                               completion->elementwise_output});
  } else if (completion->matmul) {
    ok = pin_plan_and_buffers(completion->matmul_plan,
                              {completion->matmul_activation,
                               completion->matmul_weight,
                               completion->matmul_output});
    if (ok && completion->matmul_f16_staging_workspace_acquired) {
      ok = graph_span_pin_shared_workspace(span, true, false);
    }
  } else if (completion->gdn_bundle) {
    ok = pin_node(completion->gdn_bundle_plan);
    for (Buffer *const buffer : completion->gdn_bundle_weights) {
      ok = ok && pin_buffer(buffer);
    }
    for (Buffer *const buffer : completion->gdn_bundle_outputs) {
      ok = ok && pin_buffer(buffer);
    }
    ok = ok && pin_buffer(completion->gdn_bundle_activation);
  } else if (completion->mlp_bundle) {
    ok = pin_node(completion->mlp_bundle_plan);
    ok = ok && pin_buffer(completion->mlp_bundle_activation);
    for (Buffer *const buffer : completion->mlp_bundle_weights) {
      ok = ok && pin_buffer(buffer);
    }
    for (Buffer *const buffer : completion->mlp_bundle_outputs) {
      ok = ok && pin_buffer(buffer);
    }
  } else if (completion->qwen38_projection_pack2) {
    ok = pin_node(completion->qwen38_projection_pack2_plan);
    ok = ok && pin_buffer(completion->qwen38_projection_pack2_activation);
    for (Buffer *const buffer : completion->qwen38_projection_pack2_weights) {
      ok = ok && pin_buffer(buffer);
    }
    for (Buffer *const buffer : completion->qwen38_projection_pack2_outputs) {
      ok = ok && pin_buffer(buffer);
    }
  } else if (completion->argmax) {
    ok = pin_plan_and_buffers(
        completion->argmax_plan,
        {completion->argmax_logits, completion->argmax_output});
  } else if (completion->token_selector) {
    ok = pin_node(completion->token_selector_plan);
    ok = ok && pin_buffer(completion->token_selector_logits);
    ok = ok && pin_buffer(completion->token_selector_additive);
    ok = ok && pin_buffer(completion->token_selector_valid_mask);
    ok = ok && pin_buffer(completion->token_selector_output);
    if (completion->token_selector_workspace != nullptr) {
      ok = ok && pin_buffer(completion->token_selector_workspace);
    }
  } else if (completion->array_operation) {
    ok = pin_node(completion->array_operation_plan);
    for (Buffer *const buffer : completion->array_operation_buffers) {
      ok = ok && pin_buffer(buffer);
    }
  } else if (completion->kv_state_append) {
    ok = graph_span_register_kv_state(span, completion->kv_state);
    ok = ok && pin_node(completion->kv_state);
    ok = ok && pin_buffer(completion->kv_key_input);
    ok = ok && pin_buffer(completion->kv_value_input);
    ok = ok && pin_buffer(completion->kv_key_buffer);
    ok = ok && pin_buffer(completion->kv_value_buffer);
  } else if (completion->causal_attention) {
    ok = graph_span_register_kv_state(span, completion->causal_attention_state);
    ok = ok && pin_node(completion->causal_attention_state);
    for (Buffer *const buffer : completion->causal_attention_buffers) {
      ok = ok && pin_buffer(buffer);
    }
    if (ok && completion->causal_scaled_prefill_plan_acquired) {
      ok = graph_span_pin_shared_workspace(span, false, true);
    }
  } else if (completion->linear_attention) {
    ok = graph_span_register_linear_state(span,
                                          completion->linear_attention_state);
    ok = ok && pin_node(completion->linear_attention_state);
    for (Buffer *const buffer : completion->linear_attention_buffers) {
      ok = ok && pin_buffer(buffer);
    }
  } else if (completion->d2d_copy) {
    ok = pin_buffer(completion->d2d_source);
    ok = ok && pin_buffer(completion->d2d_destination);
  } else if (completion->buffer != nullptr) {
    ok = pin_buffer(completion->buffer);
  }
  if (!ok) {
    graph_span_rollback_completion_pins(span, buffer_count, node_count,
                                        matmul_count, causal_count,
                                        kv_state_count, linear_state_count);
  }
  return ok;
}

void graph_span_unregister_and_delete(GraphSpan *const span,
                                      const sllm_graph_span_t *const raw_span,
                                      const bool unpin) noexcept {
  if (span == nullptr) {
    return;
  }
  {
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    if (raw_span != nullptr) {
      unregister_handle(raw_span);
    }
  }
  if (unpin) {
    graph_span_unpin(span);
  }
  delete span;
}

/* The stream capture has ended before callers use this helper.  A destroy
 * failure leaves the graph and its plan/queue pins in the durable quarantine;
 * successful destruction releases the temporary capture ownership. */
bool graph_span_abort_after_capture(std::unique_ptr<GraphSpan> &owner,
                                    hipError_t *const failure) noexcept {
  GraphSpan *const span = owner.get();
  if (span == nullptr) {
    return true;
  }
  graph_span_quarantine_markers(span);
  if (!graph_span_destroy_graph_checked(span, failure)) {
    graph_span_finish_capture(span);
    retain_poisoned(owner, span->context);
    return false;
  }
  graph_span_unpin(span);
  return true;
}

bool release_graph_span_active(Completion *const completion) noexcept {
  if (completion == nullptr || completion->queue == nullptr ||
      completion->graph_span_owner == nullptr ||
      completion->queue->accounting.active_submissions == 0U) {
    return false;
  }
  if (completion->graph_span_owner->whole_capture &&
      completion->graph_span_owner->whole_replay_in_flight == 0U) {
    return false;
  }
  --completion->queue->accounting.active_submissions;
  if (completion->graph_span_owner->whole_capture) {
    --completion->graph_span_owner->whole_replay_in_flight;
  } else {
    completion->graph_span_owner->in_flight = false;
  }
  return true;
}

bool rollback_graph_span_completion(Completion *const completion) noexcept {
  if (completion == nullptr || completion->context == nullptr ||
      completion->queue == nullptr || completion->graph_span_owner == nullptr) {
    return false;
  }
  const bool released =
      sllm_public_runtime::AccountingState::rollback_queue_fence(
          completion->context->accounting, completion->queue->accounting);
  if (released) {
    if (completion->graph_span_owner->whole_capture) {
      if (completion->graph_span_owner->whole_replay_in_flight == 0U) {
        completion->context->poisoned.store(true);
        return false;
      }
      --completion->graph_span_owner->whole_replay_in_flight;
    } else {
      completion->graph_span_owner->in_flight = false;
    }
  }
  return released;
}

bool release_graph_span_completion(Completion *const completion) noexcept {
  if (completion == nullptr || completion->context == nullptr ||
      completion->queue == nullptr || completion->graph_span_owner == nullptr) {
    return false;
  }
  return sllm_public_runtime::AccountingState::release_queue_fence_completion(
      completion->context->accounting, completion->queue->accounting);
}

} // namespace

/* Capture-aware execute entry points transfer their completion owner here.
 * The caller retains the handle only on failure; success consumes it and
 * GraphSpan keeps the operation/state/buffer pins until final fence release. */
extern "C" sllm_status_t
sllm_graph_span_capture_marker(sllm_graph_span_t *const raw_span,
                               sllm_completion_t **const raw_marker,
                               sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_span == nullptr || raw_marker == nullptr ||
        *raw_marker == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "whole graph capture span or marker is null");
    }
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    GraphSpan *const span = lookup<GraphSpan>(raw_span, HandleKind::GraphSpan);
    Completion *const completion =
        lookup<Completion>(*raw_marker, HandleKind::Completion);
    if (span == nullptr || completion == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "whole graph capture span or marker handle is stale");
    }
    if (!span->whole_capture || !span->capture_owner ||
        span->queue == nullptr || completion->queue != span->queue ||
        span->release_active || span->capture_failed) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "whole graph capture marker registration is outside active capture");
    }
    if (std::find(span->capture_markers.begin(), span->capture_markers.end(),
                  *raw_marker) != span->capture_markers.end()) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "whole graph capture marker was registered twice");
    }
    const std::size_t buffer_count = span->capture_buffers.size();
    const std::size_t node_count = span->capture_graph_nodes.size();
    const uint64_t matmul_count = span->matmul_workspace_pins;
    const uint64_t causal_count = span->causal_workspace_pins;
    const std::size_t kv_state_count = span->kv_states.size();
    const std::size_t linear_state_count = span->linear_states.size();
    if (!graph_span_pin_completion_resources(span, completion)) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "whole graph capture marker resource pin failed");
    }
    if (!rollback_submission_references(completion)) {
      graph_span_rollback_completion_pins(span, buffer_count, node_count,
                                          matmul_count, causal_count,
                                          kv_state_count, linear_state_count);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "whole graph capture marker accounting rollback failed");
    }
    {
      std::lock_guard<std::mutex> completion_lock(completion->state_mutex);
      completion->terminal = true;
      completion->success = true;
      completion->safe_to_release = true;
      completion->event_destroyed = true;
      completion->safety.observe_positive_completion();
    }
    try {
      span->capture_markers.reserve(span->capture_markers.size() + 1U);
      span->capture_markers.push_back(*raw_marker);
    } catch (...) {
      graph_span_rollback_completion_pins(span, buffer_count, node_count,
                                          matmul_count, causal_count,
                                          kv_state_count, linear_state_count);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "whole graph capture marker registration allocation failed");
    }
    *raw_marker = nullptr;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "whole graph capture marker registration failed");
  }
}

extern "C" sllm_status_t sllm_graph_span_decode_command(
    const sllm_graph_span_t *const raw_span,
    const sllm_graph_span_decode_command_desc_t *const command,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_span == nullptr || command == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "whole graph decode command span or descriptor is null");
    }
    if (command->struct_size != sizeof(*command) ||
        command->abi_version != SLLM_HIP_ABI_VERSION ||
        command->info_version != SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_VERSION ||
        command->reserved0 != 0U) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ABI_VERSION,
          "whole graph decode command ABI is unsupported");
    }

    GraphSpan *span = nullptr;
    Queue *queue = nullptr;
    std::array<Buffer *, 4> buffers{};
    std::array<void *, 4> pointers{};
    const std::array<const sllm_tensor_binding_t *, 4> bindings = {
        &command->input0, &command->input1, &command->output,
        &command->stop_ids};
    std::array<bool, 4> binding_used{};
    switch (command->opcode) {
    case SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_BEGIN_PHASE:
      if (!graph_span_binding_is_null(command->input0) ||
          !graph_span_binding_is_null(command->input1) ||
          !graph_span_binding_is_null(command->output) ||
          !graph_span_binding_is_null(command->stop_ids) ||
          command->rows == 0U) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INVALID_ARGUMENT,
            "begin-phase decode command has unexpected bindings or rows");
      }
      break;
    case SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_GATHER_SELECTOR:
    case SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_GATHER_DECISION:
    case SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_GATHER_RESULT:
      binding_used[0] = true;
      binding_used[2] = true;
      if (!graph_span_binding_is_null(command->input1) ||
          !graph_span_binding_is_null(command->stop_ids) ||
          command->input0.dtype != SLLM_TENSOR_DTYPE_U8 ||
          command->output.dtype != SLLM_TENSOR_DTYPE_I32 ||
          command->token_capacity == 0U) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INVALID_ARGUMENT,
            "gather decode command has unexpected bindings or capacity");
      }
      break;
    case SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_GATHER_HIDDEN:
      binding_used[0] = true;
      binding_used[2] = true;
      if (!graph_span_binding_is_null(command->input1) ||
          !graph_span_binding_is_null(command->stop_ids) ||
          command->input0.dtype != SLLM_TENSOR_DTYPE_BF16 ||
          command->output.dtype != SLLM_TENSOR_DTYPE_BF16 ||
          command->row_count == 0U || command->hidden_width == 0U) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INVALID_ARGUMENT,
            "gather-hidden decode command has unexpected bindings or shape");
      }
      break;
    case SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_GATHER_ACTIVE_TOKEN:
      binding_used[0] = true;
      binding_used[2] = true;
      if (!graph_span_binding_is_null(command->input1) ||
          !graph_span_binding_is_null(command->stop_ids) ||
          command->input0.dtype != SLLM_TENSOR_DTYPE_I32 ||
          command->output.dtype != SLLM_TENSOR_DTYPE_I32 ||
          command->token_capacity == 0U) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INVALID_ARGUMENT,
            "active-token gather metadata or bindings are invalid");
      }
      break;
    case SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_GATHER_ACTIVE_HIDDEN:
      binding_used[0] = true;
      binding_used[1] = true;
      binding_used[2] = true;
      if (!graph_span_binding_is_null(command->stop_ids) ||
          command->input0.dtype != SLLM_TENSOR_DTYPE_BF16 ||
          command->input1.dtype != SLLM_TENSOR_DTYPE_BF16 ||
          command->output.dtype != SLLM_TENSOR_DTYPE_BF16 ||
          command->row_count == 0U || command->hidden_width == 0U) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INVALID_ARGUMENT,
            "active-hidden gather metadata or bindings are invalid");
      }
      break;
    case SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_COMMIT_STEP:
      binding_used[0] = true;
      binding_used[2] = true;
      if (command->input_kind > 1U || command->vocabulary_size == 0U ||
          command->stop_count > 16U ||
          command->input0.dtype != SLLM_TENSOR_DTYPE_U8 ||
          command->output.dtype != SLLM_TENSOR_DTYPE_U8 ||
          (command->stop_count != 0U &&
           command->stop_ids.dtype != SLLM_TENSOR_DTYPE_I32) ||
          (command->stop_count == 0U &&
           !graph_span_binding_is_null(command->stop_ids)) ||
          (command->stop_count != 0U && command->stop_ids.buffer == nullptr)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INVALID_ARGUMENT,
            "commit decode command metadata is invalid");
      }
      if (command->stop_count != 0U) {
        binding_used[3] = true;
      }
      break;
    case SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_VERIFY_FIXED_K20_PQ:
      binding_used[0] = true;
      binding_used[1] = true;
      binding_used[2] = true;
      binding_used[3] = true;
      if (command->token_capacity == 0U ||
          command->token_capacity > sllm_token_selector_pq::kMaxWidth ||
          command->row_count != command->token_capacity + 1U ||
          command->phase_kind != 0U || command->phase_index != 0U ||
          command->rows != 0U || command->input_kind != 0U ||
          command->hidden_width != 0U || command->vocabulary_size != 0U ||
          command->stop_count != 0U ||
          command->input0.dtype != SLLM_TENSOR_DTYPE_U8 ||
          command->input1.dtype != SLLM_TENSOR_DTYPE_U8 ||
          command->output.dtype != SLLM_TENSOR_DTYPE_U8 ||
          command->stop_ids.dtype != SLLM_TENSOR_DTYPE_I32) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INVALID_ARGUMENT,
            "graph fixed-K20 PQ command metadata or binding dtypes are "
            "invalid");
      }
      break;
    default:
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "unknown whole graph decode command opcode");
    }

    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      span = lookup<GraphSpan>(raw_span, HandleKind::GraphSpan);
      if (span == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "whole graph decode command span handle is stale");
      }
      queue = span->queue;
      if (!span->whole_capture || !span->capture_owner ||
          span->capture_failed || queue == nullptr ||
          graph_capture_queue != queue || graph_capture_span != span ||
          span->graph != nullptr || span->context->poisoned.load()) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "whole graph decode command is outside active capture");
      }
      if (whole_graph_capture_device_control() == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INVALID_ARGUMENT,
            "whole graph decode command has no device control plane");
      }
      for (std::size_t index = 0U; index != bindings.size(); ++index) {
        if (!binding_used[index]) {
          continue;
        }
        uint64_t end_offset = 0U;
        if (!graph_span_binding_end(*bindings[index], &end_offset)) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_INVALID_ARGUMENT,
              "whole graph decode command binding is malformed");
        }
        buffers[index] =
            lookup<Buffer>(bindings[index]->buffer, HandleKind::Buffer);
        if (buffers[index] == nullptr) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
              "whole graph decode command binding buffer is stale");
        }
        if (buffers[index]->context != span->context ||
            buffers[index]->release_active) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
              "whole graph decode command binding context differs");
        }
        if (end_offset > buffers[index]->size_bytes) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
              "whole graph decode command binding exceeds its buffer");
        }
        pointers[index] =
            static_cast<char *>(buffers[index]->device_pointer) +
            static_cast<std::size_t>(bindings[index]->byte_offset);
      }
      if (command->opcode ==
          SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_VERIFY_FIXED_K20_PQ) {
        const uint64_t width = command->token_capacity;
        const uint64_t target_bytes =
            (width + 1U) * sizeof(sllm_token_selector_pq::SupportRecordV1);
        const uint64_t draft_bytes =
            width * sizeof(sllm_token_selector_pq::SupportRecordV1);
        const uint64_t decision_bytes =
            sizeof(sllm_token_selector_pq::DecisionRecordV1);
        const uint64_t draft_ids_bytes =
            width * sizeof(uint32_t) +
            sizeof(sllm_decode_control::SelectorRecordV1);
        if (buffers[0]->size_bytes - bindings[0]->byte_offset < target_bytes ||
            buffers[1]->size_bytes - bindings[1]->byte_offset < draft_bytes ||
            buffers[2]->size_bytes - bindings[2]->byte_offset <
                decision_bytes ||
            buffers[3]->size_bytes - bindings[3]->byte_offset <
                draft_ids_bytes) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
              "graph fixed-K20 PQ binding is smaller than its records");
        }
        const auto address_aligned = [](const void *const pointer,
                                        const uint64_t alignment) noexcept {
          return pointer != nullptr &&
                 (reinterpret_cast<uintptr_t>(pointer) % alignment) == 0U;
        };
        if (!address_aligned(pointers[0], 16U) ||
            !address_aligned(pointers[1], 16U) ||
            !address_aligned(pointers[2], 16U) ||
            !address_aligned(pointers[3], 4U)) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_MISALIGNED_OFFSET,
              "graph fixed-K20 PQ records have unsupported alignment");
        }
        const auto ranges_overlap =
            [](const void *const left, const uint64_t left_bytes,
               const void *const right, const uint64_t right_bytes) noexcept {
              const uint64_t left_address =
                  static_cast<uint64_t>(reinterpret_cast<uintptr_t>(left));
              const uint64_t right_address =
                  static_cast<uint64_t>(reinterpret_cast<uintptr_t>(right));
              if (left_address > UINT64_MAX - left_bytes ||
                  right_address > UINT64_MAX - right_bytes) {
                return true;
              }
              return left_address < right_address + right_bytes &&
                     right_address < left_address + left_bytes;
            };
        const std::array<uint64_t, 4> bytes = {target_bytes, draft_bytes,
                                               decision_bytes, draft_ids_bytes};
        for (std::size_t left = 0U; left != bytes.size(); ++left) {
          for (std::size_t right = left + 1U; right != bytes.size(); ++right) {
            if (ranges_overlap(pointers[left], bytes[left], pointers[right],
                               bytes[right])) {
              return sllm_public_runtime::write_error(
                  error_sink, SLLM_STATUS_ALIAS_OVERLAP,
                  "graph fixed-K20 PQ record ranges overlap");
            }
          }
        }
      }
    }

    for (std::size_t index = 0U; index != buffers.size(); ++index) {
      if (!binding_used[index] || buffers[index] == nullptr) {
        continue;
      }
      bool already_pinned = false;
      for (std::size_t prior = 0U; prior != index; ++prior) {
        already_pinned = already_pinned || (binding_used[prior] &&
                                            buffers[prior] == buffers[index]);
      }
      if (!already_pinned &&
          !graph_span_pin_command_buffer(span, buffers[index])) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "whole graph decode command buffer pin accounting exhausted");
      }
    }

    auto *const control = static_cast<sllm_decode_control::ControlV1 *>(
        whole_graph_capture_device_control());
    hipError_t launch_status = hipSuccess;
    switch (command->opcode) {
    case SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_BEGIN_PHASE:
      launch_status = sllm_decode_control::launch_begin_phase(
          control, command->phase_kind, command->phase_index, command->rows,
          queue->stream);
      break;
    case SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_GATHER_SELECTOR:
      if (command->token_capacity < 1U ||
          buffers[0]->size_bytes - bindings[0]->byte_offset <
              sizeof(sllm_decode_control::SelectorRecordV1) ||
          buffers[2]->size_bytes - bindings[2]->byte_offset < sizeof(int32_t)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
            "selector gather buffers are too small");
      }
      launch_status = sllm_decode_control::launch_gather_selector_token(
          static_cast<const sllm_decode_control::SelectorRecordV1 *>(
              pointers[0]),
          static_cast<int32_t *>(pointers[2]), queue->stream);
      break;
    case SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_GATHER_DECISION:
      if (buffers[0]->size_bytes - bindings[0]->byte_offset <
              sizeof(sllm_token_selector_pq::DecisionRecordV1) ||
          command->token_capacity >
              (buffers[2]->size_bytes - bindings[2]->byte_offset) /
                  sizeof(int32_t)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
            "decision gather buffers are too small");
      }
      launch_status = sllm_decode_control::launch_gather_decision_tokens(
          control,
          static_cast<const sllm_token_selector_pq::DecisionRecordV1 *>(
              pointers[0]),
          static_cast<int32_t *>(pointers[2]), command->token_capacity,
          queue->stream);
      break;
    case SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_GATHER_RESULT:
      if (buffers[0]->size_bytes - bindings[0]->byte_offset <
              sizeof(sllm_decode_control::ResultV1) ||
          command->token_capacity >
              (buffers[2]->size_bytes - bindings[2]->byte_offset) /
                  sizeof(int32_t)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
            "result gather buffers are too small");
      }
      launch_status = sllm_decode_control::launch_gather_result_tokens(
          static_cast<const sllm_decode_control::ResultV1 *>(pointers[0]),
          static_cast<int32_t *>(pointers[2]), command->token_capacity,
          queue->stream);
      break;
    case SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_GATHER_HIDDEN: {
      const uint64_t elements =
          static_cast<uint64_t>(command->row_count) * command->hidden_width;
      const uint64_t bytes = elements * sizeof(uint16_t);
      if (command->row_count == 0U || command->hidden_width == 0U ||
          elements / command->hidden_width != command->row_count ||
          bytes / sizeof(uint16_t) != elements ||
          bytes > buffers[0]->size_bytes - bindings[0]->byte_offset ||
          static_cast<uint64_t>(command->hidden_width) * sizeof(uint16_t) >
              buffers[2]->size_bytes - bindings[2]->byte_offset) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
            "hidden gather buffers are too small");
      }
      launch_status = sllm_decode_control::launch_gather_hidden(
          control, static_cast<const uint16_t *>(pointers[0]),
          command->row_count, command->hidden_width,
          static_cast<uint16_t *>(pointers[2]), queue->stream);
      break;
    }
    case SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_GATHER_ACTIVE_TOKEN:
      if (static_cast<uint64_t>(command->token_capacity) * sizeof(int32_t) >
              buffers[0]->size_bytes - bindings[0]->byte_offset ||
          buffers[2]->size_bytes - bindings[2]->byte_offset < sizeof(int32_t)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
            "active-token gather buffers are too small");
      }
      launch_status = sllm_decode_control::launch_gather_active_token(
          control, static_cast<const int32_t *>(pointers[0]),
          command->token_capacity, static_cast<int32_t *>(pointers[2]),
          queue->stream);
      break;
    case SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_GATHER_ACTIVE_HIDDEN: {
      const uint64_t row_bytes =
          static_cast<uint64_t>(command->hidden_width) * sizeof(uint16_t);
      const uint64_t target_bytes =
          static_cast<uint64_t>(command->row_count) * row_bytes;
      if (row_bytes == 0U ||
          target_bytes > buffers[0]->size_bytes - bindings[0]->byte_offset ||
          row_bytes > buffers[1]->size_bytes - bindings[1]->byte_offset ||
          row_bytes > buffers[2]->size_bytes - bindings[2]->byte_offset) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
            "active-hidden gather buffers are too small");
      }
      launch_status = sllm_decode_control::launch_gather_active_hidden(
          control, static_cast<const uint16_t *>(pointers[0]),
          command->row_count, static_cast<const uint16_t *>(pointers[1]),
          command->hidden_width, static_cast<uint16_t *>(pointers[2]),
          queue->stream);
      break;
    }
    case SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_COMMIT_STEP: {
      const uint64_t input_bytes =
          command->input_kind == 0U
              ? sizeof(sllm_decode_control::SelectorRecordV1)
              : sizeof(sllm_token_selector_pq::DecisionRecordV1);
      if (buffers[0]->size_bytes - bindings[0]->byte_offset < input_bytes ||
          buffers[2]->size_bytes - bindings[2]->byte_offset <
              sizeof(sllm_decode_control::ResultV1) * 2U ||
          (command->stop_count != 0U &&
           buffers[3]->size_bytes - bindings[3]->byte_offset <
               static_cast<uint64_t>(command->stop_count) * sizeof(uint32_t))) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
            "commit-step buffers are too small");
      }
      launch_status = sllm_decode_control::launch_commit_step(
          control, pointers[0],
          command->input_kind == 0U ? sllm_decode_control::InputKind::Selector
                                    : sllm_decode_control::InputKind::Decision,
          command->stop_count == 0U
              ? nullptr
              : static_cast<const uint32_t *>(pointers[3]),
          command->stop_count, command->vocabulary_size,
          static_cast<sllm_decode_control::ResultV1 *>(pointers[2]),
          queue->stream);
      break;
    }
    case SLLM_HIP_GRAPH_SPAN_DECODE_COMMAND_VERIFY_FIXED_K20_PQ:
      launch_status = sllm_token_selector_pq_kernel::launch_graph(
          static_cast<const sllm_token_selector_pq::SupportRecordV1 *>(
              pointers[0]),
          command->row_count,
          static_cast<const sllm_token_selector_pq::SupportRecordV1 *>(
              pointers[1]),
          command->token_capacity, static_cast<const uint32_t *>(pointers[3]),
          command->token_capacity, command->token_capacity,
          reinterpret_cast<const sllm_decode_control::SelectorRecordV1 *>(
              static_cast<const uint32_t *>(pointers[3]) +
              command->token_capacity),
          control, 0U,
          static_cast<sllm_token_selector_pq::DecisionRecordV1 *>(pointers[2]),
          queue->stream);
      break;
    default:
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "unknown whole graph decode command opcode");
    }
    if (launch_status != hipSuccess) {
      return hip_failure(error_sink, launch_status,
                         "whole graph decode command launch");
    }
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected whole graph decode command failure");
  }
}

extern "C" sllm_status_t sllm_graph_span_bind_linear_state(
    sllm_graph_span_t *const raw_span,
    const sllm_linear_attention_state_t *const raw_state,
    const sllm_tensor_binding_t *const checkpoint_conv,
    const sllm_tensor_binding_t *const checkpoint_recurrent,
    const uint32_t token_count, const uint32_t checkpoint_rows,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_span == nullptr || raw_state == nullptr || token_count == 0U ||
        token_count > 9U || checkpoint_rows > 8U ||
        checkpoint_rows >= token_count ||
        ((checkpoint_conv == nullptr) != (checkpoint_recurrent == nullptr)) ||
        (checkpoint_rows != 0U &&
         (checkpoint_conv == nullptr || checkpoint_recurrent == nullptr))) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "linear graph binding state, checkpoints, or dimensions are invalid");
    }
    if ((checkpoint_conv != nullptr &&
         checkpoint_conv->dtype != SLLM_TENSOR_DTYPE_BF16) ||
        (checkpoint_recurrent != nullptr &&
         checkpoint_recurrent->dtype != SLLM_TENSOR_DTYPE_F32)) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_SHAPE_MISMATCH,
          "linear graph checkpoint dtypes must be BF16 and F32");
    }

    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    GraphSpan *const span = lookup<GraphSpan>(raw_span, HandleKind::GraphSpan);
    LinearAttentionState *const state = lookup<LinearAttentionState>(
        raw_state, HandleKind::LinearAttentionState);
    if (span == nullptr || state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "linear graph binding handle is stale");
    }
    if (!graph_span_state_is_capture_owner(span) ||
        state->context != span->context || state->release_active ||
        state->transition_token != 0U ||
        state->accounting.active_submissions != 0U ||
        state->accounting.completion_references != 0U) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "linear graph binding requires a quiescent capture-owned state");
    }
    if (state->conv_kernel_size < 2U || state->qkv_width == 0U ||
        state->value_heads == 0U || state->head_dim == 0U ||
        state->active_slot > 1U ||
        state->conv_state[state->active_slot] == nullptr ||
        state->conv_state[1U - state->active_slot] == nullptr ||
        state->recurrent_state[state->active_slot] == nullptr ||
        state->recurrent_state[1U - state->active_slot] == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "linear graph binding state planes are invalid");
    }
    uint64_t conv_elements = 0U;
    uint64_t recurrent_elements = 0U;
    if (static_cast<uint64_t>(state->conv_kernel_size - 1U) >
            UINT64_MAX / state->qkv_width ||
        static_cast<uint64_t>(state->value_heads) >
            UINT64_MAX / state->head_dim ||
        static_cast<uint64_t>(state->value_heads) * state->head_dim >
            UINT64_MAX / state->head_dim) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_METADATA_OVERFLOW,
          "linear graph checkpoint plane shape overflowed");
    }
    conv_elements =
        static_cast<uint64_t>(state->conv_kernel_size - 1U) * state->qkv_width;
    recurrent_elements = static_cast<uint64_t>(state->value_heads) *
                         state->head_dim * state->head_dim;
    if (conv_elements > UINT64_MAX / sizeof(uint16_t) ||
        recurrent_elements > UINT64_MAX / sizeof(float) ||
        (checkpoint_rows != 0U &&
         (conv_elements > UINT64_MAX / (static_cast<uint64_t>(checkpoint_rows) *
                                        sizeof(uint16_t)) ||
          recurrent_elements >
              UINT64_MAX /
                  (static_cast<uint64_t>(checkpoint_rows) * sizeof(float))))) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_METADATA_OVERFLOW,
          "linear graph checkpoint byte size overflowed");
    }
    const uint64_t conv_bytes = static_cast<uint64_t>(checkpoint_rows) *
                                conv_elements * sizeof(uint16_t);
    const uint64_t recurrent_bytes = static_cast<uint64_t>(checkpoint_rows) *
                                     recurrent_elements * sizeof(float);
    Buffer *conv_buffer = nullptr;
    Buffer *recurrent_buffer = nullptr;
    uint64_t conv_end = 0U;
    uint64_t recurrent_end = 0U;
    const void *conv_pointer = nullptr;
    const void *recurrent_pointer = nullptr;
    if (checkpoint_conv != nullptr) {
      if (!graph_span_binding_end(*checkpoint_conv, &conv_end) ||
          !graph_span_binding_end(*checkpoint_recurrent, &recurrent_end)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INVALID_ARGUMENT,
            "linear graph checkpoint binding is malformed");
      }
      conv_buffer = lookup<Buffer>(checkpoint_conv->buffer, HandleKind::Buffer);
      recurrent_buffer =
          lookup<Buffer>(checkpoint_recurrent->buffer, HandleKind::Buffer);
      if (conv_buffer == nullptr || recurrent_buffer == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "linear graph checkpoint buffer is stale");
      }
      if (conv_buffer->context != span->context ||
          recurrent_buffer->context != span->context ||
          conv_buffer->release_active || recurrent_buffer->release_active ||
          conv_end > conv_buffer->size_bytes ||
          recurrent_end > recurrent_buffer->size_bytes ||
          conv_end - checkpoint_conv->byte_offset < conv_bytes ||
          recurrent_end - checkpoint_recurrent->byte_offset < recurrent_bytes) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
            "linear graph checkpoint binding exceeds its backing buffer");
      }
      conv_pointer = static_cast<const char *>(conv_buffer->device_pointer) +
                     static_cast<std::size_t>(checkpoint_conv->byte_offset);
      recurrent_pointer =
          static_cast<const char *>(recurrent_buffer->device_pointer) +
          static_cast<std::size_t>(checkpoint_recurrent->byte_offset);
    }
    const auto ranges_overlap =
        [](const void *const left, const uint64_t left_bytes,
           const void *const right, const uint64_t right_bytes) noexcept {
          if (left == nullptr || right == nullptr || left_bytes == 0U ||
              right_bytes == 0U) {
            return false;
          }
          const uint64_t left_address =
              static_cast<uint64_t>(reinterpret_cast<uintptr_t>(left));
          const uint64_t right_address =
              static_cast<uint64_t>(reinterpret_cast<uintptr_t>(right));
          if (left_address > UINT64_MAX - left_bytes ||
              right_address > UINT64_MAX - right_bytes) {
            return true;
          }
          return left_address < right_address + right_bytes &&
                 right_address < left_address + left_bytes;
        };
    const uint64_t conv_plane_bytes = conv_elements * sizeof(uint16_t);
    const uint64_t recurrent_plane_bytes = recurrent_elements * sizeof(float);
    if (ranges_overlap(conv_pointer, conv_bytes, recurrent_pointer,
                       recurrent_bytes) ||
        ranges_overlap(conv_pointer, conv_bytes,
                       state->conv_state[state->active_slot],
                       conv_plane_bytes) ||
        ranges_overlap(conv_pointer, conv_bytes,
                       state->conv_state[1U - state->active_slot],
                       conv_plane_bytes) ||
        ranges_overlap(conv_pointer, conv_bytes,
                       state->recurrent_state[state->active_slot],
                       recurrent_plane_bytes) ||
        ranges_overlap(conv_pointer, conv_bytes,
                       state->recurrent_state[1U - state->active_slot],
                       recurrent_plane_bytes) ||
        ranges_overlap(recurrent_pointer, recurrent_bytes,
                       state->conv_state[state->active_slot],
                       conv_plane_bytes) ||
        ranges_overlap(recurrent_pointer, recurrent_bytes,
                       state->conv_state[1U - state->active_slot],
                       conv_plane_bytes) ||
        ranges_overlap(recurrent_pointer, recurrent_bytes,
                       state->recurrent_state[state->active_slot],
                       recurrent_plane_bytes) ||
        ranges_overlap(recurrent_pointer, recurrent_bytes,
                       state->recurrent_state[1U - state->active_slot],
                       recurrent_plane_bytes)) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_ALIAS_OVERLAP,
          "linear graph checkpoints overlap state or each other");
    }
    GraphLinearStateBinding *const existing_binding =
        graph_span_find_linear_state(span, state);
    if (existing_binding != nullptr &&
        existing_binding->checkpoint_conv != nullptr) {
      if (existing_binding->checkpoint_conv !=
              const_cast<void *>(conv_pointer) ||
          existing_binding->checkpoint_recurrent !=
              const_cast<void *>(recurrent_pointer) ||
          existing_binding->token_count != token_count ||
          existing_binding->checkpoint_rows != checkpoint_rows ||
          existing_binding->conv_elements != conv_elements ||
          existing_binding->recurrent_elements != recurrent_elements) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INVALID_ARGUMENT,
            "linear graph state is already bound to different checkpoints");
      }
      return SLLM_STATUS_OK;
    }
    const std::size_t buffer_count = span->capture_buffers.size();
    const std::size_t node_count = span->capture_graph_nodes.size();
    const uint64_t matmul_count = span->matmul_workspace_pins;
    const uint64_t causal_count = span->causal_workspace_pins;
    const std::size_t kv_state_count = span->kv_states.size();
    const std::size_t linear_state_count = span->linear_states.size();
    if (!graph_span_register_linear_state(span, state) ||
        !graph_span_pin_completion_node(span, state) ||
        (conv_buffer != nullptr &&
         !graph_span_pin_command_buffer(span, conv_buffer)) ||
        (recurrent_buffer != nullptr &&
         !graph_span_pin_command_buffer(span, recurrent_buffer))) {
      graph_span_rollback_completion_pins(span, buffer_count, node_count,
                                          matmul_count, causal_count,
                                          kv_state_count, linear_state_count);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "linear graph checkpoint resource pin failed");
    }
    GraphLinearStateBinding *const binding =
        graph_span_find_linear_state(span, state);
    if (binding == nullptr) {
      graph_span_rollback_completion_pins(span, buffer_count, node_count,
                                          matmul_count, causal_count,
                                          kv_state_count, linear_state_count);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "linear graph binding registration disappeared");
    }
    if (binding->checkpoint_conv != nullptr ||
        binding->checkpoint_recurrent != nullptr) {
      if (binding->checkpoint_conv != const_cast<void *>(conv_pointer) ||
          binding->checkpoint_recurrent !=
              const_cast<void *>(recurrent_pointer) ||
          binding->token_count != token_count ||
          binding->checkpoint_rows != checkpoint_rows ||
          binding->conv_elements != conv_elements ||
          binding->recurrent_elements != recurrent_elements) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INVALID_ARGUMENT,
            "linear graph state is already bound to different checkpoints");
      }
      return SLLM_STATUS_OK;
    }
    binding->checkpoint_conv = const_cast<void *>(conv_pointer);
    binding->checkpoint_recurrent = const_cast<void *>(recurrent_pointer);
    binding->token_count = token_count;
    binding->checkpoint_rows = checkpoint_rows;
    binding->conv_elements = conv_elements;
    binding->recurrent_elements = recurrent_elements;
    binding->anchor_slot = state->active_slot;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception binding linear graph state");
  }
}

extern "C" sllm_status_t sllm_graph_span_select_linear_state(
    sllm_graph_span_t *const raw_span,
    const sllm_linear_attention_state_t *const raw_state,
    const uint32_t token_count, sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_span == nullptr || raw_state == nullptr || token_count == 0U) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "linear graph state selection argument is null or empty");
    }
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    GraphSpan *const span = lookup<GraphSpan>(raw_span, HandleKind::GraphSpan);
    LinearAttentionState *const state = lookup<LinearAttentionState>(
        raw_state, HandleKind::LinearAttentionState);
    if (span == nullptr || state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "linear graph state selection handle is stale");
    }
    GraphLinearStateBinding *const binding =
        graph_span_find_linear_state(span, state);
    if (!graph_span_state_is_capture_owner(span) || binding == nullptr ||
        (binding->checkpoint_rows != 0U &&
         (binding->checkpoint_conv == nullptr ||
          binding->checkpoint_recurrent == nullptr)) ||
        binding->token_count != token_count ||
        state->active_slot != binding->anchor_slot ||
        state->conv_state[state->active_slot] == nullptr ||
        state->conv_state[1U - state->active_slot] == nullptr ||
        state->recurrent_state[state->active_slot] == nullptr ||
        state->recurrent_state[1U - state->active_slot] == nullptr ||
        whole_graph_capture_device_control() == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "linear graph state selection is outside its active binding");
    }
    // Target-only M=1 always commits its sole state row when the generation is
    // valid. The control-aware recurrent kernel already wrote that row into
    // the generation-parity slot, so there is no prefix repair to capture.
    if (binding->token_count == 1U && binding->checkpoint_rows == 0U) {
      return SLLM_STATUS_OK;
    }
    auto *const control = static_cast<sllm_decode_control::ControlV1 *>(
        whole_graph_capture_device_control());
    const hipError_t launch_status =
        ::sllm_linear_attention_kernel::launch_decode_state_select(
            control,
            static_cast<const uint16_t *>(
                state->conv_state[1U - state->active_slot]),
            static_cast<const float *>(
                state->recurrent_state[1U - state->active_slot]),
            binding->checkpoint_conv != nullptr
                ? static_cast<const uint16_t *>(binding->checkpoint_conv)
                : static_cast<const uint16_t *>(
                      state->conv_state[1U - state->active_slot]),
            binding->checkpoint_recurrent != nullptr
                ? static_cast<const float *>(binding->checkpoint_recurrent)
                : static_cast<const float *>(
                      state->recurrent_state[1U - state->active_slot]),
            static_cast<uint16_t *>(state->conv_state[state->active_slot]),
            static_cast<float *>(state->recurrent_state[state->active_slot]),
            binding->conv_elements, binding->recurrent_elements, token_count,
            binding->checkpoint_rows, span->queue->stream);
    if (launch_status != hipSuccess) {
      return hip_failure(error_sink, launch_status,
                         "linear graph state selection launch");
    }
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception selecting linear graph state");
  }
}

extern "C" sllm_status_t
sllm_graph_span_prepare_paged_kv(sllm_graph_span_t *const raw_span,
                                 const uint64_t conservative_end,
                                 sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK)
      return sink_status;
    if (raw_span == nullptr || conservative_end == 0U)
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "graph paged KV preparation requires a span and end position");
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    GraphSpan *const span = lookup<GraphSpan>(raw_span, HandleKind::GraphSpan);
    if (span == nullptr)
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "graph paged KV preparation span is stale");
    if (!span->whole_capture || span->capture_owner || span->release_active ||
        span->graph_exec == nullptr || span->queue == nullptr ||
        span->context == nullptr || span->context->poisoned.load())
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "graph paged KV preparation requires an instantiated whole graph");

    const sllm_status_t device_status =
        select_context_device(span->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      return device_status;
    }

    std::lock_guard<std::mutex> accounting_lock(
        span->context->accounting_mutex);
    for (GraphKvStateBinding &binding : span->kv_states) {
      KvState *const state = binding.state;
      if (state == nullptr || !state->paged) {
        continue;
      }
      if (state->context != span->context || state->release_active ||
          state->graph_pins == 0U || state->transition_token != 0U ||
          state->accounting.active_submissions != 0U ||
          state->accounting.completion_references != 0U ||
          state->published_length != binding.initial_published_length ||
          state->generation != binding.initial_generation ||
          state->paged_host_state == nullptr ||
          state->paged_device_pool == nullptr ||
          state->paged_logical_table == nullptr ||
          state->paged_device_status == nullptr ||
          state->paged_host_state->published_length() !=
              state->published_length ||
          state->paged_host_state->generation() != state->generation) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "paged graph state changed after capture binding");
      }

      const uint64_t target =
          std::min(conservative_end, state->capacity_tokens);
      const uint64_t staged_end =
          state->paged_host_state->graph_staged_length();
      if (target <= staged_end) {
        continue;
      }
      const sllm_paged_kv::DeviceStatus pending =
          state->paged_logical_table->poll_pending();
      if (pending == sllm_paged_kv::DeviceStatus::Busy) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "paged graph logical table update is still in flight");
      }
      if (pending != sllm_paged_kv::DeviceStatus::Ok) {
        state->context->poisoned.store(true);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "paged graph logical table is poisoned");
      }

      sllm_paged_kv::State::Append reservation;
      bool reservation_prepared = false;
      bool table_update_queued = false;
      std::vector<sllm_paged_kv::LogicalEntryUpdate> updates;
      std::vector<sllm_paged_kv::LogicalEntryUpdate> combined;
      try {
        const sllm_paged_kv::Status prepare =
            state->paged_host_state->prepare_graph_through(target, reservation);
        reservation_prepared = prepare == sllm_paged_kv::Status::Ok;
        if (!reservation_prepared) {
          const sllm_status_t status =
              prepare == sllm_paged_kv::Status::PoolExhausted ||
                      prepare == sllm_paged_kv::Status::Capacity
                  ? SLLM_STATUS_KV_CAPACITY_EXCEEDED
              : prepare == sllm_paged_kv::Status::Busy
                  ? SLLM_STATUS_PUBLIC_BUSY
                  : SLLM_STATUS_INTERNAL_ERROR;
          return sllm_public_runtime::write_error(
              error_sink, status,
              "paged graph conservative reservation could not be prepared");
        }
        updates = make_paged_table_updates(reservation);
        combined = binding.paged_table_updates;
        combined.reserve(combined.size() + updates.size());
        combined.insert(combined.end(), updates.begin(), updates.end());

        std::vector<uint32_t> physical_ids;
        physical_ids.reserve(reservation.changes.size());
        for (const sllm_paged_kv::State::Change &change : reservation.changes) {
          if (change.new_physical != sllm_paged_kv::kInvalidBlock) {
            physical_ids.push_back(change.new_physical);
          }
        }
        const sllm_paged_kv::DeviceStatus ensure =
            state->paged_device_pool->ensure_blocks(physical_ids,
                                                    span->queue->stream);
        if (ensure != sllm_paged_kv::DeviceStatus::Ok) {
          if (state->paged_device_pool->poisoned()) {
            state->context->poisoned.store(true);
          }
          throw std::runtime_error("paged graph physical slab reservation");
        }
        const sllm_paged_kv::DeviceStatus table =
            state->paged_logical_table->update_entries(updates,
                                                       span->queue->stream);
        if (table != sllm_paged_kv::DeviceStatus::Ok) {
          throw std::runtime_error("paged graph logical table update");
        }
        table_update_queued = !updates.empty();
        const hipError_t cow =
            copy_paged_cow_planes(state, reservation, span->queue->stream);
        if (cow != hipSuccess) {
          (void)hip_failure(error_sink, cow, "paged graph COW copy");
          throw std::runtime_error("paged graph COW copy");
        }
        const sllm_paged_kv::Status commit =
            state->paged_host_state->commit_graph_reservation(reservation);
        if (commit != sllm_paged_kv::Status::Ok) {
          throw std::runtime_error("paged graph host reservation commit");
        }
        reservation_prepared = false;
        binding.paged_table_updates.swap(combined);
      } catch (...) {
        if (reservation_prepared &&
            !graph_span_rollback_paged_preflight(
                state, span->queue, reservation, updates, table_update_queued,
                error_sink)) {
          span->context->poisoned.store(true);
          unregister_handle(raw_span);
          poison_owner.retain(span);
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_INTERNAL_ERROR,
              "paged graph preflight rollback could not be proven safe");
        }
        if (!reservation_prepared) {
          /* Commit is deliberately the last fallible host operation.  Any
           * failure after it means host and device ownership can no longer
           * be reconciled without a queue-wide recovery protocol. */
          if (state->paged_host_state->graph_staged() &&
              binding.paged_table_updates.empty()) {
            span->context->poisoned.store(true);
            unregister_handle(raw_span);
            poison_owner.retain(span);
          }
        }
        return sllm_public_runtime::write_error(error_sink,
                                                SLLM_STATUS_INTERNAL_ERROR,
                                                "paged graph preflight failed");
      }
    }
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected graph paged KV preparation exception");
  }
}

extern "C" sllm_status_t sllm_graph_span_publish_state_metadata(
    sllm_graph_span_t *const raw_span, const uint64_t expected_initial_position,
    const uint64_t final_position, const uint64_t successful_generations,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_span == nullptr || final_position < expected_initial_position) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "graph state metadata publication range is invalid");
    }
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    GraphSpan *const span = lookup<GraphSpan>(raw_span, HandleKind::GraphSpan);
    if (span == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "graph state metadata publication span is stale");
    }
    if (!span->whole_capture || span->capture_owner || span->release_active ||
        span->whole_replay_in_flight != 0U || span->graph_exec == nullptr ||
        span->queue == nullptr ||
        span->queue->graph_capture_active.load(std::memory_order_acquire)) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "graph state metadata publication requires a drained graph");
    }
    std::lock_guard<std::mutex> accounting_lock(
        span->context->accounting_mutex);
    if (final_position != expected_initial_position &&
        successful_generations == 0U) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "graph state metadata moved without a successful generation");
    }
    for (const GraphKvStateBinding &binding : span->kv_states) {
      KvState *const state = binding.state;
      if (state == nullptr || state->context != span->context ||
          state->release_active || state->graph_pins == 0U ||
          state->published_length != binding.initial_published_length ||
          state->generation != binding.initial_generation ||
          state->transition_token != 0U ||
          state->accounting.active_submissions != 0U ||
          state->accounting.completion_references != 0U ||
          binding.initial_published_length != expected_initial_position ||
          final_position > state->capacity_tokens ||
          binding.initial_generation > UINT64_MAX - successful_generations) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_KV_LENGTH_MISMATCH,
            "captured KV state metadata no longer matches its initial "
            "snapshot");
      }
      if (state->paged) {
        if (state->paged_host_state == nullptr ||
            state->paged_logical_table == nullptr ||
            !state->paged_host_state->graph_staged() ||
            final_position > state->paged_host_state->graph_staged_length() ||
            state->paged_host_state->published_length() !=
                state->published_length ||
            state->paged_host_state->generation() != state->generation ||
            state->paged_logical_table->poll_pending() !=
                sllm_paged_kv::DeviceStatus::Ok) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_PUBLIC_BUSY,
              "paged graph state is not ready for metadata publication");
        }
      }
    }
    for (const GraphLinearStateBinding &binding : span->linear_states) {
      LinearAttentionState *const state = binding.state;
      if (state == nullptr || state->context != span->context ||
          state->release_active || state->graph_pins == 0U ||
          state->published_length != binding.initial_published_length ||
          state->generation != binding.initial_generation ||
          state->active_slot != binding.anchor_slot ||
          state->transition_token != 0U ||
          state->accounting.active_submissions != 0U ||
          state->accounting.completion_references != 0U ||
          binding.initial_published_length != expected_initial_position ||
          final_position > state->capacity_tokens ||
          binding.initial_generation > UINT64_MAX - successful_generations) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_LINEAR_ATTENTION_LENGTH_MISMATCH,
            "captured linear state metadata no longer matches its initial "
            "snapshot");
      }
    }
    if (!graph_span_finish_paged_states(
            span, final_position, successful_generations, false, error_sink)) {
      span->context->poisoned.store(true);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "paged graph state finalization could not be completed safely");
    }
    /* Every check above is complete before the first state is mutated. */
    for (const GraphKvStateBinding &binding : span->kv_states) {
      KvState *const state = binding.state;
      state->last_published_start = expected_initial_position;
      state->last_published_end = final_position;
      state->published_length = final_position;
      state->generation = binding.initial_generation + successful_generations;
      // A whole-graph generation does not preserve the eager two-slot
      // rollback snapshot. Never expose a scratch/one-ahead slot as a valid
      // previous public state through the ordinary rewind API.
      state->last_published_generation = 0U;
      state->retained_start = state->sliding_window == 0U ? 0U
                              : final_position > state->sliding_window
                                  ? final_position - state->sliding_window
                                  : 0U;
    }
    for (const GraphLinearStateBinding &binding : span->linear_states) {
      LinearAttentionState *const state = binding.state;
      state->last_published_start = expected_initial_position;
      state->last_published_end = final_position;
      state->published_length = final_position;
      state->generation = binding.initial_generation + successful_generations;
      state->active_slot = binding.anchor_slot ^
                           static_cast<uint32_t>(successful_generations & 1U);
      // A whole-graph generation does not preserve the eager two-slot
      // rollback snapshot. Never expose a scratch/one-ahead slot as a valid
      // previous public state through the ordinary rewind API.
      state->last_published_generation = 0U;
      state->checkpoint_conv_state = {nullptr, nullptr};
      state->checkpoint_recurrent_state = {nullptr, nullptr};
      state->checkpoint_rows = 0U;
      state->checkpoint_start = 0U;
      state->checkpoint_end = 0U;
      state->checkpoint_generation = 0U;
      state->checkpoint_armed = false;
    }
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception publishing graph state metadata");
  }
}

extern "C" sllm_status_t sllm_graph_span_begin_capture(
    const sllm_context_t *const raw_context,
    const sllm_queue_t *const raw_queue,
    const sllm_tensor_binding_t *const control_binding,
    sllm_graph_span_t **const raw_span,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (raw_span != nullptr) {
      *raw_span = nullptr;
    }
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_context == nullptr || raw_queue == nullptr || raw_span == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "whole graph capture context, queue, or output is null");
    }

    Context *context = nullptr;
    Queue *queue = nullptr;
    std::unique_ptr<GraphSpan> candidate;
    uintptr_t token = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      context = lookup<Context>(raw_context, HandleKind::Context);
      queue = lookup<Queue>(raw_queue, HandleKind::Queue);
      if (context == nullptr || queue == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "whole graph capture context or queue handle is stale");
      }
      if (context != queue->context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
            "whole graph capture context and queue differ");
      }
      const char *const arch_name = context->device_properties.gcnArchName;
      if (!matches_runtime_gcn_arch(arch_name, "gfx1030") &&
          !matches_runtime_gcn_arch(arch_name, "gfx1201")) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
            "whole graph capture requires exact gfx1030 or gfx1201");
      }
      if (context->poisoned.load() || context->release_active ||
          queue->release_active ||
          queue->graph_capture_active.load(std::memory_order_acquire) ||
          queue->completion_mode != SLLM_QUEUE_COMPLETION_MODE_DEFERRED) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "whole graph capture queue is not an idle deferred queue");
      }
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (queue->accounting.active_submissions != 0U ||
          queue->accounting.completion_references != 0U) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "whole graph capture queue has active submissions or references");
      }
      if (!preallocate_pinned_d2h_slots(queue)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "whole graph capture could not preallocate pinned result staging");
      }
      candidate = std::make_unique<GraphSpan>(context, queue, raw_queue);
      candidate->whole_capture = true;
      candidate->plans.reserve(1U);
      candidate->capture_markers.reserve(64U);
      if (!graph_span_validate_control_binding(
              control_binding, context, &candidate->control_buffer,
              &candidate->control_byte_offset, error_sink)) {
        return SLLM_STATUS_INVALID_ARGUMENT;
      }
      if (candidate->control_buffer != nullptr) {
        if (!sllm_public_runtime::AccountingState::
                reserve_two_buffer_prepared_plan(
                    context->accounting, candidate->control_buffer->accounting,
                    candidate->control_buffer->accounting)) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_INTERNAL_ERROR,
              "whole graph capture control-buffer pin accounting exhausted");
        }
        candidate->control_buffer_pinned = true;
      }
      ++queue->graph_pins;
      queue->graph_capture_active.store(true, std::memory_order_release);
      candidate->capture_owner = true;
      graph_capture_queue = queue;
      graph_capture_span = candidate.get();
      token = register_handle(candidate.get(), HandleKind::GraphSpan);
      if (token == 0U) {
        graph_capture_span = nullptr;
        graph_capture_queue = nullptr;
        queue->graph_capture_active.store(false, std::memory_order_release);
        candidate->capture_owner = false;
        if (candidate->control_buffer_pinned) {
          (void)sllm_public_runtime::AccountingState::
              release_two_buffer_prepared_plan(
                  context->accounting, candidate->control_buffer->accounting,
                  candidate->control_buffer->accounting);
          candidate->control_buffer_pinned = false;
        }
        if (queue->graph_pins != 0U) {
          --queue->graph_pins;
        }
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "whole graph capture handle allocation failed");
      }
    }

    // Paged GQA6 decode may request its P128 workspace while the queue is
    // already inside HIP capture. Reserve the maximum five-row workspace
    // before BeginCapture; subsequent provider selection only reuses this
    // context-owned pointer and never calls hipMalloc during capture. This
    // runs outside the context accounting lock held during handle setup.
    const uint64_t max_paged_gqa6_workspace =
        static_cast<uint64_t>(kDecodeGqa6SplitP128WorkspaceBytes) * 5U;
    void *capture_workspace = nullptr;
    uint64_t capture_workspace_bytes = 0U;
    hipError_t workspace_status = hipSuccess;
    {
      // Keep the registry lease while the plan helper temporarily acquires
      // the context accounting lock. This blocks state/queue release without
      // inverting the plan-mutex -> accounting-mutex order.
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      GraphSpan *const registered = lookup<GraphSpan>(
          reinterpret_cast<const sllm_graph_span_t *>(token),
          HandleKind::GraphSpan);
      if (registered != candidate.get() || context->poisoned.load() ||
          context->release_active || queue->release_active ||
          !queue->graph_capture_active.load(std::memory_order_acquire) ||
          queue->completion_mode != SLLM_QUEUE_COMPLETION_MODE_DEFERRED) {
        workspace_status = hipErrorNotReady;
      } else {
        workspace_status = prepare_causal_workspace_plan_for_bytes(
            context, queue, max_paged_gqa6_workspace, &capture_workspace,
            &capture_workspace_bytes);
      }
    }
    if (workspace_status != hipSuccess) {
      {
        std::lock_guard<std::mutex> registry_lock(registry_mutex);
        unregister_handle_token(token);
      }
      graph_span_unpin(candidate.get());
      return hip_failure(error_sink, workspace_status,
                         "preallocate paged GQA6 graph workspace");
    }
    (void)capture_workspace;
    (void)capture_workspace_bytes;
    candidate->causal_workspace_pins = 1U;

    const sllm_status_t device_status =
        select_context_device(context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      {
        std::lock_guard<std::mutex> registry_lock(registry_mutex);
        unregister_handle_token(token);
      }
      graph_span_unpin(candidate.get());
      return device_status;
    }
    const hipError_t begin_status =
        hipStreamBeginCapture(queue->stream, hipStreamCaptureModeThreadLocal);
    if (begin_status != hipSuccess) {
      {
        std::lock_guard<std::mutex> registry_lock(registry_mutex);
        unregister_handle_token(token);
      }
      graph_span_unpin(candidate.get());
      return hip_failure(error_sink, begin_status,
                         "hipStreamBeginCapture whole graph");
    }
    queue->whole_graph_result_readback_required.store(
        true, std::memory_order_release);
    *raw_span = reinterpret_cast<sllm_graph_span_t *>(token);
    (void)candidate.release();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception beginning whole graph capture");
  }
}

extern "C" sllm_status_t
sllm_graph_span_end_capture(sllm_graph_span_t *const raw_span,
                            sllm_graph_span_capture_info_t *const info,
                            sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_span == nullptr || !graph_span_capture_info_valid(info)) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "whole graph capture span or info is invalid");
    }
    GraphSpan *span = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      span = lookup<GraphSpan>(raw_span, HandleKind::GraphSpan);
      if (span == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "whole graph capture span handle is stale");
      }
      std::lock_guard<std::mutex> accounting_lock(
          span->context->accounting_mutex);
      if (!span->whole_capture || span->capture_failed ||
          !span->capture_owner || span->release_active || span->in_flight ||
          span->whole_replay_in_flight != 0U || span->graph != nullptr ||
          span->graph_exec != nullptr || span->queue == nullptr ||
          span->queue->graph_capture_active.load(std::memory_order_acquire) ==
              false) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "whole graph capture span is not actively capturing");
      }
    }

    hipGraph_t graph = nullptr;
    const hipError_t end_status =
        hipStreamEndCapture(span->queue->stream, &graph);
    graph_span_finish_capture(span);
    if (end_status != hipSuccess || graph == nullptr) {
      if (graph != nullptr) {
        span->graph = graph;
        hipError_t destroy_status = hipSuccess;
        if (!graph_span_destroy_graph_checked(span, &destroy_status)) {
          span->capture_failed = true;
          graph_span_quarantine_markers(span);
          retain_poisoned(span, span->context);
          return hip_failure(error_sink, destroy_status,
                             "whole graph capture failure graph destruction");
        }
      }
      span->capture_failed = true;
      graph_span_quarantine_markers(span);
      return end_status != hipSuccess
                 ? hip_failure(error_sink, end_status,
                               "hipStreamEndCapture whole graph")
                 : sllm_public_runtime::write_error(
                       error_sink, SLLM_STATUS_UNSUPPORTED,
                       "whole graph capture returned a null graph");
    }
    span->graph = graph;
    if (!graph_span_collect_whole_capture_nodes(span, info, error_sink)) {
      span->capture_failed = true;
      hipError_t destroy_status = hipSuccess;
      if (!graph_span_destroy_graph_checked(span, &destroy_status)) {
        graph_span_quarantine_markers(span);
        retain_poisoned(span, span->context);
        return hip_failure(error_sink, destroy_status,
                           "whole graph capture node validation destruction");
      }
      graph_span_quarantine_markers(span);
      return SLLM_STATUS_UNSUPPORTED;
    }
    hipGraphNode_t error_node = nullptr;
    std::array<char, 4096> log{};
    const hipError_t instantiate_status = hipGraphInstantiate(
        &span->graph_exec, span->graph, &error_node, log.data(), log.size());
    if (instantiate_status != hipSuccess || span->graph_exec == nullptr) {
      span->capture_failed = true;
      hipError_t destroy_status = hipSuccess;
      if (!graph_span_destroy_graph_checked(span, &destroy_status)) {
        graph_span_quarantine_markers(span);
        retain_poisoned(span, span->context);
        return hip_failure(error_sink, destroy_status,
                           "whole graph capture instantiate destruction");
      }
      graph_span_quarantine_markers(span);
      return hip_failure(error_sink, instantiate_status,
                         "hipGraphInstantiate whole graph capture");
    }
    /* Do not fence or read a token here. Captured operation markers stay in the
     * GraphSpan until the replay owner finalizes them after a queue fence. */
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception ending whole graph capture");
  }
}

extern "C" sllm_status_t
sllm_graph_span_abort_capture(sllm_graph_span_t **const raw_span,
                              sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_span == nullptr || *raw_span == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "whole graph capture span is null");
    }
    GraphSpan *span = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      span = lookup<GraphSpan>(*raw_span, HandleKind::GraphSpan);
      if (span == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "whole graph capture span handle is stale");
      }
      std::lock_guard<std::mutex> accounting_lock(
          span->context->accounting_mutex);
      if (!span->whole_capture || span->release_active || span->in_flight ||
          span->whole_replay_in_flight != 0U) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "whole graph capture span is not abortable");
      }
      span->release_active = true;
    }

    hipError_t capture_status = hipSuccess;
    if (span->capture_owner) {
      hipGraph_t graph = nullptr;
      capture_status = hipStreamEndCapture(span->queue->stream, &graph);
      if (graph != nullptr) {
        span->graph = graph;
      }
      graph_span_finish_capture(span);
    }
    graph_span_quarantine_markers(span);
    hipError_t destroy_status = hipSuccess;
    const bool destroyed =
        graph_span_destroy_graph_checked(span, &destroy_status);
    if (!destroyed) {
      {
        std::lock_guard<std::mutex> accounting_lock(
            span->context->accounting_mutex);
        span->release_active = false;
      }
      retain_poisoned(span, span->context);
      return hip_failure(error_sink, destroy_status,
                         "whole graph capture abort destruction");
    }
    graph_span_unregister_and_delete(span, *raw_span, true);
    *raw_span = nullptr;
    if (capture_status != hipSuccess) {
      return hip_failure(error_sink, capture_status,
                         "hipStreamEndCapture whole graph abort");
    }
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception aborting whole graph capture");
  }
}

extern "C" sllm_status_t
sllm_graph_span_create(const sllm_queue_t *const raw_queue,
                       const void *const *const prepared_plan_handles,
                       const uint64_t plan_count,
                       sllm_graph_span_t **const raw_span,
                       sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (raw_span != nullptr) {
      *raw_span = nullptr;
    }
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_queue == nullptr || prepared_plan_handles == nullptr ||
        raw_span == nullptr || plan_count == 0U ||
        plan_count > kGraphSpanMaxPlans) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "graph span queue, plans, count, or output is invalid");
    }
    Queue *queue = nullptr;
    std::unique_ptr<GraphSpan> candidate;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      queue = lookup<Queue>(raw_queue, HandleKind::Queue);
      if (queue == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "graph span queue handle is stale or has the wrong kind");
      }
      const char *const arch_name =
          queue->context->device_properties.gcnArchName;
      if (!matches_runtime_gcn_arch(arch_name, "gfx1030") &&
          !matches_runtime_gcn_arch(arch_name, "gfx1201")) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
            "graph spans require exact gfx1030 or gfx1201");
      }
      if (queue->context->poisoned.load() || queue->context->release_active ||
          queue->release_active ||
          queue->graph_capture_active.load(std::memory_order_acquire) ||
          queue->completion_mode != SLLM_QUEUE_COMPLETION_MODE_DEFERRED) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "graph span queue is not an idle deferred request queue");
      }
      std::lock_guard<std::mutex> accounting_lock(
          queue->context->accounting_mutex);
      if (queue->accounting.active_submissions != 0U ||
          queue->accounting.completion_references != 0U) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "graph span queue has active submissions or completion references");
      }
      try {
        candidate =
            std::make_unique<GraphSpan>(queue->context, queue, raw_queue);
        candidate->plans.reserve(static_cast<std::size_t>(plan_count));
        candidate->capture_markers.reserve(
            static_cast<std::size_t>(plan_count));
        for (uint64_t index = 0U; index != plan_count; ++index) {
          const void *const raw = prepared_plan_handles[index];
          for (const GraphSpanPlanRef &prior : candidate->plans) {
            if (prior.raw == raw) {
              return sllm_public_runtime::write_error(
                  error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                  "graph span contains a duplicate prepared plan");
            }
          }
          GraphSpanPlanRef ref{};
          const sllm_status_t resolve_status =
              graph_span_resolve_plan(raw, queue, &ref, error_sink);
          if (resolve_status != SLLM_STATUS_OK) {
            return resolve_status;
          }
          auto *const plan_context =
              ref.node == nullptr
                  ? nullptr
                  : (ref.kind == GraphSpanPlanKind::RmsNorm ||
                             ref.kind == GraphSpanPlanKind::ResidualRmsNorm
                         ? static_cast<RmsNormPlan *>(ref.node)->context
                         : static_cast<ElementwisePlan *>(ref.node)->context);
          if (plan_context != queue->context || ref.node == nullptr ||
              ref.node->graph_pins != 0U ||
              ((ref.kind == GraphSpanPlanKind::RmsNorm ||
                ref.kind == GraphSpanPlanKind::ResidualRmsNorm)
                   ? static_cast<RmsNormPlan *>(ref.node)->release_active ||
                         static_cast<RmsNormPlan *>(ref.node)->in_flight
                   : static_cast<ElementwisePlan *>(ref.node)->release_active ||
                         static_cast<ElementwisePlan *>(ref.node)->in_flight)) {
            return sllm_public_runtime::write_error(
                error_sink, SLLM_STATUS_PUBLIC_BUSY,
                "graph span plan is releasing, in flight, or already pinned");
          }
          candidate->plans.push_back(ref);
        }
        for (const GraphSpanPlanRef &ref : candidate->plans) {
          ++ref.node->graph_pins;
        }
        ++queue->graph_pins;
        queue->graph_capture_active.store(true, std::memory_order_release);
        candidate->capture_owner = true;
        graph_capture_queue = queue;
      } catch (...) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "graph span plan vector allocation failed");
      }
    }

    const sllm_status_t device_status =
        select_context_device(queue->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      graph_span_unpin(candidate.get());
      return device_status;
    }
    bool capture_started = false;
    bool capture_ended = false;
    hipGraph_t graph = nullptr;
    const hipError_t begin_status =
        hipStreamBeginCapture(queue->stream, hipStreamCaptureModeThreadLocal);
    if (begin_status != hipSuccess) {
      graph_span_unpin(candidate.get());
      return hip_failure(error_sink, begin_status, "hipStreamBeginCapture");
    }
    capture_started = true;
    sllm_status_t capture_status = SLLM_STATUS_OK;
    for (const GraphSpanPlanRef &ref : candidate->plans) {
      capture_status =
          graph_span_capture_plan(candidate.get(), ref, error_sink);
      if (capture_status != SLLM_STATUS_OK) {
        break;
      }
    }
    const hipError_t end_status = hipStreamEndCapture(queue->stream, &graph);
    capture_ended = true;
    if (end_status != hipSuccess || capture_status != SLLM_STATUS_OK ||
        graph == nullptr) {
      if (graph != nullptr) {
        candidate->graph = graph;
        hipError_t destroy_status = hipSuccess;
        if (!graph_span_destroy_graph_checked(candidate.get(),
                                              &destroy_status)) {
          graph_span_quarantine_markers(candidate.get());
          retain_poisoned(candidate, candidate->context);
          return hip_failure(error_sink, destroy_status,
                             "graph span destruction after capture failure");
        }
      }
      graph_span_quarantine_markers(candidate.get());
      graph_span_unpin(candidate.get());
      return end_status != hipSuccess
                 ? hip_failure(error_sink, end_status, "hipStreamEndCapture")
                 : capture_status;
    }
    candidate->graph = graph;
    std::size_t total_node_count = 0U;
    hipError_t graph_status =
        hipGraphGetNodes(candidate->graph, nullptr, &total_node_count);
    if (graph_status != hipSuccess || total_node_count == 0U) {
      hipError_t destroy_status = hipSuccess;
      if (!graph_span_abort_after_capture(candidate, &destroy_status)) {
        return hip_failure(error_sink, destroy_status,
                           "graph span destruction after node query failure");
      }
      return graph_status != hipSuccess
                 ? hip_failure(error_sink, graph_status, "hipGraphGetNodes")
                 : sllm_public_runtime::write_error(
                       error_sink, SLLM_STATUS_UNSUPPORTED,
                       "graph span capture produced no native nodes");
    }
    std::unique_ptr<hipGraphNode_t[]> graph_nodes;
    try {
      graph_nodes = std::make_unique<hipGraphNode_t[]>(total_node_count);
    } catch (...) {
      hipError_t destroy_status = hipSuccess;
      if (!graph_span_abort_after_capture(candidate, &destroy_status)) {
        return hip_failure(
            error_sink, destroy_status,
            "graph span destruction after node-list allocation failure");
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "graph span native node-list allocation failed");
    }
    std::size_t listed_node_count = total_node_count;
    graph_status = hipGraphGetNodes(candidate->graph, graph_nodes.get(),
                                    &listed_node_count);
    if (graph_status != hipSuccess || listed_node_count != total_node_count) {
      hipError_t destroy_status = hipSuccess;
      if (!graph_span_abort_after_capture(candidate, &destroy_status)) {
        return hip_failure(error_sink, destroy_status,
                           "graph span destruction after node-list query");
      }
      return graph_status != hipSuccess
                 ? hip_failure(error_sink, graph_status, "hipGraphGetNodes")
                 : sllm_public_runtime::write_error(
                       error_sink, SLLM_STATUS_UNSUPPORTED,
                       "graph span native node-list size changed");
    }
    uint64_t kernel_node_count = 0U;
    for (std::size_t index = 0U; index != listed_node_count; ++index) {
      hipGraphNodeType node_type = hipGraphNodeTypeCount;
      graph_status = hipGraphNodeGetType(graph_nodes[index], &node_type);
      if (graph_status != hipSuccess || node_type != hipGraphNodeTypeKernel) {
        hipError_t destroy_status = hipSuccess;
        if (!graph_span_abort_after_capture(candidate, &destroy_status)) {
          return hip_failure(
              error_sink, destroy_status,
              "graph span destruction after node-type query failure");
        }
        return graph_status != hipSuccess
                   ? hip_failure(error_sink, graph_status,
                                 "hipGraphNodeGetType")
                   : sllm_public_runtime::write_error(
                         error_sink, SLLM_STATUS_UNSUPPORTED,
                         "graph span contains a non-kernel native node");
      }
      ++kernel_node_count;
    }
    if (kernel_node_count == 0U) {
      hipError_t destroy_status = hipSuccess;
      if (!graph_span_abort_after_capture(candidate, &destroy_status)) {
        return hip_failure(error_sink, destroy_status,
                           "graph span destruction after empty kernel list");
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_UNSUPPORTED,
          "graph span capture produced no kernel nodes");
    }
    candidate->actual_node_count = kernel_node_count;
    candidate->logical_plan_count = plan_count;
    hipGraphNode_t error_node = nullptr;
    std::array<char, 4096> log{};
    graph_status = hipGraphInstantiate(&candidate->graph_exec, candidate->graph,
                                       &error_node, log.data(), log.size());
    if (graph_status != hipSuccess || candidate->graph_exec == nullptr) {
      hipError_t destroy_status = hipSuccess;
      if (!graph_span_abort_after_capture(candidate, &destroy_status)) {
        return hip_failure(
            error_sink, destroy_status,
            "graph span destruction after graph instantiation failure");
      }
      return hip_failure(error_sink, graph_status,
                         "hipGraphInstantiate graph span");
    }
    if (!graph_span_finalize_capture_markers(candidate.get(), error_sink)) {
      hipError_t destroy_status = hipSuccess;
      if (!graph_span_abort_after_capture(candidate, &destroy_status)) {
        return hip_failure(
            error_sink, destroy_status,
            "graph span destruction after marker finalization failure");
      }
      return SLLM_STATUS_INTERNAL_ERROR;
    }
    graph_span_finish_capture(candidate.get());
    uintptr_t token = 0U;
    try {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      token = register_handle(candidate.get(), HandleKind::GraphSpan);
    } catch (...) {
      hipError_t destroy_status = hipSuccess;
      if (!graph_span_destroy_graph_checked(candidate.get(), &destroy_status)) {
        retain_poisoned(candidate, candidate->context);
        return hip_failure(error_sink, destroy_status,
                           "graph span destruction after registry failure");
      }
      graph_span_unpin(candidate.get());
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "graph span registry allocation failed");
    }
    if (token == 0U) {
      hipError_t destroy_status = hipSuccess;
      if (!graph_span_destroy_graph_checked(candidate.get(), &destroy_status)) {
        retain_poisoned(candidate, candidate->context);
        return hip_failure(error_sink, destroy_status,
                           "graph span destruction after handle allocation");
      }
      graph_span_unpin(candidate.get());
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "graph span handle token allocation failed");
    }
    *raw_span = reinterpret_cast<sllm_graph_span_t *>(token);
    (void)candidate.release();
    (void)capture_started;
    (void)capture_ended;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in graph span create");
  }
}

extern "C" sllm_status_t
sllm_graph_span_execute(const sllm_graph_span_t *const raw_span,
                        sllm_completion_t **const completion_output,
                        sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (completion_output != nullptr) {
      *completion_output = nullptr;
    }
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_span == nullptr || completion_output == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "graph span or completion output is null");
    }
    GraphSpan *span = nullptr;
    uint64_t submission_serial = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      span = lookup<GraphSpan>(raw_span, HandleKind::GraphSpan);
      if (span == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "graph span handle is stale or has the wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(
          span->context->accounting_mutex);
      if (span->release_active || ((!span->whole_capture) && span->in_flight) ||
          span->queue == nullptr || span->queue->release_active ||
          span->queue->completion_mode != SLLM_QUEUE_COMPLETION_MODE_DEFERRED ||
          span->queue->graph_capture_active.load(std::memory_order_acquire) ||
          span->context->poisoned.load() || span->graph_exec == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "graph span is releasing, in flight, poisoned, or incomplete");
      }
      const bool reserved =
          sllm_public_runtime::AccountingState::reserve_queue_fence(
              span->context->accounting, span->queue->accounting);
      if (!reserved) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "graph span submission accounting is exhausted");
      }
      if (!next_queue_submission_serial(span->queue, &submission_serial)) {
        (void)sllm_public_runtime::AccountingState::rollback_queue_fence(
            span->context->accounting, span->queue->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "graph span submission serial is exhausted");
      }
      if (span->whole_capture) {
        if (span->whole_replay_in_flight == UINT64_MAX) {
          (void)sllm_public_runtime::AccountingState::rollback_queue_fence(
              span->context->accounting, span->queue->accounting);
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_INTERNAL_ERROR,
              "whole graph replay count is exhausted");
        }
        ++span->whole_replay_in_flight;
      } else {
        span->in_flight = true;
      }
    }
    std::unique_ptr<Completion> candidate;
    try {
      candidate =
          std::make_unique<Completion>(span->context, span->queue, nullptr, 0U,
                                       false, std::vector<uint8_t>{});
      candidate->graph_span = true;
      candidate->graph_span_owner = span;
      candidate->submission_serial = submission_serial;
    } catch (...) {
      std::lock_guard<std::mutex> accounting_lock(
          span->context->accounting_mutex);
      (void)sllm_public_runtime::AccountingState::rollback_queue_fence(
          span->context->accounting, span->queue->accounting);
      if (span->whole_capture) {
        if (span->whole_replay_in_flight != 0U) {
          --span->whole_replay_in_flight;
        } else {
          span->context->poisoned.store(true);
        }
      } else {
        span->in_flight = false;
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "graph span completion allocation failed");
    }
    uintptr_t token = 0U;
    try {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      token = register_handle(candidate.get(), HandleKind::Completion);
    } catch (...) {
      std::lock_guard<std::mutex> accounting_lock(
          span->context->accounting_mutex);
      (void)rollback_graph_span_completion(candidate.get());
      if (!span->whole_capture) {
        span->in_flight = false;
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "graph span completion registry allocation failed");
    }
    if (token == 0U) {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      unregister_handle(candidate.get());
      std::lock_guard<std::mutex> accounting_lock(
          span->context->accounting_mutex);
      (void)rollback_graph_span_completion(candidate.get());
      if (!span->whole_capture) {
        span->in_flight = false;
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "graph span completion handle allocation failed");
    }
    const sllm_status_t device_status =
        select_context_device(span->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      {
        std::lock_guard<std::mutex> registry_lock(registry_mutex);
        unregister_handle_token(token);
      }
      bool rolled_back = false;
      {
        std::lock_guard<std::mutex> accounting_lock(
            span->context->accounting_mutex);
        rolled_back = rollback_graph_span_completion(candidate.get());
      }
      if (!rolled_back) {
        candidate->orphaned = true;
        retain_poisoned(candidate, span->context);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "graph span rollback failed after device selection failure");
      }
      return device_status;
    }
    const hipError_t launch_status =
        hipGraphLaunch(span->graph_exec, span->queue->stream);
    if (launch_status != hipSuccess) {
      return cleanup_failed_submission(candidate, token, launch_status,
                                       "hipGraphLaunch graph span", span->queue,
                                       error_sink);
    }
    *completion_output = reinterpret_cast<sllm_completion_t *>(token);
    (void)candidate.release();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in graph span execute");
  }
}

extern "C" sllm_status_t
sllm_graph_span_node_count(const sllm_graph_span_t *const raw_span,
                           uint64_t *const node_count,
                           sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (node_count == nullptr || raw_span == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "graph span or node-count output is null");
    }
    *node_count = 0U;
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    GraphSpan *const span = lookup<GraphSpan>(raw_span, HandleKind::GraphSpan);
    if (span == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "graph span handle is stale or has the wrong kind");
    }
    *node_count = span->actual_node_count;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in graph span node-count query");
  }
}

extern "C" sllm_status_t
sllm_graph_span_release(sllm_graph_span_t **const raw_span,
                        sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_span == nullptr || *raw_span == nullptr) {
      return sllm_public_runtime::write_error(error_sink,
                                              SLLM_STATUS_INVALID_ARGUMENT,
                                              "graph span handle is null");
    }
    GraphSpan *span = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      span = lookup<GraphSpan>(*raw_span, HandleKind::GraphSpan);
      if (span == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "graph span handle is stale or has the wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(
          span->context->accounting_mutex);
      if (span->release_active || span->in_flight ||
          span->whole_replay_in_flight != 0U || span->graph_exec == nullptr ||
          span->queue == nullptr ||
          span->queue->graph_capture_active.load(std::memory_order_acquire)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "graph span is releasing, in flight, incomplete, or its queue is "
            "capturing another graph");
      }
      span->release_active = true;
    }
    const sllm_status_t device_status =
        select_context_device(span->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      std::lock_guard<std::mutex> accounting_lock(
          span->context->accounting_mutex);
      span->release_active = false;
      return device_status;
    }
    if (span->whole_capture && !span->capture_markers.empty() &&
        !graph_span_finalize_capture_markers(span, error_sink)) {
      std::lock_guard<std::mutex> accounting_lock(
          span->context->accounting_mutex);
      span->release_active = false;
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "whole graph capture marker finalization failed at release");
    }
    bool paged_cleanup_failed = false;
    {
      std::lock_guard<std::mutex> accounting_lock(
          span->context->accounting_mutex);
      bool paged_staged = false;
      for (const GraphKvStateBinding &binding : span->kv_states) {
        if (binding.state != nullptr && binding.state->paged &&
            binding.state->paged_host_state != nullptr &&
            binding.state->paged_host_state->graph_staged()) {
          paged_staged = true;
          break;
        }
      }
      if (paged_staged && !graph_span_abort_paged_states(span, error_sink)) {
        span->release_active = false;
        paged_cleanup_failed = true;
      }
    }
    if (paged_cleanup_failed) {
      {
        std::lock_guard<std::mutex> registry_lock(registry_mutex);
        unregister_handle(*raw_span);
        *raw_span = nullptr;
      }
      retain_poisoned(span, span->context);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "paged graph staged-state cleanup could not be proven safe");
    }
    hipError_t destroy_status = hipSuccess;
    if (!graph_span_destroy_graph_checked(span, &destroy_status)) {
      {
        std::lock_guard<std::mutex> registry_lock(registry_mutex);
        unregister_handle(*raw_span);
        *raw_span = nullptr;
      }
      retain_poisoned(span, span->context);
      return hip_failure(error_sink, destroy_status,
                         "graph span HIP graph destruction");
    }
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      unregister_handle(*raw_span);
      *raw_span = nullptr;
    }
    graph_span_unpin(span);
    delete span;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in graph span release");
  }
}

#else

namespace {
bool release_graph_span_active(Completion *const) noexcept { return false; }
bool rollback_graph_span_completion(Completion *const) noexcept {
  return false;
}
bool release_graph_span_completion(Completion *const) noexcept { return false; }
void invalidate_whole_graph_capture() noexcept {}
[[maybe_unused]] bool whole_graph_capture_active() noexcept { return false; }
[[maybe_unused]] void *whole_graph_capture_device_control() noexcept {
  return nullptr;
}
[[maybe_unused]] bool
whole_graph_pin_buffer(Buffer *const buffer,
                       sllm_error_sink_t *const error_sink) noexcept {
  (void)buffer;
  (void)sllm_public_runtime::write_error(
      error_sink, SLLM_STATUS_HIP_UNAVAILABLE,
      "whole graph buffer pin is unavailable in the host runtime");
  return false;
}
} // namespace

extern "C" sllm_status_t
sllm_graph_span_capture_marker(sllm_graph_span_t *const span,
                               sllm_completion_t **const marker,
                               sllm_error_sink_t *const error_sink) noexcept {
  const sllm_status_t sink_status =
      sllm_public_runtime::validate_error_sink(error_sink);
  if (sink_status != SLLM_STATUS_OK) {
    return sink_status;
  }
  if (span == nullptr || marker == nullptr || *marker == nullptr) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INVALID_ARGUMENT,
        "whole graph capture span or marker is null");
  }
  return sllm_public_runtime::write_error(
      error_sink, SLLM_STATUS_HIP_UNAVAILABLE,
      "whole graph capture is unavailable in the host runtime");
}

extern "C" sllm_status_t sllm_graph_span_decode_command(
    const sllm_graph_span_t *const span,
    const sllm_graph_span_decode_command_desc_t *const command,
    sllm_error_sink_t *const error_sink) noexcept {
  const sllm_status_t sink_status =
      sllm_public_runtime::validate_error_sink(error_sink);
  if (sink_status != SLLM_STATUS_OK) {
    return sink_status;
  }
  if (span == nullptr || command == nullptr) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INVALID_ARGUMENT,
        "whole graph decode command span or descriptor is null");
  }
  return sllm_public_runtime::write_error(
      error_sink, SLLM_STATUS_HIP_UNAVAILABLE,
      "whole graph decode commands are unavailable in the host runtime");
}

extern "C" sllm_status_t sllm_graph_span_bind_linear_state(
    sllm_graph_span_t *const span,
    const sllm_linear_attention_state_t *const state,
    const sllm_tensor_binding_t *const checkpoint_conv,
    const sllm_tensor_binding_t *const checkpoint_recurrent,
    const uint32_t token_count, const uint32_t checkpoint_rows,
    sllm_error_sink_t *const error_sink) noexcept {
  const sllm_status_t sink_status =
      sllm_public_runtime::validate_error_sink(error_sink);
  if (sink_status != SLLM_STATUS_OK) {
    return sink_status;
  }
  if (span == nullptr || state == nullptr || token_count == 0U ||
      checkpoint_rows >= token_count ||
      ((checkpoint_conv == nullptr) != (checkpoint_recurrent == nullptr)) ||
      (checkpoint_rows != 0U &&
       (checkpoint_conv == nullptr || checkpoint_recurrent == nullptr))) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INVALID_ARGUMENT,
        "linear graph binding state, checkpoints, or dimensions are invalid");
  }
  return sllm_public_runtime::write_error(
      error_sink, SLLM_STATUS_HIP_UNAVAILABLE,
      "linear graph state binding is unavailable in the host runtime");
}

extern "C" sllm_status_t sllm_graph_span_select_linear_state(
    sllm_graph_span_t *const span,
    const sllm_linear_attention_state_t *const state,
    const uint32_t token_count, sllm_error_sink_t *const error_sink) noexcept {
  const sllm_status_t sink_status =
      sllm_public_runtime::validate_error_sink(error_sink);
  if (sink_status != SLLM_STATUS_OK) {
    return sink_status;
  }
  if (span == nullptr || state == nullptr || token_count == 0U) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INVALID_ARGUMENT,
        "linear graph state selection argument is null or empty");
  }
  return sllm_public_runtime::write_error(
      error_sink, SLLM_STATUS_HIP_UNAVAILABLE,
      "linear graph state selection is unavailable in the host runtime");
}

extern "C" sllm_status_t
sllm_graph_span_prepare_paged_kv(sllm_graph_span_t *const span,
                                 const uint64_t conservative_end,
                                 sllm_error_sink_t *const error_sink) noexcept {
  const sllm_status_t sink_status =
      sllm_public_runtime::validate_error_sink(error_sink);
  if (sink_status != SLLM_STATUS_OK)
    return sink_status;
  if (span == nullptr || conservative_end == 0U)
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INVALID_ARGUMENT,
        "graph paged KV preparation requires a span and end position");
  return sllm_public_runtime::write_error(
      error_sink, SLLM_STATUS_HIP_UNAVAILABLE,
      "graph paged KV preparation is unavailable in the host runtime");
}

extern "C" sllm_status_t sllm_graph_span_publish_state_metadata(
    sllm_graph_span_t *const span, const uint64_t expected_initial_position,
    const uint64_t final_position, const uint64_t successful_generations,
    sllm_error_sink_t *const error_sink) noexcept {
  const sllm_status_t sink_status =
      sllm_public_runtime::validate_error_sink(error_sink);
  if (sink_status != SLLM_STATUS_OK) {
    return sink_status;
  }
  if (span == nullptr || final_position < expected_initial_position) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INVALID_ARGUMENT,
        "graph state metadata publication range is invalid");
  }
  (void)successful_generations;
  return sllm_public_runtime::write_error(
      error_sink, SLLM_STATUS_HIP_UNAVAILABLE,
      "graph state metadata publication is unavailable in the host runtime");
}

extern "C" sllm_status_t sllm_graph_span_begin_capture(
    const sllm_context_t *const context, const sllm_queue_t *const queue,
    const sllm_tensor_binding_t *const control_binding,
    sllm_graph_span_t **const span,
    sllm_error_sink_t *const error_sink) noexcept {
  if (span != nullptr) {
    *span = nullptr;
  }
  const sllm_status_t sink_status =
      sllm_public_runtime::validate_error_sink(error_sink);
  if (sink_status != SLLM_STATUS_OK) {
    return sink_status;
  }
  if (context == nullptr || queue == nullptr || span == nullptr) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INVALID_ARGUMENT,
        "whole graph capture context, queue, or output is null");
  }
  (void)control_binding;
  return sllm_public_runtime::write_error(
      error_sink, SLLM_STATUS_HIP_UNAVAILABLE,
      "whole graph capture is unavailable in the host runtime");
}

extern "C" sllm_status_t
sllm_graph_span_end_capture(sllm_graph_span_t *const span,
                            sllm_graph_span_capture_info_t *const info,
                            sllm_error_sink_t *const error_sink) noexcept {
  const sllm_status_t sink_status =
      sllm_public_runtime::validate_error_sink(error_sink);
  if (sink_status != SLLM_STATUS_OK) {
    return sink_status;
  }
  if (span == nullptr || info == nullptr) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INVALID_ARGUMENT,
        "whole graph capture span or info is null");
  }
  return sllm_public_runtime::write_error(
      error_sink, SLLM_STATUS_HIP_UNAVAILABLE,
      "whole graph capture is unavailable in the host runtime");
}

extern "C" sllm_status_t
sllm_graph_span_abort_capture(sllm_graph_span_t **const span,
                              sllm_error_sink_t *const error_sink) noexcept {
  const sllm_status_t sink_status =
      sllm_public_runtime::validate_error_sink(error_sink);
  if (sink_status != SLLM_STATUS_OK) {
    return sink_status;
  }
  if (span == nullptr || *span == nullptr) {
    return sllm_public_runtime::write_error(error_sink,
                                            SLLM_STATUS_INVALID_ARGUMENT,
                                            "whole graph capture span is null");
  }
  return sllm_public_runtime::write_error(
      error_sink, SLLM_STATUS_HIP_UNAVAILABLE,
      "whole graph capture is unavailable in the host runtime");
}

extern "C" sllm_status_t sllm_graph_span_create(
    const sllm_queue_t *const queue, const void *const *const plan_handles,
    const uint64_t plan_count, sllm_graph_span_t **const span,
    sllm_error_sink_t *const error_sink) noexcept {
  if (span != nullptr) {
    *span = nullptr;
  }
  const sllm_status_t sink_status =
      sllm_public_runtime::validate_error_sink(error_sink);
  if (sink_status != SLLM_STATUS_OK) {
    return sink_status;
  }
  if (queue == nullptr || plan_handles == nullptr || span == nullptr ||
      plan_count == 0U) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INVALID_ARGUMENT,
        "graph span queue, plans, count, or output is null");
  }
  return sllm_public_runtime::write_error(
      error_sink, SLLM_STATUS_HIP_UNAVAILABLE,
      "public HIP graph spans are unavailable in the host runtime");
}

extern "C" sllm_status_t
sllm_graph_span_execute(const sllm_graph_span_t *const span,
                        sllm_completion_t **const completion,
                        sllm_error_sink_t *const error_sink) noexcept {
  if (completion != nullptr) {
    *completion = nullptr;
  }
  const sllm_status_t sink_status =
      sllm_public_runtime::validate_error_sink(error_sink);
  if (sink_status != SLLM_STATUS_OK) {
    return sink_status;
  }
  if (span == nullptr || completion == nullptr) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INVALID_ARGUMENT,
        "graph span or completion output is null");
  }
  return sllm_public_runtime::write_error(
      error_sink, SLLM_STATUS_HIP_UNAVAILABLE,
      "public HIP graph spans are unavailable in the host runtime");
}

extern "C" sllm_status_t
sllm_graph_span_release(sllm_graph_span_t **const span,
                        sllm_error_sink_t *const error_sink) noexcept {
  const sllm_status_t sink_status =
      sllm_public_runtime::validate_error_sink(error_sink);
  if (sink_status != SLLM_STATUS_OK) {
    return sink_status;
  }
  if (span == nullptr || *span == nullptr) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INVALID_ARGUMENT, "graph span handle is null");
  }
  return sllm_public_runtime::write_error(
      error_sink, SLLM_STATUS_HIP_UNAVAILABLE,
      "public HIP graph spans are unavailable in the host runtime");
}

extern "C" sllm_status_t
sllm_graph_span_node_count(const sllm_graph_span_t *const span,
                           uint64_t *const node_count,
                           sllm_error_sink_t *const error_sink) noexcept {
  if (node_count != nullptr) {
    *node_count = 0U;
  }
  const sllm_status_t sink_status =
      sllm_public_runtime::validate_error_sink(error_sink);
  if (sink_status != SLLM_STATUS_OK) {
    return sink_status;
  }
  if (span == nullptr || node_count == nullptr) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INVALID_ARGUMENT,
        "graph span or node-count output is null");
  }
  return sllm_public_runtime::write_error(
      error_sink, SLLM_STATUS_HIP_UNAVAILABLE,
      "public HIP graph spans are unavailable in the host runtime");
}

#endif
