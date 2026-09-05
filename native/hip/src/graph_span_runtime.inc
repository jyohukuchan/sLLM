/* Bounded request-owned HIP graph spans.  This file is included by the
 * production runtime after the public completion state machine has been
 * defined.  The public API intentionally accepts opaque prepared-plan
 * handles: the native runtime resolves and pins the exact plan objects while
 * the request queue is idle. */

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
  std::vector<GraphSpanPlanRef> plans;
  std::vector<sllm_completion_t *> capture_markers;

  GraphSpan(Context *const context_value, Queue *const queue_value,
            const sllm_queue_t *const raw_queue_value)
      : QuarantineNode(HandleKind::GraphSpan), context(context_value),
        queue(queue_value), raw_queue(raw_queue_value), graph(nullptr),
        graph_exec(nullptr), actual_node_count(0U), logical_plan_count(0U),
        release_active(false), in_flight(false), capture_owner(false), plans(),
        capture_markers() {}
};

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
          ::sllm_matmul_kernel::KernelVariant::Fp8Native ||
      plan->fp8_lt_plan != nullptr) {
    return exact_gfx1201 && plan->matmul_metadata.fp8_outer &&
           plan->matmul_metadata.fp8_dtype == SLLM_TENSOR_DTYPE_F8_E4M3_FN &&
           plan->matmul_kernel_variant ==
               ::sllm_matmul_kernel::KernelVariant::Fp8Native &&
           plan->fp8_lt_plan != nullptr && plan->matmul_workspace != nullptr &&
           plan->matmul_workspace_bytes != 0U;
  }
  if (plan->matmul_kernel_variant ==
      ::sllm_matmul_kernel::KernelVariant::HipBlas) {
    return false;
  }
  return true;
}

bool graph_span_qwen38_pack(const ElementwisePlan *const plan,
                            const char *const arch_name) noexcept {
  if (plan == nullptr || !plan->qwen38_projection_pack2 ||
      arch_name == nullptr) {
    return false;
  }

  const uint32_t role = plan->qwen38_projection_pack2_role;
  if (role == SLLM_HIP_QWEN38_PROJECTION_PACK2_ROLE_NVFP4_MLP_GATE_UP) {
    return plan->qwen38_projection_pack2_metadata[0].m == 1U &&
           plan->qwen38_projection_pack2_metadata[0].k == UINT64_C(5120) &&
           plan->qwen38_projection_pack2_metadata[0].n == UINT64_C(17408) &&
           plan->qwen38_projection_pack2_metadata[1].m == 1U &&
           plan->qwen38_projection_pack2_metadata[1].k == UINT64_C(5120) &&
           plan->qwen38_projection_pack2_metadata[1].n == UINT64_C(17408);
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
    return plan->qwen38_projection_pack2_kernel_variants[0] ==
               ::sllm_matmul_kernel::KernelVariant::Fp8Native &&
           plan->qwen38_projection_pack2_kernel_variants[1] ==
               ::sllm_matmul_kernel::KernelVariant::Fp8Native &&
           plan->qwen38_projection_pack2_fp8_lt_plans[0] != nullptr &&
           plan->qwen38_projection_pack2_fp8_lt_plans[1] != nullptr;
  }
  if (exact_gfx1030) {
    const auto variant = plan->qwen38_projection_pack2_kernel_variants[0];
    return variant == plan->qwen38_projection_pack2_kernel_variants[1] &&
           variant != ::sllm_matmul_kernel::KernelVariant::Fp8Native &&
           (variant == ::sllm_matmul_kernel::KernelVariant::
                           Fp8OuterDecodeGfx1030Half2Wave4Col32 ||
            variant == ::sllm_matmul_kernel::KernelVariant::
                           Fp8OuterDecodeGfx1030Dword8Wave4Col32 ||
            variant == ::sllm_matmul_kernel::KernelVariant::
                           Fp8OuterDecodeGfx1030LdsLutWave4Col32 ||
            variant == ::sllm_matmul_kernel::KernelVariant::
                           Fp8OuterDecodeGfx1030ActivationSharedWave4Col32 ||
            variant == ::sllm_matmul_kernel::KernelVariant::
                           Fp8OuterDecodeGfx1030ActivationSharedWave8Col64);
  }
  return false;
}

bool graph_span_elementwise(const ElementwisePlan *const plan) noexcept {
  return plan != nullptr && !plan->embedding && !plan->matmul &&
         !plan->gdn_bundle && !plan->mlp_bundle &&
         !plan->qwen38_projection_pack2 && !plan->argmax &&
         plan->metadata.operation != SLLM_ELEMENTWISE_OPERATION_COPY &&
         plan->metadata.element_count != 0U &&
         graph_span_tensor_is_m1(plan->metadata.input0) &&
         graph_span_tensor_is_m1(plan->metadata.input1) &&
         graph_span_tensor_is_m1(plan->metadata.output);
}

bool graph_span_plan_m1(const RmsNormPlan *const plan) noexcept {
  if (plan == nullptr || plan->fused) {
    return plan != nullptr && plan->fused &&
           graph_span_tensor_is_m1(plan->residual_metadata.residual);
  }
  return graph_span_tensor_is_m1(plan->metadata.activation);
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
  }
  const bool owned_capture = span->capture_owner;
  if (span->queue != nullptr && owned_capture) {
    span->capture_owner = false;
    span->queue->graph_capture_active.store(false, std::memory_order_release);
  }
  if (owned_capture && graph_capture_queue == span->queue) {
    graph_capture_queue = nullptr;
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
  --completion->queue->accounting.active_submissions;
  completion->graph_span_owner->in_flight = false;
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
    completion->graph_span_owner->in_flight = false;
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
      if (span->release_active || span->in_flight || span->queue == nullptr ||
          span->queue->release_active ||
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
      span->in_flight = true;
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
      span->in_flight = false;
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
      span->in_flight = false;
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
      span->in_flight = false;
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
          span->graph_exec == nullptr || span->queue == nullptr ||
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
} // namespace

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
