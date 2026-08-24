namespace {
using GdnBundlePlan = ElementwisePlan;

bool gdn_bundle_binding_overlap(const Buffer *const left_buffer,
                                const sllm_matmul::TensorMetadata &left,
                                const Buffer *const right_buffer,
                                const sllm_matmul::TensorMetadata &right) {
  return left_buffer == right_buffer &&
         sllm_matmul::intervals_overlap(left, right);
}
} // namespace

extern "C" sllm_status_t sllm_gdn_projection_bundle_prepare(
    const sllm_context_t *const raw_context,
    const sllm_gdn_projection_bundle_desc_t *const descriptor,
    sllm_gdn_projection_bundle_plan_t **const raw_plan,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (raw_plan != nullptr) *raw_plan = nullptr;
    const auto sink_status = sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) return sink_status;
    if (raw_context == nullptr || descriptor == nullptr || raw_plan == nullptr)
      return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                                               "GDN bundle context, descriptor, or output is null");
    if (descriptor->struct_size != sizeof(*descriptor) ||
        descriptor->abi_version != SLLM_HIP_ABI_VERSION ||
        descriptor->op_version != SLLM_HIP_GDN_PROJECTION_BUNDLE_VERSION)
      return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INVALID_ABI_VERSION,
                                               "GDN bundle descriptor ABI/version is unsupported");
    constexpr uint64_t widths[4] = {8192U, 4096U, 32U, 32U};
    std::array<sllm_matmul::DescriptorMetadata, 4> metadata{};
    std::array<Buffer *, 4> weights{};
    std::array<Buffer *, 4> outputs{};
    Context *context = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      context = lookup<Context>(raw_context, HandleKind::Context);
      if (context == nullptr)
        return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                                                 "GDN bundle context handle is stale");
      for (uint32_t index = 0U; index != 4U; ++index) {
        sllm_matmul_desc_t matmul = {};
        matmul.struct_size = sizeof(matmul);
        matmul.abi_version = SLLM_HIP_ABI_VERSION;
        matmul.op_version = SLLM_HIP_MATMUL_VERSION;
        matmul.activation = descriptor->activation;
        matmul.weight = descriptor->weights[index];
        matmul.output = descriptor->outputs[index];
        const sllm_status_t status =
            sllm_matmul::validate_and_copy_descriptor(&matmul, &metadata[index], error_sink);
        if (status != SLLM_STATUS_OK) return status;
        if (metadata[index].m != 1U || metadata[index].k != 2560U ||
            metadata[index].n != widths[index])
          return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_UNSUPPORTED,
                                                   "GDN bundle requires M=1,K=2560 and fixed role widths");
        weights[index] = lookup<Buffer>(descriptor->weights[index].buffer, HandleKind::Buffer);
        outputs[index] = lookup<Buffer>(descriptor->outputs[index].buffer, HandleKind::Buffer);
        if (weights[index] == nullptr || outputs[index] == nullptr ||
            weights[index]->context != context || outputs[index]->context != context)
          return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                                                   "GDN bundle buffer handle/context mismatch");
        if (weights[index]->release_active || outputs[index]->release_active)
          return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_PUBLIC_BUSY,
                                                   "GDN bundle buffer is releasing");
      }
      Buffer *activation = lookup<Buffer>(descriptor->activation.buffer, HandleKind::Buffer);
      if (activation == nullptr || activation->context != context || activation->release_active)
        return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                                                 "GDN bundle activation handle/context mismatch");
      for (uint32_t i = 0U; i != 4U; ++i) {
        if (gdn_bundle_binding_overlap(activation, metadata[i].activation, weights[i], metadata[i].weight) ||
            gdn_bundle_binding_overlap(activation, metadata[i].activation, outputs[i], metadata[i].output) ||
            gdn_bundle_binding_overlap(weights[i], metadata[i].weight, outputs[i], metadata[i].output))
          return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_ALIAS_OVERLAP,
                                                   "GDN bundle tensor intervals overlap");
      }
      if (!sllm_public_runtime::AccountingState::reserve_five_buffer_prepared_plan(
              context->accounting, activation->accounting, weights[0]->accounting,
              weights[1]->accounting, weights[2]->accounting, weights[3]->accounting) ||
          !sllm_public_runtime::AccountingState::reserve_five_buffer_prepared_plan(
              context->accounting, activation->accounting, outputs[0]->accounting,
              outputs[1]->accounting, outputs[2]->accounting, outputs[3]->accounting)) {
        return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                                                 "GDN bundle prepared-plan accounting exhausted");
      }
      try {
        auto candidate = std::make_unique<GdnBundlePlan>(context, activation, weights, outputs, metadata);
        const uintptr_t token = register_handle(candidate.get(), HandleKind::MatmulPlan);
        if (token == 0U) {
          (void)sllm_public_runtime::AccountingState::release_five_buffer_prepared_plan(
              context->accounting, activation->accounting, weights[0]->accounting,
              weights[1]->accounting, weights[2]->accounting, weights[3]->accounting);
          (void)sllm_public_runtime::AccountingState::release_five_buffer_prepared_plan(
              context->accounting, activation->accounting, outputs[0]->accounting,
              outputs[1]->accounting, outputs[2]->accounting, outputs[3]->accounting);
          return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                                                   "GDN bundle plan handle allocation failed");
        }
        *raw_plan = reinterpret_cast<sllm_gdn_projection_bundle_plan_t *>(token);
        (void)candidate.release();
        return SLLM_STATUS_OK;
      } catch (...) {
        (void)sllm_public_runtime::AccountingState::release_five_buffer_prepared_plan(
            context->accounting, activation->accounting, weights[0]->accounting,
            weights[1]->accounting, weights[2]->accounting, weights[3]->accounting);
        (void)sllm_public_runtime::AccountingState::release_five_buffer_prepared_plan(
            context->accounting, activation->accounting, outputs[0]->accounting,
            outputs[1]->accounting, outputs[2]->accounting, outputs[3]->accounting);
        throw;
      }
    }
  } catch (...) {
    return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                                             "unexpected exception in GDN bundle prepare");
  }
}

extern "C" sllm_status_t sllm_gdn_projection_bundle_plan_release(
    sllm_gdn_projection_bundle_plan_t **const raw_plan,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const auto sink_status = sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) return sink_status;
    if (raw_plan == nullptr || *raw_plan == nullptr)
      return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                                               "GDN bundle plan is null");
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    auto *plan = lookup<GdnBundlePlan>(*raw_plan, HandleKind::MatmulPlan);
    if (plan == nullptr || !plan->gdn_bundle)
      return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                                               "GDN bundle plan handle is stale");
    std::lock_guard<std::mutex> accounting_lock(plan->context->accounting_mutex);
    if (plan->release_active || plan->in_flight)
      return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_PUBLIC_BUSY,
                                               "GDN bundle plan is releasing or in flight");
    plan->release_active = true;
    const bool released_weights = sllm_public_runtime::AccountingState::release_five_buffer_prepared_plan(
        plan->context->accounting, plan->input0->accounting, plan->gdn_weights[0]->accounting,
        plan->gdn_weights[1]->accounting, plan->gdn_weights[2]->accounting, plan->gdn_weights[3]->accounting);
    const bool released_outputs = released_weights && sllm_public_runtime::AccountingState::release_five_buffer_prepared_plan(
        plan->context->accounting, plan->input0->accounting, plan->gdn_outputs[0]->accounting,
        plan->gdn_outputs[1]->accounting, plan->gdn_outputs[2]->accounting, plan->gdn_outputs[3]->accounting);
    unregister_handle(*raw_plan); *raw_plan = nullptr;
    if (!released_outputs) { plan->context->poisoned.store(true); poison_owner.retain(plan); return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR, "GDN bundle accounting release failed; plan quarantined"); }
    delete plan; return SLLM_STATUS_OK;
  } catch (...) { return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR, "unexpected exception in GDN bundle plan release"); }
}

extern "C" sllm_status_t sllm_gdn_projection_bundle_execute(
    const sllm_gdn_projection_bundle_plan_t *const raw_plan,
    const sllm_queue_t *const raw_queue, sllm_completion_t **const completion_output,
    sllm_gdn_projection_bundle_dispatch_info_t *const dispatch_info,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const auto sink_status = sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) return sink_status;
    if (raw_plan == nullptr || raw_queue == nullptr || completion_output == nullptr || dispatch_info == nullptr)
      return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT, "GDN bundle execute argument is null");
    if (dispatch_info->struct_size != sizeof(*dispatch_info) || dispatch_info->abi_version != SLLM_HIP_ABI_VERSION || dispatch_info->info_version != SLLM_HIP_GDN_PROJECTION_BUNDLE_DISPATCH_INFO_VERSION)
      return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INVALID_ABI_VERSION, "GDN bundle dispatch info ABI is unsupported");
    GdnBundlePlan *plan = nullptr; Queue *queue = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup<GdnBundlePlan>(raw_plan, HandleKind::MatmulPlan);
      queue = lookup<Queue>(raw_queue, HandleKind::Queue);
      if (plan == nullptr || queue == nullptr || !plan->gdn_bundle)
        return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE, "GDN bundle plan or queue handle is stale");
      if (plan->context != queue->context) return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH, "GDN bundle plan and queue devices differ");
      std::lock_guard<std::mutex> accounting_lock(plan->context->accounting_mutex);
      if (plan->context->poisoned.load() || plan->context->release_active || queue->release_active || plan->release_active || plan->in_flight)
        return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_PUBLIC_BUSY, "GDN bundle owner graph is busy");
      if (!sllm_public_runtime::AccountingState::reserve_five_buffer_submission(
              plan->context->accounting, queue->accounting, plan->input0->accounting,
              plan->gdn_weights[0]->accounting, plan->gdn_weights[1]->accounting,
              plan->gdn_weights[2]->accounting, plan->gdn_weights[3]->accounting) ||
          !sllm_public_runtime::AccountingState::reserve_five_buffer_submission(
              plan->context->accounting, queue->accounting, plan->input0->accounting,
              plan->gdn_outputs[0]->accounting, plan->gdn_outputs[1]->accounting,
              plan->gdn_outputs[2]->accounting, plan->gdn_outputs[3]->accounting))
        return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR, "GDN bundle submission accounting exhausted");
      plan->in_flight = true; *completion_output = nullptr;
    }
    const auto rollback = [&]() {
      std::lock_guard<std::mutex> accounting_lock(plan->context->accounting_mutex);
      (void)sllm_public_runtime::AccountingState::rollback_five_buffer_submission(
          plan->context->accounting, queue->accounting, plan->input0->accounting,
          plan->gdn_weights[0]->accounting, plan->gdn_weights[1]->accounting,
          plan->gdn_weights[2]->accounting, plan->gdn_weights[3]->accounting);
      (void)sllm_public_runtime::AccountingState::rollback_five_buffer_submission(
          plan->context->accounting, queue->accounting, plan->input0->accounting,
          plan->gdn_outputs[0]->accounting, plan->gdn_outputs[1]->accounting,
          plan->gdn_outputs[2]->accounting, plan->gdn_outputs[3]->accounting);
      plan->in_flight = false;
    };
    const sllm_status_t device_status = select_context_device(plan->context, error_sink);
    if (device_status != SLLM_STATUS_OK) { rollback(); return device_status; }
    hipDeviceProp_t properties = {};
    const sllm_status_t property_status = get_context_device_properties(plan->context, &properties, error_sink);
    if (property_status != SLLM_STATUS_OK) { rollback(); return property_status; }
    char arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME] = {};
    if (!copy_property(arch_name, sizeof(arch_name), properties.gcnArchName,
                       sizeof(properties.gcnArchName)) ||
        !(matches_runtime_gcn_arch(arch_name, "gfx1030") ||
          matches_runtime_gcn_arch(arch_name, "gfx1201"))) {
      rollback();
      return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
                                               "GDN bundle requires exact gfx1030 or gfx1201");
    }
    std::unique_ptr<Completion> candidate;
    NativeEventGuard event_guard;
    try {
      candidate = std::make_unique<Completion>(plan->context, queue, plan->input0,
                                                0U, false, std::vector<uint8_t>{});
      candidate->gdn_bundle = true;
      candidate->gdn_bundle_plan = plan;
      candidate->gdn_bundle_activation = plan->input0;
      candidate->gdn_bundle_weights = plan->gdn_weights;
      candidate->gdn_bundle_outputs = plan->gdn_outputs;

      /* Deferred queues are finalized by a queue fence and must remain
       * eventless. Profiled queues use the same start/end event pair as the
       * other numeric operations so kernel_elapsed_ns remains available. */
      if (queue_profiles_completions(queue)) {
        hipEvent_t native_event = nullptr;
        hipError_t event_status = hipEventCreateWithFlags(&native_event, 0U);
        if (event_status != hipSuccess) {
          rollback();
          return hip_failure(error_sink, event_status,
                             "GDN bundle event creation");
        }
        event_guard.adopt(plan->context, native_event);
        candidate->event = native_event;
        hipEvent_t timing_start_event = nullptr;
        event_status = hipEventCreateWithFlags(&timing_start_event, 0U);
        if (event_status != hipSuccess) {
          return rollback_unpublished_submission(
              candidate, event_guard,
              "GDN bundle timing event creation failed before enqueue",
              error_sink);
        }
        candidate->timing_start_event = timing_start_event;
      }

      uint64_t dispatch_id = 0U;
      uintptr_t token = 0U;
      try {
        std::lock_guard<std::mutex> registry_lock(registry_mutex);
        dispatch_id = plan->context->next_dispatch_id;
        if (dispatch_id != 0U) {
          plan->context->next_dispatch_id =
              dispatch_id == std::numeric_limits<uint64_t>::max()
                  ? 0U
                  : dispatch_id + 1U;
          token = register_handle(candidate.get(), HandleKind::Completion);
        }
      } catch (...) {
        return rollback_unpublished_submission(
            candidate, event_guard,
            "GDN bundle completion registry allocation failed", error_sink);
      }
      if (dispatch_id == 0U) {
        return rollback_unpublished_submission(
            candidate, event_guard, "GDN bundle dispatch id exhausted",
            error_sink);
      }
      if (token == 0U) {
        return rollback_unpublished_submission(
            candidate, event_guard,
            "GDN bundle completion handle allocation failed", error_sink);
      }
      if (candidate->event != nullptr) {
        event_guard.release();
      }

      if (candidate->timing_start_event != nullptr) {
        const hipError_t timing_status =
            hipEventRecord(candidate->timing_start_event, queue->stream);
        if (timing_status != hipSuccess) {
          return cleanup_failed_submission(
              candidate, token, timing_status,
              "GDN bundle timing event record", queue, error_sink);
        }
      }
      const auto ptr = [](Buffer *buffer, uint64_t offset) -> uint16_t * {
        return reinterpret_cast<uint16_t *>(
            static_cast<uint8_t *>(buffer->device_pointer) + offset);
      };
      const hipError_t launch_status =
          ::sllm_gdn_projection_bundle_kernel::launch(
              ptr(plan->input0, plan->gdn_metadata[0].activation.byte_offset),
              ptr(plan->gdn_weights[0], plan->gdn_metadata[0].weight.byte_offset),
              ptr(plan->gdn_weights[1], plan->gdn_metadata[1].weight.byte_offset),
              ptr(plan->gdn_weights[2], plan->gdn_metadata[2].weight.byte_offset),
              ptr(plan->gdn_weights[3], plan->gdn_metadata[3].weight.byte_offset),
              ptr(plan->gdn_outputs[0], plan->gdn_metadata[0].output.byte_offset),
              ptr(plan->gdn_outputs[1], plan->gdn_metadata[1].output.byte_offset),
              ptr(plan->gdn_outputs[2], plan->gdn_metadata[2].output.byte_offset),
              ptr(plan->gdn_outputs[3], plan->gdn_metadata[3].output.byte_offset),
              2560U, queue->stream);
      if (launch_status != hipSuccess) {
        return cleanup_failed_submission(candidate, token, launch_status,
                                         "GDN bundle kernel launch", queue,
                                         error_sink);
      }
      if (candidate->event != nullptr) {
        const hipError_t event_status = hipEventRecord(candidate->event,
                                                       queue->stream);
        if (event_status != hipSuccess) {
          return cleanup_failed_submission(candidate, token, event_status,
                                           "GDN bundle event record", queue,
                                           error_sink);
        }
      }
      dispatch_info->backend = 1U;
      dispatch_info->dispatch_id = dispatch_id;
      dispatch_info->dispatch_count = 1U;
      dispatch_info->kernel_id = 100U;
      dispatch_info->workgroup_size_x = 256U;
      dispatch_info->grid_size_x = 12352U;
      dispatch_info->fallback_allowed = 0U;
      dispatch_info->fallback_used = 0U;
      dispatch_info->m = 1U;
      dispatch_info->k = 2560U;
      dispatch_info->widths[0] = 8192U;
      dispatch_info->widths[1] = 4096U;
      dispatch_info->widths[2] = 32U;
      dispatch_info->widths[3] = 32U;
      std::strncpy(dispatch_info->kernel_symbol,
                   "sllm_gdn_projection_bundle_bf16_fp32_decode_v1",
                   sizeof(dispatch_info->kernel_symbol) - 1U);
      std::strncpy(dispatch_info->device_symbol,
                   "sllm_gdn_projection_bundle_bf16_fp32_decode_v1",
                   sizeof(dispatch_info->device_symbol) - 1U);
      std::strncpy(dispatch_info->gcn_arch_name, arch_name,
                   sizeof(dispatch_info->gcn_arch_name) - 1U);
      *completion_output = reinterpret_cast<sllm_completion_t *>(token);
      (void)candidate.release();
      return SLLM_STATUS_OK;
    } catch (...) {
      rollback();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "unexpected exception in GDN bundle execution");
    }
  } catch (...) { return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR, "unexpected exception in GDN bundle execute"); }
}
