namespace {
using MlpBundlePlan = ElementwisePlan;

constexpr uint64_t kMlpWidth = 9216U;

bool mlp_bundle_binding_overlap(const Buffer *const left_buffer,
                                const sllm_matmul::TensorMetadata &left,
                                const Buffer *const right_buffer,
                                const sllm_matmul::TensorMetadata &right) {
  return left_buffer == right_buffer &&
         sllm_matmul::intervals_overlap(left, right);
}
} // namespace

extern "C" sllm_status_t sllm_mlp_gate_up_silu_bundle_prepare(
    const sllm_context_t *const raw_context,
    const sllm_mlp_gate_up_silu_bundle_desc_t *const descriptor,
    sllm_mlp_gate_up_silu_bundle_plan_t **const raw_plan,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (raw_plan != nullptr) *raw_plan = nullptr;
    const auto sink_status = sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) return sink_status;
    if (raw_context == nullptr || descriptor == nullptr || raw_plan == nullptr)
      return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                                               "MLP gate/up/SiLU bundle context, descriptor, or output is null");
    if (descriptor->struct_size != sizeof(*descriptor) ||
        descriptor->abi_version != SLLM_HIP_ABI_VERSION ||
        descriptor->op_version != SLLM_HIP_MLP_GATE_UP_SILU_BUNDLE_VERSION)
      return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INVALID_ABI_VERSION,
                                               "MLP gate/up/SiLU bundle descriptor ABI/version is unsupported");
    std::array<sllm_matmul::DescriptorMetadata, 2> metadata{};
    sllm_rmsnorm::TensorMetadata silu_metadata{};
    std::array<Buffer *, 2> weights{};
    std::array<Buffer *, 3> outputs{};
    Context *context = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      context = lookup<Context>(raw_context, HandleKind::Context);
      if (context == nullptr)
        return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                                                 "MLP gate/up/SiLU bundle context handle is stale");
      for (uint32_t index = 0U; index != 2U; ++index) {
        sllm_matmul_desc_t matmul = {};
        matmul.struct_size = sizeof(matmul);
        matmul.abi_version = SLLM_HIP_ABI_VERSION;
        matmul.op_version = SLLM_HIP_MATMUL_VERSION;
        matmul.activation = descriptor->activation;
        matmul.weight = index == 0U ? descriptor->gate_weight : descriptor->up_weight;
        matmul.output = index == 0U ? descriptor->gate_output : descriptor->up_output;
        const sllm_status_t status =
            sllm_matmul::validate_and_copy_descriptor(&matmul, &metadata[index], error_sink);
        if (status != SLLM_STATUS_OK) return status;
        if (metadata[index].m != 1U || metadata[index].k != 2560U ||
            metadata[index].n != kMlpWidth)
          return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_UNSUPPORTED,
                                                   "MLP gate/up/SiLU bundle requires M=1,K=2560 and fixed role widths");
        weights[index] = lookup<Buffer>((index == 0U ? descriptor->gate_weight : descriptor->up_weight).buffer, HandleKind::Buffer);
        outputs[index] = lookup<Buffer>((index == 0U ? descriptor->gate_output : descriptor->up_output).buffer, HandleKind::Buffer);
        if (weights[index] == nullptr || outputs[index] == nullptr ||
            weights[index]->context != context || outputs[index]->context != context)
          return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                                                   "MLP gate/up/SiLU bundle buffer handle/context mismatch");
        if (weights[index]->release_active || outputs[index]->release_active)
          return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_PUBLIC_BUSY,
                                                   "MLP gate/up/SiLU bundle buffer is releasing");
      }
      outputs[2] = lookup<Buffer>(descriptor->silu_output.buffer, HandleKind::Buffer);
      const sllm_status_t silu_status = sllm_rmsnorm::validate_tensor_binding(
          &descriptor->silu_output, &silu_metadata, error_sink);
      if (silu_status != SLLM_STATUS_OK) return silu_status;
      if (silu_metadata.shape[0] != 1U || silu_metadata.shape[1] != kMlpWidth ||
          silu_metadata.payload_bytes != metadata[0].output.payload_bytes)
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INVALID_ARGUMENT,
            "MLP gate/up/SiLU bundle SiLU output shape is unsupported");
      sllm_matmul::TensorMetadata silu_matmul_metadata{
          silu_metadata.byte_offset, silu_metadata.payload_bytes,
          silu_metadata.end_offset, {silu_metadata.shape[0], silu_metadata.shape[1]}};
      if (outputs[2] == nullptr || outputs[2]->context != context ||
          outputs[2]->release_active)
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "MLP gate/up/SiLU bundle SiLU output handle/context mismatch");
      Buffer *activation = lookup<Buffer>(descriptor->activation.buffer, HandleKind::Buffer);
      if (activation == nullptr || activation->context != context || activation->release_active)
        return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                                                 "MLP gate/up/SiLU bundle activation handle/context mismatch");
      for (uint32_t i = 0U; i != 2U; ++i) {
        if (mlp_bundle_binding_overlap(activation, metadata[i].activation, weights[i], metadata[i].weight) ||
            mlp_bundle_binding_overlap(activation, metadata[i].activation, outputs[i], metadata[i].output) ||
            mlp_bundle_binding_overlap(weights[i], metadata[i].weight, outputs[i], metadata[i].output))
          return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_ALIAS_OVERLAP,
                                                   "MLP gate/up/SiLU bundle tensor intervals overlap");
      }
      if (mlp_bundle_binding_overlap(activation, metadata[0].activation,
                                     outputs[2], silu_matmul_metadata) ||
          mlp_bundle_binding_overlap(weights[0], metadata[0].weight,
                                     weights[1], metadata[1].weight) ||
          mlp_bundle_binding_overlap(outputs[0], metadata[0].output,
                                     outputs[1], metadata[1].output) ||
          mlp_bundle_binding_overlap(weights[0], metadata[0].weight,
                                     outputs[1], metadata[1].output) ||
          mlp_bundle_binding_overlap(weights[1], metadata[1].weight,
                                     outputs[0], metadata[0].output) ||
          mlp_bundle_binding_overlap(weights[0], metadata[0].weight,
                                     outputs[2], silu_matmul_metadata) ||
          mlp_bundle_binding_overlap(weights[1], metadata[1].weight,
                                     outputs[2], silu_matmul_metadata) ||
          mlp_bundle_binding_overlap(outputs[0], metadata[0].output,
                                     outputs[2], silu_matmul_metadata) ||
          mlp_bundle_binding_overlap(outputs[1], metadata[1].output,
                                     outputs[2], silu_matmul_metadata))
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_ALIAS_OVERLAP,
            "MLP gate/up/SiLU bundle tensor intervals overlap");
      const bool reserved_weights =
          sllm_public_runtime::AccountingState::reserve_five_buffer_prepared_plan(
              context->accounting, activation->accounting, weights[0]->accounting,
              weights[1]->accounting, activation->accounting, activation->accounting);
      if (!reserved_weights) {
        return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                                                 "MLP gate/up/SiLU bundle prepared-plan accounting exhausted");
      }
      const bool reserved_outputs =
          sllm_public_runtime::AccountingState::reserve_five_buffer_prepared_plan(
              context->accounting, activation->accounting, outputs[0]->accounting,
              outputs[1]->accounting, outputs[2]->accounting, activation->accounting);
      if (!reserved_outputs) {
        (void)sllm_public_runtime::AccountingState::release_five_buffer_prepared_plan(
            context->accounting, activation->accounting, weights[0]->accounting,
            weights[1]->accounting, activation->accounting, activation->accounting);
        return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                                                 "MLP gate/up/SiLU bundle prepared-plan accounting exhausted");
      }
      try {
        auto candidate = std::make_unique<MlpBundlePlan>(context, activation, weights, outputs, metadata);
        candidate->mlp_silu_metadata.byte_offset = silu_metadata.byte_offset;
        candidate->mlp_silu_metadata.payload_bytes = silu_metadata.payload_bytes;
        candidate->mlp_silu_metadata.end_offset = silu_metadata.end_offset;
        candidate->mlp_silu_metadata.shape[0] = silu_metadata.shape[0];
        candidate->mlp_silu_metadata.shape[1] = silu_metadata.shape[1];
        const uintptr_t token = register_handle(candidate.get(), HandleKind::MatmulPlan);
        if (token == 0U) {
          (void)sllm_public_runtime::AccountingState::release_five_buffer_prepared_plan(
              context->accounting, activation->accounting, weights[0]->accounting,
              weights[1]->accounting, activation->accounting, activation->accounting);
          (void)sllm_public_runtime::AccountingState::release_five_buffer_prepared_plan(
              context->accounting, activation->accounting, outputs[0]->accounting,
              outputs[1]->accounting, outputs[2]->accounting, activation->accounting);
          return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                                                   "MLP gate/up/SiLU bundle plan handle allocation failed");
        }
        *raw_plan = reinterpret_cast<sllm_mlp_gate_up_silu_bundle_plan_t *>(token);
        (void)candidate.release();
        return SLLM_STATUS_OK;
      } catch (...) {
        (void)sllm_public_runtime::AccountingState::release_five_buffer_prepared_plan(
            context->accounting, activation->accounting, weights[0]->accounting,
              weights[1]->accounting, activation->accounting, activation->accounting);
        (void)sllm_public_runtime::AccountingState::release_five_buffer_prepared_plan(
            context->accounting, activation->accounting, outputs[0]->accounting,
              outputs[1]->accounting, outputs[2]->accounting, activation->accounting);
        throw;
      }
    }
  } catch (...) {
    return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR,
                                             "unexpected exception in MLP gate/up/SiLU bundle prepare");
  }
}

extern "C" sllm_status_t sllm_mlp_gate_up_silu_bundle_plan_release(
    sllm_mlp_gate_up_silu_bundle_plan_t **const raw_plan,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const auto sink_status = sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) return sink_status;
    if (raw_plan == nullptr || *raw_plan == nullptr)
      return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT,
                                               "MLP gate/up/SiLU bundle plan is null");
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    auto *plan = lookup<MlpBundlePlan>(*raw_plan, HandleKind::MatmulPlan);
    if (plan == nullptr || !plan->mlp_bundle)
      return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                                               "MLP gate/up/SiLU bundle plan handle is stale");
    std::lock_guard<std::mutex> accounting_lock(plan->context->accounting_mutex);
    if (plan->release_active || plan->in_flight)
      return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_PUBLIC_BUSY,
                                               "MLP gate/up/SiLU bundle plan is releasing or in flight");
    plan->release_active = true;
    const bool released_weights = sllm_public_runtime::AccountingState::release_five_buffer_prepared_plan(
        plan->context->accounting, plan->input0->accounting, plan->mlp_weights[0]->accounting,
        plan->mlp_weights[1]->accounting, plan->input0->accounting, plan->input0->accounting);
    const bool released_outputs = released_weights && sllm_public_runtime::AccountingState::release_five_buffer_prepared_plan(
        plan->context->accounting, plan->input0->accounting, plan->mlp_outputs[0]->accounting,
        plan->mlp_outputs[1]->accounting, plan->mlp_outputs[2]->accounting, plan->input0->accounting);
    unregister_handle(*raw_plan); *raw_plan = nullptr;
    if (!released_outputs) { plan->context->poisoned.store(true); poison_owner.retain(plan); return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR, "MLP gate/up/SiLU bundle accounting release failed; plan quarantined"); }
    delete plan; return SLLM_STATUS_OK;
  } catch (...) { return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR, "unexpected exception in MLP gate/up/SiLU bundle plan release"); }
}

extern "C" sllm_status_t sllm_mlp_gate_up_silu_bundle_execute(
    const sllm_mlp_gate_up_silu_bundle_plan_t *const raw_plan,
    const sllm_queue_t *const raw_queue, sllm_completion_t **const completion_output,
    sllm_mlp_gate_up_silu_bundle_dispatch_info_t *const dispatch_info,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const auto sink_status = sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) return sink_status;
    if (raw_plan == nullptr || raw_queue == nullptr || completion_output == nullptr || dispatch_info == nullptr)
      return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INVALID_ARGUMENT, "MLP gate/up/SiLU bundle execute argument is null");
    if (dispatch_info->struct_size != sizeof(*dispatch_info) || dispatch_info->abi_version != SLLM_HIP_ABI_VERSION || dispatch_info->info_version != SLLM_HIP_MLP_GATE_UP_SILU_BUNDLE_DISPATCH_INFO_VERSION)
      return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INVALID_ABI_VERSION, "MLP gate/up/SiLU bundle dispatch info ABI is unsupported");
    MlpBundlePlan *plan = nullptr; Queue *queue = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup<MlpBundlePlan>(raw_plan, HandleKind::MatmulPlan);
      queue = lookup<Queue>(raw_queue, HandleKind::Queue);
      if (plan == nullptr || queue == nullptr || !plan->mlp_bundle)
        return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE, "MLP gate/up/SiLU bundle plan or queue handle is stale");
      if (plan->context != queue->context) return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH, "MLP gate/up/SiLU bundle plan and queue devices differ");
      std::lock_guard<std::mutex> accounting_lock(plan->context->accounting_mutex);
      if (plan->context->poisoned.load() || plan->context->release_active || queue->release_active || plan->release_active || plan->in_flight)
        return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_PUBLIC_BUSY, "MLP gate/up/SiLU bundle owner graph is busy");
      const bool reserved_weights =
          sllm_public_runtime::AccountingState::reserve_five_buffer_submission(
              plan->context->accounting, queue->accounting, plan->input0->accounting,
              plan->mlp_weights[0]->accounting, plan->mlp_weights[1]->accounting,
              plan->input0->accounting, plan->input0->accounting);
      if (!reserved_weights)
        return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR, "MLP gate/up/SiLU bundle submission accounting exhausted");
      const bool reserved_outputs =
          sllm_public_runtime::AccountingState::reserve_five_buffer_submission(
              plan->context->accounting, queue->accounting, plan->input0->accounting,
              plan->mlp_outputs[0]->accounting, plan->mlp_outputs[1]->accounting,
              plan->mlp_outputs[2]->accounting, plan->input0->accounting);
      if (!reserved_outputs) {
        (void)sllm_public_runtime::AccountingState::rollback_five_buffer_submission(
            plan->context->accounting, queue->accounting, plan->input0->accounting,
            plan->mlp_weights[0]->accounting, plan->mlp_weights[1]->accounting,
            plan->input0->accounting, plan->input0->accounting);
        return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR, "MLP gate/up/SiLU bundle submission accounting exhausted");
      }
      plan->in_flight = true; *completion_output = nullptr;
    }
    const auto rollback = [&]() {
      std::lock_guard<std::mutex> accounting_lock(plan->context->accounting_mutex);
      (void)sllm_public_runtime::AccountingState::rollback_five_buffer_submission(
          plan->context->accounting, queue->accounting, plan->input0->accounting,
          plan->mlp_weights[0]->accounting, plan->mlp_weights[1]->accounting,
          plan->input0->accounting, plan->input0->accounting);
      (void)sllm_public_runtime::AccountingState::rollback_five_buffer_submission(
          plan->context->accounting, queue->accounting, plan->input0->accounting,
          plan->mlp_outputs[0]->accounting, plan->mlp_outputs[1]->accounting,
          plan->mlp_outputs[2]->accounting, plan->input0->accounting);
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
                                               "MLP gate/up/SiLU bundle requires exact gfx1030 or gfx1201");
    }
    std::unique_ptr<Completion> candidate;
    NativeEventGuard event_guard;
    try {
      candidate = std::make_unique<Completion>(plan->context, queue, plan->input0,
                                                0U, false, std::vector<uint8_t>{});
      candidate->mlp_bundle = true;
      candidate->mlp_bundle_plan = plan;
      candidate->mlp_bundle_activation = plan->input0;
      candidate->mlp_bundle_weights = plan->mlp_weights;
      candidate->mlp_bundle_outputs = plan->mlp_outputs;

      /* Deferred queues are finalized by a queue fence and must remain
       * eventless. Profiled queues use the same start/end event pair as the
       * other numeric operations so kernel_elapsed_ns remains available. */
      if (queue_profiles_completions(queue)) {
        hipEvent_t native_event = nullptr;
        hipError_t event_status = hipEventCreateWithFlags(&native_event, 0U);
        if (event_status != hipSuccess) {
          rollback();
          return hip_failure(error_sink, event_status,
                             "MLP gate/up/SiLU bundle event creation");
        }
        event_guard.adopt(plan->context, native_event);
        candidate->event = native_event;
        hipEvent_t timing_start_event = nullptr;
        event_status = hipEventCreateWithFlags(&timing_start_event, 0U);
        if (event_status != hipSuccess) {
          return rollback_unpublished_submission(
              candidate, event_guard,
              "MLP gate/up/SiLU bundle timing event creation failed before enqueue",
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
            "MLP gate/up/SiLU bundle completion registry allocation failed", error_sink);
      }
      if (dispatch_id == 0U) {
        return rollback_unpublished_submission(
            candidate, event_guard, "MLP gate/up/SiLU bundle dispatch id exhausted",
            error_sink);
      }
      if (token == 0U) {
        return rollback_unpublished_submission(
            candidate, event_guard,
            "MLP gate/up/SiLU bundle completion handle allocation failed", error_sink);
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
              "MLP gate/up/SiLU bundle timing event record", queue, error_sink);
        }
      }
      const auto ptr = [](Buffer *buffer, uint64_t offset) -> uint16_t * {
        return reinterpret_cast<uint16_t *>(
            static_cast<uint8_t *>(buffer->device_pointer) + offset);
      };
      const hipError_t launch_status =
          ::sllm_mlp_gate_up_silu_bundle_kernel::launch(
              ptr(plan->input0, plan->mlp_metadata[0].activation.byte_offset),
              ptr(plan->mlp_weights[0], plan->mlp_metadata[0].weight.byte_offset),
              ptr(plan->mlp_weights[1], plan->mlp_metadata[1].weight.byte_offset),
              ptr(plan->mlp_outputs[0], plan->mlp_metadata[0].output.byte_offset),
              ptr(plan->mlp_outputs[1], plan->mlp_metadata[1].output.byte_offset),
              ptr(plan->mlp_outputs[2], plan->mlp_silu_metadata.byte_offset),
              2560U, queue->stream);
      if (launch_status != hipSuccess) {
        return cleanup_failed_submission(candidate, token, launch_status,
                                         "MLP gate/up/SiLU bundle kernel launch", queue,
                                         error_sink);
      }
      if (candidate->event != nullptr) {
        const hipError_t event_status = hipEventRecord(candidate->event,
                                                       queue->stream);
        if (event_status != hipSuccess) {
          return cleanup_failed_submission(candidate, token, event_status,
                                           "MLP gate/up/SiLU bundle event record", queue,
                                           error_sink);
        }
      }
      dispatch_info->backend = 1U;
      dispatch_info->dispatch_id = dispatch_id;
      dispatch_info->dispatch_count = 1U;
      dispatch_info->kernel_id =
          ::sllm_mlp_gate_up_silu_bundle_kernel::kBaselineKernelId;
      dispatch_info->workgroup_size_x = 256U;
      dispatch_info->grid_size_x = static_cast<uint32_t>(kMlpWidth);
      dispatch_info->fallback_allowed = 0U;
      dispatch_info->fallback_used = 0U;
      dispatch_info->m = 1U;
      dispatch_info->k = 2560U;
      dispatch_info->n = kMlpWidth;
      dispatch_info->output_elements = kMlpWidth * 3U;
      sllm_public_runtime::copy_fixed_string(
          dispatch_info->kernel_symbol,
          sizeof(dispatch_info->kernel_symbol),
          ::sllm_mlp_gate_up_silu_bundle_kernel::kBaselineLogicalKernelId);
      sllm_public_runtime::copy_fixed_string(
          dispatch_info->device_symbol,
          sizeof(dispatch_info->device_symbol),
          ::sllm_mlp_gate_up_silu_bundle_kernel::kBaselineDeviceSymbol);
      std::strncpy(dispatch_info->gcn_arch_name, arch_name,
                   sizeof(dispatch_info->gcn_arch_name) - 1U);
      *completion_output = reinterpret_cast<sllm_completion_t *>(token);
      (void)candidate.release();
      return SLLM_STATUS_OK;
    } catch (...) {
      rollback();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "unexpected exception in MLP gate/up/SiLU bundle execution");
    }
  } catch (...) { return sllm_public_runtime::write_error(error_sink, SLLM_STATUS_INTERNAL_ERROR, "unexpected exception in MLP gate/up/SiLU bundle execute"); }
}
