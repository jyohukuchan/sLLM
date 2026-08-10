extern "C" sllm_status_t
sllm_matmul_prepare(const sllm_context_t *const raw_context,
                    const sllm_matmul_desc_t *const descriptor,
                    sllm_matmul_plan_t **const raw_plan,
                    sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (raw_plan != nullptr) {
      *raw_plan = nullptr;
    }
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_context == nullptr || raw_plan == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "matmul context or plan output is null");
    }
    const sllm_status_t prefix_status =
        sllm_matmul::validate_descriptor_prefix(descriptor, error_sink);
    if (prefix_status != SLLM_STATUS_OK) {
      return prefix_status;
    }
    const sllm_matmul_desc_t descriptor_copy = *descriptor;
    sllm_matmul::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_matmul::validate_and_copy_descriptor(&descriptor_copy, &metadata,
                                                  error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }

    std::unique_ptr<ElementwisePlan> candidate;
    uintptr_t token = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      Context *const context =
          lookup<Context>(raw_context, HandleKind::Context);
      Buffer *const activation =
          lookup<Buffer>(descriptor_copy.activation.buffer, HandleKind::Buffer);
      Buffer *const weight =
          lookup<Buffer>(descriptor_copy.weight.buffer, HandleKind::Buffer);
      Buffer *const output =
          lookup<Buffer>(descriptor_copy.output.buffer, HandleKind::Buffer);
      if (context == nullptr || activation == nullptr || weight == nullptr ||
          output == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "matmul context or buffer handle is stale or wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (context->poisoned.load() || context->release_active ||
          activation->release_active || weight->release_active ||
          output->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "matmul cannot prepare while a context or buffer is releasing");
      }
      if (activation->context != context || weight->context != context ||
          output->context != context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH,
            "matmul buffers must belong to the supplied context and device");
      }
      if (metadata.activation.end_offset > activation->size_bytes ||
          metadata.weight.end_offset > weight->size_bytes ||
          metadata.output.end_offset > output->size_bytes) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
            "matmul tensor interval exceeds its backing buffer");
      }
      const auto overlaps_if_same =
          [](const Buffer *const left_buffer,
             const sllm_matmul::TensorMetadata &left,
             const Buffer *const right_buffer,
             const sllm_matmul::TensorMetadata &right) {
            return left_buffer == right_buffer &&
                   sllm_matmul::intervals_overlap(left, right);
          };
      if (overlaps_if_same(activation, metadata.activation, weight,
                           metadata.weight) ||
          overlaps_if_same(activation, metadata.activation, output,
                           metadata.output) ||
          overlaps_if_same(weight, metadata.weight, output, metadata.output)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_ALIAS_OVERLAP,
            "matmul tensor intervals overlap within one backing buffer");
      }
      if (!sllm_public_runtime::AccountingState::reserve_prepared_plan(
              context->accounting, activation->accounting, weight->accounting,
              output->accounting)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "matmul prepared-plan accounting is exhausted");
      }
      try {
        candidate = std::make_unique<ElementwisePlan>(context, activation,
                                                      weight, output, metadata);
        token = register_handle(candidate.get(), HandleKind::MatmulPlan);
      } catch (...) {
        (void)sllm_public_runtime::AccountingState::release_prepared_plan(
            context->accounting, activation->accounting, weight->accounting,
            output->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "matmul prepared-plan allocation or registration failed");
      }
      if (token == 0U) {
        (void)sllm_public_runtime::AccountingState::release_prepared_plan(
            context->accounting, activation->accounting, weight->accounting,
            output->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "matmul prepared-plan handle allocation failed");
      }
    }
    *raw_plan = reinterpret_cast<sllm_matmul_plan_t *>(token);
    (void)candidate.release();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in matmul prepare");
  }
}

extern "C" sllm_status_t
sllm_matmul_plan_release(sllm_matmul_plan_t **const raw_plan,
                         sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_plan == nullptr || *raw_plan == nullptr) {
      return sllm_public_runtime::write_error(error_sink,
                                              SLLM_STATUS_INVALID_ARGUMENT,
                                              "matmul plan handle is null");
    }
    ElementwisePlan *plan = nullptr;
    bool quarantine_plan = false;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup<ElementwisePlan>(*raw_plan, HandleKind::MatmulPlan);
      if (plan == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "matmul plan handle is stale or has the wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      if (plan->release_active || plan->in_flight) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "matmul plan is releasing or has an in-flight completion");
      }
      plan->release_active = true;
      const bool accounting_released =
          !sllm_public_runtime::FaultInjector::consume(
              sllm_public_runtime::FaultPoint::AccountingFailure) &&
          sllm_public_runtime::AccountingState::release_prepared_plan(
              plan->context->accounting, plan->input0->accounting,
              plan->input1->accounting, plan->output->accounting);
      unregister_handle(*raw_plan);
      *raw_plan = nullptr;
      if (!accounting_released) {
        plan->context->poisoned.store(true);
        quarantine_plan = true;
      }
    }
    if (quarantine_plan) {
      poison_owner.retain(plan);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "matmul prepared-plan accounting release failed; plan quarantined");
    }
    delete plan;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in matmul plan release");
  }
}

extern "C" sllm_status_t
sllm_matmul_execute(const sllm_matmul_plan_t *const raw_plan,
                    const sllm_queue_t *const raw_queue,
                    sllm_completion_t **const completion_output,
                    sllm_matmul_dispatch_info_t *const dispatch_info,
                    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_plan == nullptr || raw_queue == nullptr ||
        completion_output == nullptr || dispatch_info == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "matmul execute plan, queue, completion, or dispatch info is null");
    }
    const sllm_status_t dispatch_status =
        validate_matmul_dispatch_info(dispatch_info, error_sink);
    if (dispatch_status != SLLM_STATUS_OK) {
      return dispatch_status;
    }

    ElementwisePlan *plan = nullptr;
    Queue *queue = nullptr;
    uint64_t dispatch_id = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup<ElementwisePlan>(raw_plan, HandleKind::MatmulPlan);
      queue = lookup<Queue>(raw_queue, HandleKind::Queue);
      if (plan == nullptr || queue == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "matmul execute plan or queue handle is stale or wrong kind");
      }
      if (plan->context != queue->context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
            "matmul plan and queue belong to different contexts");
      }
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      if (plan->context->poisoned.load() || plan->context->release_active ||
          queue->release_active || plan->input0->release_active ||
          plan->input1->release_active || plan->output->release_active ||
          plan->release_active || plan->in_flight) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "matmul execute owner graph is releasing or already in flight");
      }
      const auto &metadata = plan->matmul_metadata;
      if (metadata.m == 0U || metadata.k == 0U || metadata.n == 0U ||
          metadata.m > SLLM_HIP_MATMUL_MAX_M ||
          metadata.k > SLLM_HIP_MATMUL_MAX_K ||
          metadata.n > SLLM_HIP_MATMUL_MAX_N ||
          metadata.output_elements == 0U ||
          metadata.output_elements > SLLM_HIP_MATMUL_MAX_OUTPUT_ELEMENTS ||
          metadata.output_elements > UINT32_MAX) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_UNSUPPORTED,
            "matmul shape exceeds the baseline kernel launch contract");
      }
      if (!sllm_public_runtime::AccountingState::reserve_rmsnorm_submission(
              plan->context->accounting, queue->accounting,
              plan->input0->accounting, plan->input1->accounting,
              plan->output->accounting)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "matmul submission accounting is exhausted");
      }
      plan->in_flight = true;
      dispatch_id = plan->context->next_dispatch_id;
      if (dispatch_id == 0U) {
        plan->in_flight = false;
        (void)sllm_public_runtime::AccountingState::rollback_rmsnorm_submission(
            plan->context->accounting, queue->accounting,
            plan->input0->accounting, plan->input1->accounting,
            plan->output->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "matmul context dispatch id is exhausted");
      }
      plan->context->next_dispatch_id =
          dispatch_id == std::numeric_limits<uint64_t>::max()
              ? 0U
              : dispatch_id + 1U;
      *completion_output = nullptr;
    }

    std::unique_ptr<Completion> candidate;
    NativeEventGuard event_guard;
    ElementwiseExecuteScopeGuard execute_guard(plan, queue, &candidate,
                                               &event_guard, error_sink);
    const sllm_status_t device_status =
        select_context_device(plan->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      if (!rollback_reserved_elementwise_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return device_status;
    }
    hipDeviceProp_t properties = {};
    const sllm_status_t property_status = get_device_properties(
        plan->context->device_index, &properties, error_sink);
    if (property_status != SLLM_STATUS_OK) {
      if (!rollback_reserved_elementwise_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return property_status;
    }
    char arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME] = {};
    if (!copy_property(arch_name, sizeof(arch_name), properties.gcnArchName,
                       sizeof(properties.gcnArchName))) {
      if (!rollback_reserved_elementwise_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
          "HIP gcnArchName is not NUL terminated or is too long");
    }
#if defined(SLLM_HIP_COMPILE_TARGET)
    if (std::strcmp(arch_name, SLLM_HIP_COMPILE_TARGET) != 0) {
      if (!rollback_reserved_elementwise_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
          "matmul device target does not match the compiled exact target");
    }
#endif

    try {
      candidate = std::make_unique<Completion>(
          plan->context, queue, plan->input0, 0U, false, std::vector<uint8_t>{},
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          nullptr, plan, plan->input0, plan->input1, plan->output);
      execute_guard.candidate_allocated();
    } catch (...) {
      if (!rollback_reserved_elementwise_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "matmul completion allocation failed before enqueue");
    }

    hipEvent_t native_event = nullptr;
    const hipError_t event_status = hipEventCreateWithFlags(&native_event, 0U);
    if (event_status != hipSuccess) {
      if (!rollback_reserved_elementwise_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return hip_failure(error_sink, event_status, "hipEventCreateWithFlags");
    }
    event_guard.adopt(plan->context, native_event);
    candidate->event = native_event;
    hipEvent_t timing_start_event = nullptr;
    const hipError_t timing_event_status =
        hipEventCreateWithFlags(&timing_start_event, 0U);
    if (timing_event_status != hipSuccess) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "matmul timing event creation failed before enqueue", error_sink);
    }
    candidate->timing_start_event = timing_start_event;

    uintptr_t token = 0U;
    try {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      token = register_handle(candidate.get(), HandleKind::Completion);
    } catch (...) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "matmul completion registry allocation failed before enqueue",
          error_sink);
    }
    if (token == 0U) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "matmul completion handle token allocation failed", error_sink);
    }
    event_guard.release();
    execute_guard.completion_registered(token);

    const hipError_t timing_record_status =
        hipEventRecord(candidate->timing_start_event, queue->stream);
    if (timing_record_status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, timing_record_status,
                                       "hipEventRecord timing start", queue,
                                       error_sink);
    }
    const auto byte_pointer = [](Buffer *const buffer,
                                 const uint64_t offset) -> void * {
      return static_cast<char *>(buffer->device_pointer) +
             static_cast<std::size_t>(offset);
    };
    const uint16_t *const activation =
        static_cast<const uint16_t *>(byte_pointer(
            plan->input0, plan->matmul_metadata.activation.byte_offset));
    const uint16_t *const weight = static_cast<const uint16_t *>(
        byte_pointer(plan->input1, plan->matmul_metadata.weight.byte_offset));
    uint16_t *const output = static_cast<uint16_t *>(
        byte_pointer(plan->output, plan->matmul_metadata.output.byte_offset));
    const hipError_t launch_status = ::sllm_matmul_kernel::launch(
        activation, weight, output, plan->matmul_metadata.m,
        plan->matmul_metadata.k, plan->matmul_metadata.n, queue->stream);
    if (launch_status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, launch_status,
                                       "matmul kernel launch", queue,
                                       error_sink);
    }
    const hipError_t record_status =
        hipEventRecord(candidate->event, queue->stream);
    if (record_status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, record_status,
                                       "hipEventRecord", queue, error_sink);
    }
    initialize_matmul_dispatch_info(dispatch_info, dispatch_id,
                                    plan->matmul_metadata, arch_name);
    *completion_output = reinterpret_cast<sllm_completion_t *>(token);
    (void)candidate.release();
    execute_guard.disarm();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in matmul execute");
  }
}
