extern "C" sllm_status_t
sllm_argmax_prepare(const sllm_context_t *const raw_context,
                    const sllm_argmax_desc_t *const descriptor,
                    sllm_argmax_plan_t **const raw_plan,
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
          "argmax context or plan output is null");
    }
    const sllm_status_t prefix_status =
        sllm_argmax::validate_descriptor_prefix(descriptor, error_sink);
    if (prefix_status != SLLM_STATUS_OK) {
      return prefix_status;
    }
    const sllm_argmax_desc_t descriptor_copy = *descriptor;
    sllm_argmax::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_argmax::validate_and_copy_descriptor(&descriptor_copy, &metadata,
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
      Buffer *const logits =
          lookup<Buffer>(descriptor_copy.logits.buffer, HandleKind::Buffer);
      Buffer *const output =
          lookup<Buffer>(descriptor_copy.output.buffer, HandleKind::Buffer);
      if (context == nullptr || logits == nullptr || output == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "argmax context or buffer handle is stale or wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (context->poisoned.load() || context->release_active ||
          logits->release_active || output->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "argmax cannot prepare while a context or buffer is releasing");
      }
      if (logits->context != context || output->context != context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH,
            "argmax buffers must belong to the supplied context and device");
      }
      if (metadata.logits.end_offset > logits->size_bytes ||
          metadata.output.end_offset > output->size_bytes) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
            "argmax tensor interval exceeds its backing buffer");
      }
      if (logits == output &&
          sllm_argmax::intervals_overlap(metadata.logits, metadata.output)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_ALIAS_OVERLAP,
            "argmax tensor intervals overlap within one backing buffer");
      }
      if (!sllm_public_runtime::AccountingState::
              reserve_two_buffer_prepared_plan(context->accounting,
                                               logits->accounting,
                                               output->accounting)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "argmax prepared-plan accounting is exhausted");
      }
      try {
        candidate = std::make_unique<ElementwisePlan>(context, logits, output,
                                                      metadata);
        token = register_handle(candidate.get(), HandleKind::ArgmaxPlan);
      } catch (...) {
        (void)sllm_public_runtime::AccountingState::
            release_two_buffer_prepared_plan(
                context->accounting, logits->accounting, output->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "argmax prepared-plan allocation or registration failed");
      }
      if (token == 0U) {
        (void)sllm_public_runtime::AccountingState::
            release_two_buffer_prepared_plan(
                context->accounting, logits->accounting, output->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "argmax prepared-plan handle allocation failed");
      }
    }
    *raw_plan = reinterpret_cast<sllm_argmax_plan_t *>(token);
    (void)candidate.release();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in argmax prepare");
  }
}

extern "C" sllm_status_t
sllm_argmax_plan_release(sllm_argmax_plan_t **const raw_plan,
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
                                              "argmax plan handle is null");
    }
    ElementwisePlan *plan = nullptr;
    bool quarantine_plan = false;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup<ElementwisePlan>(*raw_plan, HandleKind::ArgmaxPlan);
      if (plan == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "argmax plan handle is stale or has the wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      if (plan->release_active || plan->in_flight) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "argmax plan is releasing or has an in-flight completion");
      }
      plan->release_active = true;
      const bool accounting_released =
          !sllm_public_runtime::FaultInjector::consume(
              sllm_public_runtime::FaultPoint::AccountingFailure) &&
          sllm_public_runtime::AccountingState::
              release_two_buffer_prepared_plan(plan->context->accounting,
                                               plan->input0->accounting,
                                               plan->output->accounting);
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
          "argmax prepared-plan accounting release failed; plan quarantined");
    }
    delete plan;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in argmax plan release");
  }
}

extern "C" sllm_status_t
sllm_argmax_execute(const sllm_argmax_plan_t *const raw_plan,
                    const sllm_queue_t *const raw_queue,
                    sllm_completion_t **const completion_output,
                    sllm_argmax_dispatch_info_t *const dispatch_info,
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
          "argmax execute plan, queue, completion, or dispatch info is null");
    }
    const sllm_status_t dispatch_status =
        validate_argmax_dispatch_info(dispatch_info, error_sink);
    if (dispatch_status != SLLM_STATUS_OK) {
      return dispatch_status;
    }

    ElementwisePlan *plan = nullptr;
    Queue *queue = nullptr;
    uint64_t dispatch_id = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup<ElementwisePlan>(raw_plan, HandleKind::ArgmaxPlan);
      queue = lookup<Queue>(raw_queue, HandleKind::Queue);
      if (plan == nullptr || queue == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "argmax execute plan or queue handle is stale or wrong kind");
      }
      if (plan->context != queue->context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
            "argmax plan and queue belong to different contexts");
      }
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      if (plan->context->poisoned.load() || plan->context->release_active ||
          queue->release_active || plan->input0->release_active ||
          plan->output->release_active || plan->release_active ||
          plan->in_flight) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "argmax execute owner graph is releasing or already in flight");
      }
      const auto &metadata = plan->argmax_metadata;
      if (metadata.m == 0U || metadata.v == 0U ||
          metadata.m > SLLM_HIP_ARGMAX_MAX_M ||
          metadata.v > SLLM_HIP_ARGMAX_MAX_V) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_UNSUPPORTED,
            "argmax shape exceeds the baseline kernel launch contract");
      }
      if (!sllm_public_runtime::AccountingState::reserve_two_buffer_submission(
              plan->context->accounting, queue->accounting,
              plan->input0->accounting, plan->output->accounting)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "argmax submission accounting is exhausted");
      }
      plan->in_flight = true;
      dispatch_id = plan->context->next_dispatch_id;
      if (dispatch_id == 0U) {
        plan->in_flight = false;
        (void)sllm_public_runtime::AccountingState::
            rollback_two_buffer_submission(
                plan->context->accounting, queue->accounting,
                plan->input0->accounting, plan->output->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "argmax context dispatch id is exhausted");
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
                                               &event_guard, error_sink, true);
    const sllm_status_t device_status =
        select_context_device(plan->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      if (!rollback_reserved_argmax_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return device_status;
    }
    hipDeviceProp_t properties = {};
    const sllm_status_t property_status = get_context_device_properties(
        plan->context, &properties, error_sink);
    if (property_status != SLLM_STATUS_OK) {
      if (!rollback_reserved_argmax_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return property_status;
    }
    char arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME] = {};
    if (!copy_property(arch_name, sizeof(arch_name), properties.gcnArchName,
                       sizeof(properties.gcnArchName))) {
      if (!rollback_reserved_argmax_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
          "HIP gcnArchName is not NUL terminated or is too long");
    }
#if defined(SLLM_HIP_COMPILE_TARGET)
    if (!matches_runtime_gcn_arch(arch_name, SLLM_HIP_COMPILE_TARGET)) {
      if (!rollback_reserved_argmax_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
          "argmax device target does not match the compiled exact target");
    }
#endif

    try {
      candidate = std::make_unique<Completion>(
          plan->context, queue, plan->input0, plan, plan->input0, plan->output,
          ArgmaxCompletionTag{});
      execute_guard.candidate_allocated();
    } catch (...) {
      if (!rollback_reserved_argmax_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "argmax completion allocation failed before enqueue");
    }

    hipEvent_t native_event = nullptr;
    const hipError_t event_status = queue_profiles_completions(queue)
                                        ? hipEventCreateWithFlags(&native_event, 0U)
                                        : hipSuccess;
    if (event_status != hipSuccess) {
      if (!rollback_reserved_argmax_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return hip_failure(error_sink, event_status, "hipEventCreateWithFlags");
    }
    event_guard.adopt(plan->context, native_event);
    candidate->event = native_event;
    hipEvent_t timing_start_event = nullptr;
    const hipError_t timing_event_status = queue_profiles_completions(queue)
                                               ? hipEventCreateWithFlags(
                                                     &timing_start_event, 0U)
                                               : hipSuccess;
    if (timing_event_status != hipSuccess) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "argmax timing event creation failed before enqueue", error_sink);
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
          "argmax completion registry allocation failed before enqueue",
          error_sink);
    }
    if (token == 0U) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "argmax completion handle token allocation failed", error_sink);
    }
    if (candidate->event != nullptr) {
      event_guard.release();
    }
    execute_guard.completion_registered(token);
    const hipError_t timing_record_status =
        candidate->timing_start_event != nullptr
            ? hipEventRecord(candidate->timing_start_event, queue->stream)
            : hipSuccess;
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
    const uint16_t *const logits = static_cast<const uint16_t *>(
        byte_pointer(plan->input0, plan->argmax_metadata.logits.byte_offset));
    int32_t *const output = static_cast<int32_t *>(
        byte_pointer(plan->output, plan->argmax_metadata.output.byte_offset));
    const hipError_t launch_status =
        ::sllm_argmax_kernel::launch(logits, output, plan->argmax_metadata.m,
                                     plan->argmax_metadata.v, queue->stream);
    if (launch_status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, launch_status,
                                       "argmax kernel launch", queue,
                                       error_sink);
    }
    const hipError_t record_status =
        candidate->event != nullptr
            ? hipEventRecord(candidate->event, queue->stream)
            : hipSuccess;
    if (record_status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, record_status,
                                       "hipEventRecord", queue, error_sink);
    }
    initialize_argmax_dispatch_info(dispatch_info, dispatch_id,
                                    plan->argmax_metadata, arch_name);
    *completion_output = reinterpret_cast<sllm_completion_t *>(token);
    (void)candidate.release();
    execute_guard.disarm();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in argmax execute");
  }
}
