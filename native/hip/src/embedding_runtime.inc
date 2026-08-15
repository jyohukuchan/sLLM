extern "C" sllm_status_t sllm_embedding_prepare(
    const sllm_context_t *const raw_context,
    const sllm_embedding_desc_t *const descriptor,
    sllm_embedding_plan_t **const raw_plan,
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
          "embedding context or plan output is null");
    }
    const sllm_status_t prefix_status =
        sllm_embedding::validate_descriptor_prefix(descriptor, error_sink);
    if (prefix_status != SLLM_STATUS_OK) {
      return prefix_status;
    }
    const sllm_embedding_desc_t descriptor_copy = *descriptor;
    sllm_embedding::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_embedding::validate_and_copy_descriptor(
            &descriptor_copy, &metadata, error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }

    std::unique_ptr<ElementwisePlan> candidate;
    uintptr_t token = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      Context *const context =
          lookup<Context>(raw_context, HandleKind::Context);
      Buffer *const weight =
          lookup<Buffer>(descriptor_copy.weight.buffer, HandleKind::Buffer);
      Buffer *const token_ids = lookup<Buffer>(descriptor_copy.token_ids.buffer,
                                               HandleKind::Buffer);
      Buffer *const output =
          lookup<Buffer>(descriptor_copy.output.buffer, HandleKind::Buffer);
      if (context == nullptr || weight == nullptr || token_ids == nullptr ||
          output == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "embedding context or buffer handle is stale or wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (context->poisoned.load() || context->release_active ||
          weight->release_active || token_ids->release_active ||
          output->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "embedding cannot prepare while a context or buffer is releasing");
      }
      if (weight->context != context || token_ids->context != context ||
          output->context != context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH,
            "embedding buffers must belong to the supplied context and device");
      }
      if (metadata.weight.end_offset > weight->size_bytes ||
          metadata.token_ids.end_offset > token_ids->size_bytes ||
          metadata.output.end_offset > output->size_bytes) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
            "embedding tensor interval exceeds its backing buffer");
      }
      const auto overlaps_if_same = [](const Buffer *const left_buffer,
                                       const sllm_embedding::TensorMetadata &left,
                                       const Buffer *const right_buffer,
                                       const sllm_embedding::TensorMetadata &right) {
        return left_buffer == right_buffer &&
               sllm_embedding::intervals_overlap(left, right);
      };
      if (overlaps_if_same(weight, metadata.weight, token_ids,
                           metadata.token_ids) ||
          overlaps_if_same(weight, metadata.weight, output, metadata.output) ||
          overlaps_if_same(token_ids, metadata.token_ids, output,
                           metadata.output)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_ALIAS_OVERLAP,
            "embedding tensor intervals overlap within one backing buffer");
      }
      if (!sllm_public_runtime::AccountingState::reserve_prepared_plan(
              context->accounting, weight->accounting, token_ids->accounting,
              output->accounting)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "embedding prepared-plan accounting is exhausted");
      }
      try {
        candidate = std::make_unique<ElementwisePlan>(
            context, weight, token_ids, output, metadata);
        token = register_handle(candidate.get(), HandleKind::EmbeddingPlan);
      } catch (...) {
        (void)sllm_public_runtime::AccountingState::release_prepared_plan(
            context->accounting, weight->accounting, token_ids->accounting,
            output->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "embedding prepared-plan allocation or registration failed");
      }
      if (token == 0U) {
        (void)sllm_public_runtime::AccountingState::release_prepared_plan(
            context->accounting, weight->accounting, token_ids->accounting,
            output->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "embedding prepared-plan handle allocation failed");
      }
    }
    *raw_plan = reinterpret_cast<sllm_embedding_plan_t *>(token);
    (void)candidate.release();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in embedding prepare");
  }
}

extern "C" sllm_status_t sllm_embedding_plan_release(
    sllm_embedding_plan_t **const raw_plan,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_plan == nullptr || *raw_plan == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "embedding plan handle is null");
    }
    ElementwisePlan *plan = nullptr;
    bool quarantine_plan = false;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup<ElementwisePlan>(*raw_plan, HandleKind::EmbeddingPlan);
      if (plan == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "embedding plan handle is stale or has the wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      if (plan->release_active || plan->in_flight) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "embedding plan is releasing or has an in-flight completion");
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
          "embedding prepared-plan accounting release failed; plan quarantined");
    }
    delete plan;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in embedding plan release");
  }
}

extern "C" sllm_status_t sllm_embedding_execute(
    const sllm_embedding_plan_t *const raw_plan,
    const sllm_queue_t *const raw_queue,
    sllm_completion_t **const completion_output,
    sllm_embedding_dispatch_info_t *const dispatch_info,
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
          "embedding execute plan, queue, completion, or dispatch info is null");
    }
    const sllm_status_t dispatch_status =
        validate_embedding_dispatch_info(dispatch_info, error_sink);
    if (dispatch_status != SLLM_STATUS_OK) {
      return dispatch_status;
    }

    ElementwisePlan *plan = nullptr;
    Queue *queue = nullptr;
    uint64_t dispatch_id = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup<ElementwisePlan>(raw_plan, HandleKind::EmbeddingPlan);
      queue = lookup<Queue>(raw_queue, HandleKind::Queue);
      if (plan == nullptr || queue == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "embedding execute plan or queue handle is stale or wrong kind");
      }
      if (plan->context != queue->context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
            "embedding plan and queue belong to different contexts");
      }
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      if (plan->context->poisoned.load() || plan->context->release_active ||
          queue->release_active || plan->input0->release_active ||
          plan->input1->release_active || plan->output->release_active ||
          plan->release_active || plan->in_flight) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "embedding execute owner graph is releasing or already in flight");
      }
      const auto &metadata = plan->embedding_metadata;
      if (metadata.vocab_size > SLLM_HIP_EMBEDDING_MAX_VOCAB ||
          metadata.hidden_size > SLLM_HIP_EMBEDDING_MAX_HIDDEN ||
          metadata.token_count > SLLM_HIP_EMBEDDING_MAX_TOKENS ||
          metadata.output_elements == 0U ||
          metadata.output_elements > UINT32_MAX) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_UNSUPPORTED,
            "embedding shape exceeds the baseline kernel launch contract");
      }
      if (!sllm_public_runtime::AccountingState::reserve_rmsnorm_submission(
              plan->context->accounting, queue->accounting,
              plan->input0->accounting, plan->input1->accounting,
              plan->output->accounting)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "embedding submission accounting is exhausted");
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
            "embedding context dispatch id is exhausted");
      }
      plan->context->next_dispatch_id =
          dispatch_id == std::numeric_limits<uint64_t>::max()
              ? 0U
              : dispatch_id + 1U;
      *completion_output = nullptr;
    }

    ElementwiseExecuteScopeGuard execute_guard(plan, queue, nullptr, nullptr,
                                               error_sink);
    const sllm_status_t device_status =
        select_context_device(plan->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      execute_guard.disarm();
      (void)rollback_reserved_elementwise_submission(plan, queue, error_sink);
      return device_status;
    }
    hipDeviceProp_t properties = {};
    const sllm_status_t property_status = get_device_properties(
        plan->context->device_index, &properties, error_sink);
    if (property_status != SLLM_STATUS_OK) {
      execute_guard.disarm();
      (void)rollback_reserved_elementwise_submission(plan, queue, error_sink);
      return property_status;
    }
    char arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME] = {};
    if (!copy_property(arch_name, sizeof(arch_name), properties.gcnArchName,
                       sizeof(properties.gcnArchName))) {
      execute_guard.disarm();
      (void)rollback_reserved_elementwise_submission(plan, queue, error_sink);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
          "HIP gcnArchName is not NUL terminated or is too long");
    }
#if defined(SLLM_HIP_COMPILE_TARGET)
    if (!matches_runtime_gcn_arch(arch_name, SLLM_HIP_COMPILE_TARGET)) {
      execute_guard.disarm();
      (void)rollback_reserved_elementwise_submission(plan, queue, error_sink);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
          "embedding device target does not match the compiled exact target");
    }
#endif

    const auto byte_pointer = [](Buffer *const buffer,
                                 const uint64_t offset) -> void * {
      return static_cast<char *>(buffer->device_pointer) +
             static_cast<std::size_t>(offset);
    };
    const int32_t *const token_ids = static_cast<const int32_t *>(
        byte_pointer(plan->input1,
                     plan->embedding_metadata.token_ids.byte_offset));
    std::vector<int32_t> host_token_ids(
        static_cast<std::size_t>(plan->embedding_metadata.token_count));
    const hipError_t id_copy_status = hipMemcpyAsync(
        host_token_ids.data(), token_ids,
        host_token_ids.size() * sizeof(int32_t), hipMemcpyDeviceToHost,
        queue->stream);
    if (id_copy_status != hipSuccess) {
      execute_guard.disarm();
      (void)rollback_reserved_elementwise_submission(plan, queue, error_sink);
      return hip_failure(error_sink, id_copy_status,
                         "embedding token ID validation copy");
    }
    const hipError_t id_sync_status = hipStreamSynchronize(queue->stream);
    if (id_sync_status != hipSuccess) {
      execute_guard.disarm();
      (void)rollback_reserved_elementwise_submission(plan, queue, error_sink);
      return hip_failure(error_sink, id_sync_status,
                         "embedding token ID validation synchronize");
    }
    for (const int32_t token_id : host_token_ids) {
      if (token_id < 0 ||
          static_cast<uint64_t>(token_id) >=
              plan->embedding_metadata.vocab_size) {
        execute_guard.disarm();
        (void)rollback_reserved_elementwise_submission(plan, queue, error_sink);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_TOKEN_ID_OUT_OF_RANGE,
            "embedding token ID is negative or outside the vocabulary");
      }
    }

    std::unique_ptr<Completion> candidate;
    NativeEventGuard event_guard;
    execute_guard.disarm();
    ElementwiseExecuteScopeGuard publish_guard(plan, queue, &candidate,
                                               &event_guard, error_sink);
    try {
      candidate = std::make_unique<Completion>(
          plan->context, queue, plan->input0, 0U, false,
          std::vector<uint8_t>{}, nullptr, nullptr, nullptr, nullptr, plan,
          plan->input0, plan->input1, plan->output);
      publish_guard.candidate_allocated();
    } catch (...) {
      publish_guard.disarm();
      (void)rollback_reserved_elementwise_submission(plan, queue, error_sink);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "embedding completion allocation failed before enqueue");
    }
    hipEvent_t native_event = nullptr;
    hipError_t status = hipEventCreateWithFlags(&native_event, 0U);
    if (status != hipSuccess) {
      publish_guard.disarm();
      (void)rollback_reserved_elementwise_submission(plan, queue, error_sink);
      return hip_failure(error_sink, status, "hipEventCreateWithFlags");
    }
    event_guard.adopt(plan->context, native_event);
    candidate->event = native_event;
    hipEvent_t timing_start_event = nullptr;
    status = hipEventCreateWithFlags(&timing_start_event, 0U);
    if (status != hipSuccess) {
      publish_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "embedding timing event creation failed before enqueue", error_sink);
    }
    candidate->timing_start_event = timing_start_event;
    uintptr_t token = 0U;
    try {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      token = register_handle(candidate.get(), HandleKind::Completion);
    } catch (...) {
      publish_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "embedding completion registry allocation failed", error_sink);
    }
    if (token == 0U) {
      publish_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "embedding completion handle allocation failed", error_sink);
    }
    event_guard.release();
    publish_guard.completion_registered(token);
    status = hipEventRecord(candidate->timing_start_event, queue->stream);
    if (status != hipSuccess) {
      publish_guard.disarm();
      return cleanup_failed_submission(candidate, token, status,
                                       "hipEventRecord timing start", queue,
                                       error_sink);
    }
    const uint16_t *const weight = static_cast<const uint16_t *>(
        byte_pointer(plan->input0,
                     plan->embedding_metadata.weight.byte_offset));
    uint16_t *const output = static_cast<uint16_t *>(byte_pointer(
        plan->output, plan->embedding_metadata.output.byte_offset));
    status = ::sllm_embedding_kernel::launch_gather(
        weight, token_ids, output, plan->embedding_metadata.token_count,
        plan->embedding_metadata.hidden_size, queue->stream);
    if (status != hipSuccess) {
      publish_guard.disarm();
      return cleanup_failed_submission(candidate, token, status,
                                       "embedding kernel launch", queue,
                                       error_sink);
    }
    status = hipEventRecord(candidate->event, queue->stream);
    if (status != hipSuccess) {
      publish_guard.disarm();
      return cleanup_failed_submission(candidate, token, status,
                                       "hipEventRecord", queue, error_sink);
    }
    initialize_embedding_dispatch_info(dispatch_info, dispatch_id,
                                       plan->embedding_metadata, arch_name);
    *completion_output = reinterpret_cast<sllm_completion_t *>(token);
    (void)candidate.release();
    publish_guard.disarm();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in embedding execute");
  }
}
