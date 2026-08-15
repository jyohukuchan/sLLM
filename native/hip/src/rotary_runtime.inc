extern "C" sllm_status_t sllm_rotary_prepare(
    const sllm_context_t *const raw_context,
    const sllm_rotary_desc_t *const descriptor,
    sllm_rotary_plan_t **const raw_plan,
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
          "rotary context or plan output is null");
    }
    const sllm_status_t prefix_status =
        sllm_rotary::validate_descriptor_prefix(descriptor, error_sink);
    if (prefix_status != SLLM_STATUS_OK) {
      return prefix_status;
    }
    const sllm_rotary_desc_t descriptor_copy = *descriptor;
    sllm_rotary::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_rotary::validate_and_copy_descriptor(&descriptor_copy, &metadata,
                                                  error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }

    RotaryBuffers resolved = {nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr};
    const std::array<const sllm_rotary::TensorMetadata *, 5> tensor_metadata = {
        &metadata.query, &metadata.key, &metadata.positions,
        &metadata.query_output, &metadata.key_output};
    std::unique_ptr<RotaryPlan> candidate;
    uintptr_t token = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      Context *const context =
          lookup<Context>(raw_context, HandleKind::Context);
      resolved[0] =
          lookup<Buffer>(descriptor_copy.query.buffer, HandleKind::Buffer);
      resolved[1] =
          lookup<Buffer>(descriptor_copy.key.buffer, HandleKind::Buffer);
      resolved[2] =
          lookup<Buffer>(descriptor_copy.positions.buffer, HandleKind::Buffer);
      resolved[3] = lookup<Buffer>(descriptor_copy.query_output.buffer,
                                   HandleKind::Buffer);
      resolved[4] = lookup<Buffer>(descriptor_copy.key_output.buffer,
                                   HandleKind::Buffer);
      if (context == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "rotary context handle is stale or has the wrong kind");
      }
      for (std::size_t index = 0U; index != tensor_metadata.size(); ++index) {
        if (resolved[index] == nullptr) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
              "rotary buffer handle is stale or has the wrong kind");
        }
      }
      for (std::size_t left = 0U; left != tensor_metadata.size(); ++left) {
        for (std::size_t right = left + 1U; right != tensor_metadata.size();
             ++right) {
          if (resolved[left] == resolved[right] &&
              sllm_rotary::intervals_overlap(*tensor_metadata[left],
                                             *tensor_metadata[right])) {
            return sllm_public_runtime::write_error(
                error_sink, SLLM_STATUS_ALIAS_OVERLAP,
                "rotary tensor byte ranges overlap");
          }
        }
      }
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (context->poisoned.load() || context->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "rotary cannot prepare while the context is releasing");
      }
      const std::array<uint64_t, 5> end_offsets = {
          metadata.query.end_offset, metadata.key.end_offset,
          metadata.positions.end_offset, metadata.query_output.end_offset,
          metadata.key_output.end_offset};
      for (std::size_t index = 0U; index != tensor_metadata.size(); ++index) {
        if (resolved[index]->release_active) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_PUBLIC_BUSY,
              "rotary cannot prepare while a buffer is releasing");
        }
        if (resolved[index]->context != context) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH,
              "rotary buffers must belong to the supplied context");
        }
        if (end_offsets[index] > resolved[index]->size_bytes) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
              "rotary tensor interval exceeds its backing buffer");
        }
      }
      /* The accounting helper uses an eight-entry retained-buffer envelope.
       * Duplicate padding does not add references because it counts each
       * unique buffer identity once. */
      resolved[5] = resolved[0];
      resolved[6] = resolved[1];
      resolved[7] = resolved[2];
      if (!reserve_attention_prepared_plan(context->accounting, resolved)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "rotary prepared-plan accounting is exhausted");
      }
      try {
        candidate = std::make_unique<RotaryPlan>(context, resolved, metadata);
        token = register_handle(candidate.get(), HandleKind::RotaryPlan);
      } catch (...) {
        (void)release_attention_prepared_plan(context->accounting, resolved);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "rotary prepared-plan allocation or registration failed");
      }
      if (token == 0U) {
        (void)release_attention_prepared_plan(context->accounting, resolved);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "rotary prepared-plan handle allocation failed");
      }
    }
    *raw_plan = reinterpret_cast<sllm_rotary_plan_t *>(token);
    (void)candidate.release();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in rotary prepare");
  }
}

extern "C" sllm_status_t
sllm_rotary_plan_release(sllm_rotary_plan_t **const raw_plan,
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
          "rotary plan handle is null");
    }
    RotaryPlan *plan = nullptr;
    bool quarantine_plan = false;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup<RotaryPlan>(*raw_plan, HandleKind::RotaryPlan);
      if (plan == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "rotary plan handle is stale or has the wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      if (plan->release_active || plan->in_flight) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "rotary plan is releasing or has an in-flight completion");
      }
      plan->release_active = true;
      const bool accounting_released =
          !sllm_public_runtime::FaultInjector::consume(
              sllm_public_runtime::FaultPoint::AccountingFailure) &&
          release_attention_prepared_plan(plan->context->accounting,
                                          plan->buffers);
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
          "rotary prepared-plan accounting release failed; plan quarantined");
    }
    delete plan;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in rotary plan release");
  }
}

extern "C" sllm_status_t sllm_rotary_execute(
    const sllm_rotary_plan_t *const raw_plan,
    const sllm_queue_t *const raw_queue,
    sllm_completion_t **const completion_output,
    sllm_rotary_dispatch_info_t *const dispatch_info,
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
          "rotary execute plan, queue, completion, or dispatch info is null");
    }
    const sllm_status_t dispatch_status =
        validate_rotary_dispatch_info(dispatch_info, error_sink);
    if (dispatch_status != SLLM_STATUS_OK) {
      return dispatch_status;
    }

    RotaryPlan *plan = nullptr;
    Queue *queue = nullptr;
    uint64_t dispatch_id = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup<RotaryPlan>(raw_plan, HandleKind::RotaryPlan);
      queue = lookup<Queue>(raw_queue, HandleKind::Queue);
      if (plan == nullptr || queue == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "rotary plan or queue handle is stale or has the wrong kind");
      }
      if (plan->context != queue->context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
            "rotary plan and queue belong to different contexts");
      }
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      if (plan->context->poisoned.load() || plan->context->release_active ||
          queue->release_active || plan->release_active || plan->in_flight) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "rotary execute owner graph is releasing or already in flight");
      }
      if (!reserve_attention_submission(plan->context->accounting,
                                        queue->accounting, plan->buffers)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "rotary submission accounting is exhausted");
      }
      plan->in_flight = true;
      dispatch_id = plan->context->next_dispatch_id;
      if (dispatch_id == 0U) {
        plan->in_flight = false;
        (void)rollback_attention_submission(plan->context->accounting,
                                            queue->accounting, plan->buffers);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "rotary context dispatch id is exhausted");
      }
      plan->context->next_dispatch_id =
          dispatch_id == std::numeric_limits<uint64_t>::max()
              ? 0U
              : dispatch_id + 1U;
      *completion_output = nullptr;
    }

    std::unique_ptr<Completion> candidate;
    NativeEventGuard event_guard;
    ArrayOperationExecuteScopeGuard execute_guard(
        plan, queue, &candidate, &event_guard, error_sink);

    const sllm_status_t device_status =
        select_context_device(plan->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      return device_status;
    }
    hipDeviceProp_t properties = {};
    const sllm_status_t property_status = get_device_properties(
        plan->context->device_index, &properties, error_sink);
    if (property_status != SLLM_STATUS_OK) {
      return property_status;
    }
    char arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME] = {};
    if (!copy_property(arch_name, sizeof(arch_name), properties.gcnArchName,
                       sizeof(properties.gcnArchName))) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
          "HIP gcnArchName is not NUL terminated or is too long");
    }
#if defined(SLLM_HIP_COMPILE_TARGET)
    if (!matches_runtime_gcn_arch(arch_name, SLLM_HIP_COMPILE_TARGET)) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
          "rotary device target does not match the compiled exact target");
    }
#endif

    const auto byte_pointer = [](Buffer *const buffer,
                                 const uint64_t offset) -> void * {
      return static_cast<char *>(buffer->device_pointer) +
             static_cast<std::size_t>(offset);
    };
    const int32_t *const positions = static_cast<const int32_t *>(
        byte_pointer(plan->buffers[2], plan->metadata.positions.byte_offset));
    std::vector<int32_t> host_positions(
        static_cast<std::size_t>(plan->metadata.token_count));
    hipError_t status = hipMemcpyAsync(
        host_positions.data(), positions,
        host_positions.size() * sizeof(int32_t), hipMemcpyDeviceToHost,
        queue->stream);
    if (status != hipSuccess) {
      return hip_failure(error_sink, status,
                         "rotary position validation copy");
    }
    status = hipStreamSynchronize(queue->stream);
    if (status != hipSuccess) {
      return hip_failure(error_sink, status,
                         "rotary position validation synchronize");
    }
    for (std::size_t index = 0U; index != host_positions.size(); ++index) {
      const int64_t expected =
          static_cast<int64_t>(plan->metadata.start_position) +
          static_cast<int64_t>(index);
      if (host_positions[index] != static_cast<int32_t>(expected)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_POSITION_PAYLOAD_MISMATCH,
            "rotary position payload does not equal the requested contiguous "
            "range");
      }
    }

    try {
      candidate = std::make_unique<Completion>(
          plan->context, queue, plan->buffers[0], 0U, false,
          std::vector<uint8_t>{}, nullptr, nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, plan,
          plan->buffers);
      execute_guard.candidate_allocated();
    } catch (...) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "rotary completion allocation failed before enqueue");
    }
    hipEvent_t native_event = nullptr;
    status = hipEventCreateWithFlags(&native_event, 0U);
    if (status != hipSuccess) {
      return hip_failure(error_sink, status, "hipEventCreateWithFlags");
    }
    event_guard.adopt(plan->context, native_event);
    candidate->event = native_event;
    hipEvent_t timing_start_event = nullptr;
    status = hipEventCreateWithFlags(&timing_start_event, 0U);
    if (status != hipSuccess) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "rotary timing event creation failed before enqueue", error_sink);
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
          "rotary completion registry allocation failed", error_sink);
    }
    if (token == 0U) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "rotary completion handle allocation failed", error_sink);
    }
    event_guard.release();
    execute_guard.completion_registered(token);
    status = hipEventRecord(candidate->timing_start_event, queue->stream);
    if (status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, status,
                                       "hipEventRecord timing start", queue,
                                       error_sink);
    }

    const uint16_t *const query = static_cast<const uint16_t *>(
        byte_pointer(plan->buffers[0], plan->metadata.query.byte_offset));
    const uint16_t *const key = static_cast<const uint16_t *>(
        byte_pointer(plan->buffers[1], plan->metadata.key.byte_offset));
    uint16_t *const query_output = static_cast<uint16_t *>(byte_pointer(
        plan->buffers[3], plan->metadata.query_output.byte_offset));
    uint16_t *const key_output = static_cast<uint16_t *>(
        byte_pointer(plan->buffers[4], plan->metadata.key_output.byte_offset));
    float theta = 0.0F;
    std::memcpy(&theta, &plan->metadata.theta_bits, sizeof(theta));
    status = ::sllm_rotary_kernel::launch(
        query, key, positions, query_output, key_output,
        static_cast<uint32_t>(plan->metadata.token_count),
        plan->metadata.q_heads, plan->metadata.kv_heads,
        plan->metadata.head_dim, plan->metadata.rotary_dim, theta,
        queue->stream);
    if (status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, status,
                                       "rotary kernel launch", queue,
                                       error_sink);
    }
    status = hipEventRecord(candidate->event, queue->stream);
    if (status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, status,
                                       "hipEventRecord", queue, error_sink);
    }
    initialize_rotary_dispatch_info(dispatch_info, dispatch_id,
                                    plan->metadata, arch_name);
    *completion_output = reinterpret_cast<sllm_completion_t *>(token);
    (void)candidate.release();
    execute_guard.disarm();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in rotary execute");
  }
}
