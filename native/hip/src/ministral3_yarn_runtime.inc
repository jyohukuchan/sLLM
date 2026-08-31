extern "C" sllm_status_t sllm_ministral3_yarn_prepare(
    const sllm_context_t *const raw_context,
    const sllm_ministral3_yarn_desc_t *const descriptor,
    sllm_ministral3_yarn_plan_t **const raw_plan,
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
          "Ministral3 YaRN context or plan output is null");
    }
    const sllm_status_t prefix_status =
        sllm_ministral3_yarn::validate_descriptor_prefix(descriptor,
                                                         error_sink);
    if (prefix_status != SLLM_STATUS_OK) {
      return prefix_status;
    }
    const sllm_ministral3_yarn_desc_t descriptor_copy = *descriptor;
    sllm_ministral3_yarn::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_ministral3_yarn::validate_and_copy_descriptor(
            &descriptor_copy, &metadata, error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }

    AttentionBuffers resolved = {nullptr, nullptr, nullptr, nullptr,
                                 nullptr, nullptr, nullptr, nullptr};
    const std::array<const sllm_rotary::TensorMetadata *, 5U> tensor_metadata =
        {&metadata.query, &metadata.key, &metadata.positions,
         &metadata.query_output, &metadata.key_output};
    std::unique_ptr<Ministral3YarnPlan> candidate;
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
      resolved[4] =
          lookup<Buffer>(descriptor_copy.key_output.buffer, HandleKind::Buffer);
      if (context == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "Ministral3 YaRN context handle is stale or wrong kind");
      }
      for (std::size_t index = 0U; index != tensor_metadata.size(); ++index) {
        if (resolved[index] == nullptr) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
              "Ministral3 YaRN buffer handle is stale or wrong kind");
        }
      }
      for (std::size_t left = 0U; left != tensor_metadata.size(); ++left) {
        for (std::size_t right = left + 1U; right != tensor_metadata.size();
             ++right) {
          if (resolved[left] == resolved[right] &&
              sllm_ministral3_yarn::intervals_overlap(
                  *tensor_metadata[left], *tensor_metadata[right])) {
            return sllm_public_runtime::write_error(
                error_sink, SLLM_STATUS_ALIAS_OVERLAP,
                "Ministral3 YaRN tensor byte ranges overlap");
          }
        }
      }
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (context->poisoned.load() || context->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "Ministral3 YaRN cannot prepare while context is releasing");
      }
      const std::array<uint64_t, 5U> end_offsets = {
          metadata.query.end_offset, metadata.key.end_offset,
          metadata.positions.end_offset, metadata.query_output.end_offset,
          metadata.key_output.end_offset};
      for (std::size_t index = 0U; index != tensor_metadata.size(); ++index) {
        if (resolved[index]->release_active) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_PUBLIC_BUSY,
              "Ministral3 YaRN cannot prepare while a buffer is releasing");
        }
        if (resolved[index]->context != context) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH,
              "Ministral3 YaRN buffers belong to another context");
        }
        if (end_offsets[index] > resolved[index]->size_bytes) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
              "Ministral3 YaRN tensor interval exceeds its buffer");
        }
      }
      resolved[5] = resolved[0];
      resolved[6] = resolved[1];
      resolved[7] = resolved[2];
      if (!reserve_attention_prepared_plan(context->accounting, resolved)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "Ministral3 YaRN prepared-plan accounting is exhausted");
      }
      try {
        candidate =
            std::make_unique<Ministral3YarnPlan>(context, resolved, metadata);
        token =
            register_handle(candidate.get(), HandleKind::Ministral3YarnPlan);
      } catch (...) {
        (void)release_attention_prepared_plan(context->accounting, resolved);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "Ministral3 YaRN plan allocation or registration failed");
      }
      if (token == 0U) {
        (void)release_attention_prepared_plan(context->accounting, resolved);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "Ministral3 YaRN plan handle allocation failed");
      }
    }
    *raw_plan = reinterpret_cast<sllm_ministral3_yarn_plan_t *>(token);
    (void)candidate.release();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in Ministral3 YaRN prepare");
  }
}

extern "C" sllm_status_t sllm_ministral3_yarn_plan_release(
    sllm_ministral3_yarn_plan_t **const raw_plan,
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
          "Ministral3 YaRN plan handle is null");
    }
    Ministral3YarnPlan *plan = nullptr;
    bool quarantine_plan = false;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan =
          lookup<Ministral3YarnPlan>(*raw_plan, HandleKind::Ministral3YarnPlan);
      if (plan == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "Ministral3 YaRN plan handle is stale or wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      if (plan->release_active || plan->in_flight) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "Ministral3 YaRN plan is releasing or in flight");
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
          "Ministral3 YaRN accounting release failed; plan quarantined");
    }
    delete plan;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in Ministral3 YaRN plan release");
  }
}

extern "C" sllm_status_t sllm_ministral3_yarn_execute(
    const sllm_ministral3_yarn_plan_t *const raw_plan,
    const sllm_queue_t *const raw_queue,
    sllm_completion_t **const completion_output,
    sllm_ministral3_yarn_dispatch_info_t *const dispatch_info,
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
          "Ministral3 YaRN execute plan, queue, completion, or info is null");
    }
    const sllm_status_t dispatch_status =
        validate_ministral3_yarn_dispatch_info(dispatch_info, error_sink);
    if (dispatch_status != SLLM_STATUS_OK) {
      return dispatch_status;
    }
    Ministral3YarnPlan *plan = nullptr;
    Queue *queue = nullptr;
    uint64_t dispatch_id = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan =
          lookup<Ministral3YarnPlan>(raw_plan, HandleKind::Ministral3YarnPlan);
      queue = lookup<Queue>(raw_queue, HandleKind::Queue);
      if (plan == nullptr || queue == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "Ministral3 YaRN plan or queue handle is stale or wrong kind");
      }
      if (plan->context != queue->context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
            "Ministral3 YaRN plan and queue belong to different contexts");
      }
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      if (plan->context->poisoned.load() || plan->context->release_active ||
          queue->release_active || plan->release_active || plan->in_flight) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "Ministral3 YaRN execute owner graph is busy");
      }
      if (!reserve_attention_submission(plan->context->accounting,
                                        queue->accounting, plan->buffers)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "Ministral3 YaRN submission accounting is exhausted");
      }
      plan->in_flight = true;
      dispatch_id = plan->context->next_dispatch_id;
      if (dispatch_id == 0U) {
        plan->in_flight = false;
        (void)rollback_attention_submission(plan->context->accounting,
                                            queue->accounting, plan->buffers);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "Ministral3 YaRN context dispatch id is exhausted");
      }
      plan->context->next_dispatch_id =
          dispatch_id == std::numeric_limits<uint64_t>::max()
              ? 0U
              : dispatch_id + 1U;
      *completion_output = nullptr;
    }

    std::unique_ptr<Completion> candidate;
    NativeEventGuard event_guard;
    ArrayOperationExecuteScopeGuard execute_guard(plan, queue, &candidate,
                                                  &event_guard, error_sink);
    const sllm_status_t device_status =
        select_context_device(plan->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      return device_status;
    }
    hipDeviceProp_t properties = {};
    const sllm_status_t property_status =
        get_context_device_properties(plan->context, &properties, error_sink);
    if (property_status != SLLM_STATUS_OK) {
      return property_status;
    }
    char arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME] = {};
    if (!copy_property(arch_name, sizeof(arch_name), properties.gcnArchName,
                       sizeof(properties.gcnArchName))) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
          "HIP gcnArchName is not NUL terminated or too long");
    }
#if defined(SLLM_HIP_COMPILE_TARGET)
    if (!matches_runtime_gcn_arch(arch_name, SLLM_HIP_COMPILE_TARGET)) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
          "Ministral3 YaRN device target does not match compiled target");
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
    hipError_t status = hipMemcpyAsync(host_positions.data(), positions,
                                       host_positions.size() * sizeof(int32_t),
                                       hipMemcpyDeviceToHost, queue->stream);
    if (status != hipSuccess) {
      return hip_failure(error_sink, status,
                         "Ministral3 YaRN position validation copy");
    }
    status = hipStreamSynchronize(queue->stream);
    if (status != hipSuccess) {
      return hip_failure(error_sink, status,
                         "Ministral3 YaRN position validation synchronize");
    }
    for (std::size_t index = 0U; index != host_positions.size(); ++index) {
      const int64_t expected =
          static_cast<int64_t>(plan->metadata.start_position) +
          static_cast<int64_t>(index);
      const bool valid =
          plan->metadata.position_payload_mode ==
                  SLLM_HIP_POSITION_PAYLOAD_MODE_EXPLICIT_V1
              ? host_positions[index] >= 0 &&
                    static_cast<uint64_t>(host_positions[index]) <
                        static_cast<uint64_t>(
                            SLLM_HIP_MINISTRAL3_YARN_MAX_POSITION)
              : host_positions[index] == static_cast<int32_t>(expected);
      if (!valid) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_POSITION_PAYLOAD_MISMATCH,
            "Ministral3 YaRN position payload is invalid");
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
          "Ministral3 YaRN completion allocation failed");
    }
    hipEvent_t native_event = nullptr;
    status = queue_profiles_completions(queue)
                 ? hipEventCreateWithFlags(&native_event, 0U)
                 : hipSuccess;
    if (status != hipSuccess) {
      return hip_failure(error_sink, status, "hipEventCreateWithFlags");
    }
    event_guard.adopt(plan->context, native_event);
    candidate->event = native_event;
    hipEvent_t timing_start_event = nullptr;
    status = queue_profiles_completions(queue)
                 ? hipEventCreateWithFlags(&timing_start_event, 0U)
                 : hipSuccess;
    if (status != hipSuccess) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "Ministral3 YaRN timing event creation failed", error_sink);
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
          "Ministral3 YaRN completion registry allocation failed", error_sink);
    }
    if (token == 0U) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "Ministral3 YaRN completion handle allocation failed", error_sink);
    }
    if (candidate->event != nullptr) {
      event_guard.release();
    }
    execute_guard.completion_registered(token);
    status = candidate->timing_start_event != nullptr
                 ? hipEventRecord(candidate->timing_start_event, queue->stream)
                 : hipSuccess;
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
    status = ::sllm_ministral3_yarn_kernel::launch(
        query, key, positions, query_output, key_output,
        static_cast<uint32_t>(plan->metadata.token_count),
        SLLM_HIP_MINISTRAL3_YARN_Q_HEADS, SLLM_HIP_MINISTRAL3_YARN_KV_HEADS,
        queue->stream);
    if (status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, status,
                                       "Ministral3 YaRN kernel launch", queue,
                                       error_sink);
    }
    status = candidate->event != nullptr
                 ? hipEventRecord(candidate->event, queue->stream)
                 : hipSuccess;
    if (status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, status,
                                       "hipEventRecord", queue, error_sink);
    }
    initialize_ministral3_yarn_dispatch_info(dispatch_info, dispatch_id,
                                             plan->metadata, arch_name);
    *completion_output = reinterpret_cast<sllm_completion_t *>(token);
    (void)candidate.release();
    execute_guard.disarm();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in Ministral3 YaRN execute");
  }
}
