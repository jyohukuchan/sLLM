namespace {

bool use_rdna_wave32_attention_preprocess(
    const sllm_attention_preprocess::DescriptorMetadata &metadata,
    const char *const arch_name) noexcept {
  constexpr uint32_t kQwen27QHeads = 24U;
  const bool gfx1030 = std::strcmp(arch_name, "gfx1030") == 0;
  const bool gfx1201 = std::strcmp(arch_name, "gfx1201") == 0;
  const bool existing_gfx1030_shape =
      gfx1030 && metadata.q_heads == SLLM_HIP_ATTENTION_PREPROCESS_Q_HEADS;
  const bool qwen27_shape = metadata.q_heads == kQwen27QHeads;
  if ((!existing_gfx1030_shape && !qwen27_shape) || (!gfx1030 && !gfx1201)) {
    return false;
  }
  const char *const rollback_name =
      gfx1030 ? "SLLM_ATTENTION_PREPROCESS_GFX1030_WAVE32"
              : "SLLM_ATTENTION_PREPROCESS_GFX1201_WAVE32";
  const char *const opt_in = std::getenv(rollback_name);
  // Each exact RDNA target has an independent fail-closed rollback. An unset
  // variable selects the accepted default; explicit zero and unknown values
  // retain the scalar baseline.
  const bool enabled = opt_in == nullptr || std::strcmp(opt_in, "1") == 0;
  return enabled && metadata.k_heads == SLLM_HIP_ATTENTION_PREPROCESS_K_HEADS &&
         metadata.head_dim == SLLM_HIP_ATTENTION_PREPROCESS_Q_HEAD_DIM &&
         (metadata.position_components == 1U ||
          metadata.position_components == 3U);
}

} // namespace

extern "C" sllm_status_t sllm_attention_preprocess_prepare(
    const sllm_context_t *const raw_context,
    const sllm_attention_preprocess_desc_t *const descriptor,
    sllm_attention_preprocess_plan_t **const raw_plan,
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
          "attention preprocess context or plan output is null");
    }
    const sllm_status_t prefix_status =
        sllm_attention_preprocess::validate_descriptor_prefix(descriptor,
                                                              error_sink);
    if (prefix_status != SLLM_STATUS_OK) {
      return prefix_status;
    }
    const sllm_attention_preprocess_desc_t descriptor_copy = *descriptor;
    sllm_attention_preprocess::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_attention_preprocess::validate_and_copy_descriptor(
            &descriptor_copy, &metadata, error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }

    const AttentionBuffers buffers = {nullptr, nullptr, nullptr, nullptr,
                                      nullptr, nullptr, nullptr, nullptr};
    AttentionBuffers resolved = buffers;
    const std::array<const sllm_attention_preprocess::TensorMetadata *, 8>
        tensor_metadata = {&metadata.packed_q_gate, &metadata.k,
                           &metadata.q_raw_scale,   &metadata.k_raw_scale,
                           &metadata.positions,     &metadata.q_output,
                           &metadata.gate_output,   &metadata.k_output};
    std::unique_ptr<AttentionPreprocessPlan> candidate;
    uintptr_t token = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      Context *const context =
          lookup<Context>(raw_context, HandleKind::Context);
      resolved[0] = lookup<Buffer>(descriptor_copy.packed_q_gate.buffer,
                                   HandleKind::Buffer);
      resolved[1] =
          lookup<Buffer>(descriptor_copy.k.buffer, HandleKind::Buffer);
      resolved[2] = lookup<Buffer>(descriptor_copy.q_raw_scale.buffer,
                                   HandleKind::Buffer);
      resolved[3] = lookup<Buffer>(descriptor_copy.k_raw_scale.buffer,
                                   HandleKind::Buffer);
      resolved[4] =
          lookup<Buffer>(descriptor_copy.positions.buffer, HandleKind::Buffer);
      resolved[5] =
          lookup<Buffer>(descriptor_copy.q_output.buffer, HandleKind::Buffer);
      resolved[6] = lookup<Buffer>(descriptor_copy.gate_output.buffer,
                                   HandleKind::Buffer);
      resolved[7] =
          lookup<Buffer>(descriptor_copy.k_output.buffer, HandleKind::Buffer);
      if (context == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "attention preprocess context handle is stale or has the wrong "
            "kind");
      }
      for (Buffer *const buffer : resolved) {
        if (buffer == nullptr) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
              "attention preprocess buffer handle is stale or has the wrong "
              "kind");
        }
      }
      for (std::size_t left = 0U; left != resolved.size(); ++left) {
        for (std::size_t right = left + 1U; right != resolved.size(); ++right) {
          if (resolved[left] == resolved[right] &&
              sllm_attention_preprocess::intervals_overlap(
                  *tensor_metadata[left], *tensor_metadata[right])) {
            return sllm_public_runtime::write_error(
                error_sink, SLLM_STATUS_ALIAS_OVERLAP,
                "attention preprocess tensor byte ranges overlap");
          }
        }
      }
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (context->poisoned.load() || context->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "attention preprocess cannot prepare while the context is "
            "releasing");
      }
      for (Buffer *const buffer : resolved) {
        if (buffer->release_active) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_PUBLIC_BUSY,
              "attention preprocess cannot prepare while a buffer is "
              "releasing");
        }
        if (buffer->context != context) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH,
              "attention preprocess buffers must belong to the supplied "
              "context");
        }
      }
      const std::array<uint64_t, 8> end_offsets = {
          metadata.packed_q_gate.end_offset, metadata.k.end_offset,
          metadata.q_raw_scale.end_offset,   metadata.k_raw_scale.end_offset,
          metadata.positions.end_offset,     metadata.q_output.end_offset,
          metadata.gate_output.end_offset,   metadata.k_output.end_offset};
      for (std::size_t index = 0U; index != resolved.size(); ++index) {
        if (end_offsets[index] > resolved[index]->size_bytes) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
              "attention preprocess tensor interval exceeds its backing "
              "buffer");
        }
      }
      if (!reserve_attention_prepared_plan(context->accounting, resolved)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "attention preprocess prepared-plan accounting is exhausted");
      }
      try {
        candidate = std::make_unique<AttentionPreprocessPlan>(context, resolved,
                                                              metadata);
        token = register_handle(candidate.get(),
                                HandleKind::AttentionPreprocessPlan);
      } catch (...) {
        (void)release_attention_prepared_plan(context->accounting, resolved);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "attention preprocess prepared-plan allocation or registration "
            "failed");
      }
      if (token == 0U) {
        (void)release_attention_prepared_plan(context->accounting, resolved);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "attention preprocess prepared-plan handle allocation failed");
      }
    }
    *raw_plan = reinterpret_cast<sllm_attention_preprocess_plan_t *>(token);
    (void)candidate.release();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in attention preprocess prepare");
  }
}

extern "C" sllm_status_t sllm_attention_preprocess_plan_release(
    sllm_attention_preprocess_plan_t **const raw_plan,
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
          "attention preprocess plan handle is null");
    }
    AttentionPreprocessPlan *plan = nullptr;
    bool quarantine_plan = false;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup<AttentionPreprocessPlan>(
          *raw_plan, HandleKind::AttentionPreprocessPlan);
      if (plan == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "attention preprocess plan handle is stale or has the wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      if (plan->release_active || plan->in_flight) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "attention preprocess plan is releasing or has an in-flight "
            "completion");
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
          "attention preprocess prepared-plan accounting release failed; plan "
          "quarantined");
    }
    delete plan;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in attention preprocess plan release");
  }
}

extern "C" sllm_status_t sllm_attention_preprocess_execute(
    const sllm_attention_preprocess_plan_t *const raw_plan,
    const sllm_queue_t *const raw_queue,
    sllm_completion_t **const completion_output,
    sllm_attention_preprocess_dispatch_info_t *const dispatch_info,
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
          "attention preprocess execute plan, queue, completion, or dispatch "
          "info is null");
    }
    const sllm_status_t dispatch_status =
        validate_attention_preprocess_dispatch_info(dispatch_info, error_sink);
    if (dispatch_status != SLLM_STATUS_OK) {
      return dispatch_status;
    }

    AttentionPreprocessPlan *plan = nullptr;
    Queue *queue = nullptr;
    uint64_t dispatch_id = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup<AttentionPreprocessPlan>(
          raw_plan, HandleKind::AttentionPreprocessPlan);
      queue = lookup<Queue>(raw_queue, HandleKind::Queue);
      if (plan == nullptr || queue == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "attention preprocess plan or queue handle is stale or wrong kind");
      }
      if (plan->context != queue->context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
            "attention preprocess plan and queue belong to different contexts");
      }
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      if (plan->context->poisoned.load() || plan->context->release_active ||
          queue->release_active || plan->release_active || plan->in_flight) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "attention preprocess execute owner graph is releasing or already "
            "in flight");
      }
      if (plan->metadata.m == 0U ||
          plan->metadata.m > SLLM_HIP_ATTENTION_PREPROCESS_MAX_M) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_UNSUPPORTED,
            "attention preprocess shape exceeds the baseline kernel launch "
            "contract");
      }
      if (!reserve_attention_submission(plan->context->accounting,
                                        queue->accounting, plan->buffers)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "attention preprocess submission accounting is exhausted");
      }
      plan->in_flight = true;
      dispatch_id = plan->context->next_dispatch_id;
      if (dispatch_id == 0U) {
        plan->in_flight = false;
        (void)rollback_attention_submission(plan->context->accounting,
                                            queue->accounting, plan->buffers);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "attention preprocess context dispatch id is exhausted");
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
          "HIP gcnArchName is not NUL terminated or is too long");
    }
#if defined(SLLM_HIP_COMPILE_TARGET)
    if (!matches_runtime_gcn_arch(arch_name, SLLM_HIP_COMPILE_TARGET)) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
          "attention preprocess device target does not match the compiled "
          "exact target");
    }
#endif

    const auto byte_pointer = [](Buffer *const buffer,
                                 const uint64_t offset) -> void * {
      return static_cast<char *>(buffer->device_pointer) +
             static_cast<std::size_t>(offset);
    };
    const int32_t *const positions = static_cast<const int32_t *>(
        byte_pointer(plan->buffers[4], plan->metadata.positions.byte_offset));
    if (plan->metadata.position_payload_mode !=
        SLLM_HIP_POSITION_PAYLOAD_MODE_DERIVED_CONTIGUOUS_V1) {
      std::vector<int32_t> host_positions(
          static_cast<std::size_t>(plan->metadata.m) *
          plan->metadata.position_components);
      const hipError_t position_copy_status =
          hipMemcpyAsync(host_positions.data(), positions,
                         host_positions.size() * sizeof(int32_t),
                         hipMemcpyDeviceToHost, queue->stream);
      if (position_copy_status != hipSuccess) {
        return hip_failure(error_sink, position_copy_status,
                           "attention preprocess position validation copy");
      }
      const hipError_t position_sync_status =
          hipStreamSynchronize(queue->stream);
      if (position_sync_status != hipSuccess) {
        return hip_failure(
            error_sink, position_sync_status,
            "attention preprocess position validation synchronize");
      }
      for (std::size_t index = 0U; index != host_positions.size(); ++index) {
        const int64_t expected =
            static_cast<int64_t>(plan->metadata.start_position) +
            static_cast<int64_t>(index / plan->metadata.position_components);
        const bool valid =
            plan->metadata.position_payload_mode ==
                    SLLM_HIP_POSITION_PAYLOAD_MODE_EXPLICIT_V1
                ? host_positions[index] >= 0 &&
                      static_cast<uint64_t>(host_positions[index]) <
                          static_cast<uint64_t>(
                              SLLM_HIP_ATTENTION_PREPROCESS_MAX_POSITION)
            : plan->metadata.position_components == 1U
                ? host_positions[index] == static_cast<int32_t>(expected)
                // The mRoPE ABI carries signed I32 components while its
                // public exclusive upper bound is UINT32_MAX. Every
                // non-negative I32 value is therefore representable and
                // below that bound.
                : host_positions[index] >= 0;
        if (!valid) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_POSITION_PAYLOAD_MISMATCH,
              "attention preprocess position payload does not equal the "
              "requested contiguous range");
        }
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
          "attention preprocess completion allocation failed before enqueue");
    }
    hipEvent_t native_event = nullptr;
    hipError_t status = queue_profiles_completions(queue)
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
          "attention preprocess timing event creation failed before enqueue",
          error_sink);
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
          "attention preprocess completion registry allocation failed",
          error_sink);
    }
    if (token == 0U) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "attention preprocess completion handle allocation failed",
          error_sink);
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
    const uint16_t *const packed_q_gate =
        static_cast<const uint16_t *>(byte_pointer(
            plan->buffers[0], plan->metadata.packed_q_gate.byte_offset));
    const uint16_t *const k = static_cast<const uint16_t *>(
        byte_pointer(plan->buffers[1], plan->metadata.k.byte_offset));
    const uint16_t *const q_raw_scale = static_cast<const uint16_t *>(
        byte_pointer(plan->buffers[2], plan->metadata.q_raw_scale.byte_offset));
    const uint16_t *const k_raw_scale = static_cast<const uint16_t *>(
        byte_pointer(plan->buffers[3], plan->metadata.k_raw_scale.byte_offset));
    uint16_t *const q_output = static_cast<uint16_t *>(
        byte_pointer(plan->buffers[5], plan->metadata.q_output.byte_offset));
    uint16_t *const gate_output = static_cast<uint16_t *>(
        byte_pointer(plan->buffers[6], plan->metadata.gate_output.byte_offset));
    uint16_t *const k_output = static_cast<uint16_t *>(
        byte_pointer(plan->buffers[7], plan->metadata.k_output.byte_offset));
    const bool use_wave32 =
        use_rdna_wave32_attention_preprocess(plan->metadata, arch_name);
    status = ::sllm_attention_preprocess_kernel::launch(
        packed_q_gate, k, q_raw_scale, k_raw_scale, positions, q_output,
        gate_output, k_output, static_cast<uint32_t>(plan->metadata.m),
        plan->metadata.q_heads, plan->metadata.k_heads, plan->metadata.head_dim,
        plan->metadata.position_components, plan->metadata.start_position,
        plan->metadata.position_payload_mode, use_wave32, queue->stream);
    if (status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, status,
                                       "attention preprocess kernel launch",
                                       queue, error_sink);
    }
    status = candidate->event != nullptr
                 ? hipEventRecord(candidate->event, queue->stream)
                 : hipSuccess;
    if (status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, status,
                                       "hipEventRecord", queue, error_sink);
    }
    initialize_attention_preprocess_dispatch_info(
        dispatch_info, dispatch_id, plan->metadata, use_wave32, arch_name);
    *completion_output = reinterpret_cast<sllm_completion_t *>(token);
    (void)candidate.release();
    execute_guard.disarm();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in attention preprocess execute");
  }
}
