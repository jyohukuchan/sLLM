namespace {

struct DeepSeekV4MoeRoutePlan final : ArrayOperationPlan {
  sllm_deepseek_v4_moe_route::DescriptorMetadata metadata;

  DeepSeekV4MoeRoutePlan(
      Context *const context_value, const AttentionBuffers buffers_value,
      const sllm_deepseek_v4_moe_route::DescriptorMetadata &metadata_value)
      : ArrayOperationPlan(HandleKind::MoeRoutePlan, context_value,
                           buffers_value),
        metadata(metadata_value) {}
};

std::unordered_map<uintptr_t, DeepSeekV4MoeRoutePlan *>
    deepseek_v4_moe_route_plans;

DeepSeekV4MoeRoutePlan *
lookup_deepseek_v4_moe_route_plan(const void *const raw) noexcept {
  const auto found = deepseek_v4_moe_route_plans.find(handle_key(raw));
  return found == deepseek_v4_moe_route_plans.end() ? nullptr : found->second;
}

void initialize_deepseek_v4_moe_route_dispatch_info(
    sllm_deepseek_v4_moe_route_dispatch_info_t *const info,
    const uint64_t dispatch_id,
    const sllm_deepseek_v4_moe_route::DescriptorMetadata &metadata,
    const char *const arch_name) noexcept {
  const uint32_t struct_size = info->struct_size;
  const uint32_t abi_version = info->abi_version;
  std::memset(info, 0, sizeof(*info));
  info->struct_size = struct_size;
  info->abi_version = abi_version;
  info->info_version = SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_DISPATCH_INFO_VERSION;
  info->backend = SLLM_BACKEND_HIP;
  info->dispatch_id = dispatch_id;
  info->dispatch_count = 2U;
  info->kernel_id = metadata.mode == SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_SCORE
                        ? SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_KERNEL_ID_SCORE_V1
                        : SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_KERNEL_ID_HASH_V1;
  info->workgroup_size_x = SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_WORKGROUP_SIZE;
  info->grid_size_x = static_cast<uint32_t>(metadata.token_count);
  info->token_count = metadata.token_count;
  info->expert_count = metadata.expert_count;
  info->pair_count = metadata.pair_count;
  info->selected_expert_count = metadata.selected_expert_count;
  info->mode = metadata.mode;
  info->renormalize = metadata.renormalize;
  info->fallback_allowed = 0U;
  info->fallback_used = 0U;
  sllm_public_runtime::copy_fixed_string(
      info->kernel_symbol, SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_KERNEL_SYMBOL_MAX,
      metadata.mode == SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_SCORE
          ? ::sllm_deepseek_v4_moe_route_kernel::kLogicalKernelIdScore
          : ::sllm_deepseek_v4_moe_route_kernel::kLogicalKernelIdHash);
  sllm_public_runtime::copy_fixed_string(
      info->device_symbol, SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_DEVICE_SYMBOL_MAX,
      ::sllm_deepseek_v4_moe_route_kernel::kDeviceSymbol);
  sllm_public_runtime::copy_fixed_string(info->gcn_arch_name,
                                         SLLM_HIP_MAX_GCN_ARCH_NAME, arch_name);
}

} // namespace

extern "C" sllm_status_t sllm_deepseek_v4_moe_route_prepare(
    const sllm_context_t *const raw_context,
    const sllm_deepseek_v4_moe_route_desc_t *const descriptor,
    sllm_deepseek_v4_moe_route_plan_t **const raw_plan,
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
          "DeepSeek V4 route context or plan output is null");
    }
    const sllm_status_t prefix_status =
        sllm_deepseek_v4_moe_route::validate_descriptor_prefix(descriptor,
                                                               error_sink);
    if (prefix_status != SLLM_STATUS_OK) {
      return prefix_status;
    }
    const sllm_deepseek_v4_moe_route_desc_t descriptor_copy = *descriptor;
    sllm_deepseek_v4_moe_route::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_deepseek_v4_moe_route::validate_and_copy_descriptor(
            &descriptor_copy, &metadata, error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }

    AttentionBuffers resolved{};
    std::array<const sllm_deepseek_v4_moe_route::TensorMetadata *, 4>
        tensor_metadata{};
    std::size_t active_tensor_count = 0U;
    std::unique_ptr<DeepSeekV4MoeRoutePlan> candidate;
    uintptr_t token = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      Context *const context =
          lookup<Context>(raw_context, HandleKind::Context);
      Buffer *const logits =
          lookup<Buffer>(descriptor_copy.logits.buffer, HandleKind::Buffer);
      Buffer *const bias =
          metadata.mode == SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_SCORE
              ? lookup<Buffer>(descriptor_copy.selection_bias.buffer,
                               HandleKind::Buffer)
              : logits;
      Buffer *const hash_ids =
          metadata.mode == SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_HASH
              ? lookup<Buffer>(descriptor_copy.hash_expert_ids.buffer,
                               HandleKind::Buffer)
              : logits;
      Buffer *const output =
          lookup<Buffer>(descriptor_copy.metadata.buffer, HandleKind::Buffer);
      if (context == nullptr || logits == nullptr || bias == nullptr ||
          hash_ids == nullptr || output == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "DeepSeek V4 route context or buffer handle is stale or wrong "
            "kind");
      }
      resolved = {logits, bias,   hash_ids, output,
                  logits, logits, logits,   logits};
      tensor_metadata[active_tensor_count++] = &metadata.logits;
      if (metadata.mode == SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_SCORE) {
        tensor_metadata[active_tensor_count++] = &metadata.selection_bias;
      } else {
        tensor_metadata[active_tensor_count++] = &metadata.hash_expert_ids;
      }
      tensor_metadata[active_tensor_count++] = &metadata.output;
      const std::array<Buffer *, 4> active_buffers = {
          logits,
          metadata.mode == SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_SCORE ? bias
                                                                 : hash_ids,
          output, nullptr};
      for (std::size_t left = 0U; left != active_tensor_count; ++left) {
        for (std::size_t right = left + 1U; right != active_tensor_count;
             ++right) {
          if (active_buffers[left] == active_buffers[right] &&
              sllm_deepseek_v4_moe_route::intervals_overlap(
                  *tensor_metadata[left], *tensor_metadata[right])) {
            return sllm_public_runtime::write_error(
                error_sink, SLLM_STATUS_ALIAS_OVERLAP,
                "DeepSeek V4 route tensor byte ranges overlap");
          }
        }
      }
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (context->poisoned.load() || context->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "DeepSeek V4 route cannot prepare while the context is releasing");
      }
      for (Buffer *const buffer : resolved) {
        if (buffer->release_active) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_PUBLIC_BUSY,
              "DeepSeek V4 route cannot prepare while a buffer is releasing");
        }
        if (buffer->context != context) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH,
              "DeepSeek V4 route buffers must belong to the supplied context");
        }
      }
      const uint64_t bias_or_hash_end =
          metadata.mode == SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_SCORE
              ? metadata.selection_bias.end_offset
              : metadata.hash_expert_ids.end_offset;
      if (metadata.logits.end_offset > logits->size_bytes ||
          bias_or_hash_end > active_buffers[1]->size_bytes ||
          metadata.output.end_offset > output->size_bytes) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
            "DeepSeek V4 route tensor interval exceeds its backing buffer");
      }
      if (!reserve_attention_prepared_plan(context->accounting, resolved)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "DeepSeek V4 route prepared-plan accounting is exhausted");
      }
      try {
        candidate = std::make_unique<DeepSeekV4MoeRoutePlan>(context, resolved,
                                                             metadata);
        /* A null-state placeholder reserves a globally unique opaque token
         * while guaranteeing that every pre-existing API rejects this new
         * plan kind.  The DeepSeek registry owns the typed association. */
        token = register_handle(nullptr, HandleKind::Context);
        if (token != 0U) {
          const auto inserted =
              deepseek_v4_moe_route_plans.emplace(token, candidate.get());
          if (!inserted.second) {
            unregister_handle_token(token);
            token = 0U;
          }
        }
      } catch (...) {
        if (token != 0U) {
          deepseek_v4_moe_route_plans.erase(token);
          unregister_handle_token(token);
        }
        (void)release_attention_prepared_plan(context->accounting, resolved);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "DeepSeek V4 route prepared-plan allocation or registration "
            "failed");
      }
      if (token == 0U) {
        (void)release_attention_prepared_plan(context->accounting, resolved);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "DeepSeek V4 route prepared-plan handle allocation failed");
      }
    }
    *raw_plan = reinterpret_cast<sllm_deepseek_v4_moe_route_plan_t *>(token);
    (void)candidate.release();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in DeepSeek V4 route prepare");
  }
}

extern "C" sllm_status_t sllm_deepseek_v4_moe_route_plan_release(
    sllm_deepseek_v4_moe_route_plan_t **const raw_plan,
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
          "DeepSeek V4 route plan handle is null");
    }
    DeepSeekV4MoeRoutePlan *plan = nullptr;
    bool quarantine_plan = false;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup_deepseek_v4_moe_route_plan(*raw_plan);
      if (plan == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "DeepSeek V4 route plan handle is stale or has the wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      if (plan->release_active || plan->in_flight) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "DeepSeek V4 route plan is releasing or has an in-flight "
            "completion");
      }
      plan->release_active = true;
      const bool accounting_released =
          !sllm_public_runtime::FaultInjector::consume(
              sllm_public_runtime::FaultPoint::AccountingFailure) &&
          release_attention_prepared_plan(plan->context->accounting,
                                          plan->buffers);
      deepseek_v4_moe_route_plans.erase(handle_key(*raw_plan));
      unregister_handle(*raw_plan);
      *raw_plan = nullptr;
      if (!accounting_released) {
        plan->context->poisoned.store(true);
        quarantine_plan = true;
      }
    }
    if (quarantine_plan) {
      /* Keep dependency pointers alive after an accounting ambiguity. */
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "DeepSeek V4 route accounting release failed; plan quarantined");
    }
    delete plan;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in DeepSeek V4 route plan release");
  }
}

extern "C" sllm_status_t sllm_deepseek_v4_moe_route_execute(
    const sllm_deepseek_v4_moe_route_plan_t *const raw_plan,
    const sllm_queue_t *const raw_queue,
    sllm_completion_t **const completion_output,
    sllm_deepseek_v4_moe_route_dispatch_info_t *const dispatch_info,
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
    if (raw_plan == nullptr || raw_queue == nullptr ||
        completion_output == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "DeepSeek V4 route execute input or output is null");
    }
    const sllm_status_t dispatch_status =
        sllm_deepseek_v4_moe_route::validate_dispatch_info(dispatch_info,
                                                           error_sink);
    if (dispatch_status != SLLM_STATUS_OK) {
      return dispatch_status;
    }

    DeepSeekV4MoeRoutePlan *plan = nullptr;
    Queue *queue = nullptr;
    uint64_t dispatch_id = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup_deepseek_v4_moe_route_plan(raw_plan);
      queue = lookup<Queue>(raw_queue, HandleKind::Queue);
      if (plan == nullptr || queue == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "DeepSeek V4 route plan or queue handle is stale or wrong kind");
      }
      if (plan->context != queue->context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
            "DeepSeek V4 route plan and queue belong to different contexts");
      }
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      if (plan->context->poisoned.load() || plan->context->release_active ||
          queue->release_active || plan->release_active || plan->in_flight) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "DeepSeek V4 route owner graph is releasing or already in flight");
      }
      for (Buffer *const buffer : plan->buffers) {
        if (buffer->release_active) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_PUBLIC_BUSY,
              "DeepSeek V4 route buffer is releasing");
        }
      }
      if (plan->metadata.token_count == 0U ||
          plan->metadata.token_count >
              SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_MAX_TOKENS ||
          plan->metadata.expert_count !=
              SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_EXPERT_COUNT ||
          plan->metadata.selected_expert_count !=
              SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_SELECTED_EXPERT_COUNT) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_UNSUPPORTED,
            "DeepSeek V4 route shape exceeds its launch contract");
      }
      if (!reserve_attention_submission(plan->context->accounting,
                                        queue->accounting, plan->buffers)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "DeepSeek V4 route submission accounting is exhausted");
      }
      plan->in_flight = true;
      dispatch_id = plan->context->next_dispatch_id;
      if (dispatch_id == 0U) {
        plan->in_flight = false;
        (void)rollback_attention_submission(plan->context->accounting,
                                            queue->accounting, plan->buffers);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "DeepSeek V4 route context dispatch id is exhausted");
      }
      plan->context->next_dispatch_id =
          dispatch_id == std::numeric_limits<uint64_t>::max()
              ? 0U
              : dispatch_id + 1U;
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
          "DeepSeek V4 route device target does not match the compiled exact "
          "target");
    }
#endif

    try {
      candidate = std::make_unique<Completion>(
          plan->context, queue, plan->buffers[0], 0U, false,
          std::vector<uint8_t>(sizeof(int32_t), 0U), nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, plan, plan->buffers);
      candidate->validator_kind =
          CompletionValidatorKind::DeepSeekV4MoeRouteStatus;
      execute_guard.candidate_allocated();
    } catch (...) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "DeepSeek V4 route completion allocation failed before enqueue");
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
          "DeepSeek V4 route timing event creation failed before enqueue",
          error_sink);
    }
    candidate->timing_start_event = timing_start_event;
    uintptr_t completion_token = 0U;
    try {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      completion_token =
          register_handle(candidate.get(), HandleKind::Completion);
    } catch (...) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "DeepSeek V4 route completion registry allocation failed",
          error_sink);
    }
    if (completion_token == 0U) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "DeepSeek V4 route completion handle allocation failed", error_sink);
    }
    if (candidate->event != nullptr) {
      event_guard.release();
    }
    execute_guard.completion_registered(completion_token);
    status = candidate->timing_start_event != nullptr
                 ? hipEventRecord(candidate->timing_start_event, queue->stream)
                 : hipSuccess;
    if (status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, completion_token, status,
                                       "DeepSeek V4 route timing start record",
                                       queue, error_sink);
    }

    const auto byte_pointer = [](Buffer *const buffer,
                                 const uint64_t offset) -> void * {
      return static_cast<char *>(buffer->device_pointer) +
             static_cast<std::size_t>(offset);
    };
    const auto &metadata = plan->metadata;
    const uint16_t *const logits = static_cast<const uint16_t *>(
        byte_pointer(plan->buffers[0], metadata.logits.byte_offset));
    const float *const bias =
        metadata.mode == SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_SCORE
            ? static_cast<const float *>(byte_pointer(
                  plan->buffers[1], metadata.selection_bias.byte_offset))
            : nullptr;
    const int32_t *const hash_ids =
        metadata.mode == SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_HASH
            ? static_cast<const int32_t *>(byte_pointer(
                  plan->buffers[2], metadata.hash_expert_ids.byte_offset))
            : nullptr;
    const uint64_t base = metadata.output.byte_offset;
    int32_t *const expert_ids = static_cast<int32_t *>(
        byte_pointer(plan->buffers[3], base + metadata.ids_offset));
    float *const expert_weights = static_cast<float *>(
        byte_pointer(plan->buffers[3], base + metadata.weights_offset));
    int32_t *const expert_counts = static_cast<int32_t *>(
        byte_pointer(plan->buffers[3], base + metadata.counts_offset));
    int32_t *const expert_offsets = static_cast<int32_t *>(
        byte_pointer(plan->buffers[3], base + metadata.offsets_offset));
    int32_t *const grouped_tokens = static_cast<int32_t *>(
        byte_pointer(plan->buffers[3], base + metadata.grouped_tokens_offset));
    int32_t *const grouped_slots = static_cast<int32_t *>(
        byte_pointer(plan->buffers[3], base + metadata.grouped_slots_offset));
    int32_t *const device_status_ptr = static_cast<int32_t *>(
        byte_pointer(plan->buffers[3], base + metadata.status_offset));
    status = ::sllm_deepseek_v4_moe_route_kernel::launch(
        logits, bias, hash_ids, expert_ids, expert_weights, expert_counts,
        expert_offsets, grouped_tokens, grouped_slots, device_status_ptr,
        metadata.token_count, metadata.mode, metadata.renormalize,
        metadata.routed_scale, queue->stream);
    if (status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, completion_token, status,
                                       "DeepSeek V4 route kernel launch", queue,
                                       error_sink);
    }
    status = hipMemcpyAsync(candidate->host_storage.data(), device_status_ptr,
                            sizeof(*device_status_ptr), hipMemcpyDeviceToHost,
                            queue->stream);
    if (status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(
          candidate, completion_token, status,
          "DeepSeek V4 route device status readback", queue, error_sink);
    }
    status = candidate->event != nullptr
                 ? hipEventRecord(candidate->event, queue->stream)
                 : hipSuccess;
    if (status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, completion_token, status,
                                       "hipEventRecord", queue, error_sink);
    }
    initialize_deepseek_v4_moe_route_dispatch_info(dispatch_info, dispatch_id,
                                                   metadata, arch_name);
    *completion_output =
        reinterpret_cast<sllm_completion_t *>(completion_token);
    (void)candidate.release();
    execute_guard.disarm();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in DeepSeek V4 route execute");
  }
}
