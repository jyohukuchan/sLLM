namespace {

sllm_status_t validate_token_selector_dispatch_info(
    const sllm_token_selector_dispatch_info_t *const info,
    sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "token selector dispatch info output is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, info, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_token_selector_dispatch_info_t)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "token selector dispatch info has an unsupported struct size");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "token selector dispatch info ABI is unsupported");
  }
  if (info->info_version != SLLM_HIP_TOKEN_SELECTOR_DISPATCH_INFO_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "token selector dispatch info version is unsupported");
  }
  for (const uint32_t value : info->reserved) {
    if (value != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_RESERVED_NONZERO,
          "token selector dispatch info reserved fields must be zero");
    }
  }
  return SLLM_STATUS_OK;
}

void initialize_token_selector_dispatch_info(
    sllm_token_selector_dispatch_info_t *const info,
    const uint64_t dispatch_id,
    const sllm_token_selector::DescriptorMetadata &metadata,
    const char *const arch_name) noexcept {
  const uint32_t struct_size = info->struct_size;
  const uint32_t abi_version = info->abi_version;
  std::memset(info, 0, sizeof(*info));
  info->struct_size = struct_size;
  info->abi_version = abi_version;
  info->info_version = SLLM_HIP_TOKEN_SELECTOR_DISPATCH_INFO_VERSION;
  info->backend = SLLM_BACKEND_HIP;
  info->dispatch_id = dispatch_id;
  info->dispatch_count = 1U;
  info->kernel_id = SLLM_HIP_TOKEN_SELECTOR_KERNEL_ID_BF16_F32_MASK_V1;
  info->workgroup_size_x = SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE;
  info->grid_size_x = 1U;
  info->vocab_size = metadata.vocab_size;
  info->fallback_allowed = 0U;
  info->fallback_used = 0U;
  info->result_status = SLLM_STATUS_OK;
  info->token_id = -1;
  sllm_public_runtime::copy_fixed_string(
      info->kernel_symbol, SLLM_HIP_TOKEN_SELECTOR_KERNEL_SYMBOL_MAX,
      ::sllm_token_selector_kernel::kLogicalKernelId);
  sllm_public_runtime::copy_fixed_string(
      info->device_symbol, SLLM_HIP_TOKEN_SELECTOR_DEVICE_SYMBOL_MAX,
      ::sllm_token_selector_kernel::kDeviceSymbol);
  sllm_public_runtime::copy_fixed_string(
      info->gcn_arch_name, SLLM_HIP_MAX_GCN_ARCH_NAME, arch_name);
}

bool token_selector_intervals_overlap(const Buffer *const left_buffer,
                                      const sllm_token_selector::TensorMetadata &left,
                                      const Buffer *const right_buffer,
                                      const sllm_token_selector::TensorMetadata &right) noexcept {
  return left_buffer == right_buffer &&
         sllm_token_selector::intervals_overlap(left, right);
}

} // namespace

extern "C" sllm_status_t sllm_token_selector_prepare(
    const sllm_context_t *const raw_context,
    const sllm_token_selector_desc_t *const descriptor,
    sllm_token_selector_plan_t **const raw_plan,
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
          "token selector context or plan output is null");
    }
    sllm_token_selector::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_token_selector::validate_and_copy_descriptor(descriptor, &metadata,
                                                          error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }

    std::unique_ptr<TokenSelectorPlan> candidate;
    uintptr_t token = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      Context *const context = lookup<Context>(raw_context, HandleKind::Context);
      Buffer *const logits =
          lookup<Buffer>(descriptor->logits.buffer, HandleKind::Buffer);
      Buffer *const additive =
          lookup<Buffer>(descriptor->additive_logits.buffer, HandleKind::Buffer);
      Buffer *const valid_mask =
          lookup<Buffer>(descriptor->valid_mask.buffer, HandleKind::Buffer);
      Buffer *const output =
          lookup<Buffer>(descriptor->output.buffer, HandleKind::Buffer);
      if (context == nullptr || logits == nullptr || additive == nullptr ||
          valid_mask == nullptr || output == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "token selector context or buffer handle is stale or wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (context->poisoned.load() || context->release_active ||
          logits->release_active || additive->release_active ||
          valid_mask->release_active || output->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "token selector cannot prepare while an owner is releasing");
      }
      if (logits->context != context || additive->context != context ||
          valid_mask->context != context || output->context != context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH,
            "token selector buffers must belong to the supplied context");
      }
      if (metadata.logits.end_offset > logits->size_bytes ||
          metadata.additive_logits.end_offset > additive->size_bytes ||
          metadata.valid_mask.end_offset > valid_mask->size_bytes ||
          metadata.output.end_offset > output->size_bytes) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
            "token selector tensor interval exceeds its backing buffer");
      }
      const Buffer *const buffers[] = {logits, additive, valid_mask, output};
      const sllm_token_selector::TensorMetadata metadata_values[] = {
          metadata.logits, metadata.additive_logits, metadata.valid_mask,
          metadata.output};
      for (std::size_t index = 0U; index != 4U; ++index) {
        for (std::size_t prior = 0U; prior != index; ++prior) {
          if (token_selector_intervals_overlap(
                  buffers[index], metadata_values[index], buffers[prior],
                  metadata_values[prior])) {
            return sllm_public_runtime::write_error(
                error_sink, SLLM_STATUS_ALIAS_OVERLAP,
                "token selector tensor intervals overlap in one buffer");
          }
        }
      }
      if (!sllm_public_runtime::AccountingState::
              reserve_token_selector_prepared_plan(
                  context->accounting, logits->accounting, additive->accounting,
                  valid_mask->accounting, output->accounting)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "token selector prepared-plan accounting is exhausted");
      }
      try {
        candidate = std::make_unique<TokenSelectorPlan>(
            context, logits, additive, valid_mask, output, metadata);
        token = register_handle(candidate.get(), HandleKind::TokenSelectorPlan);
      } catch (...) {
        (void)sllm_public_runtime::AccountingState::
            release_token_selector_prepared_plan(
                context->accounting, logits->accounting, additive->accounting,
                valid_mask->accounting, output->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "token selector prepared-plan allocation or registration failed");
      }
      if (token == 0U) {
        (void)sllm_public_runtime::AccountingState::
            release_token_selector_prepared_plan(
                context->accounting, logits->accounting, additive->accounting,
                valid_mask->accounting, output->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "token selector prepared-plan handle allocation failed");
      }
    }
    *raw_plan = reinterpret_cast<sllm_token_selector_plan_t *>(token);
    (void)candidate.release();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in token selector prepare");
  }
}

extern "C" sllm_status_t sllm_token_selector_plan_release(
    sllm_token_selector_plan_t **const raw_plan,
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
          "token selector plan handle is null");
    }
    TokenSelectorPlan *plan = nullptr;
    bool quarantine_plan = false;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup<TokenSelectorPlan>(*raw_plan, HandleKind::TokenSelectorPlan);
      if (plan == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "token selector plan handle is stale or has the wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(plan->context->accounting_mutex);
      if (plan->release_active || plan->in_flight) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "token selector plan is releasing or has an in-flight completion");
      }
      plan->release_active = true;
      const bool accounting_released =
          !sllm_public_runtime::FaultInjector::consume(
              sllm_public_runtime::FaultPoint::AccountingFailure) &&
          sllm_public_runtime::AccountingState::
              release_token_selector_prepared_plan(
                  plan->context->accounting, plan->logits->accounting,
                  plan->additive_logits->accounting,
                  plan->valid_mask->accounting, plan->output->accounting);
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
          "token selector prepared-plan accounting release failed; plan quarantined");
    }
    delete plan;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in token selector plan release");
  }
}

extern "C" sllm_status_t sllm_token_selector_execute(
    const sllm_token_selector_plan_t *const raw_plan,
    const sllm_queue_t *const raw_queue,
    sllm_completion_t **const completion_output,
    sllm_token_selector_dispatch_info_t *const dispatch_info,
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
          "token selector execute plan, queue, completion, or dispatch info is null");
    }
    const sllm_status_t dispatch_status =
        validate_token_selector_dispatch_info(dispatch_info, error_sink);
    if (dispatch_status != SLLM_STATUS_OK) {
      return dispatch_status;
    }
    TokenSelectorPlan *plan = nullptr;
    Queue *queue = nullptr;
    uint64_t dispatch_id = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup<TokenSelectorPlan>(raw_plan, HandleKind::TokenSelectorPlan);
      queue = lookup<Queue>(raw_queue, HandleKind::Queue);
      if (plan == nullptr || queue == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "token selector plan or queue handle is stale or wrong kind");
      }
      if (plan->context != queue->context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
            "token selector plan and queue belong to different contexts");
      }
      std::lock_guard<std::mutex> accounting_lock(plan->context->accounting_mutex);
      if (plan->context->poisoned.load() || plan->context->release_active ||
          queue->release_active || plan->release_active || plan->in_flight ||
          plan->logits->release_active || plan->additive_logits->release_active ||
          plan->valid_mask->release_active || plan->output->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "token selector owner graph is releasing or already in flight");
      }
      if (!sllm_public_runtime::AccountingState::reserve_token_selector_submission(
              plan->context->accounting, queue->accounting,
              plan->logits->accounting, plan->additive_logits->accounting,
              plan->valid_mask->accounting, plan->output->accounting)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "token selector submission accounting is exhausted");
      }
      plan->in_flight = true;
      dispatch_id = plan->context->next_dispatch_id;
      if (dispatch_id == 0U) {
        plan->in_flight = false;
        (void)sllm_public_runtime::AccountingState::
            rollback_token_selector_submission(
                plan->context->accounting, queue->accounting,
                plan->logits->accounting, plan->additive_logits->accounting,
                plan->valid_mask->accounting, plan->output->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "token selector context dispatch id is exhausted");
      }
      plan->context->next_dispatch_id =
          dispatch_id == std::numeric_limits<uint64_t>::max()
              ? 0U
              : dispatch_id + 1U;
      *completion_output = nullptr;
    }

    std::unique_ptr<Completion> candidate;
    NativeEventGuard event_guard;
    TokenSelectorExecuteScopeGuard execute_guard(plan, queue, &candidate,
                                                 &event_guard, error_sink);
    const sllm_status_t device_status = select_context_device(plan->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      if (!rollback_reserved_token_selector_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return device_status;
    }
    hipDeviceProp_t properties = {};
    const sllm_status_t property_status =
        get_device_properties(plan->context->device_index, &properties, error_sink);
    if (property_status != SLLM_STATUS_OK) {
      if (!rollback_reserved_token_selector_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return property_status;
    }
    char arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME] = {};
    if (!copy_property(arch_name, sizeof(arch_name), properties.gcnArchName,
                       sizeof(properties.gcnArchName))) {
      if (!rollback_reserved_token_selector_submission(plan, queue, error_sink)) {
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
      if (!rollback_reserved_token_selector_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
          "token selector device target does not match the compiled exact target");
    }
#endif

    try {
      candidate = std::make_unique<Completion>(
          plan->context, queue, plan->logits, 0U, false,
          std::vector<uint8_t>{});
      candidate->token_selector = true;
      candidate->token_selector_plan = plan;
      candidate->token_selector_logits = plan->logits;
      candidate->token_selector_additive = plan->additive_logits;
      candidate->token_selector_valid_mask = plan->valid_mask;
      candidate->token_selector_output = plan->output;
      execute_guard.candidate_allocated();
    } catch (...) {
      if (!rollback_reserved_token_selector_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "token selector completion allocation failed before enqueue");
    }

    hipEvent_t native_event = nullptr;
    const hipError_t event_status = queue_profiles_completions(queue)
                                        ? hipEventCreateWithFlags(&native_event, 0U)
                                        : hipSuccess;
    if (event_status != hipSuccess) {
      if (!rollback_reserved_token_selector_submission(plan, queue, error_sink)) {
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
          "token selector timing event creation failed before enqueue",
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
          "token selector completion registry allocation failed before enqueue",
          error_sink);
    }
    if (token == 0U) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "token selector completion handle token allocation failed",
          error_sink);
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
    const hipError_t launch_status = ::sllm_token_selector_kernel::launch(
        static_cast<const uint16_t *>(byte_pointer(
            plan->logits, plan->metadata.logits.byte_offset)),
        static_cast<const float *>(byte_pointer(
            plan->additive_logits, plan->metadata.additive_logits.byte_offset)),
        static_cast<const uint8_t *>(byte_pointer(
            plan->valid_mask, plan->metadata.valid_mask.byte_offset)),
        plan->metadata.vocab_size, plan->metadata.temperature,
        plan->metadata.seed, plan->metadata.counter,
        static_cast<sllm_token_selector_record_t *>(byte_pointer(
            plan->output, plan->metadata.output.byte_offset)), queue->stream);
    if (launch_status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, launch_status,
                                       "token selector kernel launch", queue,
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
    initialize_token_selector_dispatch_info(dispatch_info, dispatch_id,
                                            plan->metadata, arch_name);
    *completion_output = reinterpret_cast<sllm_completion_t *>(token);
    (void)candidate.release();
    execute_guard.disarm();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in token selector execute");
  }
}
