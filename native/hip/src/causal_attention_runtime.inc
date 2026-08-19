namespace {

class CausalAttentionExecuteScopeGuard final {
public:
  CausalAttentionExecuteScopeGuard(KvState *const state, Queue *const queue,
                                   const CausalAttentionBuffers buffers,
                                   std::unique_ptr<Completion> *const candidate,
                                   NativeEventGuard *const event_guard,
                                   sllm_error_sink_t *const sink) noexcept
      : state_(state), queue_(queue), buffers_(buffers), candidate_(candidate),
        event_guard_(event_guard), sink_(sink), token_(0U),
        phase_(Phase::Reserved) {}

  CausalAttentionExecuteScopeGuard(const CausalAttentionExecuteScopeGuard &) =
      delete;
  CausalAttentionExecuteScopeGuard &
  operator=(const CausalAttentionExecuteScopeGuard &) = delete;

  ~CausalAttentionExecuteScopeGuard() { cleanup(); }

  void candidate_allocated() noexcept { phase_ = Phase::Candidate; }
  void completion_registered(const uintptr_t token) noexcept {
    token_ = token;
    phase_ = Phase::Registered;
  }
  void disarm() noexcept { phase_ = Phase::Disarmed; }

private:
  enum class Phase : uint8_t { Reserved, Candidate, Registered, Disarmed };

  void cleanup() noexcept {
    switch (phase_) {
    case Phase::Reserved:
      (void)rollback_reserved_causal_attention(state_, queue_, buffers_, sink_);
      break;
    case Phase::Candidate:
      if (candidate_ != nullptr && candidate_->get() != nullptr) {
        (void)rollback_unpublished_submission(
            *candidate_, *event_guard_,
            "causal attention unwound before completion registration", sink_);
      } else {
        state_->context->poisoned.store(true);
      }
      break;
    case Phase::Registered:
      if (candidate_ != nullptr && candidate_->get() != nullptr) {
        (void)cleanup_failed_submission(
            *candidate_, token_, hipErrorUnknown,
            "causal attention unwound after completion registration", queue_,
            sink_);
      } else {
        state_->context->poisoned.store(true);
      }
      break;
    case Phase::Disarmed:
      break;
    }
    phase_ = Phase::Disarmed;
  }

  KvState *const state_;
  Queue *const queue_;
  CausalAttentionBuffers buffers_;
  std::unique_ptr<Completion> *const candidate_;
  NativeEventGuard *const event_guard_;
  sllm_error_sink_t *const sink_;
  uintptr_t token_;
  Phase phase_;
};

bool causal_intervals_overlap(const Buffer *const left_buffer,
                              const uint64_t left_start,
                              const uint64_t left_end,
                              const Buffer *const right_buffer,
                              const uint64_t right_start,
                              const uint64_t right_end) noexcept {
  return left_buffer == right_buffer && left_start < right_end &&
         right_start < left_end;
}

sllm_status_t validate_causal_attention_dispatch_info(
    const sllm_causal_attention_dispatch_info_t *const info,
    sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr || info->struct_size != sizeof(*info)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "causal attention dispatch info has an unsupported struct size");
  }
  if (info->abi_version != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "causal attention dispatch info ABI version is unsupported");
  }
  if (info->info_version != SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION ||
      !std::all_of(std::begin(info->reserved), std::end(info->reserved),
                   [](const uint32_t value) { return value == 0U; })) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "causal attention dispatch info version or reserved fields are "
        "invalid");
  }
  return SLLM_STATUS_OK;
}

void fill_causal_attention_dispatch_info(
    sllm_causal_attention_dispatch_info_t *const info,
    const uint64_t dispatch_id,
    const sllm_causal_attention::DescriptorMetadata &metadata,
    const uint32_t kv_heads, const uint32_t encoding,
    const char *const arch_name) noexcept {
  const bool use_gfx1201_wave_provider =
      std::strcmp(arch_name, "gfx1201") == 0 &&
      (metadata.query_count == 1U || metadata.query_count >= 32U);
  const uint32_t struct_size = info->struct_size;
  const uint32_t abi_version = info->abi_version;
  std::memset(info, 0, sizeof(*info));
  info->struct_size = struct_size;
  info->abi_version = abi_version;
  info->info_version = SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION;
  info->backend = SLLM_BACKEND_HIP;
  info->dispatch_id = dispatch_id;
  info->dispatch_count = 1U;
  info->kernel_id = encoding == SLLM_HIP_KV_ENCODING_FP16_V1
                        ? SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_ONLINE_SOFTMAX_V2
                        : SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PACKED_KV_V3;
  info->workgroup_size_x = SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE;
  info->grid_size_x =
      static_cast<uint32_t>(metadata.query_count * metadata.query.q_heads);
  info->query_count = metadata.query_count;
  info->start_position = metadata.start_position;
  info->committed_kv_length = metadata.expected_kv_length;
  info->q_heads = metadata.query.q_heads;
  info->kv_heads = kv_heads;
  info->head_dim = metadata.query.head_dim;
  info->scale_denominator = SLLM_HIP_CAUSAL_ATTENTION_SCALE_DENOMINATOR;
  info->fallback_allowed = 0U;
  info->fallback_used = 0U;
  sllm_public_runtime::copy_fixed_string(
      info->kernel_symbol, SLLM_HIP_CAUSAL_ATTENTION_KERNEL_SYMBOL_MAX,
      use_gfx1201_wave_provider
          ? (encoding == SLLM_HIP_KV_ENCODING_FP16_V1
                 ? ::sllm_causal_attention_kernel::kGfx1201WaveLogicalKernelId
                 : ::sllm_causal_attention_kernel::
                       kGfx1201WavePackedKvLogicalKernelId)
          : (encoding == SLLM_HIP_KV_ENCODING_FP16_V1
                 ? ::sllm_causal_attention_kernel::kLogicalKernelId
                 : ::sllm_causal_attention_kernel::kPackedKvLogicalKernelId));
  sllm_public_runtime::copy_fixed_string(
      info->device_symbol, SLLM_HIP_CAUSAL_ATTENTION_DEVICE_SYMBOL_MAX,
      use_gfx1201_wave_provider
          ? (encoding == SLLM_HIP_KV_ENCODING_FP16_V1
                 ? ::sllm_causal_attention_kernel::kGfx1201WaveDeviceSymbol
                 : ::sllm_causal_attention_kernel::
                       kGfx1201WavePackedKvDeviceSymbol)
          : (encoding == SLLM_HIP_KV_ENCODING_FP16_V1
                 ? ::sllm_causal_attention_kernel::kDeviceSymbol
                 : ::sllm_causal_attention_kernel::kPackedKvDeviceSymbol));
  sllm_public_runtime::copy_fixed_string(info->gcn_arch_name,
                                         SLLM_HIP_MAX_GCN_ARCH_NAME, arch_name);
}

} // namespace

extern "C" sllm_status_t sllm_causal_attention_execute(
    const sllm_context_t *const raw_context,
    const sllm_queue_t *const raw_queue,
    const sllm_causal_attention_desc_t *const descriptor,
    sllm_completion_t **const completion_output,
    sllm_causal_attention_dispatch_info_t *const dispatch_info,
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
    const sllm_status_t info_status =
        validate_causal_attention_dispatch_info(dispatch_info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    if (raw_context == nullptr || raw_queue == nullptr ||
        descriptor == nullptr || completion_output == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "causal attention context, queue, descriptor, or completion output "
          "is null");
    }
    sllm_causal_attention::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_causal_attention::validate_and_copy_descriptor(
            descriptor, &metadata, error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }

    Context *context = nullptr;
    Queue *queue = nullptr;
    KvState *state = nullptr;
    Buffer *query = nullptr;
    Buffer *output = nullptr;
    uint64_t dispatch_id = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      context = lookup<Context>(raw_context, HandleKind::Context);
      queue = lookup<Queue>(raw_queue, HandleKind::Queue);
      state = lookup<KvState>(descriptor->kv_state, HandleKind::KvState);
      query = lookup<Buffer>(descriptor->query.buffer, HandleKind::Buffer);
      output = lookup<Buffer>(descriptor->output.buffer, HandleKind::Buffer);
      if (context == nullptr || queue == nullptr || state == nullptr ||
          query == nullptr || output == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "causal attention handle is stale or has the wrong kind");
      }
      if (context != queue->context || context != state->context ||
          query->context != context || output->context != context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
            "causal attention resources must share one context");
      }
      const CausalAttentionBuffers resolved = {query, output};
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (context->poisoned.load() || context->release_active ||
          queue->release_active || state->release_active ||
          query->release_active || output->release_active ||
          state->transition_token != 0U ||
          state->accounting.active_submissions != 0U ||
          state->accounting.completion_references != 0U) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_CAUSAL_ATTENTION_STATE_BUSY,
            "causal attention requires an immutable committed KV snapshot");
      }
      if (metadata.expected_kv_length != state->published_length ||
          metadata.expected_kv_length > state->capacity_tokens ||
          metadata.start_position + metadata.query_count !=
              state->published_length) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_CAUSAL_ATTENTION_LENGTH_MISMATCH,
            "causal attention length does not equal the committed KV snapshot");
      }
      const uint32_t gqa_ratio = metadata.query.q_heads / state->head_count;
      if (metadata.query.q_heads % state->head_count != 0U ||
          (gqa_ratio != 2U && gqa_ratio != 4U && gqa_ratio != 8U &&
           gqa_ratio != 16U) ||
          metadata.query.head_dim != state->head_dim) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_SHAPE_MISMATCH,
            "causal attention query layout differs from the KV state layout");
      }
      if (metadata.query.end_offset > query->size_bytes ||
          metadata.output.end_offset > output->size_bytes) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
            "causal attention tensor interval exceeds its backing buffer");
      }
      if (causal_intervals_overlap(query, metadata.query.byte_offset,
                                   metadata.query.end_offset, output,
                                   metadata.output.byte_offset,
                                   metadata.output.end_offset) ||
          causal_intervals_overlap(query, metadata.query.byte_offset,
                                   metadata.query.end_offset, state->key_buffer,
                                   0U, state->key_buffer->size_bytes) ||
          causal_intervals_overlap(
              query, metadata.query.byte_offset, metadata.query.end_offset,
              state->value_buffer, 0U, state->value_buffer->size_bytes) ||
          causal_intervals_overlap(
              output, metadata.output.byte_offset, metadata.output.end_offset,
              state->key_buffer, 0U, state->key_buffer->size_bytes) ||
          causal_intervals_overlap(
              output, metadata.output.byte_offset, metadata.output.end_offset,
              state->value_buffer, 0U, state->value_buffer->size_bytes)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_ALIAS_OVERLAP,
            "causal attention Q/output must not alias KV storage or each "
            "other");
      }
      if (context->next_dispatch_id == 0U) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "causal attention context dispatch id is exhausted");
      }
      dispatch_id = context->next_dispatch_id;
      context->next_dispatch_id =
          dispatch_id == std::numeric_limits<uint64_t>::max()
              ? 0U
              : dispatch_id + 1U;
      if (!sllm_public_runtime::AccountingState::reserve_causal_attention(
              context->accounting, queue->accounting, state->accounting,
              query->accounting, output->accounting)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "causal attention submission accounting is exhausted");
      }
      dispatch_info->dispatch_id = dispatch_id;
      (void)resolved;
    }

    std::unique_ptr<Completion> candidate;
    NativeEventGuard event_guard;
    CausalAttentionExecuteScopeGuard execute_guard(
        state, queue, {query, output}, &candidate, &event_guard, error_sink);
    const sllm_status_t device_status =
        select_context_device(context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      execute_guard.disarm();
      (void)rollback_reserved_causal_attention(state, queue, {query, output},
                                               error_sink);
      return device_status;
    }
    hipDeviceProp_t properties = {};
    const sllm_status_t property_status =
        get_device_properties(context->device_index, &properties, error_sink);
    if (property_status != SLLM_STATUS_OK) {
      execute_guard.disarm();
      (void)rollback_reserved_causal_attention(state, queue, {query, output},
                                               error_sink);
      return property_status;
    }
    char arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME] = {};
    if (!copy_property(arch_name, sizeof(arch_name), properties.gcnArchName,
                       sizeof(properties.gcnArchName))) {
      execute_guard.disarm();
      (void)rollback_reserved_causal_attention(state, queue, {query, output},
                                               error_sink);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
          "HIP gcnArchName is not NUL terminated or is too long");
    }
#if defined(SLLM_HIP_COMPILE_TARGET)
    if (!matches_runtime_gcn_arch(arch_name, SLLM_HIP_COMPILE_TARGET)) {
      execute_guard.disarm();
      (void)rollback_reserved_causal_attention(state, queue, {query, output},
                                               error_sink);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
          "causal attention device target does not match the compiled exact "
          "target");
    }
#endif
    try {
      candidate = std::make_unique<Completion>(context, queue, nullptr, 0U,
                                               false, std::vector<uint8_t>{});
      candidate->causal_attention = true;
      candidate->causal_attention_state = state;
      candidate->causal_attention_buffers = {query, output};
      execute_guard.candidate_allocated();
    } catch (...) {
      execute_guard.disarm();
      (void)rollback_reserved_causal_attention(state, queue, {query, output},
                                               error_sink);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "causal attention completion allocation failed before enqueue");
    }
    hipEvent_t native_event = nullptr;
    const hipError_t event_status = queue_profiles_completions(queue)
                                        ? hipEventCreateWithFlags(&native_event, 0U)
                                        : hipSuccess;
    if (event_status != hipSuccess) {
      execute_guard.disarm();
      (void)rollback_reserved_causal_attention(state, queue, {query, output},
                                               error_sink);
      return hip_failure(error_sink, event_status, "hipEventCreateWithFlags");
    }
    event_guard.adopt(context, native_event);
    candidate->event = native_event;
    hipEvent_t timing_start_event = nullptr;
    const hipError_t timing_status = queue_profiles_completions(queue)
                                         ? hipEventCreateWithFlags(
                                               &timing_start_event, 0U)
                                         : hipSuccess;
    if (timing_status != hipSuccess) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "causal attention timing event creation failed", error_sink);
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
          "causal attention completion registry allocation failed", error_sink);
    }
    if (token == 0U) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "causal attention completion token allocation failed", error_sink);
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
      return cleanup_failed_submission(
          candidate, token, timing_record_status,
          "hipEventRecord causal attention timing start", queue, error_sink);
    }
    const auto byte_pointer = [](Buffer *const buffer,
                                 const uint64_t offset) -> void * {
      return static_cast<char *>(buffer->device_pointer) +
             static_cast<std::size_t>(offset);
    };
    const uint16_t *const query_pointer = static_cast<const uint16_t *>(
        byte_pointer(query, metadata.query.byte_offset));
    uint16_t *const output_pointer = static_cast<uint16_t *>(
        byte_pointer(output, metadata.output.byte_offset));
    const hipError_t launch_status = ::sllm_causal_attention_kernel::launch(
        query_pointer, state->key_buffer->device_pointer,
        state->value_buffer->device_pointer, state->key_scale_plane.address,
        state->value_scale_plane.address,
        static_cast<const float *>(state->key_outer_scale_plane.address),
        static_cast<const float *>(state->value_outer_scale_plane.address),
        output_pointer, static_cast<uint32_t>(metadata.query_count),
        state->capacity_tokens, metadata.start_position,
        metadata.expected_kv_length, metadata.query.q_heads, state->head_count,
        state->head_dim, state->encoding,
        std::strcmp(arch_name, "gfx1201") == 0 &&
            (metadata.query_count == 1U || metadata.query_count >= 32U),
        queue->stream);
    if (launch_status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, launch_status,
                                       "causal attention kernel launch", queue,
                                       error_sink);
    }
    const hipError_t record_status =
        candidate->event != nullptr
            ? hipEventRecord(candidate->event, queue->stream)
            : hipSuccess;
    if (record_status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, record_status,
                                       "hipEventRecord causal attention", queue,
                                       error_sink);
    }
    fill_causal_attention_dispatch_info(dispatch_info, dispatch_id, metadata,
                                        state->head_count, state->encoding,
                                        arch_name);
    *completion_output = reinterpret_cast<sllm_completion_t *>(token);
    (void)candidate.release();
    execute_guard.disarm();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in causal attention execute");
  }
}
