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

constexpr uint32_t kDecodeGqa6SplitP32Partitions = 32U;
constexpr uint32_t kDecodeGqa6SplitP32QHeads = 24U;
constexpr uint32_t kDecodeGqa6SplitP32KvHeads = 4U;
constexpr uint32_t kDecodeGqa6SplitP32HeadDim = 256U;
constexpr uint32_t kDecodeGqa6SplitP32WorkgroupSize = 192U;
constexpr uint64_t kDecodeGqa6SplitP32WorkspaceBytes =
    static_cast<uint64_t>(kDecodeGqa6SplitP32QHeads) *
    kDecodeGqa6SplitP32Partitions * (kDecodeGqa6SplitP32HeadDim + 2U) *
    sizeof(float);
constexpr uint32_t kDecodeGqa6SplitP128Partitions = 128U;
constexpr uint32_t kDecodeGqa6SplitP128QHeads = 24U;
constexpr uint32_t kDecodeGqa6SplitP128KvHeads = 4U;
constexpr uint32_t kDecodeGqa6SplitP128HeadDim = 256U;
constexpr uint32_t kDecodeGqa6SplitP128WorkgroupSize = 192U;
constexpr uint64_t kDecodeGqa6SplitP128WorkspaceBytes =
    static_cast<uint64_t>(kDecodeGqa6SplitP128QHeads) *
    kDecodeGqa6SplitP128Partitions * (kDecodeGqa6SplitP128HeadDim + 2U) *
    sizeof(float);

bool paged_decode_workspace_bytes(const uint32_t query_count,
                                  const uint64_t committed_kv_length,
                                  uint64_t *const result) noexcept {
  if (result == nullptr || query_count == 0U) {
    return false;
  }
  const uint64_t splits = committed_kv_length >= 8192U ? 128U : 32U;
  const uint64_t per_query = static_cast<uint64_t>(kDecodeGqa6SplitP32QHeads) *
                             splits * (kDecodeGqa6SplitP32HeadDim + 2U) *
                             sizeof(float);
  if (per_query == 0U ||
      static_cast<uint64_t>(query_count) > UINT64_MAX / per_query) {
    return false;
  }
  *result = static_cast<uint64_t>(query_count) * per_query;
  return *result <=
         static_cast<uint64_t>(std::numeric_limits<std::size_t>::max());
}
// Acquire the context-owned scaled-prefill workspace cache so requests reuse
// one allocation and a resize cannot race an in-flight submission.
hipError_t prepare_causal_workspace_plan_for_bytes(
    Context *const context, Queue *const queue, const uint64_t required,
    void **const workspace, uint64_t *const workspace_bytes) noexcept {
  if (context == nullptr || queue == nullptr || workspace == nullptr ||
      workspace_bytes == nullptr) {
    return hipErrorInvalidValue;
  }
  if (required == 0U ||
      required >
          static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return hipErrorOutOfMemory;
  }
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
  (void)required;
  return hipErrorNotSupported;
#else
  std::lock_guard<std::mutex> plan_lock(
      context->causal_scaled_prefill_plan_mutex);
  if (context->poisoned.load()) {
    return hipErrorContextIsDestroyed;
  }
  {
    std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
    if (context->causal_scaled_prefill_in_flight != 0U &&
        context->causal_scaled_prefill_owner_queue != queue) {
      return hipErrorNotReady;
    }
    if (context->causal_scaled_prefill_workspace != nullptr &&
        context->causal_scaled_prefill_workspace_bytes >= required) {
      if (context->causal_scaled_prefill_owner_queue == nullptr) {
        context->causal_scaled_prefill_owner_queue = queue;
      }
      if (context->causal_scaled_prefill_in_flight == UINT64_MAX) {
        return hipErrorOutOfMemory;
      }
      ++context->causal_scaled_prefill_in_flight;
      *workspace = context->causal_scaled_prefill_workspace;
      *workspace_bytes = context->causal_scaled_prefill_workspace_bytes;
      return hipSuccess;
    }
    if (context->causal_scaled_prefill_in_flight != 0U) {
      return hipErrorNotReady;
    }
  }
  void *new_workspace = nullptr;
  const hipError_t allocation_status =
      hipMalloc(&new_workspace, static_cast<std::size_t>(required));
  if (allocation_status != hipSuccess) {
    return allocation_status;
  }
  NativeAllocationGuard new_guard(context, new_workspace);
  void *const old_workspace = context->causal_scaled_prefill_workspace;
  if (old_workspace != nullptr) {
    const hipError_t free_status =
        free_allocation_with_fault_injection(old_workspace);
    if (free_status != hipSuccess) {
      context->causal_scaled_prefill_workspace = nullptr;
      context->causal_scaled_prefill_workspace_bytes = 0U;
      context->causal_scaled_prefill_owner_queue = nullptr;
      context->poisoned.store(true);
      orphan_owner.retain_allocation(context, old_workspace);
      (void)new_guard.cleanup();
      return free_status;
    }
    context->causal_scaled_prefill_workspace = nullptr;
    context->causal_scaled_prefill_workspace_bytes = 0U;
  }
  context->causal_scaled_prefill_workspace = new_workspace;
  context->causal_scaled_prefill_workspace_bytes = required;
  context->causal_scaled_prefill_workspace_high_water_bytes = std::max(
      context->causal_scaled_prefill_workspace_high_water_bytes, required);
  new_guard.release();
  std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
  if (context->poisoned.load() ||
      context->causal_scaled_prefill_in_flight != 0U) {
    return hipErrorNotReady;
  }
  context->causal_scaled_prefill_owner_queue = queue;
  ++context->causal_scaled_prefill_in_flight;
  *workspace = context->causal_scaled_prefill_workspace;
  *workspace_bytes = context->causal_scaled_prefill_workspace_bytes;
  return hipSuccess;
#endif
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
  const uint32_t struct_size = info->struct_size;
  const uint32_t abi_version = info->abi_version;
  std::memset(info, 0, sizeof(*info));
  info->struct_size = struct_size;
  info->abi_version = abi_version;
  info->info_version = SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION;
  info->backend = SLLM_BACKEND_HIP;
  info->dispatch_id = dispatch_id;
  info->dispatch_count = 1U;
  info->kernel_id = 0U;
  info->workgroup_size_x = SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE;
  info->grid_size_x = static_cast<uint32_t>(
      static_cast<uint64_t>(metadata.query_count) * metadata.query.q_heads);
  info->query_count = metadata.query_count;
  info->start_position = metadata.start_position;
  info->committed_kv_length = metadata.expected_kv_length;
  info->q_heads = metadata.query.q_heads;
  info->kv_heads = kv_heads;
  info->head_dim = metadata.query.head_dim;
  const uint32_t implicit_denominator = static_cast<uint32_t>(
      std::sqrt(static_cast<float>(metadata.query.head_dim)));
  const bool integer_implicit_scale =
      static_cast<uint64_t>(implicit_denominator) * implicit_denominator ==
      metadata.query.head_dim;
  info->scale_denominator =
      metadata.explicit_score_scale || !integer_implicit_scale
          ? 0U
          : implicit_denominator;
  info->fallback_allowed = 0U;
  info->fallback_used = 0U;
  const bool sliding = metadata.sliding_window != 0U;
  const uint64_t retained_start =
      metadata.expected_kv_length > metadata.sliding_window && sliding
          ? metadata.expected_kv_length - metadata.sliding_window
          : 0U;
  info->reserved[0] = static_cast<uint32_t>(metadata.sliding_window);
  info->reserved[1] =
      static_cast<uint32_t>(metadata.sliding_window >> UINT32_C(32));
  info->reserved[2] = static_cast<uint32_t>(retained_start);
  info->reserved[3] = static_cast<uint32_t>(retained_start >> UINT32_C(32));
  if (metadata.explicit_score_scale || !integer_implicit_scale) {
    const float published_scale =
        metadata.explicit_score_scale
            ? metadata.score_scale
            : 1.0F / std::sqrt(static_cast<float>(metadata.query.head_dim));
    std::memcpy(&info->reserved[4], &published_scale, sizeof(float));
    info->reserved[5] = 1U;
  }
  sllm_public_runtime::copy_fixed_string(
      info->gcn_arch_name, SLLM_HIP_MAX_GCN_ARCH_NAME, arch_name);
  (void)encoding;
}

} // namespace

#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
extern "C" uint32_t sllm_test_select_causal_attention_providers_with_position(
    const uint64_t expected_kv_length, const uint32_t query_count,
    const uint32_t query_heads, const uint32_t kv_heads,
    const uint32_t head_dim, const uint32_t encoding,
    const uint64_t start_position, const uint64_t sliding_window,
    const uint32_t explicit_score_scale, const char *const arch_name) noexcept {
  (void)expected_kv_length; (void)query_count; (void)query_heads;
  (void)kv_heads; (void)head_dim; (void)encoding; (void)start_position;
  (void)sliding_window; (void)explicit_score_scale; (void)arch_name;
  return 0U;
}
extern "C" uint32_t sllm_test_select_causal_attention_providers_with_semantics(
    const uint64_t expected_kv_length, const uint32_t query_count,
    const uint32_t query_heads, const uint32_t kv_heads,
    const uint32_t head_dim, const uint32_t encoding,
    const uint64_t sliding_window, const uint32_t explicit_score_scale,
    const char *const arch_name) noexcept {
  (void)expected_kv_length; (void)query_count; (void)query_heads;
  (void)kv_heads; (void)head_dim; (void)encoding; (void)sliding_window;
  (void)explicit_score_scale; (void)arch_name;
  return 0U;
}
extern "C" uint32_t sllm_test_select_causal_attention_providers(
    const uint64_t expected_kv_length, const uint32_t query_count,
    const uint32_t query_heads, const uint32_t kv_heads,
    const uint32_t head_dim, const uint32_t encoding,
    const char *const arch_name) noexcept {
  (void)expected_kv_length; (void)query_count; (void)query_heads;
  (void)kv_heads; (void)head_dim; (void)encoding; (void)arch_name;
  return 0U;
}
extern "C" uint32_t sllm_test_select_causal_attention_gqa4(
    const uint64_t expected_kv_length, const uint32_t query_count,
    const uint32_t query_heads, const uint32_t kv_heads,
    const uint32_t head_dim, const uint32_t encoding,
    const char *const arch_name) noexcept {
  (void)expected_kv_length; (void)query_count; (void)query_heads;
  (void)kv_heads; (void)head_dim; (void)encoding; (void)arch_name;
  return 0U;
}
#endif

/* Shared implementation for the ordinary immutable-snapshot route and the
 * additive append->attention chain.  A non-null dependency is already
 * claimed by the chain entry point and permits the causal dispatch to read
 * the append's private transition before it is published. */
static sllm_status_t causal_attention_execute_impl(
    const sllm_context_t *const raw_context,
    const sllm_queue_t *const raw_queue,
    const sllm_causal_attention_desc_t *const descriptor,
    sllm_completion_t **const completion_output,
    sllm_causal_attention_dispatch_info_t *const dispatch_info,
    sllm_error_sink_t *const error_sink,
    Completion *const dependency) noexcept {
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
    uint64_t submission_serial = 0U;
    const bool chained = dependency != nullptr;
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
      // The append claim is mutable until its dependent attention is
      // admitted.  Keep its state lock while validating the exact claim so
      // finalize/cancel cannot race these fields after the dependency was
      // pinned by the chain entry point.
      std::unique_lock<std::mutex> dependency_lock;
      if (chained) {
        dependency_lock = std::unique_lock<std::mutex>(dependency->state_mutex);
      }
      const bool whole_decode_control_capture =
          graph_capture_queue == queue && whole_graph_capture_active();
      const CausalAttentionBuffers resolved = {query, output};
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (context->poisoned.load() || context->release_active ||
          queue->release_active || state->release_active ||
          query->release_active || output->release_active ||
          (!chained && state->transition_token != 0U &&
           !whole_decode_control_capture) ||
          (!chained && state->accounting.active_submissions != 0U &&
           !whole_decode_control_capture) ||
          (!chained && state->accounting.completion_references != 0U &&
           !whole_decode_control_capture)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_CAUSAL_ATTENTION_STATE_BUSY,
            "causal attention requires an immutable committed KV snapshot");
      }
      const bool range_valid =
          !chained
              ? (whole_decode_control_capture
                     ? metadata.query_count != 0U &&
                           metadata.query_count <= 9U &&
                           metadata.start_position >= state->published_length &&
                           metadata.start_position - state->published_length <=
                               8U &&
                           metadata.start_position <=
                               UINT64_MAX - metadata.query_count
                     : metadata.expected_kv_length == state->published_length &&
                           metadata.expected_kv_length <=
                               state->capacity_tokens &&
                           metadata.start_position + metadata.query_count ==
                               state->published_length)
              : dependency->kv_state == state &&
                    dependency->context == context &&
                    dependency->queue == queue && dependency->kv_state_append &&
                    !dependency->terminal &&
                    dependency->dependent_attention_claimed &&
                    dependency->kv_append_token != 0U &&
                    state->transition_token == dependency->kv_append_token &&
                    state->transition_start == dependency->kv_append_start &&
                    state->transition_count == dependency->kv_append_count &&
                    state->transition_end == dependency->kv_append_end &&
                    metadata.start_position == dependency->kv_append_start &&
                    metadata.query_count == dependency->kv_append_count &&
                    metadata.expected_kv_length == dependency->kv_append_end &&
                    state->published_length == dependency->kv_append_start &&
                    metadata.expected_kv_length <= state->capacity_tokens;
      if (!range_valid) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_CAUSAL_ATTENTION_LENGTH_MISMATCH,
            chained ? "causal attention does not exactly claim the KV append"
                    : "causal attention length does not equal the committed KV "
                      "snapshot");
      }
      if (metadata.sliding_window != state->sliding_window ||
          (state->sliding_window != 0U &&
           state->retained_start !=
               state->published_length -
                   std::min(state->published_length, state->sliding_window))) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INVALID_CAUSAL_ATTENTION_DESCRIPTOR,
            "causal attention sliding window differs from the KV state");
      }
      if (metadata.explicit_score_scale &&
          (state->encoding != SLLM_HIP_KV_ENCODING_FP8_STATIC_V1 ||
           state->static_key_scale != 1.0F ||
           state->static_value_scale != 1.0F)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INVALID_CAUSAL_ATTENTION_DESCRIPTOR,
            "explicit score scaling requires unit-scale static FP8 KV");
      }
      const uint32_t gqa_ratio = metadata.query.q_heads / state->head_count;
      if (metadata.query.q_heads % state->head_count != 0U ||
          (gqa_ratio != 2U && gqa_ratio != 4U && gqa_ratio != 6U &&
           gqa_ratio != 8U && gqa_ratio != 16U) ||
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
      if (!next_queue_submission_serial(queue, &submission_serial)) {
        (void)sllm_public_runtime::AccountingState::rollback_causal_attention(
            context->accounting, queue->accounting, state->accounting,
            query->accounting, output->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "causal attention submission serial is exhausted");
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
        get_context_device_properties(context, &properties, error_sink);
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
    bool use_paged_decode_gqa6 = false;
    bool use_paged_decode_gqa6_c1 = false;
    bool use_paged_prefill_gqa6 = false;
    bool use_paged_decode_fp16 = false;
    [[maybe_unused]] bool use_paged_prefill_gqa4 = false;
    bool use_paged_prefill_fp16 = false;
    bool use_paged_generic_formats = false;
    bool use_paged_sliding_static_fp8 = false;
    bool whole_paged_decode_gqa6_capture = false;
    bool whole_paged_decode_gqa6_c1_capture = false;
    bool whole_paged_decode_fp16_capture = false;
#if !defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
    if (state->paged) {
      const bool exact_paged_target = std::strcmp(arch_name, "gfx1030") == 0 ||
                                      std::strcmp(arch_name, "gfx1201") == 0;
      const bool paged_tables_ready =
          state->paged_device_status != nullptr &&
          state->paged_device_pool != nullptr &&
          state->paged_logical_table != nullptr &&
          state->paged_logical_table->device_table() != nullptr &&
          state->paged_device_pool->device_descriptor_table() != nullptr &&
          state->paged_token_block_size == SLLM_HIP_KV_PAGED_TOKEN_BLOCK_SIZE &&
          state->paged_physical_layout_version ==
              SLLM_HIP_KV_PAGED_LAYOUT_VERSION &&
          state->paged_logical_table->size() <=
              std::numeric_limits<uint32_t>::max() &&
          state->paged_device_pool->physical_capacity() != 0U;
      const bool paged_sliding_static_shape =
          state->sliding_window == sllm_paged_kv::kSlidingWindowTokens &&
          state->encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1 &&
          state->head_count != 0U && metadata.query.q_heads != 0U &&
          metadata.query.q_heads % state->head_count == 0U &&
          metadata.query.head_dim == state->head_dim &&
          state->head_dim <= SLLM_HIP_KV_MAX_HEAD_DIM &&
          state->paged_sliding_ring_tags_device != nullptr &&
          state->paged_sliding_host_state != nullptr &&
          metadata.sliding_window == sllm_paged_kv::kSlidingWindowTokens &&
          (!metadata.explicit_score_scale || metadata.score_scale == 1.0F) &&
          metadata.expected_kv_length != 0U;
      const bool paged_mxfp8_gqa6_shape =
          state->encoding == SLLM_HIP_KV_ENCODING_MXFP8_E4_V1 &&
          state->head_count == 4U && state->head_dim == 256U &&
          metadata.query.q_heads == 24U && metadata.query.head_dim == 256U &&
          metadata.sliding_window == 0U && !metadata.explicit_score_scale &&
          metadata.expected_kv_length != 0U;
      const bool paged_fp16_shape =
          state->encoding == SLLM_HIP_KV_ENCODING_FP16_V1 &&
          state->head_count == 4U && state->head_dim == 256U &&
          (metadata.query.q_heads == 16U || metadata.query.q_heads == 24U) &&
          metadata.query.head_dim == 256U && metadata.sliding_window == 0U &&
          !metadata.explicit_score_scale && metadata.expected_kv_length != 0U;
      const bool paged_generic_encoding =
          state->encoding == SLLM_HIP_KV_ENCODING_FP16_V1 ||
          state->encoding == SLLM_HIP_KV_ENCODING_FP8_V1 ||
          state->encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1 ||
          state->encoding == SLLM_HIP_KV_ENCODING_NVFP4_V1 ||
          state->encoding == SLLM_HIP_KV_ENCODING_MXFP8_E4_V1 ||
          (state->encoding == SLLM_HIP_KV_ENCODING_MXFP8_E5_V1 &&
           std::strcmp(arch_name, "gfx1030") == 0);
      const bool paged_generic_shape =
          paged_generic_encoding && state->head_count != 0U &&
          metadata.query.q_heads % state->head_count == 0U &&
          metadata.query.head_dim == state->head_dim &&
          metadata.sliding_window == 0U && metadata.expected_kv_length != 0U;
      whole_paged_decode_gqa6_capture =
          whole_graph_capture_active() && graph_capture_queue == queue &&
          paged_mxfp8_gqa6_shape && metadata.query_count <= 5U;
      whole_paged_decode_gqa6_c1_capture =
          whole_paged_decode_gqa6_capture && metadata.query_count == 3U &&
          metadata.expected_kv_length >= 8192U;
      whole_paged_decode_gqa6_capture =
          whole_paged_decode_gqa6_capture &&
          !whole_paged_decode_gqa6_c1_capture;
      whole_paged_decode_fp16_capture =
          whole_graph_capture_active() && graph_capture_queue == queue &&
          paged_fp16_shape && metadata.query_count <= 5U;
      if (!exact_paged_target || !paged_tables_ready ||
          (!paged_sliding_static_shape && !paged_mxfp8_gqa6_shape &&
           !paged_fp16_shape && !paged_generic_shape)) {
        execute_guard.disarm();
        (void)rollback_reserved_causal_attention(state, queue, {query, output},
                                                 error_sink);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_UNSUPPORTED,
            "paged causal attention format, shape, or exact target is "
            "unsupported");
      }
      if (whole_graph_capture_active() && graph_capture_queue == queue &&
          !whole_paged_decode_gqa6_capture &&
          !whole_paged_decode_gqa6_c1_capture &&
          !whole_paged_decode_fp16_capture) {
        execute_guard.disarm();
        (void)rollback_reserved_causal_attention(state, queue, {query, output},
                                                 error_sink);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_UNSUPPORTED,
            "paged causal attention is not available during graph capture");
      }
      use_paged_decode_gqa6 =
          paged_mxfp8_gqa6_shape && metadata.query_count <= 5U;
      use_paged_decode_gqa6_c1 =
          use_paged_decode_gqa6 && metadata.query_count == 3U &&
          metadata.expected_kv_length >= 8192U;
      use_paged_decode_gqa6 =
          use_paged_decode_gqa6 && !use_paged_decode_gqa6_c1;
      use_paged_prefill_gqa6 = paged_mxfp8_gqa6_shape &&
                               !use_paged_decode_gqa6 &&
                               !use_paged_decode_gqa6_c1;
      use_paged_decode_fp16 = paged_fp16_shape && metadata.query_count <= 5U;
      use_paged_prefill_gqa4 =
          paged_fp16_shape && !use_paged_decode_fp16 &&
          metadata.query_count >= 64U && metadata.query.q_heads == 16U &&
          state->head_count == 4U;
      use_paged_prefill_fp16 =
          paged_fp16_shape && !use_paged_decode_fp16 &&
          !use_paged_prefill_gqa4;
      use_paged_generic_formats =
          paged_generic_shape && !paged_mxfp8_gqa6_shape && !paged_fp16_shape;
      use_paged_sliding_static_fp8 = paged_sliding_static_shape;
    } else {
      execute_guard.disarm();
      (void)rollback_reserved_causal_attention(state, queue, {query, output},
                                               error_sink);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_UNSUPPORTED,
          "legacy contiguous KV attention is unsupported; use Paged KV");
    }
#endif
    void *scaled_prefill_workspace = nullptr;
    uint64_t scaled_prefill_workspace_bytes = 0U;
    try {
      candidate = std::make_unique<Completion>(context, queue, nullptr, 0U,
                                               false, std::vector<uint8_t>{});
      candidate->causal_attention = true;
      candidate->causal_attention_state = state;
      candidate->causal_attention_buffers = {query, output};
      candidate->submission_serial = submission_serial;
      candidate->dependent_append = chained ? dependency : nullptr;
#if !defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
      if (use_paged_decode_gqa6 || use_paged_decode_gqa6_c1 ||
          use_paged_prefill_gqa6 ||
          use_paged_decode_fp16 || use_paged_prefill_gqa4 ||
          use_paged_prefill_fp16 ||
          use_paged_generic_formats || use_paged_sliding_static_fp8) {
        if (!whole_paged_decode_gqa6_capture &&
            !whole_paged_decode_gqa6_c1_capture &&
            !whole_paged_decode_fp16_capture &&
            !arm_completion_device_status(candidate.get(),
                                          state->paged_device_status)) {
          throw std::runtime_error(
              "paged causal attention device status arm failed");
        }
      }
#endif
      execute_guard.candidate_allocated();
    } catch (...) {
      execute_guard.disarm();
      (void)rollback_reserved_causal_attention(state, queue, {query, output},
                                               error_sink);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "causal attention completion allocation failed before enqueue");
    }
    if (use_paged_decode_gqa6 || use_paged_decode_gqa6_c1) {
      uint64_t workspace_bytes = 0U;
      const bool workspace_valid =
          (whole_paged_decode_gqa6_capture ||
           whole_paged_decode_gqa6_c1_capture)
              ? metadata.query_count != 0U && metadata.query_count <= 5U &&
                    static_cast<uint64_t>(metadata.query_count) <=
                        UINT64_MAX / kDecodeGqa6SplitP128WorkspaceBytes &&
                    (workspace_bytes =
                         static_cast<uint64_t>(metadata.query_count) *
                         kDecodeGqa6SplitP128WorkspaceBytes) != 0U
              : paged_decode_workspace_bytes(
                    static_cast<uint32_t>(metadata.query_count),
                    metadata.expected_kv_length, &workspace_bytes);
      if (!workspace_valid) {
        execute_guard.disarm();
        (void)rollback_reserved_causal_attention(state, queue, {query, output},
                                                 error_sink);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_METADATA_OVERFLOW,
            "paged GQA6 decode workspace size is invalid");
      }
      const hipError_t plan_status = prepare_causal_workspace_plan_for_bytes(
          context, queue, workspace_bytes, &scaled_prefill_workspace,
          &scaled_prefill_workspace_bytes);
      if (plan_status != hipSuccess) {
        execute_guard.disarm();
        (void)rollback_reserved_causal_attention(state, queue, {query, output},
                                                 error_sink);
        return hip_failure(error_sink, plan_status,
                           "causal attention paged GQA6 decode workspace "
                           "prepare");
      }
      candidate->causal_scaled_prefill_plan_acquired = true;
    } else if (use_paged_prefill_gqa6 || use_paged_decode_fp16 ||
               use_paged_prefill_fp16 || use_paged_generic_formats ||
               use_paged_sliding_static_fp8) {
      // The paged qtile8 prefill kernel owns its temporary storage and does
      // not consume the legacy contiguous-KV workspace cache.
      scaled_prefill_workspace = nullptr;
      scaled_prefill_workspace_bytes = 0U;
    }
    hipEvent_t native_event = nullptr;
    const hipError_t event_status =
        queue_profiles_completions(queue)
            ? hipEventCreateWithFlags(&native_event, 0U)
            : hipSuccess;
    if (event_status != hipSuccess) {
      execute_guard.disarm();
      if (candidate->causal_scaled_prefill_plan_acquired) {
        std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
        (void)release_causal_scaled_prefill_plan_reference(candidate.get());
      }
      (void)rollback_reserved_causal_attention(state, queue, {query, output},
                                               error_sink);
      return hip_failure(error_sink, event_status, "hipEventCreateWithFlags");
    }
    event_guard.adopt(context, native_event);
    candidate->event = native_event;
    hipEvent_t timing_start_event = nullptr;
    const hipError_t timing_status =
        queue_profiles_completions(queue)
            ? hipEventCreateWithFlags(&timing_start_event, 0U)
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
    [[maybe_unused]] const uint16_t *const query_pointer = static_cast<const uint16_t *>(
        byte_pointer(query, metadata.query.byte_offset));
    [[maybe_unused]] uint16_t *const output_pointer = static_cast<uint16_t *>(
        byte_pointer(output, metadata.output.byte_offset));
    auto *const device_control =
        graph_capture_queue == queue
            ? static_cast<sllm_decode_control::ControlV1 *>(
                  whole_graph_capture_device_control())
            : nullptr;
    if (whole_graph_capture_active() && device_control == nullptr) {
      execute_guard.disarm();
      return cleanup_failed_submission(
          candidate, token, hipErrorInvalidValue,
          "whole-decode attention requires a device control buffer", queue,
          error_sink);
    }
    if (device_control != nullptr && !whole_paged_decode_gqa6_capture &&
        !whole_paged_decode_gqa6_c1_capture &&
        !whole_paged_decode_fp16_capture) {
      execute_guard.disarm();
      return cleanup_failed_submission(
          candidate, token, hipErrorNotSupported,
          "whole-decode attention control supports MXFP8 E4 staged32 only",
          queue, error_sink);
    }
    hipError_t launch_status = hipSuccess;
#if !defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
    if (use_paged_sliding_static_fp8) {
      launch_status = ::sllm_causal_attention_kernel::
          launch_paged_sliding_static_fp8_attention(
              query_pointer, state->paged_logical_table->device_table(),
              state->paged_sliding_ring_tags_device,
              state->paged_device_pool->device_descriptor_table(),
              sllm_paged_kv::kSlidingRingSlots,
              state->paged_device_pool->physical_capacity(),
              state->paged_device_status, output_pointer,
              static_cast<uint32_t>(metadata.query_count),
              metadata.start_position, metadata.expected_kv_length,
              state->retained_start, metadata.query.q_heads, state->head_count,
              state->head_dim, state->static_key_scale,
              state->static_value_scale,
              metadata.explicit_score_scale
                  ? metadata.score_scale
                  : 1.0F / std::sqrt(static_cast<float>(state->head_dim)),
              queue->stream);
    } else if (use_paged_decode_fp16 || use_paged_prefill_gqa4 ||
               use_paged_prefill_fp16) {
      const auto *const table = state->paged_logical_table->device_table();
      const auto *const descriptors =
          state->paged_device_pool->device_descriptor_table();
      const uint32_t logical_count =
          static_cast<uint32_t>(state->paged_logical_table->size());
      const uint32_t physical_count =
          state->paged_device_pool->physical_capacity();
      launch_status =
          use_paged_decode_fp16
              ? (whole_paged_decode_fp16_capture
                     ? ::sllm_causal_attention_kernel::
                           launch_paged_decode_fp16_device(
                               query_pointer, table, descriptors, logical_count,
                               physical_count, state->paged_device_status,
                               output_pointer,
                               static_cast<uint32_t>(metadata.query_count),
                               metadata.query.q_heads, state->head_count,
                               state->head_dim, device_control, queue->stream)
                     : ::sllm_causal_attention_kernel::launch_paged_decode_fp16(
                           query_pointer, table, descriptors, logical_count,
                           physical_count, state->paged_device_status,
                           output_pointer,
                           static_cast<uint32_t>(metadata.query_count),
                           metadata.start_position, metadata.expected_kv_length,
                           metadata.query.q_heads, state->head_count,
                           state->head_dim, queue->stream))
              : use_paged_prefill_gqa4
                    ? ::sllm_causal_attention_kernel::launch_paged_prefill_gqa4(
                          query_pointer, table, descriptors, logical_count,
                          physical_count, state->paged_device_status,
                          output_pointer,
                          static_cast<uint32_t>(metadata.query_count),
                          metadata.start_position, metadata.expected_kv_length,
                          metadata.query.q_heads, state->head_count,
                          state->head_dim, queue->stream)
                    : ::sllm_causal_attention_kernel::launch_paged_prefill_fp16(
                          query_pointer, table, descriptors, logical_count,
                          physical_count, state->paged_device_status,
                          output_pointer,
                          static_cast<uint32_t>(metadata.query_count),
                          metadata.start_position, metadata.expected_kv_length,
                          metadata.query.q_heads, state->head_count,
                          state->head_dim, queue->stream);
    } else if (use_paged_generic_formats) {
      launch_status = ::sllm_causal_attention_kernel::launch_paged_attention(
          query_pointer, state->paged_logical_table->device_table(),
          state->paged_device_pool->device_descriptor_table(),
          static_cast<uint32_t>(state->paged_logical_table->size()),
          state->paged_device_pool->physical_capacity(),
          state->paged_device_status, output_pointer,
          static_cast<uint32_t>(metadata.query_count), metadata.start_position,
          metadata.expected_kv_length, metadata.query.q_heads,
          state->head_count, state->head_dim, state->encoding,
          state->static_key_scale, state->static_value_scale,
          metadata.explicit_score_scale
              ? metadata.score_scale
              : 1.0F / std::sqrt(static_cast<float>(state->head_dim)),
          queue->stream);
    } else if (use_paged_decode_gqa6_c1) {
      const bool c1_capture = whole_paged_decode_gqa6_c1_capture;
      launch_status =
          c1_capture
              ? ::sllm_causal_attention_kernel::
                    launch_paged_decode_gqa6_c1_device(
                        query_pointer,
                        state->paged_logical_table->device_table(),
                        state->paged_device_pool->device_descriptor_table(),
                        static_cast<uint32_t>(
                            state->paged_logical_table->size()),
                        state->paged_device_pool->physical_capacity(),
                        state->paged_device_status,
                        static_cast<float *>(scaled_prefill_workspace),
                        scaled_prefill_workspace_bytes, output_pointer,
                        static_cast<uint32_t>(metadata.query_count),
                        metadata.start_position, metadata.expected_kv_length,
                        metadata.query.q_heads, state->head_count,
                        state->head_dim, state->encoding, device_control,
                        queue->stream)
              : ::sllm_causal_attention_kernel::launch_paged_decode_gqa6_c1(
                    query_pointer,
                    state->paged_logical_table->device_table(),
                    state->paged_device_pool->device_descriptor_table(),
                    static_cast<uint32_t>(state->paged_logical_table->size()),
                    state->paged_device_pool->physical_capacity(),
                    state->paged_device_status,
                    static_cast<float *>(scaled_prefill_workspace),
                    scaled_prefill_workspace_bytes, output_pointer,
                    static_cast<uint32_t>(metadata.query_count),
                    metadata.start_position, metadata.expected_kv_length,
                    metadata.query.q_heads, state->head_count, state->head_dim,
                    state->encoding, queue->stream);
    } else if (use_paged_decode_gqa6) {
      const bool use_gqa_shared = std::strcmp(arch_name, "gfx1030") == 0;
      launch_status =
          whole_paged_decode_gqa6_capture
              ? ::sllm_causal_attention_kernel::launch_paged_decode_gqa6_device(
                    query_pointer, state->paged_logical_table->device_table(),
                    state->paged_device_pool->device_descriptor_table(),
                    static_cast<uint32_t>(state->paged_logical_table->size()),
                    state->paged_device_pool->physical_capacity(),
                    state->paged_device_status,
                    static_cast<float *>(scaled_prefill_workspace),
                    scaled_prefill_workspace_bytes, output_pointer,
                    static_cast<uint32_t>(metadata.query_count),
                    metadata.start_position, metadata.expected_kv_length,
                    metadata.query.q_heads, state->head_count, state->head_dim,
                    state->encoding, use_gqa_shared, device_control,
                    queue->stream)
              : ::sllm_causal_attention_kernel::launch_paged_decode_gqa6(
                    query_pointer, state->paged_logical_table->device_table(),
                    state->paged_device_pool->device_descriptor_table(),
                    static_cast<uint32_t>(state->paged_logical_table->size()),
                    state->paged_device_pool->physical_capacity(),
                    state->paged_device_status,
                    static_cast<float *>(scaled_prefill_workspace),
                    scaled_prefill_workspace_bytes, output_pointer,
                    static_cast<uint32_t>(metadata.query_count),
                    metadata.start_position, metadata.expected_kv_length,
                    metadata.query.q_heads, state->head_count, state->head_dim,
                    state->encoding, use_gqa_shared, queue->stream);
    } else if (use_paged_prefill_gqa6) {
      const bool gfx1201 = std::strcmp(arch_name, "gfx1201") == 0;
      const bool wave_local_kv = gfx1201 && metadata.query_count == 2048U;
      const bool use_gfx1201_wave_provider =
          gfx1201 && metadata.query_count < 128U;
      const bool use_gfx1201_qtile4_provider = gfx1201 &&
                                               metadata.query_count >= 128U &&
                                               metadata.start_position < 1024U;
      launch_status = ::sllm_causal_attention_kernel::launch_paged_prefill_gqa6(
          query_pointer, state->paged_logical_table->device_table(),
          state->paged_device_pool->device_descriptor_table(),
          static_cast<uint32_t>(state->paged_logical_table->size()),
          state->paged_device_pool->physical_capacity(),
          state->paged_device_status, output_pointer,
          static_cast<uint32_t>(metadata.query_count), metadata.start_position,
          metadata.expected_kv_length, metadata.query.q_heads,
          state->head_count, state->head_dim, state->encoding,
          state->static_key_scale, state->static_value_scale, wave_local_kv,
          use_gfx1201_wave_provider, use_gfx1201_qtile4_provider,
          queue->stream);
    } else {
      launch_status = hipErrorNotSupported;
    }
#else
    launch_status = hipErrorNotSupported;
#endif
    if (launch_status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, launch_status,
                                       "causal attention kernel launch", queue,
                                       error_sink);
    }
#if !defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
    if (use_paged_decode_gqa6 || use_paged_decode_gqa6_c1 ||
        use_paged_prefill_gqa6 ||
          use_paged_decode_fp16 || use_paged_prefill_gqa4 ||
          use_paged_prefill_fp16 ||
        use_paged_generic_formats || use_paged_sliding_static_fp8) {
      /* Whole capture keeps the status word device-resident.  The paged
       * device-control kernels publish invalid-table failures directly to
       * ControlV1, and inserting a D2H node here would make the captured
       * graph host-dependent.  Eager paged submissions retain the ordinary
       * asynchronous status readback and completion validation. */
      if (!whole_paged_decode_gqa6_capture &&
          !whole_paged_decode_gqa6_c1_capture &&
          !whole_paged_decode_fp16_capture) {
        const hipError_t status_copy = enqueue_completion_device_status_copy(
            candidate.get(), queue->stream);
        if (status_copy != hipSuccess) {
          execute_guard.disarm();
          return cleanup_failed_submission(
              candidate, token, status_copy,
              "paged causal attention device status readback", queue,
              error_sink);
        }
      }
    }
#endif
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
#if !defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
    if (use_paged_decode_gqa6 || use_paged_decode_gqa6_c1 ||
        use_paged_prefill_gqa6) {
      const bool use_gqa_shared = std::strcmp(arch_name, "gfx1030") == 0;
      const bool paged_gfx1201_wave = !use_paged_decode_gqa6 &&
                                      std::strcmp(arch_name, "gfx1201") == 0 &&
                                      metadata.query_count < 128U;
      const bool paged_gfx1201_qtile4 =
          !use_paged_decode_gqa6 && std::strcmp(arch_name, "gfx1201") == 0 &&
          metadata.query_count >= 128U && metadata.start_position < 1024U;
      const uint32_t splits = metadata.expected_kv_length >= 8192U ? 128U : 32U;
      dispatch_info->dispatch_count =
          (use_paged_decode_gqa6 || use_paged_decode_gqa6_c1) ? 2U : 1U;
      dispatch_info->kernel_id =
          use_paged_decode_gqa6_c1
              ? SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_DECODE_GQA6_C1_M3_V1
          : use_paged_decode_gqa6
              ? SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_DECODE_GQA6_V1
              : SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_PREFILL_GQA6_V1;
      dispatch_info->workgroup_size_x =
          (use_paged_decode_gqa6 || use_paged_decode_gqa6_c1)
              ? (use_paged_decode_gqa6_c1 ? 192U
                                          : (use_gqa_shared ? 192U : 32U))
          : (paged_gfx1201_wave || paged_gfx1201_qtile4) ? 256U
                                                         : 512U;
      dispatch_info->grid_size_x =
          use_paged_decode_gqa6_c1
              ? static_cast<uint32_t>(4U * splits)
          : use_paged_decode_gqa6
              ? static_cast<uint32_t>(
                    static_cast<uint64_t>(metadata.query_count) *
                    (use_gqa_shared ? state->head_count
                                    : metadata.query.q_heads) *
                    splits)
          : (paged_gfx1201_wave || paged_gfx1201_qtile4)
              ? static_cast<uint32_t>(
                    static_cast<uint64_t>(metadata.query_count) *
                    metadata.query.q_heads)
              : static_cast<uint32_t>((metadata.query_count + 7U) / 8U *
                                      state->head_count);
      sllm_public_runtime::copy_fixed_string(
          dispatch_info->kernel_symbol,
          SLLM_HIP_CAUSAL_ATTENTION_KERNEL_SYMBOL_MAX,
          use_paged_decode_gqa6_c1
              ? ::sllm_causal_attention_kernel::
                    kPagedDecodeGqa6C1M3LogicalKernelId
          : use_paged_decode_gqa6
              ? ::sllm_causal_attention_kernel::kPagedDecodeGqa6LogicalKernelId
          : paged_gfx1201_wave   ? ::sllm_causal_attention_kernel::
                                       kPagedPrefillGqa6Gfx1201WaveLogicalKernelId
          : paged_gfx1201_qtile4 ? ::sllm_causal_attention_kernel::
                                       kPagedPrefillGqa6QTile4LogicalKernelId
                                 : ::sllm_causal_attention_kernel::
                                       kPagedPrefillGqa6LogicalKernelId);
      sllm_public_runtime::copy_fixed_string(
          dispatch_info->device_symbol,
          SLLM_HIP_CAUSAL_ATTENTION_DEVICE_SYMBOL_MAX,
          use_paged_decode_gqa6_c1
              ? ::sllm_causal_attention_kernel::
                    kPagedDecodeGqa6C1M3DeviceSymbol
          : use_paged_decode_gqa6
              ? ::sllm_causal_attention_kernel::kPagedDecodeGqa6DeviceSymbol
          : paged_gfx1201_wave ? ::sllm_causal_attention_kernel::
                                     kPagedPrefillGqa6Gfx1201WaveDeviceSymbol
          : paged_gfx1201_qtile4
              ? ::sllm_causal_attention_kernel::
                    kPagedPrefillGqa6QTile4DeviceSymbol
              : ::sllm_causal_attention_kernel::kPagedPrefillGqa6DeviceSymbol);
    } else if (use_paged_decode_fp16 || use_paged_prefill_gqa4 ||
               use_paged_prefill_fp16) {
      dispatch_info->dispatch_count = 1U;
      dispatch_info->kernel_id =
          use_paged_decode_fp16
              ? SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_DECODE_FP16_V1
              : SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_PREFILL_FP16_V1;
      dispatch_info->workgroup_size_x =
          SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE;
      dispatch_info->grid_size_x =
          static_cast<uint32_t>(metadata.query_count * metadata.query.q_heads);
      sllm_public_runtime::copy_fixed_string(
          dispatch_info->kernel_symbol,
          SLLM_HIP_CAUSAL_ATTENTION_KERNEL_SYMBOL_MAX,
          use_paged_decode_fp16
              ? ::sllm_causal_attention_kernel::kPagedDecodeFp16LogicalKernelId
          : use_paged_prefill_gqa4
              ? ::sllm_causal_attention_kernel::
                    kPagedPrefillGqa4LogicalKernelId
              : ::sllm_causal_attention_kernel::
                    kPagedPrefillFp16LogicalKernelId);
      sllm_public_runtime::copy_fixed_string(
          dispatch_info->device_symbol,
          SLLM_HIP_CAUSAL_ATTENTION_DEVICE_SYMBOL_MAX,
          use_paged_decode_fp16
              ? ::sllm_causal_attention_kernel::kPagedDecodeFp16DeviceSymbol
          : use_paged_prefill_gqa4
              ? ::sllm_causal_attention_kernel::kPagedPrefillGqa4DeviceSymbol
              : ::sllm_causal_attention_kernel::kPagedPrefillFp16DeviceSymbol);
    } else if (use_paged_sliding_static_fp8) {
      dispatch_info->dispatch_count = 1U;
      dispatch_info->kernel_id =
          SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_SLIDING_STATIC_FP8_V1;
      dispatch_info->workgroup_size_x =
          SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE;
      dispatch_info->grid_size_x = static_cast<uint32_t>(
          static_cast<uint64_t>(metadata.query_count) * metadata.query.q_heads);
      sllm_public_runtime::copy_fixed_string(
          dispatch_info->kernel_symbol,
          SLLM_HIP_CAUSAL_ATTENTION_KERNEL_SYMBOL_MAX,
          ::sllm_causal_attention_kernel::kPagedSlidingStaticFp8LogicalKernelId);
      sllm_public_runtime::copy_fixed_string(
          dispatch_info->device_symbol,
          SLLM_HIP_CAUSAL_ATTENTION_DEVICE_SYMBOL_MAX,
          ::sllm_causal_attention_kernel::kPagedSlidingStaticFp8DeviceSymbol);
    } else if (use_paged_generic_formats) {
      dispatch_info->dispatch_count = 1U;
      dispatch_info->kernel_id =
          SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_GENERIC_FORMATS_V1;
      dispatch_info->workgroup_size_x =
          SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE;
      dispatch_info->grid_size_x =
          static_cast<uint32_t>(metadata.query_count * metadata.query.q_heads);
      sllm_public_runtime::copy_fixed_string(
          dispatch_info->kernel_symbol,
          SLLM_HIP_CAUSAL_ATTENTION_KERNEL_SYMBOL_MAX,
          ::sllm_causal_attention_kernel::
              kPagedAttentionFormatsLogicalKernelId);
      sllm_public_runtime::copy_fixed_string(
          dispatch_info->device_symbol,
          SLLM_HIP_CAUSAL_ATTENTION_DEVICE_SYMBOL_MAX,
          ::sllm_causal_attention_kernel::kPagedAttentionFormatsDeviceSymbol);
    }
#endif
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

extern "C" sllm_status_t sllm_causal_attention_execute(
    const sllm_context_t *const raw_context,
    const sllm_queue_t *const raw_queue,
    const sllm_causal_attention_desc_t *const descriptor,
    sllm_completion_t **const completion_output,
    sllm_causal_attention_dispatch_info_t *const dispatch_info,
    sllm_error_sink_t *const error_sink) noexcept {
  return causal_attention_execute_impl(raw_context, raw_queue, descriptor,
                                       completion_output, dispatch_info,
                                       error_sink, nullptr);
}

/* Enqueue attention behind one already-submitted append on the same HIP
 * queue.  The append completion remains caller-owned and separately
 * observable; its claim prevents cancellation/release until the dependent
 * attention reaches a terminal state. */
extern "C" sllm_status_t sllm_causal_attention_execute_after_kv_append(
    const sllm_context_t *const raw_context,
    const sllm_queue_t *const raw_queue,
    sllm_completion_t *const raw_append_completion,
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
    if (raw_append_completion == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "KV append dependency completion is null");
    }
    Completion *dependency = nullptr;
    const sllm_status_t pin_status =
        pin_completion(raw_append_completion, &dependency, error_sink);
    if (pin_status != SLLM_STATUS_OK) {
      return pin_status;
    }
    CompletionPin dependency_pin(dependency);
    {
      std::lock_guard<std::mutex> state_lock(dependency->state_mutex);
      if (!dependency->kv_state_append || dependency->terminal) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "KV append dependency is not a pending append completion");
      }
      if (dependency->dependent_attention_claimed) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "KV append completion already has an attention dependent");
      }
      dependency->dependent_attention_claimed = true;
    }
    const sllm_status_t status = causal_attention_execute_impl(
        raw_context, raw_queue, descriptor, completion_output, dispatch_info,
        error_sink, dependency);
    if (status != SLLM_STATUS_OK) {
      std::lock_guard<std::mutex> state_lock(dependency->state_mutex);
      dependency->dependent_attention_claimed = false;
    }
    return status;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in causal attention append chain");
  }
}
