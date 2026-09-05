namespace {

bool short_column_state_opt_in(const uint64_t token_count,
                               const uint32_t qk_heads,
                               const uint32_t value_heads,
                               const uint32_t head_dim,
                               const char *const arch_name,
                               const char *const force_baseline) noexcept {
  const char *const opt_in =
      std::getenv("SLLM_LINEAR_ATTENTION_GFX1030_SHORT_COLUMN_STATE");
  return (force_baseline == nullptr || std::strcmp(force_baseline, "1") != 0) &&
         (opt_in == nullptr || std::strcmp(opt_in, "1") == 0) &&
         token_count >= 17U && token_count < 128U &&
         qk_heads == ::sllm_linear_attention_kernel::kQkHeads &&
         value_heads == ::sllm_linear_attention_kernel::kValueHeads &&
         head_dim == ::sllm_linear_attention_kernel::kHeadDim &&
         arch_name != nullptr && std::strcmp(arch_name, "gfx1030") == 0;
}

bool short_column_state_shape_opt_in(
    const uint64_t token_count, const uint32_t qk_heads,
    const uint32_t value_heads, const uint32_t head_dim,
    const char *const force_baseline) noexcept {
  const char *const opt_in =
      std::getenv("SLLM_LINEAR_ATTENTION_GFX1030_SHORT_COLUMN_STATE");
  return (force_baseline == nullptr || std::strcmp(force_baseline, "1") != 0) &&
         (opt_in == nullptr || std::strcmp(opt_in, "1") == 0) &&
         token_count >= 17U && token_count < 128U &&
         qk_heads == ::sllm_linear_attention_kernel::kQkHeads &&
         value_heads == ::sllm_linear_attention_kernel::kValueHeads &&
         head_dim == ::sllm_linear_attention_kernel::kHeadDim;
}

// The row32-LDS provider is deliberately opt-in. Unlike the historical short
// column route, an unset variable must not change the production default: the
// provider is an exact gfx1030 Qwen3.8 M=1 contract and has an explicit
// rollback switch (set the variable to anything other than "1", or force the
// generic path with SLLM_GDN_FORCE_BASELINE=1).
bool row32_lds_state_opt_in(const uint64_t token_count, const uint32_t qk_heads,
                            const uint32_t value_heads, const uint32_t head_dim,
                            const char *const arch_name,
                            const char *const force_baseline) noexcept {
  const char *const opt_in =
      std::getenv("SLLM_LINEAR_ATTENTION_GFX1030_ROW32_LDS");
  return (force_baseline == nullptr || std::strcmp(force_baseline, "1") != 0) &&
         opt_in != nullptr && std::strcmp(opt_in, "1") == 0 &&
         token_count == 1U &&
         qk_heads == ::sllm_linear_attention_kernel::kRow32LdsQkHeads &&
         value_heads == ::sllm_linear_attention_kernel::kRow32LdsValueHeads &&
         head_dim == ::sllm_linear_attention_kernel::kRow32LdsHeadDim &&
         arch_name != nullptr && std::strcmp(arch_name, "gfx1030") == 0;
}

bool gfx942_wave64_column_state_opt_in(
    const uint64_t token_count, const uint32_t qk_heads,
    const uint32_t value_heads, const uint32_t head_dim,
    const char *const arch_name, const char *const force_baseline) noexcept {
  const char *const opt_in =
      std::getenv("SLLM_LINEAR_ATTENTION_GFX942_WAVE64_COLUMN_STATE");
  return (force_baseline == nullptr || std::strcmp(force_baseline, "1") != 0) &&
         opt_in != nullptr && std::strcmp(opt_in, "1") == 0 &&
         token_count >= 128U &&
         qk_heads == ::sllm_linear_attention_kernel::kQkHeads &&
         value_heads == ::sllm_linear_attention_kernel::kValueHeads &&
         head_dim == ::sllm_linear_attention_kernel::kHeadDim &&
         arch_name != nullptr &&
         std::strcmp(arch_name, "gfx942:sramecc+:xnack-") == 0;
}

bool gfx942_wave64_column_state_shape_opt_in(
    const uint64_t token_count, const uint32_t qk_heads,
    const uint32_t value_heads, const uint32_t head_dim,
    const char *const force_baseline) noexcept {
  const char *const opt_in =
      std::getenv("SLLM_LINEAR_ATTENTION_GFX942_WAVE64_COLUMN_STATE");
  return (force_baseline == nullptr || std::strcmp(force_baseline, "1") != 0) &&
         opt_in != nullptr && std::strcmp(opt_in, "1") == 0 &&
         token_count >= 128U &&
         qk_heads == ::sllm_linear_attention_kernel::kQkHeads &&
         value_heads == ::sllm_linear_attention_kernel::kValueHeads &&
         head_dim == ::sllm_linear_attention_kernel::kHeadDim;
}

class LinearAttentionExecuteScopeGuard final {
public:
  LinearAttentionExecuteScopeGuard(LinearAttentionState *const state,
                                   Queue *const queue,
                                   const LinearAttentionBuffers buffers,
                                   std::unique_ptr<Completion> *const candidate,
                                   NativeEventGuard *const event_guard,
                                   sllm_error_sink_t *const sink) noexcept
      : state_(state), queue_(queue), buffers_(buffers), candidate_(candidate),
        event_guard_(event_guard), sink_(sink), token_(0U),
        phase_(Phase::Reserved) {}

  LinearAttentionExecuteScopeGuard(const LinearAttentionExecuteScopeGuard &) =
      delete;
  LinearAttentionExecuteScopeGuard &
  operator=(const LinearAttentionExecuteScopeGuard &) = delete;

  ~LinearAttentionExecuteScopeGuard() { cleanup(); }

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
      (void)rollback_reserved_linear_attention(state_, queue_, buffers_, sink_);
      break;
    case Phase::Candidate:
      if (candidate_ != nullptr && candidate_->get() != nullptr) {
        (void)rollback_unpublished_submission(
            *candidate_, *event_guard_,
            "linear attention unwound before completion registration", sink_);
      } else {
        state_->context->poisoned.store(true);
      }
      break;
    case Phase::Registered:
      if (candidate_ != nullptr && candidate_->get() != nullptr) {
        (void)cleanup_failed_submission(
            *candidate_, token_, hipErrorUnknown,
            "linear attention unwound after completion registration", queue_,
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

  LinearAttentionState *const state_;
  Queue *const queue_;
  LinearAttentionBuffers buffers_;
  std::unique_ptr<Completion> *const candidate_;
  NativeEventGuard *const event_guard_;
  sllm_error_sink_t *const sink_;
  uintptr_t token_;
  Phase phase_;
};

sllm_status_t validate_linear_attention_view_info_input(
    const sllm_linear_attention_view_info_t *const info,
    sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr || info->struct_size != sizeof(*info)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "linear attention view info has an unsupported struct size");
  }
  if (info->abi_version != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "linear attention view info ABI version is unsupported");
  }
  if (info->info_version != SLLM_HIP_LINEAR_ATTENTION_VIEW_INFO_VERSION ||
      info->reserved0 != 0U ||
      !std::all_of(std::begin(info->reserved), std::end(info->reserved),
                   [](const uint32_t value) { return value == 0U; })) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "linear attention view info version or reserved fields are invalid");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t validate_linear_attention_dispatch_info(
    const sllm_linear_attention_dispatch_info_t *const info,
    sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr || info->struct_size != sizeof(*info)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "linear attention dispatch info has an unsupported struct size");
  }
  if (info->abi_version != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "linear attention dispatch info ABI version is unsupported");
  }
  if (info->info_version != SLLM_HIP_LINEAR_ATTENTION_DISPATCH_INFO_VERSION ||
      !std::all_of(std::begin(info->reserved), std::end(info->reserved),
                   [](const uint32_t value) { return value == 0U; })) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "linear attention dispatch info version or reserved fields are "
        "invalid");
  }
  return SLLM_STATUS_OK;
}

void fill_linear_attention_view_info(
    const LinearAttentionState *const state,
    sllm_linear_attention_view_info_t *const info) noexcept {
  sllm_linear_attention::initialize_view_info(info);
  info->session_id = state->session_id;
  info->layer_id = state->layer_id;
  info->conv_state_dtype = SLLM_TENSOR_DTYPE_BF16;
  info->recurrent_state_dtype = SLLM_TENSOR_DTYPE_F32;
  info->encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  info->active_slot = state->active_slot;
  info->capacity_tokens = state->capacity_tokens;
  info->observed_length = state->published_length;
  info->generation = state->generation;
  info->context_identity = state->context_identity;
  info->state_identity = state->state_identity;
  info->conv_state_shape[0] = state->conv_kernel_size - 1U;
  info->conv_state_shape[1] = state->qkv_width;
  info->recurrent_state_shape[0] = state->value_heads;
  info->recurrent_state_shape[1] = state->head_dim;
  info->recurrent_state_shape[2] = state->head_dim;
}

void fill_linear_attention_dispatch_info(
    sllm_linear_attention_dispatch_info_t *const info,
    const uint64_t dispatch_id,
    const sllm_linear_attention::DescriptorMetadata &metadata,
    const char *const arch_name, const bool use_column_provider,
    const bool use_gfx942_wave64_column_provider,
    const bool use_decode_pair_provider,
    const bool use_row32_lds_provider) noexcept {
  const uint32_t struct_size = info->struct_size;
  const uint32_t abi_version = info->abi_version;
  std::memset(info, 0, sizeof(*info));
  info->struct_size = struct_size;
  info->abi_version = abi_version;
  info->info_version = SLLM_HIP_LINEAR_ATTENTION_DISPATCH_INFO_VERSION;
  info->backend = SLLM_BACKEND_HIP;
  info->dispatch_id = dispatch_id;
  info->dispatch_count = use_column_provider ? 4U : 2U;
  info->conv_kernel_id =
      SLLM_HIP_LINEAR_ATTENTION_KERNEL_ID_CAUSAL_CONV_SILU_V1;
  info->recurrent_kernel_id =
      use_row32_lds_provider
          ? SLLM_HIP_LINEAR_ATTENTION_KERNEL_ID_RECURRENT_GATED_NORM_ROW32_LDS_V1
          : SLLM_HIP_LINEAR_ATTENTION_KERNEL_ID_RECURRENT_GATED_NORM_V1;
  info->workgroup_size_x =
      use_row32_lds_provider
          ? ::sllm_linear_attention_kernel::kRow32LdsWorkgroupSize
      : use_gfx942_wave64_column_provider
          ? ::sllm_linear_attention_kernel::kGfx942Wave64ColumnWorkgroupSize
      : use_decode_pair_provider
          ? ::sllm_linear_attention_kernel::kDecodePairWorkgroupSize
          : SLLM_HIP_LINEAR_ATTENTION_WORKGROUP_SIZE;
  const uint64_t conv_elements =
      (metadata.token_count + metadata.conv_kernel_size - 1U) *
      metadata.qkv_width;
  info->conv_grid_size_x = static_cast<uint32_t>(
      (conv_elements + SLLM_HIP_LINEAR_ATTENTION_WORKGROUP_SIZE - 1U) /
      SLLM_HIP_LINEAR_ATTENTION_WORKGROUP_SIZE);
  info->recurrent_grid_size_x =
      use_row32_lds_provider     ? metadata.value_heads
      : use_column_provider      ? metadata.value_heads * metadata.head_dim / 4U
      : use_decode_pair_provider ? metadata.qk_heads
                                 : metadata.value_heads;
  info->token_count = metadata.token_count;
  info->start_position = metadata.start_position;
  info->expected_length = metadata.expected_length;
  info->qk_heads = metadata.qk_heads;
  info->value_heads = metadata.value_heads;
  info->head_dim = metadata.head_dim;
  info->fallback_allowed = 0U;
  info->fallback_used = 0U;
  sllm_public_runtime::copy_fixed_string(
      info->kernel_symbol, SLLM_HIP_LINEAR_ATTENTION_KERNEL_SYMBOL_MAX,
      use_row32_lds_provider
          ? ::sllm_linear_attention_kernel::kRow32LdsLogicalKernelId
      : use_gfx942_wave64_column_provider
          ? ::sllm_linear_attention_kernel::kGfx942Wave64ColumnLogicalKernelId
      : use_column_provider
          ? ::sllm_linear_attention_kernel::kColumnLogicalKernelId
      : use_decode_pair_provider
          ? ::sllm_linear_attention_kernel::kDecodePairLogicalKernelId
          : ::sllm_linear_attention_kernel::kLogicalKernelId);
  sllm_public_runtime::copy_fixed_string(
      info->conv_device_symbol, SLLM_HIP_LINEAR_ATTENTION_DEVICE_SYMBOL_MAX,
      ::sllm_linear_attention_kernel::kConvDeviceSymbol);
  sllm_public_runtime::copy_fixed_string(
      info->recurrent_device_symbol,
      SLLM_HIP_LINEAR_ATTENTION_DEVICE_SYMBOL_MAX,
      use_row32_lds_provider
          ? ::sllm_linear_attention_kernel::kRow32LdsDeviceSymbol
      : use_gfx942_wave64_column_provider
          ? ::sllm_linear_attention_kernel::
                kGfx942Wave64ColumnRecurrentDeviceSymbol
      : use_column_provider
          ? ::sllm_linear_attention_kernel::kColumnRecurrentDeviceSymbol
      : use_decode_pair_provider
          ? ::sllm_linear_attention_kernel::kDecodePairDeviceSymbol
          : ::sllm_linear_attention_kernel::kRecurrentDeviceSymbol);
  sllm_public_runtime::copy_fixed_string(info->gcn_arch_name,
                                         SLLM_HIP_MAX_GCN_ARCH_NAME, arch_name);
}

hipError_t cleanup_linear_allocations(Context *const context,
                                      void **const allocations,
                                      const std::size_t count) noexcept {
  hipError_t first_error = hipSuccess;
  for (std::size_t index = 0U; index != count; ++index) {
    if (allocations[index] == nullptr) {
      continue;
    }
    const hipError_t status =
        free_allocation_with_fault_injection(allocations[index]);
    if (status != hipSuccess) {
      if (first_error == hipSuccess) {
        first_error = status;
      }
      orphan_owner.retain_allocation(context, allocations[index]);
    }
    allocations[index] = nullptr;
  }
  return first_error;
}

void *linear_binding_pointer(Buffer *const buffer,
                             const uint64_t offset) noexcept {
  return static_cast<char *>(buffer->device_pointer) +
         static_cast<std::size_t>(offset);
}

} // namespace

#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
// The host runtime links the fake HIP launchers from public_runtime.hip.cpp.
// Keep the new provider's stub beside its selector so host-only contract tests
// do not need to link a device kernel object.
namespace sllm_linear_attention_kernel {
hipError_t launch_row32_lds(const uint16_t *const, const uint16_t *const,
                            const uint16_t *const, const uint16_t *const,
                            const float *const, const uint16_t *const,
                            const float *const, const float *const,
                            float *const, uint16_t *const, const uint32_t,
                            const uint32_t, const uint32_t, const uint32_t,
                            const uint32_t, const uint32_t,
                            const hipStream_t) noexcept {
  return hipSuccess;
}
} // namespace sllm_linear_attention_kernel
#endif

#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
extern "C" uint32_t sllm_test_select_linear_attention_gfx942_wave64_column(
    const uint64_t token_count, const uint32_t qk_heads,
    const uint32_t value_heads, const uint32_t head_dim,
    const char *const arch_name) noexcept {
  return gfx942_wave64_column_state_opt_in(
             token_count, qk_heads, value_heads, head_dim, arch_name,
             std::getenv("SLLM_GDN_FORCE_BASELINE"))
             ? 1U
             : 0U;
}

extern "C" uint32_t sllm_test_select_linear_attention_gfx1030_row32_lds(
    const uint64_t token_count, const uint32_t qk_heads,
    const uint32_t value_heads, const uint32_t head_dim,
    const char *const arch_name) noexcept {
  return row32_lds_state_opt_in(token_count, qk_heads, value_heads, head_dim,
                                arch_name,
                                std::getenv("SLLM_GDN_FORCE_BASELINE"))
             ? 1U
             : 0U;
}
#endif

extern "C" sllm_status_t sllm_linear_attention_state_create(
    const sllm_context_t *const raw_context,
    const sllm_linear_attention_state_create_info_t *const info,
    sllm_linear_attention_state_t **const raw_state,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (raw_state != nullptr) {
      *raw_state = nullptr;
    }
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t info_status =
        sllm_linear_attention::validate_state_create_info(info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    if (raw_context == nullptr || raw_state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "linear attention state context or output is null");
    }

    Context *context = nullptr;
    const uintptr_t context_token = handle_key(raw_context);
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      context = lookup<Context>(raw_context, HandleKind::Context);
      if (context == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "linear attention context handle is stale or has the wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (context->poisoned.load() || context->release_active ||
          !sllm_public_runtime::AccountingState::can_increment(
              context->accounting.child_count)) {
        return sllm_public_runtime::write_error(
            error_sink,
            context->release_active ? SLLM_STATUS_PUBLIC_BUSY
                                    : SLLM_STATUS_INTERNAL_ERROR,
            "linear attention context is unavailable for state creation");
      }
      ++context->accounting.child_count;
    }
    const sllm_status_t device_status =
        select_context_device(context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      (void)rollback_child(
          context, error_sink,
          "linear attention provisional accounting rollback failed");
      return device_status;
    }

    const uint32_t qk_heads = info->qk_heads == 0U
                                  ? SLLM_HIP_LINEAR_ATTENTION_QK_HEADS
                                  : info->qk_heads;
    const uint32_t value_heads = info->value_heads == 0U
                                     ? SLLM_HIP_LINEAR_ATTENTION_VALUE_HEADS
                                     : info->value_heads;
    const uint32_t head_dim = info->head_dim == 0U
                                  ? SLLM_HIP_LINEAR_ATTENTION_HEAD_DIM
                                  : info->head_dim;
    const uint32_t conv_kernel_size =
        info->conv_kernel_size == 0U
            ? SLLM_HIP_LINEAR_ATTENTION_CONV_KERNEL_SIZE
            : info->conv_kernel_size;
    const uint32_t qkv_width = (2U * qk_heads + value_heads) * head_dim;
    const std::size_t conv_bytes =
        static_cast<std::size_t>(conv_kernel_size - 1U) * qkv_width *
        sizeof(uint16_t);
    const std::size_t recurrent_bytes = static_cast<std::size_t>(value_heads) *
                                        head_dim * head_dim * sizeof(float);
    void *allocations[4] = {};
    std::array<void *, 2> conv_views{};
    std::array<void *, 2> recurrent_views{};
    const bool compact_phase78_state = qk_heads == 16U && value_heads == 48U &&
                                       head_dim == 128U &&
                                       conv_kernel_size == 4U;
    if (compact_phase78_state) {
      const std::size_t total_bytes =
          (conv_bytes * 2U) + (recurrent_bytes * 2U);
      const hipError_t allocation_status =
          sllm_public_runtime::FaultInjector::consume(
              sllm_public_runtime::FaultPoint::NativeCreationFailure)
              ? hipErrorUnknown
              : hipMalloc(&allocations[0], total_bytes);
      if (allocation_status != hipSuccess) {
        (void)rollback_child(
            context, error_sink,
            "linear attention provisional accounting rollback failed");
        return hip_failure(error_sink, allocation_status,
                           "hipMalloc compact linear attention state");
      }
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
      std::memset(allocations[0], 0, total_bytes);
      const hipError_t zero_status = hipSuccess;
#else
      const hipError_t zero_status = hipMemset(allocations[0], 0, total_bytes);
#endif
      if (zero_status != hipSuccess) {
        const hipError_t cleanup_status =
            cleanup_linear_allocations(context, allocations, 4U);
        (void)rollback_child(
            context, error_sink,
            "linear attention provisional accounting rollback failed");
        return hip_failure(error_sink,
                           cleanup_status == hipSuccess ? zero_status
                                                        : cleanup_status,
                           "hipMemset linear attention state");
      }
      auto *const base = static_cast<char *>(allocations[0]);
      conv_views = {base, base + conv_bytes};
      auto *const recurrent_base = base + (conv_bytes * 2U);
      recurrent_views = {recurrent_base, recurrent_base + recurrent_bytes};
    } else {
      for (std::size_t index = 0U; index != 4U; ++index) {
        const std::size_t bytes = index < 2U ? conv_bytes : recurrent_bytes;
        const hipError_t allocation_status =
            sllm_public_runtime::FaultInjector::consume(
                sllm_public_runtime::FaultPoint::NativeCreationFailure)
                ? hipErrorUnknown
                : hipMalloc(&allocations[index], bytes);
        if (allocation_status != hipSuccess) {
          const hipError_t cleanup_status =
              cleanup_linear_allocations(context, allocations, index);
          (void)rollback_child(
              context, error_sink,
              "linear attention provisional accounting rollback failed");
          return hip_failure(error_sink,
                             cleanup_status == hipSuccess ? allocation_status
                                                          : cleanup_status,
                             "hipMalloc linear attention state");
        }
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
        std::memset(allocations[index], 0, bytes);
        const hipError_t zero_status = hipSuccess;
#else
        const hipError_t zero_status = hipMemset(allocations[index], 0, bytes);
#endif
        if (zero_status != hipSuccess) {
          const hipError_t cleanup_status =
              cleanup_linear_allocations(context, allocations, index + 1U);
          (void)rollback_child(
              context, error_sink,
              "linear attention provisional accounting rollback failed");
          return hip_failure(error_sink,
                             cleanup_status == hipSuccess ? zero_status
                                                          : cleanup_status,
                             "hipMemset linear attention state");
        }
      }
      conv_views = {allocations[0], allocations[1]};
      recurrent_views = {allocations[2], allocations[3]};
    }

    std::unique_ptr<LinearAttentionState> candidate;
    try {
      candidate = std::make_unique<LinearAttentionState>(
          context, info->session_id, info->layer_id, info->capacity_tokens,
          qk_heads, value_heads, head_dim, conv_kernel_size, conv_views,
          recurrent_views, context_token);
      candidate->state_backing =
          compact_phase78_state ? allocations[0] : nullptr;
    } catch (...) {
      (void)cleanup_linear_allocations(context, allocations, 4U);
      (void)rollback_child(
          context, error_sink,
          "linear attention provisional accounting rollback failed");
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "linear attention state host allocation failed");
    }
    {
      std::lock_guard<std::mutex> lock(context->accounting_mutex);
      if (!reserve_linear_attention_state(context->accounting)) {
        poison_context_locked(context);
        poison_owner.retain(std::move(candidate));
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "linear attention state accounting is exhausted; state "
            "quarantined");
      }
      --context->accounting.child_count;
    }
    uintptr_t token = 0U;
    try {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      token =
          register_handle(candidate.get(), HandleKind::LinearAttentionState);
    } catch (...) {
      bool accounting_released = false;
      {
        std::lock_guard<std::mutex> lock(context->accounting_mutex);
        accounting_released =
            release_linear_attention_state(context->accounting);
        if (!accounting_released) {
          poison_context_locked(context);
          poison_owner.retain(std::move(candidate));
        }
      }
      if (!accounting_released) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "linear attention state accounting rollback failed");
      }
      (void)cleanup_linear_allocations(context, allocations, 4U);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "linear attention state registry allocation failed");
    }
    if (token == 0U) {
      {
        std::lock_guard<std::mutex> lock(context->accounting_mutex);
        (void)release_linear_attention_state(context->accounting);
      }
      (void)cleanup_linear_allocations(context, allocations, 4U);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "linear attention state token allocation failed");
    }
    candidate->state_identity = token;
    *raw_state = reinterpret_cast<sllm_linear_attention_state_t *>(token);
    (void)candidate.release();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in linear attention state create");
  }
}

extern "C" sllm_status_t sllm_linear_attention_state_release(
    sllm_linear_attention_state_t **const raw_state,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_state == nullptr || *raw_state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "linear attention state handle is null");
    }
    LinearAttentionState *state = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      state = lookup<LinearAttentionState>(*raw_state,
                                           HandleKind::LinearAttentionState);
      if (state == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "linear attention state handle is stale or has the wrong kind");
      }
      std::lock_guard<std::mutex> lock(state->context->accounting_mutex);
      if (state->release_active || state->transition_token != 0U ||
          state->accounting.active_submissions != 0U ||
          state->accounting.completion_references != 0U) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "linear attention state has an in-flight transition");
      }
      if (state->context->poisoned.load()) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "linear attention state context is poisoned");
      }
      state->release_active = true;
    }
    const sllm_status_t device_status =
        select_context_device(state->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      state->release_active = false;
      return device_status;
    }
    void *allocations[5] = {};
    std::size_t allocation_count = 0U;
    if (state->state_backing != nullptr) {
      allocations[allocation_count++] = state->state_backing;
      allocations[allocation_count++] = state->scratch;
    } else {
      allocations[allocation_count++] = state->conv_state[0];
      allocations[allocation_count++] = state->conv_state[1];
      allocations[allocation_count++] = state->recurrent_state[0];
      allocations[allocation_count++] = state->recurrent_state[1];
      allocations[allocation_count++] = state->scratch;
    }
    const hipError_t cleanup_status = cleanup_linear_allocations(
        state->context, allocations, allocation_count);
    state->conv_state = {nullptr, nullptr};
    state->recurrent_state = {nullptr, nullptr};
    state->state_backing = nullptr;
    state->scratch = nullptr;
    state->scratch_bytes = 0U;
    if (cleanup_status != hipSuccess) {
      {
        std::lock_guard<std::mutex> registry_lock(registry_mutex);
        unregister_handle(*raw_state);
        *raw_state = nullptr;
      }
      retain_poisoned(state, state->context);
      return hip_failure(error_sink, cleanup_status,
                         "hipFree linear attention state");
    }
    bool released = false;
    {
      std::lock_guard<std::mutex> lock(state->context->accounting_mutex);
      released = !sllm_public_runtime::FaultInjector::consume(
                     sllm_public_runtime::FaultPoint::AccountingFailure) &&
                 release_linear_attention_state(state->context->accounting);
    }
    if (!released) {
      {
        std::lock_guard<std::mutex> registry_lock(registry_mutex);
        unregister_handle(*raw_state);
        *raw_state = nullptr;
      }
      retain_poisoned(state, state->context);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "linear attention state accounting release failed");
    }
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      unregister_handle(*raw_state);
      *raw_state = nullptr;
    }
    delete state;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in linear attention state release");
  }
}

extern "C" sllm_status_t sllm_linear_attention_state_query(
    const sllm_linear_attention_state_t *const raw_state,
    sllm_linear_attention_view_info_t *const info,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t info_status =
        validate_linear_attention_view_info_input(info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    LinearAttentionState *const state = lookup<LinearAttentionState>(
        raw_state, HandleKind::LinearAttentionState);
    if (state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "linear attention state handle is stale or has the wrong kind");
    }
    std::lock_guard<std::mutex> lock(state->context->accounting_mutex);
    if (state->context->poisoned.load() || state->release_active) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "linear attention state is releasing or context is poisoned");
    }
    fill_linear_attention_view_info(state, info);
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in linear attention state query");
  }
}

extern "C" sllm_status_t sllm_linear_attention_state_rewind_last(
    const sllm_linear_attention_state_t *const raw_state,
    const uint64_t expected_length, const uint64_t rewind_length,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    LinearAttentionState *const state = lookup<LinearAttentionState>(
        raw_state, HandleKind::LinearAttentionState);
    if (state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "linear attention rewind state handle is stale or wrong kind");
    }
    std::lock_guard<std::mutex> lock(state->context->accounting_mutex);
    if (state->context->poisoned.load() || state->release_active ||
        state->transition_token != 0U) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "linear attention rewind requires a quiescent state");
    }
    if (rewind_length >= expected_length ||
        state->published_length != expected_length ||
        state->last_published_start != rewind_length ||
        state->last_published_end != expected_length ||
        state->last_published_generation != state->generation ||
        state->generation == std::numeric_limits<uint64_t>::max()) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "linear attention rewind does not identify the most recent published "
          "transition");
    }
    state->active_slot = 1U - state->active_slot;
    state->published_length = rewind_length;
    ++state->generation;
    state->last_published_start = 0U;
    state->last_published_end = 0U;
    state->last_published_generation = 0U;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in linear attention state rewind");
  }
}

extern "C" sllm_status_t sllm_linear_attention_execute(
    const sllm_context_t *const raw_context,
    const sllm_queue_t *const raw_queue,
    const sllm_linear_attention_desc_t *const descriptor,
    sllm_completion_t **const completion_output,
    sllm_linear_attention_dispatch_info_t *const dispatch_info,
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
        validate_linear_attention_dispatch_info(dispatch_info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    if (raw_context == nullptr || raw_queue == nullptr ||
        descriptor == nullptr || completion_output == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "linear attention context, queue, descriptor, or completion output "
          "is null");
    }
    sllm_linear_attention::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_linear_attention::validate_and_copy_descriptor(
            descriptor, &metadata, error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }

    Context *context = nullptr;
    Queue *queue = nullptr;
    LinearAttentionState *state = nullptr;
    LinearAttentionBuffers buffers{};
    uint64_t dispatch_id = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      context = lookup<Context>(raw_context, HandleKind::Context);
      queue = lookup<Queue>(raw_queue, HandleKind::Queue);
      state = lookup<LinearAttentionState>(descriptor->state,
                                           HandleKind::LinearAttentionState);
      const std::array<const sllm_buffer_t *, 9> raw_buffers = {
          descriptor->qkv.buffer,         descriptor->z.buffer,
          descriptor->b_input.buffer,     descriptor->a_input.buffer,
          descriptor->conv_weight.buffer, descriptor->a_log.buffer,
          descriptor->dt_bias.buffer,     descriptor->norm_weight.buffer,
          descriptor->output.buffer};
      for (std::size_t index = 0U; index != buffers.size(); ++index) {
        buffers[index] = lookup<Buffer>(raw_buffers[index], HandleKind::Buffer);
      }
      if (context == nullptr || queue == nullptr || state == nullptr ||
          std::any_of(
              buffers.begin(), buffers.end(),
              [](const Buffer *const buffer) { return buffer == nullptr; })) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "linear attention handle is stale or has the wrong kind");
      }
      if (context != queue->context || context != state->context ||
          std::any_of(buffers.begin(), buffers.end(),
                      [context](const Buffer *const buffer) {
                        return buffer->context != context;
                      })) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
            "linear attention resources must share one context");
      }
      const std::array<const sllm_linear_attention::TensorMetadata *, 9>
          tensor_metadata = {
              &metadata.qkv,     &metadata.z,           &metadata.b_input,
              &metadata.a_input, &metadata.conv_weight, &metadata.a_log,
              &metadata.dt_bias, &metadata.norm_weight, &metadata.output};
      for (std::size_t index = 0U; index != buffers.size(); ++index) {
        if (tensor_metadata[index]->end_offset > buffers[index]->size_bytes) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
              "linear attention tensor interval exceeds its backing buffer");
        }
      }
      std::lock_guard<std::mutex> lock(context->accounting_mutex);
      if (context->poisoned.load() || context->release_active ||
          queue->release_active || state->release_active ||
          state->transition_token != 0U ||
          state->accounting.active_submissions != 0U ||
          state->accounting.completion_references != 0U ||
          std::any_of(buffers.begin(), buffers.end(),
                      [](const Buffer *const buffer) {
                        return buffer->release_active;
                      })) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_LINEAR_ATTENTION_STATE_BUSY,
            "linear attention state or dependency is busy");
      }
      if (metadata.start_position != state->published_length ||
          metadata.expected_length > state->capacity_tokens) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_LINEAR_ATTENTION_LENGTH_MISMATCH,
            "linear attention range does not start at the committed state "
            "length");
      }
      if (metadata.qk_heads != state->qk_heads ||
          metadata.value_heads != state->value_heads ||
          metadata.head_dim != state->head_dim ||
          metadata.conv_kernel_size != state->conv_kernel_size) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_SHAPE_MISMATCH,
            "linear attention tensor layout differs from the state layout");
      }
      if (context->next_dispatch_id == 0U) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "linear attention context dispatch id is exhausted");
      }
      dispatch_id = context->next_dispatch_id;
      context->next_dispatch_id =
          dispatch_id == std::numeric_limits<uint64_t>::max()
              ? 0U
              : dispatch_id + 1U;
      if (!reserve_linear_attention_submission(context, queue, state,
                                               buffers)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "linear attention submission accounting is exhausted");
      }
      state->transition_token = dispatch_id;
      state->transition_start = metadata.start_position;
      state->transition_count = metadata.token_count;
      state->transition_end = metadata.expected_length;
      state->commit_allowed = true;
      dispatch_info->dispatch_id = dispatch_id;
    }

    std::unique_ptr<Completion> candidate;
    NativeEventGuard event_guard;
    LinearAttentionExecuteScopeGuard execute_guard(
        state, queue, buffers, &candidate, &event_guard, error_sink);
    const sllm_status_t device_status =
        select_context_device(context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      return device_status;
    }
    const char *const force_baseline = std::getenv("SLLM_GDN_FORCE_BASELINE");
    const bool column_candidate_by_shape =
        (metadata.token_count >= 128U &&
         (force_baseline == nullptr ||
          std::strcmp(force_baseline, "1") != 0)) ||
        short_column_state_shape_opt_in(metadata.token_count, metadata.qk_heads,
                                        metadata.value_heads, metadata.head_dim,
                                        force_baseline) ||
        gfx942_wave64_column_state_shape_opt_in(
            metadata.token_count, metadata.qk_heads, metadata.value_heads,
            metadata.head_dim, force_baseline);
    constexpr uint64_t kMaxU64 = std::numeric_limits<uint64_t>::max();
    if (metadata.qkv_width == 0U || metadata.value_heads == 0U ||
        metadata.token_count >
            kMaxU64 / metadata.qkv_width / sizeof(uint16_t) ||
        (column_candidate_by_shape &&
         metadata.token_count >
             kMaxU64 / metadata.value_heads / sizeof(float) / 2U)) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_METADATA_OVERFLOW,
          "linear attention scratch size overflowed");
    }
    const uint64_t convolved_scratch =
        metadata.token_count * metadata.qkv_width * sizeof(uint16_t);
    const uint64_t scalar_scratch =
        column_candidate_by_shape
            ? metadata.token_count * metadata.value_heads * sizeof(float) * 2U
            : 0U;
    if (convolved_scratch > kMaxU64 - scalar_scratch) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_METADATA_OVERFLOW,
          "linear attention scratch planes overflowed");
    }
    const uint64_t required_scratch = convolved_scratch + scalar_scratch;
    if (required_scratch > state->scratch_bytes) {
      void *new_scratch = nullptr;
      const hipError_t allocation_status =
          hipMalloc(&new_scratch, static_cast<std::size_t>(required_scratch));
      if (allocation_status != hipSuccess) {
        return hip_failure(error_sink, allocation_status,
                           "hipMalloc linear attention scratch");
      }
      if (state->scratch != nullptr) {
        const hipError_t free_status = hipFree(state->scratch);
        if (free_status != hipSuccess) {
          orphan_owner.retain_allocation(context, state->scratch);
          context->poisoned.store(true);
          void *cleanup[] = {new_scratch};
          (void)cleanup_linear_allocations(context, cleanup, 1U);
          return hip_failure(error_sink, free_status,
                             "hipFree prior linear attention scratch");
        }
      }
      state->scratch = new_scratch;
      state->scratch_bytes = required_scratch;
    }
    hipDeviceProp_t properties = {};
    const sllm_status_t property_status =
        get_context_device_properties(context, &properties, error_sink);
    if (property_status != SLLM_STATUS_OK) {
      return property_status;
    }
    char arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME] = {};
    if (!copy_property(arch_name, sizeof(arch_name), properties.gcnArchName,
                       sizeof(properties.gcnArchName))) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
          "linear attention gcnArchName is not NUL terminated or is too long");
    }
#if defined(SLLM_HIP_COMPILE_TARGET)
    if (!matches_runtime_gcn_arch(arch_name, SLLM_HIP_COMPILE_TARGET)) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
          "linear attention device target does not match the compiled exact "
          "target");
    }
#endif
    const bool use_short_column_provider = short_column_state_opt_in(
        metadata.token_count, metadata.qk_heads, metadata.value_heads,
        metadata.head_dim, arch_name, force_baseline);
    const bool use_long_column_provider =
        metadata.token_count >= 128U &&
        (force_baseline == nullptr || std::strcmp(force_baseline, "1") != 0) &&
        (std::strcmp(arch_name, "gfx1030") == 0 ||
         std::strcmp(arch_name, "gfx1201") == 0);
    const bool use_gfx942_wave64_column_provider =
        gfx942_wave64_column_state_opt_in(
            metadata.token_count, metadata.qk_heads, metadata.value_heads,
            metadata.head_dim, arch_name, force_baseline);
    const bool use_column_provider = use_short_column_provider ||
                                     use_long_column_provider ||
                                     use_gfx942_wave64_column_provider;
    const bool use_decode_pair_provider =
        (force_baseline == nullptr || std::strcmp(force_baseline, "1") != 0) &&
        metadata.token_count == 1U && metadata.qk_heads == 16U &&
        metadata.value_heads == 32U && metadata.head_dim == 128U &&
        std::strcmp(arch_name, "gfx1030") == 0;
    const bool use_row32_lds_provider = row32_lds_state_opt_in(
        metadata.token_count, metadata.qk_heads, metadata.value_heads,
        metadata.head_dim, arch_name, force_baseline);
    try {
      candidate = std::make_unique<Completion>(context, queue, nullptr, 0U,
                                               false, std::vector<uint8_t>{});
      candidate->linear_attention = true;
      candidate->linear_attention_state = state;
      candidate->linear_attention_buffers = buffers;
      candidate->linear_attention_token = dispatch_id;
      candidate->linear_attention_start = metadata.start_position;
      candidate->linear_attention_count = metadata.token_count;
      candidate->linear_attention_end = metadata.expected_length;
      execute_guard.candidate_allocated();
    } catch (...) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "linear attention completion allocation failed");
    }
    hipEvent_t native_event = nullptr;
    const hipError_t event_status =
        queue_profiles_completions(queue)
            ? hipEventCreateWithFlags(&native_event, 0U)
            : hipSuccess;
    if (event_status != hipSuccess) {
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
      return rollback_unpublished_submission(
          candidate, event_guard,
          "linear attention timing event creation failed", error_sink);
    }
    candidate->timing_start_event = timing_start_event;
    uintptr_t token = 0U;
    try {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      token = register_handle(candidate.get(), HandleKind::Completion);
    } catch (...) {
      return rollback_unpublished_submission(
          candidate, event_guard,
          "linear attention completion registry allocation failed", error_sink);
    }
    if (token == 0U) {
      return rollback_unpublished_submission(
          candidate, event_guard,
          "linear attention completion token allocation failed", error_sink);
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
      return cleanup_failed_submission(
          candidate, token, timing_record_status,
          "hipEventRecord linear attention timing start", queue, error_sink);
    }
    const uint32_t active_slot = state->active_slot;
    const uint32_t inactive_slot = 1U - active_slot;
    const auto *const qkv = static_cast<const uint16_t *>(
        linear_binding_pointer(buffers[0], metadata.qkv.byte_offset));
    const auto *const z = static_cast<const uint16_t *>(
        linear_binding_pointer(buffers[1], metadata.z.byte_offset));
    const auto *const b_input = static_cast<const uint16_t *>(
        linear_binding_pointer(buffers[2], metadata.b_input.byte_offset));
    const auto *const a_input = static_cast<const uint16_t *>(
        linear_binding_pointer(buffers[3], metadata.a_input.byte_offset));
    const auto *const conv_weight = static_cast<const uint16_t *>(
        linear_binding_pointer(buffers[4], metadata.conv_weight.byte_offset));
    const auto *const a_log = static_cast<const float *>(
        linear_binding_pointer(buffers[5], metadata.a_log.byte_offset));
    const auto *const dt_bias = static_cast<const uint16_t *>(
        linear_binding_pointer(buffers[6], metadata.dt_bias.byte_offset));
    const auto *const norm_weight = static_cast<const float *>(
        linear_binding_pointer(buffers[7], metadata.norm_weight.byte_offset));
    auto *const output = static_cast<uint16_t *>(
        linear_binding_pointer(buffers[8], metadata.output.byte_offset));
    const hipError_t conv_status =
        ::sllm_linear_attention_kernel::launch_convolution(
            qkv, conv_weight,
            static_cast<const uint16_t *>(state->conv_state[active_slot]),
            static_cast<uint16_t *>(state->scratch),
            static_cast<uint16_t *>(state->conv_state[inactive_slot]),
            static_cast<uint32_t>(metadata.token_count), metadata.qkv_width,
            metadata.conv_kernel_size, queue->stream);
    if (conv_status != hipSuccess) {
      return cleanup_failed_submission(candidate, token, conv_status,
                                       "linear attention convolution launch",
                                       queue, error_sink);
    }
    if (use_row32_lds_provider) {
      const hipError_t recurrent_status =
          ::sllm_linear_attention_kernel::launch_row32_lds(
              static_cast<const uint16_t *>(state->scratch), z, b_input,
              a_input, a_log, dt_bias, norm_weight,
              static_cast<const float *>(state->recurrent_state[active_slot]),
              static_cast<float *>(state->recurrent_state[inactive_slot]),
              output, static_cast<uint32_t>(metadata.token_count),
              metadata.qk_heads, metadata.value_heads, metadata.head_dim,
              metadata.qkv_width, metadata.output_width, queue->stream);
      if (recurrent_status != hipSuccess) {
        return cleanup_failed_submission(
            candidate, token, recurrent_status,
            "linear attention row32-LDS recurrent launch", queue, error_sink);
      }
    } else if (use_decode_pair_provider) {
      const hipError_t recurrent_status =
          ::sllm_linear_attention_kernel::launch_decode_pair(
              static_cast<const uint16_t *>(state->scratch), z, b_input,
              a_input, a_log, dt_bias, norm_weight,
              static_cast<const float *>(state->recurrent_state[active_slot]),
              static_cast<float *>(state->recurrent_state[inactive_slot]),
              output, static_cast<uint32_t>(metadata.token_count),
              metadata.qk_heads, metadata.value_heads, metadata.head_dim,
              metadata.qkv_width, metadata.output_width, queue->stream);
      if (recurrent_status != hipSuccess) {
        return cleanup_failed_submission(
            candidate, token, recurrent_status,
            "linear attention decode-pair recurrent launch", queue, error_sink);
      }
    } else if (use_gfx942_wave64_column_provider) {
      auto *const scalar_base = reinterpret_cast<float *>(
          static_cast<char *>(state->scratch) + convolved_scratch);
      float *const beta = scalar_base;
      float *const decay =
          scalar_base + metadata.token_count * metadata.value_heads;
      const hipError_t preprocess_status = ::sllm_linear_attention_kernel::
          launch_gfx942_wave64_column_preprocess(
              static_cast<uint16_t *>(state->scratch), b_input, a_input, a_log,
              dt_bias, beta, decay, static_cast<uint32_t>(metadata.token_count),
              metadata.qk_heads, metadata.value_heads, metadata.head_dim,
              metadata.qkv_width, queue->stream);
      if (preprocess_status != hipSuccess) {
        return cleanup_failed_submission(
            candidate, token, preprocess_status,
            "linear attention gfx942 wave64 column preprocess launch", queue,
            error_sink);
      }
      const hipError_t recurrent_status =
          ::sllm_linear_attention_kernel::launch_gfx942_wave64_column_recurrent(
              static_cast<const uint16_t *>(state->scratch), beta, decay,
              static_cast<const float *>(state->recurrent_state[active_slot]),
              static_cast<float *>(state->recurrent_state[inactive_slot]),
              output, static_cast<uint32_t>(metadata.token_count),
              metadata.qk_heads, metadata.value_heads, metadata.head_dim,
              metadata.qkv_width, metadata.output_width, queue->stream);
      if (recurrent_status != hipSuccess) {
        return cleanup_failed_submission(
            candidate, token, recurrent_status,
            "linear attention gfx942 wave64 column recurrent launch", queue,
            error_sink);
      }
      const hipError_t postprocess_status = ::sllm_linear_attention_kernel::
          launch_gfx942_wave64_column_postprocess(
              z, norm_weight, output,
              static_cast<uint32_t>(metadata.token_count), metadata.value_heads,
              metadata.head_dim, metadata.output_width, queue->stream);
      if (postprocess_status != hipSuccess) {
        return cleanup_failed_submission(
            candidate, token, postprocess_status,
            "linear attention gfx942 wave64 column postprocess launch", queue,
            error_sink);
      }
    } else if (use_column_provider) {
      auto *const scalar_base = reinterpret_cast<float *>(
          static_cast<char *>(state->scratch) + convolved_scratch);
      float *const beta = scalar_base;
      float *const decay =
          scalar_base + metadata.token_count * metadata.value_heads;
      const hipError_t preprocess_status =
          ::sllm_linear_attention_kernel::launch_column_preprocess(
              static_cast<uint16_t *>(state->scratch), b_input, a_input, a_log,
              dt_bias, beta, decay, static_cast<uint32_t>(metadata.token_count),
              metadata.qk_heads, metadata.value_heads, metadata.head_dim,
              metadata.qkv_width, queue->stream);
      if (preprocess_status != hipSuccess) {
        return cleanup_failed_submission(
            candidate, token, preprocess_status,
            "linear attention column preprocess launch", queue, error_sink);
      }
      const hipError_t recurrent_status =
          ::sllm_linear_attention_kernel::launch_column_recurrent(
              static_cast<const uint16_t *>(state->scratch), beta, decay,
              static_cast<const float *>(state->recurrent_state[active_slot]),
              static_cast<float *>(state->recurrent_state[inactive_slot]),
              output, static_cast<uint32_t>(metadata.token_count),
              metadata.qk_heads, metadata.value_heads, metadata.head_dim,
              metadata.qkv_width, metadata.output_width, queue->stream);
      if (recurrent_status != hipSuccess) {
        return cleanup_failed_submission(
            candidate, token, recurrent_status,
            "linear attention column recurrent launch", queue, error_sink);
      }
      const hipError_t postprocess_status =
          ::sllm_linear_attention_kernel::launch_column_postprocess(
              z, norm_weight, output,
              static_cast<uint32_t>(metadata.token_count), metadata.value_heads,
              metadata.head_dim, metadata.output_width, queue->stream);
      if (postprocess_status != hipSuccess) {
        return cleanup_failed_submission(
            candidate, token, postprocess_status,
            "linear attention column postprocess launch", queue, error_sink);
      }
    } else {
      const hipError_t recurrent_status =
          ::sllm_linear_attention_kernel::launch_recurrent(
              static_cast<const uint16_t *>(state->scratch), z, b_input,
              a_input, a_log, dt_bias, norm_weight,
              static_cast<const float *>(state->recurrent_state[active_slot]),
              static_cast<float *>(state->recurrent_state[inactive_slot]),
              output, static_cast<uint32_t>(metadata.token_count),
              metadata.qk_heads, metadata.value_heads, metadata.head_dim,
              metadata.qkv_width, metadata.output_width, queue->stream);
      if (recurrent_status != hipSuccess) {
        return cleanup_failed_submission(candidate, token, recurrent_status,
                                         "linear attention recurrent launch",
                                         queue, error_sink);
      }
    }
    const hipError_t record_status =
        candidate->event != nullptr
            ? hipEventRecord(candidate->event, queue->stream)
            : hipSuccess;
    if (record_status != hipSuccess) {
      return cleanup_failed_submission(candidate, token, record_status,
                                       "hipEventRecord linear attention", queue,
                                       error_sink);
    }
    fill_linear_attention_dispatch_info(
        dispatch_info, dispatch_id, metadata, arch_name, use_column_provider,
        use_gfx942_wave64_column_provider, use_decode_pair_provider,
        use_row32_lds_provider);
    *completion_output = reinterpret_cast<sllm_completion_t *>(token);
    (void)candidate.release();
    execute_guard.disarm();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in linear attention execute");
  }
}

extern "C" sllm_status_t sllm_linear_attention_cancel(
    const sllm_linear_attention_state_t *const raw_state,
    sllm_completion_t *const raw_completion,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_state == nullptr || raw_completion == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "linear attention cancel input is null");
    }
    LinearAttentionState *state = nullptr;
    Completion *completion = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      state = lookup<LinearAttentionState>(raw_state,
                                           HandleKind::LinearAttentionState);
      completion = lookup<Completion>(raw_completion, HandleKind::Completion);
      if (state == nullptr || completion == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "linear attention state or completion handle is stale");
      }
    }
    std::lock_guard<std::mutex> state_lock(completion->state_mutex);
    if (!completion->linear_attention ||
        completion->linear_attention_state != state) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "completion does not belong to the supplied linear attention state");
    }
    if (completion->terminal) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "linear attention completion is already terminal");
    }
    if (!revoke_linear_attention_commit(completion)) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "linear attention cancellation failed; context poisoned");
    }
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in linear attention cancel");
  }
}
