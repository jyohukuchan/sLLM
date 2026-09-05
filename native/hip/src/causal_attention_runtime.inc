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

constexpr uint64_t kScaledPrefillChunkRows = 256U;
constexpr uint64_t kScaledPrefillQGroup = 4U;
constexpr uint64_t kScaledPrefillPackedRows =
    kScaledPrefillChunkRows * kScaledPrefillQGroup;
constexpr uint64_t kScaledPrefillHeadDim = 256U;
// The prototype processes at most 256 query rows per chunk.  Keep its
// temporary allocation explicitly bounded so pathological lengths fail before
// hipMalloc rather than attempting an M×M score matrix or overflowing a size.
constexpr uint64_t kScaledPrefillMaxWorkspaceBytes = UINT64_C(1) << 30;
constexpr uint32_t kDecodeGqa4SplitPartitions = 16U;
constexpr uint32_t kDecodeGqa4SplitP32Partitions = 32U;
constexpr uint32_t kDecodeGqa4SplitHeadDim = 256U;
constexpr uint64_t kDecodeGqa4SplitWorkspaceBytes =
    static_cast<uint64_t>(16U) * kDecodeGqa4SplitPartitions *
    (kDecodeGqa4SplitHeadDim + 2U) * sizeof(float);
constexpr uint64_t kDecodeGqa4SplitP32WorkspaceBytes =
    static_cast<uint64_t>(16U) * kDecodeGqa4SplitP32Partitions *
    (kDecodeGqa4SplitHeadDim + 2U) * sizeof(float);
constexpr uint32_t kDecodeGqa6SplitP32Partitions = 32U;
constexpr uint32_t kDecodeGqa6SplitP32QHeads = 24U;
constexpr uint32_t kDecodeGqa6SplitP32KvHeads = 4U;
constexpr uint32_t kDecodeGqa6SplitP32HeadDim = 256U;
constexpr uint32_t kDecodeGqa6SplitP32WorkgroupSize = 192U;
constexpr uint64_t kDecodeGqa6SplitP32WorkspaceBytes =
    static_cast<uint64_t>(kDecodeGqa6SplitP32QHeads) *
    kDecodeGqa6SplitP32Partitions * (kDecodeGqa6SplitP32HeadDim + 2U) *
    sizeof(float);
constexpr uint32_t kDecodeGqa6SplitP64Partitions = 64U;
constexpr uint32_t kDecodeGqa6SplitP64QHeads = 24U;
constexpr uint32_t kDecodeGqa6SplitP64KvHeads = 4U;
constexpr uint32_t kDecodeGqa6SplitP64HeadDim = 256U;
constexpr uint32_t kDecodeGqa6SplitP64WorkgroupSize = 192U;
constexpr uint64_t kDecodeGqa6SplitP64WorkspaceBytes =
    static_cast<uint64_t>(kDecodeGqa6SplitP64QHeads) *
    kDecodeGqa6SplitP64Partitions * (kDecodeGqa6SplitP64HeadDim + 2U) *
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
constexpr uint32_t kLongPrefillV2Partitions = 16U;
constexpr uint32_t kLongPrefillV2ChunkRows = 256U;
constexpr uint32_t kLongPrefillV2QHeads = 16U;
constexpr uint32_t kLongPrefillV2HeadDim = 256U;
constexpr uint64_t kLongPrefillV2WorkspaceBytes =
    static_cast<uint64_t>(kLongPrefillV2ChunkRows) * kLongPrefillV2QHeads *
    kLongPrefillV2Partitions * (kLongPrefillV2HeadDim + 2U) * sizeof(float);

bool causal_scaled_prefill_workspace_bytes(const uint64_t key_length,
                                           uint64_t *const result) noexcept {
  if (result == nullptr || key_length == 0U ||
      key_length > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return false;
  }
  const auto checked_mul = [](const uint64_t left, const uint64_t right,
                              uint64_t *const output) noexcept -> bool {
    if (output == nullptr || (right != 0U && left > UINT64_MAX / right)) {
      return false;
    }
    *output = left * right;
    return true;
  };
  uint64_t score = 0U;
  uint64_t probability = 0U;
  uint64_t residual = 0U;
  uint64_t kv = 0U;
  uint64_t query = 0U;
  uint64_t scales = 0U;
  uint64_t output = 0U;
  uint64_t special = 0U;
  if (!checked_mul(key_length, kScaledPrefillPackedRows * sizeof(float),
                   &score) ||
      !checked_mul(key_length, kScaledPrefillPackedRows * sizeof(uint16_t),
                   &probability) ||
      !checked_mul(key_length, kScaledPrefillPackedRows * sizeof(uint16_t),
                   &residual) ||
      !checked_mul(key_length, kScaledPrefillHeadDim * sizeof(uint16_t), &kv) ||
      !checked_mul(kScaledPrefillPackedRows * kScaledPrefillHeadDim,
                   sizeof(uint16_t), &query) ||
      !checked_mul(kScaledPrefillPackedRows, sizeof(float), &scales) ||
      !checked_mul(kScaledPrefillPackedRows * kScaledPrefillHeadDim,
                   sizeof(float), &output)) {
    return false;
  }
  if (!checked_mul(4U * kScaledPrefillHeadDim * 3U, sizeof(uint64_t),
                   &special)) {
    return false;
  }
  const auto add = [](uint64_t *const total, const uint64_t value) noexcept {
    if (total == nullptr || *total > UINT64_MAX - value) {
      return false;
    }
    *total += value;
    return true;
  };
  uint64_t total = score;
  if (!add(&total, probability) || !add(&total, residual) || !add(&total, kv) ||
      !add(&total, kv) || !add(&total, query) || !add(&total, scales) ||
      !add(&total, output) || !add(&total, output) || !add(&total, special) ||
      total > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      total > kScaledPrefillMaxWorkspaceBytes) {
    return false;
  }
  *result = total;
  return true;
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

hipError_t prepare_causal_scaled_prefill_plan(
    Context *const context, Queue *const queue, const uint64_t key_length,
    void **const workspace, uint64_t *const workspace_bytes) noexcept {
  uint64_t required = 0U;
  if (!causal_scaled_prefill_workspace_bytes(key_length, &required)) {
    return hipErrorOutOfMemory;
  }
  return prepare_causal_workspace_plan_for_bytes(context, queue, required,
                                                 workspace, workspace_bytes);
}

#if !defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
// The generic context constructor only creates a rocBLAS handle for matmul
// providers which request one at context-open time.  The gfx1201 GQA6
// attention provider is a causal-attention opt-in, so keep its handle
// initialization in this owned runtime path.  The handle is stored on the
// context and is therefore reused by later attention and matmul submissions;
// Context::~Context() owns its eventual destruction.
hipError_t ensure_gqa6_rocblas_handle(Context *const context) noexcept {
  if (context == nullptr) {
    return hipErrorInvalidValue;
  }
  std::lock_guard<std::mutex> lock(context->matmul_blas_mutex);
  if (context->matmul_rocblas_handle != nullptr) {
    return hipSuccess;
  }
  rocblas_handle handle = nullptr;
  if (rocblas_create_handle(&handle) != rocblas_status_success) {
    return hipErrorUnknown;
  }
  if (rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host) !=
      rocblas_status_success) {
    (void)rocblas_destroy_handle(handle);
    return hipErrorUnknown;
  }
  context->matmul_rocblas_handle = handle;
  return hipSuccess;
}
#endif

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

struct CausalAttentionProviderSelection final {
  bool use_gfx1201_wave_provider;
  bool use_decode_wave_split;
  bool use_decode_wave_split_fp16_pair;
  bool use_decode_gqa4_split;
  bool use_decode_gqa4_split_p32;
  bool use_decode_gqa6_split_p128;
  bool use_decode_gqa6_split_p64;
  bool use_decode_gqa6_split_p32;
  bool use_prefill_gqa6_blocksoftmax_q8;
  bool use_prefill_gqa6_blocksoftmax;
  bool use_decode_wave_split_q_preload;
  bool use_prefill_gqa4;
  bool use_prefill_gqa4_qtile4;
  bool use_prefill_gqa6_qtile4_k4_fp16;
  bool use_prefill_gqa6_qtile4_k8_fp16;
  bool use_prefill_gqa6_qtile4_k16_fp16;
  bool use_prefill_gqa6_qtile4_k32_fp16;
  bool use_gqa6_rocblas_f32;
  bool use_gfx1201_gqa6_rocblas_f32;
  bool use_gfx1201_gqa6_rocblas_f16_tail;
  bool use_scaled_prefill_gemm;
  bool use_long_prefill_v2;
  ::sllm_causal_attention_kernel::PrefillTilePolicy prefill_tile_policy =
      ::sllm_causal_attention_kernel::PrefillTilePolicy::Q4K1Control;
};

struct PrefillTileKey final {
  const char *target;
  uint32_t kv_encoding;
  uint32_t query_heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint64_t query_count;
  uint64_t context_length;
  uint64_t sliding_window;
  bool explicit_score_scale;
};

enum class Gqa6QTile4Fp16KeyTile : uint32_t {
  None = 0U,
  K4 = 4U,
  K8 = 8U,
  K16 = 16U,
  K32 = 32U,
};

// Candidate selection is intentionally a single-valued policy.  If several
// opt-ins are present, the smallest requested key tile wins in the explicit
// order K4 > K8 > K16 > K32.  This keeps process-environment composition
// deterministic and prevents multiple candidate bits from being published.
Gqa6QTile4Fp16KeyTile select_gqa6_qtile4_fp16_key_tile(
    const bool eligible, const char *const k4_opt_in,
    const char *const k8_opt_in, const char *const k16_opt_in,
    const char *const k32_opt_in) noexcept {
  if (!eligible) {
    return Gqa6QTile4Fp16KeyTile::None;
  }
  const auto enabled = [](const char *const value) noexcept {
    return value != nullptr && std::strcmp(value, "1") == 0;
  };
  if (enabled(k4_opt_in)) {
    return Gqa6QTile4Fp16KeyTile::K4;
  }
  if (enabled(k8_opt_in)) {
    return Gqa6QTile4Fp16KeyTile::K8;
  }
  if (enabled(k16_opt_in)) {
    return Gqa6QTile4Fp16KeyTile::K16;
  }
  if (enabled(k32_opt_in)) {
    return Gqa6QTile4Fp16KeyTile::K32;
  }
  return Gqa6QTile4Fp16KeyTile::None;
}

::sllm_causal_attention_kernel::PrefillTilePolicy
select_phase66_prefill_tile(const PrefillTileKey &key,
                            const bool candidate_enabled) noexcept {
  if (!candidate_enabled || std::strcmp(key.target, "gfx1201") != 0 ||
      (key.kv_encoding != SLLM_HIP_KV_ENCODING_FP16_V1 &&
       key.kv_encoding != SLLM_HIP_KV_ENCODING_MXFP8_E4_V1) ||
      key.query_heads != 16U || key.kv_heads != 4U ||
      key.head_dim != SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM ||
      key.query_count < 128U || key.context_length < key.query_count ||
      key.sliding_window != 0U || key.explicit_score_scale) {
    return ::sllm_causal_attention_kernel::PrefillTilePolicy::Q4K1Control;
  }
  if (key.context_length >= 2048U) {
    return ::sllm_causal_attention_kernel::PrefillTilePolicy::Q8K8;
  }
  return key.context_length >= 512U
             ? ::sllm_causal_attention_kernel::PrefillTilePolicy::Q4K8
             : ::sllm_causal_attention_kernel::PrefillTilePolicy::Q4K4;
}

CausalAttentionProviderSelection select_causal_attention_provider(
    const sllm_causal_attention::DescriptorMetadata &metadata,
    const uint32_t kv_heads, const uint32_t encoding,
    const char *const arch_name) noexcept {
  if (metadata.sliding_window != 0U || metadata.explicit_score_scale) {
    return {};
  }
  const bool use_phase33_common_provider =
      std::strcmp(arch_name, "gfx1030") == 0 ||
      std::strcmp(arch_name, "gfx1201") == 0;
  const char *const force_baseline =
      std::getenv("SLLM_CAUSAL_ATTENTION_FORCE_BASELINE");
  const char *const q_preload_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_GFX1030_Q_PRELOAD");
  const char *const short_decode_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_WAVE_SHORT");
  const char *const short_q_preload_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_WAVE_SHORT_Q_PRELOAD");
  const char *const fp16_pair_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_WAVE_FP16_PAIR");
  const char *const gqa4_split_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_GQA4_SPLIT");
  const char *const gqa4_split_p32_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_GQA4_SPLIT_P32");
  const char *const gfx1201_gqa4_split_p32_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_GFX1201_DECODE_GQA4_SPLIT_P32");
  const char *const gqa6_split_p32_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_GQA6_DECODE_SPLIT_P32");
  const char *const gqa6_split_p64_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_GQA6_DECODE_SPLIT_P64");
  const char *const gqa6_split_p128_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_GQA6_DECODE_SPLIT_P128");
  const char *const gqa6_blocksoftmax_gfx1030_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_BLOCKSOFTMAX_GFX1030");
  const char *const gqa6_blocksoftmax_gfx1201_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_BLOCKSOFTMAX_GFX1201");
  const char *const gqa6_blocksoftmax_q8_gfx1201_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_BLOCKSOFTMAX_Q8_GFX1201");
  const char *const phase66_tiled_prefill_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_PHASE66_TILED_PREFILL");
  const char *const gqa6_qtile4_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_GQA6_QTILE4");
  const char *const gqa6_qtile4_k4_fp16_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_GQA6_QTILE4_K4_FP16");
  const char *const gqa6_qtile4_k8_fp16_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_GQA6_QTILE4_K8_FP16");
  const char *const gqa6_qtile4_k16_fp16_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_GQA6_QTILE4_K16_FP16");
  const char *const gqa6_qtile4_k32_fp16_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_GQA6_QTILE4_K32_FP16");
  const char *const gqa6_rocblas_f32_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_GFX1030_ROCBLAS_F32");
  const char *const gfx1201_gqa6_rocblas_f32_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_GFX1201_ROCBLAS_F32");
  const char *const gfx1201_gqa6_rocblas_f16_tail_opt_in = std::getenv(
      "SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_GFX1201_ROCBLAS_F16_TAIL");
  const bool allow_prefill_provider =
      force_baseline == nullptr || std::strcmp(force_baseline, "1") != 0;
  const bool use_gqa6_rocblas_f32 =
      allow_prefill_provider && std::strcmp(arch_name, "gfx1030") == 0 &&
      gqa6_rocblas_f32_opt_in != nullptr &&
      std::strcmp(gqa6_rocblas_f32_opt_in, "1") == 0 &&
      metadata.query_count > 1U &&
      metadata.query_count <= static_cast<uint64_t>(UINT32_MAX) &&
      metadata.start_position <= UINT64_MAX - metadata.query_count &&
      metadata.start_position + metadata.query_count ==
          metadata.expected_kv_length &&
      metadata.query.q_heads == 24U && kv_heads == 4U &&
      metadata.query.head_dim == SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM &&
      encoding == SLLM_HIP_KV_ENCODING_FP16_V1;
  const bool use_gfx1201_gqa6_rocblas_f32 =
      allow_prefill_provider && std::strcmp(arch_name, "gfx1201") == 0 &&
      gfx1201_gqa6_rocblas_f32_opt_in != nullptr &&
      std::strcmp(gfx1201_gqa6_rocblas_f32_opt_in, "1") == 0 &&
      metadata.query_count > 1U &&
      metadata.query_count <= static_cast<uint64_t>(UINT32_MAX) &&
      metadata.start_position <= UINT64_MAX - metadata.query_count &&
      metadata.start_position + metadata.query_count ==
          metadata.expected_kv_length &&
      metadata.query.q_heads == 24U && kv_heads == 4U &&
      metadata.query.head_dim == SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM &&
      encoding == SLLM_HIP_KV_ENCODING_FP16_V1;
  const bool use_gfx1201_gqa6_rocblas_f16_tail =
      use_gfx1201_gqa6_rocblas_f32 && metadata.start_position > 0U &&
      gfx1201_gqa6_rocblas_f16_tail_opt_in != nullptr &&
      std::strcmp(gfx1201_gqa6_rocblas_f16_tail_opt_in, "1") == 0;
  const bool use_any_gqa6_rocblas_f32 =
      use_gqa6_rocblas_f32 || use_gfx1201_gqa6_rocblas_f32;
  const bool use_decode_wave_split_long =
      use_phase33_common_provider && metadata.query_count == 1U &&
      metadata.expected_kv_length >= 1024U &&
      metadata.query.head_dim == SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM;
  const bool use_decode_wave_split_short =
      std::strcmp(arch_name, "gfx1030") == 0 &&
      (short_decode_opt_in == nullptr ||
       std::strcmp(short_decode_opt_in, "1") == 0) &&
      metadata.query_count == 1U && metadata.expected_kv_length >= 32U &&
      metadata.expected_kv_length < 1024U && metadata.query.q_heads == 16U &&
      kv_heads == 4U &&
      metadata.query.head_dim == SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM &&
      encoding == SLLM_HIP_KV_ENCODING_FP16_V1;
  const bool use_decode_wave_split =
      use_decode_wave_split_long || use_decode_wave_split_short;
  const bool use_decode_wave_split_fp16_pair =
      std::strcmp(arch_name, "gfx1030") == 0 &&
      (fp16_pair_opt_in == nullptr ||
       std::strcmp(fp16_pair_opt_in, "1") == 0) &&
      allow_prefill_provider && metadata.query_count == 1U &&
      metadata.expected_kv_length >= 1024U && metadata.query.q_heads == 16U &&
      kv_heads == 4U && metadata.query.head_dim == 256U &&
      encoding == SLLM_HIP_KV_ENCODING_FP16_V1;
  const bool use_decode_gqa4_split =
      allow_prefill_provider && std::strcmp(arch_name, "gfx1030") == 0 &&
      gqa4_split_opt_in != nullptr &&
      std::strcmp(gqa4_split_opt_in, "1") == 0 && metadata.query_count == 1U &&
      metadata.expected_kv_length >= 4096U && metadata.query.q_heads == 16U &&
      kv_heads == 4U && metadata.query.head_dim == 256U &&
      encoding == SLLM_HIP_KV_ENCODING_FP16_V1;
  const bool use_decode_gqa4_split_p32 =
      allow_prefill_provider &&
      ((std::strcmp(arch_name, "gfx1030") == 0 &&
        (gqa4_split_p32_opt_in == nullptr ||
         std::strcmp(gqa4_split_p32_opt_in, "1") == 0)) ||
       (std::strcmp(arch_name, "gfx1201") == 0 &&
        (gfx1201_gqa4_split_p32_opt_in == nullptr ||
         std::strcmp(gfx1201_gqa4_split_p32_opt_in, "1") == 0))) &&
      metadata.query_count == 1U && metadata.expected_kv_length >= 4096U &&
      metadata.query.q_heads == 16U && kv_heads == 4U &&
      metadata.query.head_dim == 256U &&
      encoding == SLLM_HIP_KV_ENCODING_FP16_V1;
  const bool use_decode_gqa6_split_p128 =
      allow_prefill_provider && std::strcmp(arch_name, "gfx1030") == 0 &&
      gqa6_split_p128_opt_in != nullptr &&
      std::strcmp(gqa6_split_p128_opt_in, "1") == 0 &&
      metadata.query_count == 1U && metadata.expected_kv_length >= 8192U &&
      metadata.query.q_heads == kDecodeGqa6SplitP128QHeads &&
      kv_heads == kDecodeGqa6SplitP128KvHeads &&
      metadata.query.head_dim == kDecodeGqa6SplitP128HeadDim &&
      encoding == SLLM_HIP_KV_ENCODING_FP16_V1 &&
      metadata.sliding_window == 0U && !metadata.explicit_score_scale;
  const bool use_decode_gqa6_split_p64 =
      // P64 is the rollback for P128.  P128 is evaluated first above so a
      // process that requests both partition counts deterministically selects
      // the larger split; the P32 expression below is then mutually exclusive.
      allow_prefill_provider && use_phase33_common_provider &&
      gqa6_split_p64_opt_in != nullptr &&
      std::strcmp(gqa6_split_p64_opt_in, "1") == 0 &&
      metadata.query_count == 1U &&
      ((std::strcmp(arch_name, "gfx1030") == 0 &&
        metadata.expected_kv_length >= 8192U) ||
       (std::strcmp(arch_name, "gfx1201") == 0 &&
        metadata.expected_kv_length >= 4096U)) &&
      metadata.query.q_heads == kDecodeGqa6SplitP64QHeads &&
      kv_heads == kDecodeGqa6SplitP64KvHeads &&
      metadata.query.head_dim == kDecodeGqa6SplitP64HeadDim &&
      encoding == SLLM_HIP_KV_ENCODING_FP16_V1 && !use_decode_gqa6_split_p128;
  const bool use_decode_gqa6_split_p32 =
      allow_prefill_provider && use_phase33_common_provider &&
      gqa6_split_p32_opt_in != nullptr &&
      std::strcmp(gqa6_split_p32_opt_in, "1") == 0 &&
      metadata.query_count == 1U && metadata.expected_kv_length >= 4096U &&
      metadata.query.q_heads == kDecodeGqa6SplitP32QHeads &&
      kv_heads == kDecodeGqa6SplitP32KvHeads &&
      metadata.query.head_dim == kDecodeGqa6SplitP32HeadDim &&
      encoding == SLLM_HIP_KV_ENCODING_FP16_V1 && !use_decode_gqa6_split_p128 &&
      !use_decode_gqa6_split_p64;
  const bool use_prefill_gqa6_blocksoftmax_q8 =
      allow_prefill_provider && std::strcmp(arch_name, "gfx1201") == 0 &&
      metadata.query_count >= 128U && metadata.query.q_heads == 24U &&
      kv_heads == 4U && metadata.query.head_dim == 256U &&
      encoding == SLLM_HIP_KV_ENCODING_FP16_V1 &&
      gqa6_blocksoftmax_q8_gfx1201_opt_in != nullptr &&
      std::strcmp(gqa6_blocksoftmax_q8_gfx1201_opt_in, "1") == 0 &&
      !use_any_gqa6_rocblas_f32;
  const bool use_prefill_gqa6_blocksoftmax =
      allow_prefill_provider && use_phase33_common_provider &&
      metadata.query_count >= 128U && metadata.query.q_heads == 24U &&
      kv_heads == 4U && metadata.query.head_dim == 256U &&
      encoding == SLLM_HIP_KV_ENCODING_FP16_V1 &&
      ((std::strcmp(arch_name, "gfx1030") == 0 &&
        gqa6_blocksoftmax_gfx1030_opt_in != nullptr &&
        std::strcmp(gqa6_blocksoftmax_gfx1030_opt_in, "1") == 0) ||
       (std::strcmp(arch_name, "gfx1201") == 0 &&
        gqa6_blocksoftmax_gfx1201_opt_in != nullptr &&
        std::strcmp(gqa6_blocksoftmax_gfx1201_opt_in, "1") == 0)) &&
      !use_prefill_gqa6_blocksoftmax_q8 && !use_any_gqa6_rocblas_f32;
  const bool use_decode_wave_split_short_q_preload =
      use_decode_wave_split_short &&
      (short_q_preload_opt_in == nullptr ||
       std::strcmp(short_q_preload_opt_in, "1") == 0);
  const bool use_decode_wave_split_q_preload_long =
      std::strcmp(arch_name, "gfx1030") == 0 && use_decode_wave_split_long &&
      (q_preload_opt_in == nullptr || std::strcmp(q_preload_opt_in, "1") == 0);
  const bool use_decode_wave_split_q_preload =
      use_decode_wave_split_q_preload_long ||
      use_decode_wave_split_short_q_preload;
  const bool phase66_tiled_prefill_requested =
      phase66_tiled_prefill_opt_in != nullptr &&
      std::strcmp(phase66_tiled_prefill_opt_in, "1") == 0;
  const bool phase66_q4k1_control =
      phase66_tiled_prefill_requested &&
      std::strcmp(arch_name, "gfx1201") == 0 &&
      (encoding == SLLM_HIP_KV_ENCODING_FP16_V1 ||
       encoding == SLLM_HIP_KV_ENCODING_MXFP8_E4_V1) &&
      metadata.query_count >= 64U && metadata.query.q_heads == 16U &&
      kv_heads == 4U &&
      metadata.query.head_dim == SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM;
  const bool use_prefill_gqa4_qtile4_control =
      allow_prefill_provider && use_phase33_common_provider &&
      !use_prefill_gqa6_blocksoftmax_q8 && !use_any_gqa6_rocblas_f32 &&
      (metadata.query_count >= 128U || phase66_q4k1_control) &&
      (metadata.query.q_heads / kv_heads == 4U ||
       (metadata.query.q_heads / kv_heads == 6U &&
        gqa6_qtile4_opt_in != nullptr &&
        std::strcmp(gqa6_qtile4_opt_in, "1") == 0)) &&
      metadata.query.head_dim == SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM;
  const bool gqa6_qtile4_fp16_shape =
      allow_prefill_provider && use_phase33_common_provider &&
      !use_prefill_gqa6_blocksoftmax_q8 && !use_any_gqa6_rocblas_f32 &&
      metadata.query_count >= 128U && metadata.query.q_heads == 24U &&
      kv_heads == 4U && metadata.query.head_dim == 256U &&
      encoding == SLLM_HIP_KV_ENCODING_FP16_V1 &&
      metadata.sliding_window == 0U && !metadata.explicit_score_scale;
  const Gqa6QTile4Fp16KeyTile gqa6_qtile4_fp16_key_tile =
      select_gqa6_qtile4_fp16_key_tile(
          gqa6_qtile4_fp16_shape, gqa6_qtile4_k4_fp16_opt_in,
          gqa6_qtile4_k8_fp16_opt_in, gqa6_qtile4_k16_fp16_opt_in,
          gqa6_qtile4_k32_fp16_opt_in);
  const bool use_prefill_gqa6_qtile4_k4_fp16 =
      gqa6_qtile4_fp16_key_tile == Gqa6QTile4Fp16KeyTile::K4;
  const bool use_prefill_gqa6_qtile4_k8_fp16 =
      gqa6_qtile4_fp16_key_tile == Gqa6QTile4Fp16KeyTile::K8;
  const bool use_prefill_gqa6_qtile4_k16_fp16 =
      gqa6_qtile4_fp16_key_tile == Gqa6QTile4Fp16KeyTile::K16;
  const bool use_prefill_gqa6_qtile4_k32_fp16 =
      gqa6_qtile4_fp16_key_tile == Gqa6QTile4Fp16KeyTile::K32;
  const bool use_prefill_gqa4_qtile4 =
      use_prefill_gqa4_qtile4_control || use_prefill_gqa6_qtile4_k4_fp16 ||
      use_prefill_gqa6_qtile4_k8_fp16 || use_prefill_gqa6_qtile4_k16_fp16 ||
      use_prefill_gqa6_qtile4_k32_fp16;
  // Candidate-only typed tile policy. The key contains only target, KV
  // encoding, GQA shape, head dimension, query/context boundaries and the
  // attention semantics implemented by the candidate. Model, layer, prompt
  // and measured performance never participate. Unset, invalid, or
  // out-of-domain requests retain the existing q4k1 control path.
  const bool use_phase66_tiled_prefill =
      allow_prefill_provider && phase66_tiled_prefill_requested;
  const ::sllm_causal_attention_kernel::PrefillTilePolicy prefill_tile_policy =
      select_phase66_prefill_tile(
          {arch_name, encoding, metadata.query.q_heads, kv_heads,
           metadata.query.head_dim, metadata.query_count,
           metadata.expected_kv_length, metadata.sliding_window,
           metadata.explicit_score_scale},
          use_phase66_tiled_prefill);
  const char *const scaled_prefill_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_GFX1030_SCALED_PREFILL_GEMM");
  const char *const long_prefill_v2_opt_in =
      std::getenv("SLLM_CAUSAL_ATTENTION_GFX1030_LONG_PREFILL_V2");
  const bool use_long_prefill_v2 =
      allow_prefill_provider && std::strcmp(arch_name, "gfx1030") == 0 &&
      long_prefill_v2_opt_in != nullptr &&
      std::strcmp(long_prefill_v2_opt_in, "1") == 0 &&
      metadata.query_count >= 1024U && metadata.query.q_heads == 16U &&
      kv_heads == 4U && metadata.query.head_dim == 256U &&
      encoding == SLLM_HIP_KV_ENCODING_FP16_V1;
  const bool scaled_prefill_encoding_enabled =
      (encoding == SLLM_HIP_KV_ENCODING_FP16_V1 &&
       (scaled_prefill_opt_in == nullptr ||
        std::strcmp(scaled_prefill_opt_in, "1") == 0)) ||
      (encoding == SLLM_HIP_KV_ENCODING_MXFP8_E4_V1 &&
       scaled_prefill_opt_in != nullptr &&
       std::strcmp(scaled_prefill_opt_in, "1") == 0);
  const bool use_scaled_prefill_gemm =
      allow_prefill_provider && std::strcmp(arch_name, "gfx1030") == 0 &&
      scaled_prefill_encoding_enabled && metadata.query_count >= 1024U &&
      metadata.query.q_heads == 16U && kv_heads == 4U &&
      metadata.query.head_dim == 256U && !use_long_prefill_v2;
  return {
      std::strcmp(arch_name, "gfx1201") == 0 &&
          (metadata.query_count == 1U || metadata.query_count >= 32U),
      use_decode_wave_split,
      use_decode_wave_split_fp16_pair && !use_decode_gqa4_split &&
          !use_decode_gqa4_split_p32 && !use_decode_gqa6_split_p128 &&
          !use_decode_gqa6_split_p64 && !use_decode_gqa6_split_p32,
      use_decode_gqa4_split && !use_decode_gqa4_split_p32 &&
          !use_decode_gqa6_split_p128 && !use_decode_gqa6_split_p64 &&
          !use_decode_gqa6_split_p32,
      use_decode_gqa4_split_p32,
      use_decode_gqa6_split_p128,
      use_decode_gqa6_split_p64,
      use_decode_gqa6_split_p32,
      use_prefill_gqa6_blocksoftmax_q8,
      use_prefill_gqa6_blocksoftmax,
      use_decode_wave_split_q_preload,
      use_phase33_common_provider && metadata.query_count >= 64U &&
          metadata.query.q_heads / kv_heads == 4U &&
          metadata.query.head_dim == SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM,
      use_prefill_gqa4_qtile4,
      use_prefill_gqa6_qtile4_k4_fp16,
      use_prefill_gqa6_qtile4_k8_fp16,
      use_prefill_gqa6_qtile4_k16_fp16,
      use_prefill_gqa6_qtile4_k32_fp16,
      use_gqa6_rocblas_f32,
      use_gfx1201_gqa6_rocblas_f32,
      use_gfx1201_gqa6_rocblas_f16_tail,
      use_scaled_prefill_gemm,
      use_long_prefill_v2,
      prefill_tile_policy,
  };
}

// Publish the exact provider selected for workspace preparation and launch.
// Re-selecting here would allow a concurrent process-environment change to
// report a symbol/grid that was not enqueued.
void fill_causal_attention_dispatch_info(
    sllm_causal_attention_dispatch_info_t *const info,
    const uint64_t dispatch_id,
    const sllm_causal_attention::DescriptorMetadata &metadata,
    const uint32_t kv_heads, const uint32_t encoding,
    const char *const arch_name,
    const CausalAttentionProviderSelection &selection) noexcept {
  const bool use_gfx1201_wave_provider = selection.use_gfx1201_wave_provider;
  const bool use_decode_wave_split = selection.use_decode_wave_split;
  const bool use_decode_wave_split_fp16_pair =
      selection.use_decode_wave_split_fp16_pair;
  const bool use_decode_gqa4_split = selection.use_decode_gqa4_split;
  const bool use_decode_gqa4_split_p32 = selection.use_decode_gqa4_split_p32;
  const bool use_decode_gqa6_split_p32 = selection.use_decode_gqa6_split_p32;
  const bool use_decode_gqa6_split_p128 = selection.use_decode_gqa6_split_p128;
  const bool use_decode_gqa6_split_p64 = selection.use_decode_gqa6_split_p64;
  const bool use_prefill_gqa6_blocksoftmax_q8 =
      selection.use_prefill_gqa6_blocksoftmax_q8;
  const bool use_prefill_gqa6_blocksoftmax =
      selection.use_prefill_gqa6_blocksoftmax;
  const bool use_decode_wave_split_q_preload =
      selection.use_decode_wave_split_q_preload;
  const bool use_prefill_gqa4 = selection.use_prefill_gqa4;
  const bool use_prefill_gqa4_qtile4 = selection.use_prefill_gqa4_qtile4;
  const bool use_prefill_gqa6_qtile4_k4_fp16 =
      selection.use_prefill_gqa6_qtile4_k4_fp16;
  const bool use_prefill_gqa6_qtile4_k8_fp16 =
      selection.use_prefill_gqa6_qtile4_k8_fp16;
  const bool use_prefill_gqa6_qtile4_k16_fp16 =
      selection.use_prefill_gqa6_qtile4_k16_fp16;
  const bool use_prefill_gqa6_qtile4_k32_fp16 =
      selection.use_prefill_gqa6_qtile4_k32_fp16;
  const bool use_gqa6_rocblas_f32 = selection.use_gqa6_rocblas_f32;
  const bool use_gfx1201_gqa6_rocblas_f32 =
      selection.use_gfx1201_gqa6_rocblas_f32;
  const bool use_gfx1201_gqa6_rocblas_f16_tail =
      selection.use_gfx1201_gqa6_rocblas_f16_tail;
  const bool use_any_gqa6_rocblas_f32 =
      use_gqa6_rocblas_f32 || use_gfx1201_gqa6_rocblas_f32;
  const bool use_scaled_prefill_gemm = selection.use_scaled_prefill_gemm;
  const bool use_long_prefill_v2 = selection.use_long_prefill_v2;
  const auto prefill_tile_policy = selection.prefill_tile_policy;
  const bool use_typed_prefill =
      prefill_tile_policy !=
      ::sllm_causal_attention_kernel::PrefillTilePolicy::Q4K1Control;
  const bool sliding = metadata.sliding_window != 0U;
  const bool explicitly_scaled = metadata.explicit_score_scale;
  const uint32_t struct_size = info->struct_size;
  const uint32_t abi_version = info->abi_version;
  std::memset(info, 0, sizeof(*info));
  info->struct_size = struct_size;
  info->abi_version = abi_version;
  info->info_version = SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION;
  info->backend = SLLM_BACKEND_HIP;
  info->dispatch_id = dispatch_id;
  // The rocBLAS providers are one logical attention submission, but publish
  // their physical API launches: ID74 F32 uses seven (K/V/query staging, QK,
  // softmax, PV, unpack), while the FP16-score tail uses five (query pack,
  // QK, softmax, PV, unpack). rocBLAS-internal kernels are not counted.
  info->dispatch_count =
      use_gfx1201_gqa6_rocblas_f16_tail ? 5U
      : use_any_gqa6_rocblas_f32        ? 7U
      : (use_decode_gqa4_split || use_decode_gqa4_split_p32 ||
         use_decode_gqa6_split_p128 || use_decode_gqa6_split_p64 ||
         use_decode_gqa6_split_p32 || use_long_prefill_v2)
          ? 2U
          : 1U;
  info->kernel_id =
      sliding ? SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_SLIDING_STATIC_FP8_V1
      : explicitly_scaled
          ? SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_SCALED_STATIC_FP8_V1
      : use_gfx1201_gqa6_rocblas_f16_tail
          ? SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_GQA6_ROCBLAS_F16_TAIL_GFX1201_V1
      : use_gfx1201_gqa6_rocblas_f32
          ? SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_GQA6_ROCBLAS_F32_GFX1201_V1
      : use_decode_gqa6_split_p128
          ? SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_GQA6_SPLIT_P128_GFX1030_V1
      : encoding == SLLM_HIP_KV_ENCODING_FP16_V1
          ? SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_ONLINE_SOFTMAX_V2
          : SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PACKED_KV_V3;
  info->workgroup_size_x =
      use_decode_gqa6_split_p128  ? kDecodeGqa6SplitP128WorkgroupSize
      : use_decode_gqa6_split_p64 ? kDecodeGqa6SplitP64WorkgroupSize
      : use_decode_gqa6_split_p32 ? kDecodeGqa6SplitP32WorkgroupSize
      : (use_decode_gqa4_split || use_decode_gqa4_split_p32) ? 128U
      : use_any_gqa6_rocblas_f32                             ? 256U
                                 : SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE;
  info->grid_size_x = static_cast<uint32_t>(
      use_decode_gqa6_split_p128
          ? kDecodeGqa6SplitP128KvHeads * kDecodeGqa6SplitP128Partitions
      : use_decode_gqa6_split_p64
          ? kDecodeGqa6SplitP64KvHeads * kDecodeGqa6SplitP64Partitions
      : use_decode_gqa6_split_p32
          ? kDecodeGqa6SplitP32KvHeads * kDecodeGqa6SplitP32Partitions
      : use_decode_gqa4_split_p32 ? 4U * kDecodeGqa4SplitP32Partitions
      : use_decode_gqa4_split     ? 4U * kDecodeGqa4SplitPartitions
      : use_decode_wave_split_fp16_pair
          ? metadata.query_count * metadata.query.q_heads
      : use_long_prefill_v2 ? ((metadata.query_count + 7U) / 8U) * kv_heads *
                                  kLongPrefillV2Partitions
      : use_any_gqa6_rocblas_f32
          ? static_cast<uint64_t>(kv_heads) * metadata.query_count * 6U
      : use_scaled_prefill_gemm
          ? ((metadata.query_count + 255U) / 256U) * kv_heads
      : use_typed_prefill
          ? ((metadata.query_count +
              ::sllm_causal_attention_kernel::prefill_query_tile(
                  prefill_tile_policy) -
              1U) /
             ::sllm_causal_attention_kernel::prefill_query_tile(
                 prefill_tile_policy)) *
                kv_heads
      : use_prefill_gqa6_blocksoftmax_q8
          ? (metadata.query_count + 7U) / 8U * kv_heads
      : (use_prefill_gqa4_qtile4 || use_prefill_gqa6_blocksoftmax)
          ? (metadata.query_count + 3U) / 4U * kv_heads
      : use_prefill_gqa4 ? metadata.query_count * kv_heads
                         : metadata.query_count * metadata.query.q_heads);
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
      explicitly_scaled || !integer_implicit_scale ? 0U : implicit_denominator;
  info->fallback_allowed = 0U;
  info->fallback_used = 0U;
  const uint64_t retained_start =
      metadata.expected_kv_length > metadata.sliding_window && sliding
          ? metadata.expected_kv_length - metadata.sliding_window
          : 0U;
  info->reserved[0] = static_cast<uint32_t>(metadata.sliding_window);
  info->reserved[1] = static_cast<uint32_t>(metadata.sliding_window >> 32U);
  info->reserved[2] = static_cast<uint32_t>(retained_start);
  info->reserved[3] = static_cast<uint32_t>(retained_start >> 32U);
  if (explicitly_scaled || !integer_implicit_scale) {
    const float published_scale =
        explicitly_scaled
            ? metadata.score_scale
            : 1.0F / std::sqrt(static_cast<float>(metadata.query.head_dim));
    std::memcpy(&info->reserved[4], &published_scale, sizeof(float));
    info->reserved[5] = 1U;
  }
  sllm_public_runtime::copy_fixed_string(
      info->kernel_symbol, SLLM_HIP_CAUSAL_ATTENTION_KERNEL_SYMBOL_MAX,
      sliding ? ::sllm_causal_attention_kernel::kSlidingStaticFp8LogicalKernelId
      : explicitly_scaled
          ? ::sllm_causal_attention_kernel::kScaledStaticFp8LogicalKernelId
      : use_decode_gqa6_split_p128
          ? ::sllm_causal_attention_kernel::kDecodeGqa6SplitP128LogicalKernelId
      : use_decode_gqa6_split_p64
          ? ::sllm_causal_attention_kernel::kDecodeGqa6SplitP64LogicalKernelId
      : use_decode_gqa6_split_p32
          ? ::sllm_causal_attention_kernel::kDecodeGqa6SplitP32LogicalKernelId
      : use_prefill_gqa6_blocksoftmax_q8
          ? ::sllm_causal_attention_kernel::
                kPrefillGqa6BlockSoftmaxQ8Fp16LogicalKernelId
      : use_prefill_gqa6_blocksoftmax
          ? ::sllm_causal_attention_kernel::
                kPrefillGqa6BlockSoftmaxFp16LogicalKernelId
      : use_decode_gqa4_split_p32
          ? ::sllm_causal_attention_kernel::kDecodeGqa4SplitP32LogicalKernelId
      : use_decode_gqa4_split
          ? ::sllm_causal_attention_kernel::kDecodeGqa4SplitLogicalKernelId
      : use_decode_wave_split_fp16_pair
          ? ::sllm_causal_attention_kernel::
                kDecodeWaveSplitFp16PairLogicalKernelId
      : use_decode_wave_split
          ? (use_decode_wave_split_q_preload
                 ? ::sllm_causal_attention_kernel::
                       kDecodeWaveSplitQPreloadLogicalKernelId
                 : ::sllm_causal_attention_kernel::
                       kDecodeWaveSplitLogicalKernelId)
      : use_long_prefill_v2
          ? ::sllm_causal_attention_kernel::kLongPrefillV2LogicalKernelId
      : use_gfx1201_gqa6_rocblas_f16_tail
          ? ::sllm_causal_attention_kernel::
                kGfx1201Gqa6RocblasF16TailLogicalKernelId
      : use_gfx1201_gqa6_rocblas_f32 ? ::sllm_causal_attention_kernel::
                                           kGfx1201Gqa6RocblasF32LogicalKernelId
      : use_gqa6_rocblas_f32
          ? ::sllm_causal_attention_kernel::kGqa6RocblasF32LogicalKernelId
      : use_scaled_prefill_gemm
          ? ::sllm_causal_attention_kernel::kScaledPrefillGemmLogicalKernelId
      : prefill_tile_policy ==
              ::sllm_causal_attention_kernel::PrefillTilePolicy::Q4K4
          ? ::sllm_causal_attention_kernel::kPrefillTypedQ4K4LogicalKernelId
      : prefill_tile_policy ==
              ::sllm_causal_attention_kernel::PrefillTilePolicy::Q4K8
          ? ::sllm_causal_attention_kernel::kPrefillTypedQ4K8LogicalKernelId
      : prefill_tile_policy ==
              ::sllm_causal_attention_kernel::PrefillTilePolicy::Q8K8
          ? ::sllm_causal_attention_kernel::kPrefillTypedQ8K8LogicalKernelId
      : use_prefill_gqa6_qtile4_k4_fp16
          ? ::sllm_causal_attention_kernel::
                kPrefillGqa6QTile4K4Fp16LogicalKernelId
      : use_prefill_gqa6_qtile4_k8_fp16
          ? ::sllm_causal_attention_kernel::
                kPrefillGqa6QTile4K8Fp16LogicalKernelId
      : use_prefill_gqa6_qtile4_k16_fp16
          ? ::sllm_causal_attention_kernel::
                kPrefillGqa6QTile4K16Fp16LogicalKernelId
      : use_prefill_gqa6_qtile4_k32_fp16
          ? ::sllm_causal_attention_kernel::
                kPrefillGqa6QTile4K32Fp16LogicalKernelId
      : use_prefill_gqa4_qtile4 ? (metadata.query.q_heads / kv_heads == 6U
                                       ? ::sllm_causal_attention_kernel::
                                             kPrefillGqa6QTile4LogicalKernelId
                                       : ::sllm_causal_attention_kernel::
                                             kPrefillGqa4QTile4LogicalKernelId)
      : use_prefill_gqa4
          ? ::sllm_causal_attention_kernel::kPrefillGqa4LogicalKernelId
      : use_gfx1201_wave_provider
          ? (encoding == SLLM_HIP_KV_ENCODING_FP16_V1
                 ? ::sllm_causal_attention_kernel::kGfx1201WaveLogicalKernelId
                 : ::sllm_causal_attention_kernel::
                       kGfx1201WavePackedKvLogicalKernelId)
          : (encoding == SLLM_HIP_KV_ENCODING_FP16_V1
                 ? ::sllm_causal_attention_kernel::kLogicalKernelId
                 : ::sllm_causal_attention_kernel::kPackedKvLogicalKernelId));
  sllm_public_runtime::copy_fixed_string(
      info->device_symbol, SLLM_HIP_CAUSAL_ATTENTION_DEVICE_SYMBOL_MAX,
      sliding ? ::sllm_causal_attention_kernel::kSlidingStaticFp8DeviceSymbol
      : explicitly_scaled
          ? ::sllm_causal_attention_kernel::kScaledStaticFp8DeviceSymbol
      : use_decode_gqa6_split_p128
          ? ::sllm_causal_attention_kernel::kDecodeGqa6SplitP128DeviceSymbol
      : use_decode_gqa6_split_p64
          ? ::sllm_causal_attention_kernel::kDecodeGqa6SplitP64DeviceSymbol
      : use_decode_gqa6_split_p32
          ? ::sllm_causal_attention_kernel::kDecodeGqa6SplitP32DeviceSymbol
      : use_prefill_gqa6_blocksoftmax_q8
          ? ::sllm_causal_attention_kernel::
                kPrefillGqa6BlockSoftmaxQ8Fp16DeviceSymbol
      : use_prefill_gqa6_blocksoftmax
          ? ::sllm_causal_attention_kernel::
                kPrefillGqa6BlockSoftmaxFp16DeviceSymbol
      : use_decode_gqa4_split_p32
          ? ::sllm_causal_attention_kernel::kDecodeGqa4SplitP32DeviceSymbol
      : use_decode_gqa4_split
          ? ::sllm_causal_attention_kernel::kDecodeGqa4SplitDeviceSymbol
      : use_decode_wave_split_fp16_pair
          ? ::sllm_causal_attention_kernel::kDecodeWaveSplitFp16PairDeviceSymbol
      : use_decode_wave_split
          ? (use_decode_wave_split_q_preload
                 ? ::sllm_causal_attention_kernel::
                       kDecodeWaveSplitQPreloadDeviceSymbol
                 : ::sllm_causal_attention_kernel::kDecodeWaveSplitDeviceSymbol)
      : use_long_prefill_v2
          ? ::sllm_causal_attention_kernel::kLongPrefillV2DeviceSymbol
      : use_gfx1201_gqa6_rocblas_f16_tail
          ? ::sllm_causal_attention_kernel::
                kGfx1201Gqa6RocblasF16TailDeviceSymbol
      : use_gfx1201_gqa6_rocblas_f32
          ? ::sllm_causal_attention_kernel::kGfx1201Gqa6RocblasF32DeviceSymbol
      : use_gqa6_rocblas_f32
          ? ::sllm_causal_attention_kernel::kGqa6RocblasF32DeviceSymbol
      : use_scaled_prefill_gemm
          ? ::sllm_causal_attention_kernel::kScaledPrefillGemmDeviceSymbol
      : prefill_tile_policy ==
              ::sllm_causal_attention_kernel::PrefillTilePolicy::Q4K4
          ? ::sllm_causal_attention_kernel::kPrefillTypedQ4K4DeviceSymbol
      : prefill_tile_policy ==
              ::sllm_causal_attention_kernel::PrefillTilePolicy::Q4K8
          ? ::sllm_causal_attention_kernel::kPrefillTypedQ4K8DeviceSymbol
      : prefill_tile_policy ==
              ::sllm_causal_attention_kernel::PrefillTilePolicy::Q8K8
          ? ::sllm_causal_attention_kernel::kPrefillTypedQ8K8DeviceSymbol
      : use_prefill_gqa6_qtile4_k4_fp16
          ? ::sllm_causal_attention_kernel::kPrefillGqa6QTile4K4Fp16DeviceSymbol
      : use_prefill_gqa6_qtile4_k8_fp16
          ? ::sllm_causal_attention_kernel::kPrefillGqa6QTile4K8Fp16DeviceSymbol
      : use_prefill_gqa6_qtile4_k16_fp16
          ? ::sllm_causal_attention_kernel::
                kPrefillGqa6QTile4K16Fp16DeviceSymbol
      : use_prefill_gqa6_qtile4_k32_fp16
          ? ::sllm_causal_attention_kernel::
                kPrefillGqa6QTile4K32Fp16DeviceSymbol
      : use_prefill_gqa4_qtile4 ? (metadata.query.q_heads / kv_heads == 6U
                                       ? ::sllm_causal_attention_kernel::
                                             kPrefillGqa6QTile4DeviceSymbol
                                       : ::sllm_causal_attention_kernel::
                                             kPrefillGqa4QTile4DeviceSymbol)
      : use_prefill_gqa4
          ? ::sllm_causal_attention_kernel::kPrefillGqa4DeviceSymbol
      : use_gfx1201_wave_provider
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

#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
// Host-only selector introspection for target-scope contract tests.  The
// production selector remains private; this bit layout is test ABI only.
// Bits: gfx1201-wave, decode-wave-split, decode-fp16-pair, gqa4-p16,
// gqa4-p32, gqa6-p64, gqa6-p32, gqa6-blocksoftmax (bit 18),
// gqa6-blocksoftmax-q8 (bit 19), gfx1030 gqa6-rocblas-f32 (bit 20),
// gfx1201 gqa6-rocblas-f32 (bit 21), decode-q-preload,
// prefill-gqa4, prefill-gqa4-qtile4,
// scaled-prefill-gemm, long-prefill-v2, and gqa6-qtile4-k32-fp16 (bit 12).
// Bit 22 is the gfx1030-only GQA6 P128 decode provider.
// Bits 13..15 contain the mutually-exclusive GQA6 FP16 K4/K8/K16 candidates.
// Bits 10..11 contain the typed prefill tile policy (0=q4k1 control,
// 1=q4k4, 2=q4k8, 3=q8k8).
extern "C" uint32_t sllm_test_select_causal_attention_providers_with_semantics(
    const uint64_t expected_kv_length, const uint32_t query_count,
    const uint32_t query_heads, const uint32_t kv_heads,
    const uint32_t head_dim, const uint32_t encoding,
    const uint64_t sliding_window, const uint32_t explicit_score_scale,
    const char *const arch_name) noexcept {
  if (arch_name == nullptr) {
    return 0U;
  }
  sllm_causal_attention::DescriptorMetadata metadata{};
  metadata.query_count = query_count;
  metadata.expected_kv_length = expected_kv_length;
  metadata.query.query_count = query_count;
  metadata.query.q_heads = query_heads;
  metadata.query.head_dim = head_dim;
  metadata.sliding_window = sliding_window;
  metadata.explicit_score_scale = explicit_score_scale != 0U;
  const CausalAttentionProviderSelection selection =
      select_causal_attention_provider(metadata, kv_heads, encoding, arch_name);
  uint32_t mask = 0U;
  mask |= selection.use_gfx1201_wave_provider ? (1U << 0U) : 0U;
  mask |= selection.use_decode_wave_split ? (1U << 1U) : 0U;
  mask |= selection.use_decode_wave_split_fp16_pair ? (1U << 2U) : 0U;
  mask |= selection.use_decode_gqa4_split ? (1U << 3U) : 0U;
  mask |= selection.use_decode_gqa4_split_p32 ? (1U << 4U) : 0U;
  mask |= selection.use_decode_gqa6_split_p64 ? (1U << 17U) : 0U;
  mask |= selection.use_prefill_gqa6_blocksoftmax ? (1U << 18U) : 0U;
  mask |= selection.use_prefill_gqa6_blocksoftmax_q8 ? (1U << 19U) : 0U;
  mask |= selection.use_decode_gqa6_split_p32 ? (1U << 16U) : 0U;
  mask |= selection.use_decode_wave_split_q_preload ? (1U << 5U) : 0U;
  mask |= selection.use_prefill_gqa4 ? (1U << 6U) : 0U;
  mask |= selection.use_prefill_gqa4_qtile4 &&
                  !selection.use_prefill_gqa6_qtile4_k4_fp16 &&
                  !selection.use_prefill_gqa6_qtile4_k8_fp16 &&
                  !selection.use_prefill_gqa6_qtile4_k16_fp16 &&
                  !selection.use_prefill_gqa6_qtile4_k32_fp16
              ? (1U << 7U)
              : 0U;
  mask |= selection.use_prefill_gqa6_qtile4_k32_fp16 ? (1U << 12U) : 0U;
  mask |= selection.use_prefill_gqa6_qtile4_k4_fp16 ? (1U << 13U) : 0U;
  mask |= selection.use_prefill_gqa6_qtile4_k8_fp16 ? (1U << 14U) : 0U;
  mask |= selection.use_prefill_gqa6_qtile4_k16_fp16 ? (1U << 15U) : 0U;
  mask |= selection.use_scaled_prefill_gemm ? (1U << 8U) : 0U;
  mask |= selection.use_long_prefill_v2 ? (1U << 9U) : 0U;
  mask |= selection.use_gqa6_rocblas_f32 ? (1U << 20U) : 0U;
  mask |= selection.use_gfx1201_gqa6_rocblas_f32 ? (1U << 21U) : 0U;
  mask |= selection.use_decode_gqa6_split_p128 ? (1U << 22U) : 0U;
  mask |= static_cast<uint32_t>(selection.prefill_tile_policy) << 10U;
  return mask;
}

extern "C" uint32_t sllm_test_select_causal_attention_providers(
    const uint64_t expected_kv_length, const uint32_t query_count,
    const uint32_t query_heads, const uint32_t kv_heads,
    const uint32_t head_dim, const uint32_t encoding,
    const char *const arch_name) noexcept {
  return sllm_test_select_causal_attention_providers_with_semantics(
      expected_kv_length, query_count, query_heads, kv_heads, head_dim,
      encoding, 0U, 0U, arch_name);
}

extern "C" uint32_t sllm_test_select_causal_attention_gqa4(
    const uint64_t expected_kv_length, const uint32_t query_count,
    const uint32_t query_heads, const uint32_t kv_heads,
    const uint32_t head_dim, const uint32_t encoding,
    const char *const arch_name) noexcept {
  if (arch_name == nullptr) {
    return 0U;
  }
  sllm_causal_attention::DescriptorMetadata metadata{};
  metadata.query_count = query_count;
  metadata.expected_kv_length = expected_kv_length;
  metadata.query.query_count = query_count;
  metadata.query.q_heads = query_heads;
  metadata.query.head_dim = head_dim;
  const CausalAttentionProviderSelection selection =
      select_causal_attention_provider(metadata, kv_heads, encoding, arch_name);
  return (selection.use_decode_gqa4_split ? 1U : 0U) |
         (selection.use_decode_gqa4_split_p32 ? 2U : 0U);
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
      const CausalAttentionBuffers resolved = {query, output};
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (context->poisoned.load() || context->release_active ||
          queue->release_active || state->release_active ||
          query->release_active || output->release_active ||
          (!chained && state->transition_token != 0U) ||
          (!chained && state->accounting.active_submissions != 0U) ||
          (!chained && state->accounting.completion_references != 0U)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_CAUSAL_ATTENTION_STATE_BUSY,
            "causal attention requires an immutable committed KV snapshot");
      }
      const bool range_valid =
          !chained
              ? metadata.expected_kv_length == state->published_length &&
                    metadata.expected_kv_length <= state->capacity_tokens &&
                    metadata.start_position + metadata.query_count ==
                        state->published_length
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
    const CausalAttentionProviderSelection selection =
        select_causal_attention_provider(metadata, state->head_count,
                                         state->encoding, arch_name);
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
      execute_guard.candidate_allocated();
    } catch (...) {
      execute_guard.disarm();
      (void)rollback_reserved_causal_attention(state, queue, {query, output},
                                               error_sink);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "causal attention completion allocation failed before enqueue");
    }
    if (selection.use_decode_gqa4_split ||
        selection.use_decode_gqa4_split_p32 ||
        selection.use_decode_gqa6_split_p128 ||
        selection.use_decode_gqa6_split_p64 ||
        selection.use_decode_gqa6_split_p32) {
      const uint64_t workspace_bytes = selection.use_decode_gqa6_split_p128
                                           ? kDecodeGqa6SplitP128WorkspaceBytes
                                       : selection.use_decode_gqa6_split_p64
                                           ? kDecodeGqa6SplitP64WorkspaceBytes
                                       : selection.use_decode_gqa6_split_p32
                                           ? kDecodeGqa6SplitP32WorkspaceBytes
                                       : selection.use_decode_gqa4_split_p32
                                           ? kDecodeGqa4SplitP32WorkspaceBytes
                                           : kDecodeGqa4SplitWorkspaceBytes;
      const hipError_t plan_status = prepare_causal_workspace_plan_for_bytes(
          context, queue, workspace_bytes, &scaled_prefill_workspace,
          &scaled_prefill_workspace_bytes);
      if (plan_status != hipSuccess) {
        execute_guard.disarm();
        (void)rollback_reserved_causal_attention(state, queue, {query, output},
                                                 error_sink);
        return hip_failure(error_sink, plan_status,
                           "causal attention GQA split workspace prepare");
      }
      candidate->causal_scaled_prefill_plan_acquired = true;
    } else if (selection.use_long_prefill_v2) {
      const hipError_t plan_status = prepare_causal_workspace_plan_for_bytes(
          context, queue, kLongPrefillV2WorkspaceBytes,
          &scaled_prefill_workspace, &scaled_prefill_workspace_bytes);
      if (plan_status != hipSuccess) {
        execute_guard.disarm();
        (void)rollback_reserved_causal_attention(state, queue, {query, output},
                                                 error_sink);
        return hip_failure(
            error_sink, plan_status,
            "causal attention long-prefill v2 workspace prepare");
      }
      candidate->causal_scaled_prefill_plan_acquired = true;
    } else if (selection.use_gqa6_rocblas_f32 ||
               selection.use_gfx1201_gqa6_rocblas_f32) {
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
      const uint64_t workspace_bytes = 0U;
#else
      const hipError_t handle_status = ensure_gqa6_rocblas_handle(context);
      if (handle_status != hipSuccess) {
        execute_guard.disarm();
        (void)rollback_reserved_causal_attention(state, queue, {query, output},
                                                 error_sink);
        return hip_failure(error_sink, handle_status,
                           "causal attention GQA6 rocBLAS handle prepare");
      }
      const uint64_t workspace_bytes =
          selection.use_gfx1201_gqa6_rocblas_f16_tail
              ? ::sllm_causal_attention_kernel::
                    gqa6_rocblas_f16_tail_workspace_bytes(
                        static_cast<uint32_t>(metadata.query_count),
                        metadata.expected_kv_length)
              : ::sllm_causal_attention_kernel::
                    gqa6_rocblas_f32_workspace_bytes(
                        static_cast<uint32_t>(metadata.query_count),
                        metadata.expected_kv_length);
#endif
      if (workspace_bytes == 0U) {
        execute_guard.disarm();
        (void)rollback_reserved_causal_attention(state, queue, {query, output},
                                                 error_sink);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INVALID_ARGUMENT,
            "causal attention GQA6 rocBLAS workspace shape is invalid");
      }
      const hipError_t plan_status = prepare_causal_workspace_plan_for_bytes(
          context, queue, workspace_bytes, &scaled_prefill_workspace,
          &scaled_prefill_workspace_bytes);
      if (plan_status != hipSuccess) {
        execute_guard.disarm();
        (void)rollback_reserved_causal_attention(state, queue, {query, output},
                                                 error_sink);
        return hip_failure(error_sink, plan_status,
                           "causal attention GQA6 rocBLAS workspace prepare");
      }
      candidate->causal_scaled_prefill_plan_acquired = true;
    } else if (selection.use_scaled_prefill_gemm) {
      const hipError_t plan_status = prepare_causal_scaled_prefill_plan(
          context, queue, metadata.expected_kv_length,
          &scaled_prefill_workspace, &scaled_prefill_workspace_bytes);
      if (plan_status != hipSuccess) {
        execute_guard.disarm();
        (void)rollback_reserved_causal_attention(state, queue, {query, output},
                                                 error_sink);
        return hip_failure(error_sink, plan_status,
                           "causal attention scaled-prefill workspace prepare");
      }
      candidate->causal_scaled_prefill_plan_acquired = true;
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
    const uint16_t *const query_pointer = static_cast<const uint16_t *>(
        byte_pointer(query, metadata.query.byte_offset));
    uint16_t *const output_pointer = static_cast<uint16_t *>(
        byte_pointer(output, metadata.output.byte_offset));
    hipError_t launch_status = hipSuccess;
    if (selection.use_decode_gqa6_split_p128) {
      launch_status =
          ::sllm_causal_attention_kernel::launch_decode_gqa6_split_p128(
              query_pointer, state->key_buffer->device_pointer,
              state->value_buffer->device_pointer,
              state->key_scale_plane.address, state->value_scale_plane.address,
              static_cast<const float *>(state->key_outer_scale_plane.address),
              static_cast<const float *>(
                  state->value_outer_scale_plane.address),
              output_pointer, static_cast<uint32_t>(metadata.query_count),
              metadata.start_position, metadata.expected_kv_length,
              metadata.query.q_heads, state->head_count, state->head_dim,
              state->encoding, state->static_key_scale,
              state->static_value_scale, scaled_prefill_workspace,
              scaled_prefill_workspace_bytes, queue->stream);
    } else if (selection.use_decode_gqa6_split_p64) {
      launch_status =
          ::sllm_causal_attention_kernel::launch_decode_gqa6_split_p64(
              query_pointer, state->key_buffer->device_pointer,
              state->value_buffer->device_pointer,
              state->key_scale_plane.address, state->value_scale_plane.address,
              static_cast<const float *>(state->key_outer_scale_plane.address),
              static_cast<const float *>(
                  state->value_outer_scale_plane.address),
              output_pointer, static_cast<uint32_t>(metadata.query_count),
              metadata.start_position, metadata.expected_kv_length,
              metadata.query.q_heads, state->head_count, state->head_dim,
              state->encoding, state->static_key_scale,
              state->static_value_scale, scaled_prefill_workspace,
              scaled_prefill_workspace_bytes, queue->stream);
    } else if (selection.use_decode_gqa6_split_p32) {
      launch_status =
          ::sllm_causal_attention_kernel::launch_decode_gqa6_split_p32(
              query_pointer, state->key_buffer->device_pointer,
              state->value_buffer->device_pointer,
              state->key_scale_plane.address, state->value_scale_plane.address,
              static_cast<const float *>(state->key_outer_scale_plane.address),
              static_cast<const float *>(
                  state->value_outer_scale_plane.address),
              output_pointer, static_cast<uint32_t>(metadata.query_count),
              metadata.start_position, metadata.expected_kv_length,
              metadata.query.q_heads, state->head_count, state->head_dim,
              state->encoding, state->static_key_scale,
              state->static_value_scale, scaled_prefill_workspace,
              scaled_prefill_workspace_bytes, queue->stream);
    } else if (selection.use_decode_gqa4_split_p32) {
      launch_status =
          ::sllm_causal_attention_kernel::launch_decode_gqa4_split_p32(
              query_pointer, state->key_buffer->device_pointer,
              state->value_buffer->device_pointer,
              state->key_scale_plane.address, state->value_scale_plane.address,
              static_cast<const float *>(state->key_outer_scale_plane.address),
              static_cast<const float *>(
                  state->value_outer_scale_plane.address),
              output_pointer, static_cast<uint32_t>(metadata.query_count),
              metadata.start_position, metadata.expected_kv_length,
              metadata.query.q_heads, state->head_count, state->head_dim,
              state->encoding, state->static_key_scale,
              state->static_value_scale, scaled_prefill_workspace,
              scaled_prefill_workspace_bytes, queue->stream);
    } else if (selection.use_decode_gqa4_split) {
      launch_status = ::sllm_causal_attention_kernel::launch_decode_gqa4_split(
          query_pointer, state->key_buffer->device_pointer,
          state->value_buffer->device_pointer, state->key_scale_plane.address,
          state->value_scale_plane.address,
          static_cast<const float *>(state->key_outer_scale_plane.address),
          static_cast<const float *>(state->value_outer_scale_plane.address),
          output_pointer, static_cast<uint32_t>(metadata.query_count),
          metadata.start_position, metadata.expected_kv_length,
          metadata.query.q_heads, state->head_count, state->head_dim,
          state->encoding, state->static_key_scale, state->static_value_scale,
          scaled_prefill_workspace, scaled_prefill_workspace_bytes,
          queue->stream);
    } else if (selection.use_long_prefill_v2) {
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
      launch_status = hipErrorNotSupported;
#else
      launch_status = ::sllm_causal_attention_kernel::launch_long_prefill_v2(
          query_pointer, state->key_buffer->device_pointer,
          state->value_buffer->device_pointer, output_pointer,
          static_cast<uint32_t>(metadata.query_count), metadata.start_position,
          metadata.expected_kv_length, metadata.query.q_heads,
          state->head_count, state->head_dim, scaled_prefill_workspace,
          scaled_prefill_workspace_bytes, queue->stream);
#endif
    } else if (selection.use_gqa6_rocblas_f32 ||
               selection.use_gfx1201_gqa6_rocblas_f32) {
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
      launch_status = hipErrorNotSupported;
#else
      launch_status =
          selection.use_gfx1201_gqa6_rocblas_f16_tail
              ? ::sllm_causal_attention_kernel::launch_gqa6_rocblas_f16_tail(
                    query_pointer, state->key_buffer->device_pointer,
                    state->value_buffer->device_pointer, output_pointer,
                    static_cast<uint32_t>(metadata.query_count),
                    metadata.start_position, metadata.expected_kv_length,
                    metadata.query.q_heads, state->head_count, state->head_dim,
                    state->encoding, scaled_prefill_workspace,
                    scaled_prefill_workspace_bytes,
                    context->matmul_rocblas_handle, &context->matmul_blas_mutex,
                    queue->stream)
              : ::sllm_causal_attention_kernel::launch_gqa6_rocblas_f32(
                    query_pointer, state->key_buffer->device_pointer,
                    state->value_buffer->device_pointer, output_pointer,
                    static_cast<uint32_t>(metadata.query_count),
                    metadata.start_position, metadata.expected_kv_length,
                    metadata.query.q_heads, state->head_count, state->head_dim,
                    state->encoding, scaled_prefill_workspace,
                    scaled_prefill_workspace_bytes,
                    context->matmul_rocblas_handle, &context->matmul_blas_mutex,
                    queue->stream);
#endif
    } else if (selection.use_scaled_prefill_gemm) {
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
      launch_status = hipErrorNotSupported;
#else
      launch_status =
          ::sllm_causal_attention_kernel::launch_scaled_prefill_gemm(
              query_pointer, state->key_buffer->device_pointer,
              state->value_buffer->device_pointer,
              state->key_scale_plane.address, state->value_scale_plane.address,
              static_cast<const float *>(state->key_outer_scale_plane.address),
              static_cast<const float *>(
                  state->value_outer_scale_plane.address),
              output_pointer, static_cast<uint32_t>(metadata.query_count),
              metadata.start_position, metadata.expected_kv_length,
              metadata.query.q_heads, state->head_count, state->head_dim,
              state->encoding, state->static_key_scale,
              state->static_value_scale, scaled_prefill_workspace,
              scaled_prefill_workspace_bytes, context->matmul_blas_handle,
              &context->matmul_blas_mutex, queue->stream);
#endif
    } else if (selection.use_prefill_gqa6_blocksoftmax_q8) {
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
      launch_status = hipErrorNotSupported;
#else
      launch_status =
          ::sllm_causal_attention_kernel::launch_gqa6_blocksoftmax_q8_fp16(
              query_pointer, state->key_buffer->device_pointer,
              state->value_buffer->device_pointer,
              state->key_scale_plane.address, state->value_scale_plane.address,
              static_cast<const float *>(state->key_outer_scale_plane.address),
              static_cast<const float *>(
                  state->value_outer_scale_plane.address),
              output_pointer, static_cast<uint32_t>(metadata.query_count),
              metadata.start_position, metadata.query.q_heads,
              state->head_count, state->head_dim, state->static_key_scale,
              state->static_value_scale, queue->stream);
#endif
    } else if (selection.use_prefill_gqa6_blocksoftmax) {
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
      launch_status = hipErrorNotSupported;
#else
      const uint32_t key_tile =
          std::strcmp(arch_name, "gfx1030") == 0 ? 32U : 8U;
      launch_status =
          ::sllm_causal_attention_kernel::launch_gqa6_blocksoftmax_fp16(
              query_pointer, state->key_buffer->device_pointer,
              state->value_buffer->device_pointer,
              state->key_scale_plane.address, state->value_scale_plane.address,
              static_cast<const float *>(state->key_outer_scale_plane.address),
              static_cast<const float *>(
                  state->value_outer_scale_plane.address),
              output_pointer, static_cast<uint32_t>(metadata.query_count),
              metadata.start_position, metadata.query.q_heads,
              state->head_count, state->head_dim, state->static_key_scale,
              state->static_value_scale, key_tile, queue->stream);
#endif
    } else if (selection.prefill_tile_policy !=
               ::sllm_causal_attention_kernel::PrefillTilePolicy::Q4K1Control) {
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
      launch_status = hipErrorNotSupported;
#else
      launch_status = ::sllm_causal_attention_kernel::launch_typed_prefill(
          query_pointer, state->key_buffer->device_pointer,
          state->value_buffer->device_pointer, state->key_scale_plane.address,
          state->value_scale_plane.address,
          static_cast<const float *>(state->key_outer_scale_plane.address),
          static_cast<const float *>(state->value_outer_scale_plane.address),
          output_pointer, static_cast<uint32_t>(metadata.query_count),
          metadata.start_position, metadata.query.q_heads, state->head_count,
          state->head_dim, state->encoding, state->static_key_scale,
          state->static_value_scale, selection.prefill_tile_policy,
          queue->stream);
#endif
    } else if (selection.use_decode_wave_split_fp16_pair) {
      launch_status =
          ::sllm_causal_attention_kernel::launch_decode_wave_split_fp16_pair(
              query_pointer, state->key_buffer->device_pointer,
              state->value_buffer->device_pointer,
              state->key_scale_plane.address, state->value_scale_plane.address,
              static_cast<const float *>(state->key_outer_scale_plane.address),
              static_cast<const float *>(
                  state->value_outer_scale_plane.address),
              output_pointer, static_cast<uint32_t>(metadata.query_count),
              metadata.start_position, metadata.expected_kv_length,
              metadata.query.q_heads, state->head_count, state->head_dim,
              state->encoding, state->static_key_scale,
              state->static_value_scale, queue->stream);
    } else if (selection.use_prefill_gqa6_qtile4_k4_fp16 ||
               selection.use_prefill_gqa6_qtile4_k8_fp16 ||
               selection.use_prefill_gqa6_qtile4_k16_fp16 ||
               selection.use_prefill_gqa6_qtile4_k32_fp16) {
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
      launch_status = hipErrorNotSupported;
#else
      const uint32_t key_tile = selection.use_prefill_gqa6_qtile4_k4_fp16   ? 4U
                                : selection.use_prefill_gqa6_qtile4_k8_fp16 ? 8U
                                : selection.use_prefill_gqa6_qtile4_k16_fp16
                                    ? 16U
                                    : 32U;
      launch_status = ::sllm_causal_attention_kernel::launch_gqa6_qtile4_fp16(
          query_pointer, state->key_buffer->device_pointer,
          state->value_buffer->device_pointer, state->key_scale_plane.address,
          state->value_scale_plane.address,
          static_cast<const float *>(state->key_outer_scale_plane.address),
          static_cast<const float *>(state->value_outer_scale_plane.address),
          output_pointer, static_cast<uint32_t>(metadata.query_count),
          metadata.start_position, metadata.query.q_heads, state->head_count,
          state->head_dim, state->static_key_scale, state->static_value_scale,
          key_tile, queue->stream);
#endif
    } else {
      launch_status = ::sllm_causal_attention_kernel::launch(
          query_pointer, state->key_buffer->device_pointer,
          state->value_buffer->device_pointer, state->key_scale_plane.address,
          state->value_scale_plane.address,
          static_cast<const float *>(state->key_outer_scale_plane.address),
          static_cast<const float *>(state->value_outer_scale_plane.address),
          output_pointer, static_cast<uint32_t>(metadata.query_count),
          state->storage_capacity_tokens, metadata.start_position,
          metadata.expected_kv_length, metadata.query.q_heads,
          state->head_count, state->head_dim, state->encoding,
          state->static_key_scale, state->static_value_scale,
          selection.use_gfx1201_wave_provider, selection.use_decode_wave_split,
          selection.use_decode_wave_split_q_preload, selection.use_prefill_gqa4,
          selection.use_prefill_gqa4_qtile4,
          selection.use_prefill_gqa6_qtile4_k32_fp16, metadata.sliding_window,
          metadata.explicit_score_scale
              ? metadata.score_scale
              : 1.0F / std::sqrt(static_cast<float>(state->head_dim)),
          queue->stream);
    }
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
                                        arch_name, selection);
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
