/* Additive state fork/image ABI implementations for Paged KV and linear
 * attention.  Legacy KV VMM/raw-plane symbols remain ABI-reserved stubs so
 * older checkpoint records can still be identified by the core layer. */

namespace {

hipError_t state_copy(void *const destination, const void *const source,
                      const std::size_t bytes,
                      const hipMemcpyKind direction) noexcept {
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
  if (destination == nullptr || source == nullptr) {
    return hipErrorInvalidValue;
  }
  std::memcpy(destination, source, bytes);
  (void)direction;
  return hipSuccess;
#else
  return hipMemcpy(destination, source, bytes, direction);
#endif
}

bool state_reserved_zero(const uint32_t *const values,
                         const std::size_t count) noexcept {
  return std::all_of(values, values + count,
                     [](const uint32_t value) { return value == 0U; });
}

sllm_status_t validate_chunk(const sllm_state_chunk_t *const chunk,
                             sllm_error_sink_t *const sink) noexcept {
  if (chunk == nullptr || chunk->struct_size != sizeof(*chunk)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "state chunk has an unsupported struct size");
  }
  if (chunk->abi_version != SLLM_HIP_ABI_VERSION ||
      (chunk->info_version != SLLM_HIP_STATE_FORK_INFO_VERSION &&
       chunk->info_version != SLLM_HIP_STATE_IMAGE_SLIDING_VERSION)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "state chunk ABI or version is unsupported");
  }
  const bool full = chunk->info_version == SLLM_HIP_STATE_FORK_INFO_VERSION &&
                    chunk->reserved0 == 0U && chunk->reserved1 == 0U &&
                    state_reserved_zero(chunk->reserved, 4U);
  const uint64_t sliding_window =
      static_cast<uint64_t>(chunk->reserved0) |
      (static_cast<uint64_t>(chunk->reserved1) << UINT32_C(32));
  const bool sliding =
      chunk->info_version == SLLM_HIP_STATE_IMAGE_SLIDING_VERSION &&
      sliding_window == SLLM_HIP_KV_SLIDING_WINDOW_GEMMA4 &&
      chunk->reserved[2] == 0U && chunk->reserved[3] == 0U;
  if (!full && !sliding) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "state chunk reserved fields must be zero");
  }
  if (chunk->host_pointer == nullptr || chunk->byte_length == 0U ||
      chunk->byte_length > SLLM_HIP_STATE_CHUNK_MAX_BYTES ||
      chunk->host_capacity < chunk->byte_length ||
      chunk->byte_offset >
          std::numeric_limits<uint64_t>::max() - chunk->byte_length) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_BUFFER_TOO_SMALL,
        "state chunk host range is outside the bounded contract");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t validate_image_header(const sllm_state_image_info_t *const info,
                                    sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr || info->struct_size != sizeof(*info)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "state image info has an unsupported struct size");
  }
  if (info->abi_version != SLLM_HIP_ABI_VERSION ||
      (info->info_version != SLLM_HIP_STATE_FORK_INFO_VERSION &&
       info->info_version != SLLM_HIP_STATE_IMAGE_SLIDING_VERSION)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "state image info ABI or version is unsupported");
  }
  const bool full = info->info_version == SLLM_HIP_STATE_FORK_INFO_VERSION &&
                    state_reserved_zero(info->reserved, 7U);
  const uint64_t sliding_window =
      static_cast<uint64_t>(info->reserved[0]) |
      (static_cast<uint64_t>(info->reserved[1]) << UINT32_C(32));
  const bool sliding =
      info->info_version == SLLM_HIP_STATE_IMAGE_SLIDING_VERSION &&
      sliding_window == SLLM_HIP_KV_SLIDING_WINDOW_GEMMA4 &&
      state_reserved_zero(info->reserved + 4U, 3U);
  if (info->reserved0 != 0U || (!full && !sliding)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "state image info reserved fields must be zero");
  }
  return SLLM_STATUS_OK;
}

void fill_linear_image_info(const LinearAttentionState *const state,
                            sllm_state_image_info_t *const info) noexcept {
  const uint32_t struct_size = info->struct_size;
  const uint32_t abi_version = info->abi_version;
  std::memset(info, 0, sizeof(*info));
  info->struct_size = struct_size;
  info->abi_version = abi_version;
  info->info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
  info->session_id = state->session_id;
  info->layer_id = state->layer_id;
  info->dtype = SLLM_TENSOR_DTYPE_BF16;
  info->encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  info->active_slot = state->active_slot;
  info->capacity_tokens = state->capacity_tokens;
  info->published_length = state->published_length;
  info->generation = state->generation;
  info->plane_count = state->scratch == nullptr ? 4U : 5U;
}

void fill_fork_info(sllm_state_fork_info_t *const info, const uint32_t mode,
                    const uint64_t source_identity,
                    const uint64_t child_identity, const uint64_t source_owned,
                    const uint64_t child_owned, const uint64_t copied,
                    const uint64_t shared, const uint64_t published_length,
                    const uint64_t page_bytes) noexcept {
  const uint32_t struct_size = info->struct_size;
  const uint32_t abi_version = info->abi_version;
  std::memset(info, 0, sizeof(*info));
  info->struct_size = struct_size;
  info->abi_version = abi_version;
  info->info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
  info->mode = mode;
  info->source_state_identity = source_identity;
  info->child_state_identity = child_identity;
  info->source_owned_bytes = source_owned;
  info->child_owned_bytes = child_owned;
  info->copied_bytes = copied;
  info->shared_bytes = shared;
  info->published_length = published_length;
  info->page_bytes = page_bytes;
}

#if !defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
void fill_paged_fork_info(
    sllm_kv_paged_state_fork_info_t *const info, const uint64_t source_identity,
    const uint64_t child_identity, const uint64_t source_owned,
    const uint64_t child_owned, const uint64_t copied, const uint64_t shared,
    const uint64_t published_length, const uint32_t token_block_size,
    const uint32_t physical_layout_version,
    const uint64_t source_logical_capacity,
    const uint64_t child_logical_capacity,
    const uint64_t source_physical_blocks, const uint64_t child_physical_blocks,
    const uint64_t copied_physical_blocks,
    const uint64_t shared_physical_blocks,
    const uint64_t committed_bytes_total) noexcept {
  const uint32_t struct_size = info->struct_size;
  const uint32_t abi_version = info->abi_version;
  std::memset(info, 0, sizeof(*info));
  info->struct_size = struct_size;
  info->abi_version = abi_version;
  info->info_version = SLLM_HIP_KV_PAGED_STATE_FORK_INFO_VERSION;
  info->source_state_identity = source_identity;
  info->child_state_identity = child_identity;
  info->source_owned_bytes = source_owned;
  info->child_owned_bytes = child_owned;
  info->copied_bytes = copied;
  info->shared_bytes = shared;
  info->published_length = published_length;
  info->token_block_size = token_block_size;
  info->physical_layout_version = physical_layout_version;
  info->source_logical_table_capacity = source_logical_capacity;
  info->child_logical_table_capacity = child_logical_capacity;
  info->source_physical_blocks = source_physical_blocks;
  info->child_physical_blocks = child_physical_blocks;
  info->copied_physical_blocks = copied_physical_blocks;
  info->shared_physical_blocks = shared_physical_blocks;
  info->committed_bytes_total = committed_bytes_total;
}
#endif

struct LinearPlaneRef final {
  void *pointer;
  uint64_t bytes;
};

LinearPlaneRef linear_plane(LinearAttentionState *const state,
                            const uint32_t id) noexcept {
  const uint64_t conv_bytes =
      static_cast<uint64_t>(state->conv_kernel_size - 1U) * state->qkv_width *
      sizeof(uint16_t);
  const uint64_t recurrent_bytes = static_cast<uint64_t>(state->value_heads) *
                                   state->head_dim * state->head_dim *
                                   sizeof(float);
  switch (id) {
  case SLLM_HIP_LINEAR_STATE_PLANE_CONV_SLOT0:
    return {state->conv_state[0], conv_bytes};
  case SLLM_HIP_LINEAR_STATE_PLANE_CONV_SLOT1:
    return {state->conv_state[1], conv_bytes};
  case SLLM_HIP_LINEAR_STATE_PLANE_RECURRENT_SLOT0:
    return {state->recurrent_state[0], recurrent_bytes};
  case SLLM_HIP_LINEAR_STATE_PLANE_RECURRENT_SLOT1:
    return {state->recurrent_state[1], recurrent_bytes};
  case SLLM_HIP_LINEAR_STATE_PLANE_SCRATCH:
    return {state->scratch, state->scratch_bytes};
  default:
    return {nullptr, 0U};
  }
}

[[maybe_unused]] bool kv_quiescent(const KvState *const state) noexcept {
  return state != nullptr && !state->release_active &&
         state->transition_token == 0U && state->view_count == 0U &&
         state->accounting.active_submissions == 0U &&
         state->accounting.completion_references == 0U;
}

bool linear_quiescent(const LinearAttentionState *const state) noexcept {
  return state != nullptr && !state->release_active &&
         state->transition_token == 0U &&
         state->accounting.active_submissions == 0U &&
         state->accounting.completion_references == 0U;
}

} // namespace

extern "C" sllm_status_t
sllm_kv_state_fork(const sllm_kv_state_t *const raw_source,
                   const sllm_kv_state_create_info_v2_t *const destination_info,
                   sllm_kv_state_t **const raw_child,
                   sllm_state_fork_info_t *const fork_info,
                   sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (raw_child != nullptr) {
      *raw_child = nullptr;
    }
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t destination_status =
        sllm_kv_state::validate_state_create_info_v2(destination_info,
                                                     error_sink);
    if (destination_status != SLLM_STATUS_OK) {
      return destination_status;
    }
    if (fork_info == nullptr || fork_info->struct_size != sizeof(*fork_info)) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "KV fork info struct size is unsupported");
    }
    if (fork_info->abi_version != SLLM_HIP_ABI_VERSION) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ABI_VERSION,
          "KV fork info ABI is unsupported");
    }
    if (fork_info->info_version != SLLM_HIP_STATE_FORK_INFO_VERSION) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "KV fork info version is unsupported");
    }
    if (!state_reserved_zero(fork_info->reserved, 4U)) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_RESERVED_NONZERO,
          "KV fork info reserved fields must be zero");
    }
    if (raw_source == nullptr || raw_child == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "KV fork source or child output is null");
    }
    fill_fork_info(fork_info, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U);
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_UNSUPPORTED,
        "legacy KV VMM fork ABI is retired; use paged KV fork");
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in retired KV state fork stub");
  }
}

extern "C" sllm_status_t
sllm_kv_state_fork_query(const sllm_kv_state_t *const raw_state,
                         sllm_state_fork_info_t *const fork_info,
                         sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (fork_info == nullptr || fork_info->struct_size != sizeof(*fork_info)) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "KV fork info struct size is unsupported");
    }
    if (fork_info->abi_version != SLLM_HIP_ABI_VERSION) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ABI_VERSION,
          "KV fork info ABI is unsupported");
    }
    if (fork_info->info_version != SLLM_HIP_STATE_FORK_INFO_VERSION) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "KV fork info version is unsupported");
    }
    if (!state_reserved_zero(fork_info->reserved, 4U)) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_RESERVED_NONZERO,
          "KV fork info reserved fields must be zero");
    }
    if (raw_state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "KV fork query state handle is null");
    }
    fill_fork_info(fork_info, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U);
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_UNSUPPORTED,
        "legacy KV VMM fork query ABI is retired; use paged KV fork query");
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in retired KV fork query stub");
  }
}

namespace {

sllm_status_t validate_paged_fork_info_header(
    const sllm_kv_paged_state_fork_info_t *const info,
    sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr || info->struct_size != sizeof(*info)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "paged KV fork info has an unsupported struct size");
  }
  if (info->abi_version != SLLM_HIP_ABI_VERSION ||
      info->info_version != SLLM_HIP_KV_PAGED_STATE_FORK_INFO_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "paged KV fork info ABI or version is unsupported");
  }
  if (info->reserved0 != 0U || !state_reserved_zero(info->reserved, 4U)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "paged KV fork info reserved fields must be zero");
  }
  return SLLM_STATUS_OK;
}

#if !defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)

bool paged_collect_blocks(const sllm_paged_kv::State *const state,
                          std::vector<uint32_t> &blocks) {
  if (state == nullptr || state->pending()) {
    return false;
  }
  const uint64_t length = state->published_length();
  const uint64_t used =
      length / SLLM_HIP_KV_PAGED_TOKEN_BLOCK_SIZE +
      (length % SLLM_HIP_KV_PAGED_TOKEN_BLOCK_SIZE != 0U ? 1U : 0U);
  if (used > state->table_capacity()) {
    return false;
  }
  blocks.clear();
  blocks.reserve(static_cast<std::size_t>(used));
  for (uint64_t logical = 0U; logical < used; ++logical) {
    const uint32_t physical =
        state->physical_block(static_cast<uint32_t>(logical));
    if (physical == sllm_paged_kv::kInvalidBlock) {
      return false;
    }
    blocks.push_back(physical);
  }
  return true;
}

bool paged_collect_sliding_blocks(
    const sllm_paged_kv::SlidingState *const state,
    std::vector<uint32_t> &blocks, std::array<uint64_t, 9U> &tags) {
  if (state == nullptr || state->pending())
    return false;
  blocks.clear();
  tags.fill(std::numeric_limits<uint64_t>::max());
  for (uint32_t slot = 0U; slot < sllm_paged_kv::kSlidingRingSlots; ++slot) {
    tags[slot] = state->absolute_block_tag(slot);
    if (tags[slot] == std::numeric_limits<uint64_t>::max())
      continue;
    const uint32_t physical = state->physical_block_at_slot(slot);
    if (physical == sllm_paged_kv::kInvalidBlock)
      return false;
    blocks.push_back(physical);
  }
  return true;
}

uint64_t paged_block_bytes(const sllm_paged_kv::DevicePool &pool) noexcept {
  uint64_t total = 0U;
  for (const uint64_t bytes : pool.plan().plane_bytes_per_block()) {
    if (bytes > std::numeric_limits<uint64_t>::max() - total) {
      return std::numeric_limits<uint64_t>::max();
    }
    total += bytes;
  }
  return total;
}

bool paged_mul_bytes(const uint64_t blocks, const uint64_t block_bytes,
                     uint64_t *const result) noexcept {
  if (result == nullptr ||
      (blocks != 0U &&
       block_bytes > std::numeric_limits<uint64_t>::max() / blocks)) {
    return false;
  }
  *result = blocks * block_bytes;
  return true;
}

// Keep the destination recipe check in the fork owner synchronized with the
// paged create contract.  A fork may grow logical capacity, but it must keep
// the exact value/scale plane layout of the source pool.
bool paged_fork_recipe(const uint32_t encoding, uint32_t *const scale_dtype,
                       uint32_t *const quantization_block_size) noexcept {
  if (scale_dtype == nullptr || quantization_block_size == nullptr) {
    return false;
  }
  switch (encoding) {
  case SLLM_HIP_KV_ENCODING_FP16_V1:
    *scale_dtype = 0U;
    *quantization_block_size = 0U;
    return true;
  case SLLM_HIP_KV_ENCODING_FP8_V1:
  case SLLM_HIP_KV_ENCODING_FP8_STATIC_V1:
    *scale_dtype = SLLM_TENSOR_DTYPE_F32;
    *quantization_block_size = 0U;
    return true;
  case SLLM_HIP_KV_ENCODING_NVFP4_V1:
    *scale_dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
    *quantization_block_size = 16U;
    return true;
  case SLLM_HIP_KV_ENCODING_MXFP8_E4_V1:
  case SLLM_HIP_KV_ENCODING_MXFP8_E5_V1:
    *scale_dtype = SLLM_TENSOR_DTYPE_U8;
    *quantization_block_size = 32U;
    return true;
  default:
    return false;
  }
}

// SlidingState uses a fixed nine-entry device table and a separate absolute
// tag array. Keep its fork transaction separate from the monotonic table path
// so both Paged layouts retain their capacity and ABI checks.
sllm_status_t fork_paged_sliding_state(
    const sllm_kv_state_t *const raw_source,
    const sllm_kv_state_paged_create_info_t *const destination_info,
    sllm_kv_state_t **const raw_child,
    sllm_kv_paged_state_fork_info_t *const fork_info,
    sllm_error_sink_t *const error_sink) noexcept {
  KvState *source = nullptr;
  Context *context = nullptr;
  std::shared_ptr<sllm_paged_kv::DevicePool> pool;
  std::shared_ptr<sllm_paged_kv::Pool> host_pool;
  uint64_t source_identity = 0U;
  uint64_t source_length = 0U;
  uint64_t source_generation = 0U;
  uint64_t source_logical_capacity = 0U;
  uint32_t source_key_scale_bits = 0U;
  uint32_t source_value_scale_bits = 0U;
  auto unguard = [&]() noexcept {
    std::lock_guard<std::mutex> lock(registry_mutex);
    if (source != nullptr)
      source->release_active = false;
  };
  try {
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      source = lookup<KvState>(raw_source, HandleKind::KvState);
      if (source == nullptr)
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "paged sliding fork source handle is stale or wrong kind");
      context = source->context;
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (!source->paged || source->sliding_window == 0U ||
          source->paged_sliding_host_state == nullptr ||
          source->paged_sliding_host_pool == nullptr ||
          source->paged_device_pool == nullptr ||
          source->paged_logical_table == nullptr ||
          source->paged_sliding_ring_tags_device == nullptr ||
          source->paged_device_status == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_UNSUPPORTED,
            "paged sliding fork requires SlidingState ownership");
      }
      if (context->poisoned.load() || !kv_quiescent(source) ||
          source->graph_pins != 0U ||
          source->paged_sliding_host_state->pending() ||
          source->paged_logical_table->pending() ||
          source->paged_device_pool->poisoned()) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "paged sliding fork requires a quiescent source state");
      }
      const uint32_t heads = destination_info->head_count == 0U
                                 ? SLLM_HIP_KV_HEAD_COUNT
                                 : destination_info->head_count;
      const uint32_t dim = destination_info->head_dim == 0U
                               ? SLLM_HIP_KV_HEAD_DIM
                               : destination_info->head_dim;
      std::memcpy(&source_key_scale_bits, &source->static_key_scale,
                  sizeof(source_key_scale_bits));
      std::memcpy(&source_value_scale_bits, &source->static_value_scale,
                  sizeof(source_value_scale_bits));
      if (destination_info->session_id != source->session_id ||
          destination_info->layer_id != source->layer_id ||
          heads != source->head_count || dim != source->head_dim ||
          destination_info->memory_kind != SLLM_HIP_KV_MEMORY_KIND_PAGED ||
          destination_info->layout != SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR ||
          destination_info->dtype != source->dtype ||
          destination_info->encoding != source->encoding ||
          destination_info->scale_dtype != SLLM_TENSOR_DTYPE_F32 ||
          destination_info->quantization_block_size != 0U ||
          destination_info->token_block_size != source->paged_token_block_size ||
          destination_info->physical_layout_version !=
              source->paged_physical_layout_version ||
          destination_info->sliding_window_tokens !=
              sllm_paged_kv::kSlidingWindowTokens ||
          destination_info->max_physical_blocks !=
              source->paged_max_physical_blocks ||
          destination_info->capacity_tokens < source->published_length ||
          destination_info->static_key_scale_bits != source_key_scale_bits ||
          destination_info->static_value_scale_bits !=
              source_value_scale_bits) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INVALID_ARGUMENT,
            "paged sliding fork destination changes the source recipe or capacity");
      }
      source_identity = source->state_identity;
      source_length = source->published_length;
      source_generation = source->generation;
      source_logical_capacity = source->paged_logical_table_capacity;
      pool = source->paged_device_pool;
      host_pool = source->paged_sliding_host_pool;
      source->release_active = true;
    }

    const sllm_status_t device_status = select_context_device(context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      unguard();
      return device_status;
    }
    if (destination_info->logical_table_capacity <
            destination_info->capacity_tokens / SLLM_HIP_KV_PAGED_TOKEN_BLOCK_SIZE +
                (destination_info->capacity_tokens %
                         SLLM_HIP_KV_PAGED_TOKEN_BLOCK_SIZE != 0U ? 1U : 0U) ||
        destination_info->logical_table_capacity > pool->logical_capacity()) {
      unguard();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_KV_CAPACITY_EXCEEDED,
          "paged sliding fork child capacity exceeds the source pool");
    }

    hipStream_t stream = nullptr;
    hipError_t status = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    if (status != hipSuccess) {
      unguard();
      return hip_failure(error_sink, status, "create paged sliding fork stream");
    }
    std::unique_ptr<sllm_paged_kv::DevicePool::LogicalTable> child_table;
    std::unique_ptr<sllm_paged_kv::SlidingState> child_host;
    std::unique_ptr<Buffer> key_buffer;
    std::unique_ptr<Buffer> value_buffer;
    std::unique_ptr<KvState> candidate;
    uint32_t *child_status = nullptr;
    bool host_forked = false;
    auto fail = [&](const sllm_status_t result) noexcept {
      (void)hipStreamSynchronize(stream);
      if (candidate != nullptr) {
        if (candidate->paged_sliding_host_state != nullptr)
          (void)candidate->paged_sliding_host_state->release();
        if (candidate->paged_sliding_ring_tags_device != nullptr)
          (void)hipFree(candidate->paged_sliding_ring_tags_device);
        if (candidate->paged_sliding_ring_tags_host != nullptr)
          (void)hipHostFree(candidate->paged_sliding_ring_tags_host);
        candidate.reset();
      } else if (child_host != nullptr && host_forked) {
        (void)child_host->release();
      }
      child_table.reset();
      if (child_status != nullptr)
        (void)hipFree(child_status);
      (void)hipStreamDestroy(stream);
      unguard();
      return result;
    };
    try {
      child_table = pool->make_logical_table(sllm_paged_kv::kSlidingRingSlots,
                                             stream);
      if (hipMalloc(reinterpret_cast<void **>(&child_status),
                    sizeof(*child_status)) != hipSuccess)
        return fail(sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "paged sliding fork status allocation failed"));
      status = hipMemsetAsync(child_status, 0, sizeof(*child_status), stream);
      if (status != hipSuccess)
        return fail(hip_failure(error_sink, status,
                                "clear paged sliding fork status"));
      const sllm_paged_kv::Status host_status =
          source->paged_sliding_host_state->fork_into(child_host);
      if (host_status != sllm_paged_kv::Status::Ok)
        return fail(sllm_public_runtime::write_error(
            error_sink,
            host_status == sllm_paged_kv::Status::Busy
                ? SLLM_STATUS_PUBLIC_BUSY
                : SLLM_STATUS_KV_CAPACITY_EXCEEDED,
            "paged sliding host fork failed"));
      host_forked = true;
      std::vector<uint32_t> blocks;
      std::array<uint64_t, 9U> tags{};
      if (!paged_collect_sliding_blocks(child_host.get(), blocks, tags))
        return fail(sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "paged sliding fork produced an invalid ring"));
      std::vector<sllm_paged_kv::LogicalEntryUpdate> updates;
      for (uint32_t slot = 0U; slot < sllm_paged_kv::kSlidingRingSlots;
           ++slot) {
        if (tags[slot] == std::numeric_limits<uint64_t>::max())
          continue;
        updates.push_back(sllm_paged_kv::LogicalEntryUpdate{
            slot, sllm_paged_kv::kInvalidBlock, child_host->physical_block_at_slot(slot)});
      }
      if (child_table->update_entries(updates, stream) !=
          sllm_paged_kv::DeviceStatus::Ok)
        return fail(sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "paged sliding fork table publication failed"));
      if (hipStreamSynchronize(stream) != hipSuccess ||
          child_table->poll_pending() != sllm_paged_kv::DeviceStatus::Ok)
        return fail(sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "paged sliding fork table synchronization failed"));
      key_buffer = std::make_unique<Buffer>(context, nullptr, 0U);
      value_buffer = std::make_unique<Buffer>(context, nullptr, 0U);
      candidate = std::make_unique<KvState>(
          context, *destination_info, source->head_count, source->head_dim,
          key_buffer.get(), value_buffer.get(), std::move(pool),
          std::move(child_table), std::move(host_pool), nullptr,
          source->value_bytes_per_token, source->scale_bytes_per_token,
          source->context_identity, child_status);
      child_status = nullptr;
      candidate->sliding_window = sllm_paged_kv::kSlidingWindowTokens;
      candidate->paged_sliding_host_pool = candidate->paged_host_pool;
      candidate->paged_host_pool.reset();
      candidate->paged_sliding_host_state = std::move(child_host);
      candidate->paged_sliding_ring_tags_host = nullptr;
      if (hipHostMalloc(reinterpret_cast<void **>(
                            &candidate->paged_sliding_ring_tags_host),
                        sllm_paged_kv::kSlidingRingSlots * sizeof(uint64_t),
                        hipHostMallocPortable) != hipSuccess ||
          hipMalloc(reinterpret_cast<void **>(
                        &candidate->paged_sliding_ring_tags_device),
                    sllm_paged_kv::kSlidingRingSlots * sizeof(uint64_t)) !=
              hipSuccess)
        return fail(sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "paged sliding fork tag allocation failed"));
      std::memcpy(candidate->paged_sliding_ring_tags_host, tags.data(),
                  tags.size() * sizeof(uint64_t));
      status = hipMemcpyAsync(
          candidate->paged_sliding_ring_tags_device,
          candidate->paged_sliding_ring_tags_host,
          tags.size() * sizeof(uint64_t), hipMemcpyHostToDevice, stream);
      if (status != hipSuccess || hipStreamSynchronize(stream) != hipSuccess)
        return fail(hip_failure(error_sink, status,
                                "publish paged sliding fork tags"));
      candidate->published_length = source_length;
      candidate->generation = source_generation;
      candidate->retained_start =
          source_length > sllm_paged_kv::kSlidingWindowTokens
              ? source_length - sllm_paged_kv::kSlidingWindowTokens
              : 0U;
      candidate->last_published_end = source_length;
      candidate->last_published_generation = source_generation;
      candidate->static_key_scale = source->static_key_scale;
      candidate->static_value_scale = source->static_value_scale;
    } catch (const std::exception &error) {
      return fail(sllm_public_runtime::write_error_n_bounded(
          error_sink, SLLM_STATUS_INTERNAL_ERROR, error.what(),
          std::strlen(error.what()), 255U));
    }
    {
      std::lock_guard<std::mutex> lock(context->accounting_mutex);
      if (!sllm_public_runtime::AccountingState::reserve_kv_state(
              context->accounting, key_buffer->accounting,
              value_buffer->accounting)) {
        key_buffer.release();
        value_buffer.release();
        poison_context_locked(context);
        poison_owner.retain(std::move(candidate));
        unguard();
        (void)hipStreamDestroy(stream);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "paged sliding fork accounting reservation exhausted");
      }
    }
    (void)hipStreamDestroy(stream);
    uintptr_t token = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      token = register_handle(candidate.get(), HandleKind::KvState);
    }
    if (token == 0U) {
      unguard();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "paged sliding fork state registry insertion failed");
    }
    candidate->state_identity = token;
    const uint64_t shared_blocks =
        candidate->paged_sliding_host_state->retained_block_count();
    const uint64_t block_bytes = paged_block_bytes(*candidate->paged_device_pool);
    uint64_t shared_bytes = 0U;
    if (!paged_mul_bytes(shared_blocks, block_bytes, &shared_bytes)) {
      unguard();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "paged sliding fork byte accounting overflow");
    }
    fill_paged_fork_info(
        fork_info, source_identity, token, shared_bytes, 0U, 0U, shared_bytes,
        source_length, source->paged_token_block_size,
        source->paged_physical_layout_version, source_logical_capacity,
        destination_info->logical_table_capacity,
        shared_blocks, shared_blocks, 0U, shared_blocks,
        candidate->paged_device_pool->committed_bytes_total());
    *raw_child = reinterpret_cast<sllm_kv_state_t *>(token);
    key_buffer.release();
    value_buffer.release();
    (void)candidate.release();
    unguard();
    return SLLM_STATUS_OK;
  } catch (...) {
    unguard();
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected paged sliding fork failure");
  }
}

uint32_t paged_float_bits(const float value) noexcept {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

#endif

} // namespace

extern "C" sllm_status_t sllm_kv_state_fork_paged(
    const sllm_kv_state_t *const raw_source,
    const sllm_kv_state_paged_create_info_t *const destination_info,
    sllm_kv_state_t **const raw_child,
    sllm_kv_paged_state_fork_info_t *const fork_info,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (raw_child != nullptr) {
      *raw_child = nullptr;
    }
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_source == nullptr || destination_info == nullptr ||
        raw_child == nullptr || fork_info == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "paged KV fork source, destination, child, or info is null");
    }
    const sllm_status_t fork_info_status =
        validate_paged_fork_info_header(fork_info, error_sink);
    if (fork_info_status != SLLM_STATUS_OK) {
      return fork_info_status;
    }
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
    (void)raw_source;
    (void)destination_info;
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_UNSUPPORTED,
        "paged KV state fork requires the HIP runtime build");
#else
    const sllm_status_t destination_status =
        sllm_kv_state::validate_state_create_info_paged(destination_info,
                                                        error_sink);
    if (destination_status != SLLM_STATUS_OK) {
      return destination_status;
    }

    bool sliding_probe = false;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      KvState *const probe = lookup<KvState>(raw_source, HandleKind::KvState);
      sliding_probe = probe != nullptr && probe->sliding_window != 0U;
    }
    if (sliding_probe)
      return fork_paged_sliding_state(raw_source, destination_info, raw_child,
                                      fork_info, error_sink);

    KvState *source = nullptr;
    Context *context = nullptr;
    uint64_t source_length = 0U;
    uint64_t source_generation = 0U;
    uint64_t source_logical_capacity = 0U;
    uint64_t source_identity = 0U;
    std::shared_ptr<sllm_paged_kv::DevicePool> parent_device_pool;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      source = lookup<KvState>(raw_source, HandleKind::KvState);
      if (source == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "paged KV fork source handle is stale or wrong kind");
      }
      if (!source->paged) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_UNSUPPORTED,
            "paged KV fork requires a paged KV state source");
      }
      context = source->context;
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (source->paged_host_state == nullptr ||
          source->paged_host_pool == nullptr ||
          source->paged_device_pool == nullptr ||
          source->paged_logical_table == nullptr ||
          source->paged_device_status == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "paged KV fork source has incomplete ownership");
      }
      if (context->poisoned.load() || !kv_quiescent(source) ||
          source->graph_pins != 0U || source->paged_host_state->pending() ||
          source->paged_logical_table->pending() ||
          source->paged_device_pool->poisoned()) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "paged KV fork requires a quiescent source state");
      }
      const uint32_t destination_heads = destination_info->head_count == 0U
                                             ? SLLM_HIP_KV_HEAD_COUNT
                                             : destination_info->head_count;
      const uint32_t destination_dim = destination_info->head_dim == 0U
                                           ? SLLM_HIP_KV_HEAD_DIM
                                           : destination_info->head_dim;
      uint32_t expected_scale_dtype = 0U;
      uint32_t expected_quantization_block_size = 0U;
      if (!paged_fork_recipe(source->encoding, &expected_scale_dtype,
                             &expected_quantization_block_size)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_UNSUPPORTED,
            "paged KV fork source uses an unsupported encoding recipe");
      }
      if (source->encoding == SLLM_HIP_KV_ENCODING_MXFP8_E5_V1 &&
          !matches_runtime_gcn_arch(context->device_properties.gcnArchName,
                                    "gfx1030")) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_UNSUPPORTED,
            "paged KV MXFP8 E5 fork requires exact gfx1030");
      }
      if (destination_info->session_id != source->session_id ||
          destination_info->layer_id != source->layer_id ||
          destination_heads != source->head_count ||
          destination_dim != source->head_dim ||
          destination_info->memory_kind != SLLM_HIP_KV_MEMORY_KIND_PAGED ||
          destination_info->layout != SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR ||
          destination_info->dtype != source->dtype ||
          destination_info->encoding != source->encoding ||
          destination_info->scale_dtype != expected_scale_dtype ||
          destination_info->quantization_block_size !=
              expected_quantization_block_size ||
          destination_info->token_block_size !=
              source->paged_token_block_size ||
          destination_info->physical_layout_version !=
              source->paged_physical_layout_version ||
          destination_info->sliding_window_tokens != 0U ||
          destination_info->max_physical_blocks !=
              source->paged_max_physical_blocks ||
          destination_info->capacity_tokens < source->published_length) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INVALID_ARGUMENT,
            "paged KV fork destination changes the source recipe or capacity");
      }
      if (source->encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1 &&
          (destination_info->static_key_scale_bits !=
               paged_float_bits(source->static_key_scale) ||
           destination_info->static_value_scale_bits !=
               paged_float_bits(source->static_value_scale))) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INVALID_ARGUMENT,
            "paged KV fork destination changes static FP8 scale bits");
      }
      source_length = source->published_length;
      source_generation = source->generation;
      source_logical_capacity = source->paged_logical_table_capacity;
      source_identity = source->state_identity;
      parent_device_pool = source->paged_device_pool;
      source->release_active = true;
    }

    const auto unguard_source = [&]() noexcept {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      if (source != nullptr) {
        source->release_active = false;
      }
    };

    const sllm_status_t device_status =
        select_context_device(context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      unguard_source();
      return device_status;
    }

    const uint64_t required_child_blocks =
        destination_info->capacity_tokens / SLLM_HIP_KV_PAGED_TOKEN_BLOCK_SIZE +
        (destination_info->capacity_tokens %
                     SLLM_HIP_KV_PAGED_TOKEN_BLOCK_SIZE !=
                 0U
             ? 1U
             : 0U);
    if (destination_info->logical_table_capacity < required_child_blocks ||
        destination_info->logical_table_capacity >
            parent_device_pool->logical_capacity()) {
      unguard_source();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_KV_CAPACITY_EXCEEDED,
          "paged KV fork child logical table exceeds the parent pool");
    }
    if (source_logical_capacity != parent_device_pool->logical_capacity() ||
        source->paged_max_physical_blocks !=
            parent_device_pool->physical_capacity()) {
      unguard_source();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "paged KV fork source pool metadata is inconsistent");
    }

    hipStream_t stream = nullptr;
    hipError_t stream_status =
        hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    if (stream_status != hipSuccess) {
      unguard_source();
      return hip_failure(error_sink, stream_status,
                         "create paged KV fork stream");
    }

    std::unique_ptr<sllm_paged_kv::DevicePool::LogicalTable> child_table;
    std::unique_ptr<sllm_paged_kv::State> child_host_state;
    std::unique_ptr<Buffer> key_buffer;
    std::unique_ptr<Buffer> value_buffer;
    std::unique_ptr<KvState> candidate;
    uint32_t *child_device_status = nullptr;
    bool host_forked = false;
    bool cleanup_failed = false;
    auto cleanup_unpublished = [&]() noexcept {
      if (stream != nullptr) {
        (void)hipStreamSynchronize(stream);
      }
      if (candidate != nullptr) {
        if (candidate->paged_host_state != nullptr) {
          const sllm_paged_kv::Status status =
              candidate->paged_host_state->release();
          if (status != sllm_paged_kv::Status::Ok) {
            cleanup_failed = true;
            return;
          }
          candidate->paged_host_state.reset();
        }
        candidate->paged_logical_table.reset();
        candidate->paged_device_pool.reset();
        if (candidate->paged_device_status != nullptr) {
          const hipError_t status = free_allocation_with_fault_injection(
              candidate->paged_device_status);
          if (status != hipSuccess) {
            orphan_owner.retain_allocation(context,
                                           candidate->paged_device_status);
            cleanup_failed = true;
          }
          candidate->paged_device_status = nullptr;
        }
        candidate.reset();
      } else {
        if (child_host_state != nullptr && host_forked) {
          if (child_host_state->release() != sllm_paged_kv::Status::Ok) {
            cleanup_failed = true;
          } else {
            child_host_state.reset();
          }
        }
        child_table.reset();
        if (child_device_status != nullptr) {
          const hipError_t status =
              free_allocation_with_fault_injection(child_device_status);
          if (status != hipSuccess) {
            orphan_owner.retain_allocation(context, child_device_status);
            cleanup_failed = true;
          }
          child_device_status = nullptr;
        }
      }
    };
    auto fail_unpublished = [&](const sllm_status_t status) noexcept {
      cleanup_unpublished();
      if (cleanup_failed) {
        if (candidate != nullptr) {
          key_buffer.release();
          value_buffer.release();
          retain_poisoned(candidate, context);
        }
        poison_context(context);
      }
      const hipError_t destroy = hipStreamDestroy(stream);
      if (destroy != hipSuccess) {
        orphan_owner.retain_stream(context, stream);
        poison_context(context);
      }
      unguard_source();
      return status;
    };

    try {
      child_table = parent_device_pool->make_logical_table(
          destination_info->logical_table_capacity, stream);
      if (hipMalloc(reinterpret_cast<void **>(&child_device_status),
                    sizeof(*child_device_status)) != hipSuccess) {
        return fail_unpublished(sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "paged KV fork device status allocation failed"));
      }
      stream_status = hipMemsetAsync(child_device_status, 0,
                                     sizeof(*child_device_status), stream);
      if (stream_status != hipSuccess) {
        return fail_unpublished(hip_failure(
            error_sink, stream_status, "clear paged KV fork device status"));
      }

      const sllm_paged_kv::Status host_status =
          source->paged_host_state->fork_into(
              child_host_state, destination_info->capacity_tokens);
      if (host_status != sllm_paged_kv::Status::Ok) {
        const sllm_status_t mapped =
            host_status == sllm_paged_kv::Status::Capacity ||
                    host_status == sllm_paged_kv::Status::PoolExhausted
                ? SLLM_STATUS_KV_CAPACITY_EXCEEDED
            : host_status == sllm_paged_kv::Status::Busy
                ? SLLM_STATUS_PUBLIC_BUSY
                : SLLM_STATUS_INTERNAL_ERROR;
        return fail_unpublished(sllm_public_runtime::write_error(
            error_sink, mapped,
            "paged KV host fork cannot allocate the requested child"));
      }
      host_forked = true;

      std::vector<uint32_t> source_blocks;
      std::vector<uint32_t> child_blocks;
      if (!paged_collect_blocks(source->paged_host_state.get(),
                                source_blocks) ||
          !paged_collect_blocks(child_host_state.get(), child_blocks) ||
          child_blocks.size() != source_blocks.size()) {
        return fail_unpublished(sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "paged KV host fork produced an invalid block table"));
      }
      std::vector<sllm_paged_kv::LogicalEntryUpdate> updates;
      updates.reserve(child_blocks.size());
      for (std::size_t index = 0U; index < child_blocks.size(); ++index) {
        const uint32_t physical = child_blocks[index];
        if (!parent_device_pool->descriptor_ready(physical)) {
          return fail_unpublished(sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_INTERNAL_ERROR,
              "paged KV fork source block descriptor is not published"));
        }
        updates.push_back(sllm_paged_kv::LogicalEntryUpdate{
            static_cast<uint32_t>(index), sllm_paged_kv::kInvalidBlock,
            physical});
      }
      const sllm_paged_kv::DeviceStatus table_status =
          child_table->update_entries(updates, stream);
      if (table_status != sllm_paged_kv::DeviceStatus::Ok) {
        return fail_unpublished(sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "paged KV fork child table publication failed"));
      }
      stream_status = hipStreamSynchronize(stream);
      const sllm_paged_kv::DeviceStatus pending_status =
          child_table->poll_pending();
      if (stream_status != hipSuccess ||
          pending_status != sllm_paged_kv::DeviceStatus::Ok) {
        return fail_unpublished(hip_failure(
            error_sink,
            stream_status == hipSuccess ? child_table->last_hip_error()
                                        : stream_status,
            "synchronize paged KV fork child table"));
      }

      key_buffer = std::make_unique<Buffer>(context, nullptr, 0U);
      value_buffer = std::make_unique<Buffer>(context, nullptr, 0U);
      candidate = std::make_unique<KvState>(
          context, *destination_info, source->head_count, source->head_dim,
          key_buffer.get(), value_buffer.get(), std::move(parent_device_pool),
          std::move(child_table),
          std::shared_ptr<sllm_paged_kv::Pool>(source->paged_host_pool),
          std::move(child_host_state), source->value_bytes_per_token,
          source->scale_bytes_per_token, source->context_identity,
          child_device_status);
      child_device_status = nullptr;
      candidate->published_length = source_length;
      candidate->generation = source_generation;
      candidate->last_published_end = source_length;
      candidate->last_published_generation = source_generation;
      candidate->static_key_scale = source->static_key_scale;
      candidate->static_value_scale = source->static_value_scale;
    } catch (const std::exception &error) {
      return fail_unpublished(sllm_public_runtime::write_error_n_bounded(
          error_sink, SLLM_STATUS_INTERNAL_ERROR, error.what(),
          std::strlen(error.what()), 255U));
    } catch (...) {
      return fail_unpublished(sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "unexpected paged KV fork allocation failure"));
    }

    {
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (!sllm_public_runtime::AccountingState::reserve_kv_state(
              context->accounting, key_buffer->accounting,
              value_buffer->accounting)) {
        key_buffer.release();
        value_buffer.release();
        poison_context_locked(context);
        poison_owner.retain(std::move(candidate));
        const hipError_t destroy = hipStreamDestroy(stream);
        if (destroy != hipSuccess) {
          orphan_owner.retain_stream(context, stream);
        }
        unguard_source();
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "paged KV fork accounting reservation is exhausted; state "
            "quarantined");
      }
    }

    stream_status = hipStreamDestroy(stream);
    if (stream_status != hipSuccess) {
      key_buffer.release();
      value_buffer.release();
      retain_poisoned(candidate, context);
      orphan_owner.retain_stream(context, stream);
      unguard_source();
      return hip_failure(error_sink, stream_status,
                         "destroy paged KV fork stream");
    }
    stream = nullptr;

    uintptr_t token = 0U;
    try {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      token = register_handle(candidate.get(), HandleKind::KvState);
    } catch (...) {
      token = 0U;
    }
    if (token == 0U) {
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (!sllm_public_runtime::AccountingState::release_kv_state(
              context->accounting, key_buffer->accounting,
              value_buffer->accounting)) {
        key_buffer.release();
        value_buffer.release();
        poison_context_locked(context);
        poison_owner.retain(std::move(candidate));
        unguard_source();
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "paged KV fork accounting rollback failed; state quarantined");
      }
      cleanup_unpublished();
      unguard_source();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "paged KV fork state registry insertion failed");
    }

    candidate->state_identity = token;
    const uint64_t block_count =
        source_length / SLLM_HIP_KV_PAGED_TOKEN_BLOCK_SIZE +
        (source_length % SLLM_HIP_KV_PAGED_TOKEN_BLOCK_SIZE != 0U ? 1U : 0U);
    const uint64_t block_bytes =
        paged_block_bytes(*candidate->paged_device_pool);
    uint64_t shared_bytes = 0U;
    if (!paged_mul_bytes(block_count, block_bytes, &shared_bytes)) {
      {
        std::lock_guard<std::mutex> registry_lock(registry_mutex);
        unregister_handle(reinterpret_cast<sllm_kv_state_t *>(token));
      }
      key_buffer.release();
      value_buffer.release();
      retain_poisoned(candidate, context);
      unguard_source();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "paged KV fork byte accounting overflow");
    }
    fill_paged_fork_info(
        fork_info, source_identity, token, shared_bytes, 0U, 0U, shared_bytes,
        source_length, source->paged_token_block_size,
        source->paged_physical_layout_version, source_logical_capacity,
        destination_info->logical_table_capacity, block_count, block_count, 0U,
        block_count, candidate->paged_device_pool->committed_bytes_total());
    *raw_child = reinterpret_cast<sllm_kv_state_t *>(token);
    key_buffer.release();
    value_buffer.release();
    (void)candidate.release();
    unguard_source();
    return SLLM_STATUS_OK;
#endif
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in paged KV state fork");
  }
}

extern "C" sllm_status_t
sllm_kv_state_fork_query_paged(const sllm_kv_state_t *const raw_state,
                               sllm_kv_paged_state_fork_info_t *const fork_info,
                               sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t header_status =
        validate_paged_fork_info_header(fork_info, error_sink);
    if (header_status != SLLM_STATUS_OK) {
      return header_status;
    }
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
    (void)raw_state;
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_UNSUPPORTED,
        "paged KV fork query requires the HIP runtime build");
#else
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    KvState *const state = lookup<KvState>(raw_state, HandleKind::KvState);
    if (state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "paged KV fork query state handle is stale or wrong kind");
    }
    std::lock_guard<std::mutex> accounting_lock(
        state->context->accounting_mutex);
    if (!state->paged) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_UNSUPPORTED,
          "paged KV fork query requires a paged KV state source");
    }
    if (state->sliding_window != 0U) {
      if (state->context->poisoned.load() || !kv_quiescent(state) ||
          state->graph_pins != 0U || state->paged_sliding_host_state == nullptr ||
          state->paged_sliding_host_pool == nullptr ||
          state->paged_logical_table == nullptr ||
          state->paged_device_pool == nullptr ||
          state->paged_sliding_host_state->pending() ||
          state->paged_logical_table->pending() ||
          state->paged_device_pool->poisoned()) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "paged sliding fork query requires a quiescent state");
      }
      std::vector<uint32_t> blocks;
      std::array<uint64_t, 9U> tags{};
      if (!paged_collect_sliding_blocks(state->paged_sliding_host_state.get(),
                                        blocks, tags)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "paged sliding fork query found an invalid ring");
      }
      const uint64_t block_bytes = paged_block_bytes(*state->paged_device_pool);
      uint64_t shared_blocks = 0U;
      for (const uint32_t physical : blocks) {
        const uint32_t references =
            state->paged_sliding_host_pool->reference_count(physical);
        if (references == 0U)
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_INTERNAL_ERROR,
              "paged sliding fork query found an unowned physical block");
        if (references > 1U)
          ++shared_blocks;
      }
      const uint64_t total_blocks = static_cast<uint64_t>(blocks.size());
      uint64_t total_bytes = 0U, shared_bytes = 0U, owned_bytes = 0U;
      if (block_bytes == UINT64_MAX ||
          !paged_mul_bytes(total_blocks, block_bytes, &total_bytes) ||
          !paged_mul_bytes(shared_blocks, block_bytes, &shared_bytes) ||
          !paged_mul_bytes(total_blocks - shared_blocks, block_bytes,
                           &owned_bytes)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "paged sliding fork query byte accounting overflow");
      }
      fill_paged_fork_info(
          fork_info, state->state_identity, state->state_identity, total_bytes,
          owned_bytes, state->cow_copied_bytes, shared_bytes,
          state->published_length, state->paged_token_block_size,
          state->paged_physical_layout_version,
          state->paged_logical_table_capacity,
          state->paged_logical_table_capacity, total_blocks, total_blocks,
          state->paged_cow_block_count, shared_blocks,
          state->paged_device_pool->committed_bytes_total());
      return SLLM_STATUS_OK;
    }
    if (state->context->poisoned.load() || !kv_quiescent(state) ||
        state->graph_pins != 0U || state->paged_host_state == nullptr ||
        state->paged_host_pool == nullptr ||
        state->paged_logical_table == nullptr ||
        state->paged_device_pool == nullptr ||
        state->paged_host_state->pending() ||
        state->paged_logical_table->pending() ||
        state->paged_device_pool->poisoned()) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "paged KV fork query requires a quiescent state");
    }
    std::vector<uint32_t> blocks;
    if (!paged_collect_blocks(state->paged_host_state.get(), blocks)) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "paged KV fork query found an invalid host block table");
    }
    const uint64_t block_bytes = paged_block_bytes(*state->paged_device_pool);
    uint64_t shared_blocks = 0U;
    for (const uint32_t physical : blocks) {
      const uint32_t references =
          state->paged_host_pool->reference_count(physical);
      if (references == 0U) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "paged KV fork query found an unowned physical block");
      }
      if (references > 1U)
        ++shared_blocks;
    }
    const uint64_t total_blocks = static_cast<uint64_t>(blocks.size());
    uint64_t total_bytes = 0U;
    uint64_t shared_bytes = 0U;
    uint64_t owned_bytes = 0U;
    if (block_bytes == UINT64_MAX ||
        !paged_mul_bytes(total_blocks, block_bytes, &total_bytes) ||
        !paged_mul_bytes(shared_blocks, block_bytes, &shared_bytes) ||
        !paged_mul_bytes(total_blocks - shared_blocks, block_bytes,
                         &owned_bytes)) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "paged KV fork query byte accounting overflow");
    }
    fill_paged_fork_info(
        fork_info, state->state_identity, state->state_identity, total_bytes,
        owned_bytes, state->cow_copied_bytes, shared_bytes,
        state->published_length, state->paged_token_block_size,
        state->paged_physical_layout_version,
        state->paged_logical_table_capacity,
        state->paged_logical_table_capacity, total_blocks, total_blocks,
        state->paged_cow_block_count, shared_blocks,
        state->paged_device_pool->committed_bytes_total());
    return SLLM_STATUS_OK;
#endif
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in paged KV fork query");
  }
}

extern "C" sllm_status_t
sllm_kv_state_export(const sllm_kv_state_t *const raw_state,
                     const sllm_state_chunk_t *const chunk,
                     sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t chunk_status = validate_chunk(chunk, error_sink);
    if (chunk_status != SLLM_STATUS_OK) {
      return chunk_status;
    }
    if (raw_state == nullptr) {
      return sllm_public_runtime::write_error(error_sink,
                                              SLLM_STATUS_INVALID_ARGUMENT,
                                              "KV export state handle is null");
    }
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_UNSUPPORTED,
        "legacy KV raw-plane export ABI is retired; use paged KV image ABI");
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in retired KV export stub");
  }
}

extern "C" sllm_status_t
sllm_kv_state_import(const sllm_kv_state_t *const raw_state,
                     const sllm_state_chunk_t *const chunk,
                     sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t chunk_status = validate_chunk(chunk, error_sink);
    if (chunk_status != SLLM_STATUS_OK) {
      return chunk_status;
    }
    if (raw_state == nullptr) {
      return sllm_public_runtime::write_error(error_sink,
                                              SLLM_STATUS_INVALID_ARGUMENT,
                                              "KV import state handle is null");
    }
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_UNSUPPORTED,
        "legacy KV raw-plane import ABI is retired; use paged KV image ABI");
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in retired KV import stub");
  }
}

extern "C" sllm_status_t
sllm_kv_state_image_query(const sllm_kv_state_t *const raw_state,
                          sllm_state_image_info_t *const image_info,
                          sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t header_status =
        validate_image_header(image_info, error_sink);
    if (header_status != SLLM_STATUS_OK) {
      return header_status;
    }
    if (raw_state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "KV image query state handle is null");
    }
    const uint32_t struct_size = image_info->struct_size;
    const uint32_t abi_version = image_info->abi_version;
    std::memset(image_info, 0, sizeof(*image_info));
    image_info->struct_size = struct_size;
    image_info->abi_version = abi_version;
    image_info->info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_UNSUPPORTED,
        "legacy KV image query ABI is retired; use paged KV image ABI");
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in retired KV image query stub");
  }
}

extern "C" sllm_status_t
sllm_kv_state_image_plane_size(const sllm_kv_state_t *const raw_state,
                               const uint32_t plane, uint64_t *const size_bytes,
                               sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (size_bytes == nullptr || raw_state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "KV image plane size arguments are null");
    }
    (void)plane;
    *size_bytes = 0U;
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_UNSUPPORTED,
        "legacy KV image plane ABI is retired; use paged KV image ABI");
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in retired KV image plane stub");
  }
}

extern "C" sllm_status_t
sllm_kv_state_import_finalize(const sllm_kv_state_t *const raw_state,
                              const sllm_state_image_info_t *const image_info,
                              sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t header_status =
        validate_image_header(image_info, error_sink);
    if (header_status != SLLM_STATUS_OK) {
      return header_status;
    }
    if (raw_state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "KV import finalize state handle is null");
    }
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_UNSUPPORTED,
        "legacy KV image finalize ABI is retired; use paged KV image ABI");
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in retired KV image finalize stub");
  }
}

extern "C" sllm_status_t sllm_linear_attention_state_fork(
    const sllm_linear_attention_state_t *const raw_source,
    const sllm_linear_attention_state_create_info_t *const destination_info,
    sllm_linear_attention_state_t **const raw_child,
    sllm_state_fork_info_t *const fork_info,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (raw_child != nullptr) {
      *raw_child = nullptr;
    }
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_source == nullptr || destination_info == nullptr ||
        raw_child == nullptr || fork_info == nullptr ||
        fork_info->struct_size != sizeof(*fork_info) ||
        fork_info->abi_version != SLLM_HIP_ABI_VERSION) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "linear state fork arguments are invalid");
    }
    const sllm_status_t destination_status =
        sllm_linear_attention::validate_state_create_info(destination_info,
                                                          error_sink);
    if (destination_status != SLLM_STATUS_OK) {
      return destination_status;
    }
    LinearAttentionState *source = nullptr;
    Context *context = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      source = lookup<LinearAttentionState>(raw_source,
                                            HandleKind::LinearAttentionState);
      if (source == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "linear state fork source handle is stale or wrong kind");
      }
      context = source->context;
      std::lock_guard<std::mutex> lock(context->accounting_mutex);
      if (context->poisoned.load() || !linear_quiescent(source)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "linear state fork requires a quiescent source state");
      }
      if (destination_info->session_id != source->session_id ||
          destination_info->layer_id != source->layer_id ||
          destination_info->qk_heads != source->qk_heads ||
          destination_info->value_heads != source->value_heads ||
          destination_info->head_dim != source->head_dim ||
          destination_info->conv_kernel_size != source->conv_kernel_size ||
          destination_info->capacity_tokens < source->published_length) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INVALID_ARGUMENT,
            "linear fork destination changes layout or is smaller than prefix");
      }
      source->release_active = true;
    }
    const auto unguard_source = [&]() noexcept {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      if (source != nullptr) {
        source->release_active = false;
      }
    };
    const sllm_status_t device_status =
        select_context_device(context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      unguard_source();
      return device_status;
    }
    const uint64_t conv_bytes =
        static_cast<uint64_t>(source->conv_kernel_size - 1U) *
        source->qkv_width * sizeof(uint16_t);
    const uint64_t recurrent_bytes =
        static_cast<uint64_t>(source->value_heads) * source->head_dim *
        source->head_dim * sizeof(float);
    const std::size_t scratch_bytes =
        static_cast<std::size_t>(source->scratch_bytes);
    void *allocations[5] = {};
    const std::size_t bytes[5] = {static_cast<std::size_t>(conv_bytes),
                                  static_cast<std::size_t>(conv_bytes),
                                  static_cast<std::size_t>(recurrent_bytes),
                                  static_cast<std::size_t>(recurrent_bytes),
                                  scratch_bytes};
    uint64_t copied = 0U;
    hipError_t clone_status = hipSuccess;
    for (std::size_t index = 0U; index != 5U; ++index) {
      if (bytes[index] == 0U) {
        continue;
      }
      clone_status = hipMalloc(&allocations[index], bytes[index]);
      if (clone_status == hipSuccess) {
        const void *const source_ptr = index < 2U ? source->conv_state[index]
                                       : index < 4U
                                           ? source->recurrent_state[index - 2U]
                                           : source->scratch;
        clone_status = state_copy(allocations[index], source_ptr, bytes[index],
                                  hipMemcpyDeviceToDevice);
      }
      if (clone_status != hipSuccess) {
        break;
      }
      copied += bytes[index];
    }
    if (clone_status != hipSuccess) {
      (void)cleanup_linear_allocations(context, allocations, 5U);
      unguard_source();
      return hip_failure(error_sink, clone_status,
                         "fork linear attention state planes");
    }
    std::unique_ptr<LinearAttentionState> candidate;
    try {
      candidate = std::make_unique<LinearAttentionState>(
          context, source->session_id, source->layer_id,
          destination_info->capacity_tokens, destination_info->qk_heads,
          destination_info->value_heads, destination_info->head_dim,
          destination_info->conv_kernel_size,
          std::array<void *, 2>{allocations[0], allocations[1]},
          std::array<void *, 2>{allocations[2], allocations[3]},
          source->context_identity);
      candidate->scratch = allocations[4];
      candidate->scratch_bytes = source->scratch_bytes;
      candidate->published_length = source->published_length;
      candidate->generation = source->generation;
      candidate->active_slot = source->active_slot;
      candidate->last_published_start = source->last_published_start;
      candidate->last_published_end = source->last_published_end;
      candidate->last_published_generation = source->last_published_generation;
    } catch (...) {
      (void)cleanup_linear_allocations(context, allocations, 5U);
      unguard_source();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "linear state fork allocation failed");
    }
    bool accounting_ok = false;
    {
      std::lock_guard<std::mutex> lock(context->accounting_mutex);
      accounting_ok = reserve_linear_attention_state(context->accounting);
    }
    if (!accounting_ok) {
      (void)cleanup_linear_allocations(context, allocations, 5U);
      unguard_source();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "linear state fork accounting reservation is exhausted");
    }
    uintptr_t token = 0U;
    try {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      token =
          register_handle(candidate.get(), HandleKind::LinearAttentionState);
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(context->accounting_mutex);
        (void)release_linear_attention_state(context->accounting);
      }
      (void)cleanup_linear_allocations(context, allocations, 5U);
      unguard_source();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "linear state fork registry allocation failed");
    }
    if (token == 0U) {
      {
        std::lock_guard<std::mutex> lock(context->accounting_mutex);
        (void)release_linear_attention_state(context->accounting);
      }
      (void)cleanup_linear_allocations(context, allocations, 5U);
      unguard_source();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "linear state fork token allocation failed");
    }
    candidate->state_identity = token;
    *raw_child = reinterpret_cast<sllm_linear_attention_state_t *>(token);
    fill_fork_info(fork_info, SLLM_HIP_STATE_FORK_MODE_DEVICE_COPY,
                   source->state_identity, token, copied, copied, copied, 0U,
                   source->published_length, 0U);
    (void)candidate.release();
    unguard_source();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in linear state fork");
  }
}

extern "C" sllm_status_t sllm_linear_attention_state_export(
    const sllm_linear_attention_state_t *const raw_state,
    const sllm_state_chunk_t *const chunk,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t chunk_status = validate_chunk(chunk, error_sink);
    if (chunk_status != SLLM_STATUS_OK) {
      return chunk_status;
    }
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    LinearAttentionState *const state = lookup<LinearAttentionState>(
        raw_state, HandleKind::LinearAttentionState);
    if (state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "linear export state handle is stale or wrong kind");
    }
    std::lock_guard<std::mutex> lock(state->context->accounting_mutex);
    if (!linear_quiescent(state) || state->context->poisoned.load()) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "linear export requires a quiescent state");
    }
    const LinearPlaneRef ref = linear_plane(state, chunk->plane);
    if (ref.pointer == nullptr || chunk->byte_offset > ref.bytes ||
        chunk->byte_length > ref.bytes - chunk->byte_offset) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
          "linear export range is outside the exact native plane");
    }
    const sllm_status_t device_status =
        select_context_device(state->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      return device_status;
    }
    const hipError_t status = state_copy(
        chunk->host_pointer,
        static_cast<const char *>(ref.pointer) +
            static_cast<std::size_t>(chunk->byte_offset),
        static_cast<std::size_t>(chunk->byte_length), hipMemcpyDeviceToHost);
    return status == hipSuccess ? SLLM_STATUS_OK
                                : hip_failure(error_sink, status,
                                              "export linear raw state chunk");
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in linear state export");
  }
}

extern "C" sllm_status_t sllm_linear_attention_state_import(
    const sllm_linear_attention_state_t *const raw_state,
    const sllm_state_chunk_t *const chunk,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t chunk_status = validate_chunk(chunk, error_sink);
    if (chunk_status != SLLM_STATUS_OK) {
      return chunk_status;
    }
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    LinearAttentionState *const state = lookup<LinearAttentionState>(
        raw_state, HandleKind::LinearAttentionState);
    if (state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "linear import state handle is stale or wrong kind");
    }
    std::lock_guard<std::mutex> lock(state->context->accounting_mutex);
    if (!linear_quiescent(state) || state->context->poisoned.load()) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "linear import requires a quiescent state");
    }
    const LinearPlaneRef ref = linear_plane(state, chunk->plane);
    if (ref.pointer == nullptr || chunk->byte_offset > ref.bytes ||
        chunk->byte_length > ref.bytes - chunk->byte_offset) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
          "linear import range is outside the exact native plane");
    }
    const sllm_status_t device_status =
        select_context_device(state->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      return device_status;
    }
    const hipError_t status = state_copy(
        static_cast<char *>(ref.pointer) +
            static_cast<std::size_t>(chunk->byte_offset),
        chunk->host_pointer, static_cast<std::size_t>(chunk->byte_length),
        hipMemcpyHostToDevice);
    if (status == hipSuccess) {
      state->import_plane_mask |= UINT32_C(1) << (chunk->plane - 1U);
      return SLLM_STATUS_OK;
    }
    return hip_failure(error_sink, status, "import linear raw state chunk");
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in linear state import");
  }
}

extern "C" sllm_status_t sllm_linear_attention_state_image_query(
    const sllm_linear_attention_state_t *const raw_state,
    sllm_state_image_info_t *const image_info,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t header_status =
        validate_image_header(image_info, error_sink);
    if (header_status != SLLM_STATUS_OK) {
      return header_status;
    }
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    LinearAttentionState *const state = lookup<LinearAttentionState>(
        raw_state, HandleKind::LinearAttentionState);
    if (state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "linear image query state handle is stale or wrong kind");
    }
    std::lock_guard<std::mutex> lock(state->context->accounting_mutex);
    if (!linear_quiescent(state) || state->context->poisoned.load()) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "linear image query requires a quiescent state");
    }
    fill_linear_image_info(state, image_info);
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in linear state image query");
  }
}

extern "C" sllm_status_t sllm_linear_attention_state_import_finalize(
    const sllm_linear_attention_state_t *const raw_state,
    const sllm_state_image_info_t *const image_info,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t header_status =
        validate_image_header(image_info, error_sink);
    if (header_status != SLLM_STATUS_OK) {
      return header_status;
    }
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    LinearAttentionState *const state = lookup<LinearAttentionState>(
        raw_state, HandleKind::LinearAttentionState);
    if (state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "linear image finalize state handle is stale or wrong kind");
    }
    std::lock_guard<std::mutex> lock(state->context->accounting_mutex);
    const uint32_t expected_mask =
        state->scratch == nullptr ? UINT32_C(0x0f) : UINT32_C(0x1f);
    const uint32_t expected_count = state->scratch == nullptr ? 4U : 5U;
    if (!linear_quiescent(state) || state->context->poisoned.load() ||
        image_info->session_id != state->session_id ||
        image_info->layer_id != state->layer_id ||
        image_info->dtype != SLLM_TENSOR_DTYPE_BF16 ||
        image_info->encoding != SLLM_TENSOR_ENCODING_UNQUANTIZED ||
        image_info->active_slot > 1U ||
        image_info->capacity_tokens > state->capacity_tokens ||
        image_info->published_length > state->capacity_tokens ||
        image_info->plane_count != expected_count ||
        (state->import_plane_mask & expected_mask) != expected_mask) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "linear image metadata or imported plane set is incomplete");
    }
    state->active_slot = image_info->active_slot;
    state->published_length = image_info->published_length;
    state->generation = image_info->generation;
    state->last_published_start = 0U;
    state->last_published_end = image_info->published_length;
    state->last_published_generation = image_info->generation;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in linear state import finalize");
  }
}

extern "C" sllm_status_t sllm_linear_attention_state_image_plane_size(
    const sllm_linear_attention_state_t *const raw_state, const uint32_t plane,
    uint64_t *const size_bytes, sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK)
      return sink_status;
    if (size_bytes == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "linear image plane size output is null");
    }
    *size_bytes = 0U;
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    LinearAttentionState *const state = lookup<LinearAttentionState>(
        raw_state, HandleKind::LinearAttentionState);
    if (state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "linear image plane size state handle is stale or wrong kind");
    }
    std::lock_guard<std::mutex> lock(state->context->accounting_mutex);
    if (!linear_quiescent(state) || state->context->poisoned.load()) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "linear image plane size requires a quiescent state");
    }
    const LinearPlaneRef ref = linear_plane(state, plane);
    if (ref.pointer == nullptr || ref.bytes == 0U) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "linear image plane is unknown or empty");
    }
    *size_bytes = ref.bytes;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in linear image plane size");
  }
}
