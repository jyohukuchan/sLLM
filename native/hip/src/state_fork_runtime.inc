/* Phase 41 additive state fork and raw-plane persistence ABI.  This include is
 * kept separate from the legacy append/query implementations so the v1 ABI
 * remains source-compatible. */

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
      chunk->info_version != SLLM_HIP_STATE_FORK_INFO_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "state chunk ABI or version is unsupported");
  }
  if (chunk->reserved0 != 0U || chunk->reserved1 != 0U ||
      !state_reserved_zero(chunk->reserved, 4U)) {
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
      info->info_version != SLLM_HIP_STATE_FORK_INFO_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "state image info ABI or version is unsupported");
  }
  if (info->reserved0 != 0U || !state_reserved_zero(info->reserved, 7U)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "state image info reserved fields must be zero");
  }
  return SLLM_STATUS_OK;
}

void fill_kv_image_info(const KvState *const state,
                        sllm_state_image_info_t *const info) noexcept {
  const uint32_t struct_size = info->struct_size;
  const uint32_t abi_version = info->abi_version;
  std::memset(info, 0, sizeof(*info));
  info->struct_size = struct_size;
  info->abi_version = abi_version;
  info->info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
  info->session_id = state->session_id;
  info->layer_id = state->layer_id;
  info->dtype = state->dtype;
  info->encoding = state->encoding;
  info->active_slot = 0U;
  info->capacity_tokens = state->capacity_tokens;
  info->published_length = state->published_length;
  info->generation = state->generation;
  info->plane_count = (state->scale_bytes_per_token != 0U ? 4U : 2U) +
                      (state->outer_scale_bytes_per_token != 0U ? 2U : 0U);
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

template <typename T> void set_fork_info_header(T *const info) noexcept {
  /* No-op helper used to keep caller-owned struct_size/ABI fields intact. */
  (void)info;
}

struct KvPlaneRef final {
  KvVmmPlane *plane;
  uint64_t logical_bytes;
};

KvPlaneRef kv_plane(KvState *const state, const uint32_t id) noexcept {
  switch (id) {
  case SLLM_HIP_KV_STATE_PLANE_KEY:
    return {&state->key_plane,
            state->capacity_tokens * state->value_bytes_per_token};
  case SLLM_HIP_KV_STATE_PLANE_VALUE:
    return {&state->value_plane,
            state->capacity_tokens * state->value_bytes_per_token};
  case SLLM_HIP_KV_STATE_PLANE_KEY_SCALE:
    return {&state->key_scale_plane,
            state->capacity_tokens * state->scale_bytes_per_token};
  case SLLM_HIP_KV_STATE_PLANE_VALUE_SCALE:
    return {&state->value_scale_plane,
            state->capacity_tokens * state->scale_bytes_per_token};
  case SLLM_HIP_KV_STATE_PLANE_KEY_OUTER_SCALE:
    return {&state->key_outer_scale_plane,
            state->capacity_tokens * state->outer_scale_bytes_per_token};
  case SLLM_HIP_KV_STATE_PLANE_VALUE_OUTER_SCALE:
    return {&state->value_outer_scale_plane,
            state->capacity_tokens * state->outer_scale_bytes_per_token};
  default:
    return {nullptr, 0U};
  }
}

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

bool kv_quiescent(const KvState *const state) noexcept {
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

uint64_t kv_mapped_bytes(const KvState *const state) noexcept {
  return state->key_plane.mapped_bytes + state->value_plane.mapped_bytes +
         state->key_scale_plane.mapped_bytes +
         state->value_scale_plane.mapped_bytes +
         state->key_outer_scale_plane.mapped_bytes +
         state->value_outer_scale_plane.mapped_bytes;
}

uint64_t kv_visible_bytes(const KvState *const state) noexcept {
  const uint64_t value = state->published_length * state->value_bytes_per_token;
  const uint64_t scale = state->published_length * state->scale_bytes_per_token;
  const uint64_t outer =
      state->published_length * state->outer_scale_bytes_per_token;
  return value * 2U + scale * 2U + outer * 2U;
}

uint64_t kv_visible_shared_bytes(const KvState *const state) noexcept {
  const auto visible_plane = [state](const KvVmmPlane &plane,
                                     const uint64_t bytes_per_token) {
    if (bytes_per_token == 0U || plane.page_bytes == 0U) {
      return uint64_t{0};
    }
    const uint64_t visible = state->published_length * bytes_per_token;
    const uint64_t pages = (visible + plane.page_bytes - 1U) / plane.page_bytes;
    return std::min(plane.mapped_bytes, pages * plane.page_bytes);
  };
  return visible_plane(state->key_plane, state->value_bytes_per_token) +
         visible_plane(state->value_plane, state->value_bytes_per_token) +
         visible_plane(state->key_scale_plane, state->scale_bytes_per_token) +
         visible_plane(state->value_scale_plane, state->scale_bytes_per_token) +
         visible_plane(state->key_outer_scale_plane,
                       state->outer_scale_bytes_per_token) +
         visible_plane(state->value_outer_scale_plane,
                       state->outer_scale_bytes_per_token);
}

hipError_t clone_contiguous_plane(const KvVmmPlane &source,
                                  KvVmmPlane *const child,
                                  const uint64_t destination_logical_bytes,
                                  const uint64_t visible_bytes) noexcept {
  const hipError_t reserve_status =
      child->reserve_contiguous(destination_logical_bytes, source.page_bytes);
  if (reserve_status != hipSuccess) {
    return reserve_status;
  }
  const uint64_t zero_bytes = destination_logical_bytes;
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
  std::memset(child->address, 0, static_cast<std::size_t>(zero_bytes));
#else
  const hipError_t zero_status =
      hipMemset(child->address, 0, static_cast<std::size_t>(zero_bytes));
  if (zero_status != hipSuccess) {
    child->release_noexcept();
    return zero_status;
  }
#endif
  if (source.mapped_bytes == 0U || visible_bytes == 0U) {
    return hipSuccess;
  }
  const std::size_t bytes = static_cast<std::size_t>(std::min(
      visible_bytes, std::min(source.mapped_bytes, destination_logical_bytes)));
  return state_copy(child->address, source.address, bytes,
                    hipMemcpyDeviceToDevice);
}

hipError_t clone_kv_plane(const KvVmmPlane &source, KvVmmPlane *const child,
                          const uint64_t destination_logical_bytes,
                          const uint64_t visible_bytes, const int device_index,
                          const bool shared) noexcept {
  if (destination_logical_bytes == 0U) {
    return hipSuccess;
  }
  if (source.contiguous) {
    return clone_contiguous_plane(source, child, destination_logical_bytes,
                                  visible_bytes);
  }
  if (!shared) {
    return hipErrorInvalidValue;
  }
  const hipError_t status = child->clone_shared_from(
      source, destination_logical_bytes, visible_bytes, device_index);
  if (status != hipSuccess) {
    return status;
  }
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
  if (visible_bytes != 0U) {
    std::memcpy(
        child->address, source.address,
        static_cast<std::size_t>(std::min(visible_bytes, source.mapped_bytes)));
  }
#endif
  return hipSuccess;
}

hipError_t
release_candidate_planes(std::array<KvVmmPlane *, 6> planes) noexcept {
  hipError_t first = hipSuccess;
  for (KvVmmPlane *const plane : planes) {
    if (plane == nullptr) {
      continue;
    }
    const hipError_t status = plane->release_checked();
    if (first == hipSuccess && status != hipSuccess) {
      first = status;
    }
  }
  return first;
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
    if (raw_child == nullptr || raw_source == nullptr ||
        destination_info == nullptr || fork_info == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "KV state fork source, child, or info is null");
    }
    const sllm_status_t destination_status =
        sllm_kv_state::validate_state_create_info_v2(destination_info,
                                                     error_sink);
    if (destination_status != SLLM_STATUS_OK) {
      return destination_status;
    }
    if (fork_info->struct_size != sizeof(*fork_info) ||
        fork_info->abi_version != SLLM_HIP_ABI_VERSION) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "KV state fork info header is invalid");
    }
    KvState *source = nullptr;
    Context *context = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      source = lookup<KvState>(raw_source, HandleKind::KvState);
      if (source == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "KV state fork source handle is stale or wrong kind");
      }
      context = source->context;
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (context->poisoned.load() || !kv_quiescent(source)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "KV state fork requires a quiescent source state");
      }
      if (destination_info->session_id != source->session_id ||
          destination_info->layer_id != source->layer_id ||
          destination_info->head_count != source->head_count ||
          destination_info->head_dim != source->head_dim ||
          destination_info->layout != SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR ||
          destination_info->encoding != source->encoding ||
          destination_info->dtype != source->dtype ||
          destination_info->block_size !=
              (source->encoding == SLLM_HIP_KV_ENCODING_NVFP4_V1 ? 16U : 0U) ||
          destination_info->scale_dtype !=
              (source->encoding == SLLM_HIP_KV_ENCODING_FP8_V1 ||
                       source->encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1
                   ? SLLM_TENSOR_DTYPE_F32
               : source->encoding == SLLM_HIP_KV_ENCODING_NVFP4_V1
                   ? SLLM_TENSOR_DTYPE_F8_E4M3_FN
                   : 0U)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INVALID_ARGUMENT,
            "KV fork destination changes layer, layout, dtype, or encoding");
      }
      if (source->encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1) {
        float destination_key = 0.0F;
        float destination_value = 0.0F;
        std::memcpy(&destination_key, &destination_info->reserved[0],
                    sizeof(float));
        std::memcpy(&destination_value, &destination_info->reserved[1],
                    sizeof(float));
        if (destination_key != source->static_key_scale ||
            destination_value != source->static_value_scale) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_INVALID_ARGUMENT,
              "KV fork destination changes static FP8 scales");
        }
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
    const bool shared = !source->key_plane.contiguous;
    KvVmmPlane key_plane;
    KvVmmPlane value_plane;
    KvVmmPlane key_scale_plane;
    KvVmmPlane value_scale_plane;
    KvVmmPlane key_outer_scale_plane;
    KvVmmPlane value_outer_scale_plane;
    const int device_index = static_cast<int>(context->device_index);
    const std::array<std::pair<const KvVmmPlane *, KvVmmPlane *>, 6> planes = {
        std::pair{&source->key_plane, &key_plane},
        std::pair{&source->value_plane, &value_plane},
        std::pair{&source->key_scale_plane, &key_scale_plane},
        std::pair{&source->value_scale_plane, &value_scale_plane},
        std::pair{&source->key_outer_scale_plane, &key_outer_scale_plane},
        std::pair{&source->value_outer_scale_plane, &value_outer_scale_plane}};
    if (shared) {
      for (const auto &pair : planes) {
        const uint64_t visible_bytes =
            source->published_length *
            (pair.first == &source->key_scale_plane ||
                     pair.first == &source->value_scale_plane
                 ? source->scale_bytes_per_token
             : pair.first == &source->key_outer_scale_plane ||
                     pair.first == &source->value_outer_scale_plane
                 ? source->outer_scale_bytes_per_token
                 : source->value_bytes_per_token);
        const hipError_t ro_status =
            const_cast<KvVmmPlane *>(pair.first)
                ->mark_read_only(visible_bytes, device_index);
        if (ro_status != hipSuccess) {
          /* A previous plane may already be protected.  Keeping the source
           * append gate closed until the caller retries is safer than
           * exposing a partially shared state. */
          unguard_source();
          return hip_failure(error_sink, ro_status,
                             "protect all source KV VMM planes read-only");
        }
      }
    }
    hipError_t clone_status = hipSuccess;
    for (const auto &pair : planes) {
      const uint64_t bytes_per_token =
          pair.first == &source->key_scale_plane ||
                  pair.first == &source->value_scale_plane
              ? source->scale_bytes_per_token
          : pair.first == &source->key_outer_scale_plane ||
                  pair.first == &source->value_outer_scale_plane
              ? source->outer_scale_bytes_per_token
              : source->value_bytes_per_token;
      const uint64_t destination_logical =
          destination_info->capacity_tokens * bytes_per_token;
      const uint64_t visible_bytes = source->published_length * bytes_per_token;
      clone_status =
          clone_kv_plane(*pair.first, pair.second, destination_logical,
                         visible_bytes, device_index, shared);
      if (clone_status != hipSuccess) {
        break;
      }
    }
    if (clone_status != hipSuccess) {
      (void)release_candidate_planes(
          {&key_plane, &value_plane, &key_scale_plane, &value_scale_plane,
           &key_outer_scale_plane, &value_outer_scale_plane});
      unguard_source();
      return hip_failure(error_sink, clone_status, "fork KV state planes");
    }
    if (shared) {
      source->shared_page_count =
          source->key_plane.shared_page_count() +
          source->value_plane.shared_page_count() +
          source->key_scale_plane.shared_page_count() +
          source->value_scale_plane.shared_page_count() +
          source->key_outer_scale_plane.shared_page_count() +
          source->value_outer_scale_plane.shared_page_count();
    }
    std::unique_ptr<Buffer> key_buffer;
    std::unique_ptr<Buffer> value_buffer;
    std::unique_ptr<KvState> candidate;
    try {
      const uint64_t destination_value_bytes =
          destination_info->capacity_tokens * source->value_bytes_per_token;
      key_buffer = std::make_unique<Buffer>(context, key_plane.address,
                                            destination_value_bytes);
      value_buffer = std::make_unique<Buffer>(context, value_plane.address,
                                              destination_value_bytes);
      candidate = std::make_unique<KvState>(
          context, source->session_id, source->layer_id,
          destination_info->capacity_tokens, source->head_count,
          source->head_dim, key_buffer.get(), value_buffer.get(),
          std::move(key_plane), std::move(value_plane),
          std::move(key_scale_plane), std::move(value_scale_plane),
          std::move(key_outer_scale_plane), std::move(value_outer_scale_plane),
          source->dtype, source->encoding, source->static_key_scale,
          source->static_value_scale, source->value_bytes_per_token,
          source->scale_bytes_per_token, source->outer_scale_bytes_per_token,
          source->physical_page_bytes, source->context_identity);
      candidate->published_length = source->published_length;
      candidate->generation = source->generation;
      candidate->last_published_start = source->last_published_start;
      candidate->last_published_end = source->last_published_end;
      candidate->last_published_generation = source->last_published_generation;
      candidate->shared_page_count = shared ? source->shared_page_count : 0U;
      candidate->cow_copied_bytes = 0U;
    } catch (...) {
      (void)release_candidate_planes(
          {&key_plane, &value_plane, &key_scale_plane, &value_scale_plane,
           &key_outer_scale_plane, &value_outer_scale_plane});
      unguard_source();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "KV fork state allocation failed");
    }
    bool accounting_ok = false;
    {
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      accounting_ok = sllm_public_runtime::AccountingState::reserve_kv_state(
          context->accounting, key_buffer->accounting,
          value_buffer->accounting);
    }
    if (!accounting_ok) {
      (void)release_candidate_planes(
          {&candidate->key_plane, &candidate->value_plane,
           &candidate->key_scale_plane, &candidate->value_scale_plane,
           &candidate->key_outer_scale_plane,
           &candidate->value_outer_scale_plane});
      unguard_source();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "KV fork accounting reservation is exhausted");
    }
    uintptr_t token = 0U;
    try {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      token = register_handle(candidate.get(), HandleKind::KvState);
    } catch (...) {
      {
        std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
        (void)sllm_public_runtime::AccountingState::release_kv_state(
            context->accounting, key_buffer->accounting,
            value_buffer->accounting);
      }
      unguard_source();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "KV fork state registry allocation failed");
    }
    if (token == 0U) {
      {
        std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
        (void)sllm_public_runtime::AccountingState::release_kv_state(
            context->accounting, key_buffer->accounting,
            value_buffer->accounting);
      }
      unguard_source();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "KV fork state token allocation failed");
    }
    candidate->state_identity = token;
    *raw_child = reinterpret_cast<sllm_kv_state_t *>(token);
    const uint64_t owned = kv_mapped_bytes(source);
    const uint64_t visible = kv_visible_bytes(source);
    const uint64_t shared_visible = kv_visible_shared_bytes(source);
    const uint64_t destination_owned =
        destination_info->capacity_tokens *
        (source->value_bytes_per_token * 2U +
         source->scale_bytes_per_token * 2U +
         source->outer_scale_bytes_per_token * 2U);
    fill_fork_info(fork_info,
                   shared ? SLLM_HIP_STATE_FORK_MODE_SHARED_READ_ONLY_PAGES
                          : SLLM_HIP_STATE_FORK_MODE_DEVICE_COPY,
                   source->state_identity, token, owned,
                   shared ? 0U : destination_owned, shared ? 0U : visible,
                   shared ? shared_visible : 0U, source->published_length,
                   source->physical_page_bytes);
    key_buffer.release();
    value_buffer.release();
    (void)candidate.release();
    unguard_source();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in KV state fork");
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
    if (fork_info == nullptr || fork_info->struct_size != sizeof(*fork_info) ||
        fork_info->abi_version != SLLM_HIP_ABI_VERSION) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "KV fork query info header is invalid");
    }
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    KvState *const state = lookup<KvState>(raw_state, HandleKind::KvState);
    if (state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "KV fork query state handle is stale or wrong kind");
    }
    std::lock_guard<std::mutex> lock(state->context->accounting_mutex);
    if (state->context->poisoned.load() || !kv_quiescent(state)) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "KV fork query requires a quiescent state");
    }
    const uint64_t shared_bytes =
        state->shared_page_count * state->physical_page_bytes;
    fill_fork_info(fork_info, state->fork_mode, state->state_identity,
                   state->state_identity, kv_mapped_bytes(state),
                   state->cow_copied_bytes, state->cow_copied_bytes,
                   shared_bytes, state->published_length,
                   state->physical_page_bytes);
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in KV fork query");
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
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    KvState *const state = lookup<KvState>(raw_state, HandleKind::KvState);
    if (state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "KV export state handle is stale or wrong kind");
    }
    std::lock_guard<std::mutex> lock(state->context->accounting_mutex);
    if (!kv_quiescent(state) || state->context->poisoned.load()) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "KV export requires a quiescent state");
    }
    const KvPlaneRef ref = kv_plane(state, chunk->plane);
    if (ref.plane == nullptr || chunk->byte_offset > ref.logical_bytes ||
        chunk->byte_length > ref.logical_bytes - chunk->byte_offset ||
        chunk->byte_offset + chunk->byte_length > ref.plane->mapped_bytes) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
          "KV export range is outside the exact native plane");
    }
    const sllm_status_t device_status =
        select_context_device(state->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      return device_status;
    }
    const hipError_t status = state_copy(
        chunk->host_pointer,
        static_cast<const char *>(ref.plane->address) +
            static_cast<std::size_t>(chunk->byte_offset),
        static_cast<std::size_t>(chunk->byte_length), hipMemcpyDeviceToHost);
    return status == hipSuccess
               ? SLLM_STATUS_OK
               : hip_failure(error_sink, status, "export KV raw plane chunk");
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in KV state export");
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
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    KvState *const state = lookup<KvState>(raw_state, HandleKind::KvState);
    if (state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "KV import state handle is stale or wrong kind");
    }
    std::lock_guard<std::mutex> lock(state->context->accounting_mutex);
    if (!kv_quiescent(state) || state->context->poisoned.load()) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "KV import requires a quiescent state");
    }
    const KvPlaneRef ref = kv_plane(state, chunk->plane);
    if (ref.plane == nullptr || chunk->byte_offset > ref.logical_bytes ||
        chunk->byte_length > ref.logical_bytes - chunk->byte_offset ||
        chunk->byte_offset + chunk->byte_length > ref.plane->mapped_bytes) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
          "KV import range is outside the exact native plane");
    }
    const sllm_status_t device_status =
        select_context_device(state->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      return device_status;
    }
    /* Import into a shared child is a write transition, so privatize the
     * affected pages before the raw byte copy. */
    hipMemAllocationProp properties{};
    properties.type = hipMemAllocationTypePinned;
    properties.location.type = hipMemLocationTypeDevice;
    properties.location.id = static_cast<int>(state->context->device_index);
    hipMemAccessDesc access{};
    access.location = properties.location;
    access.flags = hipMemAccessFlagsProtReadWrite;
    const uint64_t shared_before =
        state->key_plane.shared_page_count() +
        state->value_plane.shared_page_count() +
        state->key_scale_plane.shared_page_count() +
        state->value_scale_plane.shared_page_count() +
        state->key_outer_scale_plane.shared_page_count() +
        state->value_outer_scale_plane.shared_page_count();
    if (ref.plane->make_private(chunk->byte_offset,
                                chunk->byte_offset + chunk->byte_length,
                                properties, access) != hipSuccess) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
          "KV import failed to privatize shared pages");
    }
    const uint64_t shared_after =
        state->key_plane.shared_page_count() +
        state->value_plane.shared_page_count() +
        state->key_scale_plane.shared_page_count() +
        state->value_scale_plane.shared_page_count() +
        state->key_outer_scale_plane.shared_page_count() +
        state->value_outer_scale_plane.shared_page_count();
    state->shared_page_count = shared_after;
    state->cow_copied_bytes +=
        (shared_before - shared_after) * state->physical_page_bytes;
    const hipError_t status = state_copy(
        static_cast<char *>(ref.plane->address) +
            static_cast<std::size_t>(chunk->byte_offset),
        chunk->host_pointer, static_cast<std::size_t>(chunk->byte_length),
        hipMemcpyHostToDevice);
    if (status == hipSuccess) {
      state->import_plane_mask |= UINT32_C(1) << (chunk->plane - 1U);
      return SLLM_STATUS_OK;
    }
    return hip_failure(error_sink, status, "import KV raw plane chunk");
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in KV state import");
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
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    KvState *const state = lookup<KvState>(raw_state, HandleKind::KvState);
    if (state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "KV image query state handle is stale or wrong kind");
    }
    std::lock_guard<std::mutex> lock(state->context->accounting_mutex);
    if (!kv_quiescent(state) || state->context->poisoned.load()) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "KV image query requires a quiescent state");
    }
    fill_kv_image_info(state, image_info);
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in KV state image query");
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
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    KvState *const state = lookup<KvState>(raw_state, HandleKind::KvState);
    if (state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "KV image finalize state handle is stale or wrong kind");
    }
    std::lock_guard<std::mutex> lock(state->context->accounting_mutex);
    const uint32_t expected_mask =
        (state->scale_bytes_per_token != 0U ? UINT32_C(0x0f) : UINT32_C(0x03)) |
        (state->outer_scale_bytes_per_token != 0U ? UINT32_C(0x30) : 0U);
    const uint32_t expected_count =
        (state->scale_bytes_per_token != 0U ? 4U : 2U) +
        (state->outer_scale_bytes_per_token != 0U ? 2U : 0U);
    if (!kv_quiescent(state) || state->context->poisoned.load() ||
        image_info->session_id != state->session_id ||
        image_info->layer_id != state->layer_id ||
        image_info->dtype != state->dtype ||
        image_info->encoding != state->encoding ||
        image_info->capacity_tokens > state->capacity_tokens ||
        image_info->published_length > state->capacity_tokens ||
        (state->mapped_token_capacity < image_info->published_length) ||
        image_info->plane_count != expected_count ||
        (state->import_plane_mask & expected_mask) != expected_mask) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "KV image metadata or imported plane set is incomplete");
    }
    state->published_length = image_info->published_length;
    state->generation = image_info->generation;
    state->last_published_start = 0U;
    state->last_published_end = image_info->published_length;
    state->last_published_generation = image_info->generation;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in KV state import finalize");
  }
}

extern "C" sllm_status_t
sllm_kv_state_image_plane_size(const sllm_kv_state_t *const raw_state,
                               const uint32_t plane, uint64_t *const size_bytes,
                               sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK)
      return sink_status;
    if (size_bytes == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "KV image plane size output is null");
    }
    *size_bytes = 0U;
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    KvState *const state = lookup<KvState>(raw_state, HandleKind::KvState);
    if (state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "KV image plane size state handle is stale or wrong kind");
    }
    std::lock_guard<std::mutex> lock(state->context->accounting_mutex);
    if (!kv_quiescent(state) || state->context->poisoned.load()) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "KV image plane size requires a quiescent state");
    }
    const KvPlaneRef ref = kv_plane(state, plane);
    if (ref.plane == nullptr || ref.logical_bytes == 0U) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "KV image plane is unknown or empty");
    }
    *size_bytes = ref.logical_bytes;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in KV image plane size");
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
