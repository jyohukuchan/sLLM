namespace {

using PqSupportRecord = sllm_token_selector_pq::SupportRecordV1;
using PqDecisionRecord = sllm_token_selector_pq::DecisionRecordV1;
using PqDraftIds = sllm_token_selector_pq::DraftIdsV1;

constexpr uint64_t kPqSupportBytes = sizeof(PqSupportRecord);
constexpr uint64_t kPqDecisionBytes = sizeof(PqDecisionRecord);

bool pq_ranges_overlap(const Buffer *const left_buffer,
                       const uint64_t left_offset, const uint64_t left_bytes,
                       const Buffer *const right_buffer,
                       const uint64_t right_offset,
                       const uint64_t right_bytes) noexcept {
  if (left_buffer != right_buffer) {
    return false;
  }
  return left_offset < right_offset + right_bytes &&
         right_offset < left_offset + left_bytes;
}

sllm_status_t validate_pq_buffer_range(const Buffer *const buffer,
                                       const uint64_t offset,
                                       const uint64_t bytes,
                                       const uint64_t alignment,
                                       sllm_error_sink_t *const error_sink,
                                       const char *const name) noexcept {
  if (buffer == nullptr || buffer->device_pointer == nullptr) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
        "p/q buffer handle is stale or has no device allocation");
  }
  if (sllm_public_runtime::add_overflows(offset, bytes) ||
      offset + bytes > buffer->size_bytes) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
        name == nullptr ? "p/q buffer range is outside its backing buffer"
                        : name);
  }
  const uintptr_t address = reinterpret_cast<uintptr_t>(buffer->device_pointer);
  if (sllm_public_runtime::add_overflows(static_cast<uint64_t>(address),
                                         offset) ||
      ((address + static_cast<uintptr_t>(offset)) %
       static_cast<uintptr_t>(alignment)) != 0U) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_MISALIGNED_OFFSET,
        name == nullptr ? "p/q buffer range is misaligned" : name);
  }
  return SLLM_STATUS_OK;
}

void pq_resource_multiplicities(const std::array<Buffer *, 3U> &buffers,
                                uint64_t (&multiplicities)[3U]) noexcept {
  multiplicities[0] = 0U;
  multiplicities[1] = 0U;
  multiplicities[2] = 0U;
  for (std::size_t index = 0U; index != buffers.size(); ++index) {
    std::size_t first = index;
    for (std::size_t prior = 0U; prior != index; ++prior) {
      if (buffers[prior] == buffers[index]) {
        first = prior;
        break;
      }
    }
    ++multiplicities[first];
  }
}

bool reserve_pq_submission_locked(
    Context *const context, Queue *const queue,
    const std::array<Buffer *, 3U> &buffers) noexcept {
  uint64_t multiplicities[3U] = {};
  pq_resource_multiplicities(buffers, multiplicities);
  if (!sllm_public_runtime::AccountingState::can_increment(
          queue->accounting.active_submissions) ||
      !sllm_public_runtime::AccountingState::can_increment(
          queue->accounting.completion_references) ||
      !sllm_public_runtime::AccountingState::can_increment(
          context->accounting.child_count) ||
      !sllm_public_runtime::AccountingState::can_increment(
          context->accounting.lifetime_guards)) {
    return false;
  }
  for (std::size_t index = 0U; index != buffers.size(); ++index) {
    if (multiplicities[index] != 0U &&
        (multiplicities[index] >
             std::numeric_limits<uint64_t>::max() -
                 buffers[index]->accounting.active_submissions ||
         multiplicities[index] >
             std::numeric_limits<uint64_t>::max() -
                 buffers[index]->accounting.completion_references)) {
      return false;
    }
  }
  ++queue->accounting.active_submissions;
  ++queue->accounting.completion_references;
  ++context->accounting.child_count;
  ++context->accounting.lifetime_guards;
  for (std::size_t index = 0U; index != buffers.size(); ++index) {
    if (multiplicities[index] != 0U) {
      buffers[index]->accounting.active_submissions += multiplicities[index];
      buffers[index]->accounting.completion_references += multiplicities[index];
    }
  }
  return true;
}

bool release_pq_submission_locked(
    Context *const context, Queue *const queue,
    const std::array<Buffer *, 3U> &buffers) noexcept {
  uint64_t multiplicities[3U] = {};
  pq_resource_multiplicities(buffers, multiplicities);
  if (queue->accounting.active_submissions == 0U ||
      queue->accounting.completion_references == 0U ||
      context->accounting.child_count == 0U ||
      context->accounting.lifetime_guards == 0U) {
    return false;
  }
  for (std::size_t index = 0U; index != buffers.size(); ++index) {
    if (multiplicities[index] != 0U &&
        (buffers[index]->accounting.active_submissions <
             multiplicities[index] ||
         buffers[index]->accounting.completion_references <
             multiplicities[index])) {
      return false;
    }
  }
  --queue->accounting.active_submissions;
  --queue->accounting.completion_references;
  --context->accounting.child_count;
  --context->accounting.lifetime_guards;
  for (std::size_t index = 0U; index != buffers.size(); ++index) {
    if (multiplicities[index] != 0U) {
      buffers[index]->accounting.active_submissions -= multiplicities[index];
      buffers[index]->accounting.completion_references -= multiplicities[index];
    }
  }
  return true;
}

} // namespace

extern "C" sllm_status_t sllm_token_selector_verify_fixed_k20_mtp_v1(
    const sllm_context_t *const raw_context,
    const sllm_queue_t *const raw_queue,
    const sllm_buffer_t *const raw_target_buffer, const uint64_t target_offset,
    const uint32_t target_rows, const sllm_buffer_t *const raw_draft_buffer,
    const uint64_t draft_offset, const uint32_t draft_rows,
    const PqDraftIds draft_ids, const uint32_t draft_id_count,
    const uint32_t width, const uint64_t seed, const uint64_t absolute_position,
    const sllm_buffer_t *const raw_decision_buffer,
    const uint64_t decision_offset,
    sllm_error_sink_t *const error_sink) noexcept {
  Context *context = nullptr;
  bool reserved = false;
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_context == nullptr || raw_queue == nullptr ||
        raw_target_buffer == nullptr || raw_draft_buffer == nullptr ||
        raw_decision_buffer == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "p/q context, queue, support, or decision handle is null");
    }
    if (width == 0U || width > sllm_token_selector_pq::kMaxWidth ||
        target_rows != width + 1U || draft_rows != width ||
        draft_id_count != width) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "p/q width, row count, or draft id count is invalid");
    }
    // The validated width bounds these products to nine support records.
    const uint64_t target_bytes =
        static_cast<uint64_t>(target_rows) * kPqSupportBytes;
    const uint64_t draft_bytes =
        static_cast<uint64_t>(draft_rows) * kPqSupportBytes;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      context = lookup<Context>(raw_context, HandleKind::Context);
      Queue *const queue = lookup<Queue>(raw_queue, HandleKind::Queue);
      Buffer *const target =
          lookup<Buffer>(raw_target_buffer, HandleKind::Buffer);
      Buffer *const draft =
          lookup<Buffer>(raw_draft_buffer, HandleKind::Buffer);
      Buffer *const decision =
          lookup<Buffer>(raw_decision_buffer, HandleKind::Buffer);
      if (context == nullptr || queue == nullptr || target == nullptr ||
          draft == nullptr || decision == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "p/q handle is stale or has the wrong kind");
      }
      if (queue->context != context || target->context != context ||
          draft->context != context || decision->context != context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH,
            "p/q queue and buffers must belong to the supplied context");
      }
      if (queue->release_active || target->release_active ||
          draft->release_active || decision->release_active ||
          context->release_active || context->poisoned.load()) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "p/q context, queue, or buffer is releasing or poisoned");
      }
      const sllm_status_t target_status = validate_pq_buffer_range(
          target, target_offset, target_bytes, alignof(PqSupportRecord),
          error_sink, "p/q target support range is invalid");
      if (target_status != SLLM_STATUS_OK) {
        return target_status;
      }
      const sllm_status_t draft_status = validate_pq_buffer_range(
          draft, draft_offset, draft_bytes, alignof(PqSupportRecord),
          error_sink, "p/q draft support range is invalid");
      if (draft_status != SLLM_STATUS_OK) {
        return draft_status;
      }
      const sllm_status_t decision_status =
          validate_pq_buffer_range(decision, decision_offset, kPqDecisionBytes,
                                   alignof(PqDecisionRecord), error_sink,
                                   "p/q decision range is invalid");
      if (decision_status != SLLM_STATUS_OK) {
        return decision_status;
      }
      if (pq_ranges_overlap(target, target_offset, target_bytes, draft,
                            draft_offset, draft_bytes) ||
          pq_ranges_overlap(target, target_offset, target_bytes, decision,
                            decision_offset, kPqDecisionBytes) ||
          pq_ranges_overlap(draft, draft_offset, draft_bytes, decision,
                            decision_offset, kPqDecisionBytes)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_ALIAS_OVERLAP,
            "p/q support and decision ranges overlap");
      }
      const std::array<Buffer *, 3U> buffers = {target, draft, decision};
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (context->poisoned.load() || context->release_active ||
          queue->release_active || target->release_active ||
          draft->release_active || decision->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "p/q owner became busy before reservation");
      }
      for (Buffer *const buffer : buffers) {
        if (buffer->accounting.active_submissions != 0U) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_PUBLIC_BUSY,
              "p/q buffer has an active submission");
        }
      }
      if (!reserve_pq_submission_locked(context, queue, buffers)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "p/q submission accounting is exhausted");
      }
      reserved = true;
      const sllm_status_t device_status =
          select_context_device(context, error_sink);
      if (device_status != SLLM_STATUS_OK) {
        if (!release_pq_submission_locked(context, queue, buffers)) {
          poison_context_locked(context);
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_INTERNAL_ERROR,
              "p/q accounting rollback failed after device selection");
        }
        reserved = false;
        return device_status;
      }
      auto byte_pointer = [](Buffer *const buffer,
                             const uint64_t offset) -> void * {
        return static_cast<char *>(buffer->device_pointer) +
               static_cast<std::size_t>(offset);
      };
      void *const decision_pointer = byte_pointer(decision, decision_offset);
      // run_sparse_pq clears the complete 144-byte record before any input or
      // RNG validation, and every subsequent error path clears it again. There
      // is therefore no separate memset submission that could race or fail.
      hipError_t operation_status = ::sllm_token_selector_pq_kernel::launch(
          static_cast<const PqSupportRecord *>(
              byte_pointer(target, target_offset)),
          target_rows,
          static_cast<const PqSupportRecord *>(
              byte_pointer(draft, draft_offset)),
          draft_rows, draft_ids, draft_id_count, width, seed, absolute_position,
          static_cast<PqDecisionRecord *>(decision_pointer), queue->stream);
      if (operation_status == hipSuccess) {
        operation_status = hipStreamSynchronize(queue->stream);
      }
      if (operation_status != hipSuccess) {
        // The stream may contain a submitted kernel. Keep all
        // reservations and poison while it is unresolved; buffer release
        // checks this poison before attempting hipFree.
        poison_context_locked(context);
        return hip_failure(error_sink, operation_status,
                           "p/q kernel submission or fence");
      }
      if (!release_pq_submission_locked(context, queue, buffers)) {
        poison_context_locked(context);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "p/q completion accounting failed; context poisoned");
      }
      reserved = false;
      return SLLM_STATUS_OK;
    }
  } catch (...) {
    if (reserved && context != nullptr) {
      poison_context(context);
    }
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in p/q verification");
  }
}
