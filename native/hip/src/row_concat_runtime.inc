#ifndef SLLM_ROW_CONCAT_RUNTIME_INC
#define SLLM_ROW_CONCAT_RUNTIME_INC

#include "row_concat_internal.hpp"

#include <array>
#include <limits>

namespace {

using RowConcatBuffers = std::array<Buffer *, 3U>;
using RowConcatMultiplicities = std::array<uint64_t, 3U>;

bool row_concat_multiply_overflows(const uint64_t left, const uint64_t right,
                                   uint64_t *const result) noexcept {
  if (left != 0U && right > std::numeric_limits<uint64_t>::max() / left) {
    return true;
  }
  *result = left * right;
  return false;
}

bool row_concat_ranges_overlap(const Buffer *const left_buffer,
                               const uint64_t left_offset,
                               const uint64_t left_bytes,
                               const Buffer *const right_buffer,
                               const uint64_t right_offset,
                               const uint64_t right_bytes) noexcept {
  if (left_buffer != right_buffer) {
    return false;
  }
  return left_offset < right_offset + right_bytes &&
         right_offset < left_offset + left_bytes;
}

sllm_status_t row_concat_validate_range(const Buffer *const buffer,
                                        const uint64_t offset,
                                        const uint64_t bytes,
                                        sllm_error_sink_t *const error_sink,
                                        const char *const name) noexcept {
  if (buffer == nullptr || buffer->device_pointer == nullptr) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
        "row concat buffer has no device allocation");
  }
  if ((offset & UINT64_C(1)) != 0U) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_MISALIGNED_OFFSET,
        name == nullptr ? "row concat range is not BF16 aligned" : name);
  }
  if (sllm_public_runtime::add_overflows(offset, bytes) ||
      offset + bytes > buffer->size_bytes) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
        name == nullptr ? "row concat range is outside its backing buffer"
                        : name);
  }
  const uintptr_t address = reinterpret_cast<uintptr_t>(buffer->device_pointer);
  if (sllm_public_runtime::add_overflows(static_cast<uint64_t>(address),
                                         offset) ||
      ((address + static_cast<uintptr_t>(offset)) &
       static_cast<uintptr_t>(1U)) != 0U) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_MISALIGNED_OFFSET,
        name == nullptr ? "row concat range address is not BF16 aligned"
                        : name);
  }
  return SLLM_STATUS_OK;
}

void row_concat_resource_multiplicities(
    const RowConcatBuffers &buffers,
    RowConcatMultiplicities &multiplicities) noexcept {
  for (std::size_t index = 0U; index != buffers.size(); ++index) {
    multiplicities[index] = 0U;
  }
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

bool row_concat_reserve_locked(Context *const context, Queue *const queue,
                               const RowConcatBuffers &buffers) noexcept {
  RowConcatMultiplicities multiplicities = {};
  row_concat_resource_multiplicities(buffers, multiplicities);
  const auto can_add = [](const uint64_t value, const uint64_t amount) {
    return amount <= std::numeric_limits<uint64_t>::max() - value;
  };
  if (!can_add(queue->accounting.active_submissions, 1U) ||
      !can_add(queue->accounting.completion_references, 1U) ||
      !can_add(context->accounting.child_count, 1U) ||
      !can_add(context->accounting.lifetime_guards, 1U)) {
    return false;
  }
  for (std::size_t index = 0U; index != buffers.size(); ++index) {
    if (multiplicities[index] != 0U &&
        (!can_add(buffers[index]->accounting.active_submissions,
                  multiplicities[index]) ||
         !can_add(buffers[index]->accounting.completion_references,
                  multiplicities[index]))) {
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

bool row_concat_release_locked(Context *const context, Queue *const queue,
                               const RowConcatBuffers &buffers) noexcept {
  RowConcatMultiplicities multiplicities = {};
  row_concat_resource_multiplicities(buffers, multiplicities);
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

extern "C" sllm_status_t sllm_concat_bf16_rows_v1(
    const sllm_context_t *const raw_context,
    const sllm_queue_t *const raw_queue,
    const sllm_buffer_t *const raw_left_buffer, const uint64_t left_offset,
    const sllm_buffer_t *const raw_right_buffer, const uint64_t right_offset,
    const sllm_buffer_t *const raw_output_buffer, const uint64_t output_offset,
    const uint64_t rows, const uint64_t left_columns,
    const uint64_t right_columns,
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
        raw_left_buffer == nullptr || raw_right_buffer == nullptr ||
        raw_output_buffer == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "row concat context, queue, or buffer handle is null");
    }
    if (rows == 0U || left_columns == 0U || right_columns == 0U) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "row concat rows and column counts must be non-zero");
    }
    uint64_t output_columns = 0U;
    uint64_t left_row_bytes = 0U;
    uint64_t right_row_bytes = 0U;
    uint64_t output_row_bytes = 0U;
    uint64_t left_bytes = 0U;
    uint64_t right_bytes = 0U;
    uint64_t output_bytes = 0U;
    if (row_concat_multiply_overflows(left_columns, 2U, &left_row_bytes) ||
        row_concat_multiply_overflows(right_columns, 2U, &right_row_bytes) ||
        sllm_public_runtime::add_overflows(left_columns, right_columns) ||
        row_concat_multiply_overflows(left_columns + right_columns, 2U,
                                      &output_row_bytes) ||
        row_concat_multiply_overflows(rows, left_row_bytes, &left_bytes) ||
        row_concat_multiply_overflows(rows, right_row_bytes, &right_bytes) ||
        row_concat_multiply_overflows(rows, output_row_bytes, &output_bytes)) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_METADATA_OVERFLOW,
          "row concat dimensions overflow u64");
    }
    output_columns = left_columns + right_columns;
    const uint64_t output_blocks =
        output_columns / sllm_row_concat_kernel::kWorkgroupSize +
        static_cast<uint64_t>(
            output_columns % sllm_row_concat_kernel::kWorkgroupSize != 0U);
    if (output_blocks == 0U ||
        output_blocks > std::numeric_limits<uint32_t>::max()) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "row concat output is too large for one launch");
    }
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      context = lookup<Context>(raw_context, HandleKind::Context);
      Queue *const queue = lookup<Queue>(raw_queue, HandleKind::Queue);
      Buffer *const left = lookup<Buffer>(raw_left_buffer, HandleKind::Buffer);
      Buffer *const right =
          lookup<Buffer>(raw_right_buffer, HandleKind::Buffer);
      Buffer *const output =
          lookup<Buffer>(raw_output_buffer, HandleKind::Buffer);
      if (context == nullptr || queue == nullptr || left == nullptr ||
          right == nullptr || output == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "row concat handle is stale or has the wrong kind");
      }
      if (queue->context != context || left->context != context ||
          right->context != context || output->context != context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH,
            "row concat queue and buffers must share one context");
      }
      if (queue->release_active || left->release_active ||
          right->release_active || output->release_active ||
          context->release_active || context->poisoned.load()) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "row concat context, queue, or buffer is releasing or poisoned");
      }
      const sllm_status_t left_range_status =
          row_concat_validate_range(left, left_offset, left_bytes, error_sink,
                                    "row concat left range is invalid");
      if (left_range_status != SLLM_STATUS_OK) {
        return left_range_status;
      }
      const sllm_status_t right_range_status = row_concat_validate_range(
          right, right_offset, right_bytes, error_sink,
          "row concat right range is invalid");
      if (right_range_status != SLLM_STATUS_OK) {
        return right_range_status;
      }
      const sllm_status_t output_range_status = row_concat_validate_range(
          output, output_offset, output_bytes, error_sink,
          "row concat output range is invalid");
      if (output_range_status != SLLM_STATUS_OK) {
        return output_range_status;
      }
      // The two inputs are read-only and may intentionally alias (for
      // example, duplicating one normalized row into both halves). Only an
      // input/output overlap is unsafe because the kernel writes each output
      // row while reading its sources.
      if (row_concat_ranges_overlap(left, left_offset, left_bytes, output,
                                    output_offset, output_bytes) ||
          row_concat_ranges_overlap(right, right_offset, right_bytes, output,
                                    output_offset, output_bytes)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_ALIAS_OVERLAP,
            "row concat source and output ranges overlap");
      }
      const RowConcatBuffers buffers = {left, right, output};
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (context->poisoned.load() || context->release_active ||
          queue->release_active || left->release_active ||
          right->release_active || output->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "row concat owner became busy before reservation");
      }
      for (Buffer *const buffer : buffers) {
        if (buffer->accounting.active_submissions != 0U) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_PUBLIC_BUSY,
              "row concat buffer has an active submission");
        }
      }
      if (!row_concat_reserve_locked(context, queue, buffers)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "row concat submission accounting is exhausted");
      }
      reserved = true;
      const sllm_status_t device_status =
          select_context_device(context, error_sink);
      if (device_status != SLLM_STATUS_OK) {
        if (!row_concat_release_locked(context, queue, buffers)) {
          poison_context_locked(context);
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_INTERNAL_ERROR,
              "row concat rollback failed after device selection");
        }
        reserved = false;
        return device_status;
      }
      const auto byte_pointer = [](Buffer *const buffer,
                                   const uint64_t offset) -> uint16_t * {
        return reinterpret_cast<uint16_t *>(
            static_cast<uint8_t *>(buffer->device_pointer) + offset);
      };
      const hipError_t launch_status = ::sllm_row_concat_kernel::launch(
          byte_pointer(left, left_offset), byte_pointer(right, right_offset),
          byte_pointer(output, output_offset), rows, left_columns,
          right_columns, queue->stream);
      if (launch_status != hipSuccess) {
        poison_context_locked(context);
        return hip_failure(error_sink, launch_status,
                           "row concat kernel launch");
      }
      const hipError_t sync_status = hipStreamSynchronize(queue->stream);
      if (sync_status != hipSuccess) {
        poison_context_locked(context);
        return hip_failure(error_sink, sync_status, "row concat kernel fence");
      }
      if (!row_concat_release_locked(context, queue, buffers)) {
        poison_context_locked(context);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "row concat accounting release failed; context poisoned");
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
        "unexpected exception in row concat");
  }
}

#endif // SLLM_ROW_CONCAT_RUNTIME_INC
