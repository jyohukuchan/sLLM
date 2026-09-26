/* Paged KV image query/export/import helpers.  The public ABI wrappers stay
 * in public_runtime.hip.cpp; these helpers require the KvState/Context
 * definitions from that translation unit and are included after them. */

#include "paged_kv_image_api.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sllm_paged_kv_image_runtime {
namespace {

using sllm_paged_kv::kInvalidBlock;

struct Topology final {
  sllm_kv_paged_image_info_t info{};
  std::vector<uint32_t> logical_table;
  std::vector<uint64_t> ring_tags;
  std::vector<uint32_t> ring_table;
  std::vector<uint32_t> physical_ids;
};

struct StagedSection final {
  std::vector<std::uint8_t> bytes;
  std::vector<std::uint8_t> seen;
};

struct PendingImport final {
  StagedSection logical_table;
  StagedSection ring_tags;
  StagedSection ring_table;
  std::array<StagedSection, 6U> planes;
  uint64_t staged_bytes = 0U;
};

// A transfer is allowed to arrive in multiple chunks and the final image
// metadata is supplied separately.  Keep an independent process-wide bound
// while waiting for that metadata so an arbitrary chunk offset cannot make a
// state retain unbounded host memory.  The finalizer still requires every
// section to match its validated image size exactly.
constexpr uint64_t kMaxPendingImportBytes =
    UINT64_C(6) * SLLM_HIP_STATE_CHUNK_MAX_BYTES;

std::mutex pending_imports_mutex;
std::unordered_map<const KvState *, PendingImport> pending_imports;

sllm_status_t invalid(sllm_error_sink_t *const sink,
                      const char *const message) noexcept {
  return sllm_public_runtime::write_error(
      sink, SLLM_STATUS_INVALID_ARGUMENT, message);
}

sllm_status_t busy(sllm_error_sink_t *const sink,
                   const char *const message) noexcept {
  return sllm_public_runtime::write_error(sink, SLLM_STATUS_PUBLIC_BUSY,
                                          message);
}

bool add_overflows(const uint64_t left, const uint64_t right,
                   uint64_t *const result) noexcept {
  if (right > UINT64_MAX - left)
    return true;
  *result = left + right;
  return false;
}

bool multiply_overflows(const uint64_t left, const uint64_t right,
                        uint64_t *const result) noexcept {
  if (left != 0U && right > UINT64_MAX / left)
    return true;
  *result = left * right;
  return false;
}

uint32_t float_bits(const float value) noexcept {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool state_quiescent(KvState *const state) noexcept {
  if (state == nullptr || !state->paged || state->context == nullptr ||
      state->context->poisoned.load() || state->context->release_active ||
      state->release_active || state->graph_pins != 0U ||
      state->transition_token != 0U || state->view_count != 0U ||
      state->accounting.active_submissions != 0U ||
      state->accounting.completion_references != 0U ||
      state->paged_device_pool == nullptr || state->paged_logical_table == nullptr ||
      state->paged_device_status == nullptr ||
      state->paged_device_pool->poisoned())
    return false;
  if (state->paged_logical_table->poll_pending() !=
      sllm_paged_kv::DeviceStatus::Ok)
    return false;
  if (state->sliding_window == 0U) {
    return state->paged_host_state != nullptr &&
           !state->paged_host_state->pending() &&
           !state->paged_host_state->graph_staged();
  }
  return state->paged_sliding_host_state != nullptr &&
         !state->paged_sliding_host_state->pending();
}

uint32_t compact_id(const std::vector<uint32_t> &physical_ids,
                    const uint32_t physical) noexcept {
  const auto found =
      std::find(physical_ids.begin(), physical_ids.end(), physical);
  return found == physical_ids.end()
             ? kInvalidBlock
             : static_cast<uint32_t>(found - physical_ids.begin());
}

bool append_physical(std::vector<uint32_t> &physical_ids,
                     const uint32_t physical) {
  if (physical == kInvalidBlock)
    return true;
  if (compact_id(physical_ids, physical) != kInvalidBlock)
    return true;
  physical_ids.push_back(physical);
  return true;
}

sllm_status_t collect_topology_locked(KvState *const state,
                                      Topology *const topology,
                                      sllm_error_sink_t *const sink) noexcept {
  if (!state_quiescent(state))
    return busy(sink, "paged KV image requires a quiescent state");
  try {
    const auto &pool = *state->paged_device_pool;
    const auto &plan = pool.plan();
    const bool sliding = state->sliding_window != 0U;
    topology->info.struct_size = sizeof(topology->info);
    topology->info.abi_version = SLLM_HIP_ABI_VERSION;
    topology->info.image_version = SLLM_HIP_KV_PAGED_IMAGE_VERSION;
    topology->info.flags = sliding ? SLLM_HIP_KV_PAGED_IMAGE_FLAG_SLIDING : 0U;
    if (state->encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1)
      topology->info.flags |= SLLM_HIP_KV_PAGED_IMAGE_FLAG_STATIC_SCALES;
    topology->info.byte_order = SLLM_HIP_KV_PAGED_IMAGE_ENDIAN_LITTLE;
    topology->info.session_id = state->session_id;
    topology->info.layer_id = state->layer_id;
    topology->info.dtype = state->dtype;
    topology->info.encoding = state->encoding;
    topology->info.head_count = state->head_count;
    topology->info.head_dim = state->head_dim;
    topology->info.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
    topology->info.token_block_size = state->paged_token_block_size;
    topology->info.physical_layout_version = state->paged_physical_layout_version;
    topology->info.capacity_tokens = state->capacity_tokens;
    topology->info.published_length = state->published_length;
    topology->info.generation = state->generation;
    topology->info.retained_start = sliding ? state->retained_start : 0U;
    topology->info.retained_length =
        state->published_length - topology->info.retained_start;
    topology->info.sliding_window_tokens = state->sliding_window;
    topology->info.logical_table_capacity = state->paged_logical_table_capacity;
    topology->info.table_entry_width =
        SLLM_HIP_KV_PAGED_IMAGE_TABLE_ENTRY_U32;
    topology->info.ring_slot_count = sliding ? sllm_paged_kv::kSlidingRingSlots
                                             : 0U;
    topology->info.plane_count = 0U;
    for (const bool present : plan.present())
      topology->info.plane_count += present ? 1U : 0U;
    if (state->encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1) {
      topology->info.static_key_scale_bits =
          float_bits(state->static_key_scale);
      topology->info.static_value_scale_bits =
          float_bits(state->static_value_scale);
    } else {
      // Non-static recipes must carry canonical zero scale bits.  The state
      // constructor initializes its runtime defaults to 1.0, but those are
      // not part of a dynamic, NVFP4, or MXFP8 image recipe.
      topology->info.static_key_scale_bits = 0U;
      topology->info.static_value_scale_bits = 0U;
    }
    for (std::size_t plane = 0U; plane != 6U; ++plane)
      topology->info.plane_block_stride[plane] =
          plan.plane_bytes_per_block()[plane];

    if (sliding) {
      topology->ring_tags.resize(sllm_paged_kv::kSlidingRingSlots,
                                  UINT64_MAX);
      topology->ring_table.resize(sllm_paged_kv::kSlidingRingSlots,
                                  kInvalidBlock);
      for (uint32_t slot = 0U; slot != sllm_paged_kv::kSlidingRingSlots;
           ++slot) {
        const uint64_t tag = state->paged_sliding_host_state->absolute_block_tag(slot);
        const uint32_t physical =
            state->paged_sliding_host_state->physical_block_at_slot(slot);
        topology->ring_tags[slot] = tag;
        if (tag == UINT64_MAX)
          continue;
        if (physical == kInvalidBlock ||
            !state->paged_device_pool->descriptor_ready(physical))
          return invalid(sink, "paged sliding image has an invalid ring block");
        append_physical(topology->physical_ids, physical);
      }
      for (uint32_t slot = 0U; slot != sllm_paged_kv::kSlidingRingSlots;
           ++slot) {
        if (topology->ring_tags[slot] != UINT64_MAX)
          topology->ring_table[slot] = compact_id(
              topology->physical_ids,
              state->paged_sliding_host_state->physical_block_at_slot(slot));
      }
      // Keep the common logical-table section present for sliding images as
      // well. The ring table and tags are authoritative for reconstruction;
      // the logical section carries the same compact IDs in the first nine
      // slots and invalid entries for any remaining capacity slots.
      topology->logical_table.assign(
          static_cast<size_t>(state->paged_logical_table_capacity),
          kInvalidBlock);
      for (uint32_t slot = 0U; slot != sllm_paged_kv::kSlidingRingSlots;
           ++slot) {
        if (slot < topology->logical_table.size())
          topology->logical_table[slot] = topology->ring_table[slot];
      }
    } else {
      topology->logical_table.assign(
          static_cast<size_t>(state->paged_logical_table_capacity),
          kInvalidBlock);
      for (uint64_t logical = 0U; logical < state->paged_logical_table_capacity;
           ++logical) {
        const uint32_t physical = state->paged_host_state->physical_block(
            static_cast<uint32_t>(logical));
        if (physical == kInvalidBlock)
          continue;
        if (!state->paged_device_pool->descriptor_ready(physical))
          return invalid(sink, "paged image has an unpublished logical block");
        append_physical(topology->physical_ids, physical);
        topology->logical_table[static_cast<size_t>(logical)] = physical;
      }
      for (uint32_t &entry : topology->logical_table)
        if (entry != kInvalidBlock)
          entry = compact_id(topology->physical_ids, entry);
    }
    topology->info.physical_block_count = topology->physical_ids.size();
    for (uint32_t plane = 0U; plane != topology->info.plane_count; ++plane) {
      if (multiply_overflows(topology->info.physical_block_count,
                             topology->info.plane_block_stride[plane],
                             &topology->info.plane_bytes[plane]))
        return sllm_public_runtime::write_error(
            sink, SLLM_STATUS_METADATA_OVERFLOW,
            "paged image plane byte count overflowed u64");
    }
    return sllm_paged_kv_image::validate_info(&topology->info, sink);
  } catch (...) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INTERNAL_ERROR,
        "paged image topology collection failed");
  }
}

sllm_status_t copy_device_range(const void *const source, void *const host,
                                const uint64_t bytes,
                                sllm_error_sink_t *const sink) noexcept {
  if (source == nullptr || host == nullptr || bytes == 0U)
    return invalid(sink, "paged image device range is invalid");
  const hipError_t status = hipMemcpy(host, source, static_cast<size_t>(bytes),
                                      hipMemcpyDeviceToHost);
  return status == hipSuccess
             ? SLLM_STATUS_OK
             : hip_failure(sink, status, "paged image D2H copy");
}

uint32_t read_le32(const std::uint8_t *const bytes) noexcept {
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8U) |
         (static_cast<uint32_t>(bytes[2]) << 16U) |
         (static_cast<uint32_t>(bytes[3]) << 24U);
}

uint64_t read_le64(const std::uint8_t *const bytes) noexcept {
  uint64_t value = 0U;
  for (uint32_t index = 0U; index != 8U; ++index)
    value |= static_cast<uint64_t>(bytes[index]) << (index * 8U);
  return value;
}

sllm_status_t stage_chunk(PendingImport &pending,
                          const sllm_kv_paged_image_chunk_t &chunk,
                          sllm_error_sink_t *const sink) noexcept {
  StagedSection *section = nullptr;
  if (chunk.section == SLLM_HIP_KV_PAGED_IMAGE_SECTION_LOGICAL_TABLE)
    section = &pending.logical_table;
  else if (chunk.section == SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TAGS)
    section = &pending.ring_tags;
  else if (chunk.section == SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TABLE)
    section = &pending.ring_table;
  else if (chunk.section == SLLM_HIP_KV_PAGED_IMAGE_SECTION_PLANE) {
    if (chunk.plane == 0U || chunk.plane > 6U)
      return invalid(sink, "paged import plane is outside the six-plane domain");
    section = &pending.planes[chunk.plane - 1U];
  } else {
    return invalid(sink, "paged import section is invalid");
  }
  uint64_t end = 0U;
  if (add_overflows(chunk.byte_offset, chunk.byte_length, &end) ||
      end > std::numeric_limits<size_t>::max())
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "paged import section range exceeds host size");
  try {
    const size_t target_size = static_cast<size_t>(end);
    const size_t old_size = section->seen.size();
    uint64_t newly_staged = 0U;
    if (chunk.byte_offset < old_size) {
      const size_t overlap_end = std::min(target_size, old_size);
      using Difference = std::vector<std::uint8_t>::difference_type;
      const auto first = section->seen.begin() +
                         static_cast<Difference>(chunk.byte_offset);
      const auto last = section->seen.begin() +
                        static_cast<Difference>(overlap_end);
      newly_staged += static_cast<uint64_t>(std::count(first, last, 0U));
    }
    if (target_size > old_size)
      newly_staged += static_cast<uint64_t>(target_size - old_size);
    if (pending.staged_bytes > kMaxPendingImportBytes ||
        newly_staged > kMaxPendingImportBytes - pending.staged_bytes)
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_BUFFER_TOO_SMALL,
          "paged image import staging exceeds the per-state bound");
    if (section->bytes.size() < target_size) {
      section->bytes.resize(target_size, 0U);
      section->seen.resize(target_size, 0U);
    }
    const auto *const source =
        static_cast<const std::uint8_t *>(chunk.host_pointer);
    std::memcpy(section->bytes.data() + chunk.byte_offset, source,
                static_cast<size_t>(chunk.byte_length));
    using Difference = std::vector<std::uint8_t>::difference_type;
    std::fill(section->seen.begin() +
                  static_cast<Difference>(chunk.byte_offset),
              section->seen.begin() + static_cast<Difference>(end), 1U);
    pending.staged_bytes += newly_staged;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INTERNAL_ERROR,
        "paged import host staging allocation failed");
  }
}

bool section_complete(const StagedSection &section,
                      const uint64_t expected) noexcept {
  if (expected == 0U)
    return section.bytes.empty() && section.seen.empty();
  if (expected > std::numeric_limits<size_t>::max() ||
      section.bytes.size() != static_cast<size_t>(expected) ||
      section.seen.size() != section.bytes.size())
    return false;
  return std::all_of(section.seen.begin(), section.seen.end(),
                     [](const std::uint8_t value) { return value != 0U; });
}

sllm_status_t device_status(const sllm_paged_kv::DeviceStatus status,
                            sllm_error_sink_t *const sink,
                            const char *const operation) noexcept {
  using sllm_paged_kv::DeviceStatus;
  switch (status) {
  case DeviceStatus::Ok:
    return SLLM_STATUS_OK;
  case DeviceStatus::Busy:
    return busy(sink, operation);
  case DeviceStatus::Capacity:
    return sllm_public_runtime::write_error(sink,
                                            SLLM_STATUS_KV_CAPACITY_EXCEEDED,
                                            operation);
  case DeviceStatus::Poisoned:
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INTERNAL_ERROR, operation);
  default:
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR, operation);
  }
}

sllm_kv_state_paged_create_info_t recipe_for_state(const KvState &state) {
  sllm_kv_state_paged_create_info_t recipe{};
  recipe.struct_size = sizeof(recipe);
  recipe.abi_version = SLLM_HIP_ABI_VERSION;
  recipe.create_info_version = SLLM_HIP_KV_PAGED_CREATE_INFO_VERSION;
  recipe.session_id = state.session_id;
  recipe.layer_id = state.layer_id;
  recipe.capacity_tokens = state.capacity_tokens;
  recipe.head_count = state.head_count;
  recipe.head_dim = state.head_dim;
  recipe.memory_kind = SLLM_HIP_KV_MEMORY_KIND_PAGED;
  recipe.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
  recipe.dtype = state.dtype;
  recipe.encoding = state.encoding;
  recipe.token_block_size = state.paged_token_block_size;
  recipe.physical_layout_version = state.paged_physical_layout_version;
  recipe.logical_table_capacity = state.paged_logical_table_capacity;
  recipe.max_physical_blocks = state.paged_max_physical_blocks;
  recipe.sliding_window_tokens = state.sliding_window;
  if (state.encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1) {
    recipe.scale_dtype = SLLM_TENSOR_DTYPE_F32;
    recipe.static_key_scale_bits = float_bits(state.static_key_scale);
    recipe.static_value_scale_bits = float_bits(state.static_value_scale);
  } else if (state.encoding == SLLM_HIP_KV_ENCODING_NVFP4_V1) {
    recipe.scale_dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
    recipe.quantization_block_size = 16U;
  } else if (state.encoding == SLLM_HIP_KV_ENCODING_FP8_V1) {
    recipe.scale_dtype = SLLM_TENSOR_DTYPE_F32;
  } else if (state.encoding == SLLM_HIP_KV_ENCODING_MXFP8_E4_V1 ||
             state.encoding == SLLM_HIP_KV_ENCODING_MXFP8_E5_V1) {
    recipe.scale_dtype = SLLM_TENSOR_DTYPE_U8;
    recipe.quantization_block_size = 32U;
  }
  if (state.encoding == SLLM_HIP_KV_ENCODING_FP16_V1)
    recipe.scale_dtype = 0U;
  return recipe;
}

bool restore_device_entries(
    KvState *const state,
    const std::vector<sllm_paged_kv::LogicalEntryUpdate> &updates,
    hipStream_t stream) noexcept {
  if (updates.empty())
    return true;
  const auto unproven = [&]() noexcept {
    if (state != nullptr && state->context != nullptr)
      poison_context(state->context);
    return false;
  };
  if (state == nullptr || state->paged_logical_table == nullptr)
    return unproven();
  try {
    // A preceding update may still own the table's pending snapshot.  Drain
    // it before asking restore_entries to invert the original update plan.
    if (hipStreamSynchronize(stream) != hipSuccess ||
        state->paged_logical_table->poll_pending() !=
            sllm_paged_kv::DeviceStatus::Ok)
      return unproven();
    const auto status = state->paged_logical_table->restore_entries(updates,
                                                                      stream);
    if (status != sllm_paged_kv::DeviceStatus::Ok ||
        hipStreamSynchronize(stream) != hipSuccess ||
        state->paged_logical_table->poll_pending() !=
                sllm_paged_kv::DeviceStatus::Ok)
      return unproven();
    return true;
  } catch (...) {
    return unproven();
  }
}

} // namespace

void discard_pending_import(const KvState *const state) noexcept {
  if (state == nullptr)
    return;
  std::lock_guard<std::mutex> lock(pending_imports_mutex);
  pending_imports.erase(state);
}

sllm_status_t paged_kv_image_query(
    KvState *const state, sllm_kv_paged_image_info_t *const info,
    sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr)
    return invalid(sink, "paged image query output is null");
  if (info->struct_size != sizeof(*info) ||
      info->abi_version != SLLM_HIP_ABI_VERSION ||
      info->image_version != SLLM_HIP_KV_PAGED_IMAGE_VERSION)
    return invalid(sink, "paged image query header is invalid");
  if (state == nullptr || !state->paged)
    return invalid(sink, "paged image query requires a Paged state");
  Topology topology;
  const sllm_status_t status = collect_topology_locked(state, &topology, sink);
  if (status == SLLM_STATUS_OK)
    *info = topology.info;
  return status;
}

sllm_status_t paged_kv_image_export(
    KvState *const state,
    const sllm_kv_paged_image_chunk_t *const chunk,
    sllm_error_sink_t *const sink) noexcept {
  if (state == nullptr || !state->paged)
    return invalid(sink, "paged image export requires a Paged state");
  if (state->context == nullptr)
    return invalid(sink, "paged image export state context is null");
  const sllm_status_t device_status = select_context_device(state->context, sink);
  if (device_status != SLLM_STATUS_OK)
    return device_status;
  Topology topology;
  sllm_status_t status = collect_topology_locked(state, &topology, sink);
  if (status != SLLM_STATUS_OK)
    return status;
  status = sllm_paged_kv_image::validate_chunk_for_info(&topology.info, chunk,
                                                        sink);
  if (status != SLLM_STATUS_OK)
    return status;
  if (chunk->section == SLLM_HIP_KV_PAGED_IMAGE_SECTION_LOGICAL_TABLE) {
    const auto *const bytes = reinterpret_cast<const std::uint8_t *>(
        topology.logical_table.data());
    std::memcpy(static_cast<std::uint8_t *>(chunk->host_pointer),
                bytes + chunk->byte_offset,
                static_cast<size_t>(chunk->byte_length));
    return SLLM_STATUS_OK;
  }
  if (chunk->section == SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TAGS) {
    const auto *const bytes = reinterpret_cast<const std::uint8_t *>(
        topology.ring_tags.data());
    std::memcpy(static_cast<std::uint8_t *>(chunk->host_pointer),
                bytes + chunk->byte_offset,
                static_cast<size_t>(chunk->byte_length));
    return SLLM_STATUS_OK;
  }
  if (chunk->section == SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TABLE) {
    const auto *const bytes = reinterpret_cast<const std::uint8_t *>(
        topology.ring_table.data());
    std::memcpy(static_cast<std::uint8_t *>(chunk->host_pointer),
                bytes + chunk->byte_offset,
                static_cast<size_t>(chunk->byte_length));
    return SLLM_STATUS_OK;
  }

  const uint32_t plane = chunk->plane;
  const uint64_t stride = topology.info.plane_block_stride[plane - 1U];
  const uint64_t request_begin = chunk->byte_offset;
  const uint64_t request_end = request_begin + chunk->byte_length;
  auto *const destination = static_cast<std::uint8_t *>(chunk->host_pointer);
  for (uint32_t compact = 0U; compact != topology.physical_ids.size();
       ++compact) {
    const uint64_t block_begin = static_cast<uint64_t>(compact) * stride;
    const uint64_t block_end = block_begin + stride;
    const uint64_t begin = std::max(request_begin, block_begin);
    const uint64_t end = std::min(request_end, block_end);
    if (begin >= end)
      continue;
    const uint64_t source_offset = begin - block_begin;
    const uint64_t bytes = end - begin;
    const sllm_paged_kv::BlockDescriptor &descriptor =
        state->paged_device_pool->host_descriptor(topology.physical_ids[compact]);
    const void *source = nullptr;
    switch (plane) {
    case 1U:
      source = descriptor.key;
      break;
    case 2U:
      source = descriptor.value;
      break;
    case 3U:
      source = descriptor.key_scale;
      break;
    case 4U:
      source = descriptor.value_scale;
      break;
    case 5U:
      source = descriptor.key_outer_scale;
      break;
    case 6U:
      source = descriptor.value_outer_scale;
      break;
    default:
      return invalid(sink, "paged image plane is invalid");
    }
    status = copy_device_range(
        static_cast<const std::uint8_t *>(source) + source_offset,
        destination + (begin - request_begin), bytes, sink);
    if (status != SLLM_STATUS_OK)
      return status;
  }
  return SLLM_STATUS_OK;
}

sllm_status_t paged_kv_image_import(
    KvState *const state,
    const sllm_kv_paged_image_chunk_t *const chunk,
    sllm_error_sink_t *const sink) noexcept {
  if (state == nullptr || !state->paged)
    return invalid(sink, "paged image import requires a Paged state");
  const sllm_status_t chunk_status =
      sllm_paged_kv_image::validate_chunk(chunk, sink);
  if (chunk_status != SLLM_STATUS_OK)
    return chunk_status;
  std::lock_guard<std::mutex> lock(pending_imports_mutex);
  try {
    return stage_chunk(pending_imports[state], *chunk, sink);
  } catch (...) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INTERNAL_ERROR,
        "paged image import staging failed");
  }
}

sllm_status_t paged_kv_image_import_finalize(
    KvState *const state,
    const sllm_kv_paged_image_info_t *const info,
    sllm_error_sink_t *const sink) noexcept {
  if (state == nullptr || !state->paged)
    return invalid(sink, "paged image finalize requires a Paged state");
  if (state->context == nullptr)
    return invalid(sink, "paged image finalize state context is null");
  PendingImport pending;
  {
    std::lock_guard<std::mutex> lock(pending_imports_mutex);
    const auto found = pending_imports.find(state);
    if (found == pending_imports.end())
      return invalid(sink, "paged image finalize has no staged sections");
    pending = std::move(found->second);
    pending_imports.erase(found);
  }

  // Consume the per-state staging record before validating any terminal
  // input.  This keeps malformed metadata, recipe mismatch, and all later
  // failure paths from leaving a pointer-keyed transaction behind.
  const sllm_status_t info_status =
      sllm_paged_kv_image::validate_info(info, sink);
  if (info_status != SLLM_STATUS_OK)
    return info_status;

  if (!state_quiescent(state) || state->published_length != 0U ||
      state->generation != 0U)
    return busy(sink, "paged image destination is not an empty quiescent state");
  const auto recipe = recipe_for_state(*state);
  sllm_status_t status =
      sllm_paged_kv_image::validate_recipe(info, &recipe, sink);
  if (status != SLLM_STATUS_OK)
    return status;

  uint64_t logical_bytes = 0U;
  uint64_t tags_bytes = 0U;
  uint64_t ring_table_bytes = 0U;
  const bool sliding =
      (info->flags & SLLM_HIP_KV_PAGED_IMAGE_FLAG_SLIDING) != 0U;
  if (multiply_overflows(info->logical_table_capacity, sizeof(uint32_t),
                         &logical_bytes) ||
      multiply_overflows(info->ring_slot_count, sizeof(uint64_t),
                         &tags_bytes) ||
      multiply_overflows(info->ring_slot_count, sizeof(uint32_t),
                         &ring_table_bytes)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "paged image import table size overflowed u64");
  }
  if (!section_complete(pending.logical_table, logical_bytes) ||
      !section_complete(pending.ring_tags, tags_bytes) ||
      !section_complete(pending.ring_table, ring_table_bytes))
    return invalid(sink, "paged image import table sections are incomplete");
  for (uint32_t plane = 0U; plane != 6U; ++plane) {
    const uint64_t expected = plane < info->plane_count
                                  ? info->plane_bytes[plane]
                                  : 0U;
    if (!section_complete(pending.planes[plane], expected))
      return invalid(sink, "paged image import plane section is incomplete");
  }

  std::vector<uint32_t> compact_table;
  std::array<uint64_t, sllm_paged_kv::kSlidingRingSlots> ring_tags{};
  std::array<uint32_t, sllm_paged_kv::kSlidingRingSlots> ring_table{};
  std::array<uint64_t, sllm_paged_kv::kSlidingRingSlots> old_ring_tags{};
  if (sliding) {
    if (state->paged_sliding_ring_tags_host == nullptr ||
        state->paged_sliding_ring_tags_device == nullptr)
      return invalid(sink, "paged sliding image ring tag storage is unavailable");
    std::memcpy(old_ring_tags.data(), state->paged_sliding_ring_tags_host,
                old_ring_tags.size() * sizeof(uint64_t));
  }
  try {
    compact_table.resize(static_cast<size_t>(info->logical_table_capacity),
                         kInvalidBlock);
    for (size_t index = 0U; index != compact_table.size(); ++index)
      compact_table[index] = read_le32(
          pending.logical_table.bytes.data() + index * sizeof(uint32_t));
    if (sliding) {
      for (uint32_t slot = 0U; slot != sllm_paged_kv::kSlidingRingSlots;
           ++slot) {
        ring_tags[slot] = read_le64(pending.ring_tags.bytes.data() +
                                    slot * sizeof(uint64_t));
        ring_table[slot] = read_le32(pending.ring_table.bytes.data() +
                                     slot * sizeof(uint32_t));
      }
      if (info->ring_slot_count != sllm_paged_kv::kSlidingRingSlots)
        return invalid(sink, "paged sliding image ring slot count is invalid");
    }
  } catch (...) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INTERNAL_ERROR,
        "paged image import table decode failed");
  }

  std::vector<sllm_paged_kv::LogicalEntryUpdate> updates;
  try {
    if (!sliding) {
      for (uint32_t logical = 0U;
           logical != static_cast<uint32_t>(compact_table.size()); ++logical) {
        const uint32_t compact = compact_table[logical];
        if (compact == kInvalidBlock)
          continue;
        if (compact >= info->physical_block_count)
          return invalid(sink, "paged image logical table ID is invalid");
      }
    } else {
      for (uint32_t slot = 0U; slot != sllm_paged_kv::kSlidingRingSlots;
           ++slot) {
        const uint32_t logical = compact_table[slot];
        if (logical != ring_table[slot])
          return invalid(sink, "paged image logical and ring tables differ");
      }
      for (size_t logical = sllm_paged_kv::kSlidingRingSlots;
           logical != compact_table.size(); ++logical) {
        if (compact_table[logical] != kInvalidBlock)
          return invalid(sink, "paged image sliding logical table has extra blocks");
      }
      compact_table.assign(sllm_paged_kv::kSlidingRingSlots, kInvalidBlock);
      for (uint32_t slot = 0U; slot != sllm_paged_kv::kSlidingRingSlots;
           ++slot) {
        if (ring_tags[slot] == UINT64_MAX) {
          if (ring_table[slot] != kInvalidBlock)
            return invalid(sink, "paged image has an untagged ring block");
          continue;
        }
        if (ring_table[slot] >= info->physical_block_count ||
            state->paged_logical_table->size() <= slot ||
            ring_tags[slot] % sllm_paged_kv::kSlidingRingSlots != slot)
          return invalid(sink, "paged image ring table entry is invalid");
        compact_table[slot] = ring_table[slot];
      }
    }
    std::vector<bool> seen(static_cast<size_t>(info->physical_block_count),
                           false);
    for (const uint32_t compact : compact_table) {
      if (compact == kInvalidBlock)
        continue;
      if (compact >= seen.size() || seen[compact])
        return invalid(sink, "paged image physical IDs are not compact");
      seen[compact] = true;
    }
    if (std::count(seen.begin(), seen.end(), true) !=
        static_cast<std::ptrdiff_t>(info->physical_block_count))
      return invalid(sink, "paged image physical block count is incomplete");
  } catch (...) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INTERNAL_ERROR,
        "paged image import topology allocation failed");
  }

  hipStream_t stream = nullptr;
  const hipError_t stream_status =
      hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
  if (stream_status != hipSuccess)
    return hip_failure(sink, stream_status,
                       "create paged image import stream");
  auto close_stream = [&]() noexcept {
    const hipError_t status = hipStreamDestroy(stream);
    stream = nullptr;
    return status;
  };

  sllm_paged_kv::State::ImageImport state_import;
  sllm_paged_kv::SlidingState::ImageImport sliding_import;
  std::vector<uint32_t> reserved_blocks;
  if (sliding) {
    const auto prepare = state->paged_sliding_host_state->prepare_image_import(
        [&] {
          std::array<uint32_t, sllm_paged_kv::kSlidingRingSlots> result{};
          result.fill(kInvalidBlock);
          for (uint32_t slot = 0U; slot != sllm_paged_kv::kSlidingRingSlots;
               ++slot)
            result[slot] = ring_table[slot];
          return result;
        }(),
        ring_tags, info->physical_block_count, info->published_length,
        info->generation, sliding_import);
    if (prepare != sllm_paged_kv::Status::Ok) {
      (void)close_stream();
      return device_status(
          prepare == sllm_paged_kv::Status::Busy
              ? sllm_paged_kv::DeviceStatus::Busy
              : prepare == sllm_paged_kv::Status::PoolExhausted
                    ? sllm_paged_kv::DeviceStatus::Capacity
                    : sllm_paged_kv::DeviceStatus::Invalid,
          sink, "paged sliding image host reservation failed");
    }
    reserved_blocks = sliding_import.reserved_blocks;
  } else {
    const auto prepare = state->paged_host_state->prepare_image_import(
        compact_table, info->physical_block_count, info->published_length,
        info->generation, state_import);
    if (prepare != sllm_paged_kv::Status::Ok) {
      (void)close_stream();
      return device_status(
          prepare == sllm_paged_kv::Status::Busy
              ? sllm_paged_kv::DeviceStatus::Busy
              : prepare == sllm_paged_kv::Status::PoolExhausted
                    ? sllm_paged_kv::DeviceStatus::Capacity
                    : sllm_paged_kv::DeviceStatus::Invalid,
          sink, "paged image host reservation failed");
    }
    reserved_blocks = state_import.reserved_blocks;
  }
  const auto rollback_host = [&]() noexcept {
    sllm_paged_kv::Status status = sllm_paged_kv::Status::Ok;
    if (sliding)
      status = state->paged_sliding_host_state->rollback_image_import(
          sliding_import);
    else
      status = state->paged_host_state->rollback_image_import(state_import);
    return status == sllm_paged_kv::Status::Ok;
  };
  const auto fail = [&](const sllm_status_t failure,
                        const bool device_restored = true) noexcept {
    const bool host_rolled_back = device_restored ? rollback_host() : false;
    const hipError_t close_status = close_stream();
    if (!host_rolled_back || close_status != hipSuccess) {
      if (state->context != nullptr)
        poison_context(state->context);
    }
    return failure;
  };

  const auto restore_ring_tags = [&]() noexcept {
    if (!sliding)
      return true;
    std::memcpy(state->paged_sliding_ring_tags_host, old_ring_tags.data(),
                old_ring_tags.size() * sizeof(uint64_t));
    const hipError_t copy = hipMemcpyAsync(
        state->paged_sliding_ring_tags_device,
        state->paged_sliding_ring_tags_host,
        old_ring_tags.size() * sizeof(uint64_t), hipMemcpyHostToDevice, stream);
    return copy == hipSuccess && hipStreamSynchronize(stream) == hipSuccess;
  };
  const auto restore_device_state = [&]() noexcept {
    const bool table_restored = restore_device_entries(state, updates, stream);
    const bool tags_restored = restore_ring_tags();
    if (!table_restored || !tags_restored) {
      if (state->context != nullptr)
        poison_context(state->context);
      return false;
    }
    return true;
  };

  status = device_status(state->paged_device_pool->ensure_blocks(reserved_blocks,
                                                                   stream),
                         sink, "paged image physical block reservation");
  if (status != SLLM_STATUS_OK)
    return fail(status, true);

  try {
    updates.reserve(compact_table.size());
    for (uint32_t logical = 0U; logical != compact_table.size(); ++logical) {
      const uint32_t compact = compact_table[logical];
      if (compact != kInvalidBlock)
        updates.push_back({logical, kInvalidBlock, reserved_blocks[compact]});
    }
  } catch (...) {
    return fail(sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INTERNAL_ERROR,
        "paged image table update allocation failed"), true);
  }
  if (!sliding) {
    status = device_status(
        state->paged_logical_table->update_entries(updates, stream), sink,
        "paged image logical table update");
  } else {
    std::vector<sllm_paged_kv::LogicalEntryUpdate> ring_updates;
    ring_updates.reserve(sllm_paged_kv::kSlidingRingSlots);
    for (uint32_t slot = 0U; slot != sllm_paged_kv::kSlidingRingSlots; ++slot)
      if (ring_table[slot] != kInvalidBlock)
        ring_updates.push_back({slot, kInvalidBlock, reserved_blocks[ring_table[slot]]});
    status = device_status(
        state->paged_logical_table->update_entries(ring_updates, stream), sink,
        "paged image ring table update");
    updates = std::move(ring_updates);
    if (status == SLLM_STATUS_OK && state->paged_sliding_ring_tags_device == nullptr)
      status = invalid(sink, "paged sliding image ring tag storage is unavailable");
    if (status == SLLM_STATUS_OK) {
      const hipError_t tags = hipMemcpyAsync(
          state->paged_sliding_ring_tags_device, ring_tags.data(),
          ring_tags.size() * sizeof(uint64_t), hipMemcpyHostToDevice, stream);
      if (tags != hipSuccess)
        status = hip_failure(sink, tags, "paged image ring tag upload");
    }
  }
  if (status != SLLM_STATUS_OK) {
    const bool restored = restore_device_state();
    return fail(status, restored);
  }

  for (uint32_t plane = 0U; plane != info->plane_count && status == SLLM_STATUS_OK;
       ++plane) {
    const uint64_t stride = info->plane_block_stride[plane];
    for (uint32_t compact = 0U;
         compact != info->physical_block_count && status == SLLM_STATUS_OK;
         ++compact) {
      const auto &descriptor =
          state->paged_device_pool->host_descriptor(reserved_blocks[compact]);
      void *destination = nullptr;
      switch (plane + 1U) {
      case 1U:
        destination = descriptor.key;
        break;
      case 2U:
        destination = descriptor.value;
        break;
      case 3U:
        destination = descriptor.key_scale;
        break;
      case 4U:
        destination = descriptor.value_scale;
        break;
      case 5U:
        destination = descriptor.key_outer_scale;
        break;
      case 6U:
        destination = descriptor.value_outer_scale;
        break;
      default:
        break;
      }
      const hipError_t copy = hipMemcpyAsync(
          destination,
          pending.planes[plane].bytes.data() + compact * stride,
          static_cast<size_t>(stride), hipMemcpyHostToDevice, stream);
      if (copy != hipSuccess)
        status = hip_failure(sink, copy, "paged image plane upload");
    }
  }
  if (status == SLLM_STATUS_OK) {
    const hipError_t sync = hipStreamSynchronize(stream);
    if (sync != hipSuccess)
      status = hip_failure(sink, sync, "paged image import stream");
  }
  if (status != SLLM_STATUS_OK) {
    const bool restored = restore_device_state();
    return fail(status, restored);
  }
  const auto commit = sliding
                          ? state->paged_sliding_host_state->commit_image_import(
                                sliding_import)
                          : state->paged_host_state->commit_image_import(state_import);
  if (commit != sllm_paged_kv::Status::Ok) {
    const bool restored = restore_device_state();
    return fail(sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INTERNAL_ERROR,
        "paged image host ownership commit failed"), restored);
  }
  if (sliding && state->paged_sliding_ring_tags_host != nullptr)
    std::memcpy(state->paged_sliding_ring_tags_host, ring_tags.data(),
                ring_tags.size() * sizeof(uint64_t));
  state->published_length = info->published_length;
  state->generation = info->generation;
  state->retained_start = info->retained_start;
  const hipError_t close_status = close_stream();
  if (close_status != hipSuccess) {
    if (state->context != nullptr)
      poison_context(state->context);
    return hip_failure(sink, close_status,
                       "destroy paged image import stream");
  }
  return SLLM_STATUS_OK;
}

} // namespace sllm_paged_kv_image_runtime
