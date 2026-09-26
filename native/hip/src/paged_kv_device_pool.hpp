#ifndef SLLM_PAGED_KV_DEVICE_POOL_HPP
#define SLLM_PAGED_KV_DEVICE_POOL_HPP

// Device-side ownership for the Stage 10 segmented Paged KV pool.
//
// DevicePool owns the physical allocation and the fixed descriptor table.
// LogicalTable owns one request/state logical block table.  A forked state can
// therefore create another LogicalTable while sharing the same DevicePool and
// its descriptors.  Neither table is resized after construction; this is the
// pointer lifetime contract required by a later HIP graph capture.

#include "paged_kv_device_layout.hpp"
#include "paged_kv_pool_state.hpp"
#include "paged_kv_slab_plan.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sllm_paged_kv {

enum class DeviceStatus : std::uint8_t {
  Ok,
  Invalid,
  Busy,
  Capacity,
  Corrupt,
  HipError,
  Poisoned,
};

struct LogicalEntryUpdate final {
  std::uint32_t logical_block;
  std::uint32_t expected_physical;
  std::uint32_t new_physical;
};

class DevicePool final {
public:
  class LogicalTable;

  explicit DevicePool(SlabPlan plan) : plan_(std::move(plan)) {
    const std::size_t physical_count =
        checked_count(plan_.pool_capacity_blocks());
    const std::size_t slab_count = checked_count(plan_.slab_count());
    slabs_.assign(slab_count, nullptr);
    slab_ready_.assign(slab_count, false);
    descriptors_.assign(physical_count, BlockDescriptor{});
    if (physical_count >
        std::numeric_limits<std::size_t>::max() / sizeof(BlockDescriptor))
      throw std::overflow_error("paged KV descriptor table size overflow");

    const hipError_t pin = hipHostMalloc(
        reinterpret_cast<void **>(&pinned_descriptors_),
        physical_count * sizeof(BlockDescriptor), hipHostMallocPortable);
    if (pin != hipSuccess) {
      last_error_ = pin;
      throw hip_exception("hipHostMalloc paged KV descriptors", pin);
    }
    std::memset(pinned_descriptors_, 0,
                physical_count * sizeof(BlockDescriptor));

    const hipError_t allocation =
        hipMalloc(reinterpret_cast<void **>(&device_descriptors_),
                  physical_count * sizeof(BlockDescriptor));
    if (allocation != hipSuccess) {
      last_error_ = allocation;
      (void)hipHostFree(pinned_descriptors_);
      pinned_descriptors_ = nullptr;
      throw hip_exception("hipMalloc paged KV descriptor table", allocation);
    }
    const hipError_t event =
        hipEventCreateWithFlags(&initialization_event_, hipEventDisableTiming);
    if (event != hipSuccess) {
      last_error_ = event;
      (void)hipFree(device_descriptors_);
      device_descriptors_ = nullptr;
      (void)hipHostFree(pinned_descriptors_);
      pinned_descriptors_ = nullptr;
      throw hip_exception("hipEventCreate paged KV descriptor table", event);
    }
    const hipError_t clear =
        hipMemsetAsync(device_descriptors_, 0,
                       physical_count * sizeof(BlockDescriptor), nullptr);
    if (clear != hipSuccess) {
      last_error_ = clear;
      (void)hipEventDestroy(initialization_event_);
      initialization_event_ = nullptr;
      (void)hipFree(device_descriptors_);
      device_descriptors_ = nullptr;
      (void)hipHostFree(pinned_descriptors_);
      pinned_descriptors_ = nullptr;
      throw hip_exception("hipMemsetAsync paged KV descriptor table", clear);
    }
    const hipError_t record = hipEventRecord(initialization_event_, nullptr);
    if (record != hipSuccess) {
      last_error_ = record;
      (void)hipStreamSynchronize(nullptr);
      (void)hipEventDestroy(initialization_event_);
      initialization_event_ = nullptr;
      (void)hipFree(device_descriptors_);
      device_descriptors_ = nullptr;
      (void)hipHostFree(pinned_descriptors_);
      pinned_descriptors_ = nullptr;
      throw hip_exception("hipEventRecord paged KV descriptor table", record);
    }
  }

  ~DevicePool() { cleanup_noexcept(); }

  DevicePool(const DevicePool &) = delete;
  DevicePool &operator=(const DevicePool &) = delete;
  DevicePool(DevicePool &&) = delete;
  DevicePool &operator=(DevicePool &&) = delete;

  const SlabPlan &plan() const noexcept { return plan_; }
  std::uint32_t physical_capacity() const noexcept {
    return static_cast<std::uint32_t>(plan_.pool_capacity_blocks());
  }
  std::uint32_t logical_capacity() const noexcept {
    return static_cast<std::uint32_t>(plan_.logical_block_count());
  }

  // This pointer is stable until release/destruction.  The table is owned by
  // this object and is shared by all LogicalTable instances made from it.
  BlockDescriptor *device_descriptor_table() const noexcept {
    return device_descriptors_;
  }

  bool descriptor_ready(const std::uint32_t physical) const noexcept {
    if (physical >= descriptors_.size())
      return false;
    const BlockLocation location = plan_.location(physical);
    return location.slab < slab_ready_.size() &&
           slab_ready_[static_cast<std::size_t>(location.slab)];
  }

  const BlockDescriptor &host_descriptor(const std::uint32_t physical) const {
    if (!descriptor_ready(physical))
      throw std::invalid_argument("paged KV descriptor is not published");
    return descriptors_[physical];
  }

  hipError_t last_hip_error() const noexcept { return last_error_; }
  bool poisoned() const noexcept { return poisoned_; }

  DeviceStatus wait_for_initialization(const hipStream_t stream) noexcept {
    if (released_ || poisoned_ || initialization_event_ == nullptr)
      return poisoned_ ? DeviceStatus::Poisoned : DeviceStatus::Invalid;
    const hipError_t status =
        hipStreamWaitEvent(stream, initialization_event_, 0U);
    if (status != hipSuccess) {
      last_error_ = status;
      poisoned_ = true;
      return DeviceStatus::HipError;
    }
    return DeviceStatus::Ok;
  }

  // A slab is committed as one full eight-slot allocation, including the
  // final slab whose addressable physical ID range may contain fewer slots.
  std::uint64_t allocated_slab_count() const noexcept {
    std::uint64_t count = 0U;
    for (const void *slab : slabs_) {
      if (slab != nullptr)
        ++count;
    }
    return count;
  }

  // Count only physical IDs that are addressable by the pool and whose slab
  // has been allocated.  This intentionally differs from the committed slot
  // count used by committed_bytes_per_plane().
  std::uint64_t allocated_physical_blocks() const noexcept {
    std::uint64_t count = 0U;
    for (std::size_t slab = 0U; slab < slabs_.size(); ++slab) {
      if (slabs_[slab] == nullptr)
        continue;
      const std::uint64_t first =
          static_cast<std::uint64_t>(slab) * kSlabBlocks;
      const std::uint64_t remaining = plan_.pool_capacity_blocks() - first;
      count += std::min<std::uint64_t>(kSlabBlocks, remaining);
    }
    return count;
  }

  // Values are counted for every allocated slab slot, including unused slots
  // in a partial final slab.  The plan has already checked each per-block
  // byte count; saturating arithmetic keeps this noexcept query fail-closed
  // if an unrealistically large synthetic plan is inspected.
  PlaneBytes committed_bytes_per_plane() const noexcept {
    const std::uint64_t slots =
        saturating_mul(allocated_slab_count(), kSlabBlocks);
    PlaneBytes committed{};
    const PlaneBytes &per_block = plan_.plane_bytes_per_block();
    for (std::size_t plane = 0U; plane < committed.size(); ++plane)
      committed[plane] = saturating_mul(per_block[plane], slots);
    return committed;
  }

  std::uint64_t committed_bytes_total() const noexcept {
    const PlaneBytes committed = committed_bytes_per_plane();
    std::uint64_t total = 0U;
    for (const std::uint64_t bytes : committed)
      total = saturating_add(total, bytes);
    return total;
  }

  // Allocate all slabs needed by physical_ids and enqueue descriptor-table
  // publication on stream.  The allocation itself is deliberately outside
  // the stream, while descriptor visibility is ordered by the supplied
  // stream.  A failed enqueue restores any ranges already queued to zero.
  DeviceStatus ensure_blocks(const std::vector<std::uint32_t> &physical_ids,
                             hipStream_t stream) noexcept {
    if (released_ || poisoned_)
      return poisoned_ ? DeviceStatus::Poisoned : DeviceStatus::Invalid;
    if (physical_ids.empty())
      return DeviceStatus::Ok;
    const DeviceStatus initialized = wait_for_initialization(stream);
    if (initialized != DeviceStatus::Ok)
      return initialized;

    std::vector<bool> needed;
    try {
      needed.assign(slab_ready_.size(), false);
    } catch (...) {
      return DeviceStatus::Capacity;
    }
    for (const std::uint32_t physical : physical_ids) {
      if (physical >= descriptors_.size())
        return DeviceStatus::Invalid;
      const BlockLocation location = plan_.location(physical);
      if (location.slab >= needed.size())
        return DeviceStatus::Corrupt;
      if (!slab_ready_[static_cast<std::size_t>(location.slab)])
        needed[static_cast<std::size_t>(location.slab)] = true;
    }

    struct Range final {
      std::size_t slab;
      std::size_t first;
      std::size_t count;
    };
    std::vector<std::size_t> newly_allocated;
    std::vector<Range> queued;
    try {
      const std::size_t count = static_cast<std::size_t>(
          std::count(needed.begin(), needed.end(), true));
      newly_allocated.reserve(count);
      queued.reserve(count);
    } catch (...) {
      return DeviceStatus::Capacity;
    }
    try {
      for (std::size_t slab = 0U; slab < needed.size(); ++slab) {
        if (!needed[slab] || slabs_[slab] != nullptr)
          continue;
        void *allocation = nullptr;
        const hipError_t status =
            hipMalloc(&allocation, plan_.slab_footprint());
        if (status != hipSuccess) {
          last_error_ = status;
          const bool restored = free_new_slabs(newly_allocated);
          return restored ? DeviceStatus::HipError : DeviceStatus::Poisoned;
        }
        slabs_[slab] = allocation;
        newly_allocated.push_back(slab);
        populate_descriptors(slab);
      }
    } catch (...) {
      const bool restored = free_new_slabs(newly_allocated);
      return restored ? DeviceStatus::Corrupt : DeviceStatus::Poisoned;
    }

    for (std::size_t slab = 0U; slab < needed.size(); ++slab) {
      if (!needed[slab])
        continue;
      const std::size_t first = slab * static_cast<std::size_t>(kSlabBlocks);
      const std::size_t count = std::min(static_cast<std::size_t>(kSlabBlocks),
                                         descriptors_.size() - first);
      const hipError_t status = hipMemcpyAsync(
          device_descriptors_ + first, pinned_descriptors_ + first,
          count * sizeof(BlockDescriptor), hipMemcpyHostToDevice, stream);
      if (status != hipSuccess) {
        last_error_ = status;
        bool restored = true;
        for (const Range &range : queued) {
          const hipError_t rollback =
              hipMemsetAsync(device_descriptors_ + range.first, 0,
                             range.count * sizeof(BlockDescriptor), stream);
          if (rollback != hipSuccess) {
            last_error_ = rollback;
            restored = false;
          }
        }
        for (const Range &range : queued)
          slab_ready_[range.slab] = false;
        if (!restored)
          poisoned_ = true;
        return poisoned_ ? DeviceStatus::Poisoned : DeviceStatus::HipError;
      }
      queued.push_back(Range{slab, first, count});
    }
    for (const Range &range : queued)
      slab_ready_[range.slab] = true;
    return DeviceStatus::Ok;
  }

  // The physical pool owns the descriptor table, while each state owns one
  // fixed logical table.  The pool must outlive every returned table.
  std::unique_ptr<LogicalTable>
  make_logical_table(const std::uint64_t logical_blocks, hipStream_t stream);

  // Wait for descriptor publication and release every slab/table allocation.
  // LogicalTable instances must already have been released or destroyed.
  DeviceStatus release(hipStream_t stream) noexcept {
    if (released_)
      return DeviceStatus::Invalid;
    if (active_tables_ != 0U)
      return DeviceStatus::Busy;
    const hipError_t wait = hipStreamSynchronize(stream);
    if (wait != hipSuccess) {
      last_error_ = wait;
      poisoned_ = true;
      return DeviceStatus::HipError;
    }
    const hipError_t cleanup = cleanup_allocations_noexcept();
    if (cleanup != hipSuccess) {
      last_error_ = cleanup;
      poisoned_ = true;
    }
    released_ = cleanup == hipSuccess;
    return released_ ? DeviceStatus::Ok : DeviceStatus::HipError;
  }

private:
  friend class LogicalTable;

  bool
  free_new_slabs(const std::vector<std::size_t> &newly_allocated) noexcept {
    bool restored = true;
    for (const std::size_t index : newly_allocated) {
      if (slabs_[index] == nullptr)
        continue;
      const hipError_t status = hipFree(slabs_[index]);
      if (status == hipSuccess) {
        slabs_[index] = nullptr;
      } else {
        last_error_ = status;
        poisoned_ = true;
        restored = false;
      }
    }
    return restored;
  }

  static std::size_t checked_count(const std::uint64_t count) {
    if (count == 0U || count > std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("paged KV device table count overflow");
    return static_cast<std::size_t>(count);
  }

  static std::uint64_t saturating_mul(const std::uint64_t lhs,
                                      const std::uint64_t rhs) noexcept {
    if (lhs != 0U && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
      return std::numeric_limits<std::uint64_t>::max();
    return lhs * rhs;
  }

  static std::uint64_t saturating_add(const std::uint64_t lhs,
                                      const std::uint64_t rhs) noexcept {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs)
      return std::numeric_limits<std::uint64_t>::max();
    return lhs + rhs;
  }

  static std::runtime_error hip_exception(const char *operation,
                                          const hipError_t error) {
    return std::runtime_error(std::string(operation) + ": " +
                              hipGetErrorString(error));
  }

  void populate_descriptors(const std::size_t slab) {
    const std::size_t first = slab * static_cast<std::size_t>(kSlabBlocks);
    const std::size_t count = std::min(static_cast<std::size_t>(kSlabBlocks),
                                       descriptors_.size() - first);
    auto *const base = reinterpret_cast<std::uint8_t *>(slabs_[slab]);
    for (std::size_t index = 0U; index < count; ++index) {
      const std::uint32_t physical = static_cast<std::uint32_t>(first + index);
      const DescriptorOffsetPlan offsets = plan_.descriptor_offsets(physical);
      BlockDescriptor descriptor{};
      std::uint8_t **const pointers[] = {&descriptor.key,
                                         &descriptor.value,
                                         &descriptor.key_scale,
                                         &descriptor.value_scale,
                                         &descriptor.key_outer_scale,
                                         &descriptor.value_outer_scale};
      for (std::size_t plane = 0U; plane < offsets.present.size(); ++plane) {
        if (offsets.present[plane])
          *pointers[plane] = base + offsets.offsets[plane];
      }
      descriptors_[physical] = descriptor;
      pinned_descriptors_[physical] = descriptor;
    }
  }

  hipError_t cleanup_allocations_noexcept() noexcept {
    hipError_t first = hipSuccess;
    for (void *&slab : slabs_) {
      if (slab != nullptr) {
        const hipError_t status = hipFree(slab);
        if (status == hipSuccess)
          slab = nullptr;
        else if (first == hipSuccess)
          first = status;
      }
    }
    if (initialization_event_ != nullptr) {
      const hipError_t synchronize = hipEventSynchronize(initialization_event_);
      if (synchronize != hipSuccess && first == hipSuccess)
        first = synchronize;
      const hipError_t status = hipEventDestroy(initialization_event_);
      if (status == hipSuccess)
        initialization_event_ = nullptr;
      else if (first == hipSuccess)
        first = status;
    }
    if (device_descriptors_ != nullptr) {
      const hipError_t status = hipFree(device_descriptors_);
      if (status == hipSuccess)
        device_descriptors_ = nullptr;
      else if (first == hipSuccess)
        first = status;
    }
    if (pinned_descriptors_ != nullptr) {
      const hipError_t status = hipHostFree(pinned_descriptors_);
      if (status == hipSuccess)
        pinned_descriptors_ = nullptr;
      else if (first == hipSuccess)
        first = status;
    }
    return first;
  }

  void cleanup_noexcept() noexcept {
    if (released_)
      return;
    (void)cleanup_allocations_noexcept();
    released_ = true;
  }

  void table_created() noexcept { ++active_tables_; }

  void table_destroyed() noexcept {
    if (active_tables_ != 0U)
      --active_tables_;
  }

  SlabPlan plan_;
  std::vector<void *> slabs_;
  std::vector<bool> slab_ready_;
  std::vector<BlockDescriptor> descriptors_;
  BlockDescriptor *pinned_descriptors_ = nullptr;
  BlockDescriptor *device_descriptors_ = nullptr;
  hipError_t last_error_ = hipSuccess;
  hipEvent_t initialization_event_ = nullptr;
  std::size_t active_tables_ = 0U;
  bool poisoned_ = false;
  bool released_ = false;
};

class DevicePool::LogicalTable final {
public:
  ~LogicalTable() { (void)cleanup_noexcept(); }

  LogicalTable(const LogicalTable &) = delete;
  LogicalTable &operator=(const LogicalTable &) = delete;
  LogicalTable(LogicalTable &&) = delete;
  LogicalTable &operator=(LogicalTable &&) = delete;

  std::uint32_t *device_table() const noexcept { return device_entries_; }
  std::size_t size() const noexcept { return host_entries_.size(); }
  std::uint32_t host_entry(const std::uint32_t logical) const {
    if (logical >= host_entries_.size())
      throw std::out_of_range("paged KV logical table index");
    return host_entries_[logical];
  }
  bool pending() const noexcept { return pending_; }
  hipError_t last_hip_error() const noexcept { return last_error_; }
  bool poisoned() const noexcept { return poisoned_; }

  // Validate expected_old and descriptor publication before enqueueing the
  // changed entries.  At most one update may be in flight; the event keeps
  // the immutable host snapshot alive until the supplied stream consumes it.
  DeviceStatus update_entries(const std::vector<LogicalEntryUpdate> &updates,
                              hipStream_t stream) noexcept {
    if (released_ || pool_.poisoned())
      return DeviceStatus::Poisoned;
    const DeviceStatus initialized = wait_for_initialization(stream);
    if (initialized != DeviceStatus::Ok)
      return initialized;
    const DeviceStatus completion = poll_pending();
    if (completion != DeviceStatus::Ok)
      return completion;
    if (updates.empty())
      return DeviceStatus::Ok;

    std::vector<bool> seen;
    std::vector<std::uint32_t> queued;
    try {
      seen.assign(host_entries_.size(), false);
      queued.reserve(updates.size());
    } catch (...) {
      return DeviceStatus::Capacity;
    }
    for (const LogicalEntryUpdate &update : updates) {
      if (update.logical_block >= host_entries_.size() ||
          seen[update.logical_block] ||
          host_entries_[update.logical_block] != update.expected_physical)
        return DeviceStatus::Invalid;
      seen[update.logical_block] = true;
      if (update.new_physical != kInvalidBlock &&
          !pool_.descriptor_ready(update.new_physical))
        return DeviceStatus::Invalid;
    }

    const std::size_t slot = pending_slot_ == 0U ? 1U : 0U;
    std::copy(host_entries_.begin(), host_entries_.end(),
              snapshots_[slot].values.begin());
    for (const LogicalEntryUpdate &update : updates) {
      snapshots_[slot].values[update.logical_block] = update.new_physical;
      snapshots_[slot].pinned_new[update.logical_block] = update.new_physical;
      snapshots_[slot].pinned_old[update.logical_block] =
          update.expected_physical;
      if (update.expected_physical == update.new_physical)
        continue;
      const hipError_t status =
          hipMemcpyAsync(device_entries_ + update.logical_block,
                         &snapshots_[slot].pinned_new[update.logical_block],
                         sizeof(std::uint32_t), hipMemcpyHostToDevice, stream);
      if (status != hipSuccess) {
        last_error_ = status;
        return rollback_enqueued(queued, slot, stream, status);
      }
      queued.push_back(update.logical_block);
    }
    if (queued.empty())
      return DeviceStatus::Ok;

    const hipError_t event_status =
        hipEventRecord(snapshots_[slot].event, stream);
    if (event_status != hipSuccess) {
      last_error_ = event_status;
      return rollback_enqueued(queued, slot, stream, event_status);
    }
    pending_slot_ = slot;
    pending_ = true;
    return DeviceStatus::Ok;
  }

  // Restore old values after a higher-level append transaction failed.  The
  // caller supplies the same change list used for update_entries.
  DeviceStatus restore_entries(const std::vector<LogicalEntryUpdate> &updates,
                               hipStream_t stream) noexcept {
    if (released_ || pending_)
      return pending_ ? DeviceStatus::Busy : DeviceStatus::Invalid;
    std::vector<LogicalEntryUpdate> restore;
    try {
      restore.reserve(updates.size());
      for (const LogicalEntryUpdate &update : updates) {
        if (update.logical_block >= host_entries_.size() ||
            host_entries_[update.logical_block] != update.new_physical)
          return DeviceStatus::Invalid;
        restore.push_back(LogicalEntryUpdate{update.logical_block,
                                             update.new_physical,
                                             update.expected_physical});
      }
    } catch (...) {
      return DeviceStatus::Capacity;
    }
    return update_entries(restore, stream);
  }

  DeviceStatus poll_pending() noexcept {
    if (!pending_)
      return poisoned_ ? DeviceStatus::Poisoned : DeviceStatus::Ok;
    const hipError_t status = hipEventQuery(snapshots_[pending_slot_].event);
    if (status == hipErrorNotReady)
      return DeviceStatus::Busy;
    if (status != hipSuccess) {
      last_error_ = status;
      poisoned_ = true;
      return DeviceStatus::Poisoned;
    }
    std::copy(snapshots_[pending_slot_].values.begin(),
              snapshots_[pending_slot_].values.end(), host_entries_.begin());
    pending_ = false;
    return DeviceStatus::Ok;
  }

  DeviceStatus wait_for_initialization(const hipStream_t stream) noexcept {
    if (released_ || poisoned_ || initialization_event_ == nullptr)
      return poisoned_ ? DeviceStatus::Poisoned : DeviceStatus::Invalid;
    const hipError_t status =
        hipStreamWaitEvent(stream, initialization_event_, 0U);
    if (status != hipSuccess) {
      last_error_ = status;
      poisoned_ = true;
      return DeviceStatus::HipError;
    }
    return DeviceStatus::Ok;
  }

  DeviceStatus release(hipStream_t stream) noexcept {
    if (released_)
      return DeviceStatus::Invalid;
    const hipError_t wait = hipStreamSynchronize(stream);
    if (wait != hipSuccess) {
      last_error_ = wait;
      poisoned_ = true;
      return DeviceStatus::HipError;
    }
    const hipError_t cleanup = cleanup_noexcept();
    if (cleanup != hipSuccess) {
      last_error_ = cleanup;
      poisoned_ = true;
      return DeviceStatus::HipError;
    }
    released_ = true;
    return DeviceStatus::Ok;
  }

private:
  friend class DevicePool;

  LogicalTable(DevicePool &pool, const std::size_t logical_blocks,
               hipStream_t stream)
      : pool_(pool), host_entries_(logical_blocks, kInvalidBlock) {
    if (logical_blocks == 0U || logical_blocks > pool.logical_capacity())
      throw std::invalid_argument("invalid paged KV logical table capacity");
    const hipError_t allocation =
        hipMalloc(reinterpret_cast<void **>(&device_entries_),
                  logical_blocks * sizeof(std::uint32_t));
    if (allocation != hipSuccess) {
      last_error_ = allocation;
      throw hip_exception("hipMalloc paged KV logical table", allocation);
    }
    try {
      for (Snapshot &snapshot : snapshots_) {
        snapshot.values.resize(logical_blocks);
        const std::size_t bytes = logical_blocks * sizeof(std::uint32_t);
        hipError_t pin =
            hipHostMalloc(reinterpret_cast<void **>(&snapshot.pinned_new),
                          bytes, hipHostMallocPortable);
        if (pin == hipSuccess)
          pin = hipHostMalloc(reinterpret_cast<void **>(&snapshot.pinned_old),
                              bytes, hipHostMallocPortable);
        if (pin != hipSuccess) {
          last_error_ = pin;
          throw hip_exception("hipHostMalloc paged KV table staging", pin);
        }
        const hipError_t event =
            hipEventCreateWithFlags(&snapshot.event, hipEventDisableTiming);
        if (event != hipSuccess) {
          last_error_ = event;
          throw hip_exception("hipEventCreate paged KV logical table", event);
        }
      }
      const hipError_t init_event = hipEventCreateWithFlags(
          &initialization_event_, hipEventDisableTiming);
      if (init_event != hipSuccess) {
        last_error_ = init_event;
        throw hip_exception("hipEventCreate paged KV logical table",
                            init_event);
      }
      const hipError_t clear =
          hipMemsetAsync(device_entries_, 0xff,
                         logical_blocks * sizeof(std::uint32_t), stream);
      if (clear != hipSuccess) {
        last_error_ = clear;
        throw hip_exception("hipMemsetAsync paged KV logical table", clear);
      }
      const hipError_t record = hipEventRecord(initialization_event_, stream);
      if (record != hipSuccess) {
        last_error_ = record;
        throw hip_exception("hipEventRecord paged KV logical table", record);
      }
    } catch (...) {
      (void)cleanup_noexcept();
      throw;
    }
    (void)stream;
  }

  static std::runtime_error hip_exception(const char *operation,
                                          const hipError_t error) {
    return std::runtime_error(std::string(operation) + ": " +
                              hipGetErrorString(error));
  }

  DeviceStatus rollback_enqueued(const std::vector<std::uint32_t> &queued,
                                 const std::size_t slot, hipStream_t stream,
                                 const hipError_t original) {
    bool restored = true;
    for (const std::uint32_t logical : queued) {
      const hipError_t status = hipMemcpyAsync(
          device_entries_ + logical, &snapshots_[slot].pinned_old[logical],
          sizeof(std::uint32_t), hipMemcpyHostToDevice, stream);
      if (status != hipSuccess) {
        last_error_ = status;
        restored = false;
      }
    }
    if (!restored)
      poisoned_ = true;
    const hipError_t sync = hipStreamSynchronize(stream);
    if (sync != hipSuccess) {
      last_error_ = sync;
      poisoned_ = true;
    }
    last_error_ = original;
    return poisoned_ ? DeviceStatus::Poisoned : DeviceStatus::HipError;
  }

  hipError_t cleanup_noexcept() noexcept {
    hipError_t first = hipSuccess;
    if (initialization_event_ != nullptr) {
      const hipError_t synchronize = hipEventSynchronize(initialization_event_);
      if (synchronize != hipSuccess && first == hipSuccess)
        first = synchronize;
      const hipError_t status = hipEventDestroy(initialization_event_);
      if (status == hipSuccess)
        initialization_event_ = nullptr;
      else if (first == hipSuccess)
        first = status;
    }
    for (Snapshot &snapshot : snapshots_) {
      if (snapshot.event != nullptr) {
        const hipError_t status = hipEventDestroy(snapshot.event);
        if (status == hipSuccess)
          snapshot.event = nullptr;
        else if (first == hipSuccess)
          first = status;
      }
      if (snapshot.pinned_new != nullptr) {
        const hipError_t status = hipHostFree(snapshot.pinned_new);
        if (status == hipSuccess)
          snapshot.pinned_new = nullptr;
        else if (first == hipSuccess)
          first = status;
      }
      if (snapshot.pinned_old != nullptr) {
        const hipError_t status = hipHostFree(snapshot.pinned_old);
        if (status == hipSuccess)
          snapshot.pinned_old = nullptr;
        else if (first == hipSuccess)
          first = status;
      }
    }
    if (device_entries_ != nullptr) {
      const hipError_t status = hipFree(device_entries_);
      if (status == hipSuccess)
        device_entries_ = nullptr;
      else if (first == hipSuccess)
        first = status;
    }
    if (first == hipSuccess && table_counted_) {
      pool_.table_destroyed();
      table_counted_ = false;
    }
    return first;
  }

  struct Snapshot final {
    std::vector<std::uint32_t> values;
    std::uint32_t *pinned_new = nullptr;
    std::uint32_t *pinned_old = nullptr;
    hipEvent_t event = nullptr;
  };

  DevicePool &pool_;
  std::vector<std::uint32_t> host_entries_;
  std::array<Snapshot, 2U> snapshots_;
  std::uint32_t *device_entries_ = nullptr;
  std::size_t pending_slot_ = 0U;
  hipError_t last_error_ = hipSuccess;
  hipEvent_t initialization_event_ = nullptr;
  bool pending_ = false;
  bool poisoned_ = false;
  bool released_ = false;
  bool table_counted_ = false;
};

inline std::unique_ptr<DevicePool::LogicalTable>
DevicePool::make_logical_table(const std::uint64_t logical_blocks,
                               hipStream_t stream) {
  if (released_ || poisoned_)
    throw std::logic_error("paged KV device pool is unavailable");
  if (logical_blocks == 0U || logical_blocks > plan_.logical_block_count())
    throw std::invalid_argument("invalid paged KV logical table capacity");
  if (logical_blocks > std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("paged KV logical table size overflow");
  std::unique_ptr<LogicalTable> table(new LogicalTable(
      *this, static_cast<std::size_t>(logical_blocks), stream));
  table_created();
  table->table_counted_ = true;
  return table;
}

} // namespace sllm_paged_kv

#endif // SLLM_PAGED_KV_DEVICE_POOL_HPP
