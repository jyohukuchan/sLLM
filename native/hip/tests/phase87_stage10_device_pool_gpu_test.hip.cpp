#include "../src/paged_kv_device_pool.hpp"

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using sllm_paged_kv::DevicePool;
using sllm_paged_kv::DeviceStatus;
using sllm_paged_kv::kInvalidBlock;
using sllm_paged_kv::LogicalEntryUpdate;
using sllm_paged_kv::SlabPlan;
using sllm_paged_kv::SlabPlanInput;

void check(const hipError_t status, const char *const operation) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
}

void expect(const bool condition, const char *const message) {
  if (!condition)
    throw std::runtime_error(message);
}

void expect_status(const DeviceStatus actual, const DeviceStatus expected,
                   const char *const operation) {
  if (actual != expected)
    throw std::runtime_error(operation);
}

__global__ void read_paged_entries(const sllm_paged_kv::BlockDescriptor *desc,
                                   const std::uint32_t *logical,
                                   std::uint8_t *output) {
  if (blockIdx.x != 0U || threadIdx.x != 0U)
    return;
  const std::uint32_t first = logical[0];
  const std::uint32_t second = logical[1];
  const sllm_paged_kv::BlockDescriptor first_desc = desc[first];
  const sllm_paged_kv::BlockDescriptor second_desc = desc[second];
  output[0] = first_desc.key[0];
  output[1] = first_desc.value[0];
  output[2] = first_desc.key_scale[0];
  output[3] = first_desc.value_scale[0];
  output[4] = first_desc.key_outer_scale[0];
  output[5] = first_desc.value_outer_scale[0];
  output[6] = second_desc.key[0];
  output[7] = second_desc.value[0];
}

void fill_block(const DevicePool &pool, const std::uint32_t physical,
                hipStream_t stream, const std::uint8_t salt) {
  const auto &descriptor = pool.host_descriptor(physical);
  const std::array<std::uint8_t *, 6U> planes = {descriptor.key,
                                                 descriptor.value,
                                                 descriptor.key_scale,
                                                 descriptor.value_scale,
                                                 descriptor.key_outer_scale,
                                                 descriptor.value_outer_scale};
  for (std::size_t plane = 0U; plane < planes.size(); ++plane) {
    check(hipMemsetAsync(planes[plane], static_cast<int>(salt + plane),
                         pool.plan().plane_bytes_per_block()[plane], stream),
          "hipMemsetAsync block plane");
  }
}

} // namespace

int main() {
  try {
    check(hipSetDevice(0), "hipSetDevice");
    hipStream_t stream = nullptr;
    check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
          "hipStreamCreate");

    // A five-block pool still commits one complete eight-slot slab.  Only
    // five physical IDs are addressable, while the byte query reports all
    // eight allocated slots for each of the six planes.
    const SlabPlan partial_plan =
        SlabPlan::make(SlabPlanInput{1U, 1U, 1U, 256U, 5U});
    DevicePool partial_pool(partial_plan);
    expect(partial_pool.allocated_slab_count() == 0U,
           "partial pool starts with an allocated slab");
    expect(partial_pool.allocated_physical_blocks() == 0U,
           "partial pool starts with allocated physical IDs");
    expect(partial_pool.committed_bytes_total() == 0U,
           "partial pool starts with committed bytes");
    expect_status(partial_pool.ensure_blocks({4U}, stream), DeviceStatus::Ok,
                  "partial pool allocation failed");
    expect(partial_pool.allocated_slab_count() == 1U,
           "partial pool did not allocate one slab");
    expect(partial_pool.allocated_physical_blocks() == 5U,
           "partial pool reported non-addressable physical slots");
    const auto partial_bytes = partial_pool.committed_bytes_per_plane();
    for (const std::uint64_t bytes : partial_bytes)
      expect(bytes == 8U * 128U,
             "partial pool did not count all eight allocated slots");
    expect(partial_pool.committed_bytes_total() == 6U * 8U * 128U,
           "partial pool total committed bytes mismatch");
    expect_status(partial_pool.release(stream), DeviceStatus::Ok,
                  "partial pool release failed");

    const SlabPlan plan = SlabPlan::make(SlabPlanInput{1U, 1U, 1U, 256U, 10U});
    DevicePool pool(plan);
    auto table = pool.make_logical_table(2U, stream);
    auto forked_table = pool.make_logical_table(2U, stream);
    const auto *const descriptor_pointer = pool.device_descriptor_table();
    const auto *const table_pointer = table->device_table();
    expect(descriptor_pointer != nullptr && table_pointer != nullptr,
           "stable device pointers were null");
    expect(forked_table->device_table() != table_pointer,
           "forked state reused the logical table pointer");
    expect(descriptor_pointer == pool.device_descriptor_table(),
           "descriptor table pointer changed before allocation");

    expect_status(pool.ensure_blocks({0U, 8U, 9U}, stream), DeviceStatus::Ok,
                  "ensure_blocks failed");
    expect(pool.descriptor_ready(0U) && pool.descriptor_ready(8U) &&
               pool.descriptor_ready(9U),
           "descriptor publication state mismatch");
    expect(pool.allocated_slab_count() == 2U &&
               pool.allocated_physical_blocks() == 10U,
           "allocated pool metrics mismatch");
    const auto committed_bytes = pool.committed_bytes_per_plane();
    for (const std::uint64_t bytes : committed_bytes)
      expect(bytes == 16U * 128U, "allocated plane byte count mismatch");
    expect(pool.committed_bytes_total() == 6U * 16U * 128U,
           "allocated total byte count mismatch");
    expect(descriptor_pointer == pool.device_descriptor_table(),
           "descriptor table pointer changed after slab allocation");
    fill_block(pool, 8U, stream, 0x80U);
    fill_block(pool, 0U, stream, 0x10U);

    const std::vector<LogicalEntryUpdate> publish = {{0U, kInvalidBlock, 8U},
                                                     {1U, kInvalidBlock, 0U}};
    expect_status(table->update_entries(publish, stream), DeviceStatus::Ok,
                  "logical entry update failed");
    check(hipStreamSynchronize(stream), "logical update synchronize");
    expect_status(table->poll_pending(), DeviceStatus::Ok,
                  "logical update completion failed");

    std::uint8_t *device_output = nullptr;
    check(hipMalloc(reinterpret_cast<void **>(&device_output), 8U),
          "hipMalloc output");
    read_paged_entries<<<1U, 1U, 0U, stream>>>(
        pool.device_descriptor_table(), table->device_table(), device_output);
    check(hipGetLastError(), "read_paged_entries launch");
    std::array<std::uint8_t, 8U> output{};
    check(hipMemcpyAsync(output.data(), device_output, output.size(),
                         hipMemcpyDeviceToHost, stream),
          "hipMemcpyAsync output");
    check(hipStreamSynchronize(stream), "output synchronize");
    check(hipFree(device_output), "hipFree output");
    expect(output[0] == 0x80U && output[1] == 0x81U && output[2] == 0x82U &&
               output[3] == 0x83U && output[4] == 0x84U && output[5] == 0x85U &&
               output[6] == 0x10U && output[7] == 0x11U,
           "descriptor indirection returned the wrong plane bytes");

    const std::vector<LogicalEntryUpdate> stale = {{0U, kInvalidBlock, 9U}};
    expect_status(table->update_entries(stale, stream), DeviceStatus::Invalid,
                  "stale logical entry was accepted");
    expect(table->host_entry(0U) == 8U,
           "stale logical entry changed the host mirror");

    expect_status(table->restore_entries(publish, stream), DeviceStatus::Ok,
                  "logical entry restore failed");
    check(hipStreamSynchronize(stream), "logical restore synchronize");
    expect_status(table->poll_pending(), DeviceStatus::Ok,
                  "logical restore completion failed");
    expect(table->host_entry(0U) == kInvalidBlock &&
               table->host_entry(1U) == kInvalidBlock,
           "logical clear did not update the host mirror");

    expect_status(pool.release(stream), DeviceStatus::Busy,
                  "device pool released while logical tables were alive");
    expect_status(forked_table->release(stream), DeviceStatus::Ok,
                  "forked logical table release failed");
    expect_status(table->release(stream), DeviceStatus::Ok,
                  "logical table release failed");
    expect_status(pool.release(stream), DeviceStatus::Ok,
                  "device pool release failed");
    check(hipStreamDestroy(stream), "hipStreamDestroy");
    std::cout << "phase87_stage10_device_pool_gpu_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "phase87_stage10_device_pool_gpu_test: FAIL: " << error.what()
              << '\n';
    return 1;
  }
}
