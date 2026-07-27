#include <hip/hip_runtime_api.h>

#include <dlfcn.h>
#include <cstdio>

using LaunchFn = hipError_t (*)(const void*, dim3, dim3, void**, size_t, hipStream_t);

extern "C" hipError_t hipLaunchKernel(const void* function_address,
                                       dim3 grid,
                                       dim3 block,
                                       void** args,
                                       size_t shared_bytes,
                                       hipStream_t stream) {
    static auto real_launch = reinterpret_cast<LaunchFn>(dlsym(RTLD_NEXT, "hipLaunchKernel"));
    hipFuncAttributes attr{};
    int active_blocks = 0;
    int device = 0;
    hipDeviceProp_t props{};
    const auto attrs_status = hipFuncGetAttributes(&attr, function_address);
    const auto occupancy_status = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, function_address, static_cast<int>(block.x * block.y * block.z), shared_bytes);
    const auto device_status = hipGetDevice(&device);
    const auto props_status = hipGetDeviceProperties(&props, device);
    const char* name = hipKernelNameRefByPtr(function_address, stream);
    std::fprintf(stderr,
                 "ULLM_HIP_LAUNCH_OCCUPANCY kernel=%s grid=%u,%u,%u block=%u,%u,%u "
                 "dynamic_shared_bytes=%zu num_regs=%d static_shared_bytes=%zu max_threads=%d "
                 "active_blocks_per_cu=%d waves_per_block=%u active_waves_per_cu=%u warp_size=%d "
                 "attrs_status=%d occupancy_status=%d device_status=%d props_status=%d\n",
                 name ? name : "<unknown>", grid.x, grid.y, grid.z, block.x, block.y, block.z,
                 shared_bytes, attr.numRegs, attr.sharedSizeBytes, attr.maxThreadsPerBlock,
                 active_blocks, (block.x * block.y * block.z) / props.warpSize,
                 active_blocks * ((block.x * block.y * block.z) / props.warpSize), props.warpSize,
                 static_cast<int>(attrs_status), static_cast<int>(occupancy_status),
                 static_cast<int>(device_status), static_cast<int>(props_status));
    return real_launch(function_address, grid, block, args, shared_bytes, stream);
}
