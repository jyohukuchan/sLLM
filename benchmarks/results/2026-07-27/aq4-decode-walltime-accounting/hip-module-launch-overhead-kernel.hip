// SPDX-License-Identifier: Apache-2.0
// Minimal independently-loaded code object used only to measure hipModuleLaunchKernel overhead.
extern "C" __global__ void ullm_module_launch_noop_kernel() {}
