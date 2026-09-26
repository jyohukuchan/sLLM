#ifndef SLLM_KV_STATE_KERNEL_INTERNAL_HPP
#define SLLM_KV_STATE_KERNEL_INTERNAL_HPP

#include "decode_control_kernel_internal.hpp"
#include "paged_kv_device_layout.hpp"

#include "sllm/hip.h"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace sllm_kv_state_kernel {

constexpr const char *kPagedFp16LogicalKernelId =
    "kv_state.bf16_to_paged_f16.v1";
constexpr const char *kPagedFp16DeviceSymbol =
    "sllm_kv_state_bf16_to_paged_f16_v1";
constexpr const char *kPagedMxfp8E4LogicalKernelId =
    "kv_state.bf16_to_paged_mxfp8_e4.v1";
constexpr const char *kPagedMxfp8E4DeviceSymbol =
    "sllm_kv_state_bf16_to_paged_mxfp8_e4_v1";
constexpr const char *kPagedFp8LogicalKernelId =
    "kv_state.bf16_to_paged_fp8.v1";
constexpr const char *kPagedFp8DeviceSymbol =
    "sllm_kv_state_bf16_to_paged_fp8_v1";
constexpr const char *kPagedFp8StaticLogicalKernelId =
    "kv_state.bf16_to_paged_fp8_static.v1";
constexpr const char *kPagedFp8StaticDeviceSymbol =
    "sllm_kv_state_bf16_to_paged_fp8_static_v1";
constexpr const char *kPagedNvfp4LogicalKernelId =
    "kv_state.bf16_to_paged_nvfp4.v1";
constexpr const char *kPagedNvfp4DeviceSymbol =
    "sllm_kv_state_bf16_to_paged_nvfp4_v1";
constexpr const char *kPagedMxfp8E5LogicalKernelId =
    "kv_state.bf16_to_paged_mxfp8_e5.v1";
constexpr const char *kPagedMxfp8E5DeviceSymbol =
    "sllm_kv_state_bf16_to_paged_mxfp8_e5_v1";

// Fixed-window ring entry point.  The table has nine slots and `ring_tags`
// carries the absolute block number currently resident in each slot.  The
// tag check is device-side so a stale slot cannot be interpreted as a valid
// KV row after the absolute block number wraps modulo nine.
constexpr uint32_t kSlidingRingSlots = 9U;
constexpr uint64_t kSlidingWindowTokens = 1024U;
constexpr const char *kPagedFp8StaticSlidingLogicalKernelId =
    "kv_state.bf16_to_paged_fp8_static_sliding_ring.v1";
constexpr const char *kPagedFp8StaticSlidingDeviceSymbol =
    "sllm_kv_state_bf16_to_paged_fp8_static_sliding_ring_v1";

// Private Phase 54 research ABI. These constants intentionally do not enter
// include/sllm/hip.h: candidates are process-local evidence controls, not a
// public KV descriptor contract.
constexpr uint32_t kPhase54KvRecipeFloor = 0U;
constexpr uint32_t kPhase54KvRecipeCeilExponent = 1U;
constexpr uint32_t kPhase54KvRecipeNearestEvenExponent = 2U;
constexpr uint32_t kPhase54KvRecipeParent32Duplicate = 3U;
constexpr int32_t kPhase54KvResearchOk = 0;
constexpr int32_t kPhase54KvResearchInvalidRecipe = 1;
constexpr int32_t kPhase54KvResearchUnsupported = 2;

// Status values written by the paged append device path.  Zero is the
// success value and is intentionally independent from the decode-control
// status enum: the append status is a small, request-local device word that
// the host can inspect before publishing a state transaction.
constexpr uint32_t kPagedAppendStatusOk = 0U;
constexpr uint32_t kPagedAppendStatusInvalidPosition = 1U;
constexpr uint32_t kPagedAppendStatusInvalidTable = 2U;
constexpr uint32_t kPagedAppendStatusInvalidDescriptor = 3U;

/* Append BF16 K/V rows into a 128-token physical block table.  The descriptor
 * and logical table pointers are device pointers.  A descriptor owns the
 * starts of the key/value and optional scale planes for one physical block;
 * rows inside a block are laid out token-major, then KV head, then dimension.
 * The launcher clears device_status before dispatch and the kernel sets it on
 * an invalid logical-table entry or descriptor. */
hipError_t launch_paged(const uint16_t *key_input, const uint16_t *value_input,
                        const sllm_paged_kv::BlockDescriptor *block_descriptors,
                        const uint32_t *logical_table, uint32_t table_capacity,
                        uint32_t descriptor_capacity, uint32_t token_count,
                        uint64_t capacity_tokens, uint64_t start_position,
                        uint32_t head_count, uint32_t head_dim,
                        uint32_t encoding, uint32_t *device_status,
                        hipStream_t stream, float static_key_scale = 1.0F,
                        float static_value_scale = 1.0F) noexcept;

/* Graph replay variant.  `control` supplies the current phase position and
 * active row count at replay time.  token_count is the captured maximum row
 * count; phase_rows may be any value in [1, token_count]. */
hipError_t launch_paged_device(
    const uint16_t *key_input, const uint16_t *value_input,
    const sllm_paged_kv::BlockDescriptor *block_descriptors,
    const uint32_t *logical_table, uint32_t table_capacity,
    uint32_t descriptor_capacity, uint32_t token_count,
    uint64_t capacity_tokens, uint32_t head_count, uint32_t head_dim,
    uint32_t encoding, sllm_decode_control::ControlV1 *control,
    uint32_t *device_status, hipStream_t stream, float static_key_scale = 1.0F,
    float static_value_scale = 1.0F) noexcept;

hipError_t launch_paged_sliding_static_fp8(
    const uint16_t *key_input, const uint16_t *value_input,
    const sllm_paged_kv::BlockDescriptor *block_descriptors,
    const uint32_t *ring_table, const uint64_t *ring_tags,
    uint32_t ring_slot_count, uint32_t descriptor_capacity,
    uint32_t token_count, uint64_t retained_start, uint64_t start_position,
    uint32_t head_count, uint32_t head_dim, float static_key_scale,
    float static_value_scale, uint32_t *device_status,
    hipStream_t stream) noexcept;

} // namespace sllm_kv_state_kernel

// The recipe pair is process-wide. A caller must change it only when no KV
// append or attention work is in flight; the atomic snapshot prevents torn
// K/V selection, but it is not a stream synchronization mechanism.
extern "C" int32_t
sllm_phase54_kv_research_set_recipe_pair_v1(uint32_t key_recipe,
                                            uint32_t value_recipe) noexcept;
extern "C" int32_t
sllm_phase54_kv_research_get_recipe_pair_v1(uint32_t *key_recipe,
                                            uint32_t *value_recipe) noexcept;

#endif // SLLM_KV_STATE_KERNEL_INTERNAL_HPP
