#ifndef SLLM_KV_STATE_KERNEL_INTERNAL_HPP
#define SLLM_KV_STATE_KERNEL_INTERNAL_HPP

#include "sllm/hip.h"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace sllm_kv_state_kernel {

constexpr const char *kLogicalKernelId = "kv_state.bf16_to_f16_token_major.v2";
constexpr const char *kDeviceSymbol =
    "sllm_kv_state_bf16_to_f16_token_major_v2";
constexpr const char *kFp8LogicalKernelId =
    "kv_state.bf16_to_fp8_token_major.v1";
constexpr const char *kFp8DeviceSymbol =
    "sllm_kv_state_bf16_to_fp8_token_major_v1";
constexpr const char *kFp8StaticLogicalKernelId =
    "kv_state.bf16_to_fp8_static_token_major.v1";
constexpr const char *kFp8StaticDeviceSymbol =
    "sllm_kv_state_bf16_to_fp8_token_major_v1";
constexpr const char *kNvfp4LogicalKernelId =
    "kv_state.bf16_to_nvfp4_token_major.v1";
constexpr const char *kNvfp4DeviceSymbol =
    "sllm_kv_state_bf16_to_nvfp4_token_major_v1";
constexpr const char *kFp8E4Block16LogicalKernelId =
    "kv_state.bf16_to_fp8_e4_block16_token_major.v1";
constexpr const char *kFp8E4Block16DeviceSymbol =
    "sllm_kv_state_bf16_to_fp8_block16_token_major_v1";
constexpr const char *kFp8E5Block16LogicalKernelId =
    "kv_state.bf16_to_fp8_e5_block16_token_major.v1";
constexpr const char *kFp8E5Block16DeviceSymbol =
    "sllm_kv_state_bf16_to_fp8_block16_token_major_v1";
constexpr const char *kFp8E4Block16LogicalKernelIdV2 =
    "kv_state.bf16_to_fp8_e4_block16_token_major.v2";
constexpr const char *kFp8E4Block16DeviceSymbolV2 =
    "sllm_kv_state_bf16_to_fp8_e4_block16_token_major_v2";
constexpr const char *kFp8E5Block16LogicalKernelIdV2 =
    "kv_state.bf16_to_fp8_e5_block16_token_major.v2";
constexpr const char *kFp8E5Block16DeviceSymbolV2 =
    "sllm_kv_state_bf16_to_fp8_e5_block16_token_major_v2";
constexpr const char *kMxfp8E4LogicalKernelId =
    "kv_state.bf16_to_mxfp8_e4_token_major.v1";
constexpr const char *kMxfp8E4DeviceSymbol =
    "sllm_kv_state_bf16_to_mxfp8_e4_token_major_v1";
constexpr const char *kMxfp8E5LogicalKernelId =
    "kv_state.bf16_to_mxfp8_e5_token_major.v1";
constexpr const char *kMxfp8E5DeviceSymbol =
    "sllm_kv_state_bf16_to_mxfp8_e5_token_major_v1";

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

hipError_t launch(const uint16_t *key_input, const uint16_t *value_input,
                  void *key_output, void *value_output, void *key_scales,
                  void *value_scales, float *key_outer_scales,
                  float *value_outer_scales, uint32_t token_count,
                  uint64_t capacity_tokens, uint64_t start_position,
                  uint32_t head_count, uint32_t head_dim, uint32_t encoding,
                  float static_key_scale, float static_value_scale,
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
