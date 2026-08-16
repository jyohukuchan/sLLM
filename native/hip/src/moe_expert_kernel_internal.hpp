#ifndef SLLM_MOE_EXPERT_KERNEL_INTERNAL_HPP
#define SLLM_MOE_EXPERT_KERNEL_INTERNAL_HPP

#include <hip/hip_runtime.h>

#include <cstdint>

namespace sllm_moe_expert_kernel {

inline constexpr uint64_t kHidden = 2048U;
inline constexpr uint64_t kIntermediate = 512U;
inline constexpr uint64_t kExperts = 256U;
inline constexpr uint64_t kTopK = 8U;
inline constexpr uint64_t kGateValuesOffset = 0U;
inline constexpr uint64_t kGateScalesOffset = 134217728U;
inline constexpr uint64_t kUpValuesOffset = 142606336U;
inline constexpr uint64_t kUpScalesOffset = 276824064U;
inline constexpr uint64_t kDownValuesOffset = 285212672U;
inline constexpr uint64_t kDownScalesOffset = 419430400U;
inline constexpr uint64_t kSharedGateOffset = 427819008U;
inline constexpr uint64_t kSharedUpOffset = 429916160U;
inline constexpr uint64_t kSharedDownOffset = 432013312U;
inline constexpr uint64_t kSharedExpertGateOffset = 434110464U;
inline constexpr uint64_t kLayerBlobBytes = 434114560U;

inline constexpr const char *kDecodeLogicalKernelId =
    "moe_expert.mxfp4.decode.active8_shared.v1";
inline constexpr const char *kPrefillLogicalKernelId =
    "moe_expert.mxfp4.prefill.grouped_shared.v1";

struct Workspace final {
  uint8_t *activation_values;
  uint8_t *activation_scales;
  uint16_t *routed_intermediate;
  uint8_t *intermediate_values;
  uint8_t *intermediate_scales;
  uint16_t *shared_intermediate;
  float *shared_gate;
};

uint64_t workspace_bytes(uint64_t token_count) noexcept;

hipError_t launch(const uint16_t *hidden, const uint8_t *routing_metadata,
                  const uint8_t *layer_blob, uint8_t *workspace,
                  uint16_t *output, uint64_t token_count,
                  hipStream_t stream) noexcept;

} // namespace sllm_moe_expert_kernel

#endif // SLLM_MOE_EXPERT_KERNEL_INTERNAL_HPP
