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

namespace gemma4 {

inline constexpr uint64_t kHidden = 2816U;
inline constexpr uint64_t kIntermediate = 704U;
inline constexpr uint64_t kExperts = 128U;
inline constexpr uint64_t kTopK = 8U;
inline constexpr uint64_t kBlockSize = 16U;
inline constexpr uint64_t kPackedWeightBytesPerExpert = 991232U;
inline constexpr uint64_t kBlockScaleBytesPerExpert = 123904U;
inline constexpr uint64_t kPlaneValuesBytes =
    kPackedWeightBytesPerExpert * kExperts;
inline constexpr uint64_t kPlaneScalesBytes =
    kBlockScaleBytesPerExpert * kExperts;
inline constexpr uint64_t kExpertF32ScalesBytes = kExperts * sizeof(float);
inline constexpr uint64_t kPlaneBytes =
    kPlaneValuesBytes + kPlaneScalesBytes + 2U * kExpertF32ScalesBytes;

inline constexpr uint64_t kGateValuesOffset = 0U;
inline constexpr uint64_t kGateScalesOffset =
    kGateValuesOffset + kPlaneValuesBytes;
inline constexpr uint64_t kGateOuterScalesOffset =
    kGateScalesOffset + kPlaneScalesBytes;
inline constexpr uint64_t kGateInputScalesOffset =
    kGateOuterScalesOffset + kExpertF32ScalesBytes;
inline constexpr uint64_t kUpValuesOffset = kGateValuesOffset + kPlaneBytes;
inline constexpr uint64_t kUpScalesOffset = kUpValuesOffset + kPlaneValuesBytes;
inline constexpr uint64_t kUpOuterScalesOffset =
    kUpScalesOffset + kPlaneScalesBytes;
inline constexpr uint64_t kUpInputScalesOffset =
    kUpOuterScalesOffset + kExpertF32ScalesBytes;
inline constexpr uint64_t kDownValuesOffset = kUpValuesOffset + kPlaneBytes;
inline constexpr uint64_t kDownScalesOffset =
    kDownValuesOffset + kPlaneValuesBytes;
inline constexpr uint64_t kDownOuterScalesOffset =
    kDownScalesOffset + kPlaneScalesBytes;
inline constexpr uint64_t kDownInputScalesOffset =
    kDownOuterScalesOffset + kExpertF32ScalesBytes;
inline constexpr uint64_t kPerExpertScalesOffset =
    kDownValuesOffset + kPlaneBytes;
inline constexpr uint64_t kLayerBlobBytes =
    kPerExpertScalesOffset + kExperts * sizeof(uint16_t);

inline constexpr uint64_t kActivationValueBytesPerPair = kHidden / 2U;
inline constexpr uint64_t kActivationScaleBytesPerPair = kHidden / kBlockSize;
inline constexpr uint64_t kIntermediateBytesPerPair =
    kIntermediate * sizeof(uint16_t);
inline constexpr uint64_t kIntermediateValueBytesPerPair = kIntermediate / 2U;
inline constexpr uint64_t kIntermediateScaleBytesPerPair =
    kIntermediate / kBlockSize;
inline constexpr uint64_t kWorkspaceBytesPerToken =
    kTopK * (kActivationValueBytesPerPair + kActivationScaleBytesPerPair +
             kIntermediateBytesPerPair + kIntermediateValueBytesPerPair +
             kIntermediateScaleBytesPerPair);

inline constexpr const char *kDecodeLogicalKernelId =
    "moe_expert.gemma4.nvfp4.decode.active8.v2";
inline constexpr const char *kPrefillLogicalKernelId =
    "moe_expert.gemma4.nvfp4.prefill.active8.v2";
inline constexpr const char *kDeviceSymbol =
    "sllm_gemma4_moe_expert_active8_nvfp4_v2";

static_assert(kPlaneValuesBytes == UINT64_C(126877696));
static_assert(kPlaneScalesBytes == UINT64_C(15859712));
static_assert(kPlaneBytes == UINT64_C(142738432));
static_assert(kGateScalesOffset == UINT64_C(126877696));
static_assert(kGateOuterScalesOffset == UINT64_C(142737408));
static_assert(kGateInputScalesOffset == UINT64_C(142737920));
static_assert(kUpValuesOffset == UINT64_C(142738432));
static_assert(kUpScalesOffset == UINT64_C(269616128));
static_assert(kUpOuterScalesOffset == UINT64_C(285475840));
static_assert(kUpInputScalesOffset == UINT64_C(285476352));
static_assert(kDownValuesOffset == UINT64_C(285476864));
static_assert(kDownScalesOffset == UINT64_C(412354560));
static_assert(kDownOuterScalesOffset == UINT64_C(428214272));
static_assert(kDownInputScalesOffset == UINT64_C(428214784));
static_assert(kPerExpertScalesOffset == UINT64_C(428215296));
static_assert(kLayerBlobBytes == UINT64_C(428215552));
static_assert(kWorkspaceBytesPerToken == UINT64_C(27104));

} // namespace gemma4

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

uint64_t gemma4_workspace_bytes(uint64_t token_count) noexcept;

hipError_t launch_gemma4(const uint16_t *hidden,
                         const uint8_t *routing_metadata,
                         const uint8_t *layer_blob, uint8_t *workspace,
                         uint16_t *output, uint64_t token_count,
                         hipStream_t stream) noexcept;

} // namespace sllm_moe_expert_kernel

#endif // SLLM_MOE_EXPERT_KERNEL_INTERNAL_HPP
