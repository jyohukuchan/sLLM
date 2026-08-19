#ifndef SLLM_CAUSAL_ATTENTION_KERNEL_INTERNAL_HPP
#define SLLM_CAUSAL_ATTENTION_KERNEL_INTERNAL_HPP

#include <hip/hip_runtime.h>

#include <cstdint>

namespace sllm_causal_attention_kernel {

constexpr const char *kLogicalKernelId =
    "causal_attention.online_softmax_gqa.v2";
constexpr const char *kDeviceSymbol =
    "sllm_causal_attention_online_softmax_gqa_v2";
constexpr const char *kPackedKvLogicalKernelId =
    "causal_attention.online_softmax_gqa.packed_kv.v3";
constexpr const char *kPackedKvDeviceSymbol =
    "sllm_causal_attention_online_softmax_gqa_packed_kv_v3";
constexpr const char *kGfx1201WaveLogicalKernelId =
    "causal_attention.online_softmax_gqa.gfx1201_wave.v4";
constexpr const char *kGfx1201WaveDeviceSymbol =
    "sllm_causal_attention_gfx1201_wave_v4";
constexpr const char *kGfx1201WavePackedKvLogicalKernelId =
    "causal_attention.online_softmax_gqa.packed_kv.gfx1201_wave.v4";
constexpr const char *kGfx1201WavePackedKvDeviceSymbol =
    "sllm_causal_attention_packed_gfx1201_wave_v4";

hipError_t launch(const uint16_t *query, const void *key, const void *value,
                  const void *key_scales, const void *value_scales,
                  const float *key_outer_scales,
                  const float *value_outer_scales, uint16_t *output,
                  uint32_t query_count, uint64_t capacity_tokens,
                  uint64_t start_position, uint64_t committed_kv_length,
                  uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                  uint32_t encoding, bool use_gfx1201_wave_provider,
                  hipStream_t stream) noexcept;

} // namespace sllm_causal_attention_kernel

#endif // SLLM_CAUSAL_ATTENTION_KERNEL_INTERNAL_HPP
