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
constexpr const char *kDecodeWaveSplitLogicalKernelId =
    "causal_attention.decode.wave8_split.v5";
constexpr const char *kDecodeWaveSplitDeviceSymbol =
    "sllm_causal_attention_decode_wave8_split_v5";
constexpr const char *kDecodeWaveSplitQPreloadLogicalKernelId =
    "causal_attention.decode.wave8_split.q_preload.v1";
constexpr const char *kDecodeWaveSplitQPreloadDeviceSymbol =
    "sllm_causal_attention_decode_wave8_split_q_preload_v1";
constexpr const char *kDecodeWaveSplitFp16PairLogicalKernelId =
    "causal_attention.decode.wave8_split.fp16_pair.v1";
constexpr const char *kDecodeWaveSplitFp16PairDeviceSymbol =
    "sllm_causal_attention_decode_wave8_split_fp16_pair_v1";
constexpr const char *kDecodeGqa4SplitLogicalKernelId =
    "causal_attention.decode.gqa4_tiled_split.v1";
constexpr const char *kDecodeGqa4SplitDeviceSymbol =
    "sllm_causal_attention_decode_gqa4_tiled_split_v1";
constexpr const char *kDecodeGqa4SplitP32LogicalKernelId =
    "causal_attention.decode.gqa4_tiled_split.p32.v1";
constexpr const char *kDecodeGqa4SplitP32DeviceSymbol =
    "sllm_causal_attention_decode_gqa4_split_p32_v1";
constexpr const char *kPrefillGqa4LogicalKernelId =
    "causal_attention.prefill.gqa4_shared.v6";
constexpr const char *kPrefillGqa4DeviceSymbol =
    "sllm_causal_attention_prefill_gqa4_shared_v6";
constexpr const char *kPrefillGqa4QTile4LogicalKernelId =
    "causal_attention.prefill.gqa4_qtile4.v7";
constexpr const char *kPrefillGqa4QTile4DeviceSymbol =
    "sllm_causal_attention_prefill_gqa4_qtile4_v7";
constexpr const char *kScaledPrefillGemmLogicalKernelId =
    "causal_attention.prefill.gfx1030_hipblas_scaled_fp16.v1";
constexpr const char *kScaledPrefillGemmDeviceSymbol =
    "sllm_causal_attention_prefill_gfx1030_hipblas_scaled_fp16_v1";
constexpr const char *kLongPrefillV2LogicalKernelId =
    "causal_attention.prefill.gfx1030_qtile8_split.v2";
constexpr const char *kLongPrefillV2DeviceSymbol =
    "sllm_causal_attention_prefill_gfx1030_qtile8_split_v2";

hipError_t launch(const uint16_t *query, const void *key, const void *value,
                  const void *key_scales, const void *value_scales,
                  const float *key_outer_scales,
                  const float *value_outer_scales, uint16_t *output,
                  uint32_t query_count, uint64_t capacity_tokens,
                  uint64_t start_position, uint64_t committed_kv_length,
                  uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                  uint32_t encoding, float static_key_scale,
                  float static_value_scale, bool use_gfx1201_wave_provider,
                  bool use_decode_wave_split,
                  bool use_decode_wave_split_q_preload, bool use_prefill_gqa4,
                  bool use_prefill_gqa4_qtile4, hipStream_t stream) noexcept;

hipError_t launch_scaled_prefill_gemm(
    const uint16_t *query, const void *key, const void *value, uint16_t *output,
    uint32_t query_count, uint64_t start_position, uint64_t committed_kv_length,
    uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, void *workspace,
    uint64_t workspace_bytes, void *blas_handle, void *blas_mutex,
    hipStream_t stream) noexcept;

hipError_t launch_long_prefill_v2(
    const uint16_t *query, const void *key, const void *value, uint16_t *output,
    uint32_t query_count, uint64_t start_position, uint64_t committed_kv_length,
    uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, void *workspace,
    uint64_t workspace_bytes, hipStream_t stream) noexcept;

hipError_t launch_decode_wave_split_fp16_pair(
    const uint16_t *query, const void *key, const void *value,
    const void *key_scales, const void *value_scales,
    const float *key_outer_scales, const float *value_outer_scales,
    uint16_t *output, uint32_t query_count, uint64_t start_position,
    uint64_t committed_kv_length, uint32_t q_heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t encoding, float static_key_scale,
    float static_value_scale, hipStream_t stream) noexcept;

hipError_t launch_decode_gqa4_split(
    const uint16_t *query, const void *key, const void *value,
    const void *key_scales, const void *value_scales,
    const float *key_outer_scales, const float *value_outer_scales,
    uint16_t *output, uint32_t query_count, uint64_t start_position,
    uint64_t committed_kv_length, uint32_t q_heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t encoding, float static_key_scale,
    float static_value_scale, void *workspace, uint64_t workspace_bytes,
    hipStream_t stream) noexcept;

hipError_t launch_decode_gqa4_split_p32(
    const uint16_t *query, const void *key, const void *value,
    const void *key_scales, const void *value_scales,
    const float *key_outer_scales, const float *value_outer_scales,
    uint16_t *output, uint32_t query_count, uint64_t start_position,
    uint64_t committed_kv_length, uint32_t q_heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t encoding, float static_key_scale,
    float static_value_scale, void *workspace, uint64_t workspace_bytes,
    hipStream_t stream) noexcept;

} // namespace sllm_causal_attention_kernel

#endif // SLLM_CAUSAL_ATTENTION_KERNEL_INTERNAL_HPP
