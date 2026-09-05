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
constexpr const char *kSlidingStaticFp8LogicalKernelId =
    "causal_attention.sliding_static_fp8_gqa.v1";
constexpr const char *kSlidingStaticFp8DeviceSymbol =
    "sllm_causal_attention_sliding_static_fp8_gqa_v1";
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
constexpr const char *kDecodeGqa6SplitP32LogicalKernelId =
    "causal_attention.decode.gqa6_split_p32.fp16.v1";
constexpr const char *kDecodeGqa6SplitP32DeviceSymbol =
    "sllm_causal_attention_decode_gqa6_split_p32_v1";
constexpr const char *kDecodeGqa6SplitP64LogicalKernelId =
    "causal_attention.decode.gqa6_split_p64.fp16.v1";
constexpr const char *kDecodeGqa6SplitP64DeviceSymbol =
    "sllm_causal_attention_decode_gqa6_split_p64_v1";
constexpr const char *kDecodeGqa6SplitP128LogicalKernelId =
    "causal_attention.decode.gqa6_split_p128.fp16.v1";
constexpr const char *kDecodeGqa6SplitP128DeviceSymbol =
    "sllm_causal_attention_decode_gqa6_split_p128_v1";
constexpr const char *kPrefillGqa4LogicalKernelId =
    "causal_attention.prefill.gqa4_shared.v6";
constexpr const char *kPrefillGqa4DeviceSymbol =
    "sllm_causal_attention_prefill_gqa4_shared_v6";
constexpr const char *kPrefillGqa4QTile4LogicalKernelId =
    "causal_attention.prefill.gqa4_qtile4.v7";
constexpr const char *kPrefillGqa4QTile4DeviceSymbol =
    "sllm_causal_attention_prefill_gqa4_qtile4_v7";
constexpr const char *kPrefillGqa6QTile4LogicalKernelId =
    "causal_attention.prefill.gqa6_qtile4.v1";
constexpr const char *kPrefillGqa6QTile4DeviceSymbol =
    "sllm_causal_attention_prefill_gqa6_qtile4_v1";
constexpr const char *kPrefillGqa6QTile4K4Fp16LogicalKernelId =
    "causal_attention.prefill.gqa6_qtile4_k4.fp16.v1";
constexpr const char *kPrefillGqa6QTile4K4Fp16DeviceSymbol =
    "sllm_causal_attention_prefill_gqa6_qtile4_k4_fp16_v1";
constexpr const char *kPrefillGqa6QTile4K8Fp16LogicalKernelId =
    "causal_attention.prefill.gqa6_qtile4_k8.fp16.v1";
constexpr const char *kPrefillGqa6QTile4K8Fp16DeviceSymbol =
    "sllm_causal_attention_prefill_gqa6_qtile4_k8_fp16_v1";
constexpr const char *kPrefillGqa6QTile4K16Fp16LogicalKernelId =
    "causal_attention.prefill.gqa6_qtile4_k16.fp16.v1";
constexpr const char *kPrefillGqa6QTile4K16Fp16DeviceSymbol =
    "sllm_causal_attention_prefill_gqa6_qtile4_k16_fp16_v1";
constexpr const char *kPrefillGqa6QTile4K32Fp16LogicalKernelId =
    "causal_attention.prefill.gqa6_qtile4_k32.fp16.v1";
constexpr const char *kPrefillGqa6QTile4K32Fp16DeviceSymbol =
    "sllm_causal_attention_prefill_gqa6_qtile4_k32_fp16_v1";
constexpr const char *kPrefillGqa6BlockSoftmaxFp16LogicalKernelId =
    "causal_attention.prefill.gqa6_blocksoftmax.fp16.v1";
constexpr const char *kPrefillGqa6BlockSoftmaxFp16DeviceSymbol =
    "sllm_causal_attention_prefill_gqa6_blocksoftmax_fp16_v1";
constexpr const char *kPrefillGqa6BlockSoftmaxQ8Fp16LogicalKernelId =
    "causal_attention.prefill.gqa6_blocksoftmax_q8.fp16.v1";
constexpr const char *kPrefillGqa6BlockSoftmaxQ8Fp16DeviceSymbol =
    "sllm_causal_attention_prefill_gqa6_blocksoftmax_q8_fp16_v1";
constexpr const char *kPrefillTypedQ4K4LogicalKernelId =
    "causal_attention.prefill.typed_q4k4.v1";
constexpr const char *kPrefillTypedQ4K4DeviceSymbol =
    "sllm_causal_attention_prefill_typed_q4k4_v1";
constexpr const char *kPrefillTypedQ4K8LogicalKernelId =
    "causal_attention.prefill.typed_q4k8.v1";
constexpr const char *kPrefillTypedQ4K8DeviceSymbol =
    "sllm_causal_attention_prefill_typed_q4k8_v1";
constexpr const char *kPrefillTypedQ8K8LogicalKernelId =
    "causal_attention.prefill.typed_q8k8.v1";
constexpr const char *kPrefillTypedQ8K8DeviceSymbol =
    "sllm_causal_attention_prefill_typed_q8k8_v1";
constexpr const char *kScaledPrefillGemmLogicalKernelId =
    "causal_attention.prefill.gfx1030_hipblas_scaled_fp16.v1";
constexpr const char *kScaledPrefillGemmDeviceSymbol =
    "sllm_causal_attention_prefill_gfx1030_hipblas_scaled_fp16_v1";
constexpr const char *kGqa6RocblasF32LogicalKernelId =
    "causal_attention.prefill.gfx1030_rocblas_gqa6_f32.v1";
constexpr const char *kGqa6RocblasF32DeviceSymbol =
    "sllm_causal_attention_prefill_gfx1030_rocblas_gqa6_f32_v1";
constexpr const char *kGfx1201Gqa6RocblasF32LogicalKernelId =
    "causal_attention.prefill.gfx1201_rocblas_gqa6_f32.v1";
constexpr const char *kGfx1201Gqa6RocblasF32DeviceSymbol =
    "sllm_causal_attention_prefill_gfx1201_rocblas_gqa6_f32_v1";
constexpr const char *kGfx1201Gqa6RocblasF16TailLogicalKernelId =
    "causal_attention.prefill.gfx1201_rocblas_gqa6_f16_tail.v1";
constexpr const char *kGfx1201Gqa6RocblasF16TailDeviceSymbol =
    "sllm_causal_attention_prefill_gfx1201_rocblas_gqa6_f16_tail_v1";
constexpr const char *kLongPrefillV2LogicalKernelId =
    "causal_attention.prefill.gfx1030_qtile8_split.v2";
constexpr const char *kLongPrefillV2DeviceSymbol =
    "sllm_causal_attention_prefill_gfx1030_qtile8_split_v2";

enum class PrefillTilePolicy : uint32_t {
  Q4K1Control = 0U,
  Q4K4 = 1U,
  Q4K8 = 2U,
  Q8K8 = 3U,
};

constexpr uint32_t prefill_query_tile(const PrefillTilePolicy policy) noexcept {
  return policy == PrefillTilePolicy::Q8K8 ? 8U : 4U;
}

hipError_t launch_typed_prefill(
    const uint16_t *query, const void *key, const void *value,
    const void *key_scales, const void *value_scales,
    const float *key_outer_scales, const float *value_outer_scales,
    uint16_t *output, uint32_t query_count, uint64_t start_position,
    uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, uint32_t encoding,
    float static_key_scale, float static_value_scale, PrefillTilePolicy policy,
    hipStream_t stream) noexcept;

hipError_t
launch(const uint16_t *query, const void *key, const void *value,
       const void *key_scales, const void *value_scales,
       const float *key_outer_scales, const float *value_outer_scales,
       uint16_t *output, uint32_t query_count, uint64_t capacity_tokens,
       uint64_t start_position, uint64_t committed_kv_length, uint32_t q_heads,
       uint32_t kv_heads, uint32_t head_dim, uint32_t encoding,
       float static_key_scale, float static_value_scale,
       bool use_gfx1201_wave_provider, bool use_decode_wave_split,
       bool use_decode_wave_split_q_preload, bool use_prefill_gqa4,
       bool use_prefill_gqa4_qtile4, bool use_prefill_gqa6_qtile4_k32_fp16,
       uint64_t sliding_window, float score_scale, hipStream_t stream) noexcept;

hipError_t
launch_gqa6_qtile4_fp16(const uint16_t *query, const void *key,
                        const void *value, const void *key_scales,
                        const void *value_scales, const float *key_outer_scales,
                        const float *value_outer_scales, uint16_t *output,
                        uint32_t query_count, uint64_t start_position,
                        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                        float static_key_scale, float static_value_scale,
                        uint32_t key_tile, hipStream_t stream) noexcept;

hipError_t launch_gqa6_blocksoftmax_fp16(
    const uint16_t *query, const void *key, const void *value,
    const void *key_scales, const void *value_scales,
    const float *key_outer_scales, const float *value_outer_scales,
    uint16_t *output, uint32_t query_count, uint64_t start_position,
    uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
    float static_key_scale, float static_value_scale, uint32_t key_tile,
    hipStream_t stream) noexcept;

hipError_t launch_gqa6_blocksoftmax_q8_fp16(
    const uint16_t *query, const void *key, const void *value,
    const void *key_scales, const void *value_scales,
    const float *key_outer_scales, const float *value_outer_scales,
    uint16_t *output, uint32_t query_count, uint64_t start_position,
    uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
    float static_key_scale, float static_value_scale,
    hipStream_t stream) noexcept;

constexpr const char *kScaledStaticFp8LogicalKernelId =
    "causal_attention.scaled_static_fp8_gqa.v1";
constexpr const char *kScaledStaticFp8DeviceSymbol =
    "sllm_causal_attention_scaled_static_fp8_gqa_v1";

hipError_t launch_scaled_prefill_gemm(
    const uint16_t *query, const void *key, const void *value,
    const void *key_scales, const void *value_scales,
    const float *key_outer_scales, const float *value_outer_scales,
    uint16_t *output, uint32_t query_count, uint64_t start_position,
    uint64_t committed_kv_length, uint32_t q_heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t encoding, float static_key_scale,
    float static_value_scale, void *workspace, uint64_t workspace_bytes,
    void *blas_handle, void *blas_mutex, hipStream_t stream) noexcept;

// Qwen3.8 GQA6 FP16-KV prefill feasibility provider for the gfx1030 and
// gfx1201 literal opt-ins.  The caller owns the persistent rocBLAS handle/mutex
// and a workspace of at least
// gqa6_rocblas_f32_workspace_bytes(query_count, committed_kv_length).
// Workspace layout is query F32 pack, interleaved K/V F32 staging, F32 scores,
// and F32 PV.  This is intentionally force-only; callers retain qtile4/K32 as
// the rollback path for every shape outside the exact contract.  gfx1201 uses
// the separately audited dispatch ID 74 while gfx1030 keeps its original ID.
uint64_t
gqa6_rocblas_f32_workspace_bytes(uint32_t query_count,
                                 uint64_t committed_kv_length) noexcept;

hipError_t launch_gqa6_rocblas_f32(
    const uint16_t *query, const void *key, const void *value, uint16_t *output,
    uint32_t query_count, uint64_t start_position, uint64_t committed_kv_length,
    uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, uint32_t encoding,
    void *workspace, uint64_t workspace_bytes, void *blas_handle,
    void *blas_mutex, hipStream_t stream) noexcept;

// FP16-score tail variant for gfx1201.  The query is converted from resident
// BF16 to FP16, QK/PV use FP32 rocBLAS accumulation, and only the score
// matrix remains FP16.  It is intended solely for start_position>0 chunks;
// the first self-context chunk stays on the ID74 F32-score provider.
uint64_t
gqa6_rocblas_f16_tail_workspace_bytes(uint32_t query_count,
                                      uint64_t committed_kv_length) noexcept;

hipError_t launch_gqa6_rocblas_f16_tail(
    const uint16_t *query, const void *key, const void *value, uint16_t *output,
    uint32_t query_count, uint64_t start_position, uint64_t committed_kv_length,
    uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, uint32_t encoding,
    void *workspace, uint64_t workspace_bytes, void *blas_handle,
    void *blas_mutex, hipStream_t stream) noexcept;

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

hipError_t launch_decode_gqa6_split_p32(
    const uint16_t *query, const void *key, const void *value,
    const void *key_scales, const void *value_scales,
    const float *key_outer_scales, const float *value_outer_scales,
    uint16_t *output, uint32_t query_count, uint64_t start_position,
    uint64_t committed_kv_length, uint32_t q_heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t encoding, float static_key_scale,
    float static_value_scale, void *workspace, uint64_t workspace_bytes,
    hipStream_t stream) noexcept;

hipError_t launch_decode_gqa6_split_p64(
    const uint16_t *query, const void *key, const void *value,
    const void *key_scales, const void *value_scales,
    const float *key_outer_scales, const float *value_outer_scales,
    uint16_t *output, uint32_t query_count, uint64_t start_position,
    uint64_t committed_kv_length, uint32_t q_heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t encoding, float static_key_scale,
    float static_value_scale, void *workspace, uint64_t workspace_bytes,
    hipStream_t stream) noexcept;

hipError_t launch_decode_gqa6_split_p128(
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
