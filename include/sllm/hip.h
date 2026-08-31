#ifndef SLLM_HIP_H
#define SLLM_HIP_H

#include <stdint.h>

#if defined(_WIN32) && defined(SLLM_HIP_SHARED)
#if defined(SLLM_HIP_BUILD)
#define SLLM_HIP_API __declspec(dllexport)
#else
#define SLLM_HIP_API __declspec(dllimport)
#endif
#else
#define SLLM_HIP_API
#endif

#ifdef __cplusplus
#define SLLM_HIP_NOEXCEPT noexcept
#else
#define SLLM_HIP_NOEXCEPT
#endif

#define SLLM_HIP_ABI_VERSION UINT32_C(1)
#define SLLM_HIP_LIBRARY_VERSION_MAJOR UINT32_C(0)
#define SLLM_HIP_LIBRARY_VERSION_MINOR UINT32_C(1)
#define SLLM_HIP_LIBRARY_VERSION_PATCH UINT32_C(0)

typedef uint32_t sllm_status_t;

#define SLLM_STATUS_OK UINT32_C(0)
#define SLLM_STATUS_INVALID_ARGUMENT UINT32_C(1)
#define SLLM_STATUS_BUFFER_TOO_SMALL UINT32_C(2)
#define SLLM_STATUS_UNSUPPORTED UINT32_C(3)
#define SLLM_STATUS_HIP_UNAVAILABLE UINT32_C(4)
#define SLLM_STATUS_INVALID_ABI_VERSION UINT32_C(5)
#define SLLM_STATUS_RESERVED_NONZERO UINT32_C(6)
#define SLLM_STATUS_INTERNAL_ERROR UINT32_C(7)

/* Public runtime statuses use a separate numeric range from private evidence
 * statuses.  They are additive to the Phase 1 status set. */
#define SLLM_STATUS_PUBLIC_PENDING UINT32_C(0x100)
#define SLLM_STATUS_PUBLIC_TIMEOUT UINT32_C(0x101)
#define SLLM_STATUS_PUBLIC_INVALID_HANDLE UINT32_C(0x102)
#define SLLM_STATUS_PUBLIC_DEVICE_MISMATCH UINT32_C(0x103)
#define SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR UINT32_C(0x104)
#define SLLM_STATUS_PUBLIC_BUSY UINT32_C(0x105)
#define SLLM_STATUS_PUBLIC_NOT_READY UINT32_C(0x106)
/* RMSNorm execution is additive to public ABI v1. */
#define SLLM_STATUS_INVALID_RMSNORM_DESCRIPTOR UINT32_C(0x107)
#define SLLM_STATUS_INVALID_TENSOR_BINDING UINT32_C(0x108)
#define SLLM_STATUS_ZERO_EXTENT UINT32_C(0x109)
#define SLLM_STATUS_SHAPE_MISMATCH UINT32_C(0x10a)
#define SLLM_STATUS_STRIDE_MISMATCH UINT32_C(0x10b)
#define SLLM_STATUS_METADATA_OVERFLOW UINT32_C(0x10c)
#define SLLM_STATUS_BUFFER_OUT_OF_BOUNDS UINT32_C(0x10d)
#define SLLM_STATUS_MISALIGNED_OFFSET UINT32_C(0x10e)
#define SLLM_STATUS_UNSUPPORTED_DTYPE UINT32_C(0x10f)
#define SLLM_STATUS_UNSUPPORTED_ENCODING UINT32_C(0x110)
#define SLLM_STATUS_INVALID_EPSILON UINT32_C(0x111)
#define SLLM_STATUS_UNSUPPORTED_SCALE_MODE UINT32_C(0x112)
#define SLLM_STATUS_ALIAS_OVERLAP UINT32_C(0x113)
#define SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH UINT32_C(0x114)
#define SLLM_STATUS_INVALID_ELEMENTWISE_DESCRIPTOR UINT32_C(0x115)
#define SLLM_STATUS_INVALID_EMBEDDING_DESCRIPTOR UINT32_C(0x116)
#define SLLM_STATUS_TOKEN_ID_OUT_OF_RANGE UINT32_C(0x117)
#define SLLM_STATUS_INVALID_MATMUL_DESCRIPTOR UINT32_C(0x118)
#define SLLM_STATUS_INVALID_ATTENTION_PREPROCESS_DESCRIPTOR UINT32_C(0x119)
#define SLLM_STATUS_POSITION_PAYLOAD_MISMATCH UINT32_C(0x11a)
#define SLLM_STATUS_INVALID_KV_STATE_DESCRIPTOR UINT32_C(0x11b)
#define SLLM_STATUS_INVALID_KV_APPEND_DESCRIPTOR UINT32_C(0x11c)
#define SLLM_STATUS_KV_LENGTH_MISMATCH UINT32_C(0x11d)
#define SLLM_STATUS_KV_CAPACITY_EXCEEDED UINT32_C(0x11e)
#define SLLM_STATUS_INVALID_CAUSAL_ATTENTION_DESCRIPTOR UINT32_C(0x11f)
#define SLLM_STATUS_CAUSAL_ATTENTION_LENGTH_MISMATCH UINT32_C(0x120)
#define SLLM_STATUS_CAUSAL_ATTENTION_STATE_BUSY UINT32_C(0x121)
#define SLLM_STATUS_INVALID_LINEAR_ATTENTION_STATE_DESCRIPTOR UINT32_C(0x122)
#define SLLM_STATUS_INVALID_LINEAR_ATTENTION_DESCRIPTOR UINT32_C(0x123)
#define SLLM_STATUS_LINEAR_ATTENTION_LENGTH_MISMATCH UINT32_C(0x124)
#define SLLM_STATUS_LINEAR_ATTENTION_STATE_BUSY UINT32_C(0x125)
#define SLLM_STATUS_INVALID_ARGMAX_DESCRIPTOR UINT32_C(0x126)
#define SLLM_STATUS_INVALID_ROTARY_DESCRIPTOR UINT32_C(0x127)
#define SLLM_STATUS_INVALID_WINDOWED_ATTENTION_DESCRIPTOR UINT32_C(0x128)
#define SLLM_STATUS_INVALID_TOKEN_SELECTOR_DESCRIPTOR UINT32_C(0x129)
#define SLLM_STATUS_TOKEN_SELECTOR_NONFINITE UINT32_C(0x12a)
#define SLLM_STATUS_TOKEN_SELECTOR_ALL_MASKED UINT32_C(0x12b)
#define SLLM_STATUS_TOKEN_SELECTOR_INVALID_TEMPERATURE UINT32_C(0x12c)
#define SLLM_STATUS_INVALID_MINISTRAL3_YARN_DESCRIPTOR UINT32_C(0x12d)

#define SLLM_HIP_RMSNORM_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_RMSNORM_KERNEL_ID_BASELINE_WAVE32_V1 UINT32_C(1)
#define SLLM_HIP_RMSNORM_KERNEL_ID_BASELINE_WAVE64_V1 UINT32_C(2)
#define SLLM_HIP_RMSNORM_KERNEL_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_RMSNORM_DEVICE_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_RMSNORM_MAX_N UINT64_C(4096)
#define SLLM_HIP_RMSNORM_WORKGROUP_SIZE UINT32_C(256)
#define SLLM_HIP_RMSNORM_MAX_ROWS UINT64_C(4294967295)

#define SLLM_HIP_RESIDUAL_RMSNORM_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_RESIDUAL_RMSNORM_KERNEL_ID_WAVE32_V1 UINT32_C(1)
#define SLLM_HIP_RESIDUAL_RMSNORM_KERNEL_ID_WAVE64_V1 UINT32_C(2)

#define SLLM_HIP_ELEMENTWISE_VERSION UINT32_C(1)
#define SLLM_HIP_ELEMENTWISE_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_ELEMENTWISE_KERNEL_ID_COPY_V1 UINT32_C(1)
#define SLLM_HIP_ELEMENTWISE_KERNEL_ID_ADD_V1 UINT32_C(2)
#define SLLM_HIP_ELEMENTWISE_KERNEL_ID_SILU_MUL_V1 UINT32_C(3)
#define SLLM_HIP_ELEMENTWISE_KERNEL_ID_SIGMOID_MUL_V1 UINT32_C(4)
#define SLLM_HIP_ELEMENTWISE_KERNEL_ID_SCALAR_MUL_V1 UINT32_C(5)
#define SLLM_HIP_ELEMENTWISE_KERNEL_ID_GELU_TANH_MUL_V1 UINT32_C(6)
#define SLLM_HIP_ELEMENTWISE_KERNEL_ID_TANH_SOFTCAP_V1 UINT32_C(7)
#define SLLM_HIP_ELEMENTWISE_KERNEL_ID_BROADCAST_ADD_V1 UINT32_C(8)
#define SLLM_HIP_ELEMENTWISE_KERNEL_ID_BROADCAST_MUL_V1 UINT32_C(9)
#define SLLM_HIP_ELEMENTWISE_KERNEL_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_ELEMENTWISE_DEVICE_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_ELEMENTWISE_WORKGROUP_SIZE UINT32_C(256)
#define SLLM_HIP_ELEMENTWISE_MAX_ELEMENTS UINT64_C(4294967295)

#define SLLM_HIP_EMBEDDING_VERSION UINT32_C(1)
#define SLLM_HIP_EMBEDDING_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_EMBEDDING_KERNEL_ID_GATHER_V1 UINT32_C(1)
#define SLLM_HIP_EMBEDDING_KERNEL_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_EMBEDDING_DEVICE_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_EMBEDDING_WORKGROUP_SIZE UINT32_C(256)
#define SLLM_HIP_EMBEDDING_MAX_VOCAB UINT64_C(1048576)
#define SLLM_HIP_EMBEDDING_MAX_HIDDEN UINT64_C(4096)
#define SLLM_HIP_EMBEDDING_MAX_TOKENS UINT64_C(65536)

#define SLLM_HIP_MATMUL_VERSION UINT32_C(1)
#define SLLM_HIP_MATMUL_FP8_VERSION UINT32_C(2)
#define SLLM_HIP_MATMUL_NVFP4_VERSION UINT32_C(3)
#define SLLM_HIP_MATMUL_NVFP4_W4A4_VERSION UINT32_C(4)
#define SLLM_HIP_MATMUL_MXFP4_W4A4_VERSION UINT32_C(5)
#define SLLM_HIP_MATMUL_MXFP8_W8A8_VERSION UINT32_C(6)
#define SLLM_HIP_MATMUL_MXFP6_W6A6_VERSION UINT32_C(7)
#define SLLM_HIP_MATMUL_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_MATMUL_KERNEL_ID_BASELINE_BF16_FP32_V1 UINT32_C(1)
#define SLLM_HIP_MATMUL_KERNEL_ID_TILED16_BF16_FP32_V2 UINT32_C(2)
#define SLLM_HIP_MATMUL_KERNEL_ID_DECODE_BF16_FP32_V2 UINT32_C(3)
#define SLLM_HIP_MATMUL_KERNEL_ID_HIPBLAS_DECODE_V1 UINT32_C(4)
#define SLLM_HIP_MATMUL_KERNEL_ID_HIPBLAS_BF16_FP32_V2 UINT32_C(4)
#define SLLM_HIP_MATMUL_KERNEL_ID_HIPBLASLT_FP8_OUTER_V1 UINT32_C(5)
#define SLLM_HIP_MATMUL_KERNEL_ID_FP8_BYTE_EMULATION_V1 UINT32_C(6)
#define SLLM_HIP_MATMUL_KERNEL_ID_DECODE_WAVE64_BF16_FP32_V1 UINT32_C(7)
#define SLLM_HIP_MATMUL_KERNEL_ID_NVFP4_PACKED_DEQUANT_V1 UINT32_C(8)
#define SLLM_HIP_MATMUL_KERNEL_ID_NVFP4_W4A4_PACKED_V1 UINT32_C(11)
#define SLLM_HIP_MATMUL_KERNEL_ID_SERIAL_ROWS_BF16_FP32_V1 UINT32_C(12)
#define SLLM_HIP_MATMUL_KERNEL_ID_SERIAL_ROWS_WAVE64_BF16_FP32_V1 UINT32_C(13)
#define SLLM_HIP_MATMUL_KERNEL_ID_MXFP4_W4A4_DECODE_V1 UINT32_C(14)
#define SLLM_HIP_MATMUL_KERNEL_ID_MXFP4_W4A4_PREFILL_V1 UINT32_C(15)
#define SLLM_HIP_MATMUL_KERNEL_ID_MXFP8_W8A8_DECODE_V1 UINT32_C(18)
#define SLLM_HIP_MATMUL_KERNEL_ID_MXFP8_W8A8_PREFILL_V1 UINT32_C(19)
#define SLLM_HIP_MATMUL_KERNEL_ID_MXFP6_W6A6_DECODE_V1 UINT32_C(20)
#define SLLM_HIP_MATMUL_KERNEL_ID_MXFP6_W6A6_PREFILL_V1 UINT32_C(21)
#define SLLM_HIP_MATMUL_KERNEL_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_MATMUL_DEVICE_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_MATMUL_WORKGROUP_SIZE UINT32_C(256)
#define SLLM_HIP_MATMUL_MAX_M UINT64_C(65536)
#define SLLM_HIP_MATMUL_MAX_K UINT64_C(16384)
#define SLLM_HIP_MATMUL_MAX_N UINT64_C(262144)
#define SLLM_HIP_MATMUL_MAX_OUTPUT_ELEMENTS UINT64_C(4294967295)

#define SLLM_HIP_ARGMAX_VERSION UINT32_C(1)
#define SLLM_HIP_ARGMAX_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_ARGMAX_KERNEL_ID_BASELINE_BF16_V1 UINT32_C(1)
#define SLLM_HIP_ARGMAX_KERNEL_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_ARGMAX_DEVICE_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_ARGMAX_WORKGROUP_SIZE UINT32_C(256)
#define SLLM_HIP_ARGMAX_MAX_V UINT64_C(1048576)
#define SLLM_HIP_ARGMAX_MAX_M UINT64_C(4294967295)

#define SLLM_HIP_TOKEN_SELECTOR_VERSION UINT32_C(1)
#define SLLM_HIP_TOKEN_SELECTOR_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_TOKEN_SELECTOR_KERNEL_ID_BF16_F32_MASK_V1 UINT32_C(1)
#define SLLM_HIP_TOKEN_SELECTOR_KERNEL_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_TOKEN_SELECTOR_DEVICE_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_TOKEN_SELECTOR_WORKGROUP_SIZE UINT32_C(256)
#define SLLM_HIP_TOKEN_SELECTOR_MAX_V UINT64_C(1048576)
#define SLLM_HIP_TOKEN_SELECTOR_OUTPUT_BYTES UINT32_C(16)

#define SLLM_HIP_MOE_ROUTE_VERSION UINT32_C(1)
#define SLLM_HIP_MOE_ROUTE_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_MOE_ROUTE_KERNEL_ID_STABLE_TOPK_V1 UINT32_C(1)
#define SLLM_HIP_MOE_ROUTE_KERNEL_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_MOE_ROUTE_DEVICE_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_MOE_ROUTE_WORKGROUP_SIZE UINT32_C(256)
#define SLLM_HIP_MOE_ROUTE_MAX_TOKENS UINT64_C(65536)
#define SLLM_HIP_MOE_ROUTE_MAX_EXPERTS UINT64_C(256)
#define SLLM_HIP_MOE_ROUTE_MAX_SELECTED UINT32_C(16)

/* DeepSeek V4 routing is deliberately separate from the generic MoE route
 * ABI.  Phase 57 fixes the reviewed shape to 256 experts and top-6 while
 * retaining an explicit shape in every tensor binding. */
#define SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_VERSION UINT32_C(1)
#define SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_QUERY_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_KERNEL_ID_SCORE_V1 UINT32_C(1)
#define SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_KERNEL_ID_HASH_V1 UINT32_C(2)
#define SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_KERNEL_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_DEVICE_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_WORKGROUP_SIZE UINT32_C(256)
#define SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_MAX_TOKENS UINT64_C(65536)
#define SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_EXPERT_COUNT UINT64_C(256)
#define SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_SELECTED_EXPERT_COUNT UINT32_C(6)

typedef uint32_t sllm_deepseek_v4_moe_route_mode_t;
#define SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_SCORE UINT32_C(1)
#define SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_HASH UINT32_C(2)

/* Device-written status stored at the final i32 of the metadata buffer. */
#define SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_OK INT32_C(0)
#define SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_NONFINITE INT32_C(1)
#define SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_EXPERT_OUT_OF_RANGE INT32_C(2)
#define SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_DUPLICATE_EXPERT INT32_C(3)
#define SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_ZERO_NORMALIZER INT32_C(4)

/* MiniMax M3 routing is a separate, fixed model semantic: F32 sigmoid
 * scores, selection-only F32 bias, stable top-4, selected-score
 * renormalization, and routed scale 2.0. */
#define SLLM_HIP_MINIMAX_M3_MOE_ROUTE_VERSION UINT32_C(1)
#define SLLM_HIP_MINIMAX_M3_MOE_ROUTE_QUERY_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_MINIMAX_M3_MOE_ROUTE_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_MINIMAX_M3_MOE_ROUTE_KERNEL_ID_SIGMOID_TOP4_V1 UINT32_C(1)
#define SLLM_HIP_MINIMAX_M3_MOE_ROUTE_KERNEL_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_MINIMAX_M3_MOE_ROUTE_DEVICE_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_MINIMAX_M3_MOE_ROUTE_WORKGROUP_SIZE UINT32_C(256)
#define SLLM_HIP_MINIMAX_M3_MOE_ROUTE_MAX_TOKENS UINT64_C(65536)
#define SLLM_HIP_MINIMAX_M3_MOE_ROUTE_EXPERT_COUNT UINT64_C(128)
#define SLLM_HIP_MINIMAX_M3_MOE_ROUTE_SELECTED_EXPERT_COUNT UINT32_C(4)

#define SLLM_MINIMAX_M3_MOE_ROUTE_STATUS_OK INT32_C(0)
#define SLLM_MINIMAX_M3_MOE_ROUTE_STATUS_NONFINITE INT32_C(1)
#define SLLM_MINIMAX_M3_MOE_ROUTE_STATUS_ZERO_NORMALIZER INT32_C(2)

#define SLLM_HIP_MOE_EXPERT_VERSION UINT32_C(1)
#define SLLM_HIP_MOE_EXPERT_GEMMA4_VERSION UINT32_C(2)
#define SLLM_HIP_MOE_EXPERT_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_MOE_EXPERT_KERNEL_ID_DECODE_V1 UINT32_C(1)
#define SLLM_HIP_MOE_EXPERT_KERNEL_ID_PREFILL_V1 UINT32_C(2)
#define SLLM_HIP_MOE_EXPERT_KERNEL_ID_GEMMA4_DECODE_V2 UINT32_C(3)
#define SLLM_HIP_MOE_EXPERT_KERNEL_ID_GEMMA4_PREFILL_V2 UINT32_C(4)
#define SLLM_HIP_MOE_EXPERT_KERNEL_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_MOE_EXPERT_DEVICE_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_MOE_EXPERT_WORKGROUP_SIZE UINT32_C(256)
#define SLLM_HIP_MOE_EXPERT_HIDDEN_SIZE UINT32_C(2048)
#define SLLM_HIP_MOE_EXPERT_INTERMEDIATE_SIZE UINT32_C(512)
#define SLLM_HIP_MOE_EXPERT_COUNT UINT32_C(256)
#define SLLM_HIP_MOE_EXPERT_TOPK UINT32_C(8)
#define SLLM_HIP_MOE_EXPERT_LAYER_BLOB_BYTES UINT64_C(434114560)
#define SLLM_HIP_MOE_EXPERT_MAX_TOKENS UINT64_C(65536)
#define SLLM_HIP_GEMMA4_MOE_EXPERT_HIDDEN_SIZE UINT32_C(2816)
#define SLLM_HIP_GEMMA4_MOE_EXPERT_INTERMEDIATE_SIZE UINT32_C(704)
#define SLLM_HIP_GEMMA4_MOE_EXPERT_COUNT UINT32_C(128)
#define SLLM_HIP_GEMMA4_MOE_EXPERT_TOPK UINT32_C(8)
#define SLLM_HIP_GEMMA4_MOE_EXPERT_LAYER_BLOB_BYTES UINT64_C(428215552)
#define SLLM_HIP_GEMMA4_MOE_EXPERT_WORKSPACE_BYTES_PER_TOKEN UINT64_C(27104)

#define SLLM_HIP_ATTENTION_PREPROCESS_VERSION UINT32_C(1)
#define SLLM_HIP_ATTENTION_PREPROCESS_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_ATTENTION_PREPROCESS_KERNEL_ID_BASELINE_BF16_V1 UINT32_C(1)
#define SLLM_HIP_ATTENTION_PREPROCESS_KERNEL_ID_WAVE32_BF16_V1 UINT32_C(2)
#define SLLM_HIP_ATTENTION_PREPROCESS_KERNEL_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_ATTENTION_PREPROCESS_DEVICE_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_ATTENTION_PREPROCESS_WORKGROUP_SIZE UINT32_C(1)
#define SLLM_HIP_ATTENTION_PREPROCESS_WAVE32_WORKGROUP_SIZE UINT32_C(32)
#define SLLM_HIP_ATTENTION_PREPROCESS_Q_HEADS UINT32_C(16)
#define SLLM_HIP_ATTENTION_PREPROCESS_K_HEADS UINT32_C(4)
#define SLLM_HIP_ATTENTION_PREPROCESS_Q_HEAD_DIM UINT32_C(256)
#define SLLM_HIP_ATTENTION_PREPROCESS_K_HEAD_DIM UINT32_C(256)
#define SLLM_HIP_ATTENTION_PREPROCESS_QGATE_HEAD_DIM UINT32_C(512)
#define SLLM_HIP_ATTENTION_PREPROCESS_ROTARY_DIM UINT32_C(64)
#define SLLM_HIP_ATTENTION_PREPROCESS_MAX_POSITION UINT32_C(4294967295)
#define SLLM_HIP_ATTENTION_PREPROCESS_MAX_M UINT64_C(262144)
#define SLLM_HIP_POSITION_PAYLOAD_MODE_CONTIGUOUS_V1 UINT32_C(0)
#define SLLM_HIP_POSITION_PAYLOAD_MODE_EXPLICIT_V1 UINT32_C(1)
#define SLLM_HIP_POSITION_PAYLOAD_MODE_DERIVED_CONTIGUOUS_V1 UINT32_C(2)

#define SLLM_HIP_ROTARY_VERSION UINT32_C(1)
#define SLLM_HIP_ROTARY_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_ROTARY_KERNEL_ID_SPLIT_HALF_BF16_FP32_V1 UINT32_C(1)
#define SLLM_HIP_ROTARY_KERNEL_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_ROTARY_DEVICE_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_ROTARY_WORKGROUP_SIZE UINT32_C(256)
#define SLLM_HIP_ROTARY_MAX_M UINT64_C(262144)
#define SLLM_HIP_ROTARY_MAX_POSITION UINT32_C(4294967295)

/* Ministral 3 3B uses a fixed BF16 YaRN RoPE contract. Version 1 preserves
 * the original source-layout split-half interpretation. Version 2 is the
 * official-GGUF interpretation: Q/K weights have already received the GGUF
 * head permutation, so rotary pairs are adjacent. */
#define SLLM_HIP_MINISTRAL3_YARN_VERSION UINT32_C(1)
#define SLLM_HIP_MINISTRAL3_YARN_ADJACENT_VERSION UINT32_C(2)
#define SLLM_HIP_MINISTRAL3_YARN_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_MINISTRAL3_YARN_KERNEL_ID_BF16_SPLIT_HALF_QSCALE_V1 UINT32_C(1)
#define SLLM_HIP_MINISTRAL3_YARN_KERNEL_ID_BF16_ADJACENT_QSCALE_V2 UINT32_C(2)
#define SLLM_HIP_MINISTRAL3_YARN_KERNEL_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_MINISTRAL3_YARN_DEVICE_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_MINISTRAL3_YARN_WORKGROUP_SIZE UINT32_C(256)
#define SLLM_HIP_MINISTRAL3_YARN_Q_HEADS UINT32_C(32)
#define SLLM_HIP_MINISTRAL3_YARN_KV_HEADS UINT32_C(8)
#define SLLM_HIP_MINISTRAL3_YARN_HEAD_DIM UINT32_C(128)
#define SLLM_HIP_MINISTRAL3_YARN_ROTARY_DIM UINT32_C(128)
#define SLLM_HIP_MINISTRAL3_YARN_ORIGINAL_CONTEXT UINT32_C(16384)
#define SLLM_HIP_MINISTRAL3_YARN_MAX_POSITION UINT32_C(262144)
#define SLLM_HIP_MINISTRAL3_YARN_MAX_M UINT64_C(262144)

#define SLLM_HIP_WINDOWED_ATTENTION_VERSION UINT32_C(1)
#define SLLM_HIP_WINDOWED_ATTENTION_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_WINDOWED_ATTENTION_KERNEL_ID_ONLINE_SOFTMAX_GQA_BF16_V1       \
  UINT32_C(1)
#define SLLM_HIP_WINDOWED_ATTENTION_KERNEL_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_WINDOWED_ATTENTION_DEVICE_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_WINDOWED_ATTENTION_WORKGROUP_SIZE UINT32_C(256)
#define SLLM_HIP_WINDOWED_ATTENTION_MAX_M UINT64_C(262144)
#define SLLM_HIP_WINDOWED_ATTENTION_MAX_KV UINT64_C(4294967295)
#define SLLM_HIP_WINDOWED_ATTENTION_MAX_HEAD_DIM UINT32_C(512)

#define SLLM_HIP_KV_STATE_VERSION UINT32_C(2)
#define SLLM_HIP_KV_STATE_CREATE_INFO_V2_VERSION UINT32_C(1)
#define SLLM_HIP_KV_STATE_CREATE_INFO_STATIC_FP8_VERSION UINT32_C(2)
#define SLLM_HIP_KV_STATE_CREATE_INFO_SLIDING_STATIC_FP8_VERSION UINT32_C(3)
#define SLLM_HIP_KV_VIEW_INFO_VERSION UINT32_C(2)
#define SLLM_HIP_KV_VIEW_INFO_SLIDING_VERSION UINT32_C(3)
#define SLLM_HIP_KV_APPEND_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_KV_HEAD_COUNT UINT32_C(4)
#define SLLM_HIP_KV_HEAD_DIM UINT32_C(256)
#define SLLM_HIP_KV_MAX_HEAD_DIM UINT32_C(512)
#define SLLM_HIP_KV_MAX_CAPACITY UINT64_C(4294967295)
#define SLLM_HIP_KV_MAX_M UINT64_C(262144)
#define SLLM_HIP_KV_SLIDING_MAX_CAPACITY UINT64_C(262144)
#define SLLM_HIP_KV_SLIDING_WINDOW_GEMMA4 UINT64_C(1024)
#define SLLM_HIP_KV_KERNEL_ID_BF16_TO_F16_TRANSPOSE_V1 UINT32_C(1)
#define SLLM_HIP_KV_KERNEL_ID_BF16_TO_F16_TOKEN_MAJOR_V2 UINT32_C(2)
#define SLLM_HIP_KV_KERNEL_ID_BF16_TO_FP8_TOKEN_MAJOR_V1 UINT32_C(3)
#define SLLM_HIP_KV_KERNEL_ID_BF16_TO_NVFP4_TOKEN_MAJOR_V1 UINT32_C(4)
#define SLLM_HIP_KV_KERNEL_ID_BF16_TO_FP8_STATIC_TOKEN_MAJOR_V1 UINT32_C(5)
#define SLLM_HIP_KV_KERNEL_ID_BF16_TO_FP8_E4_BLOCK16_TOKEN_MAJOR_V1 UINT32_C(6)
#define SLLM_HIP_KV_KERNEL_ID_BF16_TO_FP8_E5_BLOCK16_TOKEN_MAJOR_V1 UINT32_C(7)
#define SLLM_HIP_KV_KERNEL_ID_BF16_TO_MXFP8_E4_TOKEN_MAJOR_V1 UINT32_C(8)
#define SLLM_HIP_KV_KERNEL_ID_BF16_TO_MXFP8_E5_TOKEN_MAJOR_V1 UINT32_C(9)
#define SLLM_HIP_KV_KERNEL_ID_BF16_TO_FP8_E4_BLOCK16_TOKEN_MAJOR_V2 UINT32_C(10)
#define SLLM_HIP_KV_KERNEL_ID_BF16_TO_FP8_E5_BLOCK16_TOKEN_MAJOR_V2 UINT32_C(11)
#define SLLM_HIP_KV_WORKGROUP_SIZE UINT32_C(256)
#define SLLM_HIP_KV_KERNEL_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_KV_DEVICE_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS UINT32_C(1)
#define SLLM_HIP_KV_MEMORY_KIND_CAPABILITY_SELECTED UINT32_C(0)
#define SLLM_HIP_KV_MEMORY_KIND_CONTIGUOUS_RESIDENT UINT32_C(2)
#define SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR UINT32_C(1)
#define SLLM_HIP_KV_ENCODING_FP16_V1 UINT32_C(0)
#define SLLM_HIP_KV_ENCODING_FP8_V1 UINT32_C(1)
#define SLLM_HIP_KV_ENCODING_NVFP4_V1 UINT32_C(2)
#define SLLM_HIP_KV_ENCODING_FP8_STATIC_V1 UINT32_C(3)
#define SLLM_HIP_KV_ENCODING_FP8_E4_BLOCK16_V1 UINT32_C(4)
#define SLLM_HIP_KV_ENCODING_FP8_E5_BLOCK16_V1 UINT32_C(5)
#define SLLM_HIP_KV_ENCODING_MXFP8_E4_V1 UINT32_C(6)
#define SLLM_HIP_KV_ENCODING_MXFP8_E5_V1 UINT32_C(7)
/* V1 block16 IDs are retained for historical ABI inspection only. */
#define SLLM_HIP_KV_ENCODING_FP8_E4_BLOCK16_V2 UINT32_C(8)
#define SLLM_HIP_KV_ENCODING_FP8_E5_BLOCK16_V2 UINT32_C(9)

/* Additive Phase 41 state-fork and raw-plane persistence ABI. */
#define SLLM_HIP_STATE_FORK_VERSION UINT32_C(1)
#define SLLM_HIP_STATE_FORK_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_STATE_IMAGE_SLIDING_VERSION UINT32_C(2)
#define SLLM_HIP_STATE_FORK_MODE_DEVICE_COPY UINT32_C(1)
#define SLLM_HIP_STATE_FORK_MODE_SHARED_READ_ONLY_PAGES UINT32_C(2)
#define SLLM_HIP_KV_STATE_PLANE_KEY UINT32_C(1)
#define SLLM_HIP_KV_STATE_PLANE_VALUE UINT32_C(2)
#define SLLM_HIP_KV_STATE_PLANE_KEY_SCALE UINT32_C(3)
#define SLLM_HIP_KV_STATE_PLANE_VALUE_SCALE UINT32_C(4)
#define SLLM_HIP_KV_STATE_PLANE_KEY_OUTER_SCALE UINT32_C(5)
#define SLLM_HIP_KV_STATE_PLANE_VALUE_OUTER_SCALE UINT32_C(6)
#define SLLM_HIP_LINEAR_STATE_PLANE_CONV_SLOT0 UINT32_C(1)
#define SLLM_HIP_LINEAR_STATE_PLANE_CONV_SLOT1 UINT32_C(2)
#define SLLM_HIP_LINEAR_STATE_PLANE_RECURRENT_SLOT0 UINT32_C(3)
#define SLLM_HIP_LINEAR_STATE_PLANE_RECURRENT_SLOT1 UINT32_C(4)
#define SLLM_HIP_LINEAR_STATE_PLANE_SCRATCH UINT32_C(5)
#define SLLM_HIP_STATE_CHUNK_MAX_BYTES UINT64_C(1073741824)

#define SLLM_HIP_CAUSAL_ATTENTION_VERSION UINT32_C(1)
#define SLLM_HIP_CAUSAL_ATTENTION_SLIDING_VERSION UINT32_C(2)
#define SLLM_HIP_CAUSAL_ATTENTION_EXPLICIT_SCALE_VERSION UINT32_C(3)
#define SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_STABLE_SOFTMAX_V1 UINT32_C(1)
#define SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_ONLINE_SOFTMAX_V2 UINT32_C(2)
#define SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PACKED_KV_V3 UINT32_C(3)
#define SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_SLIDING_STATIC_FP8_V1 UINT32_C(4)
#define SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_SCALED_STATIC_FP8_V1 UINT32_C(5)
#define SLLM_HIP_CAUSAL_ATTENTION_KERNEL_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_CAUSAL_ATTENTION_DEVICE_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE UINT32_C(256)
#define SLLM_HIP_CAUSAL_ATTENTION_Q_HEADS UINT32_C(16)
#define SLLM_HIP_CAUSAL_ATTENTION_KV_HEADS UINT32_C(4)
#define SLLM_HIP_CAUSAL_ATTENTION_HEAD_DIM UINT32_C(256)
#define SLLM_HIP_CAUSAL_ATTENTION_SCALE_DENOMINATOR UINT32_C(16)
#define SLLM_HIP_CAUSAL_ATTENTION_MAX_M UINT64_C(262144)

#define SLLM_HIP_LINEAR_ATTENTION_VERSION UINT32_C(1)
#define SLLM_HIP_LINEAR_ATTENTION_VIEW_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_LINEAR_ATTENTION_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_LINEAR_ATTENTION_KERNEL_ID_CAUSAL_CONV_SILU_V1 UINT32_C(1)
#define SLLM_HIP_LINEAR_ATTENTION_KERNEL_ID_RECURRENT_GATED_NORM_V1 UINT32_C(2)
#define SLLM_HIP_LINEAR_ATTENTION_KERNEL_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_LINEAR_ATTENTION_DEVICE_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_LINEAR_ATTENTION_WORKGROUP_SIZE UINT32_C(128)
#define SLLM_HIP_LINEAR_ATTENTION_QK_HEADS UINT32_C(16)
#define SLLM_HIP_LINEAR_ATTENTION_VALUE_HEADS UINT32_C(32)
#define SLLM_HIP_LINEAR_ATTENTION_HEAD_DIM UINT32_C(128)
#define SLLM_HIP_LINEAR_ATTENTION_QKV_WIDTH UINT32_C(8192)
#define SLLM_HIP_LINEAR_ATTENTION_OUTPUT_WIDTH UINT32_C(4096)
#define SLLM_HIP_LINEAR_ATTENTION_CONV_KERNEL_SIZE UINT32_C(4)
#define SLLM_HIP_LINEAR_ATTENTION_CONV_HISTORY UINT32_C(3)
#define SLLM_HIP_LINEAR_ATTENTION_MAX_CAPACITY UINT64_C(4294967295)
#define SLLM_HIP_LINEAR_ATTENTION_MAX_M UINT64_C(262144)

#define SLLM_BACKEND_HIP UINT32_C(1)

typedef uint32_t sllm_access_mode_t;

#define SLLM_ACCESS_READ UINT32_C(1)
#define SLLM_ACCESS_WRITE UINT32_C(2)
#define SLLM_ACCESS_READ_WRITE UINT32_C(3)

#define SLLM_HIP_MAX_DEVICE_NAME UINT32_C(128)
#define SLLM_HIP_MAX_GCN_ARCH_NAME UINT32_C(64)
#define SLLM_HIP_MAX_TRANSFER_BYTES UINT64_C(1073741824)

#define SLLM_HIP_RMSNORM_VERSION UINT32_C(1)
#define SLLM_HIP_RESIDUAL_RMSNORM_VERSION UINT32_C(1)
#define SLLM_HIP_TENSOR_MAX_RANK UINT32_C(8)

typedef uint32_t sllm_tensor_dtype_t;
#define SLLM_TENSOR_DTYPE_BF16 UINT32_C(0)
#define SLLM_TENSOR_DTYPE_F16 UINT32_C(1)
#define SLLM_TENSOR_DTYPE_F32 UINT32_C(2)
#define SLLM_TENSOR_DTYPE_F8_E4M3_FN UINT32_C(3)
#define SLLM_TENSOR_DTYPE_F8_E4M3_FNUZ UINT32_C(4)
#define SLLM_TENSOR_DTYPE_F8_E5M2 UINT32_C(5)
#define SLLM_TENSOR_DTYPE_U8 UINT32_C(9)
#define SLLM_TENSOR_DTYPE_I32 UINT32_C(8)

typedef uint32_t sllm_tensor_encoding_t;
#define SLLM_TENSOR_ENCODING_UNQUANTIZED UINT32_C(0)
#define SLLM_TENSOR_ENCODING_FP8_OUTER_F32 UINT32_C(1)
#define SLLM_TENSOR_ENCODING_NVFP4_BLOCK16_E4M3FN_F32 UINT32_C(2)
#define SLLM_TENSOR_ENCODING_NVFP4_W4A4_BLOCK16_E4M3FN_F32 UINT32_C(3)
#define SLLM_TENSOR_ENCODING_MXFP4_W4A4_BLOCK32_E8M0 UINT32_C(4)
#define SLLM_TENSOR_ENCODING_FP8_BLOCK16_E8M0 UINT32_C(5)
#define SLLM_TENSOR_ENCODING_MXFP8_BLOCK32_E8M0 UINT32_C(6)
#define SLLM_TENSOR_ENCODING_MXFP6_E3M2_BLOCK32_E8M0 UINT32_C(7)

typedef uint32_t sllm_rmsnorm_accumulation_dtype_t;
#define SLLM_RMSNORM_ACCUMULATION_F32 UINT32_C(2)

typedef uint32_t sllm_rmsnorm_scale_mode_t;
#define SLLM_RMSNORM_SCALE_MODE_OFFSET_ONE UINT32_C(1)
#define SLLM_RMSNORM_SCALE_MODE_DIRECT UINT32_C(2)

typedef uint32_t sllm_rmsnorm_alias_policy_t;
#define SLLM_RMSNORM_ALIAS_POLICY_REJECT_OVERLAP UINT32_C(1)

typedef uint32_t sllm_elementwise_operation_t;
#define SLLM_ELEMENTWISE_OPERATION_COPY UINT32_C(1)
#define SLLM_ELEMENTWISE_OPERATION_ADD UINT32_C(2)
#define SLLM_ELEMENTWISE_OPERATION_SILU_MUL UINT32_C(3)
#define SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL UINT32_C(4)
#define SLLM_ELEMENTWISE_OPERATION_SCALAR_MUL UINT32_C(5)
#define SLLM_ELEMENTWISE_OPERATION_GELU_TANH_MUL UINT32_C(6)
#define SLLM_ELEMENTWISE_OPERATION_TANH_SOFTCAP UINT32_C(7)
#define SLLM_ELEMENTWISE_OPERATION_BROADCAST_ADD UINT32_C(8)
#define SLLM_ELEMENTWISE_OPERATION_BROADCAST_MUL UINT32_C(9)

#define SLLM_COMPLETION_STATE_PENDING UINT32_C(0)
#define SLLM_COMPLETION_STATE_SUCCESS UINT32_C(1)
#define SLLM_COMPLETION_STATE_FAILURE UINT32_C(2)

typedef uint32_t sllm_queue_completion_mode_t;
#define SLLM_QUEUE_COMPLETION_MODE_PROFILED UINT32_C(0)
#define SLLM_QUEUE_COMPLETION_MODE_DEFERRED UINT32_C(1)

/* These handles have no public layout and must not be dereferenced by callers.
 */
typedef struct sllm_context_t sllm_context_t;
typedef struct sllm_queue_t sllm_queue_t;
typedef struct sllm_buffer_t sllm_buffer_t;
typedef struct sllm_event_t sllm_event_t;
typedef struct sllm_completion_t sllm_completion_t;
typedef struct sllm_rmsnorm_plan_t sllm_rmsnorm_plan_t;
typedef struct sllm_residual_rmsnorm_plan_t sllm_residual_rmsnorm_plan_t;
typedef struct sllm_elementwise_plan_t sllm_elementwise_plan_t;
typedef struct sllm_embedding_plan_t sllm_embedding_plan_t;
typedef struct sllm_matmul_plan_t sllm_matmul_plan_t;
typedef struct sllm_gdn_projection_bundle_plan_t
    sllm_gdn_projection_bundle_plan_t;
typedef struct sllm_mlp_gate_up_silu_bundle_plan_t
    sllm_mlp_gate_up_silu_bundle_plan_t;
typedef struct sllm_argmax_plan_t sllm_argmax_plan_t;
typedef struct sllm_token_selector_plan_t sllm_token_selector_plan_t;
typedef struct sllm_moe_route_plan_t sllm_moe_route_plan_t;
typedef struct sllm_deepseek_v4_moe_route_plan_t
    sllm_deepseek_v4_moe_route_plan_t;
typedef struct sllm_minimax_m3_moe_route_plan_t
    sllm_minimax_m3_moe_route_plan_t;
typedef struct sllm_moe_expert_plan_t sllm_moe_expert_plan_t;
typedef struct sllm_attention_preprocess_plan_t
    sllm_attention_preprocess_plan_t;
typedef struct sllm_rotary_plan_t sllm_rotary_plan_t;
typedef struct sllm_ministral3_yarn_plan_t sllm_ministral3_yarn_plan_t;
typedef struct sllm_windowed_attention_plan_t sllm_windowed_attention_plan_t;
typedef struct sllm_kv_state_t sllm_kv_state_t;
typedef struct sllm_kv_view_t sllm_kv_view_t;
typedef struct sllm_linear_attention_state_t sllm_linear_attention_state_t;
typedef struct sllm_completion_timing_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t valid;
  uint32_t reserved0;
  uint64_t elapsed_ns;
  uint32_t reserved[4];
} sllm_completion_timing_t;

typedef struct sllm_error_sink_t {
  uint32_t struct_size;
  uint32_t abi_version;
  char *message;
  uint64_t message_capacity;
  uint64_t message_length;
  uint64_t reserved[2];
} sllm_error_sink_t;

/* message_capacity includes space for the terminating NUL.  On a diagnostic
 * error, message_length is the required message length excluding that NUL,
 * even when the message is truncated.  A valid sink copies at most
 * message_capacity - 1 bytes and always NUL-terminates a non-zero-capacity
 * buffer.  If the complete message does not fit, the API returns
 * SLLM_STATUS_BUFFER_TOO_SMALL.  A null sink leaves the primary operation
 * status unchanged. */

/* The argument must be a character array, not a pointer. */
// clang-format off
#define SLLM_ERROR_SINK_INIT(buffer)                                           \
  {sizeof(sllm_error_sink_t),                                                  \
   SLLM_HIP_ABI_VERSION,                                                       \
   (buffer),                                                                   \
   sizeof(buffer),                                                             \
   0,                                                                          \
   {0, 0}}
// clang-format on

typedef struct sllm_version_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t major;
  uint32_t minor;
  uint32_t patch;
  uint32_t reserved[3];
} sllm_version_info_t;

typedef struct sllm_backend_probe_result_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t backend;
  uint32_t available;
  uint32_t hip_runtime_present;
  uint32_t reserved[3];
} sllm_backend_probe_result_t;

typedef struct sllm_context_probe_result_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t context_present;
  uint32_t hip_available;
  uint32_t reserved[4];
} sllm_context_probe_result_t;

typedef struct sllm_device_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t device_index;
  uint32_t visible_device_count;
  uint64_t total_memory_bytes;
  uint32_t wavefront_size;
  uint32_t reserved0;
  char name[SLLM_HIP_MAX_DEVICE_NAME];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint64_t available_memory_bytes;
  uint32_t reserved[2];
} sllm_device_info_t;

typedef struct sllm_context_create_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t device_index;
  uint32_t flags;
  char expected_gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[4];
} sllm_context_create_info_t;

typedef struct sllm_queue_create_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t flags;
  uint32_t reserved[5];
} sllm_queue_create_info_t;

typedef struct sllm_buffer_create_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t size_bytes;
  uint64_t alignment_bytes;
  uint32_t flags;
  uint32_t reserved[5];
} sllm_buffer_create_info_t;

typedef struct sllm_transfer_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  /* H2D source is copied before the call returns.  D2H does not retain this
   * pointer; callers read staged output with sllm_completion_read(). */
  void *host_pointer;
  uint64_t buffer_offset_bytes;
  uint64_t size_bytes;
  uint32_t reserved[4];
} sllm_transfer_desc_t;

/* Device-to-device copies use independent offsets and never stage through
 * host memory.  The descriptor is copied before the asynchronous submission
 * returns. */
typedef struct sllm_buffer_copy_d2d_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t source_offset_bytes;
  uint64_t destination_offset_bytes;
  uint64_t size_bytes;
  uint32_t reserved[4];
} sllm_buffer_copy_d2d_desc_t;

typedef struct sllm_completion_result_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t state;
  uint32_t reserved0;
  uint64_t transfer_size_bytes;
  uint64_t available_bytes;
  uint32_t reserved[4];
} sllm_completion_result_t;

/* Tensor bindings are descriptors, not ownership transfers.  prepare copies
 * all metadata immediately and never retains this struct or its address. */
typedef struct sllm_tensor_binding_t {
  uint32_t struct_size;
  uint32_t abi_version;
  const sllm_buffer_t *buffer;
  uint64_t byte_offset;
  sllm_tensor_dtype_t dtype;
  sllm_tensor_encoding_t encoding;
  uint32_t rank;
  uint32_t reserved0;
  uint64_t shape[SLLM_HIP_TENSOR_MAX_RANK];
  uint64_t stride_elements[SLLM_HIP_TENSOR_MAX_RANK];
  uint64_t reserved[2];
} sllm_tensor_binding_t;

typedef struct sllm_rmsnorm_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t op_version;
  sllm_rmsnorm_accumulation_dtype_t accumulation_dtype;
  sllm_rmsnorm_scale_mode_t scale_mode;
  sllm_rmsnorm_alias_policy_t alias_policy;
  uint32_t epsilon_bits;
  uint32_t reserved[3];
  sllm_tensor_binding_t activation;
  sllm_tensor_binding_t raw_scale;
  sllm_tensor_binding_t output;
} sllm_rmsnorm_desc_t;

typedef struct sllm_rmsnorm_dispatch_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t backend;
  uint64_t dispatch_id;
  uint32_t dispatch_count;
  uint32_t kernel_id;
  uint32_t workgroup_size_x;
  uint32_t grid_size_x;
  uint64_t row_count;
  uint64_t normalized_size;
  uint32_t fallback_allowed;
  uint32_t fallback_used;
  char kernel_symbol[SLLM_HIP_RMSNORM_KERNEL_SYMBOL_MAX];
  char device_symbol[SLLM_HIP_RMSNORM_DEVICE_SYMBOL_MAX];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[8];
} sllm_rmsnorm_dispatch_info_t;

/* Fused residual-add/RMSNorm keeps the BF16-RNE add intermediate as output0
 * and writes the normalized BF16 result as output1. Both operations use F32
 * accumulation; no completion or aliasing contract is inherited implicitly. */
typedef struct sllm_residual_rmsnorm_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t op_version;
  sllm_rmsnorm_accumulation_dtype_t accumulation_dtype;
  sllm_rmsnorm_scale_mode_t scale_mode;
  sllm_rmsnorm_alias_policy_t alias_policy;
  uint32_t epsilon_bits;
  uint32_t reserved[3];
  sllm_tensor_binding_t residual;
  sllm_tensor_binding_t addend;
  sllm_tensor_binding_t raw_scale;
  sllm_tensor_binding_t residual_output;
  sllm_tensor_binding_t output;
} sllm_residual_rmsnorm_desc_t;

typedef struct sllm_residual_rmsnorm_dispatch_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t backend;
  uint64_t dispatch_id;
  uint32_t dispatch_count;
  uint32_t kernel_id;
  uint32_t workgroup_size_x;
  uint32_t grid_size_x;
  uint64_t row_count;
  uint64_t normalized_size;
  uint32_t fallback_allowed;
  uint32_t fallback_used;
  char kernel_symbol[SLLM_HIP_RMSNORM_KERNEL_SYMBOL_MAX];
  char device_symbol[SLLM_HIP_RMSNORM_DEVICE_SYMBOL_MAX];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[8];
} sllm_residual_rmsnorm_dispatch_info_t;

/* Copy leaves input1 zero-initialized. Add and SiLU-multiply supply all three
 * bindings. Every accepted binding is contiguous, unquantized BF16, non-empty,
 * and pairwise non-overlapping within a backing-buffer identity. */
typedef struct sllm_elementwise_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t op_version;
  sllm_elementwise_operation_t operation;
  uint32_t reserved[4];
  sllm_tensor_binding_t input0;
  sllm_tensor_binding_t input1;
  sllm_tensor_binding_t output;
} sllm_elementwise_desc_t;

typedef struct sllm_elementwise_dispatch_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t backend;
  uint64_t dispatch_id;
  sllm_elementwise_operation_t operation;
  uint32_t dispatch_count;
  uint32_t kernel_id;
  uint32_t workgroup_size_x;
  uint32_t grid_size_x;
  uint32_t fallback_allowed;
  uint32_t fallback_used;
  uint64_t element_count;
  char kernel_symbol[SLLM_HIP_ELEMENTWISE_KERNEL_SYMBOL_MAX];
  char device_symbol[SLLM_HIP_ELEMENTWISE_DEVICE_SYMBOL_MAX];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[8];
} sllm_elementwise_dispatch_info_t;

/* Baseline single-GPU embedding gather. The weight is BF16 [vocab, hidden],
 * token_ids is I32 [tokens], and output is BF16 [tokens, hidden]. */
typedef struct sllm_embedding_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t op_version;
  uint32_t reserved[5];
  sllm_tensor_binding_t weight;
  sllm_tensor_binding_t token_ids;
  sllm_tensor_binding_t output;
} sllm_embedding_desc_t;

typedef struct sllm_embedding_dispatch_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t backend;
  uint64_t dispatch_id;
  uint32_t dispatch_count;
  uint32_t kernel_id;
  uint32_t workgroup_size_x;
  uint32_t grid_size_x;
  uint32_t fallback_allowed;
  uint32_t fallback_used;
  uint64_t token_count;
  uint64_t hidden_size;
  uint64_t vocab_size;
  char kernel_symbol[SLLM_HIP_EMBEDDING_KERNEL_SYMBOL_MAX];
  char device_symbol[SLLM_HIP_EMBEDDING_DEVICE_SYMBOL_MAX];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[8];
} sllm_embedding_dispatch_info_t;

/* Baseline bias-free linear operation. activation is BF16 [M,K], checkpoint
 * weight storage is BF16 [N,K], and output is BF16 [M,N]. */
typedef struct sllm_matmul_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t op_version;
  uint32_t reserved[5];
  sllm_tensor_binding_t activation;
  sllm_tensor_binding_t weight;
  sllm_tensor_binding_t output;
} sllm_matmul_desc_t;

typedef struct sllm_matmul_dispatch_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t backend;
  uint64_t dispatch_id;
  uint32_t dispatch_count;
  uint32_t kernel_id;
  uint32_t workgroup_size_x;
  uint32_t grid_size_x;
  uint32_t fallback_allowed;
  uint32_t fallback_used;
  uint64_t m;
  uint64_t k;
  uint64_t n;
  uint64_t output_elements;
  char kernel_symbol[SLLM_HIP_MATMUL_KERNEL_SYMBOL_MAX];
  char device_symbol[SLLM_HIP_MATMUL_DEVICE_SYMBOL_MAX];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[8];
} sllm_matmul_dispatch_info_t;

/* Fixed-role Qwen3.5 GDN projection bundle. The four BF16 matmuls share one
 * activation but retain independent weights and outputs (qkv,z,b,a). */
#define SLLM_HIP_GDN_PROJECTION_BUNDLE_VERSION UINT32_C(1)
#define SLLM_HIP_GDN_PROJECTION_BUNDLE_DISPATCH_INFO_VERSION UINT32_C(1)
typedef struct sllm_gdn_projection_bundle_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t op_version;
  uint32_t reserved[5];
  sllm_tensor_binding_t activation;
  sllm_tensor_binding_t weights[4];
  sllm_tensor_binding_t outputs[4];
} sllm_gdn_projection_bundle_desc_t;

typedef struct sllm_gdn_projection_bundle_dispatch_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t backend;
  uint64_t dispatch_id;
  uint32_t dispatch_count;
  uint32_t kernel_id;
  uint32_t workgroup_size_x;
  uint32_t grid_size_x;
  uint32_t fallback_allowed;
  uint32_t fallback_used;
  uint64_t m;
  uint64_t k;
  uint32_t widths[4];
  uint32_t reserved0;
  char kernel_symbol[SLLM_HIP_MATMUL_KERNEL_SYMBOL_MAX];
  char device_symbol[SLLM_HIP_MATMUL_DEVICE_SYMBOL_MAX];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[8];
} sllm_gdn_projection_bundle_dispatch_info_t;

/* Fixed-role Qwen3.5 MLP gate/up/SiLU bundle.  One decode workgroup computes
 * both BF16 gate/up projections (M=1,K=2560,N=9216), stores the rounded gate
 * and up intermediates, then applies the existing BF16-RNE SiLU multiply. */
#define SLLM_HIP_MLP_GATE_UP_SILU_BUNDLE_VERSION UINT32_C(1)
#define SLLM_HIP_MLP_GATE_UP_SILU_BUNDLE_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_MLP_GATE_UP_SILU_BUNDLE_KERNEL_ID_V1 UINT32_C(1)
typedef struct sllm_mlp_gate_up_silu_bundle_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t op_version;
  uint32_t reserved[5];
  sllm_tensor_binding_t activation;
  sllm_tensor_binding_t gate_weight;
  sllm_tensor_binding_t up_weight;
  sllm_tensor_binding_t gate_output;
  sllm_tensor_binding_t up_output;
  sllm_tensor_binding_t silu_output;
} sllm_mlp_gate_up_silu_bundle_desc_t;

typedef struct sllm_mlp_gate_up_silu_bundle_dispatch_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t backend;
  uint64_t dispatch_id;
  uint32_t dispatch_count;
  uint32_t kernel_id;
  uint32_t workgroup_size_x;
  uint32_t grid_size_x;
  uint32_t fallback_allowed;
  uint32_t fallback_used;
  uint64_t m;
  uint64_t k;
  uint64_t n;
  uint64_t output_elements;
  char kernel_symbol[SLLM_HIP_MATMUL_KERNEL_SYMBOL_MAX];
  char device_symbol[SLLM_HIP_MATMUL_DEVICE_SYMBOL_MAX];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[8];
} sllm_mlp_gate_up_silu_bundle_dispatch_info_t;

/* Greedy BF16 logits reduction. logits is contiguous unquantized BF16 [M,V]
 * and output is contiguous unquantized I32 [M]. NaN rows produce -1; all
 * other ties select the smallest index. */
typedef struct sllm_argmax_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t op_version;
  uint32_t reserved[5];
  sllm_tensor_binding_t logits;
  sllm_tensor_binding_t output;
} sllm_argmax_desc_t;

typedef struct sllm_argmax_dispatch_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t backend;
  uint64_t dispatch_id;
  uint32_t dispatch_count;
  uint32_t kernel_id;
  uint32_t workgroup_size_x;
  uint32_t grid_size_x;
  uint64_t row_count;
  uint64_t vocab_size;
  uint32_t fallback_allowed;
  uint32_t fallback_used;
  char kernel_symbol[SLLM_HIP_ARGMAX_KERNEL_SYMBOL_MAX];
  char device_symbol[SLLM_HIP_ARGMAX_DEVICE_SYMBOL_MAX];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[8];
} sllm_argmax_dispatch_info_t;

/* Prepared, backend-resident categorical selection for exactly one BF16 logits
 * row (M=1).  All inputs and the fixed-size output record are tensor bindings
 * owned by the context.  The plan retains their backing buffers until release;
 * execute enqueues on the supplied queue and returns an asynchronous
 * completion.  The kernel performs finite validation, stable softmax, and
 * counter-based categorical sampling without a full-vocabulary D2H readback.
 * The selected record is the only output and is read with the normal selected
 * 16-byte D2H transfer path after completion. */
typedef struct sllm_token_selector_record_t {
  int32_t token_id;
  uint32_t status;
  float logprob;
  uint32_t reserved0;
} sllm_token_selector_record_t;

typedef struct sllm_token_selector_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t op_version;
  uint32_t reserved[4];
  sllm_tensor_binding_t logits;
  sllm_tensor_binding_t additive_logits;
  sllm_tensor_binding_t valid_mask;
  sllm_tensor_binding_t output;
  uint64_t vocab_size;
  float temperature;
  uint64_t seed;
  uint64_t counter;
} sllm_token_selector_desc_t;

typedef struct sllm_token_selector_dispatch_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t backend;
  uint64_t dispatch_id;
  uint32_t dispatch_count;
  uint32_t kernel_id;
  uint32_t workgroup_size_x;
  uint32_t grid_size_x;
  uint64_t vocab_size;
  uint32_t fallback_allowed;
  uint32_t fallback_used;
  uint32_t result_status;
  int32_t token_id;
  char kernel_symbol[SLLM_HIP_TOKEN_SELECTOR_KERNEL_SYMBOL_MAX];
  char device_symbol[SLLM_HIP_TOKEN_SELECTOR_DEVICE_SYMBOL_MAX];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[8];
} sllm_token_selector_dispatch_info_t;

/* Sparse-MoE routing consumes contiguous BF16 logits [M,E]. `metadata` is a
 * contiguous unquantized U8 byte buffer whose reviewed layout is:
 *   i32 expert_ids[M,K], f32 expert_weights[M,K], i32 counts[E],
 *   i32 offsets[E+1], i32 grouped_token_ids[M,K],
 *   i32 grouped_topk_slots[M,K], i32 status.
 * Status is zero on success and nonzero when any input row is nonfinite.
 * Ties select the smaller expert ID. Grouping is expert-major, then token,
 * then selected slot, without a host-side decision or sort. */
typedef struct sllm_moe_route_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t op_version;
  uint32_t selected_expert_count;
  uint32_t reserved[4];
  sllm_tensor_binding_t logits;
  sllm_tensor_binding_t metadata;
} sllm_moe_route_desc_t;

typedef struct sllm_moe_route_dispatch_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t backend;
  uint64_t dispatch_id;
  uint32_t dispatch_count;
  uint32_t kernel_id;
  uint32_t workgroup_size_x;
  uint32_t grid_size_x;
  uint64_t token_count;
  uint64_t expert_count;
  uint64_t pair_count;
  uint32_t selected_expert_count;
  uint32_t fallback_allowed;
  uint32_t fallback_used;
  uint32_t reserved0;
  char kernel_symbol[SLLM_HIP_MOE_ROUTE_KERNEL_SYMBOL_MAX];
  char device_symbol[SLLM_HIP_MOE_ROUTE_DEVICE_SYMBOL_MAX];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[8];
} sllm_moe_route_dispatch_info_t;

/* DeepSeek V4 score mode selects top-6 from
 *   sqrt(softplus(BF16 logits)) + F32 selection_bias.
 * The output weight always uses the unbiased sqrt(softplus(logit)) value.
 * Hash mode consumes I32 expert IDs [M,6] directly, requires the bias binding
 * to be all-zero, and still obtains weights from the unbiased BF16 logits.
 * Score mode conversely requires hash_expert_ids to be all-zero.  Duplicate
 * or out-of-range hash IDs and every nonfinite input fail closed through the
 * device-written status.
 *
 * `metadata` has the same stable grouped layout as sllm_moe_route_desc_t:
 *   i32 expert_ids[M,6], f32 expert_weights[M,6], i32 counts[256],
 *   i32 offsets[257], i32 grouped_token_ids[M,6],
 *   i32 grouped_topk_slots[M,6], i32 status.
 * If renormalize is nonzero, the six selected unbiased scores are normalized
 * before routed_scale is applied.  Otherwise each raw score is multiplied by
 * routed_scale. `routed_scale` must be finite and strictly positive.  It is
 * intentionally not fixed to one model-card default: the operator remains
 * reusable by reviewed DeepSeek V4 layer configurations while the model lock
 * owns the exact per-artifact value. */
typedef struct sllm_deepseek_v4_moe_route_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t op_version;
  sllm_deepseek_v4_moe_route_mode_t mode;
  uint32_t selected_expert_count;
  uint32_t renormalize;
  float routed_scale;
  uint32_t reserved0;
  uint32_t reserved[4];
  sllm_tensor_binding_t logits;
  sllm_tensor_binding_t selection_bias;
  sllm_tensor_binding_t hash_expert_ids;
  sllm_tensor_binding_t metadata;
} sllm_deepseek_v4_moe_route_desc_t;

typedef struct sllm_deepseek_v4_moe_route_query_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  sllm_deepseek_v4_moe_route_mode_t mode;
  uint64_t token_count;
  uint64_t expert_count;
  uint64_t pair_count;
  uint64_t metadata_bytes;
  uint32_t selected_expert_count;
  uint32_t renormalize;
  float routed_scale;
  uint32_t reserved0;
  uint32_t reserved[8];
} sllm_deepseek_v4_moe_route_query_info_t;

typedef struct sllm_deepseek_v4_moe_route_dispatch_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t backend;
  uint64_t dispatch_id;
  uint32_t dispatch_count;
  uint32_t kernel_id;
  uint32_t workgroup_size_x;
  uint32_t grid_size_x;
  uint64_t token_count;
  uint64_t expert_count;
  uint64_t pair_count;
  uint32_t selected_expert_count;
  sllm_deepseek_v4_moe_route_mode_t mode;
  uint32_t renormalize;
  uint32_t fallback_allowed;
  uint32_t fallback_used;
  uint32_t reserved0;
  char kernel_symbol[SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_KERNEL_SYMBOL_MAX];
  char device_symbol[SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_DEVICE_SYMBOL_MAX];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[8];
} sllm_deepseek_v4_moe_route_dispatch_info_t;

/* `metadata` uses the canonical grouped layout:
 *   i32 expert_ids[M,4], f32 expert_weights[M,4], i32 counts[128],
 *   i32 offsets[129], i32 grouped_token_ids[M,4],
 *   i32 grouped_topk_slots[M,4], i32 status.
 * The bias affects selection only. Weights use the unbiased sigmoid(logit),
 * are normalized over the four selected experts, then multiplied by 2.0. */
typedef struct sllm_minimax_m3_moe_route_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t op_version;
  uint32_t selected_expert_count;
  uint32_t reserved[4];
  sllm_tensor_binding_t logits;
  sllm_tensor_binding_t selection_bias;
  sllm_tensor_binding_t metadata;
} sllm_minimax_m3_moe_route_desc_t;

typedef struct sllm_minimax_m3_moe_route_query_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t selected_expert_count;
  uint64_t token_count;
  uint64_t expert_count;
  uint64_t pair_count;
  uint64_t metadata_bytes;
  uint32_t reserved[8];
} sllm_minimax_m3_moe_route_query_info_t;

typedef struct sllm_minimax_m3_moe_route_dispatch_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t backend;
  uint64_t dispatch_id;
  uint32_t dispatch_count;
  uint32_t kernel_id;
  uint32_t workgroup_size_x;
  uint32_t grid_size_x;
  uint64_t token_count;
  uint64_t expert_count;
  uint64_t pair_count;
  uint32_t selected_expert_count;
  uint32_t fallback_allowed;
  uint32_t fallback_used;
  uint32_t reserved0;
  char kernel_symbol[SLLM_HIP_MINIMAX_M3_MOE_ROUTE_KERNEL_SYMBOL_MAX];
  char device_symbol[SLLM_HIP_MINIMAX_M3_MOE_ROUTE_DEVICE_SYMBOL_MAX];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[8];
} sllm_minimax_m3_moe_route_dispatch_info_t;

/* One routed MoE layer. Version 1 is the byte-exact Qwen3.5 MXFP4 routed plus
 * BF16 shared-expert contract. Version 2 is Gemma 4 26B-A4B: 128 routed
 * experts, top-8, hidden 2816, intermediate 704, no shared branch. Its layer
 * blob stores projection-major gate/up/down planes. Each plane is all expert
 * packed E2M1 values, all block-16 E4M3FN scales, F32 outer scales, then F32
 * input scales; the three planes are followed by BF16 per-expert scales.
 * `routing_metadata` is the exact matching expert-count/top-k output byte
 * layout of sllm_moe_route_execute. */
typedef struct sllm_moe_expert_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t op_version;
  uint32_t reserved0;
  uint32_t reserved[4];
  sllm_tensor_binding_t hidden;
  sllm_tensor_binding_t routing_metadata;
  sllm_tensor_binding_t layer_blob;
  sllm_tensor_binding_t workspace;
  sllm_tensor_binding_t output;
} sllm_moe_expert_desc_t;

typedef struct sllm_moe_expert_dispatch_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t backend;
  uint64_t dispatch_id;
  uint32_t dispatch_count;
  uint32_t kernel_id;
  uint32_t workgroup_size_x;
  uint32_t grid_size_x;
  uint64_t token_count;
  uint64_t active_pair_count;
  uint64_t workspace_bytes;
  uint32_t selected_expert_count;
  uint32_t shared_expert_count;
  uint32_t fallback_allowed;
  uint32_t fallback_used;
  char kernel_symbol[SLLM_HIP_MOE_EXPERT_KERNEL_SYMBOL_MAX];
  char device_symbol[SLLM_HIP_MOE_EXPERT_DEVICE_SYMBOL_MAX];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[8];
} sllm_moe_expert_dispatch_info_t;

/* C3a1 text-only attention preprocessing. The packed Q/gate input is exactly
 * [M, 16, 512], with each head's final axis [Q 256, gate 256]. K is [M, 4,
 * 256]. All eight tensor bindings are required to be contiguous and their
 * byte ranges must not overlap; the runtime retains every unique backing
 * buffer through the asynchronous completion. reserved[0] selects the
 * position payload mode. DERIVED_CONTIGUOUS computes start_position+row in
 * the kernel and does not read the otherwise ABI-required positions payload. */
typedef struct sllm_attention_preprocess_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t op_version;
  uint32_t start_position;
  uint32_t reserved[4];
  sllm_tensor_binding_t packed_q_gate;
  sllm_tensor_binding_t k;
  sllm_tensor_binding_t q_raw_scale;
  sllm_tensor_binding_t k_raw_scale;
  sllm_tensor_binding_t positions;
  sllm_tensor_binding_t q_output;
  sllm_tensor_binding_t gate_output;
  sllm_tensor_binding_t k_output;
} sllm_attention_preprocess_desc_t;

typedef struct sllm_attention_preprocess_dispatch_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t backend;
  uint64_t dispatch_id;
  uint32_t dispatch_count;
  uint32_t kernel_id;
  uint32_t workgroup_size_x;
  uint32_t grid_size_x;
  uint64_t m;
  uint32_t q_heads;
  uint32_t k_heads;
  uint32_t q_head_dim;
  uint32_t k_head_dim;
  uint32_t rotary_dim;
  uint32_t start_position;
  uint32_t fallback_allowed;
  uint32_t fallback_used;
  char kernel_symbol[SLLM_HIP_ATTENTION_PREPROCESS_KERNEL_SYMBOL_MAX];
  char device_symbol[SLLM_HIP_ATTENTION_PREPROCESS_DEVICE_SYMBOL_MAX];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[8];
} sllm_attention_preprocess_dispatch_info_t;

/* Gemma split-half rotary position encoding. Query and key are contiguous
 * BF16 [M,heads,head_dim], positions is contiguous I32 [M], and outputs match
 * their respective inputs. The active rotary dimensions are paired across
 * the two halves of head_dim. All five byte ranges must be disjoint when they
 * share a backing buffer. */
typedef struct sllm_rotary_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t op_version;
  uint32_t reserved0;
  uint64_t start_position;
  uint32_t q_heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t rotary_dim;
  uint32_t theta_bits;
  uint32_t max_position;
  uint32_t reserved[2];
  sllm_tensor_binding_t query;
  sllm_tensor_binding_t key;
  sllm_tensor_binding_t positions;
  sllm_tensor_binding_t query_output;
  sllm_tensor_binding_t key_output;
} sllm_rotary_desc_t;

typedef struct sllm_rotary_dispatch_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t backend;
  uint64_t dispatch_id;
  uint32_t dispatch_count;
  uint32_t kernel_id;
  uint32_t workgroup_size_x;
  uint32_t grid_size_x;
  uint64_t token_count;
  uint32_t q_heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t rotary_dim;
  uint32_t start_position;
  uint32_t max_position;
  uint32_t fallback_allowed;
  uint32_t fallback_used;
  char kernel_symbol[SLLM_HIP_ROTARY_KERNEL_SYMBOL_MAX];
  char device_symbol[SLLM_HIP_ROTARY_DEVICE_SYMBOL_MAX];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[8];
} sllm_rotary_dispatch_info_t;

/* Fixed Ministral 3 BF16 Q/K split-half YaRN. Inputs and outputs are
 * contiguous [M,32,128] and [M,8,128], positions is contiguous I32 [M].
 * Parameters are carried as IEEE-754 binary32 bits and are validated against
 * the reviewed checkpoint constants. Query receives the Llama-4
 * position-dependent scale after rotation; key never receives that scale. */
typedef struct sllm_ministral3_yarn_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t op_version;
  uint32_t position_payload_mode;
  uint64_t start_position;
  uint32_t q_heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t rotary_dim;
  uint32_t theta_bits;
  uint32_t factor_bits;
  uint32_t original_context;
  uint32_t max_position;
  uint32_t beta_fast_bits;
  uint32_t beta_slow_bits;
  uint32_t query_scale_beta_bits;
  uint32_t reserved[5];
  sllm_tensor_binding_t query;
  sllm_tensor_binding_t key;
  sllm_tensor_binding_t positions;
  sllm_tensor_binding_t query_output;
  sllm_tensor_binding_t key_output;
} sllm_ministral3_yarn_desc_t;

typedef struct sllm_ministral3_yarn_dispatch_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t backend;
  uint64_t dispatch_id;
  uint32_t dispatch_count;
  uint32_t kernel_id;
  uint32_t workgroup_size_x;
  uint32_t grid_size_x;
  uint64_t token_count;
  uint32_t q_heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t rotary_dim;
  uint32_t start_position;
  uint32_t max_position;
  uint32_t fallback_allowed;
  uint32_t fallback_used;
  char kernel_symbol[SLLM_HIP_MINISTRAL3_YARN_KERNEL_SYMBOL_MAX];
  char device_symbol[SLLM_HIP_MINISTRAL3_YARN_DEVICE_SYMBOL_MAX];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[8];
} sllm_ministral3_yarn_dispatch_info_t;

/* Model-neutral BF16 GQA causal attention. K/V are complete token-major
 * histories [expected_kv_length,kv_heads,head_dim]. A zero sliding window
 * selects full attention; otherwise the window is inclusive of the current
 * token. The baseline provider supports the explicit scale 1.0 only. */
typedef struct sllm_windowed_attention_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t op_version;
  uint32_t reserved0;
  uint64_t start_position;
  uint64_t expected_kv_length;
  uint64_t sliding_window;
  uint32_t q_heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t scaling_bits;
  uint32_t reserved[4];
  sllm_tensor_binding_t query;
  sllm_tensor_binding_t key;
  sllm_tensor_binding_t value;
  sllm_tensor_binding_t output;
} sllm_windowed_attention_desc_t;

typedef struct sllm_windowed_attention_dispatch_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t backend;
  uint64_t dispatch_id;
  uint32_t dispatch_count;
  uint32_t kernel_id;
  uint32_t workgroup_size_x;
  uint32_t grid_size_x;
  uint64_t query_count;
  uint64_t start_position;
  uint64_t committed_kv_length;
  uint64_t sliding_window;
  uint32_t q_heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  uint32_t scaling_bits;
  uint32_t fallback_allowed;
  uint32_t fallback_used;
  char kernel_symbol[SLLM_HIP_WINDOWED_ATTENTION_KERNEL_SYMBOL_MAX];
  char device_symbol[SLLM_HIP_WINDOWED_ATTENTION_DEVICE_SYMBOL_MAX];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[8];
} sllm_windowed_attention_dispatch_info_t;

/* A request-local full-attention KV state owns separate K and V virtual
 * address reservations. Physical pages grow on demand. The allocations are
 * logically FP16 [capacity, head_count, head_dim]; no
 * query-head repetition is materialized.  session_id is an application
 * identity checked together with the context and is never dereferenced by
 * the runtime. */
typedef struct sllm_kv_state_create_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t session_id;
  uint32_t layer_id;
  uint32_t flags;
  uint64_t capacity_tokens;
  uint32_t head_count;
  uint32_t head_dim;
  uint32_t memory_kind;
  uint32_t layout;
} sllm_kv_state_create_info_t;

/* Additive low-bit create contract. The legacy create function remains the
 * exact FP16 v1 ABI above. Low-bit values and their scale planes are owned by
 * the opaque state and are never exposed as scheduler-visible pointers.
 * create_info_version=3 uses reserved[0:2] for the unit static FP8 decode
 * scales and reserved[2:4] for a little-endian uint64 sliding window. */
typedef struct sllm_kv_state_create_info_v2_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t create_info_version;
  uint32_t reserved0;
  uint64_t session_id;
  uint32_t layer_id;
  uint32_t flags;
  uint64_t capacity_tokens;
  uint32_t head_count;
  uint32_t head_dim;
  uint32_t memory_kind;
  uint32_t layout;
  uint32_t dtype;
  uint32_t encoding;
  uint32_t block_size;
  uint32_t scale_dtype;
  uint32_t reserved[4];
} sllm_kv_state_create_info_v2_t;

typedef struct sllm_kv_view_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t reserved0;
  uint64_t session_id;
  uint32_t layer_id;
  uint32_t dtype;
  uint32_t encoding;
  uint32_t head_count;
  uint32_t head_dim;
  uint32_t memory_kind;
  uint32_t layout;
  uint32_t reserved1;
  uint64_t capacity_tokens;
  uint64_t observed_length;
  uint64_t generation;
  uint64_t physical_page_bytes;
  uint64_t tokens_per_page;
  uint64_t mapped_token_capacity;
  uint64_t committed_bytes_per_plane;
  uint64_t context_identity;
  uint64_t state_identity;
  uint64_t k_stride_elements[3];
  uint64_t v_stride_elements[3];
  uint32_t reserved[4];
} sllm_kv_view_info_t;

/* Append inputs are independent, read-only BF16 [M, 4, 256] bindings.  The
 * expected length and start position must both equal the state's published
 * length at submission. */
typedef struct sllm_kv_append_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t append_version;
  uint32_t reserved0;
  uint64_t expected_length;
  uint64_t start_position;
  sllm_tensor_binding_t key_input;
  sllm_tensor_binding_t value_input;
  uint32_t reserved[4];
} sllm_kv_append_desc_t;

typedef struct sllm_kv_append_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t backend;
  uint64_t dispatch_id;
  uint32_t dispatch_count;
  uint32_t kernel_id;
  uint32_t workgroup_size_x;
  uint32_t grid_size_x;
  uint64_t start_position;
  uint64_t token_count;
  uint64_t end_position;
  uint32_t commit_allowed;
  uint32_t fallback_allowed;
  uint32_t fallback_used;
  uint32_t reserved0;
  char kernel_symbol[SLLM_HIP_KV_KERNEL_SYMBOL_MAX];
  char device_symbol[SLLM_HIP_KV_DEVICE_SYMBOL_MAX];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[8];
} sllm_kv_append_info_t;

/* C3b causal full/windowed attention. Q and output are contiguous unquantized
 * BF16 [M,Hq,D], including the reviewed [M,16,256] and [M,32,128] shapes. The
 * referenced state is one committed token-major FP16, FP8, or packed NVFP4
 * [capacity,Hkv,D] snapshot; no repeated or unpacked K/V payload is part of
 * this descriptor. op_version=2 stores a little-endian
 * uint64 sliding window in reserved[0:2]; reserved[2:4] remain zero.
 * op_version=3 stores an optional sliding window in reserved[0:2], an exact
 * positive finite binary32 score scale in reserved[2], and zero in
 * reserved[3]. Version 3 is baseline-provider-only. */
typedef struct sllm_causal_attention_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t op_version;
  uint32_t reserved0;
  uint64_t start_position;
  uint64_t expected_kv_length;
  const sllm_kv_state_t *kv_state;
  sllm_tensor_binding_t query;
  sllm_tensor_binding_t output;
  uint32_t reserved[4];
} sllm_causal_attention_desc_t;

typedef struct sllm_causal_attention_dispatch_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t backend;
  uint64_t dispatch_id;
  uint32_t dispatch_count;
  uint32_t kernel_id;
  uint32_t workgroup_size_x;
  uint32_t grid_size_x;
  uint64_t query_count;
  uint64_t start_position;
  uint64_t committed_kv_length;
  uint32_t q_heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  /* Exact integer denominator for the implicit 1/sqrt(head_dim) scale. When
   * sqrt(head_dim) is non-integral this is zero and reserved[4] contains the
   * exact binary32 scale bits with reserved[5] equal to one. */
  uint32_t scale_denominator;
  uint32_t fallback_allowed;
  uint32_t fallback_used;
  char kernel_symbol[SLLM_HIP_CAUSAL_ATTENTION_KERNEL_SYMBOL_MAX];
  char device_symbol[SLLM_HIP_CAUSAL_ATTENTION_DEVICE_SYMBOL_MAX];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[8];
} sllm_causal_attention_dispatch_info_t;

/* A request-local linear-attention state owns two BF16
 * [conv_kernel_size-1,qkv_width] convolution-history slots and two F32
 * [value_heads,head_dim,head_dim] recurrent slots. The
 * inactive pair is published only after successful completion. */
typedef struct sllm_linear_attention_state_create_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t session_id;
  uint32_t layer_id;
  uint32_t flags;
  uint64_t capacity_tokens;
  uint32_t qk_heads;
  uint32_t value_heads;
  uint32_t head_dim;
  uint32_t conv_kernel_size;
} sllm_linear_attention_state_create_info_t;

typedef struct sllm_linear_attention_view_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t reserved0;
  uint64_t session_id;
  uint32_t layer_id;
  uint32_t conv_state_dtype;
  uint32_t recurrent_state_dtype;
  uint32_t encoding;
  uint32_t active_slot;
  uint64_t capacity_tokens;
  uint64_t observed_length;
  uint64_t generation;
  uint64_t context_identity;
  uint64_t state_identity;
  uint64_t conv_state_shape[2];
  uint64_t recurrent_state_shape[3];
  uint32_t reserved[4];
} sllm_linear_attention_view_info_t;

typedef struct sllm_state_fork_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t mode;
  uint64_t source_state_identity;
  uint64_t child_state_identity;
  uint64_t source_owned_bytes;
  uint64_t child_owned_bytes;
  uint64_t copied_bytes;
  uint64_t shared_bytes;
  uint64_t published_length;
  uint64_t page_bytes;
  uint32_t reserved[4];
} sllm_state_fork_info_t;

/* Chunk operations copy the exact native encoding.  No FP16 conversion,
 * host replay, or scheduler-visible pointers are involved. */
typedef struct sllm_state_chunk_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t plane;
  uint32_t reserved0;
  uint32_t reserved1;
  uint64_t byte_offset;
  uint64_t byte_length;
  void *host_pointer;
  uint64_t host_capacity;
  uint32_t reserved[4];
} sllm_state_chunk_t;

typedef struct sllm_state_image_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t reserved0;
  uint64_t session_id;
  uint32_t layer_id;
  uint32_t dtype;
  uint32_t encoding;
  uint32_t active_slot;
  uint64_t capacity_tokens;
  uint64_t published_length;
  uint64_t generation;
  uint32_t plane_count;
  uint32_t reserved[7];
} sllm_state_image_info_t;

/* Projected inputs use qkv BF16 [M,8192], z BF16 [M,4096], b/a BF16
 * [M,32]. Convolution weight is BF16 [8192,1,4], A_log is F32 [32],
 * dt_bias is BF16 [32], norm_weight is raw F32 [128], and output is BF16
 * [M,4096]. All bindings are contiguous, unquantized and pairwise disjoint. */
typedef struct sllm_linear_attention_desc_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t op_version;
  uint32_t reserved0;
  uint64_t start_position;
  uint64_t expected_length;
  const sllm_linear_attention_state_t *state;
  sllm_tensor_binding_t qkv;
  sllm_tensor_binding_t z;
  sllm_tensor_binding_t b_input;
  sllm_tensor_binding_t a_input;
  sllm_tensor_binding_t conv_weight;
  sllm_tensor_binding_t a_log;
  sllm_tensor_binding_t dt_bias;
  sllm_tensor_binding_t norm_weight;
  sllm_tensor_binding_t output;
  uint32_t reserved[4];
} sllm_linear_attention_desc_t;

typedef struct sllm_linear_attention_dispatch_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t info_version;
  uint32_t backend;
  uint64_t dispatch_id;
  uint32_t dispatch_count;
  uint32_t conv_kernel_id;
  uint32_t recurrent_kernel_id;
  uint32_t workgroup_size_x;
  uint32_t conv_grid_size_x;
  uint32_t recurrent_grid_size_x;
  uint64_t token_count;
  uint64_t start_position;
  uint64_t expected_length;
  uint32_t qk_heads;
  uint32_t value_heads;
  uint32_t head_dim;
  uint32_t fallback_allowed;
  uint32_t fallback_used;
  char kernel_symbol[SLLM_HIP_LINEAR_ATTENTION_KERNEL_SYMBOL_MAX];
  char conv_device_symbol[SLLM_HIP_LINEAR_ATTENTION_DEVICE_SYMBOL_MAX];
  char recurrent_device_symbol[SLLM_HIP_LINEAR_ATTENTION_DEVICE_SYMBOL_MAX];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[8];
} sllm_linear_attention_dispatch_info_t;

#ifdef __cplusplus
extern "C" {
#endif

SLLM_HIP_API sllm_status_t sllm_get_abi_version(
    uint32_t *abi_version, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_query_version(sllm_version_info_t *version,
                                              sllm_error_sink_t *error_sink)
    SLLM_HIP_NOEXCEPT;

/* The host build returns SLLM_STATUS_HIP_UNAVAILABLE.  The HIP build reports
 * the visible device set without CPU fallback. */
SLLM_HIP_API sllm_status_t
sllm_backend_probe(uint32_t backend, sllm_backend_probe_result_t *result,
                   sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

/* A null context reports runtime availability.  An opaque non-null context is
 * validated by the HIP build and is never dereferenced by callers. */
SLLM_HIP_API sllm_status_t sllm_context_probe(
    const sllm_context_t *context, sllm_context_probe_result_t *result,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_device_count(
    uint32_t *count, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_device_query(uint32_t device_index, sllm_device_info_t *info,
                  sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_context_create(
    const sllm_context_create_info_t *info, sllm_context_t **context,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_context_release(
    sllm_context_t **context, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_queue_create(
    const sllm_context_t *context, const sllm_queue_create_info_t *info,
    sllm_queue_t **queue, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_queue_release(
    sllm_queue_t **queue, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_buffer_create(
    const sllm_context_t *context, const sllm_buffer_create_info_t *info,
    sllm_buffer_t **buffer, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_buffer_release(
    sllm_buffer_t **buffer, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_buffer_size(const sllm_buffer_t *buffer, uint64_t *size_bytes,
                 sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_event_create(const sllm_context_t *context, sllm_event_t **event,
                  sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_event_release(
    sllm_event_t **event, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_buffer_copy_h2d(
    const sllm_queue_t *queue, const sllm_buffer_t *buffer,
    const sllm_transfer_desc_t *transfer, sllm_completion_t **completion,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_buffer_copy_d2h(
    const sllm_queue_t *queue, const sllm_buffer_t *buffer,
    const sllm_transfer_desc_t *transfer, sllm_completion_t **completion,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_buffer_copy_d2d(
    const sllm_queue_t *queue, const sllm_buffer_t *source,
    const sllm_buffer_t *destination, const sllm_buffer_copy_d2d_desc_t *copy,
    sllm_completion_t **completion,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

/* Production execution may defer numeric-operation completion events until
 * one ordered queue fence closes a model-neutral segment.  PROFILED remains
 * the default and preserves standalone per-operation timing semantics. */
SLLM_HIP_API sllm_status_t sllm_queue_set_completion_mode(
    const sllm_queue_t *queue, sllm_queue_completion_mode_t mode,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_queue_fence(const sllm_queue_t *queue, sllm_completion_t **completion,
                 sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

/* Finalizes an eventless numeric completion after a successful fence on the
 * same context, queue, and stream. */
SLLM_HIP_API sllm_status_t sllm_completion_finalize_after(
    sllm_completion_t *completion, sllm_completion_t *fence,
    sllm_completion_result_t *result,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_completion_query(
    sllm_completion_t *completion, sllm_completion_result_t *result,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_completion_wait(sllm_completion_t *completion, uint32_t timeout_ms,
                     sllm_completion_result_t *result,
                     sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_completion_read(sllm_completion_t *completion, void *destination,
                     uint64_t destination_capacity, uint64_t *bytes_written,
                     sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

/* Numeric-operation completions retain a pair of timing-enabled HIP events. The
 * elapsed value is available only after successful completion and is never a
 * host-clock or CPU-fallback estimate.  Other completion kinds return
 * SLLM_STATUS_UNSUPPORTED. */
SLLM_HIP_API sllm_status_t sllm_completion_timing(
    sllm_completion_t *completion, sllm_completion_timing_t *timing,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_completion_release(sllm_completion_t **completion,
                        sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

/* RMSNorm preparation captures immutable metadata; execution is a separate
 * asynchronous baseline dispatch operation on the reusable plan. */
SLLM_HIP_API sllm_status_t sllm_rmsnorm_prepare(
    const sllm_context_t *context, const sllm_rmsnorm_desc_t *descriptor,
    sllm_rmsnorm_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_rmsnorm_plan_release(sllm_rmsnorm_plan_t **plan,
                          sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_rmsnorm_execute(
    const sllm_rmsnorm_plan_t *plan, const sllm_queue_t *queue,
    sllm_completion_t **completion, sllm_rmsnorm_dispatch_info_t *dispatch_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_residual_rmsnorm_prepare(const sllm_context_t *context,
                              const sllm_residual_rmsnorm_desc_t *descriptor,
                              sllm_residual_rmsnorm_plan_t **plan,
                              sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_residual_rmsnorm_plan_release(
    sllm_residual_rmsnorm_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_residual_rmsnorm_execute(
    const sllm_residual_rmsnorm_plan_t *plan, const sllm_queue_t *queue,
    sllm_completion_t **completion,
    sllm_residual_rmsnorm_dispatch_info_t *dispatch_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_elementwise_prepare(
    const sllm_context_t *context, const sllm_elementwise_desc_t *descriptor,
    sllm_elementwise_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_elementwise_plan_release(sllm_elementwise_plan_t **plan,
                              sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_elementwise_execute(
    const sllm_elementwise_plan_t *plan, const sllm_queue_t *queue,
    sllm_completion_t **completion,
    sllm_elementwise_dispatch_info_t *dispatch_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_embedding_prepare(
    const sllm_context_t *context, const sllm_embedding_desc_t *descriptor,
    sllm_embedding_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_embedding_plan_release(sllm_embedding_plan_t **plan,
                            sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_embedding_execute(
    const sllm_embedding_plan_t *plan, const sllm_queue_t *queue,
    sllm_completion_t **completion,
    sllm_embedding_dispatch_info_t *dispatch_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_matmul_prepare(
    const sllm_context_t *context, const sllm_matmul_desc_t *descriptor,
    sllm_matmul_plan_t **plan, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_matmul_plan_release(
    sllm_matmul_plan_t **plan, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_matmul_execute(
    const sllm_matmul_plan_t *plan, const sllm_queue_t *queue,
    sllm_completion_t **completion, sllm_matmul_dispatch_info_t *dispatch_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_gdn_projection_bundle_prepare(
    const sllm_context_t *context,
    const sllm_gdn_projection_bundle_desc_t *descriptor,
    sllm_gdn_projection_bundle_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_gdn_projection_bundle_plan_release(
    sllm_gdn_projection_bundle_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_gdn_projection_bundle_execute(
    const sllm_gdn_projection_bundle_plan_t *plan, const sllm_queue_t *queue,
    sllm_completion_t **completion,
    sllm_gdn_projection_bundle_dispatch_info_t *dispatch_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_mlp_gate_up_silu_bundle_prepare(
    const sllm_context_t *context,
    const sllm_mlp_gate_up_silu_bundle_desc_t *descriptor,
    sllm_mlp_gate_up_silu_bundle_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_mlp_gate_up_silu_bundle_plan_release(
    sllm_mlp_gate_up_silu_bundle_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_mlp_gate_up_silu_bundle_execute(
    const sllm_mlp_gate_up_silu_bundle_plan_t *plan, const sllm_queue_t *queue,
    sllm_completion_t **completion,
    sllm_mlp_gate_up_silu_bundle_dispatch_info_t *dispatch_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_argmax_prepare(
    const sllm_context_t *context, const sllm_argmax_desc_t *descriptor,
    sllm_argmax_plan_t **plan, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_argmax_plan_release(
    sllm_argmax_plan_t **plan, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_argmax_execute(
    const sllm_argmax_plan_t *plan, const sllm_queue_t *queue,
    sllm_completion_t **completion, sllm_argmax_dispatch_info_t *dispatch_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_token_selector_prepare(
    const sllm_context_t *context, const sllm_token_selector_desc_t *descriptor,
    sllm_token_selector_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_token_selector_plan_release(
    sllm_token_selector_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_token_selector_execute(
    const sllm_token_selector_plan_t *plan, const sllm_queue_t *queue,
    sllm_completion_t **completion,
    sllm_token_selector_dispatch_info_t *dispatch_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_moe_route_prepare(
    const sllm_context_t *context, const sllm_moe_route_desc_t *descriptor,
    sllm_moe_route_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_moe_route_plan_release(sllm_moe_route_plan_t **plan,
                            sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_moe_route_execute(
    const sllm_moe_route_plan_t *plan, const sllm_queue_t *queue,
    sllm_completion_t **completion,
    sllm_moe_route_dispatch_info_t *dispatch_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

/* query validates the complete copied descriptor and reports its checked
 * shape/layout without dereferencing or retaining buffer handles. */
SLLM_HIP_API sllm_status_t sllm_deepseek_v4_moe_route_query(
    const sllm_deepseek_v4_moe_route_desc_t *descriptor,
    sllm_deepseek_v4_moe_route_query_info_t *info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_deepseek_v4_moe_route_prepare(
    const sllm_context_t *context,
    const sllm_deepseek_v4_moe_route_desc_t *descriptor,
    sllm_deepseek_v4_moe_route_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_deepseek_v4_moe_route_plan_release(
    sllm_deepseek_v4_moe_route_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_deepseek_v4_moe_route_execute(
    const sllm_deepseek_v4_moe_route_plan_t *plan,
    const sllm_queue_t *queue, sllm_completion_t **completion,
    sllm_deepseek_v4_moe_route_dispatch_info_t *dispatch_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_minimax_m3_moe_route_query(
    const sllm_minimax_m3_moe_route_desc_t *descriptor,
    sllm_minimax_m3_moe_route_query_info_t *info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_minimax_m3_moe_route_prepare(
    const sllm_context_t *context,
    const sllm_minimax_m3_moe_route_desc_t *descriptor,
    sllm_minimax_m3_moe_route_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_minimax_m3_moe_route_plan_release(
    sllm_minimax_m3_moe_route_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_minimax_m3_moe_route_execute(
    const sllm_minimax_m3_moe_route_plan_t *plan,
    const sllm_queue_t *queue, sllm_completion_t **completion,
    sllm_minimax_m3_moe_route_dispatch_info_t *dispatch_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_moe_expert_prepare(
    const sllm_context_t *context, const sllm_moe_expert_desc_t *descriptor,
    sllm_moe_expert_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_moe_expert_plan_release(sllm_moe_expert_plan_t **plan,
                             sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_moe_expert_execute(
    const sllm_moe_expert_plan_t *plan, const sllm_queue_t *queue,
    sllm_completion_t **completion,
    sllm_moe_expert_dispatch_info_t *dispatch_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_attention_preprocess_prepare(
    const sllm_context_t *context,
    const sllm_attention_preprocess_desc_t *descriptor,
    sllm_attention_preprocess_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_attention_preprocess_plan_release(
    sllm_attention_preprocess_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_attention_preprocess_execute(
    const sllm_attention_preprocess_plan_t *plan, const sllm_queue_t *queue,
    sllm_completion_t **completion,
    sllm_attention_preprocess_dispatch_info_t *dispatch_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_rotary_prepare(
    const sllm_context_t *context, const sllm_rotary_desc_t *descriptor,
    sllm_rotary_plan_t **plan, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_rotary_plan_release(
    sllm_rotary_plan_t **plan, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_rotary_execute(
    const sllm_rotary_plan_t *plan, const sllm_queue_t *queue,
    sllm_completion_t **completion, sllm_rotary_dispatch_info_t *dispatch_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_ministral3_yarn_prepare(
    const sllm_context_t *context,
    const sllm_ministral3_yarn_desc_t *descriptor,
    sllm_ministral3_yarn_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_ministral3_yarn_plan_release(
    sllm_ministral3_yarn_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_ministral3_yarn_execute(
    const sllm_ministral3_yarn_plan_t *plan, const sllm_queue_t *queue,
    sllm_completion_t **completion,
    sllm_ministral3_yarn_dispatch_info_t *dispatch_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_windowed_attention_prepare(
    const sllm_context_t *context,
    const sllm_windowed_attention_desc_t *descriptor,
    sllm_windowed_attention_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_windowed_attention_plan_release(
    sllm_windowed_attention_plan_t **plan,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_windowed_attention_execute(
    const sllm_windowed_attention_plan_t *plan, const sllm_queue_t *queue,
    sllm_completion_t **completion,
    sllm_windowed_attention_dispatch_info_t *dispatch_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_kv_state_create(
    const sllm_context_t *context, const sllm_kv_state_create_info_t *info,
    sllm_kv_state_t **state, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_kv_state_create_v2(
    const sllm_context_t *context, const sllm_kv_state_create_info_v2_t *info,
    sllm_kv_state_t **state, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_kv_state_release(
    sllm_kv_state_t **state, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_kv_state_query(const sllm_kv_state_t *state, sllm_kv_view_info_t *info,
                    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

/* Rewinds a quiescent committed KV tail to an earlier length. The expected
 * current length makes stale rollback fail closed. Tail bytes are left
 * inaccessible and may be overwritten by later appends. */
SLLM_HIP_API sllm_status_t sllm_kv_state_rewind_last(
    const sllm_kv_state_t *state, uint64_t expected_length,
    uint64_t rewind_length, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_kv_state_snapshot(const sllm_kv_state_t *state, sllm_kv_view_t **view,
                       sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_kv_view_query(const sllm_kv_view_t *view, sllm_kv_view_info_t *info,
                   sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_kv_view_release(
    sllm_kv_view_t **view, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_kv_state_append(
    const sllm_kv_state_t *state, const sllm_queue_t *queue,
    const sllm_kv_append_desc_t *descriptor, sllm_completion_t **completion,
    sllm_kv_append_info_t *append_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

/* Revokes publication for a pending append.  The completion remains owned by
 * the caller and must still be queried/released after the GPU reaches a safe
 * terminal state. */
SLLM_HIP_API sllm_status_t sllm_kv_state_append_cancel(
    const sllm_kv_state_t *state, sllm_completion_t *completion,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_kv_state_fork(const sllm_kv_state_t *source,
                   const sllm_kv_state_create_info_v2_t *destination_info,
                   sllm_kv_state_t **child, sllm_state_fork_info_t *fork_info,
                   sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_kv_state_fork_query(
    const sllm_kv_state_t *state, sllm_state_fork_info_t *fork_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_kv_state_export(
    const sllm_kv_state_t *state, const sllm_state_chunk_t *chunk,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_kv_state_import(
    const sllm_kv_state_t *state, const sllm_state_chunk_t *chunk,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_kv_state_image_query(
    const sllm_kv_state_t *state, sllm_state_image_info_t *image_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_kv_state_image_plane_size(
    const sllm_kv_state_t *state, uint32_t plane, uint64_t *size_bytes,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_kv_state_import_finalize(
    const sllm_kv_state_t *state, const sllm_state_image_info_t *image_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_causal_attention_execute(
    const sllm_context_t *context, const sllm_queue_t *queue,
    const sllm_causal_attention_desc_t *descriptor,
    sllm_completion_t **completion,
    sllm_causal_attention_dispatch_info_t *dispatch_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_linear_attention_state_create(
    const sllm_context_t *context,
    const sllm_linear_attention_state_create_info_t *info,
    sllm_linear_attention_state_t **state,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_linear_attention_state_release(
    sllm_linear_attention_state_t **state,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_linear_attention_state_query(
    const sllm_linear_attention_state_t *state,
    sllm_linear_attention_view_info_t *info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

/* Rewinds exactly the most recently published transition by restoring the
 * prior double-buffer slot. */
SLLM_HIP_API sllm_status_t sllm_linear_attention_state_rewind_last(
    const sllm_linear_attention_state_t *state, uint64_t expected_length,
    uint64_t rewind_length, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_linear_attention_state_fork(
    const sllm_linear_attention_state_t *source,
    const sllm_linear_attention_state_create_info_t *destination_info,
    sllm_linear_attention_state_t **child, sllm_state_fork_info_t *fork_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_linear_attention_state_export(
    const sllm_linear_attention_state_t *state, const sllm_state_chunk_t *chunk,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_linear_attention_state_import(
    const sllm_linear_attention_state_t *state, const sllm_state_chunk_t *chunk,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_linear_attention_state_image_query(
    const sllm_linear_attention_state_t *state,
    sllm_state_image_info_t *image_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_linear_attention_state_image_plane_size(
    const sllm_linear_attention_state_t *state, uint32_t plane,
    uint64_t *size_bytes, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_linear_attention_state_import_finalize(
    const sllm_linear_attention_state_t *state,
    const sllm_state_image_info_t *image_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_linear_attention_execute(
    const sllm_context_t *context, const sllm_queue_t *queue,
    const sllm_linear_attention_desc_t *descriptor,
    sllm_completion_t **completion,
    sllm_linear_attention_dispatch_info_t *dispatch_info,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_linear_attention_cancel(
    const sllm_linear_attention_state_t *state, sllm_completion_t *completion,
    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SLLM_HIP_H */

#undef SLLM_HIP_NOEXCEPT
