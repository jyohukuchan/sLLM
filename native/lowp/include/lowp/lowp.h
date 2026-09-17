#ifndef LOWP_LOWP_H
#define LOWP_LOWP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LOWP_ABI_VERSION 1U
#define LOWP_MXFP4_W4A8_CONTRACT_VERSION 1U

/* Launch results preserve HIP error numbers; host planning needs no HIP header.
 */
typedef int32_t lowp_status_t;
enum {
  LOWP_SUCCESS = 0,
  LOWP_INVALID_ARGUMENT = 1,
  LOWP_OUT_OF_MEMORY = 2,
  LOWP_NOT_SUPPORTED = 801
};

typedef uint32_t lowp_format_t;
enum {
  LOWP_MXFP8_E4M3_W8A8 = 0,
  LOWP_MXFP6_E3M2_W6A6 = 1,
  LOWP_NVFP4_W4A16 = 2,
  LOWP_NVFP4_W4A4 = 3,
  /* Value 4 is reserved and is never a public format. */
  LOWP_FP8_OUTER_E4M3_W8A8 = 5,
  LOWP_MXFP8_E4M3_W8A16 = 6,
  LOWP_MXFP6_E3M2_W6A16 = 7,
  /* Defined contract, not implemented: E2M1/E8M0 W4 + E4M3/E8M0 A8. */
  LOWP_MXFP4_W4A8_V1 = 8
};

typedef uint32_t lowp_target_t;
enum {
  LOWP_TARGET_UNKNOWN = 0,
  LOWP_TARGET_GFX1030 = 1,
  LOWP_TARGET_GFX1201 = 2,
  LOWP_TARGET_GFX942_SRAMECC_ON_XNACK_OFF = 3
};

typedef uint32_t lowp_layout_t;
enum {
  LOWP_ROW_MAJOR = 0,
  LOWP_ROW_MAJOR_BLOCK_SCALED = 1,
  LOWP_CONSUMER_TILED_BLOCK_SCALED = 2
};

typedef struct lowp_format_info {
  uint32_t contract_version;
  uint32_t weight_bits, activation_bits;
  uint32_t weight_block_size, activation_block_size;
  uint32_t weight_scale_kind, activation_scale_kind;
  uint32_t weight_tensor_scale, activation_tensor_scale;
} lowp_format_info_t;

typedef struct lowp_matmul_request {
  uint32_t struct_size, version;
  lowp_format_t format;
  lowp_target_t target;
  lowp_layout_t weight_layout, activation_layout;
  uint64_t m, n, k;
} lowp_matmul_request_t;

/* A caller-owned prepared value. Do not modify it after successful planning.
 * Provider, variant, tile and inner_product retain their historical audit IDs.
 * Matmul variants are sampled during planning. Existing quantizer launcher
 * controls retain their behavior. */
typedef struct lowp_matmul_plan_v1 {
  uint32_t struct_size, version;
  lowp_matmul_request_t request;
  uint32_t provider, variant, tile, inner_product, activation_pack;
  uint32_t rejection, supported, selector_supported, selector_enabled, adopted;
  uint32_t launch_flags;
  uint64_t activation_value_bytes, activation_scale_offset,
      activation_scale_bytes;
  uint64_t weight_value_bytes, weight_scale_bytes, output_bytes;
  uint64_t workspace_bytes, scratch_offset, scratch_bytes,
      staging_workspace_bytes;
  const char *reason;
} lowp_matmul_plan_t;

enum { LOWP_ACTIVATION_PREQUANTIZED = 1U };

/* Optional external FP16 GEMM for the existing staging variant. The library
 * has no BLAS dependency; callers keep the GEMM handle and any locks. */
typedef lowp_status_t (*lowp_f16_gemm_fn)(void *user,
                                          const uint16_t *activation,
                                          const uint16_t *weight, float *output,
                                          uint64_t m, uint64_t n, uint64_t k,
                                          void *stream);

typedef struct lowp_matmul_buffers {
  uint32_t struct_size, flags;
  /* BF16 by default. PREQUANTIZED uses the format's packed values/scales. */
  const void *activation;
  const void *activation_scales;
  const void *weight;
  const void *weight_scales;
  const float *weight_tensor_scale, *activation_tensor_scale;
  uint16_t *output;
  void *workspace;
  uint64_t workspace_bytes;
  void *staging_workspace;
  uint64_t staging_workspace_bytes;
  lowp_f16_gemm_fn f16_gemm;
  void *f16_gemm_user;
} lowp_matmul_buffers_t;

uint32_t lowp_version(void);
lowp_target_t lowp_target_from_name(const char *name);
const char *lowp_target_name(lowp_target_t target);
uint64_t lowp_supported_formats(lowp_target_t target);
lowp_status_t lowp_get_format_info(lowp_format_t format,
                                   lowp_format_info_t *info);
lowp_status_t lowp_matmul_plan(const lowp_matmul_request_t *request,
                               lowp_matmul_plan_t *plan);
/* Asynchronous, caller's current HIP device and stream. No allocations or sync.
 */
lowp_status_t lowp_matmul_launch(const lowp_matmul_plan_t *plan,
                                 const lowp_matmul_buffers_t *buffers,
                                 void *stream);
lowp_status_t lowp_quantize_activation(const lowp_matmul_plan_t *plan,
                                       const uint16_t *activation_bf16,
                                       void *values, void *scales,
                                       const float *activation_tensor_scale,
                                       void *stream);

#ifdef __cplusplus
}
#endif
#endif
