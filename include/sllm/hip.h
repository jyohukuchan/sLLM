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

#define SLLM_HIP_RMSNORM_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_RMSNORM_KERNEL_ID_BASELINE_WAVE32_V1 UINT32_C(1)
#define SLLM_HIP_RMSNORM_KERNEL_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_RMSNORM_DEVICE_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_RMSNORM_MAX_N UINT64_C(4096)
#define SLLM_HIP_RMSNORM_WORKGROUP_SIZE UINT32_C(256)
#define SLLM_HIP_RMSNORM_MAX_ROWS UINT64_C(4294967295)

#define SLLM_HIP_ELEMENTWISE_VERSION UINT32_C(1)
#define SLLM_HIP_ELEMENTWISE_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_ELEMENTWISE_KERNEL_ID_COPY_V1 UINT32_C(1)
#define SLLM_HIP_ELEMENTWISE_KERNEL_ID_ADD_V1 UINT32_C(2)
#define SLLM_HIP_ELEMENTWISE_KERNEL_ID_SILU_MUL_V1 UINT32_C(3)
#define SLLM_HIP_ELEMENTWISE_KERNEL_ID_SIGMOID_MUL_V1 UINT32_C(4)
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
#define SLLM_HIP_MATMUL_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_MATMUL_KERNEL_ID_BASELINE_BF16_FP32_V1 UINT32_C(1)
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

#define SLLM_HIP_ATTENTION_PREPROCESS_VERSION UINT32_C(1)
#define SLLM_HIP_ATTENTION_PREPROCESS_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_ATTENTION_PREPROCESS_KERNEL_ID_BASELINE_BF16_V1 UINT32_C(1)
#define SLLM_HIP_ATTENTION_PREPROCESS_KERNEL_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_ATTENTION_PREPROCESS_DEVICE_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_ATTENTION_PREPROCESS_WORKGROUP_SIZE UINT32_C(1)
#define SLLM_HIP_ATTENTION_PREPROCESS_Q_HEADS UINT32_C(16)
#define SLLM_HIP_ATTENTION_PREPROCESS_K_HEADS UINT32_C(4)
#define SLLM_HIP_ATTENTION_PREPROCESS_Q_HEAD_DIM UINT32_C(256)
#define SLLM_HIP_ATTENTION_PREPROCESS_K_HEAD_DIM UINT32_C(256)
#define SLLM_HIP_ATTENTION_PREPROCESS_QGATE_HEAD_DIM UINT32_C(512)
#define SLLM_HIP_ATTENTION_PREPROCESS_ROTARY_DIM UINT32_C(64)
#define SLLM_HIP_ATTENTION_PREPROCESS_MAX_POSITION UINT32_C(262144)
#define SLLM_HIP_ATTENTION_PREPROCESS_MAX_M UINT64_C(262144)

#define SLLM_HIP_KV_STATE_VERSION UINT32_C(1)
#define SLLM_HIP_KV_VIEW_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_KV_APPEND_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_KV_HEAD_COUNT UINT32_C(4)
#define SLLM_HIP_KV_HEAD_DIM UINT32_C(256)
#define SLLM_HIP_KV_MAX_CAPACITY UINT64_C(262144)
#define SLLM_HIP_KV_MAX_M UINT64_C(262144)
#define SLLM_HIP_KV_KERNEL_ID_BF16_TO_F16_TRANSPOSE_V1 UINT32_C(1)
#define SLLM_HIP_KV_WORKGROUP_SIZE UINT32_C(256)
#define SLLM_HIP_KV_KERNEL_SYMBOL_MAX UINT32_C(64)
#define SLLM_HIP_KV_DEVICE_SYMBOL_MAX UINT32_C(64)

#define SLLM_HIP_CAUSAL_ATTENTION_VERSION UINT32_C(1)
#define SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION UINT32_C(1)
#define SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_STABLE_SOFTMAX_V1 UINT32_C(1)
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
#define SLLM_HIP_LINEAR_ATTENTION_MAX_CAPACITY UINT64_C(262144)
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
#define SLLM_HIP_TENSOR_MAX_RANK UINT32_C(8)

typedef uint32_t sllm_tensor_dtype_t;
#define SLLM_TENSOR_DTYPE_BF16 UINT32_C(0)
#define SLLM_TENSOR_DTYPE_F16 UINT32_C(1)
#define SLLM_TENSOR_DTYPE_F32 UINT32_C(2)
#define SLLM_TENSOR_DTYPE_I32 UINT32_C(8)

typedef uint32_t sllm_tensor_encoding_t;
#define SLLM_TENSOR_ENCODING_UNQUANTIZED UINT32_C(0)

typedef uint32_t sllm_rmsnorm_accumulation_dtype_t;
#define SLLM_RMSNORM_ACCUMULATION_F32 UINT32_C(2)

typedef uint32_t sllm_rmsnorm_scale_mode_t;
#define SLLM_RMSNORM_SCALE_MODE_OFFSET_ONE UINT32_C(1)

typedef uint32_t sllm_rmsnorm_alias_policy_t;
#define SLLM_RMSNORM_ALIAS_POLICY_REJECT_OVERLAP UINT32_C(1)

typedef uint32_t sllm_elementwise_operation_t;
#define SLLM_ELEMENTWISE_OPERATION_COPY UINT32_C(1)
#define SLLM_ELEMENTWISE_OPERATION_ADD UINT32_C(2)
#define SLLM_ELEMENTWISE_OPERATION_SILU_MUL UINT32_C(3)
#define SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL UINT32_C(4)

#define SLLM_COMPLETION_STATE_PENDING UINT32_C(0)
#define SLLM_COMPLETION_STATE_SUCCESS UINT32_C(1)
#define SLLM_COMPLETION_STATE_FAILURE UINT32_C(2)

/* These handles have no public layout and must not be dereferenced by callers.
 */
typedef struct sllm_context_t sllm_context_t;
typedef struct sllm_queue_t sllm_queue_t;
typedef struct sllm_buffer_t sllm_buffer_t;
typedef struct sllm_event_t sllm_event_t;
typedef struct sllm_completion_t sllm_completion_t;
typedef struct sllm_rmsnorm_plan_t sllm_rmsnorm_plan_t;
typedef struct sllm_elementwise_plan_t sllm_elementwise_plan_t;
typedef struct sllm_embedding_plan_t sllm_embedding_plan_t;
typedef struct sllm_matmul_plan_t sllm_matmul_plan_t;
typedef struct sllm_argmax_plan_t sllm_argmax_plan_t;
typedef struct sllm_attention_preprocess_plan_t
    sllm_attention_preprocess_plan_t;
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
  uint32_t reserved[4];
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

/* C3a1 text-only attention preprocessing. The packed Q/gate input is exactly
 * [M, 16, 512], with each head's final axis [Q 256, gate 256]. K is [M, 4,
 * 256]. All eight tensor bindings are required to be contiguous and their
 * byte ranges must not overlap; the runtime retains every unique backing
 * buffer through the asynchronous completion. */
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

/* A request-local full-attention KV state owns separate K and V device
 * allocations.  The allocations are logically FP16 [4, capacity, 256]; no
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
  uint32_t reserved[4];
} sllm_kv_state_create_info_t;

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
  uint32_t reserved1;
  uint64_t capacity_tokens;
  uint64_t observed_length;
  uint64_t generation;
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

/* C3b causal full attention. Q and output are contiguous unquantized BF16
 * [M, 16, 256]. The referenced state is one committed FP16 [4, capacity,
 * 256] snapshot; no repeated K/V payload is part of this descriptor. */
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
  uint32_t scale_denominator;
  uint32_t fallback_allowed;
  uint32_t fallback_used;
  char kernel_symbol[SLLM_HIP_CAUSAL_ATTENTION_KERNEL_SYMBOL_MAX];
  char device_symbol[SLLM_HIP_CAUSAL_ATTENTION_DEVICE_SYMBOL_MAX];
  char gcn_arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME];
  uint32_t reserved[8];
} sllm_causal_attention_dispatch_info_t;

/* A request-local linear-attention state owns two BF16 [3,8192]
 * convolution-history slots and two F32 [32,128,128] recurrent slots. The
 * inactive pair is published only after successful completion. */
typedef struct sllm_linear_attention_state_create_info_t {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t session_id;
  uint32_t layer_id;
  uint32_t flags;
  uint64_t capacity_tokens;
  uint32_t reserved[4];
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

SLLM_HIP_API sllm_status_t sllm_argmax_prepare(
    const sllm_context_t *context, const sllm_argmax_desc_t *descriptor,
    sllm_argmax_plan_t **plan, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_argmax_plan_release(
    sllm_argmax_plan_t **plan, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_argmax_execute(
    const sllm_argmax_plan_t *plan, const sllm_queue_t *queue,
    sllm_completion_t **completion,
    sllm_argmax_dispatch_info_t *dispatch_info,
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

SLLM_HIP_API sllm_status_t sllm_kv_state_create(
    const sllm_context_t *context, const sllm_kv_state_create_info_t *info,
    sllm_kv_state_t **state, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t sllm_kv_state_release(
    sllm_kv_state_t **state, sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

SLLM_HIP_API sllm_status_t
sllm_kv_state_query(const sllm_kv_state_t *state, sllm_kv_view_info_t *info,
                    sllm_error_sink_t *error_sink) SLLM_HIP_NOEXCEPT;

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
