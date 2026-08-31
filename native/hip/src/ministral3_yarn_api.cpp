#include "ministral3_yarn_api.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace sllm_ministral3_yarn {
namespace {

sllm_status_t error(sllm_error_sink_t *const sink, const sllm_status_t status,
                    const char *const message) noexcept {
  return sllm_public_runtime::write_error(sink, status, message);
}

bool multiply_overflows(const uint64_t left, const uint64_t right,
                        uint64_t *const result) noexcept {
  if (left != 0U && right > std::numeric_limits<uint64_t>::max() / left) {
    return true;
  }
  *result = left * right;
  return false;
}

bool all_zero(const uint32_t *const values, const std::size_t count) noexcept {
  for (std::size_t index = 0U; index != count; ++index) {
    if (values[index] != 0U) {
      return false;
    }
  }
  return true;
}

bool all_zero(const uint64_t *const values, const std::size_t count) noexcept {
  for (std::size_t index = 0U; index != count; ++index) {
    if (values[index] != 0U) {
      return false;
    }
  }
  return true;
}

bool same_bits(const uint32_t bits, const float expected) noexcept {
  uint32_t expected_bits = 0U;
  std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
  return bits == expected_bits;
}

sllm_status_t validate_tensor(const sllm_tensor_binding_t *const binding,
                              const uint32_t expected_dtype,
                              const uint32_t expected_rank,
                              sllm_rotary::TensorMetadata *const copied,
                              sllm_error_sink_t *const sink,
                              const char *const name) noexcept {
  if (binding == nullptr || copied == nullptr) {
    return error(sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
                 "Ministral3 tensor binding is null");
  }
  if (binding->struct_size != sizeof(*binding) ||
      binding->abi_version != SLLM_HIP_ABI_VERSION) {
    return error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                 "Ministral3 tensor binding ABI prefix is invalid");
  }
  if (binding->reserved0 != 0U || !all_zero(binding->reserved, 2U)) {
    return error(sink, SLLM_STATUS_RESERVED_NONZERO,
                 "Ministral3 tensor binding reserved fields must be zero");
  }
  if (binding->buffer == nullptr) {
    return error(sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
                 "Ministral3 tensor binding buffer is null");
  }
  if (binding->rank != expected_rank) {
    return error(sink, SLLM_STATUS_SHAPE_MISMATCH, name);
  }
  if (binding->dtype != expected_dtype) {
    return error(sink, SLLM_STATUS_UNSUPPORTED_DTYPE,
                 "Ministral3 tensor has an unsupported dtype");
  }
  if (binding->encoding != SLLM_TENSOR_ENCODING_UNQUANTIZED) {
    return error(sink, SLLM_STATUS_UNSUPPORTED_ENCODING,
                 "Ministral3 tensors must be unquantized");
  }
  const uint64_t element_bytes =
      expected_dtype == SLLM_TENSOR_DTYPE_I32 ? UINT64_C(4) : UINT64_C(2);
  if (binding->byte_offset % element_bytes != 0U) {
    return error(sink, SLLM_STATUS_MISALIGNED_OFFSET,
                 "Ministral3 tensor offset is not element aligned");
  }

  uint64_t stride = 1U;
  uint64_t elements = 1U;
  for (uint32_t backwards = 0U; backwards != binding->rank; ++backwards) {
    const uint32_t index = binding->rank - 1U - backwards;
    const uint64_t extent = binding->shape[index];
    if (extent == 0U) {
      return error(sink, SLLM_STATUS_ZERO_EXTENT,
                   "Ministral3 tensor extents must be non-zero");
    }
    if (binding->stride_elements[index] != stride) {
      return error(sink, SLLM_STATUS_STRIDE_MISMATCH,
                   "Ministral3 tensors must be row-major contiguous");
    }
    if (multiply_overflows(stride, extent, &stride) ||
        multiply_overflows(elements, extent, &elements)) {
      return error(sink, SLLM_STATUS_METADATA_OVERFLOW,
                   "Ministral3 tensor metadata overflowed u64");
    }
  }
  for (uint32_t index = binding->rank; index != SLLM_HIP_TENSOR_MAX_RANK;
       ++index) {
    if (binding->shape[index] != 0U || binding->stride_elements[index] != 0U) {
      return error(sink, SLLM_STATUS_RESERVED_NONZERO,
                   "Ministral3 unused tensor dimensions must be zero");
    }
  }
  uint64_t payload_bytes = 0U;
  if (multiply_overflows(elements, element_bytes, &payload_bytes) ||
      binding->byte_offset >
          std::numeric_limits<uint64_t>::max() - payload_bytes) {
    return error(sink, SLLM_STATUS_METADATA_OVERFLOW,
                 "Ministral3 tensor byte interval overflowed u64");
  }
  copied->byte_offset = binding->byte_offset;
  copied->payload_bytes = payload_bytes;
  copied->end_offset = binding->byte_offset + payload_bytes;
  copied->rank = binding->rank;
  copied->element_bytes = static_cast<uint32_t>(element_bytes);
  for (uint32_t index = 0U; index != SLLM_HIP_TENSOR_MAX_RANK; ++index) {
    copied->shape[index] = binding->shape[index];
    copied->strides[index] = binding->stride_elements[index];
  }
  return SLLM_STATUS_OK;
}

} // namespace

sllm_status_t
validate_descriptor_prefix(const sllm_ministral3_yarn_desc_t *const descriptor,
                           sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr) {
    return error(sink, SLLM_STATUS_INVALID_MINISTRAL3_YARN_DESCRIPTOR,
                 "Ministral3 YaRN descriptor is null");
  }
  if (descriptor->struct_size != sizeof(*descriptor)) {
    return error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                 "Ministral3 YaRN descriptor has an unsupported struct size");
  }
  if (descriptor->abi_version != SLLM_HIP_ABI_VERSION) {
    return error(sink, SLLM_STATUS_INVALID_ABI_VERSION,
                 "Ministral3 YaRN ABI version is unsupported");
  }
  if (descriptor->op_version != SLLM_HIP_MINISTRAL3_YARN_VERSION) {
    return error(sink, SLLM_STATUS_INVALID_MINISTRAL3_YARN_DESCRIPTOR,
                 "Ministral3 YaRN operation version is unsupported");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t validate_and_copy_descriptor(
    const sllm_ministral3_yarn_desc_t *const descriptor,
    DescriptorMetadata *const metadata,
    sllm_error_sink_t *const sink) noexcept {
  if (metadata == nullptr) {
    return error(sink, SLLM_STATUS_INVALID_MINISTRAL3_YARN_DESCRIPTOR,
                 "Ministral3 YaRN metadata output is null");
  }
  const sllm_status_t prefix = validate_descriptor_prefix(descriptor, sink);
  if (prefix != SLLM_STATUS_OK) {
    return prefix;
  }
  if (descriptor->position_payload_mode !=
          SLLM_HIP_POSITION_PAYLOAD_MODE_CONTIGUOUS_V1 &&
      descriptor->position_payload_mode !=
          SLLM_HIP_POSITION_PAYLOAD_MODE_EXPLICIT_V1) {
    return error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                 "Ministral3 YaRN position payload mode is unsupported");
  }
  if (!all_zero(descriptor->reserved, 5U)) {
    return error(sink, SLLM_STATUS_RESERVED_NONZERO,
                 "Ministral3 YaRN reserved fields must be zero");
  }
  if (descriptor->q_heads != kQHeads || descriptor->kv_heads != kKvHeads ||
      descriptor->head_dim != kHeadDim ||
      descriptor->rotary_dim != kRotaryDim) {
    return error(sink, SLLM_STATUS_SHAPE_MISMATCH,
                 "Ministral3 YaRN uses Q32/KV8 head_dim128 rotary_dim128");
  }
  if (descriptor->original_context != kOriginalContext ||
      descriptor->max_position != kMaxPosition ||
      !same_bits(descriptor->theta_bits, kTheta) ||
      !same_bits(descriptor->factor_bits, kFactor) ||
      !same_bits(descriptor->beta_fast_bits, kBetaFast) ||
      !same_bits(descriptor->beta_slow_bits, kBetaSlow) ||
      !same_bits(descriptor->query_scale_beta_bits, kQueryScaleBeta)) {
    return error(sink, SLLM_STATUS_UNSUPPORTED_SCALE_MODE,
                 "Ministral3 YaRN parameters differ from the fixed model");
  }
  if (descriptor->start_position >= kMaxPosition) {
    return error(sink, SLLM_STATUS_POSITION_PAYLOAD_MISMATCH,
                 "Ministral3 YaRN start position is outside context");
  }

  DescriptorMetadata copied{};
  const uint32_t token_rank = 3U;
  sllm_status_t status =
      validate_tensor(&descriptor->query, SLLM_TENSOR_DTYPE_BF16, token_rank,
                      &copied.query, sink, "Ministral3 query rank is invalid");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  if (copied.query.shape[1] != kQHeads || copied.query.shape[2] != kHeadDim) {
    return error(sink, SLLM_STATUS_SHAPE_MISMATCH,
                 "Ministral3 query shape must be [tokens,32,128]");
  }
  copied.token_count = copied.query.shape[0];
  if (copied.token_count == 0U || copied.token_count > kMaxTokens) {
    return error(sink, SLLM_STATUS_UNSUPPORTED,
                 "Ministral3 token count exceeds the fixed launch contract");
  }
  status = validate_tensor(&descriptor->key, SLLM_TENSOR_DTYPE_BF16, token_rank,
                           &copied.key, sink, "Ministral3 key rank is invalid");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  if (copied.key.shape[0] != copied.token_count ||
      copied.key.shape[1] != kKvHeads || copied.key.shape[2] != kHeadDim) {
    return error(sink, SLLM_STATUS_SHAPE_MISMATCH,
                 "Ministral3 key shape must be [tokens,8,128]");
  }
  status = validate_tensor(&descriptor->positions, SLLM_TENSOR_DTYPE_I32, 1U,
                           &copied.positions, sink,
                           "Ministral3 position rank is invalid");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  if (copied.positions.shape[0] != copied.token_count) {
    return error(sink, SLLM_STATUS_SHAPE_MISMATCH,
                 "Ministral3 position count differs from token count");
  }
  status = validate_tensor(&descriptor->query_output, SLLM_TENSOR_DTYPE_BF16,
                           token_rank, &copied.query_output, sink,
                           "Ministral3 query output rank is invalid");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor(&descriptor->key_output, SLLM_TENSOR_DTYPE_BF16,
                           token_rank, &copied.key_output, sink,
                           "Ministral3 key output rank is invalid");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  if (copied.query_output.shape[0] != copied.token_count ||
      copied.query_output.shape[1] != kQHeads ||
      copied.query_output.shape[2] != kHeadDim ||
      copied.key_output.shape[0] != copied.token_count ||
      copied.key_output.shape[1] != kKvHeads ||
      copied.key_output.shape[2] != kHeadDim) {
    return error(sink, SLLM_STATUS_SHAPE_MISMATCH,
                 "Ministral3 output shape differs from the input shape");
  }
  if (descriptor->start_position >
      static_cast<uint64_t>(kMaxPosition) - copied.token_count) {
    return error(sink, SLLM_STATUS_POSITION_PAYLOAD_MISMATCH,
                 "Ministral3 YaRN token range exceeds context");
  }
  copied.start_position = descriptor->start_position;
  copied.position_payload_mode = descriptor->position_payload_mode;
  *metadata = copied;
  return SLLM_STATUS_OK;
}

bool intervals_overlap(const sllm_rotary::TensorMetadata &left,
                       const sllm_rotary::TensorMetadata &right) noexcept {
  return left.byte_offset < right.end_offset &&
         right.byte_offset < left.end_offset;
}

} // namespace sllm_ministral3_yarn
