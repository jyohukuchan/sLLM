#include "token_selector_api.hpp"
#include "public_runtime_internal.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>

namespace sllm_token_selector {
namespace {

bool all_zero(const void *const bytes, const std::size_t size) noexcept {
  const auto *const values = static_cast<const unsigned char *>(bytes);
  for (std::size_t index = 0U; index != size; ++index) {
    if (values[index] != 0U) {
      return false;
    }
  }
  return true;
}

bool multiply_overflows(const uint64_t left, const uint64_t right,
                        uint64_t *const result) noexcept {
  if (left != 0U && right > std::numeric_limits<uint64_t>::max() / left) {
    return true;
  }
  *result = left * right;
  return false;
}

sllm_status_t validate_tensor(const sllm_tensor_binding_t &binding,
                              const uint32_t expected_dtype,
                              const uint64_t element_bytes,
                              const uint64_t expected_v, const bool output,
                              const bool workspace,
                              TensorMetadata *const copied,
                              sllm_error_sink_t *const sink) noexcept {
  if (binding.struct_size != sizeof(binding)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "token selector tensor binding has an unsupported struct size");
  }
  if (binding.abi_version != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "token selector tensor binding ABI is unsupported");
  }
  if (binding.reserved0 != 0U ||
      !all_zero(binding.reserved, sizeof(binding.reserved))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "token selector tensor binding reserved fields must be zero");
  }
  const uint32_t expected_rank = output || workspace ? 1U : 2U;
  if (binding.buffer == nullptr || binding.rank != expected_rank ||
      binding.dtype != expected_dtype ||
      binding.encoding != SLLM_TENSOR_ENCODING_UNQUANTIZED) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "token selector tensor binding has an unsupported shape or dtype");
  }
  if ((binding.byte_offset % element_bytes) != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_MISALIGNED_OFFSET,
        "token selector tensor offset is misaligned");
  }
  if (output) {
    if (binding.shape[0] != SLLM_HIP_TOKEN_SELECTOR_OUTPUT_BYTES ||
        binding.stride_elements[0] != 1U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_SHAPE_MISMATCH,
          "token selector output must be one contiguous 16-byte record");
    }
  } else if (workspace) {
    if (binding.shape[0] != expected_v || binding.stride_elements[0] != 1U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_SHAPE_MISMATCH,
          "token selector workspace must be one contiguous byte buffer");
    }
  } else {
    if (binding.shape[0] != 1U || binding.shape[1] != expected_v ||
        binding.stride_elements[1] != 1U ||
        binding.stride_elements[0] != expected_v) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_SHAPE_MISMATCH,
          "token selector inputs must be contiguous [1,V] tensors");
    }
  }
  for (uint32_t index = expected_rank; index != SLLM_HIP_TENSOR_MAX_RANK;
       ++index) {
    if (binding.shape[index] != 0U || binding.stride_elements[index] != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
          "token selector unused tensor metadata must be zero");
    }
  }
  uint64_t elements = binding.shape[0];
  if (!output && !workspace &&
      multiply_overflows(elements, binding.shape[1], &elements)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "token selector tensor element count overflowed u64");
  }
  uint64_t payload_bytes = 0U;
  if (multiply_overflows(elements, element_bytes, &payload_bytes) ||
      sllm_public_runtime::add_overflows(binding.byte_offset, payload_bytes)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "token selector tensor byte interval overflowed u64");
  }
  copied->byte_offset = binding.byte_offset;
  copied->payload_bytes = payload_bytes;
  copied->end_offset = binding.byte_offset + payload_bytes;
  return SLLM_STATUS_OK;
}

} // namespace

sllm_status_t
validate_descriptor_prefix(const sllm_token_selector_desc_t *const descriptor,
                           sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TOKEN_SELECTOR_DESCRIPTOR,
        "token selector descriptor is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, descriptor, sizeof(prefix));
  const uint32_t legacy_size =
      static_cast<uint32_t>(offsetof(sllm_token_selector_desc_t, top_k));
  if (prefix[0] != sizeof(sllm_token_selector_desc_t) &&
      prefix[0] != legacy_size) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "token selector descriptor prefix has an unsupported struct size");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "token selector descriptor ABI is unsupported");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_and_copy_descriptor(const sllm_token_selector_desc_t *const descriptor,
                             DescriptorMetadata *const metadata,
                             sllm_error_sink_t *const sink) noexcept {
  if (metadata == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TOKEN_SELECTOR_DESCRIPTOR,
        "token selector metadata output is null");
  }
  const sllm_status_t prefix_status =
      validate_descriptor_prefix(descriptor, sink);
  if (prefix_status != SLLM_STATUS_OK) {
    return prefix_status;
  }
  const bool legacy = descriptor->op_version == SLLM_HIP_TOKEN_SELECTOR_VERSION;
  const bool fixed =
      descriptor->op_version == SLLM_HIP_TOKEN_SELECTOR_VERSION_FIXED_TOPK_TOPP;
  if ((!legacy && !fixed) ||
      !all_zero(descriptor->reserved, sizeof(descriptor->reserved)) ||
      (fixed &&
       descriptor->struct_size != sizeof(sllm_token_selector_desc_t)) ||
      (fixed && descriptor->reserved_v2 != 0U)) {
    return sllm_public_runtime::write_error(
        sink,
        descriptor->op_version != SLLM_HIP_TOKEN_SELECTOR_VERSION
            ? SLLM_STATUS_INVALID_TOKEN_SELECTOR_DESCRIPTOR
            : SLLM_STATUS_RESERVED_NONZERO,
        "token selector descriptor version or reserved fields are invalid");
  }
  if (!std::isfinite(descriptor->temperature) ||
      descriptor->temperature <= 0.0F) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_TOKEN_SELECTOR_INVALID_TEMPERATURE,
        "token selector temperature must be finite and positive");
  }
  const uint64_t vocab_size = descriptor->vocab_size;
  if (vocab_size == 0U || vocab_size > SLLM_HIP_TOKEN_SELECTOR_MAX_V) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED,
        "token selector vocabulary exceeds the baseline launch contract");
  }
  uint32_t top_k = 0U;
  uint32_t flags = SLLM_HIP_TOKEN_SELECTOR_FLAG_ADDITIVE_PRESENT |
                   SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT;
  float top_p = 1.0F;
  if (fixed) {
    top_k = descriptor->top_k;
    flags = descriptor->flags;
    top_p = descriptor->top_p;
    if (top_k != 0U && top_k != 20U && top_k != 64U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INVALID_TOKEN_SELECTOR_DESCRIPTOR,
          "fixed token selector top-k must be 0, 20, or 64");
    }
    if (!std::isfinite(top_p) || top_p != 0.95F ||
        (flags & ~(SLLM_HIP_TOKEN_SELECTOR_FLAG_ADDITIVE_PRESENT |
                   SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT)) != 0U ||
        descriptor->temperature != 1.0F ||
        (top_k == 0U &&
         (flags & SLLM_HIP_TOKEN_SELECTOR_FLAG_ADDITIVE_PRESENT) != 0U)) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INVALID_TOKEN_SELECTOR_DESCRIPTOR,
          "fixed token selector requires temperature 1 and top-p 0.95");
    }
  }
  sllm_status_t status =
      validate_tensor(descriptor->logits, SLLM_TENSOR_DTYPE_BF16, UINT64_C(2),
                      vocab_size, false, false, &metadata->logits, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor(descriptor->additive_logits, SLLM_TENSOR_DTYPE_F32,
                           UINT64_C(4), vocab_size, false, false,
                           &metadata->additive_logits, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status =
      validate_tensor(descriptor->valid_mask, SLLM_TENSOR_DTYPE_U8, UINT64_C(1),
                      vocab_size, false, false, &metadata->valid_mask, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status =
      validate_tensor(descriptor->output, SLLM_TENSOR_DTYPE_U8, UINT64_C(1),
                      vocab_size, true, false, &metadata->output, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  if ((descriptor->output.byte_offset %
       alignof(sllm_token_selector_record_t)) != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_MISALIGNED_OFFSET,
        "token selector output record is not naturally aligned");
  }
  metadata->vocab_size = vocab_size;
  metadata->temperature = descriptor->temperature;
  metadata->op_version = descriptor->op_version;
  metadata->top_k = top_k;
  metadata->top_p = top_p;
  metadata->flags = flags;
  metadata->seed = descriptor->seed;
  metadata->counter = descriptor->counter;
  metadata->workspace = {};
  if (fixed) {
    const uint64_t bytes = workspace_bytes(vocab_size, top_k);
    status = validate_tensor(descriptor->workspace, SLLM_TENSOR_DTYPE_U8,
                             UINT64_C(1), bytes, false, true,
                             &metadata->workspace, sink);
    if (status != SLLM_STATUS_OK) {
      return status;
    }
  }
  return SLLM_STATUS_OK;
}

uint64_t workspace_bytes(const uint64_t vocab_size,
                         const uint32_t top_k) noexcept {
  if (vocab_size == 0U || (top_k != 0U && top_k != 20U && top_k != 64U)) {
    return 0U;
  }
  if (top_k == 0U) {
    return SLLM_HIP_TOKEN_SELECTOR_K0_WORKSPACE_BYTES;
  }
  const uint64_t blocks = (vocab_size + UINT64_C(1023)) / UINT64_C(1024);
  if (blocks > std::numeric_limits<uint64_t>::max() /
                   (static_cast<uint64_t>(top_k) * UINT64_C(16))) {
    return 0U;
  }
  return blocks * static_cast<uint64_t>(top_k) * UINT64_C(16);
}

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept {
  return left.byte_offset < right.end_offset &&
         right.byte_offset < left.end_offset;
}

} // namespace sllm_token_selector
