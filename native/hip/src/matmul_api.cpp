#include "matmul_api.hpp"

#include <cstring>
#include <limits>

namespace sllm_matmul {
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

sllm_status_t
validate_tensor(const sllm_tensor_binding_t &binding,
                TensorMetadata *const copied, const uint32_t expected_dtype,
                const uint32_t expected_encoding, const uint64_t element_bytes,
                const bool append_outer_scales, const bool packed_nvfp4,
                const bool append_input_tensor_scale,
                sllm_error_sink_t *const sink) noexcept {
  if (binding.struct_size != sizeof(binding)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "matmul tensor binding has an unsupported struct size");
  }
  if (binding.abi_version != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "matmul tensor binding ABI version is unsupported");
  }
  if (binding.reserved0 != 0U ||
      !all_zero(binding.reserved, sizeof(binding.reserved))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "matmul tensor binding reserved fields must be zero");
  }
  if (binding.buffer == nullptr || binding.rank != 2U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "matmul tensor binding requires a buffer and rank two");
  }
  if (binding.dtype != expected_dtype) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_DTYPE,
        "matmul tensor dtype differs from the selected contract");
  }
  if (binding.encoding != expected_encoding) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_ENCODING,
        "matmul tensor encoding differs from the selected contract");
  }
  if ((binding.byte_offset & (element_bytes - 1U)) != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_MISALIGNED_OFFSET,
        "matmul tensor offset does not meet its storage alignment");
  }
  if (binding.shape[0] == 0U || binding.shape[1] == 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_ZERO_EXTENT,
        "matmul tensor extents must be non-zero");
  }
  if (binding.stride_elements[1] != 1U ||
      binding.stride_elements[0] != binding.shape[1]) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_STRIDE_MISMATCH,
        "matmul tensors must be row-major contiguous");
  }
  for (uint32_t index = 2U; index != SLLM_HIP_TENSOR_MAX_RANK; ++index) {
    if (binding.shape[index] != 0U || binding.stride_elements[index] != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
          "matmul unused tensor metadata must be zero");
    }
  }
  uint64_t elements = 0U;
  uint64_t payload_bytes = 0U;
  if (multiply_overflows(binding.shape[0], binding.shape[1], &elements)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "matmul tensor byte interval overflowed u64");
  }
  if (packed_nvfp4) {
    payload_bytes = elements / UINT64_C(2) +
                    (elements % UINT64_C(2) != 0U ? UINT64_C(1) : UINT64_C(0));
  } else if (multiply_overflows(elements, element_bytes, &payload_bytes)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "matmul tensor byte interval overflowed u64");
  }
  if (sllm_public_runtime::add_overflows(binding.byte_offset, payload_bytes)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "matmul tensor byte interval overflowed u64");
  }
  if (append_outer_scales) {
    uint64_t scale_bytes = 0U;
    if (multiply_overflows(binding.shape[0], UINT64_C(4), &scale_bytes) ||
        sllm_public_runtime::add_overflows(payload_bytes, scale_bytes) ||
        sllm_public_runtime::add_overflows(binding.byte_offset,
                                           payload_bytes + scale_bytes)) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_METADATA_OVERFLOW,
          "matmul FP8 value/scale interval overflowed u64");
    }
    payload_bytes += scale_bytes;
  }
  if (packed_nvfp4) {
    const uint64_t blocks_per_row =
        binding.shape[1] / UINT64_C(16) +
        (binding.shape[1] % UINT64_C(16) != 0U ? UINT64_C(1) : UINT64_C(0));
    uint64_t block_scale_bytes = 0U;
    if (multiply_overflows(binding.shape[0], blocks_per_row,
                           &block_scale_bytes) ||
        sllm_public_runtime::add_overflows(payload_bytes, block_scale_bytes) ||
        sllm_public_runtime::add_overflows(payload_bytes + block_scale_bytes,
                                           UINT64_C(3))) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_METADATA_OVERFLOW,
          "matmul NVFP4 value/scale interval overflowed u64");
    }
    payload_bytes =
        ((payload_bytes + block_scale_bytes + UINT64_C(3)) & ~UINT64_C(3)) +
        (append_input_tensor_scale ? UINT64_C(8) : UINT64_C(4));
    if (sllm_public_runtime::add_overflows(binding.byte_offset,
                                           payload_bytes)) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_METADATA_OVERFLOW,
          "matmul NVFP4 tensor scale interval overflowed u64");
    }
  }
  copied->byte_offset = binding.byte_offset;
  copied->payload_bytes = payload_bytes;
  copied->end_offset = binding.byte_offset + payload_bytes;
  copied->shape[0] = binding.shape[0];
  copied->shape[1] = binding.shape[1];
  return SLLM_STATUS_OK;
}

} // namespace

sllm_status_t
validate_descriptor_prefix(const sllm_matmul_desc_t *const descriptor,
                           sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_MATMUL_DESCRIPTOR,
        "matmul descriptor is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, descriptor, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_matmul_desc_t)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "matmul descriptor prefix has an unsupported struct size");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "matmul public ABI version is unsupported");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_and_copy_descriptor(const sllm_matmul_desc_t *const descriptor,
                             DescriptorMetadata *const metadata,
                             sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr || metadata == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_MATMUL_DESCRIPTOR,
        "matmul descriptor or metadata output is null");
  }
  const sllm_status_t prefix_status =
      validate_descriptor_prefix(descriptor, sink);
  if (prefix_status != SLLM_STATUS_OK) {
    return prefix_status;
  }
  if (descriptor->op_version != SLLM_HIP_MATMUL_VERSION &&
      descriptor->op_version != SLLM_HIP_MATMUL_FP8_VERSION &&
      descriptor->op_version != SLLM_HIP_MATMUL_NVFP4_VERSION &&
      descriptor->op_version != SLLM_HIP_MATMUL_NVFP4_W4A4_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_MATMUL_DESCRIPTOR,
        "matmul descriptor version is unsupported");
  }
  if (!all_zero(descriptor->reserved, sizeof(descriptor->reserved))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "matmul descriptor reserved fields must be zero");
  }
  const bool fp8_outer = descriptor->op_version == SLLM_HIP_MATMUL_FP8_VERSION;
  const bool nvfp4_w4a4 =
      descriptor->op_version == SLLM_HIP_MATMUL_NVFP4_W4A4_VERSION;
  const bool nvfp4 = descriptor->op_version == SLLM_HIP_MATMUL_NVFP4_VERSION ||
                     nvfp4_w4a4;
  sllm_status_t status = validate_tensor(
      descriptor->activation, &metadata->activation, SLLM_TENSOR_DTYPE_BF16,
      SLLM_TENSOR_ENCODING_UNQUANTIZED, UINT64_C(2), false, false, false,
      sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  const uint32_t fp8_dtype = descriptor->weight.dtype;
  if (fp8_outer && fp8_dtype != SLLM_TENSOR_DTYPE_F8_E4M3_FN &&
      fp8_dtype != SLLM_TENSOR_DTYPE_F8_E4M3_FNUZ) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_DTYPE,
        "FP8 matmul weight must use OCP E4M3FN or E4M3FNUZ");
  }
  status = validate_tensor(
      descriptor->weight, &metadata->weight,
      nvfp4 ? SLLM_TENSOR_DTYPE_U8
            : (fp8_outer ? fp8_dtype : SLLM_TENSOR_DTYPE_BF16),
      nvfp4 ? (nvfp4_w4a4
                   ? SLLM_TENSOR_ENCODING_NVFP4_W4A4_BLOCK16_E4M3FN_F32
                   : SLLM_TENSOR_ENCODING_NVFP4_BLOCK16_E4M3FN_F32)
            : (fp8_outer ? SLLM_TENSOR_ENCODING_FP8_OUTER_F32
                         : SLLM_TENSOR_ENCODING_UNQUANTIZED),
      fp8_outer || nvfp4 ? UINT64_C(1) : UINT64_C(2), fp8_outer, nvfp4,
      nvfp4_w4a4, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  status = validate_tensor(
      descriptor->output, &metadata->output, SLLM_TENSOR_DTYPE_BF16,
      SLLM_TENSOR_ENCODING_UNQUANTIZED, UINT64_C(2), false, false, false,
      sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  metadata->m = metadata->activation.shape[0];
  metadata->k = metadata->activation.shape[1];
  metadata->n = metadata->weight.shape[0];
  if (metadata->weight.shape[1] != metadata->k ||
      metadata->output.shape[0] != metadata->m ||
      metadata->output.shape[1] != metadata->n) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        "matmul requires activation [M,K], weight [N,K], and output [M,N]");
  }
  if (multiply_overflows(metadata->m, metadata->n,
                         &metadata->output_elements)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_METADATA_OVERFLOW,
        "matmul output element count overflowed u64");
  }
  metadata->fp8_outer = fp8_outer;
  metadata->nvfp4 = nvfp4;
  metadata->nvfp4_w4a4 = nvfp4_w4a4;
  metadata->fp8_dtype = fp8_outer ? fp8_dtype : 0U;
  metadata->weight_value_bytes =
      nvfp4 ? metadata->n * metadata->k / UINT64_C(2) +
                  (metadata->n * metadata->k % UINT64_C(2) != 0U ? UINT64_C(1)
                                                                 : UINT64_C(0))
            : metadata->n * metadata->k;
  if (fp8_outer && (metadata->weight_value_bytes & UINT64_C(3)) != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_MISALIGNED_OFFSET,
        "matmul FP8 outer scales require a four-byte-aligned value payload");
  }
  metadata->weight_scale_offset =
      metadata->weight.byte_offset + metadata->weight_value_bytes;
  if (nvfp4) {
    const uint64_t block_scale_bytes =
        metadata->n *
        (metadata->k / UINT64_C(16) +
         (metadata->k % UINT64_C(16) != 0U ? UINT64_C(1) : UINT64_C(0)));
    metadata->weight_tensor_scale_offset =
        (metadata->weight_scale_offset + block_scale_bytes + UINT64_C(3)) &
        ~UINT64_C(3);
    metadata->input_tensor_scale_offset =
        nvfp4_w4a4 ? metadata->weight_tensor_scale_offset + UINT64_C(4) : 0U;
  } else {
    metadata->weight_tensor_scale_offset = 0U;
    metadata->input_tensor_scale_offset = 0U;
  }
  return SLLM_STATUS_OK;
}

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept {
  return left.byte_offset < right.end_offset &&
         right.byte_offset < left.end_offset;
}

} // namespace sllm_matmul
