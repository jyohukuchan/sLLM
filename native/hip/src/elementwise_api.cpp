#include "elementwise_api.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace sllm_elementwise {
namespace {

sllm_status_t exact_struct(const uint32_t struct_size,
                           const uint32_t abi_version,
                           const std::size_t expected_size,
                           sllm_error_sink_t *const sink,
                           const char *const size_message) noexcept {
  if (struct_size != expected_size) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                                            size_message);
  }
  if (abi_version != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "elementwise public ABI version is unsupported");
  }
  return SLLM_STATUS_OK;
}

bool multiply_overflows(const uint64_t left, const uint64_t right,
                        uint64_t *const result) noexcept {
  if (left != 0U && right > std::numeric_limits<uint64_t>::max() / left) {
    return true;
  }
  *result = left * right;
  return false;
}

bool all_zero(const void *const bytes, const std::size_t size) noexcept {
  const auto *const values = static_cast<const unsigned char *>(bytes);
  for (std::size_t index = 0U; index != size; ++index) {
    if (values[index] != 0U) {
      return false;
    }
  }
  return true;
}

sllm_status_t validate_tensor(const sllm_tensor_binding_t &binding,
                              const bool allow_prequantized,
                              const sllm_elementwise_operation_t operation,
                              TensorMetadata *const copied,
                              sllm_error_sink_t *const sink) noexcept {
  if (copied == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "elementwise tensor metadata output is null");
  }
  const sllm_status_t struct_status = exact_struct(
      binding.struct_size, binding.abi_version, sizeof(binding), sink,
      "elementwise tensor binding has an unsupported struct size");
  if (struct_status != SLLM_STATUS_OK) {
    return struct_status;
  }
  if (binding.reserved0 != 0U || binding.reserved[0] != 0U ||
      binding.reserved[1] != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "elementwise tensor binding reserved fields must be zero");
  }
  if (binding.buffer == nullptr || binding.rank == 0U ||
      binding.rank > SLLM_HIP_TENSOR_MAX_RANK) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
        "elementwise tensor binding requires a buffer and rank in 1..=8");
  }
  /* Phase 87 stage 7: only a producer output may be prequantized; inputs and
   * every other binding keep the historical BF16 contract. */
  const bool bf16_dtype = binding.dtype == SLLM_TENSOR_DTYPE_BF16;
  const bool encoded_dtype =
      allow_prequantized && (binding.dtype == SLLM_TENSOR_DTYPE_F8_E4M3_FN ||
                             binding.dtype == SLLM_TENSOR_DTYPE_U8);
  if (!bf16_dtype && !encoded_dtype) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_DTYPE,
        "elementwise tensors must use BF16 storage");
  }
  const bool legacy_encoding =
      bf16_dtype && binding.encoding == SLLM_TENSOR_ENCODING_UNQUANTIZED;
  const sllm_public_runtime::PrequantMode mode =
      sllm_public_runtime::prequant_mode_from_binding(binding.dtype,
                                                      binding.encoding);
  const bool prequantized =
      allow_prequantized && mode != sllm_public_runtime::PrequantMode::None;
  if (!legacy_encoding && !prequantized) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_ENCODING,
        "elementwise tensors must be unquantized or a prequantized FP8/NVFP4 "
        "producer output");
  }
  if (prequantized) {
    const bool producer_operation =
        operation == SLLM_ELEMENTWISE_OPERATION_SILU_MUL ||
        operation == SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL;
    if (!producer_operation) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_UNSUPPORTED,
          "only SiLU/Sigmoid multiply producers write prequantized output");
    }
    if (mode == sllm_public_runtime::PrequantMode::Nvfp4Block16 &&
        operation == SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_UNSUPPORTED,
          "sigmoid multiply has no NVFP4 prequant producer variant");
    }
  }
  if (legacy_encoding && (binding.byte_offset & UINT64_C(1)) != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_MISALIGNED_OFFSET,
        "elementwise BF16 tensor offset must be two-byte aligned");
  }

  uint64_t expected_stride = 1U;
  uint64_t elements = 1U;
  for (uint32_t backwards = 0U; backwards != binding.rank; ++backwards) {
    const uint32_t index = binding.rank - 1U - backwards;
    const uint64_t extent = binding.shape[index];
    if (extent == 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_ZERO_EXTENT,
          "elementwise tensor extents must be non-zero");
    }
    if (binding.stride_elements[index] != expected_stride) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_STRIDE_MISMATCH,
          "elementwise tensors must be row-major contiguous");
    }
    if (multiply_overflows(expected_stride, extent, &expected_stride) ||
        multiply_overflows(elements, extent, &elements)) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_METADATA_OVERFLOW,
          "elementwise tensor metadata overflowed u64");
    }
  }
  for (uint32_t index = binding.rank; index != SLLM_HIP_TENSOR_MAX_RANK;
       ++index) {
    if (binding.shape[index] != 0U || binding.stride_elements[index] != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INVALID_TENSOR_BINDING,
          "elementwise unused tensor metadata must be zero");
    }
  }
  uint64_t payload_bytes = 0U;
  if (!prequantized) {
    if (multiply_overflows(elements, UINT64_C(2), &payload_bytes) ||
        sllm_public_runtime::add_overflows(binding.byte_offset,
                                           payload_bytes)) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_METADATA_OVERFLOW,
          "elementwise tensor byte interval overflowed u64");
    }
  } else {
    /* Phase 87 stage 7 layout: rows are the leading extent (one for rank
     * one), width is the product of the remaining extents. Sigmoid's
     * [M, heads, width] binding therefore flattens to k = heads * width. */
    const uint64_t rows = binding.rank == 1U ? UINT64_C(1) : binding.shape[0];
    const uint64_t width = elements / rows;
    uint64_t value_bytes = 0U;
    if (!sllm_public_runtime::prequant_payload_layout(
            mode, rows, width, &value_bytes, &payload_bytes)) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_METADATA_OVERFLOW,
          "elementwise prequantized payload overflowed u64");
    }
    if (mode == sllm_public_runtime::PrequantMode::Fp8Outer) {
      if (sllm_public_runtime::add_overflows(binding.byte_offset,
                                             value_bytes)) {
        return sllm_public_runtime::write_error(
            sink, SLLM_STATUS_METADATA_OVERFLOW,
            "elementwise FP8 value/scale interval overflowed u64");
      }
      const uint64_t scale_offset = binding.byte_offset + value_bytes;
      if ((scale_offset & UINT64_C(3)) != 0U) {
        return sllm_public_runtime::write_error(
            sink, SLLM_STATUS_MISALIGNED_OFFSET,
            "elementwise FP8 output scales require a four-byte-aligned value "
            "payload end");
      }
    }
    if (sllm_public_runtime::add_overflows(binding.byte_offset,
                                           payload_bytes)) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_METADATA_OVERFLOW,
          "elementwise prequantized byte interval overflowed u64");
    }
  }

  copied->byte_offset = binding.byte_offset;
  copied->payload_bytes = payload_bytes;
  copied->end_offset = binding.byte_offset + payload_bytes;
  copied->rank = binding.rank;
  std::memcpy(copied->shape, binding.shape, sizeof(copied->shape));
  std::memcpy(copied->strides, binding.stride_elements,
              sizeof(copied->strides));
  return SLLM_STATUS_OK;
}

bool equal_layout(const TensorMetadata &left,
                  const TensorMetadata &right) noexcept {
  return left.rank == right.rank &&
         std::memcmp(left.shape, right.shape, sizeof(left.shape)) == 0 &&
         std::memcmp(left.strides, right.strides, sizeof(left.strides)) == 0;
}

} // namespace

sllm_status_t
validate_descriptor_prefix(const sllm_elementwise_desc_t *const descriptor,
                           sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ELEMENTWISE_DESCRIPTOR,
        "elementwise descriptor is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, descriptor, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_elementwise_desc_t)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "elementwise descriptor prefix has an unsupported struct size");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "elementwise public ABI version is unsupported");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_and_copy_descriptor(const sllm_elementwise_desc_t *const descriptor,
                             DescriptorMetadata *const metadata,
                             sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr || metadata == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ELEMENTWISE_DESCRIPTOR,
        "elementwise descriptor or metadata output is null");
  }
  const sllm_status_t prefix_status =
      validate_descriptor_prefix(descriptor, sink);
  if (prefix_status != SLLM_STATUS_OK) {
    return prefix_status;
  }
  if (descriptor->op_version != SLLM_HIP_ELEMENTWISE_VERSION ||
      (descriptor->operation != SLLM_ELEMENTWISE_OPERATION_COPY &&
       descriptor->operation != SLLM_ELEMENTWISE_OPERATION_ADD &&
       descriptor->operation != SLLM_ELEMENTWISE_OPERATION_SILU_MUL &&
       descriptor->operation != SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL &&
       descriptor->operation != SLLM_ELEMENTWISE_OPERATION_SCALAR_MUL &&
       descriptor->operation != SLLM_ELEMENTWISE_OPERATION_GELU_TANH_MUL &&
       descriptor->operation != SLLM_ELEMENTWISE_OPERATION_TANH_SOFTCAP &&
       descriptor->operation != SLLM_ELEMENTWISE_OPERATION_BROADCAST_ADD &&
       descriptor->operation != SLLM_ELEMENTWISE_OPERATION_BROADCAST_MUL)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ELEMENTWISE_DESCRIPTOR,
        "elementwise descriptor has an unsupported operation contract");
  }
  /* Phase 87 stage 7: reserved[0] is the NVFP4 input_global_scale_f32_bits
   * and is validated against the output encoding below; reserved[1..] stay
   * zero-only. */
  if (descriptor->reserved[1] != 0U || descriptor->reserved[2] != 0U ||
      descriptor->reserved[3] != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "elementwise descriptor reserved fields must be zero");
  }
  if (descriptor->operation == SLLM_ELEMENTWISE_OPERATION_COPY &&
      !all_zero(&descriptor->input1, sizeof(descriptor->input1))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ELEMENTWISE_DESCRIPTOR,
        "copy requires a zero-initialized second input binding");
  }

  sllm_status_t status =
      validate_tensor(descriptor->input0, false, descriptor->operation,
                      &metadata->input0, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  if (descriptor->operation != SLLM_ELEMENTWISE_OPERATION_COPY) {
    status = validate_tensor(descriptor->input1, false, descriptor->operation,
                             &metadata->input1, sink);
    if (status != SLLM_STATUS_OK) {
      return status;
    }
  } else {
    metadata->input1 = {};
  }
  status = validate_tensor(descriptor->output, true, descriptor->operation,
                           &metadata->output, sink);
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  const sllm_public_runtime::PrequantMode prequant_mode =
      sllm_public_runtime::prequant_mode_from_binding(
          descriptor->output.dtype, descriptor->output.encoding);
  metadata->output_prequant = static_cast<uint32_t>(prequant_mode);
  metadata->input_global_scale_f32_bits = descriptor->reserved[0];
  float input_global_scale = 0.0F;
  std::memcpy(&input_global_scale, &descriptor->reserved[0],
              sizeof(input_global_scale));
  if (prequant_mode == sllm_public_runtime::PrequantMode::Nvfp4Block16) {
    if (!std::isfinite(input_global_scale) || input_global_scale <= 0.0F) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INVALID_ARGUMENT,
          "elementwise NVFP4 output requires a finite positive input global "
          "scale in reserved[0]");
    }
  } else if (descriptor->reserved[0] != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "elementwise descriptor reserved fields must be zero");
  }
  const bool scalar_input =
      descriptor->operation == SLLM_ELEMENTWISE_OPERATION_SCALAR_MUL ||
      descriptor->operation == SLLM_ELEMENTWISE_OPERATION_TANH_SOFTCAP;
  const bool broadcast_input =
      descriptor->operation == SLLM_ELEMENTWISE_OPERATION_BROADCAST_ADD ||
      descriptor->operation == SLLM_ELEMENTWISE_OPERATION_BROADCAST_MUL;
  const bool input1_layout_valid =
      descriptor->operation == SLLM_ELEMENTWISE_OPERATION_COPY ||
      (broadcast_input
           ? metadata->input0.rank == 2U && metadata->input1.rank == 1U &&
                 metadata->input1.shape[0] == metadata->input0.shape[1]
           : (scalar_input ? metadata->input1.rank == 1U &&
                                 metadata->input1.shape[0] == 1U
                           : equal_layout(metadata->input0, metadata->input1)));
  if (!equal_layout(metadata->input0, metadata->output) ||
      !input1_layout_valid) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        broadcast_input
            ? (descriptor->operation == SLLM_ELEMENTWISE_OPERATION_BROADCAST_ADD
                   ? "broadcast add requires input/output [M,H] and vector [H]"
                   : "broadcast multiply requires input/output [M,H] and "
                     "vector [H]")
            : (scalar_input
                   ? "scalar elementwise operation requires equal input/output "
                     "layouts and one BF16 scalar"
                   : "elementwise operands must have exactly equal layouts"));
  }
  if (descriptor->operation == SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL &&
      (metadata->input0.rank != 3U ||
       (metadata->input0.shape[1] != 8U && metadata->input0.shape[1] != 16U &&
        metadata->input0.shape[1] != 24U) ||
       metadata->input0.shape[2] != 256U)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        "sigmoid multiply requires a reviewed contiguous BF16 [M,H,256] "
        "layout");
  }
  const auto overlaps = [](const sllm_tensor_binding_t &left_binding,
                           const TensorMetadata &left,
                           const sllm_tensor_binding_t &right_binding,
                           const TensorMetadata &right) {
    return left_binding.buffer == right_binding.buffer &&
           intervals_overlap(left, right);
  };
  if (overlaps(descriptor->input0, metadata->input0, descriptor->output,
               metadata->output) ||
      (descriptor->operation != SLLM_ELEMENTWISE_OPERATION_COPY &&
       (overlaps(descriptor->input0, metadata->input0, descriptor->input1,
                 metadata->input1) ||
        overlaps(descriptor->input1, metadata->input1, descriptor->output,
                 metadata->output)))) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_ALIAS_OVERLAP,
        "elementwise tensor intervals overlap within one binding identity");
  }
  metadata->element_count = metadata->input0.payload_bytes / UINT64_C(2);
  metadata->operation = descriptor->operation;
  return SLLM_STATUS_OK;
}

bool intervals_overlap(const TensorMetadata &left,
                       const TensorMetadata &right) noexcept {
  return left.byte_offset < right.end_offset &&
         right.byte_offset < left.end_offset;
}

} // namespace sllm_elementwise
