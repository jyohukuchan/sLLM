#include "residual_rmsnorm_api.hpp"

#include <cmath>
#include <cstring>

namespace sllm_residual_rmsnorm {
namespace {

sllm_status_t prefix_impl(const sllm_residual_rmsnorm_desc_t *const descriptor,
                          sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "residual RMSNorm descriptor is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, descriptor, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_residual_rmsnorm_desc_t)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "residual RMSNorm descriptor prefix has an unsupported struct size");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "residual RMSNorm public ABI version is unsupported");
  }
  return SLLM_STATUS_OK;
}

bool same_shape(const sllm_rmsnorm::TensorMetadata &left,
                const sllm_rmsnorm::TensorMetadata &right) noexcept {
  return left.rank == right.rank &&
         std::memcmp(left.shape, right.shape, sizeof(left.shape)) == 0 &&
         std::memcmp(left.strides, right.strides, sizeof(left.strides)) == 0;
}

bool overlaps(const sllm_tensor_binding_t &left,
              const sllm_rmsnorm::TensorMetadata &left_metadata,
              const sllm_tensor_binding_t &right,
              const sllm_rmsnorm::TensorMetadata &right_metadata) noexcept {
  return left.buffer == right.buffer &&
         sllm_rmsnorm::intervals_overlap(left_metadata, right_metadata);
}

} // namespace

sllm_status_t
validate_descriptor_prefix(const sllm_residual_rmsnorm_desc_t *const descriptor,
                           sllm_error_sink_t *const sink) noexcept {
  return prefix_impl(descriptor, sink);
}

sllm_status_t validate_and_copy_descriptor(
    const sllm_residual_rmsnorm_desc_t *const descriptor,
    DescriptorMetadata *const metadata,
    sllm_error_sink_t *const sink) noexcept {
  if (descriptor == nullptr || metadata == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "residual RMSNorm descriptor or metadata output is null");
  }
  const sllm_status_t prefix_status = prefix_impl(descriptor, sink);
  if (prefix_status != SLLM_STATUS_OK) {
    return prefix_status;
  }
  if (descriptor->op_version != SLLM_HIP_RESIDUAL_RMSNORM_VERSION ||
      descriptor->accumulation_dtype != SLLM_RMSNORM_ACCUMULATION_F32 ||
      descriptor->alias_policy != SLLM_RMSNORM_ALIAS_POLICY_REJECT_OVERLAP) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "residual RMSNorm descriptor has an unsupported operation contract");
  }
  if (descriptor->scale_mode != SLLM_RMSNORM_SCALE_MODE_OFFSET_ONE &&
      descriptor->scale_mode != SLLM_RMSNORM_SCALE_MODE_DIRECT) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_UNSUPPORTED_SCALE_MODE,
        "residual RMSNorm scale mode is unsupported");
  }
  for (const uint32_t value : descriptor->reserved) {
    if (value != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_RESERVED_NONZERO,
          "residual RMSNorm descriptor reserved fields must be zero");
    }
  }
  float epsilon = 0.0F;
  std::memcpy(&epsilon, &descriptor->epsilon_bits, sizeof(epsilon));
  if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_EPSILON,
        "residual RMSNorm epsilon must be finite and positive");
  }
  sllm_status_t status = sllm_rmsnorm::validate_tensor_binding(
      &descriptor->residual, &metadata->residual, sink);
  if (status != SLLM_STATUS_OK)
    return status;
  status = sllm_rmsnorm::validate_tensor_binding(&descriptor->addend,
                                                 &metadata->addend, sink);
  if (status != SLLM_STATUS_OK)
    return status;
  status = sllm_rmsnorm::validate_tensor_binding(&descriptor->raw_scale,
                                                 &metadata->raw_scale, sink);
  if (status != SLLM_STATUS_OK)
    return status;
  status = sllm_rmsnorm::validate_tensor_binding(
      &descriptor->residual_output, &metadata->residual_output, sink);
  if (status != SLLM_STATUS_OK)
    return status;
  status = sllm_rmsnorm::validate_tensor_binding(&descriptor->output,
                                                 &metadata->output, sink);
  if (status != SLLM_STATUS_OK)
    return status;

  if (!same_shape(metadata->residual, metadata->addend) ||
      !same_shape(metadata->residual, metadata->residual_output) ||
      !same_shape(metadata->residual, metadata->output)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        "residual RMSNorm tensors must have exactly equal layouts");
  }
  if (metadata->raw_scale.rank != 1U ||
      metadata->raw_scale.shape[0] !=
          metadata->residual.shape[metadata->residual.rank - 1U]) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_SHAPE_MISMATCH,
        "residual RMSNorm raw scale must match the final dimension");
  }
  const sllm_tensor_binding_t *bindings[5] = {
      &descriptor->residual, &descriptor->addend, &descriptor->raw_scale,
      &descriptor->residual_output, &descriptor->output};
  const sllm_rmsnorm::TensorMetadata *metas[5] = {
      &metadata->residual, &metadata->addend, &metadata->raw_scale,
      &metadata->residual_output, &metadata->output};
  for (uint32_t i = 0U; i != 5U; ++i) {
    for (uint32_t j = i + 1U; j != 5U; ++j) {
      if (overlaps(*bindings[i], *metas[i], *bindings[j], *metas[j])) {
        return sllm_public_runtime::write_error(
            sink, SLLM_STATUS_ALIAS_OVERLAP,
            "residual RMSNorm tensor intervals overlap");
      }
    }
  }
  metadata->epsilon_bits = descriptor->epsilon_bits;
  metadata->scale_mode = descriptor->scale_mode;
  return SLLM_STATUS_OK;
}

} // namespace sllm_residual_rmsnorm
