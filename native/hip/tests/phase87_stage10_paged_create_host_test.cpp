#include "../src/kv_state_api.hpp"

#include <cstdint>
#include <iostream>

namespace {

sllm_kv_state_paged_create_info_t base_info() {
  sllm_kv_state_paged_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.create_info_version = SLLM_HIP_KV_PAGED_CREATE_INFO_VERSION;
  info.session_id = 1U;
  info.capacity_tokens = 129U;
  info.head_count = 4U;
  info.head_dim = 256U;
  info.memory_kind = SLLM_HIP_KV_MEMORY_KIND_PAGED;
  info.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
  info.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
  info.encoding = SLLM_HIP_KV_ENCODING_MXFP8_E4_V1;
  info.scale_dtype = SLLM_TENSOR_DTYPE_U8;
  info.quantization_block_size = 32U;
  info.token_block_size = SLLM_HIP_KV_PAGED_TOKEN_BLOCK_SIZE;
  info.physical_layout_version = SLLM_HIP_KV_PAGED_LAYOUT_VERSION;
  info.logical_table_capacity = 2U;
  info.max_physical_blocks = 4U;
  return info;
}

bool checks() {
  auto info = base_info();
  const auto valid = [&](const sllm_kv_state_paged_create_info_t &candidate) {
    return sllm_kv_state::validate_state_create_info_paged(
               &candidate, nullptr) == SLLM_STATUS_OK;
  };
  if (!valid(info))
    return false;
  sllm_kv_state_create_info_v2_t legacy{};
  legacy.struct_size = sizeof(legacy);
  legacy.abi_version = SLLM_HIP_ABI_VERSION;
  legacy.create_info_version = SLLM_HIP_KV_STATE_CREATE_INFO_V2_VERSION;
  legacy.session_id = 1U;
  legacy.capacity_tokens = 129U;
  legacy.head_count = 4U;
  legacy.head_dim = 256U;
  legacy.memory_kind = SLLM_HIP_KV_MEMORY_KIND_PAGED;
  legacy.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
  legacy.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
  legacy.encoding = SLLM_HIP_KV_ENCODING_MXFP8_E4_V1;
  legacy.block_size = 32U;
  legacy.scale_dtype = SLLM_TENSOR_DTYPE_U8;
  if (sllm_kv_state::validate_state_create_info_v2(&legacy, nullptr) ==
      SLLM_STATUS_OK)
    return false;
  info.capacity_tokens = 128U;
  info.logical_table_capacity = 1U;
  if (!valid(info))
    return false;
  info.capacity_tokens = 129U;
  if (valid(info))
    return false;
  info = base_info();
  info.token_block_size = 16U;
  if (valid(info))
    return false;
  info = base_info();
  info.quantization_block_size = 16U;
  if (valid(info))
    return false;
  info = base_info();
  info.max_physical_blocks = UINT32_MAX;
  if (valid(info))
    return false;
  info = base_info();
  info.encoding = SLLM_HIP_KV_ENCODING_FP16_V1;
  info.dtype = SLLM_TENSOR_DTYPE_F16;
  info.scale_dtype = 0U;
  info.quantization_block_size = 0U;
  if (!valid(info))
    return false;
  info = base_info();
  info.encoding = SLLM_HIP_KV_ENCODING_NVFP4_V1;
  info.dtype = SLLM_TENSOR_DTYPE_U8;
  info.scale_dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
  info.quantization_block_size = 16U;
  if (!valid(info))
    return false;
  info = base_info();
  info.encoding = SLLM_HIP_KV_ENCODING_FP8_E4_BLOCK16_V2;
  info.quantization_block_size = 16U;
  if (valid(info))
    return false;
  info = base_info();
  info.encoding = SLLM_HIP_KV_ENCODING_FP8_STATIC_V1;
  info.scale_dtype = SLLM_TENSOR_DTYPE_F32;
  info.quantization_block_size = 0U;
  info.static_key_scale_bits = UINT32_C(0x3f800000);
  info.static_value_scale_bits = UINT32_C(0x3f800000);
  info.capacity_tokens = 1024U;
  info.logical_table_capacity = 8U;
  info.max_physical_blocks = 8U;
  info.sliding_window_tokens = 1024U;
  if (!valid(info))
    return false;
  info.static_key_scale_bits = 0U;
  if (valid(info))
    return false;
  info = base_info();
  info.reserved[0] = 1U;
  if (sllm_kv_state::validate_state_create_info_paged(&info, nullptr) !=
      SLLM_STATUS_RESERVED_NONZERO)
    return false;
  return true;
}

} // namespace

int main() {
  if (!checks()) {
    std::cerr << "phase87_stage10_paged_create_host_test: FAIL\n";
    return 1;
  }
  std::cout << "phase87_stage10_paged_create_host_test: PASS\n";
  return 0;
}
