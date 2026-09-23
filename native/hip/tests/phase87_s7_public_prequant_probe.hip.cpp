// Phase 87 bounded public-runtime probe.  This exercises the actual public
// RMSNorm -> matmul ABI handoff for a Qwen3.8 MLP projection shape, including
// a non-zero encoded activation byte offset.  The control keeps BF16 output
// and lets matmul perform its normal NVFP4 quantization; the candidate writes
// the contiguous (packed values, block scales) payload in the producer and
// passes that payload to a prequantized W4A4 matmul plan.

#include "sllm/hip.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr uint64_t kM = 17U;
constexpr uint64_t kK = 5120U;
constexpr uint64_t kN = 17408U;
constexpr uint64_t kEncodedOffset = 64U;
constexpr uint32_t kInputScaleBits = 0x44400000U; // 768.0f
constexpr float kInputScale = 768.0F;
constexpr uint32_t kTimeoutMs = 60'000U;

struct Error final {
  char message[512]{};
  sllm_error_sink_t sink{sizeof(sllm_error_sink_t),
                         SLLM_HIP_ABI_VERSION,
                         message,
                         sizeof(message),
                         0U,
                         {0U, 0U}};
};

bool ok(const sllm_status_t status, const sllm_status_t expected,
        const char *const where, const Error &error) {
  if (status == expected) {
    return true;
  }
  std::cerr << "FAIL: " << where << " status=" << status
            << " expected=" << expected << " message=" << error.message << '\n';
  return false;
}

bool wait_release(sllm_completion_t **const completion,
                  const char *const where) {
  if (completion == nullptr || *completion == nullptr) {
    std::cerr << "FAIL: " << where << " returned no completion\n";
    return false;
  }
  Error error;
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  bool valid =
      ok(sllm_completion_wait(*completion, kTimeoutMs, &result, &error.sink),
         SLLM_STATUS_OK, where, error) &&
      result.state == SLLM_COMPLETION_STATE_SUCCESS;
  valid = ok(sllm_completion_release(completion, &error.sink), SLLM_STATUS_OK,
             "sllm_completion_release", error) &&
          valid && *completion == nullptr;
  return valid;
}

bool copy_h2d(const sllm_queue_t *const queue,
              const sllm_buffer_t *const buffer, const void *const source,
              const uint64_t offset, const uint64_t bytes) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = const_cast<void *>(source);
  transfer.buffer_offset_bytes = offset;
  transfer.size_bytes = bytes;
  Error error;
  sllm_completion_t *completion = nullptr;
  return ok(sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                 &error.sink),
            SLLM_STATUS_OK, "sllm_buffer_copy_h2d", error) &&
         wait_release(&completion, "sllm_buffer_copy_h2d");
}

bool copy_d2h(const sllm_queue_t *const queue,
              const sllm_buffer_t *const buffer, void *const destination,
              const uint64_t offset, const uint64_t bytes) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = destination;
  transfer.buffer_offset_bytes = offset;
  transfer.size_bytes = bytes;
  Error error;
  sllm_completion_t *completion = nullptr;
  if (!ok(sllm_buffer_copy_d2h(queue, buffer, &transfer, &completion,
                               &error.sink),
          SLLM_STATUS_OK, "sllm_buffer_copy_d2h", error) ||
      !wait_release(&completion, "sllm_buffer_copy_d2h")) {
    return false;
  }
  return true;
}

sllm_tensor_binding_t binding(const sllm_buffer_t *const buffer,
                              const uint32_t dtype, const uint32_t encoding,
                              const uint64_t offset, const uint32_t rank,
                              const uint64_t first,
                              const uint64_t second = 0U) {
  sllm_tensor_binding_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.buffer = buffer;
  result.byte_offset = offset;
  result.dtype = dtype;
  result.encoding = encoding;
  result.rank = rank;
  result.shape[0] = first;
  result.stride_elements[0] = rank == 1U ? 1U : second;
  if (rank > 1U) {
    result.shape[1] = second;
    result.stride_elements[1] = 1U;
  }
  return result;
}

uint16_t bf16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & UINT32_C(0xffff);
  if (lower > UINT32_C(0x8000) ||
      (lower == UINT32_C(0x8000) && (upper & 1U) != 0U)) {
    ++upper;
  }
  return static_cast<uint16_t>(upper);
}

uint32_t f32_bits(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool release_buffer(sllm_buffer_t **const buffer) {
  if (buffer == nullptr || *buffer == nullptr) {
    return true;
  }
  Error error;
  return ok(sllm_buffer_release(buffer, &error.sink), SLLM_STATUS_OK,
            "sllm_buffer_release", error) &&
         *buffer == nullptr;
}

bool create_buffer(const sllm_context_t *const context, const uint64_t bytes,
                   sllm_buffer_t **const buffer) {
  sllm_buffer_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.size_bytes = bytes;
  Error error;
  return ok(sllm_buffer_create(context, &info, buffer, &error.sink),
            SLLM_STATUS_OK, "sllm_buffer_create", error);
}

bool run_plan(const sllm_queue_t *const queue,
              sllm_residual_rmsnorm_plan_t *plan,
              sllm_residual_rmsnorm_dispatch_info_t *dispatch) {
  Error error;
  sllm_completion_t *completion = nullptr;
  dispatch->struct_size = sizeof(*dispatch);
  dispatch->abi_version = SLLM_HIP_ABI_VERSION;
  dispatch->info_version = SLLM_HIP_RESIDUAL_RMSNORM_DISPATCH_INFO_VERSION;
  return ok(sllm_residual_rmsnorm_execute(plan, queue, &completion, dispatch,
                                          &error.sink),
            SLLM_STATUS_OK, "sllm_residual_rmsnorm_execute", error) &&
         wait_release(&completion, "sllm_residual_rmsnorm_execute");
}

bool run_elementwise(const sllm_queue_t *const queue,
                     sllm_elementwise_plan_t *plan,
                     sllm_elementwise_dispatch_info_t *dispatch) {
  Error error;
  sllm_completion_t *completion = nullptr;
  dispatch->struct_size = sizeof(*dispatch);
  dispatch->abi_version = SLLM_HIP_ABI_VERSION;
  dispatch->info_version = SLLM_HIP_ELEMENTWISE_DISPATCH_INFO_VERSION;
  return ok(sllm_elementwise_execute(plan, queue, &completion, dispatch,
                                     &error.sink),
            SLLM_STATUS_OK, "sllm_elementwise_execute", error) &&
         wait_release(&completion, "sllm_elementwise_execute");
}

bool run_matmul(const sllm_queue_t *const queue, sllm_matmul_plan_t *plan,
                sllm_matmul_dispatch_info_t *dispatch) {
  Error error;
  sllm_completion_t *completion = nullptr;
  dispatch->struct_size = sizeof(*dispatch);
  dispatch->abi_version = SLLM_HIP_ABI_VERSION;
  dispatch->info_version = SLLM_HIP_MATMUL_DISPATCH_INFO_VERSION;
  return ok(sllm_matmul_execute(plan, queue, &completion, dispatch,
                                &error.sink),
            SLLM_STATUS_OK, "sllm_matmul_execute", error) &&
         wait_release(&completion, "sllm_matmul_execute");
}

bool run_silu_case(const std::string &target) {
  constexpr uint64_t m = 17U;
  constexpr uint64_t k = 17408U;
  constexpr uint64_t n = 5120U;
  constexpr uint64_t encoded_offset = 64U;
  const uint64_t bf16_bytes = m * k * 2U;
  const uint64_t packed_bytes = m * ((k + 1U) / 2U);
  const uint64_t block_bytes = m * ((k + 15U) / 16U);
  const uint64_t encoded_bytes =
      encoded_offset + packed_bytes + block_bytes + 64U;
  const uint64_t weight_value_bytes = n * ((k + 1U) / 2U);
  const uint64_t weight_block_bytes = n * ((k + 15U) / 16U);
  const uint64_t weight_tensor_offset =
      (weight_value_bytes + weight_block_bytes + 3U) & ~UINT64_C(3);
  const uint64_t weight_bytes = weight_tensor_offset + 8U;
  const uint64_t output_bytes = m * n * 2U;

  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  Error error;
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  context_info.device_index = 0U;
  std::snprintf(context_info.expected_gcn_arch_name,
                sizeof(context_info.expected_gcn_arch_name), "%s",
                target.c_str());
  bool valid = ok(sllm_context_create(&context_info, &context, &error.sink),
                  SLLM_STATUS_OK, "silu context create", error);
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  valid = ok(sllm_queue_create(context, &queue_info, &queue, &error.sink),
             SLLM_STATUS_OK, "silu queue create", error) &&
          valid;

  sllm_buffer_t *gate = nullptr;
  sllm_buffer_t *up = nullptr;
  sllm_buffer_t *control_bf16 = nullptr;
  sllm_buffer_t *encoded = nullptr;
  sllm_buffer_t *weight = nullptr;
  sllm_buffer_t *control_output = nullptr;
  sllm_buffer_t *candidate_output = nullptr;
  valid = create_buffer(context, bf16_bytes, &gate) && valid;
  valid = create_buffer(context, bf16_bytes, &up) && valid;
  valid = create_buffer(context, bf16_bytes, &control_bf16) && valid;
  valid = create_buffer(context, encoded_bytes, &encoded) && valid;
  valid = create_buffer(context, weight_bytes, &weight) && valid;
  valid = create_buffer(context, output_bytes, &control_output) && valid;
  valid = create_buffer(context, output_bytes, &candidate_output) && valid;

  std::vector<uint16_t> host_gate(m * k, bf16(1.5F));
  std::vector<uint16_t> host_up(m * k, bf16(2.0F));
  std::vector<uint8_t> host_weight(static_cast<std::size_t>(weight_bytes),
                                   UINT8_C(0x22));
  std::fill(
      host_weight.begin() + static_cast<std::ptrdiff_t>(weight_value_bytes),
      host_weight.begin() + static_cast<std::ptrdiff_t>(weight_tensor_offset),
      UINT8_C(0x38));
  const float weight_tensor_scale = 1.0F;
  std::memcpy(host_weight.data() + weight_tensor_offset, &weight_tensor_scale,
              sizeof(weight_tensor_scale));
  std::memcpy(host_weight.data() + weight_tensor_offset + 4U, &kInputScale,
              sizeof(kInputScale));
  valid = copy_h2d(queue, gate, host_gate.data(), 0U, bf16_bytes) && valid;
  valid = copy_h2d(queue, up, host_up.data(), 0U, bf16_bytes) && valid;
  valid =
      copy_h2d(queue, weight, host_weight.data(), 0U, weight_bytes) && valid;

  const auto input_binding = [](const sllm_buffer_t *buffer) {
    return binding(buffer, SLLM_TENSOR_DTYPE_BF16,
                   SLLM_TENSOR_ENCODING_UNQUANTIZED, 0U, 2U, m, k);
  };
  const auto encoded_binding =
      binding(encoded, SLLM_TENSOR_DTYPE_U8,
              SLLM_TENSOR_ENCODING_NVFP4_W4A4_BLOCK16_E4M3FN_F32,
              encoded_offset, 2U, m, k);
  const auto weight_binding =
      binding(weight, SLLM_TENSOR_DTYPE_U8,
              SLLM_TENSOR_ENCODING_NVFP4_W4A4_BLOCK16_E4M3FN_F32, 0U, 2U, n, k);
  const auto output_binding = [&](const sllm_buffer_t *buffer) {
    return binding(buffer, SLLM_TENSOR_DTYPE_BF16,
                   SLLM_TENSOR_ENCODING_UNQUANTIZED, 0U, 2U, m, n);
  };

  sllm_elementwise_desc_t producer_desc{};
  producer_desc.struct_size = sizeof(producer_desc);
  producer_desc.abi_version = SLLM_HIP_ABI_VERSION;
  producer_desc.op_version = SLLM_HIP_ELEMENTWISE_VERSION;
  producer_desc.operation = SLLM_ELEMENTWISE_OPERATION_SILU_MUL;
  producer_desc.input0 = input_binding(gate);
  producer_desc.input1 = input_binding(up);
  producer_desc.output = encoded_binding;
  producer_desc.reserved[0] = kInputScaleBits;
  sllm_elementwise_desc_t control_producer = producer_desc;
  control_producer.output = input_binding(control_bf16);
  control_producer.reserved[0] = 0U;
  sllm_elementwise_plan_t *producer_plan = nullptr;
  sllm_elementwise_plan_t *control_producer_plan = nullptr;
  valid = ok(sllm_elementwise_prepare(context, &producer_desc, &producer_plan,
                                      &error.sink),
             SLLM_STATUS_OK, "silu producer prepare", error) &&
          valid;
  valid = ok(sllm_elementwise_prepare(context, &control_producer,
                                      &control_producer_plan, &error.sink),
             SLLM_STATUS_OK, "silu control producer prepare", error) &&
          valid;

  sllm_matmul_desc_t candidate_desc{};
  candidate_desc.struct_size = sizeof(candidate_desc);
  candidate_desc.abi_version = SLLM_HIP_ABI_VERSION;
  candidate_desc.op_version = SLLM_HIP_MATMUL_NVFP4_W4A4_VERSION;
  candidate_desc.activation = encoded_binding;
  candidate_desc.weight = weight_binding;
  candidate_desc.output = output_binding(candidate_output);
  sllm_matmul_desc_t control_desc = candidate_desc;
  control_desc.activation = input_binding(control_bf16);
  control_desc.output = output_binding(control_output);
  sllm_matmul_plan_t *candidate_matmul = nullptr;
  sllm_matmul_plan_t *control_matmul = nullptr;
  valid = ok(sllm_matmul_prepare(context, &candidate_desc, &candidate_matmul,
                                 &error.sink),
             SLLM_STATUS_OK, "silu candidate matmul prepare", error) &&
          valid;
  valid = ok(sllm_matmul_prepare(context, &control_desc, &control_matmul,
                                 &error.sink),
             SLLM_STATUS_OK, "silu control matmul prepare", error) &&
          valid;

  sllm_elementwise_dispatch_info_t producer_dispatch{};
  sllm_elementwise_dispatch_info_t control_producer_dispatch{};
  sllm_matmul_dispatch_info_t candidate_dispatch{};
  sllm_matmul_dispatch_info_t control_dispatch{};
  if (valid) {
    valid = run_elementwise(queue, control_producer_plan,
                            &control_producer_dispatch) &&
            valid;
    valid = run_matmul(queue, control_matmul, &control_dispatch) && valid;
    valid = run_elementwise(queue, producer_plan, &producer_dispatch) && valid;
    valid = run_matmul(queue, candidate_matmul, &candidate_dispatch) && valid;
  }
  if (valid) {
    std::vector<uint16_t> control_host(m * n);
    std::vector<uint16_t> candidate_host(m * n);
    valid = copy_d2h(queue, control_output, control_host.data(), 0U,
                     output_bytes) &&
            valid;
    valid = copy_d2h(queue, candidate_output, candidate_host.data(), 0U,
                     output_bytes) &&
            valid;
    std::size_t first_diff = control_host.size();
    for (std::size_t index = 0U; index < control_host.size(); ++index) {
      if (control_host[index] != candidate_host[index]) {
        first_diff = index;
        break;
      }
    }
    if (first_diff != control_host.size()) {
      std::cerr << "FAIL: public SiLU output mismatch index=" << first_diff
                << " control=0x" << std::hex << control_host[first_diff]
                << " candidate=0x" << candidate_host[first_diff] << std::dec
                << " producer=" << producer_dispatch.kernel_symbol
                << " candidate_matmul=" << candidate_dispatch.kernel_symbol
                << " control_matmul=" << control_dispatch.kernel_symbol << '\n';
      valid = false;
    } else {
      std::cout
          << "{\"public_silu_prequant_bitwise\":true,\"producer_kernel\":\""
          << producer_dispatch.kernel_symbol << "\",\"candidate_matmul\":\""
          << candidate_dispatch.kernel_symbol << "\",\"control_matmul\":\""
          << control_dispatch.kernel_symbol << "\"}\n";
    }
  }
  if (producer_plan != nullptr)
    valid = ok(sllm_elementwise_plan_release(&producer_plan, &error.sink),
               SLLM_STATUS_OK, "silu producer release", error) &&
            valid;
  if (control_producer_plan != nullptr)
    valid =
        ok(sllm_elementwise_plan_release(&control_producer_plan, &error.sink),
           SLLM_STATUS_OK, "silu control producer release", error) &&
        valid;
  if (candidate_matmul != nullptr)
    valid = ok(sllm_matmul_plan_release(&candidate_matmul, &error.sink),
               SLLM_STATUS_OK, "silu candidate matmul release", error) &&
            valid;
  if (control_matmul != nullptr)
    valid = ok(sllm_matmul_plan_release(&control_matmul, &error.sink),
               SLLM_STATUS_OK, "silu control matmul release", error) &&
            valid;
  valid = release_buffer(&candidate_output) && valid;
  valid = release_buffer(&control_output) && valid;
  valid = release_buffer(&weight) && valid;
  valid = release_buffer(&encoded) && valid;
  valid = release_buffer(&control_bf16) && valid;
  valid = release_buffer(&up) && valid;
  valid = release_buffer(&gate) && valid;
  if (queue != nullptr)
    valid = ok(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
               "silu queue release", error) &&
            valid;
  if (context != nullptr)
    valid = ok(sllm_context_release(&context, &error.sink), SLLM_STATUS_OK,
               "silu context release", error) &&
            valid;
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  const std::string target = argc > 1 ? argv[1] : "";
  if (target != "gfx1030" && target != "gfx1201") {
    std::cerr << "usage: probe gfx1030|gfx1201\n";
    return 2;
  }
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  bool valid = true;
  Error error;
  sllm_context_create_info_t context_info{};
  context_info.struct_size = sizeof(context_info);
  context_info.abi_version = SLLM_HIP_ABI_VERSION;
  context_info.device_index = 0U;
  std::snprintf(context_info.expected_gcn_arch_name,
                sizeof(context_info.expected_gcn_arch_name), "%s",
                target.c_str());
  valid = ok(sllm_context_create(&context_info, &context, &error.sink),
             SLLM_STATUS_OK, "sllm_context_create", error) &&
          valid;
  sllm_queue_create_info_t queue_info{};
  queue_info.struct_size = sizeof(queue_info);
  queue_info.abi_version = SLLM_HIP_ABI_VERSION;
  valid = ok(sllm_queue_create(context, &queue_info, &queue, &error.sink),
             SLLM_STATUS_OK, "sllm_queue_create", error) &&
          valid;

  const uint64_t bf16_input_bytes = kM * kK * 2U;
  const uint64_t raw_scale_bytes = kK * 2U;
  const uint64_t packed_bytes = kM * ((kK + 1U) / 2U);
  const uint64_t block_scale_bytes = kM * ((kK + 15U) / 16U);
  const uint64_t encoded_payload_bytes = packed_bytes + block_scale_bytes;
  const uint64_t encoded_bytes = kEncodedOffset + encoded_payload_bytes + 64U;
  const uint64_t residual_output_bytes = bf16_input_bytes;
  const uint64_t weight_value_bytes = kN * ((kK + 1U) / 2U);
  const uint64_t weight_block_scale_bytes = kN * ((kK + 15U) / 16U);
  const uint64_t weight_tensor_scale_offset =
      (weight_value_bytes + weight_block_scale_bytes + 3U) & ~UINT64_C(3);
  const uint64_t weight_bytes = weight_tensor_scale_offset + 8U;
  const uint64_t output_bytes = kM * kN * 2U;

  sllm_buffer_t *residual = nullptr;
  sllm_buffer_t *addend = nullptr;
  sllm_buffer_t *raw_scale = nullptr;
  sllm_buffer_t *residual_output = nullptr;
  sllm_buffer_t *control_norm_output = nullptr;
  sllm_buffer_t *encoded_output = nullptr;
  sllm_buffer_t *weight = nullptr;
  sllm_buffer_t *control_output = nullptr;
  sllm_buffer_t *candidate_output = nullptr;
  valid = create_buffer(context, bf16_input_bytes, &residual) && valid;
  valid = create_buffer(context, bf16_input_bytes, &addend) && valid;
  valid = create_buffer(context, raw_scale_bytes, &raw_scale) && valid;
  valid =
      create_buffer(context, residual_output_bytes, &residual_output) && valid;
  valid = create_buffer(context, residual_output_bytes, &control_norm_output) &&
          valid;
  valid = create_buffer(context, encoded_bytes, &encoded_output) && valid;
  valid = create_buffer(context, weight_bytes, &weight) && valid;
  valid = create_buffer(context, output_bytes, &control_output) && valid;
  valid = create_buffer(context, output_bytes, &candidate_output) && valid;

  std::vector<uint16_t> host_bf16(kM * kK, bf16(1.0F));
  std::vector<uint16_t> host_scale(kK, bf16(1.0F));
  std::vector<uint8_t> host_weight(static_cast<std::size_t>(weight_bytes),
                                   UINT8_C(0x22));
  std::fill(host_weight.begin() +
                static_cast<std::ptrdiff_t>(weight_value_bytes),
            host_weight.begin() +
                static_cast<std::ptrdiff_t>(weight_tensor_scale_offset),
            UINT8_C(0x38));
  float weight_tensor_scale = 1.0F;
  std::memcpy(host_weight.data() + weight_tensor_scale_offset,
              &weight_tensor_scale, sizeof(weight_tensor_scale));
  std::memcpy(host_weight.data() + weight_tensor_scale_offset + 4U,
              &kInputScale, sizeof(kInputScale));
  valid = copy_h2d(queue, residual, host_bf16.data(), 0U, bf16_input_bytes) &&
          valid;
  valid =
      copy_h2d(queue, addend, host_bf16.data(), 0U, bf16_input_bytes) && valid;
  valid = copy_h2d(queue, raw_scale, host_scale.data(), 0U, raw_scale_bytes) &&
          valid;
  valid =
      copy_h2d(queue, weight, host_weight.data(), 0U, weight_bytes) && valid;

  const auto bf16_input = [&](const sllm_buffer_t *buffer,
                              const uint64_t offset) {
    return binding(buffer, SLLM_TENSOR_DTYPE_BF16,
                   SLLM_TENSOR_ENCODING_UNQUANTIZED, offset, 2U, kM, kK);
  };
  const auto bf16_weight = binding(
      weight, SLLM_TENSOR_DTYPE_U8,
      SLLM_TENSOR_ENCODING_NVFP4_W4A4_BLOCK16_E4M3FN_F32, 0U, 2U, kN, kK);
  const auto bf16_output = [&](const sllm_buffer_t *buffer) {
    return binding(buffer, SLLM_TENSOR_DTYPE_BF16,
                   SLLM_TENSOR_ENCODING_UNQUANTIZED, 0U, 2U, kM, kN);
  };
  const auto encoded =
      binding(encoded_output, SLLM_TENSOR_DTYPE_U8,
              SLLM_TENSOR_ENCODING_NVFP4_W4A4_BLOCK16_E4M3FN_F32,
              kEncodedOffset, 2U, kM, kK);
  const auto residual_out = bf16_input(residual_output, 0U);
  const auto raw = binding(raw_scale, SLLM_TENSOR_DTYPE_BF16,
                           SLLM_TENSOR_ENCODING_UNQUANTIZED, 0U, 1U, kK);

  sllm_residual_rmsnorm_desc_t producer_desc{};
  producer_desc.struct_size = sizeof(producer_desc);
  producer_desc.abi_version = SLLM_HIP_ABI_VERSION;
  producer_desc.op_version = SLLM_HIP_RESIDUAL_RMSNORM_VERSION;
  producer_desc.accumulation_dtype = SLLM_RMSNORM_ACCUMULATION_F32;
  producer_desc.scale_mode = SLLM_RMSNORM_SCALE_MODE_OFFSET_ONE;
  producer_desc.alias_policy = SLLM_RMSNORM_ALIAS_POLICY_REJECT_OVERLAP;
  producer_desc.epsilon_bits = f32_bits(1.0e-6F);
  producer_desc.residual = bf16_input(residual, 0U);
  producer_desc.addend = bf16_input(addend, 0U);
  producer_desc.raw_scale = raw;
  producer_desc.residual_output = residual_out;
  producer_desc.output = encoded;
  producer_desc.reserved[0] = kInputScaleBits;

  sllm_residual_rmsnorm_plan_t *producer_plan = nullptr;
  valid = ok(sllm_residual_rmsnorm_prepare(context, &producer_desc,
                                           &producer_plan, &error.sink),
             SLLM_STATUS_OK, "producer prepare", error) &&
          valid;
  sllm_residual_rmsnorm_desc_t control_producer = producer_desc;
  control_producer.output = bf16_input(control_norm_output, 0U);
  control_producer.reserved[0] = 0U;
  sllm_residual_rmsnorm_plan_t *control_producer_plan = nullptr;
  valid = ok(sllm_residual_rmsnorm_prepare(context, &control_producer,
                                           &control_producer_plan, &error.sink),
             SLLM_STATUS_OK, "control producer prepare", error) &&
          valid;

  sllm_matmul_desc_t candidate_desc{};
  candidate_desc.struct_size = sizeof(candidate_desc);
  candidate_desc.abi_version = SLLM_HIP_ABI_VERSION;
  candidate_desc.op_version = SLLM_HIP_MATMUL_NVFP4_W4A4_VERSION;
  candidate_desc.activation = encoded;
  candidate_desc.weight = bf16_weight;
  candidate_desc.output = bf16_output(candidate_output);
  sllm_matmul_plan_t *candidate_matmul = nullptr;
  valid = ok(sllm_matmul_prepare(context, &candidate_desc, &candidate_matmul,
                                 &error.sink),
             SLLM_STATUS_OK, "candidate matmul prepare", error) &&
          valid;

  sllm_matmul_desc_t control_desc = candidate_desc;
  control_desc.activation = bf16_input(control_norm_output, 0U);
  control_desc.output = bf16_output(control_output);
  sllm_matmul_plan_t *control_matmul = nullptr;
  valid = ok(sllm_matmul_prepare(context, &control_desc, &control_matmul,
                                 &error.sink),
             SLLM_STATUS_OK, "control matmul prepare", error) &&
          valid;

  sllm_residual_rmsnorm_dispatch_info_t producer_dispatch{};
  sllm_residual_rmsnorm_dispatch_info_t control_producer_dispatch{};
  sllm_matmul_dispatch_info_t candidate_dispatch{};
  sllm_matmul_dispatch_info_t control_dispatch{};
  valid = run_plan(queue, control_producer_plan, &control_producer_dispatch) &&
          valid;
  valid = run_matmul(queue, control_matmul, &control_dispatch) && valid;
  valid = run_plan(queue, producer_plan, &producer_dispatch) && valid;
  valid = run_matmul(queue, candidate_matmul, &candidate_dispatch) && valid;

  if (!valid) {
    std::cerr << "FAIL: public prequant execution setup or submit failed\n";
  }

  std::vector<uint16_t> control_host(kM * kN);
  std::vector<uint16_t> candidate_host(kM * kN);
  if (valid) {
    valid = copy_d2h(queue, control_output, control_host.data(), 0U,
                     output_bytes) &&
            valid;
    valid = copy_d2h(queue, candidate_output, candidate_host.data(), 0U,
                     output_bytes) &&
            valid;
    std::size_t first_diff = control_host.size();
    for (std::size_t index = 0U; index < control_host.size(); ++index) {
      if (control_host[index] != candidate_host[index]) {
        first_diff = index;
        break;
      }
    }
    if (first_diff != control_host.size()) {
      std::cerr << "FAIL: public output mismatch index=" << first_diff
                << " control=0x" << std::hex << control_host[first_diff]
                << " candidate=0x" << candidate_host[first_diff] << std::dec
                << " producer_kernel=" << producer_dispatch.kernel_symbol
                << " candidate_matmul=" << candidate_dispatch.kernel_symbol
                << " control_matmul=" << control_dispatch.kernel_symbol << '\n';
      valid = false;
    } else {
      std::cout << "{\"public_prequant_bitwise\":true,\"producer_kernel\":\""
                << producer_dispatch.kernel_symbol
                << "\",\"candidate_matmul\":\""
                << candidate_dispatch.kernel_symbol
                << "\",\"control_matmul\":\"" << control_dispatch.kernel_symbol
                << "\"}\n";
    }
  }

  if (producer_plan != nullptr)
    valid = ok(sllm_residual_rmsnorm_plan_release(&producer_plan, &error.sink),
               SLLM_STATUS_OK, "producer plan release", error) &&
            valid;
  if (control_producer_plan != nullptr)
    valid = ok(sllm_residual_rmsnorm_plan_release(&control_producer_plan,
                                                  &error.sink),
               SLLM_STATUS_OK, "control producer plan release", error) &&
            valid;
  if (candidate_matmul != nullptr)
    valid = ok(sllm_matmul_plan_release(&candidate_matmul, &error.sink),
               SLLM_STATUS_OK, "candidate matmul release", error) &&
            valid;
  if (control_matmul != nullptr)
    valid = ok(sllm_matmul_plan_release(&control_matmul, &error.sink),
               SLLM_STATUS_OK, "control matmul release", error) &&
            valid;
  valid = release_buffer(&candidate_output) && valid;
  valid = release_buffer(&control_output) && valid;
  valid = release_buffer(&weight) && valid;
  valid = release_buffer(&encoded_output) && valid;
  valid = release_buffer(&control_norm_output) && valid;
  valid = release_buffer(&residual_output) && valid;
  valid = release_buffer(&raw_scale) && valid;
  valid = release_buffer(&addend) && valid;
  valid = release_buffer(&residual) && valid;
  if (queue != nullptr)
    valid = ok(sllm_queue_release(&queue, &error.sink), SLLM_STATUS_OK,
               "queue release", error) &&
            valid;
  if (context != nullptr)
    valid = ok(sllm_context_release(&context, &error.sink), SLLM_STATUS_OK,
               "context release", error) &&
            valid;
  valid = run_silu_case(target) && valid;
  return valid ? 0 : 1;
}
