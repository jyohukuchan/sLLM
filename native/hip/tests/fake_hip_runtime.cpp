#include <hip/hip_runtime.h>

#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <unordered_set>

struct FakeHipStream {};
struct FakeHipMemHandle {
  std::size_t size = 0U;
};

namespace {

struct State final {
  std::mutex mutex;
  std::condition_variable condition;
  bool event_create_gate = false;
  bool event_create_entered = false;
  bool event_query_gate = false;
  bool event_query_entered = false;
  bool completion_pending = false;
  hipError_t rmsnorm_launch_status = hipSuccess;
  hipError_t elementwise_launch_status = hipSuccess;
  hipError_t matmul_launch_status = hipSuccess;
  hipError_t argmax_launch_status = hipSuccess;
  hipError_t kv_state_append_launch_status = hipSuccess;
  hipError_t causal_attention_launch_status = hipSuccess;
  hipError_t event_record_status = hipSuccess;
  std::size_t rmsnorm_launch_calls = 0U;
  std::size_t elementwise_copy_launch_calls = 0U;
  std::size_t elementwise_add_launch_calls = 0U;
  std::size_t elementwise_silu_mul_launch_calls = 0U;
  std::size_t elementwise_sigmoid_mul_launch_calls = 0U;
  std::size_t embedding_gather_launch_calls = 0U;
  std::size_t matmul_launch_calls = 0U;
  std::size_t argmax_launch_calls = 0U;
  std::size_t attention_preprocess_launch_calls = 0U;
  uint64_t elementwise_last_element_count = 0U;
  uint64_t matmul_last_m = 0U;
  uint64_t matmul_last_k = 0U;
  uint64_t matmul_last_n = 0U;
  uint64_t matmul_last_output_elements = 0U;
  uint64_t argmax_last_m = 0U;
  uint64_t argmax_last_v = 0U;
  uint32_t attention_preprocess_last_m = 0U;
  std::size_t kv_state_append_launch_calls = 0U;
  std::size_t causal_attention_launch_calls = 0U;
  uint32_t kv_state_last_token_count = 0U;
  uint64_t kv_state_last_capacity_tokens = 0U;
  uint64_t kv_state_last_start_position = 0U;
  const uint16_t *kv_key_output = nullptr;
  const uint16_t *kv_value_output = nullptr;
  uint32_t rmsnorm_last_normalized_size = 0U;
  uint32_t rmsnorm_last_row_count = 0U;
  std::size_t event_destroy_calls = 0U;
  std::size_t stream_destroy_calls = 0U;
  std::size_t allocation_free_calls = 0U;
  std::unordered_set<hipEvent_t> events;
  std::unordered_set<hipStream_t> streams;
  std::unordered_set<void *> allocations;
  std::unordered_set<void *> reservations;
  std::unordered_set<hipMemGenericAllocationHandle_t> vmm_handles;
};

State state;

uint16_t bf16_to_f16(const uint16_t value) noexcept {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  const uint32_t sign = (bits >> 16U) & UINT32_C(0x8000);
  const uint32_t exponent = (bits >> 23U) & UINT32_C(0xff);
  const uint32_t fraction = bits & UINT32_C(0x7fffff);
  if (exponent == UINT32_C(0xff)) {
    return static_cast<uint16_t>(
        sign | (fraction == 0U ? UINT32_C(0x7c00) : UINT32_C(0x7e00)));
  }
  const int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<uint16_t>(sign | UINT32_C(0x7c00));
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    const uint32_t mantissa = fraction | UINT32_C(0x800000);
    const uint32_t shift = static_cast<uint32_t>(14 - half_exponent);
    uint32_t rounded = mantissa >> shift;
    const uint32_t remainder = mantissa & ((UINT32_C(1) << shift) - 1U);
    const uint32_t halfway = UINT32_C(1) << (shift - 1U);
    if (remainder > halfway ||
        (remainder == halfway && (rounded & UINT32_C(1)) != 0U)) {
      ++rounded;
    }
    return static_cast<uint16_t>(sign | rounded);
  }
  uint32_t rounded_fraction = fraction >> 13U;
  const uint32_t remainder = fraction & UINT32_C(0x1fff);
  if (remainder > UINT32_C(0x1000) ||
      (remainder == UINT32_C(0x1000) &&
       (rounded_fraction & UINT32_C(1)) != 0U)) {
    ++rounded_fraction;
    if (rounded_fraction == UINT32_C(0x400)) {
      rounded_fraction = 0U;
      if (half_exponent + 1 >= 31) {
        return static_cast<uint16_t>(sign | UINT32_C(0x7c00));
      }
      return static_cast<uint16_t>(
          sign | (static_cast<uint32_t>(half_exponent + 1) << 10U));
    }
  }
  return static_cast<uint16_t>(
      sign | (static_cast<uint32_t>(half_exponent) << 10U) | rounded_fraction);
}

float f16_to_f32(const uint16_t raw) noexcept {
  const uint32_t sign = (static_cast<uint32_t>(raw) & 0x8000U) << 16U;
  const uint32_t exponent = (static_cast<uint32_t>(raw) >> 10U) & 0x1fU;
  const uint32_t fraction = static_cast<uint32_t>(raw) & 0x03ffU;
  uint32_t bits = 0U;
  if (exponent == 0U) {
    if (fraction == 0U) {
      bits = sign;
    } else {
      uint32_t normalized = fraction;
      uint32_t shift = 0U;
      while ((normalized & 0x0400U) == 0U) {
        normalized <<= 1U;
        ++shift;
      }
      normalized &= 0x03ffU;
      bits = sign | ((127U - 14U - shift) << 23U) | (normalized << 13U);
    }
  } else if (exponent == 0x1fU) {
    bits = sign | 0x7f800000U | (fraction << 13U);
  } else {
    bits = sign | ((exponent + 112U) << 23U) | (fraction << 13U);
  }
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

float bf16_to_f32(const uint16_t raw) noexcept {
  const uint32_t bits = static_cast<uint32_t>(raw) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint16_t f32_to_bf16_rne(const float value) noexcept {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>(bits >> 16U);
}

} // namespace

namespace fake_hip {

uint32_t f16_to_f32_bits_for_test(const uint16_t raw) noexcept {
  const float value = f16_to_f32(raw);
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void reset() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  state.event_create_gate = false;
  state.event_create_entered = false;
  state.event_query_gate = false;
  state.event_query_entered = false;
  state.completion_pending = false;
  state.rmsnorm_launch_status = hipSuccess;
  state.elementwise_launch_status = hipSuccess;
  state.matmul_launch_status = hipSuccess;
  state.argmax_launch_status = hipSuccess;
  state.kv_state_append_launch_status = hipSuccess;
  state.causal_attention_launch_status = hipSuccess;
  state.event_record_status = hipSuccess;
  state.rmsnorm_launch_calls = 0U;
  state.elementwise_copy_launch_calls = 0U;
  state.elementwise_add_launch_calls = 0U;
  state.elementwise_silu_mul_launch_calls = 0U;
  state.elementwise_sigmoid_mul_launch_calls = 0U;
  state.embedding_gather_launch_calls = 0U;
  state.matmul_launch_calls = 0U;
  state.argmax_launch_calls = 0U;
  state.attention_preprocess_launch_calls = 0U;
  state.kv_state_append_launch_calls = 0U;
  state.causal_attention_launch_calls = 0U;
  state.elementwise_last_element_count = 0U;
  state.matmul_last_m = 0U;
  state.matmul_last_k = 0U;
  state.matmul_last_n = 0U;
  state.matmul_last_output_elements = 0U;
  state.argmax_last_m = 0U;
  state.argmax_last_v = 0U;
  state.attention_preprocess_last_m = 0U;
  state.kv_state_last_token_count = 0U;
  state.kv_state_last_capacity_tokens = 0U;
  state.kv_state_last_start_position = 0U;
  state.kv_key_output = nullptr;
  state.kv_value_output = nullptr;
  state.rmsnorm_last_normalized_size = 0U;
  state.rmsnorm_last_row_count = 0U;
  state.event_destroy_calls = 0U;
  state.stream_destroy_calls = 0U;
  state.allocation_free_calls = 0U;
}

hipError_t rmsnorm_launch(const uint16_t *const /*activation*/,
                          const uint16_t *const /*raw_scale*/,
                          uint16_t *const /*output*/,
                          const uint32_t normalized_size,
                          const uint32_t row_count, const float /*epsilon*/,
                          const hipStream_t /*stream*/) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  ++state.rmsnorm_launch_calls;
  state.rmsnorm_last_normalized_size = normalized_size;
  state.rmsnorm_last_row_count = row_count;
  return state.rmsnorm_launch_status;
}

hipError_t elementwise_copy_launch(const uint16_t *const input,
                                   uint16_t *const output,
                                   const uint64_t element_count,
                                   const hipStream_t /*stream*/) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  ++state.elementwise_copy_launch_calls;
  state.elementwise_last_element_count = element_count;
  if (state.elementwise_launch_status == hipSuccess) {
    std::memcpy(output, input,
                static_cast<std::size_t>(element_count) * sizeof(uint16_t));
  }
  return state.elementwise_launch_status;
}

hipError_t elementwise_add_launch(const uint16_t *const /*input0*/,
                                  const uint16_t *const /*input1*/,
                                  uint16_t *const /*output*/,
                                  const uint64_t element_count,
                                  const hipStream_t /*stream*/) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  ++state.elementwise_add_launch_calls;
  state.elementwise_last_element_count = element_count;
  return state.elementwise_launch_status;
}

hipError_t elementwise_silu_mul_launch(const uint16_t *const /*gate*/,
                                       const uint16_t *const /*up*/,
                                       uint16_t *const /*output*/,
                                       const uint64_t element_count,
                                       const hipStream_t /*stream*/) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  ++state.elementwise_silu_mul_launch_calls;
  state.elementwise_last_element_count = element_count;
  return state.elementwise_launch_status;
}

hipError_t elementwise_sigmoid_mul_launch(
    const uint16_t *const gate, const uint16_t *const attention_value,
    uint16_t *const output, const uint64_t element_count,
    const hipStream_t /*stream*/) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  ++state.elementwise_sigmoid_mul_launch_calls;
  state.elementwise_last_element_count = element_count;
  if (state.elementwise_launch_status == hipSuccess) {
    for (uint64_t index = 0U; index != element_count; ++index) {
      const float gate_value = bf16_to_f32(gate[index]);
      const float sigmoid = 1.0F / (1.0F + std::exp(-gate_value));
      output[index] =
          f32_to_bf16_rne(sigmoid * bf16_to_f32(attention_value[index]));
    }
  }
  return state.elementwise_launch_status;
}

hipError_t embedding_gather_launch(const uint16_t *const weight,
                                   const int32_t *const token_ids,
                                   uint16_t *const output,
                                   const uint64_t token_count,
                                   const uint64_t hidden_size,
                                   const hipStream_t /*stream*/) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  ++state.embedding_gather_launch_calls;
  for (uint64_t token = 0U; token != token_count; ++token) {
    const uint64_t row = static_cast<uint64_t>(token_ids[token]);
    std::memcpy(output + token * hidden_size, weight + row * hidden_size,
                static_cast<std::size_t>(hidden_size) * sizeof(uint16_t));
  }
  return hipSuccess;
}

hipError_t matmul_launch(const uint16_t *const /*activation*/,
                         const uint16_t *const /*weight*/,
                         uint16_t *const /*output*/, const uint64_t m,
                         const uint64_t k, const uint64_t n,
                         const hipStream_t /*stream*/) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  ++state.matmul_launch_calls;
  state.matmul_last_m = m;
  state.matmul_last_k = k;
  state.matmul_last_n = n;
  state.matmul_last_output_elements = m * n;
  return state.matmul_launch_status;
}

hipError_t argmax_launch(const uint16_t *const logits, int32_t *const output,
                         const uint64_t m, const uint64_t v,
                         const hipStream_t /*stream*/) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  ++state.argmax_launch_calls;
  state.argmax_last_m = m;
  state.argmax_last_v = v;
  if (state.argmax_launch_status != hipSuccess) {
    return state.argmax_launch_status;
  }
  for (uint64_t row = 0U; row != m; ++row) {
    float maximum = 0.0F;
    uint64_t index = 0U;
    bool valid = false;
    bool has_nan = false;
    for (uint64_t column = 0U; column != v; ++column) {
      const float value = bf16_to_f32(logits[row * v + column]);
      if (std::isnan(value)) {
        has_nan = true;
      } else if (!valid || value > maximum ||
                 (value == maximum && column < index)) {
        maximum = value;
        index = column;
        valid = true;
      }
    }
    output[row] = has_nan ? -1 : static_cast<int32_t>(index);
  }
  return hipSuccess;
}

hipError_t attention_preprocess_launch(
    const uint16_t *const /*packed_q_gate*/, const uint16_t *const /*k*/,
    const uint16_t *const /*q_raw_scale*/,
    const uint16_t *const /*k_raw_scale*/, const int32_t *const /*positions*/,
    uint16_t *const /*q_output*/, uint16_t *const /*gate_output*/,
    uint16_t *const /*k_output*/, const uint32_t m,
    const hipStream_t /*stream*/) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  ++state.attention_preprocess_launch_calls;
  state.attention_preprocess_last_m = m;
  return hipSuccess;
}

hipError_t kv_state_append_launch(
    const uint16_t *const key_input, const uint16_t *const value_input,
    uint16_t *const key_output, uint16_t *const value_output,
    const uint32_t token_count, const uint64_t capacity_tokens,
    const uint64_t start_position, const hipStream_t /*stream*/) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  ++state.kv_state_append_launch_calls;
  state.kv_state_last_token_count = token_count;
  state.kv_state_last_capacity_tokens = capacity_tokens;
  state.kv_state_last_start_position = start_position;
  if (state.kv_state_append_launch_status != hipSuccess) {
    return state.kv_state_append_launch_status;
  }
  const uint64_t elements_per_row = 4U * 256U;
  for (uint64_t element = 0U;
       element < static_cast<uint64_t>(token_count) * elements_per_row;
       ++element) {
    const uint64_t row = element / elements_per_row;
    const uint64_t within_row = element % elements_per_row;
    const uint64_t output_offset =
        (start_position + row) * elements_per_row + within_row;
    key_output[output_offset] = bf16_to_f16(key_input[element]);
    value_output[output_offset] = bf16_to_f16(value_input[element]);
  }
  state.kv_key_output = key_output;
  state.kv_value_output = value_output;
  return hipSuccess;
}

hipError_t causal_attention_launch(
    const uint16_t *const query, const uint16_t *const key,
    const uint16_t *const value, uint16_t *const output,
    const uint32_t query_count, const uint64_t /*capacity_tokens*/,
    const uint64_t start_position, const uint64_t committed_kv_length,
    const hipStream_t /*stream*/) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  ++state.causal_attention_launch_calls;
  if (state.causal_attention_launch_status != hipSuccess) {
    return state.causal_attention_launch_status;
  }
  for (uint32_t row = 0U; row != query_count; ++row) {
    const uint64_t position = start_position + row;
    if (position >= committed_kv_length) {
      return hipErrorInvalidValue;
    }
    for (uint32_t head = 0U; head != 16U; ++head) {
      const uint16_t *const query_head =
          query + (static_cast<uint64_t>(row) * 16U + head) * 256U;
      const uint32_t kv_head = head / 4U;
      float maximum = -INFINITY;
      for (uint64_t key_position = 0U; key_position <= position;
           ++key_position) {
        float dot = 0.0F;
        for (uint32_t dimension = 0U; dimension != 256U; ++dimension) {
          dot +=
              bf16_to_f32(query_head[dimension]) *
              f16_to_f32(key[(key_position * 4U + kv_head) * 256U + dimension]);
        }
        maximum = std::fmax(maximum, dot * 0.0625F);
      }
      float denominator = 0.0F;
      for (uint64_t key_position = 0U; key_position <= position;
           ++key_position) {
        float dot = 0.0F;
        for (uint32_t dimension = 0U; dimension != 256U; ++dimension) {
          dot +=
              bf16_to_f32(query_head[dimension]) *
              f16_to_f32(key[(key_position * 4U + kv_head) * 256U + dimension]);
        }
        denominator += std::exp(dot * 0.0625F - maximum);
      }
      uint16_t *const output_head =
          output + (static_cast<uint64_t>(row) * 16U + head) * 256U;
      for (uint32_t dimension = 0U; dimension != 256U; ++dimension) {
        float accumulation = 0.0F;
        for (uint64_t key_position = 0U; key_position <= position;
             ++key_position) {
          float dot = 0.0F;
          for (uint32_t dot_dimension = 0U; dot_dimension != 256U;
               ++dot_dimension) {
            dot +=
                bf16_to_f32(query_head[dot_dimension]) *
                f16_to_f32(
                    key[(key_position * 4U + kv_head) * 256U + dot_dimension]);
          }
          const float probability =
              std::exp(dot * 0.0625F - maximum) / denominator;
          accumulation +=
              probability *
              f16_to_f32(
                  value[(key_position * 4U + kv_head) * 256U + dimension]);
        }
        output_head[dimension] = f32_to_bf16_rne(accumulation);
      }
    }
  }
  return hipSuccess;
}

std::size_t embedding_gather_launch_calls() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.embedding_gather_launch_calls;
}

std::size_t matmul_launch_calls() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.matmul_launch_calls;
}

std::size_t attention_preprocess_launch_calls() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.attention_preprocess_launch_calls;
}

uint32_t attention_preprocess_last_m() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.attention_preprocess_last_m;
}

std::size_t kv_state_append_launch_calls() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.kv_state_append_launch_calls;
}

std::size_t causal_attention_launch_calls() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.causal_attention_launch_calls;
}

void set_causal_attention_launch_status(const hipError_t status) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  state.causal_attention_launch_status = status;
}

uint32_t kv_state_last_token_count() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.kv_state_last_token_count;
}

uint64_t kv_state_last_capacity_tokens() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.kv_state_last_capacity_tokens;
}

uint64_t kv_state_last_start_position() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.kv_state_last_start_position;
}

void set_kv_state_append_launch_status(const hipError_t status) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  state.kv_state_append_launch_status = status;
}

bool copy_kv_key_output(uint16_t *const destination,
                        const uint64_t element_count) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  if (destination == nullptr || state.kv_key_output == nullptr) {
    return false;
  }
  std::memcpy(destination, state.kv_key_output,
              static_cast<std::size_t>(element_count) * sizeof(uint16_t));
  return true;
}

bool copy_kv_value_output(uint16_t *const destination,
                          const uint64_t element_count) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  if (destination == nullptr || state.kv_value_output == nullptr) {
    return false;
  }
  std::memcpy(destination, state.kv_value_output,
              static_cast<std::size_t>(element_count) * sizeof(uint16_t));
  return true;
}

uint64_t matmul_last_m() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.matmul_last_m;
}

uint64_t matmul_last_k() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.matmul_last_k;
}

uint64_t matmul_last_n() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.matmul_last_n;
}

uint64_t matmul_last_output_elements() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.matmul_last_output_elements;
}

void set_elementwise_launch_status(const hipError_t status) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  state.elementwise_launch_status = status;
}

std::size_t elementwise_copy_launch_calls() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.elementwise_copy_launch_calls;
}

std::size_t elementwise_add_launch_calls() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.elementwise_add_launch_calls;
}

std::size_t elementwise_silu_mul_launch_calls() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.elementwise_silu_mul_launch_calls;
}

std::size_t elementwise_sigmoid_mul_launch_calls() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.elementwise_sigmoid_mul_launch_calls;
}

uint64_t elementwise_last_element_count() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.elementwise_last_element_count;
}

void set_rmsnorm_launch_status(const hipError_t status) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  state.rmsnorm_launch_status = status;
}

std::size_t rmsnorm_launch_calls() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.rmsnorm_launch_calls;
}

uint32_t rmsnorm_last_normalized_size() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.rmsnorm_last_normalized_size;
}

uint32_t rmsnorm_last_row_count() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.rmsnorm_last_row_count;
}

void set_matmul_launch_status(const hipError_t status) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  state.matmul_launch_status = status;
}

std::size_t argmax_launch_calls() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.argmax_launch_calls;
}

uint64_t argmax_last_m() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.argmax_last_m;
}

uint64_t argmax_last_v() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.argmax_last_v;
}

void set_argmax_launch_status(const hipError_t status) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  state.argmax_launch_status = status;
}

void set_event_record_status(const hipError_t status) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  state.event_record_status = status;
}

void set_event_create_gate(const bool enabled) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  state.event_create_gate = enabled;
  state.event_create_entered = false;
  state.condition.notify_all();
}

void wait_event_create_entered() {
  std::unique_lock<std::mutex> lock(state.mutex);
  state.condition.wait(lock, [] { return state.event_create_entered; });
}

void release_event_create_gate() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  state.event_create_gate = false;
  state.condition.notify_all();
}

void set_event_query_gate(const bool enabled) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  state.event_query_gate = enabled;
  state.event_query_entered = false;
  state.condition.notify_all();
}

void set_completion_pending(const bool enabled) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  state.completion_pending = enabled;
}

void wait_event_query_entered() {
  std::unique_lock<std::mutex> lock(state.mutex);
  state.condition.wait(lock, [] { return state.event_query_entered; });
}

void release_event_query_gate() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  state.event_query_gate = false;
  state.condition.notify_all();
}

std::size_t event_destroy_calls() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.event_destroy_calls;
}

std::size_t stream_destroy_calls() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.stream_destroy_calls;
}

std::size_t allocation_free_calls() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.allocation_free_calls;
}

std::size_t live_events() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.events.size();
}

std::size_t live_streams() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.streams.size();
}

std::size_t live_allocations() noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.allocations.size() + state.reservations.size() +
         state.vmm_handles.size();
}

} // namespace fake_hip

const char *hipGetErrorString(const hipError_t error) noexcept {
  switch (error) {
  case hipSuccess:
    return "success";
  case hipErrorInvalidValue:
    return "invalid value";
  case hipErrorNotReady:
    return "not ready";
  case hipErrorUnknown:
    return "unknown";
  }
  return "unknown";
}

hipError_t hipGetDeviceCount(int *const count) noexcept {
  if (count == nullptr) {
    return hipErrorInvalidValue;
  }
  *count = 1;
  return hipSuccess;
}

hipError_t hipGetDeviceProperties(hipDeviceProp_t *const properties,
                                  const unsigned int device) noexcept {
  if (properties == nullptr || device != 0U) {
    return hipErrorInvalidValue;
  }
  std::memset(properties, 0, sizeof(*properties));
  std::strncpy(properties->name, "fake-host-device",
               sizeof(properties->name) - 1U);
  std::strncpy(properties->gcnArchName, "gfx1201",
               sizeof(properties->gcnArchName) - 1U);
  properties->totalGlobalMem = static_cast<std::size_t>(16U) * 1024U * 1024U;
  properties->warpSize = 32;
  return hipSuccess;
}

hipError_t hipSetDevice(const int device) noexcept {
  return device == 0 ? hipSuccess : hipErrorInvalidValue;
}

hipError_t hipMemGetInfo(std::size_t *const available,
                         std::size_t *const total) noexcept {
  if (available == nullptr || total == nullptr) {
    return hipErrorInvalidValue;
  }
  *total = static_cast<std::size_t>(16U) * 1024U * 1024U;
  *available = static_cast<std::size_t>(12U) * 1024U * 1024U;
  return hipSuccess;
}

hipError_t hipMemGetAllocationGranularity(
    std::size_t *const granularity,
    const hipMemAllocationProp *const properties,
    const hipMemAllocationGranularity_flags option) noexcept {
  if (granularity == nullptr || properties == nullptr ||
      properties->type != hipMemAllocationTypePinned ||
      properties->location.type != hipMemLocationTypeDevice ||
      properties->location.id != 0) {
    return hipErrorInvalidValue;
  }
  *granularity = option == hipMemAllocationGranularityMinimum
                     ? std::size_t{4096}
                     : std::size_t{2} * 1024U * 1024U;
  return hipSuccess;
}

hipError_t hipMemAddressReserve(void **const pointer, const std::size_t size,
                                const std::size_t /*alignment*/,
                                void *const /*requested*/,
                                const unsigned long long /*flags*/) noexcept {
  if (pointer == nullptr || size == 0U) {
    return hipErrorInvalidValue;
  }
  *pointer = std::calloc(1U, size);
  if (*pointer == nullptr) {
    return hipErrorUnknown;
  }
  std::lock_guard<std::mutex> lock(state.mutex);
  state.reservations.insert(*pointer);
  return hipSuccess;
}

hipError_t hipMemAddressFree(void *const pointer,
                             const std::size_t /*size*/) noexcept {
  if (pointer == nullptr) {
    return hipErrorInvalidValue;
  }
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.reservations.erase(pointer) == 0U) {
    return hipErrorInvalidValue;
  }
  ++state.allocation_free_calls;
  std::free(pointer);
  return hipSuccess;
}

hipError_t hipMemCreate(hipMemGenericAllocationHandle_t *const handle,
                        const std::size_t size,
                        const hipMemAllocationProp *const properties,
                        const unsigned long long /*flags*/) noexcept {
  if (handle == nullptr || size == 0U || properties == nullptr ||
      properties->location.type != hipMemLocationTypeDevice) {
    return hipErrorInvalidValue;
  }
  *handle = new (std::nothrow) FakeHipMemHandle{size};
  if (*handle == nullptr) {
    return hipErrorUnknown;
  }
  std::lock_guard<std::mutex> lock(state.mutex);
  state.vmm_handles.insert(*handle);
  return hipSuccess;
}

hipError_t hipMemMap(void *const pointer, const std::size_t size,
                     const std::size_t offset,
                     const hipMemGenericAllocationHandle_t handle,
                     const unsigned long long /*flags*/) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return pointer == nullptr || size == 0U || offset != 0U ||
                 state.vmm_handles.count(handle) == 0U
             ? hipErrorInvalidValue
             : hipSuccess;
}

hipError_t hipMemSetAccess(void *const pointer, const std::size_t size,
                           const hipMemAccessDesc *const descriptors,
                           const std::size_t count) noexcept {
  return pointer == nullptr || size == 0U || descriptors == nullptr ||
                 count != 1U ||
                 descriptors->flags != hipMemAccessFlagsProtReadWrite
             ? hipErrorInvalidValue
             : hipSuccess;
}

hipError_t hipMemUnmap(void *const pointer, const std::size_t size) noexcept {
  return pointer == nullptr || size == 0U ? hipErrorInvalidValue : hipSuccess;
}

hipError_t
hipMemRelease(const hipMemGenericAllocationHandle_t handle) noexcept {
  if (handle == nullptr) {
    return hipErrorInvalidValue;
  }
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.vmm_handles.erase(handle) == 0U) {
    return hipErrorInvalidValue;
  }
  delete handle;
  return hipSuccess;
}

hipError_t hipStreamCreateWithFlags(hipStream_t *const stream,
                                    const unsigned int /*flags*/) noexcept {
  if (stream == nullptr) {
    return hipErrorInvalidValue;
  }
  *stream = new (std::nothrow) FakeHipStream;
  if (*stream == nullptr) {
    return hipErrorUnknown;
  }
  std::lock_guard<std::mutex> lock(state.mutex);
  state.streams.insert(*stream);
  return hipSuccess;
}

hipError_t hipStreamDestroy(const hipStream_t stream) noexcept {
  if (stream == nullptr) {
    return hipErrorInvalidValue;
  }
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.streams.erase(stream) == 0U) {
    return hipErrorInvalidValue;
  }
  ++state.stream_destroy_calls;
  delete stream;
  return hipSuccess;
}

hipError_t hipStreamSynchronize(const hipStream_t stream) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.streams.count(stream) == 0U ? hipErrorInvalidValue : hipSuccess;
}

hipError_t hipMalloc(void **const pointer, const std::size_t size) noexcept {
  if (pointer == nullptr || size == 0U) {
    return hipErrorInvalidValue;
  }
  /* Large public-buffer sizes are metadata-only in this fake runtime.  Keep
   * the allocation bounded so row-count overflow tests can exercise ABI
   * validation without touching or emulating tensor data. */
  const std::size_t allocation_size =
      size > (std::size_t{1} << 32U) ? std::size_t{1} : size;
  *pointer = std::malloc(allocation_size);
  if (*pointer == nullptr) {
    return hipErrorUnknown;
  }
  std::lock_guard<std::mutex> lock(state.mutex);
  state.allocations.insert(*pointer);
  return hipSuccess;
}

hipError_t hipFree(void *const pointer) noexcept {
  if (pointer == nullptr) {
    return hipErrorInvalidValue;
  }
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.allocations.erase(pointer) == 0U) {
    return hipErrorInvalidValue;
  }
  ++state.allocation_free_calls;
  std::free(pointer);
  return hipSuccess;
}

hipError_t hipEventCreateWithFlags(hipEvent_t *const event,
                                   const unsigned int /*flags*/) noexcept {
  if (event == nullptr) {
    return hipErrorInvalidValue;
  }
  *event = new (std::nothrow) FakeHipEvent;
  if (*event == nullptr) {
    return hipErrorUnknown;
  }
  std::unique_lock<std::mutex> lock(state.mutex);
  state.events.insert(*event);
  if (state.event_create_gate) {
    state.event_create_entered = true;
    state.condition.notify_all();
    state.condition.wait(lock, [] { return !state.event_create_gate; });
  }
  return hipSuccess;
}

hipError_t hipEventDestroy(const hipEvent_t event) noexcept {
  if (event == nullptr) {
    return hipErrorInvalidValue;
  }
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.events.erase(event) == 0U) {
    return hipErrorInvalidValue;
  }
  ++state.event_destroy_calls;
  delete event;
  return hipSuccess;
}

hipError_t hipEventRecord(const hipEvent_t event,
                          const hipStream_t stream) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.event_record_status != hipSuccess) {
    return state.event_record_status;
  }
  if (state.events.count(event) == 0U || state.streams.count(stream) == 0U) {
    return hipErrorInvalidValue;
  }
  event->recorded = true;
  return hipSuccess;
}

hipError_t hipEventElapsedTime(float *const milliseconds,
                               const hipEvent_t start,
                               const hipEvent_t end) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  if (milliseconds == nullptr || state.events.count(start) == 0U ||
      state.events.count(end) == 0U || !start->recorded || !end->recorded) {
    return hipErrorInvalidValue;
  }
  *milliseconds = 0.001F;
  return hipSuccess;
}

hipError_t hipEventQuery(const hipEvent_t event) noexcept {
  std::unique_lock<std::mutex> lock(state.mutex);
  if (state.events.count(event) == 0U) {
    return hipErrorInvalidValue;
  }
  if (state.event_query_gate) {
    state.event_query_entered = true;
    state.condition.notify_all();
    state.condition.wait(lock, [] { return !state.event_query_gate; });
  }
  if (state.completion_pending) {
    return hipErrorNotReady;
  }
  return event->recorded ? hipSuccess : hipErrorNotReady;
}

hipError_t hipMemcpyAsync(void *const destination, const void *const source,
                          const std::size_t size, const hipMemcpyKind kind,
                          const hipStream_t stream) noexcept {
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.streams.count(stream) == 0U || destination == nullptr ||
      source == nullptr ||
      (kind != hipMemcpyHostToDevice && kind != hipMemcpyDeviceToHost)) {
    return hipErrorInvalidValue;
  }
  std::memcpy(destination, source, size);
  return hipSuccess;
}
