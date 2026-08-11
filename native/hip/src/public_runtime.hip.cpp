#include "argmax_api.hpp"
#include "argmax_kernel_internal.hpp"
#include "attention_preprocess_api.hpp"
#include "attention_preprocess_kernel_internal.hpp"
#include "causal_attention_api.hpp"
#include "causal_attention_kernel_internal.hpp"
#include "elementwise_api.hpp"
#include "elementwise_kernel_internal.hpp"
#include "embedding_api.hpp"
#include "embedding_kernel_internal.hpp"
#include "evidence_abi.h"
#include "kv_state_api.hpp"
#include "kv_state_kernel_internal.hpp"
#include "linear_attention_api.hpp"
#include "linear_attention_kernel_internal.hpp"
#include "matmul_api.hpp"
#include "matmul_kernel_internal.hpp"
#include "public_runtime_internal.hpp"
#include "rmsnorm_api.hpp"
#include "rmsnorm_kernel_internal.hpp"

#include <hip/hip_runtime.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
namespace sllm_rmsnorm_kernel {
hipError_t launch(const uint16_t *const activation,
                  const uint16_t *const raw_scale, uint16_t *const output,
                  const uint32_t normalized_size, const uint32_t row_count,
                  const float epsilon, const hipStream_t stream) noexcept {
  return fake_hip::rmsnorm_launch(activation, raw_scale, output,
                                  normalized_size, row_count, epsilon, stream);
}
} // namespace sllm_rmsnorm_kernel

namespace sllm_elementwise_kernel {
hipError_t launch_copy(const uint16_t *const input, uint16_t *const output,
                       const uint64_t element_count,
                       const hipStream_t stream) noexcept {
  return fake_hip::elementwise_copy_launch(input, output, element_count,
                                           stream);
}

hipError_t launch_add(const uint16_t *const input0,
                      const uint16_t *const input1, uint16_t *const output,
                      const uint64_t element_count,
                      const hipStream_t stream) noexcept {
  return fake_hip::elementwise_add_launch(input0, input1, output, element_count,
                                          stream);
}

hipError_t launch_silu_mul(const uint16_t *const gate, const uint16_t *const up,
                           uint16_t *const output, const uint64_t element_count,
                           const hipStream_t stream) noexcept {
  return fake_hip::elementwise_silu_mul_launch(gate, up, output, element_count,
                                               stream);
}

hipError_t launch_sigmoid_mul(const uint16_t *const gate,
                              const uint16_t *const attention_value,
                              uint16_t *const output,
                              const uint64_t element_count,
                              const hipStream_t stream) noexcept {
  return fake_hip::elementwise_sigmoid_mul_launch(gate, attention_value, output,
                                                  element_count, stream);
}
} // namespace sllm_elementwise_kernel

namespace sllm_embedding_kernel {
hipError_t launch_gather(const uint16_t *const weight,
                         const int32_t *const token_ids, uint16_t *const output,
                         const uint64_t token_count, const uint64_t hidden_size,
                         const hipStream_t stream) noexcept {
  return fake_hip::embedding_gather_launch(weight, token_ids, output,
                                           token_count, hidden_size, stream);
}
} // namespace sllm_embedding_kernel

namespace sllm_matmul_kernel {
hipError_t launch(const uint16_t *const activation,
                  const uint16_t *const weight, uint16_t *const output,
                  const uint64_t m, const uint64_t k, const uint64_t n,
                  const hipStream_t stream) noexcept {
  return fake_hip::matmul_launch(activation, weight, output, m, k, n, stream);
}
} // namespace sllm_matmul_kernel

namespace sllm_argmax_kernel {
hipError_t launch(const uint16_t *const logits, int32_t *const output,
                  const uint64_t m, const uint64_t v,
                  const hipStream_t stream) noexcept {
  return fake_hip::argmax_launch(logits, output, m, v, stream);
}
} // namespace sllm_argmax_kernel

namespace sllm_attention_preprocess_kernel {
hipError_t launch(const uint16_t *const packed_q_gate, const uint16_t *const k,
                  const uint16_t *const q_raw_scale,
                  const uint16_t *const k_raw_scale,
                  const int32_t *const positions, uint16_t *const q_output,
                  uint16_t *const gate_output, uint16_t *const k_output,
                  const uint32_t m, const hipStream_t stream) noexcept {
  return fake_hip::attention_preprocess_launch(
      packed_q_gate, k, q_raw_scale, k_raw_scale, positions, q_output,
      gate_output, k_output, m, stream);
}
} // namespace sllm_attention_preprocess_kernel

namespace sllm_kv_state_kernel {
hipError_t launch(const uint16_t *const key_input,
                  const uint16_t *const value_input, uint16_t *const key_output,
                  uint16_t *const value_output, const uint32_t token_count,
                  const uint64_t capacity_tokens, const uint64_t start_position,
                  const hipStream_t stream) noexcept {
  return fake_hip::kv_state_append_launch(
      key_input, value_input, key_output, value_output, token_count,
      capacity_tokens, start_position, stream);
}
} // namespace sllm_kv_state_kernel

namespace sllm_causal_attention_kernel {
hipError_t launch(const uint16_t *const query, const uint16_t *const key,
                  const uint16_t *const value, uint16_t *const output,
                  const uint32_t query_count, const uint64_t capacity_tokens,
                  const uint64_t start_position,
                  const uint64_t committed_kv_length,
                  const hipStream_t stream) noexcept {
  return fake_hip::causal_attention_launch(
      query, key, value, output, query_count, capacity_tokens, start_position,
      committed_kv_length, stream);
}
} // namespace sllm_causal_attention_kernel

namespace sllm_linear_attention_kernel {
hipError_t launch_convolution(const uint16_t *const, const uint16_t *const,
                              const uint16_t *const, uint16_t *const,
                              uint16_t *const, const uint32_t,
                              const hipStream_t) noexcept {
  return hipSuccess;
}

hipError_t launch_recurrent(const uint16_t *const, const uint16_t *const,
                            const uint16_t *const, const uint16_t *const,
                            const float *const, const uint16_t *const,
                            const float *const, const float *const,
                            float *const, uint16_t *const, const uint32_t,
                            const hipStream_t) noexcept {
  return hipSuccess;
}
} // namespace sllm_linear_attention_kernel
#endif

namespace {

enum class HandleKind : uint32_t {
  Context,
  Queue,
  Buffer,
  Event,
  Completion,
  RmsNormPlan,
  ElementwisePlan,
  EmbeddingPlan,
  MatmulPlan,
  ArgmaxPlan,
  AttentionPreprocessPlan,
  KvState,
  KvView,
  LinearAttentionState,
};

struct QuarantineNode {
  HandleKind kind;
  QuarantineNode *next;
  bool poison_owned;

  explicit QuarantineNode(const HandleKind kind_value) noexcept
      : kind(kind_value), next(nullptr), poison_owned(false) {}
};

struct Context final : QuarantineNode {
  uint32_t device_index;
  sllm_public_runtime::AccountingState accounting;
  std::mutex accounting_mutex;
  bool release_active;
  std::atomic<bool> poisoned;
  uint64_t next_dispatch_id;

  explicit Context(const uint32_t device)
      : QuarantineNode(HandleKind::Context), device_index(device), accounting(),
        accounting_mutex(), release_active(false), poisoned(false),
        next_dispatch_id(1U) {}
};

struct Queue final : QuarantineNode {
  Context *context;
  hipStream_t stream;
  sllm_public_runtime::AccountingState accounting;
  bool release_active;

  Queue(Context *const context_value, const hipStream_t stream_value)
      : QuarantineNode(HandleKind::Queue), context(context_value),
        stream(stream_value), accounting(), release_active(false) {}
};

struct Buffer final : QuarantineNode {
  Context *context;
  void *device_pointer;
  uint64_t size_bytes;
  sllm_public_runtime::AccountingState accounting;
  bool release_active;

  Buffer(Context *const context_value, void *const pointer, const uint64_t size)
      : QuarantineNode(HandleKind::Buffer), context(context_value),
        device_pointer(pointer), size_bytes(size), accounting(),
        release_active(false) {}
};

struct KvState final : QuarantineNode {
  Context *context;
  uint64_t context_identity;
  uint64_t state_identity;
  uint64_t session_id;
  uint32_t layer_id;
  uint64_t capacity_tokens;
  Buffer *key_buffer;
  Buffer *value_buffer;
  sllm_public_runtime::AccountingState accounting;
  uint64_t published_length;
  uint64_t generation;
  uint64_t transition_token;
  uint64_t transition_start;
  uint64_t transition_count;
  uint64_t transition_end;
  bool commit_allowed;
  uint64_t view_count;
  bool release_active;

  KvState(Context *const context_value, const uint64_t session_value,
          const uint32_t layer_value, const uint64_t capacity_value,
          Buffer *const key_value, Buffer *const value_value,
          const uint64_t context_token)
      : QuarantineNode(HandleKind::KvState), context(context_value),
        context_identity(context_token), state_identity(0U),
        session_id(session_value), layer_id(layer_value),
        capacity_tokens(capacity_value), key_buffer(key_value),
        value_buffer(value_value), accounting(), published_length(0U),
        generation(0U), transition_token(0U), transition_start(0U),
        transition_count(0U), transition_end(0U), commit_allowed(false),
        view_count(0U), release_active(false) {}
};

struct KvView final : QuarantineNode {
  Context *context;
  KvState *state;
  uint64_t session_id;
  uint32_t layer_id;
  uint64_t capacity_tokens;
  uint64_t observed_length;
  uint64_t generation;
  uint64_t context_identity;
  uint64_t state_identity;
  bool readback_active;
  bool release_active;

  KvView(Context *const context_value, KvState *const state_value,
         const uint64_t session_value, const uint32_t layer_value,
         const uint64_t capacity_value, const uint64_t length_value,
         const uint64_t generation_value, const uint64_t context_token,
         const uint64_t state_token)
      : QuarantineNode(HandleKind::KvView), context(context_value),
        state(state_value), session_id(session_value), layer_id(layer_value),
        capacity_tokens(capacity_value), observed_length(length_value),
        generation(generation_value), context_identity(context_token),
        state_identity(state_token), readback_active(false),
        release_active(false) {}
};

struct LinearAttentionState final : QuarantineNode {
  Context *context;
  uint64_t context_identity;
  uint64_t state_identity;
  uint64_t session_id;
  uint32_t layer_id;
  uint64_t capacity_tokens;
  std::array<void *, 2> conv_state;
  std::array<void *, 2> recurrent_state;
  void *scratch;
  uint64_t scratch_bytes;
  sllm_public_runtime::AccountingState accounting;
  uint64_t published_length;
  uint64_t generation;
  uint32_t active_slot;
  uint64_t transition_token;
  uint64_t transition_start;
  uint64_t transition_count;
  uint64_t transition_end;
  bool commit_allowed;
  bool release_active;

  LinearAttentionState(Context *const context_value,
                       const uint64_t session_value, const uint32_t layer_value,
                       const uint64_t capacity_value,
                       const std::array<void *, 2> conv_value,
                       const std::array<void *, 2> recurrent_value,
                       const uint64_t context_token)
      : QuarantineNode(HandleKind::LinearAttentionState),
        context(context_value), context_identity(context_token),
        state_identity(0U), session_id(session_value), layer_id(layer_value),
        capacity_tokens(capacity_value), conv_state(conv_value),
        recurrent_state(recurrent_value), scratch(nullptr), scratch_bytes(0U),
        accounting(), published_length(0U), generation(0U), active_slot(0U),
        transition_token(0U), transition_start(0U), transition_count(0U),
        transition_end(0U), commit_allowed(false), release_active(false) {}
};

struct Event final : QuarantineNode {
  Context *context;
  hipEvent_t event;
  bool release_active;

  Event(Context *const context_value, const hipEvent_t event_value)
      : QuarantineNode(HandleKind::Event), context(context_value),
        event(event_value), release_active(false) {}
};

struct RmsNormPlan;
struct ElementwisePlan;
struct AttentionPreprocessPlan;
using AttentionBuffers = std::array<struct Buffer *, 8>;
using CausalAttentionBuffers = std::array<struct Buffer *, 2>;
using LinearAttentionBuffers = std::array<struct Buffer *, 9>;
struct ArgmaxCompletionTag final {};

struct Completion final : QuarantineNode {
  Context *context;
  Queue *queue;
  Buffer *buffer;
  hipEvent_t event;
  hipEvent_t timing_start_event;
  uint64_t timing_elapsed_ns;
  bool timing_valid;
  uint64_t transfer_size_bytes;
  bool d2h;
  bool terminal;
  bool success;
  bool safe_to_release;
  bool references_released;
  bool context_child_released;
  bool event_destroyed;
  bool release_active;
  bool lifetime_guard_reserved;
  bool orphaned;
  bool reference_accounting_failed;
  bool active_release_attempted;
  hipError_t failure_status;
  std::atomic<uint64_t> api_pins;
  bool wait_active;
  std::mutex state_mutex;
  sllm_public_runtime::CompletionSafetyState safety;
  std::vector<uint8_t> host_storage;
  bool rmsnorm;
  RmsNormPlan *rmsnorm_plan;
  Buffer *rmsnorm_activation;
  Buffer *rmsnorm_raw_scale;
  Buffer *rmsnorm_output;
  bool elementwise;
  ElementwisePlan *elementwise_plan;
  Buffer *elementwise_input0;
  Buffer *elementwise_input1;
  Buffer *elementwise_output;
  bool matmul;
  ElementwisePlan *matmul_plan;
  Buffer *matmul_activation;
  Buffer *matmul_weight;
  Buffer *matmul_output;
  bool argmax;
  ElementwisePlan *argmax_plan;
  Buffer *argmax_logits;
  Buffer *argmax_output;
  bool attention_preprocess;
  AttentionPreprocessPlan *attention_preprocess_plan;
  AttentionBuffers attention_buffers;
  bool kv_state_append;
  KvState *kv_state;
  Buffer *kv_key_input;
  Buffer *kv_value_input;
  Buffer *kv_key_buffer;
  Buffer *kv_value_buffer;
  uint64_t kv_append_token;
  uint64_t kv_append_start;
  uint64_t kv_append_count;
  uint64_t kv_append_end;
  bool causal_attention;
  KvState *causal_attention_state;
  CausalAttentionBuffers causal_attention_buffers;
  bool linear_attention;
  LinearAttentionState *linear_attention_state;
  LinearAttentionBuffers linear_attention_buffers;
  uint64_t linear_attention_token;
  uint64_t linear_attention_start;
  uint64_t linear_attention_count;
  uint64_t linear_attention_end;

  Completion(Context *const context_value, Queue *const queue_value,
             Buffer *const buffer_value, const uint64_t transfer_size,
             const bool d2h_value, std::vector<uint8_t> &&storage,
             RmsNormPlan *const rmsnorm_plan_value = nullptr,
             Buffer *const rmsnorm_activation_value = nullptr,
             Buffer *const rmsnorm_raw_scale_value = nullptr,
             Buffer *const rmsnorm_output_value = nullptr,
             ElementwisePlan *const elementwise_plan_value = nullptr,
             Buffer *const elementwise_input0_value = nullptr,
             Buffer *const elementwise_input1_value = nullptr,
             Buffer *const elementwise_output_value = nullptr,
             ElementwisePlan *const matmul_plan_value = nullptr,
             Buffer *const matmul_activation_value = nullptr,
             Buffer *const matmul_weight_value = nullptr,
             Buffer *const matmul_output_value = nullptr,
             AttentionPreprocessPlan *const attention_plan_value = nullptr,
             const AttentionBuffers attention_buffers_value = {},
             KvState *const kv_state_value = nullptr,
             Buffer *const kv_key_input_value = nullptr,
             Buffer *const kv_value_input_value = nullptr,
             Buffer *const kv_key_buffer_value = nullptr,
             Buffer *const kv_value_buffer_value = nullptr,
             const uint64_t kv_append_token_value = 0U,
             const uint64_t kv_append_start_value = 0U,
             const uint64_t kv_append_count_value = 0U,
             const uint64_t kv_append_end_value = 0U,
             KvState *const causal_attention_state_value = nullptr,
             const CausalAttentionBuffers causal_attention_buffers_value = {},
             ElementwisePlan *const argmax_plan_value = nullptr,
             Buffer *const argmax_logits_value = nullptr,
             Buffer *const argmax_output_value = nullptr)
      : QuarantineNode(HandleKind::Completion), context(context_value),
        queue(queue_value), buffer(buffer_value), event(nullptr),
        timing_start_event(nullptr), timing_elapsed_ns(0U), timing_valid(false),
        transfer_size_bytes(transfer_size), d2h(d2h_value), terminal(false),
        success(false), safe_to_release(false), references_released(false),
        context_child_released(false), event_destroyed(false),
        release_active(false), lifetime_guard_reserved(true), orphaned(false),
        reference_accounting_failed(false), active_release_attempted(false),
        failure_status(hipErrorInvalidValue), api_pins(0U), wait_active(false),
        state_mutex(), safety(), host_storage(std::move(storage)),
        rmsnorm(rmsnorm_plan_value != nullptr),
        rmsnorm_plan(rmsnorm_plan_value),
        rmsnorm_activation(rmsnorm_activation_value),
        rmsnorm_raw_scale(rmsnorm_raw_scale_value),
        rmsnorm_output(rmsnorm_output_value),
        elementwise(elementwise_plan_value != nullptr),
        elementwise_plan(elementwise_plan_value),
        elementwise_input0(elementwise_input0_value),
        elementwise_input1(elementwise_input1_value),
        elementwise_output(elementwise_output_value),
        matmul(matmul_plan_value != nullptr), matmul_plan(matmul_plan_value),
        matmul_activation(matmul_activation_value),
        matmul_weight(matmul_weight_value), matmul_output(matmul_output_value),
        argmax(argmax_plan_value != nullptr), argmax_plan(argmax_plan_value),
        argmax_logits(argmax_logits_value), argmax_output(argmax_output_value),
        attention_preprocess(attention_plan_value != nullptr),
        attention_preprocess_plan(attention_plan_value),
        attention_buffers(attention_buffers_value),
        kv_state_append(kv_state_value != nullptr), kv_state(kv_state_value),
        kv_key_input(kv_key_input_value), kv_value_input(kv_value_input_value),
        kv_key_buffer(kv_key_buffer_value),
        kv_value_buffer(kv_value_buffer_value),
        kv_append_token(kv_append_token_value),
        kv_append_start(kv_append_start_value),
        kv_append_count(kv_append_count_value),
        kv_append_end(kv_append_end_value),
        causal_attention(causal_attention_state_value != nullptr),
        causal_attention_state(causal_attention_state_value),
        causal_attention_buffers(causal_attention_buffers_value),
        linear_attention(false), linear_attention_state(nullptr),
        linear_attention_buffers(), linear_attention_token(0U),
        linear_attention_start(0U), linear_attention_count(0U),
        linear_attention_end(0U) {}

  Completion(Context *const context_value, Queue *const queue_value,
             Buffer *const buffer_value, ElementwisePlan *const plan_value,
             Buffer *const logits_value, Buffer *const output_value,
             ArgmaxCompletionTag)
      : Completion(context_value, queue_value, buffer_value, 0U, false,
                   std::vector<uint8_t>{}, nullptr, nullptr, nullptr, nullptr,
                   nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                   nullptr, nullptr, nullptr, {}, nullptr, nullptr, nullptr,
                   nullptr, nullptr, 0U, 0U, 0U, 0U, nullptr, {}, plan_value,
                   logits_value, output_value) {}
};

/* The plan stores copied descriptor metadata and its three retained buffer
 * identities.  Execution adds a single in-flight reservation to this graph. */
struct RmsNormPlan final : QuarantineNode {
  Context *context;
  Buffer *activation;
  Buffer *raw_scale;
  Buffer *output;
  sllm_rmsnorm::DescriptorMetadata metadata;
  bool release_active;
  bool in_flight;

  RmsNormPlan(Context *const context_value, Buffer *const activation_value,
              Buffer *const raw_scale_value, Buffer *const output_value,
              const sllm_rmsnorm::DescriptorMetadata &metadata_value)
      : QuarantineNode(HandleKind::RmsNormPlan), context(context_value),
        activation(activation_value), raw_scale(raw_scale_value),
        output(output_value), metadata(metadata_value), release_active(false),
        in_flight(false) {}
};

struct ElementwisePlan final : QuarantineNode {
  Context *context;
  Buffer *input0;
  Buffer *input1;
  Buffer *output;
  sllm_elementwise::DescriptorMetadata metadata;
  sllm_embedding::DescriptorMetadata embedding_metadata;
  sllm_matmul::DescriptorMetadata matmul_metadata;
  sllm_argmax::DescriptorMetadata argmax_metadata;
  bool embedding;
  bool matmul;
  bool argmax;
  bool release_active;
  bool in_flight;

  ElementwisePlan(Context *const context_value, Buffer *const input0_value,
                  Buffer *const input1_value, Buffer *const output_value,
                  const sllm_elementwise::DescriptorMetadata &metadata_value)
      : QuarantineNode(HandleKind::ElementwisePlan), context(context_value),
        input0(input0_value), input1(input1_value), output(output_value),
        metadata(metadata_value), embedding_metadata(), matmul_metadata(),
        argmax_metadata(), embedding(false), matmul(false), argmax(false),
        release_active(false), in_flight(false) {}

  ElementwisePlan(Context *const context_value, Buffer *const weight_value,
                  Buffer *const token_ids_value, Buffer *const output_value,
                  const sllm_embedding::DescriptorMetadata &metadata_value)
      : QuarantineNode(HandleKind::EmbeddingPlan), context(context_value),
        input0(weight_value), input1(token_ids_value), output(output_value),
        metadata(), embedding_metadata(metadata_value), matmul_metadata(),
        argmax_metadata(), embedding(true), matmul(false), argmax(false),
        release_active(false), in_flight(false) {}

  ElementwisePlan(Context *const context_value, Buffer *const activation_value,
                  Buffer *const weight_value, Buffer *const output_value,
                  const sllm_matmul::DescriptorMetadata &metadata_value)
      : QuarantineNode(HandleKind::MatmulPlan), context(context_value),
        input0(activation_value), input1(weight_value), output(output_value),
        metadata(), embedding_metadata(), matmul_metadata(metadata_value),
        argmax_metadata(), embedding(false), matmul(true), argmax(false),
        release_active(false), in_flight(false) {}

  ElementwisePlan(Context *const context_value, Buffer *const logits_value,
                  Buffer *const output_value,
                  const sllm_argmax::DescriptorMetadata &metadata_value)
      : QuarantineNode(HandleKind::ArgmaxPlan), context(context_value),
        input0(logits_value), input1(nullptr), output(output_value), metadata(),
        embedding_metadata(), matmul_metadata(),
        argmax_metadata(metadata_value), embedding(false), matmul(false),
        argmax(true), release_active(false), in_flight(false) {}
};

struct AttentionPreprocessPlan final : QuarantineNode {
  Context *context;
  AttentionBuffers buffers;
  sllm_attention_preprocess::DescriptorMetadata metadata;
  bool release_active;
  bool in_flight;

  AttentionPreprocessPlan(
      Context *const context_value, const AttentionBuffers buffers_value,
      const sllm_attention_preprocess::DescriptorMetadata &metadata_value)
      : QuarantineNode(HandleKind::AttentionPreprocessPlan),
        context(context_value), buffers(buffers_value),
        metadata(metadata_value), release_active(false), in_flight(false) {}
};

struct RegistryEntry {
  HandleKind kind;
  void *state;
};

std::mutex registry_mutex;
std::unordered_map<uintptr_t, RegistryEntry> registry;
sllm_public_runtime::MonotonicTokenSource token_source;

/* Native errors after a HIP destruction call are ownership-ambiguous unless
 * the pinned HIP contract returned success.  These process-lifetime owners
 * therefore deliberately never retry such resources or destroy them again.
 * They retain the state and its dependency pointers until process teardown. */
class PoisonOwner final {
public:
  void retain(QuarantineNode *const node) noexcept {
    if (node == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (node->poison_owned) {
      return;
    }
    node->poison_owned = true;
    node->next = head_;
    head_ = node;
  }

  template <typename T> void retain(std::unique_ptr<T> &&owner) noexcept {
    static_assert(std::is_base_of_v<QuarantineNode, T>);
    T *const raw = owner.release();
    retain(static_cast<QuarantineNode *>(raw));
  }

  std::size_t size() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    std::size_t count = 0U;
    for (QuarantineNode *node = head_; node != nullptr; node = node->next) {
      ++count;
    }
    return count;
  }

private:
  mutable std::mutex mutex_;
  QuarantineNode *head_ = nullptr;
};

PoisonOwner poison_owner;

enum class OrphanKind : uint32_t { Stream, Allocation, Event };

struct OrphanRecord final {
  bool occupied = false;
  OrphanKind kind = OrphanKind::Stream;
  hipStream_t stream = nullptr;
  void *allocation = nullptr;
  hipEvent_t event = nullptr;
};

/* Partial-construction cleanup is an explicit process-lifetime ownership
 * transfer.  The owner is intentionally unbounded: the public status ABI has
 * no "orphan table full" status, so a fixed-capacity owner would either abort
 * a status-returning call or lose a raw HIP handle.  Records are never
 * reclaimed or retried; allocation failure is the only remaining process
 * resource failure and is handled by the C++ runtime rather than by a silent
 * ownership drop.  The context pointer is used only while the context is
 * live to poison its accounting and is not retained in the durable record. */
class OrphanOwner final {
public:
  void retain_stream(Context *const context, const hipStream_t stream) {
    retain(OrphanKind::Stream, context, stream, nullptr, nullptr);
  }

  void retain_allocation(Context *const context, void *const allocation) {
    retain(OrphanKind::Allocation, context, nullptr, allocation, nullptr);
  }

  void retain_event(Context *const context, const hipEvent_t event) {
    retain(OrphanKind::Event, context, nullptr, nullptr, event);
  }

  std::size_t size() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return records_.size();
  }

private:
  void retain(const OrphanKind kind, Context *const context,
              const hipStream_t stream, void *const allocation,
              const hipEvent_t event) noexcept {
    try {
      if (context != nullptr) {
        std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
        context->poisoned.store(true);
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        records_.retain(OrphanRecord{true, kind, stream, allocation, event});
      }
    } catch (...) {
      /* The ABI has no status for process-wide owner exhaustion.  Terminate
       * only for an actual process allocation/locking failure, before the
       * raw handle can be dropped; normal orphan growth is unbounded. */
      std::terminate();
    }
  }

  mutable std::mutex mutex_;
  sllm_public_runtime::DurableRecordOwner<OrphanRecord> records_;
};

OrphanOwner orphan_owner;

template <std::size_t Size>
bool is_first_attention_buffer(const std::array<Buffer *, Size> &buffers,
                               const std::size_t index) noexcept {
  for (std::size_t prior = 0U; prior != index; ++prior) {
    if (buffers[prior] == buffers[index]) {
      return false;
    }
  }
  return true;
}

template <std::size_t Size>
bool reserve_attention_prepared_plan(
    sllm_public_runtime::AccountingState &context,
    const std::array<Buffer *, Size> &buffers) noexcept {
  if (!sllm_public_runtime::AccountingState::can_increment(
          context.child_count) ||
      !sllm_public_runtime::AccountingState::can_increment(
          context.lifetime_guards)) {
    return false;
  }
  for (std::size_t index = 0U; index != buffers.size(); ++index) {
    Buffer *const buffer = buffers[index];
    if (!is_first_attention_buffer(buffers, index)) {
      continue;
    }
    if (buffer == nullptr ||
        !sllm_public_runtime::AccountingState::can_increment(
            buffer->accounting.child_count)) {
      return false;
    }
  }
  ++context.child_count;
  ++context.lifetime_guards;
  for (std::size_t index = 0U; index != buffers.size(); ++index) {
    if (is_first_attention_buffer(buffers, index)) {
      ++buffers[index]->accounting.child_count;
    }
  }
  return true;
}

template <std::size_t Size>
bool release_attention_prepared_plan(
    sllm_public_runtime::AccountingState &context,
    const std::array<Buffer *, Size> &buffers) noexcept {
  if (context.child_count == 0U || context.lifetime_guards == 0U) {
    return false;
  }
  for (std::size_t index = 0U; index != buffers.size(); ++index) {
    Buffer *const buffer = buffers[index];
    if (!is_first_attention_buffer(buffers, index)) {
      continue;
    }
    if (buffer == nullptr || buffer->accounting.child_count == 0U) {
      return false;
    }
  }
  --context.child_count;
  --context.lifetime_guards;
  for (std::size_t index = 0U; index != buffers.size(); ++index) {
    if (is_first_attention_buffer(buffers, index)) {
      --buffers[index]->accounting.child_count;
    }
  }
  return true;
}

template <std::size_t Size>
bool reserve_attention_submission(
    sllm_public_runtime::AccountingState &context,
    sllm_public_runtime::AccountingState &queue,
    const std::array<Buffer *, Size> &buffers) noexcept {
  if (!sllm_public_runtime::AccountingState::can_increment(
          queue.active_submissions) ||
      !sllm_public_runtime::AccountingState::can_increment(
          queue.completion_references) ||
      !sllm_public_runtime::AccountingState::can_increment(
          context.child_count) ||
      !sllm_public_runtime::AccountingState::can_increment(
          context.lifetime_guards)) {
    return false;
  }
  for (std::size_t index = 0U; index != buffers.size(); ++index) {
    Buffer *const buffer = buffers[index];
    if (!is_first_attention_buffer(buffers, index)) {
      continue;
    }
    if (buffer == nullptr ||
        !sllm_public_runtime::AccountingState::can_increment(
            buffer->accounting.active_submissions) ||
        !sllm_public_runtime::AccountingState::can_increment(
            buffer->accounting.completion_references)) {
      return false;
    }
  }
  ++queue.active_submissions;
  ++queue.completion_references;
  ++context.child_count;
  ++context.lifetime_guards;
  for (std::size_t index = 0U; index != buffers.size(); ++index) {
    if (is_first_attention_buffer(buffers, index)) {
      ++buffers[index]->accounting.active_submissions;
      ++buffers[index]->accounting.completion_references;
    }
  }
  return true;
}

template <std::size_t Size>
bool release_attention_active(
    sllm_public_runtime::AccountingState &queue,
    const std::array<Buffer *, Size> &buffers) noexcept {
  if (queue.active_submissions == 0U) {
    return false;
  }
  for (std::size_t index = 0U; index != buffers.size(); ++index) {
    Buffer *const buffer = buffers[index];
    if (!is_first_attention_buffer(buffers, index)) {
      continue;
    }
    if (buffer == nullptr || buffer->accounting.active_submissions == 0U) {
      return false;
    }
  }
  --queue.active_submissions;
  for (std::size_t index = 0U; index != buffers.size(); ++index) {
    if (is_first_attention_buffer(buffers, index)) {
      --buffers[index]->accounting.active_submissions;
    }
  }
  return true;
}

template <std::size_t Size>
bool rollback_attention_submission(
    sllm_public_runtime::AccountingState &context,
    sllm_public_runtime::AccountingState &queue,
    const std::array<Buffer *, Size> &buffers) noexcept {
  if (queue.active_submissions == 0U || queue.completion_references == 0U ||
      context.child_count == 0U || context.lifetime_guards == 0U) {
    return false;
  }
  for (std::size_t index = 0U; index != buffers.size(); ++index) {
    Buffer *const buffer = buffers[index];
    if (!is_first_attention_buffer(buffers, index)) {
      continue;
    }
    if (buffer == nullptr || buffer->accounting.active_submissions == 0U ||
        buffer->accounting.completion_references == 0U) {
      return false;
    }
  }
  --queue.active_submissions;
  --queue.completion_references;
  --context.child_count;
  --context.lifetime_guards;
  for (std::size_t index = 0U; index != buffers.size(); ++index) {
    if (is_first_attention_buffer(buffers, index)) {
      --buffers[index]->accounting.active_submissions;
      --buffers[index]->accounting.completion_references;
    }
  }
  return true;
}

template <std::size_t Size>
bool release_attention_completion(
    sllm_public_runtime::AccountingState &context,
    sllm_public_runtime::AccountingState &queue,
    const std::array<Buffer *, Size> &buffers) noexcept {
  if (queue.completion_references == 0U || context.child_count == 0U ||
      context.lifetime_guards == 0U) {
    return false;
  }
  for (std::size_t index = 0U; index != buffers.size(); ++index) {
    Buffer *const buffer = buffers[index];
    if (!is_first_attention_buffer(buffers, index)) {
      continue;
    }
    if (buffer == nullptr || buffer->accounting.completion_references == 0U) {
      return false;
    }
  }
  --queue.completion_references;
  --context.child_count;
  --context.lifetime_guards;
  for (std::size_t index = 0U; index != buffers.size(); ++index) {
    if (is_first_attention_buffer(buffers, index)) {
      --buffers[index]->accounting.completion_references;
    }
  }
  return true;
}

bool reserve_linear_attention_state(
    sllm_public_runtime::AccountingState &context) noexcept {
  if (!sllm_public_runtime::AccountingState::can_increment(
          context.child_count) ||
      !sllm_public_runtime::AccountingState::can_increment(
          context.lifetime_guards)) {
    return false;
  }
  ++context.child_count;
  ++context.lifetime_guards;
  return true;
}

bool release_linear_attention_state(
    sllm_public_runtime::AccountingState &context) noexcept {
  if (context.child_count == 0U || context.lifetime_guards == 0U) {
    return false;
  }
  --context.child_count;
  --context.lifetime_guards;
  return true;
}

bool reserve_linear_attention_submission(
    Context *const context, Queue *const queue,
    LinearAttentionState *const state,
    const LinearAttentionBuffers &buffers) noexcept {
  if (!sllm_public_runtime::AccountingState::can_increment(
          state->accounting.active_submissions) ||
      !sllm_public_runtime::AccountingState::can_increment(
          state->accounting.completion_references) ||
      !reserve_attention_submission(context->accounting, queue->accounting,
                                    buffers)) {
    return false;
  }
  ++state->accounting.active_submissions;
  ++state->accounting.completion_references;
  return true;
}

bool release_linear_attention_active(
    Queue *const queue, LinearAttentionState *const state,
    const LinearAttentionBuffers &buffers) noexcept {
  if (state->accounting.active_submissions == 0U ||
      !release_attention_active(queue->accounting, buffers)) {
    return false;
  }
  --state->accounting.active_submissions;
  return true;
}

bool rollback_linear_attention_submission(
    Context *const context, Queue *const queue,
    LinearAttentionState *const state,
    const LinearAttentionBuffers &buffers) noexcept {
  if (state->accounting.active_submissions == 0U ||
      state->accounting.completion_references == 0U ||
      !rollback_attention_submission(context->accounting, queue->accounting,
                                     buffers)) {
    return false;
  }
  --state->accounting.active_submissions;
  --state->accounting.completion_references;
  return true;
}

bool release_linear_attention_completion(
    Context *const context, Queue *const queue,
    LinearAttentionState *const state,
    const LinearAttentionBuffers &buffers) noexcept {
  if (state->accounting.completion_references == 0U ||
      !release_attention_completion(context->accounting, queue->accounting,
                                    buffers)) {
    return false;
  }
  --state->accounting.completion_references;
  return true;
}

bool release_causal_active(Queue *const queue, KvState *const state,
                           const CausalAttentionBuffers &buffers) noexcept {
  if (queue == nullptr || state == nullptr || buffers[0] == nullptr ||
      buffers[1] == nullptr) {
    return false;
  }
  return sllm_public_runtime::AccountingState::release_causal_active(
      queue->accounting, state->accounting, buffers[0]->accounting,
      buffers[1]->accounting);
}

bool rollback_causal_attention(Completion *const completion) noexcept {
  if (completion == nullptr || completion->context == nullptr ||
      completion->queue == nullptr ||
      completion->causal_attention_state == nullptr) {
    return false;
  }
  return sllm_public_runtime::AccountingState::rollback_causal_attention(
      completion->context->accounting, completion->queue->accounting,
      completion->causal_attention_state->accounting,
      completion->causal_attention_buffers[0]->accounting,
      completion->causal_attention_buffers[1]->accounting);
}

bool rollback_reserved_causal_attention(
    KvState *const state, Queue *const queue,
    const CausalAttentionBuffers &buffers,
    sllm_error_sink_t *const sink) noexcept {
  if (state == nullptr || queue == nullptr || buffers[0] == nullptr ||
      buffers[1] == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lock(state->context->accounting_mutex);
  if (sllm_public_runtime::AccountingState::rollback_causal_attention(
          state->context->accounting, queue->accounting, state->accounting,
          buffers[0]->accounting, buffers[1]->accounting)) {
    return true;
  }
  state->context->poisoned.store(true);
  (void)sllm_public_runtime::write_error(
      sink, SLLM_STATUS_INTERNAL_ERROR,
      "causal attention accounting rollback failed; context poisoned");
  return false;
}

bool release_causal_completion(Completion *const completion) noexcept {
  if (completion == nullptr || completion->context == nullptr ||
      completion->queue == nullptr ||
      completion->causal_attention_state == nullptr) {
    return false;
  }
  return sllm_public_runtime::AccountingState::release_causal_completion(
      completion->context->accounting, completion->queue->accounting,
      completion->causal_attention_state->accounting,
      completion->causal_attention_buffers[0]->accounting,
      completion->causal_attention_buffers[1]->accounting);
}

/* These host-test-only exception seams intentionally live in this production
 * translation unit rather than a classifier mock.  They exercise the real
 * post-reservation and post-registration unwind paths without changing the
 * public C ABI or the production HIP archive. */
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
std::atomic<uint32_t> rmsnorm_throw_after_reservation{0U};
std::atomic<uint32_t> rmsnorm_throw_after_registration{0U};

bool consume_test_exception(std::atomic<uint32_t> &counter) noexcept {
  uint32_t remaining = counter.load(std::memory_order_acquire);
  while (remaining != 0U) {
    if (counter.compare_exchange_weak(remaining, remaining - 1U,
                                      std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
      return true;
    }
  }
  return false;
}

void throw_after_rmsnorm_reservation_if_requested() {
  if (consume_test_exception(rmsnorm_throw_after_reservation)) {
    throw std::bad_alloc();
  }
}

void throw_after_rmsnorm_registration_if_requested() {
  if (consume_test_exception(rmsnorm_throw_after_registration)) {
    throw std::bad_alloc();
  }
}
#else
void throw_after_rmsnorm_reservation_if_requested() {}
void throw_after_rmsnorm_registration_if_requested() {}
#endif

/* Lock order is fixed for every path that needs more than one lock:
 * registry_mutex -> Completion::state_mutex -> Context::accounting_mutex.
 * Orphan transfer is the separate Context::accounting_mutex -> orphan-owner
 * mutex path and is never entered while registry_mutex or a completion state
 * mutex is held.
 * The registry lock only protects token lookup/state transitions.  No HIP
 * call, allocation that can block on the runtime, stream synchronization, or
 * object destruction requiring HIP is performed while registry_mutex is held.
 * Queue/Buffer/Context counters are never read or written outside the one
 * context accounting mutex.  Completion API pins keep a Completion alive
 * while its state and accounting locks are used. */

uintptr_t handle_key(const void *const raw) noexcept {
  return reinterpret_cast<uintptr_t>(raw);
}

template <typename T>
T *lookup(const void *const raw, const HandleKind kind) noexcept {
  const uintptr_t key = handle_key(raw);
  if (key == 0U) {
    return nullptr;
  }
  const auto found = registry.find(key);
  if (found == registry.end() || found->second.kind != kind) {
    return nullptr;
  }
  return static_cast<T *>(found->second.state);
}

uintptr_t register_handle(void *const state, const HandleKind kind) {
  if (sllm_public_runtime::FaultInjector::consume(
          sllm_public_runtime::FaultPoint::RegistryInsertionFailure)) {
    return 0U;
  }
  const uintptr_t token = token_source.issue();
  if (token == 0U) {
    return 0U;
  }
  const auto inserted = registry.emplace(token, RegistryEntry{kind, state});
  if (!inserted.second) {
    return 0U;
  }
  /* Exercise the real unordered_map emplacement path, then inject the
   * exception after the insertion has completed.  The local rollback keeps
   * register_handle's exception guarantee, while the caller reaches its real
   * catch path and performs guarded event cleanup before accounting rollback.
   * This point is consumed only by the production-TU host test; it is a
   * no-op in normal production execution because no test configures it. */
  if (sllm_public_runtime::FaultInjector::consume(
          sllm_public_runtime::FaultPoint::RegistryInsertionException)) {
    registry.erase(token);
    throw std::bad_alloc();
  }
  return token;
}

void unregister_handle(const void *const raw) noexcept {
  registry.erase(handle_key(raw));
}

void unregister_handle_token(const uintptr_t token) noexcept {
  registry.erase(token);
}

sllm_status_t hip_failure(sllm_error_sink_t *const sink, const hipError_t error,
                          const char *const operation) noexcept {
  const char *const detail = hipGetErrorString(error);
  char message[256] = {};
  const char *const suffix = detail == nullptr ? "unknown HIP error" : detail;
  const int written =
      std::snprintf(message, sizeof(message), "%s: %s", operation, suffix);
  if (written < 0) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
        "HIP runtime operation failed");
  }
  return sllm_public_runtime::write_error_n_bounded(
      sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR, message,
      static_cast<std::size_t>(written), sizeof(message) - 1U);
}

hipError_t destroy_event_with_fault_injection(const hipEvent_t event) noexcept {
  if (sllm_public_runtime::FaultInjector::consume(
          sllm_public_runtime::FaultPoint::EventDestroyError)) {
    return hipErrorUnknown;
  }
  return hipEventDestroy(event);
}

hipError_t
destroy_stream_with_fault_injection(const hipStream_t stream) noexcept {
  if (sllm_public_runtime::FaultInjector::consume(
          sllm_public_runtime::FaultPoint::StreamDestroyError)) {
    return hipErrorUnknown;
  }
  return hipStreamDestroy(stream);
}

hipError_t
free_allocation_with_fault_injection(void *const allocation) noexcept {
  if (sllm_public_runtime::FaultInjector::consume(
          sllm_public_runtime::FaultPoint::AllocationFreeError)) {
    return hipErrorUnknown;
  }
  return hipFree(allocation);
}

sllm_status_t
validate_backend_result(const sllm_backend_probe_result_t *const result,
                        sllm_error_sink_t *const sink) noexcept {
  const sllm_status_t status = sllm_public_runtime::validate_struct(
      result, sink, "backend probe result is null");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  if (result->reserved[0] != 0U || result->reserved[1] != 0U ||
      result->reserved[2] != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "backend probe reserved fields must be zero");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_context_probe_result(const sllm_context_probe_result_t *const result,
                              sllm_error_sink_t *const sink) noexcept {
  const sllm_status_t status = sllm_public_runtime::validate_struct(
      result, sink, "context probe result is null");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  if (result->reserved[0] != 0U || result->reserved[1] != 0U ||
      result->reserved[2] != 0U || result->reserved[3] != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "context probe reserved fields must be zero");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t validate_device_info(const sllm_device_info_t *const info,
                                   sllm_error_sink_t *const sink) noexcept {
  const sllm_status_t status = sllm_public_runtime::validate_struct(
      info, sink, "device info output is null");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  if (info->reserved0 != 0U || info->reserved[0] != 0U ||
      info->reserved[1] != 0U || info->reserved[2] != 0U ||
      info->reserved[3] != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "device info reserved fields must be zero");
  }
  return SLLM_STATUS_OK;
}

void initialize_device_info(sllm_device_info_t *const info) noexcept {
  const uint32_t struct_size = info->struct_size;
  const uint32_t abi_version = info->abi_version;
  std::memset(info, 0, sizeof(*info));
  info->struct_size = struct_size;
  info->abi_version = abi_version;
}

void initialize_completion_result(
    sllm_completion_result_t *const result) noexcept {
  const uint32_t struct_size = result->struct_size;
  const uint32_t abi_version = result->abi_version;
  std::memset(result, 0, sizeof(*result));
  result->struct_size = struct_size;
  result->abi_version = abi_version;
}

sllm_status_t
validate_rmsnorm_dispatch_info(const sllm_rmsnorm_dispatch_info_t *const info,
                               sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "RMSNorm dispatch info output is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, info, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_rmsnorm_dispatch_info_t)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "RMSNorm dispatch info has an unsupported struct size");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "RMSNorm dispatch info ABI version is unsupported");
  }
  if (info->info_version != SLLM_HIP_RMSNORM_DISPATCH_INFO_VERSION ||
      info->reserved[0] != 0U || info->reserved[1] != 0U ||
      info->reserved[2] != 0U || info->reserved[3] != 0U ||
      info->reserved[4] != 0U || info->reserved[5] != 0U ||
      info->reserved[6] != 0U || info->reserved[7] != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "RMSNorm dispatch info version or reserved fields are invalid");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t validate_elementwise_dispatch_info(
    const sllm_elementwise_dispatch_info_t *const info,
    sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "elementwise dispatch info output is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, info, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_elementwise_dispatch_info_t)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "elementwise dispatch info has an unsupported struct size");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "elementwise dispatch info ABI version is unsupported");
  }
  if (info->info_version != SLLM_HIP_ELEMENTWISE_DISPATCH_INFO_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "elementwise dispatch info version is unsupported");
  }
  for (const uint32_t value : info->reserved) {
    if (value != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_RESERVED_NONZERO,
          "elementwise dispatch info reserved fields must be zero");
    }
  }
  return SLLM_STATUS_OK;
}

sllm_status_t validate_embedding_dispatch_info(
    const sllm_embedding_dispatch_info_t *const info,
    sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "embedding dispatch info output is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, info, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_embedding_dispatch_info_t)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "embedding dispatch info has an unsupported struct size");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "embedding dispatch info ABI version is unsupported");
  }
  if (info->info_version != SLLM_HIP_EMBEDDING_DISPATCH_INFO_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "embedding dispatch info version is unsupported");
  }
  for (const uint32_t value : info->reserved) {
    if (value != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_RESERVED_NONZERO,
          "embedding dispatch info reserved fields must be zero");
    }
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_matmul_dispatch_info(const sllm_matmul_dispatch_info_t *const info,
                              sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "matmul dispatch info output is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, info, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_matmul_dispatch_info_t)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "matmul dispatch info has an unsupported struct size");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "matmul dispatch info ABI version is unsupported");
  }
  if (info->info_version != SLLM_HIP_MATMUL_DISPATCH_INFO_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "matmul dispatch info version is unsupported");
  }
  for (const uint32_t value : info->reserved) {
    if (value != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_RESERVED_NONZERO,
          "matmul dispatch info reserved fields must be zero");
    }
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_argmax_dispatch_info(const sllm_argmax_dispatch_info_t *const info,
                              sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "argmax dispatch info output is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, info, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_argmax_dispatch_info_t)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "argmax dispatch info has an unsupported struct size");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "argmax dispatch info ABI version is unsupported");
  }
  if (info->info_version != SLLM_HIP_ARGMAX_DISPATCH_INFO_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "argmax dispatch info version is unsupported");
  }
  for (const uint32_t value : info->reserved) {
    if (value != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_RESERVED_NONZERO,
          "argmax dispatch info reserved fields must be zero");
    }
  }
  return SLLM_STATUS_OK;
}

sllm_status_t validate_attention_preprocess_dispatch_info(
    const sllm_attention_preprocess_dispatch_info_t *const info,
    sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "attention preprocess dispatch info output is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, info, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_attention_preprocess_dispatch_info_t)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "attention preprocess dispatch info has an unsupported struct size");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "attention preprocess dispatch info ABI version is unsupported");
  }
  if (info->info_version !=
      SLLM_HIP_ATTENTION_PREPROCESS_DISPATCH_INFO_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "attention preprocess dispatch info version is unsupported");
  }
  for (const uint32_t value : info->reserved) {
    if (value != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_RESERVED_NONZERO,
          "attention preprocess dispatch info reserved fields must be zero");
    }
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_completion_timing(const sllm_completion_timing_t *const timing,
                           sllm_error_sink_t *const sink) noexcept {
  if (timing == nullptr) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                                            "completion timing output is null");
  }
  uint32_t prefix[2] = {};
  std::memcpy(prefix, timing, sizeof(prefix));
  if (prefix[0] != sizeof(sllm_completion_timing_t)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "completion timing has an unsupported struct size");
  }
  if (prefix[1] != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "completion timing ABI version is unsupported");
  }
  if (timing->reserved0 != 0U || timing->reserved[0] != 0U ||
      timing->reserved[1] != 0U || timing->reserved[2] != 0U ||
      timing->reserved[3] != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "completion timing reserved fields must be zero");
  }
  return SLLM_STATUS_OK;
}

void initialize_rmsnorm_dispatch_info(sllm_rmsnorm_dispatch_info_t *const info,
                                      const uint64_t dispatch_id,
                                      const uint64_t row_count,
                                      const uint64_t normalized_size,
                                      const char *const arch_name) noexcept {
  const uint32_t struct_size = info->struct_size;
  const uint32_t abi_version = info->abi_version;
  std::memset(info, 0, sizeof(*info));
  info->struct_size = struct_size;
  info->abi_version = abi_version;
  info->info_version = SLLM_HIP_RMSNORM_DISPATCH_INFO_VERSION;
  info->backend = SLLM_BACKEND_HIP;
  info->dispatch_id = dispatch_id;
  info->dispatch_count = 1U;
  info->kernel_id = SLLM_HIP_RMSNORM_KERNEL_ID_BASELINE_WAVE32_V1;
  info->workgroup_size_x = SLLM_HIP_RMSNORM_WORKGROUP_SIZE;
  info->grid_size_x = static_cast<uint32_t>(row_count);
  info->row_count = row_count;
  info->normalized_size = normalized_size;
  info->fallback_allowed = 0U;
  info->fallback_used = 0U;
  sllm_public_runtime::copy_fixed_string(
      info->kernel_symbol, SLLM_HIP_RMSNORM_KERNEL_SYMBOL_MAX,
      ::sllm_rmsnorm_kernel::kLogicalKernelId);
  sllm_public_runtime::copy_fixed_string(info->device_symbol,
                                         SLLM_HIP_RMSNORM_DEVICE_SYMBOL_MAX,
                                         ::sllm_rmsnorm_kernel::kDeviceSymbol);
  sllm_public_runtime::copy_fixed_string(info->gcn_arch_name,
                                         SLLM_HIP_MAX_GCN_ARCH_NAME, arch_name);
}

void initialize_elementwise_dispatch_info(
    sllm_elementwise_dispatch_info_t *const info, const uint64_t dispatch_id,
    const sllm_elementwise_operation_t operation, const uint64_t element_count,
    const char *const arch_name) noexcept {
  const uint32_t struct_size = info->struct_size;
  const uint32_t abi_version = info->abi_version;
  std::memset(info, 0, sizeof(*info));
  info->struct_size = struct_size;
  info->abi_version = abi_version;
  info->info_version = SLLM_HIP_ELEMENTWISE_DISPATCH_INFO_VERSION;
  info->backend = SLLM_BACKEND_HIP;
  info->dispatch_id = dispatch_id;
  info->operation = operation;
  info->dispatch_count = 1U;
  info->kernel_id = operation == SLLM_ELEMENTWISE_OPERATION_COPY
                        ? SLLM_HIP_ELEMENTWISE_KERNEL_ID_COPY_V1
                    : operation == SLLM_ELEMENTWISE_OPERATION_ADD
                        ? SLLM_HIP_ELEMENTWISE_KERNEL_ID_ADD_V1
                    : operation == SLLM_ELEMENTWISE_OPERATION_SILU_MUL
                        ? SLLM_HIP_ELEMENTWISE_KERNEL_ID_SILU_MUL_V1
                        : SLLM_HIP_ELEMENTWISE_KERNEL_ID_SIGMOID_MUL_V1;
  info->workgroup_size_x = SLLM_HIP_ELEMENTWISE_WORKGROUP_SIZE;
  info->grid_size_x = static_cast<uint32_t>(
      (element_count + SLLM_HIP_ELEMENTWISE_WORKGROUP_SIZE - 1U) /
      SLLM_HIP_ELEMENTWISE_WORKGROUP_SIZE);
  info->fallback_allowed = 0U;
  info->fallback_used = 0U;
  info->element_count = element_count;
  const bool copy = operation == SLLM_ELEMENTWISE_OPERATION_COPY;
  const bool add = operation == SLLM_ELEMENTWISE_OPERATION_ADD;
  const bool silu_mul = operation == SLLM_ELEMENTWISE_OPERATION_SILU_MUL;
  sllm_public_runtime::copy_fixed_string(
      info->kernel_symbol, SLLM_HIP_ELEMENTWISE_KERNEL_SYMBOL_MAX,
      copy       ? ::sllm_elementwise_kernel::kCopyLogicalKernelId
      : add      ? ::sllm_elementwise_kernel::kAddLogicalKernelId
      : silu_mul ? ::sllm_elementwise_kernel::kSiluMulLogicalKernelId
                 : ::sllm_elementwise_kernel::kSigmoidMulLogicalKernelId);
  sllm_public_runtime::copy_fixed_string(
      info->device_symbol, SLLM_HIP_ELEMENTWISE_DEVICE_SYMBOL_MAX,
      copy       ? ::sllm_elementwise_kernel::kCopyDeviceSymbol
      : add      ? ::sllm_elementwise_kernel::kAddDeviceSymbol
      : silu_mul ? ::sllm_elementwise_kernel::kSiluMulDeviceSymbol
                 : ::sllm_elementwise_kernel::kSigmoidMulDeviceSymbol);
  sllm_public_runtime::copy_fixed_string(info->gcn_arch_name,
                                         SLLM_HIP_MAX_GCN_ARCH_NAME, arch_name);
}

void initialize_embedding_dispatch_info(
    sllm_embedding_dispatch_info_t *const info, const uint64_t dispatch_id,
    const sllm_embedding::DescriptorMetadata &metadata,
    const char *const arch_name) noexcept {
  const uint32_t struct_size = info->struct_size;
  const uint32_t abi_version = info->abi_version;
  std::memset(info, 0, sizeof(*info));
  info->struct_size = struct_size;
  info->abi_version = abi_version;
  info->info_version = SLLM_HIP_EMBEDDING_DISPATCH_INFO_VERSION;
  info->backend = SLLM_BACKEND_HIP;
  info->dispatch_id = dispatch_id;
  info->dispatch_count = 1U;
  info->kernel_id = SLLM_HIP_EMBEDDING_KERNEL_ID_GATHER_V1;
  info->workgroup_size_x = SLLM_HIP_EMBEDDING_WORKGROUP_SIZE;
  info->grid_size_x = static_cast<uint32_t>(
      (metadata.output_elements + SLLM_HIP_EMBEDDING_WORKGROUP_SIZE - 1U) /
      SLLM_HIP_EMBEDDING_WORKGROUP_SIZE);
  info->fallback_allowed = 0U;
  info->fallback_used = 0U;
  info->token_count = metadata.token_count;
  info->hidden_size = metadata.hidden_size;
  info->vocab_size = metadata.vocab_size;
  sllm_public_runtime::copy_fixed_string(
      info->kernel_symbol, SLLM_HIP_EMBEDDING_KERNEL_SYMBOL_MAX,
      ::sllm_embedding_kernel::kLogicalKernelId);
  sllm_public_runtime::copy_fixed_string(
      info->device_symbol, SLLM_HIP_EMBEDDING_DEVICE_SYMBOL_MAX,
      ::sllm_embedding_kernel::kDeviceSymbol);
  sllm_public_runtime::copy_fixed_string(info->gcn_arch_name,
                                         SLLM_HIP_MAX_GCN_ARCH_NAME, arch_name);
}

void initialize_matmul_dispatch_info(
    sllm_matmul_dispatch_info_t *const info, const uint64_t dispatch_id,
    const sllm_matmul::DescriptorMetadata &metadata,
    const char *const arch_name) noexcept {
  const uint32_t struct_size = info->struct_size;
  const uint32_t abi_version = info->abi_version;
  std::memset(info, 0, sizeof(*info));
  info->struct_size = struct_size;
  info->abi_version = abi_version;
  info->info_version = SLLM_HIP_MATMUL_DISPATCH_INFO_VERSION;
  info->backend = SLLM_BACKEND_HIP;
  info->dispatch_id = dispatch_id;
  info->dispatch_count = 1U;
  info->kernel_id = SLLM_HIP_MATMUL_KERNEL_ID_BASELINE_BF16_FP32_V1;
  info->workgroup_size_x = SLLM_HIP_MATMUL_WORKGROUP_SIZE;
  info->grid_size_x = static_cast<uint32_t>(
      (metadata.output_elements + SLLM_HIP_MATMUL_WORKGROUP_SIZE - 1U) /
      SLLM_HIP_MATMUL_WORKGROUP_SIZE);
  info->fallback_allowed = 0U;
  info->fallback_used = 0U;
  info->m = metadata.m;
  info->k = metadata.k;
  info->n = metadata.n;
  info->output_elements = metadata.output_elements;
  sllm_public_runtime::copy_fixed_string(
      info->kernel_symbol, SLLM_HIP_MATMUL_KERNEL_SYMBOL_MAX,
      ::sllm_matmul_kernel::kLogicalKernelId);
  sllm_public_runtime::copy_fixed_string(info->device_symbol,
                                         SLLM_HIP_MATMUL_DEVICE_SYMBOL_MAX,
                                         ::sllm_matmul_kernel::kDeviceSymbol);
  sllm_public_runtime::copy_fixed_string(info->gcn_arch_name,
                                         SLLM_HIP_MAX_GCN_ARCH_NAME, arch_name);
}

void initialize_argmax_dispatch_info(
    sllm_argmax_dispatch_info_t *const info, const uint64_t dispatch_id,
    const sllm_argmax::DescriptorMetadata &metadata,
    const char *const arch_name) noexcept {
  const uint32_t struct_size = info->struct_size;
  const uint32_t abi_version = info->abi_version;
  std::memset(info, 0, sizeof(*info));
  info->struct_size = struct_size;
  info->abi_version = abi_version;
  info->info_version = SLLM_HIP_ARGMAX_DISPATCH_INFO_VERSION;
  info->backend = SLLM_BACKEND_HIP;
  info->dispatch_id = dispatch_id;
  info->dispatch_count = 1U;
  info->kernel_id = SLLM_HIP_ARGMAX_KERNEL_ID_BASELINE_BF16_V1;
  info->workgroup_size_x = SLLM_HIP_ARGMAX_WORKGROUP_SIZE;
  info->grid_size_x = static_cast<uint32_t>(metadata.m);
  info->row_count = metadata.m;
  info->vocab_size = metadata.v;
  info->fallback_allowed = 0U;
  info->fallback_used = 0U;
  sllm_public_runtime::copy_fixed_string(
      info->kernel_symbol, SLLM_HIP_ARGMAX_KERNEL_SYMBOL_MAX,
      ::sllm_argmax_kernel::kLogicalKernelId);
  sllm_public_runtime::copy_fixed_string(info->device_symbol,
                                         SLLM_HIP_ARGMAX_DEVICE_SYMBOL_MAX,
                                         ::sllm_argmax_kernel::kDeviceSymbol);
  sllm_public_runtime::copy_fixed_string(info->gcn_arch_name,
                                         SLLM_HIP_MAX_GCN_ARCH_NAME, arch_name);
}

void initialize_attention_preprocess_dispatch_info(
    sllm_attention_preprocess_dispatch_info_t *const info,
    const uint64_t dispatch_id,
    const sllm_attention_preprocess::DescriptorMetadata &metadata,
    const char *const arch_name) noexcept {
  const uint32_t struct_size = info->struct_size;
  const uint32_t abi_version = info->abi_version;
  std::memset(info, 0, sizeof(*info));
  info->struct_size = struct_size;
  info->abi_version = abi_version;
  info->info_version = SLLM_HIP_ATTENTION_PREPROCESS_DISPATCH_INFO_VERSION;
  info->backend = SLLM_BACKEND_HIP;
  info->dispatch_id = dispatch_id;
  info->dispatch_count = 1U;
  info->kernel_id = SLLM_HIP_ATTENTION_PREPROCESS_KERNEL_ID_BASELINE_BF16_V1;
  info->workgroup_size_x = SLLM_HIP_ATTENTION_PREPROCESS_WORKGROUP_SIZE;
  info->grid_size_x = static_cast<uint32_t>(
      metadata.m * (SLLM_HIP_ATTENTION_PREPROCESS_Q_HEADS +
                    SLLM_HIP_ATTENTION_PREPROCESS_K_HEADS));
  info->m = metadata.m;
  info->q_heads = SLLM_HIP_ATTENTION_PREPROCESS_Q_HEADS;
  info->k_heads = SLLM_HIP_ATTENTION_PREPROCESS_K_HEADS;
  info->q_head_dim = SLLM_HIP_ATTENTION_PREPROCESS_Q_HEAD_DIM;
  info->k_head_dim = SLLM_HIP_ATTENTION_PREPROCESS_K_HEAD_DIM;
  info->rotary_dim = SLLM_HIP_ATTENTION_PREPROCESS_ROTARY_DIM;
  info->start_position = metadata.start_position;
  info->fallback_allowed = 0U;
  info->fallback_used = 0U;
  sllm_public_runtime::copy_fixed_string(
      info->kernel_symbol, SLLM_HIP_ATTENTION_PREPROCESS_KERNEL_SYMBOL_MAX,
      ::sllm_attention_preprocess_kernel::kLogicalKernelId);
  sllm_public_runtime::copy_fixed_string(
      info->device_symbol, SLLM_HIP_ATTENTION_PREPROCESS_DEVICE_SYMBOL_MAX,
      ::sllm_attention_preprocess_kernel::kDeviceSymbol);
  sllm_public_runtime::copy_fixed_string(info->gcn_arch_name,
                                         SLLM_HIP_MAX_GCN_ARCH_NAME, arch_name);
}

bool copy_property(char *const destination,
                   const std::size_t destination_capacity,
                   const char *const source,
                   const std::size_t source_capacity) noexcept {
  const void *const terminator = std::memchr(source, '\0', source_capacity);
  if (terminator == nullptr) {
    return false;
  }
  const std::size_t length =
      static_cast<std::size_t>(static_cast<const char *>(terminator) - source);
  if (length >= destination_capacity) {
    return false;
  }
  std::memset(destination, 0, destination_capacity);
  std::memcpy(destination, source, length);
  return true;
}

sllm_status_t get_device_properties(const uint32_t device_index,
                                    hipDeviceProp_t *const properties,
                                    sllm_error_sink_t *const sink) noexcept {
  const hipError_t status = hipGetDeviceProperties(properties, device_index);
  if (status != hipSuccess) {
    return hip_failure(sink, status, "hipGetDeviceProperties");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t get_device_count(uint32_t *const count,
                               sllm_error_sink_t *const sink) noexcept {
  int device_count = 0;
  const hipError_t status = hipGetDeviceCount(&device_count);
  if (status != hipSuccess) {
    return hip_failure(sink, status, "hipGetDeviceCount");
  }
  if (device_count < 0) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
        "HIP returned a negative device count");
  }
  *count = static_cast<uint32_t>(device_count);
  return SLLM_STATUS_OK;
}

sllm_status_t select_context_device(const Context *const context,
                                    sllm_error_sink_t *const sink) noexcept {
  if (sllm_public_runtime::FaultInjector::consume(
          sllm_public_runtime::FaultPoint::ContextSelectionFailure)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
        "injected HIP device selection failure");
  }
  const hipError_t status =
      hipSetDevice(static_cast<int>(context->device_index));
  if (status != hipSuccess) {
    return hip_failure(sink, status, "hipSetDevice");
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_device_handle_pair(const Queue *const queue,
                            const Buffer *const buffer,
                            sllm_error_sink_t *const sink) noexcept {
  if (queue->context != buffer->context) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
        "queue and buffer belong to different contexts");
  }
  return SLLM_STATUS_OK;
}

bool reserve_child(Context *const context) noexcept {
  std::lock_guard<std::mutex> lock(context->accounting_mutex);
  return sllm_public_runtime::AccountingState::reserve_child(
      context->accounting);
}

bool release_child_and_lifetime_guard(Context *const context) noexcept {
  std::lock_guard<std::mutex> lock(context->accounting_mutex);
  return sllm_public_runtime::AccountingState::release_child_and_lifetime_guard(
      context->accounting);
}

bool release_lifetime_guard(Context *const context) noexcept {
  std::lock_guard<std::mutex> lock(context->accounting_mutex);
  return sllm_public_runtime::AccountingState::release_lifetime_guard(
      context->accounting);
}

void poison_context_locked(Context *const context) noexcept {
  if (context != nullptr) {
    context->poisoned.store(true);
  }
}

void poison_context(Context *const context) noexcept {
  if (context != nullptr) {
    std::lock_guard<std::mutex> lock(context->accounting_mutex);
    poison_context_locked(context);
  }
}

void retain_poisoned(QuarantineNode *const node,
                     Context *const context) noexcept {
  poison_context(context);
  poison_owner.retain(node);
}

template <typename T>
void retain_poisoned(std::unique_ptr<T> &owner,
                     Context *const context) noexcept {
  poison_context(context);
  poison_owner.retain(std::move(owner));
}

bool rollback_child(Context *const context, sllm_error_sink_t *const sink,
                    const char *const operation) noexcept {
  {
    std::lock_guard<std::mutex> lock(context->accounting_mutex);
    if (!sllm_public_runtime::FaultInjector::consume(
            sllm_public_runtime::FaultPoint::AccountingFailure) &&
        sllm_public_runtime::AccountingState::release_child(
            context->accounting)) {
      return true;
    }
    /* The reservation is still held when this fails.  Poisoning while the
     * accounting lock is held closes the Context release window atomically;
     * the caller never exposes a raw Context* after this transition. */
    poison_context_locked(context);
  }
  (void)sllm_public_runtime::write_error(sink, SLLM_STATUS_INTERNAL_ERROR,
                                         operation);
  return false;
}

bool rollback_reserved_submission(Context *const context, Queue *const queue,
                                  Buffer *const buffer,
                                  sllm_error_sink_t *const sink) noexcept {
  std::lock_guard<std::mutex> lock(context->accounting_mutex);
  if (sllm_public_runtime::AccountingState::rollback_submission(
          context->accounting, queue->accounting, buffer->accounting)) {
    return true;
  }
  poison_context_locked(context);
  (void)sllm_public_runtime::write_error(
      sink, SLLM_STATUS_INTERNAL_ERROR,
      "submission accounting rollback failed; context poisoned");
  return false;
}

bool rollback_reserved_rmsnorm_submission(
    RmsNormPlan *const plan, Queue *const queue,
    sllm_error_sink_t *const sink) noexcept {
  Context *const context = plan->context;
  std::lock_guard<std::mutex> lock(context->accounting_mutex);
  if (sllm_public_runtime::AccountingState::rollback_rmsnorm_submission(
          context->accounting, queue->accounting, plan->activation->accounting,
          plan->raw_scale->accounting, plan->output->accounting)) {
    plan->in_flight = false;
    return true;
  }
  poison_context_locked(context);
  (void)sllm_public_runtime::write_error(
      sink, SLLM_STATUS_INTERNAL_ERROR,
      "RMSNorm submission accounting rollback failed; context poisoned");
  return false;
}

bool rollback_reserved_elementwise_submission(
    ElementwisePlan *const plan, Queue *const queue,
    sllm_error_sink_t *const sink) noexcept {
  Context *const context = plan->context;
  std::lock_guard<std::mutex> lock(context->accounting_mutex);
  if (sllm_public_runtime::AccountingState::rollback_rmsnorm_submission(
          context->accounting, queue->accounting, plan->input0->accounting,
          plan->input1->accounting, plan->output->accounting)) {
    plan->in_flight = false;
    return true;
  }
  poison_context_locked(context);
  (void)sllm_public_runtime::write_error(
      sink, SLLM_STATUS_INTERNAL_ERROR,
      "elementwise submission accounting rollback failed; context poisoned");
  return false;
}

bool rollback_reserved_argmax_submission(
    ElementwisePlan *const plan, Queue *const queue,
    sllm_error_sink_t *const sink) noexcept {
  Context *const context = plan->context;
  std::lock_guard<std::mutex> lock(context->accounting_mutex);
  if (sllm_public_runtime::AccountingState::rollback_two_buffer_submission(
          context->accounting, queue->accounting, plan->input0->accounting,
          plan->output->accounting)) {
    plan->in_flight = false;
    return true;
  }
  poison_context_locked(context);
  (void)sllm_public_runtime::write_error(
      sink, SLLM_STATUS_INTERNAL_ERROR,
      "argmax submission accounting rollback failed; context poisoned");
  return false;
}

bool rollback_reserved_attention_submission(
    AttentionPreprocessPlan *const plan, Queue *const queue,
    sllm_error_sink_t *const sink) noexcept {
  Context *const context = plan->context;
  std::lock_guard<std::mutex> lock(context->accounting_mutex);
  if (rollback_attention_submission(context->accounting, queue->accounting,
                                    plan->buffers)) {
    plan->in_flight = false;
    return true;
  }
  poison_context_locked(context);
  (void)sllm_public_runtime::write_error(
      sink, SLLM_STATUS_INTERNAL_ERROR,
      "attention preprocess submission accounting rollback failed; context "
      "poisoned");
  return false;
}

bool clear_kv_transition_locked(KvState *const state, const uint64_t token,
                                const bool publish) noexcept {
  if (state == nullptr || token == 0U || state->transition_token != token) {
    return false;
  }
  if (publish) {
    if (!state->commit_allowed ||
        state->published_length != state->transition_start ||
        state->transition_end < state->transition_start ||
        state->transition_end > state->capacity_tokens ||
        state->generation == std::numeric_limits<uint64_t>::max()) {
      return false;
    }
    state->published_length = state->transition_end;
    ++state->generation;
  }
  state->transition_token = 0U;
  state->transition_start = 0U;
  state->transition_count = 0U;
  state->transition_end = 0U;
  state->commit_allowed = false;
  return true;
}

bool rollback_reserved_kv_submission(KvState *const state, Queue *const queue,
                                     Buffer *const key_input,
                                     Buffer *const value_input,
                                     sllm_error_sink_t *const sink) noexcept {
  Context *const context = state->context;
  std::lock_guard<std::mutex> lock(context->accounting_mutex);
  const uint64_t token = state->transition_token;
  if (sllm_public_runtime::AccountingState::rollback_kv_append(
          context->accounting, queue->accounting, state->accounting,
          key_input->accounting, value_input->accounting,
          state->key_buffer->accounting, state->value_buffer->accounting) &&
      clear_kv_transition_locked(state, token, false)) {
    return true;
  }
  poison_context_locked(context);
  (void)sllm_public_runtime::write_error(
      sink, SLLM_STATUS_INTERNAL_ERROR,
      "KV append accounting or transition rollback failed; context poisoned");
  return false;
}

bool finalize_kv_append(Completion *const completion) noexcept {
  if (!completion->kv_state_append) {
    return true;
  }
  std::lock_guard<std::mutex> lock(completion->context->accounting_mutex);
  if (completion->kv_state == nullptr ||
      completion->kv_state->transition_token != completion->kv_append_token ||
      completion->kv_state->transition_start != completion->kv_append_start ||
      completion->kv_state->transition_count != completion->kv_append_count ||
      completion->kv_state->transition_end != completion->kv_append_end) {
    poison_context_locked(completion->context);
    return false;
  }
  const bool publish =
      completion->kv_state != nullptr && completion->kv_state->commit_allowed;
  if (!clear_kv_transition_locked(completion->kv_state,
                                  completion->kv_append_token, publish)) {
    poison_context_locked(completion->context);
    return false;
  }
  return true;
}

bool revoke_kv_append_commit(Completion *const completion) noexcept {
  if (!completion->kv_state_append) {
    return false;
  }
  std::lock_guard<std::mutex> lock(completion->context->accounting_mutex);
  if (completion->kv_state == nullptr ||
      completion->kv_state->transition_token != completion->kv_append_token ||
      completion->kv_append_token == 0U) {
    poison_context_locked(completion->context);
    return false;
  }
  completion->kv_state->commit_allowed = false;
  return true;
}

bool clear_linear_attention_transition_locked(LinearAttentionState *const state,
                                              const uint64_t token,
                                              const bool publish) noexcept {
  if (state == nullptr || token == 0U || state->transition_token != token) {
    return false;
  }
  if (publish) {
    if (!state->commit_allowed ||
        state->published_length != state->transition_start ||
        state->transition_end < state->transition_start ||
        state->transition_end > state->capacity_tokens ||
        state->generation == std::numeric_limits<uint64_t>::max()) {
      return false;
    }
    state->active_slot = 1U - state->active_slot;
    state->published_length = state->transition_end;
    ++state->generation;
  }
  state->transition_token = 0U;
  state->transition_start = 0U;
  state->transition_count = 0U;
  state->transition_end = 0U;
  state->commit_allowed = false;
  return true;
}

bool rollback_reserved_linear_attention(
    LinearAttentionState *const state, Queue *const queue,
    const LinearAttentionBuffers &buffers,
    sllm_error_sink_t *const sink) noexcept {
  Context *const context = state->context;
  std::lock_guard<std::mutex> lock(context->accounting_mutex);
  const uint64_t token = state->transition_token;
  if (rollback_linear_attention_submission(context, queue, state, buffers) &&
      clear_linear_attention_transition_locked(state, token, false)) {
    return true;
  }
  poison_context_locked(context);
  (void)sllm_public_runtime::write_error(
      sink, SLLM_STATUS_INTERNAL_ERROR,
      "linear attention accounting or transition rollback failed; context "
      "poisoned");
  return false;
}

bool finalize_linear_attention(Completion *const completion) noexcept {
  if (!completion->linear_attention) {
    return true;
  }
  std::lock_guard<std::mutex> lock(completion->context->accounting_mutex);
  LinearAttentionState *const state = completion->linear_attention_state;
  if (state == nullptr ||
      state->transition_token != completion->linear_attention_token ||
      state->transition_start != completion->linear_attention_start ||
      state->transition_count != completion->linear_attention_count ||
      state->transition_end != completion->linear_attention_end) {
    poison_context_locked(completion->context);
    return false;
  }
  const bool publish = state->commit_allowed;
  if (!clear_linear_attention_transition_locked(
          state, completion->linear_attention_token, publish)) {
    poison_context_locked(completion->context);
    return false;
  }
  return true;
}

bool revoke_linear_attention_commit(Completion *const completion) noexcept {
  if (!completion->linear_attention) {
    return false;
  }
  std::lock_guard<std::mutex> lock(completion->context->accounting_mutex);
  LinearAttentionState *const state = completion->linear_attention_state;
  if (state == nullptr ||
      state->transition_token != completion->linear_attention_token ||
      completion->linear_attention_token == 0U) {
    poison_context_locked(completion->context);
    return false;
  }
  state->commit_allowed = false;
  return true;
}

bool reserve_submission_references(Queue *const queue,
                                   Buffer *const buffer) noexcept {
  std::lock_guard<std::mutex> lock(queue->context->accounting_mutex);
  return sllm_public_runtime::AccountingState::reserve_submission(
      queue->context->accounting, queue->accounting, buffer->accounting);
}

bool release_submission_references(Completion *const completion) noexcept {
  if (completion->active_release_attempted) {
    return completion->references_released;
  }
  completion->active_release_attempted = true;
  std::lock_guard<std::mutex> lock(completion->context->accounting_mutex);
  bool released = false;
  if (!sllm_public_runtime::FaultInjector::consume(
          sllm_public_runtime::FaultPoint::AccountingFailure)) {
    if (completion->rmsnorm) {
      released = sllm_public_runtime::AccountingState::release_rmsnorm_active(
          completion->queue->accounting,
          completion->rmsnorm_activation->accounting,
          completion->rmsnorm_raw_scale->accounting,
          completion->rmsnorm_output->accounting);
    } else if (completion->elementwise) {
      released = sllm_public_runtime::AccountingState::release_rmsnorm_active(
          completion->queue->accounting,
          completion->elementwise_input0->accounting,
          completion->elementwise_input1->accounting,
          completion->elementwise_output->accounting);
    } else if (completion->matmul) {
      released = sllm_public_runtime::AccountingState::release_rmsnorm_active(
          completion->queue->accounting,
          completion->matmul_activation->accounting,
          completion->matmul_weight->accounting,
          completion->matmul_output->accounting);
    } else if (completion->argmax) {
      released =
          sllm_public_runtime::AccountingState::release_two_buffer_active(
              completion->queue->accounting,
              completion->argmax_logits->accounting,
              completion->argmax_output->accounting);
    } else if (completion->attention_preprocess) {
      released = release_attention_active(completion->queue->accounting,
                                          completion->attention_buffers);
    } else if (completion->kv_state_append) {
      released = sllm_public_runtime::AccountingState::release_kv_active(
          completion->queue->accounting, completion->kv_state->accounting,
          completion->kv_key_input->accounting,
          completion->kv_value_input->accounting,
          completion->kv_key_buffer->accounting,
          completion->kv_value_buffer->accounting);
    } else if (completion->causal_attention) {
      released = release_causal_active(completion->queue,
                                       completion->causal_attention_state,
                                       completion->causal_attention_buffers);
    } else if (completion->linear_attention) {
      released = release_linear_attention_active(
          completion->queue, completion->linear_attention_state,
          completion->linear_attention_buffers);
    } else {
      released = sllm_public_runtime::AccountingState::release_active(
          completion->queue->accounting, completion->buffer->accounting);
    }
  }
  if (!released) {
    completion->reference_accounting_failed = true;
    poison_context_locked(completion->context);
    return false;
  }
  completion->references_released = true;
  if (completion->rmsnorm && completion->rmsnorm_plan != nullptr) {
    completion->rmsnorm_plan->in_flight = false;
  }
  if (completion->elementwise && completion->elementwise_plan != nullptr) {
    completion->elementwise_plan->in_flight = false;
  }
  if (completion->matmul && completion->matmul_plan != nullptr) {
    completion->matmul_plan->in_flight = false;
  }
  if (completion->argmax && completion->argmax_plan != nullptr) {
    completion->argmax_plan->in_flight = false;
  }
  if (completion->attention_preprocess &&
      completion->attention_preprocess_plan != nullptr) {
    completion->attention_preprocess_plan->in_flight = false;
  }
  return true;
}

bool rollback_submission_references(Completion *const completion) noexcept {
  if (completion->active_release_attempted ||
      completion->context_child_released) {
    return false;
  }
  std::lock_guard<std::mutex> lock(completion->context->accounting_mutex);
  bool released = false;
  if (!sllm_public_runtime::FaultInjector::consume(
          sllm_public_runtime::FaultPoint::AccountingFailure)) {
    if (completion->rmsnorm) {
      released =
          sllm_public_runtime::AccountingState::rollback_rmsnorm_submission(
              completion->context->accounting, completion->queue->accounting,
              completion->rmsnorm_activation->accounting,
              completion->rmsnorm_raw_scale->accounting,
              completion->rmsnorm_output->accounting);
    } else if (completion->elementwise) {
      released =
          sllm_public_runtime::AccountingState::rollback_rmsnorm_submission(
              completion->context->accounting, completion->queue->accounting,
              completion->elementwise_input0->accounting,
              completion->elementwise_input1->accounting,
              completion->elementwise_output->accounting);
    } else if (completion->matmul) {
      released =
          sllm_public_runtime::AccountingState::rollback_rmsnorm_submission(
              completion->context->accounting, completion->queue->accounting,
              completion->matmul_activation->accounting,
              completion->matmul_weight->accounting,
              completion->matmul_output->accounting);
    } else if (completion->argmax) {
      released =
          sllm_public_runtime::AccountingState::rollback_two_buffer_submission(
              completion->context->accounting, completion->queue->accounting,
              completion->argmax_logits->accounting,
              completion->argmax_output->accounting);
    } else if (completion->attention_preprocess) {
      released = rollback_attention_submission(completion->context->accounting,
                                               completion->queue->accounting,
                                               completion->attention_buffers);
    } else if (completion->kv_state_append) {
      released = sllm_public_runtime::AccountingState::rollback_kv_append(
          completion->context->accounting, completion->queue->accounting,
          completion->kv_state->accounting,
          completion->kv_key_input->accounting,
          completion->kv_value_input->accounting,
          completion->kv_key_buffer->accounting,
          completion->kv_value_buffer->accounting);
    } else if (completion->causal_attention) {
      released = rollback_causal_attention(completion);
    } else if (completion->linear_attention) {
      released = rollback_linear_attention_submission(
          completion->context, completion->queue,
          completion->linear_attention_state,
          completion->linear_attention_buffers);
    } else {
      released = sllm_public_runtime::AccountingState::rollback_submission(
          completion->context->accounting, completion->queue->accounting,
          completion->buffer->accounting);
    }
  }
  if (released && completion->kv_state_append) {
    released = clear_kv_transition_locked(completion->kv_state,
                                          completion->kv_append_token, false);
  }
  if (released && completion->linear_attention) {
    released = clear_linear_attention_transition_locked(
        completion->linear_attention_state, completion->linear_attention_token,
        false);
  }
  if (released) {
    completion->active_release_attempted = true;
    completion->references_released = true;
    completion->context_child_released = true;
    if (completion->rmsnorm && completion->rmsnorm_plan != nullptr) {
      completion->rmsnorm_plan->in_flight = false;
    }
    if (completion->elementwise && completion->elementwise_plan != nullptr) {
      completion->elementwise_plan->in_flight = false;
    }
    if (completion->matmul && completion->matmul_plan != nullptr) {
      completion->matmul_plan->in_flight = false;
    }
    if (completion->argmax && completion->argmax_plan != nullptr) {
      completion->argmax_plan->in_flight = false;
    }
    if (completion->attention_preprocess &&
        completion->attention_preprocess_plan != nullptr) {
      completion->attention_preprocess_plan->in_flight = false;
    }
  } else {
    poison_context_locked(completion->context);
  }
  return released;
}

bool release_completion_child_reference(Completion *const completion) noexcept {
  if (completion->context_child_released) {
    return true;
  }
  std::lock_guard<std::mutex> lock(completion->context->accounting_mutex);
  bool released = false;
  if (!sllm_public_runtime::FaultInjector::consume(
          sllm_public_runtime::FaultPoint::AccountingFailure)) {
    if (completion->rmsnorm) {
      released =
          sllm_public_runtime::AccountingState::release_rmsnorm_completion(
              completion->context->accounting, completion->queue->accounting,
              completion->rmsnorm_activation->accounting,
              completion->rmsnorm_raw_scale->accounting,
              completion->rmsnorm_output->accounting);
    } else if (completion->elementwise) {
      released =
          sllm_public_runtime::AccountingState::release_rmsnorm_completion(
              completion->context->accounting, completion->queue->accounting,
              completion->elementwise_input0->accounting,
              completion->elementwise_input1->accounting,
              completion->elementwise_output->accounting);
    } else if (completion->matmul) {
      released =
          sllm_public_runtime::AccountingState::release_rmsnorm_completion(
              completion->context->accounting, completion->queue->accounting,
              completion->matmul_activation->accounting,
              completion->matmul_weight->accounting,
              completion->matmul_output->accounting);
    } else if (completion->argmax) {
      released =
          sllm_public_runtime::AccountingState::release_two_buffer_completion(
              completion->context->accounting, completion->queue->accounting,
              completion->argmax_logits->accounting,
              completion->argmax_output->accounting);
    } else if (completion->attention_preprocess) {
      released = release_attention_completion(completion->context->accounting,
                                              completion->queue->accounting,
                                              completion->attention_buffers);
    } else if (completion->kv_state_append) {
      released = sllm_public_runtime::AccountingState::release_kv_completion(
          completion->context->accounting, completion->queue->accounting,
          completion->kv_state->accounting,
          completion->kv_key_input->accounting,
          completion->kv_value_input->accounting,
          completion->kv_key_buffer->accounting,
          completion->kv_value_buffer->accounting);
    } else if (completion->causal_attention) {
      released = release_causal_completion(completion);
    } else if (completion->linear_attention) {
      released = release_linear_attention_completion(
          completion->context, completion->queue,
          completion->linear_attention_state,
          completion->linear_attention_buffers);
    } else {
      released = sllm_public_runtime::AccountingState::
          release_completion_and_lifetime_guard(completion->context->accounting,
                                                completion->queue->accounting,
                                                completion->buffer->accounting);
    }
  }
  if (released) {
    completion->context_child_released = true;
  } else {
    poison_context_locked(completion->context);
  }
  return released;
}

sllm_status_t poll_completion(Completion *const completion,
                              sllm_error_sink_t *const sink) noexcept {
  if (completion->terminal) {
    if (completion->success) {
      return SLLM_STATUS_OK;
    }
    if (completion->reference_accounting_failed) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INTERNAL_ERROR,
          "completion reference accounting failed; release is disabled");
    }
    return hip_failure(sink, completion->failure_status,
                       "completion event query");
  }
  hipError_t status = hipSuccess;
  if (sllm_public_runtime::FaultInjector::consume(
          sllm_public_runtime::FaultPoint::CompletionQueryPending)) {
    status = hipErrorNotReady;
  } else if (sllm_public_runtime::FaultInjector::consume(
                 sllm_public_runtime::FaultPoint::CompletionQueryFatal)) {
    status = hipErrorUnknown;
  } else {
    status = hipEventQuery(completion->event);
  }
  if (status == hipSuccess) {
    if (completion->rmsnorm || completion->elementwise || completion->matmul ||
        completion->argmax || completion->attention_preprocess ||
        completion->kv_state_append || completion->causal_attention ||
        completion->linear_attention) {
      if (completion->timing_start_event == nullptr) {
        completion->terminal = true;
        completion->success = false;
        completion->failure_status = hipErrorInvalidValue;
        completion->safety.quarantine();
        completion->safe_to_release = false;
        return sllm_public_runtime::write_error(
            sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
            "numeric operation completion has no timing start event");
      }
      float elapsed_ms = 0.0F;
      const hipError_t elapsed_status = hipEventElapsedTime(
          &elapsed_ms, completion->timing_start_event, completion->event);
      if (elapsed_status != hipSuccess || !std::isfinite(elapsed_ms) ||
          elapsed_ms <= 0.0F) {
        completion->terminal = true;
        completion->success = false;
        completion->failure_status = elapsed_status == hipSuccess
                                         ? hipErrorInvalidValue
                                         : elapsed_status;
        completion->safety.quarantine();
        completion->safe_to_release = false;
        return hip_failure(sink, completion->failure_status,
                           "hipEventElapsedTime");
      }
      const double elapsed_ns = static_cast<double>(elapsed_ms) * 1000000.0;
      if (!std::isfinite(elapsed_ns) || elapsed_ns < 1.0 ||
          elapsed_ns >
              static_cast<double>(std::numeric_limits<uint64_t>::max())) {
        completion->terminal = true;
        completion->success = false;
        completion->failure_status = hipErrorInvalidValue;
        completion->safety.quarantine();
        completion->safe_to_release = false;
        return sllm_public_runtime::write_error(
            sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
            "hipEventElapsedTime returned a non-positive or non-finite value");
      }
      completion->timing_elapsed_ns =
          static_cast<uint64_t>(std::ceil(elapsed_ns));
      completion->timing_valid = completion->timing_elapsed_ns != 0U;
      if (!completion->timing_valid) {
        completion->terminal = true;
        completion->success = false;
        completion->failure_status = hipErrorInvalidValue;
        completion->safety.quarantine();
        completion->safe_to_release = false;
        return sllm_public_runtime::write_error(
            sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
            "hipEventElapsedTime rounded to zero nanoseconds");
      }
    }
    if (!release_submission_references(completion)) {
      completion->terminal = true;
      completion->success = false;
      completion->safe_to_release = false;
      completion->safety.quarantine();
      completion->reference_accounting_failed = true;
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INTERNAL_ERROR,
          "completion reference accounting failed; release is disabled");
    }
    if (!finalize_kv_append(completion)) {
      completion->terminal = true;
      completion->success = false;
      completion->safe_to_release = false;
      completion->failure_status = hipErrorInvalidValue;
      completion->safety.quarantine();
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INTERNAL_ERROR,
          "KV append transition finalization failed; context poisoned");
    }
    if (!finalize_linear_attention(completion)) {
      completion->terminal = true;
      completion->success = false;
      completion->safe_to_release = false;
      completion->failure_status = hipErrorInvalidValue;
      completion->safety.quarantine();
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INTERNAL_ERROR,
          "linear attention transition finalization failed; context poisoned");
    }
    completion->terminal = true;
    completion->success = true;
    completion->safety.observe_positive_completion();
    completion->safe_to_release = true;
    return SLLM_STATUS_OK;
  }
  if (status == hipErrorNotReady) {
    return SLLM_STATUS_PUBLIC_PENDING;
  }
  /* A fatal query is not evidence that the stream stopped touching host
   * staging or device dependencies.  Keep active accounting and the event
   * intact; the caller will move the complete dependency graph to the
   * process-lifetime poison owner. */
  completion->terminal = true;
  completion->success = false;
  completion->failure_status = status;
  completion->safety.quarantine();
  completion->safe_to_release = false;
  return hip_failure(sink, status, "hipEventQuery");
}

void fill_completion_result(const Completion *const completion,
                            sllm_completion_result_t *const result) noexcept {
  initialize_completion_result(result);
  result->state = completion->terminal
                      ? (completion->success ? SLLM_COMPLETION_STATE_SUCCESS
                                             : SLLM_COMPLETION_STATE_FAILURE)
                      : SLLM_COMPLETION_STATE_PENDING;
  result->transfer_size_bytes = completion->transfer_size_bytes;
  result->available_bytes =
      completion->d2h && completion->terminal && completion->success
          ? static_cast<uint64_t>(completion->host_storage.size())
          : 0U;
}

void quarantine_completion(sllm_completion_t *const raw_completion,
                           Completion *const completion) noexcept {
  if (completion == nullptr) {
    return;
  }
  bool transfer = false;
  {
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    std::lock_guard<std::mutex> state_lock(completion->state_mutex);
    if (!completion->orphaned) {
      completion->orphaned = true;
      completion->safe_to_release = false;
      completion->release_active = false;
      completion->safety.quarantine();
      if (lookup<Completion>(raw_completion, HandleKind::Completion) ==
          completion) {
        unregister_handle(raw_completion);
      }
      transfer = true;
    }
  }
  if (transfer) {
    retain_poisoned(completion, completion->context);
  }
}

sllm_status_t cleanup_failed_submission(
    std::unique_ptr<Completion> &candidate, const uintptr_t token,
    const hipError_t primary_error, const char *const primary_operation,
    Queue *const queue, sllm_error_sink_t *const sink) noexcept {
  /* The caller never receives this token.  Consume it before any fallible
   * rollback so a failed submission can never leave an unreachable registry
   * entry. */
  {
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    unregister_handle_token(token);
  }
  const hipError_t synchronize_status = hipStreamSynchronize(queue->stream);
  if (synchronize_status != hipSuccess) {
    candidate->orphaned = true;
    retain_poisoned(candidate, candidate->context);
    return hip_failure(sink, synchronize_status,
                       "hipStreamSynchronize cleanup after async failure");
  }
  const hipError_t destroy_status =
      destroy_event_with_fault_injection(candidate->event);
  if (destroy_status != hipSuccess) {
    candidate->orphaned = true;
    retain_poisoned(candidate, candidate->context);
    return hip_failure(sink, destroy_status,
                       "hipEventDestroy cleanup after async failure");
  }
  candidate->event = nullptr;
  if (candidate->timing_start_event != nullptr) {
    const hipError_t timing_destroy_status =
        destroy_event_with_fault_injection(candidate->timing_start_event);
    if (timing_destroy_status != hipSuccess) {
      candidate->orphaned = true;
      retain_poisoned(candidate, candidate->context);
      return hip_failure(sink, timing_destroy_status,
                         "hipEventDestroy timing cleanup after async failure");
    }
    candidate->timing_start_event = nullptr;
  }
  if (!rollback_submission_references(candidate.get())) {
    candidate->orphaned = true;
    retain_poisoned(candidate, candidate->context);
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INTERNAL_ERROR,
        "submission counter rollback failed; resources are retained");
  }
  return hip_failure(sink, primary_error, primary_operation);
}

sllm_status_t pin_completion(sllm_completion_t *const raw_completion,
                             Completion **const pinned,
                             sllm_error_sink_t *const sink) noexcept {
  if (pinned == nullptr) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                                            "completion pin output is null");
  }
  *pinned = nullptr;
  std::lock_guard<std::mutex> lock(registry_mutex);
  Completion *const completion =
      lookup<Completion>(raw_completion, HandleKind::Completion);
  if (completion == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
        "completion handle is stale or has the wrong kind");
  }
  std::lock_guard<std::mutex> state_lock(completion->state_mutex);
  if (completion->release_active) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_PUBLIC_BUSY,
        "completion release is already in progress");
  }
  if (completion->api_pins.load(std::memory_order_acquire) ==
      std::numeric_limits<uint64_t>::max()) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_PUBLIC_BUSY,
        "completion API pin counter is exhausted");
  }
  completion->api_pins.fetch_add(1U, std::memory_order_acq_rel);
  *pinned = completion;
  return SLLM_STATUS_OK;
}

void unpin_completion(Completion *const completion) noexcept {
  std::lock_guard<std::mutex> registry_lock(registry_mutex);
  std::lock_guard<std::mutex> state_lock(completion->state_mutex);
  if (completion->api_pins.load(std::memory_order_acquire) != 0U) {
    completion->api_pins.fetch_sub(1U, std::memory_order_acq_rel);
  }
}

class CompletionPin final {
public:
  explicit CompletionPin(Completion *const completion) noexcept
      : completion_(completion) {}

  CompletionPin(const CompletionPin &) = delete;
  CompletionPin &operator=(const CompletionPin &) = delete;

  ~CompletionPin() { unpin_completion(completion_); }

private:
  Completion *const completion_;
};

class WaitActive final {
public:
  explicit WaitActive(Completion *const completion) noexcept
      : completion_(completion) {
    completion_->wait_active = true;
  }

  WaitActive(const WaitActive &) = delete;
  WaitActive &operator=(const WaitActive &) = delete;

  ~WaitActive() { completion_->wait_active = false; }

private:
  Completion *const completion_;
};

class NativeStreamGuard final {
public:
  NativeStreamGuard() noexcept : context_(nullptr), stream_(nullptr) {}
  NativeStreamGuard(Context *const context, const hipStream_t stream) noexcept
      : context_(context), stream_(stream) {}

  NativeStreamGuard(const NativeStreamGuard &) = delete;
  NativeStreamGuard &operator=(const NativeStreamGuard &) = delete;

  ~NativeStreamGuard() { (void)cleanup(); }

  hipError_t cleanup() noexcept {
    if (stream_ == nullptr) {
      return hipSuccess;
    }
    const hipError_t status = destroy_stream_with_fault_injection(stream_);
    if (status == hipSuccess) {
      stream_ = nullptr;
      return status;
    }
    orphan_owner.retain_stream(context_, stream_);
    stream_ = nullptr;
    return status;
  }

  void release() noexcept { stream_ = nullptr; }

private:
  Context *context_;
  hipStream_t stream_;
};

class NativeAllocationGuard final {
public:
  NativeAllocationGuard() noexcept : context_(nullptr), pointer_(nullptr) {}
  NativeAllocationGuard(Context *const context, void *const pointer) noexcept
      : context_(context), pointer_(pointer) {}

  NativeAllocationGuard(const NativeAllocationGuard &) = delete;
  NativeAllocationGuard &operator=(const NativeAllocationGuard &) = delete;

  ~NativeAllocationGuard() { (void)cleanup(); }

  hipError_t cleanup() noexcept {
    if (pointer_ == nullptr) {
      return hipSuccess;
    }
    const hipError_t status = free_allocation_with_fault_injection(pointer_);
    if (status == hipSuccess) {
      pointer_ = nullptr;
      return status;
    }
    orphan_owner.retain_allocation(context_, pointer_);
    pointer_ = nullptr;
    return status;
  }

  void release() noexcept { pointer_ = nullptr; }

private:
  Context *context_;
  void *pointer_;
};

class NativeEventGuard final {
public:
  NativeEventGuard() noexcept : context_(nullptr), event_(nullptr) {}
  NativeEventGuard(Context *const context, const hipEvent_t event) noexcept
      : context_(context), event_(event) {}

  NativeEventGuard(const NativeEventGuard &) = delete;
  NativeEventGuard &operator=(const NativeEventGuard &) = delete;

  ~NativeEventGuard() { (void)cleanup(); }

  hipError_t cleanup() noexcept {
    if (event_ == nullptr) {
      return hipSuccess;
    }
    const hipError_t status = destroy_event_with_fault_injection(event_);
    if (status == hipSuccess) {
      event_ = nullptr;
      return status;
    }
    orphan_owner.retain_event(context_, event_);
    event_ = nullptr;
    return status;
  }

  void release() noexcept { event_ = nullptr; }

  void adopt(Context *const context, const hipEvent_t event) noexcept {
    context_ = context;
    event_ = event;
  }

private:
  Context *context_;
  hipEvent_t event_;
};

sllm_status_t rollback_unpublished_submission(
    std::unique_ptr<Completion> &candidate, NativeEventGuard &event_guard,
    const char *const primary_message, sllm_error_sink_t *const sink) noexcept {
  /* The event guard owns the native event until this function completes.  A
   * successful destroy consumes it; an ambiguous destroy transfers it to the
   * durable orphan owner and poisons the still-live Context.  Only after that
   * ownership decision is complete may the submission reservation be rolled
   * back. */
  const hipError_t destroy_status = event_guard.cleanup();
  candidate->event = nullptr;
  hipError_t timing_destroy_status = hipSuccess;
  if (candidate->timing_start_event != nullptr) {
    timing_destroy_status =
        destroy_event_with_fault_injection(candidate->timing_start_event);
    if (timing_destroy_status == hipSuccess) {
      candidate->timing_start_event = nullptr;
    }
  }
  if (timing_destroy_status != hipSuccess) {
    candidate->orphaned = true;
    retain_poisoned(candidate, candidate->context);
    return hip_failure(sink, timing_destroy_status,
                       "hipEventDestroy timing registry rollback");
  }
  if (!rollback_submission_references(candidate.get())) {
    candidate->orphaned = true;
    retain_poisoned(candidate, candidate->context);
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INTERNAL_ERROR,
        "submission accounting rollback failed after guarded event cleanup");
  }
  if (destroy_status != hipSuccess) {
    return hip_failure(sink, destroy_status,
                       "hipEventDestroy submission registry rollback");
  }
  return sllm_public_runtime::write_error(sink, SLLM_STATUS_INTERNAL_ERROR,
                                          primary_message);
}

/* Once RMSNorm submission accounting has been reserved, every exceptional
 * exit must either roll that reservation back or transfer the complete graph
 * to the existing quarantine path.  The guard is deliberately stateful: a
 * candidate completion has a guarded event before registration, and a
 * registered completion must use the published-token cleanup path. */
class RmsNormExecuteScopeGuard final {
public:
  RmsNormExecuteScopeGuard(RmsNormPlan *const plan, Queue *const queue,
                           std::unique_ptr<Completion> *const candidate,
                           NativeEventGuard *const event_guard,
                           sllm_error_sink_t *const sink) noexcept
      : plan_(plan), queue_(queue), candidate_(candidate),
        event_guard_(event_guard), sink_(sink), token_(0U),
        phase_(Phase::Reserved) {}

  RmsNormExecuteScopeGuard(const RmsNormExecuteScopeGuard &) = delete;
  RmsNormExecuteScopeGuard &
  operator=(const RmsNormExecuteScopeGuard &) = delete;

  ~RmsNormExecuteScopeGuard() { cleanup(); }

  void candidate_allocated() noexcept { phase_ = Phase::Candidate; }

  void completion_registered(const uintptr_t token) noexcept {
    token_ = token;
    phase_ = Phase::Registered;
  }

  void disarm() noexcept { phase_ = Phase::Disarmed; }

private:
  enum class Phase : uint8_t { Reserved, Candidate, Registered, Disarmed };

  void cleanup() noexcept {
    switch (phase_) {
    case Phase::Reserved:
      (void)rollback_reserved_rmsnorm_submission(plan_, queue_, sink_);
      break;
    case Phase::Candidate:
      if (candidate_ != nullptr && candidate_->get() != nullptr) {
        (void)rollback_unpublished_submission(
            *candidate_, *event_guard_,
            "RMSNorm execute unwound before completion registration", sink_);
      } else {
        (void)rollback_reserved_rmsnorm_submission(plan_, queue_, sink_);
      }
      break;
    case Phase::Registered:
      if (candidate_ != nullptr && candidate_->get() != nullptr) {
        (void)cleanup_failed_submission(
            *candidate_, token_, hipErrorUnknown,
            "RMSNorm execute unwound after completion registration", queue_,
            sink_);
      } else {
        /* A registered completion without its owner is already impossible to
         * release safely.  Preserve the fail-closed accounting state. */
        poison_context(plan_->context);
      }
      break;
    case Phase::Disarmed:
      break;
    }
    phase_ = Phase::Disarmed;
  }

  RmsNormPlan *const plan_;
  Queue *const queue_;
  std::unique_ptr<Completion> *const candidate_;
  NativeEventGuard *const event_guard_;
  sllm_error_sink_t *const sink_;
  uintptr_t token_;
  Phase phase_;
};

class ElementwiseExecuteScopeGuard final {
public:
  ElementwiseExecuteScopeGuard(ElementwisePlan *const plan, Queue *const queue,
                               std::unique_ptr<Completion> *const candidate,
                               NativeEventGuard *const event_guard,
                               sllm_error_sink_t *const sink,
                               const bool argmax = false) noexcept
      : plan_(plan), queue_(queue), candidate_(candidate),
        event_guard_(event_guard), sink_(sink), argmax_(argmax), token_(0U),
        phase_(Phase::Reserved) {}

  ElementwiseExecuteScopeGuard(const ElementwiseExecuteScopeGuard &) = delete;
  ElementwiseExecuteScopeGuard &
  operator=(const ElementwiseExecuteScopeGuard &) = delete;

  ~ElementwiseExecuteScopeGuard() { cleanup(); }

  void candidate_allocated() noexcept { phase_ = Phase::Candidate; }

  void completion_registered(const uintptr_t token) noexcept {
    token_ = token;
    phase_ = Phase::Registered;
  }

  void disarm() noexcept { phase_ = Phase::Disarmed; }

private:
  enum class Phase : uint8_t { Reserved, Candidate, Registered, Disarmed };

  void cleanup() noexcept {
    switch (phase_) {
    case Phase::Reserved:
      if (argmax_) {
        (void)rollback_reserved_argmax_submission(plan_, queue_, sink_);
      } else {
        (void)rollback_reserved_elementwise_submission(plan_, queue_, sink_);
      }
      break;
    case Phase::Candidate:
      if (candidate_ != nullptr && candidate_->get() != nullptr) {
        (void)rollback_unpublished_submission(
            *candidate_, *event_guard_,
            "elementwise execute unwound before completion registration",
            sink_);
      } else {
        if (argmax_) {
          (void)rollback_reserved_argmax_submission(plan_, queue_, sink_);
        } else {
          (void)rollback_reserved_elementwise_submission(plan_, queue_, sink_);
        }
      }
      break;
    case Phase::Registered:
      if (candidate_ != nullptr && candidate_->get() != nullptr) {
        (void)cleanup_failed_submission(
            *candidate_, token_, hipErrorUnknown,
            "elementwise execute unwound after completion registration", queue_,
            sink_);
      } else {
        poison_context(plan_->context);
      }
      break;
    case Phase::Disarmed:
      break;
    }
    phase_ = Phase::Disarmed;
  }

  ElementwisePlan *const plan_;
  Queue *const queue_;
  std::unique_ptr<Completion> *const candidate_;
  NativeEventGuard *const event_guard_;
  sllm_error_sink_t *const sink_;
  bool argmax_;
  uintptr_t token_;
  Phase phase_;
};

class AttentionPreprocessExecuteScopeGuard final {
public:
  AttentionPreprocessExecuteScopeGuard(
      AttentionPreprocessPlan *const plan, Queue *const queue,
      std::unique_ptr<Completion> *const candidate,
      NativeEventGuard *const event_guard,
      sllm_error_sink_t *const sink) noexcept
      : plan_(plan), queue_(queue), candidate_(candidate),
        event_guard_(event_guard), sink_(sink), token_(0U),
        phase_(Phase::Reserved) {}

  AttentionPreprocessExecuteScopeGuard(
      const AttentionPreprocessExecuteScopeGuard &) = delete;
  AttentionPreprocessExecuteScopeGuard &
  operator=(const AttentionPreprocessExecuteScopeGuard &) = delete;

  ~AttentionPreprocessExecuteScopeGuard() { cleanup(); }

  void candidate_allocated() noexcept { phase_ = Phase::Candidate; }

  void completion_registered(const uintptr_t token) noexcept {
    token_ = token;
    phase_ = Phase::Registered;
  }

  void disarm() noexcept { phase_ = Phase::Disarmed; }

private:
  enum class Phase : uint8_t { Reserved, Candidate, Registered, Disarmed };

  void cleanup() noexcept {
    switch (phase_) {
    case Phase::Reserved:
      (void)rollback_reserved_attention_submission(plan_, queue_, sink_);
      break;
    case Phase::Candidate:
      if (candidate_ != nullptr && candidate_->get() != nullptr) {
        (void)rollback_unpublished_submission(
            *candidate_, *event_guard_,
            "attention preprocess execute unwound before completion "
            "registration",
            sink_);
      } else {
        (void)rollback_reserved_attention_submission(plan_, queue_, sink_);
      }
      break;
    case Phase::Registered:
      if (candidate_ != nullptr && candidate_->get() != nullptr) {
        (void)cleanup_failed_submission(*candidate_, token_, hipErrorUnknown,
                                        "attention preprocess execute unwound "
                                        "after completion registration",
                                        queue_, sink_);
      } else {
        poison_context(plan_->context);
      }
      break;
    case Phase::Disarmed:
      break;
    }
    phase_ = Phase::Disarmed;
  }

  AttentionPreprocessPlan *const plan_;
  Queue *const queue_;
  std::unique_ptr<Completion> *const candidate_;
  NativeEventGuard *const event_guard_;
  sllm_error_sink_t *const sink_;
  uintptr_t token_;
  Phase phase_;
};

class KvStateExecuteScopeGuard final {
public:
  KvStateExecuteScopeGuard(KvState *const state, Queue *const queue,
                           Buffer *const key_input, Buffer *const value_input,
                           std::unique_ptr<Completion> *const candidate,
                           NativeEventGuard *const event_guard,
                           sllm_error_sink_t *const sink) noexcept
      : state_(state), queue_(queue), key_input_(key_input),
        value_input_(value_input), candidate_(candidate),
        event_guard_(event_guard), sink_(sink), token_(0U),
        phase_(Phase::Reserved) {}

  KvStateExecuteScopeGuard(const KvStateExecuteScopeGuard &) = delete;
  KvStateExecuteScopeGuard &
  operator=(const KvStateExecuteScopeGuard &) = delete;

  ~KvStateExecuteScopeGuard() { cleanup(); }

  void candidate_allocated() noexcept { phase_ = Phase::Candidate; }

  void completion_registered(const uintptr_t token) noexcept {
    token_ = token;
    phase_ = Phase::Registered;
  }

  void disarm() noexcept { phase_ = Phase::Disarmed; }

private:
  enum class Phase : uint8_t { Reserved, Candidate, Registered, Disarmed };

  void cleanup() noexcept {
    switch (phase_) {
    case Phase::Reserved:
      (void)rollback_reserved_kv_submission(state_, queue_, key_input_,
                                            value_input_, sink_);
      break;
    case Phase::Candidate:
      if (candidate_ != nullptr && candidate_->get() != nullptr) {
        (void)rollback_unpublished_submission(
            *candidate_, *event_guard_,
            "KV append unwound before completion registration", sink_);
      } else {
        (void)rollback_reserved_kv_submission(state_, queue_, key_input_,
                                              value_input_, sink_);
      }
      break;
    case Phase::Registered:
      if (candidate_ != nullptr && candidate_->get() != nullptr) {
        (void)cleanup_failed_submission(*candidate_, token_, hipErrorUnknown,
                                        "KV append unwound after completion "
                                        "registration",
                                        queue_, sink_);
      } else {
        poison_context(state_->context);
      }
      break;
    case Phase::Disarmed:
      break;
    }
    phase_ = Phase::Disarmed;
  }

  KvState *const state_;
  Queue *const queue_;
  Buffer *const key_input_;
  Buffer *const value_input_;
  std::unique_ptr<Completion> *const candidate_;
  NativeEventGuard *const event_guard_;
  sllm_error_sink_t *const sink_;
  uintptr_t token_;
  Phase phase_;
};

sllm_status_t submit_copy(const sllm_queue_t *const raw_queue,
                          const sllm_buffer_t *const raw_buffer,
                          const sllm_transfer_desc_t *const transfer,
                          sllm_completion_t **const completion_output,
                          const bool d2h, sllm_error_sink_t *const sink) {
  if (completion_output != nullptr) {
    *completion_output = nullptr;
  }
  if (raw_queue == nullptr || raw_buffer == nullptr ||
      completion_output == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "copy queue, buffer, or completion output is null");
  }
  const sllm_status_t transfer_status =
      sllm_public_runtime::validate_transfer_desc(transfer, sink);
  if (transfer_status != SLLM_STATUS_OK) {
    return transfer_status;
  }
  if (!d2h && transfer->host_pointer == nullptr) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "H2D transfer host pointer is null");
  }
  std::vector<uint8_t> host_storage(
      static_cast<std::size_t>(transfer->size_bytes));
  if (!d2h) {
    std::memcpy(host_storage.data(), transfer->host_pointer,
                static_cast<std::size_t>(transfer->size_bytes));
  }
  Queue *queue = nullptr;
  Buffer *buffer = nullptr;
  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    queue = lookup<Queue>(raw_queue, HandleKind::Queue);
    buffer = lookup<Buffer>(raw_buffer, HandleKind::Buffer);
    if (queue == nullptr || buffer == nullptr) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "copy queue or buffer handle is stale or has the wrong kind");
    }
    if (queue->release_active || buffer->release_active) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_PUBLIC_BUSY,
          "copy queue or buffer release is already in progress");
    }
    if (queue->context->poisoned.load()) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INTERNAL_ERROR,
          "copy context is poisoned by an unresolved cleanup failure");
    }
    const sllm_status_t pair_status =
        validate_device_handle_pair(queue, buffer, sink);
    if (pair_status != SLLM_STATUS_OK) {
      return pair_status;
    }
    if (sllm_public_runtime::add_overflows(transfer->buffer_offset_bytes,
                                           transfer->size_bytes) ||
        transfer->buffer_offset_bytes + transfer->size_bytes >
            buffer->size_bytes) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INVALID_ARGUMENT,
          "copy range is outside the device buffer");
    }
    if (!reserve_submission_references(queue, buffer)) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INTERNAL_ERROR,
          "public submission accounting is exhausted");
    }
  }

  std::unique_ptr<Completion> candidate;
  try {
    candidate = std::make_unique<Completion>(queue->context, queue, buffer,
                                             transfer->size_bytes, d2h,
                                             std::move(host_storage));
  } catch (...) {
    if (!rollback_reserved_submission(queue->context, queue, buffer, sink)) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INTERNAL_ERROR,
          "public completion allocation rollback failed before enqueue");
    }
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INTERNAL_ERROR,
        "public completion allocation failed before enqueue");
  }

  const sllm_status_t device_status =
      select_context_device(queue->context, sink);
  if (device_status != SLLM_STATUS_OK) {
    if (!rollback_submission_references(candidate.get())) {
      candidate->orphaned = true;
      retain_poisoned(candidate, candidate->context);
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INTERNAL_ERROR,
          "submission rollback failed after device selection failure");
    }
    return device_status;
  }

  hipEvent_t native_event = nullptr;
  const hipError_t event_status =
      hipEventCreateWithFlags(&native_event, hipEventDisableTiming);
  if (event_status != hipSuccess) {
    if (!rollback_submission_references(candidate.get())) {
      candidate->orphaned = true;
      retain_poisoned(candidate, candidate->context);
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_INTERNAL_ERROR,
          "submission rollback failed after event creation failure");
    }
    return hip_failure(sink, event_status, "hipEventCreateWithFlags");
  }
  NativeEventGuard event_guard(queue->context, native_event);
  candidate->event = native_event;

  uintptr_t token = 0U;
  try {
    std::lock_guard<std::mutex> lock(registry_mutex);
    token = register_handle(candidate.get(), HandleKind::Completion);
  } catch (...) {
    return rollback_unpublished_submission(
        candidate, event_guard,
        "public completion registry allocation failed before enqueue", sink);
  }
  if (token == 0U) {
    return rollback_unpublished_submission(
        candidate, event_guard,
        "public completion handle token allocation failed", sink);
  }
  event_guard.release();
  void *const destination =
      static_cast<char *>(buffer->device_pointer) +
      static_cast<std::size_t>(transfer->buffer_offset_bytes);
  void *const source = candidate->host_storage.data();
  const hipMemcpyKind direction =
      d2h ? hipMemcpyDeviceToHost : hipMemcpyHostToDevice;
  const hipError_t copy_status = hipMemcpyAsync(
      d2h ? source : destination, d2h ? destination : source,
      static_cast<std::size_t>(transfer->size_bytes), direction, queue->stream);
  if (copy_status != hipSuccess) {
    return cleanup_failed_submission(candidate, token, copy_status,
                                     "hipMemcpyAsync", queue, sink);
  }
  const hipError_t record_status =
      hipEventRecord(candidate->event, queue->stream);
  if (record_status != hipSuccess) {
    return cleanup_failed_submission(candidate, token, record_status,
                                     "hipEventRecord", queue, sink);
  }
  *completion_output = reinterpret_cast<sllm_completion_t *>(token);
  (void)candidate.release();
  return SLLM_STATUS_OK;
}

} // namespace

extern "C" sllm_status_t
sllm_get_abi_version(uint32_t *const abi_version,
                     sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (abi_version == nullptr) {
      return sllm_public_runtime::write_error(error_sink,
                                              SLLM_STATUS_INVALID_ARGUMENT,
                                              "abi_version output is null");
    }
    *abi_version = SLLM_HIP_ABI_VERSION;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in ABI version query");
  }
}

extern "C" sllm_status_t
sllm_query_version(sllm_version_info_t *const version,
                   sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t struct_status = sllm_public_runtime::validate_struct(
        version, error_sink, "version output is null");
    if (struct_status != SLLM_STATUS_OK) {
      return struct_status;
    }
    if (version->reserved[0] != 0U || version->reserved[1] != 0U ||
        version->reserved[2] != 0U) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_RESERVED_NONZERO,
          "version reserved fields must be zero");
    }
    version->major = SLLM_HIP_LIBRARY_VERSION_MAJOR;
    version->minor = SLLM_HIP_LIBRARY_VERSION_MINOR;
    version->patch = SLLM_HIP_LIBRARY_VERSION_PATCH;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in version query");
  }
}

extern "C" sllm_status_t
sllm_backend_probe(const uint32_t backend,
                   sllm_backend_probe_result_t *const result,
                   sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t result_status =
        validate_backend_result(result, error_sink);
    if (result_status != SLLM_STATUS_OK) {
      return result_status;
    }
    if (backend != SLLM_BACKEND_HIP) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_UNSUPPORTED, "unknown backend identifier");
    }
    uint32_t count = 0U;
    const sllm_status_t count_status = get_device_count(&count, error_sink);
    if (count_status != SLLM_STATUS_OK) {
      result->backend = backend;
      result->available = 0U;
      result->hip_runtime_present = 0U;
      return count_status;
    }
    result->backend = backend;
    result->available = count == 0U ? 0U : 1U;
    result->hip_runtime_present = 1U;
    return count == 0U ? sllm_public_runtime::write_error(
                             error_sink, SLLM_STATUS_PUBLIC_NOT_READY,
                             "HIP runtime is present but no device is visible")
                       : SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in backend probe");
  }
}

extern "C" sllm_status_t
sllm_context_probe(const sllm_context_t *const raw_context,
                   sllm_context_probe_result_t *const result,
                   sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t result_status =
        validate_context_probe_result(result, error_sink);
    if (result_status != SLLM_STATUS_OK) {
      return result_status;
    }
    if (raw_context != nullptr) {
      std::lock_guard<std::mutex> lock(registry_mutex);
      if (lookup<Context>(raw_context, HandleKind::Context) == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "context probe handle is stale or has the wrong kind");
      }
    }
    uint32_t count = 0U;
    const sllm_status_t count_status = get_device_count(&count, error_sink);
    result->context_present = raw_context == nullptr ? 0U : 1U;
    result->hip_available =
        count_status == SLLM_STATUS_OK && count != 0U ? 1U : 0U;
    return count_status;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in context probe");
  }
}

extern "C" sllm_status_t
sllm_device_count(uint32_t *const count,
                  sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (count == nullptr) {
      return sllm_public_runtime::write_error(error_sink,
                                              SLLM_STATUS_INVALID_ARGUMENT,
                                              "device count output is null");
    }
    *count = 0U;
    return get_device_count(count, error_sink);
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in public device count");
  }
}

extern "C" sllm_status_t
sllm_device_query(const uint32_t device_index, sllm_device_info_t *const info,
                  sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t info_status = validate_device_info(info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    uint32_t count = 0U;
    const sllm_status_t count_status = get_device_count(&count, error_sink);
    if (count_status != SLLM_STATUS_OK) {
      return count_status;
    }
    if (device_index >= count) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "device index is outside the visible HIP device set");
    }
    hipDeviceProp_t properties = {};
    const sllm_status_t property_status =
        get_device_properties(device_index, &properties, error_sink);
    if (property_status != SLLM_STATUS_OK) {
      return property_status;
    }
    if (!copy_property(info->name, SLLM_HIP_MAX_DEVICE_NAME, properties.name,
                       sizeof(properties.name)) ||
        !copy_property(info->gcn_arch_name, SLLM_HIP_MAX_GCN_ARCH_NAME,
                       properties.gcnArchName,
                       sizeof(properties.gcnArchName))) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
          "HIP device property string is not NUL terminated or is too long");
    }
    initialize_device_info(info);
    info->device_index = device_index;
    info->visible_device_count = count;
    info->total_memory_bytes = static_cast<uint64_t>(properties.totalGlobalMem);
    info->wavefront_size = static_cast<uint32_t>(properties.warpSize);
    (void)copy_property(info->name, SLLM_HIP_MAX_DEVICE_NAME, properties.name,
                        sizeof(properties.name));
    (void)copy_property(info->gcn_arch_name, SLLM_HIP_MAX_GCN_ARCH_NAME,
                        properties.gcnArchName, sizeof(properties.gcnArchName));
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in public device query");
  }
}

extern "C" sllm_status_t
sllm_context_create(const sllm_context_create_info_t *const info,
                    sllm_context_t **const raw_context,
                    sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (raw_context != nullptr) {
      *raw_context = nullptr;
    }
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t info_status =
        sllm_public_runtime::validate_context_create_info(info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    if (raw_context == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT, "context output is null");
    }
    uint32_t count = 0U;
    const sllm_status_t count_status = get_device_count(&count, error_sink);
    if (count_status != SLLM_STATUS_OK) {
      return count_status;
    }
    if (info->device_index >= count) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "context device index is outside the visible HIP device set");
    }
    hipDeviceProp_t properties = {};
    const sllm_status_t property_status =
        get_device_properties(info->device_index, &properties, error_sink);
    if (property_status != SLLM_STATUS_OK) {
      return property_status;
    }
    std::size_t expected_length = 0U;
    (void)sllm_public_runtime::valid_arch_name(info->expected_gcn_arch_name,
                                               SLLM_HIP_MAX_GCN_ARCH_NAME,
                                               &expected_length);
    const void *const actual_terminator = std::memchr(
        properties.gcnArchName, '\0', sizeof(properties.gcnArchName));
    if (actual_terminator == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
          "HIP gcnArchName is not NUL terminated");
    }
    const std::size_t actual_length = static_cast<std::size_t>(
        static_cast<const char *>(actual_terminator) - properties.gcnArchName);
    if (actual_length != expected_length ||
        std::memcmp(info->expected_gcn_arch_name, properties.gcnArchName,
                    expected_length) != 0) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
          "requested device gcnArchName does not match exactly");
    }
    const hipError_t set_status =
        hipSetDevice(static_cast<int>(info->device_index));
    if (set_status != hipSuccess) {
      return hip_failure(error_sink, set_status, "hipSetDevice");
    }
    std::unique_ptr<Context> candidate(new Context(info->device_index));
    {
      std::lock_guard<std::mutex> lock(registry_mutex);
      uintptr_t token = 0U;
      try {
        token = register_handle(candidate.get(), HandleKind::Context);
      } catch (...) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "public context handle registry allocation failed");
      }
      if (token == 0U) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "public context handle token allocation failed");
      }
      *raw_context = reinterpret_cast<sllm_context_t *>(token);
    }
    (void)candidate.release();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in public context create");
  }
}

extern "C" sllm_status_t
sllm_context_release(sllm_context_t **const raw_context,
                     sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_context == nullptr || *raw_context == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT, "context handle is null");
    }
    Context *context = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      context = lookup<Context>(*raw_context, HandleKind::Context);
      if (context == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "context handle is stale or has the wrong kind");
      }
      if (context->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "context release is already in progress");
      }
      {
        std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
        if (context->poisoned.load()) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_INTERNAL_ERROR,
              "context is poisoned by an unresolved cleanup failure");
        }
        if (context->accounting.child_count != 0U ||
            context->accounting.lifetime_guards != 0U) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_PUBLIC_BUSY,
              "context cannot be released while child resources or cleanup "
              "guards exist");
        }
        context->release_active = true;
      }
    }
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      if (lookup<Context>(*raw_context, HandleKind::Context) != context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "context release handle changed during release");
      }
      unregister_handle(*raw_context);
      delete context;
      *raw_context = nullptr;
    }
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in public context release");
  }
}

extern "C" sllm_status_t
sllm_queue_create(const sllm_context_t *const raw_context,
                  const sllm_queue_create_info_t *const info,
                  sllm_queue_t **const raw_queue,
                  sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (raw_queue != nullptr) {
      *raw_queue = nullptr;
    }
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t info_status =
        sllm_public_runtime::validate_queue_create_info(info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    if (raw_context == nullptr || raw_queue == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "queue context or output is null");
    }
    Context *context = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      context = lookup<Context>(raw_context, HandleKind::Context);
      if (context == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "queue context handle is stale or has the wrong kind");
      }
      if (context->poisoned.load()) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "queue context is poisoned by an unresolved cleanup failure");
      }
      if (context->release_active || !reserve_child(context)) {
        return sllm_public_runtime::write_error(
            error_sink,
            context->release_active ? SLLM_STATUS_PUBLIC_BUSY
                                    : SLLM_STATUS_INTERNAL_ERROR,
            context->release_active
                ? "queue context release is already in progress"
                : "public context child accounting is exhausted");
      }
    }
    const sllm_status_t device_status =
        select_context_device(context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      if (!rollback_child(context, error_sink,
                          "queue child accounting rollback failed")) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return device_status;
    }
    hipStream_t stream = nullptr;
    const hipError_t stream_status =
        sllm_public_runtime::FaultInjector::consume(
            sllm_public_runtime::FaultPoint::NativeCreationFailure)
            ? hipErrorUnknown
            : hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    if (stream_status != hipSuccess) {
      if (!rollback_child(context, error_sink,
                          "queue child accounting rollback failed")) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return hip_failure(error_sink, stream_status, "hipStreamCreateWithFlags");
    }
    NativeStreamGuard stream_guard(context, stream);
    std::unique_ptr<Queue> candidate;
    if (sllm_public_runtime::FaultInjector::consume(
            sllm_public_runtime::FaultPoint::ConstructionCandidateFailure)) {
      const hipError_t cleanup_status = stream_guard.cleanup();
      if (cleanup_status != hipSuccess) {
        return hip_failure(error_sink, cleanup_status,
                           "hipStreamDestroy injected queue rollback");
      }
      if (!rollback_child(context, error_sink,
                          "queue child accounting rollback failed")) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "injected queue candidate construction failure");
    }
    try {
      candidate = std::make_unique<Queue>(context, stream);
    } catch (...) {
      const hipError_t cleanup_status = stream_guard.cleanup();
      if (cleanup_status != hipSuccess) {
        return hip_failure(error_sink, cleanup_status,
                           "hipStreamDestroy queue construction rollback");
      }
      if (!rollback_child(context, error_sink,
                          "queue child accounting rollback failed")) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "public queue allocation failed after HIP stream creation");
    }
    uintptr_t token = 0U;
    try {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      token = register_handle(candidate.get(), HandleKind::Queue);
    } catch (...) {
      const hipError_t cleanup_status = stream_guard.cleanup();
      if (cleanup_status != hipSuccess) {
        return hip_failure(error_sink, cleanup_status,
                           "hipStreamDestroy queue registry rollback");
      }
      if (!rollback_child(context, error_sink,
                          "queue child accounting rollback failed")) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "public queue registry allocation failed");
    }
    if (token == 0U) {
      const hipError_t cleanup_status = stream_guard.cleanup();
      if (cleanup_status != hipSuccess) {
        return hip_failure(error_sink, cleanup_status,
                           "hipStreamDestroy queue token rollback");
      }
      if (!rollback_child(context, error_sink,
                          "queue child accounting rollback failed")) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "public queue handle token allocation failed");
    }
    *raw_queue = reinterpret_cast<sllm_queue_t *>(token);
    stream_guard.release();
    (void)candidate.release();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in public queue create");
  }
}

extern "C" sllm_status_t
sllm_queue_release(sllm_queue_t **const raw_queue,
                   sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_queue == nullptr || *raw_queue == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT, "queue handle is null");
    }
    Queue *queue = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      queue = lookup<Queue>(*raw_queue, HandleKind::Queue);
      if (queue == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "queue handle is stale or has the wrong kind");
      }
      if (queue->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "queue release is already in progress");
      }
      std::lock_guard<std::mutex> accounting_lock(
          queue->context->accounting_mutex);
      if (queue->context->poisoned.load()) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "queue context is poisoned by an unresolved cleanup failure");
      }
      if (queue->accounting.active_submissions != 0U ||
          queue->accounting.completion_references != 0U) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "queue cannot be released while completion references exist");
      }
      if (!sllm_public_runtime::AccountingState::reserve_lifetime_guard(
              queue->context->accounting)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "queue cleanup guard accounting is exhausted");
      }
      queue->release_active = true;
    }
    const sllm_status_t device_status =
        select_context_device(queue->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      (void)release_lifetime_guard(queue->context);
      queue->release_active = false;
      return device_status;
    }
    const hipError_t destroy_status =
        destroy_stream_with_fault_injection(queue->stream);
    if (destroy_status != hipSuccess) {
      {
        std::lock_guard<std::mutex> registry_lock(registry_mutex);
        unregister_handle(*raw_queue);
        *raw_queue = nullptr;
      }
      retain_poisoned(queue, queue->context);
      return hip_failure(error_sink, destroy_status, "hipStreamDestroy");
    }
    queue->stream = nullptr;
    bool queue_handle_matches = false;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      queue_handle_matches =
          lookup<Queue>(*raw_queue, HandleKind::Queue) == queue;
      unregister_handle(*raw_queue);
      *raw_queue = nullptr;
    }
    if (!queue_handle_matches) {
      retain_poisoned(queue, queue->context);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "queue release handle changed during release");
    }
    if (!release_child_and_lifetime_guard(queue->context)) {
      retain_poisoned(queue, queue->context);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "queue context accounting release failed; resource quarantined");
    }
    delete queue;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in public queue release");
  }
}

extern "C" sllm_status_t
sllm_buffer_create(const sllm_context_t *const raw_context,
                   const sllm_buffer_create_info_t *const info,
                   sllm_buffer_t **const raw_buffer,
                   sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (raw_buffer != nullptr) {
      *raw_buffer = nullptr;
    }
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t info_status =
        sllm_public_runtime::validate_buffer_create_info(info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    if (raw_context == nullptr || raw_buffer == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "buffer context or output is null");
    }
    Context *context = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      context = lookup<Context>(raw_context, HandleKind::Context);
      if (context == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "buffer context handle is stale or has the wrong kind");
      }
      if (context->poisoned.load()) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "buffer context is poisoned by an unresolved cleanup failure");
      }
      if (context->release_active || !reserve_child(context)) {
        return sllm_public_runtime::write_error(
            error_sink,
            context->release_active ? SLLM_STATUS_PUBLIC_BUSY
                                    : SLLM_STATUS_INTERNAL_ERROR,
            context->release_active
                ? "buffer context release is already in progress"
                : "public context child accounting is exhausted");
      }
    }
    const sllm_status_t device_status =
        select_context_device(context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      if (!rollback_child(context, error_sink,
                          "buffer child accounting rollback failed")) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return device_status;
    }
    void *device_pointer = nullptr;
    const hipError_t allocation_status =
        sllm_public_runtime::FaultInjector::consume(
            sllm_public_runtime::FaultPoint::NativeCreationFailure)
            ? hipErrorUnknown
            : hipMalloc(&device_pointer,
                        static_cast<std::size_t>(info->size_bytes));
    if (allocation_status != hipSuccess) {
      if (!rollback_child(context, error_sink,
                          "buffer child accounting rollback failed")) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return hip_failure(error_sink, allocation_status, "hipMalloc");
    }
    NativeAllocationGuard allocation_guard(context, device_pointer);
    std::unique_ptr<Buffer> candidate;
    if (sllm_public_runtime::FaultInjector::consume(
            sllm_public_runtime::FaultPoint::ConstructionCandidateFailure)) {
      const hipError_t cleanup_status = allocation_guard.cleanup();
      if (cleanup_status != hipSuccess) {
        return hip_failure(error_sink, cleanup_status,
                           "hipFree injected buffer rollback");
      }
      if (!rollback_child(context, error_sink,
                          "buffer child accounting rollback failed")) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "injected buffer candidate construction failure");
    }
    try {
      candidate =
          std::make_unique<Buffer>(context, device_pointer, info->size_bytes);
    } catch (...) {
      const hipError_t cleanup_status = allocation_guard.cleanup();
      if (cleanup_status != hipSuccess) {
        return hip_failure(error_sink, cleanup_status,
                           "hipFree buffer construction rollback");
      }
      if (!rollback_child(context, error_sink,
                          "buffer child accounting rollback failed")) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "public buffer allocation failed after HIP allocation");
    }
    uintptr_t token = 0U;
    try {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      token = register_handle(candidate.get(), HandleKind::Buffer);
    } catch (...) {
      const hipError_t cleanup_status = allocation_guard.cleanup();
      if (cleanup_status != hipSuccess) {
        return hip_failure(error_sink, cleanup_status,
                           "hipFree buffer registry rollback");
      }
      if (!rollback_child(context, error_sink,
                          "buffer child accounting rollback failed")) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "public buffer registry allocation failed");
    }
    if (token == 0U) {
      const hipError_t cleanup_status = allocation_guard.cleanup();
      if (cleanup_status != hipSuccess) {
        return hip_failure(error_sink, cleanup_status,
                           "hipFree buffer token rollback");
      }
      if (!rollback_child(context, error_sink,
                          "buffer child accounting rollback failed")) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "public buffer handle token allocation failed");
    }
    *raw_buffer = reinterpret_cast<sllm_buffer_t *>(token);
    allocation_guard.release();
    (void)candidate.release();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in public buffer create");
  }
}

extern "C" sllm_status_t
sllm_buffer_release(sllm_buffer_t **const raw_buffer,
                    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_buffer == nullptr || *raw_buffer == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT, "buffer handle is null");
    }
    Buffer *buffer = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      buffer = lookup<Buffer>(*raw_buffer, HandleKind::Buffer);
      if (buffer == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "buffer handle is stale or has the wrong kind");
      }
      if (buffer->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "buffer release is already in progress");
      }
      std::lock_guard<std::mutex> accounting_lock(
          buffer->context->accounting_mutex);
      if (buffer->context->poisoned.load()) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "buffer context is poisoned by an unresolved cleanup failure");
      }
      if (buffer->accounting.active_submissions != 0U ||
          buffer->accounting.completion_references != 0U ||
          buffer->accounting.child_count != 0U) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "buffer cannot be released while completion or prepared-plan "
            "references exist");
      }
      if (!sllm_public_runtime::AccountingState::reserve_lifetime_guard(
              buffer->context->accounting)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "buffer cleanup guard accounting is exhausted");
      }
      buffer->release_active = true;
    }
    const sllm_status_t device_status =
        select_context_device(buffer->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      (void)release_lifetime_guard(buffer->context);
      buffer->release_active = false;
      return device_status;
    }
    const hipError_t free_status =
        free_allocation_with_fault_injection(buffer->device_pointer);
    if (free_status != hipSuccess) {
      {
        std::lock_guard<std::mutex> registry_lock(registry_mutex);
        unregister_handle(*raw_buffer);
        *raw_buffer = nullptr;
      }
      retain_poisoned(buffer, buffer->context);
      return hip_failure(error_sink, free_status, "hipFree");
    }
    buffer->device_pointer = nullptr;
    bool buffer_handle_matches = false;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      buffer_handle_matches =
          lookup<Buffer>(*raw_buffer, HandleKind::Buffer) == buffer;
      unregister_handle(*raw_buffer);
      *raw_buffer = nullptr;
    }
    if (!buffer_handle_matches) {
      retain_poisoned(buffer, buffer->context);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "buffer release handle changed during release");
    }
    if (!release_child_and_lifetime_guard(buffer->context)) {
      retain_poisoned(buffer, buffer->context);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "buffer context accounting release failed; resource quarantined");
    }
    delete buffer;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in public buffer release");
  }
}

extern "C" sllm_status_t
sllm_buffer_size(const sllm_buffer_t *const raw_buffer,
                 uint64_t *const size_bytes,
                 sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (size_bytes == nullptr) {
      return sllm_public_runtime::write_error(error_sink,
                                              SLLM_STATUS_INVALID_ARGUMENT,
                                              "buffer size output is null");
    }
    std::lock_guard<std::mutex> lock(registry_mutex);
    const Buffer *const buffer = lookup<Buffer>(raw_buffer, HandleKind::Buffer);
    if (buffer == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "buffer handle is stale or has the wrong kind");
    }
    if (buffer->release_active) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "buffer release is already in progress");
    }
    *size_bytes = buffer->size_bytes;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in public buffer size");
  }
}

extern "C" sllm_status_t
sllm_event_create(const sllm_context_t *const raw_context,
                  sllm_event_t **const raw_event,
                  sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (raw_event != nullptr) {
      *raw_event = nullptr;
    }
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_context == nullptr || raw_event == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "event context or output is null");
    }
    Context *context = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      context = lookup<Context>(raw_context, HandleKind::Context);
      if (context == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "event context handle is stale or has the wrong kind");
      }
      if (context->poisoned.load()) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "event context is poisoned by an unresolved cleanup failure");
      }
      if (context->release_active || !reserve_child(context)) {
        return sllm_public_runtime::write_error(
            error_sink,
            context->release_active ? SLLM_STATUS_PUBLIC_BUSY
                                    : SLLM_STATUS_INTERNAL_ERROR,
            context->release_active
                ? "event context release is already in progress"
                : "public context child accounting is exhausted");
      }
    }
    const sllm_status_t device_status =
        select_context_device(context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      if (!rollback_child(context, error_sink,
                          "event child accounting rollback failed")) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return device_status;
    }
    hipEvent_t native_event = nullptr;
    const hipError_t create_status =
        sllm_public_runtime::FaultInjector::consume(
            sllm_public_runtime::FaultPoint::NativeCreationFailure)
            ? hipErrorUnknown
            : hipEventCreateWithFlags(&native_event, hipEventDisableTiming);
    if (create_status != hipSuccess) {
      if (!rollback_child(context, error_sink,
                          "event child accounting rollback failed")) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return hip_failure(error_sink, create_status, "hipEventCreateWithFlags");
    }
    NativeEventGuard event_guard(context, native_event);
    std::unique_ptr<Event> candidate;
    if (sllm_public_runtime::FaultInjector::consume(
            sllm_public_runtime::FaultPoint::ConstructionCandidateFailure)) {
      const hipError_t cleanup_status = event_guard.cleanup();
      if (cleanup_status != hipSuccess) {
        return hip_failure(error_sink, cleanup_status,
                           "hipEventDestroy injected event rollback");
      }
      if (!rollback_child(context, error_sink,
                          "event child accounting rollback failed")) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "injected event candidate construction failure");
    }
    try {
      candidate = std::make_unique<Event>(context, native_event);
    } catch (...) {
      const hipError_t cleanup_status = event_guard.cleanup();
      if (cleanup_status != hipSuccess) {
        return hip_failure(error_sink, cleanup_status,
                           "hipEventDestroy event construction rollback");
      }
      if (!rollback_child(context, error_sink,
                          "event child accounting rollback failed")) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "public event allocation failed after HIP event creation");
    }
    uintptr_t token = 0U;
    try {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      token = register_handle(candidate.get(), HandleKind::Event);
    } catch (...) {
      const hipError_t cleanup_status = event_guard.cleanup();
      if (cleanup_status != hipSuccess) {
        return hip_failure(error_sink, cleanup_status,
                           "hipEventDestroy event registry rollback");
      }
      if (!rollback_child(context, error_sink,
                          "event child accounting rollback failed")) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "public event registry allocation failed");
    }
    if (token == 0U) {
      const hipError_t cleanup_status = event_guard.cleanup();
      if (cleanup_status != hipSuccess) {
        return hip_failure(error_sink, cleanup_status,
                           "hipEventDestroy event token rollback");
      }
      if (!rollback_child(context, error_sink,
                          "event child accounting rollback failed")) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "public event handle token allocation failed");
    }
    *raw_event = reinterpret_cast<sllm_event_t *>(token);
    event_guard.release();
    (void)candidate.release();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in public event create");
  }
}

extern "C" sllm_status_t
sllm_event_release(sllm_event_t **const raw_event,
                   sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_event == nullptr || *raw_event == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT, "event handle is null");
    }
    Event *event = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      event = lookup<Event>(*raw_event, HandleKind::Event);
      if (event == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "event handle is stale or has the wrong kind");
      }
      if (event->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "event release is already in progress");
      }
      std::lock_guard<std::mutex> accounting_lock(
          event->context->accounting_mutex);
      if (event->context->poisoned.load()) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "event context is poisoned by an unresolved cleanup failure");
      }
      if (!sllm_public_runtime::AccountingState::reserve_lifetime_guard(
              event->context->accounting)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "event cleanup guard accounting is exhausted");
      }
      event->release_active = true;
    }
    const sllm_status_t device_status =
        select_context_device(event->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      (void)release_lifetime_guard(event->context);
      event->release_active = false;
      return device_status;
    }
    const hipError_t destroy_status =
        destroy_event_with_fault_injection(event->event);
    if (destroy_status != hipSuccess) {
      {
        std::lock_guard<std::mutex> registry_lock(registry_mutex);
        unregister_handle(*raw_event);
        *raw_event = nullptr;
      }
      retain_poisoned(event, event->context);
      return hip_failure(error_sink, destroy_status, "hipEventDestroy");
    }
    event->event = nullptr;
    bool event_handle_matches = false;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      event_handle_matches =
          lookup<Event>(*raw_event, HandleKind::Event) == event;
      unregister_handle(*raw_event);
      *raw_event = nullptr;
    }
    if (!event_handle_matches) {
      retain_poisoned(event, event->context);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "event release handle changed during release");
    }
    if (!release_child_and_lifetime_guard(event->context)) {
      retain_poisoned(event, event->context);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "event context accounting release failed; resource quarantined");
    }
    delete event;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in public event release");
  }
}

extern "C" sllm_status_t
sllm_buffer_copy_h2d(const sllm_queue_t *const queue,
                     const sllm_buffer_t *const buffer,
                     const sllm_transfer_desc_t *const transfer,
                     sllm_completion_t **const completion,
                     sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    return submit_copy(queue, buffer, transfer, completion, false, error_sink);
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in public H2D copy");
  }
}

extern "C" sllm_status_t
sllm_buffer_copy_d2h(const sllm_queue_t *const queue,
                     const sllm_buffer_t *const buffer,
                     const sllm_transfer_desc_t *const transfer,
                     sllm_completion_t **const completion,
                     sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    return submit_copy(queue, buffer, transfer, completion, true, error_sink);
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in public D2H copy");
  }
}

extern "C" sllm_status_t
sllm_completion_query(sllm_completion_t *const raw_completion,
                      sllm_completion_result_t *const result,
                      sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t result_status =
        sllm_public_runtime::validate_completion_result(result, error_sink);
    if (result_status != SLLM_STATUS_OK) {
      return result_status;
    }
    Completion *completion = nullptr;
    const sllm_status_t pin_status =
        pin_completion(raw_completion, &completion, error_sink);
    if (pin_status != SLLM_STATUS_OK) {
      return pin_status;
    }
    CompletionPin pin(completion);
    std::unique_lock<std::mutex> state_lock(completion->state_mutex);
    const sllm_status_t status = poll_completion(completion, error_sink);
    if (completion->terminal && !completion->safe_to_release) {
      state_lock.unlock();
      quarantine_completion(raw_completion, completion);
      state_lock.lock();
    }
    fill_completion_result(completion, result);
    return status;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in public completion query");
  }
}

extern "C" sllm_status_t
sllm_completion_wait(sllm_completion_t *const raw_completion,
                     const uint32_t timeout_ms,
                     sllm_completion_result_t *const result,
                     sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t result_status =
        sllm_public_runtime::validate_completion_result(result, error_sink);
    if (result_status != SLLM_STATUS_OK) {
      return result_status;
    }
    Completion *completion = nullptr;
    const sllm_status_t pin_status =
        pin_completion(raw_completion, &completion, error_sink);
    if (pin_status != SLLM_STATUS_OK) {
      return pin_status;
    }
    CompletionPin pin(completion);
    std::unique_lock<std::mutex> state_lock(completion->state_mutex);
    if (completion->wait_active) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "another completion wait is already active");
    }
    WaitActive wait_active(completion);
    const auto started = std::chrono::steady_clock::now();
    for (;;) {
      const sllm_status_t status = poll_completion(completion, error_sink);
      if (completion->terminal && !completion->safe_to_release) {
        state_lock.unlock();
        quarantine_completion(raw_completion, completion);
        state_lock.lock();
      }
      if (status != SLLM_STATUS_PUBLIC_PENDING) {
        fill_completion_result(completion, result);
        return status;
      }
      const auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - started);
      if (timeout_ms != UINT32_MAX &&
          elapsed.count() >= static_cast<int64_t>(timeout_ms)) {
        fill_completion_result(completion, result);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_TIMEOUT,
            "public completion wait timed out");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in public completion wait");
  }
}

extern "C" sllm_status_t sllm_completion_read(
    sllm_completion_t *const raw_completion, void *const destination,
    const uint64_t destination_capacity, uint64_t *const bytes_written,
    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (bytes_written == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "completion bytes-written output is null");
    }
    *bytes_written = 0U;
    Completion *completion = nullptr;
    const sllm_status_t pin_status =
        pin_completion(raw_completion, &completion, error_sink);
    if (pin_status != SLLM_STATUS_OK) {
      return pin_status;
    }
    CompletionPin pin(completion);
    std::unique_lock<std::mutex> state_lock(completion->state_mutex);
    if (!completion->terminal) {
      return sllm_public_runtime::write_error(error_sink,
                                              SLLM_STATUS_PUBLIC_NOT_READY,
                                              "completion output is not ready");
    }
    if (!completion->success) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
          "completion failed and has no readable output");
    }
    if (!completion->d2h) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_UNSUPPORTED,
          "H2D completion has no host output");
    }
    if (destination_capacity < completion->host_storage.size()) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_BUFFER_TOO_SMALL,
          "completion output destination is too small");
    }
    if (destination == nullptr && !completion->host_storage.empty()) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "completion output destination is null");
    }
    std::memcpy(destination, completion->host_storage.data(),
                completion->host_storage.size());
    *bytes_written = static_cast<uint64_t>(completion->host_storage.size());
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in public completion read");
  }
}

extern "C" sllm_status_t
sllm_completion_timing(sllm_completion_t *const raw_completion,
                       sllm_completion_timing_t *const timing,
                       sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t timing_status =
        validate_completion_timing(timing, error_sink);
    if (timing_status != SLLM_STATUS_OK) {
      return timing_status;
    }
    const uint32_t struct_size = timing->struct_size;
    const uint32_t abi_version = timing->abi_version;
    std::memset(timing, 0, sizeof(*timing));
    timing->struct_size = struct_size;
    timing->abi_version = abi_version;
    if (raw_completion == nullptr) {
      return sllm_public_runtime::write_error(error_sink,
                                              SLLM_STATUS_INVALID_ARGUMENT,
                                              "completion handle is null");
    }
    Completion *completion = nullptr;
    const sllm_status_t pin_status =
        pin_completion(raw_completion, &completion, error_sink);
    if (pin_status != SLLM_STATUS_OK) {
      return pin_status;
    }
    CompletionPin pin(completion);
    std::unique_lock<std::mutex> state_lock(completion->state_mutex);
    const sllm_status_t completion_status =
        poll_completion(completion, error_sink);
    if (completion->terminal && !completion->safe_to_release) {
      state_lock.unlock();
      quarantine_completion(raw_completion, completion);
      return completion_status;
    }
    if (completion_status != SLLM_STATUS_OK) {
      return completion_status;
    }
    if (!completion->rmsnorm && !completion->elementwise &&
        !completion->matmul && !completion->argmax &&
        !completion->attention_preprocess && !completion->kv_state_append &&
        !completion->causal_attention && !completion->linear_attention) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_UNSUPPORTED,
          "completion timing is only available for numeric operations");
    }
    if (!completion->timing_valid || completion->timing_elapsed_ns == 0U) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
          "numeric operation completion has no valid HIP event timing");
    }
    timing->valid = 1U;
    timing->elapsed_ns = completion->timing_elapsed_ns;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in public completion timing");
  }
}

extern "C" sllm_status_t
sllm_completion_release(sllm_completion_t **const raw_completion,
                        sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_completion == nullptr || *raw_completion == nullptr) {
      return sllm_public_runtime::write_error(error_sink,
                                              SLLM_STATUS_INVALID_ARGUMENT,
                                              "completion handle is null");
    }
    Completion *completion = nullptr;
    hipEvent_t event_to_destroy = nullptr;
    hipEvent_t timing_event_to_destroy = nullptr;
    bool event_already_destroyed = false;
    bool timing_event_destroyed = false;
    bool guard_reservation_failed = false;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      completion = lookup<Completion>(*raw_completion, HandleKind::Completion);
      if (completion == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "completion handle is stale or has the wrong kind");
      }
      /* API pins are atomic so a release racing a pinned query can fail
       * immediately instead of waiting for the query's state lock while it
       * is inside a potentially blocking HIP call. */
      if (completion->api_pins.load(std::memory_order_acquire) != 0U) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "completion is pinned or release is already in progress");
      }
      std::unique_lock<std::mutex> state_lock(completion->state_mutex);
      if (completion->api_pins.load(std::memory_order_acquire) != 0U ||
          completion->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "completion is pinned or release is already in progress");
      }
      if (!completion->terminal && completion->kv_state_append &&
          !revoke_kv_append_commit(completion)) {
        completion->safe_to_release = false;
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "KV append revocation failed; context poisoned");
      }
      if (!completion->terminal && completion->linear_attention &&
          !revoke_linear_attention_commit(completion)) {
        completion->safe_to_release = false;
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "linear attention revocation failed; context poisoned");
      }
      if (!completion->terminal || !completion->safe_to_release ||
          !completion->safety.can_release_graph()) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "completion is not proven safe to release");
      }
      if (!completion->lifetime_guard_reserved) {
        guard_reservation_failed = true;
      } else {
        /* The submission reservation is the guard for this completion's
         * event.  Do not reserve a second guard: the matching release below
         * must consume exactly the reservation held since submission. */
        completion->lifetime_guard_reserved = true;
      }
      completion->release_active = true;
      event_already_destroyed = completion->event_destroyed;
      event_to_destroy = completion->event;
      timing_event_to_destroy = completion->timing_start_event;
    }

    if (guard_reservation_failed) {
      {
        std::lock_guard<std::mutex> state_lock(completion->state_mutex);
        completion->release_active = false;
        completion->safe_to_release = false;
        completion->safety.quarantine();
        completion->reference_accounting_failed = true;
      }
      quarantine_completion(*raw_completion, completion);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "completion cleanup guard accounting is exhausted; graph "
          "quarantined");
    }

    if (!event_already_destroyed || timing_event_to_destroy != nullptr) {
      const sllm_status_t device_status =
          select_context_device(completion->context, error_sink);
      if (device_status != SLLM_STATUS_OK) {
        std::lock_guard<std::mutex> registry_lock(registry_mutex);
        std::lock_guard<std::mutex> state_lock(completion->state_mutex);
        completion->release_active = false;
        return device_status;
      }
      if (!event_already_destroyed) {
        const hipError_t destroy_status =
            destroy_event_with_fault_injection(event_to_destroy);
        if (destroy_status != hipSuccess) {
          {
            std::lock_guard<std::mutex> registry_lock(registry_mutex);
            unregister_handle(*raw_completion);
            *raw_completion = nullptr;
          }
          {
            std::lock_guard<std::mutex> state_lock(completion->state_mutex);
            completion->safe_to_release = false;
            completion->release_active = false;
            completion->orphaned = true;
            completion->safety.quarantine();
          }
          retain_poisoned(completion, completion->context);
          /* ROCm 7.14 documents deferred destruction of incomplete events, but
           * does not document hipErrorNotReady as a non-consuming retry result.
           * Every destroy error is therefore ambiguous and is quarantined once.
           */
          return hip_failure(error_sink, destroy_status, "hipEventDestroy");
        }
      }
      if (timing_event_to_destroy != nullptr) {
        const hipError_t timing_destroy_status =
            destroy_event_with_fault_injection(timing_event_to_destroy);
        if (timing_destroy_status != hipSuccess) {
          {
            std::lock_guard<std::mutex> registry_lock(registry_mutex);
            unregister_handle(*raw_completion);
            *raw_completion = nullptr;
          }
          {
            std::lock_guard<std::mutex> state_lock(completion->state_mutex);
            completion->safe_to_release = false;
            completion->release_active = false;
            completion->orphaned = true;
            completion->safety.quarantine();
          }
          retain_poisoned(completion, completion->context);
          return hip_failure(error_sink, timing_destroy_status,
                             "hipEventDestroy timing event");
        }
        timing_event_destroyed = true;
      }
    }

    bool completion_handle_matches = false;
    bool event_destroy_observation_failed = false;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      completion_handle_matches =
          lookup<Completion>(*raw_completion, HandleKind::Completion) ==
          completion;
      std::unique_lock<std::mutex> state_lock(completion->state_mutex);
      if (!event_already_destroyed) {
        completion->event = nullptr;
        completion->event_destroyed = true;
        event_destroy_observation_failed =
            !completion->safety.observe_event_destroy_success();
      }
      if (timing_event_destroyed) {
        completion->timing_start_event = nullptr;
      }
      unregister_handle(*raw_completion);
      *raw_completion = nullptr;
    }
    if (!completion_handle_matches) {
      quarantine_completion(*raw_completion, completion);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "completion release handle changed during release");
    }
    if (event_destroy_observation_failed) {
      {
        std::lock_guard<std::mutex> state_lock(completion->state_mutex);
        completion->safe_to_release = false;
        completion->release_active = false;
        completion->orphaned = true;
        completion->safety.quarantine();
      }
      retain_poisoned(completion, completion->context);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "event destruction safety observation failed; graph quarantined");
    }

    bool references_released = false;
    {
      std::unique_lock<std::mutex> state_lock(completion->state_mutex);
      references_released = release_completion_child_reference(completion);
      completion->release_active = false;
      if (!references_released) {
        completion->safe_to_release = false;
        completion->safety.quarantine();
        completion->reference_accounting_failed = true;
      }
    }
    if (!references_released) {
      {
        std::lock_guard<std::mutex> state_lock(completion->state_mutex);
        completion->orphaned = true;
      }
      retain_poisoned(completion, completion->context);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "completion context reference accounting failed; resource "
          "quarantined");
    }
    completion->lifetime_guard_reserved = false;
    delete completion;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in public completion release");
  }
}

extern "C" sllm_status_t
sllm_rmsnorm_prepare(const sllm_context_t *const raw_context,
                     const sllm_rmsnorm_desc_t *const descriptor,
                     sllm_rmsnorm_plan_t **const raw_plan,
                     sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (raw_plan != nullptr) {
      *raw_plan = nullptr;
    }
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_context == nullptr || raw_plan == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "RMSNorm context or plan output is null");
    }
    if (descriptor == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_RMSNORM_DESCRIPTOR,
          "RMSNorm descriptor is null");
    }
    const sllm_status_t prefix_status =
        sllm_rmsnorm::validate_descriptor_prefix(descriptor, error_sink);
    if (prefix_status != SLLM_STATUS_OK) {
      return prefix_status;
    }
    const sllm_rmsnorm_desc_t descriptor_copy = *descriptor;
    sllm_rmsnorm::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_rmsnorm::validate_and_copy_descriptor(&descriptor_copy, &metadata,
                                                   error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }

    std::unique_ptr<RmsNormPlan> candidate;
    uintptr_t token = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      Context *const context =
          lookup<Context>(raw_context, HandleKind::Context);
      Buffer *const activation =
          lookup<Buffer>(descriptor_copy.activation.buffer, HandleKind::Buffer);
      Buffer *const raw_scale =
          lookup<Buffer>(descriptor_copy.raw_scale.buffer, HandleKind::Buffer);
      Buffer *const output =
          lookup<Buffer>(descriptor_copy.output.buffer, HandleKind::Buffer);
      if (context == nullptr || activation == nullptr || raw_scale == nullptr ||
          output == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "RMSNorm context or buffer handle is stale or has the wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (context->poisoned.load() || context->release_active ||
          activation->release_active || raw_scale->release_active ||
          output->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "RMSNorm cannot prepare while a context or buffer is releasing");
      }
      if (activation->context != context || raw_scale->context != context ||
          output->context != context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH,
            "RMSNorm buffers must belong to the supplied context and device");
      }
      const auto in_bounds = [](const sllm_rmsnorm::TensorMetadata &tensor,
                                const Buffer *const buffer) {
        return tensor.end_offset <= buffer->size_bytes;
      };
      if (!in_bounds(metadata.activation, activation) ||
          !in_bounds(metadata.raw_scale, raw_scale) ||
          !in_bounds(metadata.output, output)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
            "RMSNorm tensor interval exceeds its backing buffer");
      }
      const auto overlaps_if_same =
          [](const Buffer *const left_buffer,
             const sllm_rmsnorm::TensorMetadata &left,
             const Buffer *const right_buffer,
             const sllm_rmsnorm::TensorMetadata &right) {
            return left_buffer == right_buffer &&
                   sllm_rmsnorm::intervals_overlap(left, right);
          };
      if (overlaps_if_same(activation, metadata.activation, raw_scale,
                           metadata.raw_scale) ||
          overlaps_if_same(activation, metadata.activation, output,
                           metadata.output) ||
          overlaps_if_same(raw_scale, metadata.raw_scale, output,
                           metadata.output)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_ALIAS_OVERLAP,
            "RMSNorm tensor intervals overlap within one backing buffer");
      }
      if (!sllm_public_runtime::AccountingState::reserve_prepared_plan(
              context->accounting, activation->accounting,
              raw_scale->accounting, output->accounting)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "RMSNorm prepared-plan accounting is exhausted");
      }
      try {
        candidate = std::make_unique<RmsNormPlan>(context, activation,
                                                  raw_scale, output, metadata);
        token = register_handle(candidate.get(), HandleKind::RmsNormPlan);
      } catch (...) {
        (void)sllm_public_runtime::AccountingState::release_prepared_plan(
            context->accounting, activation->accounting, raw_scale->accounting,
            output->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "RMSNorm prepared-plan allocation or registration failed");
      }
      if (token == 0U) {
        (void)sllm_public_runtime::AccountingState::release_prepared_plan(
            context->accounting, activation->accounting, raw_scale->accounting,
            output->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "RMSNorm prepared-plan handle allocation failed");
      }
    }
    *raw_plan = reinterpret_cast<sllm_rmsnorm_plan_t *>(token);
    (void)candidate.release();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in RMSNorm prepare");
  }
}

extern "C" sllm_status_t
sllm_rmsnorm_plan_release(sllm_rmsnorm_plan_t **const raw_plan,
                          sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_plan == nullptr || *raw_plan == nullptr) {
      return sllm_public_runtime::write_error(error_sink,
                                              SLLM_STATUS_INVALID_ARGUMENT,
                                              "RMSNorm plan handle is null");
    }
    RmsNormPlan *plan = nullptr;
    bool quarantine_plan = false;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup<RmsNormPlan>(*raw_plan, HandleKind::RmsNormPlan);
      if (plan == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "RMSNorm plan handle is stale or has the wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      if (plan->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "RMSNorm plan release is already in progress");
      }
      if (plan->in_flight) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "RMSNorm plan has an in-flight completion");
      }
      plan->release_active = true;
      const bool accounting_released =
          !sllm_public_runtime::FaultInjector::consume(
              sllm_public_runtime::FaultPoint::AccountingFailure) &&
          sllm_public_runtime::AccountingState::release_prepared_plan(
              plan->context->accounting, plan->activation->accounting,
              plan->raw_scale->accounting, plan->output->accounting);
      if (!accounting_released) {
        /* The accounting reservation is intentionally not rolled back: its
         * exact dependency graph is now owned by the durable poison owner.
         * Consume the caller token while both registry and accounting locks
         * are held, then transfer the complete plan after unlocking. */
        plan->context->poisoned.store(true);
        unregister_handle(*raw_plan);
        *raw_plan = nullptr;
        quarantine_plan = true;
      } else {
        unregister_handle(*raw_plan);
        *raw_plan = nullptr;
      }
    }
    if (quarantine_plan) {
      /* A failed accounting transition is the only path that leaves a
       * recognized plan live here.  The token is already consumed, so no
       * caller retry can observe BUSY; retaining the plan preserves Context
       * and all three Buffer dependency pointers until process teardown. */
      poison_owner.retain(plan);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "RMSNorm prepared-plan accounting release failed; plan quarantined");
    }
    delete plan;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in RMSNorm plan release");
  }
}

extern "C" sllm_status_t
sllm_rmsnorm_execute(const sllm_rmsnorm_plan_t *const raw_plan,
                     const sllm_queue_t *const raw_queue,
                     sllm_completion_t **const completion_output,
                     sllm_rmsnorm_dispatch_info_t *const dispatch_info,
                     sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_plan == nullptr || raw_queue == nullptr ||
        completion_output == nullptr || dispatch_info == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "RMSNorm execute plan, queue, completion, or dispatch info is null");
    }
    const sllm_status_t dispatch_status =
        validate_rmsnorm_dispatch_info(dispatch_info, error_sink);
    if (dispatch_status != SLLM_STATUS_OK) {
      return dispatch_status;
    }

    RmsNormPlan *plan = nullptr;
    Queue *queue = nullptr;
    uint64_t row_count = 0U;
    uint64_t normalized_size = 0U;
    uint64_t dispatch_id = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup<RmsNormPlan>(raw_plan, HandleKind::RmsNormPlan);
      queue = lookup<Queue>(raw_queue, HandleKind::Queue);
      if (plan == nullptr || queue == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "RMSNorm execute plan or queue handle is stale or wrong kind");
      }
      if (plan->context != queue->context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
            "RMSNorm plan and queue belong to different contexts");
      }
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      if (plan->context->poisoned.load() || plan->context->release_active ||
          queue->release_active || plan->activation->release_active ||
          plan->raw_scale->release_active || plan->output->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "RMSNorm execute context, queue, buffer, or plan is releasing");
      }
      if (plan->in_flight) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "RMSNorm plan already has an in-flight completion");
      }
      const uint32_t rank = plan->metadata.activation.rank;
      normalized_size = plan->metadata.activation.shape[rank - 1U];
      row_count = 1U;
      for (uint32_t index = 0U; index + 1U < rank; ++index) {
        if (plan->metadata.activation.shape[index] != 0U &&
            row_count > std::numeric_limits<uint64_t>::max() /
                            plan->metadata.activation.shape[index]) {
          return sllm_public_runtime::write_error(
              error_sink, SLLM_STATUS_METADATA_OVERFLOW,
              "RMSNorm row count overflowed u64");
        }
        row_count *= plan->metadata.activation.shape[index];
      }
      if (normalized_size == 0U || normalized_size > SLLM_HIP_RMSNORM_MAX_N ||
          row_count == 0U || row_count > SLLM_HIP_RMSNORM_MAX_ROWS) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_UNSUPPORTED,
            "RMSNorm shape exceeds the baseline kernel launch contract");
      }
      if (!sllm_public_runtime::AccountingState::reserve_rmsnorm_submission(
              plan->context->accounting, queue->accounting,
              plan->activation->accounting, plan->raw_scale->accounting,
              plan->output->accounting)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "RMSNorm submission accounting is exhausted");
      }
      plan->in_flight = true;
      dispatch_id = plan->context->next_dispatch_id;
      if (dispatch_id == 0U) {
        plan->in_flight = false;
        (void)sllm_public_runtime::AccountingState::rollback_rmsnorm_submission(
            plan->context->accounting, queue->accounting,
            plan->activation->accounting, plan->raw_scale->accounting,
            plan->output->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "RMSNorm context dispatch id is exhausted");
      }
      if (dispatch_id == std::numeric_limits<uint64_t>::max()) {
        plan->context->next_dispatch_id = 0U;
      } else {
        ++plan->context->next_dispatch_id;
      }
      /* Preserve the legacy no-mutation behavior for validation failures, but
       * once accounting is reserved ensure every exceptional post-reservation
       * exit leaves no caller-visible completion token. */
      *completion_output = nullptr;
    }

    std::unique_ptr<Completion> candidate;
    NativeEventGuard event_guard;
    RmsNormExecuteScopeGuard execute_guard(plan, queue, &candidate,
                                           &event_guard, error_sink);
    throw_after_rmsnorm_reservation_if_requested();

    const sllm_status_t device_status =
        select_context_device(plan->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      if (!rollback_reserved_rmsnorm_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return device_status;
    }
    hipDeviceProp_t properties = {};
    const sllm_status_t property_status = get_device_properties(
        plan->context->device_index, &properties, error_sink);
    if (property_status != SLLM_STATUS_OK) {
      if (!rollback_reserved_rmsnorm_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return property_status;
    }
    char arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME] = {};
    if (!copy_property(arch_name, sizeof(arch_name), properties.gcnArchName,
                       sizeof(properties.gcnArchName))) {
      if (!rollback_reserved_rmsnorm_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
          "HIP gcnArchName is not NUL terminated or is too long");
    }
    if (properties.warpSize != 32) {
      if (!rollback_reserved_rmsnorm_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_UNSUPPORTED,
          "RMSNorm baseline requires a wave32 device");
    }
#if defined(SLLM_HIP_COMPILE_TARGET)
    if (std::strcmp(arch_name, SLLM_HIP_COMPILE_TARGET) != 0) {
      if (!rollback_reserved_rmsnorm_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
          "RMSNorm device target does not match the compiled exact target");
    }
#endif

    try {
      candidate = std::make_unique<Completion>(
          plan->context, queue, plan->activation, 0U, false,
          std::vector<uint8_t>{}, plan, plan->activation, plan->raw_scale,
          plan->output);
      execute_guard.candidate_allocated();
    } catch (...) {
      if (!rollback_reserved_rmsnorm_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "RMSNorm completion allocation failed before enqueue");
    }

    hipEvent_t native_event = nullptr;
    const hipError_t event_status = hipEventCreateWithFlags(&native_event, 0U);
    if (event_status != hipSuccess) {
      if (!rollback_reserved_rmsnorm_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return hip_failure(error_sink, event_status, "hipEventCreateWithFlags");
    }
    event_guard.adopt(plan->context, native_event);
    candidate->event = native_event;

    hipEvent_t timing_start_event = nullptr;
    const hipError_t timing_event_status =
        hipEventCreateWithFlags(&timing_start_event, 0U);
    if (timing_event_status != hipSuccess) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "RMSNorm timing event creation failed before enqueue", error_sink);
    }
    candidate->timing_start_event = timing_start_event;

    uintptr_t token = 0U;
    try {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      token = register_handle(candidate.get(), HandleKind::Completion);
    } catch (...) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "RMSNorm completion registry allocation failed before enqueue",
          error_sink);
    }
    if (token == 0U) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "RMSNorm completion handle token allocation failed", error_sink);
    }
    event_guard.release();
    execute_guard.completion_registered(token);
    throw_after_rmsnorm_registration_if_requested();

    const hipError_t timing_record_status =
        hipEventRecord(candidate->timing_start_event, queue->stream);
    if (timing_record_status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, timing_record_status,
                                       "hipEventRecord timing start", queue,
                                       error_sink);
    }

    const auto byte_pointer = [](Buffer *const buffer,
                                 const uint64_t offset) -> void * {
      return static_cast<char *>(buffer->device_pointer) +
             static_cast<std::size_t>(offset);
    };
    const float epsilon = [&plan]() {
      float value = 0.0F;
      std::memcpy(&value, &plan->metadata.epsilon_bits, sizeof(value));
      return value;
    }();
    const hipError_t launch_status = ::sllm_rmsnorm_kernel::launch(
        static_cast<const uint16_t *>(byte_pointer(
            plan->activation, plan->metadata.activation.byte_offset)),
        static_cast<const uint16_t *>(byte_pointer(
            plan->raw_scale, plan->metadata.raw_scale.byte_offset)),
        static_cast<uint16_t *>(
            byte_pointer(plan->output, plan->metadata.output.byte_offset)),
        static_cast<uint32_t>(normalized_size),
        static_cast<uint32_t>(row_count), epsilon, queue->stream);
    if (launch_status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, launch_status,
                                       "RMSNorm kernel launch", queue,
                                       error_sink);
    }
    const hipError_t record_status =
        hipEventRecord(candidate->event, queue->stream);
    if (record_status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, record_status,
                                       "hipEventRecord", queue, error_sink);
    }
    initialize_rmsnorm_dispatch_info(dispatch_info, dispatch_id, row_count,
                                     normalized_size, arch_name);
    *completion_output = reinterpret_cast<sllm_completion_t *>(token);
    (void)candidate.release();
    execute_guard.disarm();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in RMSNorm execute");
  }
}

extern "C" sllm_status_t
sllm_elementwise_prepare(const sllm_context_t *const raw_context,
                         const sllm_elementwise_desc_t *const descriptor,
                         sllm_elementwise_plan_t **const raw_plan,
                         sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (raw_plan != nullptr) {
      *raw_plan = nullptr;
    }
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_context == nullptr || raw_plan == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "elementwise context or plan output is null");
    }
    if (descriptor == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ELEMENTWISE_DESCRIPTOR,
          "elementwise descriptor is null");
    }
    const sllm_status_t prefix_status =
        sllm_elementwise::validate_descriptor_prefix(descriptor, error_sink);
    if (prefix_status != SLLM_STATUS_OK) {
      return prefix_status;
    }
    const sllm_elementwise_desc_t descriptor_copy = *descriptor;
    sllm_elementwise::DescriptorMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_elementwise::validate_and_copy_descriptor(&descriptor_copy,
                                                       &metadata, error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }

    std::unique_ptr<ElementwisePlan> candidate;
    uintptr_t token = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      Context *const context =
          lookup<Context>(raw_context, HandleKind::Context);
      Buffer *const input0 =
          lookup<Buffer>(descriptor_copy.input0.buffer, HandleKind::Buffer);
      Buffer *const input1 =
          metadata.operation != SLLM_ELEMENTWISE_OPERATION_COPY
              ? lookup<Buffer>(descriptor_copy.input1.buffer,
                               HandleKind::Buffer)
              : input0;
      Buffer *const output =
          lookup<Buffer>(descriptor_copy.output.buffer, HandleKind::Buffer);
      if (context == nullptr || input0 == nullptr || input1 == nullptr ||
          output == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "elementwise context or buffer handle is stale or wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (context->poisoned.load() || context->release_active ||
          input0->release_active || input1->release_active ||
          output->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "elementwise cannot prepare while a context or buffer is "
            "releasing");
      }
      if (input0->context != context || input1->context != context ||
          output->context != context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH,
            "elementwise buffers must belong to the supplied context and "
            "device");
      }
      const auto in_bounds = [](const sllm_elementwise::TensorMetadata &tensor,
                                const Buffer *const buffer) {
        return tensor.end_offset <= buffer->size_bytes;
      };
      if (!in_bounds(metadata.input0, input0) ||
          !in_bounds(metadata.output, output) ||
          (metadata.operation != SLLM_ELEMENTWISE_OPERATION_COPY &&
           !in_bounds(metadata.input1, input1))) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
            "elementwise tensor interval exceeds its backing buffer");
      }
      const auto overlaps_if_same =
          [](const Buffer *const left_buffer,
             const sllm_elementwise::TensorMetadata &left,
             const Buffer *const right_buffer,
             const sllm_elementwise::TensorMetadata &right) {
            return left_buffer == right_buffer &&
                   sllm_elementwise::intervals_overlap(left, right);
          };
      if (overlaps_if_same(input0, metadata.input0, output, metadata.output) ||
          (metadata.operation != SLLM_ELEMENTWISE_OPERATION_COPY &&
           (overlaps_if_same(input0, metadata.input0, input1,
                             metadata.input1) ||
            overlaps_if_same(input1, metadata.input1, output,
                             metadata.output)))) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_ALIAS_OVERLAP,
            "elementwise tensor intervals overlap within one backing buffer");
      }
      if (!sllm_public_runtime::AccountingState::reserve_prepared_plan(
              context->accounting, input0->accounting, input1->accounting,
              output->accounting)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "elementwise prepared-plan accounting is exhausted");
      }
      try {
        candidate = std::make_unique<ElementwisePlan>(context, input0, input1,
                                                      output, metadata);
        token = register_handle(candidate.get(), HandleKind::ElementwisePlan);
      } catch (...) {
        (void)sllm_public_runtime::AccountingState::release_prepared_plan(
            context->accounting, input0->accounting, input1->accounting,
            output->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "elementwise prepared-plan allocation or registration failed");
      }
      if (token == 0U) {
        (void)sllm_public_runtime::AccountingState::release_prepared_plan(
            context->accounting, input0->accounting, input1->accounting,
            output->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "elementwise prepared-plan handle allocation failed");
      }
    }
    *raw_plan = reinterpret_cast<sllm_elementwise_plan_t *>(token);
    (void)candidate.release();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in elementwise prepare");
  }
}

extern "C" sllm_status_t
sllm_elementwise_plan_release(sllm_elementwise_plan_t **const raw_plan,
                              sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_plan == nullptr || *raw_plan == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "elementwise plan handle is null");
    }
    ElementwisePlan *plan = nullptr;
    bool quarantine_plan = false;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup<ElementwisePlan>(*raw_plan, HandleKind::ElementwisePlan);
      if (plan == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "elementwise plan handle is stale or has the wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      if (plan->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "elementwise plan release is already in progress");
      }
      if (plan->in_flight) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "elementwise plan has an in-flight completion");
      }
      plan->release_active = true;
      const bool accounting_released =
          !sllm_public_runtime::FaultInjector::consume(
              sllm_public_runtime::FaultPoint::AccountingFailure) &&
          sllm_public_runtime::AccountingState::release_prepared_plan(
              plan->context->accounting, plan->input0->accounting,
              plan->input1->accounting, plan->output->accounting);
      if (!accounting_released) {
        plan->context->poisoned.store(true);
        unregister_handle(*raw_plan);
        *raw_plan = nullptr;
        quarantine_plan = true;
      } else {
        unregister_handle(*raw_plan);
        *raw_plan = nullptr;
      }
    }
    if (quarantine_plan) {
      poison_owner.retain(plan);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "elementwise prepared-plan accounting release failed; plan "
          "quarantined");
    }
    delete plan;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in elementwise plan release");
  }
}

extern "C" sllm_status_t
sllm_elementwise_execute(const sllm_elementwise_plan_t *const raw_plan,
                         const sllm_queue_t *const raw_queue,
                         sllm_completion_t **const completion_output,
                         sllm_elementwise_dispatch_info_t *const dispatch_info,
                         sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_plan == nullptr || raw_queue == nullptr ||
        completion_output == nullptr || dispatch_info == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "elementwise execute plan, queue, completion, or dispatch info is "
          "null");
    }
    const sllm_status_t dispatch_status =
        validate_elementwise_dispatch_info(dispatch_info, error_sink);
    if (dispatch_status != SLLM_STATUS_OK) {
      return dispatch_status;
    }

    ElementwisePlan *plan = nullptr;
    Queue *queue = nullptr;
    uint64_t dispatch_id = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      plan = lookup<ElementwisePlan>(raw_plan, HandleKind::ElementwisePlan);
      queue = lookup<Queue>(raw_queue, HandleKind::Queue);
      if (plan == nullptr || queue == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "elementwise execute plan or queue handle is stale or wrong kind");
      }
      if (plan->context != queue->context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
            "elementwise plan and queue belong to different contexts");
      }
      std::lock_guard<std::mutex> accounting_lock(
          plan->context->accounting_mutex);
      if (plan->context->poisoned.load() || plan->context->release_active ||
          queue->release_active || plan->input0->release_active ||
          plan->input1->release_active || plan->output->release_active ||
          plan->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "elementwise execute context, queue, buffer, or plan is releasing");
      }
      if (plan->in_flight) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "elementwise plan already has an in-flight completion");
      }
      if (plan->metadata.element_count == 0U ||
          plan->metadata.element_count > SLLM_HIP_ELEMENTWISE_MAX_ELEMENTS) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_UNSUPPORTED,
            "elementwise shape exceeds the baseline kernel launch contract");
      }
      if (!sllm_public_runtime::AccountingState::reserve_rmsnorm_submission(
              plan->context->accounting, queue->accounting,
              plan->input0->accounting, plan->input1->accounting,
              plan->output->accounting)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "elementwise submission accounting is exhausted");
      }
      plan->in_flight = true;
      dispatch_id = plan->context->next_dispatch_id;
      if (dispatch_id == 0U) {
        plan->in_flight = false;
        (void)sllm_public_runtime::AccountingState::rollback_rmsnorm_submission(
            plan->context->accounting, queue->accounting,
            plan->input0->accounting, plan->input1->accounting,
            plan->output->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "elementwise context dispatch id is exhausted");
      }
      if (dispatch_id == std::numeric_limits<uint64_t>::max()) {
        plan->context->next_dispatch_id = 0U;
      } else {
        ++plan->context->next_dispatch_id;
      }
      *completion_output = nullptr;
    }

    std::unique_ptr<Completion> candidate;
    NativeEventGuard event_guard;
    ElementwiseExecuteScopeGuard execute_guard(plan, queue, &candidate,
                                               &event_guard, error_sink);
    const sllm_status_t device_status =
        select_context_device(plan->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      if (!rollback_reserved_elementwise_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return device_status;
    }
    hipDeviceProp_t properties = {};
    const sllm_status_t property_status = get_device_properties(
        plan->context->device_index, &properties, error_sink);
    if (property_status != SLLM_STATUS_OK) {
      if (!rollback_reserved_elementwise_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return property_status;
    }
    char arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME] = {};
    if (!copy_property(arch_name, sizeof(arch_name), properties.gcnArchName,
                       sizeof(properties.gcnArchName))) {
      if (!rollback_reserved_elementwise_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
          "HIP gcnArchName is not NUL terminated or is too long");
    }
#if defined(SLLM_HIP_COMPILE_TARGET)
    if (std::strcmp(arch_name, SLLM_HIP_COMPILE_TARGET) != 0) {
      if (!rollback_reserved_elementwise_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
          "elementwise device target does not match the compiled exact target");
    }
#endif

    try {
      candidate = std::make_unique<Completion>(
          plan->context, queue, plan->input0, 0U, false, std::vector<uint8_t>{},
          nullptr, nullptr, nullptr, nullptr, plan, plan->input0, plan->input1,
          plan->output);
      execute_guard.candidate_allocated();
    } catch (...) {
      if (!rollback_reserved_elementwise_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "elementwise completion allocation failed before enqueue");
    }

    hipEvent_t native_event = nullptr;
    const hipError_t event_status = hipEventCreateWithFlags(&native_event, 0U);
    if (event_status != hipSuccess) {
      if (!rollback_reserved_elementwise_submission(plan, queue, error_sink)) {
        execute_guard.disarm();
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      execute_guard.disarm();
      return hip_failure(error_sink, event_status, "hipEventCreateWithFlags");
    }
    event_guard.adopt(plan->context, native_event);
    candidate->event = native_event;

    hipEvent_t timing_start_event = nullptr;
    const hipError_t timing_event_status =
        hipEventCreateWithFlags(&timing_start_event, 0U);
    if (timing_event_status != hipSuccess) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "elementwise timing event creation failed before enqueue",
          error_sink);
    }
    candidate->timing_start_event = timing_start_event;

    uintptr_t token = 0U;
    try {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      token = register_handle(candidate.get(), HandleKind::Completion);
    } catch (...) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "elementwise completion registry allocation failed before enqueue",
          error_sink);
    }
    if (token == 0U) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "elementwise completion handle token allocation failed", error_sink);
    }
    event_guard.release();
    execute_guard.completion_registered(token);

    const hipError_t timing_record_status =
        hipEventRecord(candidate->timing_start_event, queue->stream);
    if (timing_record_status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, timing_record_status,
                                       "hipEventRecord timing start", queue,
                                       error_sink);
    }

    const auto byte_pointer = [](Buffer *const buffer,
                                 const uint64_t offset) -> void * {
      return static_cast<char *>(buffer->device_pointer) +
             static_cast<std::size_t>(offset);
    };
    const uint16_t *const input0 = static_cast<const uint16_t *>(
        byte_pointer(plan->input0, plan->metadata.input0.byte_offset));
    uint16_t *const output = static_cast<uint16_t *>(
        byte_pointer(plan->output, plan->metadata.output.byte_offset));
    hipError_t launch_status = hipErrorInvalidValue;
    if (plan->metadata.operation == SLLM_ELEMENTWISE_OPERATION_COPY) {
      launch_status = ::sllm_elementwise_kernel::launch_copy(
          input0, output, plan->metadata.element_count, queue->stream);
    } else if (plan->metadata.operation == SLLM_ELEMENTWISE_OPERATION_ADD) {
      const uint16_t *const input1 = static_cast<const uint16_t *>(
          byte_pointer(plan->input1, plan->metadata.input1.byte_offset));
      launch_status = ::sllm_elementwise_kernel::launch_add(
          input0, input1, output, plan->metadata.element_count, queue->stream);
    } else if (plan->metadata.operation ==
               SLLM_ELEMENTWISE_OPERATION_SILU_MUL) {
      const uint16_t *const input1 = static_cast<const uint16_t *>(
          byte_pointer(plan->input1, plan->metadata.input1.byte_offset));
      launch_status = ::sllm_elementwise_kernel::launch_silu_mul(
          input0, input1, output, plan->metadata.element_count, queue->stream);
    } else {
      const uint16_t *const input1 = static_cast<const uint16_t *>(
          byte_pointer(plan->input1, plan->metadata.input1.byte_offset));
      launch_status = ::sllm_elementwise_kernel::launch_sigmoid_mul(
          input0, input1, output, plan->metadata.element_count, queue->stream);
    }
    if (launch_status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, launch_status,
                                       "elementwise kernel launch", queue,
                                       error_sink);
    }
    const hipError_t record_status =
        hipEventRecord(candidate->event, queue->stream);
    if (record_status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, record_status,
                                       "hipEventRecord", queue, error_sink);
    }
    initialize_elementwise_dispatch_info(
        dispatch_info, dispatch_id, plan->metadata.operation,
        plan->metadata.element_count, arch_name);
    *completion_output = reinterpret_cast<sllm_completion_t *>(token);
    (void)candidate.release();
    execute_guard.disarm();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in elementwise execute");
  }
}

#include "argmax_runtime.inc"
#include "attention_preprocess_runtime.inc"
#include "causal_attention_runtime.inc"
#include "embedding_runtime.inc"
#include "matmul_runtime.inc"

namespace {

sllm_status_t
validate_kv_view_info_input(const sllm_kv_view_info_t *const info,
                            sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                                            "KV view info output is null");
  }
  if (info->struct_size != sizeof(*info)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "KV view info has an unsupported struct size");
  }
  if (info->abi_version != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "KV view info ABI version is unsupported");
  }
  if (info->info_version != SLLM_HIP_KV_VIEW_INFO_VERSION ||
      info->reserved0 != 0U || info->reserved1 != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "KV view info version or reserved fields are invalid");
  }
  for (const uint32_t value : info->reserved) {
    if (value != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_RESERVED_NONZERO,
          "KV view info reserved fields must be zero");
    }
  }
  return SLLM_STATUS_OK;
}

sllm_status_t
validate_kv_append_info_input(const sllm_kv_append_info_t *const info,
                              sllm_error_sink_t *const sink) noexcept {
  if (info == nullptr) {
    return sllm_public_runtime::write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                                            "KV append info output is null");
  }
  if (info->struct_size != sizeof(*info)) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "KV append info has an unsupported struct size");
  }
  if (info->abi_version != SLLM_HIP_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "KV append info ABI version is unsupported");
  }
  if (info->info_version != SLLM_HIP_KV_APPEND_INFO_VERSION ||
      info->reserved0 != 0U) {
    return sllm_public_runtime::write_error(
        sink, SLLM_STATUS_RESERVED_NONZERO,
        "KV append info version or reserved fields are invalid");
  }
  for (const uint32_t value : info->reserved) {
    if (value != 0U) {
      return sllm_public_runtime::write_error(
          sink, SLLM_STATUS_RESERVED_NONZERO,
          "KV append info reserved fields must be zero");
    }
  }
  return SLLM_STATUS_OK;
}

void fill_kv_view_info(const KvState *const state, const uint64_t length,
                       const uint64_t generation, const uint64_t context_id,
                       const uint64_t state_id,
                       sllm_kv_view_info_t *const info) noexcept {
  sllm_kv_state::initialize_view_info(info);
  info->session_id = state->session_id;
  info->layer_id = state->layer_id;
  info->dtype = SLLM_TENSOR_DTYPE_F16;
  info->encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  info->head_count = SLLM_HIP_KV_HEAD_COUNT;
  info->head_dim = SLLM_HIP_KV_HEAD_DIM;
  info->capacity_tokens = state->capacity_tokens;
  info->observed_length = length;
  info->generation = generation;
  info->context_identity = context_id;
  info->state_identity = state_id;
  info->k_stride_elements[0] = state->capacity_tokens * SLLM_HIP_KV_HEAD_DIM;
  info->k_stride_elements[1] = SLLM_HIP_KV_HEAD_DIM;
  info->k_stride_elements[2] = 1U;
  info->v_stride_elements[0] = state->capacity_tokens * SLLM_HIP_KV_HEAD_DIM;
  info->v_stride_elements[1] = SLLM_HIP_KV_HEAD_DIM;
  info->v_stride_elements[2] = 1U;
}

void fill_kv_append_info(sllm_kv_append_info_t *const info,
                         const uint64_t dispatch_id, const uint64_t start,
                         const uint64_t count, const uint64_t end,
                         const char *const arch_name) noexcept {
  sllm_kv_state::initialize_append_info(info);
  info->dispatch_id = dispatch_id;
  info->dispatch_count = 1U;
  info->kernel_id = SLLM_HIP_KV_KERNEL_ID_BF16_TO_F16_TRANSPOSE_V1;
  info->workgroup_size_x = SLLM_HIP_KV_WORKGROUP_SIZE;
  info->grid_size_x = static_cast<uint32_t>(count * 4U);
  info->start_position = start;
  info->token_count = count;
  info->end_position = end;
  info->commit_allowed = 1U;
  info->fallback_allowed = 0U;
  info->fallback_used = 0U;
  sllm_public_runtime::copy_fixed_string(
      info->kernel_symbol, SLLM_HIP_KV_KERNEL_SYMBOL_MAX,
      ::sllm_kv_state_kernel::kLogicalKernelId);
  sllm_public_runtime::copy_fixed_string(info->device_symbol,
                                         SLLM_HIP_KV_DEVICE_SYMBOL_MAX,
                                         ::sllm_kv_state_kernel::kDeviceSymbol);
  sllm_public_runtime::copy_fixed_string(info->gcn_arch_name,
                                         SLLM_HIP_MAX_GCN_ARCH_NAME, arch_name);
}

bool kv_intervals_overlap(const Buffer *const left_buffer,
                          const uint64_t left_start, const uint64_t left_end,
                          const Buffer *const right_buffer,
                          const uint64_t right_start,
                          const uint64_t right_end) noexcept {
  return left_buffer == right_buffer && left_start < right_end &&
         right_start < left_end;
}

bool checked_kv_plane_bytes(const uint64_t capacity,
                            uint64_t *const plane_bytes) noexcept {
  if (plane_bytes == nullptr || capacity == 0U ||
      capacity > SLLM_HIP_KV_MAX_CAPACITY) {
    return false;
  }
  if (capacity > std::numeric_limits<uint64_t>::max() /
                     static_cast<uint64_t>(SLLM_HIP_KV_HEAD_COUNT) ||
      capacity * static_cast<uint64_t>(SLLM_HIP_KV_HEAD_COUNT) >
          std::numeric_limits<uint64_t>::max() /
              static_cast<uint64_t>(SLLM_HIP_KV_HEAD_DIM) ||
      capacity * static_cast<uint64_t>(SLLM_HIP_KV_HEAD_COUNT) *
              static_cast<uint64_t>(SLLM_HIP_KV_HEAD_DIM) >
          std::numeric_limits<uint64_t>::max() / UINT64_C(2)) {
    return false;
  }
  *plane_bytes = capacity * static_cast<uint64_t>(SLLM_HIP_KV_HEAD_COUNT) *
                 static_cast<uint64_t>(SLLM_HIP_KV_HEAD_DIM) * UINT64_C(2);
  return *plane_bytes <= SLLM_HIP_KV_EVIDENCE_MAX_READBACK_BYTES;
}

sllm_status_t validate_kv_readback_request(
    const sllm_hip_kv_readback_request_t *const request,
    sllm_error_sink_t *const error_sink) noexcept {
  if (request == nullptr) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INVALID_ARGUMENT,
        "KV evidence readback request is null");
  }
  if (request->struct_size != sizeof(*request)) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INVALID_ARGUMENT,
        "KV evidence readback request struct size is unsupported");
  }
  if (request->abi_version != SLLM_HIP_KV_EVIDENCE_ABI_VERSION) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INVALID_ABI_VERSION,
        "KV evidence readback ABI is unsupported");
  }
  if (request->reserved0 != 0U || request->reserved[0] != 0U ||
      request->reserved[1] != 0U || request->reserved[2] != 0U ||
      request->reserved[3] != 0U) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_RESERVED_NONZERO,
        "KV evidence readback reserved fields must be zero");
  }
  if (request->view == nullptr ||
      (request->plane != SLLM_HIP_KV_EVIDENCE_PLANE_K &&
       request->plane != SLLM_HIP_KV_EVIDENCE_PLANE_V)) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INVALID_ARGUMENT,
        "KV evidence readback view or plane is invalid");
  }
  if (request->byte_length == 0U || request->host_output == nullptr) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INVALID_ARGUMENT,
        "KV evidence readback length or host output is invalid");
  }
  if (request->byte_offset % UINT64_C(2) != 0U ||
      request->byte_length % UINT64_C(2) != 0U) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_MISALIGNED_OFFSET,
        "KV evidence readback range must be FP16-aligned");
  }
  if (request->byte_length > SLLM_HIP_KV_EVIDENCE_MAX_READBACK_BYTES ||
      request->byte_offset >
          std::numeric_limits<uint64_t>::max() - request->byte_length) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_METADATA_OVERFLOW,
        "KV evidence readback range overflows its bounded contract");
  }
  if (request->host_capacity < request->byte_length) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_BUFFER_TOO_SMALL,
        "KV evidence readback host output is undersized");
  }
  if (request->byte_length > std::numeric_limits<std::size_t>::max()) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_METADATA_OVERFLOW,
        "KV evidence readback length does not fit the host ABI");
  }
  return SLLM_STATUS_OK;
}

class KvReadbackActivityGuard final {
public:
  explicit KvReadbackActivityGuard(KvView *const view) noexcept : view_(view) {}

  KvReadbackActivityGuard(const KvReadbackActivityGuard &) = delete;
  KvReadbackActivityGuard &operator=(const KvReadbackActivityGuard &) = delete;

  ~KvReadbackActivityGuard() noexcept {
    std::lock_guard<std::mutex> lock(registry_mutex);
    view_->readback_active = false;
  }

private:
  KvView *view_;
};

} // namespace

extern "C" sllm_status_t
sllm_kv_state_create(const sllm_context_t *const raw_context,
                     const sllm_kv_state_create_info_t *const info,
                     sllm_kv_state_t **const raw_state,
                     sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (raw_state != nullptr) {
      *raw_state = nullptr;
    }
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t info_status =
        sllm_kv_state::validate_state_create_info(info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    if (raw_context == nullptr || raw_state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "KV state context or output is null");
    }

    Context *context = nullptr;
    const uintptr_t context_token = handle_key(raw_context);
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      context = lookup<Context>(raw_context, HandleKind::Context);
      if (context == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "KV state context handle is stale or has the wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (context->poisoned.load() || context->release_active ||
          !sllm_public_runtime::AccountingState::can_increment(
              context->accounting.child_count)) {
        return sllm_public_runtime::write_error(
            error_sink,
            context->release_active ? SLLM_STATUS_PUBLIC_BUSY
                                    : SLLM_STATUS_INTERNAL_ERROR,
            context->release_active
                ? "KV state context release is already in progress"
                : "KV state context accounting is exhausted");
      }
      ++context->accounting.child_count;
    }

    const sllm_status_t device_status =
        select_context_device(context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      (void)rollback_child(context, error_sink,
                           "KV state provisional accounting rollback failed");
      return device_status;
    }
    uint64_t elements = info->capacity_tokens * SLLM_HIP_KV_HEAD_COUNT;
    elements *= SLLM_HIP_KV_HEAD_DIM;
    const std::size_t allocation_bytes =
        static_cast<std::size_t>(elements * UINT64_C(2));
    void *key_pointer = nullptr;
    const hipError_t key_status =
        sllm_public_runtime::FaultInjector::consume(
            sllm_public_runtime::FaultPoint::NativeCreationFailure)
            ? hipErrorUnknown
            : hipMalloc(&key_pointer, allocation_bytes);
    if (key_status != hipSuccess) {
      (void)rollback_child(context, error_sink,
                           "KV state provisional accounting rollback failed");
      return hip_failure(error_sink, key_status, "hipMalloc KV K buffer");
    }
    NativeAllocationGuard key_guard(context, key_pointer);
    void *value_pointer = nullptr;
    const hipError_t value_status =
        sllm_public_runtime::FaultInjector::consume(
            sllm_public_runtime::FaultPoint::NativeCreationFailure)
            ? hipErrorUnknown
            : hipMalloc(&value_pointer, allocation_bytes);
    if (value_status != hipSuccess) {
      const hipError_t cleanup_status = key_guard.cleanup();
      (void)rollback_child(context, error_sink,
                           "KV state provisional accounting rollback failed");
      return cleanup_status == hipSuccess
                 ? hip_failure(error_sink, value_status,
                               "hipMalloc KV V buffer")
                 : hip_failure(error_sink, cleanup_status,
                               "hipFree KV K buffer rollback");
    }
    NativeAllocationGuard value_guard(context, value_pointer);
    std::unique_ptr<Buffer> key_buffer;
    std::unique_ptr<Buffer> value_buffer;
    std::unique_ptr<KvState> candidate;
    try {
      key_buffer =
          std::make_unique<Buffer>(context, key_pointer, allocation_bytes);
      value_buffer =
          std::make_unique<Buffer>(context, value_pointer, allocation_bytes);
      candidate = std::make_unique<KvState>(
          context, info->session_id, info->layer_id, info->capacity_tokens,
          key_buffer.get(), value_buffer.get(), context_token);
    } catch (...) {
      (void)key_guard.cleanup();
      (void)value_guard.cleanup();
      (void)rollback_child(context, error_sink,
                           "KV state provisional accounting rollback failed");
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "KV state allocation failed after device buffers were created");
    }

    {
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (!sllm_public_runtime::AccountingState::reserve_kv_state(
              context->accounting, key_buffer->accounting,
              value_buffer->accounting)) {
        poison_context_locked(context);
        key_guard.release();
        value_guard.release();
        poison_owner.retain(std::move(candidate));
        key_buffer.release();
        value_buffer.release();
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "KV state accounting reservation is exhausted; state quarantined");
      }
      --context->accounting.child_count; // replace provisional child guard
    }

    uintptr_t token = 0U;
    try {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      token = register_handle(candidate.get(), HandleKind::KvState);
    } catch (...) {
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      if (!sllm_public_runtime::AccountingState::release_kv_state(
              context->accounting, key_buffer->accounting,
              value_buffer->accounting)) {
        poison_context_locked(context);
        key_guard.release();
        value_guard.release();
        poison_owner.retain(std::move(candidate));
        key_buffer.release();
        value_buffer.release();
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "KV state accounting rollback failed; state quarantined");
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "KV state registry allocation failed");
    }
    if (token == 0U) {
      std::lock_guard<std::mutex> accounting_lock(context->accounting_mutex);
      const bool released =
          sllm_public_runtime::AccountingState::release_kv_state(
              context->accounting, key_buffer->accounting,
              value_buffer->accounting);
      if (!released) {
        poison_context_locked(context);
        key_guard.release();
        value_guard.release();
        poison_owner.retain(std::move(candidate));
        key_buffer.release();
        value_buffer.release();
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "KV state accounting rollback failed; state quarantined");
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "KV state handle token allocation failed");
    }
    candidate->state_identity = token;
    *raw_state = reinterpret_cast<sllm_kv_state_t *>(token);
    key_guard.release();
    value_guard.release();
    key_buffer.release();
    value_buffer.release();
    (void)candidate.release();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in KV state create");
  }
}

extern "C" sllm_status_t
sllm_kv_state_release(sllm_kv_state_t **const raw_state,
                      sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_state == nullptr || *raw_state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT, "KV state handle is null");
    }
    KvState *state = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      state = lookup<KvState>(*raw_state, HandleKind::KvState);
      if (state == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "KV state handle is stale or has the wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(
          state->context->accounting_mutex);
      if (state->release_active || state->transition_token != 0U ||
          state->view_count != 0U ||
          state->accounting.active_submissions != 0U ||
          state->accounting.completion_references != 0U) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "KV state has an in-flight append or live snapshot view");
      }
      if (state->context->poisoned.load()) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "KV state context is poisoned by an unresolved cleanup failure");
      }
      state->release_active = true;
    }

    const sllm_status_t device_status =
        select_context_device(state->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      state->release_active = false;
      return device_status;
    }
    NativeAllocationGuard key_guard(state->context,
                                    state->key_buffer->device_pointer);
    NativeAllocationGuard value_guard(state->context,
                                      state->value_buffer->device_pointer);
    const hipError_t key_status = key_guard.cleanup();
    const hipError_t value_status = value_guard.cleanup();
    state->key_buffer->device_pointer = nullptr;
    state->value_buffer->device_pointer = nullptr;
    if (key_status != hipSuccess || value_status != hipSuccess) {
      {
        std::lock_guard<std::mutex> registry_lock(registry_mutex);
        unregister_handle(*raw_state);
        *raw_state = nullptr;
      }
      retain_poisoned(state, state->context);
      return hip_failure(error_sink,
                         key_status != hipSuccess ? key_status : value_status,
                         "hipFree KV state buffers");
    }

    bool accounting_released = false;
    {
      std::lock_guard<std::mutex> accounting_lock(
          state->context->accounting_mutex);
      accounting_released =
          !sllm_public_runtime::FaultInjector::consume(
              sllm_public_runtime::FaultPoint::AccountingFailure) &&
          sllm_public_runtime::AccountingState::release_kv_state(
              state->context->accounting, state->key_buffer->accounting,
              state->value_buffer->accounting);
    }
    if (!accounting_released) {
      {
        std::lock_guard<std::mutex> registry_lock(registry_mutex);
        unregister_handle(*raw_state);
        *raw_state = nullptr;
      }
      retain_poisoned(state, state->context);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "KV state accounting release failed; state quarantined");
    }
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      unregister_handle(*raw_state);
      *raw_state = nullptr;
    }
    delete state->key_buffer;
    delete state->value_buffer;
    delete state;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in KV state release");
  }
}

extern "C" sllm_status_t
sllm_kv_state_query(const sllm_kv_state_t *const raw_state,
                    sllm_kv_view_info_t *const info,
                    sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t info_status =
        validate_kv_view_info_input(info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    KvState *const state = lookup<KvState>(raw_state, HandleKind::KvState);
    if (state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "KV state handle is stale or has the wrong kind");
    }
    std::lock_guard<std::mutex> accounting_lock(
        state->context->accounting_mutex);
    if (state->context->poisoned.load() || state->release_active) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "KV state is releasing or its context is poisoned");
    }
    fill_kv_view_info(state, state->published_length, state->generation,
                      state->context_identity, state->state_identity, info);
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in KV state query");
  }
}

extern "C" sllm_status_t
sllm_kv_state_snapshot(const sllm_kv_state_t *const raw_state,
                       sllm_kv_view_t **const raw_view,
                       sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (raw_view != nullptr) {
      *raw_view = nullptr;
    }
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_state == nullptr || raw_view == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "KV state or snapshot output is null");
    }
    std::unique_ptr<KvView> candidate;
    uintptr_t token = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      KvState *const state = lookup<KvState>(raw_state, HandleKind::KvState);
      if (state == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "KV state handle is stale or has the wrong kind");
      }
      std::lock_guard<std::mutex> accounting_lock(
          state->context->accounting_mutex);
      if (state->context->poisoned.load() || state->release_active ||
          state->view_count == std::numeric_limits<uint64_t>::max()) {
        return sllm_public_runtime::write_error(
            error_sink,
            state->release_active ? SLLM_STATUS_PUBLIC_BUSY
                                  : SLLM_STATUS_INTERNAL_ERROR,
            state->release_active
                ? "KV state is releasing"
                : "KV state snapshot accounting is exhausted");
      }
      if (!sllm_public_runtime::AccountingState::reserve_kv_view(
              state->context->accounting, state->accounting)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "KV state snapshot accounting is exhausted");
      }
      ++state->view_count;
      try {
        candidate = std::make_unique<KvView>(
            state->context, state, state->session_id, state->layer_id,
            state->capacity_tokens, state->published_length, state->generation,
            state->context_identity, state->state_identity);
        token = register_handle(candidate.get(), HandleKind::KvView);
      } catch (...) {
        --state->view_count;
        (void)sllm_public_runtime::AccountingState::release_kv_view(
            state->context->accounting, state->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "KV snapshot allocation or registry insertion failed");
      }
      if (token == 0U) {
        --state->view_count;
        (void)sllm_public_runtime::AccountingState::release_kv_view(
            state->context->accounting, state->accounting);
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "KV snapshot handle token allocation failed");
      }
    }
    *raw_view = reinterpret_cast<sllm_kv_view_t *>(token);
    (void)candidate.release();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in KV state snapshot");
  }
}

extern "C" sllm_status_t
sllm_kv_view_query(const sllm_kv_view_t *const raw_view,
                   sllm_kv_view_info_t *const info,
                   sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t info_status =
        validate_kv_view_info_input(info, error_sink);
    if (info_status != SLLM_STATUS_OK) {
      return info_status;
    }
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    KvView *const view = lookup<KvView>(raw_view, HandleKind::KvView);
    if (view == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "KV snapshot handle is stale or has the wrong kind");
    }
    std::lock_guard<std::mutex> accounting_lock(
        view->context->accounting_mutex);
    if (view->context->poisoned.load() || view->release_active) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "KV snapshot is releasing or its context is poisoned");
    }
    fill_kv_view_info(view->state, view->observed_length, view->generation,
                      view->context_identity, view->state_identity, info);
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in KV snapshot query");
  }
}

extern "C" sllm_status_t
sllm_kv_view_release(sllm_kv_view_t **const raw_view,
                     sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_view == nullptr || *raw_view == nullptr) {
      return sllm_public_runtime::write_error(error_sink,
                                              SLLM_STATUS_INVALID_ARGUMENT,
                                              "KV snapshot handle is null");
    }
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    KvView *const view = lookup<KvView>(*raw_view, HandleKind::KvView);
    if (view == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "KV snapshot handle is stale or has the wrong kind");
    }
    if (view->release_active || view->readback_active) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "KV snapshot release or evidence readback is already in progress");
    }
    std::lock_guard<std::mutex> accounting_lock(
        view->context->accounting_mutex);
    if (view->context->poisoned.load() || view->state->release_active) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "KV snapshot context or state is poisoned or releasing");
    }
    if (view->state->view_count == 0U ||
        !sllm_public_runtime::AccountingState::release_kv_view(
            view->context->accounting, view->state->accounting)) {
      poison_context_locked(view->context);
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "KV snapshot accounting release failed; context poisoned");
    }
    --view->state->view_count;
    unregister_handle(*raw_view);
    *raw_view = nullptr;
    delete view;
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in KV snapshot release");
  }
}

extern "C" sllm_status_t
sllm_hip_kv_view_readback(const sllm_hip_kv_readback_request_t *const request,
                          sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t request_status =
        validate_kv_readback_request(request, error_sink);
    if (request_status != SLLM_STATUS_OK) {
      return request_status;
    }

    KvView *view = nullptr;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      view = lookup<KvView>(request->view, HandleKind::KvView);
      if (view == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "KV evidence readback view is stale or has the wrong kind");
      }
      if (view->readback_active || view->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "KV evidence readback view is already active or releasing");
      }
      view->readback_active = true;
    }
    KvReadbackActivityGuard readback_guard(view);

    std::lock_guard<std::mutex> accounting_lock(
        view->context->accounting_mutex);
    KvState *const state = view->state;
    if (state == nullptr || state->context != view->context ||
        view->context->poisoned.load() || view->context->release_active ||
        view->release_active || state->release_active ||
        state->key_buffer == nullptr || state->value_buffer == nullptr ||
        state->key_buffer->context != view->context ||
        state->value_buffer->context != view->context ||
        state->key_buffer->release_active ||
        state->value_buffer->release_active) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "KV evidence readback state is releasing or unsafe");
    }
    if (view->context_identity != state->context_identity ||
        view->state_identity != state->state_identity ||
        view->session_id != state->session_id ||
        view->layer_id != state->layer_id ||
        view->capacity_tokens != state->capacity_tokens ||
        view->observed_length != state->published_length ||
        view->generation != state->generation ||
        state->transition_token != 0U ||
        state->accounting.active_submissions != 0U ||
        state->accounting.completion_references != 0U ||
        state->key_buffer->accounting.active_submissions != 0U ||
        state->key_buffer->accounting.completion_references != 0U ||
        state->value_buffer->accounting.active_submissions != 0U ||
        state->value_buffer->accounting.completion_references != 0U) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "KV evidence readback snapshot is stale or append is in flight");
    }

    uint64_t plane_bytes = 0U;
    if (!checked_kv_plane_bytes(state->capacity_tokens, &plane_bytes)) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_METADATA_OVERFLOW,
          "KV evidence readback state storage size is invalid");
    }
    const uint64_t storage_bytes = plane_bytes;
    if (state->key_buffer->size_bytes != storage_bytes ||
        state->value_buffer->size_bytes != storage_bytes ||
        request->byte_offset > plane_bytes ||
        request->byte_length > plane_bytes - request->byte_offset) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
          "KV evidence readback range is outside one exact KV plane");
    }

    Buffer *const buffer = request->plane == SLLM_HIP_KV_EVIDENCE_PLANE_K
                               ? state->key_buffer
                               : state->value_buffer;
    const uint64_t source_offset = request->byte_offset;
    if (buffer->device_pointer == nullptr ||
        source_offset > buffer->size_bytes ||
        request->byte_length > buffer->size_bytes - source_offset) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "KV evidence readback storage is unavailable");
    }

    const sllm_status_t device_status =
        select_context_device(view->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      return device_status;
    }
    hipStream_t stream = nullptr;
    const hipError_t create_status =
        hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    if (create_status != hipSuccess) {
      return hip_failure(error_sink, create_status,
                         "hipStreamCreateWithFlags KV evidence readback");
    }
    const auto destroy_stream = [&]() noexcept -> hipError_t {
      const hipError_t status = hipStreamDestroy(stream);
      stream = nullptr;
      return status;
    };
    const void *const source =
        static_cast<const char *>(buffer->device_pointer) +
        static_cast<std::size_t>(source_offset);
    const hipError_t copy_status =
        hipMemcpyAsync(request->host_output, source,
                       static_cast<std::size_t>(request->byte_length),
                       hipMemcpyDeviceToHost, stream);
    if (copy_status != hipSuccess) {
      const hipError_t destroy_status = destroy_stream();
      if (destroy_status != hipSuccess) {
        poison_context_locked(view->context);
        return hip_failure(error_sink, destroy_status,
                           "hipStreamDestroy KV evidence readback");
      }
      return hip_failure(error_sink, copy_status,
                         "hipMemcpyAsync KV evidence readback");
    }
    const hipError_t synchronize_status = hipStreamSynchronize(stream);
    const hipError_t destroy_status = destroy_stream();
    if (destroy_status != hipSuccess) {
      poison_context_locked(view->context);
      return hip_failure(error_sink, destroy_status,
                         "hipStreamDestroy KV evidence readback");
    }
    if (synchronize_status != hipSuccess) {
      return hip_failure(error_sink, synchronize_status,
                         "hipStreamSynchronize KV evidence readback");
    }
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in KV evidence readback");
  }
}

extern "C" sllm_status_t
sllm_kv_state_append(const sllm_kv_state_t *const raw_state,
                     const sllm_queue_t *const raw_queue,
                     const sllm_kv_append_desc_t *const descriptor,
                     sllm_completion_t **const completion_output,
                     sllm_kv_append_info_t *const append_info,
                     sllm_error_sink_t *const error_sink) noexcept {
  try {
    if (completion_output != nullptr) {
      *completion_output = nullptr;
    }
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    const sllm_status_t append_info_status =
        validate_kv_append_info_input(append_info, error_sink);
    if (append_info_status != SLLM_STATUS_OK) {
      return append_info_status;
    }
    sllm_kv_state::AppendMetadata metadata{};
    const sllm_status_t descriptor_status =
        sllm_kv_state::validate_and_copy_append(descriptor, &metadata,
                                                error_sink);
    if (descriptor_status != SLLM_STATUS_OK) {
      return descriptor_status;
    }
    if (raw_state == nullptr || raw_queue == nullptr ||
        completion_output == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "KV append state, queue, or completion output is null");
    }

    KvState *state = nullptr;
    Queue *queue = nullptr;
    Buffer *key_input = nullptr;
    Buffer *value_input = nullptr;
    uint64_t dispatch_id = 0U;
    {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      state = lookup<KvState>(raw_state, HandleKind::KvState);
      queue = lookup<Queue>(raw_queue, HandleKind::Queue);
      key_input = lookup<Buffer>(metadata.key_input.buffer, HandleKind::Buffer);
      value_input =
          lookup<Buffer>(metadata.value_input.buffer, HandleKind::Buffer);
      if (state == nullptr || queue == nullptr || key_input == nullptr ||
          value_input == nullptr) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
            "KV append state, queue, or input handle is stale or wrong kind");
      }
      if (state->context != queue->context ||
          key_input->context != state->context ||
          value_input->context != state->context) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
            "KV append state, queue, and inputs must share one context");
      }
      std::lock_guard<std::mutex> accounting_lock(
          state->context->accounting_mutex);
      if (state->context->poisoned.load() || state->context->release_active ||
          queue->release_active || key_input->release_active ||
          value_input->release_active || state->release_active) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "KV append context, queue, state, or input is releasing");
      }
      if (state->transition_token != 0U ||
          state->accounting.active_submissions != 0U) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_PUBLIC_BUSY,
            "KV state already has an append in flight");
      }
      if (metadata.expected_length != state->published_length ||
          metadata.start_position != state->published_length) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_KV_LENGTH_MISMATCH,
            "KV append expected length is not the published state length");
      }
      if (metadata.end_position > state->capacity_tokens) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_KV_CAPACITY_EXCEEDED,
            "KV append exceeds state capacity");
      }
      if (metadata.key_input.end_offset > key_input->size_bytes ||
          metadata.value_input.end_offset > value_input->size_bytes) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
            "KV append input interval exceeds its backing buffer");
      }
      if (kv_intervals_overlap(key_input, metadata.key_input.byte_offset,
                               metadata.key_input.end_offset, value_input,
                               metadata.value_input.byte_offset,
                               metadata.value_input.end_offset) ||
          kv_intervals_overlap(key_input, metadata.key_input.byte_offset,
                               metadata.key_input.end_offset, state->key_buffer,
                               0U, state->key_buffer->size_bytes) ||
          kv_intervals_overlap(key_input, metadata.key_input.byte_offset,
                               metadata.key_input.end_offset,
                               state->value_buffer, 0U,
                               state->value_buffer->size_bytes) ||
          kv_intervals_overlap(value_input, metadata.value_input.byte_offset,
                               metadata.value_input.end_offset,
                               state->key_buffer, 0U,
                               state->key_buffer->size_bytes) ||
          kv_intervals_overlap(value_input, metadata.value_input.byte_offset,
                               metadata.value_input.end_offset,
                               state->value_buffer, 0U,
                               state->value_buffer->size_bytes)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_ALIAS_OVERLAP,
            "KV append inputs must not overlap each other or KV state storage");
      }
      if (state->context->next_dispatch_id == 0U) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "KV append context dispatch id is exhausted");
      }
      dispatch_id = state->context->next_dispatch_id;
      if (dispatch_id == std::numeric_limits<uint64_t>::max()) {
        state->context->next_dispatch_id = 0U;
      } else {
        ++state->context->next_dispatch_id;
      }
      if (!sllm_public_runtime::AccountingState::reserve_kv_append(
              state->context->accounting, queue->accounting, state->accounting,
              key_input->accounting, value_input->accounting,
              state->key_buffer->accounting, state->value_buffer->accounting)) {
        return sllm_public_runtime::write_error(
            error_sink, SLLM_STATUS_INTERNAL_ERROR,
            "KV append accounting reservation is exhausted");
      }
      state->transition_token = dispatch_id;
      state->transition_start = metadata.start_position;
      state->transition_count = metadata.token_count;
      state->transition_end = metadata.end_position;
      state->commit_allowed = true;
    }

    std::unique_ptr<Completion> candidate;
    NativeEventGuard event_guard;
    KvStateExecuteScopeGuard execute_guard(state, queue, key_input, value_input,
                                           &candidate, &event_guard,
                                           error_sink);
    const sllm_status_t device_status =
        select_context_device(state->context, error_sink);
    if (device_status != SLLM_STATUS_OK) {
      execute_guard.disarm();
      if (!rollback_reserved_kv_submission(state, queue, key_input, value_input,
                                           error_sink)) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return device_status;
    }
    hipDeviceProp_t properties = {};
    const sllm_status_t property_status = get_device_properties(
        state->context->device_index, &properties, error_sink);
    if (property_status != SLLM_STATUS_OK) {
      execute_guard.disarm();
      if (!rollback_reserved_kv_submission(state, queue, key_input, value_input,
                                           error_sink)) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return property_status;
    }
    char arch_name[SLLM_HIP_MAX_GCN_ARCH_NAME] = {};
    if (!copy_property(arch_name, sizeof(arch_name), properties.gcnArchName,
                       sizeof(properties.gcnArchName))) {
      execute_guard.disarm();
      if (!rollback_reserved_kv_submission(state, queue, key_input, value_input,
                                           error_sink)) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
          "HIP gcnArchName is not NUL terminated or is too long");
    }
    if (properties.warpSize != 32) {
      execute_guard.disarm();
      if (!rollback_reserved_kv_submission(state, queue, key_input, value_input,
                                           error_sink)) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_UNSUPPORTED,
          "KV state baseline requires a wave32 device");
    }
#if defined(SLLM_HIP_COMPILE_TARGET)
    if (std::strcmp(arch_name, SLLM_HIP_COMPILE_TARGET) != 0) {
      execute_guard.disarm();
      if (!rollback_reserved_kv_submission(state, queue, key_input, value_input,
                                           error_sink)) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
          "KV state device target does not match the compiled exact target");
    }
#endif
    try {
      candidate = std::make_unique<Completion>(
          state->context, queue, nullptr, 0U, false, std::vector<uint8_t>{});
      candidate->kv_state_append = true;
      candidate->kv_state = state;
      candidate->kv_key_input = key_input;
      candidate->kv_value_input = value_input;
      candidate->kv_key_buffer = state->key_buffer;
      candidate->kv_value_buffer = state->value_buffer;
      candidate->kv_append_token = dispatch_id;
      candidate->kv_append_start = metadata.start_position;
      candidate->kv_append_count = metadata.token_count;
      candidate->kv_append_end = metadata.end_position;
      execute_guard.candidate_allocated();
    } catch (...) {
      execute_guard.disarm();
      if (!rollback_reserved_kv_submission(state, queue, key_input, value_input,
                                           error_sink)) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "KV append completion allocation failed before enqueue");
    }
    hipEvent_t native_event = nullptr;
    const hipError_t event_status = hipEventCreateWithFlags(&native_event, 0U);
    if (event_status != hipSuccess) {
      execute_guard.disarm();
      if (!rollback_reserved_kv_submission(state, queue, key_input, value_input,
                                           error_sink)) {
        return SLLM_STATUS_INTERNAL_ERROR;
      }
      return hip_failure(error_sink, event_status, "hipEventCreateWithFlags");
    }
    event_guard.adopt(state->context, native_event);
    candidate->event = native_event;
    hipEvent_t timing_start_event = nullptr;
    const hipError_t timing_status =
        hipEventCreateWithFlags(&timing_start_event, 0U);
    if (timing_status != hipSuccess) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard, "KV append timing event creation failed",
          error_sink);
    }
    candidate->timing_start_event = timing_start_event;
    uintptr_t token = 0U;
    try {
      std::lock_guard<std::mutex> registry_lock(registry_mutex);
      token = register_handle(candidate.get(), HandleKind::Completion);
    } catch (...) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "KV append completion registry allocation failed", error_sink);
    }
    if (token == 0U) {
      execute_guard.disarm();
      return rollback_unpublished_submission(
          candidate, event_guard,
          "KV append completion token allocation failed", error_sink);
    }
    event_guard.release();
    execute_guard.completion_registered(token);
    const hipError_t timing_record_status =
        hipEventRecord(candidate->timing_start_event, queue->stream);
    if (timing_record_status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, timing_record_status,
                                       "hipEventRecord KV timing start", queue,
                                       error_sink);
    }
    const auto byte_pointer = [](Buffer *const buffer,
                                 const uint64_t offset) -> void * {
      return static_cast<char *>(buffer->device_pointer) +
             static_cast<std::size_t>(offset);
    };
    const uint16_t *const key_input_pointer = static_cast<const uint16_t *>(
        byte_pointer(key_input, metadata.key_input.byte_offset));
    const uint16_t *const value_input_pointer = static_cast<const uint16_t *>(
        byte_pointer(value_input, metadata.value_input.byte_offset));
    const hipError_t launch_status = ::sllm_kv_state_kernel::launch(
        key_input_pointer, value_input_pointer,
        static_cast<uint16_t *>(state->key_buffer->device_pointer),
        static_cast<uint16_t *>(state->value_buffer->device_pointer),
        static_cast<uint32_t>(metadata.token_count), state->capacity_tokens,
        metadata.start_position, queue->stream);
    if (launch_status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, launch_status,
                                       "KV state kernel launch", queue,
                                       error_sink);
    }
    const hipError_t record_status =
        hipEventRecord(candidate->event, queue->stream);
    if (record_status != hipSuccess) {
      execute_guard.disarm();
      return cleanup_failed_submission(candidate, token, record_status,
                                       "hipEventRecord KV append", queue,
                                       error_sink);
    }
    fill_kv_append_info(append_info, dispatch_id, metadata.start_position,
                        metadata.token_count, metadata.end_position, arch_name);
    *completion_output = reinterpret_cast<sllm_completion_t *>(token);
    (void)candidate.release();
    execute_guard.disarm();
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in KV state append");
  }
}

extern "C" sllm_status_t
sllm_kv_state_append_cancel(const sllm_kv_state_t *const raw_state,
                            sllm_completion_t *const raw_completion,
                            sllm_error_sink_t *const error_sink) noexcept {
  try {
    const sllm_status_t sink_status =
        sllm_public_runtime::validate_error_sink(error_sink);
    if (sink_status != SLLM_STATUS_OK) {
      return sink_status;
    }
    if (raw_state == nullptr || raw_completion == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "KV append cancel state or completion is null");
    }
    Completion *completion = nullptr;
    const sllm_status_t pin_status =
        pin_completion(raw_completion, &completion, error_sink);
    if (pin_status != SLLM_STATUS_OK) {
      return pin_status;
    }
    CompletionPin pin(completion);
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    KvState *const state = lookup<KvState>(raw_state, HandleKind::KvState);
    if (state == nullptr) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_INVALID_HANDLE,
          "KV append cancel state handle is stale or wrong kind");
    }
    std::lock_guard<std::mutex> state_lock(completion->state_mutex);
    if (!completion->kv_state_append || completion->kv_state != state) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INVALID_ARGUMENT,
          "completion does not belong to the supplied KV state");
    }
    if (completion->terminal) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_PUBLIC_BUSY,
          "KV append completion is already terminal");
    }
    if (!revoke_kv_append_commit(completion)) {
      return sllm_public_runtime::write_error(
          error_sink, SLLM_STATUS_INTERNAL_ERROR,
          "KV append cancellation failed; context poisoned");
    }
    return SLLM_STATUS_OK;
  } catch (...) {
    return sllm_public_runtime::write_error(
        error_sink, SLLM_STATUS_INTERNAL_ERROR,
        "unexpected exception in KV append cancel");
  }
}

#include "linear_attention_runtime.inc"

#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
extern "C" std::size_t sllm_test_orphan_count() noexcept {
  return orphan_owner.size();
}

extern "C" std::size_t sllm_test_poison_count() noexcept {
  return poison_owner.size();
}

extern "C" void sllm_test_rmsnorm_execute_throw_after_reservation(
    const uint32_t occurrences) noexcept {
  rmsnorm_throw_after_reservation.store(occurrences, std::memory_order_release);
}

extern "C" void sllm_test_rmsnorm_execute_throw_after_registration(
    const uint32_t occurrences) noexcept {
  rmsnorm_throw_after_registration.store(occurrences,
                                         std::memory_order_release);
}
#endif
