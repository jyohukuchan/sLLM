#ifndef SLLM_DECODE_CONTROL_KERNEL_INTERNAL_HPP
#define SLLM_DECODE_CONTROL_KERNEL_INTERNAL_HPP

#include "token_selector_pq_algorithm.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace sllm_decode_control {

constexpr std::uint32_t kVersion = 1U;
constexpr std::uint32_t kModeTargetOnly = 0U;
constexpr std::uint32_t kModeMtp = 1U;
constexpr std::uint32_t kPhaseTarget = 0U;
constexpr std::uint32_t kPhaseDraft = 1U;
constexpr std::uint32_t kPhaseMtpAlign = 2U;
constexpr std::uint64_t kQDraftSeedDomain64 = UINT64_C(0x5144524146540001);
constexpr std::uint32_t kNoStop = UINT32_MAX;
constexpr std::uint32_t kMaxStopIds = 16U;
constexpr std::uint32_t kResultRingSlots = 2U;
constexpr std::uint32_t kMaxWidth = sllm_token_selector_pq::kMaxWidth;
constexpr std::uint32_t kMaxEmitted = sllm_token_selector_pq::kMaxEmitted;

enum class Status : std::uint32_t {
  Ok = 0U,
  Noop = 1U,
  InvalidVersion = 2U,
  InvalidMode = 3U,
  InvalidPhase = 4U,
  InvalidWidth = 5U,
  InvalidGeneration = 6U,
  InvalidCounter = 7U,
  InvalidPosition = 8U,
  InvalidCapacity = 9U,
  InvalidDecision = 10U,
  InvalidSelector = 11U,
  InvalidToken = 12U,
  InvalidStopIds = 13U,
  InvalidBudget = 14U,
  NullPointer = 15U,
};

enum class InputKind : std::uint32_t {
  Selector = 0U,
  Decision = 1U,
};

enum HaltFlag : std::uint32_t {
  HaltStop = 1U << 0U,
  HaltBudget = 1U << 1U,
  HaltInvalid = 1U << 2U,
  HaltNoop = 1U << 3U,
  HaltAllAccept = 1U << 4U,
};

/* This is the device-resident control plane for one request.  Keep the field
 * order stable: the Rust/native bridge binds this object as a raw device
 * record without adding a public C ABI. */
struct alignas(16) ControlV1 final {
  std::uint32_t version;
  std::uint32_t status;
  std::uint32_t mode;
  std::uint32_t width;
  std::uint64_t model_position;
  std::uint64_t sampler_counter;
  std::uint64_t seed;
  std::uint64_t generation;
  std::uint64_t capacity;
  std::uint64_t output_limit;
  std::uint64_t output_count;
  std::uint32_t pending_token;
  std::uint32_t halted;
  std::uint32_t commit_rows;
  std::uint32_t publish_count;
  std::uint32_t accepted_count;
  std::uint32_t stop_row;
  std::uint32_t hidden_row;
  /* Active MTP draft width for the current replay. `width` remains the
   * configured graph maximum; Draft row zero refreshes this value from the
   * remaining logical budget and physical state capacity. */
  std::uint32_t active_width;
  std::uint64_t phase_position;
  std::uint64_t phase_counter;
  std::uint64_t phase_seed;
  std::uint32_t phase_rows;
  std::uint32_t phase_index;
  std::uint32_t phase_kind;
  std::uint32_t phase_active;
};

/* The public selector record is reproduced here because this private kernel
 * must also consume it without depending on the installed public header. */
struct SelectorRecordV1 final {
  std::int32_t token_id;
  std::uint32_t status;
  float logprob;
  std::uint32_t reserved;
};

/* The result is written to generation % 2.  Host consumers must validate the
 * exact generation before accepting a slot; a wrapped ring slot is not itself
 * evidence that the record is current. */
struct alignas(16) ResultV1 final {
  std::uint64_t generation;
  std::uint32_t status;
  std::uint32_t count;
  std::uint32_t commit_rows;
  std::uint32_t accepted;
  std::uint32_t width;
  std::uint32_t halt_flags;
  std::uint32_t reserved0;
  std::uint32_t reserved_padding;
  std::uint64_t model_position_before;
  std::uint64_t model_position_after;
  std::uint64_t counter_before;
  std::uint64_t counter_after;
  std::uint32_t selected_ids[kMaxEmitted];
  std::uint32_t reserved1;
  double logprobs[kMaxEmitted];
  std::uint64_t reserved_tail;
};

static_assert(std::is_standard_layout<ControlV1>::value,
              "decode control record must remain standard-layout");
static_assert(sizeof(ControlV1) == 144U,
              "decode control record ABI must remain 144 bytes");
static_assert(alignof(ControlV1) == 16U,
              "decode control record ABI must remain 16-byte aligned");
static_assert(std::is_standard_layout<SelectorRecordV1>::value,
              "selector record must remain standard-layout");
static_assert(sizeof(SelectorRecordV1) == 16U,
              "selector record ABI must remain 16 bytes");
static_assert(std::is_standard_layout<ResultV1>::value,
              "decode control result must remain standard-layout");
static_assert(sizeof(ResultV1) == 192U,
              "decode control result ABI must remain 192 bytes");
static_assert(alignof(ResultV1) == 16U,
              "decode control result ABI must remain 16-byte aligned");
static_assert(offsetof(ResultV1, selected_ids) == 72U,
              "decode control result token offset changed");
static_assert(offsetof(ResultV1, logprobs) == 112U,
              "decode control result logprob offset changed");
static_assert(offsetof(ResultV1, reserved_padding) == 36U,
              "decode control result padding offset changed");
static_assert(offsetof(ResultV1, reserved_tail) == 184U,
              "decode control result tail offset changed");

inline constexpr const char *kControlKernelId = "decode_control.v1";
inline constexpr const char *kControlDeviceSymbol = "sllm_decode_control_v1";
inline constexpr const char *kPhaseKernelId = "decode_control.phase.v1";
inline constexpr const char *kPhaseDeviceSymbol =
    "sllm_decode_control_begin_phase_v1";
inline constexpr const char *kCommitKernelId = "decode_control.commit.v1";
inline constexpr const char *kCommitDeviceSymbol =
    "sllm_decode_control_commit_v1";
inline constexpr const char *kGatherKernelId = "decode_control.gather.v1";
inline constexpr const char *kGatherDeviceSymbol =
    "sllm_decode_control_gather_v1";

hipError_t launch_begin_phase(ControlV1 *control, std::uint32_t kind,
                              std::uint32_t index, std::uint32_t rows,
                              hipStream_t stream) noexcept;

hipError_t
launch_commit_step(ControlV1 *control, const void *record, InputKind input_kind,
                   const std::uint32_t *stop_ids, std::uint32_t stop_count,
                   std::uint32_t vocabulary_size, ResultV1 *result_ring,
                   hipStream_t stream) noexcept;

hipError_t launch_gather_selector_token(const SelectorRecordV1 *selection,
                                        std::int32_t *output_token,
                                        hipStream_t stream) noexcept;

hipError_t launch_gather_decision_tokens(
    const ControlV1 *control,
    const sllm_token_selector_pq::DecisionRecordV1 *decision,
    std::int32_t *output_tokens, std::uint32_t output_capacity,
    hipStream_t stream) noexcept;

hipError_t launch_gather_result_tokens(const ResultV1 *result,
                                       std::int32_t *output_tokens,
                                       std::uint32_t output_capacity,
                                       hipStream_t stream) noexcept;

hipError_t
launch_gather_hidden(ControlV1 *control, const std::uint16_t *hidden_rows,
                     std::uint32_t row_count, std::uint32_t hidden_width,
                     std::uint16_t *output_hidden, hipStream_t stream) noexcept;

hipError_t launch_gather_active_token(ControlV1 *control,
                                      const std::int32_t *draft_ids,
                                      std::uint32_t draft_capacity,
                                      std::int32_t *output_token,
                                      hipStream_t stream) noexcept;

hipError_t launch_gather_active_hidden(ControlV1 *control,
                                       const std::uint16_t *target_hidden_rows,
                                       std::uint32_t target_row_count,
                                       const std::uint16_t *previous_hidden,
                                       std::uint32_t hidden_width,
                                       std::uint16_t *output_hidden,
                                       hipStream_t stream) noexcept;

} // namespace sllm_decode_control

#endif // SLLM_DECODE_CONTROL_KERNEL_INTERNAL_HPP
