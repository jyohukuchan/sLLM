#include "decode_control_kernel_internal.hpp"

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace {

using sllm_decode_control::ControlV1;
using sllm_decode_control::InputKind;
using sllm_decode_control::ResultV1;
using sllm_decode_control::SelectorRecordV1;
using sllm_decode_control::Status;
using DecisionRecord = sllm_token_selector_pq::DecisionRecordV1;

constexpr std::uint32_t kVocabulary = 64U;
constexpr std::uint64_t kModelPosition = 100U;
constexpr std::uint64_t kSamplerCounter = 7U;
constexpr std::uint64_t kSeed = UINT64_C(0x0123456789abcdef);
constexpr std::uint64_t kCapacity = 1U << 20U;
constexpr std::uint64_t kOutputLimit = 128U;

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::cerr << operation << ": " << hipGetErrorString(status) << '\n';
  return false;
}

template <typename T> struct DeviceAllocation final {
  T *pointer = nullptr;

  bool allocate(const std::size_t count, const char *const operation) {
    return hip_ok(
        hipMalloc(reinterpret_cast<void **>(&pointer), count * sizeof(T)),
        operation);
  }

  bool upload(const T *const source, const std::size_t count,
              hipStream_t stream, const char *const operation) const {
    return hip_ok(hipMemcpyAsync(pointer, source, count * sizeof(T),
                                 hipMemcpyHostToDevice, stream),
                  operation);
  }

  bool download(T *const destination, const std::size_t count,
                hipStream_t stream, const char *const operation) const {
    return hip_ok(hipMemcpyAsync(destination, pointer, count * sizeof(T),
                                 hipMemcpyDeviceToHost, stream),
                  operation);
  }

  ~DeviceAllocation() {
    if (pointer != nullptr) {
      (void)hipFree(pointer);
    }
  }

  DeviceAllocation() = default;
  DeviceAllocation(const DeviceAllocation &) = delete;
  DeviceAllocation &operator=(const DeviceAllocation &) = delete;
};

struct TestContext final {
  std::uint32_t checks = 0U;
  std::uint32_t failures = 0U;

  void expect(const bool condition, const std::string &description) {
    ++checks;
    if (!condition) {
      ++failures;
      std::cerr << "FAIL: " << description << '\n';
    }
  }
};

void dump_fixture(const char *const label, const ResultV1 &result) {
  if (std::getenv("SLLM_DECODE_CONTROL_DUMP_FIXTURES") == nullptr) {
    return;
  }
  const auto *const bytes = reinterpret_cast<const std::uint8_t *>(&result);
  std::cout << "FIXTURE " << label << ' ' << std::hex << std::setfill('0');
  for (std::size_t index = 0U; index < sizeof(ResultV1); ++index) {
    std::cout << std::setw(2) << static_cast<unsigned int>(bytes[index]);
  }
  std::cout << std::dec << '\n';
}

ControlV1 control_template(const std::uint32_t mode = 1U,
                           const std::uint32_t width = 2U) {
  ControlV1 control{};
  control.version = sllm_decode_control::kVersion;
  control.status = static_cast<std::uint32_t>(Status::Ok);
  control.mode = mode;
  control.width = width;
  control.model_position = kModelPosition;
  control.sampler_counter = kSamplerCounter;
  control.seed = kSeed;
  control.generation = 0U;
  control.capacity = kCapacity;
  control.output_limit = kOutputLimit;
  control.output_count = 1U;
  control.pending_token = 3U;
  control.halted = 0U;
  control.commit_rows = 0U;
  control.publish_count = 0U;
  control.accepted_count = 0U;
  control.stop_row = sllm_decode_control::kNoStop;
  control.hidden_row = sllm_decode_control::kNoStop;
  control.active_width = width;
  control.phase_active = 0U;
  return control;
}

DecisionRecord decision_template(const std::uint32_t accepted,
                                 const std::uint32_t width = 2U) {
  DecisionRecord decision{};
  decision.version = sllm_token_selector_pq::kDecisionVersion;
  decision.status = 0U;
  decision.accepted_count = accepted;
  decision.emitted_count = accepted + 1U;
  decision.rejected_at =
      accepted == width ? sllm_token_selector_pq::kNoRejection : accepted;
  decision.width = width;
  decision.draws_used = accepted == width ? width + 1U : accepted + 2U;
  decision.reserved0 = 0U;
  decision.reserved1 = 0U;
  for (std::uint32_t index = 0U; index < decision.emitted_count; ++index) {
    decision.emitted_ids[index] = 10U + index;
    decision.target_logprobs[index] = -0.25 - static_cast<double>(index);
  }
  return decision;
}

bool run_commit(TestContext *const test, hipStream_t stream,
                const ControlV1 &initial, const DecisionRecord &decision,
                const std::uint32_t *const stop_ids,
                const std::uint32_t stop_count, ControlV1 *const control_out,
                std::array<ResultV1, sllm_decode_control::kResultRingSlots>
                    *const results_out) {
  DeviceAllocation<ControlV1> device_control;
  DeviceAllocation<DecisionRecord> device_decision;
  DeviceAllocation<std::uint32_t> device_stops;
  DeviceAllocation<ResultV1> device_results;
  if (!device_control.allocate(1U, "allocate control") ||
      !device_decision.allocate(1U, "allocate decision") ||
      !device_results.allocate(sllm_decode_control::kResultRingSlots,
                               "allocate result ring")) {
    return false;
  }
  if (stop_count != 0U &&
      (!device_stops.allocate(stop_count, "allocate stop IDs") ||
       !device_stops.upload(stop_ids, stop_count, stream, "upload stop IDs"))) {
    return false;
  }
  if (!device_control.upload(&initial, 1U, stream, "upload control") ||
      !device_decision.upload(&decision, 1U, stream, "upload decision") ||
      !hip_ok(hipMemsetAsync(device_results.pointer, 0,
                             sizeof(ResultV1) *
                                 sllm_decode_control::kResultRingSlots,
                             stream),
              "clear result ring") ||
      !hip_ok(hipStreamSynchronize(stream), "synchronize setup")) {
    return false;
  }
  if (!hip_ok(sllm_decode_control::launch_commit_step(
                  device_control.pointer, device_decision.pointer,
                  InputKind::Decision, device_stops.pointer, stop_count,
                  kVocabulary, device_results.pointer, stream),
              "launch commit")) {
    return false;
  }
  if (!hip_ok(hipStreamSynchronize(stream), "synchronize commit") ||
      !device_control.download(control_out, 1U, stream, "download control") ||
      !device_results.download(results_out->data(),
                               sllm_decode_control::kResultRingSlots, stream,
                               "download result ring") ||
      !hip_ok(hipStreamSynchronize(stream), "synchronize result")) {
    return false;
  }
  (void)test;
  return true;
}

bool run_selector_commit(
    hipStream_t stream, const ControlV1 &initial,
    const SelectorRecordV1 &selector, ControlV1 *const control_out,
    std::array<ResultV1, sllm_decode_control::kResultRingSlots>
        *const results_out) {
  DeviceAllocation<ControlV1> device_control;
  DeviceAllocation<SelectorRecordV1> device_selector;
  DeviceAllocation<ResultV1> device_results;
  if (!device_control.allocate(1U, "allocate selector control") ||
      !device_selector.allocate(1U, "allocate selector record") ||
      !device_results.allocate(sllm_decode_control::kResultRingSlots,
                               "allocate selector result ring") ||
      !device_control.upload(&initial, 1U, stream, "upload selector control") ||
      !device_selector.upload(&selector, 1U, stream,
                              "upload selector record") ||
      !hip_ok(hipMemsetAsync(device_results.pointer, 0,
                             sizeof(ResultV1) *
                                 sllm_decode_control::kResultRingSlots,
                             stream),
              "clear selector result ring") ||
      !hip_ok(hipStreamSynchronize(stream), "synchronize selector setup")) {
    return false;
  }
  if (!hip_ok(sllm_decode_control::launch_commit_step(
                  device_control.pointer, device_selector.pointer,
                  InputKind::Selector, nullptr, 0U, kVocabulary,
                  device_results.pointer, stream),
              "launch selector commit") ||
      !hip_ok(hipStreamSynchronize(stream), "synchronize selector commit") ||
      !device_control.download(control_out, 1U, stream,
                               "download selector control") ||
      !device_results.download(results_out->data(),
                               sllm_decode_control::kResultRingSlots, stream,
                               "download selector result ring") ||
      !hip_ok(hipStreamSynchronize(stream), "synchronize selector result")) {
    return false;
  }
  return true;
}

bool check_phase_contract(TestContext *const test, hipStream_t stream) {
  ControlV1 host = control_template();
  DeviceAllocation<ControlV1> control;
  if (!control.allocate(1U, "phase control allocation") ||
      !control.upload(&host, 1U, stream, "phase control upload") ||
      !hip_ok(hipStreamSynchronize(stream), "phase setup synchronize")) {
    return false;
  }
  bool valid = true;
  valid =
      valid && hip_ok(sllm_decode_control::launch_begin_phase(
                          control.pointer, sllm_decode_control::kPhaseTarget,
                          2U, 3U, stream),
                      "begin target phase");
  valid = valid && hip_ok(hipStreamSynchronize(stream), "target phase sync");
  ControlV1 observed{};
  valid = valid && control.download(&observed, 1U, stream, "target phase read");
  valid =
      valid && hip_ok(hipStreamSynchronize(stream), "target phase read sync");
  test->expect(observed.phase_position == kModelPosition,
               "target phase keeps model base position");
  test->expect(observed.phase_counter == kSamplerCounter + 2U,
               "target phase advances sampler row counter");
  test->expect(observed.phase_seed == kSeed,
               "target phase preserves sampler seed");
  test->expect(observed.phase_active == 1U, "target phase becomes active");

  valid = valid && hip_ok(sllm_decode_control::launch_begin_phase(
                              control.pointer, sllm_decode_control::kPhaseDraft,
                              1U, 1U, stream),
                          "begin draft phase");
  valid = valid && hip_ok(hipStreamSynchronize(stream), "draft phase sync");
  valid = valid && control.download(&observed, 1U, stream, "draft phase read");
  valid =
      valid && hip_ok(hipStreamSynchronize(stream), "draft phase read sync");
  test->expect(observed.phase_position == kModelPosition + 1U,
               "draft phase advances model position by row");
  test->expect(observed.phase_counter == kSamplerCounter * 9U + 1U,
               "draft phase uses nine-counter stride");
  test->expect(observed.phase_seed ==
                   (kSeed ^ sllm_decode_control::kQDraftSeedDomain64),
               "draft phase uses Q-draft seed domain");
  valid =
      valid && hip_ok(sllm_decode_control::launch_begin_phase(
                          control.pointer, sllm_decode_control::kPhaseMtpAlign,
                          2U, 1U, stream),
                      "begin MTP-align phase");
  valid = valid && hip_ok(hipStreamSynchronize(stream), "MTP-align phase sync");
  valid =
      valid && control.download(&observed, 1U, stream, "MTP-align phase read");
  valid = valid &&
          hip_ok(hipStreamSynchronize(stream), "MTP-align phase read sync");
  test->expect(observed.phase_position == kModelPosition + 2U &&
                   observed.phase_index == 2U && observed.phase_rows == 1U,
               "MTP-align phase uses the accepted width row");
  return valid;
}

bool check_sticky_phase_error(TestContext *const test, hipStream_t stream) {
  ControlV1 initial = control_template();
  SelectorRecordV1 selector{22, 0U, -0.5F, 0U};
  DeviceAllocation<ControlV1> control;
  DeviceAllocation<SelectorRecordV1> device_selector;
  DeviceAllocation<ResultV1> results;
  if (!control.allocate(1U, "sticky control allocation") ||
      !device_selector.allocate(1U, "sticky selector allocation") ||
      !results.allocate(sllm_decode_control::kResultRingSlots,
                        "sticky result allocation") ||
      !control.upload(&initial, 1U, stream, "sticky control upload") ||
      !device_selector.upload(&selector, 1U, stream,
                              "sticky selector upload") ||
      !hip_ok(hipMemsetAsync(results.pointer, 0,
                             sizeof(ResultV1) *
                                 sllm_decode_control::kResultRingSlots,
                             stream),
              "sticky result clear") ||
      !hip_ok(hipStreamSynchronize(stream), "sticky setup sync")) {
    return false;
  }
  bool valid = hip_ok(
      sllm_decode_control::launch_begin_phase(
          control.pointer, sllm_decode_control::kPhaseDraft, 2U, 1U, stream),
      "sticky invalid phase");
  valid = valid && hip_ok(hipStreamSynchronize(stream), "sticky invalid sync");
  valid =
      valid && hip_ok(sllm_decode_control::launch_begin_phase(
                          control.pointer, sllm_decode_control::kPhaseTarget,
                          0U, 1U, stream),
                      "sticky valid phase after invalid");
  valid = valid && hip_ok(hipStreamSynchronize(stream), "sticky valid sync");
  valid = valid && hip_ok(sllm_decode_control::launch_commit_step(
                              control.pointer, device_selector.pointer,
                              InputKind::Selector, nullptr, 0U, kVocabulary,
                              results.pointer, stream),
                          "sticky selector commit");
  valid = valid && hip_ok(hipStreamSynchronize(stream), "sticky commit sync");
  ControlV1 observed{};
  std::array<ResultV1, sllm_decode_control::kResultRingSlots> result_host{};
  valid =
      valid && control.download(&observed, 1U, stream, "sticky control read");
  valid = valid && results.download(result_host.data(),
                                    sllm_decode_control::kResultRingSlots,
                                    stream, "sticky result read");
  valid = valid && hip_ok(hipStreamSynchronize(stream), "sticky read sync");
  test->expect(observed.status ==
                       static_cast<std::uint32_t>(Status::InvalidPhase) &&
                   observed.halted == 1U && observed.phase_active == 0U &&
                   observed.generation == 0U &&
                   observed.model_position == kModelPosition &&
                   observed.commit_rows == 0U && observed.publish_count == 0U,
               "invalid phase remains sticky through begin and commit");
  test->expect(result_host[0U].status != static_cast<std::uint32_t>(Status::Ok),
               "sticky invalid commit cannot produce an OK ring result");
  return valid;
}

bool check_graph_replay(TestContext *const test, hipStream_t stream) {
  ControlV1 initial = control_template();
  DecisionRecord decision = decision_template(2U);
  DeviceAllocation<ControlV1> control;
  DeviceAllocation<DecisionRecord> device_decision;
  DeviceAllocation<ResultV1> results;
  if (!control.allocate(1U, "graph control allocation") ||
      !device_decision.allocate(1U, "graph decision allocation") ||
      !results.allocate(sllm_decode_control::kResultRingSlots,
                        "graph result allocation") ||
      !control.upload(&initial, 1U, stream, "graph control upload") ||
      !device_decision.upload(&decision, 1U, stream, "graph decision upload") ||
      !hip_ok(hipMemsetAsync(results.pointer, 0,
                             sizeof(ResultV1) *
                                 sllm_decode_control::kResultRingSlots,
                             stream),
              "graph result clear") ||
      !hip_ok(hipStreamSynchronize(stream), "graph setup synchronize")) {
    return false;
  }

  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  bool valid =
      hip_ok(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
             "begin decode control graph");
  valid = valid && hip_ok(sllm_decode_control::launch_commit_step(
                              control.pointer, device_decision.pointer,
                              InputKind::Decision, nullptr, 0U, kVocabulary,
                              results.pointer, stream),
                          "capture decode control commit");
  valid = valid && hip_ok(hipStreamEndCapture(stream, &graph),
                          "end decode control graph");
  if (!valid || graph == nullptr) {
    return false;
  }
  std::array<char, 1024> instantiate_log{};
  hipGraphNode_t error_node = nullptr;
  valid = hip_ok(hipGraphInstantiate(&graph_exec, graph, &error_node,
                                     instantiate_log.data(),
                                     instantiate_log.size()),
                 "instantiate decode control graph");
  if (!valid || graph_exec == nullptr) {
    if (graph != nullptr) {
      (void)hipGraphDestroy(graph);
    }
    return false;
  }
  for (std::uint32_t replay = 0U; replay < 3U; ++replay) {
    valid = valid && hip_ok(hipGraphLaunch(graph_exec, stream),
                            "replay decode control graph");
  }
  valid = valid && hip_ok(hipStreamSynchronize(stream),
                          "synchronize decode control graph");
  ControlV1 observed{};
  std::array<ResultV1, sllm_decode_control::kResultRingSlots> results_host{};
  valid =
      valid && control.download(&observed, 1U, stream, "graph control read");
  valid = valid && results.download(results_host.data(),
                                    sllm_decode_control::kResultRingSlots,
                                    stream, "graph result read");
  valid = valid && hip_ok(hipStreamSynchronize(stream), "graph result sync");
  test->expect(observed.generation == 3U,
               "three graph replays advance generation exactly three times");
  test->expect(observed.model_position == kModelPosition + 9U,
               "graph replays advance model position by published prefix");
  test->expect(observed.output_count == 10U,
               "graph replays advance output count by published prefix");
  const ResultV1 &latest = results_host[1U];
  test->expect(latest.generation == 3U,
               "ring slot carries exact latest generation");
  test->expect(latest.count == 3U && latest.commit_rows == 3U,
               "ring slot preserves all-accept count and commit rows");
  test->expect(latest.selected_ids[0] == 10U && latest.selected_ids[2] == 12U,
               "ring slot preserves selected IDs");
  (void)hipGraphExecDestroy(graph_exec);
  (void)hipGraphDestroy(graph);
  return valid;
}

bool check_gather(TestContext *const test, hipStream_t stream) {
  ControlV1 control_host = control_template();
  control_host.publish_count = 2U;
  control_host.commit_rows = 2U;
  control_host.hidden_row = 1U;
  control_host.status = static_cast<std::uint32_t>(Status::Ok);
  DecisionRecord decision = decision_template(1U);
  std::array<std::uint16_t, 12> hidden{};
  for (std::uint32_t index = 0U; index < hidden.size(); ++index) {
    hidden[index] = static_cast<std::uint16_t>(100U + index);
  }
  DeviceAllocation<ControlV1> control;
  DeviceAllocation<DecisionRecord> device_decision;
  DeviceAllocation<std::uint16_t> device_hidden;
  DeviceAllocation<std::int32_t> device_tokens;
  if (!control.allocate(1U, "gather control allocation") ||
      !device_decision.allocate(1U, "gather decision allocation") ||
      !device_hidden.allocate(hidden.size(), "gather hidden allocation") ||
      !device_tokens.allocate(3U, "gather token allocation") ||
      !control.upload(&control_host, 1U, stream, "gather control upload") ||
      !device_decision.upload(&decision, 1U, stream,
                              "gather decision upload") ||
      !device_hidden.upload(hidden.data(), hidden.size(), stream,
                            "gather hidden upload") ||
      !hip_ok(hipMemsetAsync(device_tokens.pointer, 0xff,
                             3U * sizeof(std::int32_t), stream),
              "gather token clear") ||
      !hip_ok(hipStreamSynchronize(stream), "gather setup sync")) {
    return false;
  }
  bool valid = hip_ok(sllm_decode_control::launch_gather_decision_tokens(
                          control.pointer, device_decision.pointer,
                          device_tokens.pointer, 3U, stream),
                      "gather decision tokens");
  DeviceAllocation<std::uint16_t> device_hidden_out;
  valid = valid &&
          device_hidden_out.allocate(4U, "gather hidden output allocation");
  if (valid) {
    valid = hip_ok(sllm_decode_control::launch_gather_hidden(
                       control.pointer, device_hidden.pointer, 3U, 4U,
                       device_hidden_out.pointer, stream),
                   "gather hidden output");
  }
  std::array<std::int32_t, 3> tokens{};
  std::array<std::uint16_t, 4> hidden_out{};
  if (valid) {
    valid = device_tokens.download(tokens.data(), tokens.size(), stream,
                                   "gather token read");
    valid = valid &&
            device_hidden_out.download(hidden_out.data(), hidden_out.size(),
                                       stream, "gather hidden read");
    valid = valid && hip_ok(hipStreamSynchronize(stream), "gather result sync");
  }
  test->expect(tokens[0] == 10 && tokens[1] == 11,
               "decision gather clips to control publish count");
  test->expect(hidden_out[0] == 104 && hidden_out[3] == 107,
               "hidden gather uses selected hidden row");
  for (const bool noop : {true, false}) {
    control_host.status =
        static_cast<uint32_t>(noop ? Status::Noop : Status::Ok);
    control_host.halted = 1U;
    control_host.hidden_row = 0U;
    valid = valid &&
            control.upload(&control_host, 1U, stream, "halted hidden control");
    valid = valid && hip_ok(sllm_decode_control::launch_gather_hidden(
                                control.pointer, device_hidden.pointer, 3U, 4U,
                                device_hidden_out.pointer, stream),
                            "halted hidden gather");
    valid = valid &&
            device_hidden_out.download(hidden_out.data(), hidden_out.size(),
                                       stream, "halted hidden read");
    valid = valid && hip_ok(hipStreamSynchronize(stream), "halted hidden sync");
    test->expect(hidden_out[0] == (noop ? 104U : 100U) &&
                     hidden_out[3] == (noop ? 107U : 103U),
                 noop ? "noop preserves committed hidden carry"
                      : "valid halted commit publishes final hidden carry");
  }
  return valid;
}

bool check_active_width_tail_contract(TestContext *const test,
                                      hipStream_t stream) {
  DeviceAllocation<ControlV1> control;
  if (!control.allocate(1U, "active-width control allocation")) {
    return false;
  }

  bool valid = true;
  std::uint32_t capacity_cases = 0U;
  std::uint32_t budget_cases = 0U;
  for (std::uint32_t width = 1U; width <= sllm_decode_control::kMaxWidth;
       ++width) {
    for (std::uint32_t remaining = 1U; remaining <= width + 1U; ++remaining) {
      ControlV1 initial = control_template(1U, width);
      initial.capacity = kModelPosition + remaining;
      initial.output_count = 11U;
      initial.output_limit = initial.output_count + remaining;
      valid = valid && control.upload(&initial, 1U, stream,
                                      "active-width capacity upload");
      valid =
          valid && hip_ok(sllm_decode_control::launch_begin_phase(
                              control.pointer, sllm_decode_control::kPhaseDraft,
                              0U, 1U, stream),
                          "active-width capacity draft begin");
      ControlV1 observed{};
      valid = valid && control.download(&observed, 1U, stream,
                                        "active-width capacity read");
      valid = valid && hip_ok(hipStreamSynchronize(stream),
                              "active-width capacity synchronize");
      const std::uint32_t expected = remaining - 1U;
      const std::string suffix = " width=" + std::to_string(width) +
                                 " remaining=" + std::to_string(remaining);
      test->expect(observed.status == static_cast<std::uint32_t>(Status::Ok) &&
                       observed.active_width == expected &&
                       observed.phase_active == (expected == 0U ? 0U : 1U),
                   "capacity tail derives active width" + suffix);

      valid = valid &&
              hip_ok(sllm_decode_control::launch_begin_phase(
                         control.pointer, sllm_decode_control::kPhaseTarget, 0U,
                         width + 1U, stream),
                     "active-width target begin");
      valid = valid && control.download(&observed, 1U, stream,
                                        "active-width target read");
      valid = valid && hip_ok(hipStreamSynchronize(stream),
                              "active-width target synchronize");
      test->expect(observed.status == static_cast<std::uint32_t>(Status::Ok) &&
                       observed.phase_active == 1U &&
                       observed.phase_rows == expected + 1U &&
                       observed.phase_position == kModelPosition,
                   "target phase publishes active prefix" + suffix);

      valid = valid &&
              hip_ok(sllm_decode_control::launch_begin_phase(
                         control.pointer, sllm_decode_control::kPhaseMtpAlign,
                         width, 1U, stream),
                     "active-width align begin");
      valid = valid && control.download(&observed, 1U, stream,
                                        "active-width align read");
      valid = valid && hip_ok(hipStreamSynchronize(stream),
                              "active-width align synchronize");
      test->expect(observed.status == static_cast<std::uint32_t>(Status::Ok) &&
                       observed.phase_active == 1U &&
                       observed.phase_index == expected &&
                       observed.phase_position == kModelPosition + expected &&
                       observed.phase_rows == 1U,
                   "MTP align uses dynamic active row" + suffix);
      ++capacity_cases;
    }

    for (std::uint32_t budget = 1U; budget <= width + 1U; ++budget) {
      ControlV1 initial = control_template(1U, width);
      initial.capacity = kModelPosition + width + 1U;
      initial.output_count = 17U;
      initial.output_limit = initial.output_count + budget;
      valid = valid && control.upload(&initial, 1U, stream,
                                      "active-width budget upload");
      valid =
          valid && hip_ok(sllm_decode_control::launch_begin_phase(
                              control.pointer, sllm_decode_control::kPhaseDraft,
                              0U, 1U, stream),
                          "active-width budget draft begin");
      ControlV1 observed{};
      valid = valid && control.download(&observed, 1U, stream,
                                        "active-width budget read");
      valid = valid && hip_ok(hipStreamSynchronize(stream),
                              "active-width budget synchronize");
      const std::uint32_t expected = budget < width ? budget : width;
      test->expect(
          observed.status == static_cast<std::uint32_t>(Status::Ok) &&
              observed.active_width == expected && observed.phase_active == 1U,
          "logical budget bounds active width=" + std::to_string(width) +
              " budget=" + std::to_string(budget));
      ++budget_cases;
    }
  }
  std::cout << "decode_control_active_width capacity_cases=" << capacity_cases
            << " budget_cases=" << budget_cases << '\n';
  return valid;
}

bool check_dynamic_align_gathers(TestContext *const test, hipStream_t stream) {
  constexpr std::uint32_t width = sllm_decode_control::kMaxWidth;
  constexpr std::uint32_t hidden_width = 5U;
  std::array<std::int32_t, width> draft_ids{};
  std::array<std::uint16_t, (width + 1U) * hidden_width> target_hidden{};
  std::array<std::uint16_t, hidden_width> previous_hidden{};
  for (std::uint32_t index = 0U; index < width; ++index) {
    draft_ids[index] = static_cast<std::int32_t>(700U + index);
  }
  for (std::uint32_t row = 0U; row <= width; ++row) {
    for (std::uint32_t lane = 0U; lane < hidden_width; ++lane) {
      target_hidden[row * hidden_width + lane] =
          static_cast<std::uint16_t>(1000U + row * 10U + lane);
    }
  }
  for (std::uint32_t lane = 0U; lane < hidden_width; ++lane) {
    previous_hidden[lane] = static_cast<std::uint16_t>(2000U + lane);
  }

  DeviceAllocation<ControlV1> control;
  DeviceAllocation<std::int32_t> device_draft_ids;
  DeviceAllocation<std::uint16_t> device_target_hidden;
  DeviceAllocation<std::uint16_t> device_previous_hidden;
  DeviceAllocation<std::int32_t> device_token;
  DeviceAllocation<std::uint16_t> device_hidden;
  if (!control.allocate(1U, "dynamic align control allocation") ||
      !device_draft_ids.allocate(width, "dynamic align IDs allocation") ||
      !device_target_hidden.allocate(
          target_hidden.size(), "dynamic align target hidden allocation") ||
      !device_previous_hidden.allocate(previous_hidden.size(),
                                       "dynamic align carry allocation") ||
      !device_token.allocate(1U, "dynamic align token allocation") ||
      !device_hidden.allocate(hidden_width,
                              "dynamic align hidden allocation") ||
      !device_draft_ids.upload(draft_ids.data(), draft_ids.size(), stream,
                               "dynamic align IDs upload") ||
      !device_target_hidden.upload(target_hidden.data(), target_hidden.size(),
                                   stream,
                                   "dynamic align target hidden upload") ||
      !device_previous_hidden.upload(previous_hidden.data(),
                                     previous_hidden.size(), stream,
                                     "dynamic align carry upload")) {
    return false;
  }

  bool valid = true;
  for (std::uint32_t active = 0U; active <= width; ++active) {
    ControlV1 initial = control_template(1U, width);
    initial.active_width = active;
    initial.pending_token = 613U;
    valid = valid && control.upload(&initial, 1U, stream,
                                    "dynamic align control upload");
    valid = valid && hip_ok(hipMemsetAsync(device_token.pointer, 0xff,
                                           sizeof(std::int32_t), stream),
                            "dynamic align token clear");
    valid = valid &&
            hip_ok(hipMemsetAsync(device_hidden.pointer, 0xff,
                                  hidden_width * sizeof(std::uint16_t), stream),
                   "dynamic align hidden clear");
    valid = valid && hip_ok(sllm_decode_control::launch_gather_active_token(
                                control.pointer, device_draft_ids.pointer,
                                width, device_token.pointer, stream),
                            "dynamic align token gather");
    valid = valid && hip_ok(sllm_decode_control::launch_gather_active_hidden(
                                control.pointer, device_target_hidden.pointer,
                                width + 1U, device_previous_hidden.pointer,
                                hidden_width, device_hidden.pointer, stream),
                            "dynamic align hidden gather");
    std::int32_t token = -1;
    std::array<std::uint16_t, hidden_width> hidden{};
    valid = valid && device_token.download(&token, 1U, stream,
                                           "dynamic align token read");
    valid =
        valid && device_hidden.download(hidden.data(), hidden.size(), stream,
                                        "dynamic align hidden read");
    valid = valid && hip_ok(hipStreamSynchronize(stream),
                            "dynamic align gather synchronize");
    const std::int32_t expected_token =
        active == 0U ? 613 : draft_ids[active - 1U];
    const std::uint16_t expected_hidden =
        active == 0U ? previous_hidden[0U]
                     : target_hidden[(active - 1U) * hidden_width];
    test->expect(token == expected_token,
                 "dynamic align token source active=" + std::to_string(active));
    test->expect(
        hidden[0U] == expected_hidden &&
            hidden[hidden_width - 1U] ==
                static_cast<std::uint16_t>(expected_hidden + hidden_width - 1U),
        "dynamic align hidden source active=" + std::to_string(active));
  }
  return valid;
}

bool check_commit_cases(TestContext *const test, hipStream_t stream) {
  bool valid = true;
  ControlV1 observed{};
  std::array<ResultV1, sllm_decode_control::kResultRingSlots> results{};

  ControlV1 selector_control = control_template(0U, 1U);
  SelectorRecordV1 selector{22, 0U, -0.5F, 0U};
  valid = valid && run_selector_commit(stream, selector_control, selector,
                                       &observed, &results);
  dump_fixture("ordinary-selector", results[1U]);
  test->expect(observed.generation == 1U && observed.commit_rows == 1U &&
                   observed.pending_token == 22U,
               "16-byte selector record commits target-only row");

  valid = valid &&
          run_commit(test, stream, control_template(), decision_template(2U),
                     nullptr, 0U, &observed, &results);
  dump_fixture("all-accept", results[1U]);
  test->expect(observed.generation == 1U && observed.commit_rows == 3U &&
                   observed.publish_count == 3U,
               "all-accept decision commits all target rows");
  test->expect(results[1U].halt_flags & sllm_decode_control::HaltAllAccept,
               "all-accept result flag is present");

  valid = valid &&
          run_commit(test, stream, control_template(), decision_template(1U),
                     nullptr, 0U, &observed, &results);
  dump_fixture("partial", results[1U]);
  test->expect(observed.generation == 1U && observed.commit_rows == 2U &&
                   observed.pending_token == 11U,
               "partial decision commits accepted prefix and replacement");

  const std::array<std::uint32_t, 1> stop_ids = {31U};
  DecisionRecord stop_decision = decision_template(2U);
  stop_decision.emitted_ids[1] = 31U;
  valid = valid &&
          run_commit(test, stream, control_template(), stop_decision,
                     stop_ids.data(), stop_ids.size(), &observed, &results);
  dump_fixture("valid-eos", results[1U]);
  test->expect(
      observed.halted == 1U && observed.commit_rows == 2U &&
          observed.pending_token == 3U,
      "mid-block stop commits prefix without publishing stop as pending input");
  test->expect(results[1U].halt_flags & sllm_decode_control::HaltStop,
               "mid-block stop flag is present");

  ControlV1 budget = control_template();
  budget.output_limit = 2U;
  budget.commit_rows = 9U;
  budget.publish_count = 9U;
  valid = valid && run_commit(test, stream, budget, decision_template(2U),
                              nullptr, 0U, &observed, &results);
  dump_fixture("budget-noop", results[1U]);
  test->expect(observed.halted == 1U && observed.publish_count == 1U &&
                   observed.output_count == 2U,
               "budget clips output and halts without extra publication");
  test->expect(results[1U].halt_flags & sllm_decode_control::HaltBudget,
               "budget halt flag is present");

  DecisionRecord invalid = decision_template(2U);
  invalid.status = 1U;
  valid = valid && run_commit(test, stream, control_template(), invalid,
                              nullptr, 0U, &observed, &results);
  dump_fixture("error", results[0U]);
  test->expect(observed.generation == 0U &&
                   observed.model_position == kModelPosition,
               "invalid decision does not advance control state");
  test->expect(results[0U].status ==
                   static_cast<std::uint32_t>(Status::InvalidDecision),
               "invalid decision is recorded as an error");

  ControlV1 overflow = control_template();
  overflow.model_position = std::numeric_limits<std::uint64_t>::max();
  overflow.capacity = std::numeric_limits<std::uint64_t>::max();
  valid = valid && run_commit(test, stream, overflow, decision_template(2U),
                              nullptr, 0U, &observed, &results);
  test->expect(observed.generation == 0U && observed.halted == 1U &&
                   results[0U].status ==
                       static_cast<std::uint32_t>(Status::InvalidVersion),
               "invalid capacity header is rejected without publication");

  ControlV1 inactive = control_template();
  inactive.halted = 1U;
  inactive.commit_rows = 7U;
  inactive.publish_count = 7U;
  valid = valid && run_commit(test, stream, inactive, decision_template(2U),
                              nullptr, 0U, &observed, &results);
  dump_fixture("halted-noop", results[0U]);
  test->expect(
      observed.generation == 0U && observed.model_position == kModelPosition &&
          results[0U].status == static_cast<std::uint32_t>(Status::Noop),
      "zero-active halted control is a no-op");
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  int device_index = 0;
  std::string expected_target;
  if (argc > 1) {
    char *end = nullptr;
    const long parsed = std::strtol(argv[1], &end, 10);
    if (end != argv[1] && end != nullptr && *end == '\0') {
      device_index = static_cast<int>(parsed);
      if (argc > 2) {
        expected_target = argv[2];
      }
    } else {
      expected_target = argv[1];
    }
  }
  if (!expected_target.empty()) {
    // Keep the runner's exact target argument visible in the result. The
    // architecture check is performed after HIP device selection below.
  }
  if (!hip_ok(hipSetDevice(device_index), "hipSetDevice")) {
    return EXIT_FAILURE;
  }
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, device_index),
              "hipGetDeviceProperties")) {
    return EXIT_FAILURE;
  }
  if (!expected_target.empty() && expected_target != properties.gcnArchName) {
    std::cerr << "expected target " << expected_target << " but HIP reports "
              << properties.gcnArchName << '\n';
    return EXIT_FAILURE;
  }
  hipStream_t stream = nullptr;
  if (!hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
              "hipStreamCreateWithFlags")) {
    return EXIT_FAILURE;
  }
  TestContext test;
  (void)check_phase_contract(&test, stream);
  (void)check_sticky_phase_error(&test, stream);
  (void)check_commit_cases(&test, stream);
  (void)check_graph_replay(&test, stream);
  (void)check_gather(&test, stream);
  (void)check_active_width_tail_contract(&test, stream);
  (void)check_dynamic_align_gathers(&test, stream);
  (void)hipStreamSynchronize(stream);
  (void)hipStreamDestroy(stream);
  std::cout << "decode_control_gpu status="
            << (test.failures == 0U ? "PASS" : "FAIL")
            << " checks=" << test.checks << " failures=" << test.failures
            << " target=" << properties.gcnArchName
            << " device=" << device_index << '\n';
  return test.failures == 0U ? EXIT_SUCCESS : EXIT_FAILURE;
}
