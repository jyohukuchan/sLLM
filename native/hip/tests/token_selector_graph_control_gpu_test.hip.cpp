#include "decode_control_kernel_internal.hpp"
#include "token_selector_kernel_internal.hpp"
#include "token_selector_pq_internal.hpp"

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ControlV1 = sllm_decode_control::ControlV1;
using DecisionRecord = sllm_token_selector_pq::DecisionRecordV1;
using SupportRecord = sllm_token_selector_pq::SupportRecordV1;
using SelectorRecord = sllm_token_selector_record_t;

constexpr std::uint32_t kVocab = 64U;
constexpr std::uint64_t kWorkspaceBytes =
    SLLM_HIP_TOKEN_SELECTOR_K0_WORKSPACE_BYTES;
constexpr std::uint64_t kSeed = UINT64_C(0x0123456789abcdef);

bool hip_ok(const hipError_t status, const char *const operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::cerr << operation << ": " << hipGetErrorString(status) << '\n';
  return false;
}

template <typename T> struct Device final {
  T *value = nullptr;

  bool alloc(const std::size_t count, const char *const operation) {
    return hip_ok(
        hipMalloc(reinterpret_cast<void **>(&value), count * sizeof(T)),
        operation);
  }

  bool upload(const T *const host, const std::size_t count,
              const hipStream_t stream, const char *const operation) const {
    return hip_ok(hipMemcpyAsync(value, host, count * sizeof(T),
                                 hipMemcpyHostToDevice, stream),
                  operation);
  }

  bool download(T *const host, const std::size_t count,
                const hipStream_t stream, const char *const operation) const {
    return hip_ok(hipMemcpyAsync(host, value, count * sizeof(T),
                                 hipMemcpyDeviceToHost, stream),
                  operation);
  }

  ~Device() {
    if (value != nullptr) {
      (void)hipFree(value);
    }
  }

  Device() = default;
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
};

std::uint16_t bf16(const float value) {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<std::uint16_t>(bits >> 16U);
}

ControlV1 make_control(const std::uint64_t counter,
                       const std::uint32_t width = 2U) {
  ControlV1 value{};
  value.version = sllm_decode_control::kVersion;
  value.status = static_cast<std::uint32_t>(sllm_decode_control::Status::Ok);
  value.mode = sllm_decode_control::kModeMtp;
  value.width = width;
  value.active_width = value.width;
  value.model_position = 100U;
  value.sampler_counter = counter;
  value.seed = kSeed;
  value.generation = 0U;
  value.capacity = 1U << 20U;
  value.output_limit = 128U;
  value.output_count = 1U;
  value.pending_token = 3U;
  value.stop_row = sllm_decode_control::kNoStop;
  value.hidden_row = sllm_decode_control::kNoStop;
  value.phase_position = value.model_position;
  value.phase_counter = counter;
  value.phase_seed = kSeed;
  value.phase_rows = 1U;
  value.phase_index = 0U;
  value.phase_kind = sllm_decode_control::kPhaseTarget;
  value.phase_active = 1U;
  return value;
}

SupportRecord support() {
  SupportRecord value{};
  value.version = sllm_token_selector_pq::kSupportVersion;
  value.status = 0U;
  value.count = 3U;
  value.reserved = 0U;
  value.ids[0] = 10U;
  value.ids[1] = 11U;
  value.ids[2] = 12U;
  value.probabilities[0] = 0.5;
  value.probabilities[1] = 0.3;
  value.probabilities[2] = 0.2;
  return value;
}

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

bool run_fixed(TestContext *const test, const hipStream_t stream) {
  std::vector<std::uint16_t> logits(kVocab);
  std::vector<float> additive(kVocab, 0.0F);
  std::vector<std::uint8_t> mask(kVocab, 1U);
  for (std::uint32_t index = 0U; index < kVocab; ++index) {
    logits[index] = bf16(static_cast<float>((index * 13U) % 37U) / 7.0F);
  }
  Device<std::uint16_t> device_logits;
  Device<float> device_additive;
  Device<std::uint8_t> device_mask;
  Device<std::uint8_t> device_workspace;
  Device<SelectorRecord> device_output;
  Device<ControlV1> device_control;
  if (!device_logits.alloc(logits.size(), "fixed logits allocation") ||
      !device_additive.alloc(additive.size(), "fixed additive allocation") ||
      !device_mask.alloc(mask.size(), "fixed mask allocation") ||
      !device_workspace.alloc(kWorkspaceBytes, "fixed workspace allocation") ||
      !device_output.alloc(1U, "fixed output allocation") ||
      !device_control.alloc(1U, "fixed control allocation") ||
      !device_logits.upload(logits.data(), logits.size(), stream,
                            "fixed logits upload") ||
      !device_additive.upload(additive.data(), additive.size(), stream,
                              "fixed additive upload") ||
      !device_mask.upload(mask.data(), mask.size(), stream,
                          "fixed mask upload") ||
      !hip_ok(hipStreamSynchronize(stream), "fixed setup synchronize")) {
    return false;
  }

  std::array<SelectorRecord, 3> eager{};
  std::array<SelectorRecord, 3> graph{};
  std::array<ControlV1, 3> controls{};
  for (std::uint32_t index = 0U; index < controls.size(); ++index) {
    controls[index] = make_control(index);
    // BEGIN_PHASE is part of the captured graph, so launchers must not
    // inspect this still-inactive device record on the host during capture.
    controls[index].phase_rows = 0U;
    controls[index].phase_active = 0U;
    if (!hip_ok(sllm_token_selector_kernel::launch(
                    device_logits.value, device_additive.value,
                    device_mask.value, device_workspace.value, kVocab, 1.0F,
                    20U, 0.95F,
                    SLLM_HIP_TOKEN_SELECTOR_FLAG_ADDITIVE_PRESENT |
                        SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT,
                    kSeed, index, device_output.value, stream),
                "fixed eager launch") ||
        !hip_ok(hipStreamSynchronize(stream), "fixed eager synchronize") ||
        !device_output.download(&eager[index], 1U, stream,
                                "fixed eager result") ||
        !hip_ok(hipStreamSynchronize(stream),
                "fixed eager result synchronize")) {
      return false;
    }
  }

  if (!device_control.upload(&controls[0], 1U, stream,
                             "fixed graph control upload") ||
      !hip_ok(hipStreamSynchronize(stream),
              "fixed graph control synchronize") ||
      !hip_ok(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
              "fixed graph begin capture") ||
      !hip_ok(sllm_decode_control::launch_begin_phase(
                  device_control.value, sllm_decode_control::kPhaseTarget, 0U,
                  1U, stream),
              "fixed graph captured phase begin") ||
      !hip_ok(sllm_token_selector_kernel::launch_graph_fixed_k20_topp(
                  device_logits.value, device_additive.value, device_mask.value,
                  device_workspace.value, kVocab,
                  SLLM_HIP_TOKEN_SELECTOR_FLAG_ADDITIVE_PRESENT |
                      SLLM_HIP_TOKEN_SELECTOR_FLAG_MASK_PRESENT,
                  device_control.value, 0U, device_output.value, stream),
              "fixed graph captured launch")) {
    return false;
  }
  hipGraph_t graph_handle = nullptr;
  if (!hip_ok(hipStreamEndCapture(stream, &graph_handle),
              "fixed graph end capture") ||
      graph_handle == nullptr) {
    return false;
  }
  hipGraphExec_t graph_exec = nullptr;
  hipGraphNode_t error_node = nullptr;
  std::array<char, 1024> log{};
  if (!hip_ok(hipGraphInstantiate(&graph_exec, graph_handle, &error_node,
                                  log.data(), log.size()),
              "fixed graph instantiate") ||
      graph_exec == nullptr) {
    (void)hipGraphDestroy(graph_handle);
    return false;
  }
  for (std::uint32_t index = 0U; index < controls.size(); ++index) {
    if (!device_control.upload(&controls[index], 1U, stream,
                               "fixed graph counter upload") ||
        !hip_ok(hipGraphLaunch(graph_exec, stream), "fixed graph replay") ||
        !device_output.download(&graph[index], 1U, stream,
                                "fixed graph result") ||
        !hip_ok(hipStreamSynchronize(stream),
                "fixed graph replay synchronize")) {
      (void)hipGraphExecDestroy(graph_exec);
      (void)hipGraphDestroy(graph_handle);
      return false;
    }
    test->expect(
        std::memcmp(&eager[index], &graph[index], sizeof(SelectorRecord)) == 0,
        "fixed graph selector is N0 against eager baseline at counter " +
            std::to_string(index));
  }
  (void)hipGraphExecDestroy(graph_exec);
  (void)hipGraphDestroy(graph_handle);
  return true;
}

bool run_pq(TestContext *const test, const hipStream_t stream) {
  constexpr std::uint32_t width = 2U;
  const SupportRecord target_host = support();
  const SupportRecord draft_host = support();
  std::array<SupportRecord, width + 1U> target_rows{};
  std::array<SupportRecord, width> draft_rows{};
  std::array<std::uint32_t, width> draft_ids = {10U, 11U};
  target_rows.fill(target_host);
  draft_rows.fill(draft_host);
  Device<SupportRecord> target;
  Device<SupportRecord> draft;
  Device<std::uint32_t> draft_ids_device;
  Device<DecisionRecord> decision;
  Device<ControlV1> control;
  if (!target.alloc(target_rows.size(), "PQ target allocation") ||
      !draft.alloc(draft_rows.size(), "PQ draft allocation") ||
      !draft_ids_device.alloc(draft_ids.size(), "PQ draft IDs allocation") ||
      !decision.alloc(1U, "PQ decision allocation") ||
      !control.alloc(1U, "PQ control allocation") ||
      !target.upload(target_rows.data(), target_rows.size(), stream,
                     "PQ target upload") ||
      !draft.upload(draft_rows.data(), draft_rows.size(), stream,
                    "PQ draft upload") ||
      !draft_ids_device.upload(draft_ids.data(), draft_ids.size(), stream,
                               "PQ draft IDs upload") ||
      !hip_ok(hipStreamSynchronize(stream), "PQ setup synchronize")) {
    return false;
  }
  std::array<DecisionRecord, 3> eager{};
  std::array<DecisionRecord, 3> graph{};
  std::array<ControlV1, 3> controls{};
  sllm_token_selector_pq::DraftIdsV1 host_ids{};
  host_ids.ids[0] = draft_ids[0];
  host_ids.ids[1] = draft_ids[1];
  for (std::uint32_t index = 0U; index < controls.size(); ++index) {
    controls[index] = make_control(index);
    controls[index].phase_kind = sllm_decode_control::kPhaseDraft;
    controls[index].phase_index = 1U;
    controls[index].phase_rows = 1U;
    controls[index].phase_counter = index * 9U + 1U;
    controls[index].phase_seed =
        kSeed ^ sllm_decode_control::kQDraftSeedDomain64;
    if (!hip_ok(sllm_token_selector_pq_kernel::launch(
                    target.value, width + 1U, draft.value, width, host_ids,
                    width, width, kSeed, index, decision.value, stream),
                "PQ eager launch") ||
        !hip_ok(hipStreamSynchronize(stream), "PQ eager synchronize") ||
        !decision.download(&eager[index], 1U, stream, "PQ eager result") ||
        !hip_ok(hipStreamSynchronize(stream), "PQ eager result synchronize")) {
      return false;
    }
  }
  if (!control.upload(&controls[0], 1U, stream, "PQ graph control upload") ||
      !hip_ok(hipStreamSynchronize(stream), "PQ graph control synchronize") ||
      !hip_ok(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
              "PQ graph begin capture") ||
      !hip_ok(sllm_token_selector_pq_kernel::launch_graph(
                  target.value, width + 1U, draft.value, width,
                  draft_ids_device.value, width, width, nullptr, control.value,
                  0U, decision.value, stream),
              "PQ graph captured launch")) {
    return false;
  }
  hipGraph_t graph_handle = nullptr;
  if (!hip_ok(hipStreamEndCapture(stream, &graph_handle),
              "PQ graph end capture") ||
      graph_handle == nullptr) {
    return false;
  }
  hipGraphExec_t graph_exec = nullptr;
  hipGraphNode_t error_node = nullptr;
  std::array<char, 1024> log{};
  if (!hip_ok(hipGraphInstantiate(&graph_exec, graph_handle, &error_node,
                                  log.data(), log.size()),
              "PQ graph instantiate") ||
      graph_exec == nullptr) {
    (void)hipGraphDestroy(graph_handle);
    return false;
  }
  for (std::uint32_t index = 0U; index < controls.size(); ++index) {
    if (!control.upload(&controls[index], 1U, stream,
                        "PQ graph counter upload") ||
        !hip_ok(hipGraphLaunch(graph_exec, stream), "PQ graph replay") ||
        !decision.download(&graph[index], 1U, stream, "PQ graph result") ||
        !hip_ok(hipStreamSynchronize(stream), "PQ graph replay synchronize")) {
      (void)hipGraphExecDestroy(graph_exec);
      (void)hipGraphDestroy(graph_handle);
      return false;
    }
    test->expect(
        std::memcmp(&eager[index], &graph[index], sizeof(DecisionRecord)) == 0,
        "PQ graph decision is N0 against eager baseline at counter " +
            std::to_string(index));
  }
  (void)hipGraphExecDestroy(graph_exec);
  (void)hipGraphDestroy(graph_handle);
  return true;
}

bool run_active_width_pq(TestContext *const test, const hipStream_t stream) {
  std::uint32_t capacity_cases = 0U;
  std::uint32_t budget_cases = 0U;
  std::uint64_t counter = 31U;
  for (std::uint32_t width = 1U; width <= sllm_token_selector_pq::kMaxWidth;
       ++width) {
    std::vector<SupportRecord> target_rows(width + 1U, support());
    std::vector<SupportRecord> draft_rows(width, support());
    std::vector<std::uint32_t> draft_ids(width);
    sllm_token_selector_pq::DraftIdsV1 host_ids{};
    for (std::uint32_t index = 0U; index < width; ++index) {
      draft_ids[index] = 10U + (index % 3U);
      host_ids.ids[index] = draft_ids[index];
    }
    sllm_decode_control::SelectorRecordV1 target_selector{};
    target_selector.token_id = static_cast<std::int32_t>(40U + width);
    target_selector.status = 0U;
    target_selector.logprob = -0.125F * static_cast<float>(width);
    target_selector.reserved = 0U;

    Device<SupportRecord> target;
    Device<SupportRecord> draft;
    Device<std::uint32_t> device_draft_ids;
    Device<sllm_decode_control::SelectorRecordV1> device_target_selector;
    Device<DecisionRecord> decision;
    Device<ControlV1> control;
    if (!target.alloc(target_rows.size(), "active PQ target allocation") ||
        !draft.alloc(draft_rows.size(), "active PQ draft allocation") ||
        !device_draft_ids.alloc(draft_ids.size(), "active PQ IDs allocation") ||
        !device_target_selector.alloc(1U,
                                      "active PQ target selector allocation") ||
        !decision.alloc(1U, "active PQ decision allocation") ||
        !control.alloc(1U, "active PQ control allocation") ||
        !target.upload(target_rows.data(), target_rows.size(), stream,
                       "active PQ target upload") ||
        !draft.upload(draft_rows.data(), draft_rows.size(), stream,
                      "active PQ draft upload") ||
        !device_draft_ids.upload(draft_ids.data(), draft_ids.size(), stream,
                                 "active PQ IDs upload") ||
        !device_target_selector.upload(&target_selector, 1U, stream,
                                       "active PQ target selector upload")) {
      return false;
    }
    ControlV1 capture_control = make_control(counter, width);
    if (!control.upload(&capture_control, 1U, stream,
                        "active PQ capture control upload") ||
        !hip_ok(hipStreamSynchronize(stream),
                "active PQ capture setup synchronize") ||
        !hip_ok(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
                "active PQ begin capture") ||
        !hip_ok(sllm_decode_control::launch_begin_phase(
                    control.value, sllm_decode_control::kPhaseDraft, 0U, 1U,
                    stream),
                "active PQ captured draft begin") ||
        !hip_ok(sllm_decode_control::launch_begin_phase(
                    control.value, sllm_decode_control::kPhaseTarget, 0U,
                    width + 1U, stream),
                "active PQ captured target begin") ||
        !hip_ok(sllm_token_selector_pq_kernel::launch_graph(
                    target.value, width + 1U, draft.value, width,
                    device_draft_ids.value, width, width,
                    device_target_selector.value, control.value, 0U,
                    decision.value, stream),
                "active PQ captured launch")) {
      return false;
    }
    hipGraph_t graph_handle = nullptr;
    if (!hip_ok(hipStreamEndCapture(stream, &graph_handle),
                "active PQ end capture") ||
        graph_handle == nullptr) {
      return false;
    }
    hipGraphExec_t graph_exec = nullptr;
    hipGraphNode_t error_node = nullptr;
    std::array<char, 1024> log{};
    if (!hip_ok(hipGraphInstantiate(&graph_exec, graph_handle, &error_node,
                                    log.data(), log.size()),
                "active PQ instantiate") ||
        graph_exec == nullptr) {
      (void)hipGraphDestroy(graph_handle);
      return false;
    }

    const auto run_case = [&](const ControlV1 &initial,
                              const std::uint32_t expected_active,
                              const std::string &description) {
      DecisionRecord graph_decision{};
      ControlV1 observed{};
      bool valid = control.upload(&initial, 1U, stream,
                                  "active PQ replay control upload") &&
                   hip_ok(hipGraphLaunch(graph_exec, stream),
                          "active PQ graph replay") &&
                   decision.download(&graph_decision, 1U, stream,
                                     "active PQ graph decision read") &&
                   control.download(&observed, 1U, stream,
                                    "active PQ graph control read") &&
                   hip_ok(hipStreamSynchronize(stream),
                          "active PQ graph replay synchronize");
      test->expect(
          valid &&
              observed.status ==
                  static_cast<std::uint32_t>(sllm_decode_control::Status::Ok) &&
              observed.active_width == expected_active &&
              observed.phase_kind == sllm_decode_control::kPhaseTarget &&
              observed.phase_rows == expected_active + 1U &&
              observed.phase_active == 1U,
          "active PQ device phase " + description);
      if (!valid) {
        return false;
      }

      DecisionRecord expected{};
      if (expected_active == 0U) {
        sllm_token_selector_pq::clear_decision(&expected);
        expected.width = 0U;
        expected.accepted_count = 0U;
        expected.emitted_count = 1U;
        expected.rejected_at = sllm_token_selector_pq::kNoRejection;
        expected.draws_used = 1U;
        expected.emitted_ids[0] =
            static_cast<std::uint32_t>(target_selector.token_id);
        expected.target_logprobs[0] =
            static_cast<double>(target_selector.logprob);
      } else {
        valid = hip_ok(sllm_token_selector_pq_kernel::launch(
                           target.value, expected_active + 1U, draft.value,
                           expected_active, host_ids, expected_active,
                           expected_active, initial.seed,
                           initial.sampler_counter, decision.value, stream),
                       "active PQ eager prefix launch") &&
                decision.download(&expected, 1U, stream,
                                  "active PQ eager prefix read") &&
                hip_ok(hipStreamSynchronize(stream),
                       "active PQ eager prefix synchronize");
      }
      test->expect(valid && std::memcmp(&expected, &graph_decision,
                                        sizeof(DecisionRecord)) == 0,
                   (expected_active == 0U
                        ? "active PQ width-zero target selector is exact "
                        : "active PQ prefix matches eager device oracle ") +
                       description);
      return valid;
    };

    bool valid = true;
    for (std::uint32_t remaining = 1U; remaining <= width + 1U; ++remaining) {
      ControlV1 initial = make_control(counter++, width);
      initial.capacity = initial.model_position + remaining;
      initial.output_count = 9U;
      initial.output_limit = initial.output_count + remaining;
      valid = run_case(initial, remaining - 1U,
                       "width=" + std::to_string(width) +
                           " remaining=" + std::to_string(remaining)) &&
              valid;
      ++capacity_cases;
    }
    for (std::uint32_t budget = 1U; budget <= width + 1U; ++budget) {
      ControlV1 initial = make_control(counter++, width);
      initial.capacity = initial.model_position + width + 1U;
      initial.output_count = 15U;
      initial.output_limit = initial.output_count + budget;
      const std::uint32_t expected_active = budget < width ? budget : width;
      valid = run_case(initial, expected_active,
                       "width=" + std::to_string(width) +
                           " budget=" + std::to_string(budget)) &&
              valid;
      ++budget_cases;
    }
    (void)hipGraphExecDestroy(graph_exec);
    (void)hipGraphDestroy(graph_handle);
    if (!valid) {
      return false;
    }
  }
  std::cout << "token_selector_active_width capacity_cases=" << capacity_cases
            << " budget_cases=" << budget_cases << '\n';
  return true;
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
      if (argc > 2)
        expected_target = argv[2];
    } else {
      expected_target = argv[1];
    }
  }
  if (!hip_ok(hipSetDevice(device_index), "hipSetDevice"))
    return EXIT_FAILURE;
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, device_index),
              "hipGetDeviceProperties"))
    return EXIT_FAILURE;
  if (!expected_target.empty() && expected_target != properties.gcnArchName) {
    std::cerr << "expected target " << expected_target << " but HIP reports "
              << properties.gcnArchName << '\n';
    return EXIT_FAILURE;
  }
  hipStream_t stream = nullptr;
  if (!hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
              "hipStreamCreateWithFlags"))
    return EXIT_FAILURE;
  TestContext test;
  (void)run_fixed(&test, stream);
  (void)run_pq(&test, stream);
  (void)run_active_width_pq(&test, stream);
  (void)hipStreamSynchronize(stream);
  (void)hipStreamDestroy(stream);
  std::cout << "token_selector_graph_control_gpu status="
            << (test.failures == 0U ? "PASS" : "FAIL")
            << " checks=" << test.checks << " failures=" << test.failures
            << " target=" << properties.gcnArchName
            << " device=" << device_index << '\n';
  return test.failures == 0U ? EXIT_SUCCESS : EXIT_FAILURE;
}
