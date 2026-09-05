#include <hip/hip_runtime.h>
#include <hip/hip_version.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#ifndef SLLM_TEST_EXPECTED_TARGET
#define SLLM_TEST_EXPECTED_TARGET "unknown"
#endif

namespace {

constexpr uint64_t kM = UINT64_C(1);
constexpr uint64_t kK = UINT64_C(6144);
constexpr uint64_t kN = UINT64_C(5120);
constexpr int kRequestedAlgorithms = 32;
constexpr std::size_t kAlgorithmRank = 7U;
constexpr std::size_t kWarmupCount = 3U;
constexpr std::size_t kReplayCount = 1000U;
constexpr std::array<std::size_t, 7> kCheckpoints = {0U,   1U,   7U,  31U,
                                                     127U, 511U, 999U};

static_assert(HIP_VERSION_MAJOR == 7 && HIP_VERSION_MINOR == 14,
              "the graph probe is pinned to ROCm/HIP 7.14");
static_assert(kAlgorithmRank < static_cast<std::size_t>(kRequestedAlgorithms));

struct Resources {
  hipblasLtHandle_t handle = nullptr;
  hipblasLtMatmulDesc_t operation = nullptr;
  hipblasLtMatrixLayout_t a = nullptr;
  hipblasLtMatrixLayout_t b = nullptr;
  hipblasLtMatrixLayout_t c = nullptr;
  hipblasLtMatrixLayout_t d = nullptr;
  hipblasLtMatmulPreference_t preference = nullptr;
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  std::vector<void *> allocations;
  float alpha = 1.0F;
  float beta = 0.0F;
};

struct StatusReport {
  int heuristic_query = -1;
  int heuristic_count = 0;
  int selected_state = -1;
  std::size_t selected_workspace = static_cast<std::size_t>(-1);
  int warmup_matmul = -1;
  int warmup_sync = -1;
  int capture_begin = -1;
  int capture_matmul = -1;
  int capture_end = -1;
  int node_query = -1;
  std::size_t node_count = 0U;
  int instantiate = -1;
  int replay_launch = -1;
  int replay_sync = -1;
  std::size_t replay_count = 0U;
  std::size_t checkpoint_count = 0U;
  std::size_t bf16_mismatch_count = 0U;
  std::size_t nonfinite_count = 0U;
  std::size_t distinct_checkpoint_outputs = 0U;
};

const char *hip_status_name(const int status) {
  if (status < 0) {
    return "not-run";
  }
  return hipGetErrorName(static_cast<hipError_t>(status));
}

bool hip_success(const hipError_t status, const char *const operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::cerr << operation << " failed: " << hipGetErrorName(status) << " ("
            << hipGetErrorString(status) << ")\n";
  return false;
}

bool lt_success(const hipblasStatus_t status, const char *const operation) {
  if (status == HIPBLAS_STATUS_SUCCESS) {
    return true;
  }
  std::cerr << operation << " failed: hipBLAS status "
            << static_cast<int>(status) << "\n";
  return false;
}

bool allocate_device(Resources &resources, void **const pointer,
                     const std::size_t bytes, const char *const label) {
  const hipError_t status = hipMalloc(pointer, bytes);
  if (!hip_success(status, label)) {
    return false;
  }
  resources.allocations.push_back(*pointer);
  return true;
}

bool is_checkpoint(const std::size_t replay) {
  return std::find(kCheckpoints.begin(), kCheckpoints.end(), replay) !=
         kCheckpoints.end();
}

uint64_t hash_bf16_bits(const std::vector<uint16_t> &values) {
  uint64_t hash = UINT64_C(1469598103934665603);
  for (const uint16_t value : values) {
    hash ^= static_cast<uint64_t>(value & UINT16_C(0xff));
    hash *= UINT64_C(1099511628211);
    hash ^= static_cast<uint64_t>(value >> 8U);
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

bool bf16_is_finite(const uint16_t bits) {
  return (bits & UINT16_C(0x7f80)) != UINT16_C(0x7f80);
}

void fill_weight(std::vector<uint8_t> &weight) {
  // Positive, finite OCP E4M3FN values keep the reduction well inside BF16
  // range after the deliberately small outer scales are applied.
  constexpr std::array<uint8_t, 8> codes = {
      UINT8_C(0x28), UINT8_C(0x2a), UINT8_C(0x2c), UINT8_C(0x2e),
      UINT8_C(0x30), UINT8_C(0x32), UINT8_C(0x34), UINT8_C(0x36)};
  for (std::size_t column = 0U; column < static_cast<std::size_t>(kN);
       ++column) {
    const std::size_t base = column * static_cast<std::size_t>(kK);
    for (std::size_t row = 0U; row < static_cast<std::size_t>(kK); ++row) {
      weight[base + row] = codes[(row + column * 3U) % codes.size()];
    }
  }
}

void fill_activation(std::vector<uint8_t> &activation,
                     const std::size_t replay) {
  constexpr std::array<uint8_t, 8> codes = {
      UINT8_C(0x30), UINT8_C(0x32), UINT8_C(0x34), UINT8_C(0x36),
      UINT8_C(0x38), UINT8_C(0x39), UINT8_C(0x3a), UINT8_C(0x3b)};
  for (std::size_t row = 0U; row < activation.size(); ++row) {
    uint8_t value = codes[(row * 5U + replay * 3U) % codes.size()];
    if (((row + replay * 11U) % 29U) == 0U) {
      value = static_cast<uint8_t>(value | UINT8_C(0x80));
    }
    activation[row] = value;
  }
}

float activation_scale_for_replay(const std::size_t replay) {
  const float step = static_cast<float>((replay * 7U) % 23U);
  return 0x1p-7F * (1.0F + step * 0.03125F);
}

bool check_runtime_identity() {
  int runtime_version = 0;
  if (!hip_success(hipRuntimeGetVersion(&runtime_version),
                   "hipRuntimeGetVersion")) {
    return false;
  }
  const int runtime_major = runtime_version / 10000000;
  const int runtime_minor = (runtime_version / 100000) % 100;
  int device_count = 0;
  if (!hip_success(hipGetDeviceCount(&device_count), "hipGetDeviceCount") ||
      device_count != 1) {
    std::cerr << "expected exactly one visible GPU, got " << device_count
              << "\n";
    return false;
  }
  hipDeviceProp_t properties{};
  if (!hip_success(hipGetDeviceProperties(&properties, 0),
                   "hipGetDeviceProperties")) {
    return false;
  }
  std::cout << "identity compile_target=" << SLLM_TEST_EXPECTED_TARGET
            << " runtime_target=" << properties.gcnArchName
            << " hip_header=" << HIP_VERSION_MAJOR << '.' << HIP_VERSION_MINOR
            << '.' << HIP_VERSION_PATCH << " hip_runtime=" << runtime_version
            << " visible_devices=" << device_count << '\n';
  if (runtime_major != 7 || runtime_minor != 14 ||
      std::strcmp(SLLM_TEST_EXPECTED_TARGET, "gfx1201") != 0 ||
      std::strcmp(properties.gcnArchName, "gfx1201") != 0) {
    std::cerr << "the probe requires exact gfx1201 and ROCm/HIP 7.14\n";
    return false;
  }
  return hip_success(hipSetDevice(0), "hipSetDevice(0)");
}

bool create_plan(Resources &resources, StatusReport &report,
                 const float *const weight_scales,
                 float *const activation_scales,
                 hipblasLtMatmulAlgo_t &algorithm) {
  if (!lt_success(hipblasLtCreate(&resources.handle), "hipblasLtCreate") ||
      !lt_success(hipblasLtMatmulDescCreate(&resources.operation,
                                            HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  "hipblasLtMatmulDescCreate")) {
    return false;
  }

  const hipblasOperation_t trans_a = HIPBLAS_OP_T;
  const hipblasOperation_t trans_b = HIPBLAS_OP_N;
  if (!lt_success(hipblasLtMatmulDescSetAttribute(resources.operation,
                                                  HIPBLASLT_MATMUL_DESC_TRANSA,
                                                  &trans_a, sizeof(trans_a)),
                  "set TRANSA") ||
      !lt_success(hipblasLtMatmulDescSetAttribute(resources.operation,
                                                  HIPBLASLT_MATMUL_DESC_TRANSB,
                                                  &trans_b, sizeof(trans_b)),
                  "set TRANSB")) {
    return false;
  }

  void *weight_scale_pointer = const_cast<float *>(weight_scales);
  void *activation_scale_pointer = activation_scales;
  const hipblasLtMatmulMatrixScale_t scale_mode =
      HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
  if (!lt_success(hipblasLtMatmulDescSetAttribute(
                      resources.operation,
                      HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                      &weight_scale_pointer, sizeof(weight_scale_pointer)),
                  "set A scale pointer") ||
      !lt_success(
          hipblasLtMatmulDescSetAttribute(
              resources.operation, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
              &activation_scale_pointer, sizeof(activation_scale_pointer)),
          "set B scale pointer") ||
      !lt_success(hipblasLtMatmulDescSetAttribute(
                      resources.operation, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                      &scale_mode, sizeof(scale_mode)),
                  "set A outer scale mode") ||
      !lt_success(hipblasLtMatmulDescSetAttribute(
                      resources.operation, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                      &scale_mode, sizeof(scale_mode)),
                  "set B outer scale mode") ||
      !lt_success(hipblasLtMatrixLayoutCreate(&resources.a, HIP_R_8F_E4M3, kK,
                                              kN, static_cast<int64_t>(kK)),
                  "create A layout") ||
      !lt_success(hipblasLtMatrixLayoutCreate(&resources.b, HIP_R_8F_E4M3, kK,
                                              kM, static_cast<int64_t>(kK)),
                  "create B layout") ||
      !lt_success(hipblasLtMatrixLayoutCreate(&resources.c, HIP_R_16BF, kN, kM,
                                              static_cast<int64_t>(kN)),
                  "create C layout") ||
      !lt_success(hipblasLtMatrixLayoutCreate(&resources.d, HIP_R_16BF, kN, kM,
                                              static_cast<int64_t>(kN)),
                  "create D layout") ||
      !lt_success(hipblasLtMatmulPreferenceCreate(&resources.preference),
                  "hipblasLtMatmulPreferenceCreate")) {
    return false;
  }

  const uint64_t workspace_limit = 0U;
  if (!lt_success(hipblasLtMatmulPreferenceSetAttribute(
                      resources.preference,
                      HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                      &workspace_limit, sizeof(workspace_limit)),
                  "set zero workspace limit")) {
    return false;
  }

  std::array<hipblasLtMatmulHeuristicResult_t,
             static_cast<std::size_t>(kRequestedAlgorithms)>
      heuristics{};
  int solution_count = 0;
  const hipblasStatus_t query = hipblasLtMatmulAlgoGetHeuristic(
      resources.handle, resources.operation, resources.a, resources.b,
      resources.c, resources.d, resources.preference, kRequestedAlgorithms,
      heuristics.data(), &solution_count);
  report.heuristic_query = static_cast<int>(query);
  report.heuristic_count = solution_count;
  if (!lt_success(query, "hipblasLtMatmulAlgoGetHeuristic") ||
      solution_count <= static_cast<int>(kAlgorithmRank)) {
    std::cerr << "heuristic rank " << kAlgorithmRank
              << " unavailable; solution_count=" << solution_count << '\n';
    return false;
  }
  const hipblasLtMatmulHeuristicResult_t &selected = heuristics[kAlgorithmRank];
  report.selected_state = static_cast<int>(selected.state);
  report.selected_workspace = selected.workspaceSize;
  if (selected.state != HIPBLAS_STATUS_SUCCESS ||
      selected.workspaceSize != 0U) {
    std::cerr << "heuristic rank " << kAlgorithmRank
              << " is unusable: state=" << static_cast<int>(selected.state)
              << " workspace=" << selected.workspaceSize << '\n';
    return false;
  }
  algorithm = selected.algo;
  return true;
}

hipblasStatus_t launch_matmul(const Resources &resources,
                              const hipblasLtMatmulAlgo_t &algorithm,
                              const uint8_t *const weight,
                              const uint8_t *const activation,
                              uint16_t *const output) {
  return hipblasLtMatmul(resources.handle, resources.operation,
                         &resources.alpha, weight, resources.a, activation,
                         resources.b, &resources.beta, output, resources.c,
                         output, resources.d, &algorithm, nullptr, 0U,
                         resources.stream);
}

bool run_probe(Resources &resources, StatusReport &report) {
  if (!check_runtime_identity()) {
    return false;
  }

  const std::size_t weight_elements = static_cast<std::size_t>(kK * kN);
  const std::size_t activation_elements = static_cast<std::size_t>(kK * kM);
  const std::size_t output_elements = static_cast<std::size_t>(kN * kM);

  std::vector<uint8_t> host_weight(weight_elements);
  std::vector<uint8_t> host_activation(activation_elements);
  std::vector<float> host_weight_scales(static_cast<std::size_t>(kN));
  std::vector<uint16_t> host_graph_output(output_elements);
  std::vector<uint16_t> host_eager_output(output_elements);
  fill_weight(host_weight);
  fill_activation(host_activation, 0U);
  for (std::size_t column = 0U; column < host_weight_scales.size(); ++column) {
    host_weight_scales[column] =
        0x1p-6F * (1.0F + static_cast<float>(column % 13U) * 0.015625F);
  }
  float host_activation_scale = activation_scale_for_replay(0U);

  uint8_t *device_weight = nullptr;
  uint8_t *device_activation = nullptr;
  float *device_weight_scales = nullptr;
  float *device_activation_scale = nullptr;
  uint16_t *device_output = nullptr;
  if (!allocate_device(resources, reinterpret_cast<void **>(&device_weight),
                       host_weight.size(), "allocate weight") ||
      !allocate_device(resources, reinterpret_cast<void **>(&device_activation),
                       host_activation.size(), "allocate activation") ||
      !allocate_device(resources,
                       reinterpret_cast<void **>(&device_weight_scales),
                       host_weight_scales.size() * sizeof(float),
                       "allocate weight scales") ||
      !allocate_device(resources,
                       reinterpret_cast<void **>(&device_activation_scale),
                       sizeof(float), "allocate activation scale") ||
      !allocate_device(resources, reinterpret_cast<void **>(&device_output),
                       host_graph_output.size() * sizeof(uint16_t),
                       "allocate output") ||
      !hip_success(hipStreamCreate(&resources.stream), "hipStreamCreate") ||
      !hip_success(hipMemcpy(device_weight, host_weight.data(),
                             host_weight.size(), hipMemcpyHostToDevice),
                   "copy weight") ||
      !hip_success(hipMemcpy(device_weight_scales, host_weight_scales.data(),
                             host_weight_scales.size() * sizeof(float),
                             hipMemcpyHostToDevice),
                   "copy weight scales") ||
      !hip_success(hipMemcpy(device_activation, host_activation.data(),
                             host_activation.size(), hipMemcpyHostToDevice),
                   "copy initial activation") ||
      !hip_success(hipMemcpy(device_activation_scale, &host_activation_scale,
                             sizeof(float), hipMemcpyHostToDevice),
                   "copy initial activation scale")) {
    return false;
  }

  hipblasLtMatmulAlgo_t algorithm{};
  if (!create_plan(resources, report, device_weight_scales,
                   device_activation_scale, algorithm)) {
    return false;
  }

  for (std::size_t warmup = 0U; warmup < kWarmupCount; ++warmup) {
    const hipblasStatus_t launch = launch_matmul(
        resources, algorithm, device_weight, device_activation, device_output);
    report.warmup_matmul = static_cast<int>(launch);
    if (!lt_success(launch, "eager warmup hipblasLtMatmul")) {
      return false;
    }
  }
  const hipError_t warmup_sync = hipStreamSynchronize(resources.stream);
  report.warmup_sync = static_cast<int>(warmup_sync);
  if (!hip_success(warmup_sync, "eager warmup synchronize")) {
    return false;
  }

  const hipError_t capture_begin =
      hipStreamBeginCapture(resources.stream, hipStreamCaptureModeThreadLocal);
  report.capture_begin = static_cast<int>(capture_begin);
  if (!hip_success(capture_begin, "hipStreamBeginCapture")) {
    return false;
  }
  const hipblasStatus_t capture_matmul = launch_matmul(
      resources, algorithm, device_weight, device_activation, device_output);
  report.capture_matmul = static_cast<int>(capture_matmul);
  const hipError_t capture_end =
      hipStreamEndCapture(resources.stream, &resources.graph);
  report.capture_end = static_cast<int>(capture_end);
  if (!lt_success(capture_matmul, "captured hipblasLtMatmul") ||
      !hip_success(capture_end, "hipStreamEndCapture") ||
      resources.graph == nullptr) {
    return false;
  }

  std::size_t node_count = 0U;
  const hipError_t node_query =
      hipGraphGetNodes(resources.graph, nullptr, &node_count);
  report.node_query = static_cast<int>(node_query);
  report.node_count = node_count;
  if (!hip_success(node_query, "hipGraphGetNodes") || node_count == 0U) {
    std::cerr << "captured graph contains no nodes\n";
    return false;
  }

  std::array<char, 4096> instantiate_log{};
  hipGraphNode_t error_node = nullptr;
  const hipError_t instantiate =
      hipGraphInstantiate(&resources.graph_exec, resources.graph, &error_node,
                          instantiate_log.data(), instantiate_log.size());
  report.instantiate = static_cast<int>(instantiate);
  if (!hip_success(instantiate, "hipGraphInstantiate") ||
      resources.graph_exec == nullptr) {
    std::cerr << "graph instantiate log: " << instantiate_log.data() << '\n';
    return false;
  }

  std::set<uint64_t> checkpoint_hashes;
  for (std::size_t replay = 0U; replay < kReplayCount; ++replay) {
    fill_activation(host_activation, replay);
    host_activation_scale = activation_scale_for_replay(replay);
    if (!hip_success(hipMemcpyAsync(device_activation, host_activation.data(),
                                    host_activation.size(),
                                    hipMemcpyHostToDevice, resources.stream),
                     "update activation in place") ||
        !hip_success(hipMemcpyAsync(device_activation_scale,
                                    &host_activation_scale, sizeof(float),
                                    hipMemcpyHostToDevice, resources.stream),
                     "update activation scale in place")) {
      return false;
    }
    const hipError_t graph_launch =
        hipGraphLaunch(resources.graph_exec, resources.stream);
    report.replay_launch = static_cast<int>(graph_launch);
    if (!hip_success(graph_launch, "hipGraphLaunch")) {
      return false;
    }

    if (is_checkpoint(replay)) {
      if (!hip_success(
              hipMemcpyAsync(host_graph_output.data(), device_output,
                             host_graph_output.size() * sizeof(uint16_t),
                             hipMemcpyDeviceToHost, resources.stream),
              "copy graph checkpoint output") ||
          !hip_success(hipStreamSynchronize(resources.stream),
                       "synchronize graph checkpoint")) {
        return false;
      }
      const hipblasStatus_t eager =
          launch_matmul(resources, algorithm, device_weight, device_activation,
                        device_output);
      if (!lt_success(eager, "eager checkpoint hipblasLtMatmul") ||
          !hip_success(
              hipMemcpyAsync(host_eager_output.data(), device_output,
                             host_eager_output.size() * sizeof(uint16_t),
                             hipMemcpyDeviceToHost, resources.stream),
              "copy eager checkpoint output") ||
          !hip_success(hipStreamSynchronize(resources.stream),
                       "synchronize eager checkpoint")) {
        return false;
      }
      for (std::size_t index = 0U; index < host_graph_output.size(); ++index) {
        if (host_graph_output[index] != host_eager_output[index]) {
          ++report.bf16_mismatch_count;
          if (report.bf16_mismatch_count == 1U) {
            std::cerr << "first BF16 mismatch: replay=" << replay
                      << " index=" << index << " graph=0x" << std::hex
                      << host_graph_output[index] << " eager=0x"
                      << host_eager_output[index] << std::dec << '\n';
          }
        }
        if (!bf16_is_finite(host_graph_output[index]) ||
            !bf16_is_finite(host_eager_output[index])) {
          ++report.nonfinite_count;
        }
      }
      checkpoint_hashes.insert(hash_bf16_bits(host_graph_output));
      ++report.checkpoint_count;
      if (report.bf16_mismatch_count != 0U || report.nonfinite_count != 0U) {
        return false;
      }
    } else {
      const hipError_t replay_sync = hipStreamSynchronize(resources.stream);
      report.replay_sync = static_cast<int>(replay_sync);
      if (!hip_success(replay_sync, "synchronize graph replay")) {
        return false;
      }
    }
    ++report.replay_count;
  }
  report.replay_sync = static_cast<int>(hipSuccess);
  report.distinct_checkpoint_outputs = checkpoint_hashes.size();
  if (report.checkpoint_count != kCheckpoints.size() ||
      report.distinct_checkpoint_outputs < 2U) {
    std::cerr << "checkpoint coverage/content-change check failed\n";
    return false;
  }
  return true;
}

bool cleanup(Resources &resources, std::size_t &freed_allocations) {
  bool ok = true;
  const auto check_hip_cleanup = [&](const hipError_t status,
                                     const char *const operation) {
    if (!hip_success(status, operation)) {
      ok = false;
    }
  };
  const auto check_lt_cleanup = [&](const hipblasStatus_t status,
                                    const char *const operation) {
    if (!lt_success(status, operation)) {
      ok = false;
    }
  };

  if (resources.graph_exec != nullptr) {
    check_hip_cleanup(hipGraphExecDestroy(resources.graph_exec),
                      "hipGraphExecDestroy");
    resources.graph_exec = nullptr;
  }
  if (resources.graph != nullptr) {
    check_hip_cleanup(hipGraphDestroy(resources.graph), "hipGraphDestroy");
    resources.graph = nullptr;
  }
  if (resources.stream != nullptr) {
    check_hip_cleanup(hipStreamDestroy(resources.stream), "hipStreamDestroy");
    resources.stream = nullptr;
  }
  if (resources.preference != nullptr) {
    check_lt_cleanup(hipblasLtMatmulPreferenceDestroy(resources.preference),
                     "hipblasLtMatmulPreferenceDestroy");
    resources.preference = nullptr;
  }
  if (resources.d != nullptr) {
    check_lt_cleanup(hipblasLtMatrixLayoutDestroy(resources.d),
                     "destroy D layout");
    resources.d = nullptr;
  }
  if (resources.c != nullptr) {
    check_lt_cleanup(hipblasLtMatrixLayoutDestroy(resources.c),
                     "destroy C layout");
    resources.c = nullptr;
  }
  if (resources.b != nullptr) {
    check_lt_cleanup(hipblasLtMatrixLayoutDestroy(resources.b),
                     "destroy B layout");
    resources.b = nullptr;
  }
  if (resources.a != nullptr) {
    check_lt_cleanup(hipblasLtMatrixLayoutDestroy(resources.a),
                     "destroy A layout");
    resources.a = nullptr;
  }
  if (resources.operation != nullptr) {
    check_lt_cleanup(hipblasLtMatmulDescDestroy(resources.operation),
                     "hipblasLtMatmulDescDestroy");
    resources.operation = nullptr;
  }
  if (resources.handle != nullptr) {
    check_lt_cleanup(hipblasLtDestroy(resources.handle), "hipblasLtDestroy");
    resources.handle = nullptr;
  }
  for (auto allocation = resources.allocations.rbegin();
       allocation != resources.allocations.rend(); ++allocation) {
    const hipError_t status = hipFree(*allocation);
    if (status == hipSuccess) {
      ++freed_allocations;
    } else {
      hip_success(status, "hipFree");
      ok = false;
    }
  }
  resources.allocations.clear();
  return ok;
}

void print_report(const StatusReport &report, const std::size_t allocations,
                  const std::size_t frees, const bool cleanup_ok,
                  const bool probe_ok) {
  std::cout << "plan m=" << kM << " k=" << kK << " n=" << kN
            << " input=OCP_E4M3FN scales=F32_outer output=BF16"
            << " compute=F32 workspace_limit=0 transA=T transB=N\n";
  std::cout << "algorithm requested=" << kRequestedAlgorithms
            << " available=" << report.heuristic_count
            << " rank=" << kAlgorithmRank
            << " query_status=" << report.heuristic_query
            << " selected_state=" << report.selected_state
            << " workspace=" << report.selected_workspace << '\n';
  std::cout << "capture begin=" << hip_status_name(report.capture_begin)
            << " matmul_status=" << report.capture_matmul
            << " end=" << hip_status_name(report.capture_end)
            << " node_query=" << hip_status_name(report.node_query)
            << " graph_nodes=" << report.node_count
            << " instantiate=" << hip_status_name(report.instantiate) << '\n';
  std::cout << "replay successful=" << report.replay_count << '/'
            << kReplayCount
            << " launch_status=" << hip_status_name(report.replay_launch)
            << " sync_status=" << hip_status_name(report.replay_sync)
            << " checkpoints=" << report.checkpoint_count
            << " bf16_bit_mismatches=" << report.bf16_mismatch_count
            << " nonfinite=" << report.nonfinite_count
            << " distinct_outputs=" << report.distinct_checkpoint_outputs
            << '\n';
  std::cout << "allocation_cleanup allocations=" << allocations
            << " frees=" << frees << " status="
            << (cleanup_ok && allocations == frees ? "PASS" : "FAIL") << '\n';
  std::cout << "HIPBLASLT_FP8_GRAPH_CAPTURE_RESULT="
            << (probe_ok && cleanup_ok && allocations == frees ? "PASS"
                                                               : "FAIL")
            << '\n';
}

} // namespace

int main() {
  Resources resources;
  StatusReport report;
  const bool probe_ok = run_probe(resources, report);
  const std::size_t allocations = resources.allocations.size();
  std::size_t freed_allocations = 0U;
  const bool cleanup_ok = cleanup(resources, freed_allocations);
  print_report(report, allocations, freed_allocations, cleanup_ok, probe_ok);
  return probe_ok && cleanup_ok && allocations == freed_allocations ? 0 : 1;
}
