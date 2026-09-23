// Phase 87 stage 7 C1 probe: FP8 producer-side activation quantization.
//
// The control is the decomposed production sequence
//
//   SiLU-multiply -> BF16 -> FP8 row quantization
//
// and the candidate is the production fused producer launcher.  Every timing
// sample runs both variants in one process, alternating control/candidate and
// candidate/control order.  This keeps the comparison independent of process
// startup and records both the byte-exact contract and the measured kernel
// chain cost.
//
// Suggested compile-only command (one exact target per object):
//   amdclang++ -D__HIP_ROCclr__=1 -O3 -ffp-contract=off -DNDEBUG \
//     -std=gnu++17 --offload-arch=gfx1030 -mcode-object-version=6 \
//     -mno-wavefrontsize64 -x hip -I native/lowp/include -I include \
//     -I native/hip/src -c \
//     native/hip/tests/phase87_s7_c1_perf_probe.hip.cpp \
//     -o phase87-s7-c1-perf-gfx1030.o
//
// A runnable binary additionally needs the lowp runtime archive and amdhip64:
//   amdclang++ -O3 -ffp-contract=off -DNDEBUG --offload-arch=gfx1030 \
//     -mcode-object-version=6 -mno-wavefrontsize64 --hip-link \
//     phase87-s7-c1-perf-gfx1030.o /path/to/libsllm_lowp.a \
//     -L"${ROCM_PATH}/lib" -lamdhip64 -o phase87-s7-c1-perf-gfx1030
//
// Usage:
//   phase87-s7-c1-perf-gfx1030 --target=gfx1030 \
//     [--warmup-ms=200] [--samples=7] [--reps=100] [--graph=true]

#include "../src/elementwise_kernel.hip.cpp"

// Keep the RMSNorm production TU in a nested namespace: both production TUs
// have small anonymous-namespace helpers with the same names, while this
// standalone probe needs both producer families in one translation unit.
namespace s7_c1_rmsnorm_impl {
#include "../src/rmsnorm_kernel.hip.cpp"
} // namespace s7_c1_rmsnorm_impl

#include <hip/hip_runtime.h>
#include <lowp/detail/lowp_kernel_internal.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#define S7_C1_CHECK(call)                                                      \
  do {                                                                         \
    const hipError_t s7_c1_status_ = (call);                                   \
    if (s7_c1_status_ != hipSuccess) {                                         \
      std::cerr << "FAIL: " << #call << " -> "                                 \
                << hipGetErrorString(s7_c1_status_) << '\n';                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

struct DeviceBuffer final {
  void *pointer = nullptr;

  DeviceBuffer() = default;
  explicit DeviceBuffer(const std::size_t bytes) { allocate(bytes); }
  void allocate(const std::size_t bytes) {
    S7_C1_CHECK(hipMalloc(&pointer, bytes == 0U ? 1U : bytes));
  }
  ~DeviceBuffer() {
    if (pointer != nullptr) {
      (void)hipFree(pointer);
    }
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

struct Shape final {
  uint32_t m;
  uint32_t k;
  const char *name;
};

// Qwen3.8 projection rows and nearby shape choices exercise the producer
// launcher for the decode row and the small prefill/decode boundaries.
constexpr Shape kShapes[] = {
    {1U, 5120U, "m1k5120"}, {3U, 5120U, "m3k5120"},   {17U, 5120U, "m17k5120"},
    {1U, 6144U, "m1k6144"}, {1U, 17408U, "m1k17408"},
};

std::string arg_value(const int argc, char **const argv, const std::string &key,
                      const std::string &fallback) {
  const std::string prefix = key + "=";
  for (int index = 1; index < argc; ++index) {
    const std::string current(argv[index]);
    if (current.rfind(prefix, 0) == 0) {
      return current.substr(prefix.size());
    }
  }
  return fallback;
}

void need(const bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(2);
  }
}

void fill_bf16(std::vector<uint16_t> *const values, const uint32_t seed) {
  // Finite BF16 values in [-2, 2].  The values are intentionally not all
  // powers of two so FP8 amax and rounding both participate in the probe.
  static constexpr uint16_t kPattern[] = {
      UINT16_C(0x3f80), UINT16_C(0xbf80), UINT16_C(0x3f00), UINT16_C(0xbf00),
      UINT16_C(0x3e80), UINT16_C(0xbe80), UINT16_C(0x4000), UINT16_C(0xc000),
      UINT16_C(0x3d00), UINT16_C(0xbd00), UINT16_C(0x0000), UINT16_C(0x8000),
  };
  uint32_t state = seed == 0U ? 1U : seed;
  for (uint16_t &value : *values) {
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    value = kPattern[(state >> 16U) % (sizeof(kPattern) / sizeof(kPattern[0]))];
  }
}

void upload(void *const destination, const void *const source,
            const std::size_t bytes) {
  S7_C1_CHECK(hipMemcpy(destination, source, bytes, hipMemcpyHostToDevice));
}

void launch_decomposed(const uint16_t *const gate, const uint16_t *const up,
                       uint16_t *const bf16, uint8_t *const codes,
                       float *const scales, const Shape &shape,
                       const hipStream_t stream) {
  S7_C1_CHECK(sllm_elementwise_kernel::launch_silu_mul(
      gate, up, bf16, static_cast<uint64_t>(shape.m) * shape.k, stream));
  S7_C1_CHECK(sllm_matmul_kernel::launch_fp8_quantize(
      bf16, codes, scales, shape.m, shape.k, false, stream));
}

void launch_fused(const uint16_t *const gate, const uint16_t *const up,
                  uint8_t *const codes, float *const scales, const Shape &shape,
                  const hipStream_t stream) {
  S7_C1_CHECK(sllm_elementwise_kernel::launch_silu_mul_prequant_fp8(
      gate, up, codes, scales, shape.m, shape.k, stream));
}

enum class FamilyKind { kRmsNorm, kSilu, kSigmoid };

struct Family final {
  const char *name;
  uint32_t node_count;
  Shape shape;
  FamilyKind kind;
};

// These are the 88 C1 graph nodes selected for Stage 7.  The two RMSNorm
// rows use the m1k5120 representative because both producer families have
// the same per-row FP8 contract at this stage; SiLU and Sigmoid use their
// actual projection widths.
constexpr Family kFamilies[] = {
    {"full_attention_input_rmsnorm_qkv",
     48U,
     {1U, 5120U, "m1k5120"},
     FamilyKind::kRmsNorm},
    {"fp8_mlp_post_attention_rmsnorm",
     16U,
     {1U, 5120U, "m1k5120"},
     FamilyKind::kRmsNorm},
    {"fp8_mlp_silu_mul", 8U, {1U, 17408U, "m1k17408"}, FamilyKind::kSilu},
    {"full_attention_sigmoid_mul",
     16U,
     {1U, 6144U, "m1k6144"},
     FamilyKind::kSigmoid},
};

struct FamilyBuffers final {
  DeviceBuffer input0;
  DeviceBuffer input1;
  DeviceBuffer raw_scale;
  DeviceBuffer bf16;
  DeviceBuffer residual_output;
  DeviceBuffer control_codes;
  DeviceBuffer fused_codes;
  DeviceBuffer control_scales;
  DeviceBuffer fused_scales;

  void initialize(const Shape &shape) {
    const std::size_t elements = static_cast<std::size_t>(shape.m) * shape.k;
    input0.allocate(elements * sizeof(uint16_t));
    input1.allocate(elements * sizeof(uint16_t));
    raw_scale.allocate(static_cast<std::size_t>(shape.k) * sizeof(uint16_t));
    bf16.allocate(elements * sizeof(uint16_t));
    residual_output.allocate(elements * sizeof(uint16_t));
    control_codes.allocate(elements);
    fused_codes.allocate(elements);
    control_scales.allocate(static_cast<std::size_t>(shape.m) * sizeof(float));
    fused_scales.allocate(static_cast<std::size_t>(shape.m) * sizeof(float));
  }
};

double median(std::vector<double> values);
double time_variant(int reps, hipStream_t stream,
                    const std::function<void()> &launch);

void launch_family_control(const Family &family, const FamilyBuffers &buffers,
                           const hipStream_t stream) {
  const Shape &shape = family.shape;
  const auto *input0 = static_cast<const uint16_t *>(buffers.input0.pointer);
  const auto *input1 = static_cast<const uint16_t *>(buffers.input1.pointer);
  auto *const bf16 = static_cast<uint16_t *>(buffers.bf16.pointer);
  auto *const control_codes =
      static_cast<uint8_t *>(buffers.control_codes.pointer);
  auto *const control_scales =
      static_cast<float *>(buffers.control_scales.pointer);
  if (family.kind == FamilyKind::kRmsNorm) {
    S7_C1_CHECK(s7_c1_rmsnorm_impl::sllm_rmsnorm_kernel::launch_residual_fused(
        input0, input1,
        static_cast<const uint16_t *>(buffers.raw_scale.pointer),
        static_cast<uint16_t *>(buffers.residual_output.pointer), bf16, shape.k,
        shape.m, 1.0e-5F, SLLM_RMSNORM_SCALE_MODE_DIRECT, stream));
  } else if (family.kind == FamilyKind::kSilu) {
    S7_C1_CHECK(sllm_elementwise_kernel::launch_silu_mul(
        input0, input1, bf16, static_cast<uint64_t>(shape.m) * shape.k,
        stream));
  } else {
    S7_C1_CHECK(sllm_elementwise_kernel::launch_sigmoid_mul(
        input0, input1, bf16, static_cast<uint64_t>(shape.m) * shape.k,
        stream));
  }
  S7_C1_CHECK(sllm_matmul_kernel::launch_fp8_quantize(
      bf16, control_codes, control_scales, shape.m, shape.k, false, stream));
}

void launch_family_fused(const Family &family, const FamilyBuffers &buffers,
                         const hipStream_t stream) {
  const Shape &shape = family.shape;
  const auto *input0 = static_cast<const uint16_t *>(buffers.input0.pointer);
  const auto *input1 = static_cast<const uint16_t *>(buffers.input1.pointer);
  auto *const fused_codes = static_cast<uint8_t *>(buffers.fused_codes.pointer);
  auto *const fused_scales = static_cast<float *>(buffers.fused_scales.pointer);
  if (family.kind == FamilyKind::kRmsNorm) {
    S7_C1_CHECK(
        s7_c1_rmsnorm_impl::sllm_rmsnorm_kernel::launch_residual_prequant_fp8(
            input0, input1,
            static_cast<const uint16_t *>(buffers.raw_scale.pointer),
            static_cast<uint16_t *>(buffers.residual_output.pointer),
            fused_codes, fused_scales, shape.k, shape.m, 1.0e-5F,
            SLLM_RMSNORM_SCALE_MODE_DIRECT, 0U, stream));
  } else if (family.kind == FamilyKind::kSilu) {
    S7_C1_CHECK(sllm_elementwise_kernel::launch_silu_mul_prequant_fp8(
        input0, input1, fused_codes, fused_scales, shape.m, shape.k, stream));
  } else {
    S7_C1_CHECK(sllm_elementwise_kernel::launch_sigmoid_mul_prequant_fp8(
        input0, input1, fused_codes, fused_scales, shape.m, shape.k, stream));
  }
}

void run_c1_replay(const hipStream_t stream, const int warmup_ms,
                   const int samples, const int reps, const bool graph,
                   std::size_t *const failures) {
  constexpr std::size_t family_count = sizeof(kFamilies) / sizeof(kFamilies[0]);
  FamilyBuffers buffers[family_count];
  for (std::size_t index = 0; index < family_count; ++index) {
    const Family &family = kFamilies[index];
    buffers[index].initialize(family.shape);
    const std::size_t elements =
        static_cast<std::size_t>(family.shape.m) * family.shape.k;
    std::vector<uint16_t> host(elements);
    fill_bf16(&host, UINT32_C(0x7200) + static_cast<uint32_t>(index));
    upload(buffers[index].input0.pointer, host.data(),
           host.size() * sizeof(uint16_t));
    fill_bf16(&host, UINT32_C(0x9100) + static_cast<uint32_t>(index));
    upload(buffers[index].input1.pointer, host.data(),
           host.size() * sizeof(uint16_t));
    std::vector<uint16_t> raw_scale(family.shape.k, UINT16_C(0x3f80));
    upload(buffers[index].raw_scale.pointer, raw_scale.data(),
           raw_scale.size() * sizeof(uint16_t));
  }

  for (std::size_t index = 0; index < family_count; ++index) {
    const Family &family = kFamilies[index];
    FamilyBuffers &family_buffers = buffers[index];
    launch_family_control(family, family_buffers, stream);
    launch_family_fused(family, family_buffers, stream);
  }
  S7_C1_CHECK(hipStreamSynchronize(stream));

  for (std::size_t index = 0; index < family_count; ++index) {
    const Family &family = kFamilies[index];
    FamilyBuffers &family_buffers = buffers[index];
    const std::size_t elements =
        static_cast<std::size_t>(family.shape.m) * family.shape.k;
    std::vector<uint8_t> control_codes(elements);
    std::vector<uint8_t> fused_codes(elements);
    std::vector<float> control_scales(family.shape.m);
    std::vector<float> fused_scales(family.shape.m);
    S7_C1_CHECK(hipMemcpy(control_codes.data(),
                          family_buffers.control_codes.pointer, elements,
                          hipMemcpyDeviceToHost));
    S7_C1_CHECK(hipMemcpy(fused_codes.data(),
                          family_buffers.fused_codes.pointer, elements,
                          hipMemcpyDeviceToHost));
    S7_C1_CHECK(hipMemcpy(
        control_scales.data(), family_buffers.control_scales.pointer,
        control_scales.size() * sizeof(float), hipMemcpyDeviceToHost));
    S7_C1_CHECK(
        hipMemcpy(fused_scales.data(), family_buffers.fused_scales.pointer,
                  fused_scales.size() * sizeof(float), hipMemcpyDeviceToHost));
    std::size_t code_mismatches = 0U;
    for (std::size_t element = 0; element < elements; ++element) {
      code_mismatches +=
          control_codes[element] != fused_codes[element] ? 1U : 0U;
    }
    std::size_t scale_mismatches = 0U;
    for (std::size_t row = 0; row < family.shape.m; ++row) {
      scale_mismatches += std::memcmp(&control_scales[row], &fused_scales[row],
                                      sizeof(float)) != 0
                              ? 1U
                              : 0U;
    }
    const bool bitwise = code_mismatches == 0U && scale_mismatches == 0U;
    if (!bitwise) {
      ++*failures;
    }
    std::cout << "{\"kind\":\"aggregate_correctness\",\"family\":\""
              << family.name << "\",\"node_count\":" << family.node_count
              << ",\"bitwise\":" << (bitwise ? "true" : "false")
              << ",\"code_mismatches\":" << code_mismatches
              << ",\"scale_mismatches\":" << scale_mismatches << "}\n";
  }

  const auto control = [&]() {
    for (std::size_t index = 0; index < family_count; ++index) {
      for (uint32_t count = 0; count < kFamilies[index].node_count; ++count) {
        launch_family_control(kFamilies[index], buffers[index], stream);
      }
    }
  };
  const auto fused = [&]() {
    for (std::size_t index = 0; index < family_count; ++index) {
      for (uint32_t count = 0; count < kFamilies[index].node_count; ++count) {
        launch_family_fused(kFamilies[index], buffers[index], stream);
      }
    }
  };

  const auto warmup_start = std::chrono::steady_clock::now();
  while (std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - warmup_start)
             .count() < warmup_ms) {
    control();
    fused();
  }
  S7_C1_CHECK(hipStreamSynchronize(stream));

  std::vector<double> control_samples;
  std::vector<double> fused_samples;
  control_samples.reserve(static_cast<std::size_t>(samples));
  fused_samples.reserve(static_cast<std::size_t>(samples));
  for (int sample = 0; sample < samples; ++sample) {
    const bool control_first = (sample % 2) == 0;
    for (int order = 0; order < 2; ++order) {
      const bool run_control = (order == 0) == control_first;
      const double elapsed = run_control ? time_variant(reps, stream, control)
                                         : time_variant(reps, stream, fused);
      if (run_control) {
        control_samples.push_back(elapsed);
      } else {
        fused_samples.push_back(elapsed);
      }
    }
  }
  const double control_ms = median(control_samples);
  const double fused_ms = median(fused_samples);
  constexpr uint32_t total_nodes = 88U;
  std::cout
      << "{\"kind\":\"aggregate_timing\",\"scope\":\"c1_88_nodes\","
      << "\"node_count\":" << total_nodes
      << ",\"decomposed_launches_per_replay\":" << total_nodes * 2U
      << ",\"fused_launches_per_replay\":" << total_nodes
      << ",\"control_ms\":" << control_ms << ",\"fused_ms\":" << fused_ms
      << ",\"delta_ms\":" << (control_ms - fused_ms)
      << ",\"speedup\":" << (control_ms / std::max(fused_ms, 1.0e-12))
      << ",\"control_ms_per_node\":" << (control_ms / total_nodes)
      << ",\"fused_ms_per_node\":" << (fused_ms / total_nodes)
      << ",\"samples_per_variant\":" << samples
      << ",\"reps_per_sample\":" << reps
      << ",\"same_process_ab_ba\":true,\"graph_capture\":false,\"families\":[";
  for (std::size_t index = 0; index < family_count; ++index) {
    if (index != 0U) {
      std::cout << ',';
    }
    std::cout << "{\"name\":\"" << kFamilies[index].name
              << "\",\"node_count\":" << kFamilies[index].node_count
              << ",\"shape\":\"" << kFamilies[index].shape.name << "\"}";
  }
  std::cout << "],\"paired_rounds\":[";
  double minimum_delta = control_samples[0] - fused_samples[0];
  bool all_fused_faster = true;
  for (std::size_t index = 0; index < control_samples.size(); ++index) {
    const double delta = control_samples[index] - fused_samples[index];
    minimum_delta = std::min(minimum_delta, delta);
    all_fused_faster = all_fused_faster && delta > 0.0;
    if (index != 0U) {
      std::cout << ',';
    }
    std::cout << "{\"round\":" << index << ",\"order\":\""
              << ((index % 2U) == 0U ? "AB" : "BA")
              << "\",\"control_ms\":" << control_samples[index]
              << ",\"fused_ms\":" << fused_samples[index]
              << ",\"delta_ms\":" << delta << "}";
  }
  std::cout << "],\"minimum_delta_ms\":" << minimum_delta
            << ",\"all_fused_faster\":" << (all_fused_faster ? "true" : "false")
            << "}\n";

  if (!graph) {
    return;
  }

  hipGraph_t control_graph = nullptr;
  hipGraph_t fused_graph = nullptr;
  hipGraphExec_t control_exec = nullptr;
  hipGraphExec_t fused_exec = nullptr;
  S7_C1_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
  control();
  S7_C1_CHECK(hipStreamEndCapture(stream, &control_graph));
  S7_C1_CHECK(
      hipGraphInstantiate(&control_exec, control_graph, nullptr, nullptr, 0));
  S7_C1_CHECK(hipGraphDestroy(control_graph));
  S7_C1_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
  fused();
  S7_C1_CHECK(hipStreamEndCapture(stream, &fused_graph));
  S7_C1_CHECK(
      hipGraphInstantiate(&fused_exec, fused_graph, nullptr, nullptr, 0));
  S7_C1_CHECK(hipGraphDestroy(fused_graph));
  S7_C1_CHECK(hipStreamSynchronize(stream));
  const auto control_graph_launch = [&]() {
    S7_C1_CHECK(hipGraphLaunch(control_exec, stream));
  };
  const auto fused_graph_launch = [&]() {
    S7_C1_CHECK(hipGraphLaunch(fused_exec, stream));
  };
  control_samples.clear();
  fused_samples.clear();
  for (int sample = 0; sample < samples; ++sample) {
    const bool control_first = (sample % 2) == 0;
    for (int order = 0; order < 2; ++order) {
      const bool run_control = (order == 0) == control_first;
      const double elapsed =
          run_control ? time_variant(reps, stream, control_graph_launch)
                      : time_variant(reps, stream, fused_graph_launch);
      if (run_control) {
        control_samples.push_back(elapsed);
      } else {
        fused_samples.push_back(elapsed);
      }
    }
  }
  const double control_graph_ms = median(control_samples);
  const double fused_graph_ms = median(fused_samples);
  std::cout << "{\"kind\":\"aggregate_graph_timing\",\"scope\":\"c1_88_nodes\","
            << "\"node_count\":" << total_nodes
            << ",\"control_ms\":" << control_graph_ms
            << ",\"fused_ms\":" << fused_graph_ms
            << ",\"delta_ms\":" << (control_graph_ms - fused_graph_ms)
            << ",\"speedup\":"
            << (control_graph_ms / std::max(fused_graph_ms, 1.0e-12))
            << ",\"samples_per_variant\":" << samples
            << ",\"reps_per_sample\":" << reps
            << ",\"same_process_ab_ba\":true,\"graph_capture\":true,"
            << "\"paired_rounds\":[";
  minimum_delta = control_samples[0] - fused_samples[0];
  all_fused_faster = true;
  for (std::size_t index = 0; index < control_samples.size(); ++index) {
    const double delta = control_samples[index] - fused_samples[index];
    minimum_delta = std::min(minimum_delta, delta);
    all_fused_faster = all_fused_faster && delta > 0.0;
    if (index != 0U) {
      std::cout << ',';
    }
    std::cout << "{\"round\":" << index << ",\"order\":\""
              << ((index % 2U) == 0U ? "AB" : "BA")
              << "\",\"control_ms\":" << control_samples[index]
              << ",\"fused_ms\":" << fused_samples[index]
              << ",\"delta_ms\":" << delta << "}";
  }
  std::cout << "],\"minimum_delta_ms\":" << minimum_delta
            << ",\"all_fused_faster\":" << (all_fused_faster ? "true" : "false")
            << "}\n";
  S7_C1_CHECK(hipGraphExecDestroy(control_exec));
  S7_C1_CHECK(hipGraphExecDestroy(fused_exec));
}

double median(std::vector<double> values) {
  const std::size_t middle = values.size() / 2U;
  std::nth_element(values.begin(),
                   values.begin() + static_cast<std::ptrdiff_t>(middle),
                   values.end());
  return values[middle];
}

double time_variant(const int reps, const hipStream_t stream,
                    const std::function<void()> &launch) {
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  S7_C1_CHECK(hipEventCreate(&start));
  S7_C1_CHECK(hipEventCreate(&stop));
  S7_C1_CHECK(hipEventRecord(start, stream));
  for (int repeat = 0; repeat < reps; ++repeat) {
    launch();
  }
  S7_C1_CHECK(hipEventRecord(stop, stream));
  S7_C1_CHECK(hipEventSynchronize(stop));
  float elapsed_ms = 0.0F;
  S7_C1_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
  S7_C1_CHECK(hipEventDestroy(start));
  S7_C1_CHECK(hipEventDestroy(stop));
  return static_cast<double>(elapsed_ms) / reps;
}

} // namespace

int main(const int argc, char **const argv) {
  const std::string target = arg_value(argc, argv, "--target", "");
  const int warmup_ms =
      std::atoi(arg_value(argc, argv, "--warmup-ms", "200").c_str());
  const int samples =
      std::atoi(arg_value(argc, argv, "--samples", "7").c_str());
  const int reps = std::atoi(arg_value(argc, argv, "--reps", "100").c_str());
  const std::string graph_arg = arg_value(argc, argv, "--graph", "false");
  const bool graph = graph_arg == "1" || graph_arg == "true";
  need(target == "gfx1030" || target == "gfx1201",
       "--target=gfx1030|gfx1201 is required");
  need(warmup_ms > 0 && samples >= 3 && reps > 0,
       "warmup-ms>0, samples>=3 and reps>0 are required");

  int device = 0;
  S7_C1_CHECK(hipGetDevice(&device));
  hipDeviceProp_t properties{};
  S7_C1_CHECK(hipGetDeviceProperties(&properties, device));
  const std::string arch(properties.gcnArchName);
  need(arch.find(target) != std::string::npos,
       "runtime target " + arch + " does not match --target=" + target);

  std::cout << "{\"kind\":\"meta\",\"target\":\"" << target
            << "\",\"device\":\"" << properties.name << "\",\"gcn_arch\":\""
            << arch << "\",\"warmup_ms\":" << warmup_ms
            << ",\"samples\":" << samples << ",\"reps\":" << reps
            << ",\"protocol\":\"same_process_ab_ba\",\"format\":\"fp8_e4m3fn\""
            << ",\"aggregate_graph_requested\":" << (graph ? "true" : "false")
            << "}\n";

  hipStream_t stream = nullptr;
  S7_C1_CHECK(hipStreamCreate(&stream));
  std::size_t failures = 0U;

  for (const Shape &shape : kShapes) {
    const std::size_t elements = static_cast<std::size_t>(shape.m) * shape.k;
    DeviceBuffer gate(elements * sizeof(uint16_t));
    DeviceBuffer up(elements * sizeof(uint16_t));
    DeviceBuffer bf16(elements * sizeof(uint16_t));
    DeviceBuffer control_codes(elements);
    DeviceBuffer fused_codes(elements);
    DeviceBuffer control_scales(static_cast<std::size_t>(shape.m) *
                                sizeof(float));
    DeviceBuffer fused_scales(static_cast<std::size_t>(shape.m) *
                              sizeof(float));

    std::vector<uint16_t> host(elements);
    fill_bf16(&host, UINT32_C(0x1234) + shape.k + shape.m);
    upload(gate.pointer, host.data(), host.size() * sizeof(uint16_t));
    fill_bf16(&host, UINT32_C(0x5678) + shape.k + shape.m);
    upload(up.pointer, host.data(), host.size() * sizeof(uint16_t));

    launch_decomposed(static_cast<const uint16_t *>(gate.pointer),
                      static_cast<const uint16_t *>(up.pointer),
                      static_cast<uint16_t *>(bf16.pointer),
                      static_cast<uint8_t *>(control_codes.pointer),
                      static_cast<float *>(control_scales.pointer), shape,
                      stream);
    launch_fused(static_cast<const uint16_t *>(gate.pointer),
                 static_cast<const uint16_t *>(up.pointer),
                 static_cast<uint8_t *>(fused_codes.pointer),
                 static_cast<float *>(fused_scales.pointer), shape, stream);
    S7_C1_CHECK(hipStreamSynchronize(stream));

    std::vector<uint8_t> control_host(elements);
    std::vector<uint8_t> fused_host(elements);
    std::vector<float> control_scale_host(shape.m);
    std::vector<float> fused_scale_host(shape.m);
    S7_C1_CHECK(hipMemcpy(control_host.data(), control_codes.pointer,
                          control_host.size(), hipMemcpyDeviceToHost));
    S7_C1_CHECK(hipMemcpy(fused_host.data(), fused_codes.pointer,
                          fused_host.size(), hipMemcpyDeviceToHost));
    S7_C1_CHECK(hipMemcpy(control_scale_host.data(), control_scales.pointer,
                          control_scale_host.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
    S7_C1_CHECK(hipMemcpy(fused_scale_host.data(), fused_scales.pointer,
                          fused_scale_host.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
    std::size_t code_mismatches = 0U;
    for (std::size_t index = 0; index < elements; ++index) {
      code_mismatches += control_host[index] != fused_host[index] ? 1U : 0U;
    }
    std::size_t scale_mismatches = 0U;
    for (std::size_t index = 0; index < shape.m; ++index) {
      scale_mismatches +=
          std::memcmp(&control_scale_host[index], &fused_scale_host[index],
                      sizeof(float)) != 0
              ? 1U
              : 0U;
    }
    const bool bitwise = code_mismatches == 0U && scale_mismatches == 0U;
    if (!bitwise) {
      ++failures;
    }
    std::cout << "{\"kind\":\"correctness\",\"shape\":\"" << shape.name
              << "\",\"bitwise\":" << (bitwise ? "true" : "false")
              << ",\"code_mismatches\":" << code_mismatches
              << ",\"scale_mismatches\":" << scale_mismatches << "}\n";

    const auto control = [&]() {
      launch_decomposed(static_cast<const uint16_t *>(gate.pointer),
                        static_cast<const uint16_t *>(up.pointer),
                        static_cast<uint16_t *>(bf16.pointer),
                        static_cast<uint8_t *>(control_codes.pointer),
                        static_cast<float *>(control_scales.pointer), shape,
                        stream);
    };
    const auto fused = [&]() {
      launch_fused(static_cast<const uint16_t *>(gate.pointer),
                   static_cast<const uint16_t *>(up.pointer),
                   static_cast<uint8_t *>(fused_codes.pointer),
                   static_cast<float *>(fused_scales.pointer), shape, stream);
    };

    const auto warmup_start = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - warmup_start)
               .count() < warmup_ms) {
      control();
      fused();
    }
    S7_C1_CHECK(hipStreamSynchronize(stream));

    std::vector<double> control_samples;
    std::vector<double> fused_samples;
    control_samples.reserve(static_cast<std::size_t>(samples));
    fused_samples.reserve(static_cast<std::size_t>(samples));
    for (int sample = 0; sample < samples; ++sample) {
      const bool control_first = (sample % 2) == 0;
      for (int order = 0; order < 2; ++order) {
        const bool run_control = (order == 0) == control_first;
        const double elapsed = run_control ? time_variant(reps, stream, control)
                                           : time_variant(reps, stream, fused);
        if (run_control) {
          control_samples.push_back(elapsed);
        } else {
          fused_samples.push_back(elapsed);
        }
      }
    }
    const double control_ms = median(control_samples);
    const double fused_ms = median(fused_samples);
    std::cout << "{\"kind\":\"timing\",\"shape\":\"" << shape.name
              << "\",\"control_ms\":" << control_ms
              << ",\"fused_ms\":" << fused_ms
              << ",\"delta_ms\":" << (control_ms - fused_ms)
              << ",\"speedup\":" << (control_ms / std::max(fused_ms, 1.0e-12))
              << ",\"samples_per_variant\":" << samples
              << ",\"reps_per_sample\":" << reps << "}\n";
  }

  run_c1_replay(stream, warmup_ms, samples, reps, graph, &failures);

  S7_C1_CHECK(hipStreamDestroy(stream));
  std::cout << "{\"kind\":\"cleanup\",\"failures\":" << failures << "}\n";
  return failures == 0U ? 0 : 1;
}
