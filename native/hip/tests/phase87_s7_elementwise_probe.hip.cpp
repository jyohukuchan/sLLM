// Phase 87 stage 7 probe: elementwise producer fusion for FP8 / NVFP4
// activation quantization.
//
// The control is the decomposed chain (the production SiLU-multiply or
// Sigmoid-multiply kernel followed by the production activation quantizer).
// The candidate is the fused producer. The probe fails closed on any byte
// difference between the two, then times both with an AB/BA protocol.
//
// Suggested build (one exact target per binary):
//   amdclang++ -D__HIP_ROCclr__=1 -O3 -ffp-contract=off -DNDEBUG \
//     -std=gnu++17 --offload-arch=gfx1030 -mcode-object-version=6 \
//     -mno-wavefrontsize64 -x hip -I native/lowp/include -I include \
//     -I native/hip/src \
//     -c native/hip/tests/phase87_s7_elementwise_probe.hip.cpp \
//     -o phase87-s7-elementwise-gfx1030.o
//   amdclang++ -O3 -ffp-contract=off -DNDEBUG --offload-arch=gfx1030 \
//     -mcode-object-version=6 -mno-wavefrontsize64 --hip-link \
//     phase87-s7-elementwise-gfx1030.o /path/to/libsllm_lowp.a \
//     -L"${ROCM_PATH}/lib" -lamdhip64 -o phase87-s7-elementwise-gfx1030
//
// Usage: phase87-s7-elementwise-gfx1030 --target=gfx1030 \
//   [--warmup-ms=300] [--samples=9] [--reps=100] [--shape=<name>]
//
// Scope notes:
//   * FNUZ FP8 is out of scope. The production launcher routes fnuz to the v1
//     quantizer while the fused candidate mirrors v2. Phase 87 covers exact
//     gfx1030 / gfx1201 with OCP E4M3FN only.
//   * Stage 7 has no fused Sigmoid-multiply NVFP4 candidate, so sigmoid runs
//     FP8 correctness and timing only.
//   * The fused candidates write no BF16 activation (that buffer is what the
//     removed quantizer launch consumed), so byte identity is checked on the
//     FP8 codes and row scales and on the NVFP4 packed codes and block
//     scales.

// Include the production TU so the control chain and the fused candidates
// share one translation unit, one rounding helper, and one compiler flag set.
#include "../src/elementwise_kernel.hip.cpp"

#include <hip/hip_runtime.h>
#include <lowp/detail/lowp_kernel_internal.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#define S7_HIP_CHECK(call)                                                     \
  do {                                                                         \
    const hipError_t s7_status_ = (call);                                      \
    if (s7_status_ != hipSuccess) {                                            \
      std::cerr << "FAIL: " << #call << " -> "                                 \
                << hipGetErrorString(s7_status_) << '\n';                      \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

int g_live_buffers = 0;

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "FAIL: " << message << '\n';
  std::exit(1);
}

void need(const bool condition, const std::string &message) {
  if (!condition) {
    fail(message);
  }
}

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

// Deterministic BF16 pattern. "normal" spans a moderate exponent range so
// expf stays finite on both sides; "specials" also injects zeros, signed
// zeros, infinities, NaNs and subnormals; "zeros" exercises the amax==0
// branch of both quantizers.
void fill_bf16(uint16_t *const data, const size_t elements,
               const std::string &pattern, const uint32_t seed,
               const uint32_t exponent_base, const uint32_t exponent_span) {
  if (pattern == "zeros") {
    std::memset(data, 0, elements * sizeof(uint16_t));
    return;
  }
  uint32_t state = seed == 0U ? 1U : seed;
  if (pattern == "specials") {
    static const uint16_t table[] = {
        UINT16_C(0x0000), UINT16_C(0x8000), UINT16_C(0x7f80), UINT16_C(0xff80),
        UINT16_C(0x7fc0), UINT16_C(0xffc0), UINT16_C(0x0001), UINT16_C(0x8001),
        UINT16_C(0x7f7f), UINT16_C(0xff7f), UINT16_C(0x3f80), UINT16_C(0xbf80),
        UINT16_C(0x0080), UINT16_C(0x8080), UINT16_C(0x3f00), UINT16_C(0xbf00)};
    constexpr uint32_t table_size =
        static_cast<uint32_t>(sizeof(table) / sizeof(table[0]));
    for (size_t index = 0; index < elements; ++index) {
      state = state * UINT32_C(1664525) + UINT32_C(1013904223);
      data[index] = table[(state >> 16) % table_size];
    }
    return;
  }
  for (size_t index = 0; index < elements; ++index) {
    state = state * UINT32_C(1664525) + UINT32_C(101390423);
    const uint32_t sign = (state & 1U) != 0U ? UINT32_C(0x8000) : 0U;
    const uint32_t exponent =
        exponent_base + ((state >> 1) & (exponent_span - 1U));
    const uint32_t mantissa = (state >> 9) & UINT32_C(0x7f);
    data[index] = static_cast<uint16_t>(sign | (exponent << 7) | mantissa);
  }
}

enum class Producer { kSilu, kSigmoid };

const char *producer_name(const Producer producer) {
  return producer == Producer::kSilu ? "silu" : "sigmoid";
}

constexpr Producer kProducers[] = {Producer::kSilu, Producer::kSigmoid};

struct Shape {
  uint32_t m;
  uint64_t k;
  const char *name;
  Producer producer; // the producer this shape times; correctness runs both
};

// Production first, then the non-aligned boundaries around each quantizer's
// grid split: 16 (NVFP4 block), 32 (wave) and the 5120/17408/6144 row
// lengths. m1h24d256 is the sigmoid row: 24 heads * head_dim 256 = 6144,
// contiguous.
constexpr Shape kShapes[] = {
    {1U, UINT64_C(5120), "m1k5120", Producer::kSilu},
    {3U, UINT64_C(5120), "m3k5120", Producer::kSilu},
    {17U, UINT64_C(5120), "m17k5120", Producer::kSilu},
    {1U, UINT64_C(17408), "m1k17408", Producer::kSilu},
    {17U, UINT64_C(17408), "m17k17408", Producer::kSilu},
    {2U, UINT64_C(17408), "m2k17408", Producer::kSilu},
    {1U, UINT64_C(6144), "m1k6144", Producer::kSilu},
    {1U, UINT64_C(5119), "m1k5119", Producer::kSilu},
    {1U, UINT64_C(5121), "m1k5121", Producer::kSilu},
    {1U, UINT64_C(31), "m1k31", Producer::kSilu},
    {1U, UINT64_C(32), "m1k32", Producer::kSilu},
    {1U, UINT64_C(33), "m1k33", Producer::kSilu},
    {1U, UINT64_C(17), "m1k17", Producer::kSilu},
    {1U, UINT64_C(6144), "m1h24d256", Producer::kSigmoid},
};

constexpr Shape kTimedShapes[] = {
    {1U, UINT64_C(5120), "m1k5120", Producer::kSilu},
    {3U, UINT64_C(5120), "m3k5120", Producer::kSilu},
    {1U, UINT64_C(17408), "m1k17408", Producer::kSilu},
    {2U, UINT64_C(17408), "m2k17408", Producer::kSilu},
    {1U, UINT64_C(6144), "m1h24d256", Producer::kSigmoid},
};

constexpr const char *kPatterns[] = {"normal", "specials", "zeros"};

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(const std::size_t elements) : elements_(elements) {
    if (elements_ != 0U) {
      S7_HIP_CHECK(hipMalloc(&pointer_, elements_ * sizeof(T)));
      ++g_live_buffers;
    }
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      (void)hipFree(pointer_);
      --g_live_buffers;
    }
  }
  T *get() const { return pointer_; }

private:
  T *pointer_ = nullptr;
  std::size_t elements_ = 0U;
};

void run_control_producer(const Producer producer, const uint16_t *const gate,
                          const uint16_t *const partner, uint16_t *const output,
                          const uint64_t element_count,
                          const hipStream_t stream) {
  if (producer == Producer::kSilu) {
    S7_HIP_CHECK(sllm_elementwise_kernel::launch_silu_mul(
        gate, partner, output, element_count, stream));
  } else {
    S7_HIP_CHECK(sllm_elementwise_kernel::launch_sigmoid_mul(
        gate, partner, output, element_count, stream));
  }
}

void run_fused_fp8(const Producer producer, const uint16_t *const gate,
                   const uint16_t *const partner, uint8_t *const codes,
                   float *const scales, const uint32_t m, const uint32_t k,
                   const hipStream_t stream) {
  if (producer == Producer::kSilu) {
    S7_HIP_CHECK(sllm_elementwise_kernel::launch_silu_mul_prequant_fp8(
        gate, partner, codes, scales, m, k, stream));
  } else {
    S7_HIP_CHECK(sllm_elementwise_kernel::launch_sigmoid_mul_prequant_fp8(
        gate, partner, codes, scales, m, k, stream));
  }
}

uint64_t fp8_code_bytes(const uint32_t m, const uint64_t k) {
  return static_cast<uint64_t>(m) * k;
}

uint64_t nvfp4_packed_bytes(const uint32_t m, const uint64_t k) {
  return static_cast<uint64_t>(m) * ((k + UINT64_C(1)) / UINT64_C(2));
}

uint64_t nvfp4_block_bytes(const uint32_t m, const uint64_t k) {
  return static_cast<uint64_t>(m) * ((k + UINT64_C(15)) / UINT64_C(16));
}

size_t byte_diff(const std::vector<uint8_t> &left,
                 const std::vector<uint8_t> &right) {
  const size_t bound = std::min(left.size(), right.size());
  size_t differences = left.size() == right.size() ? 0U : 1U;
  for (size_t index = 0; index < bound; ++index) {
    if (left[index] != right[index]) {
      ++differences;
    }
  }
  return differences;
}

double median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  const size_t middle = values.size() / 2U;
  std::nth_element(values.begin(),
                   values.begin() + static_cast<std::ptrdiff_t>(middle),
                   values.end());
  return values[middle];
}

void emit(const std::string &line) { std::cout << line << '\n'; }

void upload(uint16_t *const device, const std::vector<uint16_t> &host) {
  S7_HIP_CHECK(hipMemcpy(device, host.data(), host.size() * sizeof(uint16_t),
                         hipMemcpyHostToDevice));
}

} // namespace

int main(const int argc, char **const argv) {
  const std::string target = arg_value(argc, argv, "--target", "");
  const int warmup_ms =
      std::atoi(arg_value(argc, argv, "--warmup-ms", "300").c_str());
  const int samples =
      std::atoi(arg_value(argc, argv, "--samples", "9").c_str());
  const int reps = std::atoi(arg_value(argc, argv, "--reps", "100").c_str());
  need(!target.empty(), "--target=gfx1030|gfx1201 is required");
  need(warmup_ms > 0 && samples >= 3 && reps > 0,
       "warmup-ms>0, samples>=3 and reps>0 are required");

  // Optional single-shape filter, used to attribute a regression to one
  // (m, k) pair instead of the whole matrix.
  const std::string shape_filter = arg_value(argc, argv, "--shape", "");
  std::vector<Shape> shapes(std::begin(kShapes), std::end(kShapes));
  std::vector<Shape> timed_shapes(std::begin(kTimedShapes),
                                  std::end(kTimedShapes));
  if (!shape_filter.empty()) {
    const auto mismatch = [&shape_filter](const Shape &shape) {
      return shape_filter != shape.name;
    };
    shapes.erase(std::remove_if(shapes.begin(), shapes.end(), mismatch),
                 shapes.end());
    timed_shapes.erase(
        std::remove_if(timed_shapes.begin(), timed_shapes.end(), mismatch),
        timed_shapes.end());
    need(!shapes.empty() || !timed_shapes.empty(),
         "--shape=" + shape_filter + " is not a known shape");
  }

  // Pin the control to the same production kernels the fused candidates
  // mirror: FP8 v2 and the NVFP4 wave8 quantizer.
  ::unsetenv("SLLM_FP8_QUANT_FORCE_BASELINE");
  ::unsetenv("SLLM_NVFP4_FORCE_BASELINE");
  ::unsetenv("SLLM_NVFP4_W4A4_FORCE_BASELINE");
  ::setenv("SLLM_NVFP4_ACTIVATION_QUANTIZE_WAVE8", "1", 1);

  int device = 0;
  S7_HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t properties{};
  S7_HIP_CHECK(hipGetDeviceProperties(&properties, device));
  const std::string arch(properties.gcnArchName);
  need(arch.find(target) != std::string::npos,
       "runtime gfx target " + arch + " does not match --target=" + target);

  float input_global_scale = 836.0F;
  float *global_scale = nullptr;
  S7_HIP_CHECK(hipMalloc(&global_scale, sizeof(float)));

  emit("{\"kind\":\"meta\",\"target\":\"" + target + "\",\"device\":\"" +
       std::string(properties.name) + "\",\"gcn_arch\":\"" + arch +
       "\",\"warmup_ms\":" + std::to_string(warmup_ms) + ",\"samples\":" +
       std::to_string(samples) + ",\"reps\":" + std::to_string(reps) +
       ",\"fp8_control\":\"v2\",\"nvfp4_control\":\"wave8\"}");

  size_t failures = 0U;
  const std::vector<std::string> patterns(std::begin(kPatterns),
                                          std::end(kPatterns));
  const float global_candidates[] = {836.0F, 1.0F,  60.75F,
                                     19.0F,  74.0F, 504.0F};

  for (const Shape &shape : shapes) {
    const uint32_t m = shape.m;
    const uint64_t k = shape.k;
    const size_t row_elements = static_cast<size_t>(m) * k;

    for (const std::string &pattern : patterns) {
      DeviceBuffer<uint16_t> gate(row_elements);
      DeviceBuffer<uint16_t> partner(row_elements);
      DeviceBuffer<uint16_t> control_bf16_output(row_elements);
      DeviceBuffer<uint8_t> control_codes(fp8_code_bytes(m, k));
      DeviceBuffer<float> control_scales(m);
      DeviceBuffer<uint8_t> fused_codes(fp8_code_bytes(m, k));
      DeviceBuffer<float> fused_scales(m);
      DeviceBuffer<uint8_t> control_packed(nvfp4_packed_bytes(m, k));
      DeviceBuffer<uint8_t> control_blocks(nvfp4_block_bytes(m, k));
      DeviceBuffer<uint8_t> fused_packed(nvfp4_packed_bytes(m, k));
      DeviceBuffer<uint8_t> fused_blocks(nvfp4_block_bytes(m, k));

      std::vector<uint16_t> host(row_elements);
      fill_bf16(host.data(), host.size(), pattern, UINT32_C(0x1234), 119U,
                UINT32_C(16));
      upload(gate.get(), host);
      fill_bf16(host.data(), host.size(), pattern, UINT32_C(0x5678), 119U,
                UINT32_C(16));
      upload(partner.get(), host);

      for (const Producer producer : kProducers) {
        const std::string producer_label = producer_name(producer);

        // ---- FP8 producer fusion ----
        run_control_producer(producer, gate.get(), partner.get(),
                             control_bf16_output.get(),
                             static_cast<uint64_t>(row_elements), nullptr);
        S7_HIP_CHECK(sllm_matmul_kernel::launch_fp8_quantize(
            control_bf16_output.get(), control_codes.get(),
            control_scales.get(), m, k, false, nullptr));
        run_fused_fp8(producer, gate.get(), partner.get(), fused_codes.get(),
                      fused_scales.get(), m, k, nullptr);
        S7_HIP_CHECK(hipDeviceSynchronize());

        std::vector<uint8_t> host_control_codes(
            static_cast<size_t>(fp8_code_bytes(m, k)));
        std::vector<uint8_t> host_fused_codes(
            static_cast<size_t>(fp8_code_bytes(m, k)));
        S7_HIP_CHECK(hipMemcpy(host_control_codes.data(), control_codes.get(),
                               host_control_codes.size(),
                               hipMemcpyDeviceToHost));
        S7_HIP_CHECK(hipMemcpy(host_fused_codes.data(), fused_codes.get(),
                               host_fused_codes.size(), hipMemcpyDeviceToHost));
        const size_t code_diff =
            byte_diff(host_control_codes, host_fused_codes);

        std::vector<float> host_control_scales(m);
        std::vector<float> host_fused_scales(m);
        S7_HIP_CHECK(hipMemcpy(host_control_scales.data(), control_scales.get(),
                               static_cast<size_t>(m) * sizeof(float),
                               hipMemcpyDeviceToHost));
        S7_HIP_CHECK(hipMemcpy(host_fused_scales.data(), fused_scales.get(),
                               static_cast<size_t>(m) * sizeof(float),
                               hipMemcpyDeviceToHost));
        size_t scale_diff = 0U;
        for (size_t index = 0; index < m; ++index) {
          if (std::memcmp(&host_control_scales[index],
                          &host_fused_scales[index], sizeof(float)) != 0) {
            ++scale_diff;
          }
        }

        const bool fp8_ok = code_diff == 0U && scale_diff == 0U;
        if (!fp8_ok) {
          ++failures;
        }
        emit("{\"kind\":\"correctness\",\"format\":\"fp8\",\"shape\":\"" +
             std::string(shape.name) + "\",\"pattern\":\"" + pattern +
             "\",\"producer\":\"" + producer_label +
             "\",\"bitwise\":" + (fp8_ok ? "true" : "false") +
             ",\"code_mismatches\":" + std::to_string(code_diff) +
             ",\"scale_mismatches\":" + std::to_string(scale_diff) + "}");

        // ---- NVFP4 producer fusion (SiLU only in stage 7) ----
        if (producer != Producer::kSilu) {
          continue;
        }
        for (const float global : global_candidates) {
          input_global_scale = global;
          S7_HIP_CHECK(hipMemcpy(global_scale, &input_global_scale,
                                 sizeof(float), hipMemcpyHostToDevice));

          run_control_producer(producer, gate.get(), partner.get(),
                               control_bf16_output.get(),
                               static_cast<uint64_t>(row_elements), nullptr);
          S7_HIP_CHECK(sllm_matmul_kernel::launch_nvfp4_quantize(
              control_bf16_output.get(), control_packed.get(),
              control_blocks.get(), global_scale, m, k, nullptr));
          S7_HIP_CHECK(sllm_elementwise_kernel::launch_silu_mul_prequant_nvfp4(
              gate.get(), partner.get(), fused_packed.get(), fused_blocks.get(),
              global_scale, m, k, nullptr));
          S7_HIP_CHECK(hipDeviceSynchronize());

          std::vector<uint8_t> host_control_packed(
              static_cast<size_t>(nvfp4_packed_bytes(m, k)));
          std::vector<uint8_t> host_fused_packed(
              static_cast<size_t>(nvfp4_packed_bytes(m, k)));
          S7_HIP_CHECK(
              hipMemcpy(host_control_packed.data(), control_packed.get(),
                        host_control_packed.size(), hipMemcpyDeviceToHost));
          S7_HIP_CHECK(hipMemcpy(host_fused_packed.data(), fused_packed.get(),
                                 host_fused_packed.size(),
                                 hipMemcpyDeviceToHost));
          const size_t packed_diff =
              byte_diff(host_control_packed, host_fused_packed);

          std::vector<uint8_t> host_control_blocks(
              static_cast<size_t>(nvfp4_block_bytes(m, k)));
          std::vector<uint8_t> host_fused_blocks(
              static_cast<size_t>(nvfp4_block_bytes(m, k)));
          S7_HIP_CHECK(
              hipMemcpy(host_control_blocks.data(), control_blocks.get(),
                        host_control_blocks.size(), hipMemcpyDeviceToHost));
          S7_HIP_CHECK(hipMemcpy(host_fused_blocks.data(), fused_blocks.get(),
                                 host_fused_blocks.size(),
                                 hipMemcpyDeviceToHost));
          const size_t block_diff =
              byte_diff(host_control_blocks, host_fused_blocks);

          const bool nvfp4_ok = packed_diff == 0U && block_diff == 0U;
          if (!nvfp4_ok) {
            ++failures;
          }
          emit("{\"kind\":\"correctness\",\"format\":\"nvfp4\",\"shape\":\"" +
               std::string(shape.name) + "\",\"pattern\":\"" + pattern +
               "\",\"producer\":\"" + producer_label +
               "\",\"global\":" + std::to_string(global) +
               ",\"bitwise\":" + (nvfp4_ok ? "true" : "false") +
               ",\"packed_mismatches\":" + std::to_string(packed_diff) +
               ",\"block_mismatches\":" + std::to_string(block_diff) + "}");
        }
      }
    }
  }

  // ---- AB/BA timing ----
  hipStream_t stream = nullptr;
  S7_HIP_CHECK(hipStreamCreate(&stream));
  input_global_scale = 836.0F;
  S7_HIP_CHECK(hipMemcpy(global_scale, &input_global_scale, sizeof(float),
                         hipMemcpyHostToDevice));
  for (const Shape &shape : timed_shapes) {
    const uint32_t m = shape.m;
    const uint64_t k = shape.k;
    const size_t row_elements = static_cast<size_t>(m) * k;
    const Producer producer = shape.producer;

    DeviceBuffer<uint16_t> gate(row_elements);
    DeviceBuffer<uint16_t> partner(row_elements);
    DeviceBuffer<uint16_t> bf16_output(row_elements);
    DeviceBuffer<uint8_t> codes(static_cast<size_t>(fp8_code_bytes(m, k)));
    DeviceBuffer<float> scales(m);
    DeviceBuffer<uint8_t> packed(static_cast<size_t>(nvfp4_packed_bytes(m, k)));
    DeviceBuffer<uint8_t> blocks(static_cast<size_t>(nvfp4_block_bytes(m, k)));

    std::vector<uint16_t> host(row_elements);
    fill_bf16(host.data(), host.size(), "normal", UINT32_C(0x2468), 119U,
              UINT32_C(16));
    upload(gate.get(), host);
    fill_bf16(host.data(), host.size(), "normal", UINT32_C(0x1357), 119U,
              UINT32_C(16));
    upload(partner.get(), host);

    // Producer-only baseline so the two controls below can be decomposed into
    // producer time plus quantizer time.
    {
      auto producer_only = [&]() {
        run_control_producer(producer, gate.get(), partner.get(),
                             bf16_output.get(),
                             static_cast<uint64_t>(row_elements), stream);
      };
      const auto producer_start = std::chrono::steady_clock::now();
      while (std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - producer_start)
                 .count() < warmup_ms) {
        producer_only();
      }
      S7_HIP_CHECK(hipStreamSynchronize(stream));
      std::vector<double> producer_samples;
      producer_samples.reserve(static_cast<size_t>(samples));
      for (int sample = 0; sample < samples; ++sample) {
        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;
        S7_HIP_CHECK(hipEventCreate(&start));
        S7_HIP_CHECK(hipEventCreate(&stop));
        S7_HIP_CHECK(hipEventRecord(start, stream));
        for (int rep = 0; rep < reps; ++rep) {
          producer_only();
        }
        S7_HIP_CHECK(hipEventRecord(stop, stream));
        S7_HIP_CHECK(hipEventSynchronize(stop));
        float elapsed = 0.0F;
        S7_HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
        producer_samples.push_back(static_cast<double>(elapsed) / reps);
        S7_HIP_CHECK(hipEventDestroy(start));
        S7_HIP_CHECK(hipEventDestroy(stop));
      }
      emit("{\"kind\":\"timing\",\"format\":\"producer\",\"shape\":\"" +
           std::string(shape.name) + "\",\"control_ms\":" +
           std::to_string(median(producer_samples)) + "}");
    }

    // Stage 7 fuses NVFP4 only for SiLU-multiply; sigmoid times FP8 only.
    const int variant_count = producer == Producer::kSilu ? 2 : 1;
    for (int variant = 0; variant < variant_count; ++variant) {
      const bool is_fp8 = variant == 0;
      const std::string format = is_fp8 ? "fp8" : "nvfp4";

      auto control = [&]() {
        run_control_producer(producer, gate.get(), partner.get(),
                             bf16_output.get(),
                             static_cast<uint64_t>(row_elements), stream);
        if (is_fp8) {
          S7_HIP_CHECK(sllm_matmul_kernel::launch_fp8_quantize(
              bf16_output.get(), codes.get(), scales.get(), m, k, false,
              stream));
        } else {
          S7_HIP_CHECK(sllm_matmul_kernel::launch_nvfp4_quantize(
              bf16_output.get(), packed.get(), blocks.get(), global_scale, m, k,
              stream));
        }
      };
      auto fused = [&]() {
        if (is_fp8) {
          run_fused_fp8(producer, gate.get(), partner.get(), codes.get(),
                        scales.get(), m, k, stream);
        } else {
          S7_HIP_CHECK(sllm_elementwise_kernel::launch_silu_mul_prequant_nvfp4(
              gate.get(), partner.get(), packed.get(), blocks.get(),
              global_scale, m, k, stream));
        }
      };

      const auto warmup_start = std::chrono::steady_clock::now();
      while (std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - warmup_start)
                 .count() < warmup_ms) {
        control();
        fused();
      }
      S7_HIP_CHECK(hipStreamSynchronize(stream));

      std::vector<double> control_samples;
      std::vector<double> fused_samples;
      control_samples.reserve(static_cast<size_t>(samples));
      fused_samples.reserve(static_cast<size_t>(samples));
      for (int sample = 0; sample < samples; ++sample) {
        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;
        S7_HIP_CHECK(hipEventCreate(&start));
        S7_HIP_CHECK(hipEventCreate(&stop));
        const bool control_first = (sample % 2) == 0;
        for (int order = 0; order < 2; ++order) {
          const bool run_control = (order == 0) == control_first;
          S7_HIP_CHECK(hipEventRecord(start, stream));
          for (int rep = 0; rep < reps; ++rep) {
            if (run_control) {
              control();
            } else {
              fused();
            }
          }
          S7_HIP_CHECK(hipEventRecord(stop, stream));
          S7_HIP_CHECK(hipEventSynchronize(stop));
          float elapsed = 0.0F;
          S7_HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
          const double per_call = static_cast<double>(elapsed) / reps;
          if (run_control) {
            control_samples.push_back(per_call);
          } else {
            fused_samples.push_back(per_call);
          }
        }
        S7_HIP_CHECK(hipEventDestroy(start));
        S7_HIP_CHECK(hipEventDestroy(stop));
      }

      const double control_ms = median(control_samples);
      const double fused_ms = median(fused_samples);
      emit(
          "{\"kind\":\"timing\",\"format\":\"" + format + "\",\"shape\":\"" +
          shape.name + "\",\"control_ms\":" + std::to_string(control_ms) +
          ",\"fused_ms\":" + std::to_string(fused_ms) + ",\"delta_ms\":" +
          std::to_string(control_ms - fused_ms) + ",\"speedup\":" +
          std::to_string(control_ms / (fused_ms > 1.0e-9 ? fused_ms : 1.0e-9)) +
          "}");
    }
  }
  S7_HIP_CHECK(hipStreamDestroy(stream));
  S7_HIP_CHECK(hipFree(global_scale));

  if (g_live_buffers != 0) {
    fail("device buffers leaked: " + std::to_string(g_live_buffers));
  }
  emit("{\"kind\":\"cleanup\",\"live\":0,\"failures\":" +
       std::to_string(failures) + "}");

  if (failures != 0U) {
    std::cerr << "FAIL: " << failures << " bitwise comparison(s) failed\n";
    return 1;
  }
  return 0;
}
