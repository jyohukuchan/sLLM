// Phase 87 WU-D2: measure how long the predecessor-dependent NVFP4 delay
// persists, and whether representative intervening work clears it.
//
// The WU-D1 translation unit is included deliberately so this probe calls the
// same production NVFP4 launcher and uses the same synthetic KV readers,
// shapes, payload, four-copy weight pool, and 37-column oracle.  This file
// owns a different main and does not alter a selector or production source.
//
// Suggested direct build (one exact target per binary):
//   amdclang++ -D__HIP_ROCclr__=1 -O3 -ffp-contract=off -DNDEBUG \
//     -std=gnu++17 --offload-arch=gfx1030 -mcode-object-version=6 \
//     -mno-wavefrontsize64 -x hip -I native/lowp/include -I include \
//     -I native/hip/src -c
//     native/hip/tests/phase87_wud2_persistence_probe.hip.cpp \ -o
//     phase87-wud2-gfx1030.o
//   amdclang++ -O3 -ffp-contract=off -DNDEBUG --offload-arch=gfx1030 \
//     -mcode-object-version=6 -mno-wavefrontsize64 --hip-link \
//     phase87-wud2-gfx1030.o /path/to/libsllm_lowp.a /path/to/libsllm_hip.a \
//     -L/opt/rocm/lib -lrocprofiler-sdk-roctx -lamdhip64 \
//     -o phase87-wud2-gfx1030
//
// The normal run is both shapes, 300 ms continuous warmup, same-process ABBA,
// three rounds, and nine samples.  Bounded profiling can use
// --shape=wide --warmup-ms=0 --rounds=1 --samples=1 --n-list=1,2.

#define main phase87_wud1_embedded_main
#include "phase87_wud1_nvfp4_neighbor_probe.hip.cpp"
#undef main

#include "../src/elementwise_kernel_internal.hpp"
#include "../src/rmsnorm_kernel_internal.hpp"
#include "sllm/hip.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kDefaultCopyBytes = 1U << 20U;
constexpr uint32_t kRmsNormWidth = 256U;
constexpr float kRmsNormEpsilon = 1.0e-5F;

enum class Intervention : uint32_t {
  None,
  Quantize,
  RmsNorm,
  Copy,
};

struct Wud2Options final {
  std::string target;
  std::string shape = "both";
  std::string predecessor = "all";
  std::string intervention = "all";
  std::string n_list = "1,2,4,8,16,32";
  uint32_t warmup_ms = kDefaultWarmupMs;
  uint32_t rounds = kDefaultRounds;
  uint32_t samples = kDefaultSamples;
};

struct HostIntervention final {
  std::vector<uint16_t> bf16;
  std::vector<uint8_t> quantized;
  std::vector<uint8_t> quant_scales;
  std::vector<uint16_t> rms_input;
  std::vector<uint16_t> rms_scale;
  std::vector<uint16_t> rms_expected;
  std::vector<uint16_t> copy_input;
};

struct InterventionBuffers final {
  DeviceBytes bf16;
  DeviceBytes quantized;
  DeviceBytes quant_scales;
  DeviceBytes tensor_scale;
  DeviceBytes rms_input;
  DeviceBytes rms_scale;
  DeviceBytes rms_output;
  DeviceBytes copy_input;
  DeviceBytes copy_output;

  explicit InterventionBuffers(const Shape shape)
      : bf16(static_cast<size_t>(shape.k * sizeof(uint16_t))),
        quantized(static_cast<size_t>(shape.k / 2U)),
        quant_scales(static_cast<size_t>(shape.k / 16U)),
        tensor_scale(sizeof(float)),
        rms_input(static_cast<size_t>(kRmsNormWidth * sizeof(uint16_t))),
        rms_scale(static_cast<size_t>(kRmsNormWidth * sizeof(uint16_t))),
        rms_output(static_cast<size_t>(kRmsNormWidth * sizeof(uint16_t))),
        copy_input(kDefaultCopyBytes), copy_output(kDefaultCopyBytes) {}
};

struct SequenceTiming final {
  std::vector<float> dispatch_us;
  float sum_us = 0.0F;
  float wall_us = 0.0F;
};

uint8_t host_e4m3fn_encode(const float value) {
  if (std::isnan(value))
    return 0x7fU;
  const uint8_t sign = std::signbit(value) ? 0x80U : 0U;
  const float magnitude = std::fabs(value);
  if (magnitude == 0.0F)
    return sign;
  if (!std::isfinite(magnitude) || magnitude >= 448.0F)
    return static_cast<uint8_t>(sign | 0x7eU);
  if (magnitude < 0.015625F) {
    const float scaled = magnitude * 512.0F;
    const uint32_t floor = static_cast<uint32_t>(scaled);
    const float fraction = scaled - static_cast<float>(floor);
    const uint32_t rounded =
        floor + static_cast<uint32_t>(fraction > 0.5F ||
                                      (fraction == 0.5F && (floor & 1U)));
    return static_cast<uint8_t>(sign | static_cast<uint8_t>(rounded));
  }
  uint32_t bits = 0U;
  std::memcpy(&bits, &magnitude, sizeof(bits));
  const uint32_t rounded =
      bits + UINT32_C(0x0007ffff) + ((bits >> 20U) & UINT32_C(1));
  const uint32_t exponent = ((rounded >> 23U) & UINT32_C(0xff)) - 120U;
  const uint32_t code = (exponent << 3U) | ((rounded >> 20U) & 0x07U);
  return static_cast<uint8_t>(sign |
                              static_cast<uint8_t>(std::min(code, 0x7eU)));
}

uint8_t host_e2m1_encode(const float value) {
  static constexpr std::array<float, 8> positive = {0.0F, 0.5F, 1.0F, 1.5F,
                                                    2.0F, 3.0F, 4.0F, 6.0F};
  const uint8_t sign = std::signbit(value) ? 0x08U : 0U;
  if (std::isnan(value))
    return sign;
  const float magnitude = std::min(std::fabs(value), 6.0F);
  uint8_t selected = 0U;
  float selected_error = magnitude;
  for (uint8_t code = 1U; code != 8U; ++code) {
    const float error = std::fabs(magnitude - positive[code]);
    if (error < selected_error ||
        (error == selected_error && (code & 1U) == 0U &&
         (selected & 1U) != 0U)) {
      selected = code;
      selected_error = error;
    }
  }
  return static_cast<uint8_t>(sign | selected);
}

std::vector<uint16_t> make_bf16_fixture(const uint64_t count,
                                        const uint32_t salt) {
  std::vector<uint16_t> result(static_cast<size_t>(count));
  for (uint64_t index = 0U; index < count; ++index) {
    const int32_t bucket =
        static_cast<int32_t>((index * 17U + salt * 13U) % 31U) - 15;
    const float value = static_cast<float>(bucket) * 0.125F +
                        static_cast<float>((index + salt) % 5U) * 0.03125F;
    result[static_cast<size_t>(index)] = bf16_rne(value);
  }
  return result;
}

HostIntervention make_host_intervention(const Shape shape) {
  HostIntervention host;
  host.bf16 = make_bf16_fixture(shape.k, shape.k == 5120U ? 3U : 7U);
  host.quantized.assign(static_cast<size_t>(shape.k / 2U), 0U);
  host.quant_scales.assign(static_cast<size_t>(shape.k / 16U), 0U);
  constexpr float input_tensor_scale = 1.0F;
  for (uint64_t block = 0U; block < shape.k / 16U; ++block) {
    float maximum = 0.0F;
    for (uint32_t index = 0U; index < 16U; ++index) {
      maximum = std::max(
          maximum, std::fabs(bf16_to_float(host.bf16[block * 16U + index])));
    }
    const uint8_t encoded = host_e4m3fn_encode(
        maximum == 0.0F ? 0.0F : maximum / (6.0F * input_tensor_scale));
    host.quant_scales[static_cast<size_t>(block)] = encoded;
    const float decoded_scale = e4m3fn(encoded) * input_tensor_scale;
    for (uint32_t pair = 0U; pair < 8U; ++pair) {
      const uint64_t first = block * 16U + pair * 2U;
      const uint8_t low =
          decoded_scale > 0.0F
              ? host_e2m1_encode(bf16_to_float(host.bf16[first]) /
                                 decoded_scale)
              : 0U;
      const uint8_t high =
          decoded_scale > 0.0F
              ? host_e2m1_encode(bf16_to_float(host.bf16[first + 1U]) /
                                 decoded_scale)
              : 0U;
      host.quantized[static_cast<size_t>(first / 2U)] =
          static_cast<uint8_t>(low | static_cast<uint8_t>(high << 4U));
    }
  }

  host.rms_input = make_bf16_fixture(kRmsNormWidth, 23U);
  host.rms_scale.assign(kRmsNormWidth, bf16_rne(1.0F));
  host.rms_expected.resize(kRmsNormWidth);
  float sum = 0.0F;
  for (const uint16_t value : host.rms_input) {
    const float decoded = bf16_to_float(value);
    sum += decoded * decoded;
  }
  const float inverse_rms =
      1.0F /
      std::sqrt(sum / static_cast<float>(kRmsNormWidth) + kRmsNormEpsilon);
  for (size_t index = 0U; index < host.rms_input.size(); ++index)
    host.rms_expected[index] =
        bf16_rne(bf16_to_float(host.rms_input[index]) * inverse_rms);

  host.copy_input = make_bf16_fixture(kDefaultCopyBytes / sizeof(uint16_t),
                                      shape.k == 5120U ? 31U : 37U);
  return host;
}

void upload_intervention(const HostIntervention &host,
                         InterventionBuffers *const device) {
  device->bf16.upload(host.bf16.data());
  device->quantized.upload(host.quantized.data());
  device->quant_scales.upload(host.quant_scales.data());
  constexpr float input_tensor_scale = 1.0F;
  device->tensor_scale.upload(&input_tensor_scale);
  device->rms_input.upload(host.rms_input.data());
  device->rms_scale.upload(host.rms_scale.data());
  device->copy_input.upload(host.copy_input.data());
  check(hipMemset(device->rms_output.ptr, 0, device->rms_output.bytes),
        "clear RMSNorm output");
  check(hipMemset(device->copy_output.ptr, 0, device->copy_output.bytes),
        "clear copy output");
}

void launch_intervention(const Intervention kind, const Shape shape,
                         InterventionBuffers *const device,
                         hipStream_t stream) {
  switch (kind) {
  case Intervention::None:
    return;
  case Intervention::Quantize:
    check(sllm_matmul_kernel::launch_nvfp4_quantize(
              reinterpret_cast<const uint16_t *>(device->bf16.ptr),
              device->quantized.ptr, device->quant_scales.ptr,
              reinterpret_cast<const float *>(device->tensor_scale.ptr), 1U,
              shape.k, stream),
          "launch production NVFP4 activation quantizer");
    return;
  case Intervention::RmsNorm:
    check(sllm_rmsnorm_kernel::launch(
              reinterpret_cast<const uint16_t *>(device->rms_input.ptr),
              reinterpret_cast<const uint16_t *>(device->rms_scale.ptr),
              reinterpret_cast<uint16_t *>(device->rms_output.ptr),
              kRmsNormWidth, 1U, kRmsNormEpsilon,
              SLLM_RMSNORM_SCALE_MODE_DIRECT, stream),
          "launch production RMSNorm light kernel");
    return;
  case Intervention::Copy:
    check(sllm_elementwise_kernel::launch_copy(
              reinterpret_cast<const uint16_t *>(device->copy_input.ptr),
              reinterpret_cast<uint16_t *>(device->copy_output.ptr),
              kDefaultCopyBytes / sizeof(uint16_t), stream),
          "launch production fixed-size copy");
    return;
  }
}

bool compare_quantizer(const Shape shape, const HostIntervention &host,
                       InterventionBuffers *const device) {
  std::vector<uint8_t> quantized(host.quantized.size());
  std::vector<uint8_t> scales(host.quant_scales.size());
  check(hipMemcpy(quantized.data(), device->quantized.ptr, quantized.size(),
                  hipMemcpyDeviceToHost),
        "download quantized intervention");
  check(hipMemcpy(scales.data(), device->quant_scales.ptr, scales.size(),
                  hipMemcpyDeviceToHost),
        "download quantizer scales");
  const size_t packed_mismatch = std::inner_product(
      quantized.begin(), quantized.end(), host.quantized.begin(), size_t{0},
      std::plus<size_t>(), [](const uint8_t lhs, const uint8_t rhs) {
        return static_cast<size_t>(lhs != rhs);
      });
  const size_t scale_mismatch = std::inner_product(
      scales.begin(), scales.end(), host.quant_scales.begin(), size_t{0},
      std::plus<size_t>(), [](const uint8_t lhs, const uint8_t rhs) {
        return static_cast<size_t>(lhs != rhs);
      });
  const bool meaningful =
      std::any_of(quantized.begin(), quantized.end(),
                  [](const uint8_t value) { return value != 0U; }) &&
      std::any_of(scales.begin(), scales.end(),
                  [](const uint8_t value) { return value != 0U; });
  const bool pass = packed_mismatch == 0U && scale_mismatch == 0U && meaningful;
  std::printf("intervention_oracle shape=%s kind=quantize packed_mismatch=%zu "
              "scale_mismatch=%zu meaningful=%s status=%s\n",
              shape.name, packed_mismatch, scale_mismatch,
              meaningful ? "true" : "false", pass ? "PASS" : "FAIL");
  return pass;
}

bool compare_rmsnorm(const Shape shape, const HostIntervention &host,
                     InterventionBuffers *const device) {
  std::vector<uint16_t> observed(kRmsNormWidth);
  check(hipMemcpy(observed.data(), device->rms_output.ptr,
                  observed.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
        "download RMSNorm intervention");
  uint32_t max_ulp = 0U;
  bool differs = false;
  bool finite = true;
  for (size_t index = 0U; index < observed.size(); ++index) {
    max_ulp = std::max(
        max_ulp, bf16_ulp_distance(observed[index], host.rms_expected[index]));
    differs = differs || observed[index] != host.rms_input[index];
    finite = finite && std::isfinite(bf16_to_float(observed[index]));
  }
  const bool pass = max_ulp <= 2U && differs && finite;
  std::printf("intervention_oracle shape=%s kind=rmsnorm max_ulp=%u differs=%s "
              "finite=%s status=%s\n",
              shape.name, max_ulp, differs ? "true" : "false",
              finite ? "true" : "false", pass ? "PASS" : "FAIL");
  return pass;
}

bool compare_copy(const Shape shape, const HostIntervention &host,
                  InterventionBuffers *const device) {
  std::vector<uint16_t> observed(host.copy_input.size());
  check(hipMemcpy(observed.data(), device->copy_output.ptr,
                  observed.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
        "download fixed-size copy intervention");
  const bool pass = observed == host.copy_input;
  std::printf("intervention_oracle shape=%s kind=copy bytes=%u status=%s\n",
              shape.name, kDefaultCopyBytes, pass ? "PASS" : "FAIL");
  return pass;
}

bool check_finite_output(const Shape shape, const std::vector<uint16_t> &output,
                         const char *const label) {
  const bool finite =
      std::all_of(output.begin(), output.end(), [](const uint16_t value) {
        return std::isfinite(bf16_to_float(value));
      });
  std::printf("output_finite shape=%s label=%s elements=%zu status=%s\n",
              shape.name, label, output.size(), finite ? "PASS" : "FAIL");
  return finite;
}

bool check_intervention(const Intervention kind, const Shape shape,
                        const HostIntervention &host,
                        InterventionBuffers *const device, hipStream_t stream) {
  launch_intervention(kind, shape, device, stream);
  check(hipStreamSynchronize(stream), "intervention oracle synchronize");
  switch (kind) {
  case Intervention::None:
    return true;
  case Intervention::Quantize:
    return compare_quantizer(shape, host, device);
  case Intervention::RmsNorm:
    return compare_rmsnorm(shape, host, device);
  case Intervention::Copy:
    return compare_copy(shape, host, device);
  }
  return false;
}

std::string intervention_name(const Intervention kind) {
  switch (kind) {
  case Intervention::None:
    return "none";
  case Intervention::Quantize:
    return "quantize";
  case Intervention::RmsNorm:
    return "rmsnorm";
  case Intervention::Copy:
    return "copy";
  }
  return "unknown";
}

std::vector<Intervention> selected_interventions(const Wud2Options &options) {
  if (options.intervention == "all")
    return {Intervention::None, Intervention::Quantize, Intervention::RmsNorm,
            Intervention::Copy};
  const std::array<std::pair<std::string_view, Intervention>, 4> names = {{
      {"none", Intervention::None},
      {"quantize", Intervention::Quantize},
      {"rmsnorm", Intervention::RmsNorm},
      {"copy", Intervention::Copy},
  }};
  for (const auto &[name, value] : names)
    if (options.intervention == name)
      return {value};
  fail("unknown --intervention: " + options.intervention);
}

std::vector<Mode> selected_predecessors(const Wud2Options &options) {
  if (options.predecessor == "all")
    return {Mode::Isolated, Mode::Staged32, Mode::Gqa128};
  const std::array<std::pair<std::string_view, Mode>, 3> names = {{
      {"isolated", Mode::Isolated},
      {"staged32", Mode::Staged32},
      {"gqa128", Mode::Gqa128},
  }};
  for (const auto &[name, value] : names)
    if (options.predecessor == name)
      return {value};
  fail("unknown --predecessor: " + options.predecessor);
}

std::vector<uint32_t> parse_n_list(const std::string &value) {
  std::vector<uint32_t> result;
  size_t begin = 0U;
  while (begin < value.size()) {
    const size_t comma = value.find(',', begin);
    const size_t end = comma == std::string::npos ? value.size() : comma;
    const uint32_t n =
        static_cast<uint32_t>(std::stoul(value.substr(begin, end - begin)));
    if (n == 0U || n > 32U || (n & (n - 1U)) != 0U)
      fail("--n-list values must be powers of two in 1..32");
    result.push_back(n);
    begin = comma == std::string::npos ? value.size() : comma + 1U;
  }
  if (result.empty())
    fail("--n-list must not be empty");
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

std::vector<size_t> selected_shapes_wud2(const Wud2Options &options) {
  if (options.shape == "both")
    return {0U, 1U};
  if (options.shape == "wide")
    return {0U};
  if (options.shape == "down")
    return {1U};
  fail("unknown --shape: " + options.shape);
}

SequenceTiming measure_sequence(const Shape shape, const Mode predecessor,
                                const Intervention intervention,
                                const uint32_t n, DeviceMatrix *const device,
                                InterventionBuffers *const intervention_device,
                                const uint32_t weight_offset) {
  // Create every event before enqueueing the predecessor.  HIP event creation
  // is a host operation; doing it after the predecessor could leave a
  // predecessor-only gap at the exact cache-state boundary being measured.
  std::vector<hipEvent_t> starts(n);
  std::vector<hipEvent_t> stops(n);
  for (uint32_t index = 0U; index < n; ++index) {
    check(hipEventCreate(&starts[index]), "create sequence start event");
    check(hipEventCreate(&stops[index]), "create sequence stop event");
  }
  hipEvent_t wall_start = nullptr;
  hipEvent_t wall_stop = nullptr;
  check(hipEventCreate(&wall_start), "create sequence wall start event");
  check(hipEventCreate(&wall_stop), "create sequence wall stop event");
  if (predecessor != Mode::Isolated)
    launch_predecessor(predecessor, device);
  launch_intervention(intervention, shape, intervention_device, device->stream);
  check(hipEventRecord(wall_start, device->stream),
        "record sequence wall start");
  for (uint32_t index = 0U; index < n; ++index) {
    check(hipEventRecord(starts[index], device->stream),
          "record sequence start");
    launch_matmul(shape, device, (weight_offset + index) % kPoolCopies);
    check(hipEventRecord(stops[index], device->stream), "record sequence stop");
  }
  check(hipEventRecord(wall_stop, device->stream), "record sequence wall stop");
  check(hipStreamSynchronize(device->stream), "sequence synchronize");

  SequenceTiming timing;
  timing.dispatch_us.resize(n);
  for (uint32_t index = 0U; index < n; ++index) {
    float elapsed_ms = 0.0F;
    check(hipEventElapsedTime(&elapsed_ms, starts[index], stops[index]),
          "sequence dispatch elapsed");
    timing.dispatch_us[index] = elapsed_ms * 1000.0F;
    timing.sum_us += timing.dispatch_us[index];
    (void)hipEventDestroy(starts[index]);
    (void)hipEventDestroy(stops[index]);
  }
  float elapsed_ms = 0.0F;
  check(hipEventElapsedTime(&elapsed_ms, wall_start, wall_stop),
        "sequence wall elapsed");
  timing.wall_us = elapsed_ms * 1000.0F;
  (void)hipEventDestroy(wall_start);
  (void)hipEventDestroy(wall_stop);
  return timing;
}

void warmup_sequence(const Shape shape, const Mode predecessor,
                     const Intervention intervention, const uint32_t n,
                     DeviceMatrix *const device,
                     InterventionBuffers *const intervention_device,
                     const uint32_t warmup_ms) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(warmup_ms);
  uint32_t launch_count = 0U;
  do {
    for (uint32_t batch = 0U; batch < 8U; ++batch) {
      if (predecessor != Mode::Isolated)
        launch_predecessor(predecessor, device);
      launch_intervention(intervention, shape, intervention_device,
                          device->stream);
      for (uint32_t index = 0U; index < n; ++index)
        launch_matmul(shape, device, (launch_count + index) % kPoolCopies);
      launch_count += n;
    }
    check(hipStreamSynchronize(device->stream), "sequence warmup synchronize");
  } while (std::chrono::steady_clock::now() < deadline);
  std::printf("warmup shape=%s predecessor=%s intervention=%s n=%u ms=%u "
              "nvfp4_launches=%u weight_pool=%u\n",
              shape.name, mode_name(predecessor).c_str(),
              intervention_name(intervention).c_str(), n, warmup_ms,
              launch_count, kPoolCopies);
}

struct Summary final {
  std::vector<float> dispatch_medians;
  float sum_median = 0.0F;
  float wall_median = 0.0F;
};

struct Cell final {
  Mode predecessor;
  Intervention intervention;
  uint32_t n;
};

Summary summarize(const std::vector<SequenceTiming> &timings,
                  const uint32_t n) {
  Summary summary;
  summary.dispatch_medians.resize(n);
  for (uint32_t index = 0U; index < n; ++index) {
    std::vector<float> values;
    values.reserve(timings.size());
    for (const SequenceTiming &timing : timings)
      values.push_back(timing.dispatch_us[index]);
    std::sort(values.begin(), values.end());
    summary.dispatch_medians[index] = values[values.size() / 2U];
  }
  std::vector<float> sums;
  std::vector<float> walls;
  for (const SequenceTiming &timing : timings) {
    sums.push_back(timing.sum_us);
    walls.push_back(timing.wall_us);
  }
  std::sort(sums.begin(), sums.end());
  std::sort(walls.begin(), walls.end());
  summary.sum_median = sums[sums.size() / 2U];
  summary.wall_median = walls[walls.size() / 2U];
  return summary;
}

void parse_options(const int argc, char **const argv,
                   Wud2Options *const options) {
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    const auto value = [&](const std::string_view prefix) -> std::string {
      if (argument.substr(0U, prefix.size()) != prefix)
        return {};
      return std::string(argument.substr(prefix.size()));
    };
    if (argument == "--help") {
      std::printf(
          "options: --target=<gfx1030|gfx1201> --shape=<wide|down|both> "
          "--predecessor=<all|isolated|staged32|gqa128> "
          "--intervention=<all|none|quantize|rmsnorm|copy> "
          "--n-list=1,2,4,8,16,32 --warmup-ms=<N> --rounds=<N> "
          "--samples=<N>\n");
      std::exit(EXIT_SUCCESS);
    }
    if (const std::string target = value("--target="); !target.empty()) {
      options->target = target;
      continue;
    }
    if (const std::string shape = value("--shape="); !shape.empty()) {
      options->shape = shape;
      continue;
    }
    if (const std::string predecessor = value("--predecessor=");
        !predecessor.empty()) {
      options->predecessor = predecessor;
      continue;
    }
    if (const std::string intervention = value("--intervention=");
        !intervention.empty()) {
      options->intervention = intervention;
      continue;
    }
    if (const std::string n_list = value("--n-list="); !n_list.empty()) {
      options->n_list = n_list;
      continue;
    }
    if (const std::string warmup = value("--warmup-ms="); !warmup.empty()) {
      options->warmup_ms = static_cast<uint32_t>(std::stoul(warmup));
      continue;
    }
    if (const std::string rounds = value("--rounds="); !rounds.empty()) {
      options->rounds = static_cast<uint32_t>(std::stoul(rounds));
      continue;
    }
    if (const std::string samples = value("--samples="); !samples.empty()) {
      options->samples = static_cast<uint32_t>(std::stoul(samples));
      continue;
    }
    fail("unknown argument: " + std::string(argument));
  }
  if (options->rounds == 0U || options->samples == 0U)
    fail("--rounds and --samples must be nonzero");
}

int run_wud2(const Wud2Options &options) {
  int device_index = 0;
  check(hipGetDevice(&device_index), "get current HIP device");
  hipDeviceProp_t properties{};
  check(hipGetDeviceProperties(&properties, device_index),
        "get HIP device properties");
  const std::string runtime_target(properties.gcnArchName);
  if (options.target.empty())
    fail("--target=<gfx1030|gfx1201> is required for exact-target evidence");
  if (options.target != "gfx1030" && options.target != "gfx1201")
    fail("--target must be exactly gfx1030 or gfx1201");
  if (runtime_target != options.target)
    fail("runtime gfx target " + runtime_target +
         " does not match --target=" + options.target);
  const std::vector<uint32_t> n_values = parse_n_list(options.n_list);
  const std::vector<Mode> predecessors = selected_predecessors(options);
  const std::vector<Intervention> interventions =
      selected_interventions(options);
  std::printf("probe=phase87_wud2_nvfp4_persistence target=%s runtime_gfx=%s "
              "predecessor_selection=%s "
              "intervention_selection=%s copy_bytes=%u "
              "n_list=%s warmup_ms=%u rounds=%u samples=%u weight_pool=%u "
              "event_mode=one_stream_sequence\n",
              options.target.empty() ? "unspecified" : options.target.c_str(),
              runtime_target.c_str(), options.predecessor.c_str(),
              options.intervention.c_str(), kDefaultCopyBytes,
              options.n_list.c_str(), options.warmup_ms, options.rounds,
              options.samples, kPoolCopies);

  bool all_ok = true;
  for (const size_t shape_index : selected_shapes_wud2(options)) {
    const Shape shape = kShapes[shape_index];
    DeviceMatrix device(shape);
    InterventionBuffers intervention_device(shape);
    upload_matrix(shape, &device);
    const HostIntervention host = make_host_intervention(shape);
    upload_intervention(host, &intervention_device);
    for (const Intervention intervention : interventions) {
      all_ok = check_intervention(intervention, shape, host,
                                  &intervention_device, device.stream) &&
               all_ok;
    }

    // The predecessor checksum and the 37-column production output are
    // checked once per predecessor.  This keeps every intervening operation
    // observable without adding D2H work to measured sequences.
    launch_matmul(shape, &device, 0U);
    check(hipStreamSynchronize(device.stream),
          "baseline correctness synchronize");
    const std::vector<uint16_t> control = download_output(shape, &device);
    std::vector<uint8_t> oracle_activation;
    std::vector<uint8_t> oracle_activation_scales;
    std::vector<uint8_t> oracle_weights;
    std::vector<uint8_t> oracle_weight_scales;
    fill_host_matrix(shape, &oracle_activation, &oracle_activation_scales,
                     &oracle_weights, &oracle_weight_scales);
    const std::vector<uint16_t> oracle =
        host_oracle_prefix(shape, oracle_activation, oracle_activation_scales,
                           oracle_weights, oracle_weight_scales);
    uint32_t oracle_max_ulp = 0U;
    for (size_t index = 0U; index < oracle.size(); ++index)
      oracle_max_ulp = std::max(
          oracle_max_ulp, bf16_ulp_distance(control[index], oracle[index]));
    std::printf("numerical_oracle shape=%s columns=%zu max_ulp=%u status=%s\n",
                shape.name, oracle.size(), oracle_max_ulp,
                oracle_max_ulp <= 4U ? "PASS" : "FAIL");
    all_ok = oracle_max_ulp <= 4U && all_ok;
    all_ok = check_finite_output(shape, control, "isolated") && all_ok;
    for (const Mode predecessor : predecessors) {
      if (predecessor != Mode::Isolated)
        launch_predecessor(predecessor, &device);
      launch_matmul(shape, &device, 0U);
      check(hipStreamSynchronize(device.stream),
            "predecessor correctness synchronize");
      all_ok = check_neighbor_checksum(predecessor, &device) && all_ok;
      const std::vector<uint16_t> observed = download_output(shape, &device);
      all_ok = check_outputs(shape, control, observed,
                             mode_name(predecessor).c_str()) &&
               all_ok;
      all_ok = check_finite_output(shape, observed,
                                   mode_name(predecessor).c_str()) &&
               all_ok;
    }

    std::vector<Cell> cells;
    // Measure each intervention after all three predecessor arrangements so
    // that GQA recovery is compared with isolated and staged matched controls.
    for (const Mode predecessor : predecessors) {
      for (const Intervention intervention : interventions) {
        for (const uint32_t n : n_values)
          cells.push_back(Cell{predecessor, intervention, n});
      }
    }
    // Match D1's same-process ABBA ordering: every round measures the cells in
    // forward order and then in reverse order.  The warmup belongs to the cell
    // immediately before its samples, so each cell retains the same cache and
    // stream preparation in both halves.
    for (uint32_t round = 0U; round < options.rounds; ++round) {
      for (const bool reverse : {false, true}) {
        for (size_t position = 0U; position < cells.size(); ++position) {
          const size_t cell_index =
              reverse ? cells.size() - 1U - position : position;
          const Cell cell = cells[cell_index];
          warmup_sequence(shape, cell.predecessor, cell.intervention, cell.n,
                          &device, &intervention_device, options.warmup_ms);
          std::vector<SequenceTiming> timings;
          timings.reserve(options.samples);
          for (uint32_t sample = 0U; sample < options.samples; ++sample) {
            const std::string region = "phase87_wud2_" +
                                       std::string(shape.name) + "_" +
                                       mode_name(cell.predecessor) + "_" +
                                       intervention_name(cell.intervention) +
                                       "_n" + std::to_string(cell.n);
            (void)roctxRangePushA(region.c_str());
            (void)roctxProfilerResume(0);
            // Advance by the whole preceding sequence so the first dispatch
            // of this sample does not reuse the previous sample's last slot
            // when n>1 (the four-copy pool is still rotated continuously).
            const uint32_t weight_offset = (sample * cell.n) % kPoolCopies;
            const SequenceTiming timing = measure_sequence(
                shape, cell.predecessor, cell.intervention, cell.n, &device,
                &intervention_device, weight_offset);
            (void)roctxProfilerPause(0);
            (void)roctxRangePop();
            timings.push_back(timing);
            const std::vector<uint16_t> observed =
                download_output(shape, &device);
            const std::string output_label =
                mode_name(cell.predecessor) + "_" +
                intervention_name(cell.intervention) + "_n" +
                std::to_string(cell.n);
            all_ok =
                check_outputs(shape, control, observed, output_label.c_str()) &&
                all_ok;
            all_ok =
                check_finite_output(shape, observed, output_label.c_str()) &&
                all_ok;
            std::printf("sample shape=%s predecessor=%s intervention=%s "
                        "order=%s round=%u sample=%u n=%u nvfp4_us=",
                        shape.name, mode_name(cell.predecessor).c_str(),
                        intervention_name(cell.intervention).c_str(),
                        reverse ? "BA" : "AB", round, sample, cell.n);
            for (size_t index = 0U; index < timing.dispatch_us.size(); ++index)
              std::printf("%s%.3f", index == 0U ? "" : ",",
                          timing.dispatch_us[index]);
            std::printf(" sum_us=%.3f wall_us=%.3f\n", timing.sum_us,
                        timing.wall_us);
          }
          const Summary summary = summarize(timings, cell.n);
          std::printf("median shape=%s predecessor=%s intervention=%s "
                      "order=%s round=%u n=%u nvfp4_us=",
                      shape.name, mode_name(cell.predecessor).c_str(),
                      intervention_name(cell.intervention).c_str(),
                      reverse ? "BA" : "AB", round, cell.n);
          for (size_t index = 0U; index < summary.dispatch_medians.size();
               ++index)
            std::printf("%s%.3f", index == 0U ? "" : ",",
                        summary.dispatch_medians[index]);
          std::printf(" sum_us=%.3f wall_us=%.3f samples=%u\n",
                      summary.sum_median, summary.wall_median, options.samples);
        }
      }
    }
  }
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace

int main(const int argc, char **const argv) {
  try {
    Wud2Options options;
    parse_options(argc, argv, &options);
    return run_wud2(options);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "phase87_wud2_persistence_probe: %s\n", error.what());
    return EXIT_FAILURE;
  }
}
