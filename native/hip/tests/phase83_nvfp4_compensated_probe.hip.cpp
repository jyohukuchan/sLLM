// Reuse the Phase78 encoded-fixture, allocation, and timing helpers. The
// candidate below is the actual production translation unit's kernel, not a
// second implementation. The included helper retains its provenance header.
#define main phase78_tile_probe_unused_main
#include "phase78_nvfp4_gfx1030_dp4a_tile_probe.hip.cpp"
#undef main

extern "C" __global__ void sllm_nvfp4_w4a4_prefill_compensated64x64_v1(
    const uint8_t *, const uint8_t *, const uint8_t *, const uint8_t *,
    const float *, const float *, uint16_t *, uint64_t, uint64_t, uint64_t);

namespace {
uint32_t mix(uint32_t x) {
  x ^= x >> 16U;
  x *= 0x7feb352dU;
  x ^= x >> 15U;
  x *= 0x846ca68bU;
  return x ^ (x >> 16U);
}

long double reference(const HostInputs &h, uint64_t row, uint64_t col) {
  long double sum = 0;
  for (uint64_t j = 0; j < h.k; ++j) {
    const auto a = h.activation[row * h.k / 2 + j / 2];
    const auto w = h.weight[col * h.k / 2 + j / 2];
    const auto ac = (j & 1) ? a >> 4 : a & 15;
    const auto wc = (j & 1) ? w >> 4 : w & 15;
    sum += static_cast<long double>(host_e2m1(ac)) * host_e2m1(wc) *
           host_e4m3(h.activation_scales[row * (h.k / 16) + j / 16]) *
           host_e4m3(h.weight_scales[col * (h.k / 16) + j / 16]);
  }
  return sum * 0.75L * 1.125L;
}

bool candidate(const HostInputs &h, Buffers *b) {
  hipLaunchKernelGGL(
      sllm_nvfp4_w4a4_prefill_compensated64x64_v1,
      dim3(static_cast<uint32_t>(((h.m + 63) / 64) * ((h.n + 63) / 64))),
      dim3(256), 0, b->stream, b->activation, b->activation_scales, b->weight,
      b->weight_scales, b->weight_tensor_scale, b->input_tensor_scale,
      b->output, h.m, h.k, h.n);
  return hip_ok(hipGetLastError(), "compensated launch");
}

bool check(uint64_t m, uint64_t k, uint64_t n, uint32_t seed) {
  HostInputs h = make_inputs(m, k, n);
  for (size_t i = 0; i < h.activation.size(); ++i)
    h.activation[i] = static_cast<uint8_t>(mix(i + seed));
  for (size_t i = 0; i < h.weight.size(); ++i)
    h.weight[i] = static_cast<uint8_t>(mix(i + seed + 7919));
  for (size_t i = 0; i < h.activation_scales.size(); ++i)
    h.activation_scales[i] = 1 + mix(i + seed) % 126;
  for (size_t i = 0; i < h.weight_scales.size(); ++i)
    h.weight_scales[i] = 1 + mix(i + seed + 104729) % 126;
  Buffers b;
  bool ok = make_buffers(h, &b) && upload(h, &b) && candidate(h, &b) &&
            hip_ok(hipStreamSynchronize(b.stream), "candidate sync");
  std::vector<uint16_t> first(m * n), again(m * n);
  if (ok)
    ok = hip_ok(hipMemcpy(first.data(), b.output, first.size() * 2,
                          hipMemcpyDeviceToHost),
                "candidate read");
  if (ok)
    ok = candidate(h, &b) &&
         hip_ok(hipStreamSynchronize(b.stream), "repeat sync") &&
         hip_ok(hipMemcpy(again.data(), b.output, again.size() * 2,
                          hipMemcpyDeviceToHost),
                "repeat read");
  uint32_t max_ulp = 0;
  uint64_t samples = 0, nonfinite = 0;
  if (ok) {
    ok = first == again;
    for (auto v : first)
      nonfinite += (v & 0x7f80U) == 0x7f80U;
    const bool exhaustive = m * n <= 8192;
    for (uint64_t i = 0; i < (exhaustive ? m * n : 32); ++i) {
      const uint64_t pos =
          exhaustive ? i : (i == 31 ? m * n - 1 : mix(i + seed) % (m * n));
      const auto expected =
          host_bf16_rne(static_cast<float>(reference(h, pos / n, pos % n)));
      max_ulp = std::max(max_ulp, ulp_distance(first[pos], expected));
      ++samples;
    }
    ok = ok && nonfinite == 0 && max_ulp <= 1;
  }
  float control_us = 0;
  if (ok)
    ok = measure_tile<64, 64, 32>(h, &b, &control_us);
  std::array<float, 3> timings{};
  if (ok) {
    for (auto &t : timings) {
      ok = candidate(h, &b) && hip_ok(hipStreamSynchronize(b.stream), "warmup");
      if (!ok)
        break;
      ok = hip_ok(hipEventRecord(b.start, b.stream), "start") &&
           candidate(h, &b) &&
           hip_ok(hipEventRecord(b.stop, b.stream), "stop") &&
           hip_ok(hipEventSynchronize(b.stop), "event sync") &&
           hip_ok(hipEventElapsedTime(&t, b.start, b.stop), "elapsed");
      if (!ok)
        break;
    }
  }
  std::sort(timings.begin(), timings.end());
  cleanup(&b);
  ok = hip_ok(hipDeviceSynchronize(), "cleanup sync") && ok;
  std::printf(
      "m=%llu k=%llu n=%llu seed=%u oracle_samples=%llu max_ulp=%u "
      "nonfinite=%llu repeat=%d control_ms=%.6f candidate_ms=%.6f state=%s\n",
      (unsigned long long)m, (unsigned long long)k, (unsigned long long)n, seed,
      (unsigned long long)samples, max_ulp, (unsigned long long)nonfinite,
      first == again, control_us / 1000.0F, timings[1], ok ? "PASS" : "FAIL");
  std::fflush(stdout);
  return ok;
}
} // namespace

int main() {
  hipDeviceProp_t p{};
  if (!hip_ok(hipSetDevice(0), "device") ||
      !hip_ok(hipGetDeviceProperties(&p, 0), "properties"))
    return 1;
  if (std::string_view(p.gcnArchName) != "gfx1030" &&
      std::string_view(p.gcnArchName) != "gfx1201")
    return 1;
  std::printf(
      "target=%s pci=%04x:%02x:%02x oracle=host-long-double-encoded-dot\n",
      p.gcnArchName, p.pciDomainID, p.pciBusID, p.pciDeviceID);
  bool ok = true;
  for (const auto shape : {std::array<uint64_t, 3>{63, 48, 37},
                           {64, 48, 37},
                           {65, 48, 37},
                           {33, 80, 65},
                           {17, 5120, 17408},
                           {17, 17408, 5120},
                           {1024, 5120, 17408},
                           {1024, 17408, 5120}})
    for (uint32_t seed : {0U, 7919U})
      ok = check(shape[0], shape[1], shape[2], seed) && ok;
  return ok ? 0 : 1;
}
