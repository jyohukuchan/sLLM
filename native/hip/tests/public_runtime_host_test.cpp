#include "evidence_abi.h"
#include "low_precision_matmul_provider.hpp"
#include "matmul_kernel_internal.hpp"
#include "public_runtime_internal.hpp"
#include "rmsnorm_kernel_internal.hpp"

#include "sllm/hip.h"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern "C" std::size_t sllm_test_orphan_count() noexcept;
extern "C" std::size_t sllm_test_poison_count() noexcept;
extern "C" void sllm_test_rmsnorm_execute_throw_after_reservation(
    uint32_t occurrences) noexcept;
extern "C" void sllm_test_rmsnorm_execute_throw_after_registration(
    uint32_t occurrences) noexcept;
extern "C" uint32_t sllm_test_select_causal_attention_gqa4(
    uint64_t expected_kv_length, uint32_t query_count, uint32_t query_heads,
    uint32_t kv_heads, uint32_t head_dim, uint32_t encoding,
    const char *arch_name) noexcept;
extern "C" uint32_t sllm_test_select_causal_attention_providers(
    uint64_t expected_kv_length, uint32_t query_count, uint32_t query_heads,
    uint32_t kv_heads, uint32_t head_dim, uint32_t encoding,
    const char *arch_name) noexcept;
extern "C" uint32_t sllm_test_select_causal_attention_providers_with_semantics(
    uint64_t expected_kv_length, uint32_t query_count, uint32_t query_heads,
    uint32_t kv_heads, uint32_t head_dim, uint32_t encoding,
    uint64_t sliding_window, uint32_t explicit_score_scale,
    const char *arch_name) noexcept;
extern "C" uint32_t sllm_test_select_linear_attention_gfx942_wave64_column(
    uint64_t token_count, uint32_t qk_heads, uint32_t value_heads,
    uint32_t head_dim, const char *arch_name) noexcept;
extern "C" void
sllm_test_deepseek_v4_moe_route_device_status(int32_t status) noexcept;
extern "C" void
sllm_test_minimax_m3_moe_route_device_status(int32_t status) noexcept;
extern "C" uint32_t sllm_test_matmul_prepared_kernel_id(
    const sllm_matmul_plan_t *raw_plan) noexcept;
extern "C" uint32_t
sllm_test_matmul_prepared_provider_semantics(const sllm_matmul_plan_t *raw_plan,
                                             uint32_t *provider, uint32_t *tile,
                                             uint32_t *inner_product) noexcept;

namespace {

struct Error final {
  char message[256]{};
  sllm_error_sink_t sink{sizeof(sllm_error_sink_t),
                         SLLM_HIP_ABI_VERSION,
                         message,
                         sizeof(message),
                         0U,
                         {0U, 0U}};
};

bool expect_status(const sllm_status_t actual, const sllm_status_t expected,
                   const char *const operation, const Error &error) {
  if (actual == expected) {
    return true;
  }
  std::cerr << operation << " returned " << actual << ", expected " << expected
            << ": " << error.message << '\n';
  return false;
}

bool create_context_for_arch(const char *const arch,
                             sllm_context_t **const context) {
  sllm_context_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.device_index = 0U;
  std::strncpy(info.expected_gcn_arch_name, arch,
               sizeof(info.expected_gcn_arch_name) - 1U);
  Error error;
  return expect_status(sllm_context_create(&info, context, &error.sink),
                       SLLM_STATUS_OK, "sllm_context_create", error);
}

bool create_context(sllm_context_t **const context) {
  return create_context_for_arch("gfx1201", context);
}

bool create_queue(const sllm_context_t *const context,
                  sllm_queue_t **const queue) {
  sllm_queue_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  Error error;
  return expect_status(sllm_queue_create(context, &info, queue, &error.sink),
                       SLLM_STATUS_OK, "sllm_queue_create", error);
}

bool linear_attention_gfx942_wave64_column_selector_contract() {
  constexpr const char *const opt_in_name =
      "SLLM_LINEAR_ATTENTION_GFX942_WAVE64_COLUMN_STATE";
  constexpr const char *const force_name = "SLLM_GDN_FORCE_BASELINE";
  const char *const old_opt_in = std::getenv(opt_in_name);
  const char *const old_force = std::getenv(force_name);
  const bool had_opt_in = old_opt_in != nullptr;
  const bool had_force = old_force != nullptr;
  const std::string old_opt_in_value = had_opt_in ? old_opt_in : "";
  const std::string old_force_value = had_force ? old_force : "";
  const auto restore = [&]() {
    if (had_opt_in) {
      setenv(opt_in_name, old_opt_in_value.c_str(), 1);
    } else {
      unsetenv(opt_in_name);
    }
    if (had_force) {
      setenv(force_name, old_force_value.c_str(), 1);
    } else {
      unsetenv(force_name);
    }
  };
  const auto select = [](const uint64_t tokens, const uint32_t qk_heads,
                         const uint32_t value_heads, const uint32_t head_dim,
                         const char *const target) {
    return sllm_test_select_linear_attention_gfx942_wave64_column(
        tokens, qk_heads, value_heads, head_dim, target);
  };
  unsetenv(force_name);
  unsetenv(opt_in_name);
  bool valid = select(128U, 16U, 32U, 128U, "gfx942:sramecc+:xnack-") == 0U;
  setenv(opt_in_name, "1", 1);
  valid = valid &&
          select(127U, 16U, 32U, 128U, "gfx942:sramecc+:xnack-") == 0U &&
          select(128U, 16U, 32U, 128U, "gfx942:sramecc+:xnack-") == 1U &&
          select(129U, 16U, 32U, 128U, "gfx942:sramecc+:xnack-") == 1U;
  constexpr const char *const rejected_targets[] = {"gfx942",
                                                    "gfx942:sramecc-:xnack-",
                                                    "gfx942:sramecc+:xnack+",
                                                    "gfx1030",
                                                    "gfx1201",
                                                    "unknown"};
  for (const char *const target : rejected_targets) {
    valid = valid && select(128U, 16U, 32U, 128U, target) == 0U;
  }
  valid = valid &&
          select(128U, 8U, 32U, 128U, "gfx942:sramecc+:xnack-") == 0U &&
          select(128U, 16U, 16U, 128U, "gfx942:sramecc+:xnack-") == 0U &&
          select(128U, 16U, 32U, 64U, "gfx942:sramecc+:xnack-") == 0U &&
          select(128U, 16U, 32U, 256U, "gfx942:sramecc+:xnack-") == 0U;
  setenv(opt_in_name, "0", 1);
  valid = valid && select(128U, 16U, 32U, 128U, "gfx942:sramecc+:xnack-") == 0U;
  setenv(opt_in_name, "unexpected", 1);
  valid = valid && select(128U, 16U, 32U, 128U, "gfx942:sramecc+:xnack-") == 0U;
  setenv(opt_in_name, "1", 1);
  setenv(force_name, "1", 1);
  valid = valid && select(128U, 16U, 32U, 128U, "gfx942:sramecc+:xnack-") == 0U;
  restore();
  return valid;
}

bool causal_attention_gqa4_p32_selector_contract() {
  constexpr const char *const p16_name =
      "SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_GQA4_SPLIT";
  constexpr const char *const p32_name =
      "SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_GQA4_SPLIT_P32";
  constexpr const char *const gfx1201_p32_name =
      "SLLM_CAUSAL_ATTENTION_GFX1201_DECODE_GQA4_SPLIT_P32";
  constexpr const char *const force_name =
      "SLLM_CAUSAL_ATTENTION_FORCE_BASELINE";
  const char *const old_p16 = std::getenv(p16_name);
  const char *const old_p32 = std::getenv(p32_name);
  const char *const old_gfx1201_p32 = std::getenv(gfx1201_p32_name);
  const char *const old_force = std::getenv(force_name);
  const bool had_p16 = old_p16 != nullptr;
  const bool had_p32 = old_p32 != nullptr;
  const bool had_gfx1201_p32 = old_gfx1201_p32 != nullptr;
  const bool had_force = old_force != nullptr;
  const std::string old_p16_value = had_p16 ? old_p16 : "";
  const std::string old_p32_value = had_p32 ? old_p32 : "";
  const std::string old_gfx1201_p32_value =
      had_gfx1201_p32 ? old_gfx1201_p32 : "";
  const std::string old_force_value = had_force ? old_force : "";
  const auto restore_environment = [&]() {
    if (had_p16) {
      setenv(p16_name, old_p16_value.c_str(), 1);
    } else {
      unsetenv(p16_name);
    }
    if (had_p32) {
      setenv(p32_name, old_p32_value.c_str(), 1);
    } else {
      unsetenv(p32_name);
    }
    if (had_gfx1201_p32) {
      setenv(gfx1201_p32_name, old_gfx1201_p32_value.c_str(), 1);
    } else {
      unsetenv(gfx1201_p32_name);
    }
    if (had_force) {
      setenv(force_name, old_force_value.c_str(), 1);
    } else {
      unsetenv(force_name);
    }
  };
  const auto select =
      [](const uint64_t expected_kv_length, const uint32_t query_count = 1U,
         const uint32_t query_heads = 16U, const uint32_t kv_heads = 4U,
         const uint32_t head_dim = 256U,
         const char *const arch_name = "gfx1030") {
        return sllm_test_select_causal_attention_gqa4(
            expected_kv_length, query_count, query_heads, kv_heads, head_dim,
            SLLM_HIP_KV_ENCODING_FP16_V1, arch_name);
      };

  unsetenv(p16_name);
  unsetenv(p32_name);
  unsetenv(gfx1201_p32_name);
  unsetenv(force_name);
  bool valid = select(4095U) == 0U && select(4096U) == 2U &&
               select(4097U) == 2U &&
               select(4096U, 1U, 16U, 4U, 256U, "gfx1201") == 2U;

  setenv(p32_name, "0", 1);
  valid = valid && select(4096U) == 0U;
  setenv(p32_name, "unknown", 1);
  valid = valid && select(4096U) == 0U;
  setenv(p32_name, "1", 1);
  valid = valid && select(4096U) == 2U;
  unsetenv(p32_name);
  setenv(p16_name, "1", 1);
  valid = valid && select(4096U) == 2U;
  setenv(p32_name, "0", 1);
  valid = valid && select(4096U) == 1U;
  setenv(force_name, "1", 1);
  valid = valid && select(4096U) == 0U;

  unsetenv(force_name);
  unsetenv(p16_name);
  unsetenv(p32_name);
  setenv(gfx1201_p32_name, "0", 1);
  valid = valid && select(4095U, 1U, 16U, 4U, 256U, "gfx1201") == 0U &&
          select(4096U, 1U, 16U, 4U, 256U, "gfx1201") == 0U;
  setenv(gfx1201_p32_name, "unknown", 1);
  valid = valid && select(4096U, 1U, 16U, 4U, 256U, "gfx1201") == 0U;
  setenv(gfx1201_p32_name, "1", 1);
  valid = valid && select(4095U, 1U, 16U, 4U, 256U, "gfx1201") == 0U &&
          select(4096U, 1U, 16U, 4U, 256U, "gfx1201") == 2U &&
          select(4097U, 1U, 16U, 4U, 256U, "gfx1201") == 2U;
  setenv(force_name, "1", 1);
  valid = valid && select(4096U, 1U, 16U, 4U, 256U, "gfx1201") == 0U;
  unsetenv(force_name);
  valid = valid && select(4096U, 2U, 16U, 4U, 256U, "gfx1201") == 0U &&
          select(4096U, 1U, 8U, 4U, 256U, "gfx1201") == 0U &&
          select(4096U, 1U, 16U, 8U, 256U, "gfx1201") == 0U &&
          select(4096U, 1U, 16U, 4U, 128U, "gfx1201") == 0U &&
          select(4096U, 1U, 16U, 4U, 256U, "gfx942") == 0U &&
          select(4096U, 1U, 16U, 4U, 256U, "gfx9999") == 0U;
  unsetenv(gfx1201_p32_name);
  valid = valid && select(4096U, 1U, 16U, 4U, 256U, "gfx1201") == 2U &&
          select(4096U, 1U, 16U, 4U, 256U, "gfx942") == 0U &&
          select(4096U, 2U) == 0U && select(4096U, 1U, 8U) == 0U &&
          select(4096U, 1U, 16U, 8U) == 0U &&
          select(4096U, 1U, 16U, 4U, 128U) == 0U;
  restore_environment();
  return valid;
}

bool causal_attention_target_scoped_selector_contract() {
  constexpr uint32_t kGfx1201Wave = 1U << 0U;
  constexpr uint32_t kDecodeWaveSplit = 1U << 1U;
  constexpr uint32_t kDecodeWaveQPreload = 1U << 5U;
  constexpr uint32_t kDecodeGqa4SplitP32 = 1U << 4U;
  constexpr uint32_t kPrefillGqa4 = 1U << 6U;
  constexpr uint32_t kPrefillGqa4QTile4 = 1U << 7U;
  constexpr uint32_t kTypedQ4K4 = 1U << 10U;
  constexpr uint32_t kTypedQ4K8 = 2U << 10U;
  constexpr uint32_t kTypedQ8K8 = 3U << 10U;
  constexpr const char *const kForceBaseline =
      "SLLM_CAUSAL_ATTENTION_FORCE_BASELINE";
  constexpr const char *const kGfx1201Gqa4SplitP32 =
      "SLLM_CAUSAL_ATTENTION_GFX1201_DECODE_GQA4_SPLIT_P32";
  constexpr const char *const kPhase66TiledPrefill =
      "SLLM_CAUSAL_ATTENTION_PHASE66_TILED_PREFILL";
  constexpr std::array<const char *const, 10> kCandidateVariables = {
      "SLLM_CAUSAL_ATTENTION_GFX1030_Q_PRELOAD",
      "SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_WAVE_SHORT",
      "SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_WAVE_SHORT_Q_PRELOAD",
      "SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_WAVE_FP16_PAIR",
      "SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_GQA4_SPLIT",
      "SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_GQA4_SPLIT_P32",
      "SLLM_CAUSAL_ATTENTION_GFX1030_SCALED_PREFILL_GEMM",
      "SLLM_CAUSAL_ATTENTION_GFX1030_LONG_PREFILL_V2",
      kPhase66TiledPrefill,
      kForceBaseline};
  std::array<bool, kCandidateVariables.size()> had_old_value{};
  std::array<std::string, kCandidateVariables.size()> old_values{};
  for (std::size_t index = 0U; index < kCandidateVariables.size(); ++index) {
    const char *const old_value = std::getenv(kCandidateVariables[index]);
    had_old_value[index] = old_value != nullptr;
    if (old_value != nullptr) {
      old_values[index] = old_value;
    }
    unsetenv(kCandidateVariables[index]);
  }
  const char *const old_gfx1201_p32 = std::getenv(kGfx1201Gqa4SplitP32);
  const bool had_old_gfx1201_p32 = old_gfx1201_p32 != nullptr;
  const std::string old_gfx1201_p32_value =
      had_old_gfx1201_p32 ? old_gfx1201_p32 : "";
  unsetenv(kGfx1201Gqa4SplitP32);
  const auto restore_environment = [&]() {
    for (std::size_t index = 0U; index < kCandidateVariables.size(); ++index) {
      if (had_old_value[index]) {
        setenv(kCandidateVariables[index], old_values[index].c_str(), 1);
      } else {
        unsetenv(kCandidateVariables[index]);
      }
    }
    if (had_old_gfx1201_p32) {
      setenv(kGfx1201Gqa4SplitP32, old_gfx1201_p32_value.c_str(), 1);
    } else {
      unsetenv(kGfx1201Gqa4SplitP32);
    }
  };
  const auto clear_environment = [&]() {
    for (const char *const variable : kCandidateVariables) {
      unsetenv(variable);
    }
    unsetenv(kGfx1201Gqa4SplitP32);
  };
  const auto select = [](const uint64_t expected_kv_length,
                         const uint32_t query_count, const uint32_t query_heads,
                         const uint32_t kv_heads, const uint32_t head_dim,
                         const uint32_t encoding, const char *const arch_name) {
    return sllm_test_select_causal_attention_providers(
        expected_kv_length, query_count, query_heads, kv_heads, head_dim,
        encoding, arch_name);
  };
  const auto select_with_semantics = [](const uint64_t sliding_window,
                                        const bool explicit_score_scale,
                                        const uint32_t encoding) {
    return sllm_test_select_causal_attention_providers_with_semantics(
        2048U, 128U, 16U, 4U, 256U, encoding, sliding_window,
        explicit_score_scale ? 1U : 0U, "gfx1201");
  };
  const auto expect_gfx942_zero =
      [&](const uint64_t expected_kv_length, const uint32_t query_count = 1U,
          const uint32_t query_heads = 16U, const uint32_t kv_heads = 4U,
          const uint32_t head_dim = 256U,
          const uint32_t encoding = SLLM_HIP_KV_ENCODING_FP16_V1) {
        const uint32_t actual =
            select(expected_kv_length, query_count, query_heads, kv_heads,
                   head_dim, encoding, "gfx942");
        if (actual != 0U) {
          std::cerr << "gfx942 selector mask " << actual << " at kv "
                    << expected_kv_length << ", q " << query_count
                    << ", qheads " << query_heads << ", kvheads " << kv_heads
                    << ", dim " << head_dim << ", encoding " << encoding
                    << '\n';
        }
        return actual == 0U;
      };
  const auto expect_gfx1201 =
      [&](const uint64_t expected_kv_length, const uint32_t query_count,
          const uint32_t expected_mask, const uint32_t query_heads = 16U,
          const uint32_t kv_heads = 4U, const uint32_t head_dim = 256U,
          const uint32_t encoding = SLLM_HIP_KV_ENCODING_FP16_V1) {
        const uint32_t actual =
            select(expected_kv_length, query_count, query_heads, kv_heads,
                   head_dim, encoding, "gfx1201");
        if (actual != expected_mask) {
          std::cerr << "gfx1201 selector mask " << actual << ", expected "
                    << expected_mask << " at kv " << expected_kv_length
                    << ", q " << query_count << ", qheads " << query_heads
                    << ", kvheads " << kv_heads << ", dim " << head_dim
                    << ", encoding " << encoding << '\n';
        }
        return actual == expected_mask;
      };

  bool valid = true;
  // Exercise every selector boundary with all target-gated candidates unset.
  for (const uint64_t expected_kv_length :
       {31U, 32U, 33U, 1023U, 1024U, 1025U, 4095U, 4096U, 4097U, 10001U}) {
    for (const uint32_t query_count : {1U, 2U, 32U, 64U, 128U, 1024U}) {
      valid = valid && expect_gfx942_zero(expected_kv_length, query_count);
    }
  }
  valid = valid && expect_gfx942_zero(4096U, 1U, 8U, 4U, 256U) &&
          expect_gfx942_zero(4096U, 1U, 16U, 8U, 256U) &&
          expect_gfx942_zero(4096U, 1U, 16U, 4U, 128U) &&
          expect_gfx942_zero(4096U, 1U, 16U, 4U, 256U,
                             SLLM_HIP_KV_ENCODING_FP8_V1) &&
          select(4096U, 1U, 16U, 4U, 256U, SLLM_HIP_KV_ENCODING_FP16_V1,
                 "gfx9999") == 0U;

  // gfx1201 keeps the wave/prefill routes and now defaults the exact-shape
  // GQA4 P32 candidate on.  The 4095/4096 boundary distinguishes it.
  valid =
      valid && expect_gfx1201(4095U, 1U, kGfx1201Wave | kDecodeWaveSplit) &&
      expect_gfx1201(4096U, 1U,
                     kGfx1201Wave | kDecodeWaveSplit | kDecodeGqa4SplitP32) &&
      expect_gfx1201(4097U, 1U,
                     kGfx1201Wave | kDecodeWaveSplit | kDecodeGqa4SplitP32) &&
      expect_gfx1201(4096U, 2U, 0U) &&
      expect_gfx1201(4096U, 32U, kGfx1201Wave) &&
      expect_gfx1201(4096U, 64U, kGfx1201Wave | kPrefillGqa4) &&
      expect_gfx1201(4096U, 128U,
                     kGfx1201Wave | kPrefillGqa4 | kPrefillGqa4QTile4) &&
      expect_gfx1201(4096U, 1U, kGfx1201Wave | kDecodeWaveSplit, 16U, 4U, 256U,
                     SLLM_HIP_KV_ENCODING_FP8_V1) &&
      expect_gfx1201(4096U, 1U, kGfx1201Wave | kDecodeWaveSplit, 8U, 4U,
                     256U) &&
      expect_gfx1201(4096U, 1U, kGfx1201Wave, 16U, 4U, 128U) &&
      // Explicit OCP MXFP8 uses the packed-KV generic routes.  It must not
      // select any FP16-only GQA4/specialized provider.
      expect_gfx1201(4096U, 1U, kGfx1201Wave | kDecodeWaveSplit, 16U, 4U, 256U,
                     SLLM_HIP_KV_ENCODING_MXFP8_E4_V1) &&
      select(4096U, 1U, 16U, 4U, 256U, SLLM_HIP_KV_ENCODING_MXFP8_E5_V1,
             "gfx1030") == (kDecodeWaveSplit | kDecodeWaveQPreload);

  // Every environment spelling must remain inert for gfx942.  Test each
  // candidate independently and then all candidates together.
  constexpr std::array<const char *const, 4> kEnvironmentValues = {
      "1", "0", "unknown", nullptr};
  for (const char *const variable : kCandidateVariables) {
    for (const char *const value : kEnvironmentValues) {
      clear_environment();
      if (value != nullptr) {
        setenv(variable, value, 1);
      }
      valid = valid && expect_gfx942_zero(4095U) && expect_gfx942_zero(4096U) &&
              expect_gfx942_zero(4097U) && expect_gfx942_zero(1024U, 1024U) &&
              expect_gfx942_zero(10001U, 128U, 16U, 4U, 256U,
                                 SLLM_HIP_KV_ENCODING_FP8_V1);
    }
  }
  // The same environment matrix must not expose gfx1030-only candidates on
  // gfx1201. Phase66 is the one model-independent gfx1201 opt-in in this set;
  // FORCE_BASELINE removes both it and the qtile4 control.
  for (const char *const variable : kCandidateVariables) {
    for (const char *const value : kEnvironmentValues) {
      clear_environment();
      if (value != nullptr) {
        setenv(variable, value, 1);
      }
      const bool force_baseline = std::strcmp(variable, kForceBaseline) == 0 &&
                                  value != nullptr &&
                                  std::strcmp(value, "1") == 0;
      const bool phase66 = std::strcmp(variable, kPhase66TiledPrefill) == 0 &&
                           value != nullptr && std::strcmp(value, "1") == 0;
      valid = valid &&
              expect_gfx1201(4095U, 1U, kGfx1201Wave | kDecodeWaveSplit) &&
              expect_gfx1201(4096U, 2U, 0U) &&
              expect_gfx1201(4096U, 64U,
                             kGfx1201Wave | kPrefillGqa4 |
                                 (phase66 ? kPrefillGqa4QTile4 : 0U)) &&
              expect_gfx1201(
                  4096U, 128U,
                  kGfx1201Wave | kPrefillGqa4 |
                      (force_baseline
                           ? 0U
                           : kPrefillGqa4QTile4 | (phase66 ? kTypedQ8K8 : 0U)));
    }
  }
  clear_environment();
  for (const char *const variable : kCandidateVariables) {
    setenv(variable, "1", 1);
  }
  valid = valid && expect_gfx942_zero(4095U) && expect_gfx942_zero(4096U) &&
          expect_gfx942_zero(4097U) && expect_gfx942_zero(1024U, 1024U) &&
          expect_gfx942_zero(10001U, 128U);

  // FORCE_BASELINE suppresses baseline-gated candidates but cannot make
  // gfx942 select one; gfx1201 keeps its existing common prefill route.
  clear_environment();
  setenv(kForceBaseline, "1", 1);
  valid = valid && expect_gfx942_zero(4096U) &&
          expect_gfx1201(4096U, 1U, kGfx1201Wave | kDecodeWaveSplit) &&
          expect_gfx1201(4096U, 64U, kGfx1201Wave | kPrefillGqa4) &&
          expect_gfx1201(4096U, 128U, kGfx1201Wave | kPrefillGqa4);

  // Phase66 chooses tiles from typed query/context boundaries only. Each
  // rejected target, encoding or shape falls back to the existing q4k1 path.
  clear_environment();
  setenv(kPhase66TiledPrefill, "1", 1);
  valid = valid &&
          expect_gfx1201(127U, 127U,
                         kGfx1201Wave | kPrefillGqa4 | kPrefillGqa4QTile4) &&
          expect_gfx1201(127U, 128U,
                         kGfx1201Wave | kPrefillGqa4 | kPrefillGqa4QTile4) &&
          expect_gfx1201(128U, 128U,
                         kGfx1201Wave | kPrefillGqa4 | kPrefillGqa4QTile4 |
                             kTypedQ4K4) &&
          expect_gfx1201(511U, 128U,
                         kGfx1201Wave | kPrefillGqa4 | kPrefillGqa4QTile4 |
                             kTypedQ4K4) &&
          expect_gfx1201(512U, 128U,
                         kGfx1201Wave | kPrefillGqa4 | kPrefillGqa4QTile4 |
                             kTypedQ4K8) &&
          expect_gfx1201(513U, 128U,
                         kGfx1201Wave | kPrefillGqa4 | kPrefillGqa4QTile4 |
                             kTypedQ4K8) &&
          expect_gfx1201(2047U, 128U,
                         kGfx1201Wave | kPrefillGqa4 | kPrefillGqa4QTile4 |
                             kTypedQ4K8) &&
          expect_gfx1201(2048U, 128U,
                         kGfx1201Wave | kPrefillGqa4 | kPrefillGqa4QTile4 |
                             kTypedQ8K8) &&
          expect_gfx1201(2049U, 128U,
                         kGfx1201Wave | kPrefillGqa4 | kPrefillGqa4QTile4 |
                             kTypedQ8K8) &&
          expect_gfx1201(2048U, 128U,
                         kGfx1201Wave | kPrefillGqa4 | kPrefillGqa4QTile4, 16U,
                         4U, 256U, SLLM_HIP_KV_ENCODING_MXFP8_E5_V1) &&
          expect_gfx1201(2048U, 128U,
                         kGfx1201Wave | kPrefillGqa4 | kPrefillGqa4QTile4 |
                             kTypedQ8K8,
                         16U, 4U, 256U, SLLM_HIP_KV_ENCODING_MXFP8_E4_V1) &&
          select(2048U, 128U, 16U, 4U, 256U, SLLM_HIP_KV_ENCODING_MXFP8_E4_V1,
                 "gfx1030") == (kPrefillGqa4 | kPrefillGqa4QTile4) &&
          expect_gfx1201(2048U, 128U, kGfx1201Wave, 8U, 4U, 256U) &&
          expect_gfx1201(2048U, 128U, kGfx1201Wave, 16U, 8U, 256U) &&
          expect_gfx1201(2048U, 128U, kGfx1201Wave, 16U, 4U, 128U);

  // The typed candidate implements full, implicitly-scaled causal attention
  // only. Sliding-window and explicit-score-scale semantics must keep the
  // typed policy bits clear for every accepted candidate KV encoding.
  constexpr uint32_t kTypedPolicyMask = 3U << 10U;
  for (const uint32_t encoding :
       {SLLM_HIP_KV_ENCODING_FP16_V1, SLLM_HIP_KV_ENCODING_MXFP8_E4_V1}) {
    const uint32_t sliding = select_with_semantics(1024U, false, encoding);
    const uint32_t explicitly_scaled =
        select_with_semantics(0U, true, encoding);
    if ((sliding & kTypedPolicyMask) != 0U ||
        (explicitly_scaled & kTypedPolicyMask) != 0U) {
      std::cerr << "Phase66 typed prefill selected unsupported attention "
                   "semantics for encoding "
                << encoding << ": sliding mask " << sliding
                << ", explicit-scale mask " << explicitly_scaled << '\n';
      valid = false;
    }
  }
  setenv(kForceBaseline, "1", 1);
  valid = valid && expect_gfx1201(2048U, 128U, kGfx1201Wave | kPrefillGqa4);

  restore_environment();
  return valid;
}

bool create_buffer(const sllm_context_t *const context,
                   sllm_buffer_t **const buffer) {
  sllm_buffer_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.size_bytes = 64U;
  Error error;
  return expect_status(sllm_buffer_create(context, &info, buffer, &error.sink),
                       SLLM_STATUS_OK, "sllm_buffer_create", error);
}

bool create_buffer_sized(const sllm_context_t *const context,
                         const uint64_t size_bytes,
                         sllm_buffer_t **const buffer) {
  sllm_buffer_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.size_bytes = size_bytes;
  Error error;
  return expect_status(sllm_buffer_create(context, &info, buffer, &error.sink),
                       SLLM_STATUS_OK, "sllm_buffer_create", error);
}

bool mlp_gate_up_silu_bundle_abi_negative_contract() {
  sllm_mlp_gate_up_silu_bundle_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_MLP_GATE_UP_SILU_BUNDLE_VERSION;
  Error error;
  sllm_mlp_gate_up_silu_bundle_plan_t *plan = nullptr;
  if (!expect_status(sllm_mlp_gate_up_silu_bundle_prepare(nullptr, &descriptor,
                                                          &plan, &error.sink),
                     SLLM_STATUS_INVALID_ARGUMENT, "MLP bundle null context",
                     error) ||
      plan != nullptr) {
    return false;
  }
  descriptor.struct_size = sizeof(descriptor) - 1U;
  if (!expect_status(sllm_mlp_gate_up_silu_bundle_prepare(
                         reinterpret_cast<const sllm_context_t *>(1),
                         &descriptor, &plan, &error.sink),
                     SLLM_STATUS_INVALID_ABI_VERSION,
                     "MLP bundle malformed descriptor", error) ||
      plan != nullptr) {
    return false;
  }
  return true;
}

bool release_context(sllm_context_t **const context,
                     const sllm_status_t expected = SLLM_STATUS_OK) {
  Error error;
  return expect_status(sllm_context_release(context, &error.sink), expected,
                       "sllm_context_release", error);
}

bool release_queue(sllm_queue_t **const queue,
                   const sllm_status_t expected = SLLM_STATUS_OK) {
  Error error;
  return expect_status(sllm_queue_release(queue, &error.sink), expected,
                       "sllm_queue_release", error);
}

bool release_buffer(sllm_buffer_t **const buffer,
                    const sllm_status_t expected = SLLM_STATUS_OK) {
  Error error;
  return expect_status(sllm_buffer_release(buffer, &error.sink), expected,
                       "sllm_buffer_release", error);
}

bool submit_h2d(const sllm_queue_t *const queue,
                const sllm_buffer_t *const buffer,
                sllm_completion_t **const completion) {
  uint8_t payload[17] = {};
  for (std::size_t index = 0U; index != sizeof(payload); ++index) {
    payload[index] = static_cast<uint8_t>(index + 1U);
  }
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = payload;
  transfer.size_bytes = sizeof(payload);
  Error error;
  return expect_status(
      sllm_buffer_copy_h2d(queue, buffer, &transfer, completion, &error.sink),
      SLLM_STATUS_OK, "sllm_buffer_copy_h2d", error);
}

bool submit_d2h(const sllm_queue_t *const queue,
                const sllm_buffer_t *const buffer, const std::size_t size,
                sllm_completion_t **const completion) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.size_bytes = size;
  Error error;
  return expect_status(
      sllm_buffer_copy_d2h(queue, buffer, &transfer, completion, &error.sink),
      SLLM_STATUS_OK, "sllm_buffer_copy_d2h", error);
}

bool query_completion(sllm_completion_t *const completion,
                      const sllm_status_t expected) {
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  Error error;
  return expect_status(sllm_completion_query(completion, &result, &error.sink),
                       expected, "sllm_completion_query", error);
}

bool release_completion(sllm_completion_t **const completion,
                        const sllm_status_t expected = SLLM_STATUS_OK) {
  Error error;
  return expect_status(sllm_completion_release(completion, &error.sink),
                       expected, "sllm_completion_release", error);
}

bool read_completion(sllm_completion_t *const completion,
                     void *const destination, const std::size_t capacity,
                     const uint8_t *const expected,
                     const std::size_t expected_size) {
  uint64_t bytes_written = 0U;
  Error error;
  const sllm_status_t status = sllm_completion_read(
      completion, destination, capacity, &bytes_written, &error.sink);
  if (!expect_status(status, SLLM_STATUS_OK, "sllm_completion_read", error) ||
      bytes_written != expected_size ||
      std::memcmp(destination, expected, expected_size) != 0) {
    std::cerr << "D2H completion read was not byte exact\n";
    return false;
  }
  return true;
}

bool bounded_counter_cas_contention_is_fail_closed() {
  using SafetyState = sllm_public_runtime::CompletionSafetyState;
  using Injector = sllm_public_runtime::FaultInjector;

  SafetyState::reset_quarantine_cas_failures();
  SafetyState::force_quarantine_cas_failures(1U);
  SafetyState::force_quarantine_counter_cas_contention(true);
  if (!SafetyState::consume_forced_quarantine_cas_failure_for_test()) {
    std::cerr << "quarantine counter CAS exhaustion did not fail closed\n";
    return false;
  }
  SafetyState::force_quarantine_counter_cas_contention(false);
  if (!SafetyState::consume_forced_quarantine_cas_failure_for_test() ||
      SafetyState::consume_forced_quarantine_cas_failure_for_test()) {
    std::cerr
        << "quarantine counter did not recover after bounded contention\n";
    return false;
  }
  SafetyState::reset_quarantine_cas_failures();

  Injector::reset();
  Injector::set(sllm_public_runtime::FaultPoint::AccountingFailure, 1U);
  Injector::force_cas_contention(true);
  if (!Injector::consume(sllm_public_runtime::FaultPoint::AccountingFailure)) {
    std::cerr << "fault injector CAS exhaustion did not fail closed\n";
    return false;
  }
  Injector::force_cas_contention(false);
  if (!Injector::consume(sllm_public_runtime::FaultPoint::AccountingFailure) ||
      Injector::consume(sllm_public_runtime::FaultPoint::AccountingFailure)) {
    std::cerr
        << "fault injector counter did not recover after bounded contention\n";
    return false;
  }
  Injector::reset();
  return true;
}

bool completion_safety_quarantine_is_bounded_and_fail_closed() {
  using SafetyState = sllm_public_runtime::CompletionSafetyState;

  SafetyState exhausted;
  exhausted.observe_positive_completion();
  if (!exhausted.can_release_graph()) {
    std::cerr << "positive completion must initially be releasable\n";
    return false;
  }
  SafetyState::force_quarantine_cas_failures(
      static_cast<uint32_t>(SafetyState::quarantine_cas_attempt_bound() + 1U));
  exhausted.quarantine();
  SafetyState::reset_quarantine_cas_failures();
  if (exhausted.can_release_graph() || exhausted.event_destroyed() ||
      exhausted.observe_event_destroy_success()) {
    std::cerr << "bounded quarantine CAS exhaustion was not fail closed\n";
    return false;
  }
  exhausted.quarantine();
  if (exhausted.can_release_graph()) {
    std::cerr << "repeat quarantine re-enabled release\n";
    return false;
  }

  SafetyState destroyed;
  destroyed.observe_positive_completion();
  if (!destroyed.observe_event_destroy_success() ||
      !destroyed.event_destroyed()) {
    std::cerr << "positive completion could not reach EventDestroyed\n";
    return false;
  }
  destroyed.quarantine();
  if (!destroyed.event_destroyed() || destroyed.can_release_graph()) {
    std::cerr << "quarantine overwrote EventDestroyed or enabled release\n";
    return false;
  }

  SafetyState concurrent;
  concurrent.observe_positive_completion();
  std::thread quarantine_thread([&concurrent]() { concurrent.quarantine(); });
  std::thread destroy_thread(
      [&concurrent]() { (void)concurrent.observe_event_destroy_success(); });
  quarantine_thread.join();
  destroy_thread.join();
  if (concurrent.can_release_graph()) {
    std::cerr << "concurrent safety transition enabled release\n";
    return false;
  }
  return true;
}

bool successful_completion_lifecycle() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *buffer = nullptr;
  sllm_event_t *event = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer(context, &buffer)) {
    return false;
  }
  Error event_error;
  if (!expect_status(sllm_event_create(context, &event, &event_error.sink),
                     SLLM_STATUS_OK, "sllm_event_create", event_error)) {
    return false;
  }
  if (!expect_status(sllm_event_release(&event, &event_error.sink),
                     SLLM_STATUS_OK, "sllm_event_release", event_error)) {
    return false;
  }

  sllm_completion_t *completion = nullptr;
  if (!submit_h2d(queue, buffer, &completion) || completion == nullptr) {
    return false;
  }
  sllm_public_runtime::FaultInjector::set(
      sllm_public_runtime::FaultPoint::CompletionQueryPending, 1U);
  if (!query_completion(completion, SLLM_STATUS_PUBLIC_PENDING) ||
      !query_completion(completion, SLLM_STATUS_OK) ||
      !release_completion(&completion) || completion != nullptr ||
      !release_queue(&queue) || !release_buffer(&buffer) ||
      !release_context(&context)) {
    return false;
  }
  if (fake_hip::live_events() != 0U || fake_hip::live_streams() != 0U ||
      fake_hip::live_allocations() != 0U) {
    std::cerr << "successful lifecycle left fake HIP resources live\n";
    return false;
  }
  return true;
}

bool d2h_staging_and_completion_read_is_byte_exact() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *buffer = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer(context, &buffer)) {
    return false;
  }
  constexpr std::size_t transfer_size = 37U;
  uint8_t expected[transfer_size] = {};
  for (std::size_t index = 0U; index != transfer_size; ++index) {
    expected[index] = static_cast<uint8_t>((index * 13U) ^ 0x5AU);
  }
  sllm_transfer_desc_t h2d{};
  h2d.struct_size = sizeof(h2d);
  h2d.abi_version = SLLM_HIP_ABI_VERSION;
  h2d.host_pointer = expected;
  h2d.size_bytes = transfer_size;
  sllm_completion_t *h2d_completion = nullptr;
  Error error;
  if (!expect_status(sllm_buffer_copy_h2d(queue, buffer, &h2d, &h2d_completion,
                                          &error.sink),
                     SLLM_STATUS_OK, "D2H setup H2D", error) ||
      !query_completion(h2d_completion, SLLM_STATUS_OK) ||
      !release_completion(&h2d_completion)) {
    return false;
  }

  sllm_completion_t *d2h_completion = nullptr;
  if (!submit_d2h(queue, buffer, transfer_size, &d2h_completion) ||
      !query_completion(d2h_completion, SLLM_STATUS_OK)) {
    return false;
  }
  uint8_t actual[transfer_size] = {};
  if (!read_completion(d2h_completion, actual, sizeof(actual), expected,
                       transfer_size) ||
      !release_completion(&d2h_completion) || !release_queue(&queue) ||
      !release_buffer(&buffer) || !release_context(&context)) {
    return false;
  }
  return fake_hip::live_events() == 0U && fake_hip::live_streams() == 0U &&
         fake_hip::live_allocations() == 0U;
}

bool positive_completion_with_deferred_event_destroy_retains_dependencies() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *buffer = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer(context, &buffer)) {
    return false;
  }
  sllm_completion_t *completion = nullptr;
  if (!submit_h2d(queue, buffer, &completion) ||
      !query_completion(completion, SLLM_STATUS_OK)) {
    return false;
  }
  const std::size_t poison_before = sllm_test_poison_count();
  const std::size_t destroy_before = fake_hip::event_destroy_calls();
  sllm_public_runtime::FaultInjector::set(
      sllm_public_runtime::FaultPoint::EventDestroyError, 1U);
  if (!release_completion(&completion, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR) ||
      completion != nullptr || sllm_test_poison_count() != poison_before + 1U ||
      fake_hip::event_destroy_calls() != destroy_before ||
      fake_hip::live_events() == 0U) {
    std::cerr << "positive completion did not retain deferred event cleanup"
              << " poison=" << sllm_test_poison_count()
              << " expected_poison=" << poison_before + 1U
              << " destroy=" << fake_hip::event_destroy_calls()
              << " expected_destroy=" << destroy_before
              << " live_events=" << fake_hip::live_events() << '\n';
    return false;
  }
  Error queue_error;
  Error buffer_error;
  Error context_error;
  return expect_status(sllm_queue_release(&queue, &queue_error.sink),
                       SLLM_STATUS_INTERNAL_ERROR,
                       "deferred-destroy queue retention", queue_error) &&
         expect_status(sllm_buffer_release(&buffer, &buffer_error.sink),
                       SLLM_STATUS_INTERNAL_ERROR,
                       "deferred-destroy buffer retention", buffer_error) &&
         expect_status(sllm_context_release(&context, &context_error.sink),
                       SLLM_STATUS_INTERNAL_ERROR,
                       "deferred-destroy context retention", context_error);
}

bool concurrent_pin_and_release() {
  sllm_public_runtime::FaultInjector::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *buffer = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer(context, &buffer)) {
    return false;
  }
  sllm_completion_t *completion = nullptr;
  if (!submit_h2d(queue, buffer, &completion)) {
    return false;
  }
  fake_hip::set_event_query_gate(true);
  sllm_status_t query_status = SLLM_STATUS_INTERNAL_ERROR;
  std::thread query_thread([&]() {
    sllm_completion_result_t result{};
    result.struct_size = sizeof(result);
    result.abi_version = SLLM_HIP_ABI_VERSION;
    Error error;
    query_status = sllm_completion_query(completion, &result, &error.sink);
  });
  fake_hip::wait_event_query_entered();
  Error release_error;
  const sllm_status_t release_status =
      sllm_completion_release(&completion, &release_error.sink);
  if (!expect_status(release_status, SLLM_STATUS_PUBLIC_BUSY,
                     "concurrent sllm_completion_release", release_error)) {
    fake_hip::release_event_query_gate();
    query_thread.join();
    return false;
  }
  fake_hip::release_event_query_gate();
  query_thread.join();
  if (query_status != SLLM_STATUS_OK || !release_completion(&completion) ||
      !release_queue(&queue) || !release_buffer(&buffer) ||
      !release_context(&context)) {
    return false;
  }
  return true;
}

bool fatal_completion_is_quarantined() {
  sllm_public_runtime::FaultInjector::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *buffer = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer(context, &buffer)) {
    return false;
  }
  sllm_completion_t *completion = nullptr;
  if (!submit_h2d(queue, buffer, &completion)) {
    return false;
  }
  const std::size_t destroy_before = fake_hip::event_destroy_calls();
  sllm_public_runtime::FaultInjector::set(
      sllm_public_runtime::FaultPoint::CompletionQueryFatal, 1U);
  if (!query_completion(completion, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR) ||
      !release_completion(&completion, SLLM_STATUS_PUBLIC_INVALID_HANDLE) ||
      completion == nullptr ||
      fake_hip::event_destroy_calls() != destroy_before ||
      fake_hip::live_events() == 0U) {
    return false;
  }
  Error queue_error;
  Error buffer_error;
  Error context_error;
  if (!expect_status(sllm_queue_release(&queue, &queue_error.sink),
                     SLLM_STATUS_INTERNAL_ERROR, "fatal queue release",
                     queue_error) ||
      !expect_status(sllm_buffer_release(&buffer, &buffer_error.sink),
                     SLLM_STATUS_INTERNAL_ERROR, "fatal buffer release",
                     buffer_error) ||
      !expect_status(sllm_context_release(&context, &context_error.sink),
                     SLLM_STATUS_INTERNAL_ERROR, "fatal context release",
                     context_error)) {
    return false;
  }
  return true;
}

bool registry_failure_destroys_or_orphans_before_rollback() {
  sllm_public_runtime::FaultInjector::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *buffer = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer(context, &buffer)) {
    return false;
  }
  const std::size_t orphan_before = sllm_test_orphan_count();
  const std::size_t event_destroy_before = fake_hip::event_destroy_calls();
  sllm_public_runtime::FaultInjector::set(
      sllm_public_runtime::FaultPoint::RegistryInsertionFailure, 1U);
  sllm_public_runtime::FaultInjector::set(
      sllm_public_runtime::FaultPoint::EventDestroyError, 1U);
  fake_hip::set_event_create_gate(true);
  sllm_completion_t *completion = nullptr;
  sllm_status_t submit_status = SLLM_STATUS_OK;
  std::thread submit_thread([&]() {
    uint8_t payload[17] = {};
    sllm_transfer_desc_t transfer{};
    transfer.struct_size = sizeof(transfer);
    transfer.abi_version = SLLM_HIP_ABI_VERSION;
    transfer.host_pointer = payload;
    transfer.size_bytes = sizeof(payload);
    Error error;
    submit_status = sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                         &error.sink);
  });
  fake_hip::wait_event_create_entered();
  Error queue_error;
  Error buffer_error;
  Error context_error;
  const bool concurrent_releases =
      expect_status(sllm_queue_release(&queue, &queue_error.sink),
                    SLLM_STATUS_PUBLIC_BUSY, "concurrent queue release",
                    queue_error) &&
      expect_status(sllm_buffer_release(&buffer, &buffer_error.sink),
                    SLLM_STATUS_PUBLIC_BUSY, "concurrent buffer release",
                    buffer_error) &&
      expect_status(sllm_context_release(&context, &context_error.sink),
                    SLLM_STATUS_PUBLIC_BUSY, "concurrent context release",
                    context_error);
  fake_hip::release_event_create_gate();
  submit_thread.join();
  if (!concurrent_releases ||
      submit_status != SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR ||
      completion != nullptr || sllm_test_orphan_count() != orphan_before + 1U ||
      fake_hip::event_destroy_calls() != event_destroy_before ||
      fake_hip::live_events() == 0U) {
    std::cerr << "registry zero-token rollback did not retain exactly one "
                 "ambiguous event\n";
    return false;
  }
  return true;
}

bool registry_exception_reaches_real_catch_before_rollback() {
  sllm_public_runtime::FaultInjector::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *buffer = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer(context, &buffer)) {
    return false;
  }
  const std::size_t orphan_before = sllm_test_orphan_count();
  const std::size_t event_destroy_before = fake_hip::event_destroy_calls();
  sllm_public_runtime::FaultInjector::set(
      sllm_public_runtime::FaultPoint::RegistryInsertionException, 1U);
  sllm_public_runtime::FaultInjector::set(
      sllm_public_runtime::FaultPoint::EventDestroyError, 1U);
  fake_hip::set_event_create_gate(true);
  sllm_completion_t *completion = nullptr;
  sllm_status_t submit_status = SLLM_STATUS_OK;
  std::thread submit_thread([&]() {
    uint8_t payload[17] = {};
    sllm_transfer_desc_t transfer{};
    transfer.struct_size = sizeof(transfer);
    transfer.abi_version = SLLM_HIP_ABI_VERSION;
    transfer.host_pointer = payload;
    transfer.size_bytes = sizeof(payload);
    Error error;
    submit_status = sllm_buffer_copy_h2d(queue, buffer, &transfer, &completion,
                                         &error.sink);
  });
  fake_hip::wait_event_create_entered();
  Error queue_error;
  Error buffer_error;
  Error context_error;
  const bool concurrent_releases =
      expect_status(sllm_queue_release(&queue, &queue_error.sink),
                    SLLM_STATUS_PUBLIC_BUSY,
                    "exception concurrent queue release", queue_error) &&
      expect_status(sllm_buffer_release(&buffer, &buffer_error.sink),
                    SLLM_STATUS_PUBLIC_BUSY,
                    "exception concurrent buffer release", buffer_error) &&
      expect_status(sllm_context_release(&context, &context_error.sink),
                    SLLM_STATUS_PUBLIC_BUSY,
                    "exception concurrent context release", context_error);
  fake_hip::release_event_create_gate();
  submit_thread.join();
  if (!concurrent_releases ||
      submit_status != SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR ||
      completion != nullptr || sllm_test_orphan_count() != orphan_before + 1U ||
      fake_hip::event_destroy_calls() != event_destroy_before ||
      fake_hip::live_events() == 0U) {
    std::cerr << "registry exception did not use guarded event rollback\n";
    return false;
  }
  return true;
}

bool production_orphan_owner_grows_past_128() {
  sllm_public_runtime::FaultInjector::reset();
  const std::size_t before = sllm_test_orphan_count();
  for (std::size_t index = 0U; index != 129U; ++index) {
    sllm_context_t *context = nullptr;
    sllm_queue_t *queue = nullptr;
    if (!create_context(&context)) {
      return false;
    }
    sllm_public_runtime::FaultInjector::set(
        sllm_public_runtime::FaultPoint::ConstructionCandidateFailure, 1U);
    sllm_public_runtime::FaultInjector::set(
        sllm_public_runtime::FaultPoint::StreamDestroyError, 1U);
    sllm_queue_create_info_t info{};
    info.struct_size = sizeof(info);
    info.abi_version = SLLM_HIP_ABI_VERSION;
    Error error;
    const sllm_status_t status =
        sllm_queue_create(context, &info, &queue, &error.sink);
    if (!expect_status(status, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
                       "orphan-growth queue create", error) ||
        queue != nullptr) {
      return false;
    }
  }
  if (sllm_test_orphan_count() < before + 129U) {
    std::cerr << "production orphan owner did not grow beyond 128 records\n";
    return false;
  }
  return true;
}

sllm_tensor_binding_t rmsnorm_binding(const sllm_buffer_t *const buffer,
                                      const uint64_t offset,
                                      const uint64_t rows,
                                      const uint64_t columns) {
  sllm_tensor_binding_t binding{};
  binding.struct_size = sizeof(binding);
  binding.abi_version = SLLM_HIP_ABI_VERSION;
  binding.buffer = buffer;
  binding.byte_offset = offset;
  binding.dtype = SLLM_TENSOR_DTYPE_BF16;
  binding.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  binding.rank = rows == 1U ? 1U : 2U;
  if (binding.rank == 1U) {
    binding.shape[0] = columns;
    binding.stride_elements[0] = 1U;
  } else {
    binding.shape[0] = rows;
    binding.shape[1] = columns;
    binding.stride_elements[0] = columns;
    binding.stride_elements[1] = 1U;
  }
  return binding;
}

sllm_tensor_binding_t
rmsnorm_binding_rank(const sllm_buffer_t *const buffer, const uint64_t offset,
                     const uint32_t rank, const uint64_t columns,
                     const uint64_t *const shape = nullptr) {
  sllm_tensor_binding_t binding{};
  binding.struct_size = sizeof(binding);
  binding.abi_version = SLLM_HIP_ABI_VERSION;
  binding.buffer = buffer;
  binding.byte_offset = offset;
  binding.dtype = SLLM_TENSOR_DTYPE_BF16;
  binding.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  binding.rank = rank;
  uint64_t stride = 1U;
  for (uint32_t backwards = 0U; backwards != rank; ++backwards) {
    const uint32_t index = rank - 1U - backwards;
    binding.shape[index] =
        shape == nullptr ? (index == rank - 1U ? columns : 1U) : shape[index];
    binding.stride_elements[index] = stride;
    stride *= binding.shape[index];
  }
  return binding;
}

bool rmsnorm_bf16_rne_bit_contract() {
  struct ConversionCase final {
    uint32_t input_bits;
    uint16_t expected_bits;
  };
  constexpr ConversionCase cases[] = {
      {UINT32_C(0x00000000), UINT16_C(0x0000)},
      {UINT32_C(0x80000000), UINT16_C(0x8000)},
      {UINT32_C(0x00008000), UINT16_C(0x0000)},
      {UINT32_C(0x80008000), UINT16_C(0x8000)},
      {UINT32_C(0x3f807fff), UINT16_C(0x3f80)},
      {UINT32_C(0x3f808000), UINT16_C(0x3f80)},
      {UINT32_C(0x3f808001), UINT16_C(0x3f81)},
      {UINT32_C(0x3f818000), UINT16_C(0x3f82)},
      {UINT32_C(0xbf808000), UINT16_C(0xbf80)},
      {UINT32_C(0xbf818000), UINT16_C(0xbf82)},
      {UINT32_C(0x7f800000), UINT16_C(0x7f80)},
      {UINT32_C(0xff800000), UINT16_C(0xff80)},
      {UINT32_C(0x7f800001), UINT16_C(0x7fc0)},
      {UINT32_C(0xff800001), UINT16_C(0xffc0)},
      {UINT32_C(0x7f900001), UINT16_C(0x7fd0)},
      {UINT32_C(0xffa12345), UINT16_C(0xffe1)},
      {UINT32_C(0x7fc12345), UINT16_C(0x7fc1)},
  };

  for (const ConversionCase &test_case : cases) {
    float value = 0.0F;
    std::memcpy(&value, &test_case.input_bits, sizeof(value));
    const uint16_t actual = sllm_rmsnorm_kernel::float_to_bf16_rne_bits(value);
    if (actual != test_case.expected_bits) {
      std::cerr << "BF16 conversion contract mismatch\n";
      return false;
    }
  }

  constexpr uint32_t nan_bits[] = {
      UINT32_C(0x7f800001), UINT32_C(0x7f900001), UINT32_C(0x7fc12345),
      UINT32_C(0xff800001), UINT32_C(0xffa12345), UINT32_C(0xffc12345),
  };
  for (const uint32_t input_bits : nan_bits) {
    float value = 0.0F;
    std::memcpy(&value, &input_bits, sizeof(value));
    const uint16_t actual = sllm_rmsnorm_kernel::float_to_bf16_rne_bits(value);
    if ((actual & UINT16_C(0x7f80)) != UINT16_C(0x7f80) ||
        (actual & UINT16_C(0x0040)) == 0U ||
        (actual & UINT16_C(0x007f)) == 0U) {
      std::cerr << "BF16 NaN was not a quiet nonzero NaN\n";
      return false;
    }
  }
  return true;
}

sllm_rmsnorm_desc_t rmsnorm_descriptor(
    const sllm_buffer_t *const activation, const uint64_t activation_offset,
    const sllm_buffer_t *const scale, const uint64_t scale_offset,
    const sllm_buffer_t *const output, const uint64_t output_offset,
    const uint64_t rows = 2U, const uint64_t columns = 3U) {
  sllm_rmsnorm_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_RMSNORM_VERSION;
  descriptor.accumulation_dtype = SLLM_RMSNORM_ACCUMULATION_F32;
  descriptor.scale_mode = SLLM_RMSNORM_SCALE_MODE_OFFSET_ONE;
  descriptor.alias_policy = SLLM_RMSNORM_ALIAS_POLICY_REJECT_OVERLAP;
  float epsilon = 1.0e-6F;
  std::memcpy(&descriptor.epsilon_bits, &epsilon, sizeof(epsilon));
  descriptor.activation =
      rmsnorm_binding(activation, activation_offset, rows, columns);
  descriptor.raw_scale = rmsnorm_binding(scale, scale_offset, 1U, columns);
  descriptor.output = rmsnorm_binding(output, output_offset, rows, columns);
  return descriptor;
}

sllm_elementwise_desc_t
elementwise_descriptor(const sllm_elementwise_operation_t operation,
                       const sllm_buffer_t *const input0,
                       const sllm_buffer_t *const input1,
                       const sllm_buffer_t *const output, const uint64_t size) {
  sllm_elementwise_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_ELEMENTWISE_VERSION;
  descriptor.operation = operation;
  const auto binding = [operation](const sllm_buffer_t *const buffer,
                                   const uint64_t logical_size) {
    auto result = rmsnorm_binding(buffer, 0U, 1U, logical_size);
    if (operation == SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL) {
      result.rank = 3U;
      result.shape[0] = logical_size;
      result.shape[1] = 16U;
      result.shape[2] = 256U;
      result.stride_elements[0] = 4096U;
      result.stride_elements[1] = 256U;
      result.stride_elements[2] = 1U;
    } else if (operation == SLLM_ELEMENTWISE_OPERATION_BROADCAST_ADD ||
               operation == SLLM_ELEMENTWISE_OPERATION_BROADCAST_MUL) {
      result.rank = 2U;
      result.shape[0] = 1U;
      result.shape[1] = logical_size;
      result.stride_elements[0] = logical_size;
      result.stride_elements[1] = 1U;
    }
    return result;
  };
  descriptor.input0 = binding(input0, size);
  if (operation != SLLM_ELEMENTWISE_OPERATION_COPY) {
    descriptor.input1 =
        operation == SLLM_ELEMENTWISE_OPERATION_SCALAR_MUL ||
                operation == SLLM_ELEMENTWISE_OPERATION_TANH_SOFTCAP
            ? rmsnorm_binding(input1, 0U, 1U, 1U)
        : operation == SLLM_ELEMENTWISE_OPERATION_BROADCAST_ADD ||
                operation == SLLM_ELEMENTWISE_OPERATION_BROADCAST_MUL
            ? rmsnorm_binding(input1, 0U, 1U, size)
            : binding(input1, size);
  }
  descriptor.output = binding(output, size);
  return descriptor;
}

sllm_elementwise_desc_t
broadcast_elementwise_descriptor(const sllm_buffer_t *const input,
                                 const sllm_buffer_t *const vector,
                                 const sllm_buffer_t *const output,
                                 const uint64_t rows, const uint64_t columns,
                                 const sllm_elementwise_operation_t operation =
                                     SLLM_ELEMENTWISE_OPERATION_BROADCAST_ADD) {
  sllm_elementwise_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_ELEMENTWISE_VERSION;
  descriptor.operation = operation;
  descriptor.input0 = rmsnorm_binding(input, 0U, rows, columns);
  descriptor.input1 = rmsnorm_binding(vector, 0U, 1U, columns);
  descriptor.output = rmsnorm_binding(output, 0U, rows, columns);
  return descriptor;
}

sllm_elementwise_dispatch_info_t elementwise_dispatch_info() {
  sllm_elementwise_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_ELEMENTWISE_DISPATCH_INFO_VERSION;
  return info;
}

bool elementwise_prepare_execute_and_negative_contract() {
  fake_hip::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *input0 = nullptr;
  sllm_buffer_t *input1 = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer_sized(context, 32768U, &input0) ||
      !create_buffer_sized(context, 32768U, &input1) ||
      !create_buffer_sized(context, 32768U, &output)) {
    return false;
  }
  Error error;
  const auto run = [&](const sllm_elementwise_operation_t operation,
                       const uint64_t size, const uint32_t kernel_id,
                       const char *const symbol) {
    const uint64_t elements =
        operation == SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL ? size * 4096U
                                                            : size;
    auto descriptor =
        elementwise_descriptor(operation, input0, input1, output, size);
    sllm_elementwise_plan_t *plan = nullptr;
    if (!expect_status(
            sllm_elementwise_prepare(context, &descriptor, &plan, &error.sink),
            SLLM_STATUS_OK, "elementwise prepare", error) ||
        plan == nullptr) {
      return false;
    }
    sllm_completion_t *completion = nullptr;
    auto info = elementwise_dispatch_info();
    const bool executed =
        expect_status(sllm_elementwise_execute(plan, queue, &completion, &info,
                                               &error.sink),
                      SLLM_STATUS_OK, "elementwise execute", error) &&
        completion != nullptr && info.operation == operation &&
        info.dispatch_count == 1U && info.kernel_id == kernel_id &&
        info.workgroup_size_x == SLLM_HIP_ELEMENTWISE_WORKGROUP_SIZE &&
        info.grid_size_x ==
            (elements + SLLM_HIP_ELEMENTWISE_WORKGROUP_SIZE - 1U) /
                SLLM_HIP_ELEMENTWISE_WORKGROUP_SIZE &&
        info.element_count == elements && info.fallback_allowed == 0U &&
        info.fallback_used == 0U &&
        std::strcmp(info.kernel_symbol, symbol) == 0 &&
        query_completion(completion, SLLM_STATUS_OK) &&
        release_completion(&completion) &&
        expect_status(sllm_elementwise_plan_release(&plan, &error.sink),
                      SLLM_STATUS_OK, "elementwise plan release", error);
    return executed && plan == nullptr && completion == nullptr;
  };
  const auto upload_words = [&](const sllm_buffer_t *const buffer,
                                const std::vector<uint16_t> &words) {
    sllm_transfer_desc_t transfer{};
    transfer.struct_size = sizeof(transfer);
    transfer.abi_version = SLLM_HIP_ABI_VERSION;
    transfer.host_pointer = const_cast<uint16_t *>(words.data());
    transfer.size_bytes = words.size() * sizeof(uint16_t);
    sllm_completion_t *completion = nullptr;
    return expect_status(sllm_buffer_copy_h2d(queue, buffer, &transfer,
                                              &completion, &error.sink),
                         SLLM_STATUS_OK, "elementwise test upload", error) &&
           query_completion(completion, SLLM_STATUS_OK) &&
           release_completion(&completion);
  };
  std::vector<uint16_t> sigmoid_gate(3U * 16U * 256U, UINT16_C(0x4000));
  std::vector<uint16_t> attention_value(sigmoid_gate.size(), UINT16_C(0x3f80));
  if (!upload_words(input0, sigmoid_gate) ||
      !upload_words(input1, attention_value)) {
    return false;
  }
  if (!run(SLLM_ELEMENTWISE_OPERATION_COPY, 257U,
           SLLM_HIP_ELEMENTWISE_KERNEL_ID_COPY_V1,
           "elementwise.copy.bf16.v1") ||
      !run(SLLM_ELEMENTWISE_OPERATION_ADD, 17U,
           SLLM_HIP_ELEMENTWISE_KERNEL_ID_ADD_V1,
           "elementwise.add.bf16_fp32.v1") ||
      !run(SLLM_ELEMENTWISE_OPERATION_BROADCAST_ADD, 17U,
           SLLM_HIP_ELEMENTWISE_KERNEL_ID_BROADCAST_ADD_V1,
           "elementwise.broadcast_add.bf16_fp32.v1") ||
      !run(SLLM_ELEMENTWISE_OPERATION_BROADCAST_MUL, 17U,
           SLLM_HIP_ELEMENTWISE_KERNEL_ID_BROADCAST_MUL_V1,
           "elementwise.broadcast_mul.bf16_fp32.v1") ||
      !run(SLLM_ELEMENTWISE_OPERATION_SILU_MUL, 255U,
           SLLM_HIP_ELEMENTWISE_KERNEL_ID_SILU_MUL_V1,
           "elementwise.silu_mul.bf16_fp32.v1") ||
      !run(SLLM_ELEMENTWISE_OPERATION_SCALAR_MUL, 257U,
           SLLM_HIP_ELEMENTWISE_KERNEL_ID_SCALAR_MUL_V1,
           "elementwise.scalar_mul.bf16_fp32.v1") ||
      !run(SLLM_ELEMENTWISE_OPERATION_GELU_TANH_MUL, 17U,
           SLLM_HIP_ELEMENTWISE_KERNEL_ID_GELU_TANH_MUL_V1,
           "elementwise.gelu_tanh_mul.bf16_fp32.v1") ||
      !run(SLLM_ELEMENTWISE_OPERATION_TANH_SOFTCAP, 255U,
           SLLM_HIP_ELEMENTWISE_KERNEL_ID_TANH_SOFTCAP_V1,
           "elementwise.tanh_softcap.bf16_fp32.v1") ||
      !run(SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL, 3U,
           SLLM_HIP_ELEMENTWISE_KERNEL_ID_SIGMOID_MUL_V1,
           "elementwise.sigmoid_mul.bf16_fp32.v1") ||
      fake_hip::elementwise_copy_launch_calls() != 1U ||
      fake_hip::elementwise_add_launch_calls() != 1U ||
      fake_hip::elementwise_broadcast_add_launch_calls() != 1U ||
      fake_hip::elementwise_broadcast_mul_launch_calls() != 1U ||
      fake_hip::elementwise_silu_mul_launch_calls() != 1U ||
      fake_hip::elementwise_sigmoid_mul_launch_calls() != 1U ||
      fake_hip::elementwise_scalar_mul_launch_calls() != 1U ||
      fake_hip::elementwise_gelu_tanh_mul_launch_calls() != 1U ||
      fake_hip::elementwise_tanh_softcap_launch_calls() != 1U ||
      fake_hip::elementwise_last_element_count() != 3U * 16U * 256U) {
    return false;
  }

  sllm_completion_t *readback = nullptr;
  const std::size_t sigmoid_bytes = sigmoid_gate.size() * sizeof(uint16_t);
  if (!submit_d2h(queue, output, sigmoid_bytes, &readback) ||
      !query_completion(readback, SLLM_STATUS_OK)) {
    return false;
  }
  std::vector<uint16_t> sigmoid_output(sigmoid_gate.size());
  uint64_t bytes_written = 0U;
  const float sigmoid = 1.0F / (1.0F + std::exp(-2.0F));
  const uint16_t expected_sigmoid =
      sllm_rmsnorm_kernel::float_to_bf16_rne_bits(sigmoid);
  const uint16_t forbidden_silu =
      sllm_rmsnorm_kernel::float_to_bf16_rne_bits(2.0F * sigmoid);
  if (!expect_status(sllm_completion_read(readback, sigmoid_output.data(),
                                          sigmoid_bytes, &bytes_written,
                                          &error.sink),
                     SLLM_STATUS_OK, "sigmoid output read", error) ||
      bytes_written != sigmoid_bytes ||
      sigmoid_output.front() != expected_sigmoid ||
      sigmoid_output.front() == forbidden_silu ||
      !release_completion(&readback)) {
    return false;
  }

  auto broadcast_descriptor =
      broadcast_elementwise_descriptor(input0, input1, output, 3U, 17U);
  sllm_elementwise_plan_t *broadcast_plan = nullptr;
  if (!expect_status(sllm_elementwise_prepare(context, &broadcast_descriptor,
                                              &broadcast_plan, &error.sink),
                     SLLM_STATUS_OK, "broadcast add M=3 prepare", error) ||
      broadcast_plan == nullptr) {
    return false;
  }
  sllm_completion_t *broadcast_completion = nullptr;
  auto broadcast_info = elementwise_dispatch_info();
  if (!expect_status(sllm_elementwise_execute(broadcast_plan, queue,
                                              &broadcast_completion,
                                              &broadcast_info, &error.sink),
                     SLLM_STATUS_OK, "broadcast add M=3 execute", error) ||
      broadcast_completion == nullptr ||
      broadcast_info.operation != SLLM_ELEMENTWISE_OPERATION_BROADCAST_ADD ||
      broadcast_info.element_count != 3U * 17U ||
      broadcast_info.fallback_allowed != 0U ||
      broadcast_info.fallback_used != 0U ||
      !query_completion(broadcast_completion, SLLM_STATUS_OK)) {
    return false;
  }
  if (!release_completion(&broadcast_completion) ||
      !expect_status(
          sllm_elementwise_plan_release(&broadcast_plan, &error.sink),
          SLLM_STATUS_OK, "broadcast add M=3 release", error)) {
    return false;
  }
  if (fake_hip::elementwise_broadcast_add_launch_calls() != 2U) {
    return false;
  }
  sllm_completion_t *broadcast_readback = nullptr;
  const std::size_t broadcast_bytes = 3U * 17U * sizeof(uint16_t);
  if (!submit_d2h(queue, output, broadcast_bytes, &broadcast_readback) ||
      !query_completion(broadcast_readback, SLLM_STATUS_OK)) {
    return false;
  }
  std::vector<uint16_t> broadcast_output(3U * 17U);
  uint64_t broadcast_written = 0U;
  const uint16_t expected_broadcast = UINT16_C(0x4040);
  if (!expect_status(sllm_completion_read(
                         broadcast_readback, broadcast_output.data(),
                         broadcast_bytes, &broadcast_written, &error.sink),
                     SLLM_STATUS_OK, "broadcast add M=3 read", error) ||
      broadcast_written != broadcast_bytes ||
      std::any_of(broadcast_output.begin(), broadcast_output.end(),
                  [expected_broadcast](const uint16_t value) {
                    return value != expected_broadcast;
                  }) ||
      !release_completion(&broadcast_readback)) {
    return false;
  }

  auto broadcast_mul_descriptor = broadcast_elementwise_descriptor(
      input0, input1, output, 3U, 17U,
      SLLM_ELEMENTWISE_OPERATION_BROADCAST_MUL);
  sllm_elementwise_plan_t *broadcast_mul_plan = nullptr;
  if (!expect_status(sllm_elementwise_prepare(context,
                                              &broadcast_mul_descriptor,
                                              &broadcast_mul_plan, &error.sink),
                     SLLM_STATUS_OK, "broadcast mul M=3 prepare", error) ||
      broadcast_mul_plan == nullptr) {
    return false;
  }
  sllm_completion_t *broadcast_mul_completion = nullptr;
  auto broadcast_mul_info = elementwise_dispatch_info();
  if (!expect_status(sllm_elementwise_execute(broadcast_mul_plan, queue,
                                              &broadcast_mul_completion,
                                              &broadcast_mul_info, &error.sink),
                     SLLM_STATUS_OK, "broadcast mul M=3 execute", error) ||
      broadcast_mul_completion == nullptr ||
      broadcast_mul_info.operation !=
          SLLM_ELEMENTWISE_OPERATION_BROADCAST_MUL ||
      broadcast_mul_info.kernel_id !=
          SLLM_HIP_ELEMENTWISE_KERNEL_ID_BROADCAST_MUL_V1 ||
      broadcast_mul_info.element_count != 3U * 17U ||
      broadcast_mul_info.fallback_allowed != 0U ||
      broadcast_mul_info.fallback_used != 0U ||
      std::strcmp(broadcast_mul_info.kernel_symbol,
                  "elementwise.broadcast_mul.bf16_fp32.v1") != 0 ||
      std::strcmp(broadcast_mul_info.device_symbol,
                  "sllm_elementwise_broadcast_mul_bf16_fp32_v1") != 0 ||
      !query_completion(broadcast_mul_completion, SLLM_STATUS_OK) ||
      !release_completion(&broadcast_mul_completion) ||
      !expect_status(
          sllm_elementwise_plan_release(&broadcast_mul_plan, &error.sink),
          SLLM_STATUS_OK, "broadcast mul M=3 release", error) ||
      fake_hip::elementwise_broadcast_mul_launch_calls() != 2U) {
    return false;
  }
  sllm_completion_t *broadcast_mul_readback = nullptr;
  if (!submit_d2h(queue, output, broadcast_bytes, &broadcast_mul_readback) ||
      !query_completion(broadcast_mul_readback, SLLM_STATUS_OK)) {
    return false;
  }
  std::vector<uint16_t> broadcast_mul_output(3U * 17U);
  uint64_t broadcast_mul_written = 0U;
  const uint16_t expected_broadcast_mul = UINT16_C(0x4000);
  if (!expect_status(sllm_completion_read(
                         broadcast_mul_readback, broadcast_mul_output.data(),
                         broadcast_bytes, &broadcast_mul_written, &error.sink),
                     SLLM_STATUS_OK, "broadcast mul M=3 read", error) ||
      broadcast_mul_written != broadcast_bytes ||
      std::any_of(broadcast_mul_output.begin(), broadcast_mul_output.end(),
                  [expected_broadcast_mul](const uint16_t value) {
                    return value != expected_broadcast_mul;
                  }) ||
      !release_completion(&broadcast_mul_readback)) {
    return false;
  }

  broadcast_mul_descriptor.input1 = rmsnorm_binding(input1, 0U, 1U, 16U);
  sllm_elementwise_plan_t *plan = nullptr;
  if (!expect_status(sllm_elementwise_prepare(context,
                                              &broadcast_mul_descriptor, &plan,
                                              &error.sink),
                     SLLM_STATUS_SHAPE_MISMATCH,
                     "broadcast mul vector shape rejection", error) ||
      plan != nullptr) {
    return false;
  }

  auto descriptor = elementwise_descriptor(SLLM_ELEMENTWISE_OPERATION_COPY,
                                           input0, input1, output, 3U);
  descriptor.input0.dtype = SLLM_TENSOR_DTYPE_F32;
  if (!expect_status(
          sllm_elementwise_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_UNSUPPORTED_DTYPE, "elementwise dtype rejection",
          error) ||
      plan != nullptr) {
    return false;
  }
  descriptor = elementwise_descriptor(SLLM_ELEMENTWISE_OPERATION_SCALAR_MUL,
                                      input0, input1, output, 3U);
  descriptor.input1 = rmsnorm_binding(input1, 0U, 1U, 3U);
  if (!expect_status(
          sllm_elementwise_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_SHAPE_MISMATCH, "scalar multiplier shape rejection",
          error) ||
      plan != nullptr) {
    return false;
  }
  descriptor = elementwise_descriptor(SLLM_ELEMENTWISE_OPERATION_ADD, input0,
                                      input0, output, 3U);
  if (!expect_status(
          sllm_elementwise_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_ALIAS_OVERLAP, "elementwise alias rejection", error) ||
      plan != nullptr) {
    return false;
  }
  descriptor = elementwise_descriptor(SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL,
                                      input0, input1, input0, 3U);
  if (!expect_status(
          sllm_elementwise_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_ALIAS_OVERLAP, "sigmoid alias rejection", error) ||
      plan != nullptr) {
    return false;
  }
  descriptor = elementwise_descriptor(SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL,
                                      input0, input1, output, 3U);
  descriptor.input0.rank = 2U;
  descriptor.input0.shape[0] = 3U;
  descriptor.input0.shape[1] = 4096U;
  descriptor.input0.shape[2] = 0U;
  descriptor.input0.stride_elements[0] = 4096U;
  descriptor.input0.stride_elements[1] = 1U;
  descriptor.input0.stride_elements[2] = 0U;
  descriptor.input1 = descriptor.input0;
  descriptor.input1.buffer = input1;
  descriptor.output = descriptor.input0;
  descriptor.output.buffer = output;
  if (!expect_status(
          sllm_elementwise_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_SHAPE_MISMATCH, "sigmoid flat shape rejection", error) ||
      plan != nullptr) {
    return false;
  }
  descriptor = elementwise_descriptor(SLLM_ELEMENTWISE_OPERATION_SIGMOID_MUL,
                                      input0, input1, output, 3U);
  descriptor.input0.shape[1] = 4U;
  descriptor.input0.stride_elements[0] = 1024U;
  descriptor.input1 = descriptor.input0;
  descriptor.input1.buffer = input1;
  descriptor.output = descriptor.input0;
  descriptor.output.buffer = output;
  if (!expect_status(
          sllm_elementwise_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_SHAPE_MISMATCH, "sigmoid GQA head rejection", error) ||
      plan != nullptr) {
    return false;
  }
  descriptor = elementwise_descriptor(SLLM_ELEMENTWISE_OPERATION_COPY, input0,
                                      input1, output, 3U);
  descriptor.input1 = rmsnorm_binding(input1, 0U, 1U, 3U);
  if (!expect_status(
          sllm_elementwise_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_INVALID_ELEMENTWISE_DESCRIPTOR,
          "copy second-input rejection", error) ||
      plan != nullptr || !release_queue(&queue) || !release_buffer(&input0) ||
      !release_buffer(&input1) || !release_buffer(&output) ||
      !release_context(&context)) {
    return false;
  }
  return fake_hip::live_events() == 0U && fake_hip::live_streams() == 0U &&
         fake_hip::live_allocations() == 0U;
}

bool embedding_prepare_execute_and_token_range_contract() {
  fake_hip::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *weight = nullptr;
  sllm_buffer_t *token_ids = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer_sized(context, 24U, &weight) ||
      !create_buffer_sized(context, 12U, &token_ids) ||
      !create_buffer_sized(context, 18U, &output)) {
    return false;
  }
  uint16_t weight_words[12] = {0U, 1U, 2U, 3U, 4U,  5U,
                               6U, 7U, 8U, 9U, 10U, 11U};
  int32_t ids[3] = {2, 0, 2};
  const auto upload = [&](const sllm_buffer_t *const buffer, void *const bytes,
                          const uint64_t size) {
    sllm_transfer_desc_t transfer{};
    transfer.struct_size = sizeof(transfer);
    transfer.abi_version = SLLM_HIP_ABI_VERSION;
    transfer.host_pointer = bytes;
    transfer.size_bytes = size;
    sllm_completion_t *completion = nullptr;
    Error error;
    return expect_status(sllm_buffer_copy_h2d(queue, buffer, &transfer,
                                              &completion, &error.sink),
                         SLLM_STATUS_OK, "embedding input upload", error) &&
           query_completion(completion, SLLM_STATUS_OK) &&
           release_completion(&completion);
  };
  if (!upload(weight, weight_words, sizeof(weight_words)) ||
      !upload(token_ids, ids, sizeof(ids))) {
    return false;
  }
  sllm_embedding_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_EMBEDDING_VERSION;
  descriptor.weight = rmsnorm_binding(weight, 0U, 4U, 3U);
  descriptor.token_ids = rmsnorm_binding(token_ids, 0U, 1U, 3U);
  descriptor.token_ids.dtype = SLLM_TENSOR_DTYPE_I32;
  descriptor.token_ids.rank = 1U;
  descriptor.token_ids.shape[0] = 3U;
  descriptor.token_ids.shape[1] = 0U;
  descriptor.token_ids.stride_elements[0] = 1U;
  descriptor.token_ids.stride_elements[1] = 0U;
  descriptor.output = rmsnorm_binding(output, 0U, 3U, 3U);
  sllm_embedding_plan_t *plan = nullptr;
  Error error;
  if (!expect_status(
          sllm_embedding_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "embedding prepare", error) ||
      plan == nullptr) {
    return false;
  }
  sllm_embedding_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_EMBEDDING_DISPATCH_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  if (!expect_status(
          sllm_embedding_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_OK, "embedding execute", error) ||
      info.dispatch_count != 1U ||
      info.kernel_id != SLLM_HIP_EMBEDDING_KERNEL_ID_GATHER_V1 ||
      info.token_count != 3U || info.hidden_size != 3U ||
      info.vocab_size != 4U || info.fallback_allowed != 0U ||
      info.fallback_used != 0U ||
      std::strcmp(info.kernel_symbol, "embedding.gather.bf16_i32.v1") != 0 ||
      fake_hip::embedding_gather_launch_calls() != 1U ||
      !query_completion(completion, SLLM_STATUS_OK) ||
      !release_completion(&completion)) {
    return false;
  }
  ids[1] = -1;
  if (!upload(token_ids, ids, sizeof(ids)) ||
      !expect_status(
          sllm_embedding_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_TOKEN_ID_OUT_OF_RANGE,
          "embedding negative token rejection", error) ||
      completion != nullptr ||
      fake_hip::embedding_gather_launch_calls() != 1U ||
      !expect_status(sllm_embedding_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "embedding plan release", error) ||
      !release_queue(&queue) || !release_buffer(&weight) ||
      !release_buffer(&token_ids) || !release_buffer(&output) ||
      !release_context(&context)) {
    return false;
  }
  return true;
}

bool rmsnorm_prepare_lifecycle_and_negative_contract() {
  fake_hip::reset();
  sllm_context_t *context = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *scale = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_buffer(context, &activation) ||
      !create_buffer(context, &scale) || !create_buffer(context, &output)) {
    return false;
  }
  sllm_rmsnorm_desc_t descriptor =
      rmsnorm_descriptor(activation, 2U, scale, 4U, output, 6U);
  sllm_rmsnorm_plan_t *plan = nullptr;
  Error error;
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "rmsnorm prepare", error) ||
      plan == nullptr ||
      !release_buffer(&activation, SLLM_STATUS_PUBLIC_BUSY) ||
      !release_context(&context, SLLM_STATUS_PUBLIC_BUSY)) {
    return false;
  }
  if (!expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "rmsnorm plan release", error) ||
      plan != nullptr ||
      !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_INVALID_ARGUMENT, "rmsnorm double release",
                     error)) {
    return false;
  }
  if (!release_buffer(&activation) || !release_buffer(&scale) ||
      !release_buffer(&output) || !release_context(&context)) {
    return false;
  }

  if (!create_context(&context) || !create_buffer(context, &activation) ||
      !create_buffer(context, &scale) || !create_buffer(context, &output)) {
    return false;
  }
  descriptor = rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U);
  descriptor.activation.struct_size -= 1U;
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_INVALID_ARGUMENT, "rmsnorm binding size", error)) {
    return false;
  }
  descriptor = rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U);
  descriptor.epsilon_bits = 0U;
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_INVALID_EPSILON, "rmsnorm epsilon", error)) {
    return false;
  }
  descriptor = rmsnorm_descriptor(activation, 1U, scale, 0U, output, 0U);
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_MISALIGNED_OFFSET, "rmsnorm alignment", error)) {
    return false;
  }
  descriptor = rmsnorm_descriptor(activation, 60U, scale, 0U, output, 0U);
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_BUFFER_OUT_OF_BOUNDS, "rmsnorm bounds", error)) {
    return false;
  }
  descriptor = rmsnorm_descriptor(activation, 0U, activation, 2U, output, 0U);
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_ALIAS_OVERLAP, "rmsnorm activation-scale alias", error)) {
    return false;
  }
  descriptor = rmsnorm_descriptor(activation, 0U, scale, 0U, activation, 2U);
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_ALIAS_OVERLAP, "rmsnorm activation-output alias",
          error)) {
    return false;
  }
  descriptor = rmsnorm_descriptor(activation, 0U, scale, 0U, scale, 2U);
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_ALIAS_OVERLAP, "rmsnorm scale-output alias", error)) {
    return false;
  }
  descriptor =
      rmsnorm_descriptor(activation, 0U, activation, 16U, activation, 32U);
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "rmsnorm disjoint alias", error) ||
      !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "rmsnorm disjoint release", error)) {
    return false;
  }
  /* Half-open intervals that exactly touch are valid aliases.  The three
   * intervals below are [0,12), [12,18), and [18,30); none overlaps. */
  descriptor =
      rmsnorm_descriptor(activation, 0U, activation, 12U, activation, 18U);
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "rmsnorm touching alias", error) ||
      !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "rmsnorm touching release", error)) {
    return false;
  }
  return release_buffer(&activation) && release_buffer(&scale) &&
         release_buffer(&output) && release_context(&context);
}

bool rmsnorm_plan_accounting_failure_is_consumed_and_quarantined() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  sllm_context_t *context = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *scale = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_buffer(context, &activation) ||
      !create_buffer(context, &scale) || !create_buffer(context, &output)) {
    return false;
  }
  sllm_rmsnorm_desc_t descriptor =
      rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U);
  sllm_rmsnorm_plan_t *plan = nullptr;
  Error error;
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "fault plan prepare", error) ||
      plan == nullptr) {
    return false;
  }
  sllm_rmsnorm_plan_t *stale = plan;
  const std::size_t poison_before = sllm_test_poison_count();
  sllm_public_runtime::FaultInjector::set(
      sllm_public_runtime::FaultPoint::AccountingFailure, 1U);
  if (!expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_INTERNAL_ERROR, "fault plan release", error) ||
      plan != nullptr || sllm_test_poison_count() != poison_before + 1U) {
    std::cerr << "RMSNorm accounting failure did not consume and quarantine "
                 "the plan\n";
    return false;
  }

  if (!expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_INVALID_ARGUMENT, "null plan retry", error) ||
      !expect_status(sllm_rmsnorm_plan_release(&stale, &error.sink),
                     SLLM_STATUS_PUBLIC_INVALID_HANDLE, "stale plan token",
                     error)) {
    return false;
  }
  sllm_rmsnorm_plan_t *forged = reinterpret_cast<sllm_rmsnorm_plan_t *>(
      static_cast<uintptr_t>(0xfeedfaceU));
  if (!expect_status(sllm_rmsnorm_plan_release(&forged, &error.sink),
                     SLLM_STATUS_PUBLIC_INVALID_HANDLE, "forged plan token",
                     error)) {
    return false;
  }
  sllm_rmsnorm_plan_t *wrong_kind =
      reinterpret_cast<sllm_rmsnorm_plan_t *>(context);
  if (!expect_status(sllm_rmsnorm_plan_release(&wrong_kind, &error.sink),
                     SLLM_STATUS_PUBLIC_INVALID_HANDLE, "wrong-kind plan token",
                     error)) {
    return false;
  }

  /* The poison owner retains all three distinct Buffer dependencies and the
   * Context.  Their callers must see INTERNAL_ERROR, never a retryable BUSY
   * caused by the plan's permanently active release flag. */
  if (!expect_status(sllm_buffer_release(&activation, &error.sink),
                     SLLM_STATUS_INTERNAL_ERROR, "quarantined activation",
                     error) ||
      !expect_status(sllm_buffer_release(&scale, &error.sink),
                     SLLM_STATUS_INTERNAL_ERROR, "quarantined scale", error) ||
      !expect_status(sllm_buffer_release(&output, &error.sink),
                     SLLM_STATUS_INTERNAL_ERROR, "quarantined output", error) ||
      !expect_status(sllm_context_release(&context, &error.sink),
                     SLLM_STATUS_INTERNAL_ERROR, "quarantined context",
                     error)) {
    return false;
  }
  sllm_public_runtime::FaultInjector::reset();
  return true;
}

bool rmsnorm_guard_page_prefix_is_fail_closed() {
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    std::cerr << "guard-page test could not determine page size\n";
    return false;
  }
  void *const mapping =
      mmap(nullptr, static_cast<std::size_t>(page_size) * 2U,
           PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mapping == MAP_FAILED) {
    std::cerr << "guard-page test mmap failed\n";
    return false;
  }
  void *const guard = static_cast<char *>(mapping) + page_size;
  if (mprotect(guard, static_cast<std::size_t>(page_size), PROT_NONE) != 0) {
    (void)munmap(mapping, static_cast<std::size_t>(page_size) * 2U);
    std::cerr << "guard-page test mprotect failed\n";
    return false;
  }
  auto *const prefix = reinterpret_cast<uint32_t *>(guard) - 2;
  prefix[0] = sizeof(uint32_t) * 2U;
  prefix[1] = SLLM_HIP_ABI_VERSION;
  sllm_rmsnorm_plan_t *plan = nullptr;
  Error error;
  const sllm_status_t status = sllm_rmsnorm_prepare(
      reinterpret_cast<const sllm_context_t *>(static_cast<uintptr_t>(1U)),
      reinterpret_cast<const sllm_rmsnorm_desc_t *>(prefix), &plan,
      &error.sink);
  const bool pass =
      expect_status(status, SLLM_STATUS_INVALID_ARGUMENT,
                    "guard-page truncated RMSNorm descriptor", error) &&
      plan == nullptr;
  (void)munmap(mapping, static_cast<std::size_t>(page_size) * 2U);
  return pass;
}

bool rmsnorm_table_driven_negative_contract() {
  constexpr uint32_t ranks[] = {3U, 4U, 5U, 6U, 7U, 8U};
  constexpr uint64_t columns[] = {1U, 3U, 17U, 255U, 256U, 257U, 2560U};
  for (const uint32_t rank : ranks) {
    for (const uint64_t column : columns) {
      fake_hip::reset();
      sllm_context_t *context = nullptr;
      sllm_buffer_t *activation = nullptr;
      sllm_buffer_t *scale = nullptr;
      sllm_buffer_t *output = nullptr;
      const uint64_t bytes = column * 2U + 64U;
      if (!create_context(&context) ||
          !create_buffer_sized(context, bytes, &activation) ||
          !create_buffer_sized(context, bytes, &scale) ||
          !create_buffer_sized(context, bytes, &output)) {
        return false;
      }
      sllm_rmsnorm_desc_t descriptor{};
      descriptor.struct_size = sizeof(descriptor);
      descriptor.abi_version = SLLM_HIP_ABI_VERSION;
      descriptor.op_version = SLLM_HIP_RMSNORM_VERSION;
      descriptor.accumulation_dtype = SLLM_RMSNORM_ACCUMULATION_F32;
      descriptor.scale_mode = SLLM_RMSNORM_SCALE_MODE_OFFSET_ONE;
      descriptor.alias_policy = SLLM_RMSNORM_ALIAS_POLICY_REJECT_OVERLAP;
      float epsilon = 1.0e-6F;
      std::memcpy(&descriptor.epsilon_bits, &epsilon, sizeof(epsilon));
      descriptor.activation =
          rmsnorm_binding_rank(activation, 0U, rank, column);
      descriptor.raw_scale = rmsnorm_binding(scale, 0U, 1U, column);
      descriptor.output = rmsnorm_binding_rank(output, 0U, rank, column);
      sllm_rmsnorm_plan_t *plan = nullptr;
      Error error;
      if (!expect_status(
              sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
              SLLM_STATUS_OK, "rank/N RMSNorm prepare", error) ||
          !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                         SLLM_STATUS_OK, "rank/N RMSNorm release", error) ||
          !release_buffer(&activation) || !release_buffer(&scale) ||
          !release_buffer(&output) || !release_context(&context)) {
        return false;
      }
    }
  }

  sllm_context_t *context = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *scale = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_buffer(context, &activation) ||
      !create_buffer(context, &scale) || !create_buffer(context, &output)) {
    return false;
  }
  const auto valid = rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U);
  Error error;
  sllm_rmsnorm_plan_t *plan = nullptr;
  const auto expect = [&](sllm_rmsnorm_desc_t descriptor,
                          const sllm_status_t status, const char *name) {
    plan = nullptr;
    return expect_status(
               sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
               status, name, error) &&
           plan == nullptr;
  };
  if (!expect_status(sllm_rmsnorm_prepare(context, nullptr, &plan, &error.sink),
                     SLLM_STATUS_INVALID_RMSNORM_DESCRIPTOR,
                     "null RMSNorm descriptor", error) ||
      plan != nullptr) {
    return false;
  }
  auto direct = valid;
  direct.scale_mode = SLLM_RMSNORM_SCALE_MODE_DIRECT;
  if (!expect_status(sllm_rmsnorm_prepare(context, &direct, &plan, &error.sink),
                     SLLM_STATUS_OK, "direct-scale RMSNorm prepare", error) ||
      plan == nullptr ||
      !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "direct-scale RMSNorm release", error)) {
    return false;
  }
  auto descriptor = valid;
  descriptor.activation.dtype = SLLM_TENSOR_DTYPE_F32;
  if (!expect(descriptor, SLLM_STATUS_UNSUPPORTED_DTYPE, "negative dtype")) {
    return false;
  }
  descriptor = valid;
  descriptor.activation.encoding = UINT32_C(99);
  if (!expect(descriptor, SLLM_STATUS_UNSUPPORTED_ENCODING,
              "negative encoding")) {
    return false;
  }
  descriptor = valid;
  descriptor.activation.shape[1] = 0U;
  if (!expect(descriptor, SLLM_STATUS_ZERO_EXTENT, "negative zero extent")) {
    return false;
  }
  descriptor = valid;
  descriptor.output.shape[1] = 4U;
  descriptor.output.stride_elements[0] = 4U;
  if (!expect(descriptor, SLLM_STATUS_SHAPE_MISMATCH, "negative shape")) {
    return false;
  }
  descriptor = valid;
  descriptor.activation.stride_elements[0] = 1U;
  if (!expect(descriptor, SLLM_STATUS_STRIDE_MISMATCH, "negative stride")) {
    return false;
  }
  descriptor = valid;
  descriptor.struct_size -= 1U;
  if (!expect(descriptor, SLLM_STATUS_INVALID_ARGUMENT,
              "negative descriptor size")) {
    return false;
  }
  descriptor = valid;
  descriptor.struct_size = sizeof(uint32_t) * 2U;
  if (!expect(descriptor, SLLM_STATUS_INVALID_ARGUMENT,
              "malformed top-level prefix")) {
    return false;
  }
  descriptor = valid;
  descriptor.abi_version += 1U;
  if (!expect(descriptor, SLLM_STATUS_INVALID_ABI_VERSION,
              "negative descriptor ABI")) {
    return false;
  }
  descriptor = valid;
  descriptor.reserved[0] = 1U;
  if (!expect(descriptor, SLLM_STATUS_RESERVED_NONZERO,
              "negative descriptor reserved")) {
    return false;
  }
  descriptor = valid;
  descriptor.op_version += 1U;
  if (!expect(descriptor, SLLM_STATUS_INVALID_RMSNORM_DESCRIPTOR,
              "negative op version")) {
    return false;
  }
  descriptor = valid;
  descriptor.accumulation_dtype = SLLM_TENSOR_DTYPE_BF16;
  if (!expect(descriptor, SLLM_STATUS_INVALID_RMSNORM_DESCRIPTOR,
              "negative accumulation")) {
    return false;
  }
  descriptor = valid;
  descriptor.scale_mode = 0U;
  if (!expect(descriptor, SLLM_STATUS_UNSUPPORTED_SCALE_MODE,
              "negative scale mode")) {
    return false;
  }
  descriptor = valid;
  descriptor.alias_policy = 0U;
  if (!expect(descriptor, SLLM_STATUS_INVALID_RMSNORM_DESCRIPTOR,
              "negative alias policy")) {
    return false;
  }
  for (const float epsilon : {0.0F, -1.0F, NAN, INFINITY}) {
    descriptor = valid;
    std::memcpy(&descriptor.epsilon_bits, &epsilon, sizeof(epsilon));
    if (!expect(descriptor, SLLM_STATUS_INVALID_EPSILON, "negative epsilon")) {
      return false;
    }
  }
  descriptor = valid;
  descriptor.raw_scale.rank = 2U;
  descriptor.raw_scale.shape[1] = 1U;
  descriptor.raw_scale.stride_elements[0] = 1U;
  descriptor.raw_scale.stride_elements[1] = 1U;
  if (!expect(descriptor, SLLM_STATUS_SHAPE_MISMATCH,
              "negative raw-scale rank")) {
    return false;
  }
  descriptor = valid;
  descriptor.raw_scale.shape[0] = 2U;
  if (!expect(descriptor, SLLM_STATUS_SHAPE_MISMATCH,
              "negative raw-scale shape")) {
    return false;
  }
  descriptor = valid;
  descriptor.activation.reserved0 = 1U;
  if (!expect(descriptor, SLLM_STATUS_RESERVED_NONZERO,
              "negative nested reserved")) {
    return false;
  }
  descriptor = valid;
  descriptor.activation.abi_version += 1U;
  if (!expect(descriptor, SLLM_STATUS_INVALID_ABI_VERSION,
              "negative nested ABI")) {
    return false;
  }
  descriptor = valid;
  descriptor.activation.buffer = nullptr;
  if (!expect(descriptor, SLLM_STATUS_INVALID_TENSOR_BINDING,
              "negative nested null")) {
    return false;
  }
  descriptor = valid;
  descriptor.activation.rank = 0U;
  if (!expect(descriptor, SLLM_STATUS_INVALID_TENSOR_BINDING,
              "negative rank zero")) {
    return false;
  }
  descriptor = valid;
  descriptor.activation.rank = 9U;
  if (!expect(descriptor, SLLM_STATUS_INVALID_TENSOR_BINDING,
              "negative rank nine")) {
    return false;
  }
  descriptor = valid;
  descriptor.activation.shape[2] = 2U;
  if (!expect(descriptor, SLLM_STATUS_INVALID_TENSOR_BINDING,
              "negative unused shape")) {
    return false;
  }
  descriptor = valid;
  descriptor.activation.stride_elements[2] = 1U;
  if (!expect(descriptor, SLLM_STATUS_INVALID_TENSOR_BINDING,
              "negative unused stride")) {
    return false;
  }
  return release_buffer(&activation) && release_buffer(&scale) &&
         release_buffer(&output) && release_context(&context);
}

bool rmsnorm_prepare_required_shape_and_context_cases() {
  constexpr uint64_t dimensions[] = {1U, 3U, 17U, 255U, 256U, 257U, 2560U};
  for (const uint64_t columns : dimensions) {
    fake_hip::reset();
    sllm_context_t *context = nullptr;
    sllm_buffer_t *activation = nullptr;
    sllm_buffer_t *scale = nullptr;
    sllm_buffer_t *output = nullptr;
    const uint64_t bytes = columns * 4U;
    if (!create_context(&context) ||
        !create_buffer_sized(context, bytes, &activation) ||
        !create_buffer_sized(context, bytes, &scale) ||
        !create_buffer_sized(context, bytes, &output)) {
      return false;
    }
    sllm_rmsnorm_desc_t descriptor =
        rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U, 1U, columns);
    sllm_rmsnorm_plan_t *plan = nullptr;
    Error error;
    if (!expect_status(
            sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
            SLLM_STATUS_OK, "rmsnorm rank-one prepare", error) ||
        !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                       SLLM_STATUS_OK, "rmsnorm rank-one release", error)) {
      return false;
    }
    descriptor =
        rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U, 2U, columns);
    if (!expect_status(
            sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
            SLLM_STATUS_OK, "rmsnorm rank-two prepare", error) ||
        !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                       SLLM_STATUS_OK, "rmsnorm rank-two release", error) ||
        !release_buffer(&activation) || !release_buffer(&scale) ||
        !release_buffer(&output) || !release_context(&context)) {
      return false;
    }
  }

  sllm_context_t *first = nullptr;
  sllm_context_t *second = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *scale = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&first) || !create_context(&second) ||
      !create_buffer_sized(first, 4096U, &activation) ||
      !create_buffer_sized(first, 4096U, &scale) ||
      !create_buffer_sized(second, 4096U, &output)) {
    return false;
  }
  Error error;
  sllm_rmsnorm_plan_t *plan = nullptr;
  sllm_rmsnorm_desc_t descriptor =
      rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U, 2U, 257U);
  if (!expect_status(
          sllm_rmsnorm_prepare(first, &descriptor, &plan, &error.sink),
          SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH, "rmsnorm context mismatch",
          error)) {
    return false;
  }
  descriptor = rmsnorm_descriptor(activation, 0U, scale, 0U, activation, 0U);
  descriptor.activation.shape[0] = UINT64_MAX;
  descriptor.activation.rank = 1U;
  descriptor.activation.stride_elements[0] = 1U;
  descriptor.activation.shape[1] = 0U;
  descriptor.activation.stride_elements[1] = 0U;
  if (!expect_status(
          sllm_rmsnorm_prepare(first, &descriptor, &plan, &error.sink),
          SLLM_STATUS_METADATA_OVERFLOW, "rmsnorm metadata overflow", error)) {
    return false;
  }
  return release_buffer(&activation) && release_buffer(&scale) &&
         release_buffer(&output) && release_context(&first) &&
         release_context(&second);
}

sllm_rmsnorm_dispatch_info_t rmsnorm_dispatch_info() {
  sllm_rmsnorm_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_RMSNORM_DISPATCH_INFO_VERSION;
  return info;
}

sllm_residual_rmsnorm_desc_t residual_rmsnorm_descriptor(
    const sllm_buffer_t *const residual, const sllm_buffer_t *const addend,
    const sllm_buffer_t *const scale,
    const sllm_buffer_t *const residual_output,
    const sllm_buffer_t *const output, const uint64_t rows = 2U,
    const uint64_t columns = 3U) {
  sllm_residual_rmsnorm_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_RESIDUAL_RMSNORM_VERSION;
  descriptor.accumulation_dtype = SLLM_RMSNORM_ACCUMULATION_F32;
  descriptor.scale_mode = SLLM_RMSNORM_SCALE_MODE_OFFSET_ONE;
  descriptor.alias_policy = SLLM_RMSNORM_ALIAS_POLICY_REJECT_OVERLAP;
  float epsilon = 1.0e-6F;
  std::memcpy(&descriptor.epsilon_bits, &epsilon, sizeof(epsilon));
  descriptor.residual = rmsnorm_binding(residual, 0U, rows, columns);
  descriptor.addend = rmsnorm_binding(addend, 0U, rows, columns);
  descriptor.raw_scale = rmsnorm_binding(scale, 0U, 1U, columns);
  descriptor.residual_output =
      rmsnorm_binding(residual_output, 0U, rows, columns);
  descriptor.output = rmsnorm_binding(output, 0U, rows, columns);
  return descriptor;
}

sllm_residual_rmsnorm_dispatch_info_t residual_rmsnorm_dispatch_info() {
  sllm_residual_rmsnorm_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_RESIDUAL_RMSNORM_DISPATCH_INFO_VERSION;
  return info;
}

bool residual_rmsnorm_prepare_execute_lifetime_contract() {
  fake_hip::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *residual = nullptr;
  sllm_buffer_t *addend = nullptr;
  sllm_buffer_t *scale = nullptr;
  sllm_buffer_t *residual_output = nullptr;
  sllm_buffer_t *output = nullptr;
  constexpr uint64_t rows = 2U;
  constexpr uint64_t columns = 257U;
  const uint64_t matrix_bytes = rows * columns * sizeof(uint16_t);
  const uint64_t scale_bytes = columns * sizeof(uint16_t);
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer_sized(context, matrix_bytes, &residual) ||
      !create_buffer_sized(context, matrix_bytes, &addend) ||
      !create_buffer_sized(context, scale_bytes, &scale) ||
      !create_buffer_sized(context, matrix_bytes, &residual_output) ||
      !create_buffer_sized(context, matrix_bytes, &output)) {
    return false;
  }
  sllm_residual_rmsnorm_desc_t descriptor = residual_rmsnorm_descriptor(
      residual, addend, scale, residual_output, output, rows, columns);
  sllm_residual_rmsnorm_plan_t *plan = nullptr;
  Error error;
  if (!expect_status(sllm_residual_rmsnorm_prepare(context, &descriptor, &plan,
                                                   &error.sink),
                     SLLM_STATUS_OK, "residual RMSNorm prepare", error) ||
      plan == nullptr ||
      !expect_status(sllm_buffer_release(&residual, &error.sink),
                     SLLM_STATUS_PUBLIC_BUSY, "residual dependency busy",
                     error)) {
    return false;
  }
  sllm_completion_t *completion = nullptr;
  sllm_residual_rmsnorm_dispatch_info_t info = residual_rmsnorm_dispatch_info();
  if (!expect_status(sllm_residual_rmsnorm_execute(plan, queue, &completion,
                                                   &info, &error.sink),
                     SLLM_STATUS_OK, "residual RMSNorm execute", error) ||
      completion == nullptr || info.dispatch_count != 1U ||
      info.row_count != rows || info.normalized_size != columns ||
      info.kernel_id != 1U ||
      std::strcmp(info.kernel_symbol, "rmsnorm.residual_fused.wave32.v1") !=
          0 ||
      fake_hip::residual_rmsnorm_launch_calls() != 1U) {
    return false;
  }
  if (!expect_status(sllm_residual_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_PUBLIC_BUSY, "residual in-flight release",
                     error) ||
      !query_completion(completion, SLLM_STATUS_OK) ||
      !release_completion(&completion) ||
      !expect_status(sllm_residual_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "residual plan release", error) ||
      !release_queue(&queue) || !release_buffer(&residual) ||
      !release_buffer(&addend) || !release_buffer(&scale) ||
      !release_buffer(&residual_output) || !release_buffer(&output) ||
      !release_context(&context)) {
    return false;
  }
  return true;
}

bool rmsnorm_execute_metadata_and_reuse() {
  fake_hip::reset();
  sllm_context_t *context = nullptr;
  sllm_context_t *other_context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_queue_t *other_queue = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *scale = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_context(&other_context) ||
      !create_queue(context, &queue) ||
      !create_queue(other_context, &other_queue) ||
      !create_buffer_sized(context, 2U * 2U * 257U + 64U, &activation) ||
      !create_buffer_sized(context, 2U * 257U + 64U, &scale) ||
      !create_buffer_sized(context, 2U * 2U * 257U + 64U, &output)) {
    return false;
  }
  sllm_rmsnorm_desc_t descriptor =
      rmsnorm_descriptor(activation, 2U, scale, 4U, output, 6U, 2U, 257U);
  sllm_rmsnorm_plan_t *plan = nullptr;
  Error error;
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "execute prepare", error)) {
    return false;
  }
  sllm_completion_t *sentinel_completion =
      reinterpret_cast<sllm_completion_t *>(static_cast<uintptr_t>(0x55U));
  sllm_rmsnorm_dispatch_info_t invalid_info = rmsnorm_dispatch_info();
  invalid_info.struct_size = sizeof(invalid_info) - 1U;
  if (!expect_status(sllm_rmsnorm_execute(plan, queue, &sentinel_completion,
                                          &invalid_info, &error.sink),
                     SLLM_STATUS_INVALID_ARGUMENT, "truncated dispatch info",
                     error) ||
      sentinel_completion != reinterpret_cast<sllm_completion_t *>(
                                 static_cast<uintptr_t>(0x55U)) ||
      invalid_info.struct_size != sizeof(invalid_info) - 1U) {
    return false;
  }
  invalid_info = rmsnorm_dispatch_info();
  invalid_info.reserved[0] = 1U;
  if (!expect_status(sllm_rmsnorm_execute(plan, queue, &sentinel_completion,
                                          &invalid_info, &error.sink),
                     SLLM_STATUS_RESERVED_NONZERO, "reserved dispatch info",
                     error) ||
      sentinel_completion != reinterpret_cast<sllm_completion_t *>(
                                 static_cast<uintptr_t>(0x55U)) ||
      invalid_info.reserved[0] != 1U) {
    return false;
  }
  invalid_info = rmsnorm_dispatch_info();
  if (!expect_status(sllm_rmsnorm_execute(plan, queue, &sentinel_completion,
                                          nullptr, &error.sink),
                     SLLM_STATUS_INVALID_ARGUMENT, "null dispatch info",
                     error) ||
      sentinel_completion != reinterpret_cast<sllm_completion_t *>(
                                 static_cast<uintptr_t>(0x55U))) {
    return false;
  }
  invalid_info.abi_version = SLLM_HIP_ABI_VERSION + 1U;
  if (!expect_status(sllm_rmsnorm_execute(plan, queue, nullptr, &invalid_info,
                                          &error.sink),
                     SLLM_STATUS_INVALID_ARGUMENT, "null completion output",
                     error) ||
      invalid_info.abi_version != SLLM_HIP_ABI_VERSION + 1U) {
    return false;
  }
  invalid_info = rmsnorm_dispatch_info();
  invalid_info.abi_version = SLLM_HIP_ABI_VERSION + 1U;
  if (!expect_status(sllm_rmsnorm_execute(plan, queue, &sentinel_completion,
                                          &invalid_info, &error.sink),
                     SLLM_STATUS_INVALID_ABI_VERSION, "wrong dispatch ABI",
                     error) ||
      sentinel_completion != reinterpret_cast<sllm_completion_t *>(
                                 static_cast<uintptr_t>(0x55U)) ||
      invalid_info.abi_version != SLLM_HIP_ABI_VERSION + 1U) {
    return false;
  }
  invalid_info = rmsnorm_dispatch_info();
  if (!expect_status(
          sllm_rmsnorm_execute(plan, other_queue, &sentinel_completion,
                               &invalid_info, &error.sink),
          SLLM_STATUS_PUBLIC_DEVICE_MISMATCH, "wrong RMSNorm queue", error) ||
      sentinel_completion != reinterpret_cast<sllm_completion_t *>(
                                 static_cast<uintptr_t>(0x55U))) {
    return false;
  }
  sllm_completion_t *completion = nullptr;
  sllm_rmsnorm_dispatch_info_t info = rmsnorm_dispatch_info();
  if (!expect_status(
          sllm_rmsnorm_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_OK, "RMSNorm execute", error) ||
      completion == nullptr || info.dispatch_id == 0U ||
      info.dispatch_count != 1U || info.kernel_id != 1U ||
      info.workgroup_size_x != 256U || info.grid_size_x != 2U ||
      info.row_count != 2U || info.normalized_size != 257U ||
      info.backend != SLLM_BACKEND_HIP || info.fallback_allowed != 0U ||
      info.fallback_used != 0U ||
      std::strcmp(info.kernel_symbol, "rmsnorm.baseline.wave32.v1") != 0 ||
      std::strcmp(info.device_symbol, "sllm_rmsnorm_baseline_wave32_v1") != 0 ||
      fake_hip::rmsnorm_launch_calls() != 1U ||
      fake_hip::rmsnorm_last_normalized_size() != 257U ||
      fake_hip::rmsnorm_last_row_count() != 2U) {
    return false;
  }
  sllm_completion_t *second_completion =
      reinterpret_cast<sllm_completion_t *>(static_cast<uintptr_t>(0x77U));
  sllm_rmsnorm_dispatch_info_t second_info = rmsnorm_dispatch_info();
  if (!expect_status(sllm_rmsnorm_execute(plan, queue, &second_completion,
                                          &second_info, &error.sink),
                     SLLM_STATUS_PUBLIC_BUSY, "second in-flight execute",
                     error) ||
      second_completion != reinterpret_cast<sllm_completion_t *>(
                               static_cast<uintptr_t>(0x77U))) {
    return false;
  }
  if (!expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_PUBLIC_BUSY, "in-flight plan release",
                     error)) {
    return false;
  }
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect_status(sllm_completion_query(completion, &result, &error.sink),
                     SLLM_STATUS_OK, "RMSNorm completion query", error) ||
      result.state != SLLM_COMPLETION_STATE_SUCCESS ||
      result.transfer_size_bytes != 0U || result.available_bytes != 0U ||
      !expect_status(sllm_completion_release(&completion, &error.sink),
                     SLLM_STATUS_OK, "RMSNorm completion release", error) ||
      completion != nullptr) {
    return false;
  }
  const uint64_t first_dispatch = info.dispatch_id;
  completion = nullptr;
  info = rmsnorm_dispatch_info();
  if (!expect_status(
          sllm_rmsnorm_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_OK, "RMSNorm plan reuse", error) ||
      info.dispatch_id <= first_dispatch ||
      !expect_status(sllm_completion_query(completion, &result, &error.sink),
                     SLLM_STATUS_OK, "RMSNorm reused completion query",
                     error) ||
      !expect_status(sllm_completion_release(&completion, &error.sink),
                     SLLM_STATUS_OK, "RMSNorm reused completion release",
                     error) ||
      !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "RMSNorm reused plan release", error) ||
      !release_queue(&queue) || !release_buffer(&activation) ||
      !release_buffer(&scale) || !release_buffer(&output) ||
      !release_context(&context) || !release_queue(&other_queue) ||
      !release_context(&other_context)) {
    return false;
  }
  return true;
}

bool rmsnorm_direct_scale_numerical_contract() {
  fake_hip::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *scale = nullptr;
  sllm_buffer_t *output = nullptr;
  constexpr uint64_t rows = 2U;
  constexpr uint64_t columns = 3U;
  constexpr uint64_t activation_bytes = rows * columns * sizeof(uint16_t);
  constexpr uint64_t scale_bytes = columns * sizeof(uint16_t);
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer_sized(context, activation_bytes, &activation) ||
      !create_buffer_sized(context, scale_bytes, &scale) ||
      !create_buffer_sized(context, activation_bytes, &output)) {
    return false;
  }

  const std::array<float, rows * columns> activation_values = {
      1.0F, 2.0F, 3.0F, -1.0F, -2.0F, -3.0F};
  const std::array<float, columns> scale_values = {1.0F, 2.0F, 0.5F};
  std::array<uint16_t, rows * columns> activation_words{};
  std::array<uint16_t, columns> scale_words{};
  for (std::size_t index = 0U; index != activation_words.size(); ++index) {
    activation_words[index] =
        sllm_rmsnorm_kernel::float_to_bf16_rne_bits(activation_values[index]);
  }
  for (std::size_t index = 0U; index != scale_words.size(); ++index) {
    scale_words[index] =
        sllm_rmsnorm_kernel::float_to_bf16_rne_bits(scale_values[index]);
  }

  Error error;
  const auto upload = [&](const sllm_buffer_t *const buffer,
                          const void *const source, const uint64_t bytes) {
    sllm_transfer_desc_t transfer{};
    transfer.struct_size = sizeof(transfer);
    transfer.abi_version = SLLM_HIP_ABI_VERSION;
    transfer.host_pointer = const_cast<void *>(source);
    transfer.size_bytes = bytes;
    sllm_completion_t *completion = nullptr;
    return expect_status(sllm_buffer_copy_h2d(queue, buffer, &transfer,
                                              &completion, &error.sink),
                         SLLM_STATUS_OK, "direct RMSNorm upload", error) &&
           query_completion(completion, SLLM_STATUS_OK) &&
           release_completion(&completion);
  };
  if (!upload(activation, activation_words.data(), activation_bytes) ||
      !upload(scale, scale_words.data(), scale_bytes)) {
    return false;
  }
  fake_hip::set_rmsnorm_numerical_execution(true);

  auto descriptor =
      rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U, rows, columns);
  descriptor.scale_mode = SLLM_RMSNORM_SCALE_MODE_DIRECT;
  sllm_rmsnorm_plan_t *plan = nullptr;
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "direct RMSNorm numerical prepare", error)) {
    return false;
  }
  sllm_completion_t *completion = nullptr;
  auto info = rmsnorm_dispatch_info();
  if (!expect_status(
          sllm_rmsnorm_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_OK, "direct RMSNorm numerical execute", error) ||
      !query_completion(completion, SLLM_STATUS_OK) ||
      !release_completion(&completion) ||
      fake_hip::rmsnorm_last_scale_mode() != SLLM_RMSNORM_SCALE_MODE_DIRECT) {
    return false;
  }

  sllm_completion_t *readback = nullptr;
  std::array<uint16_t, rows * columns> observed{};
  uint64_t bytes_written = 0U;
  if (!submit_d2h(queue, output, activation_bytes, &readback) ||
      !query_completion(readback, SLLM_STATUS_OK) ||
      !expect_status(sllm_completion_read(readback, observed.data(),
                                          activation_bytes, &bytes_written,
                                          &error.sink),
                     SLLM_STATUS_OK, "direct RMSNorm readback", error) ||
      bytes_written != activation_bytes || !release_completion(&readback)) {
    return false;
  }

  std::array<uint16_t, rows * columns> expected{};
  for (uint64_t row = 0U; row != rows; ++row) {
    float sum = 0.0F;
    for (uint64_t column = 0U; column != columns; ++column) {
      const float value = activation_values[row * columns + column];
      sum += value * value;
    }
    const float inverse_rms =
        1.0F / std::sqrt(sum / static_cast<float>(columns) + 1.0e-6F);
    for (uint64_t column = 0U; column != columns; ++column) {
      expected[row * columns + column] =
          sllm_rmsnorm_kernel::float_to_bf16_rne_bits(
              activation_values[row * columns + column] * inverse_rms *
              scale_values[column]);
    }
  }
  const uint16_t forbidden_offset_one =
      sllm_rmsnorm_kernel::float_to_bf16_rne_bits(
          activation_values[0] /
          std::sqrt((1.0F + 4.0F + 9.0F) / 3.0F + 1.0e-6F) *
          (1.0F + scale_values[0]));
  const bool numerical_match =
      observed == expected && observed.front() != forbidden_offset_one;
  return numerical_match &&
         expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                       SLLM_STATUS_OK, "direct RMSNorm numerical release",
                       error) &&
         !plan && release_queue(&queue) && release_buffer(&activation) &&
         release_buffer(&scale) && release_buffer(&output) &&
         release_context(&context);
}

bool rmsnorm_execute_boundaries_and_failures() {
  constexpr uint64_t columns[] = {1U,   3U,    17U,   255U,  256U,
                                  257U, 2560U, 4095U, 4096U, 4097U};
  constexpr uint64_t rows[] = {1U, 2U, 3U};
  for (const uint64_t row_count : rows) {
    for (const uint64_t column_count : columns) {
      fake_hip::reset();
      sllm_context_t *context = nullptr;
      sllm_queue_t *queue = nullptr;
      sllm_buffer_t *activation = nullptr;
      sllm_buffer_t *scale = nullptr;
      sllm_buffer_t *output = nullptr;
      const uint64_t row_bytes = row_count * column_count * 2U + 64U;
      const uint64_t scale_bytes = column_count * 2U + 64U;
      if (!create_context(&context) || !create_queue(context, &queue) ||
          !create_buffer_sized(context, row_bytes, &activation) ||
          !create_buffer_sized(context, scale_bytes, &scale) ||
          !create_buffer_sized(context, row_bytes, &output)) {
        return false;
      }
      sllm_rmsnorm_desc_t descriptor = rmsnorm_descriptor(
          activation, 0U, scale, 0U, output, 0U, row_count, column_count);
      sllm_rmsnorm_plan_t *plan = nullptr;
      Error error;
      if (!expect_status(
              sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
              SLLM_STATUS_OK, "boundary prepare", error)) {
        return false;
      }
      sllm_completion_t *completion = nullptr;
      sllm_rmsnorm_dispatch_info_t info = rmsnorm_dispatch_info();
      const sllm_status_t expected =
          column_count == 4097U ? SLLM_STATUS_UNSUPPORTED : SLLM_STATUS_OK;
      if (!expect_status(sllm_rmsnorm_execute(plan, queue, &completion, &info,
                                              &error.sink),
                         expected, "boundary execute", error) ||
          (expected == SLLM_STATUS_OK &&
           (!expect_status(
                sllm_completion_query(completion, nullptr, &error.sink),
                SLLM_STATUS_INVALID_ARGUMENT, "boundary null completion result",
                error) ||
            !expect_status(
                sllm_completion_query(
                    completion,
                    reinterpret_cast<sllm_completion_result_t *>(&info),
                    &error.sink),
                SLLM_STATUS_RESERVED_NONZERO,
                "boundary wrong completion result", error))) ||
          (expected == SLLM_STATUS_OK &&
           !expect_status(sllm_completion_release(&completion, &error.sink),
                          SLLM_STATUS_PUBLIC_BUSY,
                          "boundary unqueried completion release", error))) {
        return false;
      }
      if (expected == SLLM_STATUS_OK) {
        sllm_completion_result_t result{};
        result.struct_size = sizeof(result);
        result.abi_version = SLLM_HIP_ABI_VERSION;
        if (!expect_status(
                sllm_completion_query(completion, &result, &error.sink),
                SLLM_STATUS_OK, "boundary query", error) ||
            !expect_status(sllm_completion_release(&completion, &error.sink),
                           SLLM_STATUS_OK, "boundary release", error)) {
          return false;
        }
      }
      if (!expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                         SLLM_STATUS_OK, "boundary plan release", error) ||
          !release_queue(&queue) || !release_buffer(&activation) ||
          !release_buffer(&scale) || !release_buffer(&output) ||
          !release_context(&context)) {
        return false;
      }
    }
  }

  fake_hip::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *scale = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer_sized(context, 4096U, &activation) ||
      !create_buffer_sized(context, 2048U, &scale) ||
      !create_buffer_sized(context, 4096U, &output)) {
    return false;
  }
  sllm_rmsnorm_desc_t descriptor =
      rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U, 2U, 256U);
  sllm_rmsnorm_plan_t *plan = nullptr;
  Error error;
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "failure prepare", error)) {
    return false;
  }
  sllm_completion_t *completion = nullptr;
  sllm_rmsnorm_dispatch_info_t info = rmsnorm_dispatch_info();
  const std::size_t events_before_failures = fake_hip::live_events();
  fake_hip::set_rmsnorm_launch_status(hipErrorUnknown);
  if (!expect_status(
          sllm_rmsnorm_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR, "launch failure", error) ||
      completion != nullptr ||
      fake_hip::live_events() != events_before_failures ||
      !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "launch rollback plan release", error)) {
    return false;
  }
  fake_hip::set_rmsnorm_launch_status(hipSuccess);
  plan = nullptr;
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "event failure prepare", error)) {
    return false;
  }
  fake_hip::set_event_record_status(hipErrorUnknown);
  info = rmsnorm_dispatch_info();
  if (!expect_status(
          sllm_rmsnorm_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR, "event failure", error) ||
      completion != nullptr ||
      fake_hip::live_events() != events_before_failures ||
      !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "event rollback plan release", error) ||
      !release_queue(&queue) || !release_buffer(&activation) ||
      !release_buffer(&scale) || !release_buffer(&output) ||
      !release_context(&context)) {
    return false;
  }
  return true;
}

bool rmsnorm_execute_exception_scope_guards_restore_plan_reuse() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  sllm_test_rmsnorm_execute_throw_after_reservation(0U);
  sllm_test_rmsnorm_execute_throw_after_registration(0U);

  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *scale = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer_sized(context, 4096U, &activation) ||
      !create_buffer_sized(context, 2048U, &scale) ||
      !create_buffer_sized(context, 4096U, &output)) {
    return false;
  }
  const sllm_rmsnorm_desc_t descriptor =
      rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U, 2U, 257U);
  sllm_rmsnorm_plan_t *plan = nullptr;
  Error error;
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "scope-guard prepare", error) ||
      plan == nullptr) {
    return false;
  }
  const std::size_t events_before = fake_hip::live_events();
  sllm_completion_t *completion =
      reinterpret_cast<sllm_completion_t *>(static_cast<uintptr_t>(0x9U));
  sllm_rmsnorm_dispatch_info_t info = rmsnorm_dispatch_info();

  sllm_test_rmsnorm_execute_throw_after_reservation(1U);
  if (!expect_status(
          sllm_rmsnorm_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_INTERNAL_ERROR, "scope-guard reservation exception",
          error) ||
      completion != nullptr || fake_hip::live_events() != events_before ||
      fake_hip::rmsnorm_launch_calls() != 0U) {
    std::cerr
        << "reservation exception leaked RMSNorm accounting or a handle\n";
    return false;
  }

  const auto execute_success = [&]() {
    completion = nullptr;
    info = rmsnorm_dispatch_info();
    sllm_completion_result_t result{};
    result.struct_size = sizeof(result);
    result.abi_version = SLLM_HIP_ABI_VERSION;
    return expect_status(sllm_rmsnorm_execute(plan, queue, &completion, &info,
                                              &error.sink),
                         SLLM_STATUS_OK, "scope-guard plan reuse", error) &&
           completion != nullptr &&
           expect_status(
               sllm_completion_query(completion, &result, &error.sink),
               SLLM_STATUS_OK, "scope-guard completion query", error) &&
           expect_status(sllm_completion_release(&completion, &error.sink),
                         SLLM_STATUS_OK, "scope-guard completion release",
                         error) &&
           completion == nullptr;
  };
  if (!execute_success()) {
    return false;
  }

  const uint64_t launches_before_registration_fault =
      fake_hip::rmsnorm_launch_calls();
  completion =
      reinterpret_cast<sllm_completion_t *>(static_cast<uintptr_t>(0xaU));
  info = rmsnorm_dispatch_info();
  sllm_test_rmsnorm_execute_throw_after_registration(1U);
  if (!expect_status(
          sllm_rmsnorm_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_INTERNAL_ERROR, "scope-guard registration exception",
          error) ||
      completion != nullptr || fake_hip::live_events() != events_before ||
      fake_hip::rmsnorm_launch_calls() != launches_before_registration_fault) {
    std::cerr
        << "registration exception leaked RMSNorm accounting or a handle\n";
    return false;
  }
  if (!execute_success() ||
      !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "scope-guard plan release", error) ||
      !release_queue(&queue) || !release_buffer(&activation) ||
      !release_buffer(&scale) || !release_buffer(&output) ||
      !release_context(&context)) {
    return false;
  }
  sllm_test_rmsnorm_execute_throw_after_reservation(0U);
  sllm_test_rmsnorm_execute_throw_after_registration(0U);
  return true;
}

bool rmsnorm_registered_exception_with_event_destroy_failure_is_quarantined() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  sllm_test_rmsnorm_execute_throw_after_reservation(0U);
  sllm_test_rmsnorm_execute_throw_after_registration(0U);

  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *scale = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer_sized(context, 4096U, &activation) ||
      !create_buffer_sized(context, 2048U, &scale) ||
      !create_buffer_sized(context, 4096U, &output)) {
    return false;
  }
  const sllm_rmsnorm_desc_t descriptor =
      rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U, 2U, 257U);
  sllm_rmsnorm_plan_t *plan = nullptr;
  Error error;
  if (!expect_status(
          sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "ambiguous-cleanup prepare", error) ||
      plan == nullptr) {
    return false;
  }

  const std::size_t events_before = fake_hip::live_events();
  const std::size_t destroy_before = fake_hip::event_destroy_calls();
  const std::size_t poison_before = sllm_test_poison_count();
  const std::size_t launches_before = fake_hip::rmsnorm_launch_calls();
  sllm_completion_t *completion =
      reinterpret_cast<sllm_completion_t *>(static_cast<uintptr_t>(0xbU));
  sllm_rmsnorm_dispatch_info_t info = rmsnorm_dispatch_info();

  /* EventDestroyError models an ownership-ambiguous native cleanup result:
   * the injection returns an error without calling fake hipEventDestroy. */
  sllm_public_runtime::FaultInjector::set(
      sllm_public_runtime::FaultPoint::EventDestroyError, 1U);
  sllm_test_rmsnorm_execute_throw_after_registration(1U);
  if (!expect_status(
          sllm_rmsnorm_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_INTERNAL_ERROR, "ambiguous-cleanup registered exception",
          error) ||
      completion != nullptr ||
      sllm_public_runtime::FaultInjector::consume(
          sllm_public_runtime::FaultPoint::EventDestroyError) ||
      sllm_test_poison_count() != poison_before + 1U ||
      fake_hip::event_destroy_calls() != destroy_before ||
      fake_hip::live_events() != events_before + 2U ||
      fake_hip::rmsnorm_launch_calls() != launches_before) {
    std::cerr
        << "registered exception did not fail closed on ambiguous event cleanup"
        << " completion=" << (completion == nullptr ? "null" : "set")
        << " poison=" << sllm_test_poison_count()
        << " expected_poison=" << poison_before + 1U
        << " destroy=" << fake_hip::event_destroy_calls()
        << " expected_destroy=" << destroy_before
        << " live_events=" << fake_hip::live_events()
        << " expected_live_events=" << events_before + 2U << '\n';
    return false;
  }
  sllm_test_rmsnorm_execute_throw_after_registration(0U);

  /* The poison owner retains the completion graph and both live timing events.
   * The plan remains in-flight, and the poisoned Context rejects all reuse and
   * cleanup attempts; this test deliberately does not claim safe reuse. */
  completion = nullptr;
  info = rmsnorm_dispatch_info();
  if (!expect_status(
          sllm_rmsnorm_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_PUBLIC_BUSY, "ambiguous-cleanup execute reuse", error) ||
      completion != nullptr ||
      !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                     SLLM_STATUS_PUBLIC_BUSY, "ambiguous-cleanup plan release",
                     error) ||
      plan == nullptr ||
      !expect_status(sllm_queue_release(&queue, &error.sink),
                     SLLM_STATUS_INTERNAL_ERROR,
                     "ambiguous-cleanup queue release", error) ||
      queue == nullptr ||
      !expect_status(sllm_buffer_release(&activation, &error.sink),
                     SLLM_STATUS_INTERNAL_ERROR,
                     "ambiguous-cleanup activation release", error) ||
      activation == nullptr ||
      !expect_status(sllm_buffer_release(&scale, &error.sink),
                     SLLM_STATUS_INTERNAL_ERROR,
                     "ambiguous-cleanup scale release", error) ||
      scale == nullptr ||
      !expect_status(sllm_buffer_release(&output, &error.sink),
                     SLLM_STATUS_INTERNAL_ERROR,
                     "ambiguous-cleanup output release", error) ||
      output == nullptr ||
      !expect_status(sllm_context_release(&context, &error.sink),
                     SLLM_STATUS_INTERNAL_ERROR,
                     "ambiguous-cleanup context release", error) ||
      context == nullptr || fake_hip::event_destroy_calls() != destroy_before ||
      fake_hip::live_events() != events_before + 2U ||
      fake_hip::rmsnorm_launch_calls() != launches_before) {
    std::cerr
        << "ambiguous cleanup was retried or the poisoned graph was reusable\n";
    return false;
  }
  sllm_public_runtime::FaultInjector::reset();
  return true;
}

bool deferred_segment_uses_one_fence_event_and_finalizes_exactly() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *scale = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer_sized(context, 4096U, &activation) ||
      !create_buffer_sized(context, 2048U, &scale) ||
      !create_buffer_sized(context, 4096U, &output)) {
    return false;
  }
  const sllm_rmsnorm_desc_t descriptor =
      rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U, 2U, 257U);
  std::array<sllm_rmsnorm_plan_t *, 17> plans{};
  Error error;
  for (auto &plan : plans) {
    if (!expect_status(
            sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
            SLLM_STATUS_OK, "deferred segment prepare", error)) {
      return false;
    }
  }
  if (!expect_status(
          sllm_queue_set_completion_mode(
              queue, SLLM_QUEUE_COMPLETION_MODE_DEFERRED, &error.sink),
          SLLM_STATUS_OK, "set deferred completion mode", error)) {
    return false;
  }
  const std::size_t events_before = fake_hip::live_events();
  std::array<sllm_completion_t *, 17> operations{};
  for (std::size_t index = 0U; index != operations.size(); ++index) {
    auto info = rmsnorm_dispatch_info();
    if (!expect_status(sllm_rmsnorm_execute(plans[index], queue,
                                            &operations[index], &info,
                                            &error.sink),
                       SLLM_STATUS_OK, "deferred segment execute", error) ||
        operations[index] == nullptr ||
        fake_hip::live_events() != events_before) {
      std::cerr << "deferred execute invariant failed at " << index
                << " events=" << fake_hip::live_events()
                << " baseline=" << events_before << '\n';
      return false;
    }
  }
  if (!expect_status(
          sllm_queue_set_completion_mode(
              queue, SLLM_QUEUE_COMPLETION_MODE_PROFILED, &error.sink),
          SLLM_STATUS_PUBLIC_BUSY, "completion mode change with active segment",
          error)) {
    return false;
  }
  sllm_completion_result_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect_status(sllm_completion_query(operations[0], &result, &error.sink),
                     SLLM_STATUS_PUBLIC_PENDING, "eventless completion query",
                     error) ||
      result.state != SLLM_COMPLETION_STATE_PENDING) {
    std::cerr << "deferred pending-query invariant failed, state="
              << result.state << '\n';
    return false;
  }
  sllm_completion_t *fence = nullptr;
  if (!expect_status(sllm_queue_fence(queue, &fence, &error.sink),
                     SLLM_STATUS_OK, "deferred segment fence", error) ||
      fence == nullptr || fake_hip::live_events() != events_before + 1U) {
    std::cerr << "deferred fence creation invariant failed, events="
              << fake_hip::live_events() << '\n';
    return false;
  }
  result = {};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect_status(sllm_completion_finalize_after(operations[0], fence,
                                                    &result, &error.sink),
                     SLLM_STATUS_PUBLIC_NOT_READY,
                     "finalize before fence success", error) ||
      result.state != SLLM_COMPLETION_STATE_PENDING) {
    return false;
  }
  result = {};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  if (!expect_status(sllm_completion_wait(fence, 1000U, &result, &error.sink),
                     SLLM_STATUS_OK, "deferred segment fence wait", error) ||
      result.state != SLLM_COMPLETION_STATE_SUCCESS) {
    std::cerr << "deferred fence wait invariant failed, state=" << result.state
              << '\n';
    return false;
  }
  for (std::size_t index = 0U; index != operations.size(); ++index) {
    auto *const operation = operations[index];
    result = {};
    result.struct_size = sizeof(result);
    result.abi_version = SLLM_HIP_ABI_VERSION;
    if (!expect_status(sllm_completion_finalize_after(operation, fence, &result,
                                                      &error.sink),
                       SLLM_STATUS_OK, "deferred completion finalize", error) ||
        result.state != SLLM_COMPLETION_STATE_SUCCESS) {
      std::cerr << "deferred finalize invariant failed at " << index
                << ", state=" << result.state << '\n';
      return false;
    }
  }
  for (auto &operation : operations) {
    if (!release_completion(&operation)) {
      std::cerr << "deferred operation release invariant failed\n";
      return false;
    }
  }
  if (!release_completion(&fence) || fake_hip::live_events() != events_before ||
      !expect_status(
          sllm_queue_set_completion_mode(
              queue, SLLM_QUEUE_COMPLETION_MODE_PROFILED, &error.sink),
          SLLM_STATUS_OK, "restore profiled completion mode", error)) {
    std::cerr << "deferred fence cleanup invariant failed, events="
              << fake_hip::live_events() << '\n';
    return false;
  }
  fake_hip::set_event_record_status(hipErrorUnknown);
  fence = reinterpret_cast<sllm_completion_t *>(UINTPTR_MAX);
  if (!expect_status(sllm_queue_fence(queue, &fence, &error.sink),
                     SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
                     "queue fence record failure", error) ||
      fence != nullptr || fake_hip::live_events() != events_before) {
    return false;
  }
  fake_hip::set_event_record_status(hipSuccess);
  for (auto &plan : plans) {
    if (!expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                       SLLM_STATUS_OK, "deferred segment plan release",
                       error)) {
      return false;
    }
  }
  const bool released = release_queue(&queue) && release_buffer(&activation) &&
                        release_buffer(&scale) && release_buffer(&output) &&
                        release_context(&context);
  if (!released || fake_hip::live_events() != events_before) {
    std::cerr << "deferred final cleanup invariant failed, events="
              << fake_hip::live_events() << '\n';
    return false;
  }
  return true;
}

bool rmsnorm_execute_row_limit_and_overflow() {
  constexpr uint64_t rows[] = {UINT64_C(4294967295), UINT64_C(4294967296)};
  for (const uint64_t row_count : rows) {
    fake_hip::reset();
    sllm_context_t *context = nullptr;
    sllm_queue_t *queue = nullptr;
    sllm_buffer_t *activation = nullptr;
    sllm_buffer_t *scale = nullptr;
    sllm_buffer_t *output = nullptr;
    constexpr uint64_t activation_bytes = UINT64_C(8589934592);
    if (!create_context(&context) || !create_queue(context, &queue) ||
        !create_buffer_sized(context, activation_bytes, &activation) ||
        !create_buffer_sized(context, 2U, &scale) ||
        !create_buffer_sized(context, activation_bytes, &output)) {
      return false;
    }
    sllm_rmsnorm_desc_t descriptor = rmsnorm_descriptor(
        activation, 0U, scale, 0U, output, 0U, row_count, 1U);
    sllm_rmsnorm_plan_t *plan = nullptr;
    Error error;
    if (!expect_status(
            sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
            SLLM_STATUS_OK, "row-limit prepare", error)) {
      return false;
    }
    sllm_completion_t *completion = nullptr;
    sllm_rmsnorm_dispatch_info_t info = rmsnorm_dispatch_info();
    const sllm_status_t expected = row_count == UINT64_C(4294967296)
                                       ? SLLM_STATUS_UNSUPPORTED
                                       : SLLM_STATUS_OK;
    if (!expect_status(
            sllm_rmsnorm_execute(plan, queue, &completion, &info, &error.sink),
            expected, "row-limit execute", error) ||
        fake_hip::rmsnorm_last_row_count() !=
            (expected == SLLM_STATUS_OK ? UINT32_C(4294967295) : 0U)) {
      return false;
    }
    if (expected == SLLM_STATUS_OK) {
      sllm_completion_result_t result{};
      result.struct_size = sizeof(result);
      result.abi_version = SLLM_HIP_ABI_VERSION;
      if (!expect_status(
              sllm_completion_query(completion, &result, &error.sink),
              SLLM_STATUS_OK, "row-limit query", error) ||
          !expect_status(sllm_completion_release(&completion, &error.sink),
                         SLLM_STATUS_OK, "row-limit completion release",
                         error)) {
        return false;
      }
    }
    if (!expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                       SLLM_STATUS_OK, "row-limit plan release", error) ||
        !release_queue(&queue) || !release_buffer(&activation) ||
        !release_buffer(&scale) || !release_buffer(&output) ||
        !release_context(&context)) {
      return false;
    }
  }
  return true;
}

bool rmsnorm_execute_flattens_rank_one_through_eight() {
  constexpr std::size_t rank_count = 8U;
  constexpr uint32_t ranks[] = {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U};
  constexpr uint64_t shapes[rank_count][8U] = {
      {17U, 0U, 0U, 0U, 0U, 0U, 0U, 0U}, {3U, 17U, 0U, 0U, 0U, 0U, 0U, 0U},
      {2U, 3U, 17U, 0U, 0U, 0U, 0U, 0U}, {2U, 2U, 3U, 17U, 0U, 0U, 0U, 0U},
      {2U, 3U, 2U, 3U, 17U, 0U, 0U, 0U}, {2U, 2U, 3U, 2U, 3U, 17U, 0U, 0U},
      {2U, 3U, 2U, 2U, 3U, 2U, 17U, 0U}, {2U, 2U, 3U, 2U, 2U, 3U, 2U, 17U},
  };
  constexpr uint64_t expected_rows[] = {1U, 3U, 6U, 12U, 36U, 72U, 144U, 288U};
  for (std::size_t rank_index = 0U; rank_index != rank_count; ++rank_index) {
    fake_hip::reset();
    sllm_context_t *context = nullptr;
    sllm_queue_t *queue = nullptr;
    sllm_buffer_t *activation = nullptr;
    sllm_buffer_t *scale = nullptr;
    sllm_buffer_t *output = nullptr;
    if (!create_context(&context) || !create_queue(context, &queue) ||
        !create_buffer_sized(context, 32768U, &activation) ||
        !create_buffer_sized(context, 32768U, &scale) ||
        !create_buffer_sized(context, 32768U, &output)) {
      return false;
    }
    sllm_rmsnorm_desc_t descriptor =
        rmsnorm_descriptor(activation, 0U, scale, 0U, output, 0U, 1U, 17U);
    descriptor.activation = rmsnorm_binding_rank(
        activation, 0U, ranks[rank_index], 17U, shapes[rank_index]);
    descriptor.output = rmsnorm_binding_rank(output, 0U, ranks[rank_index], 17U,
                                             shapes[rank_index]);
    sllm_rmsnorm_plan_t *plan = nullptr;
    Error error;
    if (!expect_status(
            sllm_rmsnorm_prepare(context, &descriptor, &plan, &error.sink),
            SLLM_STATUS_OK, "rank-flatten prepare", error)) {
      return false;
    }
    sllm_completion_t *completion = nullptr;
    sllm_rmsnorm_dispatch_info_t info = rmsnorm_dispatch_info();
    if (!expect_status(
            sllm_rmsnorm_execute(plan, queue, &completion, &info, &error.sink),
            SLLM_STATUS_OK, "rank-flatten execute", error) ||
        info.row_count != expected_rows[rank_index] ||
        info.normalized_size != 17U ||
        fake_hip::rmsnorm_last_row_count() != expected_rows[rank_index]) {
      return false;
    }
    sllm_completion_result_t result{};
    result.struct_size = sizeof(result);
    result.abi_version = SLLM_HIP_ABI_VERSION;
    if (!expect_status(sllm_completion_query(completion, &result, &error.sink),
                       SLLM_STATUS_OK, "rank-flatten query", error) ||
        !expect_status(sllm_completion_release(&completion, &error.sink),
                       SLLM_STATUS_OK, "rank-flatten completion release",
                       error) ||
        !expect_status(sllm_rmsnorm_plan_release(&plan, &error.sink),
                       SLLM_STATUS_OK, "rank-flatten plan release", error) ||
        !release_queue(&queue) || !release_buffer(&activation) ||
        !release_buffer(&scale) || !release_buffer(&output) ||
        !release_context(&context)) {
      return false;
    }
  }
  return true;
}

sllm_tensor_binding_t matmul_binding(const sllm_buffer_t *const buffer,
                                     const uint64_t offset, const uint64_t rows,
                                     const uint64_t columns) {
  sllm_tensor_binding_t binding{};
  binding.struct_size = sizeof(binding);
  binding.abi_version = SLLM_HIP_ABI_VERSION;
  binding.buffer = buffer;
  binding.byte_offset = offset;
  binding.dtype = SLLM_TENSOR_DTYPE_BF16;
  binding.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  binding.rank = 2U;
  binding.shape[0] = rows;
  binding.shape[1] = columns;
  binding.stride_elements[0] = columns;
  binding.stride_elements[1] = 1U;
  return binding;
}

sllm_matmul_desc_t matmul_descriptor(
    const sllm_buffer_t *const activation, const uint64_t activation_offset,
    const sllm_buffer_t *const weight, const uint64_t weight_offset,
    const sllm_buffer_t *const output, const uint64_t output_offset,
    const uint64_t m = 3U, const uint64_t k = 5U, const uint64_t n = 7U) {
  sllm_matmul_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_MATMUL_VERSION;
  descriptor.activation = matmul_binding(activation, activation_offset, m, k);
  descriptor.weight = matmul_binding(weight, weight_offset, n, k);
  descriptor.output = matmul_binding(output, output_offset, m, n);
  return descriptor;
}

sllm_matmul_dispatch_info_t matmul_dispatch_info() {
  sllm_matmul_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_MATMUL_DISPATCH_INFO_VERSION;
  return info;
}

bool gdn_projection_bundle_launch_failure_rolls_back_accounting() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  fake_hip::set_gcn_arch_name("gfx1030");

  constexpr std::array<uint64_t, 4> widths = {8192U, 4096U, 32U, 32U};
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *activation = nullptr;
  std::array<sllm_buffer_t *, 4> weights{};
  std::array<sllm_buffer_t *, 4> outputs{};
  sllm_gdn_projection_bundle_plan_t *plan = nullptr;

  bool valid = create_context_for_arch("gfx1030", &context) &&
               create_queue(context, &queue) &&
               create_buffer_sized(context, UINT64_C(2560) * sizeof(uint16_t),
                                   &activation);
  for (std::size_t index = 0U; index != widths.size() && valid; ++index) {
    valid = create_buffer_sized(
                context, widths[index] * UINT64_C(2560) * sizeof(uint16_t),
                &weights[index]) &&
            create_buffer_sized(context, widths[index] * sizeof(uint16_t),
                                &outputs[index]);
  }

  if (valid) {
    sllm_gdn_projection_bundle_desc_t descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = SLLM_HIP_ABI_VERSION;
    descriptor.op_version = SLLM_HIP_GDN_PROJECTION_BUNDLE_VERSION;
    descriptor.activation = matmul_binding(activation, 0U, 1U, 2560U);
    for (std::size_t index = 0U; index != widths.size(); ++index) {
      descriptor.weights[index] =
          matmul_binding(weights[index], 0U, widths[index], 2560U);
      descriptor.outputs[index] =
          matmul_binding(outputs[index], 0U, 1U, widths[index]);
    }

    Error error;
    valid = expect_status(sllm_gdn_projection_bundle_prepare(
                              context, &descriptor, &plan, &error.sink),
                          SLLM_STATUS_OK, "GDN bundle launch-failure prepare",
                          error) &&
            plan != nullptr;
    if (valid) {
      sllm_gdn_projection_bundle_dispatch_info_t info{};
      info.struct_size = sizeof(info);
      info.abi_version = SLLM_HIP_ABI_VERSION;
      info.info_version = SLLM_HIP_GDN_PROJECTION_BUNDLE_DISPATCH_INFO_VERSION;
      sllm_completion_t *completion =
          reinterpret_cast<sllm_completion_t *>(static_cast<uintptr_t>(0xA1U));
      valid = expect_status(sllm_gdn_projection_bundle_execute(
                                plan, queue, &completion, &info, &error.sink),
                            SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
                            "GDN bundle stub launch failure", error) &&
              completion == nullptr;
    }
    if (plan != nullptr) {
      valid = expect_status(
                  sllm_gdn_projection_bundle_plan_release(&plan, &error.sink),
                  SLLM_STATUS_OK, "GDN bundle launch-failure plan release",
                  error) &&
              valid;
    }
  }

  if (queue != nullptr) {
    valid = release_queue(&queue) && valid;
  }
  for (sllm_buffer_t *&buffer : weights) {
    if (buffer != nullptr) {
      valid = release_buffer(&buffer) && valid;
    }
  }
  for (sllm_buffer_t *&buffer : outputs) {
    if (buffer != nullptr) {
      valid = release_buffer(&buffer) && valid;
    }
  }
  if (activation != nullptr) {
    valid = release_buffer(&activation) && valid;
  }
  if (context != nullptr) {
    valid = release_context(&context) && valid;
  }
  fake_hip::set_gcn_arch_name("gfx1201");
  return valid && fake_hip::live_events() == 0U &&
         fake_hip::live_streams() == 0U && fake_hip::live_allocations() == 0U;
}

bool mlp_gate_up_silu_bundle_launch_failure_rolls_back_accounting() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  fake_hip::set_gcn_arch_name("gfx1030");

  constexpr uint64_t hidden_size = 2560U;
  constexpr uint64_t mlp_width = 9216U;
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *gate_weight = nullptr;
  sllm_buffer_t *up_weight = nullptr;
  std::array<sllm_buffer_t *, 3> outputs{};
  sllm_mlp_gate_up_silu_bundle_plan_t *plan = nullptr;

  const uint64_t activation_bytes = hidden_size * sizeof(uint16_t);
  const uint64_t weight_bytes = mlp_width * hidden_size * sizeof(uint16_t);
  const uint64_t output_bytes = mlp_width * sizeof(uint16_t);
  bool valid = create_context_for_arch("gfx1030", &context) &&
               create_queue(context, &queue) &&
               create_buffer_sized(context, activation_bytes, &activation) &&
               create_buffer_sized(context, weight_bytes, &gate_weight) &&
               create_buffer_sized(context, weight_bytes, &up_weight);
  for (sllm_buffer_t *&output : outputs) {
    if (valid) {
      valid = create_buffer_sized(context, output_bytes, &output);
    }
  }

  if (valid) {
    sllm_mlp_gate_up_silu_bundle_desc_t descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = SLLM_HIP_ABI_VERSION;
    descriptor.op_version = SLLM_HIP_MLP_GATE_UP_SILU_BUNDLE_VERSION;
    descriptor.activation = matmul_binding(activation, 0U, 1U, hidden_size);
    descriptor.gate_weight =
        matmul_binding(gate_weight, 0U, mlp_width, hidden_size);
    descriptor.up_weight =
        matmul_binding(up_weight, 0U, mlp_width, hidden_size);
    descriptor.gate_output = matmul_binding(outputs[0], 0U, 1U, mlp_width);
    descriptor.up_output = matmul_binding(outputs[1], 0U, 1U, mlp_width);
    descriptor.silu_output = matmul_binding(outputs[2], 0U, 1U, mlp_width);

    Error error;
    valid = expect_status(sllm_mlp_gate_up_silu_bundle_prepare(
                              context, &descriptor, &plan, &error.sink),
                          SLLM_STATUS_OK, "MLP bundle launch-failure prepare",
                          error) &&
            plan != nullptr;
    if (valid) {
      sllm_mlp_gate_up_silu_bundle_dispatch_info_t info{};
      info.struct_size = sizeof(info);
      info.abi_version = SLLM_HIP_ABI_VERSION;
      info.info_version =
          SLLM_HIP_MLP_GATE_UP_SILU_BUNDLE_DISPATCH_INFO_VERSION;
      sllm_completion_t *completion =
          reinterpret_cast<sllm_completion_t *>(static_cast<uintptr_t>(0xA2U));
      valid = expect_status(sllm_mlp_gate_up_silu_bundle_execute(
                                plan, queue, &completion, &info, &error.sink),
                            SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
                            "MLP bundle stub launch failure", error) &&
              completion == nullptr;
    }
    if (plan != nullptr) {
      valid = expect_status(
                  sllm_mlp_gate_up_silu_bundle_plan_release(&plan, &error.sink),
                  SLLM_STATUS_OK, "MLP bundle launch-failure plan release",
                  error) &&
              valid;
    }
  }

  if (queue != nullptr) {
    valid = release_queue(&queue) && valid;
  }
  for (sllm_buffer_t *&output : outputs) {
    if (output != nullptr) {
      valid = release_buffer(&output) && valid;
    }
  }
  if (gate_weight != nullptr) {
    valid = release_buffer(&gate_weight) && valid;
  }
  if (up_weight != nullptr) {
    valid = release_buffer(&up_weight) && valid;
  }
  if (activation != nullptr) {
    valid = release_buffer(&activation) && valid;
  }
  if (context != nullptr) {
    valid = release_context(&context) && valid;
  }
  fake_hip::set_gcn_arch_name("gfx1201");
  return valid && fake_hip::live_events() == 0U &&
         fake_hip::live_streams() == 0U && fake_hip::live_allocations() == 0U;
}

bool context_device_property_snapshot_contract() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  fake_hip::set_gcn_arch_name("gfx9999");
  sllm_context_t *rejected_context = nullptr;
  sllm_context_create_info_t rejected_info{};
  rejected_info.struct_size = sizeof(rejected_info);
  rejected_info.abi_version = SLLM_HIP_ABI_VERSION;
  rejected_info.device_index = 0U;
  std::strncpy(rejected_info.expected_gcn_arch_name, "gfx1201",
               sizeof(rejected_info.expected_gcn_arch_name) - 1U);
  Error rejected_error;
  const sllm_status_t rejected_status = sllm_context_create(
      &rejected_info, &rejected_context, &rejected_error.sink);
  if (!expect_status(rejected_status, SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
                     "context property mismatch", rejected_error) ||
      rejected_context != nullptr || fake_hip::device_property_calls() != 1U) {
    std::cerr << "context property mismatch cleanup contract failed\n";
    if (rejected_context != nullptr) {
      (void)release_context(&rejected_context);
    }
    fake_hip::set_gcn_arch_name("gfx1201");
    return false;
  }

  fake_hip::set_gcn_arch_name("gfx1201");
  sllm_context_t *first_context = nullptr;
  if (!create_context(&first_context)) {
    fake_hip::set_gcn_arch_name("gfx1201");
    return false;
  }
  fake_hip::set_gcn_arch_name("gfx1030");
  sllm_context_t *second_context = nullptr;
  if (!create_context_for_arch("gfx1030", &second_context)) {
    (void)release_context(&first_context);
    fake_hip::set_gcn_arch_name("gfx1201");
    return false;
  }
  if (fake_hip::device_property_calls() != 3U) {
    std::cerr << "context property snapshot queried an unexpected count\n";
    (void)release_context(&second_context);
    (void)release_context(&first_context);
    fake_hip::set_gcn_arch_name("gfx1201");
    return false;
  }

  sllm_queue_t *first_queue = nullptr;
  sllm_queue_t *second_queue = nullptr;
  sllm_buffer_t *first_input0 = nullptr;
  sllm_buffer_t *first_input1 = nullptr;
  sllm_buffer_t *first_output = nullptr;
  sllm_buffer_t *second_input0 = nullptr;
  sllm_buffer_t *second_input1 = nullptr;
  sllm_buffer_t *second_output = nullptr;
  const uint64_t buffer_bytes = 64U;
  const bool resources_ready =
      create_queue(first_context, &first_queue) &&
      create_queue(second_context, &second_queue) &&
      create_buffer_sized(first_context, buffer_bytes, &first_input0) &&
      create_buffer_sized(first_context, buffer_bytes, &first_input1) &&
      create_buffer_sized(first_context, buffer_bytes, &first_output) &&
      create_buffer_sized(second_context, buffer_bytes, &second_input0) &&
      create_buffer_sized(second_context, buffer_bytes, &second_input1) &&
      create_buffer_sized(second_context, buffer_bytes, &second_output);
  bool valid = resources_ready;
  sllm_elementwise_plan_t *first_plan = nullptr;
  sllm_elementwise_plan_t *second_plan = nullptr;
  sllm_completion_t *first_completion = nullptr;
  sllm_completion_t *second_completion = nullptr;
  if (resources_ready) {
    Error error;
    auto first_descriptor =
        elementwise_descriptor(SLLM_ELEMENTWISE_OPERATION_COPY, first_input0,
                               first_input1, first_output, 7U);
    auto second_descriptor =
        elementwise_descriptor(SLLM_ELEMENTWISE_OPERATION_COPY, second_input0,
                               second_input1, second_output, 7U);
    auto first_info = elementwise_dispatch_info();
    auto second_info = elementwise_dispatch_info();
    valid =
        expect_status(sllm_elementwise_prepare(first_context, &first_descriptor,
                                               &first_plan, &error.sink),
                      SLLM_STATUS_OK, "snapshot first elementwise prepare",
                      error) &&
        expect_status(
            sllm_elementwise_execute(first_plan, first_queue, &first_completion,
                                     &first_info, &error.sink),
            SLLM_STATUS_OK, "snapshot first elementwise execute", error) &&
        first_completion != nullptr &&
        std::strcmp(first_info.gcn_arch_name, "gfx1201") == 0;
    if (first_completion != nullptr) {
      valid = query_completion(first_completion, SLLM_STATUS_OK) &&
              release_completion(&first_completion) && valid;
    }
    if (first_plan != nullptr) {
      valid =
          expect_status(sllm_elementwise_plan_release(&first_plan, &error.sink),
                        SLLM_STATUS_OK,
                        "snapshot first elementwise plan release", error) &&
          valid;
    }
    if (valid) {
      valid = expect_status(sllm_elementwise_prepare(second_context,
                                                     &second_descriptor,
                                                     &second_plan, &error.sink),
                            SLLM_STATUS_OK,
                            "snapshot second elementwise prepare", error) &&
              expect_status(sllm_elementwise_execute(second_plan, second_queue,
                                                     &second_completion,
                                                     &second_info, &error.sink),
                            SLLM_STATUS_OK,
                            "snapshot second elementwise execute", error) &&
              second_completion != nullptr &&
              std::strcmp(second_info.gcn_arch_name, "gfx1030") == 0;
    }
    if (second_completion != nullptr) {
      valid = query_completion(second_completion, SLLM_STATUS_OK) &&
              release_completion(&second_completion) && valid;
    }
    if (second_plan != nullptr) {
      valid = expect_status(
                  sllm_elementwise_plan_release(&second_plan, &error.sink),
                  SLLM_STATUS_OK, "snapshot second elementwise plan release",
                  error) &&
              valid;
    }
  }
  valid = fake_hip::device_property_calls() == 3U && valid;
  if (first_queue != nullptr && !release_queue(&first_queue)) {
    valid = false;
  }
  if (second_queue != nullptr && !release_queue(&second_queue)) {
    valid = false;
  }
  if (first_input0 != nullptr && !release_buffer(&first_input0)) {
    valid = false;
  }
  if (first_input1 != nullptr && !release_buffer(&first_input1)) {
    valid = false;
  }
  if (first_output != nullptr && !release_buffer(&first_output)) {
    valid = false;
  }
  if (second_input0 != nullptr && !release_buffer(&second_input0)) {
    valid = false;
  }
  if (second_input1 != nullptr && !release_buffer(&second_input1)) {
    valid = false;
  }
  if (second_output != nullptr && !release_buffer(&second_output)) {
    valid = false;
  }
  if (second_context != nullptr && !release_context(&second_context)) {
    valid = false;
  }
  if (first_context != nullptr && !release_context(&first_context)) {
    valid = false;
  }
  fake_hip::set_gcn_arch_name("gfx1201");
  return valid && fake_hip::live_events() == 0U &&
         fake_hip::live_streams() == 0U && fake_hip::live_allocations() == 0U;
}

bool matmul_prepare_execute_and_negative_contract() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  sllm_context_t *context = nullptr;
  sllm_context_t *other_context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *weight = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_context(&other_context) ||
      !create_queue(context, &queue) ||
      !create_buffer_sized(context, 1024U, &activation) ||
      !create_buffer_sized(context, 1024U, &weight) ||
      !create_buffer_sized(context, 1024U, &output)) {
    return false;
  }
  Error error;
  auto descriptor = matmul_descriptor(activation, 0U, weight, 0U, output, 0U);
  sllm_matmul_plan_t *plan = nullptr;
  descriptor.reserved[0] = 1U;
  if (!expect_status(
          sllm_matmul_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_RESERVED_NONZERO, "matmul descriptor reserved rejection",
          error) ||
      plan != nullptr) {
    return false;
  }

  descriptor = matmul_descriptor(activation, 0U, weight, 0U, output, 0U);
  descriptor.activation.dtype = SLLM_TENSOR_DTYPE_F32;
  if (!expect_status(
          sllm_matmul_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_UNSUPPORTED_DTYPE, "matmul dtype rejection", error) ||
      plan != nullptr) {
    return false;
  }

  descriptor = matmul_descriptor(activation, 0U, weight, 0U, output, 0U);
  descriptor.weight.shape[1] = 4U;
  descriptor.weight.stride_elements[0] = 4U;
  if (!expect_status(
          sllm_matmul_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_SHAPE_MISMATCH, "matmul shape rejection", error) ||
      plan != nullptr) {
    return false;
  }

  descriptor = matmul_descriptor(activation, 0U, weight, 0U, activation, 0U);
  if (!expect_status(
          sllm_matmul_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_ALIAS_OVERLAP, "matmul overlap rejection", error) ||
      plan != nullptr) {
    return false;
  }

  descriptor = matmul_descriptor(activation, 0U, weight, 0U, output, 1024U);
  if (!expect_status(
          sllm_matmul_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_BUFFER_OUT_OF_BOUNDS, "matmul bounds rejection", error) ||
      plan != nullptr) {
    return false;
  }

  descriptor = matmul_descriptor(activation, 1U, weight, 0U, output, 0U);
  if (!expect_status(
          sllm_matmul_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_MISALIGNED_OFFSET, "matmul alignment rejection", error) ||
      plan != nullptr) {
    return false;
  }

  descriptor = matmul_descriptor(activation, 0U, weight, 0U, output, 0U);
  if (!expect_status(
          sllm_matmul_prepare(other_context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_CONTEXT_OR_DEVICE_MISMATCH, "matmul context rejection",
          error) ||
      plan != nullptr) {
    return false;
  }

  if (!expect_status(
          sllm_matmul_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "matmul prepare", error) ||
      plan == nullptr) {
    return false;
  }
  sllm_completion_t *sentinel_completion =
      reinterpret_cast<sllm_completion_t *>(static_cast<uintptr_t>(0x55U));
  auto invalid_info = matmul_dispatch_info();
  invalid_info.reserved[0] = 1U;
  if (!expect_status(sllm_matmul_execute(plan, queue, &sentinel_completion,
                                         &invalid_info, &error.sink),
                     SLLM_STATUS_RESERVED_NONZERO,
                     "matmul dispatch reserved rejection", error) ||
      sentinel_completion != reinterpret_cast<sllm_completion_t *>(
                                 static_cast<uintptr_t>(0x55U)) ||
      fake_hip::matmul_launch_calls() != 0U) {
    return false;
  }

  sllm_completion_t *completion = nullptr;
  auto info = matmul_dispatch_info();
  if (!expect_status(
          sllm_matmul_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_OK, "matmul execute", error) ||
      completion == nullptr || info.backend != SLLM_BACKEND_HIP ||
      info.dispatch_count != 1U ||
      info.kernel_id != SLLM_HIP_MATMUL_KERNEL_ID_SERIAL_ROWS_BF16_FP32_V1 ||
      info.workgroup_size_x != SLLM_HIP_MATMUL_WORKGROUP_SIZE ||
      info.grid_size_x != 7U || info.m != 3U || info.k != 5U || info.n != 7U ||
      info.output_elements != 21U || info.fallback_allowed != 0U ||
      info.fallback_used != 0U ||
      std::strcmp(info.kernel_symbol,
                  "matmul.bf16_fp32.decode.serial_rows.v1") != 0 ||
      std::strcmp(info.device_symbol,
                  "sllm_matmul_bf16_fp32_decode_serial_rows_v1") != 0 ||
      std::strcmp(info.gcn_arch_name, "gfx1201") != 0 ||
      fake_hip::matmul_launch_calls() != 1U ||
      fake_hip::matmul_last_m() != 3U || fake_hip::matmul_last_k() != 5U ||
      fake_hip::matmul_last_n() != 7U ||
      fake_hip::matmul_last_output_elements() != 21U) {
    std::cerr << "matmul dispatch mismatch completion=" << completion
              << " backend=" << info.backend
              << " dispatch_count=" << info.dispatch_count
              << " kernel_id=" << info.kernel_id << " grid=" << info.grid_size_x
              << " mkn=" << info.m << "," << info.k << "," << info.n
              << " launches=" << fake_hip::matmul_launch_calls() << "\n";
    return false;
  }
  if (!query_completion(completion, SLLM_STATUS_OK) ||
      !release_completion(&completion) ||
      !expect_status(sllm_matmul_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "matmul plan release", error) ||
      plan != nullptr || !release_queue(&queue) ||
      !release_buffer(&activation) || !release_buffer(&weight) ||
      !release_buffer(&output) || !release_context(&other_context) ||
      !release_context(&context)) {
    return false;
  }
  return fake_hip::live_events() == 0U && fake_hip::live_streams() == 0U &&
         fake_hip::live_allocations() == 0U;
}

bool matmul_mxfp_weight_activation_descriptor_contract() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  sllm_context_t *context = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *mxfp8_weight = nullptr;
  sllm_buffer_t *mxfp6_weight = nullptr;
  sllm_buffer_t *output = nullptr;
  constexpr uint64_t m = 3U;
  constexpr uint64_t n = 7U;
  constexpr uint64_t k = 64U;
  constexpr uint64_t blocks = n * k / 32U;
  if (!create_context(&context) ||
      !create_buffer_sized(context, m * k * sizeof(uint16_t), &activation) ||
      !create_buffer_sized(context, n * k + blocks, &mxfp8_weight) ||
      !create_buffer_sized(context, n * k * 3U / 4U + blocks, &mxfp6_weight) ||
      !create_buffer_sized(context, m * n * sizeof(uint16_t), &output)) {
    return false;
  }

  Error error;
  sllm_matmul_plan_t *plan = nullptr;
  auto prepare_and_release = [&](const uint32_t version,
                                 sllm_buffer_t *const weight,
                                 const uint32_t dtype, const uint32_t encoding,
                                 const char *const label) {
    auto descriptor =
        matmul_descriptor(activation, 0U, weight, 0U, output, 0U, m, k, n);
    descriptor.op_version = version;
    descriptor.weight.dtype = dtype;
    descriptor.weight.encoding = encoding;
    return expect_status(
               sllm_matmul_prepare(context, &descriptor, &plan, &error.sink),
               SLLM_STATUS_OK, label, error) &&
           plan != nullptr &&
           expect_status(sllm_matmul_plan_release(&plan, &error.sink),
                         SLLM_STATUS_OK, label, error) &&
           plan == nullptr;
  };
  bool valid = prepare_and_release(SLLM_HIP_MATMUL_MXFP8_W8A8_VERSION,
                                   mxfp8_weight, SLLM_TENSOR_DTYPE_F8_E4M3_FN,
                                   SLLM_TENSOR_ENCODING_MXFP8_BLOCK32_E8M0,
                                   "MXFP8 W8A8 prepare");
  valid =
      valid && prepare_and_release(SLLM_HIP_MATMUL_MXFP6_W6A6_VERSION,
                                   mxfp6_weight, SLLM_TENSOR_DTYPE_U8,
                                   SLLM_TENSOR_ENCODING_MXFP6_E3M2_BLOCK32_E8M0,
                                   "MXFP6 W6A6 prepare");

  for (const uint64_t nonaligned_k : {UINT64_C(31), UINT64_C(33)}) {
    auto nonaligned = matmul_descriptor(activation, 0U, mxfp8_weight, 0U,
                                        output, 0U, 1U, nonaligned_k, 1U);
    nonaligned.op_version = SLLM_HIP_MATMUL_MXFP8_W8A8_VERSION;
    nonaligned.weight.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
    nonaligned.weight.encoding = SLLM_TENSOR_ENCODING_MXFP8_BLOCK32_E8M0;
    valid =
        valid &&
        expect_status(
            sllm_matmul_prepare(context, &nonaligned, &plan, &error.sink),
            SLLM_STATUS_SHAPE_MISMATCH, "MXFP8 non-block K rejection", error) &&
        plan == nullptr;
    nonaligned.op_version = SLLM_HIP_MATMUL_MXFP6_W6A6_VERSION;
    nonaligned.weight.dtype = SLLM_TENSOR_DTYPE_U8;
    nonaligned.weight.encoding = SLLM_TENSOR_ENCODING_MXFP6_E3M2_BLOCK32_E8M0;
    valid =
        valid &&
        expect_status(
            sllm_matmul_prepare(context, &nonaligned, &plan, &error.sink),
            SLLM_STATUS_SHAPE_MISMATCH, "MXFP6 non-block K rejection", error) &&
        plan == nullptr;
  }

  valid = release_buffer(&activation) && release_buffer(&mxfp8_weight) &&
          release_buffer(&mxfp6_weight) && release_buffer(&output) &&
          release_context(&context) && valid;
  return valid && fake_hip::live_events() == 0U &&
         fake_hip::live_streams() == 0U && fake_hip::live_allocations() == 0U;
}

bool matmul_mxfp_prefill_selector_contract() {
  constexpr const char *const baseline = "SLLM_MX_WA_PREFILL_FORCE_BASELINE";
  constexpr const char *const mxfp8_tiled16 =
      "SLLM_MXFP8_PREFILL_FORCE_TILED16";
  constexpr const char *const mxfp6_row8 = "SLLM_MXFP6_PREFILL_FORCE_ROW8";
  constexpr const char *const mmq_columns =
      "SLLM_MX_WA_PREFILL_FORCE_MMQ_COLUMNS";
  constexpr const char *const gfx1030_mmq_columns =
      "SLLM_MXFP8_PREFILL_FORCE_MMQ_GFX1030_COLUMNS";
  constexpr const char *const gfx1030_phase69 =
      "SLLM_MXFP8_PREFILL_FORCE_MMQ_GFX1030_PHASE69";
  constexpr const char *const mxfp8_row8 = "SLLM_MXFP8_PREFILL_FORCE_ROW8";
  constexpr const char *const mxfp8_wmma =
      "SLLM_MXFP8_PREFILL_FORCE_WMMA_GFX1201";
  constexpr const char *const mxfp8_wmma_n16 =
      "SLLM_MXFP8_PREFILL_FORCE_WMMA_N16_GFX1201";
  constexpr const char *const mxfp8_wmma_4wave =
      "SLLM_MXFP8_PREFILL_FORCE_WMMA_4W_GFX1201";
  constexpr const char *const mxfp8_wmma_lds_pad =
      "SLLM_MXFP8_PREFILL_FORCE_WMMA_LDS_PAD_GFX1201";
  constexpr const char *const mxfp8_wmma_direct_weight =
      "SLLM_MXFP8_PREFILL_FORCE_WMMA_DIRECT_WEIGHT_GFX1201";
  constexpr const char *const mxfp8_wmma_direct_activation =
      "SLLM_MXFP8_PREFILL_FORCE_WMMA_DIRECT_ACTIVATION_GFX1201";
  constexpr const char *const mxfp8_wmma_direct_both =
      "SLLM_MXFP8_PREFILL_FORCE_WMMA_DIRECT_BOTH_GFX1201";
  constexpr const char *const mxfp8_wmma_n128_direct_both =
      "SLLM_MXFP8_PREFILL_FORCE_WMMA_N128_DIRECT_BOTH_GFX1201";
  const char *const old_baseline = std::getenv(baseline);
  const bool had_baseline = old_baseline != nullptr;
  const std::string old_baseline_value = had_baseline ? old_baseline : "";
  const char *const old_mxfp8_tiled16 = std::getenv(mxfp8_tiled16);
  const bool had_mxfp8_tiled16 = old_mxfp8_tiled16 != nullptr;
  const std::string old_mxfp8_tiled16_value =
      had_mxfp8_tiled16 ? old_mxfp8_tiled16 : "";
  const char *const old_mxfp6_row8 = std::getenv(mxfp6_row8);
  const bool had_mxfp6_row8 = old_mxfp6_row8 != nullptr;
  const std::string old_mxfp6_row8_value = had_mxfp6_row8 ? old_mxfp6_row8 : "";
  const char *const old_mmq_columns = std::getenv(mmq_columns);
  const bool had_mmq_columns = old_mmq_columns != nullptr;
  const std::string old_mmq_columns_value =
      had_mmq_columns ? old_mmq_columns : "";
  const char *const old_gfx1030_mmq_columns = std::getenv(gfx1030_mmq_columns);
  const bool had_gfx1030_mmq_columns = old_gfx1030_mmq_columns != nullptr;
  const std::string old_gfx1030_mmq_columns_value =
      had_gfx1030_mmq_columns ? old_gfx1030_mmq_columns : "";
  const char *const old_gfx1030_phase69 = std::getenv(gfx1030_phase69);
  const bool had_gfx1030_phase69 = old_gfx1030_phase69 != nullptr;
  const std::string old_gfx1030_phase69_value =
      had_gfx1030_phase69 ? old_gfx1030_phase69 : "";
  const char *const old_mxfp8_row8 = std::getenv(mxfp8_row8);
  const bool had_mxfp8_row8 = old_mxfp8_row8 != nullptr;
  const std::string old_mxfp8_row8_value = had_mxfp8_row8 ? old_mxfp8_row8 : "";
  const char *const old_mxfp8_wmma = std::getenv(mxfp8_wmma);
  const bool had_mxfp8_wmma = old_mxfp8_wmma != nullptr;
  const std::string old_mxfp8_wmma_value = had_mxfp8_wmma ? old_mxfp8_wmma : "";
  const char *const old_mxfp8_wmma_n16 = std::getenv(mxfp8_wmma_n16);
  const bool had_mxfp8_wmma_n16 = old_mxfp8_wmma_n16 != nullptr;
  const std::string old_mxfp8_wmma_n16_value =
      had_mxfp8_wmma_n16 ? old_mxfp8_wmma_n16 : "";
  const char *const old_mxfp8_wmma_4wave = std::getenv(mxfp8_wmma_4wave);
  const bool had_mxfp8_wmma_4wave = old_mxfp8_wmma_4wave != nullptr;
  const std::string old_mxfp8_wmma_4wave_value =
      had_mxfp8_wmma_4wave ? old_mxfp8_wmma_4wave : "";
  const char *const old_mxfp8_wmma_lds_pad = std::getenv(mxfp8_wmma_lds_pad);
  const bool had_mxfp8_wmma_lds_pad = old_mxfp8_wmma_lds_pad != nullptr;
  const std::string old_mxfp8_wmma_lds_pad_value =
      had_mxfp8_wmma_lds_pad ? old_mxfp8_wmma_lds_pad : "";
  const char *const old_mxfp8_wmma_direct_weight =
      std::getenv(mxfp8_wmma_direct_weight);
  const bool had_mxfp8_wmma_direct_weight =
      old_mxfp8_wmma_direct_weight != nullptr;
  const std::string old_mxfp8_wmma_direct_weight_value =
      had_mxfp8_wmma_direct_weight ? old_mxfp8_wmma_direct_weight : "";
  const char *const old_mxfp8_wmma_direct_activation =
      std::getenv(mxfp8_wmma_direct_activation);
  const bool had_mxfp8_wmma_direct_activation =
      old_mxfp8_wmma_direct_activation != nullptr;
  const std::string old_mxfp8_wmma_direct_activation_value =
      had_mxfp8_wmma_direct_activation ? old_mxfp8_wmma_direct_activation : "";
  const char *const old_mxfp8_wmma_direct_both =
      std::getenv(mxfp8_wmma_direct_both);
  const bool had_mxfp8_wmma_direct_both = old_mxfp8_wmma_direct_both != nullptr;
  const std::string old_mxfp8_wmma_direct_both_value =
      had_mxfp8_wmma_direct_both ? old_mxfp8_wmma_direct_both : "";
  const char *const old_mxfp8_wmma_n128_direct_both =
      std::getenv(mxfp8_wmma_n128_direct_both);
  const bool had_mxfp8_wmma_n128_direct_both =
      old_mxfp8_wmma_n128_direct_both != nullptr;
  const std::string old_mxfp8_wmma_n128_direct_both_value =
      had_mxfp8_wmma_n128_direct_both ? old_mxfp8_wmma_n128_direct_both : "";
  const auto restore_environment = [&]() {
    if (had_baseline) {
      setenv(baseline, old_baseline_value.c_str(), 1);
    } else {
      unsetenv(baseline);
    }
    if (had_mxfp8_tiled16) {
      setenv(mxfp8_tiled16, old_mxfp8_tiled16_value.c_str(), 1);
    } else {
      unsetenv(mxfp8_tiled16);
    }
    if (had_mxfp6_row8) {
      setenv(mxfp6_row8, old_mxfp6_row8_value.c_str(), 1);
    } else {
      unsetenv(mxfp6_row8);
    }
    if (had_mmq_columns) {
      setenv(mmq_columns, old_mmq_columns_value.c_str(), 1);
    } else {
      unsetenv(mmq_columns);
    }
    if (had_gfx1030_mmq_columns) {
      setenv(gfx1030_mmq_columns, old_gfx1030_mmq_columns_value.c_str(), 1);
    } else {
      unsetenv(gfx1030_mmq_columns);
    }
    if (had_gfx1030_phase69) {
      setenv(gfx1030_phase69, old_gfx1030_phase69_value.c_str(), 1);
    } else {
      unsetenv(gfx1030_phase69);
    }
    if (had_mxfp8_row8) {
      setenv(mxfp8_row8, old_mxfp8_row8_value.c_str(), 1);
    } else {
      unsetenv(mxfp8_row8);
    }
    if (had_mxfp8_wmma) {
      setenv(mxfp8_wmma, old_mxfp8_wmma_value.c_str(), 1);
    } else {
      unsetenv(mxfp8_wmma);
    }
    if (had_mxfp8_wmma_n16) {
      setenv(mxfp8_wmma_n16, old_mxfp8_wmma_n16_value.c_str(), 1);
    } else {
      unsetenv(mxfp8_wmma_n16);
    }
    if (had_mxfp8_wmma_4wave) {
      setenv(mxfp8_wmma_4wave, old_mxfp8_wmma_4wave_value.c_str(), 1);
    } else {
      unsetenv(mxfp8_wmma_4wave);
    }
    if (had_mxfp8_wmma_lds_pad) {
      setenv(mxfp8_wmma_lds_pad, old_mxfp8_wmma_lds_pad_value.c_str(), 1);
    } else {
      unsetenv(mxfp8_wmma_lds_pad);
    }
    if (had_mxfp8_wmma_direct_weight) {
      setenv(mxfp8_wmma_direct_weight,
             old_mxfp8_wmma_direct_weight_value.c_str(), 1);
    } else {
      unsetenv(mxfp8_wmma_direct_weight);
    }
    if (had_mxfp8_wmma_direct_activation) {
      setenv(mxfp8_wmma_direct_activation,
             old_mxfp8_wmma_direct_activation_value.c_str(), 1);
    } else {
      unsetenv(mxfp8_wmma_direct_activation);
    }
    if (had_mxfp8_wmma_direct_both) {
      setenv(mxfp8_wmma_direct_both, old_mxfp8_wmma_direct_both_value.c_str(),
             1);
    } else {
      unsetenv(mxfp8_wmma_direct_both);
    }
    if (had_mxfp8_wmma_n128_direct_both) {
      setenv(mxfp8_wmma_n128_direct_both,
             old_mxfp8_wmma_n128_direct_both_value.c_str(), 1);
    } else {
      unsetenv(mxfp8_wmma_n128_direct_both);
    }
  };

  unsetenv(baseline);
  unsetenv(mxfp8_tiled16);
  unsetenv(mxfp6_row8);
  unsetenv(mmq_columns);
  unsetenv(gfx1030_mmq_columns);
  unsetenv(mxfp8_row8);
  unsetenv(mxfp8_wmma);
  unsetenv(mxfp8_wmma_n16);
  unsetenv(mxfp8_wmma_4wave);
  unsetenv(mxfp8_wmma_lds_pad);
  unsetenv(mxfp8_wmma_direct_weight);
  unsetenv(mxfp8_wmma_direct_activation);
  unsetenv(mxfp8_wmma_direct_both);
  unsetenv(mxfp8_wmma_n128_direct_both);
  bool valid =
      sllm_matmul_kernel::select_mxfp8_variant(1U) ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8Decode &&
      sllm_matmul_kernel::select_mxfp6_variant(1U) ==
          sllm_matmul_kernel::KernelVariant::Mxfp6W6A6Decode &&
      sllm_matmul_kernel::select_mxfp8_variant(2U) ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8 &&
      sllm_matmul_kernel::select_mxfp6_variant(17U) ==
          sllm_matmul_kernel::KernelVariant::Mxfp6W6A6PrefillTiled16 &&
      sllm_matmul_kernel::select_mxfp8_variant(127U, 2560U, 9216U, "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8 &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 2560U, 9216U, "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::
              Mxfp8W8A8PrefillWmmaN128DirectBoth &&
      sllm_matmul_kernel::select_mxfp8_variant(129U, 2560U, 9216U, "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillWmmaDirectWeight &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 4096U, 12288U,
                                               "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::
              Mxfp8W8A8PrefillWmmaN128DirectBoth &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 12288U, 4096U,
                                               "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::
              Mxfp8W8A8PrefillWmmaN128DirectBoth &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 11264U, 4096U,
                                               "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::
              Mxfp8W8A8PrefillWmmaN128DirectBoth &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 2560U, 1024U, "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::
              Mxfp8W8A8PrefillWmmaN128DirectBoth &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 2560U, 512U, "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::
              Mxfp8W8A8PrefillWmmaN128DirectBoth &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 2560U, 32U, "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8 &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 2560U, 248320U,
                                               "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8;

  for (const char *const target : {"gfx1200", "gfx942", "gfx9999"}) {
    valid =
        valid &&
        sllm_matmul_kernel::select_mxfp8_variant(128U, 2560U, 9216U, target) ==
            sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8;
  }
  valid =
      valid &&
      sllm_matmul_kernel::select_mxfp8_variant(127U, 2560U, 9216U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8 &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 2047U, 9216U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8 &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 2048U, 2559U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8 &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 2048U, 2560U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::
              Mxfp8W8A8PrefillMmqGfx1030Vector32 &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 2560U, 9216U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::
              Mxfp8W8A8PrefillMmqGfx1030Vector32 &&
      sllm_matmul_kernel::select_mxfp8_variant(2048U, 9216U, 2560U,
                                               "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::
              Mxfp8W8A8PrefillMmqGfx1030Vector32 &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 2560U, 1024U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8 &&
      sllm_matmul_kernel::select_mxfp8_variant(511U, 2560U, 1024U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8 &&
      sllm_matmul_kernel::select_mxfp8_variant(512U, 2560U, 1024U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::
              Mxfp8W8A8PrefillMmqGfx1030Vector32 &&
      sllm_matmul_kernel::select_mxfp8_variant(512U, 2560U, 1025U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8 &&
      sllm_matmul_kernel::select_mxfp8_variant(512U, 2560U, 16384U,
                                               "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::
              Mxfp8W8A8PrefillMmqGfx1030Vector32 &&
      sllm_matmul_kernel::select_mxfp8_variant(512U, 2560U, 16385U,
                                               "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8 &&
      sllm_matmul_kernel::select_mxfp8_variant(512U, 2560U, 248320U,
                                               "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8;

  setenv(mxfp8_wmma, "1", 1);
  valid = valid &&
          sllm_matmul_kernel::select_mxfp8_variant(3U, 64U, 7U, "gfx1201") ==
              sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillWmmaN64 &&
          sllm_matmul_kernel::select_mxfp8_variant(1U, 64U, 7U, "gfx1201") ==
              sllm_matmul_kernel::KernelVariant::Mxfp8W8A8Decode &&
          sllm_matmul_kernel::select_mxfp8_variant(3U, 64U, 7U, "gfx1030") ==
              sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8;
  setenv(mxfp8_row8, "1", 1);
  valid = valid && sllm_matmul_kernel::select_mxfp8_variant(128U, 2560U, 9216U,
                                                            "gfx1201") ==
                       sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8;
  unsetenv(mxfp8_row8);
  unsetenv(mxfp8_wmma);

  setenv(mxfp8_wmma_n16, "1", 1);
  valid = valid &&
          sllm_matmul_kernel::select_mxfp8_variant(3U, 64U, 7U, "gfx1201") ==
              sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillWmmaN16;
  unsetenv(mxfp8_wmma_n16);

  setenv(mxfp8_wmma_4wave, "1", 1);
  valid =
      valid &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 2560U, 9216U, "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillWmma4Wave &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 2560U, 9217U, "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8 &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 2560U, 9216U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Vector32;
  unsetenv(mxfp8_wmma_4wave);

  setenv(mxfp8_wmma_lds_pad, "1", 1);
  valid =
      valid &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 2560U, 9216U, "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillWmmaLdsPad;
  unsetenv(mxfp8_wmma_lds_pad);

  setenv(mxfp8_wmma_direct_weight, "1", 1);
  valid =
      valid &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 9216U, 2560U, "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillWmmaDirectWeight;
  unsetenv(mxfp8_wmma_direct_weight);

  setenv(mxfp8_wmma_direct_activation, "1", 1);
  valid =
      valid &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 9216U, 2560U, "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::
              Mxfp8W8A8PrefillWmmaDirectActivation &&
      sllm_matmul_kernel::select_mxfp8_variant(127U, 9216U, 2560U, "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillWmmaN64 &&
      sllm_matmul_kernel::select_mxfp8_variant(129U, 9216U, 2560U, "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillWmmaN64 &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 9216U, 2560U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Vector32;
  unsetenv(mxfp8_wmma_direct_activation);

  setenv(mxfp8_wmma_direct_both, "1", 1);
  valid =
      valid &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 2560U, 9216U, "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillWmmaDirectBoth &&
      sllm_matmul_kernel::select_mxfp8_variant(129U, 2560U, 9216U, "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillWmmaN64;
  unsetenv(mxfp8_wmma_direct_both);

  setenv(mxfp8_wmma_n128_direct_both, "1", 1);
  valid =
      valid &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 2560U, 128U, "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::
              Mxfp8W8A8PrefillWmmaN128DirectBoth &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 2560U, 127U, "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8 &&
      sllm_matmul_kernel::select_mxfp8_variant(127U, 2560U, 128U, "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8 &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 2560U, 128U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8;
  unsetenv(mxfp8_wmma_n128_direct_both);

  setenv(mmq_columns, "4", 1);
  valid = valid &&
          sllm_matmul_kernel::select_mxfp8_variant(17U) ==
              sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillMmqCol4 &&
          sllm_matmul_kernel::select_mxfp6_variant(17U) ==
              sllm_matmul_kernel::KernelVariant::Mxfp6W6A6PrefillMmqCol4;
  setenv(mmq_columns, "8", 1);
  valid = valid &&
          sllm_matmul_kernel::select_mxfp8_variant(17U) ==
              sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillMmqCol8 &&
          sllm_matmul_kernel::select_mxfp6_variant(17U) ==
              sllm_matmul_kernel::KernelVariant::Mxfp6W6A6PrefillMmqCol8;

  unsetenv(mmq_columns);
  setenv(gfx1030_mmq_columns, "16", 1);
  valid =
      valid &&
      sllm_matmul_kernel::select_mxfp8_variant(2U, 31U, 17U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8 &&
      sllm_matmul_kernel::select_mxfp8_variant(2U, 32U, 17U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Col16 &&
      sllm_matmul_kernel::select_mxfp8_variant(2U, 33U, 17U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8 &&
      sllm_matmul_kernel::select_mxfp8_variant(1U, 32U, 17U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8Decode &&
      sllm_matmul_kernel::select_mxfp8_variant(2U, 32U, 17U, "gfx1201") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8 &&
      sllm_matmul_kernel::select_mxfp8_variant(2U, 32U, 17U, "gfx942") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8 &&
      sllm_matmul_kernel::select_mxfp8_variant(2U, 32U, 17U, "gfx9999") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8;
  setenv(gfx1030_mmq_columns, "32", 1);
  valid =
      valid &&
      sllm_matmul_kernel::select_mxfp8_variant(9U, 32U, 33U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Col32;

  setenv(mmq_columns, "4", 1);
  valid = valid &&
          sllm_matmul_kernel::select_mxfp8_variant(9U, 32U, 33U, "gfx1030") ==
              sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillMmqCol4;
  unsetenv(mmq_columns);
  setenv(mxfp8_row8, "1", 1);
  valid = valid &&
          sllm_matmul_kernel::select_mxfp8_variant(9U, 32U, 33U, "gfx1030") ==
              sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8;
  unsetenv(mxfp8_row8);
  setenv(mxfp8_tiled16, "1", 1);
  valid = valid &&
          sllm_matmul_kernel::select_mxfp8_variant(9U, 32U, 33U, "gfx1030") ==
              sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillTiled16;
  unsetenv(mxfp8_tiled16);
  setenv(baseline, "1", 1);
  valid = valid &&
          sllm_matmul_kernel::select_mxfp8_variant(9U, 32U, 33U, "gfx1030") ==
              sllm_matmul_kernel::KernelVariant::Mxfp8W8A8Prefill;
  unsetenv(baseline);
  unsetenv(gfx1030_mmq_columns);

  setenv(gfx1030_phase69, "regscale", 1);
  valid = valid &&
          sllm_matmul_kernel::select_mxfp8_variant(2U, 31U, 17U, "gfx1030") ==
              sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8 &&
          sllm_matmul_kernel::select_mxfp8_variant(2U, 32U, 17U, "gfx1030") ==
              sllm_matmul_kernel::KernelVariant::
                  Mxfp8W8A8PrefillMmqGfx1030Regscale &&
          sllm_matmul_kernel::select_mxfp8_variant(2U, 32U, 17U, "gfx1201") ==
              sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8;
  setenv(gfx1030_phase69, "vector32", 1);
  valid =
      valid &&
      sllm_matmul_kernel::select_mxfp8_variant(9U, 32U, 33U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Vector32;
  setenv(gfx1030_phase69, "combined", 1);
  valid =
      valid &&
      sllm_matmul_kernel::select_mxfp8_variant(9U, 32U, 33U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::
              Mxfp8W8A8PrefillMmqGfx1030RegscaleVector32 &&
      sllm_matmul_kernel::select_mxfp8_variant(7U, 256U, 7U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::
              Mxfp8W8A8PrefillMmqGfx1030RegscaleVector32 &&
      sllm_matmul_kernel::select_mxfp8_variant(8U, 256U, 8U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::
              Mxfp8W8A8PrefillMmqGfx1030RegscaleVector32 &&
      sllm_matmul_kernel::select_mxfp8_variant(9U, 256U, 9U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::
              Mxfp8W8A8PrefillMmqGfx1030RegscaleVector32 &&
      sllm_matmul_kernel::select_mxfp8_variant(127U, 256U, 31U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::
              Mxfp8W8A8PrefillMmqGfx1030RegscaleVector32 &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 256U, 32U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::
              Mxfp8W8A8PrefillMmqGfx1030RegscaleVector32 &&
      sllm_matmul_kernel::select_mxfp8_variant(129U, 256U, 33U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::
              Mxfp8W8A8PrefillMmqGfx1030RegscaleVector32 &&
      sllm_matmul_kernel::select_mxfp8_variant(8U, 255U, 8U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8 &&
      sllm_matmul_kernel::select_mxfp8_variant(8U, 257U, 8U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8;
  setenv(mxfp8_row8, "1", 1);
  valid = valid &&
          sllm_matmul_kernel::select_mxfp8_variant(9U, 32U, 33U, "gfx1030") ==
              sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8;
  unsetenv(mxfp8_row8);
  setenv(baseline, "1", 1);
  valid = valid &&
          sllm_matmul_kernel::select_mxfp8_variant(9U, 32U, 33U, "gfx1030") ==
              sllm_matmul_kernel::KernelVariant::Mxfp8W8A8Prefill;
  unsetenv(baseline);
  unsetenv(gfx1030_phase69);

  valid =
      valid &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 2560U, 9216U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillMmqGfx1030Vector32;
  setenv(gfx1030_phase69, "control", 1);
  valid =
      valid &&
      sllm_matmul_kernel::select_mxfp8_variant(128U, 2560U, 9216U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillMmqCol8 &&
      sllm_matmul_kernel::select_mxfp8_variant(17U, 2560U, 9216U, "gfx1030") ==
          sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillRow8;
  unsetenv(gfx1030_phase69);

  setenv(mxfp8_tiled16, "1", 1);
  setenv(mxfp6_row8, "1", 1);
  valid = valid &&
          sllm_matmul_kernel::select_mxfp8_variant(17U) ==
              sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillTiled16 &&
          sllm_matmul_kernel::select_mxfp6_variant(17U) ==
              sllm_matmul_kernel::KernelVariant::Mxfp6W6A6PrefillRow8;

  setenv(mmq_columns, "4", 1);
  setenv(baseline, "1", 1);
  valid = valid &&
          sllm_matmul_kernel::select_mxfp8_variant(17U) ==
              sllm_matmul_kernel::KernelVariant::Mxfp8W8A8Prefill &&
          sllm_matmul_kernel::select_mxfp6_variant(17U) ==
              sllm_matmul_kernel::KernelVariant::Mxfp6W6A6Prefill;

  unsetenv(baseline);
  unsetenv(mmq_columns);
  unsetenv(mxfp8_tiled16);

  setenv(gfx1030_mmq_columns, "16", 1);
  fake_hip::reset();
  fake_hip::set_gcn_arch_name("gfx1030");
  {
    sllm_context_t *gfx1030_context = nullptr;
    sllm_buffer_t *gfx1030_activation = nullptr;
    sllm_buffer_t *gfx1030_weight = nullptr;
    sllm_buffer_t *gfx1030_output = nullptr;
    sllm_matmul_plan_t *gfx1030_plan = nullptr;
    constexpr uint64_t gfx1030_m = 9U;
    constexpr uint64_t gfx1030_k = 32U;
    constexpr uint64_t gfx1030_n = 33U;
    constexpr uint64_t gfx1030_weight_blocks =
        gfx1030_n * gfx1030_k / UINT64_C(32);
    Error gfx1030_error;
    if (!create_context_for_arch("gfx1030", &gfx1030_context) ||
        !create_buffer_sized(gfx1030_context,
                             gfx1030_m * gfx1030_k * sizeof(uint16_t),
                             &gfx1030_activation) ||
        !create_buffer_sized(gfx1030_context,
                             gfx1030_n * gfx1030_k + gfx1030_weight_blocks,
                             &gfx1030_weight) ||
        !create_buffer_sized(gfx1030_context,
                             gfx1030_m * gfx1030_n * sizeof(uint16_t),
                             &gfx1030_output)) {
      valid = false;
    } else {
      auto descriptor = matmul_descriptor(gfx1030_activation, 0U,
                                          gfx1030_weight, 0U, gfx1030_output,
                                          0U, gfx1030_m, gfx1030_k, gfx1030_n);
      descriptor.op_version = SLLM_HIP_MATMUL_MXFP8_W8A8_VERSION;
      descriptor.weight.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
      descriptor.weight.encoding = SLLM_TENSOR_ENCODING_MXFP8_BLOCK32_E8M0;
      valid =
          valid &&
          expect_status(sllm_matmul_prepare(gfx1030_context, &descriptor,
                                            &gfx1030_plan, &gfx1030_error.sink),
                        SLLM_STATUS_OK, "gfx1030 staged MMQ provider",
                        gfx1030_error) &&
          gfx1030_plan != nullptr;
      uint32_t prepared_provider = 0U;
      uint32_t prepared_tile = 0U;
      uint32_t prepared_inner_product = 0U;
      valid =
          valid &&
          sllm_test_matmul_prepared_kernel_id(gfx1030_plan) ==
              static_cast<uint32_t>(sllm_matmul_kernel::KernelVariant::
                                        Mxfp8W8A8PrefillMmqGfx1030Col16) &&
          sllm_test_matmul_prepared_provider_semantics(
              gfx1030_plan, &prepared_provider, &prepared_tile,
              &prepared_inner_product) == 1U &&
          prepared_provider ==
              static_cast<uint32_t>(sllm_lowp::ProviderKind::Mxfp8Block32) &&
          prepared_tile ==
              static_cast<uint32_t>(sllm_lowp::TilePolicy::BlockRow8Column16) &&
          prepared_inner_product ==
              static_cast<uint32_t>(
                  sllm_lowp::InnerProduct::DecodedBlockScaledFp32);
      setenv(gfx1030_mmq_columns, "32", 1);
      setenv(baseline, "1", 1);
      valid = valid &&
              sllm_test_matmul_prepared_kernel_id(gfx1030_plan) ==
                  static_cast<uint32_t>(sllm_matmul_kernel::KernelVariant::
                                            Mxfp8W8A8PrefillMmqGfx1030Col16) &&
              sllm_test_matmul_prepared_provider_semantics(
                  gfx1030_plan, &prepared_provider, &prepared_tile,
                  &prepared_inner_product) == 1U &&
              prepared_tile == static_cast<uint32_t>(
                                   sllm_lowp::TilePolicy::BlockRow8Column16);
    }
    if (gfx1030_plan != nullptr) {
      valid = expect_status(
                  sllm_matmul_plan_release(&gfx1030_plan, &gfx1030_error.sink),
                  SLLM_STATUS_OK, "gfx1030 staged MMQ provider release",
                  gfx1030_error) &&
              valid;
    }
    valid = release_buffer(&gfx1030_activation) &&
            release_buffer(&gfx1030_weight) &&
            release_buffer(&gfx1030_output) &&
            release_context(&gfx1030_context) && valid;
  }
  unsetenv(baseline);
  unsetenv(gfx1030_mmq_columns);

  setenv(mxfp8_wmma, "1", 1);
  fake_hip::reset();
  fake_hip::set_gcn_arch_name("gfx1201");
  sllm_context_t *context = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *weight = nullptr;
  sllm_buffer_t *output = nullptr;
  sllm_matmul_plan_t *plan = nullptr;
  constexpr uint64_t prepared_m = 3U;
  constexpr uint64_t prepared_k = 64U;
  constexpr uint64_t prepared_n = 7U;
  constexpr uint64_t weight_blocks = prepared_n * prepared_k / 32U;
  Error error;
  if (!create_context(&context) ||
      !create_buffer_sized(context, prepared_m * prepared_k * sizeof(uint16_t),
                           &activation) ||
      !create_buffer_sized(context, prepared_n * prepared_k + weight_blocks,
                           &weight) ||
      !create_buffer_sized(context, prepared_m * prepared_n * sizeof(uint16_t),
                           &output)) {
    valid = false;
  } else {
    auto descriptor = matmul_descriptor(activation, 0U, weight, 0U, output, 0U,
                                        prepared_m, prepared_k, prepared_n);
    descriptor.op_version = SLLM_HIP_MATMUL_MXFP8_W8A8_VERSION;
    descriptor.weight.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
    descriptor.weight.encoding = SLLM_TENSOR_ENCODING_MXFP8_BLOCK32_E8M0;
    valid = valid &&
            expect_status(
                sllm_matmul_prepare(context, &descriptor, &plan, &error.sink),
                SLLM_STATUS_OK, "MXFP8 prepared provider", error) &&
            plan != nullptr;
    uint32_t prepared_provider = 0U;
    uint32_t prepared_tile = 0U;
    uint32_t prepared_inner_product = 0U;
    valid =
        valid &&
        sllm_test_matmul_prepared_provider_semantics(
            plan, &prepared_provider, &prepared_tile,
            &prepared_inner_product) == 1U &&
        prepared_provider ==
            static_cast<uint32_t>(sllm_lowp::ProviderKind::Mxfp8Gfx1201Wmma) &&
        prepared_tile ==
            static_cast<uint32_t>(sllm_lowp::TilePolicy::Wmma128x64x32) &&
        prepared_inner_product ==
            static_cast<uint32_t>(sllm_lowp::InnerProduct::E4M3WmmaFp32);
    unsetenv(mxfp8_wmma);
    setenv(baseline, "1", 1);
    valid =
        valid &&
        sllm_test_matmul_prepared_kernel_id(plan) ==
            static_cast<uint32_t>(
                sllm_matmul_kernel::KernelVariant::Mxfp8W8A8PrefillWmmaN64) &&
        sllm_test_matmul_prepared_provider_semantics(
            plan, &prepared_provider, &prepared_tile,
            &prepared_inner_product) == 1U &&
        prepared_provider ==
            static_cast<uint32_t>(sllm_lowp::ProviderKind::Mxfp8Gfx1201Wmma) &&
        prepared_tile ==
            static_cast<uint32_t>(sllm_lowp::TilePolicy::Wmma128x64x32) &&
        prepared_inner_product ==
            static_cast<uint32_t>(sllm_lowp::InnerProduct::E4M3WmmaFp32);
  }
  if (plan != nullptr) {
    valid = expect_status(sllm_matmul_plan_release(&plan, &error.sink),
                          SLLM_STATUS_OK, "MXFP8 prepared provider release",
                          error) &&
            valid;
  }
  valid = release_buffer(&activation) && release_buffer(&weight) &&
          release_buffer(&output) && release_context(&context) && valid;

  unsetenv(baseline);
  fake_hip::reset();
  fake_hip::set_gcn_arch_name("gfx1201");
  context = nullptr;
  activation = nullptr;
  weight = nullptr;
  output = nullptr;
  plan = nullptr;
  constexpr uint64_t large_m = 129U;
  constexpr uint64_t large_k = 2560U;
  constexpr uint64_t large_n = 9216U;
  constexpr uint64_t large_weight_blocks = large_n * large_k / 32U;
  if (!create_context(&context) ||
      !create_buffer_sized(context, large_m * large_k * sizeof(uint16_t),
                           &activation) ||
      !create_buffer_sized(context, large_n * large_k + large_weight_blocks,
                           &weight) ||
      !create_buffer_sized(context, large_m * large_n * sizeof(uint16_t),
                           &output)) {
    valid = false;
  } else {
    const auto prepare_large = [&](const uint64_t m,
                                   sllm_matmul_plan_t **const output_plan) {
      auto descriptor = matmul_descriptor(activation, 0U, weight, 0U, output,
                                          0U, m, large_k, large_n);
      descriptor.op_version = SLLM_HIP_MATMUL_MXFP8_W8A8_VERSION;
      descriptor.weight.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
      descriptor.weight.encoding = SLLM_TENSOR_ENCODING_MXFP8_BLOCK32_E8M0;
      return sllm_matmul_prepare(context, &descriptor, output_plan,
                                 &error.sink);
    };
    valid = valid &&
            expect_status(prepare_large(128U, &plan), SLLM_STATUS_OK,
                          "MXFP8 prepared ID37 provider", error) &&
            plan != nullptr;
    uint32_t prepared_provider = 0U;
    uint32_t prepared_tile = 0U;
    uint32_t prepared_inner_product = 0U;
    valid =
        valid &&
        sllm_test_matmul_prepared_kernel_id(plan) ==
            static_cast<uint32_t>(sllm_matmul_kernel::KernelVariant::
                                      Mxfp8W8A8PrefillWmmaN128DirectBoth) &&
        sllm_test_matmul_prepared_provider_semantics(
            plan, &prepared_provider, &prepared_tile,
            &prepared_inner_product) == 1U &&
        prepared_provider ==
            static_cast<uint32_t>(sllm_lowp::ProviderKind::Mxfp8Gfx1201Wmma) &&
        prepared_tile ==
            static_cast<uint32_t>(sllm_lowp::TilePolicy::Wmma128x128x32) &&
        prepared_inner_product ==
            static_cast<uint32_t>(sllm_lowp::InnerProduct::E4M3WmmaFp32);
    setenv(baseline, "1", 1);
    valid = valid &&
            sllm_test_matmul_prepared_kernel_id(plan) ==
                static_cast<uint32_t>(sllm_matmul_kernel::KernelVariant::
                                          Mxfp8W8A8PrefillWmmaN128DirectBoth) &&
            sllm_test_matmul_prepared_provider_semantics(
                plan, &prepared_provider, &prepared_tile,
                &prepared_inner_product) == 1U &&
            prepared_tile ==
                static_cast<uint32_t>(sllm_lowp::TilePolicy::Wmma128x128x32);
    unsetenv(baseline);
    if (plan != nullptr) {
      valid = expect_status(sllm_matmul_plan_release(&plan, &error.sink),
                            SLLM_STATUS_OK,
                            "MXFP8 prepared ID37 provider release", error) &&
              valid;
    }

    valid = valid &&
            expect_status(prepare_large(large_m, &plan), SLLM_STATUS_OK,
                          "MXFP8 prepared M129 provider", error) &&
            plan != nullptr;
    valid =
        valid &&
        sllm_test_matmul_prepared_kernel_id(plan) ==
            static_cast<uint32_t>(sllm_matmul_kernel::KernelVariant::
                                      Mxfp8W8A8PrefillWmmaDirectWeight) &&
        sllm_test_matmul_prepared_provider_semantics(
            plan, &prepared_provider, &prepared_tile,
            &prepared_inner_product) == 1U &&
        prepared_provider ==
            static_cast<uint32_t>(sllm_lowp::ProviderKind::Mxfp8Gfx1201Wmma) &&
        prepared_tile ==
            static_cast<uint32_t>(sllm_lowp::TilePolicy::Wmma128x64x32) &&
        prepared_inner_product ==
            static_cast<uint32_t>(sllm_lowp::InnerProduct::E4M3WmmaFp32);
    setenv(mxfp8_row8, "1", 1);
    valid = valid &&
            sllm_test_matmul_prepared_kernel_id(plan) ==
                static_cast<uint32_t>(sllm_matmul_kernel::KernelVariant::
                                          Mxfp8W8A8PrefillWmmaDirectWeight) &&
            sllm_test_matmul_prepared_provider_semantics(
                plan, &prepared_provider, &prepared_tile,
                &prepared_inner_product) == 1U &&
            prepared_tile ==
                static_cast<uint32_t>(sllm_lowp::TilePolicy::Wmma128x64x32);
    unsetenv(mxfp8_row8);
  }
  if (plan != nullptr) {
    valid = expect_status(sllm_matmul_plan_release(&plan, &error.sink),
                          SLLM_STATUS_OK,
                          "MXFP8 prepared M129 provider release", error) &&
            valid;
  }
  valid = release_buffer(&activation) && release_buffer(&weight) &&
          release_buffer(&output) && release_context(&context) && valid;

  restore_environment();
  return valid && fake_hip::live_events() == 0U &&
         fake_hip::live_streams() == 0U && fake_hip::live_allocations() == 0U;
}

bool matmul_short_mixed_metadata_dispatch_contract() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  fake_hip::set_gcn_arch_name("gfx1030");
  unsetenv("SLLM_MATMUL_GFX1030_SHORT_MIXED");
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *weight = nullptr;
  sllm_buffer_t *output = nullptr;
  constexpr uint64_t m = 9U;
  constexpr uint64_t k = 4096U;
  constexpr uint64_t n = 2560U;
  if (!create_context_for_arch("gfx1030", &context) ||
      !create_queue(context, &queue) ||
      !create_buffer_sized(context, m * k * sizeof(uint16_t), &activation) ||
      !create_buffer_sized(context, n * k * sizeof(uint16_t), &weight) ||
      !create_buffer_sized(context, m * n * sizeof(uint16_t), &output)) {
    return false;
  }
  Error error;
  auto descriptor =
      matmul_descriptor(activation, 0U, weight, 0U, output, 0U, m, k, n);
  sllm_matmul_plan_t *plan = nullptr;
  sllm_completion_t *completion = nullptr;
  auto info = matmul_dispatch_info();
  const bool valid =
      expect_status(
          sllm_matmul_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "short mixed metadata prepare", error) &&
      plan != nullptr &&
      expect_status(
          sllm_matmul_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_OK, "short mixed metadata execute", error) &&
      completion != nullptr && info.dispatch_count == 2U &&
      info.kernel_id == 17U && info.grid_size_x == n && info.m == m &&
      info.k == k && info.n == n && info.output_elements == m * n &&
      std::strcmp(info.kernel_symbol,
                  "matmul.bf16_fp32.prefill.short_mixed_bss.v2") == 0 &&
      std::strcmp(info.device_symbol, "hipblasGemmExBbsF32Output") == 0 &&
      std::strcmp(info.gcn_arch_name, "gfx1030") == 0 &&
      fake_hip::matmul_launch_calls() == 1U &&
      query_completion(completion, SLLM_STATUS_OK) &&
      release_completion(&completion) &&
      expect_status(sllm_matmul_plan_release(&plan, &error.sink),
                    SLLM_STATUS_OK, "short mixed metadata plan release",
                    error) &&
      plan == nullptr && release_queue(&queue) && release_buffer(&activation) &&
      release_buffer(&weight) && release_buffer(&output) &&
      release_context(&context);
  fake_hip::set_gcn_arch_name("gfx1201");
  return valid && fake_hip::live_events() == 0U &&
         fake_hip::live_streams() == 0U && fake_hip::live_allocations() == 0U;
}

bool matmul_short_mixed_rocblas_solution_selector_contract() {
  const char *const solution_environment =
      sllm_matmul_kernel::kPhase49Gfx1030ShortMixedRocblasSolutionEnvironment;
  const char *const old_solution = std::getenv(solution_environment);
  const bool had_solution = old_solution != nullptr;
  const std::string old_solution_value = had_solution ? old_solution : "";
  const char *const old_force = std::getenv("SLLM_MATMUL_FORCE_BASELINE");
  const bool had_force = old_force != nullptr;
  const std::string old_force_value = had_force ? old_force : "";
  const char *const old_short_mixed =
      std::getenv("SLLM_MATMUL_GFX1030_SHORT_MIXED");
  const bool had_short_mixed = old_short_mixed != nullptr;
  const std::string old_short_mixed_value =
      had_short_mixed ? old_short_mixed : "";

  const auto restore_environment = [&]() {
    if (had_solution) {
      setenv(solution_environment, old_solution_value.c_str(), 1);
    } else {
      unsetenv(solution_environment);
    }
    if (had_force) {
      setenv("SLLM_MATMUL_FORCE_BASELINE", old_force_value.c_str(), 1);
    } else {
      unsetenv("SLLM_MATMUL_FORCE_BASELINE");
    }
    if (had_short_mixed) {
      setenv("SLLM_MATMUL_GFX1030_SHORT_MIXED", old_short_mixed_value.c_str(),
             1);
    } else {
      unsetenv("SLLM_MATMUL_GFX1030_SHORT_MIXED");
    }
  };

  constexpr uint64_t m17 = 17U;
  constexpr uint64_t m32 = 32U;
  constexpr uint64_t k2560 = 2560U;
  constexpr uint64_t k4096 = 4096U;
  constexpr uint64_t n9216 = 9216U;
  constexpr uint64_t n4096 = 4096U;
  constexpr uint64_t n8192 = 8192U;
  constexpr uint64_t nVocab = 248320U;
  bool valid = sllm_matmul_kernel::phase49_gfx1030_short_mixed_rocblas_solution(
                   m17, k2560, n9216) == -473 &&
               sllm_matmul_kernel::phase49_gfx1030_short_mixed_rocblas_solution(
                   m17, k2560, n4096) == -472 &&
               sllm_matmul_kernel::phase49_gfx1030_short_mixed_rocblas_solution(
                   m32, k2560, n8192) == -473 &&
               sllm_matmul_kernel::phase49_gfx1030_short_mixed_rocblas_solution(
                   m32, k2560, nVocab) == 0 &&
               sllm_matmul_kernel::phase49_gfx1030_short_mixed_rocblas_solution(
                   16U, k2560, n9216) == 0;

  unsetenv(solution_environment);
  unsetenv("SLLM_MATMUL_FORCE_BASELINE");
  unsetenv("SLLM_MATMUL_GFX1030_SHORT_MIXED");
  valid = valid &&
          sllm_matmul_kernel::phase49_gfx1030_short_mixed_rocblas_enabled(
              "gfx1030", m17, k4096, 2560U) &&
          sllm_matmul_kernel::select_variant(m17, k2560, n9216, "gfx1030") ==
              sllm_matmul_kernel::KernelVariant::PrefillShortMixed;

  setenv(solution_environment, "1", 1);
  valid =
      valid && sllm_matmul_kernel::phase49_gfx1030_short_mixed_rocblas_enabled(
                   "gfx1030", m17, k4096, 2560U);
  setenv(solution_environment, "0", 1);
  valid =
      valid && !sllm_matmul_kernel::phase49_gfx1030_short_mixed_rocblas_enabled(
                   "gfx1030", m17, k4096, 2560U);
  setenv(solution_environment, "unknown", 1);
  valid =
      valid && !sllm_matmul_kernel::phase49_gfx1030_short_mixed_rocblas_enabled(
                   "gfx1030", m17, k4096, 2560U);
  unsetenv(solution_environment);
  setenv("SLLM_MATMUL_FORCE_BASELINE", "1", 1);
  valid = valid &&
          !sllm_matmul_kernel::phase49_gfx1030_short_mixed_rocblas_enabled(
              "gfx1030", m17, k4096, 2560U) &&
          sllm_matmul_kernel::select_variant(m17, k2560, n9216, "gfx1030") ==
              sllm_matmul_kernel::KernelVariant::Baseline;
  unsetenv("SLLM_MATMUL_FORCE_BASELINE");
  valid =
      valid && !sllm_matmul_kernel::phase49_gfx1030_short_mixed_rocblas_enabled(
                   "gfx1201", m17, k4096, 2560U);

  restore_environment();
  return valid;
}

bool matmul_async_lifetime_and_cleanup() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *activation = nullptr;
  sllm_buffer_t *weight = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer_sized(context, 1024U, &activation) ||
      !create_buffer_sized(context, 1024U, &weight) ||
      !create_buffer_sized(context, 1024U, &output)) {
    return false;
  }
  Error error;
  auto descriptor = matmul_descriptor(activation, 0U, weight, 0U, output, 0U);
  sllm_matmul_plan_t *plan = nullptr;
  if (!expect_status(
          sllm_matmul_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "matmul async prepare", error)) {
    return false;
  }
  sllm_completion_t *completion = nullptr;
  auto info = matmul_dispatch_info();
  if (!expect_status(
          sllm_matmul_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_OK, "matmul async execute", error) ||
      !expect_status(sllm_matmul_plan_release(&plan, &error.sink),
                     SLLM_STATUS_PUBLIC_BUSY, "matmul in-flight plan release",
                     error) ||
      !release_queue(&queue, SLLM_STATUS_PUBLIC_BUSY) ||
      !release_buffer(&activation, SLLM_STATUS_PUBLIC_BUSY) ||
      !release_buffer(&weight, SLLM_STATUS_PUBLIC_BUSY) ||
      !release_buffer(&output, SLLM_STATUS_PUBLIC_BUSY) ||
      !release_context(&context, SLLM_STATUS_PUBLIC_BUSY) ||
      !query_completion(completion, SLLM_STATUS_OK) ||
      !release_completion(&completion) ||
      !expect_status(sllm_matmul_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "matmul async plan release", error) ||
      !release_queue(&queue) || !release_buffer(&activation) ||
      !release_buffer(&weight) || !release_buffer(&output) ||
      !release_context(&context)) {
    return false;
  }

  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer_sized(context, 1024U, &activation) ||
      !create_buffer_sized(context, 1024U, &weight) ||
      !create_buffer_sized(context, 1024U, &output)) {
    return false;
  }
  descriptor = matmul_descriptor(activation, 0U, weight, 0U, output, 0U);
  if (!expect_status(
          sllm_matmul_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "matmul cleanup prepare", error)) {
    return false;
  }
  fake_hip::set_matmul_launch_status(hipErrorUnknown);
  completion = nullptr;
  info = matmul_dispatch_info();
  const bool cleanup_failed =
      expect_status(
          sllm_matmul_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR, "matmul launch cleanup",
          error) &&
      completion == nullptr && fake_hip::matmul_launch_calls() == 2U;
  fake_hip::set_matmul_launch_status(hipSuccess);
  return cleanup_failed &&
         expect_status(sllm_matmul_plan_release(&plan, &error.sink),
                       SLLM_STATUS_OK, "matmul cleanup plan release", error) &&
         release_queue(&queue) && release_buffer(&activation) &&
         release_buffer(&weight) && release_buffer(&output) &&
         release_context(&context) && fake_hip::live_events() == 0U &&
         fake_hip::live_streams() == 0U && fake_hip::live_allocations() == 0U;
}

using AttentionBuffers = std::array<sllm_buffer_t *, 8>;

void release_attention_buffers(AttentionBuffers *const buffers) {
  if (buffers == nullptr) {
    return;
  }
  for (sllm_buffer_t *&buffer : *buffers) {
    if (buffer != nullptr) {
      release_buffer(&buffer);
    }
  }
}

bool create_attention_resources(sllm_context_t **const context,
                                sllm_queue_t **const queue,
                                AttentionBuffers *const buffers,
                                const uint64_t m) {
  if (!create_context(context) || !create_queue(*context, queue)) {
    return false;
  }
  const uint64_t sizes[8] = {
      m * 16U * 512U * sizeof(uint16_t), m * 4U * 256U * sizeof(uint16_t),
      16U * 256U * sizeof(uint16_t),     4U * 256U * sizeof(uint16_t),
      m * 3U * sizeof(int32_t),          m * 16U * 256U * sizeof(uint16_t),
      m * 16U * 256U * sizeof(uint16_t), m * 4U * 256U * sizeof(uint16_t),
  };
  for (std::size_t index = 0U; index != buffers->size(); ++index) {
    if (!create_buffer_sized(*context, sizes[index], &(*buffers)[index])) {
      return false;
    }
  }
  return true;
}

sllm_tensor_binding_t attention_binding(const sllm_buffer_t *const buffer,
                                        const uint32_t dtype,
                                        const uint32_t rank,
                                        const uint64_t *const shape) {
  sllm_tensor_binding_t binding{};
  binding.struct_size = sizeof(binding);
  binding.abi_version = SLLM_HIP_ABI_VERSION;
  binding.buffer = buffer;
  binding.dtype = dtype;
  binding.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  binding.rank = rank;
  uint64_t stride = 1U;
  for (uint32_t backwards = 0U; backwards != rank; ++backwards) {
    const uint32_t index = rank - 1U - backwards;
    binding.shape[index] = shape[index];
    binding.stride_elements[index] = stride;
    stride *= shape[index];
  }
  return binding;
}

sllm_attention_preprocess_desc_t
attention_preprocess_descriptor(const AttentionBuffers &buffers,
                                const uint64_t m,
                                const uint32_t start_position) {
  uint64_t packed_shape[] = {m, 16U, 512U};
  uint64_t k_shape[] = {m, 4U, 256U};
  constexpr uint64_t scale_q_shape[] = {16U, 256U};
  constexpr uint64_t scale_k_shape[] = {4U, 256U};
  uint64_t positions_shape[] = {m};
  uint64_t output_q_shape[] = {m, 16U, 256U};
  uint64_t output_k_shape[] = {m, 4U, 256U};
  sllm_attention_preprocess_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_ATTENTION_PREPROCESS_VERSION;
  descriptor.start_position = start_position;
  descriptor.packed_q_gate =
      attention_binding(buffers[0], SLLM_TENSOR_DTYPE_BF16, 3U, packed_shape);
  descriptor.k =
      attention_binding(buffers[1], SLLM_TENSOR_DTYPE_BF16, 3U, k_shape);
  descriptor.q_raw_scale =
      attention_binding(buffers[2], SLLM_TENSOR_DTYPE_BF16, 2U, scale_q_shape);
  descriptor.k_raw_scale =
      attention_binding(buffers[3], SLLM_TENSOR_DTYPE_BF16, 2U, scale_k_shape);
  descriptor.positions =
      attention_binding(buffers[4], SLLM_TENSOR_DTYPE_I32, 1U, positions_shape);
  descriptor.q_output =
      attention_binding(buffers[5], SLLM_TENSOR_DTYPE_BF16, 3U, output_q_shape);
  descriptor.gate_output =
      attention_binding(buffers[6], SLLM_TENSOR_DTYPE_BF16, 3U, output_q_shape);
  descriptor.k_output =
      attention_binding(buffers[7], SLLM_TENSOR_DTYPE_BF16, 3U, output_k_shape);
  return descriptor;
}

bool upload_attention_positions(const sllm_queue_t *const queue,
                                const sllm_buffer_t *const buffer,
                                std::vector<int32_t> &positions) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = positions.data();
  transfer.size_bytes =
      static_cast<uint64_t>(positions.size() * sizeof(int32_t));
  sllm_completion_t *completion = nullptr;
  Error error;
  return expect_status(sllm_buffer_copy_h2d(queue, buffer, &transfer,
                                            &completion, &error.sink),
                       SLLM_STATUS_OK, "attention position upload", error) &&
         query_completion(completion, SLLM_STATUS_OK) &&
         release_completion(&completion);
}

sllm_attention_preprocess_dispatch_info_t attention_preprocess_dispatch_info() {
  sllm_attention_preprocess_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_ATTENTION_PREPROCESS_DISPATCH_INFO_VERSION;
  return info;
}

bool attention_preprocess_prepare_validation_and_old_abi() {
  fake_hip::reset();
  uint32_t abi_version = 0U;
  Error error;
  if (!expect_status(sllm_get_abi_version(&abi_version, &error.sink),
                     SLLM_STATUS_OK, "old ABI version query", error) ||
      abi_version != SLLM_HIP_ABI_VERSION) {
    return false;
  }
  constexpr uint64_t m = 2U;
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  AttentionBuffers buffers{};
  if (!create_attention_resources(&context, &queue, &buffers, m)) {
    release_attention_buffers(&buffers);
    release_queue(&queue);
    release_context(&context);
    return false;
  }
  auto descriptor = attention_preprocess_descriptor(buffers, m, 0U);
  sllm_attention_preprocess_plan_t *plan = nullptr;
  auto expect = [&](const sllm_attention_preprocess_desc_t &candidate,
                    const sllm_status_t expected, const char *const name) {
    plan = nullptr;
    return expect_status(sllm_attention_preprocess_prepare(context, &candidate,
                                                           &plan, &error.sink),
                         expected, name, error) &&
           plan == nullptr;
  };
  auto flat = descriptor;
  flat.packed_q_gate.rank = 2U;
  flat.packed_q_gate.shape[1] = 4096U;
  flat.packed_q_gate.shape[2] = 0U;
  flat.packed_q_gate.stride_elements[0] = 4096U;
  flat.packed_q_gate.stride_elements[1] = 1U;
  flat.packed_q_gate.stride_elements[2] = 0U;
  if (!expect(flat, SLLM_STATUS_SHAPE_MISMATCH, "flat Q/gate rejection")) {
    return false;
  }
  auto reserved = descriptor;
  reserved.reserved[1] = 1U;
  if (!expect(reserved, SLLM_STATUS_RESERVED_NONZERO,
              "attention reserved rejection")) {
    return false;
  }
  auto unknown_position_mode = descriptor;
  unknown_position_mode.reserved[0] = 3U;
  if (!expect(unknown_position_mode, SLLM_STATUS_RESERVED_NONZERO,
              "unknown attention position mode rejection")) {
    return false;
  }
  auto derived_overflow = descriptor;
  derived_overflow.reserved[0] =
      SLLM_HIP_POSITION_PAYLOAD_MODE_DERIVED_CONTIGUOUS_V1;
  derived_overflow.start_position =
      static_cast<uint32_t>(std::numeric_limits<int32_t>::max());
  if (!expect(derived_overflow, SLLM_STATUS_SHAPE_MISMATCH,
              "derived attention I32 position overflow rejection")) {
    return false;
  }
  auto alias = descriptor;
  alias.q_output.buffer = alias.packed_q_gate.buffer;
  if (!expect(alias, SLLM_STATUS_ALIAS_OVERLAP, "attention alias rejection")) {
    return false;
  }
  const uint64_t packed_bytes = m * 16U * 512U * sizeof(uint16_t);
  const uint64_t q_output_bytes = m * 16U * 256U * sizeof(uint16_t);
  if (!release_buffer(&buffers[0]) ||
      !create_buffer_sized(context, packed_bytes + q_output_bytes,
                           &buffers[0])) {
    return false;
  }
  auto shared_nonoverlap = attention_preprocess_descriptor(buffers, m, 0U);
  shared_nonoverlap.q_output.buffer = buffers[0];
  shared_nonoverlap.q_output.byte_offset = packed_bytes;
  if (!expect_status(sllm_attention_preprocess_prepare(
                         context, &shared_nonoverlap, &plan, &error.sink),
                     SLLM_STATUS_OK, "attention shared nonoverlap prepare",
                     error) ||
      plan == nullptr ||
      !expect_status(sllm_attention_preprocess_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "attention shared nonoverlap plan release",
                     error)) {
    return false;
  }
  release_attention_buffers(&buffers);
  return release_queue(&queue) && release_context(&context) &&
         fake_hip::attention_preprocess_launch_calls() == 0U;
}

bool attention_preprocess_position_payload_mismatch_is_pre_dispatch() {
  fake_hip::reset();
  constexpr uint64_t m = 2U;
  constexpr uint32_t start_position = 9U;
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  AttentionBuffers buffers{};
  if (!create_attention_resources(&context, &queue, &buffers, m)) {
    release_attention_buffers(&buffers);
    release_queue(&queue);
    release_context(&context);
    return false;
  }
  std::vector<int32_t> positions = {static_cast<int32_t>(start_position),
                                    static_cast<int32_t>(start_position + 2U)};
  Error error;
  auto descriptor = attention_preprocess_descriptor(buffers, m, start_position);
  sllm_attention_preprocess_plan_t *plan = nullptr;
  sllm_completion_t *completion = nullptr;
  auto info = attention_preprocess_dispatch_info();
  const bool valid =
      upload_attention_positions(queue, buffers[4], positions) &&
      expect_status(sllm_attention_preprocess_prepare(context, &descriptor,
                                                      &plan, &error.sink),
                    SLLM_STATUS_OK, "attention mismatch prepare", error) &&
      plan != nullptr &&
      expect_status(sllm_attention_preprocess_execute(plan, queue, &completion,
                                                      &info, &error.sink),
                    SLLM_STATUS_POSITION_PAYLOAD_MISMATCH,
                    "attention position mismatch", error) &&
      completion == nullptr &&
      fake_hip::attention_preprocess_launch_calls() == 0U &&
      expect_status(sllm_attention_preprocess_plan_release(&plan, &error.sink),
                    SLLM_STATUS_OK, "attention mismatch plan release", error);
  release_attention_buffers(&buffers);
  return valid && release_queue(&queue) && release_context(&context);
}

bool attention_preprocess_success_metadata_and_dispatch() {
  fake_hip::reset();
  constexpr uint64_t m = 3U;
  constexpr uint32_t start_position = 17U;
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  AttentionBuffers buffers{};
  if (!create_attention_resources(&context, &queue, &buffers, m)) {
    release_attention_buffers(&buffers);
    release_queue(&queue);
    release_context(&context);
    return false;
  }
  std::vector<int32_t> positions;
  for (uint64_t index = 0U; index != m; ++index) {
    positions.push_back(static_cast<int32_t>(start_position + index));
  }
  Error error;
  auto descriptor = attention_preprocess_descriptor(buffers, m, start_position);
  sllm_attention_preprocess_plan_t *plan = nullptr;
  sllm_completion_t *completion = nullptr;
  auto info = attention_preprocess_dispatch_info();
  const bool valid =
      upload_attention_positions(queue, buffers[4], positions) &&
      expect_status(sllm_attention_preprocess_prepare(context, &descriptor,
                                                      &plan, &error.sink),
                    SLLM_STATUS_OK, "attention success prepare", error) &&
      plan != nullptr &&
      expect_status(sllm_attention_preprocess_execute(plan, queue, &completion,
                                                      &info, &error.sink),
                    SLLM_STATUS_OK, "attention success execute", error) &&
      completion != nullptr && info.dispatch_id != 0U &&
      info.dispatch_count == 1U &&
      info.kernel_id ==
          SLLM_HIP_ATTENTION_PREPROCESS_KERNEL_ID_BASELINE_BF16_V1 &&
      info.workgroup_size_x == SLLM_HIP_ATTENTION_PREPROCESS_WORKGROUP_SIZE &&
      info.grid_size_x == m * 20U && info.m == m &&
      info.q_heads == SLLM_HIP_ATTENTION_PREPROCESS_Q_HEADS &&
      info.k_heads == SLLM_HIP_ATTENTION_PREPROCESS_K_HEADS &&
      info.q_head_dim == SLLM_HIP_ATTENTION_PREPROCESS_Q_HEAD_DIM &&
      info.k_head_dim == SLLM_HIP_ATTENTION_PREPROCESS_K_HEAD_DIM &&
      info.rotary_dim == SLLM_HIP_ATTENTION_PREPROCESS_ROTARY_DIM &&
      info.start_position == start_position && info.fallback_allowed == 0U &&
      info.fallback_used == 0U &&
      std::strcmp(info.kernel_symbol,
                  "attention_preprocess.headwise_norm_rope.v1") == 0 &&
      std::strcmp(info.device_symbol,
                  "sllm_attention_preprocess_headwise_norm_rope_v1") == 0 &&
      std::strcmp(info.gcn_arch_name, "gfx1201") == 0 &&
      fake_hip::attention_preprocess_launch_calls() == 1U &&
      fake_hip::attention_preprocess_last_m() == m &&
      query_completion(completion, SLLM_STATUS_OK) &&
      release_completion(&completion) &&
      expect_status(sllm_attention_preprocess_plan_release(&plan, &error.sink),
                    SLLM_STATUS_OK, "attention success plan release", error);
  release_attention_buffers(&buffers);
  return valid && release_queue(&queue) && release_context(&context);
}

bool attention_preprocess_derived_positions_skip_payload_validation() {
  fake_hip::reset();
  constexpr uint64_t m = 3U;
  constexpr uint32_t start_position = 17U;
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  AttentionBuffers buffers{};
  if (!create_attention_resources(&context, &queue, &buffers, m)) {
    release_attention_buffers(&buffers);
    release_queue(&queue);
    release_context(&context);
    return false;
  }
  std::vector<int32_t> malformed_positions = {-1, -1, -1};
  Error error;
  auto descriptor = attention_preprocess_descriptor(buffers, m, start_position);
  descriptor.reserved[0] = SLLM_HIP_POSITION_PAYLOAD_MODE_DERIVED_CONTIGUOUS_V1;
  sllm_attention_preprocess_plan_t *plan = nullptr;
  sllm_completion_t *completion = nullptr;
  auto info = attention_preprocess_dispatch_info();
  const bool valid =
      upload_attention_positions(queue, buffers[4], malformed_positions) &&
      expect_status(sllm_attention_preprocess_prepare(context, &descriptor,
                                                      &plan, &error.sink),
                    SLLM_STATUS_OK, "derived attention prepare", error) &&
      plan != nullptr &&
      expect_status(sllm_attention_preprocess_execute(plan, queue, &completion,
                                                      &info, &error.sink),
                    SLLM_STATUS_OK, "derived attention execute", error) &&
      completion != nullptr && info.start_position == start_position &&
      fake_hip::attention_preprocess_launch_calls() == 1U &&
      query_completion(completion, SLLM_STATUS_OK) &&
      release_completion(&completion) &&
      expect_status(sllm_attention_preprocess_plan_release(&plan, &error.sink),
                    SLLM_STATUS_OK, "derived attention plan release", error);
  release_attention_buffers(&buffers);
  return valid && release_queue(&queue) && release_context(&context);
}

bool attention_preprocess_mrope_positions_dispatch() {
  fake_hip::reset();
  constexpr uint64_t m = 2U;
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  AttentionBuffers buffers{};
  if (!create_attention_resources(&context, &queue, &buffers, m)) {
    release_attention_buffers(&buffers);
    release_queue(&queue);
    release_context(&context);
    return false;
  }
  std::vector<int32_t> positions = {0, 0, 0, 1, 7, 11};
  Error error;
  auto descriptor = attention_preprocess_descriptor(buffers, m, 0U);
  const uint64_t positions_shape[] = {m, 3U};
  descriptor.positions =
      attention_binding(buffers[4], SLLM_TENSOR_DTYPE_I32, 2U, positions_shape);
  sllm_attention_preprocess_plan_t *plan = nullptr;
  sllm_completion_t *completion = nullptr;
  auto info = attention_preprocess_dispatch_info();
  const bool valid =
      upload_attention_positions(queue, buffers[4], positions) &&
      expect_status(sllm_attention_preprocess_prepare(context, &descriptor,
                                                      &plan, &error.sink),
                    SLLM_STATUS_OK, "attention mRoPE prepare", error) &&
      plan != nullptr &&
      expect_status(sllm_attention_preprocess_execute(plan, queue, &completion,
                                                      &info, &error.sink),
                    SLLM_STATUS_OK, "attention mRoPE execute", error) &&
      completion != nullptr && info.dispatch_count == 1U &&
      fake_hip::attention_preprocess_launch_calls() == 1U &&
      query_completion(completion, SLLM_STATUS_OK) &&
      release_completion(&completion) &&
      expect_status(sllm_attention_preprocess_plan_release(&plan, &error.sink),
                    SLLM_STATUS_OK, "attention mRoPE plan release", error);
  release_attention_buffers(&buffers);
  return valid && release_queue(&queue) && release_context(&context);
}

using RotaryTestBuffers = std::array<sllm_buffer_t *, 5>;

void release_rotary_buffers(RotaryTestBuffers *const buffers) {
  if (buffers == nullptr) {
    return;
  }
  for (sllm_buffer_t *&buffer : *buffers) {
    if (buffer != nullptr) {
      release_buffer(&buffer);
    }
  }
}

bool create_rotary_resources(sllm_context_t **const context,
                             sllm_queue_t **const queue,
                             RotaryTestBuffers *const buffers,
                             const uint64_t token_count, const uint32_t q_heads,
                             const uint32_t kv_heads, const uint32_t head_dim) {
  if (!create_context(context) || !create_queue(*context, queue)) {
    return false;
  }
  const uint64_t query_bytes = token_count * q_heads * head_dim * 2U;
  const uint64_t key_bytes = token_count * kv_heads * head_dim * 2U;
  const uint64_t sizes[] = {query_bytes, key_bytes,
                            token_count * sizeof(int32_t), query_bytes,
                            key_bytes};
  for (std::size_t index = 0U; index != buffers->size(); ++index) {
    if (!create_buffer_sized(*context, sizes[index], &(*buffers)[index])) {
      return false;
    }
  }
  return true;
}

sllm_rotary_desc_t
rotary_descriptor(const RotaryTestBuffers &buffers, const uint64_t token_count,
                  const uint64_t start_position, const uint32_t q_heads,
                  const uint32_t kv_heads, const uint32_t head_dim,
                  const uint32_t rotary_dim, const float theta) {
  const uint64_t query_shape[] = {token_count, q_heads, head_dim};
  const uint64_t key_shape[] = {token_count, kv_heads, head_dim};
  const uint64_t positions_shape[] = {token_count};
  sllm_rotary_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_ROTARY_VERSION;
  descriptor.start_position = start_position;
  descriptor.q_heads = q_heads;
  descriptor.kv_heads = kv_heads;
  descriptor.head_dim = head_dim;
  descriptor.rotary_dim = rotary_dim;
  std::memcpy(&descriptor.theta_bits, &theta, sizeof(theta));
  descriptor.max_position = SLLM_HIP_ROTARY_MAX_POSITION;
  descriptor.query =
      attention_binding(buffers[0], SLLM_TENSOR_DTYPE_BF16, 3U, query_shape);
  descriptor.key =
      attention_binding(buffers[1], SLLM_TENSOR_DTYPE_BF16, 3U, key_shape);
  descriptor.positions =
      attention_binding(buffers[2], SLLM_TENSOR_DTYPE_I32, 1U, positions_shape);
  descriptor.query_output =
      attention_binding(buffers[3], SLLM_TENSOR_DTYPE_BF16, 3U, query_shape);
  descriptor.key_output =
      attention_binding(buffers[4], SLLM_TENSOR_DTYPE_BF16, 3U, key_shape);
  return descriptor;
}

sllm_rotary_dispatch_info_t rotary_dispatch_info() {
  sllm_rotary_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_ROTARY_DISPATCH_INFO_VERSION;
  return info;
}

bool upload_rotary_positions(const sllm_queue_t *const queue,
                             const sllm_buffer_t *const buffer,
                             const std::vector<int32_t> &positions) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = const_cast<int32_t *>(positions.data());
  transfer.size_bytes =
      static_cast<uint64_t>(positions.size() * sizeof(int32_t));
  Error error;
  sllm_completion_t *completion = nullptr;
  return expect_status(sllm_buffer_copy_h2d(queue, buffer, &transfer,
                                            &completion, &error.sink),
                       SLLM_STATUS_OK, "rotary position upload", error) &&
         query_completion(completion, SLLM_STATUS_OK) &&
         release_completion(&completion);
}

bool rotary_prepare_execute_lifetime_and_negative_contract() {
  fake_hip::reset();
  constexpr uint64_t token_count = 3U;
  constexpr uint64_t start_position = 17U;
  constexpr uint32_t q_heads = 3U;
  constexpr uint32_t kv_heads = 1U;
  constexpr uint32_t head_dim = 6U;
  constexpr uint32_t rotary_dim = 4U;
  constexpr float theta = 10'000.0F;
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  RotaryTestBuffers buffers{};
  if (!create_rotary_resources(&context, &queue, &buffers, token_count, q_heads,
                               kv_heads, head_dim)) {
    release_rotary_buffers(&buffers);
    release_queue(&queue);
    release_context(&context);
    return false;
  }
  Error error;
  auto descriptor =
      rotary_descriptor(buffers, token_count, start_position, q_heads, kv_heads,
                        head_dim, rotary_dim, theta);
  sllm_rotary_plan_t *plan = nullptr;
  auto alias = descriptor;
  alias.query_output.buffer = alias.query.buffer;
  if (!expect_status(sllm_rotary_prepare(context, &alias, &plan, &error.sink),
                     SLLM_STATUS_ALIAS_OVERLAP, "rotary alias rejection",
                     error) ||
      plan != nullptr) {
    return false;
  }
  auto odd_head_dim = descriptor;
  odd_head_dim.head_dim = 5U;
  if (!expect_status(
          sllm_rotary_prepare(context, &odd_head_dim, &plan, &error.sink),
          SLLM_STATUS_INVALID_ROTARY_DESCRIPTOR,
          "rotary odd head dimension rejection", error) ||
      plan != nullptr) {
    return false;
  }
  std::vector<int32_t> positions = {static_cast<int32_t>(start_position),
                                    static_cast<int32_t>(start_position + 2U),
                                    static_cast<int32_t>(start_position + 2U)};
  if (!upload_attention_positions(queue, buffers[2], positions) ||
      !expect_status(
          sllm_rotary_prepare(context, &descriptor, &plan, &error.sink),
          SLLM_STATUS_OK, "rotary prepare", error) ||
      plan == nullptr ||
      !release_buffer(&buffers[0], SLLM_STATUS_PUBLIC_BUSY)) {
    return false;
  }
  sllm_completion_t *completion = nullptr;
  auto info = rotary_dispatch_info();
  if (!expect_status(
          sllm_rotary_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_POSITION_PAYLOAD_MISMATCH, "rotary position mismatch",
          error) ||
      completion != nullptr || fake_hip::rotary_launch_calls() != 0U) {
    return false;
  }
  positions[1] = static_cast<int32_t>(start_position + 1U);
  if (!upload_attention_positions(queue, buffers[2], positions)) {
    return false;
  }
  info = rotary_dispatch_info();
  const bool valid =
      expect_status(
          sllm_rotary_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_OK, "rotary execute", error) &&
      completion != nullptr &&
      expect_status(sllm_rotary_plan_release(&plan, &error.sink),
                    SLLM_STATUS_PUBLIC_BUSY, "rotary active plan release",
                    error) &&
      plan != nullptr && info.dispatch_id != 0U && info.dispatch_count == 1U &&
      info.kernel_id == SLLM_HIP_ROTARY_KERNEL_ID_SPLIT_HALF_BF16_FP32_V1 &&
      info.workgroup_size_x == SLLM_HIP_ROTARY_WORKGROUP_SIZE &&
      info.grid_size_x == token_count * (q_heads + kv_heads) &&
      info.token_count == token_count && info.q_heads == q_heads &&
      info.kv_heads == kv_heads && info.head_dim == head_dim &&
      info.rotary_dim == rotary_dim && info.start_position == start_position &&
      info.max_position == SLLM_HIP_ROTARY_MAX_POSITION &&
      info.fallback_allowed == 0U && info.fallback_used == 0U &&
      std::strcmp(info.kernel_symbol, "rotary.split_half.bf16_fp32.v1") == 0 &&
      std::strcmp(info.device_symbol, "sllm_rotary_split_half_bf16_fp32_v1") ==
          0 &&
      std::strcmp(info.gcn_arch_name, "gfx1201") == 0 &&
      fake_hip::rotary_launch_calls() == 1U &&
      fake_hip::rotary_last_token_count() == token_count &&
      query_completion(completion, SLLM_STATUS_OK) &&
      release_completion(&completion) &&
      expect_status(sllm_rotary_plan_release(&plan, &error.sink),
                    SLLM_STATUS_OK, "rotary plan release", error);
  if (!valid) {
    release_rotary_buffers(&buffers);
    release_queue(&queue);
    release_context(&context);
    return false;
  }
  auto explicit_descriptor = descriptor;
  explicit_descriptor.reserved0 = SLLM_HIP_POSITION_PAYLOAD_MODE_EXPLICIT_V1;
  positions = {static_cast<int32_t>(start_position),
               static_cast<int32_t>(start_position + 2U),
               static_cast<int32_t>(start_position + 2U)};
  if (!upload_rotary_positions(queue, buffers[2], positions) ||
      !expect_status(sllm_rotary_prepare(context, &explicit_descriptor, &plan,
                                         &error.sink),
                     SLLM_STATUS_OK, "rotary explicit prepare", error) ||
      plan == nullptr ||
      !expect_status(
          sllm_rotary_execute(plan, queue, &completion, &info, &error.sink),
          SLLM_STATUS_OK, "rotary explicit execute", error) ||
      completion == nullptr || !query_completion(completion, SLLM_STATUS_OK) ||
      !release_completion(&completion) ||
      !expect_status(sllm_rotary_plan_release(&plan, &error.sink),
                     SLLM_STATUS_OK, "rotary explicit plan release", error)) {
    release_rotary_buffers(&buffers);
    release_queue(&queue);
    release_context(&context);
    return false;
  }
  release_rotary_buffers(&buffers);
  return valid && release_queue(&queue) && release_context(&context) &&
         fake_hip::live_events() == 0U && fake_hip::live_streams() == 0U &&
         fake_hip::live_allocations() == 0U;
}

bool windowed_attention_prepare_execute_lifetime_and_negative_contract() {
  fake_hip::reset();
  constexpr uint64_t query_count = 3U;
  constexpr uint64_t start_position = 2U;
  constexpr uint64_t kv_length = start_position + query_count;
  constexpr uint64_t sliding_window = 4U;
  constexpr uint32_t q_heads = 3U;
  constexpr uint32_t kv_heads = 1U;
  constexpr uint32_t head_dim = 6U;
  const uint64_t query_bytes = query_count * q_heads * head_dim * 2U;
  const uint64_t kv_bytes = kv_length * kv_heads * head_dim * 2U;
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  std::array<sllm_buffer_t *, 4> buffers{};
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer_sized(context, query_bytes, &buffers[0]) ||
      !create_buffer_sized(context, kv_bytes, &buffers[1]) ||
      !create_buffer_sized(context, kv_bytes, &buffers[2]) ||
      !create_buffer_sized(context, query_bytes, &buffers[3])) {
    return false;
  }
  std::vector<uint16_t> query(query_bytes / 2U);
  std::vector<uint16_t> key(kv_bytes / 2U);
  std::vector<uint16_t> value(kv_bytes / 2U);
  for (std::size_t index = 0U; index != query.size(); ++index) {
    query[index] = sllm_rmsnorm_kernel::float_to_bf16_rne_bits(
        static_cast<float>(static_cast<int32_t>(index % 11U) - 5) / 16.0F);
  }
  for (std::size_t index = 0U; index != key.size(); ++index) {
    key[index] = sllm_rmsnorm_kernel::float_to_bf16_rne_bits(
        static_cast<float>(static_cast<int32_t>(index % 13U) - 6) / 16.0F);
    value[index] = sllm_rmsnorm_kernel::float_to_bf16_rne_bits(
        static_cast<float>(static_cast<int32_t>(index % 7U) - 3) / 8.0F);
  }
  const auto upload = [&](const sllm_buffer_t *const buffer,
                          const std::vector<uint16_t> &words) {
    sllm_transfer_desc_t transfer{};
    transfer.struct_size = sizeof(transfer);
    transfer.abi_version = SLLM_HIP_ABI_VERSION;
    transfer.host_pointer = const_cast<uint16_t *>(words.data());
    transfer.size_bytes = words.size() * sizeof(uint16_t);
    sllm_completion_t *completion = nullptr;
    Error upload_error;
    return expect_status(sllm_buffer_copy_h2d(queue, buffer, &transfer,
                                              &completion, &upload_error.sink),
                         SLLM_STATUS_OK, "windowed attention upload",
                         upload_error) &&
           query_completion(completion, SLLM_STATUS_OK) &&
           release_completion(&completion);
  };
  if (!upload(buffers[0], query) || !upload(buffers[1], key) ||
      !upload(buffers[2], value)) {
    return false;
  }

  const uint64_t query_shape[] = {query_count, q_heads, head_dim};
  const uint64_t kv_shape[] = {kv_length, kv_heads, head_dim};
  sllm_windowed_attention_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_WINDOWED_ATTENTION_VERSION;
  descriptor.start_position = start_position;
  descriptor.expected_kv_length = kv_length;
  descriptor.sliding_window = sliding_window;
  descriptor.q_heads = q_heads;
  descriptor.kv_heads = kv_heads;
  descriptor.head_dim = head_dim;
  descriptor.scaling_bits = UINT32_C(0x3f800000);
  descriptor.query =
      attention_binding(buffers[0], SLLM_TENSOR_DTYPE_BF16, 3U, query_shape);
  descriptor.key =
      attention_binding(buffers[1], SLLM_TENSOR_DTYPE_BF16, 3U, kv_shape);
  descriptor.value =
      attention_binding(buffers[2], SLLM_TENSOR_DTYPE_BF16, 3U, kv_shape);
  descriptor.output =
      attention_binding(buffers[3], SLLM_TENSOR_DTYPE_BF16, 3U, query_shape);
  Error error;
  sllm_windowed_attention_plan_t *plan = nullptr;
  auto alias = descriptor;
  alias.output.buffer = alias.query.buffer;
  if (!expect_status(
          sllm_windowed_attention_prepare(context, &alias, &plan, &error.sink),
          SLLM_STATUS_ALIAS_OVERLAP, "windowed attention alias rejection",
          error) ||
      plan != nullptr) {
    return false;
  }
  auto unsupported_scale = descriptor;
  unsupported_scale.scaling_bits = UINT32_C(0x3f000000);
  if (!expect_status(sllm_windowed_attention_prepare(
                         context, &unsupported_scale, &plan, &error.sink),
                     SLLM_STATUS_INVALID_WINDOWED_ATTENTION_DESCRIPTOR,
                     "windowed attention scale rejection", error) ||
      plan != nullptr ||
      !expect_status(sllm_windowed_attention_prepare(context, &descriptor,
                                                     &plan, &error.sink),
                     SLLM_STATUS_OK, "windowed attention prepare", error) ||
      plan == nullptr ||
      !release_buffer(&buffers[0], SLLM_STATUS_PUBLIC_BUSY)) {
    return false;
  }
  sllm_windowed_attention_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_WINDOWED_ATTENTION_DISPATCH_INFO_VERSION;
  sllm_completion_t *completion = nullptr;
  const bool valid =
      expect_status(sllm_windowed_attention_execute(plan, queue, &completion,
                                                    &info, &error.sink),
                    SLLM_STATUS_OK, "windowed attention execute", error) &&
      completion != nullptr &&
      expect_status(sllm_windowed_attention_plan_release(&plan, &error.sink),
                    SLLM_STATUS_PUBLIC_BUSY,
                    "windowed attention active plan release", error) &&
      plan != nullptr && info.dispatch_id != 0U && info.dispatch_count == 1U &&
      info.kernel_id ==
          SLLM_HIP_WINDOWED_ATTENTION_KERNEL_ID_ONLINE_SOFTMAX_GQA_BF16_V1 &&
      info.workgroup_size_x == SLLM_HIP_WINDOWED_ATTENTION_WORKGROUP_SIZE &&
      info.grid_size_x == query_count * q_heads &&
      info.query_count == query_count &&
      info.start_position == start_position &&
      info.committed_kv_length == kv_length &&
      info.sliding_window == sliding_window && info.q_heads == q_heads &&
      info.kv_heads == kv_heads && info.head_dim == head_dim &&
      info.scaling_bits == UINT32_C(0x3f800000) &&
      info.fallback_allowed == 0U && info.fallback_used == 0U &&
      std::strcmp(info.kernel_symbol,
                  "gemma_causal_attention.online_softmax_gqa_bf16.v1") == 0 &&
      std::strcmp(info.device_symbol,
                  "sllm_gemma_causal_attention_online_softmax_gqa_bf16_v1") ==
          0 &&
      std::strcmp(info.gcn_arch_name, "gfx1201") == 0 &&
      fake_hip::windowed_attention_launch_calls() == 1U &&
      query_completion(completion, SLLM_STATUS_OK) &&
      release_completion(&completion) &&
      expect_status(sllm_windowed_attention_plan_release(&plan, &error.sink),
                    SLLM_STATUS_OK, "windowed attention plan release", error);
  for (sllm_buffer_t *&buffer : buffers) {
    release_buffer(&buffer);
  }
  return valid && release_queue(&queue) && release_context(&context) &&
         fake_hip::live_events() == 0U && fake_hip::live_streams() == 0U &&
         fake_hip::live_allocations() == 0U;
}

bool create_kv_state(const sllm_context_t *const context,
                     const uint64_t capacity, sllm_kv_state_t **const state) {
  sllm_kv_state_create_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.session_id = 0x1234U;
  info.layer_id = 7U;
  info.capacity_tokens = capacity;
  info.memory_kind = SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS;
  info.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
  Error error;
  return expect_status(sllm_kv_state_create(context, &info, state, &error.sink),
                       SLLM_STATUS_OK, "sllm_kv_state_create", error);
}

sllm_tensor_binding_t kv_input_binding(const sllm_buffer_t *const buffer,
                                       const uint64_t token_count,
                                       const uint64_t byte_offset = 0U) {
  sllm_tensor_binding_t binding{};
  binding.struct_size = sizeof(binding);
  binding.abi_version = SLLM_HIP_ABI_VERSION;
  binding.buffer = buffer;
  binding.byte_offset = byte_offset;
  binding.dtype = SLLM_TENSOR_DTYPE_BF16;
  binding.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  binding.rank = 3U;
  binding.shape[0] = token_count;
  binding.shape[1] = SLLM_HIP_KV_HEAD_COUNT;
  binding.shape[2] = SLLM_HIP_KV_HEAD_DIM;
  binding.stride_elements[0] = SLLM_HIP_KV_HEAD_COUNT * SLLM_HIP_KV_HEAD_DIM;
  binding.stride_elements[1] = SLLM_HIP_KV_HEAD_DIM;
  binding.stride_elements[2] = 1U;
  return binding;
}

sllm_kv_append_desc_t kv_append_descriptor(const sllm_buffer_t *const key,
                                           const sllm_buffer_t *const value,
                                           const uint64_t token_count,
                                           const uint64_t position) {
  sllm_kv_append_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.append_version = SLLM_HIP_KV_STATE_VERSION;
  descriptor.expected_length = position;
  descriptor.start_position = position;
  descriptor.key_input = kv_input_binding(key, token_count);
  descriptor.value_input = kv_input_binding(value, token_count);
  return descriptor;
}

sllm_kv_append_info_t kv_append_info() {
  sllm_kv_append_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_KV_APPEND_INFO_VERSION;
  return info;
}

bool upload_kv_words(const sllm_queue_t *const queue,
                     const sllm_buffer_t *const buffer,
                     const std::vector<uint16_t> &words) {
  sllm_transfer_desc_t transfer{};
  transfer.struct_size = sizeof(transfer);
  transfer.abi_version = SLLM_HIP_ABI_VERSION;
  transfer.host_pointer = const_cast<uint16_t *>(words.data());
  transfer.size_bytes = words.size() * sizeof(uint16_t);
  sllm_completion_t *completion = nullptr;
  Error error;
  return expect_status(sllm_buffer_copy_h2d(queue, buffer, &transfer,
                                            &completion, &error.sink),
                       SLLM_STATUS_OK, "KV input upload", error) &&
         completion != nullptr &&
         query_completion(completion, SLLM_STATUS_OK) &&
         release_completion(&completion);
}

bool kv_query(const sllm_kv_state_t *const state, uint64_t length,
              uint64_t generation) {
  sllm_kv_view_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_KV_VIEW_INFO_VERSION;
  Error error;
  return expect_status(sllm_kv_state_query(state, &info, &error.sink),
                       SLLM_STATUS_OK, "KV state query", error) &&
         info.dtype == SLLM_TENSOR_DTYPE_F16 &&
         info.encoding == SLLM_TENSOR_ENCODING_UNQUANTIZED &&
         info.head_count == 4U && info.head_dim == 256U &&
         info.memory_kind == SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS &&
         info.layout == SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR &&
         info.observed_length == length && info.generation == generation &&
         info.physical_page_bytes == 2U * 1024U * 1024U &&
         info.tokens_per_page == 1024U &&
         info.mapped_token_capacity >= length &&
         info.k_stride_elements[0] == 4U * 256U &&
         info.k_stride_elements[1] == 256U && info.k_stride_elements[2] == 1U &&
         info.v_stride_elements[0] == 4U * 256U &&
         info.v_stride_elements[1] == 256U && info.v_stride_elements[2] == 1U &&
         info.context_identity != 0U && info.state_identity != 0U;
}

sllm_tensor_binding_t
causal_attention_binding(const sllm_buffer_t *const buffer,
                         const uint64_t query_count) {
  const uint64_t shape[] = {query_count, 16U, 256U};
  return attention_binding(buffer, SLLM_TENSOR_DTYPE_BF16, 3U, shape);
}

sllm_causal_attention_dispatch_info_t causal_attention_dispatch_info() {
  sllm_causal_attention_dispatch_info_t info{};
  info.struct_size = sizeof(info);
  info.abi_version = SLLM_HIP_ABI_VERSION;
  info.info_version = SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION;
  return info;
}

sllm_causal_attention_desc_t causal_attention_descriptor(
    const sllm_kv_state_t *const state, const sllm_buffer_t *const query,
    const sllm_buffer_t *const output, const uint64_t query_count,
    const uint64_t start_position, const uint64_t expected_kv_length) {
  sllm_causal_attention_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_CAUSAL_ATTENTION_VERSION;
  descriptor.start_position = start_position;
  descriptor.expected_kv_length = expected_kv_length;
  descriptor.kv_state = state;
  descriptor.query = causal_attention_binding(query, query_count);
  descriptor.output = causal_attention_binding(output, query_count);
  return descriptor;
}

uint16_t causal_float_to_bf16_rne(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += UINT32_C(0x7fff) + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>(bits >> 16U);
}

bool causal_attention_numerical_gqa_and_lifetime_contract() {
  fake_hip::reset();
  if (fake_hip::f16_to_f32_bits_for_test(UINT16_C(0x0000)) !=
          UINT32_C(0x00000000) ||
      fake_hip::f16_to_f32_bits_for_test(UINT16_C(0x8000)) !=
          UINT32_C(0x80000000) ||
      fake_hip::f16_to_f32_bits_for_test(UINT16_C(0x0001)) !=
          UINT32_C(0x33800000) ||
      fake_hip::f16_to_f32_bits_for_test(UINT16_C(0x8001)) !=
          UINT32_C(0xb3800000) ||
      fake_hip::f16_to_f32_bits_for_test(UINT16_C(0x03ff)) !=
          UINT32_C(0x387fc000) ||
      fake_hip::f16_to_f32_bits_for_test(UINT16_C(0x83ff)) !=
          UINT32_C(0xb87fc000) ||
      fake_hip::f16_to_f32_bits_for_test(UINT16_C(0x7c00)) !=
          UINT32_C(0x7f800000) ||
      fake_hip::f16_to_f32_bits_for_test(UINT16_C(0xfc00)) !=
          UINT32_C(0xff800000) ||
      (fake_hip::f16_to_f32_bits_for_test(UINT16_C(0x7e01)) &
       UINT32_C(0x7f800000)) != UINT32_C(0x7f800000) ||
      (fake_hip::f16_to_f32_bits_for_test(UINT16_C(0x7e01)) &
       UINT32_C(0x007fffff)) == 0U) {
    return false;
  }
  constexpr uint64_t query_count = 3U;
  constexpr uint64_t capacity = 3U;
  const std::size_t kv_elements = query_count * 4U * 256U;
  const std::size_t query_elements = query_count * 16U * 256U;
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_kv_state_t *state = nullptr;
  sllm_buffer_t *key = nullptr;
  sllm_buffer_t *value = nullptr;
  sllm_buffer_t *query = nullptr;
  sllm_buffer_t *output = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_kv_state(context, capacity, &state) ||
      !create_buffer_sized(context, kv_elements * sizeof(uint16_t), &key) ||
      !create_buffer_sized(context, kv_elements * sizeof(uint16_t), &value) ||
      !create_buffer_sized(context, query_elements * sizeof(uint16_t),
                           &query) ||
      !create_buffer_sized(context, query_elements * sizeof(uint16_t),
                           &output)) {
    return false;
  }

  std::vector<uint16_t> key_words(kv_elements, UINT16_C(0));
  std::vector<uint16_t> value_words(kv_elements, UINT16_C(0));
  for (uint64_t token = 0U; token != query_count; ++token) {
    for (uint64_t head = 0U; head != 4U; ++head) {
      const uint16_t word =
          causal_float_to_bf16_rne(static_cast<float>(token + head + 1U));
      for (uint64_t dimension = 0U; dimension != 256U; ++dimension) {
        value_words[(token * 4U + head) * 256U + dimension] = word;
      }
    }
  }
  std::vector<uint16_t> query_words(query_elements, UINT16_C(0));
  if (!upload_kv_words(queue, key, key_words) ||
      !upload_kv_words(queue, value, value_words) ||
      !upload_kv_words(queue, query, query_words)) {
    return false;
  }

  Error error;
  sllm_completion_t *append_completion = nullptr;
  sllm_kv_append_desc_t append =
      kv_append_descriptor(key, value, query_count, 0U);
  sllm_kv_append_info_t append_info = kv_append_info();
  if (!expect_status(sllm_kv_state_append(state, queue, &append,
                                          &append_completion, &append_info,
                                          &error.sink),
                     SLLM_STATUS_OK, "causal KV append", error) ||
      !query_completion(append_completion, SLLM_STATUS_OK) ||
      !release_completion(&append_completion) ||
      !kv_query(state, query_count, 1U)) {
    return false;
  }

  sllm_completion_t *completion = nullptr;
  sllm_causal_attention_dispatch_info_t info = causal_attention_dispatch_info();
  sllm_causal_attention_desc_t descriptor = causal_attention_descriptor(
      state, query, output, query_count, 0U, query_count);
  sllm_causal_attention_desc_t wrong_length = descriptor;
  wrong_length.expected_kv_length = query_count - 1U;
  if (!expect_status(sllm_causal_attention_execute(context, queue,
                                                   &wrong_length, &completion,
                                                   &info, &error.sink),
                     SLLM_STATUS_CAUSAL_ATTENTION_LENGTH_MISMATCH,
                     "causal wrong length", error) ||
      completion != nullptr ||
      fake_hip::causal_attention_launch_calls() != 0U) {
    return false;
  }
  sllm_causal_attention_desc_t alias = descriptor;
  alias.output = alias.query;
  info = causal_attention_dispatch_info();
  if (!expect_status(sllm_causal_attention_execute(context, queue, &alias,
                                                   &completion, &info,
                                                   &error.sink),
                     SLLM_STATUS_ALIAS_OVERLAP, "causal alias", error) ||
      completion != nullptr) {
    return false;
  }

  info = causal_attention_dispatch_info();
  if (!expect_status(sllm_causal_attention_execute(context, queue, &descriptor,
                                                   &completion, &info,
                                                   &error.sink),
                     SLLM_STATUS_OK, "causal execute", error) ||
      completion == nullptr || info.backend != SLLM_BACKEND_HIP ||
      info.dispatch_count != 1U ||
      info.kernel_id != SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_ONLINE_SOFTMAX_V2 ||
      info.workgroup_size_x != 256U || info.grid_size_x != query_count * 16U ||
      info.query_count != query_count || info.start_position != 0U ||
      info.committed_kv_length != query_count || info.q_heads != 16U ||
      info.kv_heads != 4U || info.head_dim != 256U ||
      info.scale_denominator != 16U || info.fallback_allowed != 0U ||
      info.fallback_used != 0U ||
      std::strcmp(info.kernel_symbol,
                  "causal_attention.online_softmax_gqa.v2") != 0 ||
      std::strcmp(info.device_symbol,
                  "sllm_causal_attention_online_softmax_gqa_v2") != 0 ||
      std::strcmp(info.gcn_arch_name, "gfx1201") != 0 ||
      fake_hip::causal_attention_launch_calls() != 1U) {
    return false;
  }

  fake_hip::set_completion_pending(true);
  sllm_completion_t *blocked_append_completion = nullptr;
  sllm_kv_append_info_t blocked_append_info = kv_append_info();
  if (!expect_status(sllm_kv_state_append(state, queue, &append,
                                          &blocked_append_completion,
                                          &blocked_append_info, &error.sink),
                     SLLM_STATUS_PUBLIC_BUSY, "causal append while active",
                     error) ||
      blocked_append_completion != nullptr ||
      !expect_status(sllm_kv_state_release(&state, &error.sink),
                     SLLM_STATUS_PUBLIC_BUSY, "causal state while active",
                     error) ||
      state == nullptr) {
    return false;
  }
  fake_hip::set_completion_pending(false);
  if (!query_completion(completion, SLLM_STATUS_OK) ||
      !release_completion(&completion) ||
      !expect_status(sllm_kv_state_release(&state, &error.sink), SLLM_STATUS_OK,
                     "causal state release", error)) {
    return false;
  }

  sllm_completion_t *readback = nullptr;
  if (!submit_d2h(queue, output, query_elements * sizeof(uint16_t),
                  &readback) ||
      readback == nullptr || !query_completion(readback, SLLM_STATUS_OK)) {
    return false;
  }
  std::vector<uint16_t> expected(query_elements, UINT16_C(0));
  for (uint64_t row = 0U; row != query_count; ++row) {
    for (uint64_t head = 0U; head != 16U; ++head) {
      float sum = 0.0F;
      for (uint64_t token = 0U; token <= row; ++token) {
        sum += static_cast<float>(token + head / 4U + 1U);
      }
      const uint16_t word =
          causal_float_to_bf16_rne(sum / static_cast<float>(row + 1U));
      for (uint64_t dimension = 0U; dimension != 256U; ++dimension) {
        expected[(row * 16U + head) * 256U + dimension] = word;
      }
    }
  }
  std::vector<uint16_t> actual(query_elements, UINT16_C(0));
  const bool output_matches =
      read_completion(readback, actual.data(), actual.size() * sizeof(uint16_t),
                      reinterpret_cast<const uint8_t *>(expected.data()),
                      expected.size() * sizeof(uint16_t));
  const bool readback_released = release_completion(&readback);
  const bool buffers_released =
      release_buffer(&key) && release_buffer(&value) &&
      release_buffer(&query) && release_buffer(&output);
  return output_matches && readback_released && buffers_released &&
         release_queue(&queue) && release_context(&context);
}

bool kv_append_accounting_multiplicity_contract() {
  using sllm_public_runtime::AccountingState;

  const auto reservation_must_fail_without_mutation =
      [](const bool active_exhausted, const bool completion_exhausted) {
        AccountingState context{};
        AccountingState queue{};
        AccountingState state{};
        AccountingState shared_input{};
        AccountingState key_buffer{};
        AccountingState value_buffer{};
        if (active_exhausted) {
          shared_input.active_submissions = UINT64_MAX - 1U;
        }
        if (completion_exhausted) {
          shared_input.completion_references = UINT64_MAX - 1U;
        }
        const bool reserved = AccountingState::reserve_kv_append(
            context, queue, state, shared_input, shared_input, key_buffer,
            value_buffer);
        return !reserved &&
               shared_input.active_submissions ==
                   (active_exhausted ? UINT64_MAX - 1U : 0U) &&
               shared_input.completion_references ==
                   (completion_exhausted ? UINT64_MAX - 1U : 0U) &&
               queue.active_submissions == 0U &&
               queue.completion_references == 0U && context.child_count == 0U &&
               context.lifetime_guards == 0U;
      };
  if (!reservation_must_fail_without_mutation(true, false) ||
      !reservation_must_fail_without_mutation(false, true) ||
      !reservation_must_fail_without_mutation(true, true)) {
    std::cerr
        << "KV duplicate input reservation did not fail closed at max-1\n";
    return false;
  }

  AccountingState context{};
  AccountingState queue{};
  AccountingState state{};
  AccountingState shared_input{};
  AccountingState key_buffer{};
  AccountingState value_buffer{};
  const bool reserved = AccountingState::reserve_kv_append(
      context, queue, state, shared_input, shared_input, key_buffer,
      value_buffer);
  const bool active_released = AccountingState::release_kv_active(
      queue, state, shared_input, shared_input, key_buffer, value_buffer);
  const bool completion_released = AccountingState::release_kv_completion(
      context, queue, state, shared_input, shared_input, key_buffer,
      value_buffer);
  if (!reserved || shared_input.active_submissions != 0U ||
      shared_input.completion_references != 0U ||
      queue.active_submissions != 0U || queue.completion_references != 0U ||
      context.child_count != 0U || context.lifetime_guards != 0U ||
      !active_released || !completion_released) {
    std::cerr << "KV duplicate input active/completion release was asymmetric: "
              << reserved << ", " << active_released << ", "
              << completion_released << "; shared active/completion="
              << shared_input.active_submissions << "/"
              << shared_input.completion_references
              << "; queue active/"
                 "completion="
              << queue.active_submissions << "/" << queue.completion_references
              << "; context child/guard=" << context.child_count << "/"
              << context.lifetime_guards << "\n";
    return false;
  }
  if (!AccountingState::reserve_kv_append(context, queue, state, shared_input,
                                          shared_input, key_buffer,
                                          value_buffer) ||
      !AccountingState::rollback_kv_append(context, queue, state, shared_input,
                                           shared_input, key_buffer,
                                           value_buffer) ||
      shared_input.active_submissions != 0U ||
      shared_input.completion_references != 0U ||
      queue.active_submissions != 0U || queue.completion_references != 0U ||
      context.child_count != 0U || context.lifetime_guards != 0U) {
    std::cerr << "KV duplicate input rollback was asymmetric\n";
    return false;
  }
  return true;
}

bool kv_append_same_buffer_disjoint_lifecycle_contract() {
  fake_hip::reset();
  constexpr uint64_t capacity = 2U;
  constexpr uint64_t elements_per_input = 4U * 256U;
  constexpr uint64_t input_bytes = elements_per_input * sizeof(uint16_t);
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_kv_state_t *state = nullptr;
  sllm_buffer_t *shared = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_kv_state(context, capacity, &state) ||
      !create_buffer_sized(context, input_bytes * 2U, &shared)) {
    if (shared != nullptr) {
      (void)release_buffer(&shared);
    }
    if (state != nullptr) {
      Error error;
      (void)sllm_kv_state_release(&state, &error.sink);
    }
    if (queue != nullptr) {
      (void)release_queue(&queue);
    }
    if (context != nullptr) {
      (void)release_context(&context);
    }
    return false;
  }

  std::vector<uint16_t> words(elements_per_input * 2U, 0x3f80U);
  for (uint64_t index = elements_per_input; index != words.size(); ++index) {
    words[static_cast<std::size_t>(index)] = 0x4000U;
  }
  Error error;
  bool valid = upload_kv_words(queue, shared, words);
  sllm_completion_t *completion = nullptr;
  auto descriptor = kv_append_descriptor(shared, shared, 1U, 0U);
  descriptor.value_input.byte_offset = input_bytes;
  sllm_kv_append_info_t info = kv_append_info();

  fake_hip::set_completion_pending(true);
  valid = valid &&
          expect_status(sllm_kv_state_append(state, queue, &descriptor,
                                             &completion, &info, &error.sink),
                        SLLM_STATUS_OK, "KV same-buffer append", error) &&
          completion != nullptr &&
          expect_status(sllm_buffer_release(&shared, &error.sink),
                        SLLM_STATUS_PUBLIC_BUSY,
                        "KV same-buffer pending buffer release", error) &&
          shared != nullptr && kv_query(state, 0U, 0U);
  fake_hip::set_completion_pending(false);
  valid = valid && query_completion(completion, SLLM_STATUS_OK) &&
          release_completion(&completion) && kv_query(state, 1U, 1U);

  fake_hip::set_kv_state_append_launch_status(hipErrorUnknown);
  descriptor.expected_length = 1U;
  descriptor.start_position = 1U;
  info = kv_append_info();
  valid = valid &&
          expect_status(sllm_kv_state_append(state, queue, &descriptor,
                                             &completion, &info, &error.sink),
                        SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
                        "KV same-buffer launch rollback", error) &&
          completion == nullptr && kv_query(state, 1U, 1U);
  fake_hip::set_kv_state_append_launch_status(hipSuccess);
  valid = valid &&
          expect_status(sllm_kv_state_append(state, queue, &descriptor,
                                             &completion, &info, &error.sink),
                        SLLM_STATUS_OK, "KV same-buffer reuse", error) &&
          completion != nullptr &&
          query_completion(completion, SLLM_STATUS_OK) &&
          release_completion(&completion) && kv_query(state, 2U, 2U);

  const bool shared_released = release_buffer(&shared);
  const bool state_released =
      expect_status(sllm_kv_state_release(&state, &error.sink), SLLM_STATUS_OK,
                    "KV same-buffer state release", error);
  const bool queue_released = release_queue(&queue);
  const bool context_released = release_context(&context);
  return valid && shared_released && state_released && queue_released &&
         context_released;
}

bool kv_state_create_snapshot_contract() {
  fake_hip::reset();
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_kv_state_t *state = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue)) {
    release_queue(&queue);
    release_context(&context);
    return false;
  }
  sllm_kv_state_create_info_t invalid{};
  invalid.struct_size = sizeof(invalid);
  invalid.abi_version = SLLM_HIP_ABI_VERSION;
  invalid.session_id = 1U;
  invalid.capacity_tokens = 17U;
  invalid.memory_kind = SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS;
  invalid.layout = 99U;
  Error error;
  bool valid = expect_status(
                   sllm_kv_state_create(context, &invalid, &state, &error.sink),
                   SLLM_STATUS_INVALID_KV_STATE_DESCRIPTOR,
                   "KV invalid layout create", error) &&
               state == nullptr;
  invalid.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
  invalid.abi_version = SLLM_HIP_ABI_VERSION + 1U;
  valid = valid &&
          expect_status(
              sllm_kv_state_create(context, &invalid, &state, &error.sink),
              SLLM_STATUS_INVALID_ABI_VERSION, "KV old ABI create", error) &&
          state == nullptr;
  valid = valid && create_kv_state(context, 257U, &state) &&
          kv_query(state, 0U, 0U);
  sllm_kv_view_t *view = nullptr;
  sllm_kv_view_info_t view_info{};
  view_info.struct_size = sizeof(view_info);
  view_info.abi_version = SLLM_HIP_ABI_VERSION;
  view_info.info_version = SLLM_HIP_KV_VIEW_INFO_VERSION;
  valid = valid &&
          expect_status(sllm_kv_state_snapshot(state, &view, &error.sink),
                        SLLM_STATUS_OK, "KV state snapshot", error) &&
          view != nullptr && ([&]() {
            view_info.struct_size -= 1U;
            const bool result = expect_status(
                sllm_kv_view_query(view, &view_info, &error.sink),
                SLLM_STATUS_INVALID_ARGUMENT, "KV view wrong size", error);
            view_info.struct_size = sizeof(view_info);
            return result;
          })() &&
          expect_status(sllm_kv_view_query(view, &view_info, &error.sink),
                        SLLM_STATUS_OK, "KV view query", error) &&
          view_info.observed_length == 0U && view_info.generation == 0U &&
          view_info.capacity_tokens == 257U && view_info.state_identity != 0U &&
          view_info.context_identity != 0U &&
          expect_status(sllm_kv_state_release(&state, &error.sink),
                        SLLM_STATUS_PUBLIC_BUSY, "KV state live view Busy",
                        error) &&
          expect_status(sllm_kv_view_release(&view, &error.sink),
                        SLLM_STATUS_OK, "KV view release", error) &&
          view == nullptr &&
          expect_status(sllm_kv_state_release(&state, &error.sink),
                        SLLM_STATUS_OK, "KV state release", error);
  return valid && release_queue(&queue) && release_context(&context);
}

bool kv_lowbit_create_query_and_recipe_contract() {
  fake_hip::reset();
  const std::size_t baseline_allocations = fake_hip::live_allocations();
  sllm_context_t *context = nullptr;
  if (!create_context(&context)) {
    return false;
  }
  Error error;
  sllm_kv_state_create_info_v2_t create{};
  create.struct_size = sizeof(create);
  create.abi_version = SLLM_HIP_ABI_VERSION;
  create.create_info_version = SLLM_HIP_KV_STATE_CREATE_INFO_V2_VERSION;
  create.session_id = 0x16U;
  create.layer_id = 16U;
  create.capacity_tokens = 257U;
  create.head_count = 4U;
  create.head_dim = 256U;
  create.memory_kind = SLLM_HIP_KV_MEMORY_KIND_CONTIGUOUS_RESIDENT;
  create.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
  create.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
  create.encoding = SLLM_HIP_KV_ENCODING_FP8_V1;
  create.scale_dtype = SLLM_TENSOR_DTYPE_F32;

  const auto query_recipe = [&](const uint32_t expected_dtype,
                                const uint32_t expected_encoding,
                                const uint64_t bytes_per_plane) {
    sllm_kv_state_t *state = nullptr;
    if (!expect_status(
            sllm_kv_state_create_v2(context, &create, &state, &error.sink),
            SLLM_STATUS_OK, "low-bit KV create", error) ||
        state == nullptr) {
      return false;
    }
    sllm_kv_view_info_t view{};
    view.struct_size = sizeof(view);
    view.abi_version = SLLM_HIP_ABI_VERSION;
    view.info_version = SLLM_HIP_KV_VIEW_INFO_VERSION;
    const bool valid =
        expect_status(sllm_kv_state_query(state, &view, &error.sink),
                      SLLM_STATUS_OK, "low-bit KV query", error) &&
        view.dtype == expected_dtype && view.encoding == expected_encoding &&
        view.memory_kind == SLLM_HIP_KV_MEMORY_KIND_CONTIGUOUS_RESIDENT &&
        view.mapped_token_capacity == create.capacity_tokens &&
        view.tokens_per_page == 1U &&
        view.committed_bytes_per_plane == bytes_per_plane;
    const uint64_t row_stride =
        create.encoding == SLLM_HIP_KV_ENCODING_FP8_E4_BLOCK16_V2 ||
                create.encoding == SLLM_HIP_KV_ENCODING_FP8_E5_BLOCK16_V2
            ? ((static_cast<uint64_t>(create.head_dim) + 15U) / 16U) * 16U
        : create.encoding == SLLM_HIP_KV_ENCODING_MXFP8_E4_V1 ||
                create.encoding == SLLM_HIP_KV_ENCODING_MXFP8_E5_V1
            ? ((static_cast<uint64_t>(create.head_dim) + 31U) / 32U) * 32U
            : create.head_dim;
    const bool stride_valid = view.k_stride_elements[0] == 4U * row_stride &&
                              view.k_stride_elements[1] == row_stride &&
                              view.k_stride_elements[2] == 1U &&
                              view.v_stride_elements[0] == 4U * row_stride &&
                              view.v_stride_elements[1] == row_stride &&
                              view.v_stride_elements[2] == 1U;
    const bool metadata_valid = valid && stride_valid;
    if (!metadata_valid) {
      std::cerr << "low-bit KV query metadata mismatch: dtype=" << view.dtype
                << " encoding=" << view.encoding
                << " memory_kind=" << view.memory_kind
                << " mapped=" << view.mapped_token_capacity
                << " tokens_per_page=" << view.tokens_per_page
                << " committed=" << view.committed_bytes_per_plane
                << " expected_committed=" << bytes_per_plane
                << " row_stride=" << row_stride << '\n';
    }
    sllm_kv_view_t *snapshot = nullptr;
    uint8_t output[2]{};
    sllm_hip_kv_readback_request_t readback{};
    readback.struct_size = sizeof(readback);
    readback.abi_version = SLLM_HIP_KV_EVIDENCE_ABI_VERSION;
    readback.plane = SLLM_HIP_KV_EVIDENCE_PLANE_K;
    readback.byte_length = sizeof(output);
    readback.host_capacity = sizeof(output);
    readback.host_output = output;
    bool readback_rejected =
        expect_status(sllm_kv_state_snapshot(state, &snapshot, &error.sink),
                      SLLM_STATUS_OK, "low-bit KV snapshot", error) &&
        snapshot != nullptr;
    if (snapshot != nullptr) {
      readback.view = snapshot;
      readback_rejected =
          expect_status(sllm_hip_kv_view_readback(&readback, &error.sink),
                        SLLM_STATUS_UNSUPPORTED_ENCODING,
                        "low-bit KV v1 readback rejection", error) &&
          readback_rejected;
      readback_rejected =
          expect_status(sllm_kv_view_release(&snapshot, &error.sink),
                        SLLM_STATUS_OK, "low-bit KV snapshot release", error) &&
          snapshot == nullptr && readback_rejected;
    }
    return expect_status(sllm_kv_state_release(&state, &error.sink),
                         SLLM_STATUS_OK, "low-bit KV release", error) &&
           state == nullptr && metadata_valid && readback_rejected;
  };

  bool valid = query_recipe(
      SLLM_TENSOR_DTYPE_F8_E4M3_FN, SLLM_TENSOR_ENCODING_FP8_OUTER_F32,
      create.capacity_tokens * (4U * 256U + 4U * sizeof(float)));
  create.dtype = SLLM_TENSOR_DTYPE_U8;
  create.encoding = SLLM_HIP_KV_ENCODING_NVFP4_V1;
  create.block_size = 16U;
  create.scale_dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
  valid = valid && query_recipe(SLLM_TENSOR_DTYPE_U8,
                                SLLM_TENSOR_ENCODING_NVFP4_BLOCK16_E4M3FN_F32,
                                create.capacity_tokens *
                                    (4U * (256U / 2U) + 4U * (256U / 16U) +
                                     4U * sizeof(float)));
  create.block_size = 15U;
  sllm_kv_state_t *invalid_state = nullptr;
  valid = valid &&
          expect_status(sllm_kv_state_create_v2(context, &create,
                                                &invalid_state, &error.sink),
                        SLLM_STATUS_INVALID_KV_STATE_DESCRIPTOR,
                        "invalid NVFP4 KV recipe", error) &&
          invalid_state == nullptr;
  // The historical block16 encodings remain reserved ABI values but are no
  // longer accepted by state creation.
  create.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
  create.encoding = SLLM_HIP_KV_ENCODING_FP8_E4_BLOCK16_V1;
  create.block_size = 16U;
  create.scale_dtype = SLLM_TENSOR_DTYPE_U8;
  sllm_kv_state_t *historical_v1_state = nullptr;
  valid =
      valid &&
      expect_status(sllm_kv_state_create_v2(context, &create,
                                            &historical_v1_state, &error.sink),
                    SLLM_STATUS_INVALID_KV_STATE_DESCRIPTOR,
                    "historical block16 v1 encoding rejection", error) &&
      historical_v1_state == nullptr;
  create.encoding = SLLM_HIP_KV_ENCODING_FP8_E4_BLOCK16_V2;
  sllm_kv_state_t *retired_block16_state = nullptr;
  valid =
      valid &&
      expect_status(sllm_kv_state_create_v2(
                        context, &create, &retired_block16_state, &error.sink),
                    SLLM_STATUS_INVALID_KV_STATE_DESCRIPTOR,
                    "retired block16 v2 encoding rejection", error) &&
      retired_block16_state == nullptr;
  // The same canonical-tail guard applies to standard MXFP8's 32-lane rows.
  create.head_dim = 33U;
  create.encoding = SLLM_HIP_KV_ENCODING_MXFP8_E4_V1;
  create.block_size = 32U;
  create.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
  create.scale_dtype = SLLM_TENSOR_DTYPE_U8;
  sllm_kv_state_t *mx_tail_state = nullptr;
  const uint64_t mx_tail_value_bytes = create.capacity_tokens * 4U * 64U;
  const bool mx_tail_created = expect_status(
      sllm_kv_state_create_v2(context, &create, &mx_tail_state, &error.sink),
      SLLM_STATUS_OK, "MXFP8 tail import state create", error);
  const std::size_t mx_tail_allocations = fake_hip::live_allocations();
  std::vector<uint8_t> mx_tail_image(
      static_cast<std::size_t>(mx_tail_value_bytes), 0U);
  mx_tail_image[33U] = 1U; // token 0, head 0, first padded lane
  sllm_state_chunk_t mx_tail_chunk{};
  mx_tail_chunk.struct_size = sizeof(mx_tail_chunk);
  mx_tail_chunk.abi_version = SLLM_HIP_ABI_VERSION;
  mx_tail_chunk.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
  mx_tail_chunk.plane = SLLM_HIP_KV_STATE_PLANE_KEY;
  mx_tail_chunk.byte_length = mx_tail_value_bytes;
  mx_tail_chunk.host_pointer = mx_tail_image.data();
  mx_tail_chunk.host_capacity = mx_tail_value_bytes;
  const bool mx_tail_rejected =
      mx_tail_created &&
      expect_status(
          sllm_kv_state_import(mx_tail_state, &mx_tail_chunk, &error.sink),
          SLLM_STATUS_INVALID_ARGUMENT, "nonzero MXFP8 tail import rejection",
          error);
  sllm_kv_view_info_t mx_tail_view{};
  mx_tail_view.struct_size = sizeof(mx_tail_view);
  mx_tail_view.abi_version = SLLM_HIP_ABI_VERSION;
  mx_tail_view.info_version = SLLM_HIP_KV_VIEW_INFO_VERSION;
  const bool mx_tail_unchanged =
      mx_tail_created &&
      expect_status(
          sllm_kv_state_query(mx_tail_state, &mx_tail_view, &error.sink),
          SLLM_STATUS_OK, "MXFP8 tail state query", error) &&
      mx_tail_view.observed_length == 0U && mx_tail_view.generation == 0U &&
      mx_tail_view.committed_bytes_per_plane ==
          mx_tail_value_bytes + create.capacity_tokens * 4U * 2U &&
      fake_hip::live_allocations() == mx_tail_allocations;
  const bool mx_tail_released =
      mx_tail_state == nullptr ||
      (expect_status(sllm_kv_state_release(&mx_tail_state, &error.sink),
                     SLLM_STATUS_OK, "MXFP8 tail state release", error) &&
       mx_tail_state == nullptr);
  valid = valid && mx_tail_rejected && mx_tail_unchanged && mx_tail_released;
  create.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
  create.encoding = SLLM_HIP_KV_ENCODING_MXFP8_E4_V1;
  create.block_size = 32U;
  create.scale_dtype = SLLM_TENSOR_DTYPE_U8;
  for (const uint32_t head_dim :
       {15U, 16U, 17U, 31U, 32U, 33U, 255U, 256U, 257U}) {
    create.head_dim = head_dim;
    const uint64_t padded =
        ((static_cast<uint64_t>(head_dim) + 31U) / 32U) * 32U;
    const uint64_t values = create.capacity_tokens * 4U * padded;
    const uint64_t scales =
        create.capacity_tokens * 4U * ((head_dim + 31U) / 32U);
    valid = valid && query_recipe(SLLM_TENSOR_DTYPE_F8_E4M3_FN,
                                  SLLM_TENSOR_ENCODING_MXFP8_BLOCK32_E8M0,
                                  values + scales);
  }
  const bool first_context_released = release_context(&context);
  valid = valid && first_context_released;
  create.head_dim = 256U;
  fake_hip::set_gcn_arch_name("gfx1030");
  create.dtype = SLLM_TENSOR_DTYPE_F8_E5M2;
  create.encoding = SLLM_HIP_KV_ENCODING_MXFP8_E5_V1;
  create.block_size = 32U;
  create.scale_dtype = SLLM_TENSOR_DTYPE_U8;
  const bool second_context_created =
      create_context_for_arch("gfx1030", &context);
  valid = valid && second_context_created;
  if (second_context_created) {
    create.encoding = SLLM_HIP_KV_ENCODING_MXFP8_E5_V1;
    create.block_size = 32U;
    for (const uint32_t head_dim :
         {15U, 16U, 17U, 31U, 32U, 33U, 255U, 256U, 257U}) {
      create.head_dim = head_dim;
      const uint64_t padded =
          ((static_cast<uint64_t>(head_dim) + 31U) / 32U) * 32U;
      const uint64_t values = create.capacity_tokens * 4U * padded;
      const uint64_t scales =
          create.capacity_tokens * 4U * ((head_dim + 31U) / 32U);
      valid = valid && query_recipe(SLLM_TENSOR_DTYPE_F8_E5M2,
                                    SLLM_TENSOR_ENCODING_MXFP8_BLOCK32_E8M0,
                                    values + scales);
    }
  }
  const bool context_released = release_context(&context);
  const std::size_t live_allocations = fake_hip::live_allocations();
  if (!valid || !context_released || live_allocations != baseline_allocations) {
    std::cerr << "low-bit KV cleanup mismatch: valid=" << valid
              << " context_released=" << context_released
              << " live_allocations=" << live_allocations
              << " baseline_allocations=" << baseline_allocations << '\n';
  }
  return valid && context_released && live_allocations == baseline_allocations;
}

bool kv_capability_selected_contiguous_resident_contract() {
  fake_hip::reset();
  fake_hip::set_vmm_supported(false);
  const std::size_t baseline_allocations = fake_hip::live_allocations();
  constexpr uint64_t capacity = 1025U;
  constexpr uint64_t bytes_per_token = 4U * 256U * sizeof(uint16_t);
  constexpr uint64_t plane_bytes = capacity * bytes_per_token;
  sllm_context_t *context = nullptr;
  sllm_kv_state_t *state = nullptr;
  if (!create_context(&context)) {
    return false;
  }

  sllm_kv_state_create_info_t create_info{};
  create_info.struct_size = sizeof(create_info);
  create_info.abi_version = SLLM_HIP_ABI_VERSION;
  create_info.session_id = 0x942U;
  create_info.layer_id = 11U;
  create_info.capacity_tokens = capacity;
  create_info.memory_kind = SLLM_HIP_KV_MEMORY_KIND_CAPABILITY_SELECTED;
  create_info.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
  Error error;
  bool valid =
      expect_status(
          sllm_kv_state_create(context, &create_info, &state, &error.sink),
          SLLM_STATUS_OK, "capability-selected contiguous KV create", error) &&
      state != nullptr;

  sllm_kv_view_info_t view_info{};
  view_info.struct_size = sizeof(view_info);
  view_info.abi_version = SLLM_HIP_ABI_VERSION;
  view_info.info_version = SLLM_HIP_KV_VIEW_INFO_VERSION;
  valid =
      valid &&
      expect_status(sllm_kv_state_query(state, &view_info, &error.sink),
                    SLLM_STATUS_OK, "capability-selected contiguous KV query",
                    error) &&
      view_info.memory_kind == SLLM_HIP_KV_MEMORY_KIND_CONTIGUOUS_RESIDENT &&
      view_info.capacity_tokens == capacity &&
      view_info.physical_page_bytes == bytes_per_token &&
      view_info.tokens_per_page == 1U &&
      view_info.mapped_token_capacity == capacity &&
      view_info.committed_bytes_per_plane == plane_bytes;

  valid =
      valid &&
      expect_status(sllm_kv_state_release(&state, &error.sink), SLLM_STATUS_OK,
                    "capability-selected contiguous KV release", error) &&
      state == nullptr && fake_hip::live_allocations() == baseline_allocations;

  create_info.memory_kind = SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS;
  valid =
      valid &&
      expect_status(
          sllm_kv_state_create(context, &create_info, &state, &error.sink),
          SLLM_STATUS_UNSUPPORTED, "explicit virtual KV without VMM", error) &&
      state == nullptr;
  return valid && release_context(&context);
}

bool kv_evidence_readback_contract() {
  fake_hip::reset();
  constexpr uint64_t capacity = 3U;
  constexpr std::size_t input_words = 4U * 256U;
  constexpr uint64_t input_bytes = input_words * sizeof(uint16_t);
  constexpr uint64_t head_bytes = 256U * sizeof(uint16_t);
  constexpr uint64_t plane_bytes = capacity * 4U * 256U * sizeof(uint16_t);
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_kv_state_t *state = nullptr;
  sllm_buffer_t *key = nullptr;
  sllm_buffer_t *value = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_kv_state(context, capacity, &state) ||
      !create_buffer_sized(context, input_bytes, &key) ||
      !create_buffer_sized(context, input_bytes, &value)) {
    release_buffer(&key);
    release_buffer(&value);
    if (state != nullptr) {
      Error error;
      (void)sllm_kv_state_release(&state, &error.sink);
    }
    release_queue(&queue);
    release_context(&context);
    return false;
  }
  std::vector<uint16_t> key_words(input_words, 0x3f80U);
  std::vector<uint16_t> value_words(input_words, 0x4000U);
  Error error;
  bool valid = upload_kv_words(queue, key, key_words) &&
               upload_kv_words(queue, value, value_words);
  auto make_request = [](const sllm_kv_view_t *const view, const uint32_t plane,
                         const uint64_t offset, const uint64_t length,
                         const uint64_t capacity_bytes, uint8_t *const output) {
    sllm_hip_kv_readback_request_t result{};
    result.struct_size = sizeof(result);
    result.abi_version = SLLM_HIP_KV_EVIDENCE_ABI_VERSION;
    result.view = view;
    result.plane = plane;
    result.byte_offset = offset;
    result.byte_length = length;
    result.host_capacity = capacity_bytes;
    result.host_output = output;
    return result;
  };

  sllm_kv_view_t *empty_view = nullptr;
  valid = valid &&
          expect_status(sllm_kv_state_snapshot(state, &empty_view, &error.sink),
                        SLLM_STATUS_OK, "KV evidence empty snapshot", error);
  std::vector<uint8_t> output(16U, 0xa5U);
  auto empty_request =
      make_request(empty_view, SLLM_HIP_KV_EVIDENCE_PLANE_K, 0U, output.size(),
                   output.size(), output.data());
  valid = valid &&
          expect_status(sllm_hip_kv_view_readback(&empty_request, &error.sink),
                        SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
                        "KV evidence unmapped empty readback", error);
  auto wrong_kind = empty_request;
  wrong_kind.view = reinterpret_cast<const sllm_kv_view_t *>(state);
  valid = valid &&
          expect_status(sllm_hip_kv_view_readback(&wrong_kind, &error.sink),
                        SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                        "KV evidence wrong-kind handle", error);
  auto undersized = empty_request;
  undersized.host_capacity = 1U;
  valid = valid &&
          expect_status(sllm_hip_kv_view_readback(&undersized, &error.sink),
                        SLLM_STATUS_BUFFER_TOO_SMALL,
                        "KV evidence undersized host output", error);
  auto null_output = empty_request;
  null_output.host_output = nullptr;
  valid = valid &&
          expect_status(sllm_hip_kv_view_readback(&null_output, &error.sink),
                        SLLM_STATUS_INVALID_ARGUMENT,
                        "KV evidence null host output", error);
  auto reserved = empty_request;
  reserved.reserved[0] = 1U;
  valid =
      valid && expect_status(sllm_hip_kv_view_readback(&reserved, &error.sink),
                             SLLM_STATUS_RESERVED_NONZERO,
                             "KV evidence reserved field", error);
  auto out_of_bounds = empty_request;
  out_of_bounds.byte_offset = plane_bytes;
  valid =
      valid &&
      expect_status(sllm_hip_kv_view_readback(&out_of_bounds, &error.sink),
                    SLLM_STATUS_BUFFER_OUT_OF_BOUNDS,
                    "KV evidence plane bounds", error) &&
      expect_status(sllm_kv_view_release(&empty_view, &error.sink),
                    SLLM_STATUS_OK, "KV evidence empty view release", error);

  sllm_kv_append_desc_t descriptor = kv_append_descriptor(key, value, 1U, 0U);
  sllm_kv_append_info_t append_info = kv_append_info();
  sllm_completion_t *completion = nullptr;
  valid =
      valid &&
      expect_status(sllm_kv_state_append(state, queue, &descriptor, &completion,
                                         &append_info, &error.sink),
                    SLLM_STATUS_OK, "KV evidence append", error) &&
      query_completion(completion, SLLM_STATUS_OK) &&
      release_completion(&completion);
  sllm_kv_view_t *live_view = nullptr;
  valid = valid &&
          expect_status(sllm_kv_state_snapshot(state, &live_view, &error.sink),
                        SLLM_STATUS_OK, "KV evidence live snapshot", error);
  for (uint64_t head = 0U; head != 4U; ++head) {
    const uint64_t head_offset = head * head_bytes;
    std::fill(output.begin(), output.end(), 0xa5U);
    auto key_request =
        make_request(live_view, SLLM_HIP_KV_EVIDENCE_PLANE_K, head_offset,
                     output.size(), output.size(), output.data());
    valid = valid &&
            expect_status(sllm_hip_kv_view_readback(&key_request, &error.sink),
                          SLLM_STATUS_OK, "KV evidence K readback", error);
    for (std::size_t index = 0U; index != output.size(); index += 2U) {
      valid = valid && output[index] == 0x00U && output[index + 1U] == 0x3cU;
    }

    std::fill(output.begin(), output.end(), 0xa5U);
    auto value_request =
        make_request(live_view, SLLM_HIP_KV_EVIDENCE_PLANE_V, head_offset,
                     output.size(), output.size(), output.data());
    valid = valid && expect_status(
                         sllm_hip_kv_view_readback(&value_request, &error.sink),
                         SLLM_STATUS_OK, "KV evidence V readback", error);
    for (std::size_t index = 0U; index != output.size(); index += 2U) {
      valid = valid && output[index] == 0x00U && output[index + 1U] == 0x40U;
    }
  }
  const sllm_kv_view_t *const stale_view = live_view;
  valid = valid &&
          expect_status(sllm_kv_view_release(&live_view, &error.sink),
                        SLLM_STATUS_OK, "KV evidence live view release", error);
  auto stale_request =
      make_request(stale_view, SLLM_HIP_KV_EVIDENCE_PLANE_K, 0U, output.size(),
                   output.size(), output.data());
  valid = valid &&
          expect_status(sllm_hip_kv_view_readback(&stale_request, &error.sink),
                        SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                        "KV evidence stale handle", error);

  fake_hip::set_completion_pending(true);
  descriptor = kv_append_descriptor(key, value, 1U, 1U);
  append_info = kv_append_info();
  valid =
      valid &&
      expect_status(sllm_kv_state_append(state, queue, &descriptor, &completion,
                                         &append_info, &error.sink),
                    SLLM_STATUS_OK, "KV evidence pending append", error);
  sllm_kv_view_t *pending_view = nullptr;
  valid =
      valid &&
      expect_status(sllm_kv_state_snapshot(state, &pending_view, &error.sink),
                    SLLM_STATUS_OK, "KV evidence pending snapshot", error);
  auto pending_request =
      make_request(pending_view, SLLM_HIP_KV_EVIDENCE_PLANE_K, 0U,
                   output.size(), output.size(), output.data());
  valid =
      valid &&
      expect_status(sllm_hip_kv_view_readback(&pending_request, &error.sink),
                    SLLM_STATUS_PUBLIC_BUSY, "KV evidence pending readback",
                    error) &&
      expect_status(sllm_kv_view_release(&pending_view, &error.sink),
                    SLLM_STATUS_OK, "KV evidence pending view release", error);
  fake_hip::set_completion_pending(false);
  valid = valid && query_completion(completion, SLLM_STATUS_OK) &&
          release_completion(&completion) && release_buffer(&key) &&
          release_buffer(&value) &&
          expect_status(sllm_kv_state_release(&state, &error.sink),
                        SLLM_STATUS_OK, "KV evidence state release", error) &&
          release_queue(&queue) && release_context(&context);
  return valid;
}

bool kv_append_layout_and_transaction_contract() {
  fake_hip::reset();
  constexpr uint64_t capacity = 257U;
  constexpr uint64_t max_tokens = 255U;
  const std::size_t input_bytes =
      static_cast<std::size_t>(max_tokens * 4U * 256U * sizeof(uint16_t));
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_kv_state_t *state = nullptr;
  sllm_buffer_t *key = nullptr;
  sllm_buffer_t *value = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_kv_state(context, capacity, &state) ||
      !create_buffer_sized(context, input_bytes, &key) ||
      !create_buffer_sized(context, input_bytes, &value)) {
    release_buffer(&key);
    release_buffer(&value);
    release_queue(&queue);
    release_context(&context);
    return false;
  }
  const uint16_t bf16_values[4] = {0x3f80U, 0x4000U, 0x4040U, 0x4080U};
  const uint16_t f16_values[4] = {0x3c00U, 0x4000U, 0x4200U, 0x4400U};
  auto make_words = [&](const uint64_t tokens, const uint32_t shift) {
    std::vector<uint16_t> words(tokens * 4U * 256U);
    for (std::size_t index = 0U; index != words.size(); ++index) {
      words[index] = bf16_values[(index + shift) % 4U];
    }
    return words;
  };
  Error error;
  bool valid = true;
  std::vector<uint16_t> first = make_words(1U, 0U);
  const uint16_t special_bf16[] = {0x8000U, 0x7f80U, 0xff80U,
                                   0x7fc1U, 0xffc1U, 0x8001U};
  const uint16_t special_f16[] = {0x8000U, 0x7c00U, 0xfc00U,
                                  0x7e00U, 0xfe00U, 0x8000U};
  constexpr std::size_t special_count =
      sizeof(special_bf16) / sizeof(special_bf16[0]);
  for (std::size_t index = 0U; index != special_count; ++index) {
    first[index] = special_bf16[index];
  }
  std::vector<uint16_t> three_key = make_words(3U, 0U);
  std::vector<uint16_t> three_value = make_words(3U, 1U);
  valid = valid && upload_kv_words(queue, key, first) &&
          upload_kv_words(queue, value, first);
  sllm_completion_t *completion = nullptr;
  sllm_kv_append_info_t info = kv_append_info();
  auto descriptor = kv_append_descriptor(key, value, 1U, 0U);
  valid = valid &&
          expect_status(sllm_kv_state_append(state, queue, &descriptor,
                                             &completion, &info, &error.sink),
                        SLLM_STATUS_OK, "KV first append", error) &&
          completion != nullptr && info.token_count == 1U &&
          info.end_position == 1U && info.commit_allowed == 1U &&
          info.fallback_allowed == 0U && info.fallback_used == 0U &&
          info.grid_size_x == 4U &&
          std::strcmp(info.kernel_symbol,
                      "kv_state.bf16_to_f16_token_major.v2") == 0 &&
          query_completion(completion, SLLM_STATUS_OK) &&
          release_completion(&completion) && kv_query(state, 1U, 1U);
  std::vector<uint16_t> first_key_output(4U * capacity * 256U);
  std::vector<uint16_t> first_value_output(4U * capacity * 256U);
  valid = valid &&
          fake_hip::copy_kv_key_output(first_key_output.data(),
                                       first_key_output.size()) &&
          fake_hip::copy_kv_value_output(first_value_output.data(),
                                         first_value_output.size());
  for (std::size_t index = 0U; index != special_count; ++index) {
    const std::size_t offset = index;
    if (first_key_output[offset] != special_f16[index] ||
        first_value_output[offset] != special_f16[index]) {
      valid = false;
    }
  }
  valid = valid && upload_kv_words(queue, key, three_key) &&
          upload_kv_words(queue, value, three_value);
  descriptor = kv_append_descriptor(key, value, 3U, 1U);
  info = kv_append_info();
  valid =
      valid &&
      expect_status(sllm_kv_state_append(state, queue, &descriptor, &completion,
                                         &info, &error.sink),
                    SLLM_STATUS_OK, "KV non-aligned M/start append", error) &&
      completion != nullptr && info.grid_size_x == 12U &&
      fake_hip::kv_state_last_token_count() == 3U &&
      fake_hip::kv_state_last_capacity_tokens() == capacity &&
      fake_hip::kv_state_last_start_position() == 1U &&
      query_completion(completion, SLLM_STATUS_OK) &&
      release_completion(&completion);
  std::vector<uint16_t> key_output(4U * capacity * 256U);
  std::vector<uint16_t> value_output(4U * capacity * 256U);
  valid =
      valid &&
      fake_hip::copy_kv_key_output(key_output.data(), key_output.size()) &&
      fake_hip::copy_kv_value_output(value_output.data(), value_output.size());
  for (uint64_t row = 0U; row != 3U && valid; ++row) {
    for (uint64_t head = 0U; head != 4U && valid; ++head) {
      for (uint64_t dimension = 0U; dimension != 256U; ++dimension) {
        const uint64_t source = row * 1024U + head * 256U + dimension;
        const uint64_t destination =
            (1U + row) * 4U * 256U + head * 256U + dimension;
        if (key_output[destination] != f16_values[source % 4U] ||
            value_output[destination] != f16_values[(source + 1U) % 4U]) {
          valid = false;
        }
      }
    }
  }
  descriptor = kv_append_descriptor(key, value, 1U, 0U);
  info = kv_append_info();
  valid =
      valid && ([&]() {
        sllm_kv_append_desc_t wrong_size = descriptor;
        wrong_size.struct_size -= 1U;
        sllm_completion_t *wrong_completion = nullptr;
        sllm_kv_append_info_t wrong_info = kv_append_info();
        const bool size_result = expect_status(
            sllm_kv_state_append(state, queue, &wrong_size, &wrong_completion,
                                 &wrong_info, &error.sink),
            SLLM_STATUS_INVALID_ARGUMENT, "KV append wrong size", error);
        wrong_size = descriptor;
        wrong_size.append_version += 1U;
        const bool version_result = expect_status(
            sllm_kv_state_append(state, queue, &wrong_size, &wrong_completion,
                                 &wrong_info, &error.sink),
            SLLM_STATUS_INVALID_KV_APPEND_DESCRIPTOR, "KV append wrong version",
            error);
        wrong_info = kv_append_info();
        wrong_info.struct_size -= 1U;
        const bool info_size_result = expect_status(
            sllm_kv_state_append(state, queue, &descriptor, &wrong_completion,
                                 &wrong_info, &error.sink),
            SLLM_STATUS_INVALID_ARGUMENT, "KV append info wrong size", error);
        return size_result && version_result && info_size_result &&
               wrong_completion == nullptr;
      })() &&
      expect_status(sllm_kv_state_append(state, queue, &descriptor, &completion,
                                         &info, &error.sink),
                    SLLM_STATUS_KV_LENGTH_MISMATCH, "KV stale length", error) &&
      completion == nullptr && kv_query(state, 4U, 2U);

  std::vector<uint16_t> boundary_key = make_words(255U, 0U);
  std::vector<uint16_t> boundary_value = make_words(255U, 1U);
  sllm_context_t *boundary_context = nullptr;
  sllm_queue_t *boundary_queue = nullptr;
  sllm_kv_state_t *boundary_state = nullptr;
  sllm_buffer_t *boundary_key_buffer = nullptr;
  sllm_buffer_t *boundary_value_buffer = nullptr;
  valid =
      valid && create_context(&boundary_context) &&
      create_queue(boundary_context, &boundary_queue) &&
      create_kv_state(boundary_context, capacity, &boundary_state) &&
      create_buffer_sized(boundary_context, input_bytes,
                          &boundary_key_buffer) &&
      create_buffer_sized(boundary_context, input_bytes,
                          &boundary_value_buffer) &&
      upload_kv_words(boundary_queue, boundary_key_buffer, boundary_key) &&
      upload_kv_words(boundary_queue, boundary_value_buffer, boundary_value);
  auto boundary_append = [&](const uint64_t tokens, const uint64_t position,
                             const sllm_status_t expected) {
    sllm_kv_append_desc_t boundary_descriptor = kv_append_descriptor(
        boundary_key_buffer, boundary_value_buffer, tokens, position);
    sllm_kv_append_info_t boundary_info = kv_append_info();
    sllm_completion_t *boundary_completion = nullptr;
    const bool result =
        expect_status(sllm_kv_state_append(
                          boundary_state, boundary_queue, &boundary_descriptor,
                          &boundary_completion, &boundary_info, &error.sink),
                      expected, "KV boundary append", error) &&
        (expected != SLLM_STATUS_OK || boundary_completion != nullptr);
    if (expected == SLLM_STATUS_OK) {
      return result && query_completion(boundary_completion, SLLM_STATUS_OK) &&
             release_completion(&boundary_completion);
    }
    return result && boundary_completion == nullptr;
  };
  valid = valid && boundary_append(255U, 0U, SLLM_STATUS_OK) &&
          boundary_append(1U, 255U, SLLM_STATUS_OK) &&
          boundary_append(1U, 256U, SLLM_STATUS_OK) &&
          boundary_append(1U, 257U, SLLM_STATUS_KV_CAPACITY_EXCEEDED) &&
          kv_query(boundary_state, 257U, 3U);
  valid = valid && release_buffer(&boundary_key_buffer) &&
          release_buffer(&boundary_value_buffer) &&
          expect_status(sllm_kv_state_release(&boundary_state, &error.sink),
                        SLLM_STATUS_OK, "KV boundary state release", error) &&
          release_queue(&boundary_queue) && release_context(&boundary_context);
  valid = valid && release_buffer(&key) && release_buffer(&value) &&
          expect_status(sllm_kv_state_release(&state, &error.sink),
                        SLLM_STATUS_OK, "KV layout state release", error) &&
          release_queue(&queue) && release_context(&context);
  return valid;
}

bool kv_vattention_page_boundary_and_idempotent_cancel_contract() {
  fake_hip::reset();
  const std::size_t baseline_allocations = fake_hip::live_allocations();
  constexpr uint64_t capacity = 1025U;
  constexpr uint64_t page_bytes = 2U * 1024U * 1024U;
  constexpr uint64_t row_bytes = 4U * 256U * sizeof(uint16_t);
  const uint64_t input_bytes = 1023U * row_bytes;
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_kv_state_t *state = nullptr;
  sllm_buffer_t *key = nullptr;
  sllm_buffer_t *value = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_kv_state(context, capacity, &state) ||
      !create_buffer_sized(context, input_bytes, &key) ||
      !create_buffer_sized(context, input_bytes, &value)) {
    return false;
  }
  std::vector<uint16_t> words(1023U * 4U * 256U, 0x3f80U);
  bool valid = upload_kv_words(queue, key, words) &&
               upload_kv_words(queue, value, words);
  Error error;
  auto append = [&](const uint64_t tokens, const uint64_t position) {
    auto descriptor = kv_append_descriptor(key, value, tokens, position);
    auto info = kv_append_info();
    sllm_completion_t *completion = nullptr;
    return expect_status(sllm_kv_state_append(state, queue, &descriptor,
                                              &completion, &info, &error.sink),
                         SLLM_STATUS_OK, "vAttention boundary append", error) &&
           completion != nullptr &&
           query_completion(completion, SLLM_STATUS_OK) &&
           release_completion(&completion);
  };
  auto growth_is = [&](const uint64_t length, const uint64_t mapped_tokens,
                       const uint64_t committed) {
    sllm_kv_view_info_t info{};
    info.struct_size = sizeof(info);
    info.abi_version = SLLM_HIP_ABI_VERSION;
    info.info_version = SLLM_HIP_KV_VIEW_INFO_VERSION;
    const bool queried =
        expect_status(sllm_kv_state_query(state, &info, &error.sink),
                      SLLM_STATUS_OK, "vAttention boundary query", error);
    const bool matches = queried && info.observed_length == length &&
                         info.physical_page_bytes == page_bytes &&
                         info.tokens_per_page == 1024U &&
                         info.mapped_token_capacity == mapped_tokens &&
                         info.committed_bytes_per_plane == committed;
    if (!matches) {
      std::cerr << "vAttention growth mismatch length=" << info.observed_length
                << " page=" << info.physical_page_bytes
                << " tokens_per_page=" << info.tokens_per_page
                << " mapped=" << info.mapped_token_capacity
                << " committed=" << info.committed_bytes_per_plane << '\n';
    }
    return matches;
  };
  valid = valid && append(1023U, 0U) && growth_is(1023U, 1024U, page_bytes) &&
          append(1U, 1023U) && growth_is(1024U, 1024U, page_bytes) &&
          append(1U, 1024U) && growth_is(1025U, 1025U, 2U * page_bytes);
  if (!valid) {
    std::cerr << "vAttention B-1/B/B+1 append phase failed\n";
  }
  valid =
      valid && release_buffer(&key) && release_buffer(&value) &&
      expect_status(sllm_kv_state_release(&state, &error.sink), SLLM_STATUS_OK,
                    "vAttention boundary state release", error) &&
      state == nullptr &&
      fake_hip::live_allocations() == baseline_allocations &&
      release_queue(&queue) && release_context(&context);
  if (!valid) {
    std::cerr << "vAttention boundary cleanup phase failed live="
              << fake_hip::live_allocations() << '\n';
  }

  sllm_context_t *cancel_context = nullptr;
  sllm_queue_t *cancel_queue = nullptr;
  sllm_kv_state_t *cancel_state = nullptr;
  sllm_buffer_t *cancel_key = nullptr;
  sllm_buffer_t *cancel_value = nullptr;
  valid = valid && create_context(&cancel_context) &&
          create_queue(cancel_context, &cancel_queue) &&
          create_kv_state(cancel_context, capacity, &cancel_state) &&
          create_buffer_sized(cancel_context, row_bytes, &cancel_key) &&
          create_buffer_sized(cancel_context, row_bytes, &cancel_value);
  std::vector<uint16_t> one_row(4U * 256U, 0x3f80U);
  valid = valid && upload_kv_words(cancel_queue, cancel_key, one_row) &&
          upload_kv_words(cancel_queue, cancel_value, one_row);
  auto descriptor = kv_append_descriptor(cancel_key, cancel_value, 1U, 0U);
  auto info = kv_append_info();
  sllm_completion_t *completion = nullptr;
  fake_hip::set_completion_pending(true);
  valid = valid &&
          expect_status(sllm_kv_state_append(cancel_state, cancel_queue,
                                             &descriptor, &completion, &info,
                                             &error.sink),
                        SLLM_STATUS_OK, "vAttention cancel append", error) &&
          expect_status(sllm_kv_state_append_cancel(cancel_state, completion,
                                                    &error.sink),
                        SLLM_STATUS_OK, "vAttention first cancel", error) &&
          expect_status(sllm_kv_state_append_cancel(cancel_state, completion,
                                                    &error.sink),
                        SLLM_STATUS_OK, "vAttention idempotent cancel", error);
  fake_hip::set_completion_pending(false);
  valid = valid && query_completion(completion, SLLM_STATUS_OK) &&
          release_completion(&completion) && kv_query(cancel_state, 0U, 0U) &&
          release_buffer(&cancel_key) && release_buffer(&cancel_value) &&
          expect_status(sllm_kv_state_release(&cancel_state, &error.sink),
                        SLLM_STATUS_OK, "vAttention canceled state release",
                        error) &&
          release_queue(&cancel_queue) && release_context(&cancel_context) &&
          fake_hip::live_allocations() == baseline_allocations;
  if (!valid) {
    std::cerr << "vAttention idempotent cancel phase failed live="
              << fake_hip::live_allocations() << '\n';
  }
  return valid;
}

bool kv_vmm_append_transaction_failure_injection_contract() {
  constexpr uint64_t capacity = 4096U;
  constexpr uint64_t row_elements = 4U * 256U;
  constexpr uint64_t prefix_tokens = 1U;
  constexpr uint64_t append_tokens = capacity - prefix_tokens;
  const uint64_t row_bytes = row_elements * sizeof(uint16_t);
  const uint64_t input_bytes = append_tokens * row_bytes;
  struct VmmFailureCase final {
    fake_hip::VmmOperation operation;
    uint64_t successful_calls;
  };
  const std::array<VmmFailureCase, 5> failure_cases = {
      VmmFailureCase{fake_hip::VmmOperation::Create, 0U},
      VmmFailureCase{fake_hip::VmmOperation::Create, 2U},
      VmmFailureCase{fake_hip::VmmOperation::Create, 5U},
      VmmFailureCase{fake_hip::VmmOperation::Map, 0U},
      VmmFailureCase{fake_hip::VmmOperation::SetAccess, 0U}};

  for (const VmmFailureCase &failure_case : failure_cases) {
    fake_hip::reset();
    const std::size_t baseline_allocations = fake_hip::live_allocations();
    sllm_context_t *context = nullptr;
    sllm_queue_t *queue = nullptr;
    sllm_kv_state_t *state = nullptr;
    sllm_buffer_t *key = nullptr;
    sllm_buffer_t *value = nullptr;
    Error error;
    bool valid = create_context(&context) && create_queue(context, &queue) &&
                 create_kv_state(context, capacity, &state) &&
                 create_buffer_sized(context, input_bytes, &key) &&
                 create_buffer_sized(context, input_bytes, &value);
    std::vector<uint16_t> words(
        static_cast<std::size_t>(append_tokens * row_elements), 0x3f80U);
    valid = valid && upload_kv_words(queue, key, words) &&
            upload_kv_words(queue, value, words);
    sllm_completion_t *completion = nullptr;
    sllm_kv_append_info_t info = kv_append_info();
    auto descriptor = kv_append_descriptor(key, value, prefix_tokens, 0U);
    valid =
        valid &&
        expect_status(sllm_kv_state_append(state, queue, &descriptor,
                                           &completion, &info, &error.sink),
                      SLLM_STATUS_OK, "VMM transaction prefix append", error) &&
        query_completion(completion, SLLM_STATUS_OK) &&
        release_completion(&completion);

    sllm_kv_view_info_t before{};
    before.struct_size = sizeof(before);
    before.abi_version = SLLM_HIP_ABI_VERSION;
    before.info_version = SLLM_HIP_KV_VIEW_INFO_VERSION;
    valid =
        valid && expect_status(sllm_kv_state_query(state, &before, &error.sink),
                               SLLM_STATUS_OK,
                               "VMM transaction pre-failure query", error);
    const std::size_t live_before_failure = fake_hip::live_allocations();
    fake_hip::set_vmm_failure_after(failure_case.operation,
                                    failure_case.successful_calls);
    descriptor = kv_append_descriptor(key, value, append_tokens, 1U);
    info = kv_append_info();
    completion = nullptr;
    valid = valid &&
            expect_status(sllm_kv_state_append(state, queue, &descriptor,
                                               &completion, &info, &error.sink),
                          SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
                          "VMM transaction injected grow failure", error) &&
            completion == nullptr && kv_query(state, 1U, 1U) &&
            fake_hip::kv_state_append_launch_calls() == 1U &&
            fake_hip::live_allocations() == live_before_failure;
    sllm_kv_view_info_t after{};
    after.struct_size = sizeof(after);
    after.abi_version = SLLM_HIP_ABI_VERSION;
    after.info_version = SLLM_HIP_KV_VIEW_INFO_VERSION;
    valid = valid &&
            expect_status(sllm_kv_state_query(state, &after, &error.sink),
                          SLLM_STATUS_OK, "VMM transaction post-failure query",
                          error) &&
            after.mapped_token_capacity == before.mapped_token_capacity &&
            after.committed_bytes_per_plane == before.committed_bytes_per_plane;

    fake_hip::clear_vmm_failures();
    descriptor = kv_append_descriptor(key, value, append_tokens, 1U);
    info = kv_append_info();
    completion = nullptr;
    valid = valid &&
            expect_status(sllm_kv_state_append(state, queue, &descriptor,
                                               &completion, &info, &error.sink),
                          SLLM_STATUS_OK, "VMM transaction retry", error) &&
            completion != nullptr &&
            query_completion(completion, SLLM_STATUS_OK) &&
            release_completion(&completion) && kv_query(state, capacity, 2U);

    if (key != nullptr) {
      valid = release_buffer(&key) && valid;
    }
    if (value != nullptr) {
      valid = release_buffer(&value) && valid;
    }
    if (state != nullptr) {
      valid = expect_status(sllm_kv_state_release(&state, &error.sink),
                            SLLM_STATUS_OK, "VMM transaction state release",
                            error) &&
              valid;
    }
    if (queue != nullptr) {
      valid = release_queue(&queue) && valid;
    }
    if (context != nullptr) {
      valid = release_context(&context) && valid;
    }
    if (!valid || fake_hip::live_allocations() != baseline_allocations) {
      std::cerr << "VMM append transaction rollback case failed after "
                << failure_case.successful_calls
                << " successful VMM calls; live="
                << fake_hip::live_allocations()
                << " baseline=" << baseline_allocations << '\n';
      return false;
    }
  }
  return true;
}

bool kv_vmm_cow_transaction_failure_injection_contract() {
  constexpr uint64_t source_capacity = 2048U;
  constexpr uint64_t prefix_tokens = 1025U;
  constexpr uint64_t row_elements = 4U * 256U;
  const uint64_t row_bytes = row_elements * sizeof(uint16_t);
  struct VmmFailureCase final {
    fake_hip::VmmOperation operation;
    uint64_t successful_calls;
  };
  const std::array<VmmFailureCase, 4> failure_cases = {
      VmmFailureCase{fake_hip::VmmOperation::Create, 0U},
      VmmFailureCase{fake_hip::VmmOperation::Create, 1U},
      VmmFailureCase{fake_hip::VmmOperation::Map, 2U},
      VmmFailureCase{fake_hip::VmmOperation::SetAccess, 2U}};
  for (const VmmFailureCase &failure_case : failure_cases) {
    fake_hip::reset();
    const std::size_t baseline_allocations = fake_hip::live_allocations();
    sllm_context_t *context = nullptr;
    sllm_queue_t *queue = nullptr;
    sllm_kv_state_t *source = nullptr;
    sllm_kv_state_t *child = nullptr;
    sllm_buffer_t *source_key = nullptr;
    sllm_buffer_t *source_value = nullptr;
    sllm_buffer_t *child_key = nullptr;
    sllm_buffer_t *child_value = nullptr;
    Error error;
    bool valid =
        create_context(&context) && create_queue(context, &queue) &&
        create_kv_state(context, source_capacity, &source) &&
        create_buffer_sized(context, prefix_tokens * row_bytes, &source_key) &&
        create_buffer_sized(context, prefix_tokens * row_bytes,
                            &source_value) &&
        create_buffer_sized(context, row_bytes, &child_key) &&
        create_buffer_sized(context, row_bytes, &child_value);
    std::vector<uint16_t> source_words(
        static_cast<std::size_t>(prefix_tokens * row_elements), 0x3f80U);
    std::vector<uint16_t> child_words(static_cast<std::size_t>(row_elements),
                                      0x4000U);
    valid = valid && upload_kv_words(queue, source_key, source_words) &&
            upload_kv_words(queue, source_value, source_words) &&
            upload_kv_words(queue, child_key, child_words) &&
            upload_kv_words(queue, child_value, child_words);
    sllm_completion_t *completion = nullptr;
    sllm_kv_append_info_t append_info = kv_append_info();
    auto append =
        kv_append_descriptor(source_key, source_value, prefix_tokens, 0U);
    valid =
        valid &&
        expect_status(sllm_kv_state_append(source, queue, &append, &completion,
                                           &append_info, &error.sink),
                      SLLM_STATUS_OK, "COW transaction source append", error) &&
        query_completion(completion, SLLM_STATUS_OK) &&
        release_completion(&completion);

    sllm_kv_state_create_info_v2_t destination{};
    destination.struct_size = sizeof(destination);
    destination.abi_version = SLLM_HIP_ABI_VERSION;
    destination.create_info_version = SLLM_HIP_KV_STATE_CREATE_INFO_V2_VERSION;
    destination.session_id = 0x1234U;
    destination.layer_id = 7U;
    destination.capacity_tokens = source_capacity;
    destination.head_count = 4U;
    destination.head_dim = 256U;
    destination.memory_kind = SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS;
    destination.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
    destination.dtype = SLLM_TENSOR_DTYPE_F16;
    destination.encoding = SLLM_HIP_KV_ENCODING_FP16_V1;
    sllm_state_fork_info_t fork_info{};
    fork_info.struct_size = sizeof(fork_info);
    fork_info.abi_version = SLLM_HIP_ABI_VERSION;
    fork_info.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
    valid =
        valid &&
        expect_status(sllm_kv_state_fork(source, &destination, &child,
                                         &fork_info, &error.sink),
                      SLLM_STATUS_OK, "COW transaction child fork", error) &&
        fork_info.shared_bytes != 0U;
    const uint64_t shared_before = fork_info.shared_bytes;
    const std::size_t live_before_failure = fake_hip::live_allocations();
    fake_hip::set_vmm_failure_after(failure_case.operation,
                                    failure_case.successful_calls);
    append = kv_append_descriptor(child_key, child_value, 1U, prefix_tokens);
    append_info = kv_append_info();
    completion = nullptr;
    valid =
        valid &&
        expect_status(sllm_kv_state_append(child, queue, &append, &completion,
                                           &append_info, &error.sink),
                      SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
                      "COW transaction injected failure", error) &&
        completion == nullptr && kv_query(child, prefix_tokens, 1U) &&
        fake_hip::live_allocations() == live_before_failure;
    sllm_state_fork_info_t after_failure{};
    after_failure.struct_size = sizeof(after_failure);
    after_failure.abi_version = SLLM_HIP_ABI_VERSION;
    after_failure.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
    valid = valid &&
            expect_status(
                sllm_kv_state_fork_query(child, &after_failure, &error.sink),
                SLLM_STATUS_OK, "COW transaction rollback query", error) &&
            after_failure.shared_bytes == shared_before;

    fake_hip::clear_vmm_failures();
    append = kv_append_descriptor(child_key, child_value, 1U, prefix_tokens);
    append_info = kv_append_info();
    completion = nullptr;
    valid =
        valid &&
        expect_status(sllm_kv_state_append(child, queue, &append, &completion,
                                           &append_info, &error.sink),
                      SLLM_STATUS_OK, "COW transaction retry", error) &&
        completion != nullptr && query_completion(completion, SLLM_STATUS_OK) &&
        release_completion(&completion) &&
        kv_query(child, prefix_tokens + 1U, 2U);

    if (child != nullptr) {
      valid = expect_status(sllm_kv_state_release(&child, &error.sink),
                            SLLM_STATUS_OK, "COW transaction child release",
                            error) &&
              valid;
    }
    if (source != nullptr) {
      valid = expect_status(sllm_kv_state_release(&source, &error.sink),
                            SLLM_STATUS_OK, "COW transaction source release",
                            error) &&
              valid;
    }
    if (source_key != nullptr) {
      valid = release_buffer(&source_key) && valid;
    }
    if (source_value != nullptr) {
      valid = release_buffer(&source_value) && valid;
    }
    if (child_key != nullptr) {
      valid = release_buffer(&child_key) && valid;
    }
    if (child_value != nullptr) {
      valid = release_buffer(&child_value) && valid;
    }
    if (queue != nullptr) {
      valid = release_queue(&queue) && valid;
    }
    if (context != nullptr) {
      valid = release_context(&context) && valid;
    }
    if (!valid || fake_hip::live_allocations() != baseline_allocations) {
      std::cerr << "COW transaction rollback case failed after "
                << failure_case.successful_calls
                << " successful VMM calls; live="
                << fake_hip::live_allocations()
                << " baseline=" << baseline_allocations << '\n';
      return false;
    }
  }
  return true;
}

bool kv_append_lifetime_alias_and_quarantine_contract() {
  fake_hip::reset();
  constexpr uint64_t input_bytes = 17U * 4U * 256U * sizeof(uint16_t);
  sllm_context_t *context = nullptr;
  sllm_context_t *other_context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_queue_t *other_queue = nullptr;
  sllm_kv_state_t *state = nullptr;
  sllm_buffer_t *key = nullptr;
  sllm_buffer_t *value = nullptr;
  if (!create_context(&context) || !create_context(&other_context) ||
      !create_queue(context, &queue) ||
      !create_queue(other_context, &other_queue) ||
      !create_kv_state(context, 17U, &state) ||
      !create_buffer_sized(context, input_bytes, &key) ||
      !create_buffer_sized(context, input_bytes, &value)) {
    if (key != nullptr) {
      (void)release_buffer(&key);
    }
    if (value != nullptr) {
      (void)release_buffer(&value);
    }
    if (state != nullptr) {
      Error error;
      (void)sllm_kv_state_release(&state, &error.sink);
    }
    if (queue != nullptr) {
      (void)release_queue(&queue);
    }
    if (other_queue != nullptr) {
      (void)release_queue(&other_queue);
    }
    if (context != nullptr) {
      (void)release_context(&context);
    }
    if (other_context != nullptr) {
      (void)release_context(&other_context);
    }
    return false;
  }
  std::vector<uint16_t> words(4U * 256U, 0x3f80U);
  bool valid = upload_kv_words(queue, key, words) &&
               upload_kv_words(queue, value, words);
  Error error;
  sllm_kv_append_info_t info = kv_append_info();
  sllm_completion_t *completion = nullptr;
  auto descriptor = kv_append_descriptor(key, value, 1U, 0U);
  fake_hip::set_completion_pending(true);
  valid = valid &&
          expect_status(sllm_kv_state_append(state, queue, &descriptor,
                                             &completion, &info, &error.sink),
                        SLLM_STATUS_OK, "KV pending append", error) &&
          completion != nullptr && ([&]() {
            sllm_completion_t *second_completion = nullptr;
            return expect_status(sllm_kv_state_append(state, queue, &descriptor,
                                                      &second_completion, &info,
                                                      &error.sink),
                                 SLLM_STATUS_PUBLIC_BUSY,
                                 "KV double append Busy", error) &&
                   second_completion == nullptr;
          })();
  sllm_completion_result_t pending_result{};
  pending_result.struct_size = sizeof(pending_result);
  pending_result.abi_version = SLLM_HIP_ABI_VERSION;
  valid =
      valid &&
      expect_status(
          sllm_completion_wait(completion, 0U, &pending_result, &error.sink),
          SLLM_STATUS_PUBLIC_TIMEOUT, "KV append timeout", error) &&
      expect_status(
          sllm_completion_query(completion, &pending_result, &error.sink),
          SLLM_STATUS_PUBLIC_PENDING, "KV pending query", error) &&
      expect_status(sllm_completion_release(&completion, &error.sink),
                    SLLM_STATUS_PUBLIC_BUSY, "KV pending release revokes",
                    error) &&
      kv_query(state, 0U, 0U) &&
      release_buffer(&key, SLLM_STATUS_PUBLIC_BUSY) &&
      release_buffer(&value, SLLM_STATUS_PUBLIC_BUSY) &&
      expect_status(sllm_kv_state_release(&state, &error.sink),
                    SLLM_STATUS_PUBLIC_BUSY, "KV pending state Busy", error);
  fake_hip::set_completion_pending(false);
  valid = valid && query_completion(completion, SLLM_STATUS_OK) &&
          release_completion(&completion) && kv_query(state, 0U, 0U);

  fake_hip::set_completion_pending(true);
  descriptor = kv_append_descriptor(key, value, 1U, 0U);
  info = kv_append_info();
  valid =
      valid &&
      expect_status(sllm_kv_state_append(state, queue, &descriptor, &completion,
                                         &info, &error.sink),
                    SLLM_STATUS_OK, "KV explicit cancel append", error) &&
      expect_status(sllm_kv_state_append_cancel(state, completion, &error.sink),
                    SLLM_STATUS_OK, "KV explicit append cancel", error) &&
      kv_query(state, 0U, 0U);
  fake_hip::set_completion_pending(false);
  valid = valid && query_completion(completion, SLLM_STATUS_OK) &&
          release_completion(&completion) && kv_query(state, 0U, 0U);

  descriptor = kv_append_descriptor(key, value, 1U, 0U);
  info = kv_append_info();
  valid = valid &&
          expect_status(sllm_kv_state_append(state, other_queue, &descriptor,
                                             &completion, &info, &error.sink),
                        SLLM_STATUS_PUBLIC_DEVICE_MISMATCH,
                        "KV wrong context queue", error) &&
          completion == nullptr;
  descriptor = kv_append_descriptor(key, value, 1U, 0U);
  descriptor.value_input.buffer = key;
  info = kv_append_info();
  const std::size_t calls_before_alias =
      fake_hip::kv_state_append_launch_calls();
  valid =
      valid &&
      expect_status(sllm_kv_state_append(state, queue, &descriptor, &completion,
                                         &info, &error.sink),
                    SLLM_STATUS_ALIAS_OVERLAP, "KV alias rejection", error) &&
      completion == nullptr &&
      fake_hip::kv_state_append_launch_calls() == calls_before_alias;
  fake_hip::set_kv_state_append_launch_status(hipErrorUnknown);
  descriptor = kv_append_descriptor(key, value, 1U, 0U);
  info = kv_append_info();
  valid = valid &&
          expect_status(sllm_kv_state_append(state, queue, &descriptor,
                                             &completion, &info, &error.sink),
                        SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
                        "KV launch failure rollback", error) &&
          completion == nullptr && kv_query(state, 0U, 0U);
  fake_hip::set_kv_state_append_launch_status(hipSuccess);
  descriptor = kv_append_descriptor(key, value, 1U, 0U);
  info = kv_append_info();
  valid =
      valid &&
      expect_status(sllm_kv_state_append(state, queue, &descriptor, &completion,
                                         &info, &error.sink),
                    SLLM_STATUS_OK, "KV reuse after launch failure", error) &&
      query_completion(completion, SLLM_STATUS_OK) &&
      release_completion(&completion) && kv_query(state, 1U, 1U);

  const std::size_t poison_before = sllm_test_poison_count();
  descriptor = kv_append_descriptor(key, value, 1U, 1U);
  info = kv_append_info();
  valid = valid &&
          expect_status(sllm_kv_state_append(state, queue, &descriptor,
                                             &completion, &info, &error.sink),
                        SLLM_STATUS_OK, "KV quarantine append", error) &&
          query_completion(completion, SLLM_STATUS_OK);
  sllm_public_runtime::FaultInjector::set(
      sllm_public_runtime::FaultPoint::EventDestroyError, 1U);
  const sllm_status_t quarantine_status =
      sllm_completion_release(&completion, &error.sink);
  sllm_public_runtime::FaultInjector::reset();
  valid = valid &&
          expect_status(quarantine_status, SLLM_STATUS_PUBLIC_HIP_RUNTIME_ERROR,
                        "KV event cleanup quarantine", error) &&
          completion == nullptr && sllm_test_poison_count() > poison_before;
  /* This context is intentionally poisoned by the injected ambiguous event
   * cleanup.  The remaining graph is owned by the process-lifetime quarantine.
   */
  const bool other_queue_released = release_queue(&other_queue);
  const bool other_context_released = release_context(&other_context);
  return valid && other_queue_released && other_context_released;
}

bool linear_attention_transaction_and_lifetime_contract() {
  fake_hip::reset();
  constexpr uint64_t token_count = 3U;
  constexpr uint64_t capacity = 7U;
  const std::array<uint64_t, 9> sizes = {token_count * 8192U * 2U,
                                         token_count * 4096U * 2U,
                                         token_count * 32U * 2U,
                                         token_count * 32U * 2U,
                                         8192U * 4U * 2U,
                                         32U * 4U,
                                         32U * 2U,
                                         128U * 4U,
                                         token_count * 4096U * 2U};
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  std::array<sllm_buffer_t *, 9> buffers{};
  sllm_linear_attention_state_t *state = nullptr;
  const auto release_buffers = [&buffers]() {
    bool success = true;
    for (auto &buffer : buffers) {
      if (buffer != nullptr) {
        success = release_buffer(&buffer) && success;
      }
    }
    return success;
  };
  if (!create_context(&context) || !create_queue(context, &queue)) {
    return false;
  }
  for (std::size_t index = 0U; index != buffers.size(); ++index) {
    if (!create_buffer_sized(context, sizes[index], &buffers[index])) {
      release_buffers();
      release_queue(&queue);
      release_context(&context);
      return false;
    }
  }
  sllm_linear_attention_state_create_info_t create{};
  create.struct_size = sizeof(create);
  create.abi_version = SLLM_HIP_ABI_VERSION;
  create.session_id = 0x5a17U;
  create.layer_id = 9U;
  create.capacity_tokens = capacity;
  Error error;
  if (!expect_status(sllm_linear_attention_state_create(context, &create,
                                                        &state, &error.sink),
                     SLLM_STATUS_OK, "linear attention state create", error) ||
      state == nullptr) {
    return false;
  }
  const auto query_state = [&](const uint64_t length, const uint64_t generation,
                               const uint32_t active_slot) {
    sllm_linear_attention_view_info_t view{};
    view.struct_size = sizeof(view);
    view.abi_version = SLLM_HIP_ABI_VERSION;
    view.info_version = SLLM_HIP_LINEAR_ATTENTION_VIEW_INFO_VERSION;
    Error query_error;
    return expect_status(sllm_linear_attention_state_query(state, &view,
                                                           &query_error.sink),
                         SLLM_STATUS_OK, "linear attention state query",
                         query_error) &&
           view.session_id == create.session_id &&
           view.layer_id == create.layer_id &&
           view.conv_state_dtype == SLLM_TENSOR_DTYPE_BF16 &&
           view.recurrent_state_dtype == SLLM_TENSOR_DTYPE_F32 &&
           view.encoding == SLLM_TENSOR_ENCODING_UNQUANTIZED &&
           view.active_slot == active_slot &&
           view.capacity_tokens == capacity && view.observed_length == length &&
           view.generation == generation && view.context_identity != 0U &&
           view.state_identity != 0U && view.conv_state_shape[0] == 3U &&
           view.conv_state_shape[1] == 8192U &&
           view.recurrent_state_shape[0] == 32U &&
           view.recurrent_state_shape[1] == 128U &&
           view.recurrent_state_shape[2] == 128U;
  };
  const auto descriptor_for = [&](const uint64_t count, const uint64_t start) {
    const uint64_t qkv_shape[] = {count, 8192U};
    const uint64_t output_shape[] = {count, 4096U};
    const uint64_t scalar_shape[] = {count, 32U};
    constexpr uint64_t conv_shape[] = {8192U, 1U, 4U};
    constexpr uint64_t head_shape[] = {32U};
    constexpr uint64_t norm_shape[] = {128U};
    sllm_linear_attention_desc_t descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = SLLM_HIP_ABI_VERSION;
    descriptor.op_version = SLLM_HIP_LINEAR_ATTENTION_VERSION;
    descriptor.start_position = start;
    descriptor.expected_length = start + count;
    descriptor.state = state;
    descriptor.qkv =
        attention_binding(buffers[0], SLLM_TENSOR_DTYPE_BF16, 2U, qkv_shape);
    descriptor.z =
        attention_binding(buffers[1], SLLM_TENSOR_DTYPE_BF16, 2U, output_shape);
    descriptor.b_input =
        attention_binding(buffers[2], SLLM_TENSOR_DTYPE_BF16, 2U, scalar_shape);
    descriptor.a_input =
        attention_binding(buffers[3], SLLM_TENSOR_DTYPE_BF16, 2U, scalar_shape);
    descriptor.conv_weight =
        attention_binding(buffers[4], SLLM_TENSOR_DTYPE_BF16, 3U, conv_shape);
    descriptor.a_log =
        attention_binding(buffers[5], SLLM_TENSOR_DTYPE_F32, 1U, head_shape);
    descriptor.dt_bias =
        attention_binding(buffers[6], SLLM_TENSOR_DTYPE_BF16, 1U, head_shape);
    descriptor.norm_weight =
        attention_binding(buffers[7], SLLM_TENSOR_DTYPE_F32, 1U, norm_shape);
    descriptor.output =
        attention_binding(buffers[8], SLLM_TENSOR_DTYPE_BF16, 2U, output_shape);
    return descriptor;
  };
  const auto dispatch_info = []() {
    sllm_linear_attention_dispatch_info_t info{};
    info.struct_size = sizeof(info);
    info.abi_version = SLLM_HIP_ABI_VERSION;
    info.info_version = SLLM_HIP_LINEAR_ATTENTION_DISPATCH_INFO_VERSION;
    return info;
  };
  auto descriptor = descriptor_for(token_count, 0U);
  auto stale = descriptor;
  stale.start_position = 1U;
  stale.expected_length = 4U;
  auto info = dispatch_info();
  sllm_completion_t *completion = nullptr;
  if (!query_state(0U, 0U, 0U) ||
      !expect_status(sllm_linear_attention_execute(context, queue, &stale,
                                                   &completion, &info,
                                                   &error.sink),
                     SLLM_STATUS_LINEAR_ATTENTION_LENGTH_MISMATCH,
                     "linear attention stale length", error) ||
      completion != nullptr) {
    return false;
  }
  info = dispatch_info();
  if (!expect_status(sllm_linear_attention_execute(context, queue, &descriptor,
                                                   &completion, &info,
                                                   &error.sink),
                     SLLM_STATUS_OK, "linear attention execute", error) ||
      completion == nullptr || info.dispatch_id == 0U ||
      info.dispatch_count != 2U ||
      info.conv_kernel_id !=
          SLLM_HIP_LINEAR_ATTENTION_KERNEL_ID_CAUSAL_CONV_SILU_V1 ||
      info.recurrent_kernel_id !=
          SLLM_HIP_LINEAR_ATTENTION_KERNEL_ID_RECURRENT_GATED_NORM_V1 ||
      info.workgroup_size_x != SLLM_HIP_LINEAR_ATTENTION_WORKGROUP_SIZE ||
      info.recurrent_grid_size_x != 32U || info.token_count != token_count ||
      info.start_position != 0U || info.expected_length != token_count ||
      info.fallback_allowed != 0U || info.fallback_used != 0U ||
      !query_state(0U, 0U, 0U) ||
      !release_buffer(&buffers[0], SLLM_STATUS_PUBLIC_BUSY) ||
      !release_queue(&queue, SLLM_STATUS_PUBLIC_BUSY) ||
      !query_completion(completion, SLLM_STATUS_OK) ||
      !query_state(token_count, 1U, 1U) || !release_completion(&completion)) {
    return false;
  }

  descriptor = descriptor_for(1U, token_count);
  info = dispatch_info();
  if (!expect_status(
          sllm_linear_attention_execute(context, queue, &descriptor,
                                        &completion, &info, &error.sink),
          SLLM_STATUS_OK, "linear attention cancel execute", error) ||
      !expect_status(
          sllm_linear_attention_cancel(state, completion, &error.sink),
          SLLM_STATUS_OK, "linear attention cancel", error) ||
      !query_completion(completion, SLLM_STATUS_OK) ||
      !release_completion(&completion) || !query_state(token_count, 1U, 1U)) {
    return false;
  }

  const bool state_released =
      expect_status(sllm_linear_attention_state_release(&state, &error.sink),
                    SLLM_STATUS_OK, "linear attention state release", error);
  return state_released && release_buffers() && release_queue(&queue) &&
         release_context(&context);
}

bool state_fork_vmm_and_linear_image_contract() {
  fake_hip::reset();
  constexpr uint64_t source_capacity = 2048U;
  constexpr uint64_t destination_capacity = 4096U;
  constexpr uint64_t prefix_length = 1025U;
  constexpr uint64_t kv_elements_per_token = 4U * 256U;
  constexpr uint64_t kv_bytes_per_token =
      kv_elements_per_token * sizeof(uint16_t);
  const std::size_t prefix_bytes =
      static_cast<std::size_t>(prefix_length * kv_bytes_per_token);
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_kv_state_t *source = nullptr;
  sllm_kv_state_t *child = nullptr;
  sllm_buffer_t *source_key = nullptr;
  sllm_buffer_t *source_value = nullptr;
  sllm_buffer_t *child_key = nullptr;
  sllm_buffer_t *child_value = nullptr;
  Error error;
  const auto cleanup_kv = [&]() {
    bool result = true;
    if (child != nullptr) {
      result = expect_status(sllm_kv_state_release(&child, &error.sink),
                             SLLM_STATUS_OK, "fork child release", error) &&
               result;
    }
    if (source != nullptr) {
      result = expect_status(sllm_kv_state_release(&source, &error.sink),
                             SLLM_STATUS_OK, "fork source release", error) &&
               result;
    }
    result = release_buffer(&source_key) && result;
    result = release_buffer(&source_value) && result;
    result = release_buffer(&child_key) && result;
    result = release_buffer(&child_value) && result;
    result = release_queue(&queue) && result;
    result = release_context(&context) && result;
    return result;
  };
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_kv_state(context, source_capacity, &source) ||
      !create_buffer_sized(context, prefix_bytes, &source_key) ||
      !create_buffer_sized(context, prefix_bytes, &source_value)) {
    (void)cleanup_kv();
    return false;
  }
  std::vector<uint16_t> source_words(
      static_cast<std::size_t>(prefix_length * kv_elements_per_token),
      UINT16_C(0x3f80));
  if (!upload_kv_words(queue, source_key, source_words) ||
      !upload_kv_words(queue, source_value, source_words)) {
    (void)cleanup_kv();
    return false;
  }
  sllm_kv_append_desc_t append =
      kv_append_descriptor(source_key, source_value, prefix_length, 0U);
  sllm_kv_append_info_t append_info = kv_append_info();
  sllm_completion_t *completion = nullptr;
  if (!expect_status(sllm_kv_state_append(source, queue, &append, &completion,
                                          &append_info, &error.sink),
                     SLLM_STATUS_OK, "fork source append", error) ||
      completion == nullptr || !query_completion(completion, SLLM_STATUS_OK) ||
      !release_completion(&completion)) {
    (void)cleanup_kv();
    return false;
  }

  sllm_kv_state_create_info_v2_t destination_info{};
  destination_info.struct_size = sizeof(destination_info);
  destination_info.abi_version = SLLM_HIP_ABI_VERSION;
  destination_info.create_info_version =
      SLLM_HIP_KV_STATE_CREATE_INFO_V2_VERSION;
  destination_info.session_id = 0x1234U;
  destination_info.layer_id = 7U;
  destination_info.capacity_tokens = destination_capacity;
  destination_info.head_count = 4U;
  destination_info.head_dim = 256U;
  destination_info.memory_kind = SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS;
  destination_info.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
  destination_info.dtype = SLLM_TENSOR_DTYPE_F16;
  destination_info.encoding = SLLM_HIP_KV_ENCODING_FP16_V1;
  sllm_state_fork_info_t fork_info{};
  fork_info.struct_size = sizeof(fork_info);
  fork_info.abi_version = SLLM_HIP_ABI_VERSION;
  fork_info.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
  bool valid =
      expect_status(sllm_kv_state_fork(source, &destination_info, &child,
                                       &fork_info, &error.sink),
                    SLLM_STATUS_OK, "fork VMM child", error) &&
      child != nullptr &&
      fork_info.mode == SLLM_HIP_STATE_FORK_MODE_SHARED_READ_ONLY_PAGES &&
      fork_info.published_length == prefix_length &&
      fork_info.shared_bytes >= fork_info.page_bytes * 2U &&
      fork_info.child_owned_bytes == 0U && fork_info.copied_bytes == 0U &&
      fork_info.page_bytes == UINT64_C(2) * 1024U * 1024U;
  if (!valid || !create_buffer_sized(context, kv_bytes_per_token, &child_key) ||
      !create_buffer_sized(context, kv_bytes_per_token, &child_value)) {
    (void)cleanup_kv();
    return false;
  }
  std::vector<uint16_t> child_words(
      static_cast<std::size_t>(kv_elements_per_token), UINT16_C(0x4000));
  valid = valid && upload_kv_words(queue, child_key, child_words) &&
          upload_kv_words(queue, child_value, child_words);
  sllm_state_fork_info_t pre_cow_info{};
  pre_cow_info.struct_size = sizeof(pre_cow_info);
  pre_cow_info.abi_version = SLLM_HIP_ABI_VERSION;
  pre_cow_info.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
  valid =
      valid &&
      expect_status(sllm_kv_state_fork_query(child, &pre_cow_info, &error.sink),
                    SLLM_STATUS_OK, "fork pre-COW audit query", error);
  append = kv_append_descriptor(child_key, child_value, 1U, prefix_length);
  append_info = kv_append_info();
  completion = nullptr;
  valid = valid &&
          expect_status(sllm_kv_state_append(child, queue, &append, &completion,
                                             &append_info, &error.sink),
                        SLLM_STATUS_OK, "fork child append", error) &&
          completion != nullptr &&
          query_completion(completion, SLLM_STATUS_OK) &&
          release_completion(&completion);

  const uint64_t tail_offset = prefix_length * kv_bytes_per_token;
  std::vector<uint16_t> source_tail(
      static_cast<std::size_t>(kv_elements_per_token));
  std::vector<uint16_t> child_tail(
      static_cast<std::size_t>(kv_elements_per_token));
  const auto export_chunk = [&](const sllm_kv_state_t *const state,
                                const uint32_t plane, const uint64_t offset,
                                void *const host, const uint64_t bytes) {
    sllm_state_chunk_t chunk{};
    chunk.struct_size = sizeof(chunk);
    chunk.abi_version = SLLM_HIP_ABI_VERSION;
    chunk.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
    chunk.plane = plane;
    chunk.byte_offset = offset;
    chunk.byte_length = bytes;
    chunk.host_pointer = host;
    chunk.host_capacity = bytes;
    return expect_status(sllm_kv_state_export(state, &chunk, &error.sink),
                         SLLM_STATUS_OK, "fork raw KV export", error);
  };
  valid =
      valid &&
      export_chunk(source, SLLM_HIP_KV_STATE_PLANE_KEY, tail_offset,
                   source_tail.data(), kv_bytes_per_token) &&
      export_chunk(child, SLLM_HIP_KV_STATE_PLANE_KEY, tail_offset,
                   child_tail.data(), kv_bytes_per_token) &&
      source_tail ==
          std::vector<uint16_t>(source_tail.size(), UINT16_C(0x0000)) &&
      child_tail == std::vector<uint16_t>(child_tail.size(), UINT16_C(0x4000));

  sllm_state_fork_info_t dynamic_info{};
  dynamic_info.struct_size = sizeof(dynamic_info);
  dynamic_info.abi_version = SLLM_HIP_ABI_VERSION;
  dynamic_info.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
  valid =
      valid &&
      expect_status(sllm_kv_state_fork_query(child, &dynamic_info, &error.sink),
                    SLLM_STATUS_OK, "fork dynamic audit query", error) &&
      dynamic_info.copied_bytes >= dynamic_info.page_bytes &&
      dynamic_info.shared_bytes < fork_info.shared_bytes;

  sllm_state_image_info_t image_info{};
  image_info.struct_size = sizeof(image_info);
  image_info.abi_version = SLLM_HIP_ABI_VERSION;
  image_info.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
  valid =
      valid &&
      expect_status(sllm_kv_state_image_query(source, &image_info, &error.sink),
                    SLLM_STATUS_OK, "fork KV image query", error) &&
      image_info.capacity_tokens == source_capacity &&
      image_info.published_length == prefix_length &&
      image_info.generation == 1U && image_info.plane_count == 2U;
  std::vector<uint8_t> source_key_image(prefix_bytes);
  std::vector<uint8_t> source_value_image(prefix_bytes);
  const auto export_image = [&](const uint32_t plane, void *const host) {
    sllm_state_chunk_t chunk{};
    chunk.struct_size = sizeof(chunk);
    chunk.abi_version = SLLM_HIP_ABI_VERSION;
    chunk.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
    chunk.plane = plane;
    chunk.byte_length = prefix_bytes;
    chunk.host_pointer = host;
    chunk.host_capacity = prefix_bytes;
    return expect_status(sllm_kv_state_export(source, &chunk, &error.sink),
                         SLLM_STATUS_OK, "fork KV image export", error);
  };
  const auto import_image = [&](const uint32_t plane, void *const host) {
    sllm_state_chunk_t chunk{};
    chunk.struct_size = sizeof(chunk);
    chunk.abi_version = SLLM_HIP_ABI_VERSION;
    chunk.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
    chunk.plane = plane;
    chunk.byte_length = prefix_bytes;
    chunk.host_pointer = host;
    chunk.host_capacity = prefix_bytes;
    return expect_status(sllm_kv_state_import(child, &chunk, &error.sink),
                         SLLM_STATUS_OK, "fork KV image import", error);
  };
  valid =
      valid &&
      export_image(SLLM_HIP_KV_STATE_PLANE_KEY, source_key_image.data()) &&
      export_image(SLLM_HIP_KV_STATE_PLANE_VALUE, source_value_image.data()) &&
      import_image(SLLM_HIP_KV_STATE_PLANE_KEY, source_key_image.data()) &&
      import_image(SLLM_HIP_KV_STATE_PLANE_VALUE, source_value_image.data()) &&
      expect_status(
          sllm_kv_state_import_finalize(child, &image_info, &error.sink),
          SLLM_STATUS_OK, "fork KV image finalize", error);
  sllm_kv_view_info_t child_view{};
  child_view.struct_size = sizeof(child_view);
  child_view.abi_version = SLLM_HIP_ABI_VERSION;
  child_view.info_version = SLLM_HIP_KV_VIEW_INFO_VERSION;
  valid = valid &&
          expect_status(sllm_kv_state_query(child, &child_view, &error.sink),
                        SLLM_STATUS_OK, "fork KV image query after finalize",
                        error) &&
          child_view.capacity_tokens == destination_capacity &&
          child_view.observed_length == prefix_length &&
          child_view.generation == 1U;

  if (!cleanup_kv()) {
    return false;
  }

  struct LowBitRecipe final {
    const char *arch;
    uint32_t dtype;
    uint32_t encoding;
    uint32_t block_size;
    uint32_t scale_dtype;
    uint32_t plane_count;
    uint64_t value_bytes;
    uint64_t scale_bytes;
    uint64_t outer_scale_bytes;
    float static_key_scale;
    float static_value_scale;
  };
  const std::array<LowBitRecipe, 6> lowbit_recipes = {{
      {"gfx1201", SLLM_TENSOR_DTYPE_F8_E4M3_FN, SLLM_HIP_KV_ENCODING_FP8_V1, 0U,
       SLLM_TENSOR_DTYPE_F32, 4U, 4U * 256U, 4U * sizeof(float), 0U, 0.0F,
       0.0F},
      {"gfx1201", SLLM_TENSOR_DTYPE_F8_E4M3_FN,
       SLLM_HIP_KV_ENCODING_FP8_STATIC_V1, 0U, SLLM_TENSOR_DTYPE_F32, 2U,
       4U * 256U, 0U, 0U, 0.125F, 0.25F},
      {"gfx1201", SLLM_TENSOR_DTYPE_U8, SLLM_HIP_KV_ENCODING_NVFP4_V1, 16U,
       SLLM_TENSOR_DTYPE_F8_E4M3_FN, 6U, 4U * (256U / 2U), 4U * (256U / 16U),
       4U * sizeof(float), 0.0F, 0.0F},
      {"gfx1201", SLLM_TENSOR_DTYPE_F8_E4M3_FN,
       SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, 32U, SLLM_TENSOR_DTYPE_U8, 4U,
       4U * 256U, 4U * (256U / 32U), 0U, 0.0F, 0.0F},
      {"gfx942", SLLM_TENSOR_DTYPE_F8_E4M3_FN, SLLM_HIP_KV_ENCODING_MXFP8_E4_V1,
       32U, SLLM_TENSOR_DTYPE_U8, 4U, 4U * 256U, 4U * (256U / 32U), 0U, 0.0F,
       0.0F},
      {"gfx1030", SLLM_TENSOR_DTYPE_F8_E4M3_FN,
       SLLM_HIP_KV_ENCODING_MXFP8_E4_V1, 32U, SLLM_TENSOR_DTYPE_U8, 4U,
       4U * 256U, 4U * (256U / 32U), 0U, 0.0F, 0.0F},
  }};
  const auto run_lowbit_image_case = [&](const LowBitRecipe &recipe,
                                         const uint32_t case_index) {
    fake_hip::reset();
    fake_hip::set_gcn_arch_name(recipe.arch);
    constexpr uint64_t lowbit_source_capacity = 17U;
    constexpr uint64_t lowbit_destination_capacity = 33U;
    sllm_context_t *lowbit_context = nullptr;
    sllm_kv_state_t *lowbit_source = nullptr;
    sllm_kv_state_t *lowbit_child = nullptr;
    Error lowbit_error;
    sllm_kv_state_create_info_v2_t create{};
    create.struct_size = sizeof(create);
    create.abi_version = SLLM_HIP_ABI_VERSION;
    create.create_info_version =
        recipe.encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1
            ? SLLM_HIP_KV_STATE_CREATE_INFO_STATIC_FP8_VERSION
            : SLLM_HIP_KV_STATE_CREATE_INFO_V2_VERSION;
    create.session_id = 0x2000U + case_index;
    create.layer_id = 31U + case_index;
    create.capacity_tokens = lowbit_source_capacity;
    create.head_count = 4U;
    create.head_dim = 256U;
    create.memory_kind = SLLM_HIP_KV_MEMORY_KIND_CONTIGUOUS_RESIDENT;
    create.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
    create.dtype = recipe.dtype;
    create.encoding = recipe.encoding;
    create.block_size = recipe.block_size;
    create.scale_dtype = recipe.scale_dtype;
    if (recipe.encoding == SLLM_HIP_KV_ENCODING_FP8_STATIC_V1) {
      std::memcpy(&create.reserved[0], &recipe.static_key_scale,
                  sizeof(recipe.static_key_scale));
      std::memcpy(&create.reserved[1], &recipe.static_value_scale,
                  sizeof(recipe.static_value_scale));
    }
    bool case_valid =
        create_context_for_arch(recipe.arch, &lowbit_context) &&
        expect_status(
            sllm_kv_state_create_v2(lowbit_context, &create, &lowbit_source,
                                    &lowbit_error.sink),
            SLLM_STATUS_OK, "low-bit image source create", lowbit_error) &&
        lowbit_source != nullptr;
    auto release_lowbit = [&]() {
      bool released = true;
      if (lowbit_child != nullptr) {
        released =
            expect_status(
                sllm_kv_state_release(&lowbit_child, &lowbit_error.sink),
                SLLM_STATUS_OK, "low-bit image child release", lowbit_error) &&
            released;
      }
      if (lowbit_source != nullptr) {
        released =
            expect_status(
                sllm_kv_state_release(&lowbit_source, &lowbit_error.sink),
                SLLM_STATUS_OK, "low-bit image source release", lowbit_error) &&
            released;
      }
      released = release_context(&lowbit_context) && released;
      return released;
    };
    if (!case_valid) {
      (void)release_lowbit();
      return false;
    }
    sllm_state_image_info_t image{};
    image.struct_size = sizeof(image);
    image.abi_version = SLLM_HIP_ABI_VERSION;
    image.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
    case_valid =
        expect_status(sllm_kv_state_image_query(lowbit_source, &image,
                                                &lowbit_error.sink),
                      SLLM_STATUS_OK, "low-bit image query", lowbit_error) &&
        image.plane_count == recipe.plane_count &&
        image.capacity_tokens == lowbit_source_capacity;
    const std::array<uint64_t, 6> plane_bytes = {
        recipe.value_bytes, recipe.value_bytes,       recipe.scale_bytes,
        recipe.scale_bytes, recipe.outer_scale_bytes, recipe.outer_scale_bytes};
    std::array<std::vector<uint8_t>, 6> plane_images;
    for (std::size_t index = 0U; index != recipe.plane_count; ++index) {
      plane_images[index].resize(static_cast<std::size_t>(plane_bytes[index]));
      for (std::size_t byte = 0U; byte != plane_images[index].size(); ++byte) {
        plane_images[index][byte] = static_cast<uint8_t>(0x20U + index + byte);
      }
      sllm_state_chunk_t chunk{};
      chunk.struct_size = sizeof(chunk);
      chunk.abi_version = SLLM_HIP_ABI_VERSION;
      chunk.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
      chunk.plane = static_cast<uint32_t>(index + 1U);
      chunk.byte_length = plane_bytes[index];
      chunk.host_pointer = plane_images[index].data();
      chunk.host_capacity = plane_bytes[index];
      case_valid =
          case_valid &&
          expect_status(
              sllm_kv_state_import(lowbit_source, &chunk, &lowbit_error.sink),
              SLLM_STATUS_OK, "low-bit raw plane import", lowbit_error);
    }
    image.published_length = 1U;
    image.generation = 7U;
    case_valid =
        case_valid &&
        expect_status(sllm_kv_state_import_finalize(lowbit_source, &image,
                                                    &lowbit_error.sink),
                      SLLM_STATUS_OK, "low-bit image finalize", lowbit_error);
    sllm_kv_state_create_info_v2_t destination = create;
    destination.capacity_tokens = lowbit_destination_capacity;
    sllm_state_fork_info_t fork{};
    fork.struct_size = sizeof(fork);
    fork.abi_version = SLLM_HIP_ABI_VERSION;
    fork.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
    case_valid =
        case_valid &&
        expect_status(sllm_kv_state_fork(lowbit_source, &destination,
                                         &lowbit_child, &fork,
                                         &lowbit_error.sink),
                      SLLM_STATUS_OK, "low-bit image fork", lowbit_error) &&
        lowbit_child != nullptr &&
        fork.mode == SLLM_HIP_STATE_FORK_MODE_DEVICE_COPY &&
        fork.published_length == 1U;
    for (std::size_t index = 0U; index != recipe.plane_count; ++index) {
      sllm_state_chunk_t chunk{};
      chunk.struct_size = sizeof(chunk);
      chunk.abi_version = SLLM_HIP_ABI_VERSION;
      chunk.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
      chunk.plane = static_cast<uint32_t>(index + 1U);
      chunk.byte_length = plane_bytes[index];
      chunk.host_pointer = plane_images[index].data();
      chunk.host_capacity = plane_bytes[index];
      case_valid =
          case_valid &&
          expect_status(
              sllm_kv_state_export(lowbit_child, &chunk, &lowbit_error.sink),
              SLLM_STATUS_OK, "low-bit raw plane export", lowbit_error);
    }
    return release_lowbit() && case_valid;
  };
  for (const LowBitRecipe &recipe : lowbit_recipes) {
    valid =
        run_lowbit_image_case(
            recipe, static_cast<uint32_t>(&recipe - lowbit_recipes.data())) &&
        valid;
  }

  /* A linear state fork is a device copy, but must preserve the published
   * active slot and the image metadata/finalize transaction. */
  fake_hip::reset();
  sllm_context_t *linear_context = nullptr;
  sllm_queue_t *linear_queue = nullptr;
  sllm_linear_attention_state_t *linear_source = nullptr;
  sllm_linear_attention_state_t *linear_child = nullptr;
  std::array<sllm_buffer_t *, 9> buffers{};
  const std::array<uint64_t, 9> sizes = {
      8192U * 2U, 4096U * 2U, 32U * 2U,  32U * 2U,  8192U * 4U * 2U,
      32U * 4U,   32U * 2U,   128U * 4U, 4096U * 2U};
  const auto cleanup_linear = [&]() {
    bool result = true;
    if (linear_child != nullptr) {
      result =
          expect_status(
              sllm_linear_attention_state_release(&linear_child, &error.sink),
              SLLM_STATUS_OK, "linear fork child release", error) &&
          result;
    }
    if (linear_source != nullptr) {
      result =
          expect_status(
              sllm_linear_attention_state_release(&linear_source, &error.sink),
              SLLM_STATUS_OK, "linear fork source release", error) &&
          result;
    }
    for (auto &buffer : buffers) {
      result = release_buffer(&buffer) && result;
    }
    result = release_queue(&linear_queue) && result;
    result = release_context(&linear_context) && result;
    return result;
  };
  if (!create_context(&linear_context) ||
      !create_queue(linear_context, &linear_queue)) {
    (void)cleanup_linear();
    return false;
  }
  for (std::size_t index = 0U; index != buffers.size(); ++index) {
    if (!create_buffer_sized(linear_context, sizes[index], &buffers[index])) {
      (void)cleanup_linear();
      return false;
    }
  }
  sllm_linear_attention_state_create_info_t linear_create{};
  linear_create.struct_size = sizeof(linear_create);
  linear_create.abi_version = SLLM_HIP_ABI_VERSION;
  linear_create.session_id = 0x5a17U;
  linear_create.layer_id = 9U;
  linear_create.capacity_tokens = 7U;
  linear_create.qk_heads = 16U;
  linear_create.value_heads = 32U;
  linear_create.head_dim = 128U;
  linear_create.conv_kernel_size = 4U;
  valid = valid &&
          expect_status(
              sllm_linear_attention_state_create(linear_context, &linear_create,
                                                 &linear_source, &error.sink),
              SLLM_STATUS_OK, "linear fork source create", error);
  const uint64_t qkv_shape[] = {1U, 8192U};
  const uint64_t output_shape[] = {1U, 4096U};
  const uint64_t scalar_shape[] = {1U, 32U};
  constexpr uint64_t conv_shape[] = {8192U, 1U, 4U};
  constexpr uint64_t head_shape[] = {32U};
  constexpr uint64_t norm_shape[] = {128U};
  sllm_linear_attention_desc_t linear_desc{};
  linear_desc.struct_size = sizeof(linear_desc);
  linear_desc.abi_version = SLLM_HIP_ABI_VERSION;
  linear_desc.op_version = SLLM_HIP_LINEAR_ATTENTION_VERSION;
  linear_desc.expected_length = 1U;
  linear_desc.state = linear_source;
  linear_desc.qkv =
      attention_binding(buffers[0], SLLM_TENSOR_DTYPE_BF16, 2U, qkv_shape);
  linear_desc.z =
      attention_binding(buffers[1], SLLM_TENSOR_DTYPE_BF16, 2U, output_shape);
  linear_desc.b_input =
      attention_binding(buffers[2], SLLM_TENSOR_DTYPE_BF16, 2U, scalar_shape);
  linear_desc.a_input =
      attention_binding(buffers[3], SLLM_TENSOR_DTYPE_BF16, 2U, scalar_shape);
  linear_desc.conv_weight =
      attention_binding(buffers[4], SLLM_TENSOR_DTYPE_BF16, 3U, conv_shape);
  linear_desc.a_log =
      attention_binding(buffers[5], SLLM_TENSOR_DTYPE_F32, 1U, head_shape);
  linear_desc.dt_bias =
      attention_binding(buffers[6], SLLM_TENSOR_DTYPE_BF16, 1U, head_shape);
  linear_desc.norm_weight =
      attention_binding(buffers[7], SLLM_TENSOR_DTYPE_F32, 1U, norm_shape);
  linear_desc.output =
      attention_binding(buffers[8], SLLM_TENSOR_DTYPE_BF16, 2U, output_shape);
  sllm_linear_attention_dispatch_info_t linear_dispatch{};
  linear_dispatch.struct_size = sizeof(linear_dispatch);
  linear_dispatch.abi_version = SLLM_HIP_ABI_VERSION;
  linear_dispatch.info_version =
      SLLM_HIP_LINEAR_ATTENTION_DISPATCH_INFO_VERSION;
  completion = nullptr;
  valid = valid &&
          expect_status(sllm_linear_attention_execute(
                            linear_context, linear_queue, &linear_desc,
                            &completion, &linear_dispatch, &error.sink),
                        SLLM_STATUS_OK, "linear fork source execute", error) &&
          completion != nullptr &&
          query_completion(completion, SLLM_STATUS_OK) &&
          release_completion(&completion);
  sllm_linear_attention_view_info_t linear_view{};
  linear_view.struct_size = sizeof(linear_view);
  linear_view.abi_version = SLLM_HIP_ABI_VERSION;
  linear_view.info_version = SLLM_HIP_LINEAR_ATTENTION_VIEW_INFO_VERSION;
  valid = valid &&
          expect_status(sllm_linear_attention_state_query(
                            linear_source, &linear_view, &error.sink),
                        SLLM_STATUS_OK, "linear fork source query", error) &&
          linear_view.active_slot == 1U && linear_view.observed_length == 1U &&
          linear_view.generation == 1U;
  sllm_linear_attention_state_create_info_t linear_destination = linear_create;
  linear_destination.capacity_tokens = 11U;
  sllm_state_fork_info_t linear_fork_info{};
  linear_fork_info.struct_size = sizeof(linear_fork_info);
  linear_fork_info.abi_version = SLLM_HIP_ABI_VERSION;
  linear_fork_info.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
  valid = valid &&
          expect_status(sllm_linear_attention_state_fork(
                            linear_source, &linear_destination, &linear_child,
                            &linear_fork_info, &error.sink),
                        SLLM_STATUS_OK, "linear fork child", error) &&
          linear_child != nullptr &&
          linear_fork_info.mode == SLLM_HIP_STATE_FORK_MODE_DEVICE_COPY &&
          linear_fork_info.published_length == 1U &&
          linear_fork_info.copied_bytes != 0U &&
          linear_fork_info.shared_bytes == 0U;
  sllm_state_image_info_t linear_image{};
  linear_image.struct_size = sizeof(linear_image);
  linear_image.abi_version = SLLM_HIP_ABI_VERSION;
  linear_image.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
  valid = valid &&
          expect_status(sllm_linear_attention_state_image_query(
                            linear_source, &linear_image, &error.sink),
                        SLLM_STATUS_OK, "linear image query", error) &&
          linear_image.active_slot == 1U &&
          linear_image.capacity_tokens == 7U &&
          linear_image.published_length == 1U && linear_image.plane_count == 5U;
  const std::array<uint64_t, 5> linear_plane_bytes = {
      UINT64_C(3) * 8192U * sizeof(uint16_t),
      UINT64_C(3) * 8192U * sizeof(uint16_t),
      UINT64_C(32) * 128U * 128U * sizeof(float),
      UINT64_C(32) * 128U * 128U * sizeof(float),
      UINT64_C(8192) * sizeof(uint16_t)};
  std::array<std::vector<uint8_t>, 5> linear_images;
  for (std::size_t index = 0U; index != linear_images.size(); ++index) {
    linear_images[index].resize(
        static_cast<std::size_t>(linear_plane_bytes[index]));
    sllm_state_chunk_t chunk{};
    chunk.struct_size = sizeof(chunk);
    chunk.abi_version = SLLM_HIP_ABI_VERSION;
    chunk.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
    chunk.plane = static_cast<uint32_t>(index + 1U);
    chunk.byte_length = linear_plane_bytes[index];
    chunk.host_pointer = linear_images[index].data();
    chunk.host_capacity = linear_plane_bytes[index];
    valid =
        valid && expect_status(sllm_linear_attention_state_export(
                                   linear_source, &chunk, &error.sink),
                               SLLM_STATUS_OK, "linear image export", error);
    valid =
        valid && expect_status(sllm_linear_attention_state_import(
                                   linear_child, &chunk, &error.sink),
                               SLLM_STATUS_OK, "linear image import", error);
  }
  valid =
      valid && expect_status(sllm_linear_attention_state_import_finalize(
                                 linear_child, &linear_image, &error.sink),
                             SLLM_STATUS_OK, "linear image finalize", error);
  linear_view = {};
  linear_view.struct_size = sizeof(linear_view);
  linear_view.abi_version = SLLM_HIP_ABI_VERSION;
  linear_view.info_version = SLLM_HIP_LINEAR_ATTENTION_VIEW_INFO_VERSION;
  valid = valid &&
          expect_status(sllm_linear_attention_state_query(
                            linear_child, &linear_view, &error.sink),
                        SLLM_STATUS_OK, "linear image query after finalize",
                        error) &&
          linear_view.active_slot == 1U && linear_view.capacity_tokens == 11U &&
          linear_view.observed_length == 1U && linear_view.generation == 1U;
  return cleanup_linear() && valid;
}

bool sliding_static_fp8_ring_image_fork_and_scale_contract() {
  fake_hip::reset();
  const std::size_t baseline_allocations = fake_hip::live_allocations();
  const std::size_t baseline_streams = fake_hip::live_streams();
  const std::size_t baseline_events = fake_hip::live_events();
  constexpr uint64_t capacity = SLLM_HIP_KV_SLIDING_MAX_CAPACITY;
  constexpr uint64_t window = SLLM_HIP_KV_SLIDING_WINDOW_GEMMA4;
  constexpr uint64_t value_bytes_per_token = 4U * 256U;
  constexpr uint64_t input_words_per_token = 4U * 256U;
  constexpr uint64_t input_bytes =
      window * input_words_per_token * sizeof(uint16_t);
  constexpr uint64_t image_plane_bytes = window * value_bytes_per_token;
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_kv_state_t *state = nullptr;
  sllm_kv_state_t *full_state = nullptr;
  sllm_kv_state_t *child = nullptr;
  sllm_kv_state_t *restored = nullptr;
  sllm_buffer_t *key = nullptr;
  sllm_buffer_t *value = nullptr;
  sllm_buffer_t *query = nullptr;
  sllm_buffer_t *output = nullptr;
  sllm_completion_t *completion = nullptr;
  Error error;
  const auto release_all = [&]() {
    bool valid = true;
    if (completion != nullptr) {
      valid = release_completion(&completion) && valid;
    }
    for (sllm_kv_state_t **candidate :
         {&restored, &child, &full_state, &state}) {
      if (*candidate != nullptr) {
        valid = expect_status(sllm_kv_state_release(candidate, &error.sink),
                              SLLM_STATUS_OK, "sliding state release", error) &&
                valid;
      }
    }
    for (sllm_buffer_t **buffer : {&key, &value, &query, &output}) {
      valid = release_buffer(buffer) && valid;
    }
    valid = release_queue(&queue) && valid;
    valid = release_context(&context) && valid;
    return valid;
  };
  const auto create_info = []() {
    sllm_kv_state_create_info_v2_t info{};
    info.struct_size = sizeof(info);
    info.abi_version = SLLM_HIP_ABI_VERSION;
    info.create_info_version =
        SLLM_HIP_KV_STATE_CREATE_INFO_SLIDING_STATIC_FP8_VERSION;
    info.session_id = 0x5510U;
    info.layer_id = 23U;
    info.capacity_tokens = capacity;
    info.head_count = 4U;
    info.head_dim = 256U;
    info.memory_kind = SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS;
    info.layout = SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR;
    info.dtype = SLLM_TENSOR_DTYPE_F8_E4M3_FN;
    info.encoding = SLLM_HIP_KV_ENCODING_FP8_STATIC_V1;
    info.scale_dtype = SLLM_TENSOR_DTYPE_F32;
    const float unit = 1.0F;
    std::memcpy(&info.reserved[0], &unit, sizeof(unit));
    std::memcpy(&info.reserved[1], &unit, sizeof(unit));
    info.reserved[2] = static_cast<uint32_t>(window);
    info.reserved[3] = static_cast<uint32_t>(window >> 32U);
    return info;
  };
  auto create = create_info();
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !expect_status(
          sllm_kv_state_create_v2(context, &create, &state, &error.sink),
          SLLM_STATUS_OK, "sliding state create", error) ||
      !create_buffer_sized(context, input_bytes, &key) ||
      !create_buffer_sized(context, input_bytes, &value) ||
      !create_buffer_sized(context, 16U * 256U * sizeof(uint16_t), &query) ||
      !create_buffer_sized(context, 16U * 256U * sizeof(uint16_t), &output)) {
    (void)release_all();
    return false;
  }
  constexpr std::array<uint16_t, 5> input_patterns{
      UINT16_C(0x0000), UINT16_C(0x3f00), UINT16_C(0x3f80), UINT16_C(0x4000),
      UINT16_C(0xbf80)};
  constexpr std::array<uint8_t, 5> encoded_patterns{
      UINT8_C(0x00), UINT8_C(0x30), UINT8_C(0x38), UINT8_C(0x40),
      UINT8_C(0xb8)};
  std::vector<uint16_t> words(
      static_cast<std::size_t>(window * input_words_per_token));
  for (uint64_t token = 0U; token != window; ++token) {
    std::fill_n(words.begin() +
                    static_cast<std::ptrdiff_t>(token * input_words_per_token),
                static_cast<std::ptrdiff_t>(input_words_per_token),
                input_patterns[static_cast<std::size_t>(
                    token % input_patterns.size())]);
  }
  bool valid = upload_kv_words(queue, key, words) &&
               upload_kv_words(queue, value, words);
  auto full_create = create;
  full_create.create_info_version =
      SLLM_HIP_KV_STATE_CREATE_INFO_STATIC_FP8_VERSION;
  full_create.capacity_tokens = 3U;
  full_create.memory_kind = SLLM_HIP_KV_MEMORY_KIND_CONTIGUOUS_RESIDENT;
  full_create.reserved[2] = 0U;
  full_create.reserved[3] = 0U;
  valid = valid &&
          expect_status(sllm_kv_state_create_v2(context, &full_create,
                                                &full_state, &error.sink),
                        SLLM_STATUS_OK, "full static state create", error) &&
          full_state != nullptr;
  if (full_state != nullptr) {
    auto full_append = kv_append_descriptor(key, value, 1U, 0U);
    auto full_append_info = kv_append_info();
    valid = valid &&
            expect_status(sllm_kv_state_append(full_state, queue, &full_append,
                                               &completion, &full_append_info,
                                               &error.sink),
                          SLLM_STATUS_OK, "full static append", error) &&
            completion != nullptr &&
            query_completion(completion, SLLM_STATUS_OK) &&
            release_completion(&completion);
    auto full_attention =
        causal_attention_descriptor(full_state, query, output, 1U, 0U, 1U);
    full_attention.op_version =
        SLLM_HIP_CAUSAL_ATTENTION_EXPLICIT_SCALE_VERSION;
    const float unit_score_scale = 1.0F;
    std::memcpy(&full_attention.reserved[2], &unit_score_scale,
                sizeof(unit_score_scale));
    auto full_dispatch = causal_attention_dispatch_info();
    valid =
        valid &&
        expect_status(sllm_causal_attention_execute(
                          context, queue, &full_attention, &completion,
                          &full_dispatch, &error.sink),
                      SLLM_STATUS_OK, "full scaled static attention", error) &&
        completion != nullptr &&
        full_dispatch.kernel_id ==
            SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_SCALED_STATIC_FP8_V1 &&
        full_dispatch.scale_denominator == 0U &&
        full_dispatch.fallback_allowed == 0U &&
        full_dispatch.fallback_used == 0U &&
        full_dispatch.reserved[4] == UINT32_C(0x3f800000) &&
        full_dispatch.reserved[5] == 1U &&
        query_completion(completion, SLLM_STATUS_OK) &&
        release_completion(&completion);
  }
  if (!valid) {
    std::cerr << "full explicit score-scale phase failed\n";
  }
  const auto append_and_publish = [&](const uint64_t count,
                                      const uint64_t position) {
    auto descriptor = kv_append_descriptor(key, value, count, position);
    const uint64_t input_offset =
        (position % window) * input_words_per_token * sizeof(uint16_t);
    descriptor.key_input.byte_offset = input_offset;
    descriptor.value_input.byte_offset = input_offset;
    auto info = kv_append_info();
    completion = nullptr;
    return expect_status(sllm_kv_state_append(state, queue, &descriptor,
                                              &completion, &info, &error.sink),
                         SLLM_STATUS_OK, "sliding append", error) &&
           completion != nullptr &&
           query_completion(completion, SLLM_STATUS_OK) &&
           release_completion(&completion) && info.fallback_allowed == 0U &&
           info.fallback_used == 0U;
  };
  const auto query_ring = [&](const sllm_kv_state_t *const candidate,
                              const uint64_t length,
                              const uint64_t retained_start) {
    sllm_kv_view_info_t info{};
    info.struct_size = sizeof(info);
    info.abi_version = SLLM_HIP_ABI_VERSION;
    info.info_version = SLLM_HIP_KV_VIEW_INFO_VERSION;
    const bool matches =
        expect_status(sllm_kv_state_query(candidate, &info, &error.sink),
                      SLLM_STATUS_OK, "sliding state query", error) &&
        info.info_version == SLLM_HIP_KV_VIEW_INFO_SLIDING_VERSION &&
        info.observed_length == length && info.capacity_tokens == capacity &&
        info.mapped_token_capacity <= window + 1U &&
        info.reserved[0] == static_cast<uint32_t>(window) &&
        info.reserved[1] == 0U &&
        info.reserved[2] == static_cast<uint32_t>(retained_start) &&
        info.reserved[3] == static_cast<uint32_t>(retained_start >> 32U);
    if (!matches) {
      std::cerr << "sliding query mismatch version=" << info.info_version
                << " length=" << info.observed_length
                << " capacity=" << info.capacity_tokens
                << " mapped=" << info.mapped_token_capacity
                << " reserved=" << info.reserved[0] << ',' << info.reserved[1]
                << ',' << info.reserved[2] << ',' << info.reserved[3]
                << " expected_length=" << length
                << " expected_start=" << retained_start << '\n';
    }
    return matches;
  };
  valid = valid && append_and_publish(1023U, 0U) &&
          query_ring(state, 1023U, 0U) && append_and_publish(1U, 1023U) &&
          query_ring(state, 1024U, 0U) && append_and_publish(1U, 1024U) &&
          query_ring(state, 1025U, 1U);
  if (!valid) {
    std::cerr << "sliding boundary append/query phase failed\n";
  }

  auto rejected = kv_append_descriptor(key, value, 2U, 1025U);
  auto rejected_info = kv_append_info();
  completion = nullptr;
  valid =
      valid &&
      expect_status(sllm_kv_state_append(state, queue, &rejected, &completion,
                                         &rejected_info, &error.sink),
                    SLLM_STATUS_INVALID_KV_APPEND_DESCRIPTOR,
                    "saturated sliding M=2 rejection", error) &&
      completion == nullptr && query_ring(state, 1025U, 1U);
  if (!valid) {
    std::cerr << "sliding saturated rejection phase failed\n";
  }

  valid =
      valid && append_and_publish(1U, 1025U) && query_ring(state, 1026U, 2U);
  if (!valid) {
    std::cerr << "sliding wrap publication phase failed\n";
  }

  auto cancel_descriptor = kv_append_descriptor(key, value, 1U, 1026U);
  auto cancel_info = kv_append_info();
  fake_hip::set_completion_pending(true);
  valid =
      valid &&
      expect_status(sllm_kv_state_append(state, queue, &cancel_descriptor,
                                         &completion, &cancel_info,
                                         &error.sink),
                    SLLM_STATUS_OK, "sliding append before cancel", error) &&
      completion != nullptr &&
      expect_status(sllm_kv_state_append_cancel(state, completion, &error.sink),
                    SLLM_STATUS_OK, "sliding append cancel", error) &&
      expect_status(sllm_kv_state_append_cancel(state, completion, &error.sink),
                    SLLM_STATUS_OK, "sliding append cancel idempotent", error);
  fake_hip::set_completion_pending(false);
  valid = valid && query_completion(completion, SLLM_STATUS_OK) &&
          release_completion(&completion) && query_ring(state, 1026U, 2U);
  if (!valid) {
    std::cerr << "sliding cancel phase failed\n";
  }

  auto attention =
      causal_attention_descriptor(state, query, output, 1U, 1025U, 1026U);
  attention.op_version = SLLM_HIP_CAUSAL_ATTENTION_EXPLICIT_SCALE_VERSION;
  attention.reserved[0] = static_cast<uint32_t>(window);
  attention.reserved[1] = 0U;
  const float score_scale = 1.0F;
  std::memcpy(&attention.reserved[2], &score_scale, sizeof(score_scale));
  auto dispatch = causal_attention_dispatch_info();
  valid = valid &&
          expect_status(sllm_causal_attention_execute(context, queue,
                                                      &attention, &completion,
                                                      &dispatch, &error.sink),
                        SLLM_STATUS_OK, "sliding scaled attention", error) &&
          completion != nullptr &&
          dispatch.kernel_id ==
              SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_SLIDING_STATIC_FP8_V1 &&
          dispatch.dispatch_count == 1U && dispatch.scale_denominator == 0U &&
          dispatch.fallback_allowed == 0U && dispatch.fallback_used == 0U &&
          dispatch.reserved[0] == static_cast<uint32_t>(window) &&
          dispatch.reserved[2] == 2U &&
          dispatch.reserved[4] == UINT32_C(0x3f800000) &&
          dispatch.reserved[5] == 1U &&
          query_completion(completion, SLLM_STATUS_OK) &&
          release_completion(&completion);
  if (!valid) {
    std::cerr << "sliding scaled attention phase failed\n";
  }

  sllm_state_image_info_t image{};
  image.struct_size = sizeof(image);
  image.abi_version = SLLM_HIP_ABI_VERSION;
  image.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
  uint64_t key_image_size = 0U;
  uint64_t value_image_size = 0U;
  valid = valid &&
          expect_status(sllm_kv_state_image_query(state, &image, &error.sink),
                        SLLM_STATUS_OK, "sliding image query", error) &&
          image.info_version == SLLM_HIP_STATE_IMAGE_SLIDING_VERSION &&
          image.published_length == 1026U && image.reserved[0] == window &&
          image.reserved[2] == 2U &&
          expect_status(
              sllm_kv_state_image_plane_size(state, SLLM_HIP_KV_STATE_PLANE_KEY,
                                             &key_image_size, &error.sink),
              SLLM_STATUS_OK, "sliding key image size", error) &&
          expect_status(sllm_kv_state_image_plane_size(
                            state, SLLM_HIP_KV_STATE_PLANE_VALUE,
                            &value_image_size, &error.sink),
                        SLLM_STATUS_OK, "sliding value image size", error) &&
          key_image_size == image_plane_bytes &&
          value_image_size == image_plane_bytes;
  if (!valid) {
    std::cerr << "sliding image metadata phase failed key=" << key_image_size
              << " value=" << value_image_size << '\n';
  }
  std::vector<uint8_t> key_image(static_cast<std::size_t>(key_image_size));
  std::vector<uint8_t> value_image(static_cast<std::size_t>(value_image_size));
  const auto chunk_for = [&](const uint32_t plane, void *const host) {
    sllm_state_chunk_t chunk{};
    chunk.struct_size = sizeof(chunk);
    chunk.abi_version = SLLM_HIP_ABI_VERSION;
    chunk.info_version = SLLM_HIP_STATE_IMAGE_SLIDING_VERSION;
    chunk.plane = plane;
    chunk.reserved0 = static_cast<uint32_t>(window);
    chunk.byte_length = image_plane_bytes;
    chunk.host_pointer = host;
    chunk.host_capacity = image_plane_bytes;
    chunk.reserved[0] = 1026U;
    return chunk;
  };
  auto key_chunk = chunk_for(SLLM_HIP_KV_STATE_PLANE_KEY, key_image.data());
  auto value_chunk =
      chunk_for(SLLM_HIP_KV_STATE_PLANE_VALUE, value_image.data());
  valid = valid &&
          expect_status(sllm_kv_state_export(state, &key_chunk, &error.sink),
                        SLLM_STATUS_OK, "sliding key export", error) &&
          expect_status(sllm_kv_state_export(state, &value_chunk, &error.sink),
                        SLLM_STATUS_OK, "sliding value export", error);
  std::vector<uint8_t> expected_image(
      static_cast<std::size_t>(image_plane_bytes));
  for (uint64_t retained = 0U; retained != window; ++retained) {
    const uint64_t logical_token = retained + 2U;
    std::fill_n(expected_image.begin() + static_cast<std::ptrdiff_t>(
                                             retained * value_bytes_per_token),
                static_cast<std::ptrdiff_t>(value_bytes_per_token),
                encoded_patterns[static_cast<std::size_t>(
                    (logical_token % window) % encoded_patterns.size())]);
  }
  valid = valid && key_image == expected_image && value_image == expected_image;
  if (!valid) {
    const auto mismatch = std::mismatch(key_image.begin(), key_image.end(),
                                        expected_image.begin());
    if (mismatch.first != key_image.end()) {
      const auto index = static_cast<std::size_t>(
          std::distance(key_image.begin(), mismatch.first));
      std::cerr << "sliding key image mismatch index=" << index
                << " actual=" << static_cast<uint32_t>(key_image[index])
                << " expected=" << static_cast<uint32_t>(expected_image[index])
                << '\n';
    }
    std::cerr << "sliding image export phase failed\n";
  }

  sllm_state_fork_info_t fork{};
  fork.struct_size = sizeof(fork);
  fork.abi_version = SLLM_HIP_ABI_VERSION;
  fork.info_version = SLLM_HIP_STATE_FORK_INFO_VERSION;
  valid = valid &&
          expect_status(
              sllm_kv_state_fork(state, &create, &child, &fork, &error.sink),
              SLLM_STATUS_OK, "sliding state fork", error) &&
          child != nullptr && fork.published_length == 1026U &&
          fork.mode == SLLM_HIP_STATE_FORK_MODE_SHARED_READ_ONLY_PAGES &&
          fork.child_owned_bytes == 0U && fork.copied_bytes == 0U &&
          fork.shared_bytes <= 2U * fork.page_bytes &&
          query_ring(child, 1026U, 2U);
  if (!valid) {
    std::cerr << "sliding fork phase failed shared=" << fork.shared_bytes
              << " page=" << fork.page_bytes << '\n';
  }

  valid = valid &&
          expect_status(
              sllm_kv_state_create_v2(context, &create, &restored, &error.sink),
              SLLM_STATUS_OK, "sliding restore state create", error) &&
          restored != nullptr;
  if (restored != nullptr) {
    valid =
        valid &&
        expect_status(sllm_kv_state_import(restored, &key_chunk, &error.sink),
                      SLLM_STATUS_OK, "sliding key import", error) &&
        expect_status(sllm_kv_state_import(restored, &value_chunk, &error.sink),
                      SLLM_STATUS_OK, "sliding value import", error) &&
        expect_status(
            sllm_kv_state_import_finalize(restored, &image, &error.sink),
            SLLM_STATUS_OK, "sliding import finalize", error) &&
        query_ring(restored, 1026U, 2U);
  }
  const auto exported_image_matches = [&](const sllm_kv_state_t *candidate) {
    std::vector<uint8_t> observed_key(expected_image.size());
    std::vector<uint8_t> observed_value(expected_image.size());
    auto observed_key_chunk =
        chunk_for(SLLM_HIP_KV_STATE_PLANE_KEY, observed_key.data());
    auto observed_value_chunk =
        chunk_for(SLLM_HIP_KV_STATE_PLANE_VALUE, observed_value.data());
    return candidate != nullptr &&
           expect_status(sllm_kv_state_export(candidate, &observed_key_chunk,
                                              &error.sink),
                         SLLM_STATUS_OK, "sliding retained key re-export",
                         error) &&
           expect_status(sllm_kv_state_export(candidate, &observed_value_chunk,
                                              &error.sink),
                         SLLM_STATUS_OK, "sliding retained value re-export",
                         error) &&
           observed_key == expected_image && observed_value == expected_image;
  };
  valid = valid && exported_image_matches(child) &&
          exported_image_matches(restored);
  if (!valid) {
    std::cerr << "sliding restore phase failed\n";
  }
  const bool released = release_all();
  const bool clean = fake_hip::live_allocations() == baseline_allocations &&
                     fake_hip::live_streams() == baseline_streams &&
                     fake_hip::live_events() == baseline_events;
  if (!released || !clean) {
    std::cerr << "sliding cleanup failed released=" << released
              << " allocations=" << fake_hip::live_allocations()
              << " streams=" << fake_hip::live_streams()
              << " events=" << fake_hip::live_events() << '\n';
  }
  return released && valid && clean;
}

sllm_tensor_binding_t
deepseek_v4_route_binding(const sllm_buffer_t *const buffer,
                          const uint32_t dtype, const uint32_t rank,
                          const uint64_t first, const uint64_t second = 0U) {
  sllm_tensor_binding_t result{};
  result.struct_size = sizeof(result);
  result.abi_version = SLLM_HIP_ABI_VERSION;
  result.buffer = buffer;
  result.dtype = dtype;
  result.encoding = SLLM_TENSOR_ENCODING_UNQUANTIZED;
  result.rank = rank;
  result.shape[0] = first;
  result.stride_elements[0] = rank == 2U ? second : 1U;
  if (rank == 2U) {
    result.shape[1] = second;
    result.stride_elements[1] = 1U;
  }
  return result;
}

bool deepseek_v4_moe_route_descriptor_abi_and_lifetime_contract() {
  constexpr uint64_t tokens = 3U;
  constexpr uint64_t experts = SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_EXPERT_COUNT;
  constexpr uint32_t selected =
      SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_SELECTED_EXPERT_COUNT;
  constexpr uint64_t pairs = tokens * selected;
  constexpr uint64_t metadata_bytes =
      pairs * UINT64_C(16) + experts * UINT64_C(4) +
      (experts + 1U) * UINT64_C(4) + UINT64_C(4);
  const auto *const sentinel = reinterpret_cast<const sllm_buffer_t *>(1U);
  sllm_deepseek_v4_moe_route_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_VERSION;
  descriptor.mode = SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_SCORE;
  descriptor.selected_expert_count = selected;
  descriptor.renormalize = 1U;
  descriptor.routed_scale = 1.5F;
  descriptor.logits = deepseek_v4_route_binding(
      sentinel, SLLM_TENSOR_DTYPE_BF16, 2U, tokens, experts);
  descriptor.selection_bias =
      deepseek_v4_route_binding(sentinel, SLLM_TENSOR_DTYPE_F32, 1U, experts);
  descriptor.metadata = deepseek_v4_route_binding(
      sentinel, SLLM_TENSOR_DTYPE_U8, 1U, metadata_bytes);
  const auto query = [&](const sllm_deepseek_v4_moe_route_desc_t &candidate,
                         const sllm_status_t expected,
                         const char *const label) {
    sllm_deepseek_v4_moe_route_query_info_t info{};
    info.struct_size = sizeof(info);
    info.abi_version = SLLM_HIP_ABI_VERSION;
    info.info_version = SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_QUERY_INFO_VERSION;
    Error error;
    const bool valid = expect_status(
        sllm_deepseek_v4_moe_route_query(&candidate, &info, &error.sink),
        expected, label, error);
    return valid &&
           (expected != SLLM_STATUS_OK ||
            (info.mode == candidate.mode && info.token_count == tokens &&
             info.expert_count == experts && info.pair_count == pairs &&
             info.metadata_bytes == metadata_bytes &&
             info.selected_expert_count == selected &&
             info.renormalize == candidate.renormalize &&
             info.routed_scale == candidate.routed_scale));
  };
  bool valid = query(descriptor, SLLM_STATUS_OK, "DeepSeek route query");
  for (const float invalid_scale :
       {0.0F, -1.0F, std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()}) {
    auto candidate = descriptor;
    candidate.routed_scale = invalid_scale;
    valid = valid && query(candidate, SLLM_STATUS_INVALID_ARGUMENT,
                           "DeepSeek route invalid routed scale");
  }
  {
    auto candidate = descriptor;
    candidate.struct_size -= 1U;
    valid = valid && query(candidate, SLLM_STATUS_INVALID_ARGUMENT,
                           "DeepSeek route descriptor size");
  }
  {
    auto candidate = descriptor;
    candidate.abi_version += 1U;
    valid = valid && query(candidate, SLLM_STATUS_INVALID_ABI_VERSION,
                           "DeepSeek route descriptor ABI");
  }
  {
    auto candidate = descriptor;
    candidate.reserved[3] = 1U;
    valid = valid && query(candidate, SLLM_STATUS_RESERVED_NONZERO,
                           "DeepSeek route descriptor reserved");
  }
  {
    auto candidate = descriptor;
    candidate.mode = 0U;
    valid = valid && query(candidate, SLLM_STATUS_INVALID_ARGUMENT,
                           "DeepSeek route mode boundary");
  }
  {
    auto candidate = descriptor;
    candidate.renormalize = 2U;
    valid = valid && query(candidate, SLLM_STATUS_INVALID_ARGUMENT,
                           "DeepSeek route renormalize boundary");
  }
  {
    auto candidate = descriptor;
    candidate.logits.shape[0] = 0U;
    valid = valid && query(candidate, SLLM_STATUS_UNSUPPORTED,
                           "DeepSeek route zero-token boundary");
    candidate.logits.shape[0] = SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_MAX_TOKENS + 1U;
    valid = valid && query(candidate, SLLM_STATUS_UNSUPPORTED,
                           "DeepSeek route maximum-token boundary");
  }
  {
    auto candidate = descriptor;
    candidate.logits.shape[1] = experts - 1U;
    candidate.logits.stride_elements[0] = experts - 1U;
    valid = valid && query(candidate, SLLM_STATUS_SHAPE_MISMATCH,
                           "DeepSeek route E boundary");
  }
  {
    auto candidate = descriptor;
    candidate.selected_expert_count = selected - 1U;
    valid = valid && query(candidate, SLLM_STATUS_UNSUPPORTED,
                           "DeepSeek route K boundary");
  }
  {
    auto candidate = descriptor;
    candidate.hash_expert_ids = deepseek_v4_route_binding(
        sentinel, SLLM_TENSOR_DTYPE_I32, 2U, tokens, selected);
    valid = valid && query(candidate, SLLM_STATUS_INVALID_TENSOR_BINDING,
                           "DeepSeek route score inactive hash binding");
  }
  {
    auto candidate = descriptor;
    candidate.metadata.shape[0] -= 1U;
    valid = valid && query(candidate, SLLM_STATUS_SHAPE_MISMATCH,
                           "DeepSeek route metadata byte boundary");
  }
  {
    auto candidate = descriptor;
    candidate.selection_bias.byte_offset = 2U;
    valid = valid && query(candidate, SLLM_STATUS_MISALIGNED_OFFSET,
                           "DeepSeek route bias alignment");
    candidate = descriptor;
    candidate.metadata.byte_offset = 1U;
    valid = valid && query(candidate, SLLM_STATUS_MISALIGNED_OFFSET,
                           "DeepSeek route metadata alignment");
  }
  {
    auto candidate = descriptor;
    candidate.mode = SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_HASH;
    candidate.selection_bias = {};
    candidate.hash_expert_ids = deepseek_v4_route_binding(
        sentinel, SLLM_TENSOR_DTYPE_I32, 2U, tokens, selected);
    valid =
        valid && query(candidate, SLLM_STATUS_OK, "DeepSeek route hash query");
    candidate.selection_bias = descriptor.selection_bias;
    valid = valid && query(candidate, SLLM_STATUS_INVALID_TENSOR_BINDING,
                           "DeepSeek route hash inactive bias binding");
  }
  {
    sllm_deepseek_v4_moe_route_query_info_t info{};
    info.struct_size = sizeof(info) - 1U;
    info.abi_version = SLLM_HIP_ABI_VERSION;
    info.info_version = SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_QUERY_INFO_VERSION;
    Error error;
    valid = valid && expect_status(sllm_deepseek_v4_moe_route_query(
                                       &descriptor, &info, &error.sink),
                                   SLLM_STATUS_INVALID_ARGUMENT,
                                   "DeepSeek route query info size", error);
    info = {};
    info.struct_size = sizeof(info);
    info.abi_version = SLLM_HIP_ABI_VERSION + 1U;
    info.info_version = SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_QUERY_INFO_VERSION;
    valid = valid && expect_status(sllm_deepseek_v4_moe_route_query(
                                       &descriptor, &info, &error.sink),
                                   SLLM_STATUS_INVALID_ABI_VERSION,
                                   "DeepSeek route query info ABI", error);
    info = {};
    info.struct_size = sizeof(info);
    info.abi_version = SLLM_HIP_ABI_VERSION;
    info.info_version = SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_QUERY_INFO_VERSION + 1U;
    valid = valid && expect_status(sllm_deepseek_v4_moe_route_query(
                                       &descriptor, &info, &error.sink),
                                   SLLM_STATUS_INVALID_ARGUMENT,
                                   "DeepSeek route query info version", error);
  }

  sllm_context_t *context = nullptr;
  sllm_buffer_t *logits = nullptr;
  sllm_buffer_t *bias = nullptr;
  sllm_buffer_t *output = nullptr;
  sllm_deepseek_v4_moe_route_plan_t *plan = nullptr;
  valid =
      valid && create_context(&context) &&
      create_buffer_sized(context, tokens * experts * UINT64_C(2), &logits) &&
      create_buffer_sized(context, experts * UINT64_C(4), &bias) &&
      create_buffer_sized(context, metadata_bytes, &output);
  if (valid) {
    descriptor.logits = deepseek_v4_route_binding(
        logits, SLLM_TENSOR_DTYPE_BF16, 2U, tokens, experts);
    descriptor.selection_bias =
        deepseek_v4_route_binding(bias, SLLM_TENSOR_DTYPE_F32, 1U, experts);
    descriptor.metadata = deepseek_v4_route_binding(
        output, SLLM_TENSOR_DTYPE_U8, 1U, metadata_bytes);
    Error error;
    valid = expect_status(sllm_deepseek_v4_moe_route_prepare(
                              context, &descriptor, &plan, &error.sink),
                          SLLM_STATUS_OK, "DeepSeek route prepare", error) &&
            plan != nullptr;
    sllm_deepseek_v4_moe_route_dispatch_info_t dispatch{};
    dispatch.struct_size = sizeof(dispatch) - 1U;
    dispatch.abi_version = SLLM_HIP_ABI_VERSION;
    dispatch.info_version =
        SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_DISPATCH_INFO_VERSION;
    sllm_completion_t *completion = nullptr;
    const auto *const queue_sentinel =
        reinterpret_cast<const sllm_queue_t *>(1U);
    valid = valid &&
            expect_status(
                sllm_deepseek_v4_moe_route_execute(
                    plan, queue_sentinel, &completion, &dispatch, &error.sink),
                SLLM_STATUS_INVALID_ARGUMENT,
                "DeepSeek route dispatch info size", error) &&
            completion == nullptr;
    dispatch = {};
    dispatch.struct_size = sizeof(dispatch);
    dispatch.abi_version = SLLM_HIP_ABI_VERSION;
    dispatch.info_version =
        SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_DISPATCH_INFO_VERSION;
    valid = valid &&
            expect_status(
                sllm_deepseek_v4_moe_route_execute(
                    plan, queue_sentinel, &completion, &dispatch, &error.sink),
                SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                "DeepSeek route queue handle rejection", error) &&
            completion == nullptr;
    auto *wrong_kind = reinterpret_cast<sllm_moe_route_plan_t *>(plan);
    valid = valid &&
            expect_status(sllm_moe_route_plan_release(&wrong_kind, &error.sink),
                          SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                          "DeepSeek route cross-ABI handle rejection", error) &&
            wrong_kind != nullptr;
    valid = valid &&
            expect_status(sllm_buffer_release(&output, &error.sink),
                          SLLM_STATUS_PUBLIC_BUSY,
                          "DeepSeek route retained output", error) &&
            output != nullptr;
    valid = valid &&
            expect_status(
                sllm_deepseek_v4_moe_route_plan_release(&plan, &error.sink),
                SLLM_STATUS_OK, "DeepSeek route plan release", error) &&
            plan == nullptr;
  }
  Error release_error;
  const bool released =
      (output == nullptr ||
       expect_status(sllm_buffer_release(&output, &release_error.sink),
                     SLLM_STATUS_OK, "DeepSeek route output release",
                     release_error)) &&
      (bias == nullptr ||
       expect_status(sllm_buffer_release(&bias, &release_error.sink),
                     SLLM_STATUS_OK, "DeepSeek route bias release",
                     release_error)) &&
      (logits == nullptr ||
       expect_status(sllm_buffer_release(&logits, &release_error.sink),
                     SLLM_STATUS_OK, "DeepSeek route logits release",
                     release_error)) &&
      (context == nullptr ||
       expect_status(sllm_context_release(&context, &release_error.sink),
                     SLLM_STATUS_OK, "DeepSeek route context release",
                     release_error));
  return valid && released;
}

bool deepseek_v4_moe_route_device_status_completion_contract() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  constexpr uint64_t tokens = 3U;
  constexpr uint64_t experts = SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_EXPERT_COUNT;
  constexpr uint32_t selected =
      SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_SELECTED_EXPERT_COUNT;
  constexpr uint64_t pairs = tokens * selected;
  constexpr uint64_t metadata_bytes =
      pairs * UINT64_C(16) + experts * UINT64_C(4) +
      (experts + 1U) * UINT64_C(4) + UINT64_C(4);

  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *logits = nullptr;
  sllm_buffer_t *bias = nullptr;
  sllm_buffer_t *output = nullptr;
  sllm_deepseek_v4_moe_route_plan_t *plan = nullptr;
  if (!create_context(&context) || !create_queue(context, &queue) ||
      !create_buffer_sized(context, tokens * experts * UINT64_C(2), &logits) ||
      !create_buffer_sized(context, experts * UINT64_C(4), &bias) ||
      !create_buffer_sized(context, metadata_bytes, &output)) {
    return false;
  }
  sllm_deepseek_v4_moe_route_desc_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = SLLM_HIP_ABI_VERSION;
  descriptor.op_version = SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_VERSION;
  descriptor.mode = SLLM_DEEPSEEK_V4_MOE_ROUTE_MODE_SCORE;
  descriptor.selected_expert_count = selected;
  descriptor.renormalize = 1U;
  descriptor.routed_scale = 1.5F;
  descriptor.logits = deepseek_v4_route_binding(logits, SLLM_TENSOR_DTYPE_BF16,
                                                2U, tokens, experts);
  descriptor.selection_bias =
      deepseek_v4_route_binding(bias, SLLM_TENSOR_DTYPE_F32, 1U, experts);
  descriptor.metadata = deepseek_v4_route_binding(output, SLLM_TENSOR_DTYPE_U8,
                                                  1U, metadata_bytes);
  Error prepare_error;
  bool valid =
      expect_status(sllm_deepseek_v4_moe_route_prepare(
                        context, &descriptor, &plan, &prepare_error.sink),
                    SLLM_STATUS_OK, "DeepSeek status prepare", prepare_error) &&
      plan != nullptr;

  struct StatusCase final {
    int32_t device_status;
    sllm_status_t expected_status;
    const char *message_fragment;
  };
  constexpr std::array<StatusCase, 6> cases{{
      {SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_OK, SLLM_STATUS_OK, nullptr},
      {SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_NONFINITE,
       SLLM_STATUS_INVALID_ARGUMENT, "non-finite"},
      {SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_EXPERT_OUT_OF_RANGE,
       SLLM_STATUS_INVALID_ARGUMENT, "out-of-range"},
      {SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_DUPLICATE_EXPERT,
       SLLM_STATUS_INVALID_ARGUMENT, "duplicate"},
      {SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_ZERO_NORMALIZER,
       SLLM_STATUS_INVALID_ARGUMENT, "normalizer"},
      {INT32_C(99), SLLM_STATUS_INTERNAL_ERROR, "missing or unsupported"},
  }};
  const std::size_t baseline_events = fake_hip::live_events();
  for (const auto &test_case : cases) {
    if (!valid) {
      break;
    }
    sllm_test_deepseek_v4_moe_route_device_status(test_case.device_status);
    sllm_deepseek_v4_moe_route_dispatch_info_t dispatch{};
    dispatch.struct_size = sizeof(dispatch);
    dispatch.abi_version = SLLM_HIP_ABI_VERSION;
    dispatch.info_version =
        SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_DISPATCH_INFO_VERSION;
    sllm_completion_t *completion = nullptr;
    Error execute_error;
    valid = expect_status(
                sllm_deepseek_v4_moe_route_execute(
                    plan, queue, &completion, &dispatch, &execute_error.sink),
                SLLM_STATUS_OK, "DeepSeek status execute", execute_error) &&
            completion != nullptr && dispatch.fallback_allowed == 0U &&
            dispatch.fallback_used == 0U;
    if (!valid) {
      break;
    }
    sllm_completion_result_t result{};
    result.struct_size = sizeof(result);
    result.abi_version = SLLM_HIP_ABI_VERSION;
    Error wait_error;
    const bool query_first = test_case.device_status ==
                             SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_DUPLICATE_EXPERT;
    const sllm_status_t wait_status =
        query_first
            ? sllm_completion_query(completion, &result, &wait_error.sink)
            : sllm_completion_wait(completion, 1000U, &result,
                                   &wait_error.sink);
    const uint32_t expected_state = test_case.expected_status == SLLM_STATUS_OK
                                        ? SLLM_COMPLETION_STATE_SUCCESS
                                        : SLLM_COMPLETION_STATE_FAILURE;
    valid = expect_status(wait_status, test_case.expected_status,
                          query_first ? "DeepSeek first status query"
                                      : "DeepSeek status wait",
                          wait_error) &&
            result.state == expected_state &&
            (test_case.message_fragment == nullptr ||
             std::strstr(wait_error.message, test_case.message_fragment) !=
                 nullptr);
    result = {};
    result.struct_size = sizeof(result);
    result.abi_version = SLLM_HIP_ABI_VERSION;
    Error repeated_error;
    valid = valid &&
            expect_status(sllm_completion_query(completion, &result,
                                                &repeated_error.sink),
                          test_case.expected_status,
                          "DeepSeek repeated status query", repeated_error) &&
            result.state == expected_state && release_completion(&completion) &&
            completion == nullptr && fake_hip::live_events() == baseline_events;
  }

  if (valid) {
    Error mode_error;
    valid = expect_status(
        sllm_queue_set_completion_mode(
            queue, SLLM_QUEUE_COMPLETION_MODE_DEFERRED, &mode_error.sink),
        SLLM_STATUS_OK, "DeepSeek deferred mode", mode_error);
  }
  sllm_completion_t *deferred = nullptr;
  sllm_completion_t *fence = nullptr;
  if (valid) {
    sllm_test_deepseek_v4_moe_route_device_status(
        SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_DUPLICATE_EXPERT);
    sllm_deepseek_v4_moe_route_dispatch_info_t dispatch{};
    dispatch.struct_size = sizeof(dispatch);
    dispatch.abi_version = SLLM_HIP_ABI_VERSION;
    dispatch.info_version =
        SLLM_HIP_DEEPSEEK_V4_MOE_ROUTE_DISPATCH_INFO_VERSION;
    Error execute_error;
    valid = expect_status(
                sllm_deepseek_v4_moe_route_execute(
                    plan, queue, &deferred, &dispatch, &execute_error.sink),
                SLLM_STATUS_OK, "DeepSeek deferred execute", execute_error) &&
            deferred != nullptr && fake_hip::live_events() == baseline_events;
  }
  if (valid) {
    Error fence_error;
    valid =
        expect_status(sllm_queue_fence(queue, &fence, &fence_error.sink),
                      SLLM_STATUS_OK, "DeepSeek deferred fence", fence_error) &&
        fence != nullptr && fake_hip::live_events() == baseline_events + 1U;
  }
  if (valid) {
    sllm_completion_result_t result{};
    result.struct_size = sizeof(result);
    result.abi_version = SLLM_HIP_ABI_VERSION;
    Error wait_error;
    valid = expect_status(
                sllm_completion_wait(fence, 1000U, &result, &wait_error.sink),
                SLLM_STATUS_OK, "DeepSeek deferred fence wait", wait_error) &&
            result.state == SLLM_COMPLETION_STATE_SUCCESS;
    result = {};
    result.struct_size = sizeof(result);
    result.abi_version = SLLM_HIP_ABI_VERSION;
    Error finalize_error;
    valid =
        valid &&
        expect_status(sllm_completion_finalize_after(deferred, fence, &result,
                                                     &finalize_error.sink),
                      SLLM_STATUS_INVALID_ARGUMENT,
                      "DeepSeek deferred semantic failure", finalize_error) &&
        result.state == SLLM_COMPLETION_STATE_FAILURE &&
        std::strstr(finalize_error.message, "duplicate") != nullptr &&
        release_completion(&deferred) && deferred == nullptr &&
        fake_hip::live_events() == baseline_events + 1U &&
        release_completion(&fence) && fence == nullptr &&
        fake_hip::live_events() == baseline_events;
  }

  sllm_test_deepseek_v4_moe_route_device_status(
      SLLM_DEEPSEEK_V4_MOE_ROUTE_STATUS_OK);
  Error release_error;
  const bool released =
      (deferred == nullptr || release_completion(&deferred)) &&
      (fence == nullptr || release_completion(&fence)) &&
      (plan == nullptr ||
       expect_status(
           sllm_deepseek_v4_moe_route_plan_release(&plan, &release_error.sink),
           SLLM_STATUS_OK, "DeepSeek status plan release", release_error)) &&
      release_buffer(&output) && release_buffer(&bias) &&
      release_buffer(&logits) && release_queue(&queue) &&
      release_context(&context);
  return valid && released && fake_hip::live_events() == baseline_events;
}

bool minimax_m3_moe_route_public_contract() {
  fake_hip::reset();
  sllm_public_runtime::FaultInjector::reset();
  constexpr uint64_t experts = SLLM_HIP_MINIMAX_M3_MOE_ROUTE_EXPERT_COUNT;
  constexpr uint32_t selected =
      SLLM_HIP_MINIMAX_M3_MOE_ROUTE_SELECTED_EXPERT_COUNT;
  const auto metadata_bytes = [](const uint64_t tokens) {
    return tokens * selected * UINT64_C(16) + experts * UINT64_C(4) +
           (experts + 1U) * UINT64_C(4) + UINT64_C(4);
  };
  const auto *const sentinel = reinterpret_cast<const sllm_buffer_t *>(1U);
  const auto descriptor_for = [&](const uint64_t tokens) {
    sllm_minimax_m3_moe_route_desc_t descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = SLLM_HIP_ABI_VERSION;
    descriptor.op_version = SLLM_HIP_MINIMAX_M3_MOE_ROUTE_VERSION;
    descriptor.selected_expert_count = selected;
    descriptor.logits = deepseek_v4_route_binding(
        sentinel, SLLM_TENSOR_DTYPE_F32, 2U, tokens, experts);
    descriptor.selection_bias =
        deepseek_v4_route_binding(sentinel, SLLM_TENSOR_DTYPE_F32, 1U, experts);
    descriptor.metadata = deepseek_v4_route_binding(
        sentinel, SLLM_TENSOR_DTYPE_U8, 1U, metadata_bytes(tokens));
    return descriptor;
  };
  const auto query = [&](const sllm_minimax_m3_moe_route_desc_t &descriptor,
                         const sllm_status_t expected,
                         const char *const label) {
    sllm_minimax_m3_moe_route_query_info_t info{};
    info.struct_size = sizeof(info);
    info.abi_version = SLLM_HIP_ABI_VERSION;
    info.info_version = SLLM_HIP_MINIMAX_M3_MOE_ROUTE_QUERY_INFO_VERSION;
    Error error;
    const bool ok = expect_status(
        sllm_minimax_m3_moe_route_query(&descriptor, &info, &error.sink),
        expected, label, error);
    return ok && (expected != SLLM_STATUS_OK ||
                  (info.token_count == descriptor.logits.shape[0] &&
                   info.expert_count == experts &&
                   info.pair_count == descriptor.logits.shape[0] * selected &&
                   info.metadata_bytes ==
                       metadata_bytes(descriptor.logits.shape[0]) &&
                   info.selected_expert_count == selected));
  };

  bool valid = true;
  for (const uint64_t tokens :
       {UINT64_C(1), UINT64_C(3), UINT64_C(5), UINT64_C(17)}) {
    valid = valid && query(descriptor_for(tokens), SLLM_STATUS_OK,
                           "MiniMax route M query");
  }
  auto descriptor = descriptor_for(3U);
  {
    auto candidate = descriptor;
    candidate.struct_size -= 1U;
    valid = valid && query(candidate, SLLM_STATUS_INVALID_ARGUMENT,
                           "MiniMax route descriptor size");
    candidate = descriptor;
    candidate.abi_version += 1U;
    valid = valid && query(candidate, SLLM_STATUS_INVALID_ABI_VERSION,
                           "MiniMax route descriptor ABI");
    candidate = descriptor;
    candidate.reserved[0] = 1U;
    valid = valid && query(candidate, SLLM_STATUS_RESERVED_NONZERO,
                           "MiniMax route descriptor reserved");
    candidate = descriptor;
    candidate.selected_expert_count = selected - 1U;
    valid = valid && query(candidate, SLLM_STATUS_UNSUPPORTED,
                           "MiniMax route K boundary");
    candidate = descriptor;
    candidate.logits.dtype = SLLM_TENSOR_DTYPE_BF16;
    valid = valid && query(candidate, SLLM_STATUS_INVALID_TENSOR_BINDING,
                           "MiniMax route logits dtype");
    candidate = descriptor;
    candidate.logits.shape[0] = 0U;
    valid = valid &&
            query(candidate, SLLM_STATUS_UNSUPPORTED, "MiniMax route zero M");
    candidate = descriptor;
    candidate.logits.shape[1] = experts - 1U;
    candidate.logits.stride_elements[0] = experts - 1U;
    valid = valid && query(candidate, SLLM_STATUS_SHAPE_MISMATCH,
                           "MiniMax route E boundary");
    candidate = descriptor;
    candidate.selection_bias.shape[0] = experts - 1U;
    valid = valid && query(candidate, SLLM_STATUS_SHAPE_MISMATCH,
                           "MiniMax route bias shape");
    candidate = descriptor;
    candidate.metadata.shape[0] -= 1U;
    valid = valid && query(candidate, SLLM_STATUS_SHAPE_MISMATCH,
                           "MiniMax route metadata bytes");
    candidate = descriptor;
    candidate.metadata.byte_offset = 1U;
    valid = valid && query(candidate, SLLM_STATUS_MISALIGNED_OFFSET,
                           "MiniMax route metadata alignment");
  }
  {
    sllm_minimax_m3_moe_route_query_info_t info{};
    info.struct_size = sizeof(info);
    info.abi_version = SLLM_HIP_ABI_VERSION;
    info.info_version = SLLM_HIP_MINIMAX_M3_MOE_ROUTE_QUERY_INFO_VERSION;
    info.reserved[7] = 1U;
    Error error;
    valid = valid && expect_status(sllm_minimax_m3_moe_route_query(
                                       &descriptor, &info, &error.sink),
                                   SLLM_STATUS_RESERVED_NONZERO,
                                   "MiniMax route query reserved", error);
  }

  constexpr uint64_t tokens = 3U;
  sllm_context_t *context = nullptr;
  sllm_queue_t *queue = nullptr;
  sllm_buffer_t *logits = nullptr;
  sllm_buffer_t *bias = nullptr;
  sllm_buffer_t *output = nullptr;
  sllm_minimax_m3_moe_route_plan_t *plan = nullptr;
  valid =
      valid && create_context(&context) && create_queue(context, &queue) &&
      create_buffer_sized(context, tokens * experts * UINT64_C(4), &logits) &&
      create_buffer_sized(context, experts * UINT64_C(4), &bias) &&
      create_buffer_sized(context, metadata_bytes(tokens), &output);
  if (valid) {
    descriptor.logits = deepseek_v4_route_binding(logits, SLLM_TENSOR_DTYPE_F32,
                                                  2U, tokens, experts);
    descriptor.selection_bias =
        deepseek_v4_route_binding(bias, SLLM_TENSOR_DTYPE_F32, 1U, experts);
    descriptor.metadata = deepseek_v4_route_binding(
        output, SLLM_TENSOR_DTYPE_U8, 1U, metadata_bytes(tokens));
    Error error;
    valid = expect_status(sllm_minimax_m3_moe_route_prepare(
                              context, &descriptor, &plan, &error.sink),
                          SLLM_STATUS_OK, "MiniMax route prepare", error) &&
            plan != nullptr;
    sllm_minimax_m3_moe_route_dispatch_info_t invalid_dispatch{};
    invalid_dispatch.struct_size = sizeof(invalid_dispatch) - 1U;
    invalid_dispatch.abi_version = SLLM_HIP_ABI_VERSION;
    invalid_dispatch.info_version =
        SLLM_HIP_MINIMAX_M3_MOE_ROUTE_DISPATCH_INFO_VERSION;
    sllm_completion_t *invalid_completion = nullptr;
    valid = valid &&
            expect_status(sllm_minimax_m3_moe_route_execute(
                              plan, queue, &invalid_completion,
                              &invalid_dispatch, &error.sink),
                          SLLM_STATUS_INVALID_ARGUMENT,
                          "MiniMax invalid dispatch size", error) &&
            invalid_completion == nullptr;
    invalid_dispatch = {};
    invalid_dispatch.struct_size = sizeof(invalid_dispatch);
    invalid_dispatch.abi_version = SLLM_HIP_ABI_VERSION;
    invalid_dispatch.info_version =
        SLLM_HIP_MINIMAX_M3_MOE_ROUTE_DISPATCH_INFO_VERSION;
    invalid_dispatch.reserved[0] = 1U;
    valid = valid &&
            expect_status(sllm_minimax_m3_moe_route_execute(
                              plan, queue, &invalid_completion,
                              &invalid_dispatch, &error.sink),
                          SLLM_STATUS_RESERVED_NONZERO,
                          "MiniMax invalid dispatch reserved", error) &&
            invalid_completion == nullptr;
    auto *wrong_kind = reinterpret_cast<sllm_moe_route_plan_t *>(plan);
    valid = valid &&
            expect_status(sllm_moe_route_plan_release(&wrong_kind, &error.sink),
                          SLLM_STATUS_PUBLIC_INVALID_HANDLE,
                          "MiniMax cross-ABI plan rejection", error) &&
            wrong_kind != nullptr;
    valid = valid &&
            expect_status(sllm_buffer_release(&output, &error.sink),
                          SLLM_STATUS_PUBLIC_BUSY, "MiniMax retained output",
                          error) &&
            output != nullptr;
  }

  struct StatusCase final {
    int32_t device_status;
    sllm_status_t expected_status;
    const char *fragment;
    bool query_first;
  };
  constexpr std::array<StatusCase, 4> cases{{
      {SLLM_MINIMAX_M3_MOE_ROUTE_STATUS_OK, SLLM_STATUS_OK, nullptr, false},
      {SLLM_MINIMAX_M3_MOE_ROUTE_STATUS_NONFINITE, SLLM_STATUS_INVALID_ARGUMENT,
       "non-finite", true},
      {SLLM_MINIMAX_M3_MOE_ROUTE_STATUS_ZERO_NORMALIZER,
       SLLM_STATUS_INVALID_ARGUMENT, "normalizer", false},
      {INT32_C(99), SLLM_STATUS_INTERNAL_ERROR, "missing or unsupported", true},
  }};
  const std::size_t baseline_events = fake_hip::live_events();
  for (const auto &test_case : cases) {
    if (!valid) {
      break;
    }
    sllm_test_minimax_m3_moe_route_device_status(test_case.device_status);
    sllm_minimax_m3_moe_route_dispatch_info_t dispatch{};
    dispatch.struct_size = sizeof(dispatch);
    dispatch.abi_version = SLLM_HIP_ABI_VERSION;
    dispatch.info_version = SLLM_HIP_MINIMAX_M3_MOE_ROUTE_DISPATCH_INFO_VERSION;
    sllm_completion_t *completion = nullptr;
    Error execute_error;
    valid =
        expect_status(sllm_minimax_m3_moe_route_execute(plan, queue,
                                                        &completion, &dispatch,
                                                        &execute_error.sink),
                      SLLM_STATUS_OK, "MiniMax route execute", execute_error) &&
        completion != nullptr && dispatch.dispatch_count == 2U &&
        dispatch.kernel_id ==
            SLLM_HIP_MINIMAX_M3_MOE_ROUTE_KERNEL_ID_SIGMOID_TOP4_V1 &&
        dispatch.token_count == tokens && dispatch.expert_count == experts &&
        dispatch.pair_count == tokens * selected &&
        dispatch.fallback_allowed == 0U && dispatch.fallback_used == 0U;
    sllm_completion_result_t result{};
    result.struct_size = sizeof(result);
    result.abi_version = SLLM_HIP_ABI_VERSION;
    Error completion_error;
    const sllm_status_t status =
        test_case.query_first
            ? sllm_completion_query(completion, &result, &completion_error.sink)
            : sllm_completion_wait(completion, 1000U, &result,
                                   &completion_error.sink);
    valid =
        valid &&
        expect_status(status, test_case.expected_status,
                      "MiniMax route completion", completion_error) &&
        (test_case.fragment == nullptr ||
         std::strstr(completion_error.message, test_case.fragment) != nullptr);
    result = {};
    result.struct_size = sizeof(result);
    result.abi_version = SLLM_HIP_ABI_VERSION;
    Error cached_error;
    valid = valid &&
            expect_status(
                sllm_completion_query(completion, &result, &cached_error.sink),
                test_case.expected_status, "MiniMax cached completion",
                cached_error) &&
            release_completion(&completion) &&
            fake_hip::live_events() == baseline_events;
  }

  sllm_completion_t *deferred = nullptr;
  sllm_completion_t *fence = nullptr;
  if (valid) {
    Error mode_error;
    valid = expect_status(
        sllm_queue_set_completion_mode(
            queue, SLLM_QUEUE_COMPLETION_MODE_DEFERRED, &mode_error.sink),
        SLLM_STATUS_OK, "MiniMax deferred mode", mode_error);
  }
  if (valid) {
    sllm_test_minimax_m3_moe_route_device_status(
        SLLM_MINIMAX_M3_MOE_ROUTE_STATUS_ZERO_NORMALIZER);
    sllm_minimax_m3_moe_route_dispatch_info_t dispatch{};
    dispatch.struct_size = sizeof(dispatch);
    dispatch.abi_version = SLLM_HIP_ABI_VERSION;
    dispatch.info_version = SLLM_HIP_MINIMAX_M3_MOE_ROUTE_DISPATCH_INFO_VERSION;
    Error execute_error;
    valid = expect_status(
                sllm_minimax_m3_moe_route_execute(
                    plan, queue, &deferred, &dispatch, &execute_error.sink),
                SLLM_STATUS_OK, "MiniMax deferred execute", execute_error) &&
            fake_hip::live_events() == baseline_events;
    Error busy_error;
    valid =
        valid &&
        expect_status(
            sllm_minimax_m3_moe_route_plan_release(&plan, &busy_error.sink),
            SLLM_STATUS_PUBLIC_BUSY, "MiniMax in-flight plan", busy_error) &&
        plan != nullptr;
    Error fence_error;
    valid = valid && expect_status(
                         sllm_queue_fence(queue, &fence, &fence_error.sink),
                         SLLM_STATUS_OK, "MiniMax deferred fence", fence_error);
  }
  if (valid) {
    sllm_completion_result_t result{};
    result.struct_size = sizeof(result);
    result.abi_version = SLLM_HIP_ABI_VERSION;
    Error wait_error;
    valid = expect_status(
        sllm_completion_wait(fence, 1000U, &result, &wait_error.sink),
        SLLM_STATUS_OK, "MiniMax fence wait", wait_error);
    result = {};
    result.struct_size = sizeof(result);
    result.abi_version = SLLM_HIP_ABI_VERSION;
    Error finalize_error;
    valid =
        valid &&
        expect_status(sllm_completion_finalize_after(deferred, fence, &result,
                                                     &finalize_error.sink),
                      SLLM_STATUS_INVALID_ARGUMENT,
                      "MiniMax deferred semantic failure", finalize_error) &&
        std::strstr(finalize_error.message, "normalizer") != nullptr;
  }

  sllm_test_minimax_m3_moe_route_device_status(
      SLLM_MINIMAX_M3_MOE_ROUTE_STATUS_OK);
  Error release_error;
  const bool released =
      (deferred == nullptr || release_completion(&deferred)) &&
      (fence == nullptr || release_completion(&fence)) &&
      (plan == nullptr ||
       expect_status(
           sllm_minimax_m3_moe_route_plan_release(&plan, &release_error.sink),
           SLLM_STATUS_OK, "MiniMax route plan release", release_error)) &&
      release_buffer(&output) && release_buffer(&bias) &&
      release_buffer(&logits) && release_queue(&queue) &&
      release_context(&context);
  return valid && released && fake_hip::live_events() == baseline_events &&
         fake_hip::live_allocations() == 0U && fake_hip::live_streams() == 0U;
}

} // namespace

int main() {
  if (!minimax_m3_moe_route_public_contract()) {
    std::cerr << "MiniMax M3 MoE route public contract test failed\n";
    return 1;
  }
  if (!deepseek_v4_moe_route_device_status_completion_contract()) {
    std::cerr << "DeepSeek V4 MoE route status completion test failed\n";
    return 1;
  }
  if (!deepseek_v4_moe_route_descriptor_abi_and_lifetime_contract()) {
    std::cerr << "DeepSeek V4 MoE route descriptor/ABI contract test failed\n";
    return 1;
  }
  if (!mlp_gate_up_silu_bundle_abi_negative_contract()) {
    std::cerr << "MLP gate/up/SiLU bundle ABI contract test failed\n";
    return 1;
  }
  if (!rmsnorm_bf16_rne_bit_contract()) {
    return 1;
  }
  if (!elementwise_prepare_execute_and_negative_contract()) {
    std::cerr << "elementwise prepare/execute contract test failed\n";
    return 1;
  }
  if (!embedding_prepare_execute_and_token_range_contract()) {
    std::cerr << "embedding prepare/execute contract test failed\n";
    return 1;
  }
  if (!context_device_property_snapshot_contract()) {
    std::cerr << "context device-property snapshot test failed\n";
    return 1;
  }
  if (!matmul_prepare_execute_and_negative_contract()) {
    std::cerr << "matmul prepare/execute contract test failed\n";
    return 1;
  }
  if (!matmul_mxfp_weight_activation_descriptor_contract()) {
    std::cerr << "matmul MXFP descriptor contract test failed\n";
    return 1;
  }
  if (!matmul_mxfp_prefill_selector_contract()) {
    std::cerr << "matmul MXFP provider selector contract test failed\n";
    return 1;
  }
  if (!matmul_short_mixed_metadata_dispatch_contract()) {
    std::cerr << "matmul short-mixed metadata dispatch test failed\n";
    return 1;
  }
  if (!matmul_short_mixed_rocblas_solution_selector_contract()) {
    std::cerr << "matmul short-mixed rocBLAS selector test failed\n";
    return 1;
  }
  if (!matmul_async_lifetime_and_cleanup()) {
    std::cerr << "matmul async lifetime/cleanup test failed\n";
    return 1;
  }
  if (!gdn_projection_bundle_launch_failure_rolls_back_accounting() ||
      !mlp_gate_up_silu_bundle_launch_failure_rolls_back_accounting()) {
    std::cerr << "bundle launch-failure accounting test failed\n";
    return 1;
  }
  if (!attention_preprocess_prepare_validation_and_old_abi() ||
      !attention_preprocess_position_payload_mismatch_is_pre_dispatch() ||
      !attention_preprocess_derived_positions_skip_payload_validation() ||
      !attention_preprocess_success_metadata_and_dispatch() ||
      !attention_preprocess_mrope_positions_dispatch()) {
    std::cerr << "attention preprocess public ABI contract test failed\n";
    return 1;
  }
  if (!rotary_prepare_execute_lifetime_and_negative_contract()) {
    std::cerr << "rotary public ABI contract test failed\n";
    return 1;
  }
  if (!windowed_attention_prepare_execute_lifetime_and_negative_contract()) {
    std::cerr << "windowed attention public ABI contract test failed\n";
    return 1;
  }
  if (!causal_attention_gqa4_p32_selector_contract()) {
    std::cerr << "causal attention GQA4 P32 selector contract test failed\n";
    return 1;
  }
  if (!causal_attention_target_scoped_selector_contract()) {
    std::cerr
        << "causal attention target-scoped selector contract test failed\n";
    return 1;
  }
  if (!bounded_counter_cas_contention_is_fail_closed() ||
      !completion_safety_quarantine_is_bounded_and_fail_closed() ||
      !successful_completion_lifecycle() ||
      !d2h_staging_and_completion_read_is_byte_exact() ||
      !positive_completion_with_deferred_event_destroy_retains_dependencies()) {
    return 1;
  }
  if (!concurrent_pin_and_release() || !fatal_completion_is_quarantined() ||
      !registry_failure_destroys_or_orphans_before_rollback() ||
      !registry_exception_reaches_real_catch_before_rollback() ||
      !production_orphan_owner_grows_past_128() ||
      !rmsnorm_prepare_lifecycle_and_negative_contract() ||
      !rmsnorm_plan_accounting_failure_is_consumed_and_quarantined() ||
      !rmsnorm_guard_page_prefix_is_fail_closed() ||
      !rmsnorm_table_driven_negative_contract() ||
      !rmsnorm_prepare_required_shape_and_context_cases()) {
    return 1;
  }
  if (!rmsnorm_execute_metadata_and_reuse()) {
    std::cerr << "RMSNorm execute metadata/reuse test failed\n";
    return 1;
  }
  if (!residual_rmsnorm_prepare_execute_lifetime_contract()) {
    std::cerr << "residual RMSNorm prepare/execute lifetime test failed\n";
    return 1;
  }
  if (!rmsnorm_direct_scale_numerical_contract()) {
    std::cerr << "RMSNorm direct-scale numerical contract test failed\n";
    return 1;
  }
  if (!rmsnorm_execute_boundaries_and_failures()) {
    std::cerr << "RMSNorm execute boundary/failure test failed\n";
    return 1;
  }
  if (!rmsnorm_execute_exception_scope_guards_restore_plan_reuse()) {
    std::cerr << "RMSNorm execute exception-scope-guard test failed\n";
    return 1;
  }
  if (!rmsnorm_registered_exception_with_event_destroy_failure_is_quarantined()) {
    std::cerr << "RMSNorm registered-exception ambiguous-cleanup test failed\n";
    return 1;
  }
  if (!deferred_segment_uses_one_fence_event_and_finalizes_exactly()) {
    std::cerr << "deferred segment completion-mode test failed\n";
    return 1;
  }
  if (!rmsnorm_execute_row_limit_and_overflow()) {
    std::cerr << "RMSNorm execute row-limit test failed\n";
    return 1;
  }
  if (!rmsnorm_execute_flattens_rank_one_through_eight()) {
    std::cerr << "RMSNorm execute rank-flatten test failed\n";
    return 1;
  }
  if (!linear_attention_gfx942_wave64_column_selector_contract()) {
    std::cerr << "linear attention gfx942 wave64 column selector contract "
                 "test failed\n";
    return 1;
  }
#define SLLM_RUN_KV_CONTRACT(test_name)                                        \
  if (!(test_name)()) {                                                        \
    std::cerr << #test_name " failed\n";                                       \
    return 1;                                                                  \
  }
  SLLM_RUN_KV_CONTRACT(kv_append_accounting_multiplicity_contract)
  SLLM_RUN_KV_CONTRACT(causal_attention_numerical_gqa_and_lifetime_contract)
  SLLM_RUN_KV_CONTRACT(linear_attention_transaction_and_lifetime_contract)
  SLLM_RUN_KV_CONTRACT(kv_append_same_buffer_disjoint_lifecycle_contract)
  SLLM_RUN_KV_CONTRACT(kv_state_create_snapshot_contract)
  SLLM_RUN_KV_CONTRACT(kv_lowbit_create_query_and_recipe_contract)
  SLLM_RUN_KV_CONTRACT(kv_capability_selected_contiguous_resident_contract)
  SLLM_RUN_KV_CONTRACT(kv_evidence_readback_contract)
  SLLM_RUN_KV_CONTRACT(kv_append_layout_and_transaction_contract)
  SLLM_RUN_KV_CONTRACT(
      kv_vattention_page_boundary_and_idempotent_cancel_contract)
  SLLM_RUN_KV_CONTRACT(kv_vmm_append_transaction_failure_injection_contract)
  SLLM_RUN_KV_CONTRACT(kv_vmm_cow_transaction_failure_injection_contract)
  SLLM_RUN_KV_CONTRACT(kv_append_lifetime_alias_and_quarantine_contract)
  SLLM_RUN_KV_CONTRACT(state_fork_vmm_and_linear_image_contract)
  SLLM_RUN_KV_CONTRACT(sliding_static_fp8_ring_image_fork_and_scale_contract)
#undef SLLM_RUN_KV_CONTRACT
  std::cout << "production public runtime host fault test: PASS\n";
  return 0;
}
