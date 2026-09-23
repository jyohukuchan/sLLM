#ifndef SLLM_HIP_MATMUL_KERNEL_INTERNAL_COMPAT_HPP
#define SLLM_HIP_MATMUL_KERNEL_INTERNAL_COMPAT_HPP

// The low precision selector, identity, and geometry contract is owned by
// native/lowp.  Keep this include path for existing HIP callers while the
// runtime transitions to the lowp C API.
#include <lowp/detail/lowp_kernel_internal.hpp>

namespace sllm_matmul_kernel {

constexpr const char *kLogicalKernelId = "matmul.bf16_fp32.v1";
constexpr const char *kDeviceSymbol = "sllm_matmul_bf16_fp32_v1";
constexpr const char *kPrefillLogicalKernelId = "matmul.bf16_fp32.tiled16.v2";
constexpr const char *kPrefillDeviceSymbol = "sllm_matmul_bf16_fp32_tiled16_v2";
// Phase 83: gfx1030 large-prefill BF16 register tile.  The shape selector is
// kept separate from the model path so any BF16 [M,K] x [N,K] caller with the
// measured MTP dimensions can use the same provider contract.
constexpr const char *kBf16PrefillGfx1030_64x64LogicalKernelId =
    "matmul.bf16_fp32.prefill.gfx1030.64x64.k32_transposed.v1";
constexpr const char *kBf16PrefillGfx1030_64x64DeviceSymbol =
    "sllm_matmul_bf16_fp32_prefill_gfx1030_64x64_k32_transposed_v1";
static_assert(
    sizeof("matmul.bf16_fp32.prefill.gfx1030.64x64.k32_transposed.v1") <= 64U);
static_assert(
    sizeof("sllm_matmul_bf16_fp32_prefill_gfx1030_64x64_k32_transposed_v1") <=
    64U);
constexpr const char *kDecodeLogicalKernelId = "matmul.bf16_fp32.decode.v4";
constexpr const char *kDecodeDeviceSymbol = "sllm_matmul_bf16_fp32_decode_v4";
constexpr const char *kDecodeWave64LogicalKernelId =
    "matmul.bf16_fp32.decode.wave64.v1";
constexpr const char *kDecodeWave64DeviceSymbol =
    "sllm_matmul_bf16_fp32_decode_wave64_v1";
constexpr const char *kSerialRowsLogicalKernelId =
    "matmul.bf16_fp32.decode.serial_rows.v1";
constexpr const char *kSerialRowsDeviceSymbol =
    "sllm_matmul_bf16_fp32_decode_serial_rows_v1";
constexpr const char *kSerialRowsWave64LogicalKernelId =
    "matmul.bf16_fp32.decode.serial_rows.wave64.v1";
constexpr const char *kSerialRowsWave64DeviceSymbol =
    "sllm_matmul_bf16_fp32_decode_serial_rows_wave64_v1";
constexpr const char *kShortSerialLogicalKernelId =
    "matmul.bf16_fp32.prefill.short_serial.v1";
constexpr const char *kShortSerialDeviceSymbol =
    "sllm_matmul_bf16_fp32_prefill_short_serial_v1";
constexpr const char *kShortMixedLogicalKernelId =
    "matmul.bf16_fp32.prefill.short_mixed_bss.v2";
constexpr const char *kShortMixedDeviceSymbol = "hipblasGemmExBbsF32Output";
constexpr const char *kHipBlasLogicalKernelId = "matmul.hipblas.gemm_ex.v2";
constexpr const char *kHipBlasDeviceSymbol = "hipblasGemmEx";
constexpr int32_t kPhase49Gfx1030RocblasSolution445 = -445;
constexpr const char *kPhase49Gfx1030ShortMixedRocblasSolutionEnvironment =
    "SLLM_MATMUL_GFX1030_SHORT_MIXED_ROCBLAS_SOLUTION";
constexpr const char *kFp8NativeLogicalKernelId =
    "matmul.fp8.outer.hipblaslt.v1";
constexpr const char *kFp8NativeDeviceSymbol = "hipblasLtMatmul";
constexpr const char *kFp8OuterGfx1201Dot4LogicalKernelId =
    "matmul.fp8.outer.gfx1201.dot4.v1";
constexpr const char *kFp8OuterGfx1201Dot4DeviceSymbol =
    "sllm_matmul_fp8_outer_gfx1201_dot4_v1";

hipError_t launch_fp8_outer_gfx1201_dot4(
    const uint8_t *activation, const float *activation_scales,
    const uint8_t *weight, const float *weight_scales, uint16_t *output,
    uint64_t m, uint64_t k, uint64_t n, hipStream_t stream) noexcept;

// WU2 measures these operation shapes, independent of model identity. Keep
// MTP verification rows and unmeasured shapes on their existing providers.
constexpr bool fp8_outer_gfx1201_dot4_shape(const uint64_t m, const uint64_t k,
                                            const uint64_t n) noexcept {
  return m == 1U && ((k == 5120U &&
                      (n == 1024U || n == 6144U || n == 10240U || n == 12288U ||
                       n == 17408U || n == 98304U || n == 248320U)) ||
                     (k == 6144U && n == 5120U) || (k == 17408U && n == 5120U));
}

// Matmul provider IDs owned by this runtime.  They share the numeric audit ID
// space with native/lowp, which reserves these values; value 1 is lowp's
// "no low-precision specialization" result and is this runtime's BF16
// reference kernel.
struct HostKernelVariant final {
  static constexpr KernelVariant Baseline = KernelVariant::Unspecialized;
  static constexpr KernelVariant PrefillTiled16 =
      static_cast<KernelVariant>(2U);
  static constexpr KernelVariant DecodeReduction =
      static_cast<KernelVariant>(3U);
  static constexpr KernelVariant HipBlas = static_cast<KernelVariant>(4U);
  static constexpr KernelVariant Fp8Native = static_cast<KernelVariant>(5U);
  static constexpr KernelVariant Fp8OuterGfx1201Dot4 =
      static_cast<KernelVariant>(103U);
  static constexpr KernelVariant DecodeReductionWave64 =
      static_cast<KernelVariant>(7U);
  static constexpr KernelVariant SerialRowsReduction =
      static_cast<KernelVariant>(12U);
  static constexpr KernelVariant SerialRowsReductionWave64 =
      static_cast<KernelVariant>(13U);
  static constexpr KernelVariant PrefillShortSerial =
      static_cast<KernelVariant>(16U);
  static constexpr KernelVariant PrefillShortMixed =
      static_cast<KernelVariant>(17U);
  static constexpr KernelVariant Bf16PrefillGfx1030_64x64 =
      static_cast<KernelVariant>(91U);
};

constexpr const char *logical_kernel_id(const KernelVariant variant) noexcept {
  return variant == HostKernelVariant::Bf16PrefillGfx1030_64x64
             ? kBf16PrefillGfx1030_64x64LogicalKernelId
         : variant == HostKernelVariant::Fp8OuterGfx1201Dot4
             ? kFp8OuterGfx1201Dot4LogicalKernelId
         : variant == HostKernelVariant::Fp8Native ? kFp8NativeLogicalKernelId
         : variant == HostKernelVariant::PrefillShortSerial
             ? kShortSerialLogicalKernelId
         : variant == HostKernelVariant::PrefillShortMixed
             ? kShortMixedLogicalKernelId
         : variant == HostKernelVariant::HipBlas ? kHipBlasLogicalKernelId
         : variant == HostKernelVariant::DecodeReductionWave64
             ? kDecodeWave64LogicalKernelId
         : variant == HostKernelVariant::SerialRowsReductionWave64
             ? kSerialRowsWave64LogicalKernelId
         : variant == HostKernelVariant::SerialRowsReduction
             ? kSerialRowsLogicalKernelId
         : variant == HostKernelVariant::DecodeReduction
             ? kDecodeLogicalKernelId
         : variant == HostKernelVariant::PrefillTiled16
             ? kPrefillLogicalKernelId
         : lowp_logical_kernel_id(variant) != nullptr
             ? lowp_logical_kernel_id(variant)
             : kLogicalKernelId;
}

constexpr const char *device_symbol(const KernelVariant variant) noexcept {
  return variant == HostKernelVariant::Bf16PrefillGfx1030_64x64
             ? kBf16PrefillGfx1030_64x64DeviceSymbol
         : variant == HostKernelVariant::Fp8OuterGfx1201Dot4
             ? kFp8OuterGfx1201Dot4DeviceSymbol
         : variant == HostKernelVariant::Fp8Native ? kFp8NativeDeviceSymbol
         : variant == HostKernelVariant::PrefillShortSerial
             ? kShortSerialDeviceSymbol
         : variant == HostKernelVariant::PrefillShortMixed
             ? kShortMixedDeviceSymbol
         : variant == HostKernelVariant::HipBlas ? kHipBlasDeviceSymbol
         : variant == HostKernelVariant::DecodeReductionWave64
             ? kDecodeWave64DeviceSymbol
         : variant == HostKernelVariant::SerialRowsReductionWave64
             ? kSerialRowsWave64DeviceSymbol
         : variant == HostKernelVariant::SerialRowsReduction
             ? kSerialRowsDeviceSymbol
         : variant == HostKernelVariant::DecodeReduction ? kDecodeDeviceSymbol
         : variant == HostKernelVariant::PrefillTiled16  ? kPrefillDeviceSymbol
         : lowp_device_symbol(variant) != nullptr ? lowp_device_symbol(variant)
                                                  : kDeviceSymbol;
}

inline const char *
logical_kernel_id_for_target(const KernelVariant variant,
                             const char *const target) noexcept {
  (void)target;
  return logical_kernel_id(variant);
}

inline const char *device_symbol_for_target(const KernelVariant variant,
                                            const char *const target) noexcept {
  const char *const symbol = lowp_device_symbol_for_target(variant, target);
  return symbol != nullptr ? symbol : device_symbol(variant);
}

inline const char *device_symbol_for_target(const KernelVariant variant,
                                            const char *const target,
                                            const uint64_t m, const uint64_t k,
                                            const uint64_t n) noexcept {
  const char *const symbol =
      lowp_device_symbol_for_target(variant, target, m, k, n);
  return symbol != nullptr ? symbol : device_symbol(variant);
}

constexpr uint32_t grid_size_x(const KernelVariant variant, const uint64_t m,
                               const uint64_t n,
                               const uint64_t k = 0U) noexcept {
  return variant == HostKernelVariant::Fp8OuterGfx1201Dot4
             ? static_cast<uint32_t>((n + 7U) / 8U)
         : variant == HostKernelVariant::Fp8Native ? static_cast<uint32_t>(n)
         : variant == HostKernelVariant::Bf16PrefillGfx1030_64x64
             ? static_cast<uint32_t>(((m + 63U) / 64U) * ((n + 63U) / 64U))
         : variant == HostKernelVariant::PrefillShortSerial
             ? static_cast<uint32_t>(((m + 7U) / 8U) * n)
         : (variant == HostKernelVariant::PrefillShortMixed ||
            variant == HostKernelVariant::HipBlas ||
            variant == HostKernelVariant::DecodeReductionWave64 ||
            variant == HostKernelVariant::SerialRowsReductionWave64 ||
            variant == HostKernelVariant::SerialRowsReduction ||
            variant == HostKernelVariant::DecodeReduction)
             ? static_cast<uint32_t>(n)
         : variant == HostKernelVariant::PrefillTiled16
             ? static_cast<uint32_t>((n + 15U) / 16U)
             : lowp_grid_size_x(variant, m, n, k);
}

static_assert(lowp_logical_kernel_id(HostKernelVariant::Baseline) == nullptr);
static_assert(lowp_logical_kernel_id(HostKernelVariant::PrefillTiled16) ==
              nullptr);
static_assert(lowp_logical_kernel_id(HostKernelVariant::DecodeReduction) ==
              nullptr);
static_assert(lowp_logical_kernel_id(HostKernelVariant::HipBlas) == nullptr);
static_assert(lowp_logical_kernel_id(HostKernelVariant::Fp8Native) == nullptr);
static_assert(lowp_logical_kernel_id(HostKernelVariant::Fp8OuterGfx1201Dot4) ==
              nullptr);
static_assert(workgroup_size_x(HostKernelVariant::Fp8OuterGfx1201Dot4) == 256U);
static_assert(grid_size_x(HostKernelVariant::Fp8OuterGfx1201Dot4, 1U, 10240U) ==
              1280U);
static_assert(lowp_logical_kernel_id(
                  HostKernelVariant::DecodeReductionWave64) == nullptr);
static_assert(lowp_logical_kernel_id(HostKernelVariant::SerialRowsReduction) ==
              nullptr);
static_assert(lowp_logical_kernel_id(
                  HostKernelVariant::SerialRowsReductionWave64) == nullptr);
static_assert(lowp_logical_kernel_id(HostKernelVariant::PrefillShortSerial) ==
              nullptr);
static_assert(lowp_logical_kernel_id(HostKernelVariant::PrefillShortMixed) ==
              nullptr);
static_assert(lowp_logical_kernel_id(
                  HostKernelVariant::Bf16PrefillGfx1030_64x64) == nullptr);
static_assert(grid_size_x(HostKernelVariant::Bf16PrefillGfx1030_64x64, 64U,
                          17408U) == 272U);
static_assert(grid_size_x(HostKernelVariant::Bf16PrefillGfx1030_64x64, 65U,
                          17408U) == 544U);

constexpr bool phase34_gfx1030_hipblas_shape(const uint64_t m, const uint64_t k,
                                             const uint64_t n) noexcept {
  const bool main_projection =
      (k == 2560U && (n == 9216U || n == 8192U || n == 4096U)) ||
      (k == 9216U && n == 2560U) || (k == 4096U && n == 2560U);
  if (main_projection) {
    return m >= 128U;
  }
  // full-attention K/V is too small to amortize the provider until a larger
  // row block. The GDN b/a N=32 projection remains on tiled16: its measured
  // crossover was unstable and its weighted absolute contribution is small.
  return k == 2560U && n == 1024U && m >= 1024U;
}

// ROCm 7.14 exposes a faster Tensile solution for the same BF16/F32 GEMM
// contract through rocblas_gemm_algo_solution_index. Keep this accepted
// default candidate constrained to the already-adopted Phase 34 gfx1030 shape
// set; an explicit environment value of 0 (or any unknown value) rolls back
// to the ordinary hipBLAS path, as do all other targets and shapes.
constexpr bool
phase49_gfx1030_rocblas_solution_445_shape(const uint64_t m, const uint64_t k,
                                           const uint64_t n) noexcept {
  return phase34_gfx1030_hipblas_shape(m, k, n);
}

static_assert(!phase34_gfx1030_hipblas_shape(127U, 2560U, 9216U));
static_assert(phase34_gfx1030_hipblas_shape(128U, 2560U, 9216U));
static_assert(phase34_gfx1030_hipblas_shape(129U, 4096U, 2560U));
static_assert(!phase34_gfx1030_hipblas_shape(1023U, 2560U, 1024U));
static_assert(phase34_gfx1030_hipblas_shape(1024U, 2560U, 1024U));
static_assert(!phase34_gfx1030_hipblas_shape(10001U, 2560U, 32U));
static_assert(!phase34_gfx1030_hipblas_shape(10001U, 2560U, 248320U));
static_assert(!phase49_gfx1030_rocblas_solution_445_shape(127U, 2560U, 9216U));
static_assert(phase49_gfx1030_rocblas_solution_445_shape(128U, 2560U, 9216U));
static_assert(phase49_gfx1030_rocblas_solution_445_shape(10001U, 2560U, 9216U));
static_assert(!phase49_gfx1030_rocblas_solution_445_shape(10001U, 2560U, 32U));
static_assert(!phase49_gfx1030_rocblas_solution_445_shape(10001U, 2560U,
                                                          248320U));

// The short-serial rollback provider covers the five dense Qwen projection
// shapes that are known to benefit from row grouping at small prefill M.
constexpr bool phase49_gfx1030_short_serial_shape(const uint64_t m,
                                                  const uint64_t k,
                                                  const uint64_t n) noexcept {
  const bool main_projection =
      (k == 2560U && (n == 9216U || n == 8192U || n == 4096U)) ||
      (k == 9216U && n == 2560U) || (k == 4096U && n == 2560U);
  return m >= 9U && m <= 63U && main_projection;
}

static_assert(!phase49_gfx1030_short_serial_shape(8U, 2560U, 9216U));
static_assert(phase49_gfx1030_short_serial_shape(9U, 2560U, 9216U));
static_assert(phase49_gfx1030_short_serial_shape(17U, 2560U, 8192U));
static_assert(phase49_gfx1030_short_serial_shape(31U, 2560U, 4096U));
static_assert(phase49_gfx1030_short_serial_shape(33U, 9216U, 2560U));
static_assert(phase49_gfx1030_short_serial_shape(63U, 4096U, 2560U));
static_assert(!phase49_gfx1030_short_serial_shape(64U, 4096U, 2560U));
static_assert(!phase49_gfx1030_short_serial_shape(17U, 2560U, 1024U));
static_assert(!phase49_gfx1030_short_serial_shape(17U, 2560U, 248320U));

constexpr bool phase49_gfx1030_short_mixed_shape(const uint64_t m,
                                                 const uint64_t k,
                                                 const uint64_t n) noexcept {
  const bool qwen_projection =
      (k == 2560U && (n == 32U || n == 1024U || n == 4096U || n == 8192U ||
                      n == 9216U || n == 248320U)) ||
      (k == 4096U && n == 2560U) || (k == 9216U && n == 2560U);
  return m >= 9U && m <= 63U && qwen_projection;
}

// Phase 49 short-mixed solution table.  The table is deliberately narrower
// than the default short-mixed provider: M=17 keeps its per-shape fastest
// choices, while M=32 uses the -473 candidate uniformly because it is the
// measured exact-output solution across all non-vocabulary shapes.  The
// vocabulary head (N=248320) stays on the hipBLAS baseline.  A zero return
// means baseline.  The companion environment is default-on when unset or
// exactly "1"; exactly "0" and unknown values disable the candidate.
constexpr int32_t
phase49_gfx1030_short_mixed_rocblas_solution(const uint64_t m, const uint64_t k,
                                             const uint64_t n) noexcept {
  if (m != 17U && m != 32U) {
    return 0;
  }
  if (m == 32U) {
    if ((k == 2560U &&
         (n == 32U || n == 1024U || n == 4096U || n == 8192U || n == 9216U)) ||
        ((k == 4096U || k == 9216U) && n == 2560U)) {
      return -473;
    }
    return 0;
  }
  if (k == 2560U) {
    if (n == 9216U || n == 8192U || n == 32U) {
      return -473;
    }
    if (n == 4096U) {
      return -472;
    }
    if (n == 1024U && m == 17U) {
      return -473;
    }
    return 0;
  }
  if ((k == 4096U || k == 9216U) && n == 2560U) {
    return -472;
  }
  return 0;
}

inline bool
phase49_gfx1030_short_mixed_rocblas_enabled(const char *const target,
                                            const uint64_t m, const uint64_t k,
                                            const uint64_t n) noexcept {
  const char *const force_baseline = std::getenv("SLLM_MATMUL_FORCE_BASELINE");
  if (force_baseline != nullptr && std::strcmp(force_baseline, "1") == 0) {
    return false;
  }
  const char *const environment =
      std::getenv(kPhase49Gfx1030ShortMixedRocblasSolutionEnvironment);
  const bool enabled =
      environment == nullptr || std::strcmp(environment, "1") == 0;
  return target_is(target, "gfx1030") && enabled &&
         phase49_gfx1030_short_mixed_rocblas_solution(m, k, n) != 0;
}

constexpr bool
phase49_gfx1030_mixed_workspace_bytes(const uint64_t m, const uint64_t n,
                                      uint64_t *const bytes) noexcept {
  if (bytes == nullptr || m == 0U || n == 0U || m > UINT64_MAX / n) {
    return false;
  }
  const uint64_t elements = m * n;
  if (elements > UINT64_MAX / UINT64_C(4)) {
    return false;
  }
  *bytes = elements * UINT64_C(4);
  return true;
}

static_assert(!phase49_gfx1030_short_mixed_shape(8U, 2560U, 32U));
static_assert(phase49_gfx1030_short_mixed_shape(9U, 2560U, 32U));
static_assert(phase49_gfx1030_short_mixed_shape(17U, 2560U, 1024U));
static_assert(phase49_gfx1030_short_mixed_shape(32U, 2560U, 248320U));
static_assert(phase49_gfx1030_short_mixed_shape(63U, 4096U, 2560U));
static_assert(!phase49_gfx1030_short_mixed_shape(64U, 2560U, 9216U));
static_assert(!phase49_gfx1030_short_mixed_shape(17U, 2560U, 33U));
static_assert(phase49_gfx1030_short_mixed_rocblas_solution(17U, 2560U, 9216U) ==
              -473);
static_assert(phase49_gfx1030_short_mixed_rocblas_solution(17U, 2560U, 1024U) ==
              -473);
static_assert(phase49_gfx1030_short_mixed_rocblas_solution(32U, 2560U, 1024U) ==
              -473);
static_assert(phase49_gfx1030_short_mixed_rocblas_solution(17U, 2560U,
                                                           248320U) == 0);
static_assert(phase49_gfx1030_short_mixed_rocblas_solution(16U, 2560U, 9216U) ==
              0);

// Phase 83 BF16 prefill provider. These are the three real MTP companion
// matrix orientations, with M>=64 as the measured large-tile crossover. The
// M dimension is intentionally a threshold rather than an alignment
// requirement: the kernel has guarded tails and the M=63/65 boundary is part
// of its numerical contract.
constexpr bool
phase83_gfx1030_bf16_prefill_64x64_shape(const uint64_t m, const uint64_t k,
                                         const uint64_t n) noexcept {
  return m >= 64U && ((k == UINT64_C(10240) && n == UINT64_C(5120)) ||
                      (k == UINT64_C(5120) && n == UINT64_C(17408)) ||
                      (k == UINT64_C(17408) && n == UINT64_C(5120)));
}

static_assert(!phase83_gfx1030_bf16_prefill_64x64_shape(63U, 5120U, 17408U));
static_assert(phase83_gfx1030_bf16_prefill_64x64_shape(64U, 5120U, 17408U));
static_assert(phase83_gfx1030_bf16_prefill_64x64_shape(65U, 5120U, 17408U));
static_assert(phase83_gfx1030_bf16_prefill_64x64_shape(1024U, 10240U, 5120U));
static_assert(phase83_gfx1030_bf16_prefill_64x64_shape(1024U, 17408U, 5120U));
static_assert(!phase83_gfx1030_bf16_prefill_64x64_shape(64U, 5120U, 17407U));
static_assert(!phase83_gfx1030_bf16_prefill_64x64_shape(64U, 4096U, 2560U));

inline KernelVariant select_variant(const uint64_t m, const uint64_t k,
                                    const uint64_t n,
                                    const char *const target) noexcept {
  const char *const force_baseline = std::getenv("SLLM_MATMUL_FORCE_BASELINE");
  if (force_baseline != nullptr && std::strcmp(force_baseline, "1") == 0) {
    return HostKernelVariant::Baseline;
  }
  // Speculative target verification uses a small logical row block. Keep each
  // row's dot-product arithmetic identical to canonical M=1 decode; only the
  // independent row/column workgroups are grouped into one submission.
  if (m > 1U && m <= 8U) {
    return target_is(target, "gfx942")
               ? HostKernelVariant::SerialRowsReductionWave64
               : HostKernelVariant::SerialRowsReduction;
  }
  const char *const disable_short_serial =
      std::getenv("SLLM_MATMUL_GFX1030_SHORT_SERIAL");
  const char *const disable_short_mixed =
      std::getenv("SLLM_MATMUL_GFX1030_SHORT_MIXED");
  if (target_is(target, "gfx1030") &&
      !(disable_short_mixed != nullptr &&
        std::strcmp(disable_short_mixed, "0") == 0) &&
      phase49_gfx1030_short_mixed_shape(m, k, n)) {
    return HostKernelVariant::PrefillShortMixed;
  }
  if (target_is(target, "gfx1030") &&
      !(disable_short_serial != nullptr &&
        std::strcmp(disable_short_serial, "0") == 0) &&
      phase49_gfx1030_short_serial_shape(m, k, n)) {
    return HostKernelVariant::PrefillShortSerial;
  }
  if (target_is(target, "gfx1030") &&
      phase83_gfx1030_bf16_prefill_64x64_shape(m, k, n)) {
    return HostKernelVariant::Bf16PrefillGfx1030_64x64;
  }
  if (m > 1U && (target_is(target, "gfx1201") || target_is(target, "gfx942"))) {
    return HostKernelVariant::HipBlas;
  }
  if (target_is(target, "gfx1030") && phase34_gfx1030_hipblas_shape(m, k, n)) {
    return HostKernelVariant::HipBlas;
  }
  return m == 1U ? (target_is(target, "gfx942")
                        ? HostKernelVariant::DecodeReductionWave64
                        : HostKernelVariant::DecodeReduction)
                 : HostKernelVariant::PrefillTiled16;
}

// Preserve the historical sLLM FP8 outer selector at the HIP boundary.  The
// standalone lowp library owns only the software gfx1030 selector; native
// gfx1201/gfx942 FP8 remains an sLLM HIP/hipBLASLt decision.
inline KernelVariant
select_fp8_outer_variant(const uint64_t m, const uint64_t k, const uint64_t n,
                         const char *const target,
                         const bool fnuz = false) noexcept {
  if (target_is(target, "gfx1201") && !fnuz &&
      fp8_outer_gfx1201_dot4_shape(m, k, n)) {
    return HostKernelVariant::Fp8OuterGfx1201Dot4;
  }
  if (target_is(target, "gfx1201") || target_is(target, "gfx942")) {
    return HostKernelVariant::Fp8Native;
  }
  return select_fp8_software_variant(m, k, n, target, fnuz);
}

inline SelectorDecision
select_fp8_outer_decision(const uint64_t m, const uint64_t k, const uint64_t n,
                          const char *const target,
                          const bool fnuz = false) noexcept {
  if (target_is(target, "gfx1201") || target_is(target, "gfx942")) {
    return make_selector_decision(
        select_fp8_outer_variant(m, k, n, target, fnuz), true, true, true,
        kSelectorReasonAdopted);
  }
  return select_fp8_software_decision(m, k, n, target, fnuz);
}

} // namespace sllm_matmul_kernel

#endif // SLLM_HIP_MATMUL_KERNEL_INTERNAL_COMPAT_HPP
