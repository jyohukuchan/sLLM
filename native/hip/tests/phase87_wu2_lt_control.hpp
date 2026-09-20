#ifndef SLLM_PHASE87_WU2_LT_CONTROL_HPP
#define SLLM_PHASE87_WU2_LT_CONTROL_HPP

// Phase 87 WU2 hipBLASLt control wrapper.
//
// This header mirrors the production FP8 outer-vector descriptor and
// selection policy in native/hip/src/public_runtime.hip.cpp.  It is test-only
// plumbing: the wrapper owns its handle/descriptors and never changes the
// production rank or algorithm tables.

#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace phase87_wu2 {

class LtControl final {
public:
  // These fields describe the selected heuristic.  rank is the returned
  // heuristic position (therefore zero for the production fallback); index is
  // the hipBLASLt algorithm index of that selected result.
  int rank = 0;
  int32_t index = 0;
  std::string kernel_name;
  bool zero_workspace = false;

  LtControl(const uint64_t m, const uint64_t k, const uint64_t n,
            const float *const weight_scales, float *const activation_scales)
      : m_(m), k_(k), n_(n), weight_scales_(weight_scales),
        activation_scales_(activation_scales) {
    try {
      initialize();
    } catch (...) {
      cleanup();
      throw;
    }
  }

  LtControl(const LtControl &) = delete;
  LtControl &operator=(const LtControl &) = delete;
  LtControl(LtControl &&) = delete;
  LtControl &operator=(LtControl &&) = delete;

  ~LtControl() { cleanup(); }

  hipblasStatus_t launch(const uint8_t *const activation,
                         const uint8_t *const weight, uint16_t *const output,
                         const hipStream_t stream) const noexcept {
    if (activation == nullptr || weight == nullptr || output == nullptr ||
        handle_ == nullptr || operation_ == nullptr || a_ == nullptr ||
        b_ == nullptr || c_ == nullptr || d_ == nullptr || !zero_workspace) {
      return HIPBLAS_STATUS_INVALID_VALUE;
    }
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    return hipblasLtMatmul(handle_, operation_, &alpha, weight, a_, activation,
                           b_, &beta, output, c_, output, d_, &algorithm_,
                           nullptr, 0U, stream);
  }

private:
  static constexpr int32_t kPinnedVersion = 100401;
  static constexpr char kPinnedRevision[] = "cd957402";

  hipblasLtHandle_t handle_ = nullptr;
  hipblasLtMatmulDesc_t operation_ = nullptr;
  hipblasLtMatrixLayout_t a_ = nullptr;
  hipblasLtMatrixLayout_t b_ = nullptr;
  hipblasLtMatrixLayout_t c_ = nullptr;
  hipblasLtMatrixLayout_t d_ = nullptr;
  hipblasLtMatmulPreference_t preference_ = nullptr;
  hipblasLtMatmulAlgo_t algorithm_{};
  uint64_t m_;
  uint64_t k_;
  uint64_t n_;
  const float *weight_scales_;
  float *activation_scales_;

  static void require(const hipblasStatus_t status, const char *const name) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
      throw std::runtime_error(std::string("phase87 WU2 ") + name +
                               " failed: hipBLAS status " +
                               std::to_string(static_cast<int>(status)));
    }
  }

  static bool pinned_identity(const hipblasLtHandle_t handle) {
    int32_t version = 0;
    std::array<char, 64> revision{};
    require(hipblasLtGetVersion(handle, &version), "hipblasLtGetVersion");
    require(hipblasLtGetGitRevision(handle, revision.data()),
            "hipblasLtGetGitRevision");
    return version == kPinnedVersion &&
           std::string(revision.data()) == kPinnedRevision;
  }

  static int m1_rank(const uint64_t k, const uint64_t n) noexcept {
    if (k == 5120U && n == 10240U) {
      return 7;
    }
    if ((k == 5120U && n == 6144U) || (k == 6144U && n == 5120U) ||
        (k == 5120U && n == 17408U)) {
      return 8;
    }
    if ((k == 5120U && n == 12288U) || (k == 5120U && n == 1024U) ||
        (k == 17408U && n == 5120U)) {
      return 9;
    }
    return 0;
  }

  static bool m1_ranked_shape(const uint64_t k, const uint64_t n) noexcept {
    return (k == 5120U && n == 10240U) || (k == 5120U && n == 6144U) ||
           (k == 6144U && n == 5120U) || (k == 5120U && n == 12288U) ||
           (k == 5120U && n == 1024U) || (k == 5120U && n == 17408U) ||
           (k == 17408U && n == 5120U);
  }

  static int32_t m23_algorithm_index(const uint64_t k,
                                     const uint64_t n) noexcept {
    if ((k == 5120U && n == 6144U) || (k == 5120U && n == 1024U) ||
        (k == 6144U && n == 5120U)) {
      return 123373;
    }
    if ((k == 5120U && n == 10240U) || (k == 5120U && n == 12288U) ||
        (k == 17408U && n == 5120U)) {
      return 123374;
    }
    if (k == 5120U && n == 17408U) {
      return 123375;
    }
    return 0;
  }

  void initialize() {
    if (m_ == 0U || k_ == 0U || n_ == 0U || weight_scales_ == nullptr ||
        activation_scales_ == nullptr) {
      throw std::invalid_argument("phase87 WU2 invalid LtControl arguments");
    }

    require(hipblasLtCreate(&handle_), "hipblasLtCreate");
    require(
        hipblasLtMatmulDescCreate(&operation_, HIPBLAS_COMPUTE_32F, HIP_R_32F),
        "hipblasLtMatmulDescCreate");

    const hipblasOperation_t trans_a = HIPBLAS_OP_T;
    const hipblasOperation_t trans_b = HIPBLAS_OP_N;
    require(hipblasLtMatmulDescSetAttribute(operation_,
                                            HIPBLASLT_MATMUL_DESC_TRANSA,
                                            &trans_a, sizeof(trans_a)),
            "set TRANSA");
    require(hipblasLtMatmulDescSetAttribute(operation_,
                                            HIPBLASLT_MATMUL_DESC_TRANSB,
                                            &trans_b, sizeof(trans_b)),
            "set TRANSB");

    void *weight_scale_pointer = const_cast<float *>(weight_scales_);
    void *activation_scale_pointer = activation_scales_;
    const hipblasLtMatmulMatrixScale_t scale_mode =
        HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    require(hipblasLtMatmulDescSetAttribute(
                operation_, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                &weight_scale_pointer, sizeof(weight_scale_pointer)),
            "set A scale pointer");
    require(hipblasLtMatmulDescSetAttribute(
                operation_, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                &activation_scale_pointer, sizeof(activation_scale_pointer)),
            "set B scale pointer");
    require(hipblasLtMatmulDescSetAttribute(operation_,
                                            HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                            &scale_mode, sizeof(scale_mode)),
            "set A scale mode");
    require(hipblasLtMatmulDescSetAttribute(operation_,
                                            HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                            &scale_mode, sizeof(scale_mode)),
            "set B scale mode");

    // A is the [N,K] weight plane presented as KxN then transposed. B is the
    // [M,K] activation plane presented as KxM. C/D are the transposed NxM
    // view over the row-major [M,N] output buffer.
    require(hipblasLtMatrixLayoutCreate(&a_, HIP_R_8F_E4M3, k_, n_,
                                        static_cast<int64_t>(k_)),
            "create A layout");
    require(hipblasLtMatrixLayoutCreate(&b_, HIP_R_8F_E4M3, k_, m_,
                                        static_cast<int64_t>(k_)),
            "create B layout");
    require(hipblasLtMatrixLayoutCreate(&c_, HIP_R_16BF, n_, m_,
                                        static_cast<int64_t>(n_)),
            "create C layout");
    require(hipblasLtMatrixLayoutCreate(&d_, HIP_R_16BF, n_, m_,
                                        static_cast<int64_t>(n_)),
            "create D layout");
    require(hipblasLtMatmulPreferenceCreate(&preference_),
            "hipblasLtMatmulPreferenceCreate");
    const uint64_t workspace_limit = 0U;
    require(hipblasLtMatmulPreferenceSetAttribute(
                preference_, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspace_limit, sizeof(workspace_limit)),
            "set zero workspace limit");

    const bool ranked_m1 = m_ == 1U && m1_ranked_shape(k_, n_);
    const int32_t requested_index =
        (m_ >= 2U && m_ <= 3U) ? m23_algorithm_index(k_, n_) : 0;
    const bool indexed_m23 = requested_index != 0 && pinned_identity(handle_);
    const int requested_count = ranked_m1 || indexed_m23 ? 32 : 1;
    std::array<hipblasLtMatmulHeuristicResult_t, 32> heuristics{};
    int solution_count = 0;
    require(hipblasLtMatmulAlgoGetHeuristic(handle_, operation_, a_, b_, c_, d_,
                                            preference_, requested_count,
                                            heuristics.data(), &solution_count),
            "hipblasLtMatmulAlgoGetHeuristic");
    if (solution_count <= 0) {
      throw std::runtime_error("phase87 WU2 hipBLASLt returned no solution");
    }

    int selected = 0;
    if (ranked_m1) {
      selected = m1_rank(k_, n_);
      if (selected >= solution_count) {
        throw std::runtime_error("phase87 WU2 M1 rank is unavailable");
      }
    } else if (indexed_m23) {
      for (int candidate = 0; candidate < solution_count; ++candidate) {
        const auto &heuristic = heuristics[static_cast<std::size_t>(candidate)];
        hipblasLtMatmulAlgo_t candidate_algo = heuristic.algo;
        if (heuristic.state == HIPBLAS_STATUS_SUCCESS &&
            heuristic.workspaceSize == 0U &&
            hipblaslt_ext::getIndexFromAlgo(candidate_algo) ==
                requested_index) {
          selected = candidate;
          break;
        }
      }
      // Production falls back to the rank-zero result when the pinned index
      // is absent or unusable.
      if (selected == 0) {
        rank = 0;
      }
    }

    const auto &chosen = heuristics[static_cast<std::size_t>(selected)];
    if (chosen.state != HIPBLAS_STATUS_SUCCESS || chosen.workspaceSize != 0U) {
      throw std::runtime_error(
          "phase87 WU2 selected hipBLASLt solution is not zero-workspace");
    }
    rank = selected;
    algorithm_ = chosen.algo;
    index = hipblaslt_ext::getIndexFromAlgo(algorithm_);
    kernel_name = hipblaslt_ext::getKernelNameFromAlgo(handle_, algorithm_);
    zero_workspace = chosen.workspaceSize == 0U;
  }

  void cleanup() noexcept {
    if (preference_ != nullptr) {
      (void)hipblasLtMatmulPreferenceDestroy(preference_);
      preference_ = nullptr;
    }
    if (d_ != nullptr) {
      (void)hipblasLtMatrixLayoutDestroy(d_);
      d_ = nullptr;
    }
    if (c_ != nullptr) {
      (void)hipblasLtMatrixLayoutDestroy(c_);
      c_ = nullptr;
    }
    if (b_ != nullptr) {
      (void)hipblasLtMatrixLayoutDestroy(b_);
      b_ = nullptr;
    }
    if (a_ != nullptr) {
      (void)hipblasLtMatrixLayoutDestroy(a_);
      a_ = nullptr;
    }
    if (operation_ != nullptr) {
      (void)hipblasLtMatmulDescDestroy(operation_);
      operation_ = nullptr;
    }
    if (handle_ != nullptr) {
      (void)hipblasLtDestroy(handle_);
      handle_ = nullptr;
    }
  }
};

} // namespace phase87_wu2

#endif // SLLM_PHASE87_WU2_LT_CONTROL_HPP
