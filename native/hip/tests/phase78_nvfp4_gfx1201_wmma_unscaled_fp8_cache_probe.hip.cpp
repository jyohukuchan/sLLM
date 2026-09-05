// Phase 78 standalone gfx1201 NVFP4 unscaled-FP8 cache probe.
//
// The production ID64 control expands packed E2M1 values while filling each
// LDS tile.  That repeats activation expansion once per N=64 workgroup and
// weight expansion once per M=128 workgroup.  This probe moves either or both
// exact E2M1 -> E4M3 byte conversions into explicit kernels outside the timed
// GEMM.  Crucially, it does not fold the E4M3 block scales into either FP8
// operand: every K=16 WMMA contribution is scaled and accumulated in exactly
// the same order as ID64, and tensor-global scaling remains in the BF16-RNE
// epilogue.
//
// The existing standalone scaled-ingress probe owns a byte-for-byte ID64
// control plus the format/oracle helpers.  Embedding it keeps this experiment
// tied to that frozen control while this file contributes no production ABI or
// selector changes.

#define main phase78_scaled_ingress_embedded_main
#include "phase78_nvfp4_gfx1201_wmma_scaled_ingress_probe.hip.cpp"
#undef main

#include <functional>
#include <numeric>
#include <string>

namespace {

constexpr uint32_t kCacheThreads = 256U;
constexpr uint32_t kCacheWarmups = 3U;
constexpr uint32_t kCacheMeasured = 10U;
constexpr uint32_t kRequestChunks = 10U;
constexpr uint64_t kMaximumLdsBytes = UINT64_C(64) * 1024U;
constexpr int kMaximumVgpr = 128;

static_assert(kCacheWarmups == kWarmups);
static_assert(kCacheMeasured == kMeasured);

constexpr std::array<Shape, 4> kBoundaryShapes = {{
    {17U, 32U, 17U, 0U, "oracle-m17-k32-n17"},
    {127U, 16U, 63U, 0U, "boundary-m127-k16-n63"},
    {128U, 32U, 64U, 0U, "boundary-m128-k32-n64"},
    {129U, 32U, 65U, 0U, "boundary-m129-k32-n65"},
}};

enum class CacheMode : uint32_t {
  Activation = 0U,
  Weight = 1U,
  Both = 2U,
};

const char *cache_mode_name(const CacheMode mode) {
  switch (mode) {
  case CacheMode::Activation:
    return "activation-cache";
  case CacheMode::Weight:
    return "weight-cache";
  case CacheMode::Both:
    return "activation-weight-cache";
  }
  return "unknown-cache";
}

// Four adjacent E2M1 values are expanded to four byte-addressable E4M3
// values.  Rows are contiguous and every supported K is a multiple of 16, so
// both the 16-bit source load and 32-bit destination store are aligned.
__global__
__launch_bounds__(kCacheThreads) void expand_e2m1_to_e4m3_exact_kernel(
    const uint8_t *const packed, uint8_t *const expanded, const uint64_t rows,
    const uint64_t k) {
#if defined(__gfx1201__)
  const uint64_t group =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t groups = rows * k / UINT64_C(4);
  if (group < groups) {
    const uint16_t source = __builtin_nontemporal_load(
        reinterpret_cast<const uint16_t *>(packed) + group);
    reinterpret_cast<uint32_t *>(expanded)[group] =
        e2m1x4_to_e4m3fn_exact(source);
  }
#else
  (void)packed;
  (void)expanded;
  (void)rows;
  (void)k;
#endif
}

template <bool ActivationCached, bool WeightCached>
__global__ __launch_bounds__(kCacheThreads, 1) void cached_id64_kernel(
    const uint8_t *const packed_activation,
    const uint8_t *const expanded_activation,
    const uint8_t *const activation_block_scales,
    const uint8_t *const packed_weight, const uint8_t *const expanded_weight,
    const uint8_t *const weight_block_scales,
    const float *const weight_tensor_scale,
    const float *const input_tensor_scale, uint16_t *const output,
    const uint64_t m, const uint64_t k, const uint64_t n) {
#if defined(__gfx1201__)
  constexpr uint32_t wave_width = 32U;
  constexpr uint32_t waves_per_workgroup = 8U;
  constexpr uint32_t tile_m = 16U;
  constexpr uint32_t tile_n = 16U;
  constexpr uint32_t column_tiles = 4U;
  constexpr uint32_t fragment_k = 16U;
  constexpr uint32_t stage_k = 32U;
  constexpr uint32_t scale_blocks_per_stage = stage_k / fragment_k;
  constexpr uint32_t rows_per_workgroup = waves_per_workgroup * tile_m;
  constexpr uint32_t columns_per_workgroup = column_tiles * tile_n;
  constexpr uint32_t tile_values = tile_m * stage_k;
  constexpr uint32_t values_per_group = 4U;
  constexpr uint32_t groups_per_tile = tile_values / values_per_group;
  constexpr uint32_t output_values = tile_m * tile_n;

  __shared__ __align__(4)
      rocwmma::float8_t activation_tile[waves_per_workgroup][tile_values];
  __shared__ __align__(4)
      rocwmma::float8_t weight_tile[column_tiles][tile_values];
  __shared__ float activation_scale_tile[waves_per_workgroup][tile_m]
                                        [scale_blocks_per_stage];
  __shared__ float weight_scale_tile[column_tiles * tile_n]
                                    [scale_blocks_per_stage];
  __shared__ float tensor_scale;

  using AFragment =
      rocwmma::fragment<rocwmma::matrix_a, tile_m, tile_n, fragment_k,
                        rocwmma::float8_t, rocwmma::row_major>;
  using BFragment =
      rocwmma::fragment<rocwmma::matrix_b, tile_m, tile_n, fragment_k,
                        rocwmma::float8_t, rocwmma::col_major>;
  using AccumulatorFragment = rocwmma::fragment<rocwmma::accumulator, tile_m,
                                                tile_n, fragment_k, float>;

  const uint32_t thread = threadIdx.x;
  const uint32_t lane = thread & (wave_width - 1U);
  const uint32_t wave = thread / wave_width;
  const uint64_t row_group_base =
      static_cast<uint64_t>(blockIdx.y) * rows_per_workgroup;
  const uint64_t row_tile_base =
      row_group_base + static_cast<uint64_t>(wave) * tile_m;
  const uint64_t column_base =
      static_cast<uint64_t>(blockIdx.x) * columns_per_workgroup;
  const uint64_t blocks_per_row = k / fragment_k;
  const uint64_t stages =
      (blocks_per_row + scale_blocks_per_stage - 1U) / scale_blocks_per_stage;
  const uint64_t packed_row_bytes = k / UINT64_C(2);
  float accumulators[column_tiles][output_values / wave_width] = {};

  if (thread == 0U) {
    tensor_scale = weight_tensor_scale[0] * input_tensor_scale[0];
  }

  for (uint64_t stage = 0U; stage < stages; ++stage) {
    const uint64_t inner_base = stage * stage_k;
    auto *const activation_groups =
        reinterpret_cast<uint32_t *>(activation_tile);
    auto *const weight_groups = reinterpret_cast<uint32_t *>(weight_tile);

    for (uint32_t group = thread; group < waves_per_workgroup * groups_per_tile;
         group += blockDim.x) {
      const uint32_t source_wave = group / groups_per_tile;
      const uint32_t wave_group = group - source_wave * groups_per_tile;
      const uint32_t local_row = wave_group / (stage_k / values_per_group);
      const uint32_t local_group =
          wave_group - local_row * (stage_k / values_per_group);
      const uint64_t row = row_group_base +
                           static_cast<uint64_t>(source_wave) * tile_m +
                           local_row;
      uint32_t values = 0U;
      if (row < m && inner_base + local_group * values_per_group < k) {
        if constexpr (ActivationCached) {
          values = __builtin_nontemporal_load(
              reinterpret_cast<const uint32_t *>(expanded_activation + row * k +
                                                 inner_base) +
              local_group);
        } else {
          const uint16_t packed = __builtin_nontemporal_load(
              reinterpret_cast<const uint16_t *>(packed_activation +
                                                 row * packed_row_bytes +
                                                 inner_base / UINT64_C(2)) +
              local_group);
          values = e2m1x4_to_e4m3fn_exact(packed);
        }
      }
      activation_groups[group] = values;
    }

    for (uint32_t group = thread; group < column_tiles * groups_per_tile;
         group += blockDim.x) {
      const uint32_t column_tile = group / groups_per_tile;
      const uint32_t tile_group = group - column_tile * groups_per_tile;
      const uint32_t local_column = tile_group / (stage_k / values_per_group);
      const uint32_t local_group =
          tile_group - local_column * (stage_k / values_per_group);
      const uint64_t column = column_base +
                              static_cast<uint64_t>(column_tile) * tile_n +
                              local_column;
      uint32_t values = 0U;
      if (column < n && inner_base + local_group * values_per_group < k) {
        if constexpr (WeightCached) {
          values = __builtin_nontemporal_load(
              reinterpret_cast<const uint32_t *>(expanded_weight + column * k +
                                                 inner_base) +
              local_group);
        } else {
          const uint16_t packed = __builtin_nontemporal_load(
              reinterpret_cast<const uint16_t *>(packed_weight +
                                                 column * packed_row_bytes +
                                                 inner_base / UINT64_C(2)) +
              local_group);
          values = e2m1x4_to_e4m3fn_exact(packed);
        }
      }
      weight_groups[group] = values;
    }

    if (thread < waves_per_workgroup * tile_m * scale_blocks_per_stage) {
      const uint32_t scale_block = thread % scale_blocks_per_stage;
      const uint32_t row_index = thread / scale_blocks_per_stage;
      const uint32_t source_wave = row_index / tile_m;
      const uint32_t local_row = row_index - source_wave * tile_m;
      const uint64_t row = row_group_base +
                           static_cast<uint64_t>(source_wave) * tile_m +
                           local_row;
      const uint64_t block = stage * scale_blocks_per_stage + scale_block;
      activation_scale_tile[source_wave][local_row][scale_block] =
          row < m && block < blocks_per_row
              ? e4m3fn_to_float(
                    activation_block_scales[row * blocks_per_row + block])
              : 0.0F;
    }
    if (thread < column_tiles * tile_n * scale_blocks_per_stage) {
      const uint32_t scale_block = thread % scale_blocks_per_stage;
      const uint32_t local_column = thread / scale_blocks_per_stage;
      const uint64_t column = column_base + local_column;
      const uint64_t block = stage * scale_blocks_per_stage + scale_block;
      weight_scale_tile[local_column][scale_block] =
          column < n && block < blocks_per_row
              ? e4m3fn_to_float(
                    weight_block_scales[column * blocks_per_row + block])
              : 0.0F;
    }
    __syncthreads();

    // Preserve ID64's exact numerical order: one FP8 WMMA for each K=16
    // scale domain, then activation scale, then weight scale, then FP32 add.
    for (uint32_t scale_block = 0U; scale_block < scale_blocks_per_stage;
         ++scale_block) {
      AFragment activation_fragment;
      rocwmma::load_matrix_sync(
          activation_fragment, activation_tile[wave] + scale_block * fragment_k,
          stage_k);
#pragma unroll
      for (uint32_t column_tile = 0U; column_tile < column_tiles;
           ++column_tile) {
        BFragment weight_fragment;
        AccumulatorFragment contribution;
        rocwmma::fill_fragment(contribution, 0.0F);
        rocwmma::load_matrix_sync(
            weight_fragment,
            weight_tile[column_tile] + scale_block * fragment_k, stage_k);
        rocwmma::mma_sync(contribution, activation_fragment, weight_fragment,
                          contribution);
        const auto row_major =
            rocwmma::apply_data_layout<rocwmma::row_major>(contribution);
#pragma unroll
        for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
          const uint32_t local_row =
              (lane / tile_n) * (output_values / wave_width) + slot;
          const uint32_t local_column = lane % tile_n;
          float term = row_major[slot] *
                       activation_scale_tile[wave][local_row][scale_block];
          term *= weight_scale_tile[column_tile * tile_n + local_column]
                                   [scale_block];
          accumulators[column_tile][slot] += term;
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (uint32_t column_tile = 0U; column_tile < column_tiles; ++column_tile) {
#pragma unroll
    for (uint32_t slot = 0U; slot < output_values / wave_width; ++slot) {
      const uint32_t local_row =
          (lane / tile_n) * (output_values / wave_width) + slot;
      const uint32_t local_column = lane % tile_n;
      const uint64_t row = row_tile_base + local_row;
      const uint64_t column = column_base + column_tile * tile_n + local_column;
      if (row < m && column < n) {
        output[row * n + column] =
            bf16_rne(accumulators[column_tile][slot] * tensor_scale);
      }
    }
  }
#else
  (void)packed_activation;
  (void)expanded_activation;
  (void)activation_block_scales;
  (void)packed_weight;
  (void)expanded_weight;
  (void)weight_block_scales;
  (void)weight_tensor_scale;
  (void)input_tensor_scale;
  (void)output;
  (void)m;
  (void)k;
  (void)n;
#endif
}

struct PairWeight final {
  std::vector<uint8_t> values;
  std::vector<uint8_t> scales;
};

PairWeight make_pair_weight(const Shape &shape, const HostInputs &inputs) {
  PairWeight pair;
  pair.values.resize(inputs.weight.size());
  pair.scales.resize(inputs.weight_scales.size());
  for (std::size_t index = 0U; index < pair.values.size(); ++index) {
    const uint8_t source = inputs.weight[index];
    const uint8_t low =
        static_cast<uint8_t>(((source & UINT8_C(0x0f)) + 3U) & 0x0fU);
    const uint8_t high = static_cast<uint8_t>(
        ((((source >> 4U) & UINT8_C(0x0f)) + 5U) & 0x0fU) << 4U);
    pair.values[index] = static_cast<uint8_t>(low | high);
  }
  for (std::size_t index = 0U; index < pair.scales.size(); ++index) {
    pair.scales[index] = positive_finite_e4m3(
        static_cast<uint64_t>(index),
        kSeed ^ UINT32_C(0xd1b54a35) ^ static_cast<uint32_t>(shape.k));
  }
  return pair;
}

struct CacheBuffers final {
  uint8_t *activation = nullptr;
  uint8_t *activation_scales = nullptr;
  uint8_t *activation_expanded = nullptr;
  uint8_t *weight = nullptr;
  uint8_t *weight_scales = nullptr;
  uint8_t *weight_expanded = nullptr;
  uint8_t *pair_weight = nullptr;
  uint8_t *pair_weight_scales = nullptr;
  float *weight_tensor_scale = nullptr;
  float *input_tensor_scale = nullptr;
  uint16_t *output = nullptr;
  uint16_t *pair_output = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  std::size_t allocations = 0U;
};

struct ByteSizes final {
  std::size_t activation = 0U;
  std::size_t activation_scales = 0U;
  std::size_t activation_expanded = 0U;
  std::size_t weight = 0U;
  std::size_t weight_scales = 0U;
  std::size_t weight_expanded = 0U;
  std::size_t output = 0U;
};

bool cache_shape_sizes(const Shape &shape, ByteSizes *const sizes) {
  if (sizes == nullptr || shape.m == 0U || shape.k == 0U || shape.n == 0U ||
      (shape.k % UINT64_C(16)) != 0U || shape.m > SIZE_MAX / shape.k ||
      shape.n > SIZE_MAX / shape.k || shape.m > SIZE_MAX / shape.n ||
      shape.m * shape.n > SIZE_MAX / sizeof(uint16_t)) {
    return false;
  }
  sizes->activation = static_cast<std::size_t>(shape.m * shape.k / 2U);
  sizes->activation_scales = static_cast<std::size_t>(shape.m * shape.k / 16U);
  sizes->activation_expanded = static_cast<std::size_t>(shape.m * shape.k);
  sizes->weight = static_cast<std::size_t>(shape.n * shape.k / 2U);
  sizes->weight_scales = static_cast<std::size_t>(shape.n * shape.k / 16U);
  sizes->weight_expanded = static_cast<std::size_t>(shape.n * shape.k);
  sizes->output =
      static_cast<std::size_t>(shape.m * shape.n * sizeof(uint16_t));
  return true;
}

bool cache_allocate(void **const pointer, const std::size_t bytes,
                    CacheBuffers *const buffers) {
  if (bytes == 0U ||
      !hip_ok(hipMalloc(pointer, bytes), "unscaled cache hipMalloc")) {
    return false;
  }
  ++buffers->allocations;
  return true;
}

bool allocate_and_upload_cache(const HostInputs &inputs, const PairWeight &pair,
                               const ByteSizes &sizes,
                               CacheBuffers *const buffers) {
  return cache_allocate(reinterpret_cast<void **>(&buffers->activation),
                        sizes.activation, buffers) &&
         cache_allocate(reinterpret_cast<void **>(&buffers->activation_scales),
                        sizes.activation_scales, buffers) &&
         cache_allocate(
             reinterpret_cast<void **>(&buffers->activation_expanded),
             sizes.activation_expanded, buffers) &&
         cache_allocate(reinterpret_cast<void **>(&buffers->weight),
                        sizes.weight, buffers) &&
         cache_allocate(reinterpret_cast<void **>(&buffers->weight_scales),
                        sizes.weight_scales, buffers) &&
         cache_allocate(reinterpret_cast<void **>(&buffers->weight_expanded),
                        sizes.weight_expanded, buffers) &&
         cache_allocate(reinterpret_cast<void **>(&buffers->pair_weight),
                        sizes.weight, buffers) &&
         cache_allocate(reinterpret_cast<void **>(&buffers->pair_weight_scales),
                        sizes.weight_scales, buffers) &&
         cache_allocate(
             reinterpret_cast<void **>(&buffers->weight_tensor_scale),
             sizeof(float), buffers) &&
         cache_allocate(reinterpret_cast<void **>(&buffers->input_tensor_scale),
                        sizeof(float), buffers) &&
         cache_allocate(reinterpret_cast<void **>(&buffers->output),
                        sizes.output, buffers) &&
         cache_allocate(reinterpret_cast<void **>(&buffers->pair_output),
                        sizes.output, buffers) &&
         hip_ok(hipStreamCreate(&buffers->stream), "cache hipStreamCreate") &&
         hip_ok(hipEventCreate(&buffers->start), "cache start event") &&
         hip_ok(hipEventCreate(&buffers->stop), "cache stop event") &&
         hip_ok(hipMemcpy(buffers->activation, inputs.activation.data(),
                          sizes.activation, hipMemcpyHostToDevice),
                "cache upload activation") &&
         hip_ok(hipMemcpy(buffers->activation_scales,
                          inputs.activation_scales.data(),
                          sizes.activation_scales, hipMemcpyHostToDevice),
                "cache upload activation scales") &&
         hip_ok(hipMemcpy(buffers->weight, inputs.weight.data(), sizes.weight,
                          hipMemcpyHostToDevice),
                "cache upload weight") &&
         hip_ok(hipMemcpy(buffers->weight_scales, inputs.weight_scales.data(),
                          sizes.weight_scales, hipMemcpyHostToDevice),
                "cache upload weight scales") &&
         hip_ok(hipMemcpy(buffers->pair_weight, pair.values.data(),
                          sizes.weight, hipMemcpyHostToDevice),
                "cache upload pair weight") &&
         hip_ok(hipMemcpy(buffers->pair_weight_scales, pair.scales.data(),
                          sizes.weight_scales, hipMemcpyHostToDevice),
                "cache upload pair weight scales") &&
         hip_ok(hipMemcpy(buffers->weight_tensor_scale, &kWeightTensorScale,
                          sizeof(float), hipMemcpyHostToDevice),
                "cache upload weight tensor scale") &&
         hip_ok(hipMemcpy(buffers->input_tensor_scale, &kInputTensorScale,
                          sizeof(float), hipMemcpyHostToDevice),
                "cache upload input tensor scale") &&
         hip_ok(hipMemset(buffers->output, 0, sizes.output),
                "cache clear output") &&
         hip_ok(hipMemset(buffers->pair_output, 0, sizes.output),
                "cache clear pair output");
}

void cleanup_cache(CacheBuffers *const buffers, CleanupTotals *const totals) {
  totals->allocations += buffers->allocations;
  const auto destroy_event = [&](hipEvent_t *const event) {
    if (*event != nullptr) {
      totals->ok = hip_ok(hipEventDestroy(*event), "cache hipEventDestroy") &&
                   totals->ok;
      *event = nullptr;
    }
  };
  destroy_event(&buffers->stop);
  destroy_event(&buffers->start);
  if (buffers->stream != nullptr) {
    totals->ok =
        hip_ok(hipStreamDestroy(buffers->stream), "cache hipStreamDestroy") &&
        totals->ok;
    buffers->stream = nullptr;
  }
  const auto free_device = [&](auto **const pointer) {
    if (*pointer != nullptr) {
      if (hip_ok(hipFree(*pointer), "cache hipFree")) {
        ++totals->frees;
      } else {
        totals->ok = false;
      }
      *pointer = nullptr;
    }
  };
  free_device(&buffers->pair_output);
  free_device(&buffers->output);
  free_device(&buffers->input_tensor_scale);
  free_device(&buffers->weight_tensor_scale);
  free_device(&buffers->pair_weight_scales);
  free_device(&buffers->pair_weight);
  free_device(&buffers->weight_expanded);
  free_device(&buffers->weight_scales);
  free_device(&buffers->weight);
  free_device(&buffers->activation_expanded);
  free_device(&buffers->activation_scales);
  free_device(&buffers->activation);
}

bool launch_expand(const uint8_t *const packed, uint8_t *const expanded,
                   const uint64_t rows, const uint64_t k,
                   const hipStream_t stream) {
  const uint64_t groups = rows * k / UINT64_C(4);
  const uint64_t blocks =
      (groups + kCacheThreads - UINT64_C(1)) / kCacheThreads;
  if (groups == 0U || blocks == 0U || blocks > UINT32_MAX) {
    return false;
  }
  hipLaunchKernelGGL(expand_e2m1_to_e4m3_exact_kernel,
                     dim3(static_cast<uint32_t>(blocks)), dim3(kCacheThreads),
                     0U, stream, packed, expanded, rows, k);
  return hip_ok(hipGetLastError(), "expand E2M1 to E4M3 launch");
}

bool launch_id64(const Shape &shape, const CacheBuffers &buffers,
                 const bool pair) {
  const dim3 grid(static_cast<uint32_t>((shape.n + 63U) / 64U),
                  static_cast<uint32_t>((shape.m + 127U) / 128U));
  const uint8_t *const weight = pair ? buffers.pair_weight : buffers.weight;
  const uint8_t *const scales =
      pair ? buffers.pair_weight_scales : buffers.weight_scales;
  uint16_t *const output = pair ? buffers.pair_output : buffers.output;
  hipLaunchKernelGGL(id64_control_kernel, grid, dim3(kCacheThreads), 0U,
                     buffers.stream, buffers.activation,
                     buffers.activation_scales, weight, scales,
                     buffers.weight_tensor_scale, buffers.input_tensor_scale,
                     output, shape.m, shape.k, shape.n);
  return hip_ok(hipGetLastError(), "ID64 control launch");
}

template <bool ActivationCached, bool WeightCached>
bool launch_cached(const Shape &shape, const CacheBuffers &buffers,
                   const bool pair) {
  const dim3 grid(static_cast<uint32_t>((shape.n + 63U) / 64U),
                  static_cast<uint32_t>((shape.m + 127U) / 128U));
  const uint8_t *const weight = pair ? buffers.pair_weight : buffers.weight;
  const uint8_t *const scales =
      pair ? buffers.pair_weight_scales : buffers.weight_scales;
  uint16_t *const output = pair ? buffers.pair_output : buffers.output;
  // The pair path is only used for activation-cache timing.  It deliberately
  // retains distinct packed gate/up weights and therefore never aliases the
  // first projection's output or weight stream.
  const uint8_t *const expanded_weight =
      pair ? nullptr : buffers.weight_expanded;
  hipLaunchKernelGGL((cached_id64_kernel<ActivationCached, WeightCached>), grid,
                     dim3(kCacheThreads), 0U, buffers.stream,
                     buffers.activation, buffers.activation_expanded,
                     buffers.activation_scales, weight, expanded_weight, scales,
                     buffers.weight_tensor_scale, buffers.input_tensor_scale,
                     output, shape.m, shape.k, shape.n);
  return hip_ok(hipGetLastError(), "cached ID64 launch");
}

uint8_t host_e2m1_to_e4m3(const uint8_t code) {
  constexpr std::array<uint8_t, 8> positive = {
      UINT8_C(0x00), UINT8_C(0x30), UINT8_C(0x38), UINT8_C(0x3c),
      UINT8_C(0x40), UINT8_C(0x44), UINT8_C(0x48), UINT8_C(0x4c)};
  return static_cast<uint8_t>(positive[code & UINT8_C(7)] |
                              ((code & UINT8_C(8)) << 4U));
}

bool validate_expanded_bytes(const char *const label,
                             const std::vector<uint8_t> &packed,
                             const uint8_t *const device_expanded,
                             const std::size_t expanded_bytes) {
  std::vector<uint8_t> observed(expanded_bytes);
  if (!hip_ok(hipMemcpy(observed.data(), device_expanded, expanded_bytes,
                        hipMemcpyDeviceToHost),
              "copy expanded bytes")) {
    return false;
  }
  std::size_t mismatches = 0U;
  std::size_t first = SIZE_MAX;
  for (std::size_t index = 0U; index < expanded_bytes; ++index) {
    const uint8_t source = packed[index / 2U];
    const uint8_t code = (index & 1U) == 0U
                             ? source & UINT8_C(0x0f)
                             : static_cast<uint8_t>(source >> 4U);
    if (observed[index] != host_e2m1_to_e4m3(code)) {
      first = std::min(first, index);
      ++mismatches;
    }
  }
  std::cout << "expanded_oracle operand=" << label
            << " bytes=" << expanded_bytes << " mismatches=" << mismatches;
  if (first != SIZE_MAX) {
    std::cout << " first_mismatch=" << first;
  }
  std::cout << " status=" << (mismatches == 0U ? "PASS" : "FAIL") << "\n";
  return mismatches == 0U;
}

struct Timing final {
  std::array<float, kCacheMeasured> samples_us{};
  float median_us = 0.0F;
  float mad_us = 0.0F;
  float minimum_us = 0.0F;
  float maximum_us = 0.0F;
  std::size_t repeat_mismatches = 0U;
  std::vector<std::vector<uint16_t>> outputs;
  bool ran = false;
  bool deterministic = false;
};

float upper_median(std::array<float, kCacheMeasured> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2U];
}

void finish_timing(const char *const shape_name, const char *const label,
                   Timing *const timing) {
  timing->median_us = upper_median(timing->samples_us);
  std::array<float, kCacheMeasured> deviations{};
  for (std::size_t index = 0U; index < deviations.size(); ++index) {
    deviations[index] = std::abs(timing->samples_us[index] - timing->median_us);
  }
  timing->mad_us = upper_median(deviations);
  timing->minimum_us =
      *std::min_element(timing->samples_us.begin(), timing->samples_us.end());
  timing->maximum_us =
      *std::max_element(timing->samples_us.begin(), timing->samples_us.end());
  timing->ran = true;
  timing->deterministic = timing->repeat_mismatches == 0U;
  std::cout << "timing shape=" << shape_name << " measurement=" << label
            << " warmups=" << kCacheWarmups << " measured=" << kCacheMeasured
            << " samples_us=";
  for (std::size_t index = 0U; index < timing->samples_us.size(); ++index) {
    if (index != 0U) {
      std::cout << ',';
    }
    std::cout << std::fixed << std::setprecision(3)
              << timing->samples_us[index];
  }
  std::cout << " median_us=" << timing->median_us
            << " mad_us=" << timing->mad_us << " min_us=" << timing->minimum_us
            << " max_us=" << timing->maximum_us
            << " repeat_bf16_mismatches=" << timing->repeat_mismatches
            << " deterministic=" << (timing->deterministic ? "PASS" : "FAIL")
            << "\n";
}

template <typename Launch>
bool measure_sequence(const Shape &shape, const char *const label,
                      CacheBuffers *const buffers,
                      const std::span<uint16_t *const> output_pointers,
                      const std::size_t output_bytes, Launch &&launch,
                      Timing *const timing) {
  for (uint32_t warmup = 0U; warmup < kCacheWarmups; ++warmup) {
    if (!launch()) {
      return false;
    }
  }
  if (!hip_ok(hipStreamSynchronize(buffers->stream),
              "cache warmup synchronize")) {
    return false;
  }
  const std::size_t output_elements = output_bytes / sizeof(uint16_t);
  timing->outputs.assign(output_pointers.size(),
                         std::vector<uint16_t>(output_elements));
  std::vector<std::vector<uint16_t>> current(
      output_pointers.size(), std::vector<uint16_t>(output_elements));
  for (uint32_t iteration = 0U; iteration < kCacheMeasured; ++iteration) {
    if (!hip_ok(hipEventRecord(buffers->start, buffers->stream),
                "cache timing start") ||
        !launch() ||
        !hip_ok(hipEventRecord(buffers->stop, buffers->stream),
                "cache timing stop") ||
        !hip_ok(hipEventSynchronize(buffers->stop),
                "cache timing synchronize")) {
      return false;
    }
    float milliseconds = 0.0F;
    if (!hip_ok(
            hipEventElapsedTime(&milliseconds, buffers->start, buffers->stop),
            "cache elapsed time")) {
      return false;
    }
    timing->samples_us[iteration] = milliseconds * 1000.0F;
    for (std::size_t output_index = 0U; output_index < output_pointers.size();
         ++output_index) {
      if (!hip_ok(hipMemcpy(current[output_index].data(),
                            output_pointers[output_index], output_bytes,
                            hipMemcpyDeviceToHost),
                  "cache copy output")) {
        return false;
      }
      if (iteration == 0U) {
        timing->outputs[output_index] = current[output_index];
      } else {
        for (std::size_t element = 0U; element < output_elements; ++element) {
          timing->repeat_mismatches +=
              static_cast<std::size_t>(current[output_index][element] !=
                                       timing->outputs[output_index][element]);
        }
      }
    }
  }
  finish_timing(shape.name, label, timing);
  return timing->deterministic;
}

template <typename Launch>
bool measure_kernel_only(const Shape &shape, const char *const label,
                         CacheBuffers *const buffers, Launch &&launch,
                         Timing *const timing) {
  for (uint32_t warmup = 0U; warmup < kCacheWarmups; ++warmup) {
    if (!launch()) {
      return false;
    }
  }
  if (!hip_ok(hipStreamSynchronize(buffers->stream),
              "conversion warmup synchronize")) {
    return false;
  }
  for (uint32_t iteration = 0U; iteration < kCacheMeasured; ++iteration) {
    if (!hip_ok(hipEventRecord(buffers->start, buffers->stream),
                "conversion timing start") ||
        !launch() ||
        !hip_ok(hipEventRecord(buffers->stop, buffers->stream),
                "conversion timing stop") ||
        !hip_ok(hipEventSynchronize(buffers->stop),
                "conversion timing synchronize")) {
      return false;
    }
    float milliseconds = 0.0F;
    if (!hip_ok(
            hipEventElapsedTime(&milliseconds, buffers->start, buffers->stop),
            "conversion elapsed time")) {
      return false;
    }
    timing->samples_us[iteration] = milliseconds * 1000.0F;
  }
  finish_timing(shape.name, label, timing);
  timing->deterministic = true;
  return true;
}

std::size_t nonfinite_count(const std::vector<uint16_t> &values) {
  return static_cast<std::size_t>(
      std::count_if(values.begin(), values.end(), [](const uint16_t bits) {
        return (bits & UINT16_C(0x7f80)) == UINT16_C(0x7f80);
      }));
}

bool exact_compare(const Shape &shape, const char *const label,
                   const std::vector<uint16_t> &reference,
                   const std::vector<uint16_t> &candidate) {
  if (reference.size() != candidate.size()) {
    std::cout << "bit_compare shape=" << shape.name << " pair=" << label
              << " status=FAIL reason=size-mismatch\n";
    return false;
  }
  const Comparison comparison = compare(reference, candidate);
  const std::size_t nonfinite = nonfinite_count(candidate);
  std::cout << "bit_compare shape=" << shape.name << " pair=" << label
            << " bf16_mismatches=" << comparison.mismatches
            << " max_bf16_ulp=" << comparison.max_ulp
            << " nonfinite=" << nonfinite << " reference_fnv64=0x" << std::hex
            << hash_bf16(reference) << " candidate_fnv64=0x"
            << hash_bf16(candidate) << std::dec << " status="
            << (comparison.mismatches == 0U && nonfinite == 0U ? "PASS"
                                                               : "FAIL")
            << "\n";
  return comparison.mismatches == 0U && nonfinite == 0U;
}

struct CacheResource final {
  bool available = false;
  int vgpr = 0;
  std::size_t lds = 0U;
  std::size_t scratch = 0U;
  int active_blocks = 0;
  double occupancy = 0.0;
};

CacheResource print_cache_resource(const char *const name,
                                   const void *const kernel,
                                   const hipDeviceProp_t &properties) {
  CacheResource resource;
  hipFuncAttributes attributes{};
  int active_blocks = 0;
  if (!hip_ok(hipFuncGetAttributes(&attributes, kernel),
              "cache hipFuncGetAttributes") ||
      !hip_ok(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                  &active_blocks, kernel, static_cast<int>(kCacheThreads), 0U),
              "cache occupancy")) {
    return resource;
  }
  resource.available = true;
  resource.vgpr = attributes.numRegs;
  resource.lds = attributes.sharedSizeBytes;
  resource.scratch = attributes.localSizeBytes;
  resource.active_blocks = active_blocks;
  resource.occupancy =
      properties.maxThreadsPerMultiProcessor == 0
          ? 0.0
          : static_cast<double>(active_blocks * kCacheThreads) /
                properties.maxThreadsPerMultiProcessor;
  std::cout << "resources variant=" << name << " vgpr=" << resource.vgpr
            << " lds=" << resource.lds
            << " scratch_per_thread=" << resource.scratch
            << " max_threads=" << attributes.maxThreadsPerBlock
            << " active_blocks_per_cu=" << active_blocks
            << " active_waves_per_cu=" << active_blocks * 8
            << " occupancy=" << std::fixed << std::setprecision(6)
            << resource.occupancy << " status="
            << (resource.scratch == 0U && resource.lds <= kMaximumLdsBytes &&
                        resource.vgpr <= kMaximumVgpr && active_blocks > 0
                    ? "PASS"
                    : "FAIL")
            << "\n";
  return resource;
}

struct ShapeMetrics final {
  Shape shape{};
  float control_us = 0.0F;
  float activation_expand_us = 0.0F;
  float weight_expand_us = 0.0F;
  float activation_steady_us = 0.0F;
  float activation_cold_us = 0.0F;
  float weight_steady_us = 0.0F;
  float weight_cold_us = 0.0F;
  float both_steady_us = 0.0F;
  float both_cold_us = 0.0F;
  float pair_control_us = 0.0F;
  float pair_activation_steady_us = 0.0F;
  float pair_activation_cold_us = 0.0F;
  bool ok = false;
};

void print_amortization(const ShapeMetrics &metrics) {
  const double control = metrics.control_us;
  const double weight_gain = control - metrics.weight_steady_us;
  const double weight_break_even =
      weight_gain > 0.0 ? std::ceil(metrics.weight_expand_us / weight_gain)
                        : std::numeric_limits<double>::infinity();
  const double request_weight =
      metrics.weight_expand_us + kRequestChunks * metrics.weight_steady_us;
  const double request_both =
      metrics.weight_expand_us +
      kRequestChunks * (metrics.activation_expand_us + metrics.both_steady_us);
  const double request_control = kRequestChunks * control;
  std::cout << "amortization shape=" << metrics.shape.name
            << " request_chunks=" << kRequestChunks
            << " weight_break_even_chunks=";
  if (std::isfinite(weight_break_even)) {
    std::cout << static_cast<uint64_t>(weight_break_even);
  } else {
    std::cout << "never";
  }
  std::cout << " request_control_us=" << std::fixed << std::setprecision(3)
            << request_control << " request_weight_cache_us=" << request_weight
            << " request_weight_speedup=" << request_control / request_weight
            << " request_both_cache_us=" << request_both
            << " request_both_speedup=" << request_control / request_both
            << " pair_activation_cold_speedup="
            << (metrics.pair_activation_cold_us > 0.0F
                    ? metrics.pair_control_us / metrics.pair_activation_cold_us
                    : 0.0)
            << "\n";
}

bool run_cache_shape(const Shape &shape, ShapeMetrics *const metrics,
                     CleanupTotals *const cleanup_totals) {
  std::cout << "shape_begin name=" << shape.name << " m=" << shape.m
            << " k=" << shape.k << " n=" << shape.n
            << " occurrences=" << shape.occurrences << "\n";
  ByteSizes sizes;
  if (!cache_shape_sizes(shape, &sizes)) {
    std::cerr << "invalid cache probe shape " << shape.name << "\n";
    return false;
  }
  std::cout << "memory shape=" << shape.name
            << " packed_activation_bytes=" << sizes.activation
            << " activation_scale_bytes=" << sizes.activation_scales
            << " activation_cache_additional_bytes="
            << sizes.activation_expanded
            << " packed_weight_bytes=" << sizes.weight
            << " weight_scale_bytes=" << sizes.weight_scales
            << " weight_cache_additional_bytes=" << sizes.weight_expanded
            << " gate_up_weight_cache_additional_bytes="
            << sizes.weight_expanded * 2U << " output_bytes=" << sizes.output
            << "\n";

  const HostInputs inputs = make_inputs(shape);
  const PairWeight pair = make_pair_weight(shape, inputs);
  CacheBuffers buffers;
  if (!allocate_and_upload_cache(inputs, pair, sizes, &buffers)) {
    cleanup_cache(&buffers, cleanup_totals);
    return false;
  }
  bool ok = true;
  metrics->shape = shape;

  Timing activation_expansion;
  Timing weight_expansion;
  ok = measure_kernel_only(
           shape, "activation-expand-only", &buffers,
           [&]() {
             return launch_expand(buffers.activation,
                                  buffers.activation_expanded, shape.m, shape.k,
                                  buffers.stream);
           },
           &activation_expansion) &&
       ok;
  ok = measure_kernel_only(
           shape, "weight-expand-only", &buffers,
           [&]() {
             return launch_expand(buffers.weight, buffers.weight_expanded,
                                  shape.n, shape.k, buffers.stream);
           },
           &weight_expansion) &&
       ok;
  ok = hip_ok(hipStreamSynchronize(buffers.stream),
              "expanded byte oracle synchronize") &&
       ok;
  ok = validate_expanded_bytes("activation", inputs.activation,
                               buffers.activation_expanded,
                               sizes.activation_expanded) &&
       ok;
  ok = validate_expanded_bytes("weight", inputs.weight, buffers.weight_expanded,
                               sizes.weight_expanded) &&
       ok;

  const std::array<uint16_t *, 1> one_output = {buffers.output};
  Timing control;
  Timing activation_steady;
  Timing activation_cold;
  Timing weight_steady;
  Timing weight_cold;
  Timing both_steady;
  Timing both_cold;

  ok = measure_sequence(
           shape, "id64-control", &buffers, one_output, sizes.output,
           [&]() { return launch_id64(shape, buffers, false); }, &control) &&
       ok;
  ok = measure_sequence(
           shape, "activation-cache-steady", &buffers, one_output, sizes.output,
           [&]() { return launch_cached<true, false>(shape, buffers, false); },
           &activation_steady) &&
       ok;
  ok = measure_sequence(
           shape, "activation-cache-cold", &buffers, one_output, sizes.output,
           [&]() {
             return launch_expand(buffers.activation,
                                  buffers.activation_expanded, shape.m, shape.k,
                                  buffers.stream) &&
                    launch_cached<true, false>(shape, buffers, false);
           },
           &activation_cold) &&
       ok;
  ok = measure_sequence(
           shape, "weight-cache-steady", &buffers, one_output, sizes.output,
           [&]() { return launch_cached<false, true>(shape, buffers, false); },
           &weight_steady) &&
       ok;
  ok = measure_sequence(
           shape, "weight-cache-cold", &buffers, one_output, sizes.output,
           [&]() {
             return launch_expand(buffers.weight, buffers.weight_expanded,
                                  shape.n, shape.k, buffers.stream) &&
                    launch_cached<false, true>(shape, buffers, false);
           },
           &weight_cold) &&
       ok;
  ok = measure_sequence(
           shape, "both-cache-steady", &buffers, one_output, sizes.output,
           [&]() { return launch_cached<true, true>(shape, buffers, false); },
           &both_steady) &&
       ok;
  ok = measure_sequence(
           shape, "both-cache-cold", &buffers, one_output, sizes.output,
           [&]() {
             return launch_expand(buffers.activation,
                                  buffers.activation_expanded, shape.m, shape.k,
                                  buffers.stream) &&
                    launch_expand(buffers.weight, buffers.weight_expanded,
                                  shape.n, shape.k, buffers.stream) &&
                    launch_cached<true, true>(shape, buffers, false);
           },
           &both_cold) &&
       ok;

  if (control.ran) {
    const auto oracle = host_oracle(shape, inputs);
    ok = check_host_oracle(shape, Variant::Id64, oracle,
                           control.outputs.front()) &&
         ok;
    ok = nonfinite_count(control.outputs.front()) == 0U && ok;
  }
  if (control.ran && activation_steady.ran && activation_cold.ran &&
      weight_steady.ran && weight_cold.ran && both_steady.ran &&
      both_cold.ran) {
    ok = exact_compare(shape, "id64-vs-activation-steady",
                       control.outputs.front(),
                       activation_steady.outputs.front()) &&
         ok;
    ok =
        exact_compare(shape, "id64-vs-activation-cold", control.outputs.front(),
                      activation_cold.outputs.front()) &&
        ok;
    ok = exact_compare(shape, "id64-vs-weight-steady", control.outputs.front(),
                       weight_steady.outputs.front()) &&
         ok;
    ok = exact_compare(shape, "id64-vs-weight-cold", control.outputs.front(),
                       weight_cold.outputs.front()) &&
         ok;
    ok = exact_compare(shape, "id64-vs-both-steady", control.outputs.front(),
                       both_steady.outputs.front()) &&
         ok;
    ok = exact_compare(shape, "id64-vs-both-cold", control.outputs.front(),
                       both_cold.outputs.front()) &&
         ok;
  }

  // Gate/up pair: one activation expansion feeds two distinct packed weight
  // matrices and two disjoint outputs.  Weight expansion is intentionally not
  // included in this pair experiment.
  const std::array<uint16_t *, 2> pair_outputs = {buffers.output,
                                                  buffers.pair_output};
  Timing pair_control;
  Timing pair_activation_steady;
  Timing pair_activation_cold;
  ok = measure_sequence(
           shape, "pair-id64-control", &buffers, pair_outputs, sizes.output,
           [&]() {
             return launch_id64(shape, buffers, false) &&
                    launch_id64(shape, buffers, true);
           },
           &pair_control) &&
       ok;
  ok = measure_sequence(
           shape, "pair-activation-cache-steady", &buffers, pair_outputs,
           sizes.output,
           [&]() {
             return launch_cached<true, false>(shape, buffers, false) &&
                    launch_cached<true, false>(shape, buffers, true);
           },
           &pair_activation_steady) &&
       ok;
  ok = measure_sequence(
           shape, "pair-activation-cache-cold", &buffers, pair_outputs,
           sizes.output,
           [&]() {
             return launch_expand(buffers.activation,
                                  buffers.activation_expanded, shape.m, shape.k,
                                  buffers.stream) &&
                    launch_cached<true, false>(shape, buffers, false) &&
                    launch_cached<true, false>(shape, buffers, true);
           },
           &pair_activation_cold) &&
       ok;
  if (pair_control.ran && pair_activation_steady.ran &&
      pair_activation_cold.ran) {
    for (std::size_t index = 0U; index < pair_outputs.size(); ++index) {
      const std::string steady_label =
          "pair-id64-vs-activation-steady-output" + std::to_string(index);
      const std::string cold_label =
          "pair-id64-vs-activation-cold-output" + std::to_string(index);
      ok = exact_compare(shape, steady_label.c_str(),
                         pair_control.outputs[index],
                         pair_activation_steady.outputs[index]) &&
           ok;
      ok = exact_compare(shape, cold_label.c_str(), pair_control.outputs[index],
                         pair_activation_cold.outputs[index]) &&
           ok;
    }
  }

  metrics->control_us = control.median_us;
  metrics->activation_expand_us = activation_expansion.median_us;
  metrics->weight_expand_us = weight_expansion.median_us;
  metrics->activation_steady_us = activation_steady.median_us;
  metrics->activation_cold_us = activation_cold.median_us;
  metrics->weight_steady_us = weight_steady.median_us;
  metrics->weight_cold_us = weight_cold.median_us;
  metrics->both_steady_us = both_steady.median_us;
  metrics->both_cold_us = both_cold.median_us;
  metrics->pair_control_us = pair_control.median_us;
  metrics->pair_activation_steady_us = pair_activation_steady.median_us;
  metrics->pair_activation_cold_us = pair_activation_cold.median_us;
  metrics->ok = ok;
  print_amortization(*metrics);

  cleanup_cache(&buffers, cleanup_totals);
  std::cout << "shape_end name=" << shape.name
            << " status=" << (ok ? "PASS" : "FAIL") << "\n";
  return ok;
}

void print_weighted_cache_results(
    const std::array<ShapeMetrics, kShapes.size()> &results) {
  struct Field final {
    const char *name;
    float ShapeMetrics::*value;
  };
  constexpr std::array<Field, 9> fields = {{
      {"id64-control", &ShapeMetrics::control_us},
      {"activation-expand", &ShapeMetrics::activation_expand_us},
      {"weight-expand", &ShapeMetrics::weight_expand_us},
      {"activation-cache-steady", &ShapeMetrics::activation_steady_us},
      {"activation-cache-cold", &ShapeMetrics::activation_cold_us},
      {"weight-cache-steady", &ShapeMetrics::weight_steady_us},
      {"weight-cache-cold", &ShapeMetrics::weight_cold_us},
      {"both-cache-steady", &ShapeMetrics::both_steady_us},
      {"both-cache-cold", &ShapeMetrics::both_cold_us},
  }};
  double control_total = 0.0;
  uint64_t total_weight = 0U;
  for (const ShapeMetrics &result : results) {
    control_total += result.control_us * result.shape.occurrences;
    total_weight += result.shape.occurrences;
  }
  for (const Field &field : fields) {
    double total = 0.0;
    for (const ShapeMetrics &result : results) {
      total += (result.*(field.value)) * result.shape.occurrences;
    }
    std::cout << "weighted measurement=" << field.name
              << " shapes=6 qwen_projection_weight=" << total_weight
              << " weighted_total_us=" << std::fixed << std::setprecision(3)
              << total << " weighted_mean_us=" << total / total_weight
              << " speedup_vs_id64="
              << (total > 0.0 ? control_total / total : 0.0) << "\n";
  }
  for (const uint64_t m : {UINT64_C(128), UINT64_C(512), UINT64_C(1024)}) {
    for (const Field &field : fields) {
      double total = 0.0;
      double control = 0.0;
      uint64_t weight = 0U;
      for (const ShapeMetrics &result : results) {
        if (result.shape.m == m) {
          total += (result.*(field.value)) * result.shape.occurrences;
          control += result.control_us * result.shape.occurrences;
          weight += result.shape.occurrences;
        }
      }
      std::cout << "weighted_by_m m=" << m << " measurement=" << field.name
                << " qwen_projection_weight=" << weight
                << " weighted_mean_us=" << std::fixed << std::setprecision(3)
                << total / weight
                << " speedup_vs_id64=" << (total > 0.0 ? control / total : 0.0)
                << "\n";
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  int device = 0;
  if (argc > 2 || (argc == 2 && !parse_device(argv[1], &device))) {
    std::cerr << "usage: phase78_nvfp4_gfx1201_wmma_unscaled_fp8_cache_probe "
                 "[DEVICE]\n";
    return EXIT_FAILURE;
  }
  if (!hip_ok(hipSetDevice(device), "cache probe hipSetDevice")) {
    return EXIT_FAILURE;
  }
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, device),
              "cache probe hipGetDeviceProperties")) {
    return EXIT_FAILURE;
  }
  std::cout << "identity device=" << device << " name=" << properties.name
            << " pci_domain=" << properties.pciDomainID
            << " pci_bus=" << properties.pciBusID
            << " pci_device=" << properties.pciDeviceID
            << " arch=" << properties.gcnArchName
            << " total_global_mem=" << properties.totalGlobalMem << "\n";
  if (!exact_gfx1201(properties.gcnArchName)) {
    std::cerr << "unsupported exact target: expected gfx1201, observed "
              << properties.gcnArchName << "\n";
    return EXIT_FAILURE;
  }

  bool resources_ok = true;
  const ResourceInfo id64 = resource_info(Variant::Id64, properties);
  resources_ok = id64.available && id64.scratch == 0U &&
                 id64.lds <= kMaximumLdsBytes && id64.vgpr <= kMaximumVgpr &&
                 id64.active_blocks_per_cu > 0;
  const CacheResource expand = print_cache_resource(
      "unscaled-e2m1-to-e4m3-expand",
      reinterpret_cast<const void *>(expand_e2m1_to_e4m3_exact_kernel),
      properties);
  const CacheResource activation = print_cache_resource(
      cache_mode_name(CacheMode::Activation),
      reinterpret_cast<const void *>(cached_id64_kernel<true, false>),
      properties);
  const CacheResource weight = print_cache_resource(
      cache_mode_name(CacheMode::Weight),
      reinterpret_cast<const void *>(cached_id64_kernel<false, true>),
      properties);
  const CacheResource both = print_cache_resource(
      cache_mode_name(CacheMode::Both),
      reinterpret_cast<const void *>(cached_id64_kernel<true, true>),
      properties);
  for (const CacheResource *const resource :
       {&expand, &activation, &weight, &both}) {
    resources_ok = resource->available && resource->scratch == 0U &&
                   resource->lds <= kMaximumLdsBytes &&
                   resource->vgpr <= kMaximumVgpr &&
                   resource->active_blocks > 0 && resources_ok;
  }

  CleanupTotals cleanup_totals;
  bool correctness_ok = true;
  for (const Shape &shape : kBoundaryShapes) {
    ShapeMetrics ignored;
    correctness_ok =
        run_cache_shape(shape, &ignored, &cleanup_totals) && correctness_ok;
  }

  std::array<ShapeMetrics, kShapes.size()> results{};
  for (std::size_t index = 0U; index < kShapes.size(); ++index) {
    correctness_ok =
        run_cache_shape(kShapes[index], &results[index], &cleanup_totals) &&
        correctness_ok;
  }
  print_weighted_cache_results(results);

  bool two_x = true;
  for (const ShapeMetrics &result : results) {
    if (result.shape.m != 1024U) {
      continue;
    }
    const double request_both_per_chunk =
        (result.weight_expand_us +
         kRequestChunks *
             (result.activation_expand_us + result.both_steady_us)) /
        kRequestChunks;
    const double target = result.shape.k == 5120U ? 4230.0 : 4183.0;
    const bool shape_two_x =
        result.activation_cold_us <= result.control_us / 2.0F ||
        request_both_per_chunk <= result.control_us / 2.0;
    const bool absolute_target =
        result.activation_cold_us <= target || request_both_per_chunk <= target;
    std::cout << "two_x_gate shape=" << result.shape.name
              << " control_us=" << result.control_us
              << " activation_cold_us=" << result.activation_cold_us
              << " request_both_per_chunk_us=" << request_both_per_chunk
              << " half_control_us=" << result.control_us / 2.0F
              << " absolute_target_us=" << target
              << " relative=" << (shape_two_x ? "PASS" : "FAIL")
              << " absolute=" << (absolute_target ? "PASS" : "FAIL") << "\n";
    two_x = shape_two_x && absolute_target && two_x;
  }

  const bool cleanup_ok =
      cleanup_totals.ok && cleanup_totals.allocations == cleanup_totals.frees;
  std::cout << "cleanup allocations=" << cleanup_totals.allocations
            << " frees=" << cleanup_totals.frees
            << " status=" << (cleanup_ok ? "PASS" : "FAIL") << "\n";
  const bool evidence_ok = correctness_ok && resources_ok && cleanup_ok;
  std::cout << "PHASE78_NVFP4_UNSCALED_FP8_CACHE_EVIDENCE="
            << (evidence_ok ? "PASS" : "FAIL") << "\n";
  std::cout << "PHASE78_NVFP4_UNSCALED_FP8_CACHE_DECISION="
            << (evidence_ok && two_x ? "GO" : "N0") << "\n";
  return evidence_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
