// Phase 78 standalone CK Tile FMHA probe for Qwen3.8 GQA6 prefill.
//
// This file is intentionally not part of the production build.  It probes
// CK Tile's direct (header-only) FmhaFwdKernel entry point for the exact
// gfx1201 layout used by the model:
//   Q: [M, 24, 256] BF16 (staged to FP16 on device)
//   K/V: [L, 4, 256] FP16, row-major V
//   O: [M, 24, 256] BF16
//
// The selected candidate is a 64x64 sequence tile, K0=K1=32, using the
// QRKSVS pipeline in group mode.  The executable contains a
// tiny FP32 host oracle and 3+10 event timing, but GPU execution is an
// explicit operator choice: this source may be compiled without touching a
// GPU and defaults to a dry-run summary.  On ROCm 7.14/gfx1201, the requested
// QRKSVSAsync variant cannot be emitted for this shape: CK's gfx12 async LDS
// path attempts an unsupported dwordx4 load for the V distribution.  The
// compileable baseline below therefore uses QRKSVS (synchronous LDS ingress)
// while retaining the requested tile and K0/K1 values; this limitation is
// reported by the probe rather than hidden behind a fallback.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/fmha_fwd.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int kQHeads = 24;
constexpr int kKvHeads = 4;
constexpr int kHeadDim = 256;
constexpr int kGqa = 6;
constexpr int kWarmups = 3;
constexpr int kMeasured = 10;
constexpr float kScale = 1.0F / 16.0F; // 1 / sqrt(256)

using FmhaShape = ck_tile::TileFmhaShape<
    ck_tile::sequence<64, 64, 32, 256, 32, 256>, ck_tile::sequence<2, 2, 1>,
    ck_tile::sequence<16, 16, 16>, ck_tile::sequence<4, 1, 1>,
    ck_tile::sequence<16, 16, 16>,
    true>; // row-major V: [L, D]

using FmhaTraits = ck_tile::TileFmhaTraits<
    true, true, true, true, false, ck_tile::BlockAttentionBiasEnum::NO_BIAS,
    false, false, false, ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE, -1>;

using FmhaProblem = ck_tile::BlockFmhaPipelineProblem<
    ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, float, float,
    ck_tile::half_t, uint8_t, float, ck_tile::half_t, float, ck_tile::bf16_t,
    FmhaShape, true, ck_tile::StandardAttention,
    ck_tile::GenericAttentionMask<true, false>, false, FmhaTraits>;

// CK's gfx1201 host-side GetSmemSize helper calls get_n_lds_banks(), which is
// device-only for the 7.14 headers.  Keep the policy local and provide the
// gfx1201 value (32 banks) explicitly.  The descriptor itself remains CK's
// descriptor and is instantiated by the device kernel.
struct FmhaProbePolicy
    : ck_tile::BlockFmhaPipelineQXKSVSCustomPolicy<true, false, 1, 1> {
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeKV() {
    // gfx1201: V LDS is 2 * 32 * (128 + 16) half elements for K1=32,
    // N1=256, one K/V buffer; it dominates the K LDS descriptor.
    return 18432;
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize() {
    return GetSmemSizeKV<Problem>();
  }
};

using FmhaPipeline =
    ck_tile::BlockFmhaPipelineQRKSVS<FmhaProblem, FmhaProbePolicy>;
using FmhaEpilogueProblem =
    ck_tile::Default2DEpilogueProblem<float, ck_tile::bf16_t, true, true>;
using FmhaEpilogue = ck_tile::Default2DEpilogue<FmhaEpilogueProblem>;
using FmhaKernel = ck_tile::FmhaFwdKernel<FmhaPipeline, FmhaEpilogue>;

static_assert(FmhaShape::kM0 == 64 && FmhaShape::kN0 == 64 &&
                  FmhaShape::kK0 == 32 && FmhaShape::kK1 == 32 &&
                  FmhaShape::IsVLayoutRowMajor,
              "probe shape changed");
static_assert(FmhaKernel::kIsGroupMode, "the probe must use group mode");

struct DeviceCase {
  int m = 0;
  int l = 0;
  int start = 0;
  ck_tile::bf16_t *q_bf16 = nullptr;
  ck_tile::half_t *q_fp16 = nullptr;
  ck_tile::half_t *k_fp16 = nullptr;
  ck_tile::half_t *v_fp16 = nullptr;
  ck_tile::bf16_t *o_bf16 = nullptr;
  int32_t *seqstart_q = nullptr;
  int32_t *seqstart_k = nullptr;
};

[[nodiscard]] bool hip_ok(const hipError_t status, const char *where) {
  if (status == hipSuccess)
    return true;
  std::fprintf(stderr, "CK probe HIP error at %s: code=%d %s\n", where,
               static_cast<int>(status), hipGetErrorString(status));
  return false;
}

template <typename T>
bool device_alloc(T **ptr, const size_t count, const char *where) {
  return hip_ok(hipMalloc(reinterpret_cast<void **>(ptr), count * sizeof(T)),
                where);
}

template <typename T> bool device_free(T *ptr, const char *where) {
  return ptr == nullptr || hip_ok(hipFree(ptr), where);
}

__global__ void stage_bf16_to_fp16(const ck_tile::bf16_t *src,
                                   ck_tile::half_t *dst, size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    dst[i] = ck_tile::bf16_to_fp16(src[i]);
}

// The direct helper is deliberately kept in this file.  Using the public
// runner would select a different template path and hides the shape being
// measured; direct template launch also avoids an unresolved runner wrapper.
bool launch_fmha(const DeviceCase &dc, hipStream_t stream) {
  fmha_fwd_args args{};
  args.q_ptr = dc.q_fp16;
  args.k_ptr = dc.k_fp16;
  args.v_ptr = dc.v_fp16;
  args.o_ptr = dc.o_bf16;
  args.seqstart_q_ptr = dc.seqstart_q;
  args.seqstart_k_ptr = dc.seqstart_k;
  args.batch = 1;
  args.seqlen_q = dc.m;
  args.seqlen_k = dc.l;
  args.max_seqlen_q = dc.m;
  args.hdim_q = kHeadDim;
  args.hdim_v = kHeadDim;
  args.nhead_q = kQHeads;
  args.nhead_k = kKvHeads;
  args.num_head_q_total = kQHeads;
  args.head_start = 0;
  args.scale_s = kScale;
  args.logits_soft_cap = 0.0F;
  args.stride_q = kHeadDim;
  args.stride_k = kHeadDim;
  args.stride_v = kHeadDim;
  args.stride_o = kHeadDim;
  args.nhead_stride_q = kQHeads * kHeadDim;
  args.nhead_stride_k = kKvHeads * kHeadDim;
  args.nhead_stride_v = kKvHeads * kHeadDim;
  args.nhead_stride_o = kQHeads * kHeadDim;
  args.batch_stride_q = dc.m * kQHeads * kHeadDim;
  args.batch_stride_k = dc.l * kKvHeads * kHeadDim;
  args.batch_stride_v = dc.l * kKvHeads * kHeadDim;
  args.batch_stride_o = dc.m * kQHeads * kHeadDim;
  args.window_size_left = -1;
  args.window_size_right = 0;
  args.sink_size = 0;
  args.mask_type = static_cast<ck_tile::index_t>(
      ck_tile::GenericAttentionMaskEnum::MASK_FROM_BOTTOM_RIGHT);
  args.min_seqlen_q = dc.m;
  args.p_drop = 0.0F;
  args.s_randval = false;
  args.block_scale_size_q = 0;
  args.block_scale_size_kv = 0;

  auto kargs_and_grid = fmha_fwd_create_kargs_and_grids<FmhaKernel>(args);
  auto kargs = kargs_and_grid[ck_tile::number<0>{}];
  auto grid = kargs_and_grid[ck_tile::number<1>{}];
  auto kernel = ck_tile::make_kernel<FmhaKernel::kBlockPerCu>(
      FmhaKernel{}, grid, dim3(FmhaKernel::kBlockSize),
      FmhaKernel::GetSmemSize(), kargs);
  ck_tile::stream_config config{stream};
  kernel(config);
  return hip_ok(hipGetLastError(), "CK Tile FMHA launch");
}

bool launch_pipeline(const DeviceCase &dc, hipStream_t stream) {
  const size_t q_count = static_cast<size_t>(dc.m) * kQHeads * kHeadDim;
  const dim3 blocks(static_cast<unsigned>((q_count + 255U) / 256U));
  hipLaunchKernelGGL(stage_bf16_to_fp16, blocks, dim3(256), 0, stream,
                     dc.q_bf16, dc.q_fp16, q_count);
  if (!hip_ok(hipGetLastError(), "BF16->FP16 staging launch"))
    return false;
  return launch_fmha(dc, stream);
}

float fp16_to_float(const ck_tile::half_t x) { return static_cast<float>(x); }

uint16_t bf16_bits(const ck_tile::bf16_t x) {
  return ck_tile::bit_cast<uint16_t>(x);
}

uint16_t bf16_rne_bits(const float value) {
  return ck_tile::bit_cast<uint16_t>(
      ck_tile::float_to_bf16<ck_tile::bf16_rounding_mode::standard>(value));
}

void fill_inputs(std::vector<ck_tile::bf16_t> &q,
                 std::vector<ck_tile::half_t> &k,
                 std::vector<ck_tile::half_t> &v, const int m, const int l) {
  for (int row = 0; row < m; ++row) {
    for (int head = 0; head < kQHeads; ++head) {
      for (int d = 0; d < kHeadDim; ++d) {
        const float x =
            0.07F * std::sin(0.013F * (row + 1) + 0.017F * (head + 1) +
                             0.003F * (d + 1));
        q[(static_cast<size_t>(row) * kQHeads + head) * kHeadDim + d] =
            ck_tile::float_to_bf16<ck_tile::bf16_rounding_mode::standard>(x);
      }
    }
  }
  for (int row = 0; row < l; ++row) {
    for (int head = 0; head < kKvHeads; ++head) {
      for (int d = 0; d < kHeadDim; ++d) {
        const float kx =
            0.06F * std::cos(0.009F * (row + 1) + 0.011F * (head + 1) +
                             0.005F * (d + 1));
        const float vx =
            0.05F * std::sin(0.007F * (row + 1) + 0.019F * (head + 1) +
                             0.002F * (d + 1));
        k[(static_cast<size_t>(row) * kKvHeads + head) * kHeadDim + d] =
            static_cast<ck_tile::half_t>(kx);
        v[(static_cast<size_t>(row) * kKvHeads + head) * kHeadDim + d] =
            static_cast<ck_tile::half_t>(vx);
      }
    }
  }
}

void oracle(const std::vector<ck_tile::bf16_t> &q,
            const std::vector<ck_tile::half_t> &k,
            const std::vector<ck_tile::half_t> &v, std::vector<uint16_t> &out,
            const int m, const int l, const int start) {
  out.assign(static_cast<size_t>(m) * kQHeads * kHeadDim, 0);
  std::vector<float> scores(static_cast<size_t>(l));
  for (int row = 0; row < m; ++row) {
    const int key_limit = std::min(l - 1, start + row);
    for (int qhead = 0; qhead < kQHeads; ++qhead) {
      const int kvhead = qhead / kGqa;
      const auto *qr =
          q.data() + (static_cast<size_t>(row) * kQHeads + qhead) * kHeadDim;
      float maximum = -std::numeric_limits<float>::infinity();
      for (int col = 0; col <= key_limit; ++col) {
        const auto *kr =
            k.data() +
            (static_cast<size_t>(col) * kKvHeads + kvhead) * kHeadDim;
        float score = 0.0F;
        for (int d = 0; d < kHeadDim; ++d)
          score += static_cast<float>(ck_tile::bf16_to_fp16(qr[d])) *
                   fp16_to_float(kr[d]);
        scores[col] = score * kScale;
        maximum = std::max(maximum, scores[col]);
      }
      float denominator = 0.0F;
      for (int col = 0; col <= key_limit; ++col) {
        scores[col] = std::exp(scores[col] - maximum);
        denominator += scores[col];
      }
      auto *orow =
          out.data() + (static_cast<size_t>(row) * kQHeads + qhead) * kHeadDim;
      for (int d = 0; d < kHeadDim; ++d) {
        float value = 0.0F;
        for (int col = 0; col <= key_limit; ++col) {
          const auto *vr =
              v.data() +
              (static_cast<size_t>(col) * kKvHeads + kvhead) * kHeadDim;
          value += (scores[col] / denominator) * fp16_to_float(vr[d]);
        }
        orow[d] = bf16_rne_bits(value);
      }
    }
  }
}

bool allocate_case(DeviceCase &dc, const int m, const int l, const int start,
                   const std::vector<ck_tile::bf16_t> &q,
                   const std::vector<ck_tile::half_t> &k,
                   const std::vector<ck_tile::half_t> &v) {
  dc.m = m;
  dc.l = l;
  dc.start = start;
  if (!device_alloc(&dc.q_bf16, q.size(), "q_bf16 allocation") ||
      !device_alloc(&dc.q_fp16, q.size(), "q_fp16 allocation") ||
      !device_alloc(&dc.k_fp16, k.size(), "k_fp16 allocation") ||
      !device_alloc(&dc.v_fp16, v.size(), "v_fp16 allocation") ||
      !device_alloc(&dc.o_bf16, q.size(), "o_bf16 allocation") ||
      !device_alloc(&dc.seqstart_q, 2, "seqstart_q allocation") ||
      !device_alloc(&dc.seqstart_k, 2, "seqstart_k allocation"))
    return false;
  if (!hip_ok(hipMemcpy(dc.q_bf16, q.data(), q.size() * sizeof(q[0]),
                        hipMemcpyHostToDevice),
              "copy q") ||
      !hip_ok(hipMemcpy(dc.k_fp16, k.data(), k.size() * sizeof(k[0]),
                        hipMemcpyHostToDevice),
              "copy k") ||
      !hip_ok(hipMemcpy(dc.v_fp16, v.data(), v.size() * sizeof(v[0]),
                        hipMemcpyHostToDevice),
              "copy v"))
    return false;
  const std::array<int32_t, 2> sq{0, m};
  const std::array<int32_t, 2> sk{0, l};
  return hip_ok(hipMemcpy(dc.seqstart_q, sq.data(), sizeof(sq),
                          hipMemcpyHostToDevice),
                "copy seqstart_q") &&
         hip_ok(hipMemcpy(dc.seqstart_k, sk.data(), sizeof(sk),
                          hipMemcpyHostToDevice),
                "copy seqstart_k");
}

void release_case(DeviceCase &dc) {
  // Cleanup is best effort after an earlier error, but every allocation is
  // explicitly released so the probe can be repeated in one process.
  device_free(dc.q_bf16, "free q_bf16");
  device_free(dc.q_fp16, "free q_fp16");
  device_free(dc.k_fp16, "free k_fp16");
  device_free(dc.v_fp16, "free v_fp16");
  device_free(dc.o_bf16, "free o_bf16");
  device_free(dc.seqstart_q, "free seqstart_q");
  device_free(dc.seqstart_k, "free seqstart_k");
  dc = {};
}

bool run_case(const int m, const int l, const int start,
              const bool check_oracle) {
  std::vector<ck_tile::bf16_t> q(static_cast<size_t>(m) * kQHeads * kHeadDim);
  std::vector<ck_tile::half_t> k(static_cast<size_t>(l) * kKvHeads * kHeadDim);
  std::vector<ck_tile::half_t> v(static_cast<size_t>(l) * kKvHeads * kHeadDim);
  fill_inputs(q, k, v, m, l);
  DeviceCase dc;
  if (!allocate_case(dc, m, l, start, q, k, v)) {
    release_case(dc);
    return false;
  }
  hipStream_t stream = nullptr;
  if (!hip_ok(hipStreamCreate(&stream), "stream create")) {
    release_case(dc);
    return false;
  }
  bool ok = true;
  for (int i = 0; i < kWarmups && ok; ++i)
    ok = launch_pipeline(dc, stream);
  hipEvent_t begin = nullptr;
  hipEvent_t end = nullptr;
  ok = ok && hip_ok(hipEventCreate(&begin), "begin event create") &&
       hip_ok(hipEventCreate(&end), "end event create");
  float total_ms = 0.0F;
  if (ok) {
    for (int i = 0; i < kMeasured; ++i) {
      ok = hip_ok(hipEventRecord(begin, stream), "begin event record") &&
           launch_pipeline(dc, stream) &&
           hip_ok(hipEventRecord(end, stream), "end event record") &&
           hip_ok(hipEventSynchronize(end), "end event synchronize");
      float elapsed = 0.0F;
      ok = ok &&
           hip_ok(hipEventElapsedTime(&elapsed, begin, end), "event elapsed");
      total_ms += elapsed;
    }
  }
  if (ok && check_oracle) {
    std::vector<ck_tile::bf16_t> got(q.size());
    std::vector<uint16_t> expected;
    oracle(q, k, v, expected, m, l, start);
    ok = hip_ok(hipMemcpyAsync(got.data(), dc.o_bf16,
                               got.size() * sizeof(got[0]),
                               hipMemcpyDeviceToHost, stream),
                "copy output") &&
         hip_ok(hipStreamSynchronize(stream), "output synchronize");
    size_t mismatches = 0;
    uint16_t max_xor = 0;
    if (ok) {
      for (size_t i = 0; i < got.size(); ++i) {
        const uint16_t actual = bf16_bits(got[i]);
        if (actual != expected[i]) {
          ++mismatches;
          max_xor = std::max<uint16_t>(
              max_xor, static_cast<uint16_t>(actual ^ expected[i]));
        }
      }
    }
    std::printf("CK probe oracle M=%d L=%d start=%d mismatches=%zu/%zu "
                "max_xor=0x%04x\n",
                m, l, start, mismatches, got.size(), max_xor);
  }
  if (ok)
    std::printf("CK probe tile=64x64 K0=32 K1=32 M=%d L=%d start=%d "
                "avg_ms=%.6f (3+10)\n",
                m, l, start, total_ms / static_cast<float>(kMeasured));
  if (end != nullptr)
    (void)hip_ok(hipEventDestroy(end), "destroy end event");
  if (begin != nullptr)
    (void)hip_ok(hipEventDestroy(begin), "destroy begin event");
  (void)hip_ok(hipStreamDestroy(stream), "stream destroy");
  release_case(dc);
  return ok;
}

void print_resources() {
  hipDeviceProp_t props{};
  if (hipGetDeviceProperties(&props, 0) == hipSuccess) {
    std::printf(
        "CK probe GPU name=%s arch=%s sharedMemPerBlock=%zu regsPerBlock=%d\n",
        props.name, props.gcnArchName, props.sharedMemPerBlock,
        props.regsPerBlock);
  }
  std::printf("CK probe resources block_size=%d smem=%zu block_per_cu=%d\n",
              FmhaKernel::kBlockSize,
              static_cast<size_t>(FmhaKernel::GetSmemSize()),
              FmhaKernel::kBlockPerCu);
}

} // namespace

int main(int argc, char **argv) {
  bool execute = false;
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--run")
      execute = true;
  std::printf("CK Tile gfx1201 GQA6 probe: Q BF16->FP16, K/V FP16, O BF16, "
              "group QRKSVS, tile 64x64/K0=K1=32, row-major V\n");
  std::printf("requested QRKSVSAsync is N0 on ROCm 7.14 gfx1201: unsupported "
              "async dwordx4 ingress for this shape\n");
  std::printf("cases: M=L=1024 start=0; M=1024 L=9435 start=8411 (bottom-right "
              "causal)\n");
  if (!execute) {
    std::printf("dry-run: pass --run only after explicit GPU authorization\n");
    return 0;
  }
  if (!hip_ok(hipSetDevice(0), "set device"))
    return 1;
  print_resources();
  // The small non-aligned case exercises edge padding and is the only case
  // compared against the host oracle by default; exact cases report timing.
  if (!run_case(8, 17, 9, true))
    return 1;
  if (!run_case(1024, 1024, 0, false))
    return 1;
  if (!run_case(1024, 9435, 8411, false))
    return 1;
  return 0;
}
