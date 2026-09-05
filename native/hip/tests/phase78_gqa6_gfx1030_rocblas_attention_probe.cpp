// Phase 78 standalone GQA6 rocBLAS attention feasibility probe (gfx1030).
//
// This file is intentionally independent of the production sources.  It keeps
// the production FP16-KV/Qtile4/K32 kernel as a control and measures two
// explicit-GEMM pipelines:
//   * FP16 Q/K/V, FP32 rocBLAS accumulation, FP16 score/probabilities;
//   * F32 Q/K/V staging, SGEMM QK/PV, F32 score/probabilities.
//
// K/V are resident in the production interleaved layout [token,4,256].  The
// GEMMs use a pointer at each KV head, lda=4*256 and batch stride=256.  The
// score matrix is column-major [context,q_count], which is row-major
// [q_count,context] to the softmax kernel.  The output is BF16-RNE.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

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

constexpr uint32_t kThreads = 256U;
constexpr uint32_t kWave = 32U;
constexpr uint32_t kQHeads = 24U;
constexpr uint32_t kKvHeads = 4U;
constexpr uint32_t kGqa = 6U;
constexpr uint32_t kD = 256U;
constexpr uint32_t kWarmups = 3U;
constexpr uint32_t kMeasured = 10U;

__device__ __forceinline__ float bf16f(uint16_t x) {
  return __uint_as_float(static_cast<uint32_t>(x) << 16U);
}
__device__ __forceinline__ float half_f(uint16_t x) {
  return __half2float(__ushort_as_half(x));
}
__device__ __forceinline__ uint16_t bf16_rne(float x) {
  const uint32_t bits = __float_as_uint(x);
  if ((bits & 0x7f800000U) == 0x7f800000U) {
    if ((bits & 0x007fffffU) != 0U)
      return static_cast<uint16_t>(((bits >> 16U) & 0x8000U) | 0x7fc0U |
                                   ((bits >> 16U) & 0x003fU));
    return static_cast<uint16_t>(bits >> 16U);
  }
  uint32_t upper = bits >> 16U;
  const uint32_t lower = bits & 0xffffU;
  if (lower > 0x8000U || (lower == 0x8000U && (upper & 1U)))
    ++upper;
  return static_cast<uint16_t>(upper);
}

// Exact production control: one block owns four rows of one KV head, with
// FP32 online softmax and interleaved FP16 K/V.
__global__ __launch_bounds__(256, 1) void qtile4_k32_control(
    const uint16_t *q, const uint16_t *k, const uint16_t *v, uint16_t *out,
    uint32_t rows, uint64_t start) {
  constexpr uint32_t kTile = 4U;
  constexpr uint32_t kLogical = 24U;
  constexpr uint32_t kPerWave = 3U;
  constexpr uint32_t kPairs = 4U;
  const uint64_t flat = blockIdx.x;
  const uint64_t tile = flat / kKvHeads;
  const uint32_t kv = static_cast<uint32_t>(flat % kKvHeads);
  const uint64_t first = tile * kTile;
  if (first >= rows)
    return;
  const uint32_t t = threadIdx.x;
  const uint32_t lane = t & (kWave - 1U);
  const uint32_t wave = t / kWave;
  const uint32_t first_head = kv * kGqa;
  __shared__ uint16_t kt[32][kD];
  __shared__ uint16_t vt[32][kD];
  float2 qv[kPerWave][kPairs];
  float2 acc[kPerWave][kPairs] = {};
  float rmax[kPerWave];
  float rden[kPerWave];
#pragma unroll
  for (uint32_t item = 0; item < kPerWave; ++item) {
    rmax[item] = -INFINITY;
    rden[item] = 0.0F;
    const uint32_t logical = wave * kPerWave + item;
    const uint64_t row = first + logical / kGqa;
    const uint32_t head = first_head + logical % kGqa;
    const uint64_t safe = row < rows ? row : rows - 1U;
    const uint16_t *qr = q + (safe * kQHeads + head) * kD;
#pragma unroll
    for (uint32_t pair = 0; pair < kPairs; ++pair) {
      const uint32_t d = lane * 2U + pair * kWave * 2U;
      qv[item][pair] = row < rows ? make_float2(bf16f(qr[d]), bf16f(qr[d + 1U]))
                                  : make_float2(0.0F, 0.0F);
    }
  }
  const uint64_t end = first + kTile;
  const uint64_t last = (end < rows ? end : rows) - 1U;
  const uint64_t last_key = start + last;
  for (uint64_t begin = 0; begin <= last_key; begin += 32U) {
    const uint32_t count =
        static_cast<uint32_t>(std::min<uint64_t>(32U, last_key - begin + 1U));
    for (uint32_t i = t; i < 32U * kD; i += kThreads) {
      const uint32_t pos = i / kD;
      const uint32_t d = i % kD;
      if (pos < count) {
        const uint64_t src = (begin + pos) * kKvHeads + kv;
        kt[pos][d] = k[src * kD + d];
        vt[pos][d] = v[src * kD + d];
      }
    }
    __syncthreads();
    for (uint32_t pos = 0; pos < count; ++pos) {
      const uint64_t key_pos = begin + pos;
      for (uint32_t item = 0; item < kPerWave; ++item) {
        const uint32_t logical = wave * kPerWave + item;
        const uint64_t row = first + logical / kGqa;
        const bool active = row < rows && key_pos <= start + row;
        float dot = 0.0F;
#pragma unroll
        for (uint32_t pair = 0; pair < kPairs; ++pair) {
          const uint32_t d = lane * 2U + pair * kWave * 2U;
          const float2 kval =
              make_float2(half_f(kt[pos][d]), half_f(kt[pos][d + 1U]));
          const float2 qval = qv[item][pair];
          dot += active ? qval.x * kval.x + qval.y * kval.y : 0.0F;
        }
#pragma unroll
        for (uint32_t off = kWave / 2U; off; off >>= 1U)
          dot += __shfl_down(dot, off, kWave);
        float rescale = 1.0F;
        float contribution = 0.0F;
        float next = rmax[item];
        if (lane == 0U && active) {
          const float score = dot * rsqrtf(static_cast<float>(kD));
          next = fmaxf(rmax[item], score);
          rescale = expf(rmax[item] - next);
          contribution = expf(score - next);
        }
        rescale = __shfl(rescale, 0, kWave);
        contribution = __shfl(contribution, 0, kWave);
        next = __shfl(next, 0, kWave);
        rden[item] = rden[item] * rescale + contribution;
        rmax[item] = next;
#pragma unroll
        for (uint32_t pair = 0; pair < kPairs; ++pair) {
          const uint32_t d = lane * 2U + pair * kWave * 2U;
          if (active) {
            const float2 vv =
                make_float2(half_f(vt[pos][d]), half_f(vt[pos][d + 1U]));
            acc[item][pair].x =
                acc[item][pair].x * rescale + contribution * vv.x;
            acc[item][pair].y =
                acc[item][pair].y * rescale + contribution * vv.y;
          }
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (uint32_t item = 0; item < kPerWave; ++item) {
    const uint32_t logical = wave * kPerWave + item;
    const uint64_t row = first + logical / kGqa;
    const uint32_t head = first_head + logical % kGqa;
    if (row < rows) {
      uint16_t *dst = out + (row * kQHeads + head) * kD;
#pragma unroll
      for (uint32_t pair = 0; pair < kPairs; ++pair) {
        const uint32_t d = lane * 2U + pair * kWave * 2U;
        dst[d] = bf16_rne(acc[item][pair].x / rden[item]);
        dst[d + 1U] = bf16_rne(acc[item][pair].y / rden[item]);
      }
    }
  }
}

// Pack BF16 [M,24,D] into four natural row-major [M*6,D] matrices, which are
// column-major D x (M*6) to rocBLAS.
__global__ void pack_query_f16(const uint16_t *q, uint16_t *dst,
                               uint32_t rows) {
  const uint64_t i =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total = static_cast<uint64_t>(kKvHeads) * rows * kGqa * kD;
  if (i >= total)
    return;
  const uint64_t per = static_cast<uint64_t>(rows) * kGqa * kD;
  const uint32_t batch = static_cast<uint32_t>(i / per);
  const uint64_t local = i % per;
  const uint32_t qi = static_cast<uint32_t>(local / kD);
  const uint32_t d = static_cast<uint32_t>(local % kD);
  const uint32_t row = qi / kGqa;
  const uint32_t head = batch * kGqa + qi % kGqa;
  const float x =
      bf16f(q[(static_cast<uint64_t>(row) * kQHeads + head) * kD + d]);
  // Preserve the batch in the packed allocation.  `local` is only used for
  // the source-row mapping; using it as the destination index would make the
  // four batches race and leave only the last KV-head group in every matrix.
  dst[i] = __half_as_ushort(__float2half(x));
}

__global__ void pack_query_f32(const uint16_t *q, float *dst, uint32_t rows) {
  const uint64_t i =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total = static_cast<uint64_t>(kKvHeads) * rows * kGqa * kD;
  if (i >= total)
    return;
  const uint64_t per = static_cast<uint64_t>(rows) * kGqa * kD;
  const uint32_t batch = static_cast<uint32_t>(i / per);
  const uint64_t local = i % per;
  const uint32_t qi = static_cast<uint32_t>(local / kD);
  const uint32_t d = static_cast<uint32_t>(local % kD);
  const uint32_t row = qi / kGqa;
  const uint32_t head = batch * kGqa + qi % kGqa;
  dst[i] = bf16f(q[(static_cast<uint64_t>(row) * kQHeads + head) * kD + d]);
}

__global__ void convert_kv(const uint16_t *src, float *dst, uint64_t count) {
  const uint64_t i =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    dst[i] = half_f(src[i]);
}

template <typename T> __device__ __forceinline__ float score_to_float(T x) {
  return static_cast<float>(x);
}
template <> __device__ __forceinline__ float score_to_float<__half>(__half x) {
  return __half2float(x);
}
template <typename T> __device__ __forceinline__ T float_to_score(float x) {
  return static_cast<T>(x);
}
template <> __device__ __forceinline__ __half float_to_score<__half>(float x) {
  return __float2half(x);
}

template <typename T>
__global__ void causal_softmax(T *scores, uint32_t rows, uint64_t start,
                               uint64_t context) {
  const uint64_t flat = blockIdx.x;
  const uint64_t qcount = static_cast<uint64_t>(rows) * kGqa;
  const uint64_t batch = flat / qcount;
  const uint64_t qi = flat % qcount;
  const uint64_t row = qi / kGqa;
  if (batch >= kKvHeads || row >= rows)
    return;
  const uint64_t limit = std::min<uint64_t>(context - 1U, start + row);
  const uint64_t base = batch * qcount * context + qi * context;
  __shared__ float partials[kThreads / kWave];
  const uint32_t t = threadIdx.x;
  const uint32_t lane = t & (kWave - 1U);
  const uint32_t wave = t / kWave;
  float mx = -INFINITY;
  for (uint64_t pos = t; pos <= limit; pos += kThreads)
    mx = fmaxf(mx, score_to_float(scores[base + pos]));
  for (uint32_t off = kWave / 2U; off; off >>= 1U)
    mx = fmaxf(mx, __shfl_down(mx, off, kWave));
  if (lane == 0U)
    partials[wave] = mx;
  __syncthreads();
  if (wave == 0U) {
    mx = lane < kThreads / kWave ? partials[lane] : -INFINITY;
    for (uint32_t off = kWave / 2U; off; off >>= 1U)
      mx = fmaxf(mx, __shfl_down(mx, off, kWave));
    if (lane == 0U)
      partials[0] = mx;
  }
  __syncthreads();
  mx = partials[0];
  float den = 0.0F;
  for (uint64_t pos = t; pos <= limit; pos += kThreads) {
    const float p = expf(score_to_float(scores[base + pos]) - mx);
    scores[base + pos] = float_to_score<T>(p);
    den += p;
  }
  for (uint32_t off = kWave / 2U; off; off >>= 1U)
    den += __shfl_down(den, off, kWave);
  if (lane == 0U)
    partials[wave] = den;
  __syncthreads();
  if (wave == 0U) {
    den = lane < kThreads / kWave ? partials[lane] : 0.0F;
    for (uint32_t off = kWave / 2U; off; off >>= 1U)
      den += __shfl_down(den, off, kWave);
    if (lane == 0U)
      partials[0] = den;
  }
  __syncthreads();
  den = partials[0];
  for (uint64_t pos = t; pos < context; pos += kThreads)
    scores[base + pos] =
        pos <= limit
            ? float_to_score<T>(score_to_float(scores[base + pos]) / den)
            : float_to_score<T>(0.0F);
}

__global__ void unpack_bf16(const float *pv, uint16_t *out, uint32_t rows) {
  const uint64_t i =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total = static_cast<uint64_t>(kKvHeads) * rows * kGqa * kD;
  if (i >= total)
    return;
  const uint64_t per = static_cast<uint64_t>(rows) * kGqa * kD;
  const uint32_t batch = static_cast<uint32_t>(i / per);
  const uint64_t local = i - static_cast<uint64_t>(batch) * per;
  const uint32_t qi = static_cast<uint32_t>(local / kD);
  const uint32_t d = static_cast<uint32_t>(local % kD);
  const uint32_t row = qi / kGqa;
  const uint32_t head = batch * kGqa + qi % kGqa;
  const uint64_t qcount = static_cast<uint64_t>(rows) * kGqa;
  out[(static_cast<uint64_t>(row) * kQHeads + head) * kD + d] =
      bf16_rne(pv[static_cast<uint64_t>(batch) * qcount * kD +
                  static_cast<uint64_t>(qi) * kD + d]);
}

bool hip_ok(hipError_t s, const char *what) {
  if (s == hipSuccess)
    return true;
  std::fprintf(stderr, "hip_error op=%s status=%s\n", what,
               hipGetErrorString(s));
  return false;
}
bool rb_ok(rocblas_status s, const char *what) {
  if (s == rocblas_status_success)
    return true;
  std::fprintf(stderr, "rocblas_error op=%s status=%d\n", what,
               static_cast<int>(s));
  return false;
}

struct Base final {
  uint16_t *q = nullptr, *k = nullptr, *v = nullptr, *out = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr, stop = nullptr;
};
void free_base(Base *b) {
  if (!b)
    return;
  if (b->stop)
    (void)hipEventDestroy(b->stop);
  if (b->start)
    (void)hipEventDestroy(b->start);
  if (b->stream)
    (void)hipStreamDestroy(b->stream);
  if (b->out)
    (void)hipFree(b->out);
  if (b->v)
    (void)hipFree(b->v);
  if (b->k)
    (void)hipFree(b->k);
  if (b->q)
    (void)hipFree(b->q);
  *b = {};
}
bool make_base(uint64_t rows, uint64_t context, Base *b) {
  const size_t qb = static_cast<size_t>(rows * kQHeads * kD * 2U);
  const size_t kb = static_cast<size_t>(context * kKvHeads * kD * 2U);
  if (!hip_ok(hipMalloc(reinterpret_cast<void **>(&b->q), qb), "malloc q") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&b->k), kb), "malloc k") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&b->v), kb), "malloc v") ||
      !hip_ok(hipMalloc(reinterpret_cast<void **>(&b->out), qb),
              "malloc out") ||
      !hip_ok(hipStreamCreate(&b->stream), "stream") ||
      !hip_ok(hipEventCreate(&b->start), "event start") ||
      !hip_ok(hipEventCreate(&b->stop), "event stop")) {
    free_base(b);
    return false;
  }
  return true;
}

uint16_t hbf(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  uint32_t hi = u >> 16U, lo = u & 0xffffU;
  if (lo > 0x8000U || (lo == 0x8000U && (hi & 1U)))
    ++hi;
  return static_cast<uint16_t>(hi);
}
uint16_t hhalf(float x) { return __half_as_ushort(__float2half(x)); }
float fbf(uint16_t x) {
  uint32_t u = static_cast<uint32_t>(x) << 16U;
  float y;
  std::memcpy(&y, &u, 4);
  return y;
}
float fhalf(uint16_t x) {
  const uint32_t sign = static_cast<uint32_t>(x & 0x8000U) << 16U;
  const uint32_t e = (x >> 10U) & 31U, m = x & 1023U;
  uint32_t u = sign;
  if (e == 0U) {
    if (m) {
      float y = static_cast<float>(m) * 0x1p-24F;
      std::memcpy(&u, &y, 4);
      u = (u & 0x7fffffffU) | sign;
    }
  } else if (e == 31U)
    u |= 0x7f800000U | (m << 13U);
  else
    u |= ((e + 112U) << 23U) | (m << 13U);
  float y;
  std::memcpy(&y, &u, 4);
  return y;
}
uint32_t ulp(uint16_t a, uint16_t b) {
  if ((a & 0x7fffU) == 0U && (b & 0x7fffU) == 0U)
    return 0;
  const int32_t aa = (a & 0x8000U) ? 0x8000 - (a & 0x7fffU) : 0x8000 + a;
  const int32_t bb = (b & 0x8000U) ? 0x8000 - (b & 0x7fffU) : 0x8000 + b;
  return static_cast<uint32_t>(std::abs(aa - bb));
}

void fill(uint64_t rows, uint64_t context, std::vector<uint16_t> *q,
          std::vector<uint16_t> *k, std::vector<uint16_t> *v) {
  q->resize(rows * kQHeads * kD);
  k->resize(context * kKvHeads * kD);
  v->resize(context * kKvHeads * kD);
  for (uint64_t r = 0; r < rows; ++r)
    for (uint32_t h = 0; h < kQHeads; ++h)
      for (uint32_t d = 0; d < kD; ++d)
        (*q)[(r * kQHeads + h) * kD + d] =
            hbf(0.45F *
                std::sin(static_cast<float>((r * 7U + h * 13U + d * 3U) % 97U) *
                         0.071F));
  for (uint64_t r = 0; r < context; ++r)
    for (uint32_t h = 0; h < kKvHeads; ++h)
      for (uint32_t d = 0; d < kD; ++d) {
        (*k)[(r * kKvHeads + h) * kD + d] = hhalf(
            0.65F *
            std::cos(static_cast<float>((r * 11U + h * 5U + d * 17U) % 131U) *
                     0.053F));
        (*v)[(r * kKvHeads + h) * kD + d] = hhalf(
            0.55F *
            std::sin(static_cast<float>((r * 3U + h * 19U + d * 7U) % 127U) *
                     0.067F));
      }
}
bool upload(const std::vector<uint16_t> &q, const std::vector<uint16_t> &k,
            const std::vector<uint16_t> &v, Base *b) {
  return hip_ok(hipMemcpy(b->q, q.data(), q.size() * 2U, hipMemcpyHostToDevice),
                "copy q") &&
         hip_ok(hipMemcpy(b->k, k.data(), k.size() * 2U, hipMemcpyHostToDevice),
                "copy k") &&
         hip_ok(hipMemcpy(b->v, v.data(), v.size() * 2U, hipMemcpyHostToDevice),
                "copy v");
}

bool oracle(uint64_t rows, uint64_t context, uint64_t start,
            const std::vector<uint16_t> &q, const std::vector<uint16_t> &k,
            const std::vector<uint16_t> &v, std::vector<uint16_t> *out) {
  out->assign(rows * kQHeads * kD, 0U);
  const float scale = 1.0F / std::sqrt(static_cast<float>(kD));
  for (uint64_t r = 0; r < rows; ++r) {
    const uint64_t last = std::min<uint64_t>(context - 1U, start + r);
    for (uint32_t h = 0; h < kQHeads; ++h) {
      const uint32_t kv = h / kGqa;
      std::vector<float> p(last + 1U);
      float mx = -INFINITY;
      for (uint64_t pos = 0; pos <= last; ++pos) {
        float dot = 0.0F;
        const uint16_t *qr = q.data() + (r * kQHeads + h) * kD;
        const uint16_t *kr = k.data() + (pos * kKvHeads + kv) * kD;
        for (uint32_t d = 0; d < kD; ++d)
          dot = std::fmaf(fbf(qr[d]), fhalf(kr[d]), dot);
        p[pos] = dot * scale;
        mx = std::max(mx, p[pos]);
      }
      float den = 0.0F;
      for (float &x : p) {
        x = std::exp(x - mx);
        den += x;
      }
      uint16_t *dst = out->data() + (r * kQHeads + h) * kD;
      for (uint32_t d = 0; d < kD; ++d) {
        float x = 0.0F;
        for (uint64_t pos = 0; pos <= last; ++pos)
          x += p[pos] / den * fhalf(v[(pos * kKvHeads + kv) * kD + d]);
        dst[d] = hbf(x);
      }
    }
  }
  return true;
}

struct Pipe final {
  enum Kind { F16, F32 } kind;
  uint16_t *q16 = nullptr, *score16 = nullptr;
  float *q32 = nullptr, *key32 = nullptr, *value32 = nullptr,
        *score32 = nullptr, *pv = nullptr;
  hipEvent_t start = nullptr, stop = nullptr;
  rocblas_handle handle = nullptr;
};
void free_pipe(Pipe *p) {
  if (!p)
    return;
  if (p->handle)
    (void)rocblas_destroy_handle(p->handle);
  if (p->stop)
    (void)hipEventDestroy(p->stop);
  if (p->start)
    (void)hipEventDestroy(p->start);
  if (p->pv)
    (void)hipFree(p->pv);
  if (p->score32)
    (void)hipFree(p->score32);
  if (p->value32)
    (void)hipFree(p->value32);
  if (p->key32)
    (void)hipFree(p->key32);
  if (p->q32)
    (void)hipFree(p->q32);
  if (p->score16)
    (void)hipFree(p->score16);
  if (p->q16)
    (void)hipFree(p->q16);
  *p = {};
}
bool make_pipe(uint64_t rows, uint64_t context, Pipe::Kind kind,
               hipStream_t stream, Pipe *p) {
  p->kind = kind;
  const uint64_t qc = rows * kGqa, batches = kKvHeads;
  const size_t q16b = batches * qc * kD * 2U;
  const size_t q32b = batches * qc * kD * 4U;
  const size_t score16b = batches * qc * context * 2U;
  const size_t score32b = batches * qc * context * 4U;
  const size_t pvb = batches * qc * kD * 4U;
  const size_t kv32b = batches * context * kD * 4U;
  bool ok = true;
  if (kind == Pipe::F16) {
    ok = hip_ok(hipMalloc(reinterpret_cast<void **>(&p->q16), q16b),
                "malloc p q16") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&p->score16), score16b),
                "malloc p score16");
  } else {
    ok = hip_ok(hipMalloc(reinterpret_cast<void **>(&p->q32), q32b),
                "malloc p q32") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&p->key32), kv32b),
                "malloc p key32") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&p->value32), kv32b),
                "malloc p value32") &&
         hip_ok(hipMalloc(reinterpret_cast<void **>(&p->score32), score32b),
                "malloc p score32");
  }
  ok = ok &&
       hip_ok(hipMalloc(reinterpret_cast<void **>(&p->pv), pvb),
              "malloc p pv") &&
       hip_ok(hipEventCreate(&p->start), "pipe event start") &&
       hip_ok(hipEventCreate(&p->stop), "pipe event stop") &&
       rb_ok(rocblas_create_handle(&p->handle), "create handle") &&
       rb_ok(rocblas_set_stream(p->handle, stream), "set stream");
  if (!ok) {
    free_pipe(p);
    return false;
  }
  return true;
}

bool stage_f32(Base *b, Pipe *p, uint64_t context, float *elapsed_us) {
  const uint64_t total = context * kKvHeads * kD;
  const uint32_t blocks =
      static_cast<uint32_t>((total + kThreads - 1U) / kThreads);
  if (!hip_ok(hipEventRecord(p->start, b->stream), "stage event start"))
    return false;
  hipLaunchKernelGGL(convert_kv, dim3(blocks), dim3(kThreads), 0U, b->stream,
                     b->k, p->key32, total);
  hipLaunchKernelGGL(convert_kv, dim3(blocks), dim3(kThreads), 0U, b->stream,
                     b->v, p->value32, total);
  if (!hip_ok(hipGetLastError(), "stage kv launch") ||
      !hip_ok(hipEventRecord(p->stop, b->stream), "stage event stop") ||
      !hip_ok(hipEventSynchronize(p->stop), "stage sync") ||
      !hip_ok(hipEventElapsedTime(elapsed_us, p->start, p->stop),
              "stage elapsed"))
    return false;
  *elapsed_us *= 1000.0F;
  return true;
}

bool launch_pipe(Base *b, Pipe *p, uint64_t rows, uint64_t context,
                 uint64_t start) {
  const uint64_t qc = rows * kGqa;
  const uint64_t totalq = kKvHeads * qc * kD;
  const uint32_t blocks =
      static_cast<uint32_t>((totalq + kThreads - 1U) / kThreads);
  const float alpha = 1.0F / std::sqrt(static_cast<float>(kD));
  const float one = 1.0F, zero = 0.0F;
  const rocblas_stride qstride = static_cast<rocblas_stride>(qc * kD);
  const rocblas_stride kstride = static_cast<rocblas_stride>(kD);
  const rocblas_stride sstride = static_cast<rocblas_stride>(qc * context);
  const rocblas_stride pvstride = static_cast<rocblas_stride>(qc * kD);
  if (p->kind == Pipe::F16) {
    hipLaunchKernelGGL(pack_query_f16, dim3(blocks), dim3(kThreads), 0U,
                       b->stream, b->q, p->q16, static_cast<uint32_t>(rows));
    // The explicit reinterpret cast above is safe: __half is two bytes and
    // p->q16 is only used as raw FP16 by rocBLAS.
    if (!hip_ok(hipGetLastError(), "pack f16"))
      return false;
    if (!rb_ok(rocblas_gemm_strided_batched_ex(
                   p->handle, rocblas_operation_transpose,
                   rocblas_operation_none, static_cast<rocblas_int>(context),
                   static_cast<rocblas_int>(qc), static_cast<rocblas_int>(kD),
                   &alpha, b->k, rocblas_datatype_f16_r,
                   static_cast<rocblas_int>(kKvHeads * kD), kstride, p->q16,
                   rocblas_datatype_f16_r, static_cast<rocblas_int>(kD),
                   qstride, &zero, p->score16, rocblas_datatype_f16_r,
                   static_cast<rocblas_int>(context), sstride, p->score16,
                   rocblas_datatype_f16_r, static_cast<rocblas_int>(context),
                   sstride, kKvHeads, rocblas_datatype_f32_r,
                   rocblas_gemm_algo_standard, 0, 0U),
               "QK f16 gemm"))
      return false;
    hipLaunchKernelGGL((causal_softmax<__half>), dim3(kKvHeads * qc),
                       dim3(kThreads), 0U, b->stream,
                       reinterpret_cast<__half *>(p->score16),
                       static_cast<uint32_t>(rows), start, context);
    if (!hip_ok(hipGetLastError(), "softmax f16"))
      return false;
    if (!rb_ok(rocblas_gemm_strided_batched_ex(
                   p->handle, rocblas_operation_none, rocblas_operation_none,
                   static_cast<rocblas_int>(kD), static_cast<rocblas_int>(qc),
                   static_cast<rocblas_int>(context), &one, b->v,
                   rocblas_datatype_f16_r,
                   static_cast<rocblas_int>(kKvHeads * kD), kstride, p->score16,
                   rocblas_datatype_f16_r, static_cast<rocblas_int>(context),
                   sstride, &zero, p->pv, rocblas_datatype_f32_r,
                   static_cast<rocblas_int>(kD), pvstride, p->pv,
                   rocblas_datatype_f32_r, static_cast<rocblas_int>(kD),
                   pvstride, kKvHeads, rocblas_datatype_f32_r,
                   rocblas_gemm_algo_standard, 0, 0U),
               "PV f16 gemm"))
      return false;
  } else {
    hipLaunchKernelGGL(pack_query_f32, dim3(blocks), dim3(kThreads), 0U,
                       b->stream, b->q, p->q32, static_cast<uint32_t>(rows));
    if (!hip_ok(hipGetLastError(), "pack f32"))
      return false;
    if (!rb_ok(rocblas_gemm_strided_batched_ex(
                   p->handle, rocblas_operation_transpose,
                   rocblas_operation_none, static_cast<rocblas_int>(context),
                   static_cast<rocblas_int>(qc), static_cast<rocblas_int>(kD),
                   &alpha, p->key32, rocblas_datatype_f32_r,
                   static_cast<rocblas_int>(kKvHeads * kD), kstride, p->q32,
                   rocblas_datatype_f32_r, static_cast<rocblas_int>(kD),
                   qstride, &zero, p->score32, rocblas_datatype_f32_r,
                   static_cast<rocblas_int>(context), sstride, p->score32,
                   rocblas_datatype_f32_r, static_cast<rocblas_int>(context),
                   sstride, kKvHeads, rocblas_datatype_f32_r,
                   rocblas_gemm_algo_standard, 0, 0U),
               "QK f32 gemm"))
      return false;
    hipLaunchKernelGGL((causal_softmax<float>), dim3(kKvHeads * qc),
                       dim3(kThreads), 0U, b->stream, p->score32,
                       static_cast<uint32_t>(rows), start, context);
    if (!hip_ok(hipGetLastError(), "softmax f32"))
      return false;
    if (!rb_ok(rocblas_gemm_strided_batched_ex(
                   p->handle, rocblas_operation_none, rocblas_operation_none,
                   static_cast<rocblas_int>(kD), static_cast<rocblas_int>(qc),
                   static_cast<rocblas_int>(context), &one, p->value32,
                   rocblas_datatype_f32_r,
                   static_cast<rocblas_int>(kKvHeads * kD), kstride, p->score32,
                   rocblas_datatype_f32_r, static_cast<rocblas_int>(context),
                   sstride, &zero, p->pv, rocblas_datatype_f32_r,
                   static_cast<rocblas_int>(kD), pvstride, p->pv,
                   rocblas_datatype_f32_r, static_cast<rocblas_int>(kD),
                   pvstride, kKvHeads, rocblas_datatype_f32_r,
                   rocblas_gemm_algo_standard, 0, 0U),
               "PV f32 gemm"))
      return false;
  }
  hipLaunchKernelGGL(unpack_bf16, dim3(blocks), dim3(kThreads), 0U, b->stream,
                     p->pv, b->out, static_cast<uint32_t>(rows));
  return hip_ok(hipGetLastError(), "unpack");
}

bool measure_control(Base *b, uint64_t rows, uint64_t context, uint64_t start,
                     float *us) {
  const uint32_t blocks = static_cast<uint32_t>((rows + 3U) / 4U * kKvHeads);
  for (uint32_t i = 0; i < kWarmups; ++i) {
    hipLaunchKernelGGL(qtile4_k32_control, dim3(blocks), dim3(kThreads), 0U,
                       b->stream, b->q, b->k, b->v, b->out,
                       static_cast<uint32_t>(rows), start);
    if (!hip_ok(hipGetLastError(), "control launch") ||
        !hip_ok(hipStreamSynchronize(b->stream), "control warmup"))
      return false;
  }
  std::array<float, kMeasured> s{};
  for (float &x : s) {
    if (!hip_ok(hipEventRecord(b->start, b->stream), "control event start"))
      return false;
    hipLaunchKernelGGL(qtile4_k32_control, dim3(blocks), dim3(kThreads), 0U,
                       b->stream, b->q, b->k, b->v, b->out,
                       static_cast<uint32_t>(rows), start);
    if (!hip_ok(hipGetLastError(), "control launch measure") ||
        !hip_ok(hipEventRecord(b->stop, b->stream), "control event stop") ||
        !hip_ok(hipEventSynchronize(b->stop), "control event sync") ||
        !hip_ok(hipEventElapsedTime(&x, b->start, b->stop), "control elapsed"))
      return false;
    x *= 1000.0F;
  }
  std::sort(s.begin(), s.end());
  *us = s[kMeasured / 2U];
  return true;
}

bool measure_pipe(Base *b, Pipe *p, uint64_t rows, uint64_t context,
                  uint64_t start, float *us) {
  for (uint32_t i = 0; i < kWarmups; ++i)
    if (!launch_pipe(b, p, rows, context, start) ||
        !hip_ok(hipStreamSynchronize(b->stream), "pipe warmup"))
      return false;
  std::array<float, kMeasured> s{};
  for (float &x : s) {
    if (!hip_ok(hipEventRecord(p->start, b->stream), "pipe event start") ||
        !launch_pipe(b, p, rows, context, start) ||
        !hip_ok(hipEventRecord(p->stop, b->stream), "pipe event stop") ||
        !hip_ok(hipEventSynchronize(p->stop), "pipe event sync") ||
        !hip_ok(hipEventElapsedTime(&x, p->start, p->stop), "pipe elapsed"))
      return false;
    x *= 1000.0F;
  }
  std::sort(s.begin(), s.end());
  *us = s[kMeasured / 2U];
  return true;
}

bool copy_out(Base *b, uint64_t rows, std::vector<uint16_t> *out) {
  out->resize(rows * kQHeads * kD);
  return hip_ok(
      hipMemcpy(out->data(), b->out, out->size() * 2U, hipMemcpyDeviceToHost),
      "copy out");
}
void compare(const char *name, const std::vector<uint16_t> &expected,
             const std::vector<uint16_t> &actual) {
  uint32_t max = 0;
  uint64_t over = 0;
  uint32_t shown = 0;
  size_t max_index = 0;
  float max_abs = 0.0F, max_rel = 0.0F;
  for (size_t i = 0; i < expected.size(); ++i) {
    const uint32_t d = ulp(expected[i], actual[i]);
    if (d > max) {
      max = d;
      max_index = i;
    }
    const float e = fbf(expected[i]), a = fbf(actual[i]);
    const float abs_error = std::fabs(e - a);
    max_abs = std::max(max_abs, abs_error);
    max_rel = std::max(max_rel, abs_error / std::max(1.0e-6F, std::fabs(e)));
    if (d > 1U) {
      ++over;
      if (shown++ < 4U)
        std::printf("oracle_mismatch candidate=%s index=%zu "
                    "expected=0x%04x(%g) actual=0x%04x(%g) ulp=%u\n",
                    name, i, expected[i], fbf(expected[i]), actual[i],
                    fbf(actual[i]), d);
    }
  }
  std::printf("oracle candidate=%s values=%zu max_bf16_ulp=%u over1=%llu "
              "max_abs=%g max_rel=%g status=%s\n",
              name, expected.size(), max, static_cast<unsigned long long>(over),
              max_abs, max_rel, over == 0U ? "PASS" : "INFO");
  if (max > 16U)
    std::printf("oracle_max candidate=%s index=%zu expected=0x%04x(%g) "
                "actual=0x%04x(%g)\n",
                name, max_index, expected[max_index], fbf(expected[max_index]),
                actual[max_index], fbf(actual[max_index]));
}
void resources() {
  const std::array<const void *, 5> f = {
      reinterpret_cast<const void *>(qtile4_k32_control),
      reinterpret_cast<const void *>(pack_query_f16),
      reinterpret_cast<const void *>(causal_softmax<__half>),
      reinterpret_cast<const void *>(causal_softmax<float>),
      reinterpret_cast<const void *>(unpack_bf16)};
  const std::array<const char *, 5> n = {"qtile4-k32-control", "pack-f16",
                                         "softmax-f16", "softmax-f32",
                                         "unpack-bf16"};
  for (size_t i = 0; i < f.size(); ++i) {
    hipFuncAttributes a{};
    int active = 0;
    const hipError_t as = hipFuncGetAttributes(&a, f[i]);
    const hipError_t os = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, f[i], kThreads, 0U);
    std::printf(
        "resources kernel=%s vgpr=%d sgpr=unavailable(hipFuncAttributes) "
        "lds=%zu scratch=%zu active_blocks=%d attrs=%s occupancy=%s\n",
        n[i], a.numRegs, a.sharedSizeBytes, a.localSizeBytes, active,
        hipGetErrorString(as), hipGetErrorString(os));
  }
}

bool parse_u64(const char *s, uint64_t *v) {
  if (!s || !*s)
    return false;
  char *e = nullptr;
  const unsigned long long x = std::strtoull(s, &e, 10);
  if (e == s || *e)
    return false;
  *v = static_cast<uint64_t>(x);
  return true;
}

} // namespace

int main(int argc, char **argv) {
  uint32_t device = 0U;
  uint64_t parsed_device = 0U;
  if (argc > 2 || (argc == 2 && (!parse_u64(argv[1], &parsed_device) ||
                                 parsed_device > UINT32_MAX))) {
    std::fprintf(
        stderr,
        "usage: phase78_gqa6_gfx1030_rocblas_attention_probe [device]\n");
    return EXIT_FAILURE;
  }
  if (argc == 2)
    device = static_cast<uint32_t>(parsed_device);
  if (!hip_ok(hipSetDevice(device), "set device"))
    return EXIT_FAILURE;
  hipDeviceProp_t prop{};
  if (!hip_ok(hipGetDeviceProperties(&prop, device), "device properties"))
    return EXIT_FAILURE;
  std::printf("target=%s device=%u pci=%04x:%02x:%02x name=%s\n",
              prop.gcnArchName, device, prop.pciDomainID, prop.pciBusID,
              prop.pciDeviceID, prop.name);
  if (std::string(prop.gcnArchName).find("gfx1030") == std::string::npos) {
    std::fprintf(stderr, "target_error expected gfx1030\n");
    return EXIT_FAILURE;
  }
  resources();

  // Exact all-head numerical oracle with nonaligned rows/context and causal
  // start.  This exercises the interleaved pointer/stride mapping.
  {
    constexpr uint64_t rows = 7U, context = 13U, start = 5U;
    std::vector<uint16_t> q, k, v, exp, got;
    fill(rows, context, &q, &k, &v);
    oracle(rows, context, start, q, k, v, &exp);
    Base b;
    if (!make_base(rows, context, &b) || !upload(q, k, v, &b)) {
      free_base(&b);
      return EXIT_FAILURE;
    }
    for (Pipe::Kind kind : {Pipe::F16, Pipe::F32}) {
      Pipe p;
      if (!make_pipe(rows, context, kind, b.stream, &p)) {
        free_base(&b);
        return EXIT_FAILURE;
      }
      float stage = 0.0F;
      if (kind == Pipe::F32 && !stage_f32(&b, &p, context, &stage)) {
        free_pipe(&p);
        free_base(&b);
        return EXIT_FAILURE;
      }
      if (!launch_pipe(&b, &p, rows, context, start) ||
          !hip_ok(hipStreamSynchronize(b.stream), "oracle pipe sync") ||
          !copy_out(&b, rows, &got)) {
        free_pipe(&p);
        free_base(&b);
        return EXIT_FAILURE;
      }
      compare(kind == Pipe::F16 ? "rocblas-fp16-score" : "sgemm-f32-score", exp,
              got);
      free_pipe(&p);
    }
    float control_us = 0.0F;
    if (!measure_control(&b, rows, context, start, &control_us)) {
      free_base(&b);
      return EXIT_FAILURE;
    }
    if (!copy_out(&b, rows, &got)) {
      free_base(&b);
      return EXIT_FAILURE;
    }
    compare("qtile4-k32-control", exp, got);
    free_base(&b);
    if (std::getenv("SLLM_GQA6_ORACLE_ONLY") != nullptr)
      return EXIT_SUCCESS;
  }

  bool all_ok = true;
  for (uint64_t rows : {128U, 512U, 1024U}) {
    for (int long_tail = 0; long_tail < 2; ++long_tail) {
      const uint64_t context = long_tail ? 9435U : rows;
      const uint64_t start = long_tail ? context - rows : 0U;
      std::vector<uint16_t> q, k, v, control, got;
      fill(rows, context, &q, &k, &v);
      Base b;
      if (!make_base(rows, context, &b) || !upload(q, k, v, &b)) {
        free_base(&b);
        return EXIT_FAILURE;
      }
      float control_us = 0.0F;
      if (!measure_control(&b, rows, context, start, &control_us) ||
          !copy_out(&b, rows, &control)) {
        free_base(&b);
        return EXIT_FAILURE;
      }
      std::printf("control_result rows=%llu context=%llu start=%llu "
                  "median_us=%.3f ms=%.6f\n",
                  static_cast<unsigned long long>(rows),
                  static_cast<unsigned long long>(context),
                  static_cast<unsigned long long>(start), control_us,
                  control_us / 1000.0F);
      const uint64_t qc = rows * kGqa;
      for (Pipe::Kind kind : {Pipe::F16, Pipe::F32}) {
        Pipe p;
        if (!make_pipe(rows, context, kind, b.stream, &p)) {
          free_base(&b);
          return EXIT_FAILURE;
        }
        float stage_us = 0.0F;
        if (kind == Pipe::F32 && !stage_f32(&b, &p, context, &stage_us)) {
          free_pipe(&p);
          free_base(&b);
          return EXIT_FAILURE;
        }
        float us = 0.0F;
        if (!measure_pipe(&b, &p, rows, context, start, &us) ||
            !copy_out(&b, rows, &got)) {
          free_pipe(&p);
          free_base(&b);
          return EXIT_FAILURE;
        }
        compare(kind == Pipe::F16 ? "rocblas-fp16-score" : "sgemm-f32-score",
                control, got);
        const uint64_t qbytes =
            kKvHeads * qc * kD * (kind == Pipe::F16 ? 2U : 4U);
        const uint64_t scorebytes =
            kKvHeads * qc * context * (kind == Pipe::F16 ? 2U : 4U);
        const uint64_t kvbytes =
            kind == Pipe::F32 ? 2U * kKvHeads * context * kD * 4U : 0U;
        const uint64_t pvbytes = kKvHeads * qc * kD * 4U;
        const double flops = 4.0 * static_cast<double>(kKvHeads) *
                             static_cast<double>(qc) *
                             static_cast<double>(context) * kD;
        const double tflops = flops / (static_cast<double>(us) * 1.0e9);
        const double speedup = static_cast<double>(control_us) / us;
        std::printf(
            "pipeline_result candidate=%s rows=%llu context=%llu start=%llu "
            "median_us=%.3f ms=%.6f stage_kv_us=%.3f workspace_bytes=%llu "
            "qpack_bytes=%llu score_bytes=%llu pv_bytes=%llu qk_pv_tflops=%.4f "
            "speedup_vs_qtile=%.4f precision=%s classification=%s\n",
            kind == Pipe::F16 ? "rocblas-fp16-score" : "sgemm-f32-score",
            static_cast<unsigned long long>(rows),
            static_cast<unsigned long long>(context),
            static_cast<unsigned long long>(start), us, us / 1000.0F, stage_us,
            static_cast<unsigned long long>(qbytes + scorebytes + pvbytes +
                                            kvbytes),
            static_cast<unsigned long long>(qbytes),
            static_cast<unsigned long long>(scorebytes),
            static_cast<unsigned long long>(pvbytes), tflops, speedup,
            kind == Pipe::F16 ? "fp16" : "f32",
            kind == Pipe::F16 ? "N2" : "N1");
        free_pipe(&p);
      }
      free_base(&b);
    }
  }
  std::printf("summary status=%s target=gfx1030 control=qtile4-k32 "
              "pipeline=4-way-strided-batched-qk-softmax-pv warmups=%u "
              "measured=%u oracle=7x13-start5 score_candidates=fp16,f32\n",
              all_ok ? "PASS" : "FAIL", kWarmups, kMeasured);
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
