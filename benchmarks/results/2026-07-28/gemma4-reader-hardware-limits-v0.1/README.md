# Gemma4 reader hardware-limit measurement v0.1

## Decision

**Far from a hardware limit.**  The exact current F32 reader work reaches only
`10.835–17.010 GFLOPS`: `0.0227–0.0356%` of the R9700's 47.8-TFLOPS FP32-vector
peak and `0.0057–0.0089%` of its 191-TFLOPS BF16-matrix peak.  Its effective
global Q/K/V/output traffic is only `16.351–33.124 GB/s` (`2.55–5.18%` of the
640-GB/s roof).  This is not an HBM-roof result.

The large gap is **not presently closable by a safe local tuning**.  Both
kernels have a hard grid underfill (eight 256-thread CTAs for 64 CUs), while
the otherwise attractive matrix route changes the F32 attention operands to
FP16/BF16 and cannot meet the Gemma differential gate without a measured
numerical solution.  The previously rejected sliding batching variants remain
rejected; this measurement does not revive them.

The product peak inputs are 47.8 TFLOPS FP32 vector, 191 TFLOPS FP16 matrix
(the project uses the same figure for BF16), and 640 GB/s.  AMD lists the FP32,
FP16-matrix, CU, and memory figures on the [R9700 product page](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html).

## Method and scope

This is a fresh cold-prefill `rocprofv3 --kernel-trace` measurement on
HIP ordinal 1 / `gfx1201` / amd-smi GPU 2, at commit `a1ae1d9c`, with
`ullm-openai` stopped, `/run/ullm/r9700.lock` held, then released before the
service restart.  The service finished `active`, `NRestarts=0`.  The two
profile JSON files record the target's own accepted no-process preflight.  The
full trace databases are retained in the local raw directory; the compact
numbers below are reproducible by joining the `kernels` view on the two reader
names and summing `duration`.

The failed `--pmc` attempt is also retained in `raw/rocprof-n512.stderr`.
With PMC collection enabled, rocprofv3 opens the GPU before the target starts;
the resident driver's intentional `amd-smi process` guard then refuses it.
Thus there is **no valid counter attribution** for physical HBM bytes,
per-CU residency, LDS bank conflicts, or a resource-limited occupancy.  This
report does not substitute HIPRTC metadata for that missing attribution.

The current default sliding path is important: it is M=1 calls to
`ullm_paged_decode_attn_f32_kernel`, not the disabled
`ullm_gemma_sliding_attn_ring_batched_256_f32_kernel`.  The fresh trace sees
exactly `28*N` sliding launches and `7*ceil(N/128)` full launches.

| N | reader kernel | launches | summed GPU time | mean / launch | grid / block |
| --: | --- | --: | --: | --: | --- |
| 512 | sliding `ullm_paged_decode_attn_f32_kernel` | 14,336 | 2.780233 s | 193.934 us | 2,048 / 256 threads |
| 512 | full `ullm_gemma_full_attn_batched_512_f32_kernel` | 28 | 0.942675 s | 33,666.975 us | 2,048 / 256 threads |
| 2048 | sliding `ullm_paged_decode_attn_f32_kernel` | 57,344 | 19.080124 s | 332.731 us | 2,048 / 256 threads |
| 2048 | full `ullm_gemma_full_attn_batched_512_f32_kernel` | 112 | 14.147060 s | 126,313.032 us | 2,048 / 256 threads |

These agree with EA's independent attribution (19.054416 s sliding and
14.238781 s full at N=2048) within trace/run variation.  The small difference
does not change the conclusion.

## Arithmetic required

For a Q head of width `D`, one Q dot K is `D` FMAs, or `2D` FLOPs.  Applying a
softmax weight to V is also `D` multiply-adds, or `2D` FLOPs.  Exponentials,
max/denominator updates, divisions, reductions, address work, and causal
branches are deliberately excluded: the requested FLOPs are only QK and AV.

For sliding, `H=8`, `D=256`, `L=28`, window `W=512`, and the causal key count
over the prefill is:

```text
S_sliding(N) = sum_t min(t + 1, 512)
             = N(N + 1)/2                         (N <= 512)
             = 512*513/2 + (N - 512)*512          (N > 512)
QK = AV = L * H * (2D) * S_sliding
```

For full, `H=8`, `D=512`, `L=7`, and `S_full(N)=N(N+1)/2`:

```text
QK = AV = L * H * (2D) * S_full
```

| N | kind | QK FLOPs | softmax·V FLOPs | requested total |
| --: | --- | --: | --: | --: |
| 512 | sliding | 15,061,745,664 | 15,061,745,664 | 30,123,491,328 |
| 512 | full | 7,530,872,832 | 7,530,872,832 | 15,061,745,664 |
| 2048 | sliding | 105,256,058,880 | 105,256,058,880 | 210,512,117,760 |
| 2048 | full | 120,317,804,544 | 120,317,804,544 | 240,635,609,088 |

## Achieved compute and data movement

`bytes` below are the actual bulk global loads/stores issued by the current
source, not the old logical-K/V accounting and not a claim about physical HBM
transactions.  They include repeated Q loads in the inner source loop, K, V,
and output; scalar page-table reads are below 0.01% and excluded.  Physical
HBM `FETCH_SIZE` could not be validly collected for the reason above.

The M=1 generic sliding kernel reads Q, K, and V for every causal pair, so its
bulk bytes are:

```text
B_slide = 28 * [8 * 3 * 256 * 4 * S_sliding + 8 * 256 * 4 * N]
```

The full kernel retains K/V in LDS once per source per 128-row tile, but still
reads Q for each visible pair.  With `C=N/128` tiles and
`T=128*C*(C+1)/2` source tokens visited across tiles:

```text
B_full = 7 * [8*512*4*S_full + 8*2*512*4*T + 8*512*4*N]
```

| N | kind | bulk global bytes | achieved | FP32-vector peak | BF16-matrix peak | 640-GB/s roof |
| --: | --- | --: | --: | --: | --: |
| 512 | sliding | 90.488 GB | 10.835 GFLOPS, 32.547 GB/s | 0.0227% | 0.0057% | 5.085% |
| 512 | full | 15.414 GB | 15.978 GFLOPS, 16.351 GB/s | 0.0334% | 0.0084% | 2.555% |
| 2048 | sliding | 632.006 GB | 11.033 GFLOPS, 33.124 GB/s | 0.0231% | 0.0058% | 5.176% |
| 2048 | full | 244.863 GB | 17.010 GFLOPS, 17.308 GB/s | 0.0356% | 0.0089% | 2.704% |

## Launch occupancy and resources

The profiler dispatch record gives the **actual launch configuration**:
`grid_x=2048`, `workgroup_x=256`, and ROCm reports wave32 on this GPU.  Each
dispatch therefore has 8 CTAs and 64 waves.  With 64 CUs, that is exactly
`0.125 CTA/CU` and `1 wave/CU` averaged over the whole device.  At most eight
CUs can receive a CTA; if distributed one per active CU, they each have one
CTA / eight waves while 56 CUs are idle.  This hard grid underfill is
independent of the unavailable per-CU counter sample.

| kernel | dispatch-record LDS | dispatch-record scratch | VGPR / AGPR / SGPR | resource limit attribution |
| --- | --: | --: | --- | --- |
| sliding generic M=1 | 1,024 B | 0 B | 32 / 0 / 128 | not available |
| full M=128 | 6,152 B | 1,040 B per work-item | 32 / 0 / 128 | not available |

The full kernel's 1,040-B scratch record is a real warning sign (its two
128-element weighted accumulators are declared as local arrays), but it is
not evidence that scratch, VGPR, LDS, or SGPR is the active residency limit.
No such limit is asserted here.

## Candidate ranking and recoverable share

The percentages below use EA's N=2048 `33.293197 s / 69.463637 s = 47.93%`
reader-GPU share.  They are overlapping upper bounds, not additive savings.

| rank | finding | evidence | maximum reader share recovered | maximum whole-prefill share recovered | verdict |
| --: | --- | --- | --: | --: | --- |
| 1 | Grid underfill / serial scalar reduction topology | 8 CTAs, 64 waves for 64 CUs in both readers | 87.5% if 8x scaling | 41.94 percentage points; 1.72x total Amdahl bound | real headroom, but requires a different split/merge attention design, not another sliding batching attempt |
| 2 | Scalar F32 arithmetic, no matrix instructions | both sources use scalar `float` multiply/add and reductions; no WMMA builtin/intrinsic appears in either reader source | <=100% | <=47.93 pp; 1.92x free-reader ceiling | enormous theoretical gap but not an accepted precision path yet |
| 3 | Full-reader scratch/live-state pressure | trace has 1,040 B private scratch per work-item | <=42.8% (full reader only) | <=20.50 pp | plausible full-only issue; no evidence supports a numerical recovery fraction |
| 4 | Poor coalescing / float4 / LDS-bank conflict | threads index contiguous `tid` elements; no valid PMC counter sample | 0% claimed | 0 pp claimed | not supported; low achieved bandwidth follows grid/scalar topology, not an HBM roof |
| 5 | M=1 fixed kernel overhead | shortest sliding dispatch is 2.760 us; steady N=2048 median is 356.683 us | about 0.48% using 2.760 us only as a deliberately generous fixed-cost proxy | about 0.23 pp | not significant; no decomposed fixed-cost measurement exists |

The scalar source audit is unambiguous, but a WMMA replacement is not
numerically free.  WMMA consumes FP16/BF16 operands with F32 accumulation.
Round-to-nearest input quantisation alone has maximum relative error
`2^-11 = 4.8828e-4` for FP16 and `2^-8 = 3.90625e-3` for BF16 per normal
operand, before the QK reduction and softmax.  At magnitude 1 that is already
10.2x / 81.4x the project maximum-absolute gate of `4.8e-5`; QK and softmax
can amplify it.  No candidate was implemented, so no real-activation
max-abs/max-rel/logit differential is available, and this report makes no
claim that either precision would pass.

## No implementation

No runtime/kernel source was changed.  No causal, ring-rollover, tile-width,
real-activation, shared-KV, or decode validation is applicable.
