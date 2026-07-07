# Phase C4 batch width and long-context coverage v1

Date: 2026-07-07

Runtime commit: `f25d377`

Purpose:

- Fill the missing `B=2` rows in the Phase C4 runtime causal attention
  batch-width grid.
- Add an optional wider `B=8, N=4096` runtime row.
- Add cached-prefix `M=512` rows for the existing `L=4096/16384/65536`
  sweep.
- Refresh the current self-attention layer partial long-context
  `N=16384` row after the latest causal attention path.

## Runtime Causal Attention Batch

Command shape:

```bash
ULLM_REQUIRE_HIP_CAUSAL_ATTN_BATCH_KERNEL=1 \
  target/release/ullm-engine runtime-causal-attn-batch-smoke \
  2 B N REPEATS 16 4 256 256
```

| B | N | repeats | mean ms | input tok/s | attention pair/s | sampled diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 512 | 3 | 13.954804 | 73379.746182 | 18821904.895693 | 0 |
| 2 | 2048 | 3 | 230.013736 | 17807.632124 | 18243919.110634 | 0 |
| 8 | 4096 | 1 | 3698.347224 | 8860.174022 | 18150066.484942 | 0 |

Interpretation:

- The required Phase C4 batch-width component grid now has
  `B=1/2/4/8` for `N=512/2048` when combined with
  `phase-c4-cold-prefill-online-softmax-v1.md`.
- `B=2` sits on the same `18.2-18.8M pair/s` band as `B=1/4/8`.
- `B=8, N=4096` also remains in the same band at `18.15M pair/s`.
- Increasing request batch width still scales wall time roughly linearly.
  The current causal attention batch kernel is real-batch addressable, but
  not yet request-reuse efficient.

## Cached Prefix M=512

Command shape:

```bash
ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_KERNEL=1 \
  target/release/ullm-engine runtime-cached-prefix-attn-smoke \
  2 L 512 REPEATS 16 4 256 256 cached_prefix_chunked
```

| L | M | repeats | mean ms | new input tok/s | attention pair/s | sampled diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 512 | 3 | 129.396385 | 3956.833879 | 17222119.458747 | 0 |
| 16384 | 512 | 1 | 676.288522 | 757.073325 | 12598078.664420 | 0 |
| 65536 | 512 | 1 | 2607.803969 | 196.333776 | 12917289.949872 | 0 |

Interpretation:

- The cached-prefix chunk grid now includes the missing `M=512` row for
  `L=4096/16384/65536`.
- `L=65536, M=512` completes the planned long-prefix component boundary
  together with the already saved `M=1/16/128` rows.
- Pair throughput is good at `L=4096`, but long prefix rows stay closer to
  `12.6-12.9M pair/s`, so K/V read pattern and request/batch direction are
  still the next cached-prefix optimization targets.

## Package Self-Attention Layer N=16384

Command:

```bash
ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 \
ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL=1 \
ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1 \
  target/release/ullm-engine package-self-attn-layer-batch-smoke \
  /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d \
  2 1048576 3 len:16384 1
```

| component | N | repeats | mean ms | token/s | verification ms | layer diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| package self-attention layer batch | 16384 | 1 | 13279.226135 | 1233.806837 | 11677.183403 | 0 |

Interpretation:

- The current self-attention layer partial long-context row is much faster
  than the older source-shared layer row saved earlier in the plan
  (`24825.171928 ms`), and keeps the output guard intact.
- Verification is still expensive at `11.68s`, but it is separate from the
  timed GPU wall section.
- This remains a single-layer component result. It is useful for SQ prefill
  format evaluation pressure, but not a full-model total throughput result.

## Next Optimization Signal

- Runtime causal attention batch width is not the main missing control-plane
  piece anymore; `B=1/2/4/8` coverage exists for `N=512/2048`.
- Further cold-prefill improvement should focus on tiled/block causal
  attention or K/V reuse across nearby timesteps/heads.
- Cached-prefix improvement should focus on long-prefix K/V read coalescing
  and request/batch direction, not score pass count.
- Full SQ candidate comparison still needs package-level batch throughput and
  FP8 candidate runtime rows.
