# Gemma4 activation-region growth v0.1

## Step 1: dense MLP resident region

The first contiguous device-resident region is, for every dense Gemma4 MLP:

```text
host MLP input -> gate BF16 matvec -> up BF16 matvec -> GELUTanh * up
               -> down BF16 matvec -> host MLP output
```

`gate`, `up`, the activated product, and the down-projection result use
persistent F32 device workspaces.  The region therefore has two host
boundaries rather than the three independent matvec H2D/D2H pairs.  It removes
four activation transfers per layer (140 per decode token; 420 matvec result
round trips per four-token benchmark), while preserving the existing source
BF16 weight and device K/V contracts.

The new `ullm_runtime_gelu_tanh_mul_f32` kernel uses the literal
`0.7978845608f` and Transformers GELUTanh's F32 operation order.  This is the
intentional runtime translation-unit change recorded in this commit.

| path | decode tok/s | prefill tok/s | decode / 140.341 tok/s roofline |
| --- | ---: | ---: | ---: |
| prior per-primitive baseline | 15.733 | 18.544 | 11.21% |
| step 1: dense MLP region | 20.479 | 23.610 | 14.59% |

The three-repeat R9700-only evidence is
`benchmarks/results/2026-07-27/gemma4-activation-device-v0.1/raw/dg/mlp-region-benchmark.json`.
The two full-model four-step continuations, cache/full-reprefill checks, and
the expected plain text all passed.  Device output was checked against the
unchanged individual resident host path on 3,045 real MLP calls / 4,677,120
output elements: max abs `0.0000019073486328125`, max rel
`0.0416666679084301` (the relative maximum occurs at a near-zero reference).

Attention, all normalizations, RoPE, residual/PLE, and final norm/head remain
host-visible.  Absorbing the next boundary requires device RMSNorm plus its
BF16 gamma conversion and device residual/add workspaces; attention requires
device Q/K/V normalization and RoPE before its device-resident attention
output can feed the next projection.
