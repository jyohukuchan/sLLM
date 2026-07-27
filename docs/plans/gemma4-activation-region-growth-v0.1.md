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

## Step 2: MLP-adjacent direct-gamma norms and residual

The region now begins at the host attention residual and ends only after the
MLP residual:

```text
host attention residual -> direct-BF16 pre-FF RMSNorm -> dense MLP
                        -> direct-BF16 post-FF RMSNorm -> residual add
                        -> host MLP residual
```

The two Gemma gamma vectors stay in their resident checkpoint BF16 buffers.
Each is converted by the existing device BF16-row kernel into a persistent F32
workspace for the existing device RMSNorm kernel; no `+1` transform is
applied. The existing device add kernel writes the final residual. This removes
280 host-visible BF16-row D2H/synchronize boundaries per four-token decode
(1,056 -> 776) while preserving the 688 remaining host-visible matvec calls.
No runtime translation unit changed in this step.

Three R9700-only repeats measured 21.505 tok/s decode and 24.490 tok/s
prefill, or 15.32% of the 140.341 tok/s decode roofline (+5.01% decode and
+3.73% prefill versus step 1). The raw evidence is
`benchmarks/results/2026-07-27/gemma4-activation-device-v0.1/raw/dh/mlp-norm-residual-benchmark-v3.json`.

Validation against the unchanged host `pre-FF RMSNorm -> MLP -> post-FF
RMSNorm -> residual-add` sequence covered 3,045 real captured residuals /
4,677,120 output elements: max abs `0.00042724609375`, max rel
`0.1631205677986145` (near-zero maximum). The full cached-versus-reprefill
multi-step cases both passed and retained `[9079, 236761, 108, 818]` and
`[528, 496, 1902, 1298]`, respectively. The validation evidence is
`benchmarks/results/2026-07-27/gemma4-activation-device-v0.1/raw/dh/mlp-norm-residual-validation-v3.json`.

## Step 2.5: Gemma proportional RoPE validation gate

`ullm_runtime_gemma_proportional_rope_f32` remains validation-only and is not
wired into Gemma execution. The final R9700 (`gfx1201`) validation exercised
906 real normalized Q/K activations (2,646,528 values) against the unchanged
host `apply_gemma4_rope_in_place`, with
`ULLM_REQUIRE_HIP_GEMMA_PROPORTIONAL_ROPE_KERNEL=1` set so HIPRTC staging
cannot satisfy the run. It measured max abs `0.000003814697265625` and max
rel `0.02024332620203495` (near-zero reference). The rotated channels have
the same maxima; both unrotated spans, including their channels adjacent to
the active partial pairs, are bit-exact pass-through (`0.0` abs / `0.0` rel).

The frequency exponent is `2 * pair / head_dim`, never `rotary_dim`; the
generic Qwen kernel was not changed. The direct full-geometry HIP regression
also covers 8 heads × 512 channels with 128 rotary channels and asserts both
unrotated spans bit-exact. The first attempted real run exposed that Cargo did
not track the included RoPE source fragment, so it could link a stale native
runtime object. `ullm-runtime-sys/build.rs` now tracks both RoPE source
fragments; a clean native rebuild was required before the successful rerun.

The two full multi-step cases continued to match their expected cached and
full-reprefill sequences. Evidence:
`benchmarks/results/2026-07-27/gemma4-activation-device-v0.1/raw/dj/gemma-proportional-rope-real-activation-validation-v4.json`.
