# Qwen3.5-35B-A3B MoE activation-region growth v0.1

## Baseline and first region selection

This run targets only the host-side activation gap in the 35B-A3B MoE.  It
does not change MoE routing, the selected-expert GEMMs, AQ4 dequantization, or
full attention.  In particular, the router's device-to-host result copies are
recorded in the trace so the cost is not hidden, but are deliberately not a
candidate for this series.

The fresh R9700-only baseline uses the 24-token prompt / four-token decode
case and the same 262144-token F16 typed-KV guard contract as the production
loader work. Three unprofiled repetitions measured 11.032 decode tok/s and
10.955 prefill tok/s.  ROCprof is attribution only: it measured 479.112 ms
for the four-token decode and 261.149 ms in GPU kernels, leaving 217.963 ms
(45.50%) outside kernels.  This independently agrees with the prior MoE
attribution's 44.493% decode / 47.410% prefill host-gap finding.

The raw API ranking is committed as
`benchmarks/results/2026-07-27/qwen35-moe-activation-device-v0.1/raw/per-primitive-baseline.json`.
The trace database itself remains local (107 MB), with SHA-256
`b10ed6a36641d98f869436e5112e2cb5f8bbaf61fe42b30c2496a8f00af09e69`.

The largest in-scope contiguous chain is the shared expert, whose three
passthrough tensors are F32/BF16 device matvec-capable matrices but whose
existing `matvec_silu_mul_with` helper reads both projection results back to
the host for the SiLU product:

```text
resident post-attention norm -> shared gate matvec -> shared up matvec
    -> SiLU(gate) * up -> shared down matvec -> resident MoE/shared combine
```

The inputs and outputs are already device buffers, so this removes a complete
host island rather than porting an isolated primitive.  The region needs a
persistent gate workspace while the existing activation workspace holds the
up/product and feeds the down projection.  It leaves routing untouched and
does not enter the full-attention O-projection path, even though that separate
passthrough fallback also appears in the trace.

| step | resident ops | host matvecs / 4 tok | decode tok/s | prefill tok/s |
| --- | --- | ---: | ---: | ---: |
| baseline | none newly added | not yet reduced | 11.032 | 10.955 |

## Attempted step 1: shared-expert resident gate/up/product/down (rejected)

The candidate used only dead-after-use resident buffers: the normal MLP
activation buffer for gate, the layer input buffer for up, and an
attention-projection temporary for the SiLU product.  Thus it added no VRAM
allocation and left post-norm resident for the independent route verifier.
The existing device `silu_mul_f32` kernel was sufficient; no GELUTanh or RoPE
kernel applies to this SiLU shared-expert chain.

It is rejected.  A current-source 262144-token load first exposed the
prewarm's hard-coded RMS epsilon (`1e-5` versus the descriptor's `1e-6`); when
temporarily corrected for diagnosis, allocation of the final full-attention V
cache failed even with zero new region memory.  The pre-existing 262k baseline
therefore cannot be reproduced from a fresh source build and must not be used
as a promotion comparison.

At a 131072-token diagnostic context, the candidate's own raw-BF16 router
verification passed, but comparison with the unchanged host shared-expert path
failed the required multi-step gate: host generated
`[194659, 194659, 194659, 194659]`; candidate generated
`[90700, 8340, 25, 271]`.  Final hidden max abs/max rel were
`23.95993995666504` / `27510.341719335243`; logits were
`35.50808143615723` / `235438.86013986013`; all 40 final-token selected-expert
ID vectors differed.  Evidence remains local under
`raw/step1/{host,candidate}-131k-*`.  No code from this rejected attempt is
landed.
