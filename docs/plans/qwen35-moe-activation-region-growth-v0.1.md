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

## Attempted step 2: dedicated-workspace shared-expert region (rejected)

Before retrying, a fresh unchanged-host source run at context length 8,192
used the 23-token text prompt from the original baseline and produced the
non-repeating 16-token continuation
`[90700, 8340, 25, 271, 16, 13, 220, 2972, 2014, 53983, 279, 5952, 64700,
198, 262, 348]`. This is the healthy reference; it refutes the concern that
the source MoE path is intrinsically degenerate at all reproducible context
lengths. The original 262,144-token capability remains unreproducible from
fresh source and is not used here.

The retry allocated three distinct persistent F32 workspaces per layer for
the shared gate, shared up, and SiLU product. They did not alias attention,
dense-MLP, routed-expert, or combine buffers. The shared down result used the
already-owned `shared_output` handoff buffer. The chain was therefore
`post-norm -> device BF16 gate/up -> device SiLU product -> device BF16 down
-> existing device shared combine`; routing, full attention, selected-expert
GEMM, and AQ4 dequantization were not changed.

The catastrophic prior failure was largely buffer-alias corruption: the same
real-model comparison fell from final-hidden max abs `23.95994` to
`0.12071514`. It still fails the required numerical gate. The only semantic
difference remaining between the unchanged host chain and this non-aliasing
candidate was host `std::exp` versus the existing device `expf` SiLU product
(and its F32 multiplication order); after forty layers that difference was
large enough to alter two of forty final route vectors. This candidate is
therefore rejected, and all implementation code was rolled back byte-exact.

| step | decode tok/s | prefill tok/s | final route-ID agreement | result |
| --- | ---: | ---: | --- | --- |
| baseline (committed 262k record) | 11.032 | 10.955 | n/a | reference only; 262k not reproducible from fresh source |
| step 2 host, 8k / 16 decode | 11.039 | 10.455 | 40 / 40 | healthy source reference |
| step 2 dedicated resident, 8k / 16 decode | 11.499 | 10.760 | 38 / 40 | rejected |

Both runs generated the same 16 token IDs, but that is insufficient for a
promotion. On the final generated token, post-final-norm hidden max abs/max
rel were `0.12071514129638672` / `16.248566048614325`; full-vocabulary logits
were `0.09089469909667969` / `316.64006791171477`. The real activation dumps
and JSON records are retained locally in
`benchmarks/results/2026-07-27/qwen35-moe-activation-device-v0.1/raw/dm-step2/`.
The selected IDs are compared before accepting the identical text continuation.
