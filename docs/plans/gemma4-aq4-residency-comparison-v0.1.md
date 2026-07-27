# Gemma4 / AQ4 residency comparison v0.1

## Result

The AQ4-residency hypothesis is **supported**, with an important boundary:
AQ4 is resident for the *activation graph*, not merely weights and K/V. Gemma4
is resident only for source BF16 weights and K/V, then returns every primitive
result to host memory.

AQ4's `PackageSelfAttnResidentStepLayer` owns persistent device buffers for
the hidden input, normalized input, Q/K/V projections, normalized/RoPE Q/K,
K/V cache, attention output, post-attention result, MLP activation, and layer
output. `dispatch_layer_stack` passes a prior layer's device output directly
to the next layer. Its diagnostic trace path explicitly documents D2H copies
as diagnostic-only and excludes them from production sessions.

Gemma4's `Bf16MatvecRuntime`, by contrast, owns only reusable device input
and output buffers. `matvec_resident`, `bf16_row_resident`, and
`device_attention` each issue `copy_to_host` followed by
`stream.synchronize`; CPU math then builds the next device input. This exactly
matches the 473.378 ms D2H submission finding in the localisation record.

## Tested narrow port: rejected

I tested reusable HIP-registered host output staging as a deliberately narrow
way to remove pageable D2H submission stalls without changing F32 arithmetic.
The first resident Gemma4 validation run failed closed with:

```text
Gemma4 resident BF16 row returned non-finite or malformed F32 output
```

The experiment was removed in full. No runtime API or Gemma execution change
from that trial remains. The normal resident path was rebuilt and rerun on the
R9700; it restored the known generated IDs
`[236761, 108, 818, 5279]` and measured 16.637 tok/s in its one-repeat
rollback probe. The raw rollback evidence is retained at
`benchmarks/results/2026-07-27/gemma4-pinned-host-staging-v0.1/raw/gemma-benchmark-rollback-verified.json`.

The production AQ4 regression probe was then rerun on the R9700 with its full
required-kernel guard set. Its output SHA-256 is exactly
`30865287e7525f4b24449ec24be3aa7619bfbbbbf48522cf2f67f9e58379b588`, with
top-1 `220` / `8.529029846191406`; this is byte-identical to the fixed
128-token production record. The probe output is retained beside the Gemma
rollback evidence as `qwen35-aq4-production-rollback-probe.json`.

## Safe next implementation boundary

The evidence does **not** justify a small local copy tweak. A safe performance
port must retain Gemma4 activations on device through whole layer segments and
implement its exact F32 RMSNorm, RoPE, GELU, multiply/residual, PLE, BF16 norm
weight conversion, and final soft-cap contracts there. AQ4 already has the
workspace discipline and several primitives, but its Qwen-specific fused
operations cannot be substituted for Gemma4 without a new finite-precision
contract validated against `tools/architecture_hf_trace.py`.

That is a substantive implementation, rather than a safe same-session change.
The working tree is left byte-for-byte on the pre-trial Gemma execution route.
