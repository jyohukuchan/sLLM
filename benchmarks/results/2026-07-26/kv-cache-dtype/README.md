# KV cache dtype evidence — 2026-07-26

## Scope

This directory records the FP16 / FP8 E4M3FN paged-KV work from commit
`ed641675`. It is intentionally separate from pre-existing 2026-07-26 result
directories.

## CPU validation

The following targeted CPU tests completed after the typed ABI was added:

```text
cargo test -p ullm-engine decoder::tests --lib
36 passed; 0 failed

cargo test -p ullm-engine kv_cache_dtype --lib
4 passed; 0 failed
```

Coverage includes F32 regression, F16, FP8 E4M3FN, mixed K/V dtypes, physical
block-table placement, scale-aware readback, direct decode, typed causal
prefill fallback, and rejection of corrupted negative FP8 scales. `Q8_0` is
rejected by the parser. The K/V override precedence over the uniform selector
is tested without mutating the process environment.

## Capacity ledger

Fixed geometry: 16 tokens/block, 256 blocks, 4 KV heads, K/V dim 256, per
layer.

| dtype | cache bytes | same-F32-budget context |
|---|---:|---:|
| F32 | 32 MiB | 4,096 tokens |
| F16 | 16 MiB | 8,192 tokens |
| FP8 E4M3FN + per-token/head FP16 K/V scales | 8.0625 MiB | 16,256 tokens (1,016 full blocks) |

The FP8 result includes 64 KiB of scales: two `[4096, 4]` FP16 planes. This
is capacity accounting, not a speed estimate.

## Full-model / quality status

| dtype | decode | prefill | long-context generated text |
|---|---|---|---|
| F32 | not rerun; BH reference is 27.378731 tok/s | not rerun | not rerun |
| F16 | unmeasured | unmeasured | unmeasured |
| FP8 | unmeasured | unmeasured | unmeasured |

No throughput number was inferred from a kernel or profile range. The typed
HIP path is currently correctness staging, not native execution; additionally,
the AQ4_0 resident production cache path is deliberately outside this task's
allowed edit scope. Therefore a full-model F16/FP8 comparison and F32/F16/FP8
side-by-side generation are not yet valid to record.

## GPU guard note

An unintended broad runtime test invocation was found to include opportunistic
HIP tests before the mandatory R9700 lock preflight. It is excluded from this
evidence and no follow-up GPU work was run while the service held
`/run/ullm/r9700.lock`. No active manifest or systemd configuration was
changed.

See `docs/plans/kv-cache-dtype-fp16-fp8-design-v0.1.md` for the complete
route audit and native-kernel handoff.
