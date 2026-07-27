# Gemma4 512-wide paged-decode specialization v0.1

## Change

`ullm_paged_decode_attn_f32_kernel` now has an exact `head_dim == value_dim ==
512` branch.  It uses one 256-thread CTA per Q head: every thread reduces two
Q/K products and owns two V accumulators.  The host launcher selects the
one-CTA/head grid only for this exact shape.  The pre-existing `<=256` kernel
body is unmodified.

The runtime translation-unit guard was re-recorded in the same change:
`7148b11c049754ef2875aeb460b4c175f68a6f4dddd7109aaa8bdbd76730c6c6`.

## Clean R9700 measurements

The release executable was rebuilt after `cargo clean -p ullm-runtime-sys`.
Every invocation used `HIP_VISIBLE_DEVICES=1`, which the result records as
runtime index 1 / gfx1201, and acquired `/run/ullm/r9700.lock`.  Required
Gemma HIP kernels were fail-closed.

| prompt context | prefill tok/s before | prefill tok/s after | change | decode tok/s before | decode tok/s after | change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 18.683 | 18.796 | +0.6% | 8.163 | 8.206 | +0.5% |
| 512 | 5.613 | 6.744 | +20.2% | 2.948 | 3.222 | +9.3% |

The benchmark is three M=N prefill repetitions and three 128-token decode
sweeps.  N=2048 is retained as a separate long, locked run in progress at the
time this increment was committed.

## Correctness

`raw/validation.json` is a complete multi-step, real-activation full-model
check.  The causal future-token probe is exactly `0 / 0`.  The unchanged-host
comparison reports layer-output max abs `2.6702880859375e-5`, final-norm max
abs `3.4332275390625e-5`, and logits max abs `2.1219253540039062e-5`; both
known cached/re-prefill continuation pairs and their top-1 logits agree.

The slight throughput improvement despite removing the scalar fallback means
that the promoted PLE prefill route has substantial work outside this M=1
reader.  It does not invalidate the source-level 256.5x reader arithmetic
amplification established in the Job-1 audit.
