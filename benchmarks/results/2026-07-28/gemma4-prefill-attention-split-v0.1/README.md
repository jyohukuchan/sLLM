# Gemma4 prefill attention split v0.1

## Decision

Do **not** build the sliding-layer-only gather + Flash2 path.  Fresh
instrumentation shows that the seven 512-wide full-attention layers dominate
the current M=1 prefill attention region, and the existing 256-wide reader
cannot execute them.

| prompt | sliding region | full region | full share of attention | sliding-only end-to-end ceiling |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.965014 s | 4.058743 s | 80.791% | 1.155x |
| 512 | 5.713319 s | 62.063165 s | 91.570% | 1.081x |

The N=128 total attention-region time is 5.023757 s of a 7.173508 s cold
prefill; the prior DS 5.81 s figure is not reused as a split surrogate.  The
fresh run places the full layers at 56.580% of all prefill wall time.  At
N=512 they are 81.761% of all prefill wall time.  Even an impossible
zero-cost sliding path would leave the unsupported full route as the dominant
cost, so implementing a gather only adds complexity without a viable
end-to-end payoff.

## Method

`ullm-gemma4-resident --mode attention-profile` performs exactly one
cold-cache prefill: it has no warmup and no decode.  A non-semantic timer at
the existing resident attention-region boundary records all device-resident
attention work (input norm, Q/K/V/O projections, RoPE, K/V write, paged
reader, post norm, residual boundary, and the final synchronization) by the
executing Gemma4 layer kind.  The architectural split is explicit: full
layers are 4, 9, 14, 19, 24, 29, and 34; the other 28 are sliding.

The recorded reader counts validate the scope: N=128 has 3,584 sliding plus
896 full calls (4,480 total), and N=512 has 14,336 sliding plus 3,584 full
calls (17,920 total).  Both runs used HIP ordinal 1 only, which identifies the
gfx1201 R9700; the service was inactive and each command acquired
`/run/ullm/r9700.lock`.

The native runtime was clean rebuilt (`cargo clean -p ullm-runtime-sys`) and
the release driver was relinked before this window.  No `runtime/src` file or
kernel source changed, so the runtime translation-unit guard was not
re-recorded.

## AQ4 regression probe

The start and end probes are byte-identical to each other and to the protected record:
`30865287e7525f4b24449ec24be3aa7619bfbbbbf48522cf2f67f9e58379b588`, top-1
`220 / 8.529029846191406`.  Its 128 fixed IDs and all required production
guards are inherited from the existing R9700 probe script.

Raw JSON is in `raw/attention-profile-n128.json`,
`raw/attention-profile-n512.json`, and the `raw/qwen35-aq4-probe-*.json`
pair.
