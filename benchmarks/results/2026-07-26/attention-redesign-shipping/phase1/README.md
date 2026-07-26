# `AQ4_0` decode attention: applicability audit

## Scope and active identity

This audit answers whether the `SQ8_0` Qwen3-14B GQA-grouped, split-tile-20
decode redesign can be applied to the served `AQ4_0` Qwen3.5-9B model.  It is
not a cross-model output-quality comparison.

At audit start, `/etc/ullm/served-models/active.json` was the `AQ4_0` P3
manifest with SHA-256
`a98910dc5bf59dc768e5bcd20bcf58968699540eb1b33df33066dcb6f274fe49`.
It binds source `c4c9a9b344fc10e9a77ab0ded3293469d21b2f72` and the worker
`/opt/ullm/aq4-p3-deployment-v0.1/releases/aq4-p3-c4c9a9b3/ullm-aq4-worker`.

## Qwen3.5-9B structure

The served model config is
`/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B/config.json`
(SHA-256 `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05`).
Its `text_config` has 32 layers, 16 query heads, 4 KV heads, and head/value
dimension 256.  `layer_types` is exactly eight repetitions of three
`linear_attention` layers followed by one `full_attention` layer: 24 linear
layers and 8 full-attention layers.  The full-attention GQA ratio is therefore
4:1, not 1:1.

## ROCprof kernel attribution (historical P3-compatible trace)

`existing-p3-round2-trace-summary.json` is regenerated from the raw C=1339,
warmup=6, measured=32 ROCprof trace in
`benchmarks/results/2026-07-20/aq4-decode-step-profile-post-p3-round2-20260719T230555Z/`.
The accounting assigns each `KERNEL_DISPATCH` to an outer
`ullm.aq4.decode.step.v1/...` marker using the start time of its correlated
`hipModuleLaunchKernel` API call.  It deliberately does not use profiler range
time as throughput, and it excludes DMA-style records whose correlation is not
a module kernel launch.

| kernel | dispatches | inclusive GPU time | share of module-launched decode GPU time |
| --- | ---: | ---: | ---: |
| `ullm_paged_decode_attn_f32_kernel` | 0 | 0 ns | 0.00000% |
| `ullm_paged_decode_attn_split_partial_f32_kernel` | 256 | 35,642,400 ns | 8.64529% |
| `ullm_paged_decode_attn_split_merge_f32_kernel` | 256 | 1,373,896 ns | 0.33325% |
| split attention core | 512 | 37,016,296 ns | **8.97854%** |
| all module-launched decode kernels | 9,344 | 412,275,120 ns | 100% |

There are eight partial and eight merge dispatches per marked decode step,
matching the eight full-attention layers.  The partial grid is
`45,056 / 256 = 176 = 16 * ceil(1339 / 128)`, which is direct trace evidence
that this route uses a 128-token source tile.  The two raw direct-kernel rows
are outside the decode markers (warmup/setup), so they are not decode evidence.

This artifact predates the active P3 source commit and does not bind its own
worker hash.  It is consequently historical P3-compatible evidence only.  A
separate current-P3-compatible ROCprof capture is required before treating the
kernel route as current-active evidence; no service restart is needed for that
capture.

`current-p3-compatible-c1339-20260726T141806Z/`,
`current-p3-compatible-c1339-20260726T141953Z/`, and
`current-p3-compatible-c1339-20260726T142102Z/` record attempted current
captures.  Their required preflight or non-blocking `flock` found another owner
of `/run/ullm/r9700.lock`, so none started a profile.  They are lock-use
records, not profiler evidence.  A later successful attempt must be added
rather than overwriting them.

`tools/summarize-aq4-decode-attention-trace.py` was replayed against the
historical raw CSVs and produced a byte-identical copy of
`existing-p3-round2-trace-summary.json`; its output is a kernel-time summary
rather than a throughput measurement.

## Applicability verdict

**BH's `SQ8_0` grouped tile-20 implementation cannot be applied directly to
`AQ4_0`; `AQ4_0` must not be promoted for it.**

The trace shows that `AQ4_0` does not use
`ullm_paged_decode_attn_f32_kernel` in real C=1339 decode; it uses the generic
split partial/merge path.  The active `c4c9a9b3` source predates the BH grouped
implementation and only accepts the AQ4 experimental split tiles 128 or 256.
In the redesign source, the grouped body is explicitly guarded by
`q_heads / kv_heads == 5`, `head_dim == 128`, and `value_dim == 128`.
`AQ4_0` is 4:1 with dimensions 256, so enabling the selector would take the
generic fallback rather than the grouped body.  The 24 linear-attention layers
are also outside this full-attention GQA/KV-split optimization.

Extending the kernel to the 4:1/256 shape would be a new implementation, not
an application of BH's redesign.  It was intentionally not started here:
`runtime/src/ullm_runtime_parts/part_01.inc` and
`runtime/src/ullm_runtime_hiprtc_sources.inc` are concurrently owned by the
prefill-attention workstream, and no full-model AQ4 validation exists for such
a new kernel.

For scale only, if the `SQ8_0` full-model 1.790050x result could somehow apply
to every one of the measured 8.97854% split-core share, Amdahl's-law ceiling
would be `1 / ((1 - p) + p / 1.790050) = 1.0412625x` (+4.13%).  This is a
conditional upper bound, not an `AQ4_0` speed prediction.
