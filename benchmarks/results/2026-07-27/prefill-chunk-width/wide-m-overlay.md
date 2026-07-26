# Isolated wide-M execution overlay

## Purpose

The shared checkout has protected in-flight work in the layer/stack/model-head
and KV runtime files.  To determine whether the remaining bounds are real
kernel limits without editing those files, a source-only copy was built under
`/tmp/ullm-sq8-wide-m-overlay.9JkzMM`.  It is not a committed product change.

## Temporary changes

The overlay raises the selected M set through 4096 in the layer, stack,
model-head, Rust CK, and CK C++ validation paths; raises the layer-oracle
shape ceiling; and changes only the F32/typed paged-KV **API validation** from
`m <= 128` to `m <= 4096`.  It does not alter either BX-owned file:

- `runtime/src/ullm_runtime_parts/part_01.inc`
- `runtime/src/ullm_runtime_hiprtc_sources.inc`

The temporary build also contains a derivative of the historical prefill
driver that accepts `--chunk-tokens`; it writes the selected width and its
40-layer-expanded cached-prefix call count in its JSONL records.

## Build result

Both the overlay serving executable and width-aware driver compiled with
`GPU_ARCH=gfx1201` and `rocm-ck-gfx1201` into the isolated target directory
`/tmp/ullm-sq8-wide-m-target`.  This is a build/admission check only.  No
overlay GPU execution has begun: at the preflight the production gateway held
`/run/ullm/r9700.lock`, so the task is waiting rather than taking the R9700.

The overlay is evidence for the next synchronized implementation, not an
authorization to copy its temporary whitelist changes into protected files.
