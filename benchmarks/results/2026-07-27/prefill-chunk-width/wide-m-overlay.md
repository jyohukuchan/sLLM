# Isolated wide-M execution overlay

## Purpose and boundary

The shared checkout has protected in-flight BP/BX work in the layer, stack,
model-head, CK, and KV runtime files.  To distinguish a real kernel limit
from their conservative M=128 validation lists, a source-only copy was built
under `/tmp/ullm-sq8-wide-m-overlay.9JkzMM`.  It is not a committed product
change and no overlay artifact was promoted or served.

## Temporary admissions

The overlay raises the selected M set through 4096 in the layer, stack,
model-head, Rust CK, and CK C++ validation paths; raises the layer-oracle
shape ceiling; and changes only the F32/typed paged-KV **API validation** from
`m <= 128` to `m <= 4096`.  It does not alter either BX-owned kernel source:

- `runtime/src/ullm_runtime_parts/part_01.inc`
- `runtime/src/ullm_runtime_hiprtc_sources.inc`

The temporary driver writes its width and 40-layer-expanded cached-prefix
call count to JSONL.  The overlay serving executable and driver compiled for
`gfx1201` into `/tmp/ullm-sq8-wide-m-target`.

## Full-model outcome

The overlay ran all M=128/256/512/1024/2048 throughput conditions and actual
N=4095 traces in `run-20260727T024801+0900`.  It then ran hidden/logit,
decode, fixed-suite generation, and real-token N=4000 generation in
`run-20260727T044042+0900`.  The latter completed `status=0`, released the
R9700 lock, and restored `ullm-openai.service`.

The overlay proves the lower M=128 bounds are not an execution limit for
M=256..2048.  It does **not** authorize copying its temporary whitelist edits
into protected product files.  Permanent integration needs the atomic
owner-reviewed changes in [`lower-runtime-handoff.md`](lower-runtime-handoff.md).
The actual rate/trace/fidelity result is in
[`measurement-summary.md`](measurement-summary.md).
