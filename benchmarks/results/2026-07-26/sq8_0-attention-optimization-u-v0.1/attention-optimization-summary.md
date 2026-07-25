# `SQ8_0` R9700 attention optimization result v0.1

- Date: 2026-07-26
- Target: R9700 only — `gfx1201`, AMD SMI GPU 2, PCI `0000:47:00.0`.
- Scope: isolated canonical `SQ8_0` artifact/package process; this was not a
  live served-SQ8 process.  No activation, campaign, authorization, release,
  `/opt/ullm` write, external ABI, or direct legacy dispatch was changed.
- Decision: **Flash2 staged body is NO-GO; retain the normal body.**  The
  paged split measurements are an explicit-API result only; no automatic
  dispatch was changed.

## PMC disposition

The ROCm 7.2.1 SDK counter definition file contains gfx1201 definitions for
`FETCH_SIZE`, `VALUInsts`, raw `SQ_INSTS_VALU`, and raw
`GL2C_EA_RDREQ_{32B,64B,128B}`.  Names and architecture definitions are thus
not the explanation for the zero values.

The purpose-built load+FMA probe establishes the failure below.  Each raw
counter collection had matching kernel trace records, and `SQ_WAVES` was
nonzero (`32768` per dispatch), so dispatch tracing itself worked.

| collection | observation |
|---|---|
| raw SQ | `SQ_INSTS_VALU=0` while `SQ_WAVES=32768` |
| raw GL2C | all of `GL2C_EA_RDREQ_32B`, `_64B`, and `_128B` were zero |
| derived probe | `FETCH_SIZE=0`, `VALUInsts=0`, `Wavefronts=32768` |
| selected Flash2 body | all 160 observed launches reported `FETCH_SIZE=0` and `VALUInsts=0`; `Wavefronts=40960` per launch |

Raw files: `pmc/probe-{raw-sq,raw-gl2c,derived}/data/` and
`pmc/target-derived/data/`.  This rules out a derived-metric spelling/formula
problem, but the exact low-level cause (driver/firmware/permission or ROCm
counter-programming behavior) is **未確認**.  A root-only retry was deliberately
not opened after the service start-limit budget was consumed.  Therefore
physical HBM efficiency and a final memory-bound-versus-compute-bound verdict
remain **未確認**.  The admissible substitutes are the prior logical KV rates,
the captured ISA/resource metadata, and workgroup-supply geometry.

## Flash2 staged wave32 prototype

The isolated HIPRTC source exposes separate legacy, QK-only, QK+max, and
QK+max+sum symbols.  It leaves the normal runtime symbol as the default.  Its
static all-staged resource record is wave32, LDS 1296 B, VGPR 27, SGPR 48,
private/spills 0; the legacy reference is LDS 1296 B, VGPR 21, SGPR 46,
private/spills 0.  See `static/prototype.metadata.txt` and
`flash2/standalone-fixed/prototype.metadata.txt`.

Standalone differential results have no NaN/Inf.  The all-staged maximum
absolute difference versus the separate legacy symbol was `1.1920929e-7`
(short), `1.0430813e-7` (63→68 tail), `2.9802322e-8` (synthetic 896→1024
M=128 shape), and `2.6464462e-5` (adversarial score range).  The standalone
synthetic kernel timing was legacy `13.317192 ms` versus staged `12.876236 ms`
per launch (1.03425x).  This is deliberately not reported as serving
throughput.

The unprofiled canonical-artifact `raw-p0512` vLLM-source fixture baseline
completed four M=128 prefill units in `1.167487403 s` total (`438.548629`
input tokens/s); it generated token 66.  The staged full-model run also
generated token 66, but failed the frozen SQ8 vector gate:

| capture | max abs | relative L2 | cosine | result |
|---|---:|---:|---:|---|
| final hidden | 0.7760314941 | 0.0145683599 | 0.9999164687 | fail |
| logits | 0.2401080132 | 0.0084836396 | 0.9999792756 | fail |

The gate was `max_abs <= 2e-5`, `relative_l2 <= 1e-5`, and
`cosine >= 0.999999`; see `flash2/differential/`.  The staged run was also
contaminated as a performance sample by a detected brief service restart, so
its timing is not used.  The very large vector mismatch is independently
sufficient to reject the body.  The follow-up generalized the prototype's
cross-wave handoff from a fixed eight waves to `blockDim.x`; actual Flash2
records use 256 threads/eight waves, so that edit is behaviorally identical
for this measured geometry and cannot reverse the failed gate.  A later retry
aborted before HIP work because its prompt path was relative; its raw record
is retained in `flash2/serving-staged-fixed/` but is not a measurement.

Consequently the production Flash2 symbol was **not** replaced.

## Paged decode explicit source-tile split

`decode/split-bench.json` calls only the existing explicit split API; direct
legacy dispatch remains unchanged.  At M=1 / C=1036, its host-API plus stream
synchronize timing and numerical differential were:

| path | mean ms | max abs vs direct | split count | partial-WG wave supply |
|---|---:|---:|---:|---:|
| direct legacy | 0.643241770 | reference | 1 | 320 waves / 15.625% |
| tile 128 | 0.228016370 | 1.34110e-7 | 9 | 2880 waves / 140.625% |
| tile 256 | 0.227932360 | 1.26660e-7 | 5 | 1600 waves / 78.125% |
| tile 512 | 0.383530140 | 1.34110e-7 | 3 | 960 waves / 46.875% |

All split outputs were finite.  Tile 256 is marginally fastest among the two
near-tied best tiles (2.822x lower attention-call time than direct); tile 512
regresses.  This supports the workgroup-supply hypothesis, but is not a full
model end-to-end claim and is not an authorization to alter direct dispatch.

## `uint4` and operating record

`uint4` wide-load/lane re-layout was **not started**: raw physical PMC values
remain unusable, so the prerequisite lane/physical-traffic validation is
unmet.

`llama-qwen35-udq4.service` was recorded `inactive` and `disabled`, and
`gdm3.service` was `inactive` before the measurement.  R9700 telemetry is in
`telemetry/`; pre/post samples were unthrottled (for example, before the
primary stop: edge/hotspot/memory `36/37/34 C`, gfx `81 MHz`, socket `13 W`).
No in-kernel peak telemetry was captured, so in-run thermal peak is
**未確認**.

The primary test script stopped the service at 05:05:32+09:00 and restored it
active at 05:07:48+09:00.  A tool-lifecycle misread caused one manual start at
05:06:24 and a compensating stop at 05:06:51; this is why staged serving timing
is discarded.  A later path-error attempt stopped/restored at 05:10:38 without
running a GPU kernel.  Exact records are under `service/`, and the final state
was `ullm-openai.service=active/running`, `NRestarts=0`.
