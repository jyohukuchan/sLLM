# SQ8_0 R9700 decode-attention root-cause evidence

## Scope and status

This directory records the investigation of the production decode kernel
`ullm_paged_decode_attn_f32_kernel` on the R9700 (`gfx1201`). It is evidence
for the SQ8_0 Qwen3.5-14B, 40-layer, M=1 decode path at cache length 1,036;
it is not a claim that an experimental variant is eligible for default use.

The confirmed primary problem is insufficient workgroup supply: the direct
kernel emits one 256-thread workgroup per Q head, or 40 workgroups per layer.
The supplied R9700 has 64 CUs and wave32 execution, so this launch consists of
320 waves and supplies only `320 / (64 * 32) = 15.625%` of a 32-wave-per-CU
queueing envelope. That is a launch/supply proxy, **not measured achieved
occupancy**. The exact achieved occupancy and physical HBM bandwidth are
unconfirmed; the attempted counters are retained below rather than inferred.

## Check of the original arithmetic

The supplied 58x calculation is correct when its byte count is interpreted as
the *unique resident KV footprint* rather than physical HBM traffic:

| quantity at C=1,036 | value |
|---|---:|
| unique K+V footprint, `8 * 128 * 2 * 4 * 1,036` | 8,486,912 B (8.486912 MB; 8.09375 MiB) |
| ideal time at 640 GB/s for that footprint | 13.2608 us |
| observed direct attention dispatch | 769.3306 us |
| observed / unique-footprint roof | 58.0154x |
| direct semantic K+V loads, including 5-way GQA reread, `40 * 128 * 2 * 4 * 1,036` | 42,434,560 B |
| semantic-load roof at 640 GB/s | 66.304 us |
| observed / semantic-load roof | 11.6031x |
| semantic rate if all loads reached HBM | 55.1578 GB/s |

The direct kernel maps each of 40 Q heads to one of 8 KV heads, so it scans a
KV head independently for each of its five Q-head consumers. Cache reuse may
reduce actual HBM traffic; page-table effects may increase misses. Therefore
neither 42.434560 MB nor 55.1578 GB/s is a physical HBM measurement.

The timing arithmetic is also correct: `40 * 769.3306 us = 30.773224 ms` of
attention per generated token. With the valid prior full-model baseline of
15.294955751 tok/s (65.381033 ms/token), this is 47.0675% of wall time. The
recorded llama.cpp attention time is 0.763730 ms/token, so the attention-only
ratio is 40.2933x. This is distinct from full-model throughput: llama.cpp is
30.4680750229 tok/s, versus the prior uLLM baseline's 15.294955751 tok/s, or
about 1.992x.

## Measured launch and resource characteristics

The raw rocprof trace in `phase0-pmc-root/` confirms the direct dispatch as
global `(10240, 1, 1)`, workgroup `(256, 1, 1)`: 40 workgroups, eight wave32s
per workgroup, and 320 waves per layer. Its runtime metadata records 1,024 B
LDS, 32 VGPRs, 128 SGPRs, and zero scratch bytes. Independently inspected
code-object metadata reports 25 VGPRs, 52 SGPRs, 1 KiB LDS, and no private
spills. These differ because they come from different metadata layers; neither
is a direct achieved-occupancy counter.

| implementation / dispatch | workgroups per layer | wave32s | 64 CU x 32-wave supply proxy | relevant resources |
|---|---:|---:|---:|---|
| uLLM direct paged decode | 40 | 320 | 15.625% | trace: 1 KiB LDS, 32 VGPR, 128 SGPR; static: 1 KiB LDS, 25 VGPR, 52 SGPR |
| llama.cpp vector FATTN main | 400 | 1,600 | 78.125% | static: 42,496 B LDS, 248 VGPR, 128 SGPR |
| llama.cpp FATTN combine | 40 | 160 | 7.8125% | separate merge dispatch |

The llama.cpp capture is the existing direct comparable structural record in
`benchmarks/results/2026-07-26/llamacpp-attention-analysis/`: vector FATTN
uses `(32, 40, 40)` global and `(32, 4, 1)` workgroup geometry, giving ten KV
partials per Q head, then a combine launch. It emits 17,600 attention
workgroups per generated token across 40 layers (16,000 main plus 1,600
combine), while the uLLM direct path emits 1,600. It uses F16 KV and a
continuous layout, so it is strong evidence for the parallelism mechanism,
not a format-free bandwidth comparison.

### Access and softmax structure

For each Q head, the direct fast path walks all logical source positions in
sequence. It derives a physical timestep from the paged `block_table`, then
loads a contiguous K vector and, after a score reduction, a contiguous V
vector. Lanes in a wave access adjacent vector elements, so the vector loads
are coalesced inside a page. The table lookup changes page base at the
16-token block boundary; it is not a per-lane scatter. Each score reduction
uses two CTA barriers in the normal warp-reduction path. At C=1,036 that is
2,072 CTA barriers per Q-head workgroup/layer in addition to serial page
walking, reduction, `expf`, and V accumulation.

This explains the direction of the 58x gap without pretending to allocate an
exact percentage to each factor: the direct launch underfills the GPU, it
serializes 1,036 score iterations with CTA handoffs, and it repeats GQA KV
semantic loads five times. F32 KV is another twofold traffic disadvantage
against the recorded llama.cpp F16-KV path. F32 alone cannot explain 40x;
the 40-to-400 main-workgroup contrast and serial work structure are the
dominant confirmed difference.

### Physical-byte and achieved-occupancy limitation

`phase0-pmc-root/` was collected as root on isolated R9700 with
`SQ_WAVES`, `SQ_INSTS_VALU`, and `GL2C_EA_RDREQ_{32B,64B,128B}`. For every
direct row the GL2 request and VALU values are zero, while `SQ_WAVES` is
inconsistent with the known 320-wave launch (for example 54,428 then 8,632).
The same untrustworthy zero-counter behavior occurred before privilege
escalation. This is a ROCm/gfx1201 counter-collection limitation in this
setup, not a measurement of zero traffic. The available gfx1201 event list
did not expose the `TCC_EA_RDREQ_DRAM` event used by older CDNA examples.

Consequently, **physical bytes, physical HBM bandwidth, cache hit rate, and
achieved occupancy remain unconfirmed**. The raw files are kept so a future
valid counter backend can replace this limitation without revising the launch
or numerical evidence.

## Phase 1: split-KV numerical isolation

The R9700-only minimal probe uses deterministic F32 Q/K/V, a non-contiguous
but unique paged table (`logical block i -> (13*i+7) mod 256`), and a CPU F32
two-pass reference. Raw outputs and JSON diagnostics are retained in
`phase1-probe/legacy/`.

| cache length / source tile | split count | split vs direct max abs | bit differences / 5,120 | result |
|---|---:|---:|---:|---|
| 128 / 128 | 1 | 0 | 0 | exact degeneration to direct |
| 130 / 128 | 2 | 2.9802e-8 | 2,250 | differs only after a second partial, including a tail/page boundary |
| 1,036 / 128 | 9 | 1.08033e-7 | 4,934 | finite difference, no non-finite values |

The direct and split routes are both close to the independent CPU reference
(at C=1,036, 9.6858e-8 and 1.00583e-7 maximum absolute difference,
respectively). Source inspection also checked partial-state initialization,
tail/empty-tile guards, causal bounds, invalid page handling, and the merge
scales `exp(m_i - m_new)`. The exact single-tile equality and appearance of a
small finite difference only once `split_count > 1` rule out the checked
initialization, tail, empty-tile, and obvious merge-scale bug classes for these
cases. They do not prove that no latent bug exists for every shape.

The best supported conclusion is finite-FP reassociation in partial
online-softmax/merge, amplified by SQ8_0 sequential activation quantization in
feedback decode. It is not safe to call it a proven implementation bug. The
existing full-model source-tile candidates had much larger downstream evidence
(tile 128: 90 failures and 13 hard top-1 regressions; tile 256: 10 failures
and 6 hard top-1 regressions), so the existing direct fallback for multi-tile
requests remains correct. llama.cpp demonstrates that split/merge can be
operational for its own F16-KV/weight contract, but it does not refute this
SQ8_0 sensitivity result.

## Phase 2: direct-order candidate and timing status

`ULLM_EXPERIMENTAL_PAGED_DECODE_WAVE_SCALAR_SOFTMAX=1` adds an opt-in direct
candidate. It leaves token order and the single-pass recurrence unchanged,
but lets only lane 0 in each V-owning wave update the replicated scalar
max/denominator state, then broadcasts that state to the lanes accumulating V.
It is disabled by default and incompatible with the two-pass fallback.

For C=1,036, its raw 20,480-byte direct output is byte-for-byte equal to the
default output (`fnv1a64 e7ddf1d5c0230f45`; `cmp -s` passed), recorded in
`phase2-wave-scalar-probe-after-default-guard/`. The probe's host-call plus
synchronize timing was 0.666713 ms default and 0.678809 ms candidate; that is
not a model benchmark and shows no standalone speed claim.

`phase2-full-model/` contains an attempted five-repeat, 16-step steady decode
comparison. It must be **excluded**: `ullm-openai.service` was externally
started at 20:19:35 JST while the direct run's tail and all of the candidate
run were active. The stored 14.685730 and 14.959300 tok/s values are therefore
not valid candidate/control measurements, even though their generated token
sequences match. HEAD also advanced between the two runners in the shared
worktree. No valid post-change full-model decode tok/s exists, no default was
changed, and no promotion is justified. The lightweight promotion policy
exists at `docs/plans/lightweight-promotion-policy-v0.1.md`, but the required
actual-output-quality evaluation was not run in a clean window.

## GPU-window and coexistence record

Before every GPU action, the required `pgrep -af` check was issued. Its broad
match included other Codex prompt command lines, so it was supplemented with a
process listing excluding those false positives; no matching measurement or
server workload was found before the isolated runs. All accepted probe runs
used `HIP_VISIBLE_DEVICES=1` and verified `gfx1201`. A single accidental
unisolated probe invocation only queried device information, saw `gfx1030`,
and refused before context creation or a kernel launch; no V620 compute was
performed.

`ullm-openai.service` was stopped at 20:17:27 JST for the planned isolated
window. Another session started it at 20:19:35 JST; the source of that start
is unconfirmed. The service is now active/running again, so no additional
stop/start was performed after the collision, avoiding further
`StartLimitBurst=3` consumption. `llama-qwen35-udq4.service` remained
inactive and disabled throughout. No unit file, `/opt/ullm` content, active
model manifest, or V620 workload was changed.

## File map

- `phase0-pmc-root/`: raw root rocprof trace/counter attempt and profiler-run
  probe diagnostic. The profiler changed the numerical output, so it is not
  used as a correctness or timing result.
- `phase1-probe/legacy/`: R9700 minimal direct/split numerical isolation.
- `phase2-wave-scalar-probe/` and
  `phase2-wave-scalar-probe-after-default-guard/`: opt-in direct-order
  candidate diagnostics and byte-equality evidence.
- `phase2-full-model/`: retained but explicitly invalidated contaminated
  throughput attempt.
