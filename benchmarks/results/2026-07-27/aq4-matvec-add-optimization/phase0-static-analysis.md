# AQ4_0 `matvec_add` Phase 0: static ISA and traffic audit

## Scope and provenance

This audit starts from the marker-attributed, 32-token C=1339 production
capture in
`../aq4-decode-walltime-accounting/current-p3-c4c9a9b3-service-window-20260727T011830+0900/capture/`.
That capture has 292 module launches/token and attributes 64 launches/token,
3.697842250 ms/token, and 1,241,513,984 physical weight bytes/token to
`ullm_aq4_matvec_add_f32_kernel` (335.740 GB/s, 52.4594% of the deliberately
optimistic 640 GB/s *weight-payload* roofline).  It attributes 32 launches/token,
3.402800344 ms/token, and 1,811,939,328 physical weight bytes/token to
`ullm_aq4_matvec_silu_mul_f32_kernel` (532.485 GB/s, 83.2007%).

The ISA objects in `phase0-isa/` are rebuilt from the checked-in HIPRTC raw
sources with `tools/build-aq4-projection-isa.sh`, `-O3`, and gfx1201.  Their
`SHA256SUMS` bind the source, compiler, disassembly, and summary.  Counts below
are **static code-object counts**, not a claim about dynamic per-element counts;
the source-level dynamic work is stated separately.

## What differs from `silu_mul`

| property | current add | current SiLU-mul | consequence |
|---|---:|---:|---|
| grid / workgroup | 131072 / 256 = 512 WGs | 393216 / 256 = 1536 WGs | add supplies 8 WGs/CU at 64 CUs; SiLU-mul supplies 24. Both use 8 wave32 waves/WG. R9700 exposes 32 waves/CU, so no more than 4 such WGs can be resident per CU; achieved residency still needs counters. |
| static LDS / private / spills | 0 / 0 / 0 | 0 / 0 / 0 | neither is paying an LDS-tree, scratch, or spill cost. |
| static VGPR / SGPR | 30 / 99 | 54 / 65 | add is not statically worse in VGPR or LDS. Resource pressure alone does not explain 52.46%. |
| reduction | five `ds_bpermute_b32`, no `s_barrier` | ten `ds_bpermute_b32`, no `s_barrier` | both are already wave32 shuffle reductions. Replacing a barrier/LDS tree is not an available add gain. |
| static whole-function ISA | 1434 total; 399 VALU; 922 SALU | 1932 total; 659 VALU; 1164 SALU | whole-function totals are not directly comparable because SiLU owns two AQ4 streams and the activation. They do rule out a simple “add has more static code” explanation. |

Static vector-load forms from those same code objects are below.  They are
included to make the wide-load question auditable, but are **not** dynamic
load counts: each function contains group-size, scale-validity, and loop code
paths whose execution depends on the launched shape.

| body | `global_load_b128` | `b96` | `b64` | `b32` | `u8` | vector-memory instructions |
|---|---:|---:|---:|---:|---:|---:|
| current add | 1 | 0 | 32 | 70 | 4 | 108 |
| current SiLU-mul | 2 | 0 | 4 | 60 | 32 | 99 |
| add candidate | 16 | 1 | 2 | 73 | 6 | 99 |

The important source-level difference is inside the actual g8/g16 traversal:

- One add `uint4` chunk represents 32 AQ4 elements and one weight stream. It
  performs 32 useful codebook×input accumulations. In the retained generic g8
  path it walks four groups × eight byte slots, although only four byte slots
  per g8 group are valid; 16 loop slots are predicated away per chunk. The g16
  path has two groups × eight valid byte slots, but still computes dynamic
  `byte_start`, `byte_count`, packed-word selection, and shifts.
- SiLU-mul has explicit g8 and g16 loops. For every input pair it reads the two
  input floats once and uses them for both gate and up streams. Across its two
  physical weight streams it therefore issues one input-value access per two
  weight elements, whereas add issues one per weight element. This is an
  instruction/cache-pipeline difference, not proof of corresponding DRAM
  traffic: the input vector is repeatedly cacheable.
- SiLU-mul also has three times as many WGs per launch and roughly twice the
  payload per launch. Its higher reported weight-payload bandwidth therefore
  includes more work per kernel launch and more resident scheduling waves; it
  is not a pure one-stream ISA comparison.

The following is the auditable **source-level work per physical AQ4 weight
element**.  It deliberately does not mislabel static whole-function ISA counts
as dynamic per-element counters: the exact executed VALU count still depends
on the g8/g16 branch, valid scale-index predicates, and compiler scheduling.

| work | add (one AQ4 stream) | SiLU-mul (gate + up, two AQ4 streams) | implication |
|---|---:|---:|---|
| packed index payload | 0.5 B | 0.5 B per stream | same format cost per physical weight |
| scale-index payload | 1/8 B (g8) or 1/16 B (g16) | same per stream | format does not explain the gap |
| input f32 source accesses | 1 per weight element | 1 shared by two weight elements | SiLU issues half as many input accesses per physical stream element |
| codebook lookups / FMACs | 1 / 1 | 2 / 2 total (1 / 1 per stream) | the fused body performs more arithmetic, but reuses the input value |
| row reduction | 5 wave32 `ds_bpermute` + adds | 10 (one independent reduction per stream) | add is already cheaper here |

The current add source loads one 16-byte `uint4` per 32-element chunk; its
generic g8 traversal then visits 32 byte-loop slots but only 16 are valid.
The SiLU source has explicit g8/g16 element loops and keeps `input_low` /
`input_high` live while applying both codebooks.  ISA confirms that neither
body has a barrier/LDS reduction or spills, but only a locked PMC capture can
turn the source-level access count into measured VALU-per-element, cache, or
HBM traffic.  The candidate window therefore requests `SQ_INSTS_VALU`,
`SQ_WAVES`, `SQ_WAVE_CYCLES`, and occupancy as diagnostic evidence; an
unsupported/zero raw counter is reported as unconfirmed rather than treated as
a measured zero.

At the source level, both add and each physical SiLU stream retain the same
irreducible useful inner work: one codebook decode and one f32 multiply-accumulate
per AQ4 element, followed by a group-scale operation and a five-step/32-lane
row reduction.  The candidate does **not** claim to remove those 32 useful
accumulations from an add `uint4`.  It removes traversal/control work around
them: per 32 elements generic g8 executes 16 invalid byte-slot predicates
(four invalid byte slots in each of four groups), while generic g16 executes
two invalid group-slot predicates and dynamic byte-start/word-selection
addressing.  The exact dynamic VALU count per element is unconfirmed because
gfx1201 raw `SQ_INSTS_VALU` has historically reported zero even for nontrivial
kernels; static code and the source-level accounting must not be relabelled as
a hardware-counter measurement.

This explains why copying SiLU's reduction cannot help: add already has the
same no-barrier reduction.  The transferable part is the explicit g8/g16
traversal, not the two-stream fusion itself.

## Priority 2/3 scope check

The lower-priority `matvec_triple` and `matvec_qkv_z_gate_beta` paths were
also inspected before changing shared code.  Their production paired/triple
helpers already load `input_low`/`input_high` once and apply them to their
multiple AQ4 streams.  `qkv_z_gate_beta` additionally already uses its
two-wave shuffle-plus-minimal-LDS combine at RPB=4.  Thus the transferable
one-stream traversal/input-access asymmetry found in add is not present in the
same form, and no unmeasured change was made to either lower-priority kernel.

The add payload is also not evenly split between the two group formats.  Of
its 1,241,513,984 B/token, g16 `mlp_down` is 905,969,664 B (72.97%), while
g8 `linear_attn_out` is 251,658,240 B (20.27%) and g8 `attn_o` is 83,886,080 B
(6.76%).  A useful traversal candidate must therefore retain a strong g16
path; a g8-only microbenchmark would not represent the production target.

## Residual/add traffic check

The `add` family has 64 launches/token and each produces 4096 f32 values.  A
residual read plus output write is therefore at most
`64 × 4096 × 4 × 2 = 2,097,152 B/token` before cache effects.  This is 0.1689%
of the 1,241,513,984 B/token weight payload.  The distinct input vectors have
the same 2 MiB upper-bound footprint across the listed 32 small and 32 down
projections; row scales add at most another 0.5 MiB for the row-scaled calls.

Thus the residual read/write itself cannot account for a 30.74-percentage-point
gap to SiLU-mul's payload metric.  The metric deliberately excludes activation,
residual, output, scale-index, cache-miss, and instruction costs, so it is
still not total-DRAM bandwidth.  The exact split between cache behavior,
instruction issue, and occupancy is **not yet confirmed** without a
counter-based R9700 run.

## Candidate selected from the evidence

The production gfx1201/RPB=8 body now retains the previous shuffle body as a
direct-only baseline and replaces only the traverser for the host-validated
g8/g16, `cols % 32 == 0` contract:

- g8: four fixed packed words, four fixed bytes each;
- g16: x/y then z/w, four fixed bytes per word;
- low nibble remains first; g8/g16 scale-table indices are unchanged;
- group accumulation, row-scale application, f32 output, and
  `output[row] = residual[row] + value` remain in the inherited epilogue.

The static candidate is smaller (1434 → 820 instructions; 922 → 395 SALU;
399 → 321 VALU) while preserving no LDS/spills, but it raises VGPR 30 → 49.
That trade-off is intentionally not accepted on static ISA alone.  The next
gate is a locked R9700 differential against the retained shuffle baseline,
including the two production shapes, greedy/runtime checks, and a
counter/occupancy observation before any full-model decision.

The larger static count of `global_load_b128` in the candidate is not eight
loads of the same packed word.  The gfx1201 disassembly shows consecutive
offsets `-124, -108, …, -12`, i.e. eight distinct 16-byte AQ4 chunks in one
unrolled body (the other group-size arm has its own eight-chunk body).  The
generic source has one such load in its loop body.  This is a deliberate
instruction-level trade: more independent in-flight chunks and less loop
control can raise memory-level parallelism, at the cost of the observed VGPR
increase.  It must be judged by the R9700 full-model A/B rather than by the
static count alone.
