# SQ8_0 prefill chunk-width expansion

## Result status

This evidence set separates work that was executed from work that is only
analytically or scheduler-verified. It does **not** claim a wider-M model
throughput result before the layer/stack/CK execution contract accepts that M.

| question | result in this run |
| --- | --- |
| why `resident_stack_width()` was fixed | It binds the resident stack, prompt hidden buffer, and CK workspaces to one M; it is an allocation/shape contract, not a Flash2 tile restriction. |
| scheduler-selectable widths | implemented for power-of-two M=2..4096; default remains M=128 |
| executable widths today | existing measured lower contract only: M in `{1,2,4,8,16,32,128}` |
| no-padding tail | unit-tested for M=128/256/512/1024/2048; every replay contains only real tokens |
| direct M=256+ CK shape probe | passed: M=256/512/1024/2048/4096 each accepted all four SQ8_0 projection shapes; raw JSONL retained |
| wider full-model prefill rate / trace / generation | **unconfirmed**: runtime admission rejects before allocation/dispatch |
| decode behavior | no decode code or selector changed; the immediately preceding BR run measured 27.411786 tok/s against 27.378731 reference |

The exact lower blockers and the schedule are in
[`scheduler-contract.md`](scheduler-contract.md). Allocation arithmetic and
its limits are in [`memory-accounting.md`](memory-accounting.md).

## Files

| file | purpose |
| --- | --- |
| `memory-accounting.md` | SQ8_0 allocation contract and AQ4_0 co-residency calculation |
| `scheduler-contract.md` | fixed-width rationale, tail proof, planned call counts, attention-kernel finding |
| `throughput.md` | same-accounting M=128 control and explicit unmeasured wide-M rows |
| `validation-status.md` | correctness, trace, and decode evidence/status |
| `wide_m_ck_shape_probe.cpp` | direct GPU shape-admission probe for the four SQ8_0 projection shapes |
| `wide-m-ck-shape-probe.jsonl` | 24 successful direct-CK shape-admission rows |
| `lower-runtime-handoff.md` | exact BP/CK/API follow-on contract and validation sequence |

## Reproduction boundary

The requested performance protocol is the one recorded in
`../../2026-07-26/r9700-prefill-comparison/conditions.md` and `accounting.md`:
R9700 gfx1201 only, one sequence, one same-length warm-up, five unprofiled
timed repetitions, and 128/512/1024/2048/4095 prompt lengths. The preceding
M=128 control follows that protocol. This work did not substitute profiler
range duration or a kernel-only timing for throughput.

Before any GPU operation, the owner must check `/run/ullm/r9700.lock`, the
listed benchmark processes, and `ullm-openai.service`; a held lock is never
stolen. The run was deliberately paused while the gateway held that lock, so
no service start/stop or active-manifest change was made here.
