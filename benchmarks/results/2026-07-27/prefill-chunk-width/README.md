# SQ8_0 prefill chunk-width expansion

## Result status

The scheduler width is now selectable for power-of-two M=2..4096, with the
default unchanged at M=128.  `resident_stack_width()` was parameterized but
not removed: it is the allocation/shape contract shared by the resident stack,
prompt hidden buffer, activation buffer, and CK projection workspace.

The product source retains its lower measured-M admission gate until the
BP/BX-owned files can be changed atomically.  An isolated source overlay
lifted only those validation gates and produced the requested full-model
evidence for M=256..2048 on R9700.  It did not modify either BX-owned Flash2
source file and is not a production admission change.

| question | completed result |
| --- | --- |
| fixed `resident_stack_width()` reason | allocation/shape coupling, not an attention tile restriction |
| no-padding tail | preserved for M=128/256/512/1024/2048; only real-token suffix replay is used |
| widest analytical resident fit with observed AQ4_0 | M=4096, with 6.424 GiB analytical headroom; actual co-resident load remains unmeasured |
| widest useful M at N=4095 | M=2048; M=4096 cannot form a real 4096-token unit without a fabricated row |
| actual N=4095 attention calls | 1,280 / 640 / 320 / 160 / **80** for M=128/256/512/1024/2048 |
| best N=4095 rate | M=2048: **126.686 tok/s**, versus M=128 104.965 and llama.cpp 1,008.683 |
| wide-M numerical result | byte-exact whenever the selected M actually executes; M=1 fallback differences are recorded and have no non-finite values |
| text result | 10-case suite: no obvious-collapse findings; real-token N=4000 completion: identical IDs/text for all M |
| fresh decode regression | M=128: 27.552769 tok/s versus 27.378731 reference |

The full results are in [`measurement-summary.md`](measurement-summary.md).
It records the split-run provenance, all five-repeat rates, trace counts,
numerical evidence, generation results, and the kernel handoff conclusion.

## Files

| file | purpose |
| --- | --- |
| `measurement-summary.md` | final full-model result and evidence interpretation |
| `memory-accounting.md` | SQ8_0 allocation contract and AQ4_0 co-residency calculation |
| `scheduler-contract.md` | fixed-width rationale, tail proof, actual trace counts, and short-prompt behavior |
| `throughput.md` | full five-prompt, five-width rate grid against llama.cpp |
| `validation-status.md` | tests, numerical/text fidelity, trace, decode, and service-window status |
| `lower-runtime-handoff.md` | permanent lower-admission and BX wide-M performance continuation contract |
| `wide-m-overlay.md` | exact boundary of the temporary full-model execution overlay |
| `run-20260727T024801+0900/` | raw throughput and trace sweep; first numerical guard failure is recorded |
| `run-20260727T044042+0900/` | successful numerical/decode/generation continuation, thermal and service records |

## Reproduction boundary

The rate sweep follows
`../../2026-07-26/r9700-prefill-comparison/conditions.md` and `accounting.md`:
R9700 gfx1201 only, one sequence, same-length warm-up, five unprofiled timed
repetitions, and prompt lengths 128/512/1024/2048/4095.  The trace is used
only for dispatch and kernel-accounting evidence, never as a throughput timer.

The validation-only continuation used one additional service window after the
missing split-decode guard was corrected.  It completed with `status=0`,
released `/run/ullm/r9700.lock`, restored `ullm-openai.service` to active,
and retained `NRestarts=0`.  `llama-qwen35-udq4.service` remained inactive and
disabled throughout.
