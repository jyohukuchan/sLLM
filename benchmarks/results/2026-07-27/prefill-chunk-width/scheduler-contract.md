# Resident-width contract and real-token schedule

## Why `resident_stack_width()` existed

`Sq8ServingPrefillMode::resident_stack_width()` is not a Flash2 tile selector.
It is the one M used when `load_with_prefill_mode` allocates:

1. the `Qwen3Sq8StackRuntime` layer workspace and resident hidden buffer;
2. `prompt_chunk_hidden`; and
3. CK activation/projection workspaces through
   `Qwen3Sq8LayerConfig::sequence_len`.

It was therefore parameterized, not detached: a fixed resident stack requires
one coherent allocation/shape M.  The scheduler can select M=2..4096
power-of-two widths (`m<N>-chunk<N>` in the CLI, and
`ULLM_SQ8_PREFILL_CHUNK_TOKENS=<M>` in the worker), while the default remains
M=128.

The committed product path deliberately keeps its lower measured-M admission
at `{1,2,4,8,16,32,128}`.  The layer, stack, model head, Rust CK wrapper, CK
C++ API, and paged-KV API must be widened atomically by their owners.  An
isolated overlay widened exactly those guards to determine whether they hid a
runtime kernel bound; it is measurement evidence, not production admission.

## No-padding tail contract

After at least one complete M-wide unit, a remainder rewinds to the first
real token of the final M-wide range, recomputes exactly M real tokens, and
commits only the outstanding logical suffix.  It never inserts padding,
fabricates a token, or adds an attention mask.  The exact N=4095 schedule is:

| M | units/layer | calls across 40 layers | final real-token replay / logical commit |
| ---: | ---: | ---: | --- |
| 128 | 32 | 1,280 | `3967..4094` / 127 |
| 256 | 16 | 640 | `3839..4094` / 255 |
| 512 | 8 | 320 | `3583..4094` / 511 |
| 1024 | 4 | 160 | `3071..4094` / 1,023 |
| 2048 | 2 | 80 | `2047..4094` / 2,047 |
| 4096 | 4,095 M=1 units | 163,800 | no legal 4,096-real-token replay exists |

The first five rows are both scheduler-unit-tested and observed in the
isolated full-model traces.  The M=4096 row is deliberately not a useful
N=4095 width: using an M=4096 attention unit would require a nonexistent
4096th prompt row.

## Short prompt consequence

A resident M cannot issue a smaller fixed chunk for an initial `N < M`
prompt without allocating another resident shape.  It also cannot create
rows.  These requested benchmark cells therefore correctly use audited M=1
seeds:

| resident M | N=128 | N=512 | N=1024 | N=2048 | N=4095 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1x128 | 4x128 | 8x128 | 16x128 | 32x128 (last overlaps) |
| 256 | M1x128 | 2x256 | 4x256 | 8x256 | 16x256 (last overlaps) |
| 512 | M1x128 | 1x512 | 2x512 | 4x512 | 8x512 (last overlaps) |
| 1024 | M1x128 | M1x512 | 1x1024 | 2x1024 | 4x1024 (last overlaps) |
| 2048 | M1x128 | M1x512 | M1x1024 | 1x2048 | 2x2048 (last overlaps) |
| 4096 | M1x128 | M1x512 | M1x1024 | M1x2048 | M1x4095 |

This explains the intentionally slow `*` cells in the throughput table and
why default M=128 remains appropriate for ordinary short prompts.

## Kernel finding and actual result

The direct CK probe accepted M=256..4096 for Q/O, K/V, gate/up, and down
shapes.  The F32 paged-KV writer and cached-prefix Flash2 launcher already
take runtime M; their F32 kernels have dynamic bounds/grids.  The selected
grouped-GQA CTA's 12,624 B LDS (and generic 1,296 B LDS) is per CTA, not a
persistent M-sized allocation.  Hence no BX-owned kernel source change was
needed merely to execute M=256..2048.

The full-model trace proves the dispatch reduction, but also shows the limit:
at M=2048 Flash2 still accounts for 90.134% of selected kernel time and the
N=4095 rate is 126.686 tok/s versus llama.cpp 1,008.683.  A future wide-M
Flash2 redesign is required for a material further gain.  Its evidence-based
handoff is [`lower-runtime-handoff.md`](lower-runtime-handoff.md); no kernel
body was changed in this task.
