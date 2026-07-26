# Resident-width contract and 4095-token schedule

## What was fixed, and why

`Sq8ServingPrefillMode::resident_stack_width()` is not an attention-kernel
tile selector.  It is the single resident M used when `load_with_prefill_mode`
allocates all of these together:

1. `Qwen3Sq8StackRuntime`'s `Qwen3Sq8LayerWorkspace` and resident hidden;
2. the serving `prompt_chunk_hidden` buffer; and
3. the CK activation / projection workspaces whose `m` is the same
   `Qwen3Sq8LayerConfig::sequence_len`.

The original fixed-M admission is additionally repeated below the scheduler:
`Qwen3Sq8LayerConfig::validate`, `Qwen3Sq8StackRuntime`, the Rust CK helper,
and `runtime/src/ullm_runtime_api_sq8_ck.inc` each accept only the measured
set `{1,2,4,8,16,32,128}`. The latter two reject before dispatch. This is the
reason a scheduler-only width change cannot execute a full M=256 model yet;
it is not a Flash2 requirement. A direct helper probe has now shown that CK
can *shape-admit* the four Qwen3-14B projections at M=256..4096, but it did
not bypass the layer/stack/API contract in a model run.

The 2026-07-26 tail journal was correct on the allocation coupling, but also
established that the **prefix-position alignment** was only scheduler policy:
it can be rewound for a fixed-width real-token suffix without changing the
resident workspace.

## Scheduler implementation in this change

`Sq8ServingPrefillMode::fixed_chunk_tokens(M)` now accepts power-of-two
fixed widths from 2 through 4096 and retains the legacy variants for 8, 32,
and 128.  The selector is also accepted by the `sq8_ck_serving` CLI as
`m<N>-chunk<N>`.  It has two deliberate admissions:

- scheduler admission validates the power-of-two/context contract and plans
  a no-padding schedule;
- runtime admission checks the existing measured-M lower contract before any
  model allocation.  Today that accepts the legacy measured widths only and
  returns an explicit error for M>128.

This separation makes the proposed width visible and testable without
pretending that an unmeasured CK/layer/stack path has executed.

For a suffix after at least one complete fixed chunk, the planner rewinds to
the first real token of the final M-wide range, computes exactly M real
tokens, and commits only the remaining logical tokens.  It neither inserts a
fake token nor uses padding/masking.  M=4096 with a 4095-token prompt has no
earlier real M-wide range, so it intentionally remains 4095 M=1 seeds.
Likewise, a prompt shorter than its selected M remains on the audited M=1
seed path; for example M=256 cannot reduce the 128-token case without
introducing the prohibited fake rows.

## 4095-token consequences

| fixed M | execution units/layer | Flash2 calls across 40 layers | final real-token replay |
| ---: | ---: | ---: | --- |
| 128 | 32 | 1,280 | logical 3968, execution 3967..4094, commit 127 |
| 256 | 16 | 640 | logical 3840, execution 3839..4094, commit 255 |
| 512 | 8 | 320 | logical 3584, execution 3583..4094, commit 511 |
| 1024 | 4 | 160 | logical 3072, execution 3071..4094, commit 1023 |
| 2048 | 2 | 80 | logical 2048, execution 2047..4094, commit 2047 |
| 4096 | 4095 M=1 units | 163,800 | no legal fixed replay exists at N=4095 |

The first five rows are unit-tested scheduler facts, not an on-GPU trace.
The old M=128 value is the BR trace baseline.  The lower-layer contract stops
a full-model trace for the new widths before execution, so an actual M=256+
attention-dispatch count is **unconfirmed** in this evidence set.

## CK and attention-kernel finding

`wide_m_ck_shape_probe.cpp` bypassed only the public measured-M gates and
called the existing gfx1201 CK helper with zeroed buffers. For M=256, 512,
1024, 2048, and 4096 it completed activation quantization and all four
Qwen3-14B projection shapes: Q/O (5120x5120), K/V (1024x5120), gate/up
(17408x5120), and down (5120x17408). The raw 24-row result is
`wide-m-ck-shape-probe.jsonl`. It establishes that the existing helper's
generic M/N/K argument construction supports those shapes; it does not prove
real-weight numerical fidelity, full-layer correctness, or throughput.

The F32 Flash2 launcher passes `new_tokens` as a runtime argument and sets
its grid to `new_tokens * launch_heads`; its only relevant bound here is
`value_dim <= 256`.  For Qwen3-14B's R9700 grouped-GQA path,
`launch_heads=kv_heads=8`; for the generic path it is 40.  There is no
M=128 specialization or resident M-sized attention scratch allocation in
either `ullm_runtime_hiprtc_sources.inc` or `part_01.inc`.

Therefore no BX-owned attention source change is needed for wider M.  The
required next implementation work is the lower CK/layer/stack admission
contract, documented in the plan handoff; Flash2 should remain untouched
unless a full-model wide-M trace exposes a separate correctness or launch
limit.

## Paged KV-write finding after the initial handoff

The full cached-prefix path has one additional M=128 admission that is not an
attention restriction: `ullm_runtime_paged_kv_write_chunk_f32` and its typed
entry point in `runtime/src/ullm_runtime_api_attention.inc` reject `m > 128`.
The F32 implementation's existing HIP launcher is already dynamic: it passes
`m` to the kernel and launches `ceil(m * (kv_heads * head_dim + kv_heads *
value_dim) / 256)` CTAs.  Its HIPRTC writer likewise uses runtime `m` and
global bounds, rather than a 128-row tile.  Thus the source read finds an API
validation bound, not an identified need to edit either BX-owned HIPRTC or
launcher source.

This file is currently being edited as part of BX's KV-dtype work, so this
task deliberately does not modify it.  The synchronized wide-M change must
raise the two chunk-writer checks to the selected context-safe maximum and
then demonstrate real F32-KV writes at M=256 before admitting larger widths.
The usual overflow, cache-range, and block-table checks already scale with
`m` and must remain intact.
