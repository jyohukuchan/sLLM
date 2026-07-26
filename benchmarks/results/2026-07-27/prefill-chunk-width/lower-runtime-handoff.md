# Lower-runtime wide-M handoff

## What is already proven

The serving scheduler can select M=256/512/1024/2048/4096 and preserves
real-token tail replay. The direct CK helper accepted all four Qwen3-14B
projection shapes for every requested M; see `wide-m-ck-shape-probe.jsonl`.
For M other than 128, the existing C++ selector already uses:

| projections | implementation |
| --- | --- |
| Q, K, V, O | `DefaultTile16x128x128` (ID 1) |
| gate, up | `KPaddingTile16x128x256` (ID 2) |
| down | `DefaultTile16x128x256` (ID 4) |

This is the same branch selected by the direct probe. No new CK kernel or
Flash2 kernel source is required for the width change.

## Remaining synchronized changes

The following files must be changed atomically for an executable M=256 path.
The first two are BP-owned; do not change them concurrently with BP.

| file | exact contract to extend |
| --- | --- |
| `crates/ullm-engine/src/sq8_layer_runtime.rs` | prefill option list, `is_qwen3_14b_sq8_prefill_chunk_tokens`, layer `sequence_len` validation, cached-prefix admission, and their unit tests |
| `crates/ullm-engine/src/sq8_stack_runtime.rs` | stack/report/chunk admission, `validate_measured_ck_dispatch`, and fixed-width tests |
| `crates/ullm-runtime-sys/src/lib_parts/sq8_ck.rs` | activation-buffer whitelist and its diagnostic/tests |
| `runtime/src/ullm_runtime_api_sq8_ck.inc` | `sq8_ck_m_is_measured`; `sq8_ck_projection_implementation` already selects the wide-M IDs above |

Start by admitting `256` everywhere, then only add 512/1024/2048 after the
previous full-model result has been reviewed. M=4096 is a shape/capacity case,
but it cannot reduce the no-padding N=4095 schedule.

## Required evidence after each admission

1. Create a fresh resident model at the chosen width and run all five prompt
   lengths using the existing five-repeat, unprofiled accounting.
2. Capture a real N=4095 trace. Expected cached-prefix Flash2 calls are 640
   at M=256, then 320/160/80 at M=512/1024/2048.
3. Compare hidden state and logits with M=128; record any non-bitwise result,
   but do not use a new scalar threshold as the decision gate.
4. Run actual greedy/generated text and apply the lightweight-promotion policy
   for obvious breakage.
5. Re-run the decode condition to show the independent M=1 path remains near
   its 27.378731 tok/s reference.

`runtime/src/ullm_runtime_parts/part_01.inc` and
`runtime/src/ullm_runtime_hiprtc_sources.inc` are deliberately absent from
this handoff: the launch is dynamic in `new_tokens`, and BX owns those files.
