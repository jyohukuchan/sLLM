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
Flash2 kernel source is required to make a wide M *executable*.

## Remaining synchronized changes

The following files must be changed atomically for an executable M=256 path.
The first two are BP-owned; do not change them concurrently with BP.

| file | exact contract to extend |
| --- | --- |
| `crates/ullm-engine/src/sq8_layer_oracle.rs` | `QWEN3_14B_SQ8_LAYER_ORACLE_MAX_SEQUENCE_LEN` currently caps the layer configuration at 128; lift its validation ceiling consistently with the 4096-token serving context, or split the runtime shape ceiling from the host-oracle ceiling if that oracle must remain 128-only |
| `crates/ullm-engine/src/sq8_layer_runtime.rs` | prefill option list, `is_qwen3_14b_sq8_prefill_chunk_tokens`, layer `sequence_len` validation, cached-prefix admission, and their unit tests |
| `crates/ullm-engine/src/sq8_stack_runtime.rs` | stack/report/chunk admission, `validate_measured_ck_dispatch`, and fixed-width tests |
| `crates/ullm-engine/src/sq8_model_head_runtime.rs` | wide-M report validation and `selected_row_offset_bytes`, so the final real row of a chunk can reach the BF16 model head |
| `crates/ullm-runtime-sys/src/lib_parts/sq8_ck.rs` | activation-buffer whitelist and its diagnostic/tests |
| `runtime/src/ullm_runtime_api_sq8_ck.inc` | `sq8_ck_m_is_measured`; `sq8_ck_projection_implementation` already selects the wide-M IDs above |
| `runtime/src/ullm_runtime_api_attention.inc` | F32 and typed paged-KV chunk API guards currently reject `m > 128`; coordinate this edit with BX's in-flight KV-dtype work. The existing F32 HIP launcher/kernel take runtime `m`, so this is an admission/validation change, not a request to change BX-owned HIPRTC or launcher source |

The isolated overlay applied this synchronized admission set through M=4096
without changing the protected Flash2 source files.  It full-model-ran
M=256/512/1024/2048; its results below remove the need to gate the permanent
landing on an additional M=256 smoke.  Permanent product integration should
still land the listed paths atomically, preserve the default M=128, and retain
their diagnostics/tests.  M=4096 remains a shape/capacity case and cannot
reduce the no-padding N=4095 schedule.

## Evidence completed in the isolated overlay

1. The overlay created fresh resident models and ran all five widths at all
   five prompt lengths using the required five-repeat unprofiled accounting.
2. Real N=4095 traces observed exactly 640/320/160/80 cached-prefix Flash2
   calls for M=256/512/1024/2048.
3. Every prompt that actually used its selected M was F32-byte-exact versus
   M=128.  N<M M=1 fallback differences were recorded, had no non-finite
   values, and did not change the one-token greedy output.
4. The fixed 10-case text suite had no obvious-collapse finding.  The real
   N=4000 prompt produced identical token IDs/text across M=128..2048.
5. Fresh decode was 27.552769 tok/s versus the 27.378731 reference.

`runtime/src/ullm_runtime_parts/part_01.inc` and
`runtime/src/ullm_runtime_hiprtc_sources.inc` are deliberately absent from
this handoff: cached-prefix attention is dynamic in `new_tokens`, and the F32
paged-KV writer is already dynamic in `m`; BX owns both files.  A future trace
or M=256 smoke failure could invalidate that source-read conclusion, in which
case the failure must be handed to BX rather than patched concurrently.

## BX wide-M performance continuation

The overlay removes the ambiguity about functional M support but exposes a
separate performance requirement.  At N=4095, attention calls fall from
1,280 to 80 at M=2048, yet the selected grouped Flash2 kernel remains 90.134%
of selected kernel time and full-model rate reaches only 126.686 tok/s
(llama.cpp reference: 1,008.683).  Therefore a material next gain requires a
wide-M Flash2 algorithm/body investigation, not another whitelist change.

The continuation must preserve these verified contracts:

1. accept runtime `new_tokens`/M rather than reintroducing a static M=128
   guard; Qwen3-14B remains `value_dim=128 <= 256`;
2. preserve causal online-softmax and F32 paged-KV semantics for all real
   rows, including the cursor-rewound final suffix; do not use padding,
   fabricated rows, or a mask as a width workaround;
3. differential hidden/logit output before timing, record non-finite values,
   and use actual generated text rather than a new scalar numerical gate;
4. measure full-model five-repeat prefill and real N=4095 traces.  A
   kernel-only multiplier or a profiler-duration throughput claim is not a
   decision result; and
5. re-run the independent decode condition after any shared runtime change.

The evidence baseline and raw traces are
[`measurement-summary.md`](measurement-summary.md) and
`run-20260727T024801+0900/traces/`.  This file is the requested handoff
location for BX; no BX-owned kernel source was modified in this task.
