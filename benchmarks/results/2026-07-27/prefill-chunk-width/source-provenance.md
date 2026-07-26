# Source provenance

| item | value |
| --- | --- |
| scheduler/CLI implementation | `f6b58e6c` (`feat(sq8): make serving prefill width selectable`) |
| worker opt-in | `2031e968` (`ULLM_SQ8_PREFILL_CHUNK_TOKENS`) |
| overlay harness and real-token validation | `4a067e11`, `8412e170`, `23c30630`, `88607fe0`, `43cd16dd` |
| GPU | R9700 / gfx1201 only (`HIP_VISIBLE_DEVICES=1`) |
| performance/trace run | `run-20260727T024801+0900` |
| successful numerical/decode/generation run | `run-20260727T044042+0900` (`window-finished status=0`) |
| artifact/package used by full-model overlay | Qwen3-14B-FP8 SQ8_0 product artifact/package recorded in each run configuration |
| direct CK shape probe | zeroed device buffers only; its result is shape admission, not full-model fidelity |

The first run completed the timing and trace sweep, then correctly failed
closed at its first numerical smoke because a required split-decode guard was
missing.  `88607fe0` added the guard and the successful validation-only
continuation supplied numerical, decode, and generated-text evidence.  The
two runs are deliberately reported separately in
[`measurement-summary.md`](measurement-summary.md).

The shared worktree contained concurrent BP/BX/BQ/BW changes throughout this
work.  They were not staged or committed here.  Pre-existing formatting-only
changes in `crates/ullm-engine/src/sq8_serving_runtime.rs` were also preserved
and excluded from this task's commits.

The wide-M full-model execution source lives only in the isolated overlay
under `/tmp/ullm-sq8-wide-m-overlay.9JkzMM` and its target directory.  It
exists to test lower validation bounds without modifying protected source; it
is not a product artifact or a committed runtime admission change.
