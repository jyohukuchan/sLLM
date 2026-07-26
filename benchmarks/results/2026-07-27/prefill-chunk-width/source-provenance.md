# Source provenance

| item | value |
| --- | --- |
| scheduler/CLI implementation commit | `f6b58e6c` (`feat(sq8): make serving prefill width selectable`) |
| parent at implementation start | `b6dc2389` |
| GPU | R9700 / gfx1201 only (`HIP_VISIBLE_DEVICES=1`) |
| artifact/package used by direct CK probe | none; the probe uses zeroed device buffers and the existing helper |
| full-model wide-M run | not attempted because layer/stack measured-M admission remains closed |

The workspace contained unrelated concurrent edits in the BP/BX/BQ/BW-owned
areas throughout this work. They were neither staged nor committed here. The
three pre-existing formatting-only edits in
`crates/ullm-engine/src/sq8_serving_runtime.rs` were likewise preserved in the
working tree and excluded from `f6b58e6c`.
