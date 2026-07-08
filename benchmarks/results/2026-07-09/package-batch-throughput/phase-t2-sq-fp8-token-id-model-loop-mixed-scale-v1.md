# T2 SQ FP8 Model-Loop Mixed Scale v1

## 前回の要点

- k/up row-block32は、coverageをlayer3だけまで削っても `case_a` が崩れた。
- `up_proj` layer3 row-block32と `k_proj` layer3 row-block16は単体ではstrict top1を維持した。
- ただし `k/up` layer3 row-block16は組み合わせると崩れたため、mixed scale policyを試す必要があった。

## 今回の変更点

- `tools/build-sq-fp8-w8a16-artifact.py` がpolicy `scale.overrides[]` を読み、tensorごとに異なるrow-block幅をmanifestへ保存できるようにした。
- candidate-level `scale_granularity=mixed`、`scale_layout=per_tensor` を追加し、各 `fp8_tensors[]` entryのscale metadataをauthoritativeにした。
- `k_proj` row-block16 + `up_proj` row-block32 のmixed policyをlayer3、layers `3,7` でR9700 model-loop prompt bundleにかけた。

## Summary

| variant | FP8 tensors | pass | len4 SQ top1 | case_a SQ top1 | case_a AQ4 rank in SQ top8 | case_b SQ top1 | prefill tok/s | decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `kup1-layer3-k16-up32` | 2 | 3 / 3 | 110784 | 237950 | 1 | 182949 | 28.558735 | 28.202772 |
| `kup2-k16-up32` | 4 | 2 / 3 | 110784 | 193706 | 2 | 182949 | 33.122241 | 32.547642 |

## Interpretation

The mixed scale layout fixes the layer3 k/up interaction that failed when both tensors used row-block16 or row-block32. However, expanding the same k16/up32 pattern to layers `3,7` fails `case_a`, with AQ4 top1 still present in SQ top8 at rank 2. This means the current boundary is now layer coverage and cumulative interaction, not just the single-tensor scale width for layer3.

These timings are selected-layer model-loop diagnostics. They are not final full-package SQ throughput, and wrapper elapsed includes artifact read/materialization.

## Artifacts

- `results.jsonl`
- `comparison.json`
- `sq-kup1-layer3-k16-up32/*`
- `sq-kup2-k16-up32/*`
- `benchmarks/results/2026-07-09/sq-fp8-kup1-layer3-k16-up32-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-kup2-k16-up32-policy-v0.1.json`

## 次の行動

1. Treat layer3 k16/up32 as a passing mixed-scale probe, not a promoted SQ policy.
2. Investigate layer coverage interaction before widening k/up coverage further.
3. Candidate next probes are k16/up32 only on layer7, or k16/up32 with additional fallback/alternate scale for layer7.
