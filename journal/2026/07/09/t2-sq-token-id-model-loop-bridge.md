# T2 SQ token-id model-loop bridge

## 前回の要点

- T1でtoken-id model-loop pathはrequest-batch prefillとdecode ready batchを通せるようになった。
- T2では `kup6_gate5_down5` SQ FP8 policy artifactをruntimeへmaterializeする経路は確認済みだった。
- まだSQ artifactをtoken-id model-loop scheduler pathへ接続できていなかった。

## 今回の変更点

- `sq-fp8-token-ids-model-loop-smoke` を追加した。
- `Qwen3PackageModelRuntime::load_with_sq_overlay` を使い、SQ FP8 artifactをselected-layer model-loop pathへ渡した。
- stdout/JSONLへ `sq_overlay`、`sq_candidate`、`sq_artifact`、`sq_fp8_tensor_count`、`sq_passthrough_tensor_count`、`sq_row_chunk` を保存するようにした。
- `run-external-benchmark.py --parse ullm-model-loop-throughput` はSQ overlay metadataを `workload` に保持するようにした。

## 結果

保存先:

- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-real-prefill-smoke-v1.md`

AQ4/SQ selected-layer bridge comparison:

| row | final top1 | prefill tok/s | decode tok/s | end-to-end tok/s | verified |
| --- | --- | ---: | ---: | ---: | --- |
| AQ4 real-prefill bridge | `155793,23175` | 85.722441 | 84.560571 | 85.331620 | true |
| SQ FP8 real-prefill bridge | `155793,23175` | 99.526151 | 99.900180 | 99.650516 | true |

Notes:

- This is selected-layer only, not full LM throughput.
- Internal tok/s excludes SQ artifact read/materialization.
- The tiny workload proves path connectivity and top1 preservation, not final SQ speed.

## 次の行動

1. len4/case_a/case_b prompt bundleをこのSQ model-loop pathへ接続する。
2. AQ4/SQのtop1、top-k overlap、AQ4 top1 rank、logit gapを保存する。
3. full-package real batch throughputはT1aとして別途継続する。
