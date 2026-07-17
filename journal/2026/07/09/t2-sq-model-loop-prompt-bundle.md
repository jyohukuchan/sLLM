# T2 SQ model-loop prompt bundle

## 前回の要点

- `sq-fp8-token-ids-model-loop-smoke` は `kup6_gate5_down5` SQ FP8 artifactをtoken-id model-loop request-batch prefill pathへ接続済みだった。
- 小さい layers `3,7` smokeではAQ4/SQ final top1が一致していた。
- 次は `len4`、`case_a`、`case_b` prompt bundleを6 selected layersのmodel-loop pathへ流す必要があった。

## 今回の変更点

- `PackageModelLoopSmokeRun` stdoutへ `final_topk_tokens_csv` と `final_topk_logits_csv` を追加した。
- `tools/run-external-benchmark.py` のmodel-loop parserで、top-k token/logit matrixをJSONLの `workload` に保持するようにした。
- R9700でAQ4/SQを layers `3,7,11,15,19,23`、batch `3`、top-k `8`、LM head chunk rows `4096`、prompt bundle `len4/case_a/case_b` で実行した。
- 結果は `uLLM-project/benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-prompt-bundle-v1.md` に保存した。

## 結果

| case | AQ4 top1 | SQ top1 | top1 match | AQ4 top1 rank in SQ top8 | top8 common |
| --- | ---: | ---: | --- | ---: | ---: |
| len4 | 110784 | 102446 | false | 3 | 6 / 8 |
| case_a | 237950 | 111791 | false | 2 | 4 / 8 |
| case_b | 182949 | 182949 | true | 1 | 6 / 8 |

Throughput:

| row | prefill tok/s | decode tok/s | end-to-end tok/s | VRAM consumed bytes |
| --- | ---: | ---: | ---: | ---: |
| AQ4 | 33.044260 | 32.565648 | 32.981036 | 5885833216 |
| SQ FP8 W8A16 | 32.377649 | 32.021248 | 32.330712 | 5885886464 |

## 次の行動

- `kup6_gate5_down5` はselected-layer regression subsetとして扱い、strict top1 policyには昇格しない。
- 次のT2品質探索では、model-loop top1 driftを起こすFP8 coverageを削るか、scale/layout候補を変える。
- full-package real batch throughputはT1aとして別に継続する。
