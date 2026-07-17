# T1 token-id model-loop bridge

## 前回の要点

- SQ候補評価では、品質差とthroughputを同じscheduler pathで見る必要がある。
- 既存のselected-layer model-loop smokeはsynthetic residual入力だった。
- full-package real request-batch throughputはまだ未完了なので、中間gateが必要だった。

## 今回の変更点

- `package-token-ids-model-loop-smoke` を追加した。
- prompt token ID batchをembedding rowに変換し、selected-layer model-loop schedulerへ入力できるようにした。
- final RMSNormとlm_head top-kを接続し、requestごとのfinal top1 tokenをstdout/JSONLへ保存した。
- R9700 AQ4 packageで layers `3,7`、`batch=2`、prompt `2`、generated `1` のsmokeを実行した。
- 保存結果は `benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-token-id-model-loop-hybrid-smoke-v1.md`。

## 結果

| field | value |
| --- | ---: |
| input_source | `embedding_token_ids` |
| batching.mode | `hybrid` |
| prefill_real_batch | `false` |
| decode_real_batch | `true` |
| decode_request_parallelism | 2 |
| prefill total tok/s | 85.882896 |
| decode generated tok/s | 85.086896 |
| end-to-end tok/s | 85.615913 |
| final top1 tokens | `155793,23175` |
| verified | `true` |

## 次の行動

1. このrowはselected-layer bridgeとして扱い、full LM throughputとは区別する。
2. 次はSQ overlayまたはcandidate policyを同じtoken-id model-loop pathへ接続する。
3. AQ4/SQのfinal top1、top-k overlap、logit gap、throughputを同じschemaで比較する。
