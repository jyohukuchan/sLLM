# T1 token-id model-loop real-prefill bridge

## 前回の要点

- `phase-t1-token-id-model-loop-hybrid-smoke-v1` ではtoken ID embedding入力、selected-layer scheduler、decode ready batch、final lm_head top1 guardを接続した。
- ただしprefillは `prefill_real_batch=false` のままだった。
- SQ候補評価では、少なくともselected-layer bridge上でprefill request batchの実測行が必要だった。

## 今回の変更点

- `decode_runner.rs` にprefill batch input helperとprefill batch runner APIを追加した。
- `qwen3_loader.rs` にpackage model用のprefill batch helperを追加した。
- `package-token-ids-model-loop-smoke` は、layerごと・timestepごとにactive requestをまとめる `stack_prefill_request_batch_step` を使うようにした。
- stdout/JSONLへ `prefill_batch_request_counts_csv` を追加し、parserで `batching.prefill_batch_request_counts` を保持するようにした。

## 結果

保存先:

- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-token-id-model-loop-real-prefill-smoke-v1.md`

R9700 AQ4 selected-layer smoke:

| field | value |
| --- | ---: |
| batching.mode | `real` |
| prefill_real_batch | `true` |
| prefill_executor | `stack_prefill_request_batch_step` |
| prefill request parallelism | 2 |
| prefill batch request counts | `2,2,2,2` |
| decode_real_batch | `true` |
| decode request parallelism | 2 |
| prefill total tok/s | 85.722441 |
| decode generated tok/s | 84.560571 |
| end-to-end tok/s | 85.331620 |
| final top1 tokens | `155793,23175` |
| verified | `true` |

## 次の行動

1. SQ overlayまたはcandidate policyをこのtoken-id model-loop pathへ接続する。
2. AQ4/SQのfinal top1、top-k overlap、logit gap、throughputを同じscheduler pathで比較する。
3. full-package real batch throughputは最終比較用として別途継続する。
