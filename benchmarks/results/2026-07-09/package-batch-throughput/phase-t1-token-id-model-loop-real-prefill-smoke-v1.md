# T1 token-id model-loop real-prefill smoke v1

## 前回の要点

- `phase-t1-token-id-model-loop-hybrid-smoke-v1` ではtoken ID embedding入力、selected-layer model-loop scheduler、decode ready batch、final lm_head top1 guardを接続できた。
- ただしprefillはrequest-batch実行ではなく、`prefill_real_batch=false`、`batching.mode=hybrid` の中間gateだった。
- SQ候補のbatch時total throughput評価では、prefillもrequest batchとして流れる行が必要である。

## 今回の変更点

- decoder layer runnerにprefill batch input helperとprefill batch runner APIを追加した。
- `package-token-ids-model-loop-smoke` のprefill実行を、layerごと・timestepごとに実行可能requestをまとめる `stack_prefill_request_batch_step` へ変更した。
- stdout/JSONLに `prefill_batch_request_counts_csv` を追加し、parserは `batching.prefill_batch_request_counts` として保持する。
- R9700 AQ4 packageで同じ layers `3,7`、`batch=2`、prompt `2`、generated `1` のsmokeを再実行した。

## R9700 smoke

Command:

```text
target/debug/ullm-engine package-token-ids-model-loop-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d 2 1048576 3,7 len:2x2 1 2 1024 32 10000000 0
```

Result:

| field | value |
| --- | ---: |
| status | `ok` |
| input source | `embedding_token_ids` |
| layers | `3,7` |
| requests | 2 |
| prompt tokens/request | 2 |
| generated tokens/request | 1 |
| batching mode | `real` |
| prefill executor | `stack_prefill_request_batch_step` |
| prefill real batch | `true` |
| prefill request parallelism | 2 |
| prefill batch request counts | `2,2,2,2` |
| decode real batch | `true` |
| decode request parallelism | 2 |
| prefill total tok/s | 85.722441 |
| decode generated tok/s | 84.560571 |
| end-to-end tok/s | 85.331620 |
| final top1 tokens | `155793,23175` |
| VRAM consumed bytes | 1892798464 |
| verified | `true` |

Artifacts:

- `results.jsonl`
- `raw.json`
- `stdout.log`
- `stderr.log`
- `memory.jsonl`

## 次の行動

1. このrowはselected-layer T1/T2 bridgeとして扱い、full LM throughputとは区別する。
2. 次はSQ overlayまたはcandidate policyをこのtoken-id model-loop pathへ接続し、AQ4/SQ final top1、top-k overlap、logit gap、throughputを比較する。
3. full-package real batch runnerは別途T1aとして継続し、最終的なAQ4/SQ/vLLM比較にはfull-package行を使う。
