# T1 token-id model-loop hybrid smoke v1

## 前回の要点

- `package-self-attn-mlp-block-model-loop-smoke` はselected-layer stackでschedulerとdecode ready batchを使えるが、入力はsynthetic residualだった。
- SQ候補のdriftとthroughputを同じ経路で見るには、token-id embedding入力とfinal lm_head top1 guardへ近づける必要があった。

## 今回の変更点

- `package-token-ids-model-loop-smoke` を追加した。
- prompt token ID batchをembedding rowへ変換し、model-loop schedulerの初期residualとして使う。
- decode側は固定のsynthetic future token IDをembedding rowとして使う。これはgreedy generationではなく、scheduler/stack/quality guard接続の中間gateである。
- final hiddenにfinal RMSNormとlm_head top-kをかけ、requestごとのfinal top1 tokenをstdoutとJSONLへ保存した。
- `tools/run-external-benchmark.py --parse ullm-model-loop-throughput` は `input_source` と `final_top1_tokens` を保持するようにした。

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
| batching mode | `hybrid` |
| prefill real batch | `false` |
| decode real batch | `true` |
| decode request parallelism | 2 |
| prefill total tok/s | 85.882896 |
| decode generated tok/s | 85.086896 |
| end-to-end tok/s | 85.615913 |
| final top1 tokens | `155793,23175` |
| VRAM consumed bytes | 1892876288 |
| verified | `true` |

Artifacts:

- `results.jsonl`
- `raw.json`
- `stdout.log`
- `stderr.log`
- `memory.jsonl`

## 次の行動

1. このrowはselected-layer T1/T2 bridgeとして扱い、full LM throughputとは扱わない。
2. 次はSQ overlayまたはcandidate policyをこのtoken-id model-loop pathへ接続し、AQ4/SQ final top1を比較する。
3. prefillのrequest-batch化はまだ未完了なので、`batching.mode=hybrid` のまま区別する。
