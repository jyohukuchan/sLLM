# T1 mixed request-state resident throughput small grid v1

## 前回の要点

- `phase-t1-mixed-request-state-layer-batch-executor-v1` でfull mixed `manifest-all` のrequest-batch executor境界を追加した。
- ただし前回rowは `throughput_row=false` で、`total_wall_ms` がlayer load込みだった。

## 今回の変更点

- `package-token-ids-mixed-request-state-smoke` の `total_wall_ms` を resident inference区間に変更した。
- `total_wall_ms = prefill_wall_ms + decode_wall_ms + final_logits_wall_ms` とし、`outer_wall_ms` にCLI内の全体時間を残した。
- stdout/JSONLに `throughput_row=true`、`load_excluded_from_total=true`、`final_logits_in_total=true` を保存した。
- `run-external-benchmark.py --parse ullm-model-loop-throughput` は単一値CSVを1要素リストとして保存し、mixed pathの `max_new_tokens_csv` を generated token listとして読めるようにした。

## R9700 small grid

Common command shape:

```text
target/debug/ullm-engine package-token-ids-mixed-request-state-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d 2 1048576 manifest-all len:2xB 1 1 1024 32 10000000 0
```

Result:

| batch | mode | prefill real | decode real | throughput row | prefill tok/s | decode tok/s | end-to-end tok/s | total wall ms | outer wall ms | VRAM consumed bytes | final top1 tokens |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `single` | `false` | `false` | `true` | 21.577161 | 79.979289 | 8.926325 | 336.084556 | 10744.741758 | 4210278400 | `44370` |
| 4 | `real` | `true` | `true` | `true` | 49.037429 | 81.291382 | 24.096096 | 498.005974 | 12308.551404 | 5011484672 | `44370,5446,10701,25411` |
| 8 | `real` | `true` | `true` | `true` | 65.290399 | 81.616739 | 34.577530 | 694.092379 | 16088.342071 | 5401587712 | `44370,5446,10701,25411,21901,685,279,27973` |

All rows:

- `status=ok`
- `verified_all=true`
- `request_batch_executor=true`
- `fused_request_batch=false`
- `load_excluded_from_total=true`
- `final_logits_in_total=true`
- `prefill_mode=token_id_full_mixed_request_state`
- `layers_csv=0..31`

Artifacts:

- `results.jsonl`
- `batch1/raw.json`, `batch1/stdout.log`, `batch1/stderr.log`, `batch1/memory.jsonl`
- `batch4/raw.json`, `batch4/stdout.log`, `batch4/stderr.log`, `batch4/memory.jsonl`
- `batch8/raw.json`, `batch8/stdout.log`, `batch8/stderr.log`, `batch8/memory.jsonl`

## 次の行動

1. このrowをAQ4 full mixed request-batch resident baselineとして扱う。
2. `fused_request_batch=false` なので、GPU fused batchの速度改善証拠としては扱わない。
3. 次はSQ FP8 candidateを同じ full mixed resident pathに接続し、AQ4/SQのqualityとthroughputを同じschemaで比較する。
