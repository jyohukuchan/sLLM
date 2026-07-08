# T1 mixed request-state layer-batch executor v1

## 前回の要点

- `package-token-ids-mixed-request-state-smoke` は `manifest-all` で全32層のmixed layer orderを通し、final lm_head top1 guardまで到達していた。
- 直前の実装ではrequest slotごとのweight bundle共有まで進み、R9700 AQ4 `manifest-all` は `verified=true`、final top1は `44370,5446` だった。
- ただし実行順は `batching_mode=request_state_interleaved` のままで、`prefill_real_batch=false` / `decode_real_batch=false` だった。

## 今回の変更点

- full mixed request-state pathに `mixed_request_state_layer_batch_step` を追加した。
- prefill/decodeの各stepで、active requestを層単位にまとめて実行するようにした。
- stdoutとJSONL parserは `request_batch_executor=true`、`fused_request_batch=false`、`throughput_row=false` を保存する。
- `prefill_mode=token_id_full_mixed_request_state` を保存し、selected-layer model-loopとは区別できるようにした。

## R9700 smoke

Command:

```text
target/debug/ullm-engine package-token-ids-mixed-request-state-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d 2 1048576 manifest-all len:2x2 1 1 1024 32 10000000 0
```

Result:

| field | value |
| --- | ---: |
| status | `ok` |
| layers | `0..31` |
| input source | `embedding_token_ids` |
| prefill mode | `token_id_full_mixed_request_state` |
| batching mode | `real` |
| request batch executor | `true` |
| fused request batch | `false` |
| throughput row | `false` |
| prefill real batch | `true` |
| decode real batch | `true` |
| prefill executor parallelism | 2 |
| decode executor parallelism | 2 |
| prefill batch request counts | `2,2` |
| decode batch request counts | `2` |
| prefill total input tok/s | 36.164606 |
| decode generated tok/s | 81.064816 |
| end-to-end tok/s | 0.535737 |
| layer load ms | 10569.358382 |
| prefill wall ms | 110.605381 |
| decode wall ms | 24.671616 |
| final logits wall ms | 236.334417 |
| total wall ms | 11199.524049 |
| final top1 tokens | `44370,5446` |
| VRAM consumed bytes | 4373938176 |
| verified | `true` |

Artifacts:

- `results.jsonl`
- `raw.json`
- `stdout.log`
- `stderr.log`
- `memory.jsonl`

## 次の行動

1. このrowはfull mixed request-batch executor境界の証拠として扱う。
2. `fused_request_batch=false` なので、fused GPU batchの速度改善証拠としては扱わない。
3. `throughput_row=false` かつ `total_wall_ms` はlayer loadを含むため、SQ/AQ4の正式速度比較には使わない。
4. 次はresident load後の測定区間を分け、batch `1/4/8` のfull mixed AQ4 throughput rowを作る。
