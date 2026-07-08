# T2 SQ FP8 token-id model-loop real-prefill smoke v1

## 前回の要点

- T1では `package-token-ids-model-loop-smoke` がtoken ID embedding入力、selected-layer scheduler、request-batch prefill、decode ready batch、final lm_head top1 guardを通せるようになった。
- AQ4 selected-layer rowは `batching.mode=real`、`prefill_real_batch=true`、`decode_real_batch=true` で保存済みである。
- T2の次の接続点は、同じtoken-id model-loop pathへSQ FP8 policy artifactを渡し、AQ4/SQを同じschemaで比較することだった。

## 今回の変更点

- `sq-fp8-token-ids-model-loop-smoke` を追加した。
- `Qwen3PackageModelRuntime::load_with_sq_overlay` を使い、`/tmp/ullm-sq-fp8-kup6-gate5-down5-policy-v0.1-artifact` をselected-layer model-loop pathへ接続した。
- stdout/JSONLには `sq_overlay`、`sq_candidate`、`sq_artifact`、`sq_fp8_tensor_count`、`sq_passthrough_tensor_count`、`sq_row_chunk` を保存する。
- `tools/run-external-benchmark.py --parse ullm-model-loop-throughput` はSQ overlay metadataを `workload` に保持する。

## R9700 smoke

Command:

```text
target/debug/ullm-engine sq-fp8-token-ids-model-loop-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d /tmp/ullm-sq-fp8-kup6-gate5-down5-policy-v0.1-artifact 2 1048576 3,7 len:2x2 1 2 1024 32 10000000 0
```

Result:

| field | value |
| --- | ---: |
| status | `ok` |
| SQ candidate | `sq-fp8-w8a16-r9700-v0` |
| SQ artifact | `/tmp/ullm-sq-fp8-kup6-gate5-down5-policy-v0.1-artifact` |
| SQ FP8 tensors | 22 |
| SQ passthrough tensors | 753 |
| layers | `3,7` |
| requests | 2 |
| batching mode | `real` |
| prefill real batch | `true` |
| decode real batch | `true` |
| prefill batch request counts | `2,2,2,2` |
| final top1 tokens | `155793,23175` |
| final top1 logits | `5.596449,5.045245` |
| prefill total tok/s | 99.526151 |
| decode generated tok/s | 99.900180 |
| end-to-end tok/s | 99.650516 |
| VRAM consumed bytes | 1892880384 |
| verified | `true` |

## AQ4/SQ bridge comparison

| row | final top1 | final top1 logits | prefill tok/s | decode tok/s | end-to-end tok/s | VRAM consumed bytes |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| AQ4 real-prefill bridge | `155793,23175` | `5.581087,4.894406` | 85.722441 | 84.560571 | 85.331620 | 1892798464 |
| SQ FP8 real-prefill bridge | `155793,23175` | `5.596449,5.045245` | 99.526151 | 99.900180 | 99.650516 | 1892880384 |

Notes:

- This is a selected-layer bridge, not full LM throughput.
- Internal tok/s excludes SQ artifact read/materialization time. The wrapper elapsed time was `33.426s`, while model-loop timing starts after layer load.
- The token count is intentionally tiny (`prefill=4`, `decode=2`), so this row proves path connectivity and guard preservation, not final SQ format speed.

## 次の行動

1. Add a prompt-bundle version of this SQ model-loop guard, starting with the existing len4/case_a/case_b token IDs.
2. Save AQ4/SQ top1, top-k overlap, AQ4 top1 rank, and logit gap in the same JSONL/schema.
3. Keep full-package real batch throughput as the separate T1a requirement for final AQ4/SQ/vLLM comparison.
