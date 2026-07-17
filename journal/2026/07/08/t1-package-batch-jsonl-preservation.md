# T1 package batch JSONL preservation

## 前回の要点

- T0はR9700 SQ評価のstate freezeとして実質完了していた。
- T1はlogical batch raw reportとschema確認までは進んでいたが、JSONL変換で必要fieldが落ちないことをテストで固定できていなかった。
- T1 real batch runnerはまだ未完了だった。
- T2はSQ FP8 mixed row-block candidateの品質境界を狭めたが、full-target guardはまだ未完了だった。

## 今回の変更点

- `tools/run-external-benchmark.py` のpackage-batch変換で、raw `batching.prefill_executor` と `batching.resolved_prefill_executor` をJSONL `workload.*` fieldにfallback保存するようにした。
- package-batch用のmemory enrichment helperを追加し、`memory.kv_cache_bytes_total` の保持を明示した。
- `tests/test_external_benchmark_batch_parser.py` を追加し、T1で必要なtotal-throughput、prefix/chunk/context accounting、executor accounting、KV cache bytesの保存を固定した。
- 合成 `package-batch-throughput-bench-v0.1` reportを `run-external-benchmark.py --parse ullm-package-batch-throughput` のmain pathに通し、JSONL rowに必要fieldが残ることを確認した。
- `benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-jsonl-preservation-v1.md` に結果を保存した。
- `docs/plans/fp8-sq-r9700-batch-throughput-prefill-plan-v0.1.md` と `sq-r9700-state-freeze-v0.1.md/json` を更新し、T1 JSONL/schema preservationをv0.1 doneにした。

## 次の行動

1. T1の残りはreal batch runnerであり、logical batch結果をSQ性能判断に使わない。
2. T2はacceptance ruleを決めるか、6層bundleの累積driftを追加fallback/per-layer policy/stronger formatで潰す。
3. T5 throughput比較へ進むのは、T1 real batchとT2 quality guardの条件が揃ってからにする。
