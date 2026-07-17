# T1 package prefill component runner

## 前回の要点

- `run-external-benchmark.py --parse ullm-component-prefill` はsynthetic runtime componentのreal-batch stdoutをJSONLへ変換できた。
- ただし、`.ullm.d` package pathから実際のpackage-backed componentを走らせるrunnerは未整備だった。
- T1 full package total throughputはまだ未完了だった。

## 今回の変更点

- `tools/run-package-prefill-component-workload.py` を追加した。
- `ullm-package-prefill-component-workload-v0.1` manifestからpackage-backed component smokeを実行し、`run-external-benchmark.py --parse ullm-component-prefill` へ流すようにした。
- parserを拡張し、`package-prefill-aq4-matvec-batch-smoke` の `token_tps_mean` と `real_batch=true` をT1 JSONL schemaへ正規化できるようにした。
- R9700でAQ4 packageの `model.language_model.layers.3.self_attn.k_proj.weight` を `package-prefill-aq4-matvec-batch-smoke` に通し、`batching.mode=real`、`prefill_real_batch=true`、`prefill_total_input_tokens_per_second=19063.596157` を確認した。
- 結果は `benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-package-prefill-component-runner-v1.*` に保存した。

## 次の行動

1. このrunnerはpackage-backed component rows用として保持する。
2. full package throughput判断にはまだ使わない。
3. 次はrequest batch `batch=1/4/8`、decode、end-to-end total throughputへ広げる。
