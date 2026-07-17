# T1 component prefill real-batch parser

## 前回の要点

- `package-batch-throughput-bench` のlogical batch JSONL preservationは完了していた。
- ただしfull package real batch runnerは未完了だった。
- 既存のreal-batch component smokeはkey-value stdoutで、比較用JSONLへ直接流せなかった。

## 今回の変更点

- `tools/run-external-benchmark.py` に `--parse ullm-component-prefill` を追加した。
- `runtime-causal-attn-batch-smoke` のようなcomponent prefill real-batch outputをkey-value parseし、`inference-benchmark-result-v0.1` rowへ変換できるようにした。
- `batching.mode=real`、`prefill_real_batch=true`、request/token parallelism、`prefill_total_input_tokens_per_second`、`attention_pair_tps_mean`、sampled correctnessを保存する。
- R9700でB=2/N=32のsmokeをJSONLへ変換し、`prefill_real_batch=true` を確認した。
- 結果を `benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-component-prefill-real-batch-parser-v1.md` に保存した。

## 次の行動

1. component real-batch rowsはkernel/schema検証に限定する。
2. full package throughput判断にはまだ使わない。
3. 次はpackage prefillまたはdecode runnerをreal-batch executor pathへ接続する。
