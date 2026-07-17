# SQ8_0 layer batch component row

## 前回の要点

- `sq-fp8-package-self-attn-layer-batch-smoke` は、Qwen3.5 layer3の全projection artifactで `sq_fp8_batch_matvec_count=14/14` を確認済みだった。
- ただしraw stdoutだけではM10比較用のJSONL体系に載っていなかった。

## 今回の変更点

- `tools/run-external-benchmark.py` のcomponent parserでSQ8_0 projection telemetryを保存できるようにした。
- `sq-fp8-package-self-attn-layer-batch-smoke` を `--parse ullm-component-prefill` で実行し、`benchmarks/results/2026-07-09/sq8-layer-batch-component/results.jsonl` に保存した。
- JSONL rowは `batching.mode=real`、`prefill_real_batch=true`、`sq_fp8_batch_matvec_count=14`、`sq_fp8_expected_all_batch_matvec_count=14` を記録している。
- このrowはcomponent real-batch証拠であり、full-package throughput比較用rowではない。

## 次の行動

- full-package real-batch/server-style runnerで、同じdirect batch projection boundaryを使えるようにする。
- M10ではcomponent row、model-loop row、full serving rowを混同しない比較表にする。
