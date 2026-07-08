# T2 SQ FP8 pair/triple telemetry v1

## 前回の要点

- `phase-t2-sq-fp8-pair-triple-boundary-probe-v1` で、layer3 `q/k` pair候補とlayer3 `q/k/v` triple候補はB=1/4/8のfull mixed pathでAQ4 final top1と一致した。
- ただしstdout/JSONL上の `sq_execution_mode` は `direct_fp8_dequant_matvec` だけで、実際にsingle/pair/tripleのどの境界を踏んだかを列として確認できなかった。

## 今回の変更点

- engine側にSQ FP8 direct projection telemetryを追加した。
- layer load/prewarm後にtelemetryをresetし、prefill/decode/final logits測定区間で成功したSQ FP8 direct kernel呼び出しだけを数える。
- stdoutへ `sq_projection_boundary` と `sq_fp8_*_matvec_count` を追加した。
- `tools/run-external-benchmark.py --parse ullm-model-loop-throughput` がtelemetry列を `workload` に保存するようにした。
- parser unit testに `sq_projection_boundary` と各countの保持を追加した。

## R9700 telemetry rows

Common package:

```text
/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d
```

Common workload:

```text
manifest-all len:2x4 generated=1 top_k=1 lm_head_chunk_rows=1024 rotary_dim=32 rope_base=10000000 position_offset=0
```

| candidate | boundary | single count | batch count | pair count | triple count | prefill tok/s | decode tok/s | end-to-end tok/s | final top1 | top1 match |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `qk-layer3-q32-k16` | `pair` | 0 | 0 | 12 | 0 | 46.261368 | 78.439366 | 24.154962 | `44370,5446,10701,25411` | `true` |
| `qkv-layer3-q32-k16-v32` | `triple` | 0 | 0 | 0 | 12 | 48.522139 | 79.902139 | 25.686824 | `44370,5446,10701,25411` | `true` |

Raw artifacts:

- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-pair-triple-telemetry-v1/results.schema.jsonl`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-pair-triple-telemetry-v1/results.jsonl`
- per-row `raw.json`, `stdout.log`, `stderr.log`, and `memory.jsonl`

## 判断

- pair候補は測定区間でSQ FP8 pair direct kernelを12回呼び、single/batch/triple countは0だった。
- triple候補は測定区間でSQ FP8 triple direct kernelを12回呼び、single/batch/pair countは0だった。
- これで前回のpair/triple境界証拠は、環境変数や候補名だけでなく、stdout/JSONLのtelemetryでも検証できるようになった。

## 次の行動

1. `q/k/v` layer3 triple候補を、prompt bundleまたは長めのprefill gridでquality確認する。
2. layer7以降の `q/k/v` 追加でstrict top1が維持できるかを見る。
3. SQ FP8 batch matvec telemetryもcomponent/prefill runnerで同じ列に流し、single/batch/pair/tripleの呼び分けを比較表で扱えるようにする。
