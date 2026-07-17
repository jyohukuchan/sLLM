# pre-SQ runtime TPS T3 VRAM-monitored runs

- `package-token-ids-bench` を `package-token-ids-generate-smoke` と同じ引数で使えるCLI aliasとして追加した。
- incremental self-attentionのKV cache bytesをruntime JSONへ記録するようにした。
- `tools/summarize-runtime-tps.py` を追加し、raw `package-token-ids-generate-smoke` JSONからMarkdown summaryと正規化JSONLを作れるようにした。
- `tools/run-external-benchmark.py` に `--parse ullm-token-ids-generate` を追加し、uLLM stdout JSONとrocm-smi VRAM監視を1つの `inference-benchmark-result-v0.1` JSONL行へ統合できるようにした。

検証:

- `cargo check -p ullm-engine`
- `cargo build -p ullm-engine`
- `cargo build -p ullm-engine --release`
- `python3 -m py_compile tools/run-external-benchmark.py tools/summarize-runtime-tps.py`
- `target/debug/ullm-engine package-token-ids-bench ... 2 1048576 3 len:1 1 4 1024 64 10000000 0`
- `python3 tools/run-external-benchmark.py --parse ullm-token-ids-generate ... target/release/ullm-engine package-token-ids-bench ... all 1 1 ...`

代表結果:

| target | uLLM device | rocm-smi card | prompt | generated | prefill tok/s | decode tok/s | total wall s | consumed GiB | KV bytes | verified |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| R9700/RDNA4 | `2` | `card2` | 512 | 256 | 2.912 | 0.141 | 1984.922 | 26.257 | 50331648 | true |
| V620/RDNA2 | `1` | `card1` | 512 | 256 | 2.520 | 0.139 | 2037.830 | 26.247 | 50331648 | true |

Artifacts:

- `uLLM-project/benchmarks/results/2026-07-06/engine/pre-sq-runtime-bench-vram.jsonl`
- `uLLM-project/benchmarks/results/2026-07-06/engine/pre-sq-runtime-bench-vram-summary.md`
- `uLLM-project/docs/research/pre-sq-runtime-tps-results-2026-07-06.md`

注意:

- V620の `device_index=1` はrocm-smi上では `card1` にmemory peakが出た。保存済みJSONLのmetadataは `card1` に補正した。
- KV cacheは約48MiBで、VRAM consumedの大半はresident f32 materialized weights/runtime buffers。
- T3の最小 `512/256` gridは満たしたが、T5のBF16/materialized AQ baselineは未完了。
