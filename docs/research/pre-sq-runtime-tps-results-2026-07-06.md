# Pre-SQ Runtime TPS Results 2026-07-06

## 前回の要点

- `docs/plans/pre-sq-runtime-tps-plan-v0.1.md` は、sq format策定前に実推論に近いtoken/sを取ることを目的にしている。
- 数tokenのsmokeだけではなく、最低 `prompt_tokens=512`、`generated_tokens=256` をR9700/RDNA4とV620/RDNA2で測る必要がある。
- tensor parallel、batch処理、server APIは今回の範囲外にしている。

## 今回の変更点

- `ullm-engine package-token-ids-bench` を追加し、既存のhybrid incremental generate経路を正式なpre-sq計測入口として使えるようにした。
- incremental self-attentionのKV cache bytesをJSONへ記録するようにした。
- `tools/summarize-runtime-tps.py` を追加し、raw smoke JSONからMarkdown summaryと `inference-benchmark-result-v0.1` 風JSONLを生成できるようにした。
- `tools/run-external-benchmark.py` に `--parse ullm-token-ids-generate` を追加し、uLLM stdout JSON、rocm-smi VRAM監視、correctness summaryを同じJSONL行にまとめられるようにした。
- R9700とV620で `prompt_tokens=512`, `generated_tokens=256` のVRAM監視付きrunを完走した。

## 次の行動

1. BF16/materialized AQ baselineを同じschemaで最低1本作る。
2. `2048/256` または `2048/512` のstretch runをR9700優先で試す。
3. sq format案では、F32常駐を避ける保存形式とdecode時のmaterialize範囲を最優先で検討する。

## Artifacts

- Raw runtime smoke summary: `benchmarks/results/2026-07-06/engine/pre-sq-runtime-bench-summary.md`
- Raw runtime smoke JSONL: `benchmarks/results/2026-07-06/engine/pre-sq-runtime-bench-summary.jsonl`
- VRAM-monitored benchmark JSONL: `benchmarks/results/2026-07-06/engine/pre-sq-runtime-bench-vram.jsonl`
- VRAM-monitored benchmark summary: `benchmarks/results/2026-07-06/engine/pre-sq-runtime-bench-vram-summary.md`

Local raw logs are under `benchmarks/results/2026-07-06/engine/logs/`, but that directory is intentionally ignored by git. The tracked JSONL above contains the comparable metrics, memory summary, correctness summary, and artifact paths.

## Primary Result

Package:

```text
/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d
```

Conditions:

- model: Qwen3.5-9B package
- quantization policy: `qwen35_9b_p4p46_hidden3994_v1`
- layers: all decoder layers
- decode mode: `hybrid_incremental_greedy`
- prompt tokens: `512`
- generated tokens: `256`
- batch size: `1`
- tensor parallel: `1`
- sampling: greedy
- KV cache dtype in current runtime: f32

| target | uLLM device | rocm-smi card | prefill tok/s | decode tok/s | total tok/s | total wall s | TTFT ms | TPOT ms | decode p50 ms | decode p95 ms | consumed GiB | peak total GiB | KV bytes | verified |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| R9700/RDNA4 | `2` | `card2` | 2.912 | 0.141 | 0.387 | 1984.922 | 175802.661 | 7084.688 | 7037.410 | 7625.910 | 26.257 | 26.419 | 50331648 | true |
| V620/RDNA2 | `1` | `card1` | 2.520 | 0.139 | 0.377 | 2037.830 | 203177.131 | 7182.688 | 7151.547 | 7709.009 | 26.247 | 26.409 | 50331648 | true |

V620 note: the engine `device_index=1` run mapped to rocm-smi `card1` in the memory log. The initial `card0` hint in the command was corrected in the saved JSONL metadata.

## Interpretation

- The required `512/256` grid is now proven on both R9700 and V620 with the same JSONL schema.
- Decode throughput is effectively the same on R9700 and V620, around `0.14 tok/s`. This means current runtime overhead dominates GPU generation differences.
- R9700 is faster on prefill, about `2.912 tok/s` vs `2.520 tok/s`, but the end-to-end run is still decode dominated.
- KV cache is only about `48 MiB`. The measured VRAM pressure, about `26.25 GiB` consumed, is dominated by resident f32 materialized weights and runtime buffers rather than KV.
- These numbers are not a product-speed target. They are a pre-sq lower-bound measurement from the current proof path.

## SQ Design Implications

- The first sq format should prioritize avoiding whole-layer f32 residency.
- Weight storage and decode-time materialization granularity matter more than KV compression for this specific `512/256` single-request case.
- The current decode path should not be used to judge final RDNA2 vs RDNA4 hardware potential because lm_head/top-k and per-step host/runtime orchestration are still heavy.
- A useful sq prototype should make the following visible in the benchmark record:
  - compact resident bytes
  - materialized working-set bytes
  - prefill TPS
  - decode TPS
  - materialization time
  - steady-state decode time

## Remaining Plan Items

- T3 is now substantially satisfied for the minimum `512/256` grid, including VRAM.
- T4 still needs a stricter short reference check against HF/PyTorch or existing golden fixture.
- T5 is not complete: BF16/materialized AQ baseline comparison still needs to be added.
- T6 decision pack is not final until T5 exists and at least one stretch context run is attempted or explicitly deferred.
