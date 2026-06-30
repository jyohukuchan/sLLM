# ATOM Qwen3 FP8 Cause Analysis - 2026-06-30

## Scope

- Device: R9700 / gfx1201, selected with `ROCR_VISIBLE_DEVICES=1`
- ROCm runtime: `7.2.1`
- ATOM source: `cce1a6e56dcd8cb300183f81901fdaed6090d951`
- AITER source: `71829a74bc2600bfbce4c05f85ecbe0eeb994323`
- Official reference: <https://github.com/ROCm/ATOM/blob/main/recipes/Qwen3-8B-FP8.md>

The official recipe reports Qwen3-8B-FP8 on RX 9070 XT / gfx1201 at about 52.9 output tok/s for ISL/OSL 549/256 with CUDAGraph. The local question was why the earlier ATOM Qwen3-14B-FP8 row looked much slower.

## Conclusion

The large discrepancy is mostly explained by comparing different metrics and different model/engine conditions.

- The official recipe's `Output tok/s` matches a TPOT-derived metric, `1000 / mean_tpot_ms`.
- `tools/run-external-benchmark.py` currently records `metrics.decode_tokens_per_second` from ATOM benchmark `output_throughput`, which includes TTFT/request duration for a one-request run. It is useful as an end-to-end throughput number, but it is not the same as the official table's decode-only TPOT metric.
- The official-like local Qwen3-8B-FP8 CUDAGraph run reproduces the official result: `mean_tpot_ms=17.97`, so TPOT-derived output speed is `55.65 tok/s`.
- A follow-up run using the official server/benchmark settings with only the model changed to Qwen3-14B-FP8 measured `mean_tpot_ms=55.46`, so TPOT-derived output speed is `18.03 tok/s`.
- The earlier slow Qwen3-14B-FP8 representative row also used `--enforce-eager`. Removing that roughly doubled the pp512/tg128 row by the wrapper metric: `9.15 -> 18.27 tok/s`.
- Qwen3-14B-FP8 is substantially heavier than Qwen3-8B-FP8. From the configs, the rough MLP work ratio is about `1.97x` and the attention/projection work ratio is about `1.74x`. The measured TPOT gap in the official-like BF16 KV runs is `55.65 / 17.82 = 3.12x`, so model shape and ATOM kernel path efficiency likely explain the rest.
- FP8 KV did not improve this single-request 14B workload. With the correct `--block-size 128`, TPOT-derived speed fell to `9.80 tok/s`.

## Metric Table

`Wrapper tok/s` is ATOM `output_throughput`, recorded in the benchmark JSONL as `metrics.decode_tokens_per_second`. `TPOT tok/s` is `1000 / mean_tpot_ms` from ATOM `benchmark_serving` result JSON and is the comparable metric for the official recipe's output tok/s.

| Model | Condition | Workload | KV | Wrapper tok/s | TPOT tok/s | TPOT ms | TTFT ms | E2E ms | Consumed GiB | Note |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-8B-FP8 | official-like CUDAGraph | pp549/tg256 | BF16 | 35.84 | 55.65 | 17.97 | 2560.30 | 7142.77 | 27.26 | Reproduces official 52.9-class result by TPOT |
| Qwen3-14B-FP8 | official settings, model only changed | pp549/tg256 | BF16 | 17.98 | 18.03 | 55.46 | 94.74 | 14236.69 | n/a | Same settings as official 8B run, with only model path changed |
| Qwen3-8B-FP8 | official-like eager | pp549/tg256 | BF16 | 13.90 | 13.96 | 71.65 | 149.04 | 18420.57 | 27.11 | Local eager path is slower than official eager table |
| Qwen3-14B-FP8 | official-like CUDAGraph | pp549/tg256 | BF16 | 16.62 | 17.82 | 56.10 | 1098.10 | 15404.10 | 26.42 | Direct 14B vs 8B comparison |
| Qwen3-14B-FP8 | earlier representative eager | pp512/tg128 | BF16 | 9.15 | 10.97 | 91.19 | 2406.66 | 13987.37 | 24.17 | Previous slow row |
| Qwen3-14B-FP8 | same row without eager | pp512/tg128 | BF16 | 18.27 | 18.40 | 54.34 | 103.70 | 7004.37 | 24.30 | CUDAGraph/eager isolation |
| Qwen3-14B-FP8 | CUDAGraph FP8 KV | pp549/tg256 | FP8 | 7.70 | 9.80 | 102.01 | 7219.59 | 33232.70 | 26.85 | `--block-size 128`; slower here |

There is also a Qwen3-14B-FP8 FP8 KV run with `--block-size 64`, but ATOM logged that unified attention expects `--block-size 128` for FP8 KV, so it should be treated as a warning case rather than a clean datapoint.

## Model Config Comparison

| Model | Layers | Hidden | Intermediate | Heads | KV heads | FP8 block |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-8B-FP8 | 36 | 4096 | 12288 | 32 | 8 | 128x128 |
| Qwen3-14B-FP8 | 40 | 5120 | 17408 | 40 | 8 | 128x128 |

The local 8B result rules out a basic gfx1201/R9700 incompatibility as the explanation for the official-vs-local gap. The model-only 14B run confirms that the official-settings 14B TPOT speed is about `18 tok/s`. The remaining 14B slowness should be investigated as a 14B shape/kernel efficiency issue, not as proof that ATOM cannot reach the official 8B number on this class of GPU.
