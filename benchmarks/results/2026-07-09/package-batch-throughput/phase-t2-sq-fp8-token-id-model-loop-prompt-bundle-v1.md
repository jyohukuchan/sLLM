# T2 SQ FP8 Token-ID Model-Loop Prompt Bundle v1

## 前回の要点

- `sq-fp8-token-ids-model-loop-smoke` は `kup6_gate5_down5` SQ FP8 artifactをtoken-id model-loop request-batch prefill pathへ接続済みである。
- 直前のsmokeは layers `3,7`、batch `2`、prompt `2` の小さい接続確認で、AQ4/SQのfinal top1は一致していた。
- 次の確認は、既存の `len4`、`case_a`、`case_b` prompt bundleを同じscheduler pathへ流し、top-k overlap、AQ4 top1 rank、logit gapを保存することだった。

## 今回の変更点

- `PackageModelLoopSmokeRun` のstdoutに `final_topk_tokens_csv` と `final_topk_logits_csv` を追加した。
- `tools/run-external-benchmark.py --parse ullm-model-loop-throughput` は、これらを `workload.final_topk_tokens` と `workload.final_topk_logits` のlist-of-listとして保存する。
- R9700でAQ4/SQを同じ条件で実行した。条件は layers `3,7,11,15,19,23`、batch `3`、top-k `8`、LM head chunk rows `4096`、prompt bundle `len4/case_a/case_b` である。
- 比較結果を `comparison.json` に保存した。

## R9700 Results

| row | batching | prefill real | decode real | prefill tok/s | decode tok/s | end-to-end tok/s | VRAM consumed bytes | wrapper elapsed s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| AQ4 | real | true | true | 33.044260 | 32.565648 | 32.981036 | 5885833216 | 5.223 |
| SQ FP8 W8A16 | real | true | true | 32.377649 | 32.021248 | 32.330712 | 5885886464 | 92.056 |

Quality comparison:

| case | token IDs | AQ4 top1 | SQ top1 | top1 match | AQ4 top1 rank in SQ top8 | top8 common | SQ top1 minus AQ4 top1 logit |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| len4 | `1,2,3,4` | 110784 | 102446 | false | 3 | `6 / 8` | 0.161812 |
| case_a | `100,200,300,400,500,600,700,800` | 237950 | 111791 | false | 2 | `4 / 8` | 0.020830 |
| case_b | `42,314,2718,1618,12345,23456,34567,45678` | 182949 | 182949 | true | 1 | `6 / 8` | 0.000000 |

Artifacts:

- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-prompt-bundle-v1/results.jsonl`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-prompt-bundle-v1/comparison.json`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-prompt-bundle-v1/aq4/`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-prompt-bundle-v1/sq/`

## Interpretation

This selected-layer model-loop guard is stricter than the earlier direct logits prompt-bundle guard. The current `kup6_gate5_down5` candidate does not preserve strict top1 on `len4` or `case_a` once token-id embedding, six selected runtime layers, request-batch prefill, decode ready batch, final norm, and LM head are connected in one path.

The candidate remains useful as a regression subset because AQ4 top1 stays inside SQ top8 for all three cases, and SQ internal throughput is close to AQ4 on this selected-layer path. It should not be promoted as the SQ quality policy under the current strict-top1 rule.

SQ wrapper elapsed includes artifact read/materialization. Internal tok/s excludes that load phase, so throughput rows must keep those two timings separate.

## 次の行動

1. Treat `kup6_gate5_down5` as a selected-layer regression subset, not a promoted SQ policy.
2. Use `final_topk_tokens` / `final_topk_logits` in later AQ4/SQ comparisons instead of relying only on final top1.
3. Continue T2 by either reducing the FP8 coverage that causes model-loop top1 drift, or testing scale/layout candidates before widening coverage.
4. Continue T1 full-package real batch throughput separately before making final AQ4/SQ/vLLM performance claims.
