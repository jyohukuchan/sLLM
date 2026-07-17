# T2 SQ coverage reduction

## 前回の要点

- `kup6_gate5_down5` はdirect logits prompt bundleではstrict top1を維持した。
- 同じ候補はsix-layer token-id model-loop prompt bundleでは `len4` と `case_a` が崩れた。
- 現在のSQ品質昇格条件は、direct logitsではなくtoken-id model-loop上のstrict top1である。

## 今回の変更点

- k/up row-block32のcoverageを6層から1層まで削って、R9700の同じprompt bundleで評価した。
- layer3の `k_proj` 単体、`up_proj` 単体、`k_proj` row-block16、`k/up` row-block16を追加評価した。
- 結果を `uLLM-project/benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-coverage-reduction-v1.md` に保存した。
- `uLLM-project/docs/plans/fp8-sq-r9700-batch-throughput-prefill-plan-v0.1.md` と `uLLM-project/benchmarks/results/2026-07-08/sq-r9700-state-freeze-v0.1.{md,json}` に反映した。

## 結果

| variant | pass | case_a SQ top1 | case_a AQ4 rank in SQ top8 |
| --- | ---: | ---: | ---: |
| `kup6-rowblock32` | 2 / 3 | 193706 | 4 |
| `kup5-rowblock32` | 2 / 3 | 193706 | 5 |
| `kup4-rowblock32` | 2 / 3 | 193706 | 5 |
| `kup3-rowblock32` | 2 / 3 | 193706 | 4 |
| `kup2-rowblock32` | 2 / 3 | 193706 | 3 |
| `kup1-layer3-rowblock32` | 2 / 3 | 124170 | 4 |
| `k-layer3-rowblock32` | 2 / 3 | 111791 | 2 |
| `up-layer3-rowblock32` | 3 / 3 | 237950 | 1 |
| `k-layer3-rowblock16` | 3 / 3 | 237950 | 1 |
| `kup1-layer3-rowblock16` | 2 / 3 | 193706 | 3 |

## 次の行動

- `kup6_gate5_down5` はdirect logits regression subsetとして残し、SQ quality policyへは昇格しない。
- 次はper-family/per-tensor scale-layout supportをartifact builderとmanifestに入れる。
- その後、`k_proj` row-block16 + `up_proj` row-block32のような混合scale policyをmodel-loop prompt-bundle gateで再評価する。
