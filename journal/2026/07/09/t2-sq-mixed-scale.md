# T2 SQ mixed scale

## 前回の要点

- k/up row-block32はlayer3だけでも `case_a` が崩れた。
- `up_proj` layer3 row-block32と `k_proj` layer3 row-block16は単体ではstrict top1を維持した。
- `k/up` layer3 row-block16は組み合わせると崩れたため、tensorごとにscale block幅を変える必要があった。

## 今回の変更点

- `uLLM-project/tools/build-sq-fp8-w8a16-artifact.py` にpolicy `scale.overrides[]` を追加した。
- mixed layoutではcandidate-level `scale_granularity=mixed`、`scale_layout=per_tensor` を保存し、各 `fp8_tensors[]` entryのscale metadataをauthoritativeにした。
- `uLLM-project/docs/specs/sq-fp8-artifact-v0.1.md`、`uLLM-project/docs/words.txt`、policy unit testを更新した。
- R9700で `k_proj` row-block16 + `up_proj` row-block32をlayer3とlayers `3,7` の2条件でmodel-loop prompt bundleにかけた。
- 結果を `uLLM-project/benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-mixed-scale-v1.md` に保存した。

## 結果

| variant | pass | case_a SQ top1 | case_a AQ4 rank in SQ top8 |
| --- | ---: | ---: | ---: |
| `kup1-layer3-k16-up32` | 3 / 3 | 237950 | 1 |
| `kup2-k16-up32` | 2 / 3 | 193706 | 2 |

## 次の行動

- `kup1-layer3-k16-up32` はpassing mixed-scale probeとして保持する。
- `kup2-k16-up32` はfailure guardとして残す。
- 次はlayer7単体 `k16/up32`、またはlayer7の片側fallback/別scaleを試して、coverage interactionを分ける。
