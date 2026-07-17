# SQ plan update after mixed row-block guard

## 前回の要点

- SQ候補1は `sq-fp8-w8a16-r9700-v0` としてR9700/RDNA4限定で進めていた。
- FlashAttention2-style cached-prefix側は、SQ候補評価へ進むための前提作業として一旦十分な速度に達した。
- T2ではFP8 artifact writer、runtime materialize smoke、package overlay guard、row-block scaleまで進んだ。
- row-block scaleは `q` と `down` を回復したが、`v` はtested block sizeではstrict top1を維持できなかった。
- `v` fallback + `q/k/o/gate/up/down` row-block32 FP8が、layers `3,7` の短い3 promptで通る部分候補になっていた。

## 今回の変更点

- 混合候補を layers `3,7,11,15`、layers `3,7,11,15,19`、layers `3,7,11,15,19,23`、all self-attention probe layersへ広げた。
- layers `3,7,11,15` は短い3 promptで `3 / 3` strict top1一致だった。
- layers `3,7,11,15,19` はlen4でstrict top1一致だった。
- layers `3,7,11,15,19,23` はlen4でstrict top1不一致だった。
- all self-attention probe layers `3,7,11,15,19,23,27,31` もstrict top1不一致だった。
- layer `23` 単体では `q` row-block32がriskで、`q/v` fallbackならstrict top1に戻った。
- ただし6層bundleでは、layer `23` の `q` だけをfallbackしても、全6層の `q/v` をfallbackしてもstrict top1は戻らなかった。
- 結果を `benchmarks/results/2026-07-08/sq-fp8-mixed-candidate-layer-scaling-guard-v0.1.md` に整理した。
- `docs/plans/fp8-sq-r9700-batch-throughput-prefill-plan-v0.1.md`、`sq-r9700-state-freeze-v0.1.md/json`、`docs/specs/sq-fp8-artifact-v0.1.md` に現在地を反映した。

## 次の行動

1. T2 short guardの合格基準を決める。strict top1を維持するのか、top-k overlapやtext-level guardを許容するのかを先に固定する。
2. strict top1を維持する場合は、6層bundleの累積driftを追加fallback、per-layer policy、またはより強いscale/layoutで潰す。
3. strict top1を緩める場合は、短文生成品質、top-k overlap、logit driftを同じschemaで保存し、速度評価へ進む条件を明文化する。
4. overlay load timingはSQ速度ではないため、T5 throughput比較ではT1 real batch runnerとnative/materialization-aware runtime pathを使う。
