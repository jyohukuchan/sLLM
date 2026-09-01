# Phase 65: gfx1201 MXFP8 asymmetric operand staging

## 目的

Phase 64後も2,048-token GPU時間の61.67%を占めたMXFP8 WMMAについて、他推論engineとのno-copy比較から得た
shape-specific providerとoperand reuseの観点を、sLLM既存WMMA bodyへ独立実装する。attention state精度やresident
weight expansionへmemory trade-offを広げる前にmatrix残差を減らす。

## 固定baseline

- exact R9700 `gfx1201`、ROCm 7.14.0、Code Object V6、wave32。
- 開始時CLI `sha256:7da63d769c2a5d23056bec8e3c4e7abd4be3373b6d21e80756391e681e303a7c`。
- Qwen3.5-4B／9B MXFP8 GGUF、FP16 KV、2,048 direct input、chunk 2,048、最大4 output token。
- 開始profile DB `sha256:3d9dce9c9ff57a83cd6eabce2cc64600fc72a538d282964ec4bb36417c8734bb`。
  kernel-duration比はID31 35.13%、ID34 26.54%、attention 15.25%、linear recurrent 6.07%だった。

## 評価した4経路

1. ID31: activation／weight valueをともにLDS共有するPhase 63 baseline。
2. ID34: activation LDS共有／weight direct-load。
3. ID35: activation direct-load／weight LDS共有。
4. ID36: activation／weight valueをともにdirect-loadする。

全経路は同じFP32 accumulation、E8M0 scale、BF16 RNE output、arithmetic treeを維持した。ID35／36は完全な
128-row workgroupだけに限定し、M127／129は範囲外global fragmentを読まず既存zero-padded providerへ戻した。

## 結果と判断

- 33-case GPU oracleをPASSし、M127／128／129、2B／4B／9B wide、4B／9B down、K11264境界、
  N64／256／512／1024を含む全比較でBF16 output digestが一致した。fallback false、cleanup 0だった。
- 5反復operator中央値ではID36が全測定shapeでID31を短縮した。代表値は4B wide
  `550,374 -> 182,886 ns`（-66.77%）、9B wide `1,284,074 -> 842,982 ns`（-34.35%）、4B down
  `1,565,642 -> 385,090 ns`（-75.40%）、9B down `2,245,699 -> 687,938 ns`（-69.37%）だった。
  N64／512／1024もそれぞれ-83.28%／-81.58%／-82.14%だった。
- 全モデル4経路中央値（tok/s）は、4BがID31／34／35／36で
  `1,754.159 / 2,550.930 / 2,196.824 / 3,045.720`、9Bが
  `909.252 / 1,526.615 / 1,078.597 / 1,748.018`だった。
- model名を使わず、exact gfx1201、Mが128整列、K>=2,048、64<=N<=16,384、K32／N64整列のshape familyへ
  ID36を既定採用した。環境変数なしの最終中央値は4B `3,053.502 tok/s`、9B `1,761.989 tok/s`で、
  Phase 64既定値比+37.81%／+36.50%だった。
- resident／peakは4B `4,954,035,712 / 6,153,623,040 bytes`、9B
  `11,205,394,944 / 12,713,263,616 bytes`で4経路間不変だった。生成token列、HIP-only、fallback false、
  cleanup 0も維持した。
- 最終profile DB `sha256:c5325815ef0caef32a04eda7b68fb7d35fb5c277c9fcf8745c3ce353c0811a12`では、
  ID36 800 callが`1,596,572.669 us`（52.99%）だった。開始時のID31+ID34同じ800 call
  `2,610,391.830 us`から38.84%、全kernel時間は約28.82%短縮した。次点はattention 14.38%、
  linear recurrent 8.70%である。

## 完了条件

- exact gfx1201 operator oracle、4B／9B実モデル、host selector、Rust focused 3/3、ABI test、formatをPASSした。
- current sourceでgfx1030とgfx942 wave64のreal HIP release compile-onlyをPASSした。
- 第三者code、疑似code、tile table、dispatch値、symbolをcopy／adapt／portしていない。比較とlicense境界は
  [Phase 65 inference-engine comparison boundary](../../../../../provenance/phase65-inference-engine-comparison.md)を正本とする。

[全体計画](../../../../main-plan.md) /
[対応する履歴](../../../../../history/2026/09/1-10/phase65-gfx1201-mxfp8-asymmetric-staging.md) /
[追跡済み要約](../../../../../../ci/matrix/phase65-gfx1201-mxfp8-direct-both-v1.json) /
[比較・provenance境界](../../../../../provenance/phase65-inference-engine-comparison.md)
