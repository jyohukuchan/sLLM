# Phase 65 gfx1201 MXFP8 asymmetric staging履歴

## 2026-09-01: 4経路比較とdirect-both既定採用

- Phase 64後profileと、llama.cpp／SGLang／vLLM／LMDeploy／KTransformers／TensorRT-LLMのlicense確認済み
  no-copy比較を行った。外部実装はshape-specific providerという抽象的比較点だけに使い、第三者codeはreuseしなかった。
- 同じWMMA演算順のID31（両LDS）、ID34（weight direct）、ID35（activation direct）、ID36（両direct）を
  同一runnerで比較した。ID35／36は整列M限定で、非整列Mはzero-padded既存経路へ戻した。
- 33-case GPU oracleと5反復operator比較をPASSし、全経路のBF16 digestが一致した。ID36は代表4B wideを
  -66.77%、9B wideを-34.35%、4B downを-75.40%、9B downを-69.37%短縮し、N64まで勝利した。
- 全モデル4経路比較でもID36が勝ち、shape限定defaultの2,048-token prefill中央値は4B
  `3,053.502 tok/s`、9B `1,761.989 tok/s`となった。Phase 64既定値から+37.81%／+36.50%で、
  resident／peak、生成token、HIP-only、fallback、cleanupは不変だった。
- 最終profileではmatrix 800 callを`2.610 s -> 1.597 s`へ38.84%短縮した。matrixはなお52.99%で最大だが、
  attention 14.38%、linear recurrent 8.70%が相対的に上昇した。
- host selector、Rust focused test、ABI、format、gfx1030／gfx942 real HIP release compile-onlyをPASSし、Phase 65を完了した。

[全体計画](../../../../plans/main-plan.md) /
[対応する計画](../../../../plans/archive/2026/09/1-10/phase65-gfx1201-mxfp8-asymmetric-staging.md) /
[追跡済み要約](../../../../../ci/matrix/phase65-gfx1201-mxfp8-direct-both-v1.json)
