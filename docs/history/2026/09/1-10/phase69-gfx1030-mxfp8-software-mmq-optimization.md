# Phase 69: gfx1030 MXFP8 software-MMQ次段最適化

2026-09-02にcanonical Radeon Pro V620、exact `gfx1030`、ROCm 7.14.0、Code Object V6、wave32で完了した。
対象はOCP MXFP8 E4M3 W8A8 block32/E8M0、FP32 accumulation、BF16 RNE outputである。

## 実装

- MMQ本体からpacked valueのingress policyを分離し、既存scalar policyとMXFP8専用32-bit value load policyを同じ
  col8 schedule／reductionへ接続した。4個のE4M3 codeを1回のloadで取得して従来の内部MX value-plane decodeへ渡し、
  scale byte、FP32 term／accumulator順、BF16 RNE stageは変えていない。
- K32 scaleをcompile-time block順にregisterへhoistするID40、32-bit value ingressのID41、両者を組み合わせたID42を
  独立candidateとして実装した。ID40／42はVGPR増加とshape別の非単調性によりbenchmark-onlyとし、ID41を採用した。
- ID41はPhase 67の既存production scopeだけでexact `gfx1030`の既定値とした。`gfx1031`以降、gfx1201、scope外shape、
  MXFP6／NVFP4は変更していない。旧ID27へは
  `SLLM_MXFP8_PREFILL_FORCE_MMQ_GFX1030_PHASE69=control`、row8へは
  `SLLM_MXFP8_PREFILL_FORCE_ROW8=1`で戻せる。

## resourceとprofile

最終Code Objectは`sha256:91fc43718a59377663bbcaa48a2dfb9d20e3e883e1eee455f87e234a622905d6`である。
ID27／ID41はともにLDS 8,704 byte、VGPR 46、spill 0、wave32で、ID41だけSGPRが29から31へ増えた。
ID40はVGPR 53、ID42はVGPR 54だった。

rocprofv3の28 case×4 dispatchでは、ID41はID27比で平均VALUInstsを`6,298.79 -> 5,074.33`へ19.44%減らした。
MemUnitBusyは`83.30 -> 80.57`、occupancyは`78.69% -> 77.26%`、ALUStalledByLDSは`2.27 -> 3.68`、
LDSBankConflictは`0 -> 9.46`だった。value ingressの命令削減が主効果であり、追加のregister-scale化や二重bufferを
採る根拠は得られなかった。

## correctnessとoperator性能

最終runnerは2 warmup＋10 measured、28 caseでID27 control／ID41／ID27 controlを測った。全caseが独立FP32 oracleを
PASSし、control／candidateのBF16 digestと10 repeat digestは一致した。特殊値、HIP-only、fallback false、cleanup 0も
確認した。最終paired runでは28 case中25、production scope 16 case中14が両controlより高速だった。
`M128/K2560/N4096`と`M512/K2560/N8192`は別runと向きが一致せず、単体kernel時間は周波数状態に対して非単調だった。
このためoperator全shapeの一律改善とは主張せず、採否は固定primary full-model 2行のno-regressionで決めた。

## Qwen3.5-4B full-model

固定MXFP8 GGUF、FP16 KV、direct pretokenized input、最大4 output、greedy、ignore EOSを、同一最終binaryで
3 warmup＋10 measuredした。

| input | ID27 control median | ID41 median | throughput | prefill time | E2E |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 205.0009 tok/s | 254.4461 tok/s | +24.12% | -19.43% | -17.84% |
| 2,048 | 204.2416 tok/s | 249.3441 tok/s | +22.08% | -18.09% | -17.65% |

prefill中央値は512で`2.497551 -> 2.012214 s`、2,048で`10.027341 -> 8.213554 s`だった。生成token列は全sampleで
`[23066,23066,23066,23066]`、dispatch件数もcontrol/candidateで一致した。residentは`4,954,035,712` byte、peakは
512で`5,292,664,320` byte、2,048で`6,153,623,040` byteのまま、HIP-only、fallback false、cleanup 0だった。
最終CLI SHA-256は`e1ae87f2bb7745f86af666625f80d3dc20d7fe87f51d5e6e7d8408a5c352db3a`である。

## 結論

gfx1030 MXFP8の残差にはpacked E4M3 valueをscalar loadするcostが残っていた。32-bit ingressは永続展開weight、FP32
attention／KV、cross-request cacheを追加せず、primary full-model prefillを22〜24%改善したため既存Phase 67 scopeへ
N0候補として採用した。scale register化、combined候補、二重buffer、演算順を変えるblock後scaleは採用しない。

MMQのschedule／reductionとvalue ingress policyの境界は共通化した。後続MXFP6はpacked E3M2をresidentのまま保ち、
専用tile ingressでE4M3へexact変換してこの骨格を再利用できる。NVFP4はblock 16／tensor scaleを持つ独立policyが必要である。

[全体計画](../../../../plans/main-plan.md) /
[対応する計画](../../../../plans/archive/2026/09/1-10/phase69-gfx1030-mxfp8-software-mmq-optimization.md) /
[追跡要約](../../../../../ci/matrix/phase69-gfx1030-mxfp8-software-mmq-v1.json)
