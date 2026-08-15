# Phase 15 Weight NVFP4履歴

## 2026-08-15: 詳細計画作成

- Phase 13/14とcross-model RDNA performance bridgeの後にWeight NVFP4を実行する順序を固定した。
- Weight-only、BF16 activationを初期契約とし、value、block scale、tensor scale、packingをversioned encodingと
  derived sidecarの必須metadataにした。
- R9700 native/library capability、packed-dequant、V620 emulation/convertedを別providerとして評価し、sidecar loadだけで
  native FP4と表記しない方針を固定した。
- Qwen/Gemmaのslice/full-model accuracy、VRAM、performance、serviceまでをPhase完了条件に含めた。

## 2026-08-15: 開始baseline受領

- Phase 14→15共通RDNA性能bridgeで、Gemma request workspace/prepared semantic再利用とM=1 BF16 matvec
  streaming loadを採用した。R9700でGemma `3/17` `14.221 tok/s`、`32/32` `13.949 tok/s`、
  Qwen3.5-2B short-odd `66.490 tok/s`を現行BF16 baselineとする。
- device profileはQwen/GemmaともM=1 BF16 matvecが最大categoryで、Gemma attentionは`4.07%`に留まった。
  P15-A0/A4ではNVFP4のbandwidth削減とunpack/dequant overheadをこのbaselineへ比較し、FA3-likeやgraph rewriteを
  NVFP4 scopeへ持ち込まない。
- V620ではGemma full bounded `3/17` `11.768 tok/s`、`32/32` `11.398 tok/s`、Qwen short-odd
  `54.942 tok/s`をtarget別baselineとする。一方のGPUの改善を他方へ一般化しない。

[対応する計画](../../../../plans/active/2026/08/11-20/phase15-weight-nvfp4.md)
