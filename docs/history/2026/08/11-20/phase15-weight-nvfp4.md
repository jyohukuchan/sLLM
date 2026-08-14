# Phase 15 Weight NVFP4履歴

## 2026-08-15: 詳細計画作成

- Phase 13/14とcross-model RDNA performance bridgeの後にWeight NVFP4を実行する順序を固定した。
- Weight-only、BF16 activationを初期契約とし、value、block scale、tensor scale、packingをversioned encodingと
  derived sidecarの必須metadataにした。
- R9700 native/library capability、packed-dequant、V620 emulation/convertedを別providerとして評価し、sidecar loadだけで
  native FP4と表記しない方針を固定した。
- Qwen/Gemmaのslice/full-model accuracy、VRAM、performance、serviceまでをPhase完了条件に含めた。

[対応する計画](../../../../plans/active/2026/08/11-20/phase15-weight-nvfp4.md)
