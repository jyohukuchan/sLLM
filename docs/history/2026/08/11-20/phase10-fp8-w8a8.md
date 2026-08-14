# Phase 10 FP8 W8A8 history

## 2026-08-14: 詳細計画の作成

- Phase 9のdtype非依存prepared execution/provider境界を再利用し、Phase 10をmodel本体FP8 W8A8として具体化した。
- RDNA4 exact `gfx1201`はOCP E4M3 native FP8、RDNA2 exact `gfx1030`はnative FP8 matrix pathがないため
  W8A8 emulationと明示BF16 conversionを別providerとして扱う。RDNA2 pathをnativeと表記しない。
- 小型の公式Qwen3.5 FP8 modelに依存せず、verified 4B BF16 lockからblock 128の再現可能な開発用派生FP8
  lockを作る。公式27B FP8はPhase 12の追加interop spotとした。
- 詳細は[active plan](../../../../plans/active/2026/08/11-20/phase10-fp8-w8a8.md)を正とする。
