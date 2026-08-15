# Phase 15Q Unsloth NVFP4品質要因切り分け履歴

## 2026-08-15: 詳細計画作成

- ユーザーの明示指示により、Phase 16より前の次タスクとして、NVFP4の高いKLDが量子化algorithmと数値formatの
  どちらに主に由来するかを調べるPhase 15Qを追加した。
- `unsloth/gemma-4-12b-it-NVFP4` revision `b1f649734b34aa5575b03d186abd1b9be3d0d5c4`を候補に固定した。
  artifactは9,304,966,064 byte、SHA-256 `7c2ee23298e7c3a9247e8947597dca5a38f8b791a0322487466d2bfad8ce704b`である。
- remote header/configをbounded readし、MLP 144 tensorがU8 packed E2M1、E4M3 block scale、F32 global scaleを持ち、
  weight observerが`imatrix_mse`であることを確認した。一方、公開checkpoint全体はMLP W4A4、attention W8A8、KV FP8の
  mixed-precisionであり、sLLMのweight-only NVFP4と直接比較できない。
- primary比較を、exact Gemma 4 12B-it BF16 source上でMLP 144 tensorだけ入れ替える`B0/S0/U0/O0`とした。
  activation、attention、KV、runtimeを固定し、Unsloth mixed checkpoint直接実行はsecondary laneへ分ける。
- artifact/source lock、independent decoder、tensor/layer sensitivity、複数logit位置のKLD分布、generation/service回帰、
  algorithm/format/runtime/mixedの判定規則をplanへ固定した。この時点ではmodel payloadの取得、source実装、GPU実行、
  provider状態変更を行っていない。

[対応する計画](../../../../plans/active/2026/08/11-20/phase15q-unsloth-nvfp4-quality-attribution.md)
