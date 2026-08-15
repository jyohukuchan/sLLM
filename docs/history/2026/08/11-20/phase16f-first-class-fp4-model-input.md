# Phase 16F first-class FP4 model input履歴

## 2026-08-16: 詳細計画作成

- ユーザー決定に従い、提供元NVFP4 PTQ/QATとMXFP4/MXFP8 QAT/native modelをBF16/FP8と同じ操作で扱う
  official model input phaseを追加した。内部evidence分類を起動mode、許可flag、通常警告へ変換しない。
- primary full-modelを既存cache/lockとGemma adapterを再利用できる`unsloth/gemma-4-12b-it-NVFP4` revision
  `b1f649734b34aa5575b03d186abd1b9be3d0d5c4`とした。公開mixed recipeのW4A4 MLP、W8A8 attention、FP8 KV、
  BF16/ignoreを忠実に実行するため、Phase 16 KV量子化の後へ配置した。
- NVIDIA `Gemma-4-31B-IT-NVFP4` revision `4135a98a9b728a548947683219633b25682223ac`は4 shard合計
  `32,633,477,808` byteでR9700 32 GiBへworkspace込みで収まらないため、secondary schema/model-lock/reference targetとした。
- OCP MX v1.0と`moonshotai/Kimi-K3` revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`をMXFP4/MXFP8 contractへ固定した。
  Kimi full modelは未実装MoE/architectureかつ2.8T級のため、encoding/import boundaryだけを本Phaseで完成させ、Phase 18以降へ渡す。
- safetensors/compressed-tensorsと将来GGUFが同じcontainer-neutral encoding/recipe descriptorへlowerする計画、same-artifact
  reference、task oracle、AMD operator/full-model、performance/UXの受入条件を固定した。本時点ではsource実装やmodel downloadを行っていない。

[対応する計画](../../../../plans/active/2026/08/11-20/phase16f-first-class-fp4-model-input.md)
