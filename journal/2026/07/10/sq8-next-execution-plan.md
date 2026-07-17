# SQ8次期実行計画の策定

日付: 2026-07-10

## 前回の要点

直近約10時間の監査により、現Qwen3-14B-FP8 sidecarはsource `weight_scale_inv`を適用しておらず、現batch kernelはbatch間でweightを再利用しないことが分かった。接続基盤は再利用できるが、同一モデル比較と性能判断はやり直す必要がある。

## 今回の変更点

- `uLLM-project/docs/plans/sq8-recovery-plan-v0.2.md`を次のactive planとした。
- 旧`docs/plans/sq8-implementation-plan-v0.1.md`は実装履歴として残し、実行順序がv0.2へ移ったことを明記した。
- 次の10時間の到達点を、40層やserving parityではなく、source-correctなone-tensor artifact、one-linear oracle、R9700 capability判定、代表projection一つのbatch scaling証明に限定した。
- canonical artifactとkernel固有prepackを分離し、artifact変更とkernel layout変更を切り離した。
- 各phaseへentry gate、acceptance、stop conditionを設定した。

## 次の行動

P0で既存の無効なuLLM比較行を隔離し、対象tensor、入力、hash、commandを固定する。その後、P1としてF8 payloadと128x128 block scaleをbyte-exactに保持するartifact schemaとround-trip golden testを実装する。source reconstructionが一致するまでGPU性能作業へ進まない。

## 計画書

- `uLLM-project/docs/plans/sq8-recovery-plan-v0.2.md`
- `uLLM-project/docs/plans/sq8-implementation-plan-v0.1.md`は履歴参照のみ
