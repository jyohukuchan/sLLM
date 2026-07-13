# Reasoningとthinking budgetの本番計画

## 前回の要点

- Qwen3.5 9B AQ4ではreasoningが配信設定、Gateway、worker protocol、出力分離の各境界で
  未接続だった。
- OpenWebUI 0.9.4は`delta.reasoning_content`を表示できるため、UI patchは不要だった。

## 今回の変更点

- `docs/plans/generic-reasoning-thinking-budget-production-plan-v0.1.md`へ、モデル非依存の
  reasoning dialect、厳密なbudget、OpenWebUI E2E、性能・品質benchmark、release evidence、
  rollbackまでの計画を保存した。
- 見積もりは試作2〜4人日、AQ4 beta累計8〜12人日、production候補累計15〜25人日とした。
- この作業では実装、配信設定、サービス状態を変更していない。

## 次の行動

実装指示を受けたら、現行AQ4のAPI、token列、prefill/decode性能をPhase 0の基準証跡として
保存し、versioned API/manifest/worker仕様を固定する。
