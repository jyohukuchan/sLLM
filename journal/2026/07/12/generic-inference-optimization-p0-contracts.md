# 汎用推論最適化 P0 契約固定

## 前回の要点

- `generic-production-inference-optimization-plan-v0.1.md`を作成し、model固有loopではなく共通graph/state/planner/registry/executorとして実装する方針を決めた。
- 実装開始前にP0としてgraph/state、backend registry、production trace、prefill validationの契約固定が必要だった。

## 今回の変更点

- ADR 0004でModelGraph、WeightBindings、StateSchema、ExecutionBatch、ModelAdapterの責務と禁止事項を固定した。
- ADR 0005で実行可能backend operation registry、workspace事前見積り、fallback、state transaction、plugin/ABI境界を固定した。
- `ullm.production_execution_trace.v1`でcomponent/full-model/production-server scope、実operator ID、batch幅、fallback、memory、commit/reset、identityを記録する契約を固定した。
- `ullm.prefill_validation.v1`でM=1/source oracle、format別policy、性能matrix、OOM、2+10 TTFT、OpenWebUI promotionを固定した。
- 計画StatusをP0完了、P1未開始へ更新した。
- lunaへ編集を委任しようとしたが、このthreadのagent作成履歴上限により親・既存agent配下の両方でspawnが拒否された。ユーザーの「出来るだけluna」という指定に従い、制約を明示したうえで既存terra agentへ詳細指示を渡して編集した。
- コード、build、service、GPU実行は変更していない。

## 次の行動

1. P1としてtyped ModelGraph、WeightBindings、StateSchema、ExecutionBatchのRust型を追加する。
2. GPU allocation前のgraph/shape/state/workspace validationをCPU-only testで固定する。
3. CPU reference executorの最小semantic subsetへ進む。
