# SQ8計画を合格条件基準へ変更

日付: 2026-07-10

## 前回の要点

`uLLM-project/docs/plans/sq8-recovery-plan-v0.2.md`は、正しいartifact、reference kernel、R9700 component proof、1層、40層、実生成の順序を定めていた。一方、初期10時間の時間割と各phaseのtimeboxが残っていた。

## 今回の変更点

- ユーザーの方針に合わせ、固定の完了期限を撤廃した。
- 技術的な依存順序と合格条件は維持した。
- 初期10時間の時間表を、期間を定めないInitial Validation Checkpointへ変更した。
- 実生成、prefill、40層、typed result、同条件baselineまでを最低限の機能完成として定義した。
- vLLMとの完全な性能同等、全fusion、全shapeのpeak tuning、HTTP、V620 native FP8は最低限の機能完成から外した。
- source-correctな機能完成とR9700 optimized v0完成を別の状態にした。reference経路は基本機能の検証へ進めるが、flat scalingの経路をoptimizedとして40層へ昇格させない。
- M=8/M=2の`2.5x`は推奨目標とし、必須条件はmeasurement noiseを超える非flatなscaling、referenceより高速、native FP8命令、fallbackなしとした。
- profilerで支配的と確認された箇所だけを追加最適化し、推定end-to-end効果が5%未満、同一ボトルネックで2回連続3%未満、または別subsystemが支配的になった場合の終了条件を追加した。

## 次の行動

工程はP0から順に進める。時間経過によるphaseの省略は行わず、各acceptanceを満たしてから次へ進む。最初の実装対象は従来どおり、既存結果の隔離とsource-correct canonical artifactである。
