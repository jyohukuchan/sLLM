# リポジトリ衛生監査

## 前回の要点

- `2d961fbb` で、AGENTS.md が必須とする `journal/` を誤って無視していた設定を除去し、参照されるGPUアーキテクチャ文書とAQ4計画を追跡へ戻した。
- AQ4 P2の既知の5ファイル、`benchmarks/results/` の未追跡evidence、07/16のP3停止関連は他タスクの所有物としてこの監査から除外した。

## 今回の変更点

- 全ての追跡対象 `.gitignore` 規則、未追跡ファイル、空ディレクトリ、追跡済みの典型的な一時ファイルをCPU-onlyで監査した。
- 追跡済み文書からファイル名で直接参照され、内容も完結している未追跡journal成果物だけを回収した。
  - `journal/2026/06/30/compile-env-report.txt`
  - `journal/2026/06/30/reference-env-report.txt`
  - `journal/2026/07/13/qwen35-aq4-full-native-prefill-evidence.md`
  - `journal/2026/07/14/qwen35-aq4-p2-path-oracle.md`
- 根拠が不足する残りの未追跡journal、保護対象、既知の変更はstage・変更・削除していない。

## 次の行動

- 所有者が、最上位に限定されていない `build/` 規則と、既存の追跡済みraw logに対する `benchmarks/results/**/logs/` 規則の扱いを判断する。
- 直接参照が確認できない未追跡journalは、各作業の所有者または参照先を確認できるまで回収・削除を行わない。
