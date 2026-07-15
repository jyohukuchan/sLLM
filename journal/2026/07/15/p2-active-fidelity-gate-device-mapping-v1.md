# P2 active fidelity gate: attempt5 device mapping correction

## 前回の要点

- attempt5 の証跡は `benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/fidelity-calibration-active-v0.1/attempts/active-attempt5-20260715T085414Z/` に退避済みで、入力・バイナリ・archive の SHA256SUMS は読み取り専用検証済みだった。
- preflight は通過したが、実行時に `HIP_VISIBLE_DEVICES=1` と `--device-index 0` の組み合わせが `gfx1030`（V620、物理 card1）を選び、期待する `gfx1201`（R9700）に到達しなかった。GPU 計算、出力、メトリクスは未実行である。サービスは復元され、BASE の未完了出力は残っていない。

## 今回の変更点

- 読み取り専用の実測根拠に基づき、capture gate の固定を `HIP_VISIBLE_DEVICES=2`（物理 card2）と論理 `--device-index 0` に変更した。期待アーキテクチャは `gfx1201` のまま固定した。
- preflight で物理可視デバイス、論理デバイス番号、アーキテクチャの対応を静的に検査し、実行時の receipt にも `physical_card2->logical_device0` を記録する。
- GPU、サービス、常駐モデル、既存の attempt5 archive は変更していない。

## 次の行動

- `bash -n`、mock preflight、テンプレート単体テスト、入力・バイナリ・attempt5 archive の SHA256SUMS、および `git diff --check` を実行する。
- GPU を使用する次の attempt は、君による明示的な実行判断とこの mapping pin のレビュー後にだけ開始する。
