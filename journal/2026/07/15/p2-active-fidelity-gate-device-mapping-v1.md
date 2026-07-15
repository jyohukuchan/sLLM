# P2 active fidelity gate: attempt5/6 device mapping correction

## 前回の要点

- attempt5 の証跡は `benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/fidelity-calibration-active-v0.1/attempts/active-attempt5-20260715T085414Z/` に退避済みで、入力・バイナリ・archive の SHA256SUMS は読み取り専用検証済みだった。
- preflight は通過したが、`HIP_VISIBLE_DEVICES=1` と `--device-index 0` で capture を開始した。archive の observer は card1/V620 (`gfx1030`) を記録したものの、後続監査で global index0 が CPU であることが判明し、この card観測は capture 実デバイスの直接証拠ではない。capture は `gfx1201` の runtime guard を通過せず、GPU 計算、出力、メトリクスは未実行である。サービスは復元され、BASE の未完了出力は残っていない。

## 今回の変更点

- 最初の暫定案（`HIP_VISIBLE_DEVICES=2`、論理 `--device-index 0`）は、後続の index semantics 監査で誤りと判明したため採用していない。
- 最終固定は、ROCm SMI physical card2、`HIP_VISIBLE_DEVICES=1`（HIP visible token1）、filtered HIP ordinal0、global `--device-index 1`、期待アーキテクチャ `gfx1201` である。preflight と receipt はこの対応を静的に束縛する。
- GPU、サービス、常駐モデル、attempt5/6 の raw archive は変更していない。

## 次の行動

- `bash -n`、mock preflight、テンプレート単体テスト、入力・バイナリ・attempt5 archive の SHA256SUMS、および `git diff --check` を実行する。
- GPU を使用する次の attempt は、君による明示的な実行判断とこの mapping pin のレビュー後にだけ開始する。

## 実装 semantics 監査による訂正

- `runtime/src/ullm_runtime_api_core.inc` と `runtime/src/ullm_runtime_parts/part_00.inc` は、global device index `0` を CPU、`1` 以降を HIP ordinal `index-1` と定義している。したがって attempt5 の `--device-index 0` は GPU を選んでおらず、attempt5 archive の ROCm SMI card1 観測は capture が使った device の直接証拠ではない。
- サービスを停止せずに HIP runtime API を照会する最小 read-only probe を `HIP_VISIBLE_DEVICES=1` と `HIP_VISIBLE_DEVICES=2` で実行した。visible token `1` は filtered HIP ordinal 0 で Radeon Graphics / compute capability 12.0（`gfx1201`、R9700）を返し、token `2` は Radeon Pro V620 / capability 10.3（`gfx1030`）を返した。既存 resident の token1→ROCm SMI card2、BDF `0000:47:00.0`、UUID `a8ff7551-0000-1000-80e9-ddefa2d60f55` の binding と整合する。
- attempt5/6 の raw archive は変更しない。capture gate の正しい固定は、ROCm SMI physical card2、HIP visible token1、filtered HIP ordinal0、global logical device1、expected `gfx1201` である。receipt はこの全対応を記録し、test は global0=CPU / global1=filtered HIP ordinal0 の境界を固定する。
