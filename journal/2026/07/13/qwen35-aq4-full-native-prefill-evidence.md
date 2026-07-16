# Qwen3.5 AQ4 full-native prefill evidence

## 前回の要点

既存の linear-native と M1 resident smoke で、7境界ケースの token/progress/終了状態と native/M1 の速度比を確認済みだった。

## 今回の変更点

- 実機 full-native 原データを `benchmarks/results/2026-07-13/qwen35-9b-aq4-full-native-prefill-smoke/resident-evidence.json` として保持し、`summary.json` と `summary.md` を追加した。
- M1・linear-native・full-native の prompt 127/128/129/255/256 TPS と速度比を比較し、M=128 audit（physical 256 = 64 + 192、token-equivalent 8384 = 8192 + 192、新chunk ID各8）および M=256（physical 320、prefill 128、coverage complete）を機械検証した。
- 7ケースの token/progress/clean shutdown/子プロセス状態、JSON parse、source hash、独自 assert、diff check を実施した。最大 full-native prompt TPS は116.61 tok/sで、数千 tok/s未達を記録した。初回 native p8 cold overhead は定常比較から分離した。

## 次の行動

比較元証跡と `.rocprofv3/` を変更せず、次回実機再測定時は新しい source commit/binary hash と bounded stderr audit の完全ログ保存を更新する。
