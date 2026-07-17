# Qwen3.5 AQ4 native prefill evidence

## 前回の要点

resident と M1 baseline の同一 prompt 境界ケース、および native/M1 crossover の既存 JSON を読み取り検証した。

## 今回の変更点

- `benchmarks/results/2026-07-13/qwen35-9b-aq4-native-prefill-smoke/summary.json` を追加した。
- 7 ケースの prompt/generated token、累積 progress、clean shutdown、子プロセス状態を機械検証した。
- prompt 127/128/129/255/256 の速度比、resident stderr 内の operation audit、native implementation ID を記録した。
- M=128 は prefill physical 2096、token-equivalent 8192、native linear 48、self M1 splice 2048（97.71%）だった。
- crossover の width=2 は初回 cold overhead として分離し、width=3 から定常比較と明記した。最大 native prompt TPS は 102.58、generation TPS は 95.71 で、数千 tok/s には未達だった。

## 次の行動

実機の再ビルド・再測定後は、この summary の source commit と binary hash を更新して再検証する。
