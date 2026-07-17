# Pre-SQ Runtime TPS Stop Policy

## 前回の要点

- accepted packageはR9700/V620の両方で `512/256` を完走し、decodeは約 `0.14 tok/s` に張り付いた。
- materialized-AQ baseline packageはR9700 `512/256` を完走し、decodeは `0.140 tok/s` で同じ律速を示した。

## 今回の変更点

- V620 materialized-AQ baseline `512/256` は、追加情報が少ないため途中で意図的に停止した。
- 計画書へMeasurement Stop Policyを追加し、低速で安定したdecodeを長時間繰り返さない方針に変えた。
- 結果メモへR9700 baseline完走値と、V620の途中停止理由およびメモリ到達値を記録した。
- 既存package/runtimeでは真のBF16 baselineを作れないことを確認し、pre-sq範囲ではdeferする方針にした。
- R9700/V620で短いgolden prefix reference guardを実行し、accepted packageが12層fixture比較でverifiedになることを確認した。
- T6 decision packとsq format設計入力メモを作成し、pre-sq段階を閉じる判断材料をまとめた。

## 次の行動

- sq format v0.1の設計に進む。
- T5はmaterialized-AQ lower-boundとBF16 defer判断で閉じた。
- 以後の測定は、長いprefill圧力と短いdecode probeを分ける。
