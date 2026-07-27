# 依頼CK SQ8_0 GQA grouped tile 128 sweep

## 前回の要点

- CF は SQ8_0 grouped tile-20 を 512 token の 8 ケースで再判定し、
  `javascript_debug_extended` の事実誤認を根拠に quality hold とした。
- AQ4_0 は active manifest
  `a654d92fe8142fcc0904fe187c96b84c95e0dd18acac61ef25d0cfa6429a08cd` のままである。

## 今回の変更点

- R9700 (`gfx1201`) の service stop window を 1 回だけ使用し、N=1024 の full-model M=1
  decode を direct / GQA grouped tile-20 / GQA grouped tile-128 で 4 warmup + 16 timed steps x 5
  実測した。結果は 15.356923 / 27.561622 / 26.311639 tok/s。tile-128 は tile-20 の 95.46%、
  direct の 1.714x だった。
- numeric decode-oracle の出力 directory を runner より先に作成してしまい、runner の
  fail-closed create-new 契約で停止した。従って数値差と tile-128 の 8 prompt quality generation
  は未実施であり、品質合否も境界も判定していない。
- cleanup は lock 解放後に AQ4 service を start した。active manifest SHA は不変、
  `ActiveState=active`, `NRestarts=0`、実際の `restored` completion を確認した。

## 次の行動

- tile-128 の quality / numeric は、この失敗の output-directory 作成を修正した新規窓でのみ
  再試験する。今回の速度だけで SQ8_0 を昇格しない。
