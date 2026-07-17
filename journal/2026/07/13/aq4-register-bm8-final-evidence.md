# AQ4 register BM8 final evidence

## 前回の要点

BM8 register GEMMをforced ABIとtyped registryへ昇格し、M8..128の適格形状でproduction選択するようにした。

## 今回の変更点

- experimental crossoverのwidth 8 cold値とwidth 16..128定常値を分離した。
- no-env promoted rawをfull-native/M1と比較し、7ケースのtoken・progress・state・auditを検証した。
- 2つのrocprof DBのSHA256とselected kernel rowsを記録した。
- canonical guard存在とexperimental環境変数不在を確認した。

## 次の行動

反復・長文脈・同時実行でBM8のproduction性能とdecode比率を継続測定する。
