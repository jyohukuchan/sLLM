# AQ4 layer-0 QKV attempt2 と formal CPU の threshold-free 比較

## 前回の要点

- attempt2 は commit `61818b0b0b877795c2880ba4901f396e24a931e3` で保存され、HIP logical device 1 / HIP ordinal 0 / `gfx1201` の standalone single matvec を3 rows実行した。
- GPU output SHA は `24248fd1c4b4b7186f9b048a7fa4c69925904a04b265a273390089df7312545e`、formal CPU output SHA は `9683b8c5decd545c35e416da0b0f9568e6f51463ae5395fcd872dc9cbd82b473` である。

## 今回の変更点

- `benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-qkv-gpu-probe-v0.1/comparisons/attempt2-vs-cpu-formal-v1/comparison.json` に、同一 input/package identity を持つ3 rows×8192の比較を保存した。
- GPU archive `SHA256SUMS` SHA は `b353bf460d21b91998f595e43126069663f014fb2e7737b1623e824388329a00`、GPU report SHA は `4cce8b4a55c506d94801201314237cab1fc0adaaa70e27861116ad15e1f7efc1`、CPU report SHA は `2e0e623d0cec8299944ee73db7586b0790e8a83eed9d037e0416fddba1e44145` である。
- 全体の `max_abs` は `5.91278076171875e-05`、relative L2 は `8.828352279004451e-07`、cosine は `0.999999999999611`、bit mismatch element count は `23847/24576`、bit mismatch bits は `81529`、GPU/CPU/pair nonfinite は `0/0/0` だった。row別の値は comparison JSON に固定した。
- attempt2 の observer は2サンプルで `observer-failed.marker` がなく、service は `active/running` に復旧した。ただし `NRestarts=1` と `/run/ullm` mount namespace の transient failure は記録されている。親の profile diagnostic commit `6442a68f38b4ddd2200f9a516a561e4c61f576a1` と時刻が重なるが、因果関係は断定しない。
- 数値 threshold、Go/No-Go、promotion、holdout 判定は実施していない。比較は diagnostic-only とする。

## 検証

- attempt2 の `SHA256SUMS` は9/9 PASSした。
- GPU report の row SHA、output SHA、input identity、package manifest identity、device/guard identityを実ファイルから再計算して一致を確認した。
- CPU formal の `SHA256SUMS` と output/report SHAを確認した。
- comparison JSON は JSON parse と SHA256SUMS を確認した。
- GPU、サービス、holdoutの操作は行っていない。

## 次の行動

- comparison evidence と本 journal だけを通常の限定commitとして保存する。
- この比較を数値閾値やpromotionの根拠へ昇格させず、必要な判断は別の明示的な契約で行う。
