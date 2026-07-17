# AQ4 tiled prefill candidate rejection evidence（2026-07-13）

## 前回の要点

BM8/BN32/BK128 の gfx1201 AQ4 tiled GEMM 候補を、当時の選択既定候補として resident smoke で取得した。比較元は full-native AQ4 prefill resident evidence である。

## 今回の変更

- raw `resident-evidence.json` は変更せず、候補 `summary.json` / `summary.md` を追加した。
- 7ケースの prompt/generated token 列、progress、`length` outcome、reset、clean shutdown、子プロセス状態を baseline と独自 assert で exact 比較した。
- p127/128/129/255/256 の candidate/baseline prompt TPS 比率を raw 値から計算し、候補を rejected / no promotion と記録した。
- 現在の gating/default は `4ab1181`、既定 dispatch は Legacy、tiled は `ULLM_EXPERIMENTAL_HIP_AQ4_TILED_GEMM=1` の opt-in と明記した。p128=111.17 tok/s の manual check は非 raw 注記として除外した。
- JSON parse、raw SHA256、独自 assert、diff check を実施し、比較元と `.rocprofv3/` を変更していない。

## 次の行動

既定 Legacy を維持する。tiled 候補は gated opt-in のまま、性能改善後に別の resident evidence として再評価する。
