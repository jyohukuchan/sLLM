# AQ4_0 decode wall-clock accounting

## 前回の要点

- 本番 `AQ4_0` Qwen3.5-9B は active manifest SHA-256
  `a98910dc5bf59dc768e5bcd20bcf58968699540eb1b33df33066dcb6f274fe49`、
  P3 worker `c4c9a9b3` であり、直接 decode 計測は C=1339 で
  13.613452656 ms/token（73.4568 tok/s）だった。
- 既存 rocprof trace の module-kernel inclusive 合計は 412,275,120 ns
  だが、トレース対象トークン数と壁時計との同一ラン比較は未確定だった。

## 今回の変更点

- marker 範囲と層構成の両方から、既存 P3 trace は C=1339..1370 の
  **32 decode token**、9,344 module dispatch（292/token）と確定した。
- raw timestamp から historical trace の module kernel は 12.883598
  ms/token、全 GPU dispatch 間ギャップは 1.514498 ms/token と再計算した。
  次の launch API が既に戻っているギャップが 1.311078 ms/token を占める
  ため、全ギャップをホスト起動オーバーヘッドとする解釈は棄却した。
- AQ4_0 物理 payload を package から再集計し、射影は historical trace で
  10.448168 ms/token、payload-only 640 GB/s floor は 7.132979 ms/token
  （68.27%）と記録した。`matvec_add` が最初の改善候補である。
- `tools/analyze-aq4-decode-walltime-accounting.py`、roofline 解析、
  R9700 ロックを尊重する paired-capture 手順、projection handoff を
  `c0724b71` にコミットした。AQ4_0 カーネル本体は変更していない。
- R9700 は他セッションが `/run/ullm/r9700.lock` を保持している間、
  capture script が exit 75 で拒否することまで確認した。サービスは
  起動・変更していない。

## 次の行動

1. ロック解放後、同一 c4c9a9b3 profile binary の unprofiled / rocprof /
   module-launch probe を一つの排他窓で採取する。
2. current trace の kernel share、GPU gaps、D2H/sync と launch probe を
   historical 分析と置き換え、プロファイラ摂動と production wall の差を明記する。
3. 結果・温度/実クロック・lock/service 記録を保存し、最終判定と
   projection/fusion/HIP Graph の見積りを更新して別コミットにする。
