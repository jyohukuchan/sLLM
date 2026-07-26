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
- `c4c9a9b3` の active manifest / worker / profile binary の SHA-256 が
  capture 前後で一致することを確認し、C=1339 の新規 32-step paired
  capture を実行した。direct は 13.458181 ms/token（74.3042 tok/s）、
  同一 rocprof run の module kernel は 12.831467 ms/token だった。
- 同一 rocprof run の物理会計は kernel **88.9889%** / kernel 外
  1.587711 ms/token、同一窓の unprofiled-normalized 比は **95.3432%** /
  0.626714 ms/token だった。両者を同一の物理分解として混ぜず、
  profiler 摂動を明記した。
- current trace の GPU inter-dispatch gap は 1.487833 ms/token。
  その 88.53% は次の `hipModuleLaunchKernel` がすでに戻った後の穴であり、
  全量を host launch overhead とする解釈を棄却した。no-op probe の
  unprofiled base `hipModuleLaunchKernel` enqueue は 1.553198 µs/call
  （292 回で 0.453534 ms/token）だった。
- current projection は 10.382780 ms/token、AQ4_0 payload-only 640 GB/s
  floor は 7.132979 ms/token（439.681 GB/s / 68.700%）。`matvec_add` が
  絶対時間と payload 効率の両面で最優先だが、protected source は変更
  していない。
- service window は 01:18:30--01:19:03 JST に 1 回だけ使用した。stop 後に
  capture が flock を取得し、`llama-qwen35-udq4.service` は起動していない。
  復旧の最初の start は既存の `start-limit-hit` で失敗したため、記録付きの
  `reset-failed` と 1 回の start で `ullm-openai.service` を active
  （MainPID 4004158、NRestarts=0）へ復旧した。
- R9700 watcher は gfx max 3,242 MHz、memory 1,258 MHz、fabric 2,016 MHz、
  hotspot max 76 C を記録した。`THROTTLED` 表示だけでは clock loss と
  判断していない。

## 次の行動

1. protected runtime files の所有が解けた後、まず
   `ullm_aq4_matvec_add_f32_kernel` の具体的な load / lane mapping を
   kernel-focused probe で計測する。payload-only roofline は上限であり、
   cache・命令・非 payload traffic の原因は未確認である。
2. HIP Graph と normalization fusion は独立した capture/replay / numerical
   実験で評価する。現時点の 0.453534 ms/token と 0.463949 ms/token は
   profile/critical-path 未検証の上限尺度であり、実装根拠ではない。
3. 次の測定窓では service restore の `start-limit-hit` を自動的に
   `reset-failed` + 一回の再 start で扱う手順を使い、必ず post-window
   active 状態を記録する。
