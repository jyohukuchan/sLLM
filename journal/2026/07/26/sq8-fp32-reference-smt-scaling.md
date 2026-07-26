# SQ8 strict-F32 reference の CPU SMT scaling と構成判断

Date: 2026-07-26

## 今回の測定

- GPU、`ullm-openai.service`、active manifest、activation、campaign、`/opt/ullm` は操作していない。
  SMT 比較は `target/release/ullm-sq8-fp32-reference` の CPU-only strict-F32 実行で行い、
  `HIP_VISIBLE_DEVICES=-1`、`ROCR_VISIBLE_DEVICES=-1`、
  `ULLM_HIP_VISIBLE_DEVICES=-1`、空の `CUDA_VISIBLE_DEVICES` を各 child に固定した。
  `--no-capture` と `ionice -c3` を使い、比較中のファイル書込みは小さな receipt と log だけにした。
- 現行 corpus は停止前の 189 秒で 1,028 -> 1,138 forward、110 forward、
  **0.582011 forward/s** だった。これは actual capture を含む、要求された 3 分以上の
  ベースラインである。
- no-capture scaling は各 worker の forward 実時間を最低 180 秒以上取った。steady aggregate
  rate は「全 forward 数 / 最大 worker の forward 実時間」であり、初期 artifact/package
  verification は除く。wall rate は起動から最終 worker 完了までを含むので、選定には
  steady rate を使った。

| 構成 | logical CPU | sibling 配置 | steady forward/s | physical 8x8 比 | wall forward/s | process ごとの peak RSS 合計 KiB |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| physical 8 x 8 | 64 | CPU 0--63 の physical thread のみ | **0.574514** | **1.0000x** | 0.497364 | 849,692 |
| SMT 14 x 8 | 112 | 0--63 と 64--111、兄弟は別 process | 0.576390 | 1.0033x | 0.524493 | 1,487,036 |
| SMT 14 x 8 | 112 | 4 physical pair / process、兄弟は同一 process | 0.326707 | 0.5687x | 0.306924 | 1,486,856 |
| SMT 16 x 8 | 128 | 全 sibling を別 process | 0.351016 | 0.6110x | 0.312356 | 1,659,280 |
| SMT 7 x 16 | 112 | 8 physical pair / process、兄弟は同一 process | 0.280612 | 0.4884x | 0.245796 | 1,186,420 |

- 14 x 8 の sibling-split だけが物理 core の 8 x 8 より速かったが、増分は
  **0.33% (1.0033x)** に留まった。測定誤差、再開 replay、scheduler 変更のリスクに
  見合う改善ではない。全 SMT は 0.6110x、同一 process に sibling を入れる 14 x 8 は
  0.5687x、7 x 16 は 0.4884x であり、FP execution resource / cache の競合が勝つという
  実測になった。
- 表の peak RSS 合計は process ごとに観測した peak の和であり、同時刻の aggregate RSS ではない。
  measurement receipt の peak RSS は 8-thread process で約 103--106 MiB、16-thread process
  で約 170 MiB だった。各 run 前後の `MemAvailable` は 79 GiB 以上で、選定 resume 後も
  8 worker 合計 RSS は約 0.80 GiB、`MemAvailable` は約 81 GiB だった。測定中の連続した
  `MemAvailable` 最小値は取得していないため、そこは**未確認**である。

## resume と結果一致

- 停止前に既存 8 case の `--resume --verify-resume` を read-only で通した。frozen gate hash、
  worker binary hash、artifact/package、seed、job list、8-thread case plan を照合し、成功した。
  SMT 14 x 8 も同じ immutable corpus identity で read-only preflight を通過した。一方 7 x 16
  は thread count が corpus identity に束縛されているため、worker を起動する前に拒否された。
- launcher は explicit `--allow-smt` がなければ CPU 64--127 を拒否し、resume identity
  (gate/binary/artifact/package/seed/thread count/jobs) と scheduling
  (process 数、affinity、nice、memory preflight) を分離して記録するようにした。scheduler
  だけを変える compatible resume は immutable invocation record を追加するが、thread count は
  case `plan.json` とともに不変のままである。
- no-capture 20 forward の three 8-thread configuration は同じ summary sequence SHA-256
  `fc1e654d...`、12 forward の full-SMT 8-thread と 7 x 16 も同じ
  `af62648b...` だった。各 configuration 内でも全 worker sequence は一致した。
- published capture と physical 8 x 8 recomputation を
  `sequential_m1:raw-p0001-g1024` の先頭 20 forward で比較した。logits、final hidden、40 layer
  hidden の **42 payload hash/forward、計 840 hash** は mismatch 0 だった。
- 最終選定の physical 8 x 8 で `--resume` した後、234 秒時点で 8 worker 合計 72 forward を
  replay 検証済みと `progress.json` が記録した。worker は既存 capture の metadata と全 payload
  SHA-256 を検証してからしか継続できず、この時点の mismatch は 0、published forward directory
  数は 1,194 のままだった。corpus は継続中である。

## 判断と残り時間

- 採用構成は変更せず、**8 processes x 8 threads、CPU 0--63 physical-core-only、nice 10**。
  再開 launcher PID は `1595579` で、8 worker が再生・検証を継続している。
- launcher の frozen job list は合計 **28,853 forward** を宣言している。resume 時点の published
  forward は 1,194、未生成は 27,659 だった。依頼文中の 23,556 とは一致しないため、推測で
  補わず、launcher plan / job count から確認できる 28,853 を見積りに使った。
- capture baseline 0.582011 forward/s だけなら残り **13.20 h**。resume replay の最初の
  72 forward / 234 s (0.307692 forward/s) を外挿すると、既存 1,194 capture の残 replay を
  約 1.01 h と見積もり、合計の暫定残り時間は **約 14.21 h** である。後半 context、queued
  worker initialization、capture/filesystem 変動を含まないため、完走保証ではない。

詳細な raw receipt と summary は
[`smt-scaling-v1`](../../../../benchmarks/results/2026-07-26/sq8-fp32-reference/cpu-f32-parallel-reference-v1/smt-scaling-v1/) に保存した。
