# SQ8 strict-F32 reference の CPU プロセス並列 capture

Date: 2026-07-26

## 前回の要点

- artifact-F32 strict full-model reference は CPU で実装・decoder/解析 test と
  CPU F64 projection cross-check を通過し、同一 9 step の二回実行で logits、final
  hidden、40 layer hidden が byte-identical だった。
- GPU F32 diagnostic は pre-fixed gate を満たさなかったため、GPU は v0.2 control
  に使わない。今回も GPU、`ullm-openai.service`、active manifest、activation、campaign、
  `/opt/ullm` は操作していない。

## 今回の変更点

- Threadripper PRO 3995WX の 64 physical core / 128 logical CPU を確認し、SMT sibling
  を混ぜず logical CPU 0--63 の physical-core set だけに pin して CPU strict-F32
  runner を測った。各 worker は position 0--3 の四 forward、seed 0、capture off である。
  wall rate は artifact/package verification と model initialization を含み、steady rate は
  四 forward の critical path だけを用いた。

| threads x processes | steady aggregate forward/s | critical 4-forward s | RSS x processes KiB | 17 independent case の early-position ECT |
| --- | ---: | ---: | ---: | ---: |
| 64 x 1 | 0.118987 | 33.617 | 561,416 | 67.358 h |
| 32 x 2 | 0.234869 | 34.062 | 596,888 | 34.241 h |
| 16 x 4 | 0.383692 | 41.700 | 667,716 | 19.286 h |
| 12 x 5 | 0.563547 | 35.489 | 670,676 | 13.888 h |
| 10 x 6 | 0.489846 | 48.995 | 706,020 | 16.077 h |
| 8 x 8 | 0.599971 | 53.336 | 809,792 | **13.444 h** |
| 6 x 10 | 0.620157 | 64.500 | 969,584 | 15.777 h |
| 4 x 16 | **0.679429** | 94.197 | 1,556,784 | 21.447 h |

- aggregate forward/s の最大は 4 x 16 だったが、4,096-forward causal chains を持つ
  frozen case の終了時刻が悪化する。largest-case-first / earliest-completion-time で 17 case を
  配置した見積もりの最小は 8 x 8 だったため、full capture は固定 8 threads x 8 processes、
  disjoint physical CPU sets、nice 10、seed 0 で開始した。launch memory budget は 768 MiB x 8
  と 16 GiB reserve、launch preflight の `MemAvailable` は 60,398,608 KiB であり、reserve と
  worker budget 後も 37,329,936 KiB が残った。benchmark RSS x 8 は 809,792 KiB だった。
- frozen JSON は変更していない。coverage accounting は、seven primary stream の decode
  sample が合計 4,096 position、同じ stream の prompt を含む forward が 10,409、five boundary
  case が 6,028、sequential M=1 が 16,437、M=128 chunk/tail が 12,416、総計 28,853 forward
  である。raw F32 tensor payload だけでも 41,762,524,672 B (38.894 GiB) であり、metadata と
  filesystem overhead は含めない。
- selected ECT 13.444 h は position 0--3 の forward only であり、後半 context の増加、queued
  worker initialization、capture/hash I/O、filesystem contention を含まない。従って 8--12 h
  一晩完走は**確認できない**。12 h を既に超えるので、完走見込みとは判定しない。後半 context
  の CPU cost はこの時点では未測定である。
- 参考として M=128 全量を除くと 12,416 forward (43.03%)、long M=128 二 case を除くと
  8,192 (28.39%)、primary decode を 4,096 から 1,024 / 512 に短縮しても 3,072 (10.65%) /
  3,584 (12.42%) forward の削減に留まる。前者は mandatory M=128 coverage を失い、後者は
  frozen primary sample と一側 95% Wilson 下限を 99.934% から 99.737% / 99.474% に下げる。
  primary-only は 18,444 (63.92%) を削るが boundary/M=128 を失う。いずれも v0.2 非適格であり、
  基準は改訂していない。
- corpus worker を追加した。position ごとの `.staging` -> atomic rename、atomic
  `progress.json`、既存 payload/hash の検証、case completion 時の `SHA256SUMS` と `run.json`
  receipt を実装した。resume は既存 capture を再書込みせず、F32 KV を再構築するため検証済み
  position を replay してから継続する。worker binary / gate SHA / artifact identity / thread
  count / seed は case plan に束縛する。
- 8-thread serial sample (CPU 40--47) と 8-process 中の parallel sample (CPU 56--63) で
  raw-p0001 の position 0--3 を直接比較した。metadata 四 file と logits/final/40 layer hidden
  の 168 payload、計 172 file は `cmp` で mismatch 0、teacher-forced token sequence も
  `1 -> 25 -> 330 -> 16 -> 13` で一致した。receipt は
  `benchmarks/results/2026-07-26/sq8-fp32-reference/cpu-f32-parallel-reference-v1/parallel-vs-serial-t8-v1.json`
  に固定した。これは並列化が同一 8-thread serial output を変えていない直接証拠である。

## 保存状態と次の行動

- static throughput evidence は
  `benchmarks/results/2026-07-26/sq8-fp32-reference/cpu-parallel-throughput-v1/summary.json`、
  active capture root の immutable launch plan/preflight は
  `benchmarks/results/2026-07-26/sq8-fp32-reference/cpu-f32-parallel-reference-v1/` にある。
- capture は case/position 単位で中断・再開可能である。中断時は同一 binary と同一 launcher
  plan を `--resume` で再実行する。全 17 case が `run.json: status=complete` になるまで、
  full v0.2 control 完成とは扱わない。
- 予定時間内に完走しなければ、完走済み case / position と per-case SHA-256 manifest を明記して
  停止できる。未完の coverage を pass と解釈せず、frozen JSON を変更しない。
