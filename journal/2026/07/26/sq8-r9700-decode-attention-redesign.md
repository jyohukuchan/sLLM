# SQ8_0 R9700 decode attention redesign

## 前回の要点

- direct `ullm_paged_decode_attn_f32_kernel` は C=1,036 で layer あたり 40 WG、
  queued-wave supply proxy 15.625% だった。5-way GQA により semantic K+V load は
  42,434,560 B、source row ごとに 2 CTA barrier が残っていた。
- 旧「tile 128 の 9 分割なのに 1.2365x」という数字は、C≈516 / 5 tile / 200 WG の
  full-model 測定と C=1,036 / 9 tile / 360 WG の trace を混同していた。Amdahl
  再計算は 1.227x を予測し、観測と整合する。物理 HBM byte と achieved occupancy
  は未確認である。

## 今回の変更点

- `15907d84`、`0455b119`、`b65e63c3` で opt-in の generic tile 20、GQA grouped、
  GQA grouped+pipelined paged-decode attention を追加した。grouped は 5 Q head が
  1 KV head の K/V を共有し、semantic K+V load を 42,434,560 B から 8,486,912 B
  に減らす。pipeline は barrier を 2/source row から 1/source row + 1/partial
  へ下げる。既定 direct は変更していない。
- 有効な R9700 isolated full-model decode では direct 15.228021012 tok/s、
  generic tile128 22.412990396、tile20 23.872854841、grouped tile20
  **27.378731052**、pipelined 27.253516733 だった。最速 grouped は有効既存
  baseline 15.294955751 の 1.790050x、llama.cpp 30.468075023 の 89.8604% である。
  pipeline は attention-only probe で 1.080422x 速かったが、full model では
  grouped 比 0.995427x で、勝ちとは扱わない。
- split-vs-direct の max abs は最大 1.11759e-7、non-finite は 0。これは
  lightweight promotion の合否には使っていない。現行 manifest は tile 値 `20`
  を表せず、service candidate を起動できないため、10 prompt の実文章比較は未実施、
  昇格も未実施である。
- 計測窓は 2 回。第 1 窓は service/external GPU の重なりを発見して無効化し、第 2 窓
  のみを採用した。最後に verified start-limit から reset-failed と 1 回の start を
  行い、22:24:45 JST に `ullm-openai.service` が active/running へ復旧した。
  `llama-qwen35-udq4.service` は起動せず inactive/disabled のままである。

## 次の行動

- generic served-model manifest に値付き environment を安全に表現できる設計が別途
  承認・実装された後、同じ generic promotion tool で実際の 10 prompt 文章を比較し、
  明白な崩壊がなければ promotion/rollback flow を通す。
- F16/BF16 KV は format 影響を分離して評価する。semantic byte 半減は見込めるが、
  physical HBM、conversion、capacity の実測なしに full-model 効果を主張しない。
- GQA grouped の 27.38 tok/s からさらに縮める場合は、attention 以外の残差と
  grouped merge の構造を profiling で分け、プロファイラ range 時間を throughput に
  使わない。
