# SQ8_0 R9700 decode-attention redesign

## 結論

最速の有効な full-model 変種は **GQA grouped + tile 20** で、R9700
(`gfx1201`) において **27.378731052 tok/s** だった。これは有効な既存
SQ8_0 baseline `15.294955751 tok/s` の **1.790050x**、llama.cpp
`30.4680750229 tok/s` の **89.8604%** である。direct control との比較では
**1.797918x** である。

この実装はすべて環境変数でのみ選択される実験経路で、既定の direct
path は変えていない。軽量昇格は実施していない。理由は数値 gate ではなく、
現行 generic served-model manifest が tile 値 `20` を表現できず、この実装を
service 経由で起動して実際の文章 suite を取得できないためである。

## 旧「9 分割で 1.24x」の解決

これは split が WG を増やしていなかった現象ではなかった。旧 C=1,036 /
tile=128 trace の partial dispatch は `92160 / 256 = 360 WG`、すなわち
`40 Q head × 9 tile` である。したがって partial 内の逐次 tile loop 仮説は
否定された。

一方、`1.236512x` と報告された旧 full-model run は C=513--519、tile=128
では **5 tile / 200 partial WG** だった。9 tile/360 WG の trace と同じ条件では
ない。旧 direct full-model は `53.519086 ms/token`、split は
`43.282296 ms/token` で、C≈516 に線形換算した direct attention
`15.327204 ms/token` を、同じ C=1,036 の unprofiled attention probe の
`2.82103x` で短縮する Amdahl 再計算は `43.625 ms/token`、すなわち
**1.227x** を予測する。観測 `1.236512x` と整合する。

従って主因は、短い旧 full-model 区間で attention が約 29% にとどまり、
残りの層が短縮されなかった Amdahl 効果である。さらに generic split は
source row ごとの 2 CTA barrier を保持していた。旧 trace では separate
merge dispatch は存在するが、partial を支配していたことは示さなかった。
プロファイラの range 時間を throughput には使っていない。

物理 HBM byte、L2 hit/miss、達成 occupancy、launch overhead の個別寄与は
gfx1201 で有効な PMC を得られなかったため **未確認** である。semantic KV
byte は物理帯域測定ではない。詳細な既存証跡と再計算は
[`preflight/old-split-kv-explanation.md`](preflight/old-split-kv-explanation.md)
にある。

## 実装した独立の変更

| 変種 | partial / merge WG (C=1,036) | semantic K+V load | CTA barrier | 意図 |
|---|---:|---:|---:|---|
| direct | 40 / なし | 42,434,560 B | 2 / source row | 基準 |
| generic tile 128 | 360 / 40 | 42,434,560 B | 2 / source row | KV split |
| generic tile 20 | 2,080 / 40 | 42,434,560 B | 2 / source row | より細かい KV split |
| GQA grouped tile 20 | 416 / 40 | 8,486,912 B | 2 / source row | 同一 KV head の 5 Q head が K/V を共有 |
| GQA grouped+pipelined tile 20 | 416 / 40 | 8,486,912 B | 1 / source row + 1 / partial | double-buffer で CTA barrier を削減 |

WG 数から得る queued-wave supply proxy（64 CU × 32 wave32 = 2,048）は
direct 15.625%、generic tile 128 140.625%、generic tile 20 812.5%、grouped
tile 20 162.5% である。これは **achieved occupancy ではない**。なお、旧
C≈516/tile128 の 200 WG が 78.125% となるため、依頼時の「約 78%」と
C=1,036/9 tile trace の数字を混同してはいけない。

GQA grouped は K/V semantic load を 5 倍削減する。source-level CTA barrier
総数も、generic の `40 × 1,036 × 2 = 82,880` から grouped の `16,576` に
なる。pipeline は source-row `8 × 1,036` 回と initial partial `8 × 52` 回を
合わせて **8,704** にする。

実装は HIPRTC preamble と paged-decode launcher にあり、test-only selection
には `ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE=20`、
`ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE=1`、および
`ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_GROUPED_SPLIT=1` を用いる。pipeline
variant だけはさらに `ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_PIPELINED_SPLIT=1`
を用いる。

## 有効な隔離 full-model 計測

全 run は R9700 runtime index 1 / `gfx1201`、prompt 1,024 token、M=1 decode
16 step、warmup 4、repeat 5 で行った。load と M=128 seed prefill は除外した。
速度欄は profiler ではなく unprofiled runner の full-model decode tok/s である。

| 変種 | tok/s | direct control 比 | 備考 |
|---|---:|---:|---|
| 有効な既存 SQ8_0 baseline | 15.294955751 | — | 比較基準、65.381033 ms/token |
| この窓の direct control | 15.228021012 | 1.000000x | baseline の -0.438%。固定済み b65 executable を使用 |
| generic tile 128 | 22.412990396 | 1.471826x | KV split の初段 |
| generic tile 20 | 23.872854841 | 1.567704x | tile 128 より 1.065135x |
| GQA grouped tile 20 | **27.378731052** | **1.797918x** | generic tile 20 より 1.146856x、最速 |
| GQA grouped+pipelined tile 20 | 27.253516733 | 1.789695x | grouped 比 0.995427x。full model では採用しない |
| llama.cpp vector FATTN | 30.468075023 | — | 同じ比較対象 |

最速 grouped の baseline 比は **1.790050x**、llama.cpp 比は **0.898604x**
（10.1396% 低い）。pipeline は attention-only probe では grouped 比
1.080422x だったが、full model では -0.4573% であり、この範囲では barrier
削減を既定の勝ちと主張しない。

`valid-direct.json` の `runner_git_head` は起動時に共有 worktree を CWD として
読んだメタデータで `c0e5e428…` になっているが、実行 binary の SHA-256 は
固定 build 記録の `d5c59454…` で、source commit は `b65e63c3…` である。
他の valid run は `runner_git_head=b65e63c3…` を直接記録した。これは control
を無効化する理由ではないが、再現時には detached source worktree を CWD に
固定する。

## attention-only diagnostic

以下は host API call + stream synchronize の単一 M=1 attention 呼出しであり、
full-model throughput ではない。上の因果分解を補助するためだけに保存した。

| 変種 | split mean ms | 前段比 | max \|split-direct\| | non-finite |
|---|---:|---:|---:|---:|
| generic tile 128 | 0.224426850 | — | 1.08033e-7 | 0 |
| generic tile 20 | 0.174918240 | 1.283039x | 1.11759e-7 | 0 |
| GQA grouped tile 20 | 0.081783710 | 2.138791x | 1.08033e-7 | 0 |
| GQA grouped+pipelined tile 20 | 0.075696050 | 1.080422x | 1.08033e-7 | 0 |

各 probe の `selected_split_geometry`、semantic bytes、barrier 数、GPU-vs-direct
差分は [`probe/valid-*.json`](probe/) に保存した。これらの約 1 ULP 差と固定
synthetic input 上の同一 greedy token ID 列は診断であり、昇格の合否閾値には
していない。

## 数値・品質・昇格

F32 kernel probe では grouped / pipelined とも max abs
`1.08033e-7`、non-finite 0 だった。これは split-merge の明白な数値破綻を
示さないが、実際の推論文章の品質証明ではない。

[`docs/plans/lightweight-promotion-policy-v0.1.md`](../../../../docs/plans/lightweight-promotion-policy-v0.1.md)
に従い、top-1 一致率や logit の厳密一致を gate にしていない。だがこの
redesign を選べる candidate manifest がないため、generic promotion tool を
異なる worker に対して実行したり、候補固有 apparatus を作ったりはしなかった。
従って 10 prompt の実文章比較は **未実施**、昇格は **未実施** である。正確な
blocker は [`quality/promotion-blocker.md`](quality/promotion-blocker.md) に記録した。

## F16/BF16 KV の見積もり

KV format を F16/BF16 にすれば semantic K+V load は generic の
42,434,560 B から 21,217,280 B、grouped の 8,486,912 B から 4,243,456 B に
半減する。既存 direct baseline の attention `30.773224 ms/token` が完全に
半減すると仮定するだけの帯域上限は、wall `49.994421 ms/token`、
**20.002232 tok/s**（既存 baseline 比 1.307767x）である。

これは physical HBM byte、cache hit、format conversion、KV storage 容量、
または redesigned full-model の予測ではない。これらは未測定なので、この数字を
F16/BF16 KV の実効改善としては報告しない。

## 計測窓とサービス

- 計測窓は 2 回試行した。第 1 窓は `ullm-openai.service` 復帰および外部 GPU
  利用と重なったため全 timing を `preflight/contaminated-window-1.md` に無効として
  保存した。
- 第 2 窓だけを有効とした。各 valid run 前に process check、service inactive/failed、
  R9700 process table empty を確認した。pipeline の開始時は 40/41/40 C、
  `UNTHROTTLED`、終了後も外部 GPU process は空だった。開始証跡は
  [`preflight/valid-window-2-pre-pipeline.md`](preflight/valid-window-2-pre-pipeline.md)。
- この作業が発行した stop は 2 回、最終 restore は 1 回だけである。restore 前に
  `Result=exit-code`, `NRestarts=3`, `Start request repeated too quickly` を確認したため、
  方針どおり `systemctl reset-failed` を一度だけ行い、`systemctl start` を一度だけ
  発行した。22:24:45 JST に `ullm-openai.service` は `active (running)` へ復旧した。
  `llama-qwen35-udq4.service` は開始せず `inactive` / `disabled` のままである。

## Provenance

- implementation commits: `15907d84`, `0455b119`, `b65e63c3`;
- build, source commit, and executable hashes: [`build.md`](build.md);
- invalid early artifacts are retained for audit but no timing claim uses them;
- no V620 (`gfx1030`) computation was run.
