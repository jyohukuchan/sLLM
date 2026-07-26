# SQ8 numerical gate v0.2 preliminary tile evaluation

## 前回の要点

v0.2 harness は frozen JSON の SHA-256 を実行時に検証し、full admission には
4,096 primary decode position / 7 stream、M=128 と boundary coverage、control 3 /
candidate 2 repetition を要求する。CPU artifact-FP32 reference はまだ生成中だが、
8 case 全てを含む生成済み 2,160 source position は利用可能だった。

## 今回の変更点

- 進行中 reference root を書き換えず read-only snapshot を固定した。snapshot は
  `benchmarks/results/2026-07-26/sq8-gate-v0.2-preliminary/reference-snapshot-2160.json`、
  SHA-256 は `d0ac40dfc5d911f7356b7d93d7469f35439d5b823fa92c3b4231aad9d7baa540`、position manifest
  SHA-256 は `967a85d76cc64594de0ae6a071bf082babfb56b8ebf9ef4fe6b0925844ae318c` である。
- teacher-forced capture token を reasoning forced-end token と区別し、isolated capture が
  serving の reasoning accounting を壊さないようにした。さらに capture identity から mode 固有の
  runtime 設定を分離し、device identity の誤検出を防いだ。
- preliminary evaluator の exposure 集計を decode-only から全 `sequential_m1` position に修正した。
  `PagedDecodeState` は prompt 中にも実行されるためである。旧 receipt は保存し、再計算 receipt を
  authoritative とした。`selector_exposure` 以外の数値結果は同一である。

## 実行

R9700 GPU 2（`0000:47:00.0`、`gfx1201`）のみを使った。V620 は選択していない。private child
process で shared control 1、tile128 1、tile256 1 を取り、production default、active manifest、
activation、campaign、remote state は変更していない。

service window は 3 回だった。最初の二回は candidate 評価前に deterministic harness boundary error
（teacher-forced accounting、mode-specific identity comparison）を検出して restore し、修正後の三回目で
control と両 tile を同一 window にまとめた。最終 window は 16:41:32--17:16:36 JST、復旧後は
`active/running`、`Result=success`、`NRestarts=0` だった。

## 結果

| candidate | multi-tile exposure | result | key failures |
| --- | ---: | --- | --- |
| tile128 | 645 M=1 (prompt 547, decode 98) | `preliminary / fail_metric_subset` (90) | logits max-abs `2.543147087097168 > 1.990684199333191` at `raw-p0001-g1024:decode:00188`; final hidden P99 relative-L2 `0.20372229973103717 > 0.17362422008268527`; top-1 Wilson `0.9678961737926518 < 0.9731245214414421` |
| tile256 | 64 M=1 prompt | `preliminary / fail_metric_subset` (10) | layer-05 P99 relative-L2 `0.04898892478442091 > 0.04881303307700018` at `raw-p4095-g1:prompt:00287`; top-1 Wilson `0.9696649304986341 < 0.9731245214414421`; hard top-1 regression at raw-p4095 prompts 262/294/299/302/304/318 |

tile128 の hard top-1 regression は raw-p1023 prompts 140/162/183 と raw-p4095 prompts
140/162/183/262/265/294/299/304/312/317 にある。詳細の metric/scope/position は
`attempt-3/evaluations-recomputed/tile128.json` と `tile256.json` に保存した。

両候補とも数値不合格なので、end-to-end decode speed は測らなかった。したがって過去の `1.2365×` を
支持する新しい timing はない。Flash2 は優先順位と時間制限により未実施である。

## Coverage の位置づけ

source snapshot は 2,160 position だが、この capture route が materialize したのは 1,290 である。
M=128 内部の 870 position は個別 capture 不可であった。zero-error 仮定でも 2,160 position の
one-sided 95% Wilson 下限は `99.8749001%`、formal 4,096 primary position は `99.9339903%` で、
`0.0590902` percentage point の差がある。加えて今回の one-control / one-candidate capture は formal
repeat envelope を持たない。

ゆえにこれは admission pass ではなく、`preliminary` であり `not_qualified` である。数値不合格という
結論はこの測定集合について有効だが、coverage 不足を通過証拠や正式基準の置換として扱わない。

## 解釈と次の行動

llama.cpp の `(max, sum, weighted-V)` partial を global max で再重み付けする split-KV は、同じ
algorithm class が実装可能であることを示す。しかし uLLM の候補が同じ FP32 reference に対する matched
direct control より悪化している以上、これを基準側の問題とする根拠にはならない。per-tile online-softmax
merge と direct の演算 association の違いを SQ8 activation quantization が増幅した仮説とは整合する。
ただし最初に境界を越えた quantizer または lane-level step は **未確認** である。

multi-tile を再提案するには、direct と同じ数値 contract を満たす merge/accumulation を実装し、reference
完走後に frozen v0.2 full admission を control 3 / candidate 2 で再実行する。それまでは direct default を
維持する。
