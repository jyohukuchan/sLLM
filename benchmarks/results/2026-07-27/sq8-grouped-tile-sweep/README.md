# 依頼CK: SQ8_0 GQA-grouped source-tile 128 sweep

## 結論

`SQ8_0` GQA-grouped / source tile 128 の full-model M=1 decode は実測できた。
同一の R9700 exclusive window・N=1024・4 warmup + 16 timed steps x 5 で
**26.311639 tok/s** だった。tile 20 の同窓再測定は **27.561622 tok/s**、direct は
**15.356923 tok/s** である。従って tile 128 は tile 20 の **95.46%** を保ち、direct の
**1.714x** である。

ただし、tile 128 の品質は**未判定**である。numeric capture に渡した output directory を
runner が新規作成する契約に反して事前作成してしまい、最初の direct numeric capture が
`File exists (os error 17)` で失敗した。この single-window runner は fail-fast で cleanup に
入り、tile 20/128 の数値比較と isolated tile-128 gateway の 8 ケース生成は走らなかった。
このため `javascript_debug_extended` の tile-128 出力も存在しない。品質合格、品質境界、
`split_vs_direct` の tile-128 値はいずれも推測せず **未確認** とする。本結果は昇格候補の
速度証跡のみであり、`SQ8_0` を昇格・本番切替してはならない。

tile 32/64 は現行の SQ8 source-tile parser / manifest contract が受理しないため、未測定である。

## 実測

| route | decode tok/s | tile-20 比 | direct 比 |
| --- | ---: | ---: | ---: |
| direct | 15.356923 | 55.72% | 1.000x |
| GQA grouped tile 20 | 27.561622 | 100.00% | 1.795x |
| GQA grouped tile 128 | 26.311639 | 95.46% | 1.714x |

各 sample は [`bench/`](bench/) の JSON にあり、モデル load、seed prefill、warmup、
profiler range は throughput に含めていない。device は R9700 (`gfx1201`,
`HIP_VISIBLE_DEVICES=1`) のみで、F32 KV のままである。

## 数値と品質

- 過去の kernel diagnostic にある tile-20 `1.08033e-7` はこの結果の実測ではない。
  今回の full-model numeric route は direct の最初の capture 前に停止したため、tile 20/128
  とも `split_vs_direct` は未確認である。失敗の原文は
  [`numeric/direct.stderr`](numeric/direct.stderr) に保存した。
- CF の direct 8 ケースを読み取り専用で evidence copy として
  [`quality/baseline/`](quality/baseline/) に保存した。tile-128 candidate は未生成であり、
  生成文の並置比較は成立していない。
- 従って「tile 幅 20 が品質 failure の原因」という仮説は、この窓からは支持も反証もできない。
  微小な演算差が feedback trajectory を分岐させる可能性は残るが、tile-128 の実生成を得るまで
  結論に用いない。

## 本番復旧

window は 1 回。停止前・復旧後とも active manifest SHA-256 は
`a654d92fe8142fcc0904fe187c96b84c95e0dd18acac61ef25d0cfa6429a08cd` で、
`active.json` は変更していない。lock を解放してから AQ4 service を start した。
復旧後は `ActiveState=active`, `NRestarts=0` を確認し、manual gateway response probe が
model `ullm-qwen3.5-9b-aq4` で `restored` を返した。`llama-qwen35-udq4.service` は
inactive/disabled のままである。

## CO: tile-128 文章品質（2026-07-27）

この後の CO window で、CN の数値証跡を再利用して再測定せず、CF の direct 8 件を
読み取り専用 baseline として tile-128 candidate を同一 prompt / token 上限で生成した。
成果物は [`co-window/quality/comparison.md`](co-window/quality/comparison.md) と
[`comparison.json`](co-window/quality/comparison.json) にある。8/8 が HTTP 200 で、空応答、
反復、文字化け、コード要求放棄など policy の automated blocking は 0 件だった。
完全一致率 0.000 は診断値であり、policy の合否しきい値ではない。

`javascript_debug_extended` は candidate でも runnable JavaScript と expected output `2` を
生成した。説明は `NaN` が falsy なので `filter(Boolean)` で除かれる一方、`Infinity` は
truthy のまま残るため元コードは 3 となる、と正しく記す。従って tile-20 の「元の
`filter(Boolean)` は NaN を含めない」という事実誤認は再現していない。実生成文に基づく
lightweight-promotion-policy-v0.1 の判定は **pass** である。本タスクでは SQ8_0 の本番切替を
行っていない。

速度は既存の direct 15.356923、tile-20 27.561622、tile-128 26.311639 tok/s を使用した。
したがって tile-128 は direct 比 1.714x、tile-20 比 95.46% である。CN の full-model
numeric capture は tile-20 1.3091354370、tile-128 2.3758392334 maximum absolute difference
（各 471,168 F32 values、non-finite 0）で、品質合否には使っていない。
