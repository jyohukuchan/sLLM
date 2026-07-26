# SQ8_0 prefill M=1 tail fix

Date: 2026-07-26

## 前回の要点

- AY の R9700 prefill 比較では SQ8_0 が 128 から 4095 まで llama.cpp Q8_0/F32-KV に負け、4095 は 71.576 tok/s 対 1,008.683 tok/s だった。
- `m128-chunk128` の planner は残りが 128 未満になると `unwrap_or(1)` に落ち、4095 では 31 個の M=128 の後に 127 個の M=1 を実行していた。
- `resident_stack_width()` は resident hidden、prompt chunk buffer、全 layer workspace、CK/Flash2 の測定済み `sequence_len` を結ぶ固定契約であり、任意の ragged 幅へ外すことはできない。

## 今回の変更点

- scheduler に logical range と execution range を分離し、固定幅 tail は cache cursor を real-token suffix の先頭へ rewind して一つの M=128 chunk として再実行し、論理的には tail 部分だけを commit する方式を実装した。パディング・偽 token・mask は使わない。
- PagedDecodeState と resident stack の cursor rewind を追加した。`written_len` を境界に stale cache を読まない CPU causal-attention test を追加し、全 rewound slot が実 token の実行で上書きされることを検証した。
- prefix position の chunk-alignment は resident workspace の必須条件ではないことを確認して撤去した。fixed resident M は維持している。
- exact multiple (128 / 512 / 1024 / 2048) は old/new oracle の final hidden と logits が byte 一致した。129 / 1000 / 4095 は M=1 decode attention と M=128 cached-prefix attention の異なる演算順により tensor は byte 一致しないが、固定入力の top-1 と最初の生成 token は一致し、non-finite はなかった。
- isolated R9700 window で AY と同じ uLLM 条件を再測定した。4095 は 158→32 advances、71.576→99.532 tok/s (1.391×) となった。2048 は 188.425→189.685 tok/s でほぼ不変であり、tail 解消後も long cached-prefix attention が主な残差である。
- 計測中に別 Gemma4 trace を 2048 cooldown で検出した。timed driver は thermal gate 後、当該 trace 終了を確認してから開始した。attention kernel source は変更していない。
- service は外部要因で既に inactive だったため stop を追加発行せず inherited-inactive window として扱い、終了時の一回の start は成功した。active manifest SHA-256 は `c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4` のまま。AQ4_0 active manifest であり、SQ8_0 candidate は昇格しなかった。

## 次の行動

1. BH の attention 作業へ、tail 解消後も 4095 が llama.cpp F32-KV の 10.134×遅いことと cached-prefix attention が次のボトルネックであることを申し送る。
2. 端数 tail の numerical drift は M=1 と M=128 の attention 実装差として記録済み。将来 kernel 経路を統一または精度評価する場合は、full-model CPU/reference corpus との比較を別タスクで行う。
3. SQ8_0 を served candidate として昇格対象にする場合は、active manifest が競合しないことを再確認し、軽量昇格方針に従う実文章 suite を用意してから promotion tool 経由で行う。
