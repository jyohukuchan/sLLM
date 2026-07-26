# SQ8_0 prefill chunk-width expansion

## 前回の要点

BR の prefill trace は、M=128 の SQ8_0 が N=4095 で各 layer 32 回、40 layer
合計 1,280 回の cached-prefix Flash2 を呼ぶ一方、llama.cpp は 40 回であることを
示した。BK は末尾を padding せず、実トークンだけの重複 suffix を cursor rewind
で再計算する tail を実装していた。M=128 の full-model control は 105.040 tok/s、
llama.cpp Q8_0/F32-KV は 1,008.683 tok/s であった。

## 今回の変更点

`Sq8ServingPrefillMode` に `fixed_chunk_tokens(M)` を追加し、2..4096 の
power-of-two resident width を scheduler と CLI (`m<N>-chunk<N>`) で選べるように
した。既定は M=128 のままである。worker には
`ULLM_SQ8_PREFILL_CHUNK_TOKENS=<M>` の opt-in を接続した。wide M が未登録の
lower contract に届いた場合は黙って fallback せず、model allocation 前に明示的に
reject する。

`resident_stack_width()` は attention tile ではない。resident stack workspace、
resident hidden、prompt hidden、CK activation/projection workspace の M を一つに
束ねる allocation/shape contract であるため、削除せず可変化した。M=256/512/
1024/2048 の N=4095 tail は順に 16/8/4/2 unit、40 layer で 640/320/160/80 call
となる。cursor rewind は M=2048 では `2047..4094` を実トークンとして再計算し、
logical には末尾 2047 token だけを commit する。偽 token・padding・mask は一切
使わない。M=4096 は VRAM 上は選べるが、N=4095 に real 4096-row unit は存在しない
ため有効な wide chunk ではない。

R9700 (31.859 GiB) の allocation contract は幅ごとに 539,648 B/token 増える。
観測済み AQ4_0 7.426 GiB と加算しても M=4096 は SQ8_0 18.519 GiB、残り 6.424 GiB
で分析上 fit する。ただしこれは allocator/module overhead を除く計算であり、
同居 load の実測ではない。

BP/BX 所有の lower validation を共有 checkout では変更せず、isolated overlay で
M=256..2048 の full-model を実行した。R9700 の actual N=4095 attention call は
1,280 / 640 / 320 / 160 / **80** になった。M=2048 の full-model prefill は
126.686 tok/s、M=128 の 104.965 tok/s に対し 1.207x、llama.cpp との差は 9.610x
から 7.962x まで縮んだ。一方 Flash2 は M=2048 でも selected kernel time の
90.134% であり、dispatch 数だけ減らしても llama.cpp 水準には届かない。

wide M を実際に使う prompt では hidden/logits F32 bytes が M=128 と完全一致
（`max_abs=0`、non-finite なし）した。N<M の M=1 fallback のみ差があり、最大
hidden `max_abs=1.601738`、logits `max_abs=1.100868`、greedy token は全件一致。
固定 10 ケース生成は各 candidate 9/10 text exact、残る一件は「状態」と「データ」
のような軽微な語句差で obvious-collapse finding は 0。実 token 4,000 の長文は
M=128..2048 で 83 token ID・467文字の text が完全一致した。fresh decode は
27.552769 tok/s（参照 27.378731）である。

性能/trace sweep は `run-20260727T024801+0900`、数値/decode/生成 continuation は
`run-20260727T044042+0900` に保存した。前者は timing/trace 完了後、必要な
split-decode guard 未設定により最初の numerical smoke で fail-closed した。
`88607fe0` で guard を直し、後者は `status=0`、lock 解放、service active 復旧、
`NRestarts=0` で完了した。

## 次の行動

1. BP/BX 側の lower measured-M admission を原子的に本流へ入れる場合は、
   `benchmarks/results/2026-07-27/prefill-chunk-width/lower-runtime-handoff.md`
   の対象ファイルとテストをそのまま使う。既定 M=128 は維持する。
2. prefill gap の次の本丸は wide-M Flash2 本体である。runtime M、causal
   online-softmax、F32 paged-KV、real-token rewind を保ちつつ、M=2048 でも 90% を
   占める kernel を別 symbol で設計・差分・full-model 計測する。BX への仕様は同
   handoff に追記済みで、BX 所有 source はこの作業で触っていない。
3. short prompt の M=1 fallback は no-padding contract の帰結である。wide resident
   を通常サービスの default にする前に、複数 resident shape や安全な選択 policy
   を別設計として検討する。
