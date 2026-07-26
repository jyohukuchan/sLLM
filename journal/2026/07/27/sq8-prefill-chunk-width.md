# SQ8_0 prefill chunk-width expansion

## 前回の要点

BR の prefill trace は、M=128 の SQ8_0 が N=4095 で各 layer 32 回、40 layer
合計 1,280 回の cached-prefix Flash2 を呼ぶ一方、llama.cpp は 40 回であることを
示した。BK は末尾を padding せず、実トークンだけの重複 suffix を cursor rewind
で再計算する tail を実装していた。M=128 の full-model control は 105.040 tok/s、
llama.cpp Q8_0/F32-KV は 1,008.683 tok/s であった。

## 今回の変更点

`Sq8ServingPrefillMode` に `fixed_chunk_tokens(M)` を追加し、2..4096 の power-of-two
resident width を scheduler と CLI (`m<N>-chunk<N>`) で選べるようにした。既定は
M=128 のままである。M=256/512/1024/2048 の N=4095 tail は、順に 16/8/4/2 unit、
40 layer で 640/320/160/80 の予定 attention call となる。unit test は cursor
rewind、論理 commit、M=4096/N=4095 で偽トークンを作らないことを確認した。

ただし `resident_stack_width()` は単なる attention tile ではない。resident stack
workspace、resident hidden、prompt hidden、CK activation/projection workspace を同じ M
で確保する shape contract である。さらに layer/stack/Rust CK/C++ API の measured-M
list は現在 `{1,2,4,8,16,32,128}` であり、wide M は allocation 前に明示的に reject
する二段 admission にした。Flash2 は `new_tokens` が動的で、M=128 固有の attention
scratch は見つからなかったため、BX 所有カーネルへの変更は不要と判断した。

VRAM 計算では M あたり 539,648 B 増える。M=4096 は SQ8_0 要求量 18.519 GiB、観測済み
AQ4_0 7.426 GiB と合わせても R9700 31.859 GiB に対して分析上 6.424 GiB 残る。これは
co-resident load の観測ではなく、allocator/module overhead を含まない allocation
contract である。N=4095 で有効に call を減らせる最大幅は M=2048。

R9700 の短い lock 窓で direct CK probe も実行した。zero buffer ではあるが、
M=256/512/1024/2048/4096 の全てで Q/O・K/V・gate/up・down の4 projection と
activation quantization が成功した。gateway は `active` / `NRestarts=0` に復旧し、
`llama-qwen35-udq4.service` は `inactive` / `disabled` のままである。これは shape
admission の証拠であり、実重みの数値一致・full-model速度の証拠ではない。

## 次の行動

direct shape probe は完了したため、次は BP 側の layer/stack/Rust CK/C++ API
measured-M contract をまず M=256 に広げ、fresh resident full-model smoke を実行する。
成功後に 512/1024/2048 を段階的に広げる。同一五 prompt・五 repeat の full-model
prefill、actual trace、hidden/logit 記録、greedy/生成文比較、decode 再測を実施する。
数値しきい値で止めず、軽量昇格方針に従って実際の生成の破綻を判定する。

追調査では paged KV-write の公開 API にも `m <= 128` guard が見つかった。ただし F32
writer の既存 HIP launcher と HIPRTC body は runtime `m` と dynamic grid を使用しており、
Flash2 と同様に M=128 専用の kernel tile は確認できなかった。BX が KV dtype 対応で同 API
を編集中のため共有 checkout は変更せず、同じ lower guards だけを上げた隔離 source overlay
を build した。R9700 lock は gateway が保持中なので、実機実行は奪わず待機する。
