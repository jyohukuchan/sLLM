# `SQ8_0` prefill attention redesign — 2026-07-26/27 JST

## 前回の要点

BK の cursor-rewind tail 修正により 4095-token の M=1 tail 欠陥は解消済みだったが、
長い cached-prefix prefill の残差が attention かどうかは未確認だった。BH の decode
GQA grouped tile-20 は有効だった一方、decode の workgroup split を prefill にそのまま
移す根拠はなかった。

## 今回の変更点

rocprof の 512/2048/4095 prompt capture で `ullm_cached_prefix_attn_f32_flash2_kernel` が
uLLM kernel time の 59.873% / 86.319% / 93.070% を占めると確認した。prefill generic は
1 dispatch あたり 5,120 WG、40,960 wave32（machine slot に対して 2,000% queued supply）で
あり、decode 型 KV split の under-supply 仮説は棄却した。

F32 Flash2 に exact-shape `gfx1201`, Q=40/KV=8, D=128 用の serial GQA path を追加した。
CTA は `(token, KV head)` を持ち、20-token の K/V LDS staging を 5 Q head で共有する。
各 Q head の generic 256-thread reduction tree と 64-token online softmax 境界を保ったため、
128--4095 の final hidden/logits は generic control と F32 byte exact だった。BK の
`sq8_serving_runtime.rs` は変更していない。

full-model prefill は 128/512/1024/2048/4095 で
865.157/520.351/338.308/189.737/100.586 tok/s から
883.021/561.905/358.745/196.585/105.040 tok/s へ改善した。改善率は
1.020648x/1.079858x/1.060409x/1.036094x/1.044275x。曲線の 128/4095 比は
8.601x -> 8.407x にしか改善しておらず、llama.cpp Q8_0 F32-KV との差は 4095 で
9.603x 残る。従って局所改善は retain するが、長文 prefill 崩落の解決とは扱わない。

BH decode grouped tile-20 selector も現 BR worktree build で 27.411786 tok/s を記録し、
reference 27.378731 tok/s を維持した。

## 次の行動

physical HBM/TCC counter、cache behavior、実効 occupancy は未確認なので、次候補は
その測定を先に追加する。serial staging の増えた VGPR/LDS と CTA 内 5-head serial work の
trade-off を調べ、long-prefix attention が依然 90% 超を占める理由を分解する。probe 単体の
速度で採否を決めず、full-model 5-length sweep、tail、numerical、decode regression を
継続して確認する。
