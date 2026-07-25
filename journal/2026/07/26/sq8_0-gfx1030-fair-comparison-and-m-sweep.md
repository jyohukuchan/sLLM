# SQ8_0 gfx1030 equal optimization, fair comparison, and M sweep

## 前回の要点

`SQ8_1` W8A16/W8A8 は V620/card0 M=1 で `SQ8_0` fallback より 2.692x / 2.558x 高速と記録されて
いた。しかし `SQ8_0` 側は未最適化 generic fallback で、同一 process co-dispatch でもなかった。
M>1/prefill は熱停止ではなく、単に未測定だった。gfx1030 では CK/native-FP8 route がないため、generic
`SQ8_0` は実経路である。

## 今回の変更点

- `runtime/src/kernels/sq8_0/sq8_0_matvec_hiprtc.inc` に gfx1030-only の direct/batch generic
  specialization を追加した。公開 symbol、kernel parameter ABI、host launcher/dispatch は維持した。
  complete aligned K=16 は `uint4` 128-bit load、256 threads は eight logical wave32 reduction、
  LDS は eight F32 partials (32 B) と one barrier にした。scale boundary、tail、unaligned case は
  scalar semantic fallback を残した。
- static audit の direct gfx1030 ISA は `global_load_dwordx4` 0 -> 1、`ds_bpermute` 0 -> 5、
  `s_barrier` 2 -> 1、`ds_read` 3 -> 2、`ds_write` 2 -> 1、fixed LDS 1024 B -> 32 B だった。exact
  runtime source の direct metadata は 31 VGPR / 48 SGPR / 32 B LDS、batch は 31 / 52 / 32 B、
  private/spill は全て0。isolated `__launch_bounds__(256,2)` prototype は 30 VGPR / 48 SGPR /
  32 B のままで追加 constraint を採用しなかった。
- source hash gate と gfx1201 device-only normalized disassembly/metadata `cmp=0` を通し、
  gfx1201 legacy body/code object が不変であることを確認した。R9700/gfx1201 device execution はしていない。
- V620 の exact BDF `0000:03:00.0` -> `card0` -> own junction `temp2_input` を every-point thermal
  guard で使い、same-process M=1 rotating co-dispatch を three runs 実行した。paired ratio median は
  optimized `SQ8_0` / W8A16 = 2.633x、optimized `SQ8_0` / W8A8 = 2.522x だった。historical
  2.692x/2.558x は optimization gap を含むため format-only speedup ではない、と設計文書に訂正を追記した。
- current exact direct API の M={1,8,32,128} bundle は W8A16 が全点で W8A8 より速く、W8A8 crossover
  はなかった。runtime に触れない benchmark-only prequant + 2-D batch prototype では W8A8 が M=1
  から 1.415x、M=8/32/128 で 3.393x/2.214x/2.493x W8A16 より速かった。これは activation plane を
  eight-output-row CTA の外へ hoist する価値の evidence であり、production ABI/dispatch ではない。
- direct/batch numerical gate と prototype 2 batch x full 5120 output-row differential は全て pass。
  final thermal range は 40–54 C、85 C guard/cooldown timeout は0、M={1,8,32,128} は全完走した。

## 次の行動

1. W8A8 を production selection に使う前に、full-model weight-plus-activation quality gate を別 task
   で通す。今回の shape-level synthetic benchmark はその代替ではない。
2. prequant + 2-D batch prototype を採用候補にする場合は、separate ABI/dispatch design、full tail/
   alignment validation、target-GPU profiler evidence、actual prefill workload の数値/性能 gate を先に
   固定する。
3. profiler-derived occupancy、cache/DRAM traffic、run 3 の absolute clock-state change の原因は未確認
   のままなので、今回の static metadata や paired ratio を超える主張には使わない。
