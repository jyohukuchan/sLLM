# `SQ9_0` 対 INT8 ブロックスケールの評価記録と互換性方針訂正

## 前回の要点

- SQ9_0 は V620/gfx1030 のように有用な FP8 演算経路を持たない GPU 向けの、scale-free
  signed E5M3 設計入力だった。
- 実測済みの前提として、gfx1030 では __builtin_amdgcn_sdot4 が
  v_dot4c_i32_i8 を生成する。したがって、V620 は int8 dot を持つ。
- 今回は GPU を一切起動せず、Qwen3-14B-FP8 の読み取り専用 CPU 評価と
  gfx1030 向け静的コンパイルだけを許可範囲とした。
- この journal は当時の性能・品質・静的 ISA 評価を保存する。2026-07-26 の後続方針訂正では、
  その数値や評価データを変更しない。

## 今回の変更点

- Qwen3-14B-FP8 の source-correct F8_E4M3FN + BF16 [128,128] scale reconstructionから、
  layers 0/20/39 の Q/K/V/O と MLP gate/up/down、21 tensor・990,904,320 weights を
  ストリーミング CPU 比較した。
  - SQ9_0: relative L2 0.0265181390、relative MSE 0.000703211698。
  - Q8_0-style int8 + FP16 scale/g32: relative L2 0.00562448658、relative MSE
    0.0000316348493。
  - SQ9_0 は Q8_0-style g32 の 4.7148 倍の relative L2、22.2290 倍の error SSE。
  - g128 int8 scale の 8.125 bpp ablation は SQ9_0 より良いが、g32 より悪く、初期
    int8 設計は g32 を支持した。
- 固定 K=128 の W8A8/W8A16 GEMV probe を gfx1030 向けに -O3 static compileして
  disassemble した。
  - W8A8 Q8_0-style は 128 weights に 32 v_dot4c_i32_i8、67 VALU、VGPR 41。
  - W8A8 SQ9_0 は 128 int8-to-f32、132 FP32 mixed FMA、514 bitfield/shift class、
    813 VALU、VGPR 28。
  - W8A16 Q8_0-style は 399 VALU、SQ9_0 は 648 VALU。両方 spill 0、LDS 0であり、
    int8-to-floatとscale multiplyを払ってもQ8_0-styleが静的に有利だった。
  - packed FP16 companionも確認した。双方64 v_pk_fma_f16/128 weightsだが、SQ9_0は
    432 bitfield/shift classを伴い、W8A8 int8 dotの4 MAC/instructionには届かない。
- raw results, code object, disassembly, notes, hash manifest, and reproducible tools were追加した。
- 当時はこの結果から `SQ9_0` を runtime/artifact/campaign candidate として破棄すると
  記録した。これは性能単体の判定として保存するが、互換性を重視するユーザー方針により、
  runtime/artifact の対応範囲を否定する判定ではなくなった。

## 判定の訂正（後に保留化）

- 当時は `SQ9_0` を対応する future format とし、packer、reader、validator、deterministic RNE
  quantizer、generic E5M3-to-FP16/FP32 dequant kernel、runtime load selector、served-model
  manifest の explicit format/profile を将来の実装対象とした。この journal 自体は実装を行わない。
- この implementation scope は同日後続の方針で**保留**になった。上記の全 component は現在
  未実装・非選択であり、V100 または exact RDNA1 target の全着手条件が満たされるまで作業しない。
- `SQ9_0` は推奨形式でも最適化主対象でもない。V620 M=1 の `SQ8_0` 比 +6.069% が
  package-plus-KV 採算条件 +7.29% を満たさないこと、INT8 block-scale が容量・ISA・品質で
  有利という測定/評価は、変更せずに非推奨の根拠として残す。
- `gfx1030`、`gfx1100`、`gfx1201`、`gfx942`、`gfx950` は current INT8-capable scope であり、
  `SQ9_0` を explicit にも選択しない。V100/RDNA1 は named future candidates だが、V100 の
  DP4A 実用性と exact RDNA1 GFX capability は未確認である。
- gfx1201/RDNA4 は INT8 dot と INT8/FP8 WMMA を持つ。`SQ8_1` の INT8 path は成立するが、
  source-preserving FP8 WMMA path を持つ `SQ8_0` が RDNA4 の推奨最適化フォーマットである。
  正確な ISA 表は `docs/reference/amd-low-precision-isa-and-format-selection-rocm7.2.1.md` を正とする。

## 次の行動（保留）

1. `SQ9_0` compatibility plan、packer/quantizer、reader/validator、CPU oracle、generic dequant、
   runtime selector、manifest schema はいずれも保留する。GPU 実験、artifact、campaign、release、
   active manifest は作成しない。
2. `SQ8_1` の設計入力は別作業で確定する。可搬 baseline は `v_dot4_i32_i8` とし、RDNA3/RDNA4
   は VOP3P `v_dot4_i32_iu8` と INT8 WMMA を個別に評価する。別作業中の
   `docs/plans/sq8_1-format-design-input-v0.1.md` はこの作業から変更しない。
3. `SQ9_0` を保留解除するには、V100 または exact RDNA1 target の requirement、target 固有の
   capability/hardware evidence、current formats との matched comparison、新しい reviewed plan、
   および別途のユーザー承認が必要である。

## Evidence

- Quantization: benchmarks/results/2026-07-26/sq9_0-vs-q8_0-offline/quantization-error/
- ISA: benchmarks/results/2026-07-26/sq9_0-vs-q8_0-offline/isa/
- CPU evaluator: tools/evaluate-sq9-q8-weight-error.py
- static probe: tools/sq9-q8-gfx1030-isa.hip.cpp
- static build and count tools: tools/build-sq9-q8-gfx1030-isa.sh and
  tools/analyze-sq9-q8-gfx1030-isa.py
