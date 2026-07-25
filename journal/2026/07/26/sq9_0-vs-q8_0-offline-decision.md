# SQ9_0 対 int8 ブロックスケールのオフライン決着

## 前回の要点

- SQ9_0 は V620/gfx1030 のように有用な FP8 演算経路を持たない GPU 向けの、scale-free
  signed E5M3 設計入力だった。
- 実測済みの前提として、gfx1030 では __builtin_amdgcn_sdot4 が
  v_dot4c_i32_i8 を生成する。したがって、V620 は int8 dot を持つ。
- 今回は GPU を一切起動せず、Qwen3-14B-FP8 の読み取り専用 CPU 評価と
  gfx1030 向け静的コンパイルだけを許可範囲とした。

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
- SQ9_0 を runtime/artifact/campaign candidateとして破棄する判定を
  docs/plans/sq9-format-design-input-v0.1.md に追記した。

## 次の行動

1. int8 + FP16 scale/g32 の wire-incompatible exact candidateとして SQ8_1 の設計入力を
   別タスクで作成する。SQ8_0/AQ4_0 の既存 format, artifact, dispatch, campaign, releaseは
   変更しない。
2. SQ8_1 の dynamic W8A8 activation quantizationを、保持された raw activationと
   holdout promptで測る。activation error, saturation, linear output, logitsは現時点で未確認。
3. GPU 性能比較は別途明示承認された windowでのみ行う。今回の ISA は timing, occupancy,
   transaction, TPS の代替ではない。

## Evidence

- Quantization: benchmarks/results/2026-07-26/sq9_0-vs-q8_0-offline/quantization-error/
- ISA: benchmarks/results/2026-07-26/sq9_0-vs-q8_0-offline/isa/
- CPU evaluator: tools/evaluate-sq9-q8-weight-error.py
- static probe: tools/sq9-q8-gfx1030-isa.hip.cpp
- static build and count tools: tools/build-sq9-q8-gfx1030-isa.sh and
  tools/analyze-sq9-q8-gfx1030-isa.py
