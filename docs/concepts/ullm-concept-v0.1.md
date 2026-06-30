# uLLM concept v0.1

## Overview

uLLM は、低ビット LLM 推論のための「モデル形式 + 量子化ライブラリ + 実行ランタイム + ベンチマーク基盤」として設計する。

中核目標は二つに分ける。

1. `aq` は精度重視の量子化系列。
   - 重みは index 値として保存する。
   - LUT、複数 scale、family 別 codebook、layer 別量子化方式を使える。
   - 実行時は FP8、INT8、BF16、FP16 などバックエンドが得意な型へ展開して演算する。
   - 同一 bpp 帯で Unsloth Dynamic 2.0 GGUF 系を超える精度と、同等以上の実用速度を目標にする。

2. `sq` はサーバー向け速度重視の量子化系列。
   - ハードウェアが持つ低精度行列演算器へ直接乗る形式を優先する。
   - NVIDIA Blackwell 以降では NVFP4、MXFP8、FP8 を主要対象にする。
   - AMD CDNA3/4 以降では FP8/FP4 系、CDNA1/2 や RDNA 系では能力に応じて段階的に対応する。
   - dequant はシフト、スケール、単純な整数/浮動小数演算までに制限し、LUT は使わない。
   - INT8 は原則対象外にする。ただし A100 や CPU など、INT8 が実効性能上有利な例外は backend capability として扱う。

## 主要コンポーネント

### 1. Model IR

Qwen3 系以降を第一ターゲットにしつつ、モデル実装は固定クラスではなく IR で扱う。

- dense transformer
- MoE
- GQA/MQA
- RoPE 系位置埋め込み
- sliding/window attention
- MTP/drafter head
- diffusion-style executor
- multimodal adapter

Qwen3 互換だけを直接書くと、Gemma 4、DiffusionGemma、DeepSeek 系の差分で破綻しやすい。最初から layer graph と executor を分ける。

### 2. Quantization Library

量子化ライブラリは `aq` と `sq` で最適化目標を分ける。

`aq` 側:

- layer-wise mixed quantization
- importance-aware calibration
- codebook/LUT search
- per-group/per-channel/per-tensor scale
- expert/family 別 LUT
- outlier handling
- perplexity、MMLU 系、needle、coding eval を含む評価

`sq` 側:

- hardware-native block layout
- activation-aware quantization
- FP8/NVFP4/MXFP8/FP4 向け scale 生成
- batch throughput を最大化する packing
- dequant-free または near-dequant-free kernel

### 3. Runtime

ランタイムは次の層に分ける。

- loader: uLLM container 形式を読む
- planner: モデル、量子化形式、ハードウェア能力から実行計画を作る
- executor: prefill、decode、draft、verify、diffusion refinement を実行する
- scheduler: continuous batching、KV cache、prefix cache、prefill/decode 分離を管理する
- backend: CPU、CUDA/NVIDIA、ROCm/AMD、CANN/Ascend、Intel/AMD NPU など

prefill と decode は最初から別 executor として扱う。異なる GPU への分離、異なる量子化形式、異なる batch policy を許す。

### 4. File Format

拡張子は未定だが、コンテナ形式は早期に決める。

候補:

- `.ullm`
- `.uaq` / `.usq`
- `.ulq`

推奨は `.ullm` をコンテナ拡張子にし、内部 manifest で `quant_family: aq | sq` を持つ形。

保存内容:

- manifest
- tokenizer metadata
- architecture graph
- tensor table
- quant blocks
- LUT/codebook table
- scale table
- backend hints
- calibration metadata
- benchmark provenance
- compatibility version

## ハードウェア方針

初期から全対応を同じ品質で狙わない。能力階層を定義する。

Tier 0:

- CPU AVX-512/VNNI/AMX 系
- 汎用 correctness backend
- aq の基準実装、量子化検証、ベンチ基盤

Tier 1:

- NVIDIA Hopper/Blackwell 系
- FP8、NVFP4、MXFP8 を使う sq
- サーバー throughput の主要比較対象

Tier 2:

- AMD CDNA3/4 系
- FP8/FP4 系、MFMA/Matrix Core を使う sq
- ROCm backend

Tier 3:

- AMD CDNA1/2、MI50/MI60、RDNA2/3
- aq 中心、sq は機能制限つき
- 実効性能は capability detection 後に判断する

Tier 4:

- Intel/AMD NPU、Huawei Ascend
- 直接カーネル実装ではなく backend plugin として扱う
- CANN などベンダーランタイム経由を基本にする

## ベンチマーク

比較は bpp だけでは不十分。最低限、以下を固定する。

- bpp
- context length
- batch size
- prefill tokens/s
- decode tokens/s
- total tokens/s
- latency p50/p95
- memory footprint
- accuracy/perplexity
- calibration dataset
- model revision
- backend revision

`aq` の主比較対象:

- FP16/BF16 baseline
- llama.cpp GGUF/imatrix
- Unsloth Dynamic 2.0 GGUF 系

`sq` の主比較対象:

- FP16/BF16 serving
- FP8 serving
- ATOM 系
- SGLang/vLLM 系ランタイム

## リスク

- `aq` と `sq` を一つの抽象で隠しすぎると、どちらにも最適化できない。
- 古い GPU まで sq の目標に入れると、ハードウェア演算器の制約で速度目標が崩れる。
- Qwen3 だけに寄せると、MTP、MoE、拡散型生成、multimodal で再設計が必要になる。
- NPU はベンダー SDK 依存が強く、初期から第一級 backend にすると実装が分散する。
- ベンチ条件を固定しないと、Unsloth/ATOM/SGLang との比較が意味を失う。

## 次の行動

1. v0.1 では `.ullm` コンテナと manifest schema を先に決める。
2. `aq` の reference quantizer を CPU で作る。
3. Qwen3 dense small model を最初の correctness target にする。
4. その後、MoE、MTP、prefill/decode 分離、GPU backend の順に広げる。
5. `sq` は NVIDIA Blackwell/Hopper と AMD CDNA3/4 を中心に、ハードウェア native format の調査を別タスク化する。

## 参照した外部情報

- NVIDIA Transformer Engine FP8/FP4 overview: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
- NVIDIA MXFP8 docs: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html
- NVIDIA NVFP4 inference blog: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- Qwen3 repository: https://github.com/QwenLM/Qwen3
- Qwen3 blog: https://qwen.ai/blog?id=qwen3
- Unsloth Dynamic 2.0 GGUF docs: https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs
- Unsloth Dynamic v2 blog: https://unsloth.ai/blog/dynamic-v2
- ATOM paper: https://arxiv.org/abs/2310.19102
- SGLang repository: https://github.com/sgl-project/sglang
- AMD Matrix Core CDNA blog: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- AMD GPUOpen Matrix Core note: https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/
- Huawei Ascend CANN: https://www.hiascend.com/eng/cann
- Gemma 4 MTP docs: https://ai.google.dev/gemma/docs/mtp/overview
- DiffusionGemma overview: https://ai.google.dev/gemma/docs/diffusiongemma
