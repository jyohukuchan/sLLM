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
   - 初期ターゲットは FP8 とする。
   - NVIDIA Blackwell 以降の NVFP4、MXFP8 や AMD CDNA4 以降の FP4 系は、検証環境を用意できた段階で追加する。
   - AMD CDNA3/4 以降では FP8 系、CDNA1/2 や RDNA 系では能力に応じて段階的に対応する。
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
- control plane: API、routing、prefill/decode 割り当て、telemetry、速度予測を管理する
- backend: HIP/AMD、CPU、CUDA/NVIDIA、JAX/TPU、CANN/Ascend、Intel/AMD NPU など

prefill と decode は最初から別 executor として扱う。異なる GPU への分離、異なる量子化形式、異なる batch policy を許す。

大部分の演算処理は C++20 で実装する。GPU kernel は初期段階では HIP C++ を直接書き、Triton には依存しない。API、scheduler、prefill/decode の割り当て、実行計画、速度予測などの全体制御には Rust を活用する。

### 4. File Format

拡張子とコンテナ形式は早期に決める。

候補:

- `.ullm`
- `.uaq` / `.usq`
- `.ulq`

推奨は `.ullm` を最終配布用の単一ファイル拡張子にし、内部 manifest で `quant_family: aq | sq` を持つ形。開発初期は `model.ullm/` という directory container で実装し、仕様が固まった後に同じ論理構造を単一ファイルへ pack する。

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

初期から全対応を同じ品質で狙わない。調達可能なハードウェア順に開発する。

Tier 0:

- HIP on Radeon PRO V620 / Radeon AI PRO R9700
- 直接 HIP C++ kernel を書く
- aq/sq FP8 周辺の基礎実装、ベンチ、runtime 統合を進める

Tier 1:

- AMD MI300X
- FP8 sq の主要検証環境
- 発表・宣伝前のサーバー級 throughput 評価対象

Tier 2:

- CPU AVX-512/VNNI/AMX 系
- correctness backend、fallback backend、CPU 向け aq 実行
- MI300X 後に実装優先度を上げる

Tier 3:

- NVIDIA Hopper/Blackwell 系
- FP8、MXFP8、NVFP4 を使う sq
- ハードウェア確保後に順次対応

Tier 4:

- JAX/TPU、Intel/AMD NPU、Huawei Ascend
- 直接カーネル実装ではなく backend plugin として扱う
- JAX、XLA、CANN などベンダー/フレームワークランタイム経由を基本にする

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
- predicted prefill/decode speed
- prediction error

`aq` の主比較対象:

- FP16/BF16 baseline
- llama.cpp GGUF/imatrix
- Unsloth Dynamic 2.0 GGUF 系

`sq` の主比較対象:

- FP16/BF16 serving
- FP8 serving
- ATOM 系
- SGLang/vLLM 系ランタイム
- TensorRT-LLM

## リスク

- `aq` と `sq` を一つの抽象で隠しすぎると、どちらにも最適化できない。
- 古い GPU まで sq の目標に入れると、ハードウェア演算器の制約で速度目標が崩れる。
- Qwen3 だけに寄せると、MTP、MoE、拡散型生成、multimodal で再設計が必要になる。
- NPU はベンダー SDK 依存が強く、初期から第一級 backend にすると実装が分散する。
- ベンチ条件を固定しないと、Unsloth/ATOM/SGLang との比較が意味を失う。
- vLLM、SGLang、llama.cpp、ATOM、TensorRT-LLM を参照する際に、無意識のコード流用が起きると Apache-2.0 配布方針と衝突する可能性がある。参照コードは Git 管理外に隔離し、実装は clean-room 方針で行う。

## 次の行動

1. v0.1 では `.ullm` directory container と manifest schema を先に決める。
2. Qwen3-14B と Qwen3-30B-A3B を初期 correctness target にする。
3. `aq` の reference quantizer と FP8 `sq` の最小仕様を作る。
4. HIP on V620/R9700、MI300X、AVX-512 の順で backend を進める。
5. その段階で一度発表・宣伝できる benchmark と demo を作る。
6. Qwen3.5 または Gemma4 を早期の新技術検証 target に追加する。
7. JAX/TPU、NPU、Ascend、NVIDIA backend は順次 plugin/backend として対応する。

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
- Qwen3.5 blog: https://qwen.ai/blog?id=qwen3.5
- Gemma 4 overview: https://ai.google.dev/gemma/docs/core
- Gemma 4 MTP docs: https://ai.google.dev/gemma/docs/mtp/overview
- DiffusionGemma overview: https://ai.google.dev/gemma/docs/diffusiongemma
