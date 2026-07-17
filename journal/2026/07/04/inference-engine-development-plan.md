# Inference engine development plan

## 前回の要点

- AQの仕様はまだ固定しない。
- `.ullm.d` package、direct conversion、AQ policy評価、IQ4同bpp補正の検証は進んだ。
- ただし、materializeとfused dequantの実測がないため、fqや最終AQ仕様を決めるには推論エンジン側が必要になった。

## 今回の変更点

- `docs/plans/inference-engine-development-plan-v0.1.md` を追加した。
- 既存推論エンジンを実装前に参照する手順を明文化した。
- 参照対象は llama.cpp、vLLM、SGLang、ATOM、AITER、TensorRT-LLM。
- 初期方針は R9700 / HIP C++ / C++20 / Rust control plane。
- 最初は token IDs 入力のCLI、`.ullm.d` loader、単一request BF16 correctness path、paged KV/batching、AQ/fq runtime experimentの順で進める。
- `docs/words.txt` に `fq`、`control plane`、`runtime`、`backend` を追加した。

## 次の行動

- `docs/research/inference-engine-reference-notes-v0.1.md` を作り、既存エンジンの参照ファイルと採用判断を記録する。
- Rust/C++境界、KV cache block layout、runtime backend interfaceのADRを作る。
- `crates/ullm-engine`、`crates/ullm-runtime-sys`、`runtime/include/ullm_runtime.h` のskeletonを作る。
