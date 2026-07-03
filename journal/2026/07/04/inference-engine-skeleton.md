# Inference engine skeleton

## 前回の要点

- 推論エンジン開発計画を作り、既存推論エンジンを積極的に参照する方針を決めた。
- `aq` / `sq` / `fq` は、推論エンジン上の実測なしに仕様固定しない。

## 今回の変更点

- `NOTICE` を追加し、最低限の著作権表示とreference sourceの扱いを明文化した。
- `docs/decisions/0002-inference-engine-language-boundary.md` を追加し、Rust control plane / C++20 runtime / C ABI境界を決めた。
- `docs/research/inference-engine-reference-notes-v0.1.md` を追加し、llama.cpp、vLLM、SGLang、ATOM、AITER、TensorRT-LLMの参照ファイル候補を記録した。
- `crates/ullm-runtime-sys` を追加し、C++ runtimeのC ABIをRustから呼べるようにした。
- `runtime/include/ullm_runtime.h` と `runtime/src/ullm_runtime.cpp` を追加した。
- `crates/ullm-engine` を追加し、`inspect-devices`、`runtime-smoke`、`inspect-package` CLIを実装した。
- `ullm-engine` に最小の `RequestQueue` と `KvBlockAllocator` を追加した。

## 実測・検証

- `cargo run -p ullm-engine -- inspect-devices`
  - CPU fallback 1件とHIP device 3件を検出した。
  - HIP側は V620、R9700、V620 の3件として表示された。
- `cargo run -p ullm-engine -- runtime-smoke`
  - C++ runtime経由の `add_f32` smokeが成功した。
- `cargo run -p ullm-engine -- inspect-package /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d`
  - quantized tensors `255`
  - passthrough tensors `520`
  - codebooks `12`
  - referenced files `1042`
  - missing referenced files `0`
- `cargo fmt --all --check` passed。
- `cargo test --workspace` passed。

## 作成したgit checkpoints

- `4842d52 Add runtime boundary and notice policy`
- `0f6448b Add inference runtime skeleton`
- `740df4f Record inference engine reference notes`
- `5c22fc5 Add ullm package inspection CLI`
- `bc0e704 Add control plane scheduler primitives`

## 次の行動

- `docs/decisions/0003-kv-cache-block-layout.md` を作り、block-size、block table、sequence mapping、free policyを決める。
- `inspect-package` の結果を使って `.ullm.d` のmetadataをruntimeへ渡す準備をする。
- C++ runtimeにmemory handle / stream handleのABIを追加し、HIP buffer allocation smokeへ進む。
