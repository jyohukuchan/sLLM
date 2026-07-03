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
- `docs/decisions/0003-kv-cache-block-layout.md` を追加し、初期KV cache block layoutを決めた。
- C++ runtime C ABIにcontext、buffer、stream handleを追加した。
- `runtime-memory-smoke` と `runtime-stream-smoke` CLIを追加した。
- `KvBlockAllocator` にdefault block-size `16` tokensと断片化telemetryを追加した。
- C++ runtime C ABIにhost-device-copyを追加し、Rust側に `RuntimeBuffer::copy_from_host` / `copy_to_host` を追加した。
- `runtime-copy-smoke` CLIを追加し、CPU fallbackとR9700 HIPでbyte payloadの往復検証をできるようにした。
- `.ullm.d` manifestから最小の非空参照ファイルを選ぶhelperと `package-load-smoke` CLIを追加した。

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
- `cargo run -p ullm-engine -- runtime-memory-smoke 0`
  - CPU fallback buffer allocation smokeが成功した。
- `cargo run -p ullm-engine -- runtime-memory-smoke 2`
  - R9700 HIP buffer allocation smokeが成功した。
- `cargo run -p ullm-engine -- runtime-stream-smoke 0`
  - CPU fallback stream synchronize smokeが成功した。
- `cargo run -p ullm-engine -- runtime-stream-smoke 2`
  - R9700 HIP stream synchronize smokeが成功した。
- `cargo run -p ullm-engine -- runtime-copy-smoke 0`
  - CPU fallback runtime bufferへの4096B往復copyが成功した。
- `cargo run -p ullm-engine -- runtime-copy-smoke 2`
  - R9700 HIP runtime bufferへの4096B往復copyが成功した。
- `cargo run -p ullm-engine -- package-load-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d 0`
  - `.ullm.d` 内の `codebooks/attn_k__aq4_e4m3_g8_ts_flloyd16.f32` 64BをCPU fallback runtime bufferへloadし、readback検証が成功した。
- `cargo run -p ullm-engine -- package-load-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d 2`
  - 同じ64B payloadをR9700 HIP runtime bufferへloadし、readback検証が成功した。
- `cargo fmt --all --check` passed。
- `cargo test --workspace` passed。

## 作成したgit checkpoints

- `4842d52 Add runtime boundary and notice policy`
- `0f6448b Add inference runtime skeleton`
- `740df4f Record inference engine reference notes`
- `5c22fc5 Add ullm package inspection CLI`
- `bc0e704 Add control plane scheduler primitives`
- `c3af6b2 Document KV cache block layout`
- `e654a62 Add runtime memory allocation smoke`
- `2635b06 Add KV block allocator telemetry`
- `f4db981 Add runtime stream smoke`
- `ca1a97c Add runtime buffer copy smoke`
- `192d9ae Add package payload load smoke`

## 次の行動

- `package-load-smoke` は現状で最小の非空参照ファイルを選ぶため、実パッケージでは64B codebookが選ばれている。次はtensor index/scale/passthrough payloadを明示選択するloader smokeへ進める。
- `.ullm.d` manifest metadataをruntime側のweight registryへ渡す設計を具体化する。
- Qwen3系のattention/MLP最小forwardに必要なkernel境界を、既存推論エンジン実装を参照しながら切り出す。
