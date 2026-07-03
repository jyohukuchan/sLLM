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
- `.ullm.d` の参照payloadを `smallest` / `tensor-index` / `tensor-scale` / `tensor-codebook` / `codebook` / `passthrough` のpayload roleで選べるようにした。
- `package-load-smoke PACKAGE_DIR [DEVICE_INDEX] [MAX_BYTES] [PAYLOAD_ROLE]` に拡張し、role、owner index、owner nameをログへ出すようにした。
- `package-tensor-load-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR]` を追加し、1つのquantized tensorのindex、scale、codebookをまとめてruntime bufferへchunked loadできるようにした。
- tensor selectorは未指定または数値index、完全一致tensor名、または一意な部分一致を受け付けるようにした。
- `crates/ullm-engine/src/loader.rs` を追加し、`LoadOptions`、`LoadedPayload`、`LoadedTensorBundle`、`WeightRegistry` を実装した。
- `WeightRegistry` はtensor bundleのindex、scale、codebookをそれぞれruntime bufferへloadし、load済みのresident payloadとして保持する。
- `package-weight-register-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR]` を追加し、`.ullm.d` の1 tensor分のpayloadをregistryに登録して後続処理から参照できる状態を検証できるようにした。
- `WeightRegistry::load_and_insert_many` を追加し、複数の `TensorPayloadBundle` を1回のAPI呼び出しでregistryへ登録できるようにした。
- `package::list_tensor_payload_bundles` を追加し、`.ullm.d` manifest内のquantized tensor payload bundleをmanifest順で列挙できるようにした。
- `package-weight-register-many-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [MAX_TENSORS]` を追加し、package内の先頭N個のquantized tensorをweight registryへまとめて登録できるようにした。誤ってfull packageを全量loadしないよう、CLIでは件数上限を明示的に扱う。
- `LoadedPayload` が `Arc<RuntimeBuffer>` を保持するようにし、同一 `codebook_file` を参照する複数tensorでcodebook runtime bufferを共有できるようにした。
- `WeightRegistry` にcodebook poolを追加し、`codebook_payloads` と `resident_payload_bytes` を取得できるようにした。
- CLI出力では、tensorごとの参照量である `registry_payload_bytes` と、共有後の実resident量である `resident_payload_bytes` を分けて表示するようにした。

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
- `target/debug/ullm-engine package-load-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d 2 1048576 tensor-index`
  - `tensors/000-model_language_model_layers_0_linear_attn_in_proj_a_weight.idx4` 65,536BをR9700 HIP runtime bufferへloadし、readback検証が成功した。
- `target/debug/ullm-engine package-load-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d 2 1048576 tensor-scale`
  - `tensors/000-model_language_model_layers_0_linear_attn_in_proj_a_weight.scale_u8` 8,192BをR9700 HIP runtime bufferへloadし、readback検証が成功した。
- `target/debug/ullm-engine package-load-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d 2 1048576 passthrough`
  - `passthrough/005-model_language_model_layers_0_linear_attn_dt_bias.raw` 64BをR9700 HIP runtime bufferへloadし、readback検証が成功した。
- `target/debug/ullm-engine package-load-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d 2 1048576 tensor-codebook`
  - `codebooks/attn_k__aq4_e4m3_g8_ts_flloyd16.f32` 64Bをtensor由来codebookとしてR9700 HIP runtime bufferへloadし、readback検証が成功した。
- `target/debug/ullm-engine package-load-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d 2 1048576 codebook`
  - `codebooks/attn_k__aq4_e4m3_g8_ts_flloyd16.f32` 64Bをtop-level codebookとしてR9700 HIP runtime bufferへloadし、readback検証が成功した。
- `target/debug/ullm-engine package-tensor-load-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d 2 1048576 0`
  - tensor `model.language_model.layers.0.linear_attn.in_proj_a.weight` のidx4 65,536B、scale 8,192B、codebook 64BをR9700 HIP runtime bufferへchunked loadし、readback検証が成功した。
- `target/debug/ullm-engine package-tensor-load-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d 2 1048576 27`
  - tensor `model.language_model.layers.11.self_attn.k_proj.weight` のidx4 2,097,152Bを2 chunks、scale 524,288Bを1 chunk、codebook 64Bを1 chunkでR9700 HIP runtime bufferへchunked loadし、readback検証が成功した。
- `target/debug/ullm-engine package-tensor-load-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d 2 1048576 model.language_model.layers.11.self_attn.k_proj.weight`
  - tensor名指定でも同じtensor 27のbundle load検証が成功した。
- `target/debug/ullm-engine package-weight-register-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d 2 1048576 0`
  - tensor `model.language_model.layers.0.linear_attn.in_proj_a.weight` のidx4 65,536B、scale 8,192B、codebook 64BをR9700 HIP runtime bufferへloadし、weight registry上のresident payloadとして保持できた。
  - `registry_payload_bytes=73792`、index chunks `1`、scale chunks `1`、codebook chunks `1`。
- `target/debug/ullm-engine package-weight-register-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d 2 1048576 27`
  - tensor `model.language_model.layers.11.self_attn.k_proj.weight` のidx4 2,097,152B、scale 524,288B、codebook 64BをR9700 HIP runtime bufferへloadし、weight registry上のresident payloadとして保持できた。
  - `registry_payload_bytes=2621504`、index chunks `2`、scale chunks `1`、codebook chunks `1`。
- `target/debug/ullm-engine package-weight-register-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d 2 1048576 model.language_model.layers.11.self_attn.k_proj.weight`
  - tensor名指定でも同じtensor 27のregistry登録検証が成功した。
- `target/debug/ullm-engine package-weight-register-many-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d 2 1048576 2`
  - R9700 HIPでpackage内255 quantized tensorsのうち先頭2 tensorsをweight registryへ登録できた。
  - `registry_tensors=2`、`registry_payload_bytes=147584`。
  - 登録tensorは `model.language_model.layers.0.linear_attn.in_proj_a.weight` と `model.language_model.layers.0.linear_attn.in_proj_b.weight` で、それぞれpayload bytes `73792`。
- `target/debug/ullm-engine package-weight-register-many-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d 0 1048576 2`
  - CPU fallbackでも同じ先頭2 tensorsのweight registry登録が成功した。
- `target/debug/ullm-engine package-weight-register-many-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d 2 1048576 16`
  - R9700 HIPで先頭16 tensorsをweight registryへ登録できた。
  - `registry_payload_bytes=247759872`、`resident_payload_bytes=247759360`、`codebook_payloads=8`。
  - 先頭16 tensorsでは8種類のcodebookが2回ずつ現れるため、64B codebook 8個分、合計512Bの重複がresident payloadから除外された。
- `cargo fmt --all --check` passed。
- `cargo test --workspace` passed。
- `cargo test -p ullm-engine` passed。`ullm-engine` は14 tests。
- `cargo build -p ullm-engine` passed。
- codebook dedup後の `cargo test -p ullm-engine` は15 tests passed。

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
- `f767287 Add package payload role selection`
- `901aca4 Add package tensor bundle load smoke`
- `7f2cba7 Add runtime weight registry smoke`
- `746d422 Add multi tensor weight registry load`
- `d19ecd8 Add package tensor bundle listing smoke`
- `98bd2a6 Deduplicate registry codebook payloads`

## 次の行動

- `WeightRegistry` で複数tensor分のresident payloadを保持し、同一codebook payloadを共有できるようになった。
- 次はpackage全体を扱うruntime-side package handleを作り、manifest由来metadata、registry、codebook poolをまとめて保持する。
- 後続kernelから参照できる形にするため、tensor名・family・candidate id・payload buffer handleをまとめてlookupできるAPIを整える。
- Qwen3系のattention/MLP最小forwardに必要なkernel境界を、既存推論エンジン実装を参照しながら切り出す。
