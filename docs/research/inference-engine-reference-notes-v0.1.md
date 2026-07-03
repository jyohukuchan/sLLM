# Inference engine reference notes v0.1

## Purpose

uLLM推論エンジンで参照する設計要素を整理する。ここに挙げるファイルは設計調査の入口であり、実装コードのコピー元ではない。

## License Boundary

- `reference-src/` のコードは読んでよいが、uLLMへ直接コピーしない。
- 実装へ取り込むのは、公開API、挙動、データ構造上の設計判断、benchmark観点に限る。
- Apache-2.0/MITのコードであっても、意図的に取り込む場合は著作権表示、LICENSE、NOTICE要件をADRに記録してから行う。
- 参照元のthird-party、生成済みkernel、binary artifact、model artifactは個別にライセンス確認する。

## vLLM

| 参照ファイル | 見る観点 | 採用/保留判断 |
|---|---|---|
| `reference-src/vllm/vllm/v1/core/sched/scheduler.py` | request scheduling、running/waiting管理、token budget | scheduling state machineの観点だけ採用候補 |
| `reference-src/vllm/vllm/v1/core/sched/request_queue.py` | waiting queue、priority、arrival順序 | Rust schedulerのqueue設計で参照 |
| `reference-src/vllm/vllm/v1/core/kv_cache_manager.py` | KV block割当、prefix/cache連携、block lifetime | E4のKV block allocator設計で参照 |
| `reference-src/vllm/vllm/v1/core/single_type_kv_cache_manager.py` | 単一KV種別のblock管理 | 初期の単純paged KVに近いので優先参照 |
| `reference-src/vllm/vllm/v1/core/block_pool.py` | free block pool、block reuse | allocator invariantの参考にする |
| `reference-src/vllm/vllm/v1/kv_cache_interface.py` | KV cache abstraction、layer/group spec | uLLM runtime/backend境界の比較対象 |
| `reference-src/vllm/vllm/v1/attention/selector.py` | attention backend選択 | backend capability選択の参考にする |
| `reference-src/vllm/vllm/v1/attention/backends/registry.py` | attention backend registry | uLLM backend registryはRust側で簡素に実装 |
| `reference-src/vllm/vllm/model_executor/layers/attention/attention.py` | model layerからattention backendへの呼び出し境界 | C++ runtimeのlayer execution境界で参照 |
| `reference-src/vllm/docs/design/paged_attention.md` | paged attentionの設計意図 | KV layout ADRの入力にする |
| `reference-src/vllm/docs/design/hybrid_kv_cache_manager.md` | hybrid KV cache grouping | Qwen3.5/Gemma4など混在attention向けに保留 |
| `reference-src/vllm/docs/features/disagg_prefill.md` | prefill/decode分離 | E6 disaggregationで再参照 |

## SGLang

| 参照ファイル | 見る観点 | 採用/保留判断 |
|---|---|---|
| `reference-src/sglang/python/sglang/srt/managers/scheduler.py` | continuous batching、chunked prefill、scheduler本体 | vLLMとの対比でE4設計に使う |
| `reference-src/sglang/python/sglang/srt/managers/scheduler_components/load_inquirer.py` | load推定 | 速度予測機能の初期特徴量候補 |
| `reference-src/sglang/python/sglang/srt/managers/scheduler_components/kv_events_publisher.py` | KV event発行 | telemetry/event設計で参照 |
| `reference-src/sglang/python/sglang/srt/managers/scheduler_input_blocker.py` | 入力制御、backpressure | queue制御の参考にする |
| `reference-src/sglang/python/sglang/srt/mem_cache/unified_radix_cache.py` | Radix cache、prefix reuse | 初期実装では保留、cache reuse段階で採用候補 |
| `reference-src/sglang/python/sglang/srt/disaggregation/prefill.py` | prefill worker側flow | E6で参照 |
| `reference-src/sglang/python/sglang/srt/disaggregation/decode.py` | decode worker側flow | E6で参照 |
| `reference-src/sglang/python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py` | decode側KV offload管理 | KV transfer設計で参照 |
| `reference-src/sglang/python/sglang/srt/disaggregation/kv_events.py` | disaggregation event定義 | uLLM event schema候補 |
| `reference-src/sglang/python/sglang/srt/layers/attention/attention_registry.py` | attention backend登録/選択 | backend registryの比較対象 |
| `reference-src/sglang/docs/advanced_features/pd_disaggregation.md` | prefill/decode分離の設計説明 | E6の設計入力 |
| `reference-src/sglang/docs/advanced_features/attention_backend.md` | attention backend方針 | HIP backend選択の比較対象 |

## ATOM

| 参照ファイル | 見る観点 | 採用/保留判断 |
|---|---|---|
| `reference-src/atom/docs/scheduling_kv_cache_guide.md` | schedulingとKV cache設計 | R9700/MI300X向けの実装判断で優先参照 |
| `reference-src/atom/atom/model_engine/model_runner.py` | model runner境界 | Rust control planeとC++ runtime境界の比較対象 |
| `reference-src/atom/atom/models/qwen3_5.py` | Qwen3.5 model実装 | E3のmodel wiringで参照 |
| `reference-src/atom/atom/plugin/vllm/models/qwen3_5.py` | vLLM plugin側Qwen3.5 | vLLM互換のshape/metadata確認に使う |
| `reference-src/atom/atom/plugin/sglang/models/qwen3_5.py` | SGLang plugin側Qwen3.5 | SGLang互換のattention差分確認に使う |
| `reference-src/atom/atom/model_ops/paged_attention.py` | paged attention呼び出し | E4 kernel API設計で参照 |
| `reference-src/atom/atom/model_ops/v4_kernels/paged_prefill.py` | paged prefill | HIP paged prefill実測時に参照 |
| `reference-src/atom/atom/model_ops/v4_kernels/paged_decode.py` | paged decode | HIP decode kernel設計で参照 |
| `reference-src/atom/atom/plugin/sglang/attention_backend/full_attention/kv_cache.py` | SGLang pluginのKV cache | KV layout比較で参照 |
| `reference-src/atom/atom/kv_transfer/disaggregation/aggregator.py` | KV transfer aggregation | E6 disaggregationで参照 |
| `reference-src/atom/atom/kv_transfer/disaggregation/types.py` | KV transfer type定義 | uLLM event/type設計で参照 |
| `reference-src/atom/atom/kv_transfer/offload/connector.py` | offload connector | 後続のKV offloadで参照 |

## AITER

| 参照ファイル | 見る観点 | 採用/保留判断 |
|---|---|---|
| `reference-src/aiter/csrc/include/attention_common.cuh` | attention kernel共通型 | HIP kernel ABI設計の比較対象 |
| `reference-src/aiter/csrc/include/attention_dtypes.h` | dtype定義 | FP8/BF16 pathの型設計で参照 |
| `reference-src/aiter/csrc/include/hip_float8.h` | HIP FP8 helper | sq/FP8実測段階で参照 |
| `reference-src/aiter/csrc/include/dtype_fp8.cuh` | FP8 dtype helper | FP8 payload検証で参照 |
| `reference-src/aiter/op_tests/test_batch_prefill.py` | batch prefill test | E4/E5のtest shape候補 |
| `reference-src/aiter/op_tests/test_kvcache.py` | KV cache test | KV layoutのcorrectness test候補 |
| `reference-src/aiter/op_tests/test_kvcache_blockscale.py` | block scale付きKV cache | sq/FP8 KV検証で参照 |
| `reference-src/aiter/op_tests/test_mha_fp8.py` | FP8 MHA test | FP8 attention実測で参照 |
| `reference-src/aiter/op_tests/test_mha_varlen_fp8.py` | varlen FP8 attention | batching実測で参照 |
| `reference-src/aiter/op_tests/triton_tests/attention/test_pa_decode.py` | paged decode reference/test | 自前HIP kernelの比較対象 |
| `reference-src/aiter/op_tests/triton_tests/attention/test_pa_prefill.py` | paged prefill reference/test | 自前HIP kernelの比較対象 |
| `reference-src/aiter/csrc/py_itfs_cu/asm_fmha_fwd_mxfp8.cu` | MXFP8 FMHA entry | MI300X以降のsq候補で参照 |

## llama.cpp

| 参照ファイル | 見る観点 | 採用/保留判断 |
|---|---|---|
| `reference-src/llama.cpp/src/llama-model-loader.cpp` | model file loading、tensor metadata | `.ullm.d` loaderとの比較対象 |
| `reference-src/llama.cpp/src/llama-model-loader.h` | loader interface | C++ runtime loader境界で参照 |
| `reference-src/llama.cpp/src/llama-batch.cpp` | batch表現 | token IDs CLIとbatching設計で参照 |
| `reference-src/llama.cpp/src/llama-batch.h` | batch interface | 初期batch APIの比較対象 |
| `reference-src/llama.cpp/src/llama-kv-cache.cpp` | KV cache本体 | CPU fallback/単純KV設計で参照 |
| `reference-src/llama.cpp/src/llama-kv-cache.h` | KV cache interface | uLLM KV ABIの比較対象 |
| `reference-src/llama.cpp/src/llama-context.cpp` | context lifecycle | Rust/C++ runtime lifecycleで参照 |
| `reference-src/llama.cpp/src/llama-graph.cpp` | graph execution構築 | E3 layer execution planの比較対象 |
| `reference-src/llama.cpp/src/models/qwen35.cpp` | Qwen3.5 wiring | 最初のmodel targetで参照 |
| `reference-src/llama.cpp/src/models/qwen3.cpp` | Qwen3 wiring | Qwen3-14B/Qwen3-30B-A3Bで参照 |
| `reference-src/llama.cpp/src/models/qwen3next.cpp` | Qwen3 Next wiring | MTP/新技術対応で後続参照 |

## TensorRT-LLM

| 参照ファイル | 見る観点 | 採用/保留判断 |
|---|---|---|
| `reference-src/tensorrt-llm/docs/source/features/paged-attention-ifb-scheduler.md` | paged attention + inflight batching | sq/server目標の比較対象 |
| `reference-src/tensorrt-llm/docs/source/features/kvcache.md` | KV cache機能 | KV policy設計で参照 |
| `reference-src/tensorrt-llm/docs/source/features/kv-cache-connector.md` | KV connector | E6 transfer設計で参照 |
| `reference-src/tensorrt-llm/docs/source/features/overlap-scheduler.md` | overlap scheduler | prefill/decode overlapで参照 |
| `reference-src/tensorrt-llm/docs/source/torch/scheduler.md` | torch runtime scheduler | Rust scheduler比較で参照 |
| `reference-src/tensorrt-llm/docs/source/torch/kv_cache_manager.md` | torch KV manager | block manager比較で参照 |
| `reference-src/tensorrt-llm/cpp/include/tensorrt_llm/batch_manager/kvCacheManager.h` | C++ KV manager interface | C++ runtime API比較で参照 |
| `reference-src/tensorrt-llm/cpp/include/tensorrt_llm/batch_manager/kvCacheTransferManager.h` | KV transfer manager | E6 transfer設計で参照 |
| `reference-src/tensorrt-llm/cpp/include/tensorrt_llm/executor/executor.h` | executor interface | control plane/runtime境界で参照 |
| `reference-src/tensorrt-llm/cpp/include/tensorrt_llm/runtime/gptDecoderBatched.h` | batched decoder | decode batching設計で参照 |
| `reference-src/tensorrt-llm/cpp/include/tensorrt_llm/runtime/bufferManager.h` | runtime buffer管理 | C++ memory handle設計で参照 |
| `reference-src/tensorrt-llm/cpp/include/tensorrt_llm/deep_gemm/fp8_gemm.cuh` | FP8 GEMM interface | MI300X/NVIDIA sq比較で後続参照 |

## Immediate Design Inputs

- E2 `.ullm.d` loaderは llama.cpp loader と TensorRT-LLM buffer manager を参照しつつ、巨大payloadを一括読みしない設計にする。
- E3 Qwen3.5 correctness pathは llama.cpp / ATOM のQwen3.5 wiringを参照する。
- E4 KV cacheは vLLMのblock pool、SGLangのscheduler、ATOMのROCm paged attentionを比較してADR化する。
- E6 prefill/decode disaggregationは SGLang/ATOM/TensorRT-LLMのKV transfer設計を比較してから実装する。
