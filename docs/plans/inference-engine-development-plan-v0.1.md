# Inference engine development plan v0.1

## Purpose

`aq` / `sq` / `fq` の仕様は、kernel と scheduler の実測値なしに固定しない。まず uLLM の推論エンジンを、Qwen3.5-9B / Qwen3 系で実測できる最低限の形まで作る。

この計画では、思いつきで実装を進めないために、既存推論エンジンを設計入力として明示的に参照する。ただし `reference-src/` のコードは uLLM へ直接コピーしない。設計判断だけを抽出し、必要なら ADR に残す。

## 前回の要点

- `ullm-quant` は `.ullm.d` directory package を直接出力できる。
- Qwen3.5-9B の full package は p4p6 で約 `8.5 GiB`、quantized-only で約 `3.8 GiB` まで作成・検証済み。
- direct package conversion は jobs64 で約 `35-36 s`、最大RSSは約 `3.6-3.7 GiB`。
- AQ policy は full inproj248 project-text loss では all-g16 が保守的品質 baseline で、mixed policy は追加検証が必要。
- IQ4_XS replacement では同じ `4.25 bpp` 枠で `dual_codebook_l7_sel1` が有望だが、実推論kernel上の測定はまだない。

## 今回の変更点

- 推論エンジン開発を、量子化仕様決定より前に進める。
- 初期実装は R9700 / HIP C++ / C++20 を主対象にする。
- Rust は scheduler、request管理、telemetry、benchmark harness、prefill/decode割り当てに使う。
- C++ は model execution、HIP runtime、kernel、backend abstraction に使う。
- Python は correctness reference、benchmark集計、移行期の評価補助だけに限定する。

## Reference Policy

参照実装は次のように使う。

| Project | Local path | 参照する領域 | 使い方 |
|---|---|---|---|
| llama.cpp | `reference-src/llama.cpp` | model loader、Qwen/Gemma model wiring、KV cache、CPU fallback、GGUF量子化比較 | 小さく堅い実装の境界設計を見る |
| vLLM | `reference-src/vllm` | scheduler、paged KV、block manager、attention backend選択、benchmark schema | serving architecture とblock管理を抽出する |
| SGLang | `reference-src/sglang` | chunked prefill、RadixAttention、prefill/decode disaggregation、HiCache、ROCm attention backend | 高throughput servingとcache再利用を抽出する |
| ATOM | `reference-src/atom` | ROCm向けpaged attention、Qwen3.5 model、disaggregation、KV transfer | R9700/MI300Xに近いROCm実装を優先参照する |
| AITER | `reference-src/aiter` | HIP/ROCm kernel、FP8 attention/GEMM、paged attention tests | kernel形状、dtype、gfx1201/gfx942差分を見る |
| TensorRT-LLM | `reference-src/tensorrt-llm` | inflight batching、FP8 runtime、paged KV、multi-GPU execution | NVIDIA側の高throughput設計と将来sq目標を見る |

参照手順:

1. 実装前に該当領域の参照ファイルを `rg` で列挙する。
2. 重要な設計差分は `docs/research/inference-engine-reference-notes-v0.1.md` に書く。
3. 実装へ入れる前に、必要なら `docs/decisions/` に ADR を作る。
4. Apache-2.0 / MIT のコード断片を直接移植しない。必要な場合は、別途ライセンス判断を残す。
5. 参照したcommitは `docs/research/reference-source-inventory-v0.1.md` に追記または更新する。

## Target Scope

初期対象:

- hardware: R9700 first、V620は可能な範囲でllama.cpp/uLLM smoke、MI300Xは次段階
- backend: HIP C++ direct
- language standard: C++20
- control plane: Rust
- model: Qwen3.5-9B first、Qwen3-14B/Qwen3-30B-A3Bは後続
- format: existing `.ullm.d` package first、single-file `.ullm` は後回し
- dtype baseline: BF16/FP16 passthrough first
- quantized path: current all-g16/p4p6 `.ullm.d` を読み、materializeまたはfused dequantの実測へ進む
- API: 最初は token IDs 入力のCLI。tokenizer/server APIは後続

初期対象外:

- multi-node serving
- OpenAI互換HTTP API
- TPU/JAX backend
- Ascend/CANN backend
- CUDA/TensorRT backend
- NVFP4/MXFP4最適化
- MTP/speculative decodeの本実装

## Repository Layout

追加予定の構成:

```text
crates/
  ullm-engine/        # Rust control plane, scheduler, CLI, telemetry
  ullm-runtime-sys/   # C ABI binding to C++ runtime
runtime/
  include/            # C ABI and C++ runtime interfaces
  src/                # backend-independent execution code
  backends/
    hip/              # HIP C++ backend and kernels
    cpu/              # correctness/debug backend
kernels/
  hip/                # standalone HIP kernels if separated from runtime
tests/
  engine/
  runtime/
benchmarks/
  engine/
```

`ullm-quant` は変換器として維持し、推論エンジンとはcrateを分ける。

## Milestones

### E0: Reference audit and architecture notes

目的:

- 実装前に、既存エンジンから取り込む設計要素を固定する。
- 以降の実装で迷ったときの判断基準を残す。

手順:

1. vLLM の scheduler、KV block manager、attention backend docs/testsを読む。
2. SGLang の chunked prefill、RadixAttention、disaggregation、HiCache docs/testsを読む。
3. ATOM/AITER の ROCm paged attention、Qwen3.5 model、KV transferを読む。
4. llama.cpp の Qwen/Qwen3.5 model wiring、KV cache、batch処理、loaderを読む。
5. TensorRT-LLM の inflight batching、paged KV、FP8 runtimeを読む。
6. `docs/research/inference-engine-reference-notes-v0.1.md` を作る。
7. 最低限のADRを作る。

成果物:

- `docs/research/inference-engine-reference-notes-v0.1.md`
- `docs/decisions/0001-inference-engine-language-boundary.md`
- `docs/decisions/0002-kv-cache-block-layout.md`
- `docs/decisions/0003-runtime-backend-interface.md`

完了条件:

- 各参照実装について、参照したファイルと採用/不採用の理由が残っている。
- Rust/C++境界、KV cache block layout、backend interfaceの初期方針が決まっている。

### E1: Engine skeleton

目的:

- RustからC++ runtimeを呼び出し、device初期化、memory allocation、簡単なkernel起動、telemetry記録を通す。

手順:

1. `crates/ullm-runtime-sys` を作る。
2. `runtime/include/ullm_runtime.h` にC ABIを定義する。
3. HIP backendでdevice列挙、stream作成、buffer確保、copy、単純kernelを実装する。
4. `crates/ullm-engine` にCLIを作る。
5. `ullm-engine inspect-devices` と `ullm-engine runtime-smoke` を実装する。
6. build systemはCargo + CMakeまたはCargo build scriptで固定する。

成果物:

- `crates/ullm-engine`
- `crates/ullm-runtime-sys`
- `runtime/include/ullm_runtime.h`
- `runtime/backends/hip/`

完了条件:

- R9700でHIP deviceを選択できる。
- Rust CLIからC++ HIP kernelを起動できる。
- CI相当のlocal testでCPU-only環境ではHIP testをskipできる。

### E2: `.ullm.d` loader and model metadata

目的:

- 既存の `.ullm.d` full packageを推論エンジンから読めるようにする。
- 最初は全tensorをGPUへ載せる必要はなく、manifest、tensor table、payload位置を正しく扱う。

手順:

1. `ullm-engine inspect-package <model.ullm.d>` を実装する。
2. manifest、codebooks、quantized tensors、passthrough tensorsを読む。
3. tensor metadataをRust側で保持し、C++側へ必要最小限のlayoutを渡す。
4. passthrough BF16/FP16 tensorのstreaming loadを実装する。
5. quantized tensorは最初はmetadataだけ読み、materializeはE5以降に回す。

完了条件:

- p4p6 full packageのtensor数、passthrough数、codebook数、総bytesを既存summaryと一致させられる。
- loaderが巨大payloadを一括でメモリへ載せない。

### E3: Qwen3.5 single-request BF16 correctness path

目的:

- 量子化なしで、1 request / batch 1 のprefill + decodeを動かす。
- tokenizationは初期CLIでは外部で済ませ、token IDsを入力にする。

手順:

1. Qwen3.5 config parserを実装する。
2. layer execution planを作る。
3. RMSNorm、RoPE、attention、MLP、lm_headの最小実装を作る。
4. GEMMは初期段階ではrocBLAS/hipBLASLtを使う。
5. elementwise kernelはHIP C++で直接実装する。
6. PyTorch referenceとlogits差分を比較する。
7. 1 token decodeを通す。

完了条件:

- 1 promptのprefill logitsがPyTorch referenceに対して許容誤差内。
- 1 token decodeが動く。
- VRAM使用量と各layer時間がtelemetryに残る。

### E4: KV cache, paged attention, and batching

目的:

- 実測に必要なbatchingとdecode throughput測定を可能にする。
- ここで既存エンジンの設計参照が最も重要になる。

手順:

1. vLLM/SGLang/ATOMを参照し、KV block tableを設計する。
2. Rust schedulerでrequest、sequence、block allocationを管理する。
3. chunked prefillを実装する。
4. decode continuous batchingを実装する。
5. paged attention kernelをHIP C++で実装またはAITER/ATOM設計を参考に自前実装する。
6. prefix/cache reuseは最初は計測対象から外し、後でRadixAttention風のcacheへ進む。

完了条件:

- batch size、concurrent requests、context lengthを振ってdecode-tps/prefill-tpsを測れる。
- block allocatorの断片化、使用block数、KV bytesを記録できる。
- 既存エンジン benchmark schemaと同じ列で比較できる。

### E5: AQ/fq runtime experiment path

目的:

- `aq` / `fq` の仕様決定に必要な実測を取る。
- 最初から最終形式を決めず、materializeとfused dequantを比較する。

手順:

1. `.ullm.d` のAQ tensorをGPUへ転送する。
2. まず materialize path を作り、codebook-index + local-scaleからFP16/BF16 payloadへ展開する。
3. 次に fused dequant GEMV/GEMM のprototypeを作る。
4. `dual_codebook_l7_sel1`、`outlier補正`、現行all-g16/p4p6を小さく比較する。
5. prefillとdecodeで別々に測る。
6. memory bandwidth、LUT latency、GEMM occupancy、KV trafficを記録する。

完了条件:

- 同じprompt/gridでBF16 baseline、materialize AQ、fused AQを比較できる。
- BF16-error、logit差分、decode-tps、prefill-tps、VRAM消費量が同じrun recordに残る。
- この段階の結果で `fq` 仕様候補を絞れる。

### E6: Prefill/decode disaggregation and prediction

目的:

- 君が最初から要求していた prefill/decode 別GPU割り当てと速度予測を入れる。

手順:

1. ATOM/SGLangのdisaggregationとKV transferを参照する。
2. Rust schedulerにprefill worker/decode workerの概念を入れる。
3. 同一node複数GPUでKV handoffを実装する。
4. token/s予測モデルを、実測telemetryから作る。
5. 予測値と実測値の誤差を保存する。

完了条件:

- prefill専用GPUとdecode専用GPUの構成を指定できる。
- 予測decode-tps/prefill-tpsと実測値の誤差を記録できる。

### E7: MI300X and AVX-512 expansion

目的:

- R9700で動く構造をMI300Xへ移し、次にAVX-512 CPU backendを追加する。

手順:

1. MI300XでHIP backendのkernel shapeを再測定する。
2. MI300X向けFP8/sq candidateを測る。
3. AVX-512 INT8/BF16 pathを追加する。
4. CPU backendをcorrectness fallbackから実用benchmarkへ引き上げる。

完了条件:

- R9700とMI300Xの差分がbackend capabilityとして表現される。
- AVX-512 backendのdecode baselineが取れる。

## Benchmark Requirements

各runは最低限次を記録する。

- model / package path / quant policy
- hardware / ROCm version / driver / compiler
- backend / kernel variant / build commit
- context length / prompt tokens / generated tokens
- batch size / concurrent requests
- prefill-tps / decode-tps / total-tps
- TTFT / TPOT / p50 / p95
- VRAM baseline / peak / consumed
- KV cache bytes / allocated blocks / free blocks
- `decode-tps * consumed VRAM GiB`
- correctness metric: logits relative MSE、top-k agreement、loss delta

既存の `docs/specs/inference-benchmark-result-v0.1.md` と互換にする。足りない項目があればschemaを拡張する。

## Immediate Next Actions

1. `docs/research/inference-engine-reference-notes-v0.1.md` を作り、vLLM/SGLang/ATOM/AITER/llama.cpp/TensorRT-LLMの参照ファイルを読む。
2. `docs/decisions/0001-inference-engine-language-boundary.md` を作る。
3. `crates/ullm-engine` と `crates/ullm-runtime-sys` の空crateを作る。
4. `runtime/include/ullm_runtime.h` とHIP runtime smokeを作る。
5. R9700で `inspect-devices` と `runtime-smoke` を通す。

