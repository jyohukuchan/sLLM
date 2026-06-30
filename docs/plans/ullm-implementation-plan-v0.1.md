# uLLM implementation plan v0.1

## Purpose

この文書は、`docs/concepts/ullm-concept-v0.1.md` の構想を、実装可能な手順へ落とすための初期計画である。

この版では、最短で価値を確認するために次の順序を採用する。

1. 既存推論エンジンの benchmark 条件と結果 schema を固定する。
2. llama.cpp を V620/R9700 で測り、MI300X 以降で vLLM、SGLang、ROCm/ATOM、TensorRT-LLM も測れる形にする。
3. `.ullm` directory container と manifest schema を仮固定する。
4. Qwen3-14B と Qwen3-30B-A3B を最初の correctness target にする。
5. `aq` と `sq` は仕様を先に固定せず、実験 protocol と採用判定基準を作る。
6. C++20 runtime と HIP C++ backend を作り、V620/R9700 で同じ `.ullm` を実行できるようにする。
7. MI300X で FP8 系 `sq` 候補のサーバー級 benchmark を取る。
8. AVX-512 CPU backend を追加する。
9. その段階で発表・宣伝できる demo、benchmark、配布物を整える。

## Assumptions

- 構想文書は `docs/concepts/` に置く。
- 具体的な実装計画は `docs/plans/` に置く。
- 作業記録は `journal/` に置く。
- 初期実装は、正しさ検証と量子化研究を Python/PyTorch で進める。
- 演算処理の大部分は C++20 を中心に実装する。
- 初期 GPU backend は HIP C++ を直接書く。過去プロジェクトの経験を踏まえ、Triton は初期依存にしない。
- API、scheduler、prefill/decode の割り当て、telemetry、速度予測などの control plane は Rust を活用する。
- Python は reference、converter、quantizer、evaluation のために使う。
- 大きなモデルや評価データは一度にメモリへ載せず、streaming と chunk 処理を前提にする。
- 最初の GPU 検証環境は Radeon PRO V620 と Radeon AI PRO R9700 とする。
- 次のサーバー級検証環境は MI300X とする。
- CPU AVX-512 backend は MI300X 後に優先度を上げる。
- Qwen3.5 または Gemma4 は、Unsloth Dynamic 系との比較と MTP/新技術検証のために早期 target へ昇格させる。
- JAX/TPU は backend/plugin 構想に含めるが、初期実装 target にはしない。
- `aq` と `sq` の詳細な量子化仕様は、事前に決めない。最初に決めるのは実験 ID、payload metadata、測定条件、結果 schema、採用判定基準である。
- Qwen3 実装の前に、既存推論エンジンの token/s を測定する phase を挟む。

## Repository Layout

初期の推奨レイアウトは次の通り。

```text
uLLM-project/
  docs/
    concepts/
    plans/
    specs/
    research/
    decisions/
  python/
    ullm_quant/
    ullm_format/
    ullm_eval/
  crates/
    ullm_control/
    ullm_scheduler/
    ullm_api/
  runtime/
    include/
    src/
    backends/
      cpu/
      hip/
      cuda/
      jax_tpu/
      cann/
  kernels/
    cpu/
    cuda/
    hip/
  schemas/
  tools/
  tests/
    format/
    quant/
    runtime/
    eval/
  benchmarks/
  reference-src/  # ignored
  journal/
```

最初にすべてを作る必要はない。Phase 1 では `docs/specs/`、`docs/plans/`、`benchmarks/`、`tools/`、`schemas/`、`python/`、`tests/format/` だけでよい。`reference-src/` は Git 管理外に置き、参照コードを uLLM の実装へ直接コピーしない。

## Target Milestones

### M0: Planning baseline

目的:

- 構想、計画、仕様、作業記録の置き場所を分離する。
- 最初の実装対象を固定する。

手順:

1. `docs/concepts/ullm-concept-v0.1.md` を構想の基準文書にする。
2. この文書を `docs/plans/ullm-implementation-plan-v0.1.md` として置く。
3. 仕様文書用に `docs/specs/` を作る。
4. 重要な決定は `docs/decisions/` に ADR として保存する。
5. 最初の対象モデルを Qwen3-14B と Qwen3-30B-A3B に固定する。
6. Qwen3.5 または Gemma4 を早期の追加検証 target として扱う。

成果物:

- `docs/plans/ullm-implementation-plan-v0.1.md`
- 後続の ADR テンプレート

完了条件:

- 構想文書と計画文書が別ファイルになっている。
- 次に作る仕様文書の対象が明確になっている。

### M1: `.ullm` format v0.1 draft

目的:

- `.ullm` の最小仕様を決める。
- `aq` と `sq` の詳細を固定せず、実験 payload を manifest で参照できるようにする。

手順:

1. `docs/specs/ullm-container-v0.1.md` を作る。
2. `schemas/ullm-manifest-v0.1.schema.json` を作る。
3. 初期実装では、debug と差分確認を優先して directory container の `model.ullm/` を採用する。
4. single-file `.ullm` にするかどうか、単一ファイル化する場合の内部構造は後で決める。
5. directory container の標準構成を定義する。

```text
model.ullm/
  manifest.json
  tokenizer.json
  tensors/
    000000.bin
    000001.bin
  codebooks/
    000000.bin
  scales/
    000000.bin
  calibration.json
  provenance.json
```

6. `manifest.json` に最低限の項目を入れる。

- `format_version`
- `quant_family`
- `architecture`
- `model_config`
- `tensor_table`
- `quant_blocks`
- `codebooks`
- `scales`
- `tokenizer`
- `calibration`
- `backend_hints`
- `provenance`

7. alignment、endianness、checksum、tensor chunk の扱いを決める。
8. `tensors/`、`codebooks/`、`scales/` は固定仕様ではなく、manifest が解釈方法を指定する論理ストレージとして扱う。
9. `aq` 用の codebook/LUT や `sq` 用の scale metadata は、必要な実験でだけ使う。
10. 未対応機能を拒否するための capability field を定義する。
11. single-file 化は未決定事項として記録し、v0.1 仕様の決定範囲から外す。

成果物:

- `docs/specs/ullm-container-v0.1.md`
- `schemas/ullm-manifest-v0.1.schema.json`
- `tests/format/` の manifest validation test

完了条件:

- 空の `.ullm` directory container を validate できる。
- FP baseline と quantization experiment payload の manifest example を validate できる。
- manifest schema が unknown field と missing required field を検出できる。
- single-file 化を決めなくても、directory container と manifest schema の検証が進められる。

### M2.5: Existing inference engine benchmark baseline

目的:

- Qwen3 実装前に、既存推論エンジンの token/s と制約を条件別に測定する。
- uLLM の目標を、推測ではなく実測 baseline から決める。

手順:

1. `docs/specs/inference-benchmark-result-v0.1.md` を作る。
2. `docs/plans/existing-engine-benchmark-plan-v0.1.md` を作る。
3. 結果保存先を `benchmarks/results/` にする。
4. JSONL result schema を定義する。
5. 共通測定条件を定義する。

- engine
- engine commit
- model
- quantization
- context length
- prompt tokens
- generated tokens
- batch size
- concurrent requests
- tensor parallelism
- pipeline parallelism
- GPU count
- GPU model
- backend
- driver/runtime version
- KV cache dtype
- prefill tokens/s
- decode tokens/s
- total tokens/s
- latency p50/p95
- VRAM peak
- power if available
- unsupported/OOM reason

6. V620/R9700 では llama.cpp を主な実測対象にする。
7. vLLM、SGLang、ROCm/ATOM、TensorRT-LLM は V620 で無理に動かさず、MI300X/NVIDIA 環境用の測定 harness を先に作る。
8. TP/PP を指定できない engine では、その条件を `unsupported` として記録する。
9. 結果から uLLM の最初の target condition を決める。

成果物:

- `docs/specs/inference-benchmark-result-v0.1.md`
- `docs/plans/existing-engine-benchmark-plan-v0.1.md`
- `benchmarks/results/README.md`
- llama.cpp benchmark wrapper
- engine capability matrix

完了条件:

- llama.cpp で少なくとも context length と batch size を振った token/s を記録できる。
- TP/PP など未対応条件を失敗ではなく structured result として保存できる。
- Qwen3 実装前に、uLLM が狙う throughput baseline が表になっている。

### M2: Format tooling v0.1

目的:

- `.ullm` を作る、読む、検査するための最小ツールを作る。

手順:

1. `python/ullm_format/` を作る。
2. manifest reader/writer を実装する。
3. binary tensor writer/reader を実装する。
4. checksum validation を実装する。
5. CLI を仮実装する。

```bash
python -m ullm_format inspect path/to/model.ullm
python -m ullm_format validate path/to/model.ullm
python -m ullm_format dump-manifest path/to/model.ullm
```

6. 小さな dummy tensor を保存して round-trip test を通す。
7. OOM を避けるため、tensor reader は chunk iterator を持つ。

成果物:

- `python/ullm_format/`
- `tests/format/test_manifest.py`
- `tests/format/test_tensor_roundtrip.py`

完了条件:

- dummy `.ullm` container を作成、検査、読み戻しできる。
- 破損 checksum を検出できる。
- 1 tensor を一括読み込みせず chunk 単位で読める。

### M3: Qwen3 import path

目的:

- Hugging Face 形式の Qwen3-14B と Qwen3-30B-A3B から `.ullm` へ変換する。
- まだ量子化せず、FP16/BF16 tensor を保存する。

手順:

1. Qwen3-14B と Qwen3-30B-A3B の model config と tensor name を調査して `docs/research/qwen3-import-v0.1.md` にまとめる。
2. `python/ullm_format/import_hf_qwen3.py` を作る。
3. safetensors を streaming で読む。
4. tensor name mapping を manifest に保存する。
5. tokenizer metadata を保存する。
6. architecture graph の最小表現を manifest に保存する。
7. HF model と `.ullm` import 後の tensor checksum を比較する。
8. Qwen3-30B-A3B の MoE/expert tensor を manifest で表現する。

成果物:

- `docs/research/qwen3-import-v0.1.md`
- `python/ullm_format/import_hf_qwen3.py`
- Qwen3-14B FP16/BF16 `.ullm` sample
- Qwen3-30B-A3B FP16/BF16 `.ullm` sample

完了条件:

- Qwen3-14B と Qwen3-30B-A3B の全 tensor が `.ullm` に保存される。
- tensor shape、dtype、checksum が元モデルと一致する。
- tokenizer metadata が読み戻せる。
- dense と MoE の architecture graph 差分を manifest で表現できる。

### M4: Reference executor v0.1

目的:

- `.ullm` から読み込んだ Qwen3-14B と Qwen3-30B-A3B で forward correctness を確認する。

手順:

1. `python/ullm_eval/reference_executor.py` を作る。
2. `.ullm` tensor reader から PyTorch tensor を構築する。
3. Qwen3 dense path を最小実装する。
4. Qwen3 MoE path を最小実装する。
5. RoPE、GQA、RMSNorm、MLP、MoE routing、attention mask を実装する。
6. HF Transformers の logits と比較する。
7. greedy decode で短い生成を確認する。
8. 比較スクリプトを作る。

```bash
python -m ullm_eval.compare_hf_logits \
  --hf-model path/to/hf-model \
  --ullm-model path/to/model.ullm \
  --prompt "Hello"
```

成果物:

- `python/ullm_eval/reference_executor.py`
- `tests/runtime/test_qwen3_dense_logits.py`

完了条件:

- FP16/BF16 `.ullm` で HF logits と許容誤差内に一致する。
- 1 prompt の greedy decode が動く。
- failure 時に layer 単位で差分を追える debug 出力がある。
- Qwen3-30B-A3B の expert routing 差分を追える。

### M5: Quantization experiment framework v0.1

目的:

- `aq` と `sq` の詳細仕様を先に決めず、複数の量子化候補を安全に試せる実験基盤を作る。
- 保存、読み込み、dequant、評価を、特定 variant に依存せず記録できるようにする。

初期候補:

- LUT/index 系
- per-group scale 系
- per-channel scale 系
- tensor family 別 metadata
- outlier handling
- hardware-native FP8 系
- dequant-free / near-dequant-free 系

手順:

1. `docs/research/quantization-experiment-protocol-v0.1.md` を作る。
2. `docs/specs/quantization-experiment-result-v0.1.md` を作る。
3. tensor family を仮定義する。

- attention q/k/v/o
- MLP up/gate/down
- embedding
- output head
- norm

4. calibration dataset loader を streaming で作る。
5. layer/tensor ごとの統計を取る。
6. 実験 ID、seed、calibration 条件、payload layout、bpp 計算方法を保存する。
7. candidate ごとに pack/dequant hook を実装できる interface を作る。
8. `.ullm` manifest には固定 variant 名ではなく `quant_experiment_id` と `payload_schema` を保存する。
9. dequant 後の MSE、cosine similarity、max error を測る。
10. reference executor で候補 model を実行する。
11. perplexity の簡易評価を実行する。
12. 結果から採用候補を絞り、仕様化できる段階になってから `docs/specs/aq-*.md` または `docs/specs/sq-*.md` を作る。

成果物:

- `docs/research/quantization-experiment-protocol-v0.1.md`
- `docs/specs/quantization-experiment-result-v0.1.md`
- `python/ullm_quant/experiments.py`
- `python/ullm_quant/pack.py`
- `python/ullm_eval/perplexity.py`
- `tests/quant/test_experiment_roundtrip.py`

完了条件:

- 少なくとも 2 種類の量子化候補を同じ result schema で比較できる。
- `.ullm` から candidate payload を読み、dequant または native execution 用に解釈できる。
- FP16/BF16 baseline と candidate の perplexity 差分を測れる。
- bpp を manifest と評価ログへ保存できる。
- 仕様として固定しない候補を明確に experimental として扱える。

### M6: Evaluation baseline

目的:

- 量子化 candidate の改善が本当に効いているか判断できる評価基盤を固定する。

手順:

1. `docs/specs/evaluation-v0.1.md` を作る。
2. 評価対象を最小セットから始める。

- perplexity
- short generation sanity test
- long context sanity test
- coding prompt sanity test
- latency/tokens/s smoke benchmark

3. 評価条件を固定する。

- model revision
- tokenizer revision
- context length
- calibration dataset
- evaluation dataset
- random seed
- dtype
- backend
- hardware

4. 結果保存形式を定義する。
5. baseline と candidate の比較レポートを生成する。

成果物:

- `docs/specs/evaluation-v0.1.md`
- `python/ullm_eval/report.py`
- `benchmarks/results/` のサンプル JSON

完了条件:

- 同じモデル、同じ条件で baseline と candidate を比較できる。
- 評価結果に bpp、メモリ使用量、tokens/s、perplexity が含まれる。
- 評価スクリプトが巨大データを一括読みしない。

### M7: Quantization quality iteration

目的:

- Unsloth Dynamic 2.0 GGUF 系と比較できる水準へ向けて、量子化候補を実験で絞る。

手順:

1. Unsloth Dynamic 2.0 GGUF の比較対象モデルを固定する。
2. llama.cpp/imatrix の比較手順を固定する。
3. 候補 variant を複数作り、次の要素を順に試す。

- tensor family 別 LUT
- layer-wise mixed quantization
- per-channel scale
- group size 32/64/128 comparison
- outlier side tensor
- activation-aware calibration
- Hessian/importance 近似
- MoE expert 別 codebook

4. 各改善を一つずつ feature flag 化する。
5. 改善ごとに bpp、accuracy、speed、file size を記録する。
6. 採用する改善だけを安定仕様へ昇格する。

成果物:

- `docs/research/aq-quality-iteration-v0.1.md`
- 安定化した場合のみ `docs/specs/aq-*.md`
- 比較レポート

完了条件:

- 同一 bpp 帯で baseline GGUF との比較表がある。
- 採用した改善と捨てた改善の理由が記録されている。
- experimental payload を安定仕様へ昇格するか、明示的に捨てられる。

### M8: C++ runtime foundation

目的:

- Python reference から独立した runtime の土台を作る。

手順:

1. `runtime/` を作る。
2. C++20 の build system を決める。
3. manifest parser を実装する。
4. tensor mmap reader を実装する。
5. CPU backend interface を定義する。
6. FP16/BF16 baseline tensor を読めるようにする。
7. 選定済み candidate payload の CPU dequant を実装する。
8. 小さな matmul path を実装する。
9. Python reference と C++ runtime の tensor/dequant 結果を比較する。

成果物:

- `runtime/include/`
- `runtime/src/`
- `runtime/backends/cpu/`
- `tests/runtime/`

完了条件:

- C++ runtime が `.ullm` manifest と tensor table を読める。
- C++ runtime の candidate dequant が Python reference と一致する。
- 小さな dummy layer の出力が Python reference と一致する。

### M9: CPU backend v0.1

目的:

- CPU を correctness backend として成立させる。
- AVX-512/VNNI/AMX の最適化は段階的に入れる。

手順:

1. scalar reference kernel を作る。
2. AVX2 fallback を作る。
3. AVX-512 path を作る。
4. VNNI/AMX の capability detection を作る。
5. selected candidate dequant + GEMM の融合可能性を測る。
6. small batch decode の latency を測る。
7. kernel selection log を出す。

成果物:

- `kernels/cpu/`
- CPU capability detector
- CPU benchmark report

完了条件:

- 同じ `.ullm` model が CPU backend で実行できる。
- capability detection により使った kernel が記録される。
- scalar fallback と SIMD path の結果差分が許容範囲内である。

### M10: `sq` experiment protocol v0.1

目的:

- `sq` を `aq` の高速実行モードではなく、別の量子化系列として実験する。
- 詳細仕様は、FP8 候補の benchmark と hardware validation 後に固定する。

手順:

1. `docs/research/sq-experiment-protocol-v0.1.md` を作る。
2. 初期対象を FP8 に固定する。
3. 将来対象 hardware-native format を整理する。

- FP8 E4M3/E5M2
- MXFP8
- NVFP4
- AMD FP8/FP4 系

4. dequant 制約を明文化する。

- LUT 禁止
- 複雑な非線形補正は禁止
- scale、shift、単純な型変換、hardware native unpack は許容

5. `sq` manifest metadata を定義する。
6. backend capability と required capability を定義する。
7. A100 など INT8 例外扱いの判断基準を定義する。
8. FP8 candidate を複数試せるようにする。
9. FP4 系は検証環境が高額なため、v0.1 の実装対象から外す。

成果物:

- `docs/research/sq-experiment-protocol-v0.1.md`
- `schemas/ullm-manifest-v0.1.schema.json` の experimental sq examples

完了条件:

- `sq` model を `aq` loader が誤って処理しない。
- `sq` candidate が要求する hardware capability を manifest で表現できる。
- unsupported backend で明確に拒否できる。
- FP8 candidate の実測結果が stable spec 化の判断材料として保存されている。

### M11: GPU backend research and prototype

目的:

- HIP backend を最初の GPU backend として実装し、V620/R9700 で検証する。

手順:

1. `docs/research/gpu-backend-matrix-v0.1.md` を作る。
2. V620、R9700、MI300X、AVX-512 CPU、その他 GPU/NPU/TPU を別の priority tier に分ける。
3. HIP C++ backend の ABI と build path を決める。
4. kernel API を C++ runtime から呼べる形にする。
5. GEMM microbenchmark を作る。
6. FP8 `sq` candidate の prototype を作る。
7. selected candidate dequant + GEMM の prototype を作る。
8. MI300X で同じ benchmark を実行する準備をする。
9. NVIDIA backend は参照調査と interface 設計に留め、実機確保後に実装する。

成果物:

- `docs/research/gpu-backend-matrix-v0.1.md`
- `kernels/hip/` prototype
- `kernels/cuda/` interface placeholder
- GEMM microbenchmark

完了条件:

- GPU ごとの supported dtype、preferred layout、unsupported reason が記録されている。
- microbenchmark が backend、dtype、batch、shape を記録する。
- C++ runtime から HIP prototype kernel を呼べる。
- V620/R9700 で benchmark を実行できる。
- MI300X で検証すべき項目が明確になっている。

### M12: Scheduler and serving v0.1

目的:

- model format と kernel だけでなく、serving throughput を測れる runtime にする。

手順:

1. `docs/specs/runtime-scheduler-v0.1.md` を作る。
2. request、sequence、batch、KV block の data structure を決める。
3. prefill executor と decode executor を分ける。
4. continuous batching を実装する。
5. prefix cache を設計する。
6. prefill/decode を同一 GPU で分けて測る。
7. prefill/decode を異なる GPU で分ける実験を追加する。
8. OpenAI-compatible API は v0.1 では必須にしない。内部 benchmark API を先に作る。

成果物:

- `docs/specs/runtime-scheduler-v0.1.md`
- scheduler prototype
- serving benchmark

完了条件:

- prefill tokens/s、decode tokens/s、total tokens/s を分けて記録できる。
- batch size を上げた時の throughput curve を出せる。
- prefill/decode disaggregation の実験条件が保存される。

### M13: Advanced architecture support

目的:

- MTP、MoE、DiffusionGemma などを後付けで壊さず入れられるようにする。

手順:

1. MoE routing を architecture graph に追加する。
2. expert tensor と expert family codebook を manifest に追加する。
3. MTP/drafter head を executor interface に追加する。
4. draft/verify loop を scheduler に追加する。
5. diffusion-style executor を別 executor として定義する。
6. autoregressive decode と diffusion refinement の共通部分と非共通部分を分ける。
7. Gemma 4 MTP または DiffusionGemma を次の correctness target として選ぶ。

成果物:

- `docs/specs/model-ir-v0.2.md`
- `docs/specs/executor-v0.2.md`
- advanced target research note

完了条件:

- dense Qwen3 path を壊さず MoE/MTP/diffusion の metadata を表現できる。
- 未対応 executor は load 時に明確に拒否できる。

### M14: NPU and vendor backend plugin plan

目的:

- JAX/TPU、Intel/AMD NPU、Huawei Ascend を CPU/GPU backend と同じ前提で扱わない。
- vendor SDK 依存を plugin 境界に閉じる。

手順:

1. `docs/plans/vendor-backend-plugin-plan-v0.1.md` を作る。
2. backend plugin ABI を仮定義する。
3. JAX/TPU の XLA 経由 import path と制約を調査する。
4. CANN/Ascend の LLM inference API と graph import path を調査する。
5. Intel/AMD NPU の graph compiler path を調査する。
6. `.ullm` から vendor runtime が必要とする形式への変換責務を決める。
7. TPU/NPU/Ascend では `aq` と `sq` を直接全対応させず、capability に応じて拒否または変換する。

成果物:

- `docs/plans/vendor-backend-plugin-plan-v0.1.md`
- backend plugin ABI draft

完了条件:

- vendor backend が未実装でも core runtime が成立する。
- plugin が対応しない quant format を明確に拒否できる。

### M15: Rust control plane v0.1

目的:

- C++20 runtime と HIP kernels を、Rust 側の control plane から管理する。
- API、scheduler、telemetry、prefill/decode 割り当て、速度予測を C++ kernel 実装と分離する。

手順:

1. `crates/ullm_control/` を作る。
2. `crates/ullm_scheduler/` を作る。
3. `crates/ullm_api/` を作る。
4. C ABI または cxx bridge で C++ runtime を呼ぶ境界を決める。
5. model load、request enqueue、prefill dispatch、decode dispatch の state machine を Rust で定義する。
6. backend capability と runtime telemetry を Rust 側へ集約する。
7. scheduler が prefill/decode の割り当て先を選べるようにする。
8. crash 時に C++ backend と Rust control plane の責務境界が分かる log を出す。

成果物:

- `crates/ullm_control/`
- `crates/ullm_scheduler/`
- `crates/ullm_api/`
- C++ runtime FFI draft

完了条件:

- Rust から C++ runtime の dummy backend を呼べる。
- request lifecycle を Rust 側で追跡できる。
- backend capability に基づいて prefill/decode の割り当て方針を変えられる。

### M16: Speed prediction v0.1

目的:

- prefill/decode の速度を実測前または軽い probe から予測し、scheduler と配布時の hardware guidance に使う。

手順:

1. `docs/specs/speed-prediction-v0.1.md` を作る。
2. 入力特徴量を定義する。

- model architecture
- parameter count
- active parameter count
- quant family
- bpp
- tensor layout
- backend
- GPU/CPU capability
- memory bandwidth estimate
- compute throughput estimate
- context length
- batch size
- KV cache size
- prefill/decode split

3. GEMM microbenchmark と end-to-end benchmark を同じ JSON schema へ保存する。
4. roofline 近似を使った初期 predictor を作る。
5. 実測結果で補正する calibration layer を作る。
6. prediction error を benchmark report に保存する。
7. scheduler が予測値を参照して prefill/decode 割り当て候補を出せるようにする。

成果物:

- `docs/specs/speed-prediction-v0.1.md`
- `python/ullm_eval/speed_predictor.py`
- Rust scheduler から使う predictor interface
- prediction benchmark report

完了条件:

- V620/R9700 の実測値に対して prefill/decode tokens/s の予測誤差を出せる。
- MI300X と AVX-512 backend 追加時に同じ predictor schema を使える。
- scheduler が予測値を log に出せる。

### M17: Distribution plan v0.1

目的:

- 発表・宣伝前に、配布形式と導入経路を最低限決める。

手順:

1. `docs/plans/distribution-plan-v0.1.md` を作る。
2. ソース配布、binary 配布、model container 配布を分ける。
3. 初期配布物を定義する。

- source release
- Python format/quantization package
- C++/HIP runtime binary
- Rust API/scheduler binary
- sample `.ullm/` directory container
- single-file `.ullm` は未決定事項として扱う

4. Linux x86_64 + ROCm/HIP を最初の binary target にする。
5. Docker/Podman image を配布するか決める。
6. GitHub Releases、PyPI、crates.io、container registry の使い分けを決める。
7. model artifact の checksum、manifest、provenance を公開する。
8. Apache-2.0 の NOTICE、third-party notices、reference-code policy を release checklist に入れる。

成果物:

- `docs/plans/distribution-plan-v0.1.md`
- release checklist
- packaging prototype

完了条件:

- V620/R9700 で動く最小 runtime を第三者が再現できる手順がある。
- `.ullm/` directory container の配布方針が明確である。
- single-file `.ullm` を後で検討するための未決定事項が記録されている。
- license/notice の確認手順が release checklist に入っている。

### M18: Announcement gate v0.1

目的:

- V620/R9700、MI300X、AVX-512 まで到達した段階で、一度発表・宣伝できる状態を作る。

手順:

1. 発表対象の benchmark scenario を固定する。
2. Qwen3-14B、Qwen3-30B-A3B、Qwen3.5 または Gemma4 のうち、公開できる結果を選ぶ。
3. Unsloth Dynamic 2.0 GGUF、llama.cpp、vLLM、SGLang、ATOM、TensorRT-LLM との比較条件を固定する。
4. prefill/decode throughput と速度予測の結果を含める。
5. 再現手順、hardware、driver、ROCm version、commit hash を公開する。
6. 未対応ハードウェアの対応予定を明記する。

成果物:

- announcement benchmark report
- public demo script
- release notes draft

完了条件:

- 比較条件が再現可能である。
- 速度だけでなく精度、bpp、メモリ、予測誤差を提示できる。
- 未対応機能を誇張せず、次の対応範囲を示せる。

### M19: Reference source policy

目的:

- llama.cpp、vLLM、SGLang、ATOM、TensorRT-LLM を参照できるようにしつつ、uLLM の Apache-2.0 方針と実装の独立性を守る。

手順:

1. `reference-src/` を Git 管理外にする。
2. `tools/fetch-reference-sources.sh` で浅い clone を再取得できるようにする。
3. 各参照リポジトリの commit、license、用途を `docs/research/reference-source-inventory-v0.1.md` に記録する。
4. 参照コードを uLLM 実装へ直接コピーしない。
5. 実装に必要な知見は、コードではなく設計メモや仕様差分として記録する。
6. 外部コードを取り込む必要が出た場合は、事前に ADR を追加し、license、NOTICE、著作権表記、変更範囲を確認する。
7. license 未確認の参照元は、読解と比較だけに限定し、再利用しない。

成果物:

- `.gitignore`
- `tools/fetch-reference-sources.sh`
- `docs/research/reference-source-inventory-v0.1.md`
- `docs/decisions/0001-license-and-reference-code.md`

完了条件:

- 参照ソースが手元にあるが Git には入っていない。
- 各参照元の commit と license 状態が記録されている。
- uLLM 実装時の code-copy 禁止ルールが明文化されている。

## Immediate Work Queue

最初に着手する順序は次の通り。

1. `docs/specs/inference-benchmark-result-v0.1.md` を書く。
2. `docs/plans/existing-engine-benchmark-plan-v0.1.md` を書く。
3. `benchmarks/results/README.md` を作る。
4. llama.cpp benchmark wrapper を作る。
5. V620/R9700 で context length、batch、prompt/generated token 数を振って llama.cpp token/s を測る。
6. vLLM、SGLang、ROCm/ATOM、TensorRT-LLM は V620 で無理に動かさず、unsupported reason を記録できるようにする。
7. `.ullm` directory container の `tensors/`、`codebooks/`、`scales/` を「実験 payload 用の論理ストレージ」として文書化する。
8. `schemas/ullm-manifest-v0.1.schema.json` を作る。
9. `python/ullm_format/` の manifest validation を作る。
10. dummy `.ullm` directory container の round-trip test を作る。
11. 量子化実験 protocol と result schema を作る。
12. Qwen3-14B と Qwen3-30B-A3B の import 調査に進む。
13. HIP C++ backend の build skeleton を作る。
14. V620/R9700 向け GEMM microbenchmark を作る。
15. prefill/decode 速度予測の v0.1 仕様を書く。
16. 配布計画と release checklist を作る。

## Review Points

後から添削すべき点は次の通り。

- 初期 runtime を C++20 中心にする判断が妥当か。
- single-file `.ullm` を本当に必要とするか。
- single-file `.ullm` を作る場合、pack 方式を tar-like container にするか、custom binary container にするか。
- Qwen3-14B と Qwen3-30B-A3B のどちらを先に correctness target にするか。
- Qwen3.5 と Gemma4 のどちらを先に advanced target にするか。
- `aq` と `sq` の candidate を stable spec に昇格させる判断基準は何か。
- 既存推論エンジン benchmark の最初の測定 grid をどこまで広げるか。
- HIP V620/R9700 でどこまで性能を追うか。
- MI300X 調達後の発表基準をどこに置くか。
- AVX-512 backend をどこまで先に作るか。
- NVIDIA backend、JAX/TPU、NPU/Ascend をいつ plugin 計画へ進めるか。
- Rust control plane と C++ runtime の FFI 境界をどうするか。
- 配布を GitHub Releases、PyPI、crates.io、container image のどれから始めるか。
- Apache-2.0 を維持しながら外部実装を参照するための運用が十分か。

## Current Integrated Plan

現時点では、まず既存推論エンジンの benchmark schema と測定 harness を作り、llama.cpp on V620/R9700 の token/s baseline を取る。次に `.ullm` directory container と quantization experiment payload の受け皿を作る。その後、HIP C++ backend を V620/R9700 で進め、MI300X で FP8 系 `sq` candidate のサーバー級 throughput を確認し、その後 AVX-512 backend を追加する。

最初の成功条件は、既存推論エンジンの token/s を context length、batch、prompt/generated tokens、TP、PP、GPU 数などの条件つきで保存できる状態である。量子化方式は、その baseline と評価基盤ができた後に candidate として試し、結果を見てから stable spec へ昇格させる。その後、Qwen3-14B、Qwen3-30B-A3B、Qwen3.5、Gemma4 へ進む。
