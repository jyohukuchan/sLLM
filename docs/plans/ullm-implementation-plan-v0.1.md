# uLLM implementation plan v0.1

## Purpose

この文書は、`docs/concepts/ullm-concept-v0.1.md` の構想を、実装可能な手順へ落とすための初期計画である。

この版では、最短で価値を確認するために次の順序を採用する。

1. `.ullm` コンテナ仕様を仮固定する。
2. Qwen3 dense small model を最初の correctness target にする。
3. CPU/Python reference 実装で `aq` の保存、読み込み、dequant、評価を成立させる。
4. C++ runtime と CPU backend を作り、同じ `.ullm` を読めるようにする。
5. ベンチ基盤を固定してから、`aq` の改善と `sq` の GPU backend へ進む。

## Assumptions

- 構想文書は `docs/concepts/` に置く。
- 具体的な実装計画は `docs/plans/` に置く。
- 作業記録は `journal/` に置く。
- 初期実装は、正しさ検証と量子化研究を Python/PyTorch で進める。
- 長期の runtime は C++20 を中心にし、CUDA/HIP/CANN などの backend を追加する。
- Python は reference、converter、quantizer、evaluation のために使う。
- Rust を使うかどうかは v0.1 では決めない。C++ runtime が不便になった時点で CLI や manifest tooling への採用を再検討する。
- 大きなモデルや評価データは一度にメモリへ載せず、streaming と chunk 処理を前提にする。

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
  runtime/
    include/
    src/
    backends/
      cpu/
      cuda/
      hip/
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
  journal/
```

最初にすべてを作る必要はない。Phase 1 では `docs/specs/`、`schemas/`、`python/`、`tests/format/` だけでよい。

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
5. 最初の対象モデルを Qwen3 dense small 系に固定する。

成果物:

- `docs/plans/ullm-implementation-plan-v0.1.md`
- 後続の ADR テンプレート

完了条件:

- 構想文書と計画文書が別ファイルになっている。
- 次に作る仕様文書の対象が明確になっている。

### M1: `.ullm` format v0.1 draft

目的:

- `.ullm` の最小仕様を決める。
- `aq` と `sq` の違いを manifest で表現できるようにする。

手順:

1. `docs/specs/ullm-container-v0.1.md` を作る。
2. `schemas/ullm-manifest-v0.1.schema.json` を作る。
3. packed single-file ではなく、最初は directory container を採用する。
4. directory container の標準構成を定義する。

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

5. `manifest.json` に最低限の項目を入れる。

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

6. alignment、endianness、checksum、tensor chunk の扱いを決める。
7. `aq` 用に codebook/LUT 参照を表現する。
8. `sq` 用に hardware-native layout と scale metadata を表現する。
9. 未対応機能を拒否するための capability field を定義する。

成果物:

- `docs/specs/ullm-container-v0.1.md`
- `schemas/ullm-manifest-v0.1.schema.json`
- `tests/format/` の manifest validation test

完了条件:

- 空の `.ullm` directory container を validate できる。
- `aq` と `sq` の manifest example を validate できる。
- manifest schema が unknown field と missing required field を検出できる。

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

### M3: Qwen3 dense import path

目的:

- Hugging Face 形式の Qwen3 dense small model から `.ullm` へ変換する。
- まだ量子化せず、FP16/BF16 tensor を保存する。

手順:

1. Qwen3 dense small の model config と tensor name を調査して `docs/research/qwen3-dense-import-v0.1.md` にまとめる。
2. `python/ullm_format/import_hf_qwen3.py` を作る。
3. safetensors を streaming で読む。
4. tensor name mapping を manifest に保存する。
5. tokenizer metadata を保存する。
6. architecture graph の最小表現を manifest に保存する。
7. HF model と `.ullm` import 後の tensor checksum を比較する。

成果物:

- `docs/research/qwen3-dense-import-v0.1.md`
- `python/ullm_format/import_hf_qwen3.py`
- Qwen3 dense FP16/BF16 `.ullm` sample

完了条件:

- Qwen3 dense small の全 tensor が `.ullm` に保存される。
- tensor shape、dtype、checksum が元モデルと一致する。
- tokenizer metadata が読み戻せる。

### M4: Reference executor v0.1

目的:

- `.ullm` から読み込んだ Qwen3 dense model で forward correctness を確認する。

手順:

1. `python/ullm_eval/reference_executor.py` を作る。
2. `.ullm` tensor reader から PyTorch tensor を構築する。
3. Qwen3 dense の forward path を最小実装する。
4. RoPE、GQA、RMSNorm、MLP、attention mask を実装する。
5. HF Transformers の logits と比較する。
6. greedy decode で短い生成を確認する。
7. 比較スクリプトを作る。

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

### M5: `aq` reference quantizer v0.1

目的:

- `aq` の最小量子化形式を実装し、保存、読み込み、dequant、評価を成立させる。

初期 variant:

- `aq4_lut16_g64`
- 4-bit index
- group size 64
- tensor family ごとに LUT/codebook を持つ
- per-group scale
- per-tensor fallback scale
- outlier は v0.1 では別 tensor に逃がさず、統計だけ保存する

手順:

1. `docs/specs/aq-v0.1.md` を作る。
2. tensor family を定義する。

- attention q/k/v/o
- MLP up/gate/down
- embedding
- output head
- norm

3. calibration dataset loader を streaming で作る。
4. layer/tensor ごとの統計を取る。
5. LUT/codebook の初期生成を実装する。
6. 4-bit index packing を実装する。
7. per-group scale を保存する。
8. dequant reference を実装する。
9. `.ullm` manifest に `aq4_lut16_g64` metadata を保存する。
10. dequant 後の MSE、cosine similarity、max error を測る。
11. reference executor で aq model を実行する。
12. perplexity の簡易評価を実行する。

成果物:

- `docs/specs/aq-v0.1.md`
- `python/ullm_quant/aq.py`
- `python/ullm_quant/pack.py`
- `python/ullm_eval/perplexity.py`
- `tests/quant/test_aq_roundtrip.py`

完了条件:

- Qwen3 dense small を `aq4_lut16_g64` へ変換できる。
- `.ullm` から aq tensor を読み、dequant して forward できる。
- FP16/BF16 baseline と aq の perplexity 差分を測れる。
- bpp を manifest と評価ログへ保存できる。

### M6: Evaluation baseline

目的:

- `aq` の改善が本当に効いているか判断できる評価基盤を固定する。

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
5. baseline と aq の比較レポートを生成する。

成果物:

- `docs/specs/evaluation-v0.1.md`
- `python/ullm_eval/report.py`
- `benchmarks/results/` のサンプル JSON

完了条件:

- 同じモデル、同じ条件で baseline と aq を比較できる。
- 評価結果に bpp、メモリ使用量、tokens/s、perplexity が含まれる。
- 評価スクリプトが巨大データを一括読みしない。

### M7: `aq` quality iteration

目的:

- Unsloth Dynamic 2.0 GGUF 系と比較できる水準へ向けて `aq` を改善する。

手順:

1. Unsloth Dynamic 2.0 GGUF の比較対象モデルを固定する。
2. llama.cpp/imatrix の比較手順を固定する。
3. `aq4_lut16_g64` を基準に、次の改善を順に試す。

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
6. 採用する改善だけを `aq-v0.2` へ昇格する。

成果物:

- `docs/research/aq-quality-iteration-v0.1.md`
- `docs/specs/aq-v0.2.md`
- 比較レポート

完了条件:

- 同一 bpp 帯で baseline GGUF との比較表がある。
- 採用した改善と捨てた改善の理由が記録されている。
- `aq` の loader が古い v0.1 model を読めるか、明示的に拒否できる。

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
7. `aq4_lut16_g64` の CPU dequant を実装する。
8. 小さな matmul path を実装する。
9. Python reference と C++ runtime の tensor/dequant 結果を比較する。

成果物:

- `runtime/include/`
- `runtime/src/`
- `runtime/backends/cpu/`
- `tests/runtime/`

完了条件:

- C++ runtime が `.ullm` manifest と tensor table を読める。
- C++ runtime の aq dequant が Python reference と一致する。
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
5. `aq` dequant + GEMM の融合可能性を測る。
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

### M10: `sq` specification v0.1

目的:

- `sq` を `aq` の高速実行モードではなく、別の量子化系列として仕様化する。

手順:

1. `docs/specs/sq-v0.1.md` を作る。
2. 対象 hardware-native format を整理する。

- FP8 E4M3/E5M2
- MXFP8
- NVFP4
- AMD FP8/FP4 系

3. dequant 制約を明文化する。

- LUT 禁止
- 複雑な非線形補正は禁止
- scale、shift、単純な型変換、hardware native unpack は許容

4. `sq` manifest metadata を定義する。
5. backend capability と required capability を定義する。
6. A100 など INT8 例外扱いの判断基準を定義する。
7. `sq8_fp8_block` を最初の prototype variant にする。

成果物:

- `docs/specs/sq-v0.1.md`
- `schemas/ullm-manifest-v0.1.schema.json` の sq examples

完了条件:

- `sq` model を `aq` loader が誤って処理しない。
- `sq` が要求する hardware capability を manifest で表現できる。
- unsupported backend で明確に拒否できる。

### M11: GPU backend research and prototype

目的:

- GPU backend をすぐ本実装せず、まず target ごとの現実的な順序を決める。

手順:

1. `docs/research/gpu-backend-matrix-v0.1.md` を作る。
2. 実機で確認できる GPU と、仕様調査のみの GPU を分ける。
3. AMD ROCm backend の最初の対象を決める。
4. NVIDIA backend の最初の対象を決める。
5. kernel API を C++ runtime から呼べる形にする。
6. GEMM microbenchmark を作る。
7. `sq8_fp8_block` の prototype を作る。
8. `aq` dequant + GEMM の prototype を作る。

成果物:

- `docs/research/gpu-backend-matrix-v0.1.md`
- `kernels/cuda/` prototype
- `kernels/hip/` prototype
- GEMM microbenchmark

完了条件:

- GPU ごとの supported dtype、preferred layout、unsupported reason が記録されている。
- microbenchmark が backend、dtype、batch、shape を記録する。
- C++ runtime から prototype kernel を呼べる。

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

- Intel/AMD NPU、Huawei Ascend を CPU/GPU backend と同じ前提で扱わない。
- vendor SDK 依存を plugin 境界に閉じる。

手順:

1. `docs/plans/vendor-backend-plugin-plan-v0.1.md` を作る。
2. backend plugin ABI を仮定義する。
3. CANN/Ascend の LLM inference API と graph import path を調査する。
4. Intel/AMD NPU の graph compiler path を調査する。
5. `.ullm` から vendor runtime が必要とする形式への変換責務を決める。
6. NPU では `aq` と `sq` を直接全対応させず、capability に応じて拒否または変換する。

成果物:

- `docs/plans/vendor-backend-plugin-plan-v0.1.md`
- backend plugin ABI draft

完了条件:

- vendor backend が未実装でも core runtime が成立する。
- plugin が対応しない quant format を明確に拒否できる。

## Immediate Work Queue

最初に着手する順序は次の通り。

1. `docs/specs/ullm-container-v0.1.md` を書く。
2. `schemas/ullm-manifest-v0.1.schema.json` を作る。
3. `python/ullm_format/` の manifest validation を作る。
4. dummy `.ullm` directory container の round-trip test を作る。
5. Qwen3 dense small の import 調査を行う。
6. FP16/BF16 `.ullm` import tool を作る。
7. Python reference executor で HF logits と比較する。
8. `aq4_lut16_g64` の仕様を書く。
9. `aq4_lut16_g64` の pack/dequant を実装する。
10. perplexity と tokens/s の最小評価を作る。

## Review Points

後から添削すべき点は次の通り。

- 初期 runtime を C++20 中心にする判断が妥当か。
- `.ullm` を最初から single-file にするか、directory container から始めるか。
- 最初の Qwen3 target model をどれにするか。
- `aq4_lut16_g64` を最初の variant にするか。
- CPU backend をどこまで先に作るか。
- NVIDIA backend と AMD backend の優先順。
- `sq` の最初の target を FP8 にするか、FP4 系まで含めるか。
- NPU/Ascend をいつ plugin 計画へ進めるか。

## Current Integrated Plan

現時点では、まず `.ullm` container と `aq` reference pipeline を完成させる。GPU `sq` は高価値だが、format、evaluation、correctness target が固まる前に着手すると比較不能になりやすい。

最初の成功条件は、Qwen3 dense small を HF 形式から `.ullm` へ変換し、FP16/BF16 baseline と `aq4_lut16_g64` model の logits、perplexity、bpp、file size を同じ評価基盤で比較できる状態である。
