# R9700 Qwen3-14B-FP8 external engine benchmarks

## 前回の要点

- llama.cpp の R9700/V620 baseline と quant family memory sweep は完了済み。
- 次の対象は R9700 + `Qwen/Qwen3-14B-FP8` の vLLM、SGLang、ROCm/ATOM。
- Hugging Face からの取得は旧 `huggingface-cli` ではなく `hf` コマンドを使う。

## 今回の変更点

- `hf download` で `Qwen/Qwen3-14B-FP8` を `~/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8/` に配置した。
- `tools/run-external-benchmark.py` を追加し、外部エンジンの実行、標準出力/標準エラー保存、ROCm VRAM サンプリング、JSONL 行の記録をまとめた。
- vLLM ROCm nightly は `ROCR_VISIBLE_DEVICES=1` で R9700/gfx1201 を選ぶ必要があった。`HIP_VISIBLE_DEVICES=1` だけだと AITER が V620/gfx1030 を見て失敗した。
- SGLang は source commit `3add35e26dc0623d6647e226de7d17754bb61804` を使い、測定用に ignored source tree へ二つのローカルパッチを入れた。
  - `sgl-kernel/setup_rocm.py` に experimental `gfx1201` を許可。
  - `layernorm.py` の HIP 分岐を、インストール済み vLLM ROCm の 4 引数 `fused_add_rms_norm` ABI に合わせた。
- ATOM は source commit `cce1a6e56dcd8cb300183f81901fdaed6090d951` を使い、`amd-aiter==0.1.16.post2` wheel に無い `moe_shuffle_scale` を `shuffle_scale` fallback で import できるようにした。
- ATOM はモデルロード後の warmup で ModelRunner が `exitcode=-11` になり、`--enforce-eager` と warmup 縮小でも成功しなかった。ATOM Dockerfile は AITER HEAD のソースビルド前提なので、次回は AITER を `gfx1201` 向けにビルドする必要がある。
- 追加で `reference-src/aiter` を clone し、AITER commit `71829a74bc2600bfbce4c05f85ecbe0eeb994323` を `build/envs/atom-rocm` へ editable install した。
  - `AITER_USE_SYSTEM_TRITON=1 BUILD_TARGET=rocm GPU_ARCHS=gfx1201 PREBUILD_KERNELS=0` を使用。
  - `PREBUILD_KERNELS=1` は FlyDSL AOT が `GPU_ARCHS=gfx1201` にもかかわらず大きな `gfx950` kernel set をコンパイルし始めたため中止した。
  - install 後の確認では `amd-aiter==0.1.17.dev155+g71829a74b`、`aiter.__file__` は `reference-src/aiter/aiter/__init__.py`、`get_gfx()` は `gfx1201`、`moe_shuffle_scale` も export されていた。
- AITER ソース版の ATOM では、同じ `Qwen3-14B-FP8` のまま smoke と代表条件が通った。
  - ATOM `benchmark_serving` は `--metric-percentiles 50,95` だけだと結果保存後に `KeyError: 'p99_ttft_ms'` で終了コード 1 になる。`50,95,99` で正常終了した。
- ROCm/ATOM 公式の `Qwen3-8B-FP8` recipe を確認し、R9700/gfx1201 で公式相当条件の切り分けを行った。
  - 公式の `Output tok/s` は `mean_tpot_ms` からの `1000 / mean_tpot_ms` と同じ性質の値だった。
  - こちらの JSONL に入れている `decode_tokens_per_second` は ATOM `output_throughput` 由来で、1 request 測定では TTFT と request 全体時間を含む。公式表と直接比較する値ではない。
  - `Qwen3-8B-FP8` の公式相当 CUDAGraph/BF16 KV では `mean_tpot_ms=17.97`、TPOT 由来 `55.65 tok/s` になり、公式の 52.9 tok/s 級はローカルでも再現した。
  - 公式設定からモデルだけ `Qwen3-14B-FP8` に変えた再測定では、`mean_tpot_ms=55.458619`、TPOT 由来 `18.031462 tok/s`、wrapper `output_throughput=17.981162 tok/s` だった。
  - `Qwen3-14B-FP8` の pp512/tg128 は `--enforce-eager` を外すと、wrapper decode が `9.15 -> 18.27 tok/s` に改善した。
  - `Qwen3-14B-FP8` の公式相当 CUDAGraph/BF16 KV pp549/tg256 は wrapper `16.62 tok/s`、TPOT 由来 `17.82 tok/s`、consumed `26.42 GiB` だった。
  - `Qwen3-14B-FP8` の FP8 KV は、正しい `--block-size 128` でも TPOT 由来 `9.80 tok/s` で、この single request 条件では高速化しなかった。

## 結果

- vLLM representative `pp512/tg128/tp1/pp1`: prefill 94.66 tok/s、decode 23.67 tok/s、total 118.33 tok/s、consumed 28.72 GiB、decode x GiB 679.79。
- SGLang representative `pp512/tg128/tp1/pp1`: prefill 49.50 tok/s、decode 24.99 tok/s、total 74.49 tok/s、consumed 16.81 GiB、decode x GiB 420.12。
- ATOM wheel AITER: smoke 3 variants all failed before readiness。最大 consumed VRAM は約 16.13 GiB、最小は warmup 制限時の約 15.66 GiB。
- ATOM source AITER smoke `pp16/tg8/tp1/pp1`: prefill 23.72 tok/s、decode 11.86 tok/s、total 35.58 tok/s、consumed 24.11 GiB、decode x GiB 285.98。
- ATOM source AITER representative eager `pp512/tg128/tp1/pp1`: prefill 36.60 tok/s、decode 9.15 tok/s、total 45.75 tok/s、consumed 24.17 GiB、decode x GiB 221.14。
- ATOM source AITER representative CUDAGraph `pp512/tg128/tp1/pp1`: prefill 73.09 tok/s、decode 18.27 tok/s、total 91.37 tok/s、consumed 24.30 GiB、decode x GiB 444.07。
- ATOM source AITER official-like `Qwen3-8B-FP8` CUDAGraph/BF16 KV `pp549/tg256/tp1/pp1`: wrapper decode 35.84 tok/s、TPOT 由来 55.65 tok/s、consumed 27.26 GiB。
- ATOM source AITER official settings with only model changed to `Qwen3-14B-FP8` CUDAGraph/BF16 KV `pp549/tg256/tp1/pp1`: wrapper decode 17.98 tok/s、TPOT 由来 18.03 tok/s。
- ATOM source AITER official-like `Qwen3-14B-FP8` CUDAGraph/BF16 KV `pp549/tg256/tp1/pp1`: wrapper decode 16.62 tok/s、TPOT 由来 17.82 tok/s、consumed 26.42 GiB。

## 次の行動

- 外部エンジン比較は、今後は `output_throughput` と TPOT 由来 tok/s を分けて扱う。
- ATOM をさらに測る場合は、今の source AITER env を使い、CUDAGraph を有効にし、`--metric-percentiles 50,95,99` を指定する。
- SGLang/vLLM の R9700 結果は、MI300X 到着後の TP/PP/同時リクエスト拡張時の比較基準にする。
