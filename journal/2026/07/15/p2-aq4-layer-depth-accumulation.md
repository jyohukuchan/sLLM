# P2 AQ4 multi-layer depth accumulation

## 前回の要点

- layer0のQKV+Z source-BF16 combined置換は、固定3-step診断でlayer-output relative L2 `1.092549252e-03` だった。
- QKV+Z combinedはA+B combinedの約8.78倍で、単層のfamily追加よりlayer-depth accumulationを先に確認する判断になった。

## 今回の変更点

### 監査根拠

- source config `Qwen3.5-9B/config.json` の `text_config.layer_types[0..4]` は `linear_attention, linear_attention, linear_attention, full_attention`、`full_attention_interval=4` である。
- production package manifestにもlayer 0〜2は `linear_attn.in_proj_qkv/in_proj_z`、layer 3は `self_attn.q/k/v/o_proj` が存在し、同じ層種を独立に確認できる。
- source full-model checkpoint capture `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-8-seq16/metadata.json` は、source BF16 model、token IDs 1〜16、layer 0〜7のbefore/after hiddenをfloat32 sidecarとして記録する。今回の入力は `layer_0_before.f32` のtoken IDs 1,2,3を切り出した。
- linear sequence runnerは任意`layer_index`、`sequence_len`、external residualを受け、通常wrapperは全diagnostic optionを`None`にする。QKV/Z override境界はproduction matvec後のconv前およびSiLU-mul前である。`crates/ullm-engine/src/main_parts/part_05.rs:459-511`
- mixed golden-prefix CPU loopはactual-prefixのcurrent hiddenを層順に進め、layer kindを分岐できるが、monolithic fixture runnerである。`crates/ullm-engine/src/main_parts/part_01.rs:7632-7652`
- self-attention model-loop smokeはmanifestのself-attention layer集合を独立に実行する構成で、linear layerからのtrack別external residualとstateを受け渡す公開step interfaceではない。`crates/ullm-engine/src/main_parts/part_02.rs:2528-2605,2898-2930,3179-3186`

### 実装と実測

- CPU-only `package-linear-attn-depth-step-diagnostic` を追加した。既存linear runnerへexternal residualと既存QKV/Z hooksだけを渡し、live production RMSNorm、layer output、linear recurrent state、hash、finite、RSSを出力する。通常worker経路は変更していない。
- `tools/run-aq4-layer-depth-accumulation.py` は4 trackをlayer/track逐次で実行する。各非baseline trackのlive normalized inputから、対応layerのsource BF16 QKV/Zを256-row chunk、float32 accumulateで計算し、full f32 weightを保持しない。
- combined hidden relative L2はdepth1 `0.0154507748`、depth2 `0.0239759476`、depth3 `0.0293600416`。depth1比はそれぞれ1.0、1.5518、1.9002で、3 linear layerまで単調に増幅した。
- depth3はQKV-only `0.0184790259`、Z-only `0.0216196659`。combined interaction residualはdepth1 2.20% amplification、depth2 3.70% amplification、depth3 5.76% cancellationだった。
- memory high-water markはPython約831 MiB、child約701 MiB。track/layer parallelismとTorch threadはすべて1で実行した。
- depth4はlayer3 full attentionのため未実行である。既存mixed CPU loopへ、external residualとtrack-local stateを入出力しつつlinear layerでは既存QKV/Z hooksをroutingできる1-step interfaceがないことがexact blockerである。depth3からdepth4を推測していない。

## 次の行動

- 必要最小の次候補は、linear/full-attention共通のmixed-decoder diagnostic step interfaceである。external residualとtrack-local linear/KV stateを受け、layer outputと更新stateを返す。
- このhookを追加して同じ4 trackをdepth4以降へ継続する。現時点ではQKV/Z以外のprecision family追加やpromotion判断へ進めない。

## 検証

- `cargo check -p ullm-engine --bin ullm-engine`: pass
- `cargo test -p ullm-engine --bin ullm-engine -- --test-threads=1`: 26 passed
- `python3 -m py_compile tools/run-aq4-layer-depth-accumulation.py tests/test_aq4_layer_depth_accumulation.py`: pass
- `pytest -q tests/test_aq4_layer_depth_accumulation.py`: 6 passed
- artifact `SHA256SUMS`: pass
- 通常3-step smokeを2回実行し、stdout SHA256は両方 `9ac224cc444569bb9e5c4c493eacf4007c06c862c03466da31a058a123e4ad9b` でbit-exactだった。
- GPU、service、P3、Gate、holdout、promotionは実行していない。
