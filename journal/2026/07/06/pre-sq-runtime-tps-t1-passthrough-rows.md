# pre-SQ runtime TPS T1 passthrough row reader

- T1の最初の実装境界として、`crates/ullm-engine/src/loader.rs` にpassthrough tensorの行単位F32読み込みを追加した。
- 背景: Qwen3.5-9B packageでは `model.language_model.embed_tokens.weight` と `lm_head.weight` が `BF16` passthroughで、shapeは `248320x4096`。全体をF32展開すると各tensorだけで約3.8GiBになり、token-ID runtimeの初期実装としてはOOMリスクが高い。
- 追加API:
  - `PassthroughF32Rows`
  - `read_named_passthrough_f32_rows`
  - `read_passthrough_payload_f32_rows`
- 既存の `read_passthrough_payload_f32_bytes` と行読み込みで、BF16/F32のelement size解決とdecode helperを共有するようにした。
- 行読み込みは2D tensorのみを受け付け、manifestのshape/elements/payload byte数を検証し、指定されたrow順と重複を保って返す。
- 追加テスト `read_named_passthrough_f32_rows_reads_selected_2d_rows` で、BF16 embedding風payload、F32 lm_head風payload、重複row、out-of-range rowを確認した。
- subagentレビューで指摘されたリスクに対応し、`payload_encoding` / `payload_sha256` をengine側のpassthrough bundleに保持するようにした。
- 行読み込みと全体読み込みでは、`payload_encoding` が未指定または `raw_safetensors_payload` の場合だけraw payloadとして扱う。
- F16 passthrough decodeを追加した。dtype未指定の2-byte payloadは従来どおりBF16推定にしている。
- `Vec::with_capacity` / `vec![]` の直接確保を避け、主要な読み込みbufferと出力vectorは `try_reserve` で失敗を `Err` にするようにした。

検証:

- `cargo fmt --all --check`
- `git diff --check`
- `cargo check -p ullm-engine`
- `cargo test -p ullm-engine loader -- --test-threads=1`
- `cargo test -p ullm-engine package -- --test-threads=1`
- `cargo test -p ullm-engine -- --test-threads=1`

次の行動:

- `package-token-ids-logits-smoke` の最小境界を作る。
- token IDsからembedding rowを取り出し、既存 `Qwen3PackageModelRuntime` の短いCPU pathへ接続する。
- final RMSNormとlm_head/top-kを、まず短い固定token列でNaNなし・shape一致まで確認する。

追加実装:

- `package-token-ids-logits-smoke` CLIを追加した。
- 引数は `PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYERS_CSV|all] [TOKEN_IDS_CSV] [TOP_K] [LM_HEAD_CHUNK_ROWS] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]`。
- token IDsから `model.language_model.embed_tokens.weight` の行だけを読み、既存model-loop helperで指定layer列を通し、`model.language_model.norm.weight` でfinal RMSNormを行い、`lm_head.weight` をchunked row scanして正確なtop-k logitsをJSONで出す。
- default layer listはQwen3.5-9B向けに `0..36` とした。短い検証では `0` や `0,1` を指定できる。
- lm_head top-kは全語彙を走査するため、debug buildで実packageを走らせると重い。synthetic packageのunit testでchunked top-kの順位を検証済み。

追加検証:

- `cargo test -p ullm-engine package_token_ids_logits_tests -- --test-threads=1`
- `cargo build -p ullm-engine`

未実行:

- 実Qwen3.5 packageでの `package-token-ids-logits-smoke` 実行。理由はdebug buildでのfull-vocab lm_head dotが重いため。次はrelease buildまたはlm_head top-kのGPU/chunk matvec化後に短い実runを通す。

追加実行:

- lm_head scan向けにcontiguous row range readerを追加し、`package-token-ids-logits-smoke` のlm_head top-kをrange readへ切り替えた。
- 実packageで短いCPU smokeを完走した。
  - command: `target/debug/ullm-engine package-token-ids-logits-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d 0 1048576 3 1 3 4096`
  - result: `verified=true`
  - layer: `[3]`
  - token_ids: `[1]`
  - total: `81932.757116 ms`
  - lm_head_top_k: `79496.685343 ms`
  - artifact: `benchmarks/results/2026-07-06/engine/package-token-ids-logits-smoke-layer3-cpu.json`
- layer `0` はこの package ではself-attn layerではなく、`model.language_model.layers.0.self_attn.q_norm.weight` が存在しないため失敗した。短いtoken-ID smokeではself-attn layer listを明示する必要がある。

Hybrid token-ID smoke:

- `package-token-ids-logits-smoke` のlayer loopをhybrid化した。
- layer kindは既存 `package_decoder_layer_kind` で判定する。
- self-attn layerは `qwen3_package_decoder_layer_runtime_from_package` と既存self-attn sequence pathを使う。
- linear-attn layerは既存 `package_linear_attn_mlp_block_sequence_run` を使い、`residual_sequence -> layer_output` として接続する。
- 実package CPU debugでlayer 0単独が完走した。
  - command: `target/debug/ullm-engine package-token-ids-logits-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d 0 1048576 0 1 3 4096`
  - result: `verified=true`
  - layer_kinds: `["linear_attention"]`
  - total: `98001.747322 ms`
- 実package CPU debugでlayer 0 -> layer 3のhybrid 2-layer pathが完走した。
  - command: `target/debug/ullm-engine package-token-ids-logits-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d 0 1048576 0,3 1 3 4096`
  - result: `verified=true`
  - layer_kinds: `["linear_attention", "self_attention"]`
  - total: `96043.625111 ms`
  - artifact: `benchmarks/results/2026-07-06/engine/package-token-ids-logits-smoke-hybrid-layer0-layer3-cpu.json`

未実行:

- `LAYERS_CSV=all` のfull 36-layer token-ID smoke。CPU debugではlinear-attn sequence helperとlm_head全語彙top-kが重く、次はrelease buildまたはGPUで短いrunを試す。
- linear-attn sequence runtimeの正式な `qwen3_loader.rs` / `decoder.rs` 抽出。現時点ではproof優先で `main.rs` の既存smoke helperを再利用している。

Full-layer token-ID smoke:

- `LAYERS_CSV=all` は最初36層で試したが、実packageはlayer `0..31` の32層だったため layer 32 で失敗した。
- manifest上のlayer数を確認し、default layer countを `32` に修正した。
- release buildでfull 32-layer token-ID smokeが完走した。
  - command: `target/release/ullm-engine package-token-ids-logits-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d 0 1048576 all 1 3 8192`
  - result: `verified=true`
  - layers: `0..31`
  - layer kind pattern: `linear_attention, linear_attention, linear_attention, self_attention` repeated
  - total: `74910.642709 ms`
  - layers: `68484.484716 ms`
  - lm_head_top_k: `6295.389955 ms`
  - artifact: `benchmarks/results/2026-07-06/engine/package-token-ids-logits-smoke-all-layers-release-cpu.json`
- release buildでR9700/RDNA4 device `2` のfull 32-layer token-ID smokeも完走した。
  - command: `target/release/ullm-engine package-token-ids-logits-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d 2 1048576 all 1 3 8192`
  - result: `verified=true`
  - layers: `0..31`
  - layer kind pattern: `linear_attention, linear_attention, linear_attention, self_attention` repeated
  - total: `13812.1875 ms`
  - layers: `6532.664256 ms`
  - lm_head_top_k: `7113.847954 ms`
  - artifact: `benchmarks/results/2026-07-06/engine/package-token-ids-logits-smoke-all-layers-release-r9700.json`
- release buildでV620/RDNA2 device `1` のfull 32-layer token-ID smokeも完走した。
  - command: `target/release/ullm-engine package-token-ids-logits-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d 1 1048576 all 1 3 8192`
  - result: `verified=true`
  - layers: `0..31`
  - layer kind pattern: `linear_attention, linear_attention, linear_attention, self_attention` repeated
  - total: `13805.645054999999 ms`
  - layers: `7249.3505000000005 ms`
  - lm_head_top_k: `6399.606487 ms`
  - artifact: `benchmarks/results/2026-07-06/engine/package-token-ids-logits-smoke-all-layers-release-v620.json`

T1到達点:

- token IDsからembedding rowを読み、hybrid full decoder layer、final RMSNorm、lm_head top-kまでCPU/RDNA4/RDNA2で通った。
- ただしこれはfull-sequence smokeであり、長いprefill/decode TPS測定ではない。
- T2では、single requestのprefill/decode loopを作り、prompt `128` / generated `32` から始める。

T2入口:

- `package-token-ids-generate-smoke` CLIを追加した。
- 引数は `PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYERS_CSV|all] [TOKEN_IDS_CSV|len:N] [GENERATED_TOKENS] [TOP_K] [LM_HEAD_CHUNK_ROWS] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]`。
- 現時点の `decode_mode` は `full_sequence_recompute_greedy`。generated tokenはgreedy top-1で選ぶが、true incremental decodeではなく、各decode stepはfull sequenceを再計算する。
- JSONには `incremental_decode=false`、`prefill`、`decode`、`throughput`、`memory`、`correctness` を含め、最終的なincremental TPS pathと区別できるようにした。
- release buildでR9700/RDNA4 device `2` のfull 32-layer prompt1/gen1 generate smokeが完走した。
  - command: `target/release/ullm-engine package-token-ids-generate-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d 2 1048576 all 1 1 3 8192`
  - result: `verified=true`
  - generated_token_ids: `[5328]`
  - prefill wall_ms: `13382.369802`
  - prefill tps: `0.0747251805768026`
  - artifact: `benchmarks/results/2026-07-06/engine/package-token-ids-generate-smoke-all-layers-release-r9700-prompt1-gen1.json`
- release buildでR9700/RDNA4 device `2` のfull 32-layer prompt1/gen2 generate smokeも完走した。
  - command: `target/release/ullm-engine package-token-ids-generate-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d 2 1048576 all 1 2 3 8192`
  - result: `verified=true`
  - generated_token_ids: `[5328, 1438]`
  - prefill wall_ms: `13522.056525`
  - decode recompute step wall_ms: `[14619.469943]`
  - decode timed_step_tps: `0.06840193275808973`
  - artifact: `benchmarks/results/2026-07-06/engine/package-token-ids-generate-smoke-all-layers-release-r9700-prompt1-gen2.json`
- release buildでV620/RDNA2 device `1` のfull 32-layer prompt1/gen2 generate smokeも完走した。
  - command: `target/release/ullm-engine package-token-ids-generate-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d 1 1048576 all 1 2 3 8192`
  - result: `verified=true`
  - generated_token_ids: `[5328, 1438]`
  - prefill wall_ms: `14161.247898`
  - decode recompute step wall_ms: `[14758.867051]`
  - decode timed_step_tps: `0.06775587831670617`
  - artifact: `benchmarks/results/2026-07-06/engine/package-token-ids-generate-smoke-all-layers-release-v620-prompt1-gen2.json`

次のT2作業:

- `full_sequence_recompute_greedy` は成果物schemaとgreedy生成の入口として残す。
- sq format策定に使えるTPSへ進めるには、linear-attn層のstateful step APIとself-attn層のpaged KVを同じhybrid model loopに統合し、`incremental_decode=true` の経路を追加する。

Linear-attn stateful step:

- true incremental decodeへ向けて、linear-attnのconv1d履歴とrecurrent stateをstepごとに更新するhost helperを追加した。
- 単体テストで、step更新したconv1d/recurrent出力とfull-sequence host計算が一致することを確認した。
- `package-linear-attn-stateful-step-smoke` CLIを追加した。
- このCLIは実packageのlinear-attn full-sequence中間値を基準に、step更新で再構成したconv1d、gate/beta、q/k/v split、recurrent出力が一致することを確認する。
- release buildでlayer `0`, sequence_len `3` がR9700/RDNA4 device `2` とV620/RDNA2 device `1` の両方で通った。
  - R9700 command: `target/release/ullm-engine package-linear-attn-stateful-step-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d 2 1048576 0 3`
  - V620 command: `target/release/ullm-engine package-linear-attn-stateful-step-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d 1 1048576 0 3`
  - result: `verified=true`
  - max diffs: host conv `0.000000119`, runtime step conv `0.000000000`, conv activation `0.000000238`, gate `0.000000238`, beta `0.000000060`, host recurrent `0.000000001`, runtime step recurrent `0.000000003`
- `package-linear-attn-stateful-step-smoke` をattention blockまで拡張した。
- recurrent出力後のlinear-attn RMSNorm、SiLU gate、out projection、residual addをstep単位でruntime実行し、full-sequence中間値と比較する。
- R9700/RDNA4 device `2` とV620/RDNA2 device `1` の両方でlayer `0`, sequence_len `3` が通った。
  - result: `verified=true`
  - max diffs: attention norm `0.000000954`, projection input `0.000000179`, attention output `0.000000477`, attention block `0.000000477`
- `package-linear-attn-stateful-step-smoke` をMLP/layer outputまで拡張した。
- attention blockのruntime step出力を入力として、post RMSNorm、MLP gate/up projection、SiLU-mul、down projection、layer residual addをstep単位でruntime実行し、full-sequence中間値と比較する。
- R9700/RDNA4 device `2` とV620/RDNA2 device `1` の両方でlayer `0`, sequence_len `3` が通った。
  - result: `verified=true`
  - max diffs: post norm `0.000000238`, MLP gate `0.000000477`, MLP up `0.000000298`, MLP activation `0.000000477`, MLP output `0.000000238`, layer output `0.000000954`
- `PackageLinearAttnResidentStepLayer` を追加した。
- 1つのlinear-attn layerについて、重みとconv/recurrent stateを保持し、input RMSNorm、qkv/a/b/z projection、conv1d、recurrent、attention post、MLPを1 token stepで実行する。
- 既存 `package-linear-attn-stateful-step-smoke` にresident step layerのlayer output比較を追加した。
- R9700/RDNA4 device `2` とV620/RDNA2 device `1` の両方でlayer `0`, sequence_len `3` が通った。
  - result: `verified=true`
  - resident_step_layer_output_max_abs_diff: `0.000000954`

Incremental generate入口:

- `package-token-ids-generate-smoke` の標準経路を `hybrid_incremental_greedy` / `incremental_decode=true` に切り替えた。
- 旧 `full_sequence_recompute_greedy` は `ULLM_GENERATE_DECODE_MODE=full_sequence_recompute_greedy` で残した。
- linear-attn layerは `PackageLinearAttnResidentStepLayer`、self-attn layerは `Qwen3DecoderLayerRuntime` のpaged KV stateを使う。
- 現時点では選択されたlayerの重みをF32 dequant runtime bufferとして常駐させるため、full 32-layer residentはVRAM上限に当たる可能性が高い。
- release buildでR9700/RDNA4 device `2` のlayer `0` prompt1/gen2 incremental generateが完走した。
  - command: `target/release/ullm-engine package-token-ids-generate-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d 2 1048576 0 1 2 3 8192`
  - result: `verified=true`
  - generated_token_ids: `[180887, 64717]`
  - prefill wall_ms: `7313.963254`
  - decode step_wall_ms: `[6458.051357]`
- release buildでR9700/RDNA4 device `2` のhybrid layer `0,3` prompt1/gen2 incremental generateが完走した。
  - result: `verified=true`
  - generated_token_ids: `[188789, 220970]`
  - prefill wall_ms: `7339.8311029999995`
  - decode step_wall_ms: `[7000.575762]`
- release buildでV620/RDNA2 device `1` のhybrid layer `0,3` prompt1/gen2 incremental generateも完走した。
  - result: `verified=true`
  - generated_token_ids: `[188789, 220970]`
  - prefill wall_ms: `7617.804917`
  - decode step_wall_ms: `[7137.121042999999]`
